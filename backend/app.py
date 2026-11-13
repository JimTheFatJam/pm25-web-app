"""
backend/app.py -- Local Flask API for the PM2.5 web app.

Serves the frontend, runs server-side spatial cell selection (the
313,845-cell grid and its spatial index live here), AND -- as of this
integration-testing stage -- runs real PM2.5 predictions using the
retrained Baseline + Group D model artifacts and live Google Earth Engine
data. See backend/prediction/ for the prediction pipeline itself; this
file only wires it into HTTP routes and loads the model once at startup.

Every spatial/validation function below is imported directly from the
existing, already-tested modules (selection/cell_selector.py,
selection/request_validator.py, grid/grid_provider.py) -- nothing
reimplemented here.
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flask import Flask, jsonify, request, send_from_directory, Response, stream_with_context
from shapely.geometry import shape, mapping
import shapely
import geopandas as gpd

from config.settings import (
    PHILIPPINES_BOUNDARY_PATH, CELL_AREA_KM2, CELL_SIDE_METERS, DISPLAY_CRS, MIN_PREDICTION_DATE,
)
from grid.grid_provider import load_grid_with_fallback
from selection.cell_selector import select_cells, clip_polygon_to_boundary, compute_selection_area_km2, InvalidPolygonError
from selection.request_validator import validate_prediction_request
from backend.cell_geometry import cell_corners_wgs84

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="")

# ============================================================
# Public-deployment hardening (see backend/limits.py and DEPLOYMENT.md).
# All of this is inert unless PM25_PUBLIC_MODE=1, so local development and
# the test suite behave exactly as before.
# ============================================================
from backend import limits  # noqa: E402

# Behind nginx, the real client address arrives in X-Forwarded-For; without
# this every request would look like it came from 127.0.0.1 and the per-IP
# rate limits would apply to all visitors collectively.
if limits.PUBLIC_MODE:
    from werkzeug.middleware.proxy_fix import ProxyFix
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)

# A drawn polygon or a 2,000-cell id list is a few tens of KB; this simply
# stops someone streaming a huge body at the JSON parser.
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024


@app.after_request
def _security_headers(response):
    """Conservative headers for a public site. The frontend is entirely
    first-party (Leaflet and D3 are vendored, no CDN, no inline eval), so a
    strict CSP costs nothing. 'unsafe-inline' is kept for styles only
    because the map layers set inline style attributes."""
    if not limits.PUBLIC_MODE:
        return response
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    response.headers.setdefault("Permissions-Policy", "geolocation=(), camera=(), microphone=()")
    response.headers.setdefault(
        "Content-Security-Policy",
        "default-src 'self'; img-src 'self' data: blob:; style-src 'self' 'unsafe-inline'; "
        "script-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'self'; "
        "form-action 'self'",
    )
    return response


@app.route("/api/health")
def api_health():
    """Cheap liveness/readiness probe -- no GEE, no model work. Used by
    deploy/deploy.sh to verify a release before declaring success."""
    from backend.prediction.data_availability import get_availability_state
    avail = get_availability_state()
    return jsonify({
        "status": "ok",
        "model_loaded": LOADED_MODEL is not None,
        "explainer_available": EXPLAINER is not None,
        "mobile_service": MOBILE_SERVICE is not None,
        "availability_status": avail.get("status"),
        "latest_valid_date": (avail["result"]["latest_valid_date"].isoformat()
                              if avail.get("result") else None),
        "limits": limits.status(),
    })

print("Loading grid and boundary (once, at startup)...")
GRID = load_grid_with_fallback()
BOUNDARY = gpd.read_file(PHILIPPINES_BOUNDARY_PATH)

# CRITICAL PERFORMANCE FIX: the full boundary has ~393,000 vertices
# (confirmed directly during earlier grid-generation work). Using it
# directly in clip_polygon_to_boundary's polygon.intersection() call was
# tested and found to hang / not complete within 60s for a realistic
# user-drawn polygon -- the exact same expensive-exact-intersection
# problem already solved once for full-grid generation, recurring here
# for the same underlying reason. A simplified boundary (same tolerance
# used for the display boundary) is used for this fast clip check only
# -- this is a UI-level "is this obviously outside the domain" check,
# not a precision requirement, so the small simplification tolerance
# (~500m) is an acceptable, documented trade-off for keeping interactive
# polygon selection actually interactive.
BOUNDARY_FOR_CLIPPING = BOUNDARY.copy()
BOUNDARY_FOR_CLIPPING["geometry"] = BOUNDARY_FOR_CLIPPING.geometry.simplify(0.005)

GRID_SOURCE = GRID.attrs.get("grid_source", "UNKNOWN")
print(f"Grid loaded: {len(GRID):,} cells, source={GRID_SOURCE}")

# Populate config.settings.ANALYSIS_CRS from the grid's own real CRS.
# BUG FIX: this was declared in config/settings.py as "populated at
# runtime from the grid file's own .crs" but that population step was
# never actually written anywhere -- every caller (static_predictors.py,
# dynamic_predictors.py, pipeline.py) was receiving None instead of the
# real CRS, which surfaced as geopandas' "Cannot transform naive
# geometries" the first time a GeoSeries was constructed with crs=None
# and then .to_crs() was called on it.
import config.settings as _settings
_settings.ANALYSIS_CRS = GRID.crs
print(f"ANALYSIS_CRS populated from grid: {GRID.crs}")

# ============================================================
# GEE initialization and model loading -- ONCE, at startup. If either
# fails, the app does not start serving requests. See backend/prediction/
# model_loader.py for exactly what's validated.
# ============================================================
# Must come before `import ee` -- see backend/_ee_windows_compat.py's own
# docstring for exactly why (earthengine-api's CLI dependency chain pulls
# in Unix-only modules that don't exist on Windows; this shim is a no-op
# on Linux/macOS, where the real modules already exist).
from backend import _ee_windows_compat

import ee
from config.settings import GEE_PROJECT_ID
from backend.prediction.model_loader import load_models, ModelLoadError

def _initialize_earth_engine():
    """Authenticate to Earth Engine and return a short description of how.

    Two paths, because a server has no browser to complete an interactive
    OAuth flow:

      * SERVER -- GOOGLE_APPLICATION_CREDENTIALS points at a service-account
        JSON key. Note that ee.Initialize() does NOT read that variable: it
        calls get_persistent_credentials(), which only looks for the file
        `earthengine authenticate` writes, and fails with "Please authorize
        access to your Earth Engine account". The key therefore has to be
        turned into ServiceAccountCredentials explicitly, which is what this
        does. (Confirmed against earthengine-api 1.7.x.)

      * LOCAL -- no such variable, so fall back to those persistent
        credentials, i.e. exactly how a developer machine has always worked.

    The service account needs the Earth Engine API enabled on the project
    plus an Earth Engine role; see DEPLOYMENT.md.
    """
    key_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
    if key_path:
        if not Path(key_path).is_file():
            raise FileNotFoundError(
                f"GOOGLE_APPLICATION_CREDENTIALS points at '{key_path}', which does not exist. "
                f"Either fix the path or unset the variable to use interactive credentials.")
        info = json.loads(Path(key_path).read_text())
        email = info.get("client_email")
        if not email:
            raise ValueError(
                f"'{key_path}' has no client_email field -- it does not look like a service-account "
                f"key. Download a JSON key for the service account, not an OAuth client secret.")
        credentials = ee.ServiceAccountCredentials(email, key_path)
        if GEE_PROJECT_ID:
            ee.Initialize(credentials, project=GEE_PROJECT_ID)
        else:
            ee.Initialize(credentials)
        return f"service account {email}"

    if GEE_PROJECT_ID:
        ee.Initialize(project=GEE_PROJECT_ID)
    else:
        ee.Initialize()
    return "interactive credentials (earthengine authenticate)"


print("Initializing Google Earth Engine...")
try:
    _how = _initialize_earth_engine()
    print(f"  GEE initialized successfully via {_how}.")
except Exception as e:
    print(f"  FAILED to initialize GEE: {type(e).__name__}: {e}")
    print("  See DEPLOYMENT.md's 'Earth Engine authentication' section. The backend will still start "
          "(so the rest of the app remains usable for area selection), but every prediction request "
          "will fail until this is fixed and the backend is restarted.")

try:
    LOADED_MODEL = load_models()
except ModelLoadError as e:
    print(f"\nFATAL: model loading failed: {e}\n")
    print("The backend will still start (so the rest of the app remains usable for area selection), "
          "but /api/generate_prediction will refuse every request until this is fixed and the "
          "backend is restarted.")
    LOADED_MODEL = None

# SHAP explainer (application layer, optional): built ONCE from the
# already-loaded model's Philippine-only XGBoost member, per the thesis's own
# methodology -- see backend/prediction/explainer.py. None if the model
# didn't load or shap is unavailable; predictions never depend on it.
from backend.prediction.explainer import build_explainer
EXPLAINER = build_explainer(LOADED_MODEL) if LOADED_MODEL is not None else None

# Descriptive fields from ensemble_config.json (trained_utc, weighting
# method, transforms) for the prediction export's metadata block only --
# read once here, never used for inference (model_loader.py owns that).
ENSEMBLE_CONFIG_INFO = None
try:
    import json as _json
    from config.settings import MODEL_ENSEMBLE_CONFIG_PATH as _ens_path
    if _ens_path.exists():
        ENSEMBLE_CONFIG_INFO = _json.loads(_ens_path.read_text())
except Exception as _e:  # noqa: BLE001 -- metadata only; never block startup over it
    print(f"  Note: could not read ensemble_config.json for export metadata: {_e}")

# Live GEE availability check -- launched in a background daemon thread,
# NOT run at blocking module-level startup. Two separate, confirmed fixes
# over the previous version:
#   1. The check itself no longer sorts a fully unfiltered GEE collection
#      (which hung indefinitely for large collections like MODIS AOD and
#      ERA5/ERA5-Land -- see data_availability.py's module docstring for
#      the traced root cause and fix).
#   2. Even so, this check can legitimately take a while (several bounded
#      GEE queries) or fail if GEE is temporarily unreachable -- it must
#      not block the whole app from starting and serving other routes.
# Uses the exact same threading.Thread(daemon=True) pattern already used
# for prediction jobs in job_state.py. Read the current state via
# data_availability.get_availability_state() from any request handler;
# never call check_dynamic_data_availability() directly from a route.
from backend.prediction.data_availability import start_availability_check_in_background

print("\nStarting live GEE dynamic-data availability check in the background "
      "(the app will finish starting immediately; see below for progress)...")
start_availability_check_in_background()

# Periodic refresh of that check (mobile T* must advance without a
# restart) -- keeps the last good result if a refresh fails.
from backend.prediction.data_availability import start_availability_refresh_loop, get_availability_state
from config.settings import AVAILABILITY_REFRESH_INTERVAL_SECONDS
start_availability_refresh_loop(AVAILABILITY_REFRESH_INTERVAL_SECONDS)

# Mobile prediction service (backend/mobile.py): point -> cell index built
# once from the loaded grid; the real pipeline core (batched retrieval)
# and explainer are injected. None if the model didn't load.
from backend import mobile as _mobile
CELL_INDEX = _mobile.CellIndex(GRID)
MOBILE_SERVICE = None
if LOADED_MODEL is not None:
    MOBILE_SERVICE = _mobile.MobileService(
        cell_index=CELL_INDEX,
        get_availability=get_availability_state,
        compute_prediction=_mobile.make_compute_prediction(GRID, LOADED_MODEL),
        explain_row=_mobile.make_explain_row(EXPLAINER, LOADED_MODEL),
    )
    print(f"Mobile prediction service ready (scheme: {MOBILE_SERVICE.scheme['id']}, "
          f"T* strategy: {MOBILE_SERVICE.tstar_strategy.name}).")


# ============================================================
# Static frontend
# ============================================================

@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


# ============================================================
# API
# ============================================================

@app.route("/api/grid_info")
@limits.rate_limited("read")
def grid_info():
    from backend.prediction.data_availability import get_availability_state
    from config.pm25_classification import schemes_for_api
    availability = get_availability_state()
    result = availability["result"]
    return jsonify({
        "grid_source": GRID_SOURCE,
        "total_cells": len(GRID),
        "cell_area_km2": CELL_AREA_KM2,
        "min_prediction_date": MIN_PREDICTION_DATE,
        "availability_status": availability["status"],  # "pending" | "ready" | "failed"
        "max_prediction_date": result["latest_valid_date"].isoformat() if result else None,
        "availability_error": availability["error"],
        # Presentation-only classification schemes (config/pm25_classification.py)
        # -- the frontend never defines thresholds itself.
        "pm25_classification_schemes": schemes_for_api(),
    })


@app.route("/api/select_cells", methods=["POST"])
@limits.rate_limited("read")
def api_select_cells():
    """Body: {"polygon": <GeoJSON geometry, EPSG:4326>}
    Returns cell count/area/ids -- NOT every cell's full geometry (could
    be tens of thousands of polygons for a large selection; the frontend
    only needs to draw the selection's own outline, which it already has
    from the user's drawn polygon, plus the summary numbers)."""
    body = request.get_json(force=True)
    if not body or "polygon" not in body:
        return jsonify({"error": "Missing 'polygon' in request body."}), 400

    try:
        polygon = shape(body["polygon"])
        clipped_polygon, was_clipped = clip_polygon_to_boundary(polygon, BOUNDARY_FOR_CLIPPING, polygon_crs=DISPLAY_CRS)
        selected = select_cells(clipped_polygon, GRID, polygon_crs=DISPLAY_CRS)
    except InvalidPolygonError as e:
        return jsonify({"error": str(e)}), 400

    area_km2 = compute_selection_area_km2(selected, CELL_AREA_KM2)
    return jsonify({
        "cell_count": len(selected),
        "cell_ids": selected["grid_id"].tolist(),
        "area_km2": area_km2,
        "area_outside_domain_excluded": was_clipped,
        "clipped_polygon": mapping(clipped_polygon),
    })


@app.route("/api/select_cells_grid", methods=["POST"])
@limits.rate_limited("read")
def api_select_cells_grid():
    """Body: {"polygon": <GeoJSON geometry, EPSG:4326>}
    Powers the OPTIONAL, toggle-controlled 1km cell visualization only --
    never called unless the user has switched that macro on.

    Returns each selected cell's ACTUAL 4 corners, reprojected properly
    from the grid's native CRS (ESRI:102028, Albers Equal Area Conic) to
    EPSG:4326 -- not an axis-aligned lat/lon rectangle approximated from
    the centroid. That earlier approximation was the direct cause of a
    real gap/overlap bug: a square in a conic projection does NOT map to
    an axis-aligned rectangle in lat/lon -- its edges pick up a slight
    rotation/slant that varies by location, so two cells that share an
    exact edge in the native CRS were being independently approximated
    with slightly different "flat" rectangles, leaving small gaps or
    overlaps between neighbors. Reprojecting the true shared corner
    points instead guarantees adjacent cells still share the exact same
    boundary after reprojection, since reprojection is a continuous,
    one-to-one function of position -- a shared point in the native CRS
    stays a shared point in EPSG:4326.

    Computed with vectorized operations, not a per-cell Python loop
    (benchmarked directly: ~1s total for a 62,191-cell selection,
    corner extraction via shapely.get_coordinates() rather than
    iterating .exterior.coords per polygon).

    Re-runs select_cells() on the same polygon (not a lookup against the
    ids returned by /api/select_cells) specifically so the visualized
    cells are always guaranteed identical to the ones that would
    actually be in a prediction request -- same function, same inputs,
    no separate code path that could drift out of sync."""
    body = request.get_json(force=True)
    if not body or "polygon" not in body:
        return jsonify({"error": "Missing 'polygon' in request body."}), 400

    try:
        polygon = shape(body["polygon"])
        clipped_polygon, _ = clip_polygon_to_boundary(polygon, BOUNDARY_FOR_CLIPPING, polygon_crs=DISPLAY_CRS)
        selected = select_cells(clipped_polygon, GRID, polygon_crs=DISPLAY_CRS)
    except InvalidPolygonError as e:
        return jsonify({"error": str(e)}), 400

    if len(selected) == 0:
        return jsonify({"cell_ids": [], "lons": [], "lats": []})

    # Exact cell bounds in the grid's own native CRS, built directly
    # from grid_x_index/grid_y_index (the same indices the official grid
    # itself is defined by -- see grid/grid_provider.py's provenance
    # notes), not derived from the cell's own (possibly coastal-clipped)
    # stored geometry, so every cell here is the full, precise 1000m
    # square its indices define. The construction itself now lives in
    # backend/cell_geometry.py (moved verbatim) so the prediction export
    # emits the identical polygons this route draws.
    lons_per_cell, lats_per_cell = cell_corners_wgs84(selected, GRID.crs)

    return jsonify({
        "cell_ids": selected["grid_id"].tolist(),
        "lons": lons_per_cell,
        "lats": lats_per_cell,
    })


@app.route("/api/validate_request", methods=["POST"])
@limits.rate_limited("read")
def api_validate_request():
    """Body: {"cell_count": int, "start_datetime": iso str, "end_datetime": iso str, "temporal_resolution": str}"""
    from datetime import datetime
    body = request.get_json(force=True)
    cell_count = body.get("cell_count", 0)
    start = datetime.fromisoformat(body["start_datetime"]) if body.get("start_datetime") else None
    end = datetime.fromisoformat(body["end_datetime"]) if body.get("end_datetime") else None
    resolution = body.get("temporal_resolution", "hourly")

    # validate_prediction_request expects a sized collection for
    # selected_cells -- a plain range() of the right length satisfies
    # `len(...)` without needing the full GeoDataFrame round-tripped here.
    fake_cells = range(cell_count)
    result = validate_prediction_request(fake_cells, start, end, resolution)
    return jsonify({"is_valid": result.is_valid, "message": result.message})


@app.route("/api/generate_prediction", methods=["POST"])
@limits.rate_limited("predict")
def api_generate_prediction():
    """Body: {"cell_ids": [...], "start_datetime": iso str, "end_datetime": iso str, "temporal_resolution": str}
    Starts a REAL prediction job in a background thread (see
    backend/prediction/job_state.py -- a minimal in-memory tracker, not a
    production job queue) and returns a job_id immediately. Poll
    /api/prediction_status/<job_id> for progress; the frontend does this
    automatically (see frontend/main.js)."""
    from datetime import datetime
    from backend.prediction import job_state, pipeline

    if LOADED_MODEL is None:
        return jsonify({
            "status": "error",
            "message": "The Baseline + Group D model failed to load at backend startup -- see the "
                        "backend console log for the specific reason. Fix the issue and restart the "
                        "backend before requesting a prediction.",
        }), 503

    body = request.get_json(force=True)
    cell_ids = body.get("cell_ids", [])
    if not cell_ids:
        return jsonify({"status": "error", "message": "No cell_ids provided."}), 400
    # Public deployment: bound the area one request may ask for. Inert locally.
    if limits.PUBLIC_MODE and len(cell_ids) > limits.MAX_CELLS_PER_REQUEST:
        return limits.too_many_cells_error(len(cell_ids))
    try:
        start_dt = datetime.fromisoformat(body["start_datetime"])
        end_dt = datetime.fromisoformat(body["end_datetime"])
    except (KeyError, ValueError) as e:
        return jsonify({"status": "error", "message": f"Invalid start_datetime/end_datetime: {e}"}), 400
    # BUG FIX: temporal_resolution was being read for the /api/validate_request
    # row-count estimate, but never actually passed through to the real
    # prediction job -- every request silently ran at hourly resolution
    # regardless of what the user selected.
    temporal_resolution = body.get("temporal_resolution", "hourly")

    from backend.prediction.data_availability import get_availability_state
    availability = get_availability_state()["result"]  # None if still pending or if the check failed

    # Global GEE circuit breaker, then a concurrency slot. The slot is held
    # for the lifetime of the BACKGROUND job (not the request), so it is
    # released in the wrapper's finally rather than by a context manager.
    if not limits.budget_available():
        return limits.budget_error()
    slot = limits.JobSlot()
    if not slot.acquire():
        return limits.busy_error()
    limits.consume_budget(1)

    def _run_and_release(*a, **kw):
        try:
            return pipeline.run_prediction_job(*a, **kw)
        finally:
            slot.release()

    job_id = job_state.create_job()
    job_state.run_job_in_background(
        job_id, _run_and_release, cell_ids, start_dt, end_dt, GRID, LOADED_MODEL, temporal_resolution, availability,
    )
    return jsonify({"status": "started", "job_id": job_id})


@app.route("/api/prediction_status/<job_id>")
@limits.rate_limited("poll")
def api_prediction_status(job_id):
    from backend.prediction import job_state
    job = job_state.get_job(job_id)
    if job is None:
        return jsonify({"error": f"No job found with id {job_id}."}), 404
    # log_lines can grow long over a real run; the frontend only needs
    # recent lines for the live console view, not the full history on
    # every poll -- trimmed here, not in job_state itself, so nothing is
    # actually discarded server-side.
    job["log_lines"] = job["log_lines"][-50:]
    return jsonify(job)


@app.route("/api/prediction_export/<job_id>")
@limits.rate_limited("read")
def api_prediction_export(job_id):
    """GET ?format=csv|geojson[&timestamp=<one of the result's timestamps>]
    Streams a COMPLETED job's already-retained result as a file (see
    backend/export.py). Never re-runs the prediction. The optional
    timestamp filter (GeoJSON only) exports a single frame, so GIS users
    don't get the same polygon repeated once per timestamp."""
    from backend.prediction import job_state
    from backend import export as export_mod

    job = job_state.get_job(job_id)
    if job is None:
        return jsonify({"error": f"No job found with id {job_id} (it may have been evicted -- re-run the prediction)."}), 404
    if job["status"] != "complete":
        return jsonify({"error": f"Job {job_id} is not complete (status: {job['status']})."}), 409

    fmt = (request.args.get("format") or "csv").lower()
    if fmt not in ("csv", "geojson"):
        return jsonify({"error": "format must be 'csv' or 'geojson'."}), 400
    timestamp = request.args.get("timestamp")
    result = job["result"]
    if timestamp is not None:
        if fmt != "geojson":
            return jsonify({"error": "timestamp filtering is only supported for format=geojson."}), 400
        if timestamp not in set(result["timestamps"]):
            return jsonify({"error": f"timestamp {timestamp!r} is not one of this job's result timestamps."}), 400

    selected = GRID[GRID["grid_id"].isin(result["cell_ids"])]
    cells_meta = export_mod.build_cells_meta(selected, GRID.crs)
    filename = export_mod.build_filename(result, job_id, fmt)

    if fmt == "csv":
        gen = export_mod.iter_csv(result, cells_meta)
        mimetype = "text/csv"  # Flask appends charset=utf-8 itself for text/* types
    else:
        metadata = export_mod.build_metadata(result, job_id, model_info=ENSEMBLE_CONFIG_INFO)
        gen = export_mod.iter_geojson(result, cells_meta, metadata, timestamp=timestamp)
        mimetype = "application/geo+json"

    return Response(
        stream_with_context(gen), mimetype=mimetype,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.route("/api/prediction_explanation/<job_id>")
@limits.rate_limited("read")
def api_prediction_explanation(job_id):
    """GET ?grid_id=<cell>&datetime_utc=<one of the job's HOURLY feature-row timestamps>
    On-demand SHAP explanation of ONE hourly feature row, computed from the
    rows retained at job completion (backend/prediction/explanation_store.py)
    with the startup TreeExplainer on the Philippine-only XGBoost member
    (backend/prediction/explainer.py). Never recomputes features or
    predictions. For daily results the frontend asks for one underlying
    hour at a time -- SHAP values are never averaged across a day here.
    Also returns the job's retained hourly timestamps so the frontend can
    offer an hour picker."""
    from backend.prediction import job_state, explanation_store
    from backend.prediction.explainer import build_explanation_payload
    from backend.prediction.feature_labels import display_name

    if EXPLAINER is None:
        return jsonify({"error": "SHAP explanations are unavailable in this backend session (the explainer "
                                 "could not be built at startup -- see the backend console). Predictions are unaffected."}), 503
    job = job_state.get_job(job_id)
    if job is None:
        return jsonify({"error": f"No job found with id {job_id} (it may have been evicted -- re-run the prediction)."}), 404
    if job["status"] != "complete":
        return jsonify({"error": f"Job {job_id} is not complete (status: {job['status']})."}), 409
    if not explanation_store.has_job(job_id):
        return jsonify({"error": "Explanation data was not retained for this job (retention disabled, the job exceeded "
                                 "the retention budget, or it was evicted by newer jobs). Predictions are unaffected; "
                                 "re-run a smaller request to get explanations."}), 404

    grid_id = request.args.get("grid_id")
    datetime_utc = request.args.get("datetime_utc")
    if not grid_id or not datetime_utc:
        return jsonify({"error": "grid_id and datetime_utc query parameters are required.",
                        "hourly_timestamps": explanation_store.hourly_timestamps(job_id)}), 400

    x_row, info = explanation_store.get_row(job_id, grid_id, datetime_utc)
    if x_row is None:
        return jsonify({"error": info, "hourly_timestamps": explanation_store.hourly_timestamps(job_id)}), 404
    feature_order = info

    # The displayed (ensemble) value for the same row, if this job's result
    # is hourly -- reported alongside so the explained member's own
    # prediction is never mistaken for the map value. Daily results have no
    # single hourly ensemble row in the result, so this is None there.
    ensemble_val = None
    result = job["result"]
    if result.get("temporal_resolution", "hourly") == "hourly":
        from backend.export import parse_result_timestamp
        want = parse_result_timestamp(datetime_utc)
        for rec in result["predictions"]:
            if rec["grid_id"] == grid_id and parse_result_timestamp(rec["datetime_utc"]) == want:
                v = rec["pm25_prediction"]
                ensemble_val = None if v is None or v != v else float(v)
                break

    try:
        payload = build_explanation_payload(
            EXPLAINER, x_row, feature_order, display_name, LOADED_MODEL["log_clip_bounds"],
            grid_id, datetime_utc, ensemble_prediction_ugm3=ensemble_val,
        )
    except Exception as e:  # noqa: BLE001 -- surface as a clean error, never a 500 traceback
        return jsonify({"error": f"Explanation failed: {type(e).__name__}: {e}"}), 500
    payload["temporal_resolution"] = result.get("temporal_resolution", "hourly")
    payload["hourly_timestamps"] = explanation_store.hourly_timestamps(job_id)
    return jsonify(payload)


@app.route("/api/mobile/prediction", methods=["POST"])
@limits.rate_limited("mobile")
def api_mobile_prediction():
    """Body: {"lat": <float>, "lon": <float>} -- the mobile app's single
    call (SRS 4.3). Resolves the 1 km cell, picks the strict T*, runs the
    existing pipeline core (batched retrieval) unless the (cell, T*)
    reading is cached, and returns value + EPA classification/guidance +
    top-3 SHAP factors + data-latency metadata. See backend/mobile.py.
    Coordinates are never logged."""
    if MOBILE_SERVICE is None:
        return jsonify({"status": "error", "error_code": "model_unavailable",
                        "message": "The prediction model failed to load at backend startup."}), 503
    body = request.get_json(force=True, silent=True)

    # Public deployment: a cached (cell, T*) reading costs nothing, so only
    # genuine compute is charged against the global daily GEE budget, and a
    # concurrency slot is held for the duration of that compute. Checking the
    # budget up front means a walk across the 62,000-cell grid is stopped,
    # while repeat visitors in an already-computed cell are never blocked.
    if not limits.budget_available():
        return limits.budget_error()
    with limits.JobSlot() as slot:
        if not slot.acquired:
            return limits.busy_error()
        payload, status = MOBILE_SERVICE.handle(body)
    if status == 200 and not payload.get("from_cache"):
        limits.consume_budget(1)
    return jsonify(payload), status


@app.route("/api/cancel_prediction/<job_id>", methods=["POST"])
@limits.rate_limited("poll")
def api_cancel_prediction(job_id):
    from backend.prediction import job_state
    cancelled = job_state.request_cancel(job_id)
    if not cancelled:
        return jsonify({"status": "error", "message": f"Job {job_id} not found or already finished."}), 404
    return jsonify({"status": "cancel_requested"})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5050, debug=False)
