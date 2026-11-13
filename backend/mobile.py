"""
backend/mobile.py -- The mobile application's single prediction request
(SRS section 4.3): latitude/longitude in, one JSON reading out.

Application layer only. It reuses, unchanged:
  - the official grid definition (ESRI:102028, 1 km, floor-division
    indices -- grid/grid_provider.py, build_full_grid.py),
  - the live availability state (backend/prediction/data_availability.py),
  - the full prediction pipeline core (pipeline.run_prediction_core ->
    static/dynamic predictors -> research feature engineering ->
    inference), with the approved BATCHED GEE retrieval injected
    (dynamic_predictors_batched.py; equivalence-tested against the
    per-timestamp functions),
  - the SHAP explainer and retained feature rows (explainer.py,
    explanation_store.py),
  - the classification schemes (config/pm25_classification.py).

Temporal semantics (approved for this version): the prediction is for
T* = the latest hour the web app itself would accept -- 23:00 Philippine
time on the availability check's `latest_valid_date` (the latest local
calendar day fully covered by EVERY required dynamic collection). No
research ±5-day / ±3-hour fallback and no historical averaging: the same
strict same-timestamp retrieval as the web app. T* selection is behind
TStarStrategy so an approved fallback rule can be added later without
touching the request flow.

Privacy: coordinates are used only to resolve the grid cell and are
never logged or stored; only the 1 km cell id appears in logs.
"""

import math
import threading
import time
from collections import OrderedDict
from datetime import datetime, timezone

import numpy as np
from pyproj import Transformer

import config.settings as settings
from config.settings import CELL_SIDE_METERS, DISPLAY_CRS, LOCAL_TIMEZONE
from config.pm25_classification import get_scheme, classify, MOBILE_CLASSIFICATION_SCHEME_ID

MOBILE_TOP_FACTORS = 3

DATA_NOTE = ("Modelled from satellite and weather data that may lag real time by several days. "
             "This is an estimate, not a live measurement.")
EXPLANATION_NOTE = ("These factors describe what pushed the model's estimate higher or lower (model attribution), "
                    "not what caused air pollution.")


class MobileRequestError(Exception):
    """A clean, user-facing failure with an HTTP status and a stable error code."""

    def __init__(self, status, code, message):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


# ============================================================
# Point -> grid cell
# ============================================================

class CellIndex:
    """Compact (grid_x_index, grid_y_index) -> grid_id lookup for the whole
    grid (313,845 cells): sorted int64 keys + searchsorted, ~5 MB, built
    once at startup. Uses the grid's own indices, so the 915 original
    training cells (ids like PH000000828) resolve exactly like the
    PHFULL_* cells."""

    _OFFSET = 10 ** 6
    _MULT = 10 ** 7

    def __init__(self, grid_gdf):
        x = grid_gdf["grid_x_index"].to_numpy(dtype=np.int64)
        y = grid_gdf["grid_y_index"].to_numpy(dtype=np.int64)
        keys = (x + self._OFFSET) * self._MULT + (y + self._OFFSET)
        order = np.argsort(keys)
        self._keys = keys[order]
        self._ids = grid_gdf["grid_id"].to_numpy()[order]
        self.crs = grid_gdf.crs
        self._transformer = Transformer.from_crs(DISPLAY_CRS, self.crs, always_xy=True)

    def indices_for(self, lat, lon):
        x, y = self._transformer.transform(lon, lat)
        if not (math.isfinite(x) and math.isfinite(y)):
            return None
        return int(math.floor(x / CELL_SIDE_METERS)), int(math.floor(y / CELL_SIDE_METERS))

    def lookup(self, gx, gy):
        key = (np.int64(gx) + self._OFFSET) * self._MULT + (np.int64(gy) + self._OFFSET)
        pos = np.searchsorted(self._keys, key)
        if pos < len(self._keys) and self._keys[pos] == key:
            return str(self._ids[pos])
        return None

    def cell_for_point(self, lat, lon):
        idx = self.indices_for(lat, lon)
        return None if idx is None else self.lookup(*idx)


def validate_coordinates(body):
    if not isinstance(body, dict):
        raise MobileRequestError(400, "invalid_request", "Request body must be a JSON object with 'lat' and 'lon'.")
    try:
        lat = float(body.get("lat"))
        lon = float(body.get("lon"))
    except (TypeError, ValueError):
        raise MobileRequestError(400, "invalid_coordinates", "'lat' and 'lon' must be numbers.")
    if not (math.isfinite(lat) and math.isfinite(lon)) or not (-90.0 <= lat <= 90.0) or not (-180.0 <= lon <= 180.0):
        raise MobileRequestError(400, "invalid_coordinates", "'lat' must be within [-90, 90] and 'lon' within [-180, 180].")
    return lat, lon


# ============================================================
# T* selection (strict rule; strategy object so a fallback rule can be
# added later behind explicit approval)
# ============================================================

class StrictTStarStrategy:
    """T* = 23:00 Philippine time on availability['latest_valid_date'] --
    exactly the latest end timestamp the web app's own validation accepts
    (pipeline.py: end_datetime.date() <= latest_valid_date), so mobile and
    web share one definition of 'latest supported'."""

    name = "strict_same_timestamp"

    def select(self, availability_result):
        if not availability_result:
            return None
        d = availability_result["latest_valid_date"]
        return datetime(d.year, d.month, d.day, 23, 0, tzinfo=LOCAL_TIMEZONE)


# ============================================================
# Bounded cache + in-flight de-duplication
# ============================================================

class PredictionCache:
    """LRU keyed by (grid_id, tstar_utc_iso). Entries for an older T*
    simply stop being requested once T* advances and age out of the
    bound. compute_or_wait() guarantees ONE computation per key even
    under concurrent identical requests: the first caller computes, the
    others wait on its Event and then read the cached result."""

    def __init__(self, max_entries):
        self.max_entries = max_entries
        self._entries = OrderedDict()
        self._inflight = {}   # key -> {"event": Event, "error": Exception|None}
        self._lock = threading.Lock()
        self.stats = {"hits": 0, "misses": 0, "waits": 0, "computations": 0}

    def get(self, key):
        with self._lock:
            if key in self._entries:
                self._entries.move_to_end(key)
                self.stats["hits"] += 1
                return self._entries[key]
            return None

    def put(self, key, value):
        with self._lock:
            self._entries[key] = value
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)

    def compute_or_wait(self, key, compute_fn):
        cached = self.get(key)
        if cached is not None:
            return cached, True
        with self._lock:
            if key in self._entries:  # filled between the two checks
                self.stats["hits"] += 1
                return self._entries[key], True
            slot = self._inflight.get(key)
            if slot is None:
                slot = {"event": threading.Event(), "error": None}
                self._inflight[key] = slot
                owner = True
                self.stats["misses"] += 1
                self.stats["computations"] += 1
            else:
                owner = False
                self.stats["waits"] += 1
        if not owner:
            slot["event"].wait()
            if slot["error"] is not None:
                raise slot["error"]
            value = self.get(key)
            if value is None:
                raise MobileRequestError(500, "prediction_failed", "Shared computation produced no result.")
            return value, True
        try:
            value = compute_fn()
            self.put(key, value)
            return value, False
        except Exception as e:
            slot["error"] = e
            raise
        finally:
            with self._lock:
                self._inflight.pop(key, None)
            slot["event"].set()

    def __len__(self):
        with self._lock:
            return len(self._entries)


# ============================================================
# The service
# ============================================================

class MobileService:
    """Wires the pieces together. Everything expensive is injected so the
    request flow is unit-testable without GEE or the model:
      get_availability: () -> availability state dict (status/result/error)
      compute_prediction: (grid_id, tstar_local) -> {"pm25": float|None,
                          "retain_key": str, "n_rows": int, "log": [str]}
      explain_row: (retain_key, grid_id, tstar_utc_iso) -> explanation
                          payload dict or None
    """

    def __init__(self, cell_index, get_availability, compute_prediction, explain_row,
                 tstar_strategy=None, cache_max_entries=None, scheme_id=MOBILE_CLASSIFICATION_SCHEME_ID):
        self.cell_index = cell_index
        self.get_availability = get_availability
        self.compute_prediction = compute_prediction
        self.explain_row = explain_row
        self.tstar_strategy = tstar_strategy or StrictTStarStrategy()
        self.cache = PredictionCache(cache_max_entries or getattr(settings, "MOBILE_CACHE_MAX_ENTRIES", 5000))
        self.scheme = get_scheme(scheme_id)
        if self.scheme is None:
            raise ValueError(f"Unknown classification scheme id {scheme_id!r}")

    # ---- request flow ----
    def handle(self, body):
        """Returns (payload, http_status). Raises nothing to the caller;
        every failure becomes a structured error payload."""
        t0 = time.time()
        try:
            lat, lon = validate_coordinates(body)
            grid_id = self.cell_index.cell_for_point(lat, lon)
            if grid_id is None:
                raise MobileRequestError(404, "outside_coverage",
                                         "This location is outside the supported prediction grid (Philippine land area).")
            tstar_local = self.select_tstar()
            tstar_utc = tstar_local.astimezone(timezone.utc)
            key = (grid_id, tstar_utc.isoformat())

            def _compute():
                res = self.compute_prediction(grid_id, tstar_local)
                explanation = None
                if res.get("retain_key"):
                    try:
                        explanation = self.explain_row(res["retain_key"], grid_id, tstar_utc.isoformat())
                    except Exception as e:  # noqa: BLE001 -- explanation is optional
                        explanation = {"error": f"{type(e).__name__}: {e}"}
                return self.build_reading(grid_id, tstar_local, res, explanation)

            reading, from_cache = self.cache.compute_or_wait(key, _compute)
            payload = dict(reading)
            payload["from_cache"] = from_cache
            payload["request_seconds"] = round(time.time() - t0, 2)
            payload["staleness_hours"] = round((datetime.now(timezone.utc) - tstar_utc).total_seconds() / 3600, 1)
            return payload, 200
        except MobileRequestError as e:
            return {"status": "error", "error_code": e.code, "message": e.message}, e.status
        except Exception as e:  # noqa: BLE001 -- never a raw 500 traceback to the phone
            return {"status": "error", "error_code": "prediction_failed",
                    "message": f"Prediction failed: {type(e).__name__}: {e}"}, 500

    def select_tstar(self):
        state = self.get_availability()
        if state.get("status") == "pending":
            raise MobileRequestError(503, "availability_pending",
                                     "The backend is still determining which dates have complete environmental data. Try again shortly.")
        if state.get("status") == "failed" or not state.get("result"):
            raise MobileRequestError(503, "availability_failed",
                                     "Environmental-data availability could not be determined (Earth Engine unavailable). Try again later.")
        tstar = self.tstar_strategy.select(state["result"])
        if tstar is None:
            raise MobileRequestError(503, "no_supported_timestamp", "No timestamp with complete environmental data is available.")
        return tstar

    # ---- response assembly ----
    def build_reading(self, grid_id, tstar_local, res, explanation):
        value = res.get("pm25")
        value = None if value is None or (isinstance(value, float) and math.isnan(value)) else float(value)
        tstar_utc = tstar_local.astimezone(timezone.utc)
        cat = classify(value, self.scheme) if value is not None else None
        classification = None
        if cat is not None:
            classification = {
                "scheme_id": self.scheme["id"],
                "scheme_name": self.scheme["name"],
                "category": cat["label"],
                "color": cat["color"],
                "interpretation": cat["interpretation"],
                "guidance": cat.get("guidance", ""),
                "basis_note": self.scheme["basis_notes"]["hourly"],
            }
        top_factors, explanation_available, explanation_error, n_missing = [], False, None, None
        if explanation and "contributions" in explanation:
            explanation_available = True
            n_missing = explanation.get("n_missing_features")
            for c in explanation["contributions"][:MOBILE_TOP_FACTORS]:
                top_factors.append({
                    "feature": c["feature"],
                    "label": c["display_name"],
                    "direction": "higher" if c["shap_log"] > 0 else "lower",
                    "shap_log": round(float(c["shap_log"]), 4),
                })
        elif explanation and "error" in explanation:
            explanation_error = explanation["error"]

        return {
            "status": "ok" if value is not None else "unavailable",
            "cell_id": grid_id,
            "as_of_utc": tstar_utc.isoformat(),
            "as_of_local": tstar_local.isoformat(),
            "as_of_local_label": tstar_local.strftime("%b %d, %Y, %I:%M %p").replace(" 0", " ") + " PHT",
            "timestamp_strategy": self.tstar_strategy.name,
            "pm25_ugm3": None if value is None else round(value, 1),
            "units": "µg/m³",
            "classification": classification,
            "data_note": DATA_NOTE,
            "n_missing_features": n_missing,
            "n_features": explanation.get("n_features") if explanation else None,
            "explanation": {
                "available": explanation_available,
                "top_factors": top_factors,
                "note": EXPLANATION_NOTE,
                "error": explanation_error,
            },
            "computed_at_utc": datetime.now(timezone.utc).isoformat(),
            "compute_seconds": res.get("compute_seconds"),
            "message": None if value is not None else "The model could not produce a value for this location and time.",
        }


# ============================================================
# Default wiring for the running app (real pipeline)
# ============================================================

def make_compute_prediction(grid_gdf, loaded_model):
    """Runs the EXISTING pipeline core for one cell at [T*, T*] with the
    batched retrieval injected, retaining the feature rows for SHAP."""
    from backend.prediction import pipeline, dynamic_predictors_batched as dpb

    def _compute(grid_id, tstar_local):
        from backend.prediction.data_availability import get_availability_state
        availability = get_availability_state()["result"]
        retain_key = f"mobile:{grid_id}:{tstar_local.astimezone(timezone.utc).isoformat()}"
        lines = []
        t0 = time.time()
        result = pipeline.run_prediction_core(
            [grid_id], tstar_local, tstar_local, grid_gdf, loaded_model,
            temporal_resolution="hourly", availability=availability,
            log_fn=lambda m: lines.append(m) or print(f"    [mobile] {m}"),
            retain_key=retain_key,
            fetch_daily_fn=dpb.fetch_daily_predictors_batched,
            fetch_hourly_fn=dpb.fetch_hourly_predictors_batched,
        )
        preds = result["predictions"]
        value = preds[0]["pm25_prediction"] if preds else None
        return {"pm25": value, "retain_key": retain_key, "n_rows": len(preds),
                "compute_seconds": round(time.time() - t0, 1), "log": lines}

    return _compute


def make_explain_row(explainer, loaded_model):
    from backend.prediction import explanation_store
    from backend.prediction.explainer import build_explanation_payload
    from backend.prediction.feature_labels import display_name

    def _explain(retain_key, grid_id, tstar_utc_iso):
        if explainer is None:
            return None
        row, info = explanation_store.get_row(retain_key, grid_id, tstar_utc_iso)
        if row is None:
            return {"error": info}
        return build_explanation_payload(explainer, row, info, display_name, loaded_model["log_clip_bounds"],
                                         grid_id, tstar_utc_iso)

    return _explain
