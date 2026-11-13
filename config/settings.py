"""
config/settings.py -- Central configuration for the PM2.5 area-selection prototype.

All CRS, path, and timezone decisions live here so they are declared
once and never hardcoded elsewhere in the application.
"""

import os
from pathlib import Path
from zoneinfo import ZoneInfo

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# ============================================================
# Data sources
# ============================================================
PHILIPPINES_BOUNDARY_PATH = DATA_DIR / "philippines-adm0.gpkg"

# This IS the official research grid (training_grid.gpkg from the
# research project), extended to cover the FULL Philippine land area
# using the exact same grid definition (CRS, cell size, origin) --
# confirmed directly against the research pipeline's own grid-building
# script's documented design intent. See grid/grid_provider.py and
# build_full_grid.py for the full provenance statement. All 915 original
# training cells are reused with byte-identical geometry and grid_id;
# see build_full_grid.py's own output log for the verification.
OFFICIAL_GRID_PATH = DATA_DIR / "official_grid_ph_full.gpkg"

# The original 915-cell training-only subset, kept for reference/reuse
# by build_full_grid.py -- not used directly by the running app.
TRAINING_GRID_PATH = DATA_DIR / "official_grid_ph.gpkg"

# ============================================================
# CRS strategy (documented here; see README for full rationale)
# ============================================================
# Display CRS: standard geographic lat/lon, required by folium/Leaflet
# web maps and by the OpenStreetMap basemap.
DISPLAY_CRS = "EPSG:4326"

# Analysis CRS: the grid's own native CRS (Asia South Albers Equal Area
# Conic, ESRI:102028) -- a projected, equal-area CRS in meters. ALL
# intersection tests and area calculations happen in this CRS, never in
# raw lat/lon degrees, which would distort both grid alignment and area
# math. Read directly from the grid file itself at load time rather than
# hardcoded twice, so the two can never drift out of sync.
ANALYSIS_CRS = None  # populated at runtime from the grid file's own .crs

CELL_SIDE_METERS = 1000.0  # confirmed via direct inspection of official_grid_ph.gpkg
CELL_AREA_KM2 = (CELL_SIDE_METERS / 1000.0) ** 2  # = 1.0 km^2 per cell, nominal

# ============================================================
# Timezone
# ============================================================
# All user-facing date/time controls and displays use Philippine local
# time, matching the research project's own established convention.
# UTC is never the primary user-facing representation.
LOCAL_TIMEZONE_NAME = "Asia/Manila"
LOCAL_TIMEZONE = ZoneInfo(LOCAL_TIMEZONE_NAME)

# ============================================================
# Future backend limits (NOT enforced yet -- see selection/request_validator.py)
# ============================================================
# Deliberately left as a named placeholder so a real limit can be
# introduced later without restructuring any calling code.
MAX_PREDICTION_ROWS = None  # e.g. 500_000 once the backend defines the actual limit

# For THIS diagnostic-testing stage only: a deliberately small, hardcoded
# ceiling so a mistaken large request cannot accidentally trigger a huge
# number of live GEE calls before MAX_PREDICTION_ROWS above has been set
# from real stress testing. This is NOT a production limit -- see the
# deployment plan, Section 5 -- it exists purely so "the first small test"
# can't accidentally become "the first huge, expensive test."
DIAGNOSTIC_TEST_MAX_ROWS = 200000  # e.g. 5 cells x 24 hours = 120, comfortably under this

# ============================================================
# Model artifacts (Baseline + Group D deployment model)
# ============================================================
# Produced by philippine_only_full_evaluation.py (Stage A) in the research
# environment: the Philippine-only ensemble, trained from scratch on the
# finetune_train cells with no transfer learning -- the backend NEVER trains
# or fine-tunes anything itself. The superseded fine-tuned artifacts are kept
# in data/models/baseline_group_d/ for rollback and thesis comparison; the
# 157-feature Baseline + Group D configuration is unchanged between the two.
# See backend/prediction/model_loader.py,
# which validates every one of these files exists and is schema-consistent
# at Flask startup, before any request is served.
MODEL_DIR = DATA_DIR / "models" / "philippine_only"
MODEL_XGBOOST_PATH  = MODEL_DIR / "ph_only_xgboost.joblib"
MODEL_LIGHTGBM_PATH = MODEL_DIR / "ph_only_lightgbm.joblib"
MODEL_CATBOOST_PATH = MODEL_DIR / "ph_only_catboost.cbm"
MODEL_ENSEMBLE_CONFIG_PATH = MODEL_DIR / "ensemble_config.json"
MODEL_FEATURE_SCHEMA_PATH = MODEL_DIR / "feature_schema.json"
MODEL_PREPROCESSING_CONFIG_PATH = MODEL_DIR / "preprocessing_config.json"
MODEL_LANDCOVER_SCHEME_PATH = MODEL_DIR / "landcover_encoding_scheme.txt"

EXPECTED_FEATURE_COUNT = 157  # Baseline (152) + Group D (5) -- see the deployment plan, Section 1.4
EXPECTED_BASELINE_FEATURE_COUNT = 152
EXPECTED_GROUP_D_FEATURE_COUNT = 5

# ============================================================
# SHAP explanation support (application layer -- see
# backend/prediction/explanation_store.py and explainer.py)
# ============================================================
# Global, process-wide memory budget for retaining completed jobs' feature
# rows so a single (cell, hour) row can later be explained on demand.
# Rows are kept as float32 (~636 bytes/row incl. index). Oldest jobs are
# evicted first when a new job would exceed the budget; a single job
# larger than the whole budget is simply not retained (its prediction is
# unaffected -- only its explanation becomes unavailable). 0 disables
# retention entirely. Prediction never depends on this.
SHAP_FEATURE_RETENTION_MAX_MB = 64

# ============================================================
# Mobile endpoint (application layer -- see backend/mobile.py)
# ============================================================
# Completed mobile responses are cached per (grid cell, T*) so repeat
# requests for the same location are served without GEE/model work while
# T* is unchanged; entries for an older T* become unreachable once T*
# advances and are evicted by this bound (LRU).
MOBILE_CACHE_MAX_ENTRIES = 5000
# The live GEE availability check (startup, background) is re-run on this
# interval so a newer T* is discovered without restarting Flask. Each run
# is ~8 bounded GEE queries (a few seconds).
AVAILABILITY_REFRESH_INTERVAL_SECONDS = 3600

# ============================================================
# Research pipeline location (for direct reuse of validated feature-
# construction logic -- see backend/prediction/feature_builder.py)
# ============================================================
# The backend does NOT reimplement rolling-average windows, physics/
# transport formulas, calendar features, or land-cover one-hot encoding
# from memory -- per explicit instruction, it imports and calls the
# actual, validated functions from the research pipeline's own modules
# (prepare_features.py, run_physics_transport_experiment.py,
# build_additional_features.py, advanced_feature_engineering_experiment.py,
# extract_era5_hourly_batched.py). Those modules are NOT copied into this
# project -- set this to the directory on your machine where your research
# pipeline scripts actually live (the same directory
# the model-training scripts were run from). The backend adds this to
# sys.path at startup and fails clearly if the required modules aren't
# importable from it -- see backend/prediction/model_loader.py.
#
# Defaults to the verbatim copies vendored at backend/research_pipeline/
# (see that folder's README) so a fresh clone deploys standalone. Override
# with PM25_RESEARCH_PIPELINE_DIR to point at the original research project
# instead -- model_loader.py still fails loudly if the modules are missing.
RESEARCH_PIPELINE_DIR = Path(os.environ.get(
    "PM25_RESEARCH_PIPELINE_DIR", str(PROJECT_ROOT / "backend" / "research_pipeline")))

# ============================================================
# Local datasets (see data/local_datasets/*/README.txt for exact
# placement and naming instructions)
# ============================================================
LOCAL_DATASETS_DIR = DATA_DIR / "local_datasets"
WORLDPOP_DIR = LOCAL_DATASETS_DIR / "worldpop"
COASTLINE_PATH = LOCAL_DATASETS_DIR / "coastline" / "ne_10m_coastline.gpkg"
OSM_PBF_PATH = LOCAL_DATASETS_DIR / "osm" / "philippines-latest.osm.pbf"
WORLDPOP_YEAR_MIN = 2015
WORLDPOP_YEAR_MAX = 2030

# ============================================================
# Prediction-date availability policy (see the deployment plan, Section 3)
# ============================================================
# Provisional values, exactly as flagged in the deployment plan -- NOT
# re-verified against live GEE as part of this integration-testing stage
# (explicitly out of scope for this task). Kept conservative on purpose:
# it is safer for a first diagnostic test to reject a borderline-recent
# date than to silently attempt a prediction with data that may not
# actually exist yet.
MIN_PREDICTION_DATE = "2020-01-01"          # model-training-coverage floor (provisional, see plan Section 3.1.1)
MIN_DATA_AVAILABILITY_DATE = "2018-07-11"   # raw data-availability floor (Sentinel-5P + 7-day rolling window)
# PREDICTION_UNAVAILABLE_ZONE_DAYS removed -- confirmed wrong: it assumed
# ERA5T (a faster-latency preliminary product), but this pipeline's
# ERA5/ERA5-Land collections have no ERA5T variant on GEE and are only
# available up to roughly 3 months behind real time. The latest valid
# prediction date is now determined by a live GEE availability check at
# startup -- see backend/prediction/data_availability.py -- rather than
# a hardcoded day count.
PREDICTION_PRELIMINARY_ZONE_DAYS = 100      # today - N days is flagged "preliminary" (provisional, ERA5-Land finalization-derived)

# ============================================================
# Google Earth Engine
# ============================================================
# Authentication is NOT configured here -- it uses whatever `ee.Initialize()`
# picks up from the environment (application-default credentials or a service
# account, per your own GEE setup). See DEPLOYMENT.md for the server setup.
#
# The project id is read from the environment so it is never committed: it
# identifies a billable Google Cloud project, and this repository is public.
# For local development put it in config/local_settings.py (gitignored) --
# see config/local_settings.example.py.
GEE_PROJECT_ID = os.environ.get("PM25_GEE_PROJECT_ID", "")

# ============================================================
# Map defaults
# ============================================================
PHILIPPINES_CENTER = [12.8797, 121.7740]  # approximate geographic center, for initial map view only
DEFAULT_ZOOM = 6
GRID_VISIBLE_MIN_ZOOM = 9  # below this zoom, the grid is too dense to render usefully -- see map/map_renderer.py


# ============================================================
# Optional machine-local overrides -- config/local_settings.py
# ============================================================
# Gitignored. Anything defined there wins over the values above, so a
# developer can keep GEE_PROJECT_ID (and any other local path) out of the
# repository without exporting environment variables on every run.
# See config/local_settings.example.py.
try:
    from config.local_settings import *  # noqa: F401,F403
except ImportError:
    pass
