# First Real Prediction Test -- Setup and Run Guide

This covers what changed for this integration-testing stage: real model
artifacts, real local datasets, real GEE calls, real inference. It does
NOT replace README.md (the original area-selection prototype guide) --
read this alongside it.

**Honesty check before you start**: this backend has been verified in
pieces (model loading and inference were tested end-to-end with real
fitted models; the job-tracking/cancellation logic was tested end-to-end
for real), but the GEE-calling code and the feature-construction glue
code have only been verified by direct code inspection, not by an actual
run -- there is no GEE access in the environment this was built in. Your
first real run is the actual test of those two pieces. Send me the
console log from that run and we'll fix whatever it surfaces.

---

## 1. Where things go

### Model artifacts -- `data/models/philippine_only/`

The deployed ensemble is the Philippine-only, no-transfer model produced by
`philippine_only_full_evaluation.py` (Stage A). Copy its three model files
here, alongside the three config files (which are unchanged from the earlier
fine-tuned set -- the 157-feature schema is identical):

```
data/models/philippine_only/
    ph_only_xgboost.joblib
    ph_only_lightgbm.joblib
    ph_only_catboost.cbm
    ensemble_config.json            (weights = that run's Philippine-only weights)
    feature_schema.json
    preprocessing_config.json
    landcover_encoding_scheme.txt
```

Do not rename these files -- `config/settings.py` references these exact
names. The superseded fine-tuned artifacts stay in
`data/models/baseline_group_d/` for rollback and thesis comparison; switching
back is a four-line change in `config/settings.py`.

### Local datasets -- `data/local_datasets/`

Already-created folders, each with its own README:

```
data/local_datasets/worldpop/phl_pop_<YEAR>_CN_1km_R2025A_UA_v1.tif   (2015-2030, as available)
data/local_datasets/coastline/ne_10m_coastline.gpkg
data/local_datasets/osm/philippines-latest.osm.pbf
```

### Road density cache -- `data/static_layers/road_density_by_cell.csv`

**Required, not optional.** See `data/static_layers/README.txt` for exactly
why and how to generate it using your own existing road-density
extraction script. A partial cache covering just your test cells is
enough to get started.

---

## 2. Python environment

```bash
pip install flask geopandas shapely rasterstats joblib xgboost lightgbm catboost earthengine-api pandas numpy scipy
```

Plus whatever your research pipeline scripts themselves need (pandas,
numpy, scipy, shap, matplotlib -- already confirmed installed if the
model-training scripts ran successfully for you).

---

## 3. Configure `config/settings.py`

Three settings MUST be set before starting the backend -- there are no
safe defaults, by design:

```python
RESEARCH_PIPELINE_DIR = Path(r"C:\path\to\air_quality_thesis_project\OpenAQ_API")
GEE_PROJECT_ID = "your-gcp-project-id"   # see Earth Engine setup below
```

`RESEARCH_PIPELINE_DIR` must point at the SAME directory the
model-training scripts were run from -- the backend directly
imports and calls functions from `prepare_features.py`,
`run_physics_transport_experiment.py`, `build_additional_features.py`,
`advanced_feature_engineering_experiment.py`, and
`extract_era5_hourly_batched.py` in that directory. It does not
reimplement their logic.

`MIN_PREDICTION_DATE` is already set to the provisional value from the
deployment plan -- leave it as-is for now. The previous
`PREDICTION_UNAVAILABLE_ZONE_DAYS` constant has been removed: the latest
valid prediction date is now determined by a live GEE availability check
at startup (see backend/prediction/data_availability.py) rather than a
fixed day count -- watch the startup console output (Section 5 below)
for the calculated value.

---

## 4. Earth Engine authentication

The backend calls `ee.Initialize()` (or `ee.Initialize(project=...)` if
`GEE_PROJECT_ID` is set) using whatever credentials are already set up in
your own environment. If you haven't authenticated in this environment
before:

```bash
earthengine authenticate
```

This opens a browser flow and stores credentials locally. If you're using
a Google Cloud project with the Earth Engine API enabled (most accounts
now require this), set `GEE_PROJECT_ID` in `config/settings.py` to that
project's ID.

If GEE initialization fails, the backend will still start (so you can
still test area selection), but every prediction request will fail with a
clear message until this is fixed and the backend is restarted -- check
the console log for the specific error.

---

## 5. Start the backend

```bash
cd pm25_web_app
python run_app.py
```

or directly:

```bash
python backend/app.py
```

Watch the startup console output carefully -- it will tell you
immediately if GEE initialization or model loading failed, and exactly
why. It will also show the live dynamic-data availability check: the
latest available timestamp detected for each dataset (MODIS AOD, the 5
Sentinel-5P gases, ERA5-Land, ERA5), which one is the binding constraint,
and the resulting latest valid prediction date. If all three succeed,
you'll see:

```
Model loaded successfully. Ready to serve prediction requests.
```

## 6. Start the frontend

Nothing separate to start -- Flask serves the frontend directly. Open:

```
http://127.0.0.1:5050
```

---

## 7. Run the first small test

**Deliberately keep this small.** Do not select the whole Philippines.

1. Use the "Draw Boundary" tool to select **1-5 cells** (a tiny area, not
   a large region).
2. Set the date range to a **single recent date** within the range now
   shown by the date picker itself (its min/max come from
   `MIN_PREDICTION_DATE` and the latest valid prediction date calculated
   at startup -- see Section 5's console output for the exact value) --
   with **hourly resolution over 24 hours**.
3. Click **Generate PM2.5 Prediction Map**.
4. Watch the progress panel (real stages, real elapsed time, a live log
   console) and the backend's own console output at the same time --
   the backend console has the full detail (dataset queried, batch
   number, missing-data counts, GEE exceptions); the frontend log console
   shows the last 50 lines of the same thing.

The backend console will print, right at the start:

```
Cells = N
Timestamps = M
Prediction rows = N x M
```

If `N x M` exceeds 200, the request is rejected immediately with a clear
message -- this is a deliberately small ceiling for this diagnostic
stage (`config.settings.DIAGNOSTIC_TEST_MAX_ROWS`), not the eventual
production limit.

---

## 8. What to send back if something goes wrong

The full backend console output from your test run, including the
startup log. The two most likely first failure points, based on what I
could and couldn't verify without live GEE access:

- **Land-cover one-hot encoding** (`feature_builder.py`'s
  `_parse_landcover_scheme()`) -- I only confirmed
  `one_hot_encode_landcover()`'s function signature, not its actual body,
  so the dict shape I'm passing it is a best-effort guess. If this fails,
  check `one_hot_encode_landcover()`'s real implementation in
  `prepare_features.py` and I'll fix the caller to match.
- **GEE call syntax/quota behavior** -- every collection ID and band name
  was confirmed directly from your extraction scripts, but the actual
  `reduceRegions`/`ee.Image.unmask()` calls in `dynamic_predictors.py` and
  `static_predictors.py` have not been run against live GEE.

---

## 9. What this stage deliberately does NOT do

Per explicit instruction: no final row limits (only a small diagnostic
ceiling), no aggressive GEE call optimization, no production job queue
(job tracking is a simple in-memory dict + background thread -- fine for
one person testing, not for concurrent users), no scheduled static-layer
refresh (elevation/land cover/NDVI/nighttime lights are fetched live,
per request), no SHAP, no exports, no video playback, no frontend
redesign beyond the progress panel itself.
