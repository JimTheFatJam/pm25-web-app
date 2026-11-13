"""
diagnose_dynamic_retrieval.py
============================================================================
STANDALONE DIAGNOSTIC -- run this from the pm25_web_app directory, with a
real, initialized GEE session, to directly test the hypothesis behind the
"0 ok / N missing (all N variables combined)" reports.

Reuses the app's OWN real grid file, CRS, cell-geometry construction, and
exact GEE collection IDs/bands (via backend.prediction.dynamic_predictors)
-- this is not a separately-invented approximation, it exercises the
identical inputs the real app uses.

What this tests, for a small sample of real grid cells and a known
historical date (2024-06-01) / hour (2024-06-01T12:00 UTC), where every
dataset should be well established:

  1. Each dynamic dataset INDEPENDENTLY -- its own separate
     reduceRegions() call, no ee.Image.cat() combination with anything
     else -- exactly matching how extract_era5_hourly_batched.py itself
     queries ERA5-Land and (separately) ERA5.
  2. The SAME datasets combined via ee.Image.cat() with NO explicit crs=
     -- exactly matching the app's current dynamic_predictors.py code.
  3. The SAME combined image again, but WITH an explicit crs= parameter
     on reduceRegions() -- exactly matching the research pipeline's own
     validated pattern (extract_era5_hourly_batched.py always passes
     crs=working_proj).

If (1) succeeds but (2) fails while (3) succeeds, that is direct,
observed confirmation that combining differently-sourced bands via
ee.Image.cat() without an explicit crs= is the cause -- exactly the
"composites default to a degenerate WGS-84 1-degree-pixel projection"
behavior documented in Earth Engine's own guide (Reducing an
ImageCollection) and already independently confirmed for a related
.mean() case in extract_era5_hourly_batched.py's own docstring.

This script makes NO changes to the application, the trained model, the
feature schema, or any prediction logic. It is read-only diagnostics.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Must come before `import ee` -- see backend/_ee_windows_compat.py's own
# docstring; matches backend/app.py's exact required import order.
from backend import _ee_windows_compat  # noqa

import ee
from config.settings import GEE_PROJECT_ID, ANALYSIS_CRS
import config.settings as _settings
from grid.grid_provider import load_grid_with_fallback
from backend.prediction import dynamic_predictors as dp

print("Initializing GEE...")
if GEE_PROJECT_ID:
    ee.Initialize(project=GEE_PROJECT_ID)
else:
    ee.Initialize()
print("  OK.\n")

print("Loading the real grid file...")
GRID = load_grid_with_fallback()
_settings.ANALYSIS_CRS = GRID.crs
grid_crs = GRID.crs
print(f"  Loaded {len(GRID):,} cells. CRS = {grid_crs}\n")

# A small, real sample of cells -- NOT a synthetic geometry. Configurable
# via a 3rd command-line argument specifically to test whether cell
# count/the specific cell set (not the date/hour) is what determines
# success vs failure -- confirmed necessary after reproduce_real_request.py
# showed 21 real cells failing completely at the same date/hour where 5
# cells (this script's original default) succeeded completely.
N_SAMPLE_CELLS = int(sys.argv[3]) if len(sys.argv) > 3 else 5
sample_cells = GRID.iloc[:N_SAMPLE_CELLS][["grid_id", "grid_x_index", "grid_y_index"]].reset_index(drop=True)
print(f"Using {len(sample_cells)} real grid cells for this diagnostic:")
print(sample_cells.to_string(index=False))
print()

local_fc, nbhd_fc = dp._build_feature_collections(sample_cells, grid_crs)

TEST_DATE = sys.argv[1] if len(sys.argv) > 1 else "2024-06-01"
TEST_HOUR_ISO = sys.argv[2] if len(sys.argv) > 2 else f"{TEST_DATE}T12:00:00"
print(f"Testing date={TEST_DATE}, hour={TEST_HOUR_ISO}\n")


def report(label, reduced_fc):
    """Prints the raw returned properties for every feature, and a
    valid/null summary -- not just a combined ok/missing count."""
    info = reduced_fc.getInfo()
    print(f"--- {label} ---")
    for feat in info["features"]:
        props = {k: v for k, v in feat["properties"].items() if k != "grid_id"}
        print(f"  {feat['properties']['grid_id']}: {props}")
    # Per-property valid/null summary across all returned features.
    if info["features"]:
        all_keys = sorted(set(k for feat in info["features"] for k in feat["properties"] if k != "grid_id"))
        for key in all_keys:
            n_valid = sum(1 for feat in info["features"] if feat["properties"].get(key) is not None)
            print(f"    {key:35s} {n_valid}/{len(info['features'])} valid")
    else:
        print("  (zero features returned -- reduceRegions() matched nothing at all)")
    print()


# ============================================================
# PART A: each dataset INDEPENDENTLY (separate reduceRegions() calls,
# no ee.Image.cat() with anything else) -- matches
# extract_era5_hourly_batched.py's own approach.
# ============================================================
print("=" * 70)
print("PART A: each dataset tested INDEPENDENTLY (no ee.Image.cat())")
print("=" * 70)

# A1. ERA5-Land alone, all 6 bands, exact hour.
hour_start = ee.Date(TEST_HOUR_ISO)
era5_land_img = ee.ImageCollection(dp.ERA5_LAND_COLLECTION).filterDate(hour_start, hour_start.advance(1, "hour")).first()
era5_land_img = era5_land_img.select(list(dp.ERA5_LAND_BANDS.values()))
report("A1. ERA5-Land alone, local cells, no crs=", era5_land_img.reduceRegions(collection=local_fc, reducer=ee.Reducer.mean(), scale=1000))

# A2. Full ERA5 alone (the PBLH + fallback source), exact hour.
era5_img = ee.ImageCollection(dp.ERA5_COLLECTION).filterDate(hour_start, hour_start.advance(1, "hour")).first()
era5_img_bands = era5_img.select([dp.ERA5_PBLH_BAND, dp.PRECIP_RATE_BAND] + list(dp.ERA5_FALLBACK_BANDS.values()))
report("A2. Full ERA5 alone, local cells, no crs=", era5_img_bands.reduceRegions(collection=local_fc, reducer=ee.Reducer.mean(), scale=1000))

# A3. MODIS AOD alone, single day.
day_start = ee.Date(TEST_DATE)
day_end = day_start.advance(1, "day")
aod_img = ee.ImageCollection(dp.MODIS_AOD_COLLECTION).select(dp.MODIS_AOD_BAND).filterDate(day_start, day_end).mean()
report("A3. MODIS AOD alone (single day mean, no crs=)", aod_img.reduceRegions(collection=local_fc, reducer=ee.Reducer.mean(), scale=1000))

# A4. Each Sentinel-5P gas alone, single day.
for name, (coll, band) in dp.S5P_GASES.items():
    gas_img = ee.ImageCollection(coll).select(band).filterDate(day_start, day_end).mean()
    report(f"A4. Sentinel-5P {name} alone (single day mean, no crs=)", gas_img.reduceRegions(collection=local_fc, reducer=ee.Reducer.mean(), scale=1000))


# ============================================================
# PART B: the app's ACTUAL current approach -- ee.Image.cat() combining
# ERA5-Land + ERA5 PBLH into one multi-band image, NO explicit crs= on
# reduceRegions(). This is dynamic_predictors.fetch_hourly_predictors'
# real code path, called directly (not reimplemented).
# ============================================================
print("=" * 70)
print("PART B: the app's ACTUAL current combined approach (ee.Image.cat(), no crs=)")
print("=" * 70)

band_images = []
for var_name, band in dp.ERA5_LAND_BANDS.items():
    land_val_img = era5_land_img.select(band)
    if var_name == "precipitation":
        fallback_val_img = era5_img.select(dp.PRECIP_RATE_BAND).multiply(dp.PRECIP_RATE_TO_METERS_PER_HOUR)
    else:
        fallback_val_img = era5_img.select(dp.ERA5_FALLBACK_BANDS[var_name])
    band_images.append(land_val_img.unmask(fallback_val_img).rename(var_name))
band_images.append(era5_img.select(dp.ERA5_PBLH_BAND).rename("boundary_layer_height"))
combined_img_no_crs = ee.Image.cat(band_images)
report("B. ERA5-Land + ERA5 PBLH combined via ee.Image.cat(), NO crs=", combined_img_no_crs.reduceRegions(collection=local_fc, reducer=ee.Reducer.mean(), scale=1000))


# ============================================================
# PART C: the SAME combined image, but WITH an explicit crs= -- matching
# extract_era5_hourly_batched.py's own validated practice
# (crs=working_proj on every reduceRegions() call there).
# ============================================================
print("=" * 70)
print("PART C: the SAME combined image, WITH an explicit crs= (research-pipeline style)")
print("=" * 70)

# Use ERA5-Land's own native projection as the explicit crs -- the same
# role extract_era5_hourly_batched.py's working_proj plays there.
era5_land_native_proj = era5_land_img.select(list(dp.ERA5_LAND_BANDS.values())[0]).projection()
report("C. Combined image, WITH explicit crs=<ERA5-Land native projection>",
       combined_img_no_crs.reduceRegions(collection=local_fc, reducer=ee.Reducer.mean(), scale=1000, crs=era5_land_native_proj))

print("=" * 70)
print("Diagnostic complete. Compare Part A (independent) vs Part B (app's current, no crs=) vs Part C (with explicit crs=).")
print("If A succeeds, B fails, and C succeeds: the missing crs= on the combined image is the confirmed cause.")
print("=" * 70)

# ============================================================
# PART D: what does the app's OWN live availability check currently say
# about this exact test date? This calls the same
# check_dynamic_data_availability() the background thread runs at
# startup -- a fresh, live call here (diagnostic-only; the running app
# itself does NOT re-call this per request, it uses its cached result).
# Directly checks whether TEST_DATE is inside or outside what the app
# would currently consider valid.
# ============================================================
print()
print("=" * 70)
print("PART D: live availability check, for direct comparison against the tested date")
print("=" * 70)
from backend.prediction.data_availability import check_dynamic_data_availability, DataAvailabilityError
try:
    availability = check_dynamic_data_availability(log_fn=print)
    from datetime import date as _date
    tested_date = _date.fromisoformat(TEST_DATE)
    latest_valid = availability["latest_valid_date"]
    print()
    if tested_date > latest_valid:
        print(f"TEST_DATE {TEST_DATE} is AFTER the currently computed latest valid date {latest_valid} "
              f"(binding dataset: '{availability['binding_dataset']}') -- if this is the date that failed "
              f"in the real app, this is consistent with genuine (if narrower-than-expected) data latency, "
              f"not a retrieval bug.")
    else:
        print(f"TEST_DATE {TEST_DATE} is WITHIN the currently computed valid range (latest valid date "
              f"{latest_valid}) -- if this date still failed in the real app, the availability check's "
              f"computed bound does not match what dynamic_predictors.py actually found, which would need "
              f"further investigation.")
except DataAvailabilityError as e:
    print(f"Live availability check itself failed: {e}")
