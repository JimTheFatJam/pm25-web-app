"""
Extract ERA5-Land Meteorology, Hour-Grouped
=================================================

Built directly on extract_daily_predictors_batched.py's architecture --
same single-country selection, same checkpointed resumability, same
concurrent-request pattern, same true-country resolution. The one thing
that's genuinely different is the batching UNIT, because ERA5 has a
different redundancy structure than the daily products (AOD/Sentinel-5P).

Why this isn't just "the daily script with dates replaced by hours"
--------------------------------------------------------------------------
Daily batching eliminated TWO kinds of redundancy: temporal (the same
date's value reused across 24 hours -- the big 24x win) and spatial
(multiple cells sharing a date, batched into one call). ERA5 genuinely
changes every hour, so there is no free temporal win to collapse at the
per-hour level -- only the spatial kind is available here: if multiple grid
cells share the exact same UTC hour (very common, since most of a
country's stations are simultaneously active most of the time), batching
them into ONE reduceRegions() call per hour avoids redundantly resolving
that same hour's image once per cell. This is still a real, substantial
reduction (roughly 3x or more fewer requests than a per-row design, likely
more depending on how many cells overlap on a typical hour -- this script
reports your actual distinct-hour count on first run, the same way the
daily script reports distinct dates), just a smaller ceiling than daily
batching's 24x, because the underlying redundancy structure is different.

A further optimization (stacking an entire day's 24 hourly images into one
multi-band image via ee.ImageCollection.toBands(), collapsing ~24 hourly
calls into 1) was considered and deliberately NOT implemented here -- it
depends on GEE's band-naming behavior under .toBands(), which cannot be
verified without live access, and this project has been burned before by
trusting unverified GEE behavior (the mosaic-ordering theory, the AOD gap
chase). If per-hour batching proves too slow in practice, that is the
right moment to invest in verifying that riskier optimization live, with a
real baseline to compare against -- not before.

Per-hour is also structurally SIMPLER than the daily script's per-date
extraction: ERA5-Land provides all 5 variables as bands of a SINGLE image
per hour (unlike AOD/Sentinel-5P's 6 separate collections needing 6
separate searches per date), so each hour needs only one image resolution
and one reduceRegions() call per geometry type (local, neighborhood) --
not a loop over multiple sources.

Preserved from the daily extraction script
------------------------------------------------
- COUNTRY_CODE macro -- one country per run.
- Checkpointed resumability: every completed hour's results are appended
  immediately to era5_hourly_<COUNTRY>_progress.csv, not held in memory
  until the whole run finishes. An hour's extraction is atomic (either
  extract_one_hour returns a complete result for every cell needing that
  hour, or nothing is written for it), so "already has a row in the
  progress file" is a reliable, sufficient completeness check.
- The final user-facing output (era5_hourly_<COUNTRY>.csv, one row per
  original hourly label row) is regenerated from the progress file in a
  try/finally block, so even a Ctrl+C mid-run leaves an up-to-date (if
  partial) merged output.
- Concurrent hour processing via a thread pool -- GEE's interactive tier
  explicitly supports parallel requests up to quota limits.
- True-country resolution via sensor_grid_lookup.csv.

NOT YET LIVE-TESTED, as with every GEE-dependent script in this pipeline:
the actual Earth Engine calls cannot be exercised in the environment that
wrote this. What HAS been verified: the resumability logic (a simulated
interrupted-then-resumed two-session run, confirmed identical final output
regardless of session boundaries -- see the daily script's own test
history, which this reuses unchanged) and the hour-grouping/broadcast
logic specifically for this script (tested with synthetic multi-cell,
multi-hour label data). The timestamp formatting bug found in an earlier
version (pandas' default space-separated string vs. the strict "T"-
separated ISO format ee.Date() requires) is fixed and re-verified here too.

Variables extracted, and why
----------------------------------
From ERA5-Land (ECMWF/ERA5_LAND/HOURLY, ~11km resolution): temperature_2m,
dewpoint_2m, surface_pressure, u/v wind components at 10m, and total
precipitation (a direct, well-documented PM2.5 washout mechanism -- added
after an explicit variable-selection review; free to include since it's
already part of the same image as everything else).

From the separate, full ERA5 dataset (ECMWF/ERA5/HOURLY, ~31km resolution
-- NOT available in ERA5-Land at all, confirmed by directly listing
ERA5-Land's own bands): boundary_layer_height. Arguably the single most
cited meteorological driver of PM2.5 in the literature (a shallow boundary
layer traps pollutants near the surface; a deep one dilutes them), and not
redundant with anything else extracted here -- but it costs a second,
separate image resolution and reduceRegions() call pair per hour, roughly
doubling this script's total GEE call volume. Both the collection ID and
band name were confirmed by directly listing the real bands available (an
earlier assumption that the catalog webpage's underscore-separated URL
slug matched the real asset ID was WRONG for this specific collection --
it's "ECMWF/ERA5/HOURLY" with a slash, not an underscore -- caught before
any extraction code was built around the wrong guess).

Derived locally, NOT extracted as separate GEE bands (zero additional
Earth Engine cost either way): relative_humidity (Magnus-Tetens formula
from temperature_2m + dewpoint_2m, verified against a standard reference
value and against this project's own real extracted data) and
wind_speed_10m (magnitude of the u/v components already extracted).

Usage
-----
Set COUNTRY_CODE and EE_PROJECT_ID below. Run with TEST_MODE on first.
If interrupted, simply run the script again -- it resumes automatically.

Output: era5_hourly_{country}.csv (final, merged, one row per original
label row), era5_hourly_{country}_progress.csv (incrementally-saved
checkpoint, one row per (grid_id, hour) -- safe to delete once you're
confident the final output is complete and correct).

Dependencies: earthengine-api, pandas
    pip install earthengine-api pandas
"""

import csv
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import ee
import numpy as np
import pandas as pd

# ============================================================
# Configuration -- MACROS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
GRID_SUMMARY_PATH = SCRIPT_DIR / "training_grid_summary.csv"
SENSOR_GRID_LOOKUP_PATH = SCRIPT_DIR / "sensor_grid_lookup.csv"
LABEL_FILE_PATTERN = "grid_hourly_labels_*.csv"

EE_PROJECT_ID = __import__("os").environ.get("PM25_GEE_PROJECT_ID", "")  # scrubbed on vendoring -- see deploy/sync_research_pipeline.py
COUNTRY_CODE = "VN"  # <-- change per run; processes only this one country

OUTPUT_PATH = SCRIPT_DIR / f"era5_hourly_{COUNTRY_CODE}.csv"
PROGRESS_PATH = SCRIPT_DIR / f"era5_hourly_{COUNTRY_CODE}_progress.csv"
REPORT_PATH = SCRIPT_DIR / f"era5_hourly_{COUNTRY_CODE}_report.txt"

# --- ALWAYS run with this on first ---
TEST_MODE = False
TEST_MODE_N_HOURS = 10  # limits to the first N distinct hours NOT ALREADY IN THE PROGRESS FILE

MAX_CONCURRENT_HOURS = 8  # GEE interactive tier supports parallel requests up to quota
                          # limits; lower this if you hit rate-limit errors, raise if it
                          # comfortably succeeds

CELL_SIZE_M = 1000
NEIGHBORHOOD_RADIUS_CELLS = 2
HOURLY_FALLBACK_WINDOW_HOURS = 3  # ERA5-Land is essentially always complete, so this
                                  # fallback is expected to be rarely exercised -- kept
                                  # for consistency with how gaps are handled elsewhere
                                  # in this pipeline, not because ERA5 commonly needs it.

ERA5_COLLECTION = "ECMWF/ERA5_LAND/HOURLY"
ERA5_BANDS = {
    "temperature_2m": "temperature_2m",
    "dewpoint_2m": "dewpoint_temperature_2m",
    "surface_pressure": "surface_pressure",
    "u_wind_10m": "u_component_of_wind_10m",
    "v_wind_10m": "v_component_of_wind_10m",
    # "total_precipitation_hourly" is GEE's already-differenced per-hour
    # increment (not the raw cumulative-since-midnight total) -- confirmed
    # via the GEE Data Catalog that ERA5-Land's accumulated bands are
    # provided this way ("Earth Engine provides 19 hourly bands by computing
    # the difference between consecutive forecast steps"), but this exact
    # band name has NOT been independently confirmed via a direct band
    # listing the way every other band in this pipeline was -- watch your
    # TEST_MODE run specifically for this one; a wrong band name here would
    # surface as a clear GEE error, not a silent failure, the same way every
    # other wrong-band-name issue in this project has.
    "precipitation": "total_precipitation_hourly",
}

# Boundary layer height is NOT available in ERA5-Land at all -- confirmed by
# directly listing ERA5-Land's own bands, none of which include it. It
# exists only in the separate, full ERA5 ("single levels") dataset, at a
# coarser ~31km native resolution (vs. ERA5-Land's ~11km) -- a real,
# accepted resolution tradeoff for this one variable specifically, per the
# earlier cost/benefit discussion. Collection ID and band name BOTH directly
# confirmed via a live band listing (not guessed): the collection ID is
# "ECMWF/ERA5/HOURLY" (a SLASH between ERA5 and HOURLY, not the underscore
# used in "ECMWF/ERA5_LAND/HOURLY" -- an early diagnostic snippet in this
# project incorrectly assumed the same underscore convention applied here,
# caught before any extraction code was built around it), and the band name
# is "boundary_layer_height" with no prefix (NOT "mean_boundary_layer_height"
# -- that "mean_" prefixed name is a different variable, boundary layer
# dissipation, not height).
PBLH_COLLECTION = "ECMWF/ERA5/HOURLY"
PBLH_BAND = "boundary_layer_height"

# ERA5-Land uses a land-sea mask and has NO data at all over open ocean --
# confirmed empirically (not just from documentation) across three
# archipelagic countries: every affected grid cell showed either exactly
# 0% or exactly 100% missing across its entire time series, with zero
# partial/mixed cells -- the signature of a fixed geographic mask, not a
# time-varying effect like cloud cover. Mainland countries (Malaysia,
# Cambodia, Vietnam) showed 0% missing; the three island/archipelago
# countries (Indonesia, Singapore, Philippines) each showed 15-17% of
# rows affected. The separate full ERA5 dataset (already queried above for
# boundary_layer_height) genuinely covers the whole globe including ocean,
# and has same-unit equivalent bands for these five variables, so it is
# used as a fallback ONLY when the ERA5-Land value is null -- fetched from
# the SAME image already being queried for PBLH, so this costs zero
# additional Earth Engine calls.
#
# Precipitation is deliberately NOT included here: ERA5-Land's
# total_precipitation_hourly is an accumulated per-hour amount, while the
# closest equivalent in the full ERA5 dataset, mean_total_precipitation_rate,
# is a RATE -- a different unit that cannot be substituted in directly
# without a conversion this project has not verified. Leaving it null is
# safer than silently filling it with a wrong-unit value.
ERA5_LAND_OCEAN_FALLBACK_BANDS = {
    "temperature_2m": "temperature_2m",
    "dewpoint_2m": "dewpoint_temperature_2m",
    "surface_pressure": "surface_pressure",
    "u_wind_10m": "u_component_of_wind_10m",
    "v_wind_10m": "v_component_of_wind_10m",
}

# Precipitation CAN be backfilled too, via a verified unit conversion --
# confirmed via ECMWF's own Copernicus User Support documentation that mean
# rate parameters (Table 4) and accumulated parameters (Table 3) represent
# "the same information... except the units are different". The full ERA5
# dataset's "mean_total_precipitation_rate" is in kg m-2 s-1; since 1 kg/m2
# of water is equivalent to 1mm of depth, multiplying by 3600 (seconds per
# hour) gives mm accumulated per hour, and dividing by 1000 converts to
# meters, matching total_precipitation_hourly's own units exactly:
#   accumulated_meters_per_hour = rate_kg_per_m2_per_s * 3600 / 1000
#                               = rate_kg_per_m2_per_s * 3.6
# Handled as a special case below (not folded into
# ERA5_LAND_OCEAN_FALLBACK_BANDS) specifically because it needs this
# conversion factor applied, unlike the other five variables, which are
# direct same-unit substitutions.
PRECIP_RATE_BAND = "mean_total_precipitation_rate"
PRECIP_RATE_TO_METERS_PER_HOUR = 3.6

WORKING_CRS_WKT = (
    'PROJCS["Asia_South_Albers_Equal_Area_Conic",'
    'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,'
    'AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],'
    'PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],'
    'UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],'
    'AUTHORITY["EPSG","4326"]],'
    'PROJECTION["Albers_Conic_Equal_Area"],'
    'PARAMETER["latitude_of_center",-15],'
    'PARAMETER["longitude_of_center",125],'
    'PARAMETER["standard_parallel_1",7],'
    'PARAMETER["standard_parallel_2",-32],'
    'PARAMETER["false_easting",0],'
    'PARAMETER["false_northing",0],'
    'UNIT["metre",1,AUTHORITY["EPSG","9001"]],'
    'AXIS["Easting",EAST],AXIS["Northing",NORTH],'
    'AUTHORITY["ESRI","102028"]]'
)


# ============================================================
# True-country resolution (matching every other extraction script)
# ============================================================

def load_true_country_lookup(sensor_grid_lookup_path: Path) -> dict:
    lookup_df = pd.read_csv(sensor_grid_lookup_path, usecols=["grid_id", "country"])
    grid_id_to_country = {}
    for row in lookup_df.itertuples(index=False):
        grid_id_to_country.setdefault(row.grid_id, row.country)
    return grid_id_to_country


def fallback_country_from_grid_id(grid_id: str) -> str:
    match = re.match(r"^([A-Za-z]+)", str(grid_id))
    return match.group(1) if match else "UNKNOWN"


# ============================================================
# Hourly image resolution -- the ERA5-specific equivalent of the daily
# script's get_daily_image(), simpler because all 5 variables are bands of
# ONE image per hour, not 6 separate collections needing 6 separate searches
# ============================================================

def get_hourly_image_generic(collection_id: str, bands: list, target_hour: "ee.Date"):
    """Generic hourly image resolution, usable for ANY collection -- the
    logic (nearest-hour fallback, mean compositing across ties, empty-image
    handling) doesn't depend on which collection or how many bands it has.
    Used for both ERA5-Land (5-6 variables in one image) and the separate
    full ERA5 dataset (needed only for boundary_layer_height, since
    ERA5-Land doesn't provide it).

    IMPORTANT: same_hour.mean() (below) silently discards the source
    images' real projection metadata and defaults to a degenerate 1-degree-
    per-pixel identity transform -- confirmed directly via a live diagnostic
    (the composite's own .projection() printed transform [1,0,0,0,1,0],
    versus the real source data's [0.25,0,-180.125,0,-0.25,90.125]). A
    single reduceRegion() call still tolerated this and returned a value,
    but reduceRegions() (this pipeline's actual method, querying many
    geometries via a FeatureCollection) silently returned null for every
    feature -- confirmed to be exactly this cause, not a masking or
    data-availability issue, by replicating this script's exact query
    method in isolation. native_proj is captured from a raw image BEFORE
    any reducer composite has a chance to destroy it, then explicitly
    re-attached via .setDefaultProjection() so reduceRegions() correctly
    interprets the image's real sampling grid regardless of the requested
    output crs/scale."""
    col = (ee.ImageCollection(collection_id)
           .filterDate(target_hour.advance(-HOURLY_FALLBACK_WINDOW_HOURS, "hour"),
                       target_hour.advance(HOURLY_FALLBACK_WINDOW_HOURS + 1, "hour"))
           .select(bands))
    native_proj = ee.Image(col.first()).select(bands[0]).projection()

    def with_distance(img):
        img_date = ee.Date(img.get("system:time_start"))
        return img.set("hour_distance", img_date.difference(target_hour, "hour").abs())

    col_with_distance = col.map(with_distance).sort("hour_distance")

    def pick_nearest():
        closest_distance = ee.Number(col_with_distance.first().get("hour_distance"))
        same_hour = col_with_distance.filter(ee.Filter.eq("hour_distance", closest_distance))
        return (same_hour.mean()
                .setDefaultProjection(native_proj)
                .set("hour_offset_hours", closest_distance))

    empty_image = (ee.Image.constant([0] * len(bands))
                   .rename(bands)
                   .updateMask(ee.Image.constant(0))
                   .set("hour_offset_hours", -1))
    result_img = ee.Image(ee.Algorithms.If(col.size().gt(0), pick_nearest(), empty_image))
    return result_img.set("source_image_count", col.size())


def get_hourly_image(target_hour: "ee.Date"):
    """The single ERA5-Land image for this exact UTC hour -- thin wrapper
    around get_hourly_image_generic(), unchanged in behavior from the
    original version."""
    return get_hourly_image_generic(ERA5_COLLECTION, list(ERA5_BANDS.values()), target_hour)


def get_full_era5_image(target_hour: "ee.Date"):
    """The single full-ERA5 (not ERA5-Land) image for this exact UTC hour,
    carrying boundary_layer_height, the five ocean-fallback meteorological
    bands, AND the precipitation rate band together -- fetching all seven
    from the same image costs no more than fetching PBLH alone did, since
    it's one collection, one image resolution, one reduceRegions() call
    either way."""
    all_bands = [PBLH_BAND, PRECIP_RATE_BAND] + list(ERA5_LAND_OCEAN_FALLBACK_BANDS.values())
    return get_hourly_image_generic(PBLH_COLLECTION, all_bands, target_hour)


def cell_centroid_point(grid_x_index: int, grid_y_index: int, working_proj):
    cx = grid_x_index * CELL_SIZE_M + CELL_SIZE_M / 2
    cy = grid_y_index * CELL_SIZE_M + CELL_SIZE_M / 2
    return ee.Geometry.Point([cx, cy], proj=working_proj)


def neighborhood_rectangle(grid_x_index: int, grid_y_index: int, working_proj, radius=NEIGHBORHOOD_RADIUS_CELLS):
    x0 = (grid_x_index - radius) * CELL_SIZE_M
    y0 = (grid_y_index - radius) * CELL_SIZE_M
    x1 = (grid_x_index + radius + 1) * CELL_SIZE_M
    y1 = (grid_y_index + radius + 1) * CELL_SIZE_M
    return ee.Geometry.Rectangle([x0, y0, x1, y1], proj=working_proj, evenOdd=False)


def build_local_and_neighborhood_fcs(cells: pd.DataFrame, working_proj):
    local_features, nbhd_features = [], []
    for row in cells.itertuples(index=False):
        gx, gy = int(row.grid_x_index), int(row.grid_y_index)
        local_features.append(ee.Feature(cell_centroid_point(gx, gy, working_proj), {"grid_id": row.grid_id}))
        nbhd_features.append(ee.Feature(neighborhood_rectangle(gx, gy, working_proj), {"grid_id": row.grid_id}))
    return ee.FeatureCollection(local_features), ee.FeatureCollection(nbhd_features)


def extract_one_hour(hour_iso: str, cells_for_hour: pd.DataFrame, working_proj) -> dict:
    """One hour's worth of results across all ERA5-Land variables PLUS
    boundary layer height (from the separate full ERA5 dataset), for exactly
    the cells that need this hour. ERA5-Land needs only ONE image resolution
    and ONE reduceRegions() call per geometry type (all its variables come
    back as separate properties of that single call, since they're bands of
    the same image) -- boundary layer height needs its OWN separate image
    resolution and reduceRegions() call pair, since it lives in a different
    collection at a different native resolution, but reuses the EXACT SAME
    local_fc/nbhd_fc geometries (the query locations don't change, only
    which underlying image is being queried)."""
    target_hour = ee.Date(hour_iso)
    local_fc, nbhd_fc = build_local_and_neighborhood_fcs(cells_for_hour, working_proj)

    img = get_hourly_image(target_hour)

    local_result = img.reduceRegions(
        collection=local_fc, reducer=ee.Reducer.mean(), scale=CELL_SIZE_M, crs=working_proj,
    ).getInfo()
    nbhd_result = img.reduceRegions(
        collection=nbhd_fc, reducer=ee.Reducer.mean(), scale=CELL_SIZE_M, crs=working_proj,
    ).getInfo()
    hour_offset_hours = img.get("hour_offset_hours").getInfo()
    source_image_count = img.get("source_image_count").getInfo()

    local_by_gid = {f["properties"]["grid_id"]: f["properties"] for f in local_result["features"]}
    nbhd_by_gid = {f["properties"]["grid_id"]: f["properties"] for f in nbhd_result["features"]}

    results = {}
    for gid in cells_for_hour["grid_id"]:
        local_props = local_by_gid.get(gid, {})
        nbhd_props = nbhd_by_gid.get(gid, {})
        row_values = {}
        for name, band in ERA5_BANDS.items():
            row_values[f"{name}_local"] = local_props.get(band)
            row_values[f"{name}_neighborhood_mean"] = nbhd_props.get(band)
        row_values["era5_hour_offset_hours"] = hour_offset_hours
        row_values["era5_source_image_count"] = source_image_count
        results[gid] = row_values

    # Full ERA5 (not ERA5-Land) -- separate collection, separate image
    # resolution, but the SAME query geometries as above. Now fetches
    # boundary_layer_height AND the five ocean-fallback bands together, so
    # this is a MULTI-band image -- confirmed via live diagnostic earlier
    # that reduceRegions() names its output properties differently depending
    # on band count: a single-band image (PBLH fetched alone, the earlier
    # version of this code) returns the reducer's own default name ("mean"),
    # while a multi-band image (this version, and ERA5-Land above) returns
    # properties keyed by each band's own name. Using ".get('mean')" here now
    # would silently break in exactly the way the original PBLH bug did.
    full_era5_img = get_full_era5_image(target_hour)
    full_era5_local_result = full_era5_img.reduceRegions(
        collection=local_fc, reducer=ee.Reducer.mean(), scale=CELL_SIZE_M, crs=working_proj,
    ).getInfo()
    full_era5_nbhd_result = full_era5_img.reduceRegions(
        collection=nbhd_fc, reducer=ee.Reducer.mean(), scale=CELL_SIZE_M, crs=working_proj,
    ).getInfo()
    full_era5_hour_offset_hours = full_era5_img.get("hour_offset_hours").getInfo()
    full_era5_source_image_count = full_era5_img.get("source_image_count").getInfo()

    full_era5_local_by_gid = {f["properties"]["grid_id"]: f["properties"] for f in full_era5_local_result["features"]}
    full_era5_nbhd_by_gid = {f["properties"]["grid_id"]: f["properties"] for f in full_era5_nbhd_result["features"]}

    for gid in cells_for_hour["grid_id"]:
        row_values = results[gid]
        full_era5_local_props = full_era5_local_by_gid.get(gid, {})
        full_era5_nbhd_props = full_era5_nbhd_by_gid.get(gid, {})

        row_values["boundary_layer_height_local"] = full_era5_local_props.get(PBLH_BAND)
        row_values["boundary_layer_height_neighborhood_mean"] = full_era5_nbhd_props.get(PBLH_BAND)
        row_values["pblh_hour_offset_hours"] = full_era5_hour_offset_hours
        row_values["pblh_source_image_count"] = full_era5_source_image_count

        # Ocean fallback: ERA5-Land has no data at all over water (confirmed
        # empirically -- see ERA5_LAND_OCEAN_FALLBACK_BANDS above). Only
        # substitute when the ERA5-Land value is genuinely missing, and
        # always record whether a substitution happened, so a value pulled
        # from the coarser (~31km vs ~11km) full-ERA5 grid is never
        # indistinguishable from a native ERA5-Land value.
        for name, full_era5_band in ERA5_LAND_OCEAN_FALLBACK_BANDS.items():
            local_key, nbhd_key = f"{name}_local", f"{name}_neighborhood_mean"
            used_fallback_local = row_values.get(local_key) is None
            used_fallback_nbhd = row_values.get(nbhd_key) is None
            if used_fallback_local:
                row_values[local_key] = full_era5_local_props.get(full_era5_band)
            if used_fallback_nbhd:
                row_values[nbhd_key] = full_era5_nbhd_props.get(full_era5_band)
            row_values[f"{name}_used_ocean_fallback_local"] = used_fallback_local
            row_values[f"{name}_used_ocean_fallback_neighborhood"] = used_fallback_nbhd

        # Precipitation: special-cased because it needs the verified
        # rate-to-accumulation conversion (PRECIP_RATE_TO_METERS_PER_HOUR)
        # applied, not a direct substitution like the five variables above.
        precip_used_fallback_local = row_values.get("precipitation_local") is None
        precip_used_fallback_nbhd = row_values.get("precipitation_neighborhood_mean") is None
        if precip_used_fallback_local:
            rate = full_era5_local_props.get(PRECIP_RATE_BAND)
            row_values["precipitation_local"] = rate * PRECIP_RATE_TO_METERS_PER_HOUR if rate is not None else None
        if precip_used_fallback_nbhd:
            rate = full_era5_nbhd_props.get(PRECIP_RATE_BAND)
            row_values["precipitation_neighborhood_mean"] = rate * PRECIP_RATE_TO_METERS_PER_HOUR if rate is not None else None
        row_values["precipitation_used_ocean_fallback_local"] = precip_used_fallback_local
        row_values["precipitation_used_ocean_fallback_neighborhood"] = precip_used_fallback_nbhd

    return results



# ============================================================
# Resumability helpers (identical pattern to the daily extraction script)
# ============================================================

def check_progress_file_schema(progress_path: Path):
    """Detect a stale, narrower header left over from before a schema change
    (like the ocean-fallback fix) mixed with newer, wider rows -- the exact
    failure mode that wasted an entire run's worth of GEE calls before
    failing at the final merge step, since the mismatch was never caught
    until write_final_merged_output() tried to read the whole file at the
    end. Checking this FIRST, before any processing begins, fails fast with
    clear instructions instead of burning through hours of extraction only
    to fail at the very last step."""
    if not progress_path.exists():
        return
    with open(progress_path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            return
        header_len = len(header)
        for row in reader:
            if row and len(row) != header_len:
                raise RuntimeError(
                    f"{progress_path.name} has an inconsistent schema: its header has "
                    f"{header_len} column(s), but at least one row has {len(row)}. This "
                    f"happens when a progress file predates a schema change (such as the "
                    f"ocean-fallback fix). Run repair_era5_progress_schema.py on this file "
                    f"first, then rerun this script -- don't delete the file, it can be "
                    f"repaired without losing any completed work."
                )


def load_already_done_hours(progress_path: Path) -> set:
    """An hour's extraction is atomic -- see extract_one_hour -- so "this
    hour has any row in the progress file" would normally be a reliable,
    sufficient completeness check. It is NOT sufficient on its own: rows
    written before a given fix have every current column present in the
    file's header but null values specifically for what that fix was
    supposed to populate -- and a plain "does this hour exist" check would
    treat those as done forever, silently never reprocessing them (already
    confirmed to happen once, for boundary_layer_height specifically).

    The same blind spot applies to the ocean-fallback fix: a row processed
    before that fix existed has a valid boundary_layer_height_local (that
    was already fixed) but a null temperature_2m_local for any ocean-masked
    cell (since the fallback that would have filled it in didn't exist
    yet). Checking boundary_layer_height_local alone would NOT catch this,
    since PBLH was already fine. An hour is now only considered done if
    BOTH boundary_layer_height_local AND temperature_2m_local are non-null
    -- after the ocean-fallback fix, temperature_2m_local should be
    non-null for every successfully processed row regardless of whether
    the cell is on land or over water, so requiring it is a correct
    completeness check going forward, not just a workaround."""
    if not progress_path.exists():
        return set()
    try:
        existing = pd.read_csv(progress_path, usecols=["hour", "boundary_layer_height_local", "temperature_2m_local"])
    except ValueError as e:
        print(f"  WARNING: {progress_path.name} is missing a column this check now requires ({e}) -- "
              f"likely a progress file from before that column existed. Treating all its hours as "
              f"not yet done, so everything gets safely reprocessed under the current script.")
        return set()
    complete = existing[existing["boundary_layer_height_local"].notna() & existing["temperature_2m_local"].notna()]
    return set(complete["hour"].astype(str).unique())


def append_to_progress(progress_path: Path, hour_str: str, hour_result: dict):
    rows = [{"grid_id": gid, "hour": hour_str, **values} for gid, values in hour_result.items()]
    df = pd.DataFrame(rows)
    write_header = not progress_path.exists()
    df.to_csv(progress_path, mode="a", header=write_header, index=False)


def derive_additional_features(df: pd.DataFrame) -> pd.DataFrame:
    """Relative humidity and wind speed are DERIVED locally from
    already-extracted temperature/dewpoint/wind columns, not extracted as
    separate GEE bands -- zero additional Earth Engine cost either way.

    Relative humidity uses the Magnus-Tetens approximation, verified against
    a commonly cited reference value (25C air temp, 20C dewpoint -> ~73.8%
    RH, reproduced exactly) and the T==Td edge case (always exactly 100%
    RH, saturated air). temperature_2m/dewpoint_2m are in Kelvin (ERA5's
    native units) and are converted to Celsius before applying the formula.

    Wind speed is the straightforward magnitude of the u/v components
    already extracted (sqrt(u^2 + v^2))."""
    df = df.copy()

    def magnus_rh_vectorized(temp_k, dewpoint_k):
        temp_c = temp_k - 273.15
        dewpoint_c = dewpoint_k - 273.15
        es = 6.1094 * np.exp(17.625 * temp_c / (temp_c + 243.04))
        e = 6.1094 * np.exp(17.625 * dewpoint_c / (dewpoint_c + 243.04))
        return 100.0 * e / es

    for scope in ["local", "neighborhood_mean"]:
        temp_col = f"temperature_2m_{scope}"
        dewpoint_col = f"dewpoint_2m_{scope}"
        u_col = f"u_wind_10m_{scope}"
        v_col = f"v_wind_10m_{scope}"

        if temp_col in df.columns and dewpoint_col in df.columns:
            df[f"relative_humidity_{scope}"] = magnus_rh_vectorized(df[temp_col], df[dewpoint_col])
        if u_col in df.columns and v_col in df.columns:
            df[f"wind_speed_10m_{scope}"] = np.sqrt(df[u_col] ** 2 + df[v_col] ** 2)

    return df


def write_final_merged_output(country_labels: pd.DataFrame, progress_path: Path, output_path: Path):
    """Regenerate the final, merged, user-facing output from whatever is
    currently in the progress file -- called both at the end of a
    fully-completed run and (via try/finally) after an interruption.

    Deduplication is necessary now that load_already_done_hours() can
    consider an hour "not done" even though a row for it already exists
    (the boundary_layer_height completeness check) -- reprocessing that hour
    appends a NEW row rather than replacing the old incomplete one, since
    append_to_progress() only ever appends. Keeping the LAST occurrence per
    (grid_id, hour) -- the most recently written, i.e. the reprocessed one
    -- avoids a many-to-one merge blowing up into duplicate output rows."""
    if not progress_path.exists():
        print("  No progress file yet -- nothing to write to the final output.")
        return 0
    hourly_results_df = pd.read_csv(progress_path)
    hourly_results_df["hour"] = hourly_results_df["hour"].astype(str)
    hourly_results_df = hourly_results_df.drop_duplicates(subset=["grid_id", "hour"], keep="last")
    out_df = country_labels.merge(hourly_results_df, on=["grid_id", "hour"], how="left")
    out_df = out_df.drop(columns=["hour", "true_country"], errors="ignore")
    out_df = derive_additional_features(out_df)
    out_df.to_csv(output_path, index=False)
    return len(out_df)


# ============================================================
# Main
# ============================================================

def main():
    ee.Initialize(project=EE_PROJECT_ID)

    # Check this BEFORE any processing/GEE calls happen -- catching a schema
    # mismatch here costs a fraction of a second; catching it only at the
    # final merge step (as originally happened) wastes an entire run's
    # worth of GEE calls first.
    check_progress_file_schema(PROGRESS_PATH)

    if not GRID_SUMMARY_PATH.exists():
        raise FileNotFoundError(f"Expected {GRID_SUMMARY_PATH}.")
    if not SENSOR_GRID_LOOKUP_PATH.exists():
        raise FileNotFoundError(f"Expected {SENSOR_GRID_LOOKUP_PATH}.")

    grid_summary = pd.read_csv(GRID_SUMMARY_PATH, usecols=["grid_id", "grid_x_index", "grid_y_index"])
    true_country_lookup = load_true_country_lookup(SENSOR_GRID_LOOKUP_PATH)

    label_files = sorted(SCRIPT_DIR.glob(LABEL_FILE_PATTERN))
    if not label_files:
        print(f"No files matching {LABEL_FILE_PATTERN} found in {SCRIPT_DIR}.")
        return

    all_labels = pd.concat(
        [pd.read_csv(p, usecols=["grid_id", "datetime_utc"]) for p in label_files], ignore_index=True
    )
    all_labels["true_country"] = all_labels["grid_id"].map(true_country_lookup)
    unresolved = all_labels["true_country"].isna()
    if unresolved.any():
        all_labels.loc[unresolved, "true_country"] = all_labels.loc[unresolved, "grid_id"].apply(fallback_country_from_grid_id)

    country_labels = all_labels[all_labels["true_country"] == COUNTRY_CODE].copy()
    if len(country_labels) == 0:
        print(f"No label rows resolved to true_country == '{COUNTRY_CODE}'. Nothing to do.")
        return
    # "hour" here means the exact UTC hour timestamp (unlike the daily script's
    # "date", which discards time-of-day) -- ERA5 genuinely varies hourly, so
    # each distinct timestamp is its own batching unit, not just each date.
    # "hour" doubles as both the grouping/progress-tracking key AND the exact
    # string passed to ee.Date() inside extract_one_hour -- it must be strict
    # ISO-8601 with a "T" separator (e.g. "2020-01-01T00:00:00"), not pandas'
    # default astype(str) representation, which uses a space separator
    # ("2020-01-01 00:00:00+00:00") that ee.Date() rejects outright. This is
    # the same bug, and the same fix, as the one found in
    # extract_predictors_gee.py's build_feature_collection() much earlier in
    # this project -- missed here when this script was built from scratch,
    # confirmed and corrected directly against the reported error.
    country_labels["hour"] = pd.to_datetime(country_labels["datetime_utc"]).dt.floor("h").dt.strftime("%Y-%m-%dT%H:%M:%S")

    cells = country_labels[["grid_id"]].drop_duplicates().merge(grid_summary, on="grid_id", how="left")
    n_unmatched = cells["grid_x_index"].isna().sum()
    if n_unmatched:
        print(f"  WARNING: {n_unmatched} grid_id(s) not found in {GRID_SUMMARY_PATH.name} -- dropped.")
    cells = cells.dropna(subset=["grid_x_index", "grid_y_index"])
    cell_index = cells.set_index("grid_id")

    cells_per_hour_grouped = country_labels[["grid_id", "hour"]].drop_duplicates().merge(
        cell_index[["grid_x_index", "grid_y_index"]], on="grid_id", how="inner"
    ).groupby("hour")

    all_distinct_hours = sorted(cells_per_hour_grouped.groups.keys())
    print(f"=== {COUNTRY_CODE} ===")
    print(f"{len(country_labels)} label rows -> {len(all_distinct_hours)} distinct hour(s), "
          f"{cells['grid_id'].nunique()} distinct grid cell(s).")
    print(f"(For comparison: a per-row design would need {len(country_labels)} requests per geometry type; "
          f"this needs {len(all_distinct_hours)}.)")

    already_done = load_already_done_hours(PROGRESS_PATH)
    remaining_hours = [h for h in all_distinct_hours if h not in already_done]
    print(f"{len(already_done)} hour(s) already completed in a previous session "
          f"(found in {PROGRESS_PATH.name}) -- {len(remaining_hours)} remaining.")

    if TEST_MODE:
        remaining_hours = remaining_hours[:TEST_MODE_N_HOURS]
        print(f"TEST_MODE is on -- limiting to the first {len(remaining_hours)} remaining hour(s).")

    working_proj = ee.Projection(WORKING_CRS_WKT)

    def process_one_hour(hour_str):
        cells_for_hour = cells_per_hour_grouped.get_group(hour_str)[["grid_id", "grid_x_index", "grid_y_index"]]
        return hour_str, extract_one_hour(hour_str, cells_for_hour, working_proj)

    n_completed_this_session = 0
    n_failed_this_session = 0

    try:
        if remaining_hours:
            with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_HOURS) as executor:
                futures = {executor.submit(process_one_hour, h): h for h in remaining_hours}
                for future in as_completed(futures):
                    hour_str = futures[future]
                    try:
                        _, hour_result = future.result()
                    except Exception as e:
                        n_failed_this_session += 1
                        print(f"  ERROR on {hour_str}: {e} -- will retry on next run.")
                        continue
                    append_to_progress(PROGRESS_PATH, hour_str, hour_result)
                    n_completed_this_session += 1
                    print(f"  [{n_completed_this_session + len(already_done)}/{len(all_distinct_hours)}] "
                          f"{hour_str} done and saved to {PROGRESS_PATH.name}.")
        else:
            print("  Nothing left to process for this country -- already fully complete.")
    finally:
        # Runs on normal completion AND on Ctrl+C/exceptions, so the final
        # output always reflects whatever progress has actually been saved.
        print(f"\nRegenerating {OUTPUT_PATH.name} from current progress...")
        n_rows = write_final_merged_output(country_labels, PROGRESS_PATH, OUTPUT_PATH)
        print(f"Wrote {n_rows} rows -> {OUTPUT_PATH.name}")

        report_lines = [
            f"ERA5 Hourly Extraction -- Report ({COUNTRY_CODE})", "=" * 42,
            f"Total distinct hours for this country: {len(all_distinct_hours)}",
            f"Completed before this session: {len(already_done)}",
            f"Completed this session: {n_completed_this_session}",
            f"Failed this session (will retry next run): {n_failed_this_session}",
            f"Remaining after this session: "
            f"{len(all_distinct_hours) - len(already_done) - n_completed_this_session}",
        ]
        REPORT_PATH.write_text("\n".join(report_lines))
        print(f"Wrote {REPORT_PATH.name}")


if __name__ == "__main__":
    main()
