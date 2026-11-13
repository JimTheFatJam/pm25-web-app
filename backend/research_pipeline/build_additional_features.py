"""
Additional Feature Collection Pipeline
=============================================

Builds source_additional_features.csv and philippines_additional_features.csv
as SEPARATE, optional files -- source_features.csv and philippines_features.csv
are never read for writing and never modified. Joinable on (grid_id,
datetime_utc), the same keys used throughout this project.

Three feature groups, per the finalized design report:
  1. MODIS Active Fire (MOD14A1 Terra + MYD14A1 Aqua, combined) -- hotspot
     count/presence/FRP, local + 5km/10km neighborhood, 24h/72h/7d rolling
     accumulation. VIIRS explicitly NOT used -- confirmed during design
     research that GEE's VIIRS active fire collection only starts
     2023-09-03, which would leave the first ~3.5 years of this project's
     Jan 2020-Jun 2026 study period with zero fire coverage. MODIS (Terra
     since Feb 2000, Aqua since Jul 2002) covers the full study period.
  2. Consecutive dry days -- derived entirely from precipitation_local,
     already present in the EXISTING feature files. No new GEE extraction.
  3. Calendar features (is_weekend, is_public_holiday) -- computed
     entirely in pandas, no GEE extraction.
  Drought indices (SPI/PDSI/SPEI) were evaluated and explicitly EXCLUDED
  per the design report: the only global GEE option (SPEIbase) is 0.5
  degree monthly resolution, far too coarse for this project's ~1km
  hourly task and substantially redundant with consecutive dry days.

GEE extraction strategy: mirrors extract_daily_predictors_batched.py
EXACTLY, not a new design -- date-grouped batching (one reduceRegions()
call per distinct calendar date, not per row), incremental per-date
checkpointing with atomic completeness (a date either has a complete
progress-file entry or none at all), and a try/finally block that
regenerates the final merged output even after an interruption. No
Export Tasks, no Drive exports -- results streamed directly via
.getInfo(), identical to every other extraction script in this project.

IMPORTANT, stated plainly: this environment has no Earth Engine
credentials and no network access to earthengine.googleapis.com -- the
actual GEE calls in this script are UNTESTED here, exactly as already
noted in extract_daily_predictors_batched.py's own docstring for the
same reason. What IS tested (see the bottom of this file / the
accompanying test log): the resumability/checkpointing logic itself
(simulated multi-session runs), the consecutive-dry-days computation,
the calendar feature generation, and the diagnostics generation -- all
of which are pure pandas and require no GEE access to verify.

MOD14A1/MYD14A1 FireMask encoding, verified via NASA's own MODIS Fire
Product User's Guide before use here (not assumed): values 7 (low
confidence), 8 (nominal confidence), and 9 (high confidence) are fire
detections; 0-6 are non-fire categories (unprocessed, masked, water,
cloud, non-fire land, unknown). FIRE_DETECTION_THRESHOLD=7 below reflects
this directly.

Usage
-----
Set COUNTRY_CODE and EE_PROJECT_ID below, run with TEST_MODE=True first.
If interrupted, simply run again -- resumes from the progress file
automatically. After all countries are done, run
build_final_additional_features_files() to assemble the two deliverable
CSVs (source and Philippines).
"""

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import ee
import holidays
import numpy as np
import pandas as pd

# ============================================================
# Configuration -- MACROS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
SOURCE_FEATURES_PATH = SCRIPT_DIR / "source_features.csv"
SOURCE_METADATA_PATH = SCRIPT_DIR / "source_metadata.csv"
PH_FEATURES_PATH = SCRIPT_DIR / "philippines_features.csv"
PH_METADATA_PATH = SCRIPT_DIR / "philippines_metadata.csv"
GRID_SUMMARY_PATH = SCRIPT_DIR / "training_grid_summary.csv"
SENSOR_GRID_LOOKUP_PATH = SCRIPT_DIR / "sensor_grid_lookup.csv"

SOURCE_OUTPUT_PATH = SCRIPT_DIR / "source_additional_features.csv"
PH_OUTPUT_PATH = SCRIPT_DIR / "philippines_additional_features.csv"
SOURCE_ASSEMBLED_COUNTRIES_PATH = SCRIPT_DIR / "source_additional_features_assembled_countries.txt"
PH_ASSEMBLED_COUNTRIES_PATH = SCRIPT_DIR / "philippines_additional_features_assembled_countries.txt"
SUMMARY_OUTPUT_PATH = SCRIPT_DIR / "additional_feature_summary.csv"
EXTREME_STATS_OUTPUT_PATH = SCRIPT_DIR / "feature_extreme_statistics.csv"

EE_PROJECT_ID = __import__("os").environ.get("PM25_GEE_PROJECT_ID", "")  # scrubbed on vendoring -- see deploy/sync_research_pipeline.py
COUNTRY_CODE = "PH"  # <-- change per run, one country per run, matching the established pattern
IS_PHILIPPINES_RUN = True  # <-- set True when processing the Philippines target domain specifically

FIRE_PROGRESS_PATH = SCRIPT_DIR / f"fire_features_{COUNTRY_CODE}_progress.csv"
FIRE_REPORT_PATH = SCRIPT_DIR / f"fire_features_{COUNTRY_CODE}_report.txt"

TEST_MODE = False
TEST_MODE_N_DATES = 5

MAX_CONCURRENT_DATES = 8  # same rationale as extract_daily_predictors_batched.py

CELL_SIZE_M = 1000
LOCAL_FALLBACK_BUFFER_RADIUS_M = 2000
NEIGHBORHOOD_RADII_M = {"5km": 5000, "10km": 10000}

MOD14A1_COLLECTION = "MODIS/061/MOD14A1"  # Terra, since 2000-02-24
MYD14A1_COLLECTION = "MODIS/061/MYD14A1"  # Aqua, since 2002-07-04
FIRE_DETECTION_THRESHOLD = 7  # FireMask >= 7 -> fire pixel (low/nominal/high confidence); see module docstring

ROLLING_WINDOWS_FIRE = {"24h": "24h", "72h": "72h", "7day": "7D"}

# Consecutive dry days: WMO-recognized "dry day" definition (<1mm/day),
# documented here explicitly per instruction, not buried in a constant
# with no explanation.
DRY_DAY_PRECIP_THRESHOLD_MM = 1.0

# Countries covered by this project -- used to build a per-country holiday
# calendar. ISO 3166-1 alpha-2 codes, matching this project's existing
# country-code convention (see run_source_reweighting_experiment.py's
# SOURCE_COUNTRIES).
PROJECT_COUNTRIES = ["IN", "TH", "VN", "ID", "KH", "SG", "MY", "PH"]

TIMEZONE_CACHE_PATH = SCRIPT_DIR / "grid_cell_timezones.csv"

# Fixed-date holidays (month, day) -- calendar-date equality, no library
# needed. LUNAR_HOLIDAY_NAME_PATTERNS below handles the ones that move
# year to year (Lunar New Year), matched against the `holidays` library's
# own returned holiday NAMES rather than hardcoded dates, since a lunar
# date shifts by ~10-11 days each Gregorian year.
LUNAR_HOLIDAY_NAME_PATTERNS = ["lunar new year", "chinese new year", "tết", "tet"]


# ============================================================
# CORRECTION: per-cell local time resolution.
# ============================================================
# The original version of build_calendar_features and
# compute_consecutive_dry_days operated directly on datetime_utc --
# WRONG for any calendar-based feature (is_weekend, is_public_holiday,
# and everything below), because "what calendar date is this" is
# inherently a LOCAL-TIME question, not a UTC one. A single per-country
# UTC offset would ALSO have been wrong: Indonesia alone spans three
# official civil time zones (WIB=UTC+7, WITA=UTC+8, WIT=UTC+9, confirmed
# directly -- Jakarta/Java vs. Bali/Sulawesi vs. Papua/Maluku), so a
# per-country constant would misclassify a meaningful fraction of
# Indonesian rows regardless of which single offset was chosen.
#
# Fixed here using each grid cell's own centroid_latitude/
# centroid_longitude (already present in the existing metadata files --
# confirmed directly, not assumed) via `timezonefinder`, which resolves
# true IANA civil time zones from coordinates (correctly distinguishing
# Asia/Jakarta vs. Asia/Makassar vs. Asia/Jayapura -- verified directly
# against known reference points spanning all three Indonesian zones,
# plus Manila and Mumbai, before use here).
#
# This is a DIFFERENT tool from compute_local_solar_hour (used elsewhere
# in this project, e.g. additional_experiments.py, for diurnal-cycle
# analysis) deliberately -- that function approximates SOLAR time from
# longitude alone, appropriate for physical/astronomical timing.
# Calendar/holiday features are CIVIL/LEGAL constructs tied to actual
# jurisdictional time zones, which follow political boundaries, not a
# smooth longitude gradient -- timezonefinder's IANA-boundary lookup is
# the correct tool for this, not a solar approximation.
#
# NOTE: MODIS fire date-matching is UNCHANGED and does NOT need this fix
# -- MOD14A1/MYD14A1 daily composites are themselves defined and
# processed by NASA using UTC calendar days, so matching against
# country_labels' UTC date is already consistent with how the underlying
# satellite product is defined, not a bug.
# ============================================================

def resolve_cell_timezones(metadata_df: pd.DataFrame, grid_id_col: str = "grid_id") -> pd.DataFrame:
    """Returns a (grid_id, timezone) lookup, ONE row per unique cell --
    timezone lookups are cached and computed once per cell, not once per
    row, since a cell's location (and therefore its timezone) never
    changes across its many hourly observations. Cached to
    TIMEZONE_CACHE_PATH across runs, since this project's full dataset
    has millions of rows but only a few hundred distinct cells."""
    from timezonefinder import TimezoneFinder

    unique_cells = metadata_df[[grid_id_col, "centroid_latitude", "centroid_longitude"]].drop_duplicates(subset=[grid_id_col])

    if TIMEZONE_CACHE_PATH.exists():
        cached = pd.read_csv(TIMEZONE_CACHE_PATH)
        already_resolved = set(cached[grid_id_col])
        unique_cells = unique_cells[~unique_cells[grid_id_col].isin(already_resolved)]
    else:
        cached = pd.DataFrame(columns=[grid_id_col, "timezone"])

    if len(unique_cells) > 0:
        print(f"  Resolving timezone for {len(unique_cells)} new cell(s) via timezonefinder...")
        tf = TimezoneFinder()
        new_rows = []
        for row in unique_cells.itertuples(index=False):
            lat, lon = getattr(row, "centroid_latitude"), getattr(row, "centroid_longitude")
            tz = tf.timezone_at(lat=lat, lng=lon)
            if tz is None:  # can happen for coordinates just offshore; fall back to the nearest timezone
                tz = tf.closest_timezone_at(lat=lat, lng=lon)
            new_rows.append({grid_id_col: getattr(row, grid_id_col), "timezone": tz})
        new_df = pd.DataFrame(new_rows)
        cached = pd.concat([cached, new_df], ignore_index=True)
        cached.to_csv(TIMEZONE_CACHE_PATH, index=False)
        print(f"  Wrote {TIMEZONE_CACHE_PATH.name} ({len(cached)} cell(s) total).")

    return cached[[grid_id_col, "timezone"]]


def add_local_datetime(df: pd.DataFrame, timezone_lookup: pd.DataFrame) -> pd.DataFrame:
    """Adds a 'datetime_local' column (tz-naive, representing each row's
    OWN local wall-clock time) by converting datetime_utc per row using
    that row's cell's timezone. Grouped by timezone (not looped per row)
    for efficiency -- there are only a handful of distinct timezones
    across this project's countries despite millions of rows."""
    df = df.merge(timezone_lookup, on="grid_id", how="left")
    if df["timezone"].isna().any():
        n_missing = df["timezone"].isna().sum()
        raise ValueError(f"{n_missing} row(s) have no resolved timezone -- check centroid_latitude/"
                          f"centroid_longitude coverage before proceeding.")

    dt_utc = pd.to_datetime(df["datetime_utc"], utc=True)
    local_dt = pd.Series(index=df.index, dtype="datetime64[ns]")
    for tz_name, group_idx in df.groupby("timezone").groups.items():
        local_dt.loc[group_idx] = dt_utc.loc[group_idx].dt.tz_convert(tz_name).dt.tz_localize(None)
    df["datetime_local"] = local_dt
    return df.drop(columns=["timezone"])


# ============================================================
# Stage: Calendar features (is_weekend, is_public_holiday) -- pure
# pandas, no GEE. Verified directly: the `holidays` library supports all
# 8 project countries, checked here rather than assumed.
# ============================================================

def build_calendar_features(df: pd.DataFrame, country_code: str, timezone_lookup: pd.DataFrame) -> pd.DataFrame:
    """df must have 'grid_id' and 'datetime_utc' columns. CORRECTED to
    compute every calendar feature from each row's LOCAL date (via
    datetime_local, see add_local_datetime), not the UTC date -- see the
    module-level correction note above add_local_datetime for why this
    matters and how it was verified."""
    df = add_local_datetime(df, timezone_lookup)
    dt_local = df["datetime_local"]
    local_date = dt_local.dt.date

    is_weekend = dt_local.dt.dayofweek.isin([5, 6])  # Saturday=5, Sunday=6

    years_needed = sorted(dt_local.dt.year.unique())
    # holidays library returns a Mapping[date, str] -- the string names
    # are used below to identify Lunar New Year, which moves ~10-11 days
    # earlier each Gregorian year and so cannot be matched by a fixed
    # (month, day) the way New Year's/Christmas can.
    country_holidays = holidays.country_holidays(country_code, years=years_needed)
    holiday_names = local_date.map(lambda d: country_holidays.get(d))
    is_public_holiday = holiday_names.notna()

    is_lunar_new_year = holiday_names.fillna("").str.lower().apply(
        lambda name: any(pattern in name for pattern in LUNAR_HOLIDAY_NAME_PATTERNS))
    # +/- 1 day window around the exact lunar holiday date, per the
    # original recommendation (fireworks/burning activity is not
    # confined to the exact calendar day).
    lunar_dates = set(local_date[is_lunar_new_year].unique())
    lunar_window_dates = set()
    for d in lunar_dates:
        lunar_window_dates.add(d - pd.Timedelta(days=1))
        lunar_window_dates.add(d)
        lunar_window_dates.add(d + pd.Timedelta(days=1))
    is_lunar_new_year_window = local_date.isin(lunar_window_dates)

    month = dt_local.dt.month
    day = dt_local.dt.day
    is_new_years_eve = (month == 12) & (day == 31)
    is_new_years_day = (month == 1) & (day == 1)
    is_christmas_eve = (month == 12) & (day == 24)
    is_christmas = (month == 12) & (day == 25)

    # Day-after-holiday: shifted per CELL using the cell's own local
    # dates (time-based per-cell lookup, matching this project's
    # established discipline for exactly this reason -- a data gap must
    # not silently pull in a non-adjacent day's holiday status).
    holiday_by_cell_date = df.assign(_is_holiday=is_public_holiday.values, _local_date=local_date.values)
    holiday_lookup = holiday_by_cell_date.drop_duplicates(subset=["grid_id", "_local_date"]).set_index(["grid_id", "_local_date"])["_is_holiday"]
    prev_date = pd.Series(local_date.values, index=df.index) - pd.Timedelta(days=1)
    lookup_keys = list(zip(df["grid_id"].values, prev_date.values))
    is_day_after_holiday = pd.Series([holiday_lookup.get(k, False) for k in lookup_keys], index=df.index)

    return pd.DataFrame({
        "grid_id": df["grid_id"].values, "datetime_utc": df["datetime_utc"].values,
        "is_weekend": is_weekend.astype(int).values,
        "is_public_holiday": is_public_holiday.astype(int).values,
        "is_new_years_eve": is_new_years_eve.astype(int).values,
        "is_new_years_day": is_new_years_day.astype(int).values,
        "is_christmas_eve": is_christmas_eve.astype(int).values,
        "is_christmas": is_christmas.astype(int).values,
        "is_lunar_new_year": is_lunar_new_year_window.astype(int).values,
        "is_day_after_holiday": is_day_after_holiday.astype(int).values,
    })


# ============================================================
# Stage: Consecutive dry days -- derived from precipitation_local
# already present in the EXISTING feature file, no new GEE extraction.
# Time-based (not positional) per-cell computation, matching this
# project's established discipline for exactly this reason (a data gap
# must not silently break the streak count against the wrong day).
# ============================================================

def compute_consecutive_dry_days(existing_features_df: pd.DataFrame, existing_metadata_df: pd.DataFrame,
                                   timezone_lookup: pd.DataFrame) -> pd.DataFrame:
    """DRY_DAY_PRECIP_THRESHOLD_MM=1.0 mm/day is the WMO-recognized
    definition of a 'dry day' -- documented here explicitly, not an
    arbitrary unexplained constant.

    CORRECTED to bucket by each cell's LOCAL calendar date, not UTC --
    daily precipitation totals (and the meteorological "dry day" concept
    itself) are conventionally defined on local calendar days, matching
    how meteorological services actually report them. See the
    module-level correction note above add_local_datetime for the full
    rationale and verification."""
    if len(existing_features_df) != len(existing_metadata_df):
        raise ValueError("Features/metadata row count mismatch -- cannot attach datetime positionally.")
    df = pd.concat([existing_features_df[["grid_id", "precipitation_local"]].reset_index(drop=True),
                     existing_metadata_df[["datetime_utc"]].reset_index(drop=True)], axis=1)
    df = add_local_datetime(df, timezone_lookup)
    df["_date"] = df["datetime_local"].dt.date

    daily_precip = df.groupby(["grid_id", "_date"])["precipitation_local"].sum().reset_index()
    daily_precip = daily_precip.sort_values(["grid_id", "_date"])
    daily_precip["is_dry"] = daily_precip["precipitation_local"] < DRY_DAY_PRECIP_THRESHOLD_MM

    # Consecutive-dry-day streak per cell: resets to 0 on any wet day,
    # increments by 1 on each dry day, and resets fresh at every cell
    # boundary (never continues a streak across cells). Deliberately a
    # SINGLE-level groupby, not nested -- an earlier version nested
    # groupby.apply() inside a per-group function, which pandas silently
    # returns as a wide DataFrame (one row per GROUP, one column per
    # original row) rather than the expected long Series when there's
    # only one group -- confirmed directly via a minimal reproduction
    # before writing this fix, not guessed at.
    block_change = ((daily_precip["is_dry"] != daily_precip.groupby("grid_id")["is_dry"].shift())
                     | (daily_precip["grid_id"] != daily_precip["grid_id"].shift()))
    block_id = block_change.cumsum()
    streak = daily_precip.groupby(block_id).cumcount() + 1
    daily_precip["consecutive_dry_days"] = streak.where(daily_precip["is_dry"], 0)

    # Broadcast the daily streak value back to every hourly row of that
    # date -- same broadcast-merge pattern already established in
    # extract_daily_predictors_batched.py for daily-cadence features.
    df = df.merge(daily_precip[["grid_id", "_date", "consecutive_dry_days"]], on=["grid_id", "_date"], how="left")
    return df[["grid_id", "datetime_utc", "consecutive_dry_days"]]


# ============================================================
# Stage: MODIS Active Fire extraction (MOD14A1 + MYD14A1, combined).
# Mirrors extract_daily_predictors_batched.py's date-grouped batching
# EXACTLY -- one reduceRegions() call per distinct calendar date per
# geometry/reducer combination, not per row. UNTESTED against live GEE in
# this environment (no credentials, no network access here) -- see module
# docstring.
# ============================================================

# ============================================================
# True-country resolution -- MISSING from the original version of this
# script, which is the root cause of a serious bug: without this filter,
# country_labels was built directly from the FULL source_features.csv
# (all 7 source countries combined) with COUNTRY_CODE only affecting
# output file NAMING, not which rows actually got processed. Confirmed
# directly from a real run's progress file, which contained VN grid_ids
# while COUNTRY_CODE="ID" -- every prior "single-country" run has
# actually been processing all 7 source countries' dates every time.
# Ported directly from extract_daily_predictors_batched.py's own
# load_true_country_lookup/fallback_country_from_grid_id, matching this
# project's established country-resolution convention exactly rather
# than inventing a new one.
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


def filter_to_country(country_labels: pd.DataFrame, country_code: str, sensor_grid_lookup_path: Path) -> pd.DataFrame:
    """Filters country_labels down to ONLY the rows belonging to
    country_code -- the step that was missing entirely. For the
    Philippines run, every row already belongs to PH by construction (a
    separate, single-country feature file), so this is a no-op safety
    check there rather than a real filter."""
    if country_code == "PH":
        return country_labels

    if not sensor_grid_lookup_path.exists():
        raise FileNotFoundError(
            f"{sensor_grid_lookup_path} not found -- required to correctly filter source_features.csv "
            f"down to a single country. Without this, ALL source countries get processed on every run, "
            f"exactly the bug this function fixes.")

    true_country_lookup = load_true_country_lookup(sensor_grid_lookup_path)
    resolved = country_labels["grid_id"].map(true_country_lookup)
    unresolved = resolved.isna()
    if unresolved.any():
        resolved.loc[unresolved] = country_labels.loc[unresolved, "grid_id"].apply(fallback_country_from_grid_id)

    filtered = country_labels[resolved == country_code].copy()
    print(f"  Filtered to true_country == '{country_code}': {len(filtered)}/{len(country_labels)} rows "
          f"({filtered['grid_id'].nunique()} distinct cells).")
    if len(filtered) == 0:
        raise ValueError(f"No rows resolved to country '{country_code}' -- check COUNTRY_CODE and "
                          f"{sensor_grid_lookup_path.name} before proceeding.")
    return filtered


def cell_centroid_point(grid_x_index: int, grid_y_index: int, working_proj):
    cx = grid_x_index * CELL_SIZE_M + CELL_SIZE_M / 2
    cy = grid_y_index * CELL_SIZE_M + CELL_SIZE_M / 2
    return ee.Geometry.Point([cx, cy], proj=working_proj)


def build_fire_feature_collections(cells: pd.DataFrame, working_proj) -> dict:
    """Local (small buffer, matching LOCAL_FALLBACK_BUFFER_RADIUS_M) plus
    5km and 10km neighborhood buffers -- METRIC circular buffers on the
    cell centroid, not the existing pipeline's grid-cell-count rectangle,
    since the feature request specifically asks for 5km/10km radii, a
    different (and more directly interpretable) scale than the existing
    neighborhood convention."""
    local_features, nbhd_5km_features, nbhd_10km_features = [], [], []
    for row in cells.itertuples(index=False):
        gx, gy = int(row.grid_x_index), int(row.grid_y_index)
        centroid = cell_centroid_point(gx, gy, working_proj)
        local_features.append(ee.Feature(centroid.buffer(LOCAL_FALLBACK_BUFFER_RADIUS_M), {"grid_id": row.grid_id}))
        nbhd_5km_features.append(ee.Feature(centroid.buffer(NEIGHBORHOOD_RADII_M["5km"]), {"grid_id": row.grid_id}))
        nbhd_10km_features.append(ee.Feature(centroid.buffer(NEIGHBORHOOD_RADII_M["10km"]), {"grid_id": row.grid_id}))
    return {
        "local": ee.FeatureCollection(local_features),
        "5km": ee.FeatureCollection(nbhd_5km_features),
        "10km": ee.FeatureCollection(nbhd_10km_features),
    }


def get_combined_fire_image(date_iso: str):
    """Terra (MOD14A1) + Aqua (MYD14A1) combined via per-pixel MAX --
    matches the "if either satellite detected fire, count it" principle
    already used by NASA's own compositing philosophy for these products
    (confirmed during design research: MYD14A1/VNP14A1's own documented
    compositing uses a similar maximum-value-composite approach).
    Direct calendar-date match, NOT the existing pipeline's multi-day
    fallback window -- MOD14A1/MYD14A1 are true daily composites with a
    value for every day (unlike AOD, which has frequent cloud-driven
    gaps), so a wide fallback window would incorrectly smear a fire
    signal across days it didn't occur on."""
    target_date = ee.Date(date_iso)
    next_date = target_date.advance(1, "day")

    terra = ee.ImageCollection(MOD14A1_COLLECTION).filterDate(target_date, next_date).select(["FireMask", "MaxFRP"]).first()
    aqua = ee.ImageCollection(MYD14A1_COLLECTION).filterDate(target_date, next_date).select(["FireMask", "MaxFRP"]).first()

    combined = ee.ImageCollection([ee.Image(terra), ee.Image(aqua)]).max()
    fire_mask = combined.select("FireMask")
    frp = combined.select("MaxFRP")

    is_fire = fire_mask.gte(FIRE_DETECTION_THRESHOLD)
    # MaxFRP masked to fire pixels ONLY -- otherwise mean/sum FRP would be
    # diluted by the many non-fire pixels in each buffer, which carry a
    # MaxFRP value of 0 or no-data rather than genuinely low FRP.
    frp_masked = frp.updateMask(is_fire)

    return is_fire.rename("is_fire"), frp_masked.rename("frp_masked")


def extract_one_fire_date(date_iso: str, cells_for_date: pd.DataFrame, working_proj) -> dict:
    is_fire_img, frp_img = get_combined_fire_image(date_iso)
    fcs = build_fire_feature_collections(cells_for_date, working_proj)

    results = {gid: {} for gid in cells_for_date["grid_id"]}

    for scale_name, fc in fcs.items():
        suffix = "" if scale_name == "local" else f"_{scale_name}"

        count_result = is_fire_img.reduceRegions(collection=fc, reducer=ee.Reducer.sum(), scale=CELL_SIZE_M, crs=working_proj).getInfo()
        max_frp_result = frp_img.reduceRegions(collection=fc, reducer=ee.Reducer.max(), scale=CELL_SIZE_M, crs=working_proj).getInfo()
        mean_frp_result = frp_img.reduceRegions(collection=fc, reducer=ee.Reducer.mean(), scale=CELL_SIZE_M, crs=working_proj).getInfo()
        sum_frp_result = frp_img.reduceRegions(collection=fc, reducer=ee.Reducer.sum(), scale=CELL_SIZE_M, crs=working_proj).getInfo()

        count_by_gid = {f["properties"]["grid_id"]: f["properties"].get("sum", 0) for f in count_result["features"]}
        max_frp_by_gid = {f["properties"]["grid_id"]: f["properties"].get("max") for f in max_frp_result["features"]}
        mean_frp_by_gid = {f["properties"]["grid_id"]: f["properties"].get("mean") for f in mean_frp_result["features"]}
        sum_frp_by_gid = {f["properties"]["grid_id"]: f["properties"].get("sum") for f in sum_frp_result["features"]}

        for gid in cells_for_date["grid_id"]:
            hotspot_count = count_by_gid.get(gid, 0) or 0
            results[gid][f"hotspot_count{suffix}"] = hotspot_count
            results[gid][f"fire_present{suffix}"] = int(hotspot_count > 0)
            if scale_name != "local":
                results[gid][f"max_frp{suffix}"] = max_frp_by_gid.get(gid)
            else:
                results[gid]["max_frp"] = max_frp_by_gid.get(gid)
                results[gid]["mean_frp"] = mean_frp_by_gid.get(gid)
                results[gid]["sum_frp"] = sum_frp_by_gid.get(gid)

    return results


# ============================================================
# Resumability helpers -- mirrors extract_daily_predictors_batched.py's
# own load_already_done_dates / append_to_progress / try-finally pattern
# exactly, applied here to fire extraction specifically.
# ============================================================

def load_already_assembled_countries(tracker_path: Path) -> set:
    """Same atomic-completion principle as load_already_done_fire_dates,
    applied one level up: a country is either fully appended to the
    final multi-country output AND recorded here, or neither -- never a
    partial/duplicated state."""
    if not tracker_path.exists():
        return set()
    content = tracker_path.read_text().strip()
    return set(content.split("\n")) if content else set()


def append_country_to_final_output(country_df: pd.DataFrame, output_path: Path, tracker_path: Path, country_code: str):
    """Appends this country's rows to the persistent, multi-country
    output file (creating it with a header on the very first country,
    appending without a header afterward) -- FIXES a real bug in the
    previous version, which called .to_csv(output_path) directly, i.e.
    OVERWRITING the entire file with only the current run's country every
    single time. That meant running country B after country A silently
    destroyed country A's already-completed rows from the final output
    (though NOT from that country's own fire_features_{CODE}_progress.csv,
    which was and remains untouched -- no raw extraction work was ever
    lost, only the final assembly step was wrong).

    Guarded by the SAME tracker file this function writes to: if this
    country was already assembled in a previous run, this is a no-op --
    prevents duplicate rows if a country's assembly step is ever re-run
    (e.g. after an interruption between writing progress and marking
    the country complete)."""
    already_assembled = load_already_assembled_countries(tracker_path)
    if country_code in already_assembled:
        print(f"  {country_code} already assembled into {output_path.name} (found in {tracker_path.name}) -- skipping, not duplicating.")
        return

    write_header = not output_path.exists()
    country_df.to_csv(output_path, mode="a", header=write_header, index=False)
    with open(tracker_path, "a") as f:
        f.write(country_code + "\n")
    print(f"  Appended {len(country_df)} rows for {country_code} to {output_path.name} "
          f"(now includes: {sorted(already_assembled | {country_code})}).")


def load_already_done_fire_dates(progress_path: Path) -> set:
    if not progress_path.exists():
        return set()
    existing = pd.read_csv(progress_path, usecols=["date"])
    return set(existing["date"].astype(str).unique())


def append_fire_progress(progress_path: Path, date_str: str, date_result: dict):
    rows = [{"grid_id": gid, "date": date_str, **values} for gid, values in date_result.items()]
    df = pd.DataFrame(rows)
    write_header = not progress_path.exists()
    df.to_csv(progress_path, mode="a", header=write_header, index=False)


def add_rolling_fire_features(progress_df: pd.DataFrame) -> pd.DataFrame:
    """24h/72h/7-day rolling accumulation of hotspot_count, per cell,
    time-based (not positional) -- same discipline as every rolling
    feature elsewhere in this project, so a missing date correctly
    produces a smaller-but-honest window rather than silently pulling in
    a non-adjacent date's value."""
    progress_df = progress_df.sort_values(["grid_id", "date"]).reset_index(drop=True)
    dt = pd.to_datetime(progress_df["date"], utc=True)

    new_cols = {}
    for window_label, window_str in ROLLING_WINDOWS_FIRE.items():
        col_name = f"rolling_{window_label}_hotspot_count"
        s = pd.Series(progress_df["hotspot_count"].values, index=dt)
        new_cols[col_name] = (
            s.groupby(progress_df["grid_id"].values)
             .apply(lambda g: g.rolling(window_str, min_periods=1).sum())
             .values
        )
    return pd.concat([progress_df, pd.DataFrame(new_cols, index=progress_df.index)], axis=1)


def process_fire_extraction_for_country(country_labels: pd.DataFrame, grid_summary: pd.DataFrame, working_proj):
    """One country per run, matching the established convention. Returns
    the final fire-features DataFrame for this country (broadcast-merged
    back to every original hourly row, with rolling features added)."""
    cells = country_labels[["grid_id"]].drop_duplicates().merge(grid_summary, on="grid_id", how="left")
    cells = cells.dropna(subset=["grid_x_index", "grid_y_index"])
    cell_index = cells.set_index("grid_id")

    country_labels = country_labels.copy()
    country_labels["date"] = pd.to_datetime(country_labels["datetime_utc"]).dt.date.astype(str)
    cells_per_date = country_labels[["grid_id", "date"]].drop_duplicates().merge(
        cell_index[["grid_x_index", "grid_y_index"]], on="grid_id", how="inner"
    ).groupby("date")

    all_distinct_dates = sorted(cells_per_date.groups.keys())
    print(f"  {len(country_labels)} label rows -> {len(all_distinct_dates)} distinct date(s), "
          f"{cells['grid_id'].nunique()} distinct grid cell(s).")

    already_done = load_already_done_fire_dates(FIRE_PROGRESS_PATH)
    remaining_dates = [d for d in all_distinct_dates if d not in already_done]
    print(f"  {len(already_done)} date(s) already completed (found in {FIRE_PROGRESS_PATH.name}) -- "
          f"{len(remaining_dates)} remaining.")

    if TEST_MODE:
        remaining_dates = remaining_dates[:TEST_MODE_N_DATES]
        print(f"  TEST_MODE is on -- limiting to the first {len(remaining_dates)} remaining date(s).")

    def process_one_date(date_str):
        cells_for_date = cells_per_date.get_group(date_str)[["grid_id", "grid_x_index", "grid_y_index"]]
        return date_str, extract_one_fire_date(date_str, cells_for_date, working_proj)

    n_completed, n_failed = 0, 0
    try:
        if remaining_dates:
            with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_DATES) as executor:
                futures = {executor.submit(process_one_date, d): d for d in remaining_dates}
                for future in as_completed(futures):
                    date_str = futures[future]
                    try:
                        _, date_result = future.result()
                    except Exception as e:
                        n_failed += 1
                        print(f"    ERROR on {date_str}: {e} -- will retry on next run.")
                        continue
                    append_fire_progress(FIRE_PROGRESS_PATH, date_str, date_result)
                    n_completed += 1
                    print(f"    [{n_completed + len(already_done)}/{len(all_distinct_dates)}] "
                          f"{date_str} done and saved to {FIRE_PROGRESS_PATH.name}.")
        else:
            print("  Nothing left to process for fire extraction -- already fully complete.")
    finally:
        print(f"\n  Regenerating fire report from current progress...")
        report_lines = [
            f"Fire Feature Extraction -- Report ({COUNTRY_CODE})", "=" * 45,
            f"Total distinct dates: {len(all_distinct_dates)}",
            f"Completed before this session: {len(already_done)}",
            f"Completed this session: {n_completed}",
            f"Failed this session (will retry next run): {n_failed}",
        ]
        FIRE_REPORT_PATH.write_text("\n".join(report_lines))
        print(f"  Wrote {FIRE_REPORT_PATH.name}")

    if not FIRE_PROGRESS_PATH.exists():
        return None
    progress_df = pd.read_csv(FIRE_PROGRESS_PATH)
    progress_with_rolling = add_rolling_fire_features(progress_df)

    country_labels["date"] = country_labels["date"].astype(str)
    final = country_labels.merge(progress_with_rolling, on=["grid_id", "date"], how="left")
    return final.drop(columns=["date"], errors="ignore")


# ============================================================
# Diagnostics -- computed on the FINAL assembled additional-features
# file, comparing top-10% PM2.5 rows against the remaining 90%, per
# explicit instruction, to give an early read on which features look
# informative before any retraining begins.
# ============================================================

def generate_diagnostics(additional_df: pd.DataFrame, pm25_lookup_df: pd.DataFrame, target_col: str = "pm25"):
    """pm25_lookup_df must contain grid_id, datetime_utc, and target_col --
    NOT required to already be filtered/aligned to additional_df's rows.
    This function does its own explicit key-based merge and verifies
    coverage, rather than assuming positional alignment from the caller.

    FIXES a real bug: an earlier version assumed the caller had already
    aligned target_df to additional_df's exact row set, but the actual
    call site passed the FULL, unfiltered features_df (all 7 source
    countries, 7.1M rows) against a single country's already-filtered
    additional_df (146K rows) -- a length mismatch that crashed with a
    boolean-indexer error at runtime. Doing the join explicitly here,
    rather than requiring the caller to get it right, removes an entire
    class of this kind of bug."""
    feature_cols = [c for c in additional_df.columns if c not in ("grid_id", "datetime_utc")]

    merged = additional_df.merge(pm25_lookup_df[["grid_id", "datetime_utc", target_col]],
                                   on=["grid_id", "datetime_utc"], how="left")
    n_unmatched = merged[target_col].isna().sum()
    if n_unmatched > 0:
        print(f"  WARNING: {n_unmatched}/{len(merged)} rows in additional_df had no matching "
              f"{target_col} value after joining on (grid_id, datetime_utc) -- these rows are "
              f"excluded from the extreme-vs-ordinary comparison below (kept in additional_feature_summary.csv,"
              f" which doesn't need the target).")

    summary_rows = []
    for col in feature_cols:
        series = additional_df[col]
        summary_rows.append({
            "feature": col, "missing_pct": series.isna().mean() * 100,
            "mean": series.mean(), "std": series.std(), "min": series.min(), "max": series.max(),
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(SUMMARY_OUTPUT_PATH, index=False)

    high_pm_threshold = merged[target_col].quantile(0.90)
    high_pm_mask = merged[target_col] >= high_pm_threshold

    extreme_rows = []
    for col in feature_cols:
        high_vals = merged.loc[high_pm_mask, col]
        low_vals = merged.loc[~high_pm_mask, col]
        extreme_rows.append({
            "feature": col,
            "top10pct_mean": high_vals.mean(), "top10pct_std": high_vals.std(),
            "remaining90pct_mean": low_vals.mean(), "remaining90pct_std": low_vals.std(),
            "mean_difference": high_vals.mean() - low_vals.mean(),
            "top10pct_missing_pct": high_vals.isna().mean() * 100,
            "remaining90pct_missing_pct": low_vals.isna().mean() * 100,
        })
    extreme_df = pd.DataFrame(extreme_rows).sort_values("mean_difference", key=abs, ascending=False)
    extreme_df.to_csv(EXTREME_STATS_OUTPUT_PATH, index=False)

    print(f"Wrote {SUMMARY_OUTPUT_PATH.name} and {EXTREME_STATS_OUTPUT_PATH.name}")
    return summary_df, extreme_df


# ============================================================
# Final assembly: combine fire + dry-days + calendar into ONE additional
# features file per domain, joinable on (grid_id, datetime_utc). The
# ORIGINAL feature files are only ever READ here, never written.
# ============================================================

def build_additional_features_for_domain(features_path: Path, metadata_path: Path, country_code: str,
                                            country_labels: pd.DataFrame, fire_features_df: pd.DataFrame = None) -> pd.DataFrame:
    """country_labels must already be filtered to this country's own
    rows (see filter_to_country) -- an earlier version of this function
    re-read the FULL features_path/metadata_path from disk directly,
    which for a source-domain run means ALL 7 countries, not just
    country_code. That would have produced the same class of bug fixed
    in main() (all countries reprocessed every run), just for
    dry-days/calendar instead of fire -- caught and fixed in the same
    pass, before it caused the duplication it would have caused once a
    second country's run occurred."""
    existing_features = pd.read_csv(features_path)
    existing_metadata = pd.read_csv(metadata_path)
    if len(existing_features) != len(existing_metadata):
        raise ValueError(f"{features_path.name}/{metadata_path.name} row count mismatch.")
    if "centroid_latitude" not in existing_metadata.columns or "centroid_longitude" not in existing_metadata.columns:
        raise ValueError(f"{metadata_path.name} is missing centroid_latitude/centroid_longitude -- required "
                          f"for correct per-cell local-time resolution (see add_local_datetime).")

    full_base_df = pd.concat([existing_features[["grid_id", "precipitation_local"]].reset_index(drop=True),
                               existing_metadata[["datetime_utc", "centroid_latitude", "centroid_longitude"]].reset_index(drop=True)], axis=1)

    # Restrict to exactly this country's (grid_id, datetime_utc) pairs,
    # via an inner join against the already-filtered country_labels --
    # this is the actual fix.
    filtered_base = full_base_df.merge(country_labels[["grid_id", "datetime_utc"]], on=["grid_id", "datetime_utc"], how="inner")
    filtered_metadata = filtered_base[["grid_id", "datetime_utc"]]

    # Resolve each cell's timezone ONCE here (cached across runs via
    # TIMEZONE_CACHE_PATH), then reused for BOTH dry-days and calendar
    # features below -- the actual fix for the local-date bug.
    timezone_lookup = resolve_cell_timezones(filtered_base[["grid_id", "centroid_latitude", "centroid_longitude"]])

    dry_days_df = compute_consecutive_dry_days(filtered_base[["grid_id", "precipitation_local"]], filtered_metadata, timezone_lookup)
    calendar_df = build_calendar_features(filtered_metadata, country_code, timezone_lookup)

    result = filtered_metadata.merge(dry_days_df, on=["grid_id", "datetime_utc"], how="left")
    result = result.merge(calendar_df, on=["grid_id", "datetime_utc"], how="left")
    if fire_features_df is not None:
        fire_cols = [c for c in fire_features_df.columns if c not in ("grid_id", "datetime_utc")]
        result = result.merge(fire_features_df[["grid_id", "datetime_utc"] + fire_cols],
                                on=["grid_id", "datetime_utc"], how="left")
    return result


# ============================================================
# Main
# ============================================================

def main():
    print("Loading source dataset...")
    ee.Initialize(project=EE_PROJECT_ID)

    if not GRID_SUMMARY_PATH.exists() or not SENSOR_GRID_LOOKUP_PATH.exists():
        raise FileNotFoundError(f"Expected {GRID_SUMMARY_PATH} and {SENSOR_GRID_LOOKUP_PATH}.")

    features_path = PH_FEATURES_PATH if IS_PHILIPPINES_RUN else SOURCE_FEATURES_PATH
    metadata_path = PH_METADATA_PATH if IS_PHILIPPINES_RUN else SOURCE_METADATA_PATH
    output_path = PH_OUTPUT_PATH if IS_PHILIPPINES_RUN else SOURCE_OUTPUT_PATH

    grid_summary = pd.read_csv(GRID_SUMMARY_PATH, usecols=["grid_id", "grid_x_index", "grid_y_index"])
    features_df = pd.read_csv(features_path)
    metadata_df = pd.read_csv(metadata_path)
    country_labels = pd.concat([features_df[["grid_id"]].reset_index(drop=True),
                                  metadata_df[["datetime_utc"]].reset_index(drop=True)], axis=1)

    # THE FIX: restrict to COUNTRY_CODE's own rows only -- this call was
    # entirely missing before, which meant every run processed all 7
    # source countries regardless of COUNTRY_CODE. See filter_to_country's
    # own docstring for how this was discovered.
    country_labels = filter_to_country(country_labels, COUNTRY_CODE, SENSOR_GRID_LOOKUP_PATH)

    print(f"Building dry day features for {COUNTRY_CODE}...")
    print(f"Processing {COUNTRY_CODE}...")

    working_proj = ee.Projection("EPSG:4326")  # simple geographic CRS for buffer-based (not rectangle) geometries;
                                                 # metric buffers on a point are well-defined without a custom
                                                 # equal-area projection the way the rectangle-based neighborhood
                                                 # convention required one

    print("Collecting MODIS fire observations...")
    fire_df = process_fire_extraction_for_country(country_labels, grid_summary, working_proj)

    print(f"Finished {COUNTRY_CODE}...")

    additional_df = build_additional_features_for_domain(features_path, metadata_path, COUNTRY_CODE, country_labels, fire_df)
    tracker_path = PH_ASSEMBLED_COUNTRIES_PATH if IS_PHILIPPINES_RUN else SOURCE_ASSEMBLED_COUNTRIES_PATH
    append_country_to_final_output(additional_df, output_path, tracker_path, COUNTRY_CODE)
    print(f"Export complete.")

    if "pm25" in features_df.columns:
        pm25_lookup_df = pd.concat([features_df[["grid_id", "pm25"]].reset_index(drop=True),
                                      metadata_df[["datetime_utc"]].reset_index(drop=True)], axis=1)
        generate_diagnostics(additional_df, pm25_lookup_df)


if __name__ == "__main__":
    main()
