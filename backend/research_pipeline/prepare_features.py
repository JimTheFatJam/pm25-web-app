"""
Feature Preparation: Statistics, Fallback, Rolling Averages, Encoding
==========================================================================

Takes the two integrated training files (training_dataset_source_domains.csv,
training_dataset_philippines.csv) and prepares them for model training,
strictly in this order:

  1. Data statistics report -- missingness, dtype, and basic distributional
     stats for every column, written to a .txt file, computed on the RAW
     integrated data BEFORE any of the transformations below touch it.
  2. Bidirectional local <-> neighborhood fallback, with an indicator
     column recording every substitution.
  3. Rolling/lagged averages -- ONLY on genuinely hourly or daily dynamic
     predictors, never on static or annual/monthly ones (a rolling mean of
     a constant is just the constant, and would be a wasted, redundant
     column). Applied to predictors only, never to the pm25 target itself
     -- a lagged-PM2.5 feature would be undefined at any unmonitored
     location, exactly where this study's national prediction grid needs
     to produce estimates, and would silently break at the moment the
     model matters most.
  4. Cyclical (sine/cosine) time encoding of hour-of-day, day-of-week, and
     month-of-year, replacing raw integer time components as model inputs.
  5. Land cover one-hot encoding -- see below.
  6. Metadata/feature separation -- grid_id is kept in both outputs (per
     explicit instruction) as an identifier, not a feature; everything else
     non-predictive (datetime_utc, country, centroid coordinates, raw
     year/month, n_sensors_contributing) is split into its own file,
     row-aligned with the feature file, rather than discarded.

pm25 is NEVER modified by this script -- explicit decision
------------------------------------------------------------------------------
Exact-zero and extreme (>=900) pm25 values are left completely unchanged,
by deliberate decision following a dedicated data quality investigation
(see investigate_data_quality.py and its reports). Extreme values are not
threshold-filtered, since genuine extreme pollution events, though rare,
do occur, and several specific repeated values (985.00, 987.00, 999.99,
1000.00) that look like likely sensor/reporting artifacts concentrated in
a small number of grid cells (primarily India and Vietnam) are NOT
documented as official sentinel values by OpenAQ or the source monitoring
networks -- treating them as such and removing them would be an
undocumented cleaning assumption. Exact zeros are also left unchanged in
both files: the investigation found zero rows in the Philippines file
associated with physically plausible conditions (higher wind speed, lower
AOD than average), while source-domain zeros looked more suspicious
(higher AOD than average, 88% concentrated in India alone) -- but this is
not sufficient evidence to safely discriminate genuine from erroneous
zeros at the individual-row level. Both patterns are intended to be
documented in the thesis's data quality / limitations discussion, not
acted on silently here.

Land cover one-hot encoding
--------------------------------
landcover_local and landcover_neighborhood_mode are nominal IGBP class
codes (e.g. 9 = Savannas, 13 = Urban, 17 = Water) -- there is no sense in
which a larger code number means "more" of anything, so leaving them as
raw integers would let a tree-based model find splits like "code < 12.5"
that have no real ordinal meaning. One-hot encoded instead: each class
becomes its own True/False column. All one-hot columns are kept (no
reference category dropped) since tree-based ensembles do not suffer the
linear-model dummy-variable-trap the way a regression model would.

Classes below 1% prevalence are grouped into a shared "_other" column,
EXCEPT that a class is kept as its own separate column if it reaches 1%
prevalence in EITHER file, not just the pooled/combined prevalence across
both. This distinction matters concretely: land cover class 17 (Water)
is only 0.33% of the source file, but 10.08% of the Philippines file --
the actual target domain this thesis is about. Its POOLED prevalence
across both files combined is 0.92%, just under a naive 1% cutoff, which
would have silently discarded it as "other" despite it being a real,
substantial, physically meaningful category (coastal/archipelago cells)
specifically where it matters most. The encoding scheme (which classes
are kept separate vs. grouped) is determined ONCE from both files
together and applied identically to both, so the two files always share
the exact same set of one-hot columns.

Rolling average variable scope (52 new columns, confirmed with the user
before implementing)
--------------------------------------------------------------------------------
Hourly ERA5 group (7 variables: temperature_2m, dewpoint_2m,
surface_pressure, u_wind_10m, v_wind_10m, precipitation,
boundary_layer_height) x 2 scales (local, neighborhood_mean) x 2 windows
(3-hour, 24-hour) = 28 columns. Rolling computed directly on the hourly row
sequence per grid cell, using a TIME-based window (not a fixed row count),
so gaps in a cell's hourly coverage don't silently distort the window.

Daily AOD/gas group (6 variables: aod, no2, so2, co, o3, hcho) x 2 scales x
2 windows (3-day, 7-day) = 24 columns. These are already held constant
across all 24 hours of their calendar date (per the daily extraction
design), so rolling is computed ONCE per (grid_id, date) on the
deduplicated daily sequence, then broadcast back to every hourly row of
that date -- mirroring the same "compute at native cadence, broadcast to
hours" pattern already used throughout this pipeline, rather than rolling
over the raw hourly rows (which would need a 72/168-row window and gain
nothing, since 24 consecutive rows per date are identical anyway).

NOT included in this rolling scope, deliberately: relative_humidity and
wind_speed_10m (derived hourly variables, not part of the originally
confirmed 52-column scope -- can be added later if wanted, but excluded
here to match what was explicitly agreed), and every static/annual/monthly
predictor (elevation, roads, coast distance, population, land cover,
nighttime lights, NDVI) and pm25 itself.

Bidirectional fallback: an honest asymmetry
------------------------------------------------
Local-to-neighborhood substitution (already established in this pipeline)
is a reasonably strong estimate -- a genuine 25-cell spatial mean.
Neighborhood-to-local (the new, reverse direction) is a weaker
substitution -- extrapolating one point outward to stand in for a 25-cell
average -- kept anyway as a defensive last resort, but confirmed (against
the real Philippines file) to occur ZERO times across every variable
checked, consistent with the neighborhood window geometrically including
the local cell itself. Every substitution, in either direction, is flagged
so this can be checked rather than assumed.

Usage
-----
Place this script alongside training_dataset_source_domains.csv and
training_dataset_philippines.csv, and run it.

Output, per input file (using SOURCE/PHILIPPINES as the relevant name):
  - <name>_statistics_report.txt
  - <name>_features.csv (grid_id + all model-ready feature columns)
  - <name>_metadata.csv (grid_id + datetime_utc, country, centroid
    coordinates, raw year/month, n_sensors_contributing -- row-aligned
    with the features file via row order)
  - landcover_encoding_scheme.txt (which land cover classes were kept
    separate vs. grouped into "_other", and why)

Dependencies: pandas, numpy
    pip install pandas numpy
"""

from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================
# Configuration -- MACROS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent

INPUT_FILES = {
    "SOURCE": SCRIPT_DIR / "training_dataset_source_domains.csv",
    "PHILIPPINES": SCRIPT_DIR / "training_dataset_philippines.csv",
}

LANDCOVER_COLUMNS = ["landcover_local", "landcover_neighborhood_mode"]
LANDCOVER_MIN_PREVALENCE_PCT = 1.0  # kept separate if >= this in EITHER file
LANDCOVER_IGBP_LABELS = {
    1: "Evergreen_Needleleaf_Forest", 2: "Evergreen_Broadleaf_Forest", 3: "Deciduous_Needleleaf_Forest",
    4: "Deciduous_Broadleaf_Forest", 5: "Mixed_Forest", 6: "Closed_Shrublands", 7: "Open_Shrublands",
    8: "Woody_Savannas", 9: "Savannas", 10: "Grasslands", 11: "Permanent_Wetlands", 12: "Croplands",
    13: "Urban_and_Built_up", 14: "Cropland_Natural_Vegetation_Mosaic", 15: "Snow_and_Ice",
    16: "Barren_or_Sparsely_Vegetated", 17: "Water",
}

HOURLY_ERA5_VARS = ["temperature_2m", "dewpoint_2m", "surface_pressure",
                    "u_wind_10m", "v_wind_10m", "precipitation", "boundary_layer_height"]
HOURLY_ROLLING_WINDOWS = {"3h": "3h", "24h": "24h"}

DAILY_DYNAMIC_VARS = ["aod", "no2", "so2", "co", "o3", "hcho"]
DAILY_ROLLING_WINDOWS = {"3d": "3D", "7d": "7D"}

# Column densities/AOD cannot be physically negative; small negative values
# are a documented characteristic of Sentinel-5P retrieval noise near
# true-zero conditions, not a genuine measurement. Clipped to zero rather
# than left as-is (per explicit decision) -- applied generically to all six
# variables, even though only NO2/SO2/HCHO showed negatives in the data
# checked so far, since clipping an already-non-negative column is a no-op.
GAS_CLIP_VARS = DAILY_DYNAMIC_VARS

# Variables needing the bidirectional local<->neighborhood fallback.
# NDVI and population are quasi-static/annual predictors that ALSO have a
# local/neighborhood_mean pair and ALSO showed real missingness (6.86% and
# 3.95% respectively, for the Philippines) -- confirmed directly that every
# one of those missing rows has a valid neighborhood value available, so
# adding them here resolves that gap completely using the same mechanism
# already built for the hourly/daily dynamic predictors, not a new one.
FALLBACK_VARS = HOURLY_ERA5_VARS + DAILY_DYNAMIC_VARS + ["ndvi", "population"]

SCALES = ["local", "neighborhood_mean"]

# Reverse-direction fallback flags (*_neighborhood_filled_from_local)
# confirmed constant-zero across the entire Philippines file (every one of
# the 13 checked), meaning they carry no information a model could use --
# dropped from the final feature set. road_density_local_m_per_km2 is
# confirmed identical to road_length_local_m (local cell area is always
# exactly 1 km^2, so dividing by it changes nothing) -- dropped, keeping
# road_length_local_m. The neighborhood-scale versions of both are kept,
# since neighborhood area is 25 km^2, so density and length genuinely
# differ there.
COLUMNS_TO_DROP_FROM_FEATURES_SUFFIX = ["_neighborhood_filled_from_local"]
COLUMNS_TO_DROP_FROM_FEATURES_EXACT = ["road_density_local_m_per_km2"]

METADATA_COLUMNS = ["datetime_utc", "country", "centroid_latitude", "centroid_longitude",
                    "year", "month", "n_sensors_contributing"]

TARGET_COLUMN = "pm25"


# ============================================================
# Stage 0: Land cover one-hot encoding scheme (determined ONCE from both files)
# ============================================================

def determine_landcover_scheme(paths: dict) -> dict:
    """Reads ONLY the land cover columns (cheap, even at 7.1M rows) from
    every available file, computes each class's prevalence WITHIN EACH
    FILE SEPARATELY, and keeps a class as its own one-hot column if it
    reaches LANDCOVER_MIN_PREVALENCE_PCT in EITHER file -- not the pooled
    prevalence across both combined, which would incorrectly discard
    classes that are rare overall but important in one specific domain
    (confirmed concretely: Water is 0.33% of source but 10.08% of
    Philippines; pooled, it's 0.92%, just under a naive 1% cutoff)."""
    per_file_pct = {}
    for name, path in paths.items():
        if not path.exists():
            continue
        available_cols = pd.read_csv(path, nrows=0).columns
        cols = [c for c in LANDCOVER_COLUMNS if c in available_cols]
        if not cols:
            continue
        lc = pd.read_csv(path, usecols=cols)
        for col in cols:
            counts = lc[col].value_counts(normalize=True) * 100
            for code, pct in counts.items():
                per_file_pct.setdefault(col, {}).setdefault(int(code), {})[name] = pct

    scheme = {}
    report_lines = ["Land Cover One-Hot Encoding Scheme", "=" * 34, "",
                     f"A class is kept as its own column if it reaches "
                     f"{LANDCOVER_MIN_PREVALENCE_PCT}% prevalence in EITHER file "
                     f"individually (not pooled across both).", ""]
    for col, classes in per_file_pct.items():
        keep_separate = []
        report_lines.append(f"{col}:")
        for code in sorted(classes):
            pcts = classes[code]
            max_pct = max(pcts.values())
            decision = "KEEP" if max_pct >= LANDCOVER_MIN_PREVALENCE_PCT else "-> other"
            label = LANDCOVER_IGBP_LABELS.get(code, f"code_{code}")
            pct_str = ", ".join(f"{name}={pct:.2f}%" for name, pct in pcts.items())
            report_lines.append(f"  {code:>2} ({label:<32}) {pct_str:<30} {decision}")
            if max_pct >= LANDCOVER_MIN_PREVALENCE_PCT:
                keep_separate.append(code)
        scheme[col] = sorted(keep_separate)
        report_lines.append("")

    report_path = SCRIPT_DIR / "landcover_encoding_scheme.txt"
    report_path.write_text("\n".join(report_lines))
    print(f"  Wrote {report_path.name} (documents which classes were kept separate vs. grouped, and why)")
    return scheme


def one_hot_encode_landcover(df: pd.DataFrame, scheme: dict) -> pd.DataFrame:
    """Applies the shared scheme (determined once across both files) to
    one-hot encode each land cover column, dropping the original raw
    integer column afterward -- leaving it in would silently reintroduce
    the exact ordinal-encoding problem this step exists to fix. All
    one-hot columns are kept (no reference category dropped), since
    tree-based ensembles don't have the linear-model dummy-variable-trap
    issue that convention exists to avoid."""
    new_cols = {}
    for col, keep_codes in scheme.items():
        if col not in df.columns:
            continue
        for code in keep_codes:
            label = LANDCOVER_IGBP_LABELS.get(code, f"code_{code}")
            new_cols[f"{col}_{code}_{label}"] = (df[col] == code)
        new_cols[f"{col}_other"] = ~df[col].isin(keep_codes)
    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    df = df.drop(columns=[c for c in scheme if c in df.columns])
    return df




def write_statistics_report(df: pd.DataFrame, name: str):
    lines = [f"Data Statistics Report -- {name}", "=" * (24 + len(name)), "",
             f"Total rows: {len(df)}", f"Total columns: {len(df.columns)}",
             f"Distinct grid cells: {df['grid_id'].nunique()}", ""]

    lines.append(f"{'Column':<45}{'Dtype':<10}{'Missing':>10}{'Missing %':>12}")
    lines.append("-" * 77)
    for col in df.columns:
        n_missing = df[col].isna().sum()
        pct_missing = n_missing / len(df) * 100
        lines.append(f"{col:<45}{str(df[col].dtype):<10}{n_missing:>10}{pct_missing:>11.2f}%")

    lines.append("")
    lines.append("Numeric column summary statistics (mean, std, min, max):")
    lines.append("-" * 77)
    numeric_df = df.select_dtypes(include=[np.number])
    for col in numeric_df.columns:
        s = numeric_df[col]
        if s.notna().sum() == 0:
            lines.append(f"{col:<45} all values missing")
            continue
        lines.append(f"{col:<45} mean={s.mean():.4g}  std={s.std():.4g}  min={s.min():.4g}  max={s.max():.4g}")

    report_path = SCRIPT_DIR / f"{name.lower()}_statistics_report.txt"
    report_path.write_text("\n".join(lines))
    print(f"  Wrote {report_path.name}")


# ============================================================
# Stage 1.5: Clip physically impossible negative gas/AOD values to zero
# ============================================================

def clip_negative_gas_values(df: pd.DataFrame) -> pd.DataFrame:
    """Applied BEFORE fallback and rolling averages, so both operate on
    already-clean values -- otherwise a rolling mean would still be pulled
    down by the same retrieval noise this step is meant to remove, and a
    fallback substitution could copy a negative value across scales."""
    for var in GAS_CLIP_VARS:
        for scale in SCALES:
            col = f"{var}_{scale}"
            if col in df.columns:
                n_negative = (df[col] < 0).sum()
                if n_negative:
                    print(f"    Clipping {n_negative} negative value(s) in {col} to 0.")
                df[col] = df[col].clip(lower=0)
    return df


# ============================================================
# Stage 2: Bidirectional local <-> neighborhood fallback
# ============================================================

def apply_bidirectional_fallback(df: pd.DataFrame, all_vars: list) -> pd.DataFrame:
    # New flag/substitution columns are collected here and added via a
    # single pd.concat() at the end, rather than many individual
    # df[col] = ... assignments in the loop below -- repeated one-at-a-time
    # column assignment fragments a DataFrame's internal memory blocks, and
    # a later operation needing a consolidated view (even something as
    # simple as .copy()) can then trigger an expensive reconsolidation that
    # spikes memory far beyond the data's actual size. Confirmed to matter
    # in practice: this was the direct cause of an out-of-memory crash on
    # the 7.1-million-row source-domains file (which never showed up on the
    # much smaller Philippines file).
    new_cols = {}
    for var in all_vars:
        local_col, nbhd_col = f"{var}_local", f"{var}_neighborhood_mean"
        if local_col not in df.columns or nbhd_col not in df.columns:
            continue

        local_null = df[local_col].isna()
        nbhd_null = df[nbhd_col].isna()

        # Direction 1 (established): local missing, neighborhood present.
        fill_local = local_null & ~nbhd_null
        new_cols[f"{var}_local_filled_from_neighborhood"] = fill_local
        df.loc[fill_local, local_col] = df.loc[fill_local, nbhd_col]

        # Direction 2 (new, defensive): neighborhood missing, local present.
        # Confirmed to occur 0 times on the real Philippines file -- kept
        # as a safety net, not because it's expected to fire often.
        fill_nbhd = nbhd_null & ~local_null
        new_cols[f"{var}_neighborhood_filled_from_local"] = fill_nbhd
        df.loc[fill_nbhd, nbhd_col] = df.loc[fill_nbhd, local_col]

    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    return df


# ============================================================
# Stage 3: Rolling / lagged averages
# ============================================================

def add_hourly_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Rolling means for the hourly ERA5 group, computed directly on each
    grid cell's own hourly row sequence using a TIME-based window, so gaps
    in coverage don't silently distort the window width."""
    df = df.sort_values(["grid_id", "datetime_utc"]).reset_index(drop=True)
    dt = pd.to_datetime(df["datetime_utc"], utc=True)

    new_cols = {}
    for var in HOURLY_ERA5_VARS:
        for scale in SCALES:
            col = f"{var}_{scale}"
            if col not in df.columns:
                continue
            s = pd.Series(df[col].values, index=dt)
            for window_label, window_str in HOURLY_ROLLING_WINDOWS.items():
                out_col = f"{col}_rolling_{window_label}"
                new_cols[out_col] = (
                    s.groupby(df["grid_id"].values)
                     .apply(lambda g: g.rolling(window_str, min_periods=1).mean())
                     .values
                )
    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    return df


def add_daily_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Rolling means for the daily AOD/gas group, computed ONCE per
    (grid_id, date) on the deduplicated daily sequence, then broadcast back
    to every hourly row of that date -- these variables are already
    constant across a date's 24 hours, so rolling the raw hourly rows would
    need a 72/168-row window for no benefit.

    Builds the date-keyed lookup as an entirely separate small DataFrame
    rather than assigning a temporary _date column onto the main df first
    -- by this point df has already passed through several concat-based
    column additions (fallback flags, hourly rolling columns), and pandas
    warned that a further single-column assignment onto it was triggering
    a costly fragmentation-driven reconsolidation. Keeping the date key
    entirely separate until the final concat avoids ever touching df's
    columns until the very last step."""
    dt = pd.to_datetime(df["datetime_utc"], utc=True)
    date_series = dt.dt.date.astype(str)

    daily_cols = [f"{var}_{scale}" for var in DAILY_DYNAMIC_VARS for scale in SCALES
                  if f"{var}_{scale}" in df.columns]
    daily_unique = pd.DataFrame({"grid_id": df["grid_id"].values, "_date": date_series.values,
                                 **{c: df[c].values for c in daily_cols}})
    daily_unique = daily_unique.drop_duplicates(subset=["grid_id", "_date"]).copy()
    daily_unique["_date_dt"] = pd.to_datetime(daily_unique["_date"], utc=True)
    daily_unique = daily_unique.sort_values(["grid_id", "_date_dt"]).reset_index(drop=True)

    new_daily_cols = {}
    for var in DAILY_DYNAMIC_VARS:
        for scale in SCALES:
            col = f"{var}_{scale}"
            if col not in daily_unique.columns:
                continue
            s = pd.Series(daily_unique[col].values, index=daily_unique["_date_dt"])
            for window_label, window_str in DAILY_ROLLING_WINDOWS.items():
                out_col = f"{col}_rolling_{window_label}"
                new_daily_cols[out_col] = (
                    s.groupby(daily_unique["grid_id"].values)
                     .apply(lambda g: g.rolling(window_str, min_periods=1).mean())
                     .values
                )
    daily_unique = pd.concat([daily_unique, pd.DataFrame(new_daily_cols, index=daily_unique.index)], axis=1)
    rolling_cols = list(new_daily_cols.keys())

    merge_keys = pd.DataFrame({"grid_id": df["grid_id"].values, "_date": date_series.values})
    merged_rolling = merge_keys.merge(daily_unique[["grid_id", "_date"] + rolling_cols], on=["grid_id", "_date"], how="left")
    df = pd.concat([df, merged_rolling[rolling_cols].reset_index(drop=True)], axis=1)
    return df


# ============================================================
# Stage 4: Cyclical time encoding
# ============================================================

def add_cyclical_time_features(df: pd.DataFrame) -> pd.DataFrame:
    dt = pd.to_datetime(df["datetime_utc"], utc=True)

    hour = dt.dt.hour
    dow = dt.dt.dayofweek
    month = dt.dt.month

    new_cols = {
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
        "day_of_week_sin": np.sin(2 * np.pi * dow / 7),
        "day_of_week_cos": np.cos(2 * np.pi * dow / 7),
        "month_sin": np.sin(2 * np.pi * (month - 1) / 12),
        "month_cos": np.cos(2 * np.pi * (month - 1) / 12),
    }
    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    return df


# ============================================================
# Stage 5: Metadata / feature separation
# ============================================================

def split_metadata_and_features(df: pd.DataFrame):
    metadata_cols = ["grid_id"] + [c for c in METADATA_COLUMNS if c in df.columns]
    metadata_df = df[metadata_cols].copy()  # small (8 columns) -- fine at any row count tested so far

    drop_from_features = set(METADATA_COLUMNS) - {"grid_id"}
    drop_from_features |= set(COLUMNS_TO_DROP_FROM_FEATURES_EXACT)
    drop_from_features |= {c for c in df.columns for s in COLUMNS_TO_DROP_FROM_FEATURES_SUFFIX if c.endswith(s)}
    cols_to_actually_drop = [c for c in drop_from_features if c in df.columns and c != "grid_id"]

    # features_df is almost the ENTIRE dataframe (only a handful of metadata
    # columns excluded) -- df[feature_cols].copy() means holding two
    # nearly-full-sized copies in memory simultaneously (the original df
    # AND the new sliced copy), which is exactly what crashed on the
    # 7.1-million-row source-domains file (5.41 GiB requested for a single
    # array). Dropping the few metadata-only columns from df IN PLACE
    # instead means df itself becomes the features table -- never two
    # large objects at once, just one shrinking slightly.
    df.drop(columns=cols_to_actually_drop, inplace=True)
    features_df = df

    return features_df, metadata_df


# ============================================================
# Main
# ============================================================

def main():
    print("Stage 0: determining land cover one-hot encoding scheme from both files...")
    landcover_scheme = determine_landcover_scheme(INPUT_FILES)
    print()

    for name, path in INPUT_FILES.items():
        if not path.exists():
            print(f"=== {name} === SKIPPED (file not found: {path.name})")
            continue

        print(f"=== {name} ===")
        print(f"  Loading {path.name}...")
        df = pd.read_csv(path)
        print(f"  {len(df)} rows, {len(df.columns)} columns.")

        print("  Stage 1: writing data statistics report (on raw, untransformed data)...")
        write_statistics_report(df, name)

        print("  Stage 1.5: clipping physically impossible negative gas/AOD values to 0...")
        df = clip_negative_gas_values(df)

        print("  Stage 2: applying bidirectional local <-> neighborhood fallback...")
        df = apply_bidirectional_fallback(df, FALLBACK_VARS)

        print("  Stage 3a: computing hourly ERA5 rolling averages...")
        df = add_hourly_rolling_features(df)

        print("  Stage 3b: computing daily AOD/gas rolling averages...")
        df = add_daily_rolling_features(df)

        print("  Stage 4: adding cyclical time encoding...")
        df = add_cyclical_time_features(df)

        print("  Stage 5: one-hot encoding land cover (using the shared scheme from Stage 0)...")
        df = one_hot_encode_landcover(df, landcover_scheme)

        print("  Stage 6: separating metadata from features...")
        features_df, metadata_df = split_metadata_and_features(df)

        features_path = SCRIPT_DIR / f"{name.lower()}_features.csv"
        metadata_path = SCRIPT_DIR / f"{name.lower()}_metadata.csv"
        features_df.to_csv(features_path, index=False)
        metadata_df.to_csv(metadata_path, index=False)
        print(f"  Wrote {features_path.name} ({features_df.shape[1]} columns) "
              f"and {metadata_path.name} ({metadata_df.shape[1]} columns).\n")


if __name__ == "__main__":
    main()
