"""
Experiment 6: Advanced Feature Engineering with Runtime Generation,
Group Ablation, and Enhanced Environmental Evaluation
============================================================================

Single self-contained script. Generates 7 physically-justified feature
groups (A-G) entirely at runtime from EXISTING columns -- no new GEE
extraction, no new OpenAQ pulls, original CSVs never modified.

============================================================================
PRE-IMPLEMENTATION REDUNDANCY CHECK (performed before writing this script,
per explicit instruction)
============================================================================
Full inventory of the current 152-feature set (132 original +
9 calendar/dry-days + 5 physics + 6 transport, established across
Experiments 1-3) was reviewed against every feature proposed below.
Two DIRECT duplicates were found and are explicitly excluded here:
  - "AOD x RH" (Group G candidate) -- already exists as aod_times_rh_local
    (Physics/Transport experiment). NOT regenerated.
  - "wind speed x BLH" -- already exists as ventilation_coefficient_local.
    NOT regenerated.
No other proposed feature in Groups A-G duplicates an existing column --
confirmed directly against feature_groups.py's full inventory plus
combined_physics_features_loader.py's physics/transport additions.

One additional, independently-justified feature beyond the requested
groups is included (see Group C): elevation local-minus-neighborhood
contrast. Physical rationale: captures basin/valley cold-air-pooling
effects (a cell sitting lower than its surroundings can trap cold, dense,
pollutant-laden air) -- a distinct topographic mechanism from the
pollutant contrasts already requested, cheap to add (static, reuses
already-extracted elevation columns), and was independently identified
in the pre-implementation design review as high-value. No other
speculative additions were made, per explicit instruction to prefer a
small number of well-justified features over feature-count inflation.

============================================================================
ENVIRONMENTAL METRICS -- formulas verified against primary/canonical
sources before implementation, not assumed from memory:
============================================================================
Willmott's Index of Agreement (d) -- Willmott (1981), On the validation
of models, Physical Geography 2:184-194:
    d = 1 - [sum((P-O)^2)] / [sum((|P - O_bar| + |O - O_bar|)^2)]
    Range [0,1], 1=perfect agreement. Verified against Willmott's own
    formula as reproduced across multiple independent sources (R
    packages hydroGOF/metrica, arXiv:1909.12862) before implementation.

MAAPE (Mean Arctangent Absolute Percentage Error) -- Kim & Kim (2016),
"A new metric of absolute percentage error for intermittent demand
forecasts", International Journal of Forecasting 32(3):669-679:
    MAAPE = mean(arctan(|(O - P) / O|))
    Range [0, pi/2]. Specifically designed to remain finite when O is
    near zero (unlike plain MAPE) -- relevant here since PM2.5 can be
    close to zero. A small epsilon is still added to the denominator for
    numerical safety in the division itself, even though arctan bounds
    the final result regardless.

Mean Bias Error (MBE) = mean(P - O). Positive = over-prediction bias,
negative = under-prediction bias.

============================================================================
REUSE
============================================================================
Training/evaluation machinery reused directly from prior experiments,
not reimplemented: train_source_logspace, finetune_logspace,
compute_ensemble_weights, predict_ensemble_real (run_logspace_pipeline_experiment.py);
compute_magnitude_weights (run_logspace_r2_improvement_experiment.py);
load_domain_with_combined_features, compute_full_metrics
(physics_transport_ablation_experiment.py); wind_from_direction_deg
(run_physics_transport_experiment.py, for Group B's circular statistics).

============================================================================
CHECKPOINTING
============================================================================
Not explicitly requested with the same language as the GEE extraction
scripts, but added anyway given this experiment's real computational
scale (9 ablation configs x 3 validation methods, each requiring a fresh
source retrain): per-configuration results are written incrementally to
a CSV as each ablation completes, and a completed configuration is
skipped (not recomputed) on a re-run after interruption -- same
accumulate-and-resume discipline used throughout this project's
long-running scripts.
"""

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

import diagnostics_core as dc
import validation_strategies as vs
import optimize_and_validate as ov
import combined_physics_features_loader as cpl
from run_logspace_pipeline_experiment import (
    train_source_logspace, finetune_logspace, compute_ensemble_weights, predict_ensemble_real,
    IDENTIFIER_COLUMN, TARGET_COLUMN, RANDOM_SEED,
)
from run_logspace_r2_improvement_experiment import compute_magnitude_weights
from physics_transport_ablation_experiment import compute_full_metrics, load_domain_with_combined_features
from run_physics_transport_experiment import wind_from_direction_deg

ALPHA = 0.75
MAX_ADDITIONAL_ROUNDS = 400
CELLS_TO_EXCLUDE = ["PH000000828"]

OUTPUT_DIR = dc.DIAGNOSTICS_OUTPUT_ROOT / "advanced_feature_engineering_experiment"
CHECKPOINT_PATH = OUTPUT_DIR / "ablation_checkpoint.csv"
HIGH_PM_THRESHOLD = 35.0  # configurable near top of script, per instruction
NOISE_FLOOR_RMSE = 0.058
SHAP_SAMPLE_SIZE = 3000
EPSILON = 1e-6

# Data-driven persistence thresholds (Group A) -- computed from TRAINING
# data only once loaded, not hardcoded here (see build_group_a below).
LOW_WIND_PERCENTILE = 25  # "low wind" = bottom quartile of the training distribution
LOW_BLH_PERCENTILE = 25
TENDENCY_HOURS = 3  # Group D: rate of change over 3h, matching the existing "_rolling_3h" timescale
VARIABILITY_WINDOW = "24h"  # Group E


# ============================================================
# Environmental evaluation metrics -- see module docstring for verified
# formulas and citations.
# ============================================================

def willmott_d(actual: np.ndarray, predicted: np.ndarray) -> float:
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    obs_mean = actual.mean()
    numerator = np.sum((predicted - actual) ** 2)
    denominator = np.sum((np.abs(predicted - obs_mean) + np.abs(actual - obs_mean)) ** 2)
    if denominator < EPSILON:
        return np.nan  # degenerate case: actual is constant, d is undefined
    return float(1 - numerator / denominator)


def maape(actual: np.ndarray, predicted: np.ndarray) -> float:
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    ratio = (actual - predicted) / (actual + np.where(actual >= 0, EPSILON, -EPSILON))
    return float(np.mean(np.arctan(np.abs(ratio))))


def mean_bias_error(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.asarray(predicted, dtype=float) - np.asarray(actual, dtype=float)))


def compute_environmental_metrics(actual, predicted) -> dict:
    return {"willmott_d": willmott_d(actual, predicted), "maape": maape(actual, predicted),
            "mbe": mean_bias_error(actual, predicted)}


# ============================================================
# Group A: Persistence (data-driven thresholds, time-aware streak)
# ============================================================

def compute_streak(df: pd.DataFrame, condition: pd.Series, out_col: str) -> pd.DataFrame:
    """Time-aware streak: resets at cell boundaries, value changes, AND
    time gaps (a missing hour means the condition's continuity through
    that gap is unknown, so the streak must not silently continue across
    it) -- single-level groupby throughout, avoiding the nested-groupby
    pitfall already caught and fixed twice elsewhere in this project."""
    dt = pd.to_datetime(df["datetime_utc"], utc=True)
    time_gap = dt.diff() != pd.Timedelta(hours=1)
    cell_change = df[IDENTIFIER_COLUMN] != df[IDENTIFIER_COLUMN].shift()
    value_change = condition != condition.shift()
    block_change = value_change | cell_change | time_gap
    block_change.iloc[0] = True  # first row always starts a new block
    block_id = block_change.cumsum()
    streak = df.groupby(block_id).cumcount() + 1
    df[out_col] = streak.where(condition, 0)
    return df


def build_group_a(df: pd.DataFrame, train_cell_ids: set) -> pd.DataFrame:
    """Data-driven thresholds: LOW_WIND_PERCENTILE / LOW_BLH_PERCENTILE of
    the TRAINING population only (not the full df) -- avoids threshold
    leakage from validation/holdout populations."""
    df = df.sort_values([IDENTIFIER_COLUMN, "datetime_utc"]).reset_index(drop=True)
    train_mask = df[IDENTIFIER_COLUMN].isin(train_cell_ids)
    wind_threshold = df.loc[train_mask, "wind_speed_10m_local"].quantile(LOW_WIND_PERCENTILE / 100)
    blh_threshold = df.loc[train_mask, "boundary_layer_height_local"].quantile(LOW_BLH_PERCENTILE / 100)

    is_low_wind = df["wind_speed_10m_local"] < wind_threshold
    is_low_blh = df["boundary_layer_height_local"] < blh_threshold
    df = compute_streak(df, is_low_wind, "cumulative_low_wind_duration")
    df = compute_streak(df, is_low_blh, "cumulative_low_blh_duration")
    return df, {"wind_threshold": float(wind_threshold), "blh_threshold": float(blh_threshold)}


GROUP_A_FEATURES = ["cumulative_low_wind_duration", "cumulative_low_blh_duration"]


# ============================================================
# Group B: Wind direction consistency (proper circular statistics --
# mean resultant length, NOT linear averaging of angles, which is
# mathematically wrong for circular data: e.g. averaging 350deg and
# 10deg linearly gives 180deg, the OPPOSITE direction, instead of the
# correct 0deg).
# ============================================================

def build_group_b(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values([IDENTIFIER_COLUMN, "datetime_utc"]).reset_index(drop=True)
    wind_dir_deg = wind_from_direction_deg(df["u_wind_10m_local"].values, df["v_wind_10m_local"].values)
    wind_dir_rad = np.radians(wind_dir_deg)
    dt = pd.to_datetime(df["datetime_utc"], utc=True)

    consistency = np.full(len(df), np.nan)
    for cell_id, group_idx in df.groupby(IDENTIFIER_COLUMN).groups.items():
        cell_dt = dt.loc[group_idx]
        cell_cos = pd.Series(np.cos(wind_dir_rad[group_idx]), index=cell_dt)
        cell_sin = pd.Series(np.sin(wind_dir_rad[group_idx]), index=cell_dt)
        mean_cos = cell_cos.rolling(VARIABILITY_WINDOW, min_periods=1).mean()
        mean_sin = cell_sin.rolling(VARIABILITY_WINDOW, min_periods=1).mean()
        # Mean resultant length R = sqrt(mean(cos)^2 + mean(sin)^2):
        # standard circular-statistics measure, 0=maximally variable
        # direction, 1=perfectly consistent direction.
        consistency[group_idx] = np.sqrt(mean_cos.values ** 2 + mean_sin.values ** 2)

    df[f"wind_direction_consistency_{VARIABILITY_WINDOW}"] = consistency
    return df


GROUP_B_FEATURES = [f"wind_direction_consistency_{VARIABILITY_WINDOW}"]


# ============================================================
# Group C: Spatial contrast (local vs. neighborhood)
# ============================================================

CONTRAST_VARIABLES = ["no2", "co", "aod"]


def build_group_c(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for var in CONTRAST_VARIABLES:
        local_col, nbhd_col = f"{var}_local", f"{var}_neighborhood_mean"
        df[f"{var}_local_minus_neighborhood"] = df[local_col] - df[nbhd_col]
        # Ratio version: epsilon-protected denominator (avoids literal
        # division-by-zero for near-zero neighborhood values, common for
        # gases/AOD in clean conditions) AND clipped to the 1st-99th
        # percentile of the ratio's OWN distribution (protects against a
        # few near-zero-denominator rows producing extreme outlier
        # ratios that would otherwise dominate a tree split).
        ratio = df[local_col] / (df[nbhd_col].abs() + EPSILON)
        lower, upper = ratio.quantile(0.01), ratio.quantile(0.99)
        df[f"{var}_local_over_neighborhood_ratio"] = ratio.clip(lower=lower, upper=upper)
    # Additional, independently-justified feature (see module docstring).
    df["elevation_local_minus_neighborhood"] = df["elevation_local"] - df["elevation_neighborhood_mean"]
    return df


GROUP_C_FEATURES = ([f"{v}_local_minus_neighborhood" for v in CONTRAST_VARIABLES] +
                     [f"{v}_local_over_neighborhood_ratio" for v in CONTRAST_VARIABLES] +
                     ["elevation_local_minus_neighborhood"])


# ============================================================
# Group D: Atmospheric tendency (first-order temporal derivative).
# Time-based lookup (NOT positional shift), mirroring lag_features.py's
# already-validated approach exactly, for the same reason: a positional
# shift silently returns the wrong value across a data gap.
# ============================================================

TENDENCY_VARIABLES = ["boundary_layer_height_local", "wind_speed_10m_local", "surface_pressure_local",
                       "relative_humidity_local", "temperature_2m_local"]


def build_group_d(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values([IDENTIFIER_COLUMN, "datetime_utc"]).reset_index(drop=True)
    new_cols = {f"{v}_tendency_{TENDENCY_HOURS}h": np.full(len(df), np.nan) for v in TENDENCY_VARIABLES}

    for cell_id, group_idx in df.groupby(IDENTIFIER_COLUMN).groups.items():
        cell_rows = df.loc[group_idx]
        cell_dt = pd.to_datetime(cell_rows["datetime_utc"], utc=True)
        for var in TENDENCY_VARIABLES:
            cell_series = pd.Series(cell_rows[var].values, index=cell_dt)
            cell_series = cell_series[~cell_series.index.duplicated(keep="first")]
            lookup_times = cell_dt - pd.Timedelta(hours=TENDENCY_HOURS)
            lagged_values = cell_series.reindex(lookup_times).values
            new_cols[f"{var}_tendency_{TENDENCY_HOURS}h"][group_idx] = cell_rows[var].values - lagged_values

    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


GROUP_D_FEATURES = [f"{v}_tendency_{TENDENCY_HOURS}h" for v in TENDENCY_VARIABLES]


# ============================================================
# Group E: Atmospheric variability (rolling standard deviation) --
# same time-based rolling pattern as the project's existing rolling
# MEANS, with a std reducer instead.
# ============================================================

VARIABILITY_VARIABLES = ["wind_speed_10m_local", "boundary_layer_height_local", "relative_humidity_local", "temperature_2m_local"]


def build_group_e(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values([IDENTIFIER_COLUMN, "datetime_utc"]).reset_index(drop=True)
    dt = pd.to_datetime(df["datetime_utc"], utc=True)
    new_cols = {}
    for var in VARIABILITY_VARIABLES:
        s = pd.Series(df[var].values, index=dt)
        new_cols[f"{var}_rolling_std_{VARIABILITY_WINDOW}"] = (
            s.groupby(df[IDENTIFIER_COLUMN].values).apply(lambda g: g.rolling(VARIABILITY_WINDOW, min_periods=2).std()).values
        )
    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


GROUP_E_FEATURES = [f"{v}_rolling_std_{VARIABILITY_WINDOW}" for v in VARIABILITY_VARIABLES]


# ============================================================
# Group F: Standardized (z-score) anomaly. Only AOD, per instruction to
# avoid redundancy -- reuses the EXISTING aod_local_rolling_7d as the
# mean reference (not recomputed) and a freshly-computed rolling std as
# the normalizer, epsilon-protected against near-zero local variance.
# ============================================================

def build_group_f(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values([IDENTIFIER_COLUMN, "datetime_utc"]).reset_index(drop=True)
    dt = pd.to_datetime(df["datetime_utc"], utc=True)
    s = pd.Series(df["aod_local"].values, index=dt)
    aod_rolling_std = s.groupby(df[IDENTIFIER_COLUMN].values).apply(lambda g: g.rolling("7D", min_periods=2).std()).values
    df["aod_zscore_anomaly"] = (df["aod_local"] - df["aod_local_rolling_7d"]) / (aod_rolling_std + EPSILON)
    return df


GROUP_F_FEATURES = ["aod_zscore_anomaly"]


# ============================================================
# Group G: Physically motivated interactions (depends on Group A's
# duration columns -- computed as a prerequisite even when Group G is
# tested in isolation, per the ablation design below).
# ============================================================

def build_group_g(df_with_group_a: pd.DataFrame) -> pd.DataFrame:
    df = df_with_group_a.copy()
    df["population_x_low_wind_duration"] = df["population_local"] * df["cumulative_low_wind_duration"]
    df["population_x_low_blh_duration"] = df["population_local"] * df["cumulative_low_blh_duration"]
    return df


GROUP_G_FEATURES = ["population_x_low_wind_duration", "population_x_low_blh_duration"]

ALL_GROUP_FEATURES = {
    "A": GROUP_A_FEATURES, "B": GROUP_B_FEATURES, "C": GROUP_C_FEATURES, "D": GROUP_D_FEATURES,
    "E": GROUP_E_FEATURES, "F": GROUP_F_FEATURES, "G": GROUP_G_FEATURES,
}


# ============================================================
# Master generation: builds ALL groups once (Group A computed as a
# prerequisite for Group G regardless of which groups are "active" for a
# given ablation config -- exposure to the model is controlled separately,
# see select_ablation_features below), entirely in memory, original df
# untouched (this function always returns a NEW DataFrame).
# ============================================================

def generate_all_engineered_features(df: pd.DataFrame, train_cell_ids: set) -> tuple:
    """Returns (engineered_df, threshold_metadata). engineered_df is a
    NEW DataFrame -- df itself is never mutated in place anywhere in this
    pipeline (every build_group_* function either copies or reassigns)."""
    df, threshold_meta = build_group_a(df, train_cell_ids)
    df = build_group_b(df)
    df = build_group_c(df)
    df = build_group_d(df)
    df = build_group_e(df)
    df = build_group_f(df)
    df = build_group_g(df)  # depends on Group A's columns, already present at this point
    return df, threshold_meta


def select_ablation_features(engineered_df_cols: list, active_groups: list) -> list:
    """Which of the newly-engineered columns are EXPOSED to the model for
    a given ablation config. Group A's columns are always computed (as a
    Group G prerequisite) but only exposed as standalone features when
    "A" is itself an active group -- e.g. "Baseline + Group G" alone does
    NOT include cumulative_low_wind_duration as a separate feature, only
    the two interaction terms that depend on it."""
    selected = []
    for group in active_groups:
        selected.extend(ALL_GROUP_FEATURES[group])
    return selected


ABLATION_CONFIGS = {
    "baseline": [], "group_A": ["A"], "group_B": ["B"], "group_C": ["C"], "group_D": ["D"],
    "group_E": ["E"], "group_F": ["F"], "group_G": ["G"], "all_groups": ["A", "B", "C", "D", "E", "F", "G"],
}


# ============================================================
# Feature quality validation -- automatic checks before any training,
# per explicit instruction. Removes only features that CLEARLY fail.
# ============================================================

def validate_feature_quality(df: pd.DataFrame, candidate_cols: list) -> tuple:
    """Returns (retained_cols, quality_report_df, removed_report_lines)."""
    report_rows = []
    removed = []
    retained = []

    for col in candidate_cols:
        series = df[col]
        n_nan = series.isna().sum()
        n_inf = int(np.isinf(series.astype(float)).sum())
        n_unique = series.nunique(dropna=True)
        is_constant = n_unique <= 1
        is_near_constant = n_unique <= max(2, int(0.001 * len(series)))
        pct_nan = n_nan / len(series) * 100

        status = "RETAINED"
        reason = ""
        if is_constant:
            status, reason = "REMOVED", "constant column (zero variance)"
        elif pct_nan > 50:
            status, reason = "REMOVED", f"excessive missingness ({pct_nan:.1f}%% NaN)"
        elif n_inf > 0:
            status, reason = "REMOVED", f"{n_inf} infinite value(s) found"
        elif is_near_constant:
            status, reason = "FLAGGED (retained)", f"near-constant ({n_unique} unique values) -- retained but flagged, not automatically removed"

        report_rows.append({"feature": col, "pct_nan": pct_nan, "n_inf": n_inf, "n_unique": n_unique,
                             "is_constant": is_constant, "is_near_constant": is_near_constant, "status": status, "reason": reason})
        if status == "REMOVED":
            removed.append((col, reason))
        else:
            retained.append(col)

    quality_df = pd.DataFrame(report_rows)

    # Duplicated columns (exact value equality) and extremely high pairwise
    # correlations among RETAINED engineered columns only.
    dup_pairs = []
    corr_pairs = []
    if len(retained) > 1:
        sub = df[retained].astype(float)
        for i, col_a in enumerate(retained):
            for col_b in retained[i + 1:]:
                if sub[col_a].equals(sub[col_b]):
                    dup_pairs.append((col_a, col_b))
        corr_matrix = sub.corr().abs()
        for i, col_a in enumerate(retained):
            for col_b in retained[i + 1:]:
                r = corr_matrix.loc[col_a, col_b]
                if pd.notna(r) and r > 0.98:
                    corr_pairs.append((col_a, col_b, float(r)))

    removed_lines = [f"  {col}: {reason}" for col, reason in removed]
    if dup_pairs:
        removed_lines.append(f"  NOTE: {len(dup_pairs)} exact-duplicate column pair(s) found (reported, not auto-removed): {dup_pairs}")
    if corr_pairs:
        removed_lines.append(f"  NOTE: {len(corr_pairs)} pair(s) with |correlation|>0.98 found (reported, not auto-removed): "
                              f"{[(a,b,round(r,3)) for a,b,r in corr_pairs]}")

    return retained, quality_df, removed_lines


# ============================================================
# Segmented evaluation: overall, normal-PM, high-PM -- all metrics
# (existing 4 + 3 environmental) computed for each segment.
# ============================================================

def compute_segmented_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    normal_mask = actual < HIGH_PM_THRESHOLD
    high_mask = ~normal_mask

    def segment_metrics(mask, label):
        if mask.sum() == 0:
            return {f"{label}_{k}": np.nan for k in ("rmse", "mae", "r2", "willmott_d", "maape", "mbe")}
        a, p = actual[mask], predicted[mask]
        full = compute_full_metrics(pd.Series(a), p)
        env = compute_environmental_metrics(a, p)
        return {f"{label}_rmse": full["rmse"], f"{label}_mae": full["mae"], f"{label}_r2": full["r2"],
                f"{label}_willmott_d": env["willmott_d"], f"{label}_maape": env["maape"], f"{label}_mbe": env["mbe"]}

    result = {}
    result.update(segment_metrics(np.ones(len(actual), dtype=bool), "overall"))
    result["overall_high_pm_mae"] = compute_full_metrics(pd.Series(actual), predicted)["high_pm_mae"]  # backward-compat, established threshold
    result.update(segment_metrics(normal_mask, "normal_pm"))
    result.update(segment_metrics(high_mask, "high_pm"))
    return result


# ============================================================
# Per-config training/evaluation -- reuses the established
# finetune-once-per-context pattern from every prior experiment.
# ============================================================

def finetune_and_predict(source_models, train_df, val_df, target_df, feature_cols):
    weight_train = compute_magnitude_weights(train_df[TARGET_COLUMN], ALPHA)
    replication_counts = np.maximum(1, np.round(weight_train)).astype(int)
    train_weighted = train_df.loc[train_df.index.repeat(replication_counts)].reset_index(drop=True)

    import run_logspace_pipeline_experiment as lp
    original_cap = lp.MAX_ADDITIONAL_ROUNDS
    lp.MAX_ADDITIONAL_ROUNDS = MAX_ADDITIONAL_ROUNDS
    try:
        finetuned = finetune_logspace(source_models, train_weighted[feature_cols], train_weighted[TARGET_COLUMN],
                                       val_df[feature_cols], val_df[TARGET_COLUMN])
    finally:
        lp.MAX_ADDITIONAL_ROUNDS = original_cap

    weights = compute_ensemble_weights(finetuned, val_df[feature_cols], val_df[TARGET_COLUMN])
    preds = predict_ensemble_real(finetuned, weights, target_df[feature_cols])
    preds_df = pd.DataFrame({IDENTIFIER_COLUMN: target_df[IDENTIFIER_COLUMN].values,
                              "actual_pm25": target_df[TARGET_COLUMN].values, "predicted": preds["ensemble"]})
    return finetuned, weights, preds_df


def run_one_ablation_config(config_name: str, active_groups: list, ph_engineered_df: pd.DataFrame,
                              source_train_base: pd.DataFrame, base_feature_cols: list,
                              source_train_cell_ids: set) -> dict:
    """Trains source ONCE for this config, evaluates across all three
    validation methods. Reuses the already-trained alpha_capacity source
    model for the baseline config specifically (byte-for-byte identical
    to the established best pipeline); every other config requires a
    fresh source retrain since its feature set differs."""
    active_new_cols = select_ablation_features(list(ph_engineered_df.columns), active_groups)
    feature_cols = base_feature_cols + active_new_cols
    print(f"\n{'#'*60}\n# Ablation: {config_name} ({len(active_new_cols)} new feature(s): {active_groups})\n{'#'*60}")

    # NOTE: "baseline" here still requires a fresh source retrain -- there
    # is no persisted source model anywhere in this project trained on the
    # 152-feature "improved" set with mean regression specifically (the
    # saved alpha_capacity model predates Experiments 1-3's feature
    # additions and only has the original 132 features -- confirmed
    # directly via an XGBoost feature-mismatch error when this was
    # incorrectly assumed reusable).
    if config_name == "baseline":
        # No Group A-G columns are referenced by feature_cols here, so
        # skip generating them entirely for the (large, ~4.9M-row) source
        # population -- avoids real wasted computation that would
        # otherwise be immediately discarded.
        source_weight = compute_magnitude_weights(source_train_base[TARGET_COLUMN], ALPHA)
        source_models = train_source_logspace(source_train_base[feature_cols], source_train_base[TARGET_COLUMN], source_weight)
    else:
        source_engineered, _ = generate_all_engineered_features(source_train_base, source_train_cell_ids)
        source_weight = compute_magnitude_weights(source_engineered[TARGET_COLUMN], ALPHA)
        source_models = train_source_logspace(source_engineered[feature_cols], source_engineered[TARGET_COLUMN], source_weight)

    frozen_split = pd.read_csv(dc.SCRIPT_DIR / "finetune_outputs" / "philippines_finetune_split_assignment.csv")
    ph_with_split = ph_engineered_df.merge(frozen_split[[IDENTIFIER_COLUMN, "split"]], on=IDENTIFIER_COLUMN, how="left")
    spatial_train = ph_with_split[ph_with_split["split"] == "finetune_train"]
    spatial_val = ph_with_split[ph_with_split["split"] == "finetune_val"]
    spatial_holdout = ph_with_split[ph_with_split["split"] == "holdout"]
    spatial_finetuned, spatial_weights, spatial_preds = finetune_and_predict(source_models, spatial_train, spatial_val, spatial_holdout, feature_cols)
    spatial_metrics = compute_segmented_metrics(spatial_preds["actual_pm25"].values, spatial_preds["predicted"].values)

    random_split = vs.random_holdout_split(ph_engineered_df, seed=RANDOM_SEED)
    ph_with_random = ph_engineered_df.copy()
    ph_with_random["split"] = random_split["split"].values
    random_train = ph_with_random[ph_with_random["split"] == "finetune_train"]
    random_val = ph_with_random[ph_with_random["split"] == "finetune_val"]
    random_holdout = ph_with_random[ph_with_random["split"] == "holdout"]
    _, _, random_preds = finetune_and_predict(source_models, random_train, random_val, random_holdout, feature_cols)
    random_metrics = compute_segmented_metrics(random_preds["actual_pm25"].values, random_preds["predicted"].values)

    outer_folds = vs.kfold_cell_splits(ph_engineered_df, k=5, seed=RANDOM_SEED)
    fold_preds = []
    for fold_idx, fold_assignment in enumerate(outer_folds):
        fold_df = ph_engineered_df.merge(fold_assignment[[IDENTIFIER_COLUMN, "split"]], on=IDENTIFIER_COLUMN, how="left")
        f_train = fold_df[fold_df["split"] == "finetune_train"]
        f_val = fold_df[fold_df["split"] == "finetune_val"]
        f_target = fold_df[fold_df["split"] == "holdout"]
        _, _, f_preds = finetune_and_predict(source_models, f_train, f_val, f_target, feature_cols)
        fold_preds.append(f_preds)
    pooled_5fold = pd.concat(fold_preds, ignore_index=True)
    fold5_metrics = compute_segmented_metrics(pooled_5fold["actual_pm25"].values, pooled_5fold["predicted"].values)

    return {"config_name": config_name, "active_groups": active_groups, "feature_cols": feature_cols,
            "spatial_metrics": spatial_metrics, "random_metrics": random_metrics, "fold5_metrics": fold5_metrics,
            "spatial_finetuned_models": spatial_finetuned, "spatial_holdout_df": spatial_holdout,
            "spatial_preds": spatial_preds}


# ============================================================
# Checkpointing: per-config results appended to CHECKPOINT_PATH as each
# completes; a completed config is skipped (not recomputed) on a re-run.
# ============================================================

def flatten_result_for_checkpoint(result: dict) -> dict:
    row = {"config_name": result["config_name"], "active_groups": ",".join(result["active_groups"])}
    for method in ("spatial", "random", "fold5"):
        for k, v in result[f"{method}_metrics"].items():
            row[f"{method}_{k}"] = v
    return row


def load_completed_configs(checkpoint_path: Path) -> set:
    if not checkpoint_path.exists():
        return set()
    return set(pd.read_csv(checkpoint_path)["config_name"].unique())


def append_checkpoint(checkpoint_path: Path, row: dict):
    df = pd.DataFrame([row])
    write_header = not checkpoint_path.exists()
    df.to_csv(checkpoint_path, mode="a", header=write_header, index=False)


# ============================================================
# SHAP: baseline, best ablation, full engineered model.
# ============================================================

def compute_shap_for_config(result: dict, label: str) -> pd.DataFrame:
    finetuned = result["spatial_finetuned_models"]
    holdout_df = result["spatial_holdout_df"]
    feature_cols = result["feature_cols"]
    sample_n = min(SHAP_SAMPLE_SIZE, len(holdout_df))
    sample_idx = np.random.RandomState(RANDOM_SEED).choice(len(holdout_df), size=sample_n, replace=False)
    sample_features = holdout_df.iloc[sample_idx][feature_cols]

    explainer = shap.TreeExplainer(finetuned["xgboost"])
    shap_values = explainer.shap_values(sample_features)
    shap_df = pd.DataFrame({"feature": feature_cols, "mean_abs_shap": np.abs(shap_values).mean(axis=0)})
    shap_df = shap_df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    shap_df["rank"] = shap_df.index + 1
    shap_df.to_csv(OUTPUT_DIR / "tables" / f"shap_{label}.csv", index=False)
    return shap_df


# ============================================================
# Main
# ============================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "tables").mkdir(exist_ok=True)
    (OUTPUT_DIR / "figures").mkdir(exist_ok=True)

    print("Loading Philippine data with the current best (improved) feature set...")
    ph_df, base_feature_cols = load_domain_with_combined_features("philippines", include_new=True)
    ph_df = ph_df[~ph_df[IDENTIFIER_COLUMN].isin(CELLS_TO_EXCLUDE)].reset_index(drop=True)

    frozen_split = pd.read_csv(dc.SCRIPT_DIR / "finetune_outputs" / "philippines_finetune_split_assignment.csv")
    train_cell_ids = set(frozen_split[frozen_split["split"] == "finetune_train"][IDENTIFIER_COLUMN])

    print("Generating all engineered features (Groups A-G) at runtime...")
    ph_engineered_df, threshold_meta = generate_all_engineered_features(ph_df, train_cell_ids)
    print(f"  Data-driven thresholds: {threshold_meta}")

    all_new_cols = [c for cols in ALL_GROUP_FEATURES.values() for c in cols]
    retained_cols, quality_df, removed_lines = validate_feature_quality(ph_engineered_df, all_new_cols)
    quality_df.to_csv(OUTPUT_DIR / "tables" / "feature_quality_report.csv", index=False)
    quality_summary = ["Feature Quality Report", "=" * 30, "",
                        f"Engineered features created: {len(all_new_cols)}",
                        f"Retained: {len(retained_cols)}", f"Removed: {len(all_new_cols) - len(retained_cols)}",
                        "", "Removal/flag details:"] + removed_lines
    (OUTPUT_DIR / "feature_quality_report.txt").write_text("\n".join(quality_summary))
    print("\n".join(quality_summary))

    # Source train base (raw, un-engineered) -- engineered fresh per
    # non-baseline config inside run_one_ablation_config, since alpha
    # weighting on the source population must use that config's own
    # feature set consistently.
    source_df_full, _, _ = dc.load_data()
    source_metadata_full = pd.read_csv(dc.SCRIPT_DIR / "source_metadata.csv",
                                         usecols=[IDENTIFIER_COLUMN, "datetime_utc", "centroid_latitude", "centroid_longitude"])
    if len(source_metadata_full) != len(source_df_full):
        raise ValueError("source_features.csv/source_metadata.csv row count mismatch.")
    source_df_full = source_df_full.copy()
    for col in ["datetime_utc", "centroid_latitude", "centroid_longitude"]:
        source_df_full[col] = source_metadata_full[col].values
    source_train_raw = source_df_full[source_df_full["split"] == "train"].drop(columns=["split"])
    source_train_cell_ids = set(source_train_raw[IDENTIFIER_COLUMN].unique())

    # Build the "improved" (152-feature: original + non-fire additional +
    # physics + transport) source base ONCE here -- identical across
    # every ablation config (only Group A-G generation varies per config),
    # so this must not be rebuilt inside the per-config loop.
    print("\nBuilding the improved (152-feature) source base once, shared across all ablation configs...")
    source_train_metadata = source_train_raw[[IDENTIFIER_COLUMN, "datetime_utc", "centroid_latitude", "centroid_longitude"]]
    source_train_raw_features_only = source_train_raw.drop(columns=["datetime_utc", "centroid_latitude", "centroid_longitude"])
    source_train_base, _ = cpl.build_combined_features(source_train_raw_features_only, source_train_metadata, "source")

    completed = load_completed_configs(CHECKPOINT_PATH)
    print(f"\n{len(completed)} ablation config(s) already completed (checkpoint found) -- will be skipped.")

    all_results = {}
    for config_name, active_groups in ABLATION_CONFIGS.items():
        if config_name in completed:
            print(f"Skipping {config_name} (already in checkpoint).")
            continue
        result = run_one_ablation_config(config_name, active_groups, ph_engineered_df, source_train_base,
                                           base_feature_cols, source_train_cell_ids)
        all_results[config_name] = result
        append_checkpoint(CHECKPOINT_PATH, flatten_result_for_checkpoint(result))
        print(f"  Checkpointed {config_name}.")

    # Reload the FULL checkpoint (covers configs from this run AND any prior run).
    checkpoint_df = pd.read_csv(CHECKPOINT_PATH)
    checkpoint_df.to_csv(OUTPUT_DIR / "tables" / "ablation_full_results.csv", index=False)

    # --- Ablation report ---
    baseline_row = checkpoint_df[checkpoint_df["config_name"] == "baseline"].iloc[0]
    checkpoint_df["spatial_rmse_delta"] = checkpoint_df["spatial_overall_rmse"] - baseline_row["spatial_overall_rmse"]
    checkpoint_df["fold5_rmse_delta"] = checkpoint_df["fold5_overall_rmse"] - baseline_row["fold5_overall_rmse"]
    non_baseline = checkpoint_df[checkpoint_df["config_name"] != "baseline"]
    best_config_row = non_baseline.sort_values("spatial_rmse_delta").iloc[0]
    worst_config_row = non_baseline.sort_values("spatial_rmse_delta", ascending=False).iloc[0]

    ablation_lines = ["Ablation Report", "=" * 30, "",
                       checkpoint_df[["config_name", "spatial_overall_rmse", "fold5_overall_rmse",
                                       "spatial_rmse_delta", "fold5_rmse_delta"]].to_string(index=False),
                       f"\nBest group by spatial RMSE delta: {best_config_row['config_name']} ({best_config_row['spatial_rmse_delta']:+.4f})",
                       f"Worst group by spatial RMSE delta: {worst_config_row['config_name']} ({worst_config_row['spatial_rmse_delta']:+.4f})",
                       f"Groups with NEGATIVE impact (RMSE got worse): "
                       f"{non_baseline[non_baseline['spatial_rmse_delta'] > 0]['config_name'].tolist()}",
                       f"Groups within noise floor of baseline (likely redundant/no effect): "
                       f"{non_baseline[non_baseline['spatial_rmse_delta'].abs() <= NOISE_FLOOR_RMSE]['config_name'].tolist()}",
                       ]
    (OUTPUT_DIR / "ablation_report.txt").write_text("\n".join(ablation_lines))
    print("\n" + "\n".join(ablation_lines))

    # --- SHAP: baseline, best ablation, full ---
    print("\nComputing SHAP for baseline, best ablation, and full engineered model...")
    shap_results = {}
    for label, config_name in [("baseline", "baseline"), ("best_ablation", best_config_row["config_name"]), ("full_engineered", "all_groups")]:
        if config_name in all_results:
            shap_results[label] = compute_shap_for_config(all_results[config_name], label)
        else:
            print(f"  {label} ({config_name}) not in this run's results (already checkpointed from a prior run) -- "
                  f"SHAP for it requires re-running that config's training; skipped here, not fabricated.")

    if "full_engineered" in shap_results:
        full_shap = shap_results["full_engineered"]
        for group, cols in ALL_GROUP_FEATURES.items():
            group_shap = full_shap[full_shap["feature"].isin(cols)]
            if len(group_shap):
                print(f"  Group {group} in full model: best rank {group_shap['rank'].min()}, "
                      f"{(group_shap['mean_abs_shap']>0).sum()}/{len(group_shap)} used (non-zero SHAP)")

    # --- Final verdict ---
    meaningful = abs(best_config_row["spatial_rmse_delta"]) > NOISE_FLOOR_RMSE and abs(best_config_row["fold5_rmse_delta"]) > NOISE_FLOOR_RMSE and best_config_row["spatial_rmse_delta"] < 0
    verdict_lines = ["Experiment 6 -- Final Verdict", "=" * 40, "",
                      f"1. Did feature engineering produce a practically meaningful improvement? {'YES' if meaningful else 'NO'}",
                      f"2. Best-contributing group: {best_config_row['config_name']}",
                      f"3. Recommend for permanent inclusion: {best_config_row['config_name'] if meaningful else '(none meet the noise floor bar)'}",
                      f"4. Recommend discarding: {non_baseline[non_baseline['spatial_rmse_delta'].abs() <= NOISE_FLOOR_RMSE]['config_name'].tolist()}",
                      f"5. See ablation_full_results.csv (normal_pm_* vs high_pm_* columns) for whether improvements "
                      f"concentrate in normal conditions, episodic pollution, or both.",
                      f"6. See feature_quality_report.txt and ablation_full_results.csv for the complete evidence base.",
                      ]
    (OUTPUT_DIR / "final_verdict.txt").write_text("\n".join(verdict_lines))
    print("\n" + "\n".join(verdict_lines))
    print(f"\nExperiment 6 complete. Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
