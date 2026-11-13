"""
Physics and Wind-Transport Feature Engineering Experiment
================================================================

Isolates exactly one variable: feature representation. The training
pipeline (log-space source-to-Philippines transfer, magnitude-weighted
fine-tuning at alpha=1.0, same splits, same evaluation) is completely
unchanged from the current best configuration -- reused via import from
the already-validated modules, never reimplemented -- so any performance
difference between the four ablation configurations below is
attributable only to the new features, not to anything else.

Group 1: Aerosol and Ventilation Physics (pure row-wise arithmetic on
existing columns -- no new data, no spatial computation, no leakage risk)
------------------------------------------------------------------------------
  - aod_over_blh_local = aod_local / boundary_layer_height_local
        A standard approximation of surface-level aerosol concentration:
        AOD integrates aerosol loading through the whole atmospheric
        column, but PM2.5 is a SURFACE measurement. Dividing by mixing
        depth (boundary layer height) approximates "how concentrated is
        this column's aerosol load within the volume near the surface" --
        directly closer to the physical quantity being predicted than
        column AOD alone.
  - aod_times_rh_local = aod_local * relative_humidity_local
        Aerosol hygroscopic growth: particles swell and scatter more
        light under higher humidity, inflating AOD relative to actual
        dry particulate mass -- this interaction is a standard,
        documented correction term in aerosol-PM2.5 literature.
  - aod_over_blh_times_rh_local = aod_over_blh_local * relative_humidity_local
        Combines both corrections above.
  - ventilation_coefficient_local = wind_speed_10m_local * boundary_layer_height_local
        A standard, named atmospheric science quantity (wind speed times
        mixing depth) -- directly measures how effectively pollutants are
        diluted and dispersed; low ventilation coefficient is a
        well-established precursor to pollution buildup.
  - stagnation_indicator_local = 1 / (ventilation_coefficient_local + epsilon)
        The direct inverse of ventilation: high stagnation score
        corresponds to simultaneously low wind AND shallow boundary
        layer, the textbook conditions for pollution accumulation. A
        single continuous score is used rather than a hard threshold, to
        avoid an arbitrary, undocumented cutoff choice.

Scope decision: LOCAL scale only, not neighborhood or rolling variants --
stated explicitly, since computing every physics interaction across every
existing scale/window combination would multiply the new feature count
without a specific physical justification for each variant. Local-scale
interactions are the most directly physically motivated (surface
conditions at the exact grid cell), and this keeps the ablation focused
and interpretable rather than an uncontrolled feature-count expansion.

Group 2: Wind-Aware Transport (genuine spatial geometry, computed from
data already in the dataset -- no new external data source)
------------------------------------------------------------------------------
IMPORTANT LIMITATION, stated upfront: the existing feature table's
"neighborhood_mean" columns are already-aggregated symmetric averages
computed once during the original Earth Engine extraction (Chapter 4) --
individual neighboring grid cells' own values are NOT present in the
per-row feature table. Building a genuine wind-weighted directional
feature therefore requires reconstructing cell-to-cell spatial
relationships from what IS available: each grid cell's own centroid
latitude/longitude (in the metadata files) and every cell's own hourly
local pollutant values. This is more computation on EXISTING data, not a
new external data source -- consistent with the stated constraint.

Method:
  1. A STATIC cell-adjacency table is built ONCE per domain (source,
     Philippines) from centroid coordinates: for each grid cell, every
     OTHER cell within ADJACENCY_RADIUS_KM, with the great-circle bearing
     FROM that neighbor TO the cell (i.e. the direction pollution
     transported from the neighbor would travel to reach this cell).
     This is O(n_cells^2) -- for a few hundred cells, trivially cheap;
     it does NOT depend on time and is computed only once.
  2. For each row (grid cell, hour), the wind vector (from u/v components)
     gives a wind-FROM direction. A neighbor's contribution is weighted by
     max(0, cos(angle between the wind-from direction and the bearing
     from that neighbor to the cell)) -- a neighbor directly upwind gets
     weight 1, a neighbor directly downwind gets weight 0, partial
     alignments get a smooth intermediate weight. This is applied
     PER GRID CELL (looping over the ~hundreds of cells, not the millions
     of rows), vectorized across all of that cell's own hours at once,
     using a wide (datetime x grid_id) pivot of each transportable
     variable so that a given row's neighbors' CONCURRENT values can be
     looked up efficiently without a per-row join.
  3. Computed for aod_local, no2_local, so2_local, co_local, o3_local,
     hcho_local -- the six daily dynamic pollutant variables, matching
     "pollution arriving from upwind cells" specifically, not every
     existing feature.

No target leakage: every new feature is built from other PREDICTOR
columns (AOD, gases, meteorology, geometry) at the same or earlier time
as the row it's attached to -- pm25 itself is never used in constructing
any of these features.

Usage
-----
Place this script alongside source_features.csv, source_metadata.csv,
philippines_features.csv, philippines_metadata.csv,
baseline_model_outputs/ (source cell split), and
finetune_outputs/philippines_finetune_split_assignment.csv (Philippine
fine-tune split), and run it.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from catboost import CatBoostRegressor

from train_baseline_models import compute_metrics
from run_logspace_pipeline_experiment import (
    train_source_logspace, finetune_logspace, compute_ensemble_weights, predict_ensemble_real,
    IDENTIFIER_COLUMN, TARGET_COLUMN, RANDOM_SEED,
)
from run_logspace_r2_improvement_experiment import compute_magnitude_weights

# ============================================================
# Configuration -- MACROS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
SOURCE_FEATURES_PATH = SCRIPT_DIR / "source_features.csv"
SOURCE_METADATA_PATH = SCRIPT_DIR / "source_metadata.csv"
PH_FEATURES_PATH = SCRIPT_DIR / "philippines_features.csv"
PH_METADATA_PATH = SCRIPT_DIR / "philippines_metadata.csv"
BASELINE_DIR = SCRIPT_DIR / "baseline_model_outputs"
PH_SPLIT_PATH = SCRIPT_DIR / "finetune_outputs" / "philippines_finetune_split_assignment.csv"

OUTPUT_ROOT = SCRIPT_DIR / "finetune_outputs_physics_transport"

ALPHA = 1.0  # magnitude weighting, fixed at the current best value -- not swept here
ADJACENCY_RADIUS_KM = 15.0  # captures a meaningfully "regional" upwind scale, larger than the
                             # existing ~2.5km-radius neighborhood buffer, without being so large
                             # that "upwind" loses spatial specificity
TRANSPORT_VARIABLES = ["aod_local", "no2_local", "so2_local", "co_local", "o3_local", "hcho_local"]
EPSILON = 1e-6
SHAP_SAMPLE_SIZE = 3000


# ============================================================
# Group 1: Aerosol and ventilation physics features
# ============================================================

def add_physics_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["aod_over_blh_local"] = df["aod_local"] / (df["boundary_layer_height_local"] + EPSILON)
    df["aod_times_rh_local"] = df["aod_local"] * df["relative_humidity_local"]
    df["aod_over_blh_times_rh_local"] = df["aod_over_blh_local"] * df["relative_humidity_local"]
    df["ventilation_coefficient_local"] = df["wind_speed_10m_local"] * df["boundary_layer_height_local"]
    df["stagnation_indicator_local"] = 1.0 / (df["ventilation_coefficient_local"] + EPSILON)
    return df


PHYSICS_FEATURE_NAMES = ["aod_over_blh_local", "aod_times_rh_local", "aod_over_blh_times_rh_local",
                          "ventilation_coefficient_local", "stagnation_indicator_local"]


# ============================================================
# Group 2: Wind-aware transport features
# ============================================================

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def bearing_deg(lat1, lon1, lat2, lon2):
    """Bearing FROM (lat1,lon1) TO (lat2,lon2), in degrees, 0=North, 90=East."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return (np.degrees(np.arctan2(x, y)) + 360) % 360


def build_cell_adjacency(centroids: pd.DataFrame, radius_km: float) -> dict:
    """Returns {grid_id: [(neighbor_grid_id, bearing_neighbor_to_cell_deg), ...]}.
    O(n_cells^2), computed once -- centroids do not change over time."""
    ids = centroids[IDENTIFIER_COLUMN].values
    lats = centroids["centroid_latitude"].values
    lons = centroids["centroid_longitude"].values
    n = len(ids)
    adjacency = {gid: [] for gid in ids}

    for i in range(n):
        dists = haversine_km(lats[i], lons[i], lats, lons)
        within = np.where((dists > 0) & (dists <= radius_km))[0]
        for j in within:
            # Bearing FROM neighbor j TO cell i -- the direction pollution
            # transported from j would travel to arrive at i.
            b = bearing_deg(lats[j], lons[j], lats[i], lons[i])
            adjacency[ids[i]].append((ids[j], b))
    return adjacency


def wind_from_direction_deg(u, v):
    """Meteorological wind-FROM direction (the direction the wind is
    blowing FROM), in degrees, 0=North, 90=East, from u/v components."""
    return (np.degrees(np.arctan2(-u, -v)) + 360) % 360


def add_wind_transport_features(df: pd.DataFrame, adjacency: dict) -> pd.DataFrame:
    """Adds transport_<var>_local columns: a wind-upwind-weighted average
    of each cell's neighbors' concurrent local values, computed by
    looping over grid cells (not rows) and vectorizing across each
    cell's own hours using a wide (datetime x grid_id) pivot per
    transportable variable, so neighbor lookups don't require a per-row
    join."""
    df = df.copy()
    df["_wind_from_deg"] = wind_from_direction_deg(df["u_wind_10m_local"].values, df["v_wind_10m_local"].values)

    # One wide pivot per transportable variable: index=datetime_utc,
    # columns=grid_id. Requires datetime_utc as a column -- attached by
    # the caller before this function runs.
    pivots = {}
    for var in TRANSPORT_VARIABLES:
        if var not in df.columns:
            continue
        pivots[var] = df.pivot_table(index="datetime_utc", columns=IDENTIFIER_COLUMN, values=var, aggfunc="first")

    for var in TRANSPORT_VARIABLES:
        if var in pivots:
            df[f"transport_{var}"] = np.nan

    for grid_id, neighbors in adjacency.items():
        cell_mask = df[IDENTIFIER_COLUMN].values == grid_id
        if not cell_mask.any() or len(neighbors) == 0:
            continue
        cell_rows = df.loc[cell_mask]
        wind_from = cell_rows["_wind_from_deg"].values
        # A neighbor is upwind when the wind is blowing FROM that
        # neighbor's direction TOWARD this cell -- i.e. when the wind's
        # TRAVEL direction (wind_from + 180) matches the bearing FROM the
        # neighbor TO this cell. Comparing wind_from directly against that
        # bearing (as an earlier version of this function did) is off by
        # 180 degrees and silently computes a DOWNWIND-weighted feature
        # instead of upwind -- confirmed directly with a hand-checkable
        # 3-cell synthetic test (a due-north neighbor received weight 0
        # instead of 1 under a due-north wind) before this fix was applied.
        wind_to = (wind_from + 180) % 360
        cell_datetimes = cell_rows["datetime_utc"].values

        neighbor_ids = [n[0] for n in neighbors]
        neighbor_bearings = np.array([n[1] for n in neighbors])
        angle_diff = np.radians(wind_to[:, None] - neighbor_bearings[None, :])
        weights = np.clip(np.cos(angle_diff), 0, None)  # shape: (n_hours_this_cell, n_neighbors)

        for var, pivot in pivots.items():
            valid_neighbor_cols = [nid for nid in neighbor_ids if nid in pivot.columns]
            if not valid_neighbor_cols:
                continue
            col_idx = [neighbor_ids.index(nid) for nid in valid_neighbor_cols]
            neighbor_values = pivot.reindex(cell_datetimes)[valid_neighbor_cols].values  # (n_hours, n_valid_neighbors)
            these_weights = weights[:, col_idx]
            with np.errstate(invalid="ignore", divide="ignore"):
                weight_sum = these_weights.sum(axis=1)
                weighted_avg = np.where(weight_sum > 0,
                                          np.nansum(these_weights * neighbor_values, axis=1) / np.where(weight_sum > 0, weight_sum, 1),
                                          np.nan)
            df.loc[cell_mask, f"transport_{var}"] = weighted_avg

    df = df.drop(columns=["_wind_from_deg"])
    return df


TRANSPORT_FEATURE_NAMES = [f"transport_{v}" for v in TRANSPORT_VARIABLES]


# ============================================================
# Ablation configuration runner (reuses the validated log-space +
# magnitude-weighting pipeline unchanged; only the feature set varies)
# ============================================================

def run_configuration(config_name: str, source_df, ph_df, feature_cols) -> dict:
    run_dir = OUTPUT_ROOT / config_name
    models_dir = run_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    source_train_df = source_df[source_df["split"] == "train"]
    source_val_df = source_df[source_df["split"] == "val"]
    ph_train_df = ph_df[ph_df["split"] == "finetune_train"]
    ph_val_df = ph_df[ph_df["split"] == "finetune_val"]
    ph_holdout_df = ph_df[ph_df["split"] == "holdout"]

    print(f"  [{config_name}] Training source models (log-space, magnitude-weighted alpha={ALPHA})...")
    source_weight = compute_magnitude_weights(source_train_df[TARGET_COLUMN], ALPHA)
    source_models = train_source_logspace(source_train_df[feature_cols], source_train_df[TARGET_COLUMN], source_weight)

    print(f"  [{config_name}] Fine-tuning (magnitude-weighted via row replication, matching the validated R2 experiment)...")
    ph_weight_train = compute_magnitude_weights(ph_train_df[TARGET_COLUMN], ALPHA)
    replication_counts = np.maximum(1, np.round(ph_weight_train)).astype(int)
    ph_train_weighted = ph_train_df.loc[ph_train_df.index.repeat(replication_counts)].reset_index(drop=True)

    finetuned_models = finetune_logspace(source_models, ph_train_weighted[feature_cols], ph_train_weighted[TARGET_COLUMN],
                                          ph_val_df[feature_cols], ph_val_df[TARGET_COLUMN])

    joblib.dump(finetuned_models["xgboost"], models_dir / "finetuned_xgboost.joblib")
    joblib.dump(finetuned_models["lightgbm"], models_dir / "finetuned_lightgbm.joblib")
    finetuned_models["catboost"].save_model(str(models_dir / "finetuned_catboost.cbm"))

    ensemble_weights = compute_ensemble_weights(finetuned_models, ph_val_df[feature_cols], ph_val_df[TARGET_COLUMN])
    holdout_preds = predict_ensemble_real(finetuned_models, ensemble_weights, ph_holdout_df[feature_cols])
    metrics = compute_metrics(ph_holdout_df[TARGET_COLUMN], holdout_preds["ensemble"])

    n_negative = int((holdout_preds["ensemble"] < 0).sum())
    abs_error = np.abs(holdout_preds["ensemble"] - ph_holdout_df[TARGET_COLUMN].values)

    predictions_df = pd.DataFrame({
        IDENTIFIER_COLUMN: ph_holdout_df[IDENTIFIER_COLUMN].values, "actual_pm25": ph_holdout_df[TARGET_COLUMN].values,
        "ensemble_pred": holdout_preds["ensemble"],
    })
    predictions_df.to_csv(run_dir / "predictions_holdout.csv", index=False)
    (run_dir / "feature_list.json").write_text(json.dumps({"n_features": len(feature_cols), "features": feature_cols}, indent=2))

    # --- Feature importance: gain-based (native) ---
    xgb_gain = finetuned_models["xgboost"].get_booster().get_score(importance_type="gain")
    lgbm_gain = dict(zip(feature_cols, finetuned_models["lightgbm"].booster_.feature_importance(importance_type="gain")))
    cb_gain = dict(zip(feature_cols, finetuned_models["catboost"].get_feature_importance()))
    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "xgboost_gain": [xgb_gain.get(f"f{i}", 0.0) for i in range(len(feature_cols))],
        "lightgbm_gain": [lgbm_gain.get(f, 0.0) for f in feature_cols],
        "catboost_gain": [cb_gain.get(f, 0.0) for f in feature_cols],
    })
    importance_df.to_csv(run_dir / "feature_importance_gain.csv", index=False)

    # --- SHAP (XGBoost, sample of holdout) ---
    rng = np.random.RandomState(RANDOM_SEED)
    shap_sample = ph_holdout_df[feature_cols].sample(n=min(SHAP_SAMPLE_SIZE, len(ph_holdout_df)), random_state=RANDOM_SEED)
    explainer = shap.TreeExplainer(finetuned_models["xgboost"])
    shap_values = explainer.shap_values(shap_sample)
    mean_abs_shap = pd.DataFrame({"feature": feature_cols, "mean_abs_shap": np.abs(shap_values).mean(axis=0)})
    mean_abs_shap = mean_abs_shap.sort_values("mean_abs_shap", ascending=False)
    mean_abs_shap.to_csv(run_dir / "shap_importance.csv", index=False)

    return {
        "configuration": config_name, "n_features": len(feature_cols),
        "rmse": metrics["RMSE"], "mae": metrics["MAE"], "r2": metrics["R2"],
        "n_negative": n_negative, "pct_negative": n_negative / len(predictions_df) * 100,
        "pred_std": float(holdout_preds["ensemble"].std()),
        "p95_abs_error": float(pd.Series(abs_error).quantile(0.95)), "max_abs_error": float(abs_error.max()),
        "importance_df": importance_df, "shap_df": mean_abs_shap,
    }


# ============================================================
# Main
# ============================================================

def main():
    OUTPUT_ROOT.mkdir(exist_ok=True)

    print("Loading source features and the existing baseline source cell split...")
    source_df = pd.read_csv(SOURCE_FEATURES_PATH)
    source_cell_split = pd.read_csv(BASELINE_DIR / "baseline_cell_split_assignment.csv")
    source_meta = pd.read_csv(SOURCE_METADATA_PATH)
    if len(source_df) != len(source_meta):
        raise ValueError("Source features and metadata row counts differ.")
    source_df = pd.concat([source_df.reset_index(drop=True),
                            source_meta[["datetime_utc", "centroid_latitude", "centroid_longitude"]].reset_index(drop=True)], axis=1)
    source_df = source_df.merge(source_cell_split[[IDENTIFIER_COLUMN, "split"]], on=IDENTIFIER_COLUMN, how="left")

    print("Loading Philippines features, metadata, and the existing (unmodified) fine-tune split...")
    ph_df = pd.read_csv(PH_FEATURES_PATH)
    ph_meta = pd.read_csv(PH_METADATA_PATH)
    if len(ph_df) != len(ph_meta):
        raise ValueError("Philippines features and metadata row counts differ.")
    ph_df = pd.concat([ph_df.reset_index(drop=True),
                        ph_meta[["datetime_utc", "centroid_latitude", "centroid_longitude"]].reset_index(drop=True)], axis=1)
    ph_split = pd.read_csv(PH_SPLIT_PATH)
    ph_df = ph_df.merge(ph_split[[IDENTIFIER_COLUMN, "split"]], on=IDENTIFIER_COLUMN, how="left")
    if ph_df["split"].isna().any():
        raise ValueError("Some Philippine rows have no split assignment.")

    base_feature_cols = [c for c in source_df.columns
                         if c not in (IDENTIFIER_COLUMN, TARGET_COLUMN, "split", "datetime_utc", "centroid_latitude", "centroid_longitude")]

    print("\nBuilding Group 1 (physics) features for both domains...")
    source_df_physics = add_physics_features(source_df)
    ph_df_physics = add_physics_features(ph_df)

    print("Building cell adjacency tables (once per domain) for Group 2 (transport) features...")
    source_centroids = source_df[[IDENTIFIER_COLUMN, "centroid_latitude", "centroid_longitude"]].drop_duplicates(subset=[IDENTIFIER_COLUMN])
    ph_centroids = ph_df[[IDENTIFIER_COLUMN, "centroid_latitude", "centroid_longitude"]].drop_duplicates(subset=[IDENTIFIER_COLUMN])
    source_adjacency = build_cell_adjacency(source_centroids, ADJACENCY_RADIUS_KM)
    ph_adjacency = build_cell_adjacency(ph_centroids, ADJACENCY_RADIUS_KM)

    print("Computing Group 2 (wind transport) features for both domains (this loops over grid cells)...")
    source_df_transport = add_wind_transport_features(source_df, source_adjacency)
    ph_df_transport = add_wind_transport_features(ph_df, ph_adjacency)

    source_df_both = add_physics_features(source_df_transport)
    ph_df_both = add_physics_features(ph_df_transport)

    configs = {
        "1_baseline": (source_df, ph_df, base_feature_cols),
        "2_physics": (source_df_physics, ph_df_physics, base_feature_cols + PHYSICS_FEATURE_NAMES),
        "3_transport": (source_df_transport, ph_df_transport, base_feature_cols + TRANSPORT_FEATURE_NAMES),
        "4_both": (source_df_both, ph_df_both, base_feature_cols + PHYSICS_FEATURE_NAMES + TRANSPORT_FEATURE_NAMES),
    }

    results = []
    for config_name, (s_df, p_df, feat_cols) in configs.items():
        print(f"\n=== Configuration {config_name} ({len(feat_cols)} features) ===")
        result = run_configuration(config_name, s_df, p_df, feat_cols)
        results.append(result)
        print(f"  RMSE={result['rmse']:.4f}  MAE={result['mae']:.4f}  R2={result['r2']:.4f}  "
              f"neg={result['pct_negative']:.2f}%  pred_std={result['pred_std']:.2f}")

    # --- Report ---
    lines = ["Physics and Wind-Transport Feature Ablation -- Comparison Report", "=" * 65, ""]
    for r in results:
        lines.append(f"{r['configuration']} ({r['n_features']} features):")
        lines.append(f"  RMSE={r['rmse']:.4f}  MAE={r['mae']:.4f}  R2={r['r2']:.4f}")
        lines.append(f"  Negative predictions: {r['n_negative']} ({r['pct_negative']:.2f}%)")
        lines.append(f"  Prediction std={r['pred_std']:.2f}  P95 abs error={r['p95_abs_error']:.2f}  max abs error={r['max_abs_error']:.2f}")
        lines.append("")

    baseline_result = results[0]
    lines.append("=" * 65)
    lines.append("ABLATION DELTAS (vs. 1_baseline)")
    lines.append("=" * 65)
    for r in results[1:]:
        lines.append(f"\n{r['configuration']}: RMSE delta={r['rmse']-baseline_result['rmse']:+.4f}  "
                      f"R2 delta={r['r2']-baseline_result['r2']:+.4f}  "
                      f"max_abs_error delta={r['max_abs_error']-baseline_result['max_abs_error']:+.2f}")

    lines.append("\n" + "=" * 65)
    lines.append("NEW-FEATURE USAGE (are the engineered physics/transport features actually used?)")
    lines.append("=" * 65)
    for r in results[1:]:
        new_feats = [f for f in (PHYSICS_FEATURE_NAMES if "physics" in r["configuration"] else []) +
                     (TRANSPORT_FEATURE_NAMES if "transport" in r["configuration"] else []) +
                     (PHYSICS_FEATURE_NAMES + TRANSPORT_FEATURE_NAMES if "both" in r["configuration"] else [])]
        new_feats = list(dict.fromkeys(new_feats))  # dedupe, preserve order
        if not new_feats:
            continue
        lines.append(f"\n{r['configuration']} -- SHAP ranking of new features (vs. all {r['n_features']} features):")
        shap_df = r["shap_df"].reset_index(drop=True)
        shap_df["rank"] = shap_df.index + 1
        for feat in new_feats:
            match = shap_df[shap_df["feature"] == feat]
            if len(match):
                lines.append(f"  {feat:<30} SHAP rank {int(match['rank'].iloc[0])}/{len(shap_df)}  "
                              f"mean|SHAP|={match['mean_abs_shap'].iloc[0]:.4f}")

    report_path = OUTPUT_ROOT / "physics_transport_ablation_report.txt"
    report_path.write_text("\n".join(lines))
    print(f"\nWrote {report_path}")
    print("\nPhysics and wind-transport ablation experiment complete.")


if __name__ == "__main__":
    main()
