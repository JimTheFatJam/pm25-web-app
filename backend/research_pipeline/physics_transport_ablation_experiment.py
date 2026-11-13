"""
Experiment: Physics & Wind-Transport Feature Engineering (Combined)
==========================================================================

Extends Experiment 1 rather than replacing it: baseline feature set vs.
baseline + (consecutive_dry_days + calendar/holiday features, already
validated in Experiment 1) + (NEW physics interaction features) + (NEW
wind-transport features). Fire features explicitly excluded, per
Experiment 1's own finding (zero SHAP importance across all 14 fire
columns).

Reuses, not reimplements:
  - additional_features_loader.py's integrity-checked merge machinery
    (via combined_physics_features_loader.py, which also excludes fire).
  - run_physics_transport_experiment.py's already-validated
    add_physics_features / add_wind_transport_features / build_cell_adjacency
    -- including a previously-fixed 180-degree wind-direction sign bug,
    reused directly rather than risking reintroducing it.
  - feature_ablation_experiment.py's generic _finetune_and_eval /
    compute_full_metrics helpers (feature-set-agnostic, imported
    directly, not duplicated).

METHODOLOGICAL CORRECTION, flagged directly before implementation began:
the original prompt for this experiment specified alpha=1.0, carried
over from run_physics_transport_experiment.py, which fixed that value
because it WAS the established best value at the time that script was
built. A later alpha-sensitivity experiment revised the established best
value to alpha=0.75, used in every experiment since (including
Experiment 1, immediately prior to this one). This experiment uses
alpha=0.75 to actually satisfy its own stated principle ("exactly one
variable changes, everything else matches the current best validated
pipeline") -- using 1.0 would have silently changed a second variable.
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
    train_source_logspace, IDENTIFIER_COLUMN, TARGET_COLUMN, RANDOM_SEED,
)
from run_logspace_r2_improvement_experiment import compute_magnitude_weights
from feature_ablation_experiment import _finetune_and_eval, compute_full_metrics, HIGH_PM_QUANTILE

ALPHA = 0.75  # see module docstring -- corrected from the originally-specified 1.0
MAX_ADDITIONAL_ROUNDS = 400
CELLS_TO_EXCLUDE = ["PH000000828"]

OUTPUT_DIR = dc.DIAGNOSTICS_OUTPUT_ROOT / "physics_transport_ablation_experiment"
SHAP_SAMPLE_SIZE = 3000
NOISE_FLOOR_RMSE = 0.058


def load_domain_with_combined_features(domain: str, include_new: bool) -> tuple:
    """domain: 'source' or 'philippines'. Mirrors
    feature_ablation_experiment.py's load_data_with_optional_additional_features,
    but building the COMBINED (non-fire additional + physics + transport)
    feature set instead of the full additional-features set."""
    source_df, ph_df, all_feature_cols = dc.load_data()
    if domain == "source":
        base_df = source_df.drop(columns=["split"], errors="ignore")
        metadata_path = dc.SCRIPT_DIR / "source_metadata.csv"
    else:
        base_df = ph_df.drop(columns=["split"], errors="ignore")
        metadata_path = dc.SCRIPT_DIR / "philippines_metadata.csv"

    metadata_df = pd.read_csv(metadata_path, usecols=[IDENTIFIER_COLUMN, "datetime_utc", "centroid_latitude", "centroid_longitude"])
    if len(metadata_df) != len(base_df):
        raise ValueError(f"{domain} features/metadata row count mismatch.")

    if not include_new:
        result_df = base_df.copy()
        result_df["datetime_utc"] = metadata_df["datetime_utc"].values
        return result_df, all_feature_cols

    combined_df, new_feature_cols = cpl.build_combined_features(base_df, metadata_df, domain)
    return combined_df, all_feature_cols + new_feature_cols


def train_and_evaluate_pipeline(ph_df: pd.DataFrame, feature_cols: list, label: str) -> dict:
    """Mirrors feature_ablation_experiment.py's train_and_evaluate_pipeline
    exactly in structure (reuse the trained alpha_capacity source model
    for BASELINE, fresh source training for the expanded feature set),
    adapted here to build the combined (non-fire additional + physics +
    transport) source-domain feature set for the WITH arm."""
    print(f"\n{'#'*60}\n# {label}: training source ensemble ({len(feature_cols)} features)\n{'#'*60}")

    source_df_full, _, _ = dc.load_data()
    if label.startswith("WITH"):
        source_metadata_full = pd.read_csv(dc.SCRIPT_DIR / "source_metadata.csv",
                                             usecols=[IDENTIFIER_COLUMN, "datetime_utc", "centroid_latitude", "centroid_longitude"])
        if len(source_metadata_full) != len(source_df_full):
            raise ValueError("source_features.csv/source_metadata.csv row count mismatch.")
        source_df_full = source_df_full.copy()
        for col in ["datetime_utc", "centroid_latitude", "centroid_longitude"]:
            source_df_full[col] = source_metadata_full[col].values

    source_train = source_df_full[source_df_full["split"] == "train"]
    if label.startswith("WITHOUT"):
        stage_a_dir = dc.DIAGNOSTICS_OUTPUT_ROOT / ov.STAGE_A_NAME
        source_models = ov.load_source_models(stage_a_dir / "alpha_capacity" / "models")
    else:
        source_train_base = source_train.drop(columns=["split", "datetime_utc", "centroid_latitude", "centroid_longitude"]).reset_index(drop=True)
        source_train_metadata = pd.concat([
            source_train[[IDENTIFIER_COLUMN]].reset_index(drop=True),
            source_train[["datetime_utc", "centroid_latitude", "centroid_longitude"]].reset_index(drop=True),
        ], axis=1)
        source_train, new_cols = cpl.build_combined_features(source_train_base, source_train_metadata, "source")
        source_weight = compute_magnitude_weights(source_train[TARGET_COLUMN], ALPHA)
        source_models = train_source_logspace(source_train[feature_cols], source_train[TARGET_COLUMN], source_weight)

    results = {}

    print(f"\n  --- {label}: spatial holdout (frozen split) ---")
    frozen_split = pd.read_csv(dc.SCRIPT_DIR / "finetune_outputs" / "philippines_finetune_split_assignment.csv")
    ph_with_split = ph_df.merge(frozen_split[[IDENTIFIER_COLUMN, "split"]], on=IDENTIFIER_COLUMN, how="left")
    spatial_train = ph_with_split[ph_with_split["split"] == "finetune_train"]
    spatial_val = ph_with_split[ph_with_split["split"] == "finetune_val"]
    spatial_holdout = ph_with_split[ph_with_split["split"] == "holdout"]
    spatial_finetuned, spatial_weights, spatial_metrics, spatial_preds = _finetune_and_eval(
        source_models, spatial_train, spatial_val, spatial_holdout, feature_cols)
    results["spatial_holdout"] = spatial_metrics

    print(f"\n  --- {label}: random holdout ---")
    random_split = vs.random_holdout_split(ph_df, seed=RANDOM_SEED)
    ph_with_random = ph_df.copy()
    ph_with_random["split"] = random_split["split"].values
    random_train = ph_with_random[ph_with_random["split"] == "finetune_train"]
    random_val = ph_with_random[ph_with_random["split"] == "finetune_val"]
    random_holdout = ph_with_random[ph_with_random["split"] == "holdout"]
    _, _, random_metrics, _ = _finetune_and_eval(source_models, random_train, random_val, random_holdout, feature_cols)
    results["random_holdout"] = random_metrics

    print(f"\n  --- {label}: 5-fold ---")
    outer_folds = vs.kfold_cell_splits(ph_df, k=5, seed=RANDOM_SEED)
    fold_preds = []
    for fold_idx, fold_assignment in enumerate(outer_folds):
        fold_df = ph_df.merge(fold_assignment[[IDENTIFIER_COLUMN, "split"]], on=IDENTIFIER_COLUMN, how="left")
        f_train = fold_df[fold_df["split"] == "finetune_train"]
        f_val = fold_df[fold_df["split"] == "finetune_val"]
        f_target = fold_df[fold_df["split"] == "holdout"]
        _, _, _, f_preds = _finetune_and_eval(source_models, f_train, f_val, f_target, feature_cols)
        fold_preds.append(f_preds)
    pooled_5fold = pd.concat(fold_preds, ignore_index=True)
    results["5fold"] = compute_full_metrics(pooled_5fold["actual_pm25"], pooled_5fold["predicted"])

    return {"results": results, "spatial_finetuned_models": spatial_finetuned, "spatial_weights": spatial_weights,
            "spatial_holdout_df": spatial_holdout, "spatial_preds": spatial_preds, "feature_cols": feature_cols}


# ============================================================
# SHAP analysis: answers the 5 specific questions from the experiment
# design, split by feature GROUP (non-fire-additional / physics /
# transport / existing), not just new-vs-existing.
# ============================================================

def run_shap_analysis(with_result: dict, existing_feature_cols: list, non_fire_additional_cols: list):
    print("\n" + "#" * 60 + "\n# SHAP Analysis (WITH-features model)\n" + "#" * 60)
    finetuned_models = with_result["spatial_finetuned_models"]
    holdout_df = with_result["spatial_holdout_df"]
    feature_cols = with_result["feature_cols"]
    preds_df = with_result["spatial_preds"]

    sample_n = min(SHAP_SAMPLE_SIZE, len(holdout_df))
    sample_idx = np.random.RandomState(RANDOM_SEED).choice(len(holdout_df), size=sample_n, replace=False)
    sample_features = holdout_df.iloc[sample_idx][feature_cols]

    explainer = shap.TreeExplainer(finetuned_models["xgboost"])
    shap_values = explainer.shap_values(sample_features)
    shap_df = pd.DataFrame({"feature": feature_cols, "mean_abs_shap": np.abs(shap_values).mean(axis=0)})
    shap_df = shap_df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    shap_df["rank"] = shap_df.index + 1

    def classify(f):
        if f in cpl.PHYSICS_FEATURE_NAMES:
            return "physics"
        if f in cpl.TRANSPORT_FEATURE_NAMES:
            return "transport"
        if f in non_fire_additional_cols:
            return "calendar_dry_days"
        return "existing"
    shap_df["group"] = shap_df["feature"].apply(classify)
    shap_df.to_csv(OUTPUT_DIR / "tables" / "shap_full_ranking.csv", index=False)

    top20 = shap_df.head(20)
    physics_in_top20 = top20[top20["group"] == "physics"]
    transport_in_top20 = top20[top20["group"] == "transport"]
    calendar_in_top20 = top20[top20["group"] == "calendar_dry_days"]

    high_pm_threshold = preds_df["actual_pm25"].quantile(HIGH_PM_QUANTILE)
    high_pm_row_ids = preds_df[preds_df["actual_pm25"] >= high_pm_threshold].index
    high_pm_sample_idx = np.intersect1d(sample_idx, high_pm_row_ids.values)
    if len(high_pm_sample_idx) >= 10:
        high_pm_features = holdout_df.iloc[high_pm_sample_idx][feature_cols]
        high_pm_shap_values = explainer.shap_values(high_pm_features)
        high_pm_shap_df = pd.DataFrame({"feature": feature_cols, "mean_abs_shap_high_pm": np.abs(high_pm_shap_values).mean(axis=0)})
        high_pm_shap_df["group"] = high_pm_shap_df["feature"].apply(classify)
        high_pm_shap_df = high_pm_shap_df.sort_values("mean_abs_shap_high_pm", ascending=False).reset_index(drop=True)
        high_pm_shap_df.to_csv(OUTPUT_DIR / "tables" / "shap_high_pm_only.csv", index=False)
    else:
        high_pm_shap_df = None
        print(f"  Only {len(high_pm_sample_idx)} high-PM rows in the SHAP sample -- skipping high-PM SHAP, not fabricating one.")

    # Q4: do physics features outperform calendar/dry-days features?
    physics_best_rank = shap_df[shap_df["group"] == "physics"]["rank"].min() if (shap_df["group"] == "physics").any() else None
    calendar_best_rank = shap_df[shap_df["group"] == "calendar_dry_days"]["rank"].min() if (shap_df["group"] == "calendar_dry_days").any() else None

    lines = ["Physics/Transport Experiment -- SHAP Interpretability Report", "=" * 60, "",
             f"Q1. Physics features in Top 20: {len(physics_in_top20)}/5",
             f"    {physics_in_top20[['feature','rank','mean_abs_shap']].to_string(index=False) if len(physics_in_top20) else '(none)'}",
             f"\nQ2. Transport features in Top 20: {len(transport_in_top20)}/6",
             f"    {transport_in_top20[['feature','rank','mean_abs_shap']].to_string(index=False) if len(transport_in_top20) else '(none)'}",
             f"\nQ3. Top features specifically during high-PM episodes:",
             f"    {high_pm_shap_df.head(10)[['feature','mean_abs_shap_high_pm','group']].to_string(index=False) if high_pm_shap_df is not None else '(insufficient high-PM sample)'}",
             f"\nQ4. Physics vs. calendar/dry-days -- best rank comparison:",
             f"    Best physics feature rank: {physics_best_rank}  |  Best calendar/dry-days feature rank: {calendar_best_rank}",
             f"    ({'Physics features rank BETTER (lower)' if (physics_best_rank or 999) < (calendar_best_rank or 999) else 'Calendar/dry-days features rank BETTER (lower), or comparable'})",
             f"\nQ5. Does the model gain extreme-vs-normal discriminative information?",
             f"    See Q3 above -- compare which GROUPS dominate the high-PM-specific ranking vs. the overall ranking; "
             f"a group ranking meaningfully higher in the high-PM list than in the overall list is evidence of that "
             f"group specifically helping distinguish extreme from ordinary conditions, not just adding general signal.",
             ]
    (OUTPUT_DIR / "shap_interpretation.txt").write_text("\n".join(lines))
    print("\n".join(lines))

    fig, ax = plt.subplots(figsize=(8, 6))
    top15 = shap_df.head(15)
    color_map = {"physics": "tab:red", "transport": "tab:green", "calendar_dry_days": "tab:orange", "existing": "tab:blue"}
    colors = [color_map[g] for g in top15["group"]]
    ax.barh(top15["feature"], top15["mean_abs_shap"], color=colors)
    ax.invert_yaxis()
    ax.set_xlabel("mean |SHAP value|")
    ax.set_title("Top 15 (red=physics, green=transport, orange=calendar/dry-days, blue=existing)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "figures" / "shap_top15.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    return shap_df


# ============================================================
# Main
# ============================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "tables").mkdir(exist_ok=True)
    (OUTPUT_DIR / "figures").mkdir(exist_ok=True)

    print("Loading data WITHOUT new features (baseline)...")
    ph_df_without, feature_cols_without = load_domain_with_combined_features("philippines", include_new=False)
    ph_df_without = ph_df_without[~ph_df_without[IDENTIFIER_COLUMN].isin(CELLS_TO_EXCLUDE)].reset_index(drop=True)

    print("Loading data WITH combined features (non-fire additional + physics + transport)...")
    ph_df_with, feature_cols_with = load_domain_with_combined_features("philippines", include_new=True)
    ph_df_with = ph_df_with[~ph_df_with[IDENTIFIER_COLUMN].isin(CELLS_TO_EXCLUDE)].reset_index(drop=True)
    new_feature_cols = [c for c in feature_cols_with if c not in feature_cols_without]
    non_fire_additional_cols = [c for c in new_feature_cols if c not in cpl.PHYSICS_FEATURE_NAMES and c not in cpl.TRANSPORT_FEATURE_NAMES]
    print(f"  {len(new_feature_cols)} new feature(s): {non_fire_additional_cols} + "
          f"{cpl.PHYSICS_FEATURE_NAMES} + {cpl.TRANSPORT_FEATURE_NAMES}")

    without_result = train_and_evaluate_pipeline(ph_df_without, feature_cols_without, "WITHOUT new features (baseline)")
    with_result = train_and_evaluate_pipeline(ph_df_with, feature_cols_with, "WITH combined features")

    comparison_rows = []
    for method in ["spatial_holdout", "random_holdout", "5fold"]:
        wo, wi = without_result["results"][method], with_result["results"][method]
        comparison_rows.append({
            "validation_method": method,
            "without_rmse": wo["rmse"], "with_rmse": wi["rmse"], "delta_rmse": wi["rmse"] - wo["rmse"],
            "without_mae": wo["mae"], "with_mae": wi["mae"], "delta_mae": wi["mae"] - wo["mae"],
            "without_r2": wo["r2"], "with_r2": wi["r2"], "delta_r2": wi["r2"] - wo["r2"],
            "without_high_pm_mae": wo["high_pm_mae"], "with_high_pm_mae": wi["high_pm_mae"],
            "delta_high_pm_mae": wi["high_pm_mae"] - wo["high_pm_mae"],
        })
    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(OUTPUT_DIR / "tables" / "with_vs_without_comparison.csv", index=False)
    print("\n" + comparison_df.to_string(index=False))

    shap_df = run_shap_analysis(with_result, feature_cols_without, non_fire_additional_cols)

    spatial_row = comparison_df[comparison_df["validation_method"] == "spatial_holdout"].iloc[0]
    fold5_row = comparison_df[comparison_df["validation_method"] == "5fold"].iloc[0]
    meaningful = (spatial_row["delta_rmse"] < -NOISE_FLOOR_RMSE) and (fold5_row["delta_rmse"] < -NOISE_FLOOR_RMSE)
    verdict = ("MEANINGFUL IMPROVEMENT" if meaningful else
               "NO MEANINGFUL IMPROVEMENT beyond the noise floor on both spatial holdout and 5-fold")
    (OUTPUT_DIR / "verdict.txt").write_text(
        f"Physics/Transport Experiment Verdict\n{'='*40}\nSpatial holdout RMSE delta: {spatial_row['delta_rmse']:+.4f}\n"
        f"5-fold RMSE delta: {fold5_row['delta_rmse']:+.4f}\nNoise floor: {NOISE_FLOOR_RMSE}\n\n{verdict}")
    print(f"\n{verdict}")
    print(f"\nExperiment complete. Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
