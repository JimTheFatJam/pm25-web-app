"""
Experiment 1: Runtime Feature Ablation (Additional Features)
====================================================================

Research question: do the newly collected additional features (fire,
consecutive dry days, calendar/holiday indicators) improve PM2.5
prediction beyond the current best pipeline, with EVERYTHING else held
identical?

WITHOUT-features run reuses the already-trained "alpha_capacity" Stage A
source model directly (no retraining) -- it is, by definition, the
current best pipeline exactly as established. WITH-features run requires
a fresh source retrain (new columns mean the previously-trained source
model cannot be reused, same reasoning as run_lag_feature_experiment.py's
own lag-feature retrain) -- everything else (alpha=0.75, capacity=400,
log-space, ensemble reconstruction, all three validation methods) stays
byte-for-byte identical between the two runs, so the ONLY variable that
differs is whether additional_features_loader's merge is applied.

The runtime merge itself (additional_features_loader.py) is a separate,
reusable module -- imported here, not reimplemented -- following
lag_features.py's established in-memory-merge-only philosophy. Its own
integrity checks (row count unchanged, no duplicate/missing keys) run
automatically before any training begins and STOP immediately on
failure, per explicit instruction that these are correctness checks, not
experimental diagnostics.

Per explicit instruction, this experiment implements ONLY the WITH vs
WITHOUT comparison -- quantile regression, gated quantile regression,
and feature-group ablations are deliberately NOT implemented here; they
are contingent on this experiment's own results.
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
import additional_features_loader as afl
from run_logspace_pipeline_experiment import (
    train_source_logspace, finetune_logspace, compute_ensemble_weights, predict_ensemble_real,
    IDENTIFIER_COLUMN, TARGET_COLUMN, RANDOM_SEED,
)
from run_logspace_r2_improvement_experiment import compute_magnitude_weights

ALPHA = 0.75
MAX_ADDITIONAL_ROUNDS = 400
CELLS_TO_EXCLUDE = ["PH000000828"]

OUTPUT_DIR = dc.DIAGNOSTICS_OUTPUT_ROOT / "feature_ablation_experiment"
HIGH_PM_QUANTILE = 0.90
SHAP_SAMPLE_SIZE = 3000
NOISE_FLOOR_RMSE = 0.058

# INCLUDE_ADDITIONAL_FEATURES: the single flag controlling the runtime
# merge, per explicit instruction ("clean optional module controlled by
# a flag"). Toggled internally by this experiment's own WITH/WITHOUT
# loop -- exposed at module level so other experiments can import and
# set it without duplicating the merge logic.
INCLUDE_ADDITIONAL_FEATURES = False


def load_data_with_optional_additional_features(domain: str, include_additional: bool) -> tuple:
    """domain: 'source' or 'philippines'. Returns (df, feature_cols) where
    df has grid_id, datetime_utc, target, and every feature column
    (original, plus new ones if include_additional=True)."""
    source_df, ph_df, all_feature_cols = dc.load_data()
    if domain == "source":
        base_df = source_df.drop(columns=["split"], errors="ignore")
        metadata_path = dc.SCRIPT_DIR / "source_metadata.csv"
    else:
        base_df = ph_df.drop(columns=["split"], errors="ignore")
        metadata_path = dc.SCRIPT_DIR / "philippines_metadata.csv"

    metadata_df = pd.read_csv(metadata_path, usecols=[IDENTIFIER_COLUMN, "datetime_utc"])
    if len(metadata_df) != len(base_df):
        raise ValueError(f"{domain} features/metadata row count mismatch.")

    if not include_additional:
        result_df = base_df.copy()
        result_df["datetime_utc"] = metadata_df["datetime_utc"].values
        return result_df, all_feature_cols

    merged_df, new_feature_cols = afl.load_and_merge_features(base_df, metadata_df, domain)
    return merged_df, all_feature_cols + new_feature_cols


def train_and_evaluate_pipeline(ph_df: pd.DataFrame, feature_cols: list, label: str) -> dict:
    """Trains source (fresh -- see module docstring for why WITH-features
    cannot reuse the existing alpha_capacity model) then evaluates via
    ALL THREE validation methods (spatial holdout frozen, random holdout,
    5-fold), exactly mirroring how every prior experiment in this project
    has done this. Returns metrics per method plus the fitted models/
    weights for the spatial-holdout run, used later for SHAP."""
    print(f"\n{'#'*60}\n# {label}: training source ensemble ({len(feature_cols)} features)\n{'#'*60}")

    source_df_full, _, _ = dc.load_data()
    if label.startswith("WITH"):
        # Attach datetime_utc via a SAFE positional concat done ONCE,
        # BEFORE filtering -- boolean masking on split afterward then
        # naturally preserves row correspondence, unlike an earlier
        # version of this function which filtered first and then tried
        # to re-look-up matching metadata rows via source_train.index
        # positionally -- exactly the kind of index-alignment fragility
        # that has caused real bugs elsewhere in this project. Concat
        # BEFORE filtering avoids that risk entirely rather than relying
        # on index preservation through a later lookup.
        source_metadata_full = pd.read_csv(dc.SCRIPT_DIR / "source_metadata.csv", usecols=[IDENTIFIER_COLUMN, "datetime_utc"])
        if len(source_metadata_full) != len(source_df_full):
            raise ValueError("source_features.csv/source_metadata.csv row count mismatch.")
        source_df_full = source_df_full.copy()
        source_df_full["datetime_utc"] = source_metadata_full["datetime_utc"].values

    source_train = source_df_full[source_df_full["split"] == "train"]
    if label.startswith("WITH"):
        source_train_base = source_train.drop(columns=["split", "datetime_utc"]).reset_index(drop=True)
        source_train_metadata = source_train[["datetime_utc"]].reset_index(drop=True)
        # source_train_metadata needs grid_id too, for load_and_merge_features's own merge keys
        source_train_metadata = pd.concat([source_train[[IDENTIFIER_COLUMN]].reset_index(drop=True), source_train_metadata], axis=1)
        source_train, new_cols = afl.load_and_merge_features(source_train_base, source_train_metadata, "source")

    source_weight = compute_magnitude_weights(source_train[TARGET_COLUMN], ALPHA)
    if label.startswith("WITHOUT"):
        # Reuse the already-trained "alpha_capacity" Stage A source model
        # directly -- this IS the current best pipeline exactly, so
        # reusing it (rather than retraining from scratch with a
        # different random seed's tree-building order) guarantees this
        # experiment's WITHOUT arm is byte-for-byte the established
        # baseline, not a fresh approximation of it. An earlier version
        # of this function trained from scratch for BOTH arms, which
        # contradicted this function's own docstring and introduced
        # unnecessary training-order noise into what should be an exact
        # reference point -- caught and fixed before running for real.
        stage_a_dir = dc.DIAGNOSTICS_OUTPUT_ROOT / ov.STAGE_A_NAME
        source_models = ov.load_source_models(stage_a_dir / "alpha_capacity" / "models")
    else:
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


def _finetune_and_eval(source_models, train_df, val_df, target_df, feature_cols):
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
    metrics = compute_full_metrics(preds_df["actual_pm25"], preds_df["predicted"])
    return finetuned, weights, metrics, preds_df


def compute_full_metrics(actual, predicted) -> dict:
    m = dc.compute_metrics(actual, predicted)
    actual_arr, predicted_arr = np.asarray(actual), np.asarray(predicted)
    high_pm_threshold = pd.Series(actual_arr).quantile(HIGH_PM_QUANTILE)
    mask = actual_arr >= high_pm_threshold
    high_pm_mae = np.abs(predicted_arr[mask] - actual_arr[mask]).mean() if mask.sum() > 0 else np.nan
    return {"rmse": m["RMSE"], "mae": m["MAE"], "r2": m["R2"], "high_pm_mae": high_pm_mae}


# ============================================================
# SHAP analysis: existing vs. new features, top-20, high-PM-specific
# contribution -- WITH-features model only, per instruction.
# ============================================================

def run_shap_analysis(with_result: dict, original_feature_cols: list, new_feature_cols: list):
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
    shap_df["group"] = shap_df["feature"].apply(lambda f: "new" if f in new_feature_cols else "existing")
    shap_df.to_csv(OUTPUT_DIR / "tables" / "shap_full_ranking.csv", index=False)

    existing_df = shap_df[shap_df["group"] == "existing"]
    new_df = shap_df[shap_df["group"] == "new"]
    existing_df.to_csv(OUTPUT_DIR / "tables" / "shap_existing_features.csv", index=False)
    new_df.to_csv(OUTPUT_DIR / "tables" / "shap_new_features.csv", index=False)

    top20 = shap_df.head(20)
    new_in_top20 = top20[top20["group"] == "new"]

    # High-PM-specific SHAP: same explainer, restricted to the top-10%
    # actual-PM2.5 sample rows only, to answer "which new features
    # contribute most during extreme events" directly rather than
    # inferring it from the overall ranking alone.
    high_pm_threshold = preds_df["actual_pm25"].quantile(HIGH_PM_QUANTILE)
    high_pm_row_ids = preds_df[preds_df["actual_pm25"] >= high_pm_threshold].index
    high_pm_sample_idx = np.intersect1d(sample_idx, high_pm_row_ids.values)
    if len(high_pm_sample_idx) >= 10:
        high_pm_features = holdout_df.iloc[high_pm_sample_idx][feature_cols]
        high_pm_shap_values = explainer.shap_values(high_pm_features)
        high_pm_shap_df = pd.DataFrame({"feature": feature_cols, "mean_abs_shap_high_pm": np.abs(high_pm_shap_values).mean(axis=0)})
        high_pm_shap_df = high_pm_shap_df.sort_values("mean_abs_shap_high_pm", ascending=False).reset_index(drop=True)
        high_pm_shap_df["group"] = high_pm_shap_df["feature"].apply(lambda f: "new" if f in new_feature_cols else "existing")
        high_pm_shap_df.to_csv(OUTPUT_DIR / "tables" / "shap_high_pm_only.csv", index=False)
        new_high_pm = high_pm_shap_df[high_pm_shap_df["group"] == "new"].head(10)
    else:
        high_pm_shap_df, new_high_pm = None, None
        print(f"  Only {len(high_pm_sample_idx)} high-PM rows in the SHAP sample -- skipping high-PM-specific SHAP "
              f"(too few for a meaningful estimate), not fabricating one.")

    # Specific feature-group questions, per instruction.
    fire_cols = [c for c in new_feature_cols if "fire" in c or "hotspot" in c or "frp" in c]
    dry_days_cols = [c for c in new_feature_cols if "dry_days" in c]
    calendar_cols = [c for c in new_feature_cols if c.startswith("is_")]

    def group_summary(cols, name):
        sub = shap_df[shap_df["feature"].isin(cols)]
        if len(sub) == 0:
            return f"{name}: no matching columns found."
        best = sub.iloc[0]
        return (f"{name}: {len(sub)} feature(s), best-ranked = {best['feature']} (rank {best['rank']}/{len(shap_df)}, "
                f"mean|SHAP|={best['mean_abs_shap']:.4f}), {(sub['mean_abs_shap']>0).sum()}/{len(sub)} used (non-zero SHAP)")

    lines = ["Experiment 1 -- SHAP Interpretability Report", "=" * 50, "",
             f"New features in the overall Top 20: {len(new_in_top20)}/20",
             f"  {new_in_top20[['feature','rank','mean_abs_shap']].to_string(index=False) if len(new_in_top20) else '(none)'}",
             "", "Feature-group summaries (per explicit questions):",
             "  " + group_summary(fire_cols, "Fire-related variables"),
             "  " + group_summary(dry_days_cols, "Consecutive dry days"),
             "  " + group_summary(calendar_cols, "Holiday/weekend indicators"),
             ]
    if new_high_pm is not None:
        lines.append(f"\nTop new features specifically during high-PM episodes (top 10%):")
        lines.append("  " + new_high_pm[["feature", "mean_abs_shap_high_pm"]].to_string(index=False))

    (OUTPUT_DIR / "shap_interpretation.txt").write_text("\n".join(lines))
    print("\n".join(lines))

    fig, ax = plt.subplots(figsize=(8, 6))
    top15 = shap_df.head(15)
    colors = ["tab:orange" if g == "new" else "tab:blue" for g in top15["group"]]
    ax.barh(top15["feature"], top15["mean_abs_shap"], color=colors)
    ax.invert_yaxis()
    ax.set_xlabel("mean |SHAP value|")
    ax.set_title("Top 15 Features (orange = new additional features)")
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

    print("Loading data WITHOUT additional features...")
    ph_df_without, feature_cols_without = load_data_with_optional_additional_features("philippines", include_additional=False)
    ph_df_without = ph_df_without[~ph_df_without[IDENTIFIER_COLUMN].isin(CELLS_TO_EXCLUDE)].reset_index(drop=True)

    print("Loading data WITH additional features (runtime merge, integrity-checked)...")
    ph_df_with, feature_cols_with = load_data_with_optional_additional_features("philippines", include_additional=True)
    ph_df_with = ph_df_with[~ph_df_with[IDENTIFIER_COLUMN].isin(CELLS_TO_EXCLUDE)].reset_index(drop=True)
    new_feature_cols = [c for c in feature_cols_with if c not in feature_cols_without]
    print(f"  {len(new_feature_cols)} new feature(s): {new_feature_cols}")

    without_result = train_and_evaluate_pipeline(ph_df_without, feature_cols_without, "WITHOUT additional features")
    with_result = train_and_evaluate_pipeline(ph_df_with, feature_cols_with, "WITH additional features")

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

    shap_df = run_shap_analysis(with_result, feature_cols_without, new_feature_cols)

    spatial_row = comparison_df[comparison_df["validation_method"] == "spatial_holdout"].iloc[0]
    fold5_row = comparison_df[comparison_df["validation_method"] == "5fold"].iloc[0]
    meaningful = (spatial_row["delta_rmse"] < -NOISE_FLOOR_RMSE) and (fold5_row["delta_rmse"] < -NOISE_FLOOR_RMSE)
    verdict = ("MEANINGFUL IMPROVEMENT -- proceed to Experiment 2" if meaningful else
               "NO MEANINGFUL IMPROVEMENT beyond the noise floor on both spatial holdout and 5-fold -- "
               "Experiments 2-4 are not yet justified by this result")
    (OUTPUT_DIR / "verdict.txt").write_text(
        f"Experiment 1 Verdict\n{'='*30}\nSpatial holdout RMSE delta: {spatial_row['delta_rmse']:+.4f}\n"
        f"5-fold RMSE delta: {fold5_row['delta_rmse']:+.4f}\nNoise floor: {NOISE_FLOOR_RMSE}\n\n{verdict}")
    print(f"\n{verdict}")
    print(f"\nExperiment 1 complete. Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
