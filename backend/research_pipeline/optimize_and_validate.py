"""
Optimization + Validation Strategy Pipeline
=================================================

One end-to-end script, two internal stages:

Stage A (optimization): full 2x2x2 factorial over {alpha in [1.0, 0.75]},
{Cambodia in [included, excluded]}, {capacity in [200, 400]} -- 8
configurations, reusing diagnostics_core.run_experiment_suite() exactly
as every other diagnostics experiment does. Determines the winner by
fine-tuned Philippine holdout RMSE and saves it to best_pipeline.json.

Stage B (validation strategy comparison): loads best_pipeline.json
automatically, and evaluates BOTH the baseline and optimized pipelines
under four validation methods -- spatial holdout (existing), random
holdout (row-level, literature-comparison artifact only), cell-level
5-fold, cell-level 10-fold. Rolling/expanding-window is deliberately
excluded from this comparison (see the preceding critical review: it
measures temporal, not spatial, generalization -- a different question
from what this comparison is answering).

No source-domain retraining happens in Stage B at all -- both source
models it needs (baseline's, and the Stage A winner's) were already
trained and saved by Stage A. Stage B only repeats Philippine-side
fine-tuning, once per (pipeline x validation-method x fold) combination.

Every statistical comparison (baseline vs. optimized, per validation
method) aggregates to per-cell mean error before running the Wilcoxon
test -- including under random holdout, where the SPLIT itself is
row-level by design, but the STATISTICAL TEST must not be, to avoid
reintroducing the row-level autocorrelation problem this project has
been careful to avoid everywhere else.

Output
------
diagnostics_outputs/
  optimization_stage_a/            -- standard run_experiment_suite() output
  best_pipeline.json
  validation_strategy_comparison/
    <method>_<pipeline>/predictions.csv
    consolidated_comparison.csv    -- one row per (method x pipeline)
    consolidated_comparison_report.txt
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from scipy.stats import wilcoxon

import diagnostics_core as dc
import validation_strategies as vs
from run_logspace_pipeline_experiment import finetune_logspace, compute_ensemble_weights, predict_ensemble_real
from run_logspace_r2_improvement_experiment import compute_magnitude_weights

IDENTIFIER_COLUMN = dc.IDENTIFIER_COLUMN
TARGET_COLUMN = dc.TARGET_COLUMN
RANDOM_SEED = dc.RANDOM_SEED

STAGE_A_NAME = "optimization_stage_a"
BEST_PIPELINE_PATH = dc.DIAGNOSTICS_OUTPUT_ROOT / "best_pipeline.json"
STAGE_B_OUTPUT_DIR = dc.DIAGNOSTICS_OUTPUT_ROOT / "validation_strategy_comparison"

EXISTING_SPATIAL_SPLIT_PATH = dc.SCRIPT_DIR / "finetune_outputs" / "philippines_finetune_split_assignment.csv"


# ============================================================
# Stage A: factorial optimization
# ============================================================

def generate_stage_a_configs() -> list:
    return [
        dc.make_config("baseline"),
        dc.make_config("alpha_0.75", alpha=0.75),
        dc.make_config("exclude_KH", excluded_countries=["KH"]),
        dc.make_config("capacity_400", max_additional_rounds=400),
        dc.make_config("alpha_KH", alpha=0.75, excluded_countries=["KH"]),
        dc.make_config("alpha_capacity", alpha=0.75, max_additional_rounds=400),
        dc.make_config("KH_capacity", excluded_countries=["KH"], max_additional_rounds=400),
        dc.make_config("alpha_KH_capacity", alpha=0.75, excluded_countries=["KH"], max_additional_rounds=400),
    ]


def run_stage_a() -> dict:
    print("\n" + "#" * 60 + "\n# STAGE A: Factorial Optimization\n" + "#" * 60)
    configs = generate_stage_a_configs()
    comparison_df = dc.run_experiment_suite(STAGE_A_NAME, configs)

    winner_row = comparison_df.iloc[0]  # already sorted by rank_by_finetuned_rmse
    winner_name = winner_row["configuration"]
    winner_config = next(c for c in configs if c["name"] == winner_name)

    best_pipeline = {
        "configuration_name": winner_name,
        "alpha": winner_config["alpha"],
        "max_additional_rounds": winner_config["max_additional_rounds"],
        "excluded_countries": winner_config["excluded_countries"],
        "excluded_feature_groups": winner_config["excluded_feature_groups"],
        "finetuned_rmse": float(winner_row["finetuned_rmse"]),
        "finetuned_mae": float(winner_row["finetuned_mae"]),
        "finetuned_r2": float(winner_row["finetuned_r2"]),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    dc.DIAGNOSTICS_OUTPUT_ROOT.mkdir(exist_ok=True)
    BEST_PIPELINE_PATH.write_text(json.dumps(best_pipeline, indent=2))
    print(f"\nStage A winner: {winner_name} (RMSE={winner_row['finetuned_rmse']:.4f}) -- wrote best_pipeline.json")
    return best_pipeline


# ============================================================
# Stage B: validation strategy comparison
# ============================================================

def load_source_models(models_dir: Path) -> dict:
    xgb = joblib.load(models_dir / "source_xgboost.joblib")
    lgbm = joblib.load(models_dir / "source_lightgbm.joblib")
    cb = CatBoostRegressor()
    cb.load_model(str(models_dir / "source_catboost.cbm"))
    return {"xgboost": xgb, "lightgbm": lgbm, "catboost": cb}


def attach_split(ph_df: pd.DataFrame, split_assignment: pd.DataFrame, row_level: bool) -> pd.DataFrame:
    if row_level:
        result = ph_df.copy()
        result["split"] = split_assignment["split"].values  # same row order/length by construction
        return result
    return ph_df.merge(split_assignment[[IDENTIFIER_COLUMN, "split"]], on=IDENTIFIER_COLUMN, how="left")


def finetune_and_evaluate_fold(source_models: dict, ph_df_with_split: pd.DataFrame, feature_cols: list,
                                alpha: float) -> pd.DataFrame:
    """Fine-tunes the given (already-trained, unmodified) source models on
    one fold's Philippine split, returns holdout predictions for that fold.
    No source-domain training happens here at all."""
    train_df = ph_df_with_split[ph_df_with_split["split"] == "finetune_train"]
    val_df = ph_df_with_split[ph_df_with_split["split"] == "finetune_val"]
    holdout_df = ph_df_with_split[ph_df_with_split["split"] == "holdout"]

    weight_train = compute_magnitude_weights(train_df[TARGET_COLUMN], alpha)
    replication_counts = np.maximum(1, np.round(weight_train)).astype(int)
    train_weighted = train_df.loc[train_df.index.repeat(replication_counts)].reset_index(drop=True)

    finetuned = finetune_logspace(source_models, train_weighted[feature_cols], train_weighted[TARGET_COLUMN],
                                   val_df[feature_cols], val_df[TARGET_COLUMN])
    ensemble_weights = compute_ensemble_weights(finetuned, val_df[feature_cols], val_df[TARGET_COLUMN])
    holdout_preds = predict_ensemble_real(finetuned, ensemble_weights, holdout_df[feature_cols])

    return pd.DataFrame({
        IDENTIFIER_COLUMN: holdout_df[IDENTIFIER_COLUMN].values, "actual_pm25": holdout_df[TARGET_COLUMN].values,
        "ensemble_pred": holdout_preds["ensemble"],
    })


def run_one_pipeline_one_method(pipeline_name: str, source_models: dict, feature_cols: list, alpha: float,
                                 method_name: str, folds: list, ph_df_original: pd.DataFrame,
                                 row_level: bool) -> pd.DataFrame:
    print(f"  [{method_name} :: {pipeline_name}] {len(folds)} fold(s)...")
    fold_predictions = []
    for fold_assignment in folds:
        ph_with_split = attach_split(ph_df_original, fold_assignment, row_level=row_level)
        fold_preds = finetune_and_evaluate_fold(source_models, ph_with_split, feature_cols, alpha)
        fold_predictions.append(fold_preds)
    return pd.concat(fold_predictions, ignore_index=True)


def wilcoxon_vs_baseline_predictions(baseline_preds: pd.DataFrame, other_preds: pd.DataFrame) -> dict:
    """Same aggregate-before-combine discipline as diagnostics_core's
    wilcoxon_vs_baseline() -- applies even to random holdout, where the
    SPLIT is row-level by design but the TEST must not be."""
    b = baseline_preds.copy()
    o = other_preds.copy()
    b["abs_error"] = (b["ensemble_pred"] - b["actual_pm25"]).abs()
    o["abs_error"] = (o["ensemble_pred"] - o["actual_pm25"]).abs()
    b_per_cell = b.groupby(IDENTIFIER_COLUMN)["abs_error"].mean().rename("baseline_mae")
    o_per_cell = o.groupby(IDENTIFIER_COLUMN)["abs_error"].mean().rename("other_mae")
    per_cell = pd.concat([b_per_cell, o_per_cell], axis=1).dropna()

    if len(per_cell) < 2 or (per_cell["baseline_mae"] == per_cell["other_mae"]).all():
        return {"wilcoxon_pvalue": np.nan, "n_cells": len(per_cell)}
    stat, pval = wilcoxon(per_cell["baseline_mae"], per_cell["other_mae"])
    return {"wilcoxon_pvalue": float(pval), "n_cells": len(per_cell)}


def run_stage_b(best_pipeline: dict) -> pd.DataFrame:
    print("\n" + "#" * 60 + "\n# STAGE B: Validation Strategy Comparison\n" + "#" * 60)
    STAGE_B_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading source models saved by Stage A (no new source training)...")
    stage_a_dir = dc.DIAGNOSTICS_OUTPUT_ROOT / STAGE_A_NAME
    baseline_source_models = load_source_models(stage_a_dir / "baseline" / "models")
    optimized_source_models = load_source_models(stage_a_dir / best_pipeline["configuration_name"] / "models")

    source_df, ph_df_original, all_feature_cols = dc.load_data()
    # dc.load_data() already attaches the EXISTING spatial split as a
    # "split" column -- dropped here so each validation method's own
    # split can be attached cleanly below without a column-name collision
    # (confirmed directly: without this, merge() silently produces
    # split_x/split_y instead of "split", causing a KeyError downstream).
    ph_df_original = ph_df_original.drop(columns=["split"])
    baseline_feature_cols = dc.resolve_feature_cols(all_feature_cols, [])
    optimized_feature_cols = dc.resolve_feature_cols(all_feature_cols, best_pipeline["excluded_feature_groups"])

    print("Generating validation method splits...")
    spatial_split = pd.read_csv(EXISTING_SPATIAL_SPLIT_PATH)
    random_split = vs.random_holdout_split(ph_df_original, seed=RANDOM_SEED)
    folds_5 = vs.kfold_cell_splits(ph_df_original, k=5, seed=RANDOM_SEED)
    folds_10 = vs.kfold_cell_splits(ph_df_original, k=10, seed=RANDOM_SEED)

    validation_methods = {
        "spatial_holdout": {"folds": [spatial_split], "row_level": False},
        "random_holdout": {"folds": [random_split], "row_level": True},
        "5fold": {"folds": folds_5, "row_level": False},
        "10fold": {"folds": folds_10, "row_level": False},
    }

    pipelines = {
        "baseline": {"source_models": baseline_source_models, "feature_cols": baseline_feature_cols, "alpha": 1.0},
        "optimized": {"source_models": optimized_source_models, "feature_cols": optimized_feature_cols,
                       "alpha": best_pipeline["alpha"]},
    }

    all_predictions = {}
    results = []
    for method_name, method_info in validation_methods.items():
        for pipeline_name, pipeline_info in pipelines.items():
            preds = run_one_pipeline_one_method(
                pipeline_name, pipeline_info["source_models"], pipeline_info["feature_cols"], pipeline_info["alpha"],
                method_name, method_info["folds"], ph_df_original, method_info["row_level"])
            key = f"{method_name}_{pipeline_name}"
            all_predictions[key] = preds
            (STAGE_B_OUTPUT_DIR / key).mkdir(exist_ok=True)
            preds.to_csv(STAGE_B_OUTPUT_DIR / key / "predictions.csv", index=False)

            metrics = dc.compute_metrics(preds["actual_pm25"], preds["ensemble_pred"])
            n_negative = int((preds["ensemble_pred"] < 0).sum())
            results.append({
                "validation_method": method_name, "pipeline": pipeline_name,
                "n_rows": len(preds), "n_cells": preds[IDENTIFIER_COLUMN].nunique(),
                "rmse": metrics["RMSE"], "mae": metrics["MAE"], "r2": metrics["R2"],
                "pct_negative": n_negative / len(preds) * 100,
            })
            print(f"    {key}: RMSE={metrics['RMSE']:.4f}  MAE={metrics['MAE']:.4f}  R2={metrics['R2']:.4f}")

    comparison_df = pd.DataFrame(results)

    wilcoxon_rows = []
    for method_name in validation_methods:
        baseline_preds = all_predictions[f"{method_name}_baseline"]
        optimized_preds = all_predictions[f"{method_name}_optimized"]
        w = wilcoxon_vs_baseline_predictions(baseline_preds, optimized_preds)
        # Renamed to avoid a column-name collision with comparison_df's own
        # "n_cells" (confirmed directly -- without this, merge() silently
        # produced n_cells_x/n_cells_y instead of a plain column, causing
        # a KeyError in the report writer).
        wilcoxon_rows.append({"validation_method": method_name, "wilcoxon_pvalue": w["wilcoxon_pvalue"],
                                "wilcoxon_n_cells": w["n_cells"]})
    wilcoxon_df = pd.DataFrame(wilcoxon_rows)
    comparison_df = comparison_df.merge(wilcoxon_df, on="validation_method", how="left")

    for metric in ["rmse", "mae", "r2"]:
        baseline_lookup = comparison_df[comparison_df["pipeline"] == "baseline"].set_index("validation_method")[metric]
        comparison_df[f"delta_{metric}_vs_baseline"] = comparison_df.apply(
            lambda row: row[metric] - baseline_lookup[row["validation_method"]] if row["pipeline"] == "optimized" else np.nan,
            axis=1)

    comparison_df.to_csv(STAGE_B_OUTPUT_DIR / "consolidated_comparison.csv", index=False)
    _write_stage_b_report(comparison_df, best_pipeline)
    print(f"\nWrote consolidated_comparison.csv and consolidated_comparison_report.txt to {STAGE_B_OUTPUT_DIR}/")
    return comparison_df


def _write_stage_b_report(comparison_df: pd.DataFrame, best_pipeline: dict):
    lines = ["Validation Strategy Comparison -- Consolidated Report", "=" * 55, ""]
    lines.append(f"Optimized pipeline (from Stage A): {best_pipeline['configuration_name']} "
                  f"(alpha={best_pipeline['alpha']}, max_additional_rounds={best_pipeline['max_additional_rounds']}, "
                  f"excluded_countries={best_pipeline['excluded_countries']})\n")
    lines.append("NOTE on random_holdout: row-level by design, retained ONLY as a literature-comparison "
                  "artifact -- do not cite this number as this model's actual performance. It measures a "
                  "materially easier task (interpolating at partially-seen locations) than every other "
                  "method here.\n")
    lines.append("Rolling/expanding-window validation is deliberately excluded from this comparison -- it "
                  "measures temporal, not spatial, generalization, a different question from what this "
                  "table answers.\n")

    for method in ["spatial_holdout", "random_holdout", "5fold", "10fold"]:
        subset = comparison_df[comparison_df["validation_method"] == method]
        lines.append(f"\n--- {method} ---")
        for _, row in subset.iterrows():
            lines.append(f"  {row['pipeline']}: RMSE={row['rmse']:.4f}  MAE={row['mae']:.4f}  R2={row['r2']:.4f}  "
                          f"(n_rows={int(row['n_rows'])}, n_cells={int(row['n_cells'])})")
        opt_row = subset[subset["pipeline"] == "optimized"].iloc[0]
        lines.append(f"  Optimized vs. baseline: delta_RMSE={opt_row['delta_rmse_vs_baseline']:+.4f}  "
                      f"delta_R2={opt_row['delta_r2_vs_baseline']:+.4f}  "
                      f"Wilcoxon p={opt_row['wilcoxon_pvalue']:.4g} (n_cells={opt_row['wilcoxon_n_cells']})" if not pd.isna(opt_row['wilcoxon_pvalue'])
                      else f"  Optimized vs. baseline: delta_RMSE={opt_row['delta_rmse_vs_baseline']:+.4f}  Wilcoxon: not computable (identical or n<2)")

    (STAGE_B_OUTPUT_DIR / "consolidated_comparison_report.txt").write_text("\n".join(lines))


# ============================================================
# Main
# ============================================================

def main():
    best_pipeline = run_stage_a()
    run_stage_b(best_pipeline)
    print("\nOptimization + validation strategy pipeline complete.")


if __name__ == "__main__":
    main()
