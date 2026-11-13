"""
Diagnostics Framework Core
================================

Single shared pipeline used by every diagnostic experiment (alpha
sensitivity, fine-tuning capacity, country ablation, feature-group
ablation). Every experiment is just a list of ExperimentConfig dicts fed
to run_experiment_suite() -- the training/evaluation logic itself lives
here exactly once, so every experiment runs under identical conditions
by construction, not by discipline.

ExperimentConfig fields (all except "name" have defaults matching the
current best pipeline, so a config that only sets "name" reproduces the
baseline exactly):
    name: str
    alpha: float = 1.0
    max_additional_rounds: int = 200
    excluded_countries: list[str] = []
    excluded_feature_groups: list[str] = []

Two different exclusion semantics, deliberately
------------------------------------------------------
excluded_countries removes cells from source TRAIN, VAL, AND TEST alike
-- a country excluded from training shouldn't remain in the evaluation
population either, since keeping it would measure "generalize to an
unseen country" (already what zero-shot PH transfer measures), not "how
well does a model trained on the remaining countries generalize within
those same countries." excluded_feature_groups and alpha/capacity
variations only ever affect training -- val/test cell composition never
changes for those, so results stay apples-to-apples across configs
within those experiments (this was Experiment 1's own design principle,
applied consistently here).

Anti-holdout-optimization discipline
------------------------------------------
Every experiment module fully materializes its config list BEFORE any
training starts (no adaptive generation based on partial results). The
Philippine and source splits are loaded unchanged from the existing,
already-validated split files -- never regenerated here.

Output, per experiment
-----------------------
diagnostics_outputs/<experiment_name>/
    <config_name>/models/, predictions_holdout.csv, config.json
    comparison.csv          -- every config's metrics + delta-vs-baseline
    report.txt               -- ranked, human-readable summary
    experiment_metadata.json -- reproducibility record
Plus one accumulated row per configuration appended to the shared
diagnostics_outputs/master_summary.csv across ALL experiments.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from train_baseline_models import compute_metrics
from run_logspace_pipeline_experiment import (
    train_source_logspace, finetune_logspace, compute_ensemble_weights, predict_ensemble_real,
    IDENTIFIER_COLUMN, TARGET_COLUMN, RANDOM_SEED,
)
from run_logspace_r2_improvement_experiment import compute_magnitude_weights
import feature_groups as fg

SCRIPT_DIR = Path(__file__).resolve().parent  # flat placement, matching every other script in this project --
                                                # this file lives directly alongside source_features.csv, etc.,
                                                # not in a nested subfolder
SOURCE_FEATURES_PATH = SCRIPT_DIR / "source_features.csv"
SOURCE_METADATA_PATH = SCRIPT_DIR / "source_metadata.csv"
PH_FEATURES_PATH = SCRIPT_DIR / "philippines_features.csv"
BASELINE_DIR = SCRIPT_DIR / "baseline_model_outputs"
PH_SPLIT_PATH = SCRIPT_DIR / "finetune_outputs" / "philippines_finetune_split_assignment.csv"

DIAGNOSTICS_OUTPUT_ROOT = SCRIPT_DIR / "diagnostics_outputs"
MASTER_SUMMARY_PATH = DIAGNOSTICS_OUTPUT_ROOT / "master_summary.csv"

DEFAULT_ALPHA = 1.0
DEFAULT_MAX_ADDITIONAL_ROUNDS = 200
N_BOOTSTRAP = 10_000  # kept for parity with earlier experiments' bootstrap use; Wilcoxon below is the primary test


# ============================================================
# ExperimentConfig
# ============================================================

def make_config(name: str, alpha: float = DEFAULT_ALPHA, max_additional_rounds: int = DEFAULT_MAX_ADDITIONAL_ROUNDS,
                 excluded_countries: list = None, excluded_feature_groups: list = None) -> dict:
    return {
        "name": name, "alpha": alpha, "max_additional_rounds": max_additional_rounds,
        "excluded_countries": excluded_countries or [], "excluded_feature_groups": excluded_feature_groups or [],
    }


# ============================================================
# Data loading (shared, unchanged across every experiment)
# ============================================================

def load_data():
    source_df = pd.read_csv(SOURCE_FEATURES_PATH)
    source_meta = pd.read_csv(SOURCE_METADATA_PATH, usecols=[IDENTIFIER_COLUMN, "country"]).drop_duplicates()
    source_cell_split = pd.read_csv(BASELINE_DIR / "baseline_cell_split_assignment.csv")
    source_df = source_df.merge(source_cell_split[[IDENTIFIER_COLUMN, "split"]], on=IDENTIFIER_COLUMN, how="left")
    source_df = source_df.merge(source_meta, on=IDENTIFIER_COLUMN, how="left")

    ph_df = pd.read_csv(PH_FEATURES_PATH)
    ph_split = pd.read_csv(PH_SPLIT_PATH)
    ph_df = ph_df.merge(ph_split[[IDENTIFIER_COLUMN, "split"]], on=IDENTIFIER_COLUMN, how="left")
    if ph_df["split"].isna().any():
        raise ValueError("Some Philippine rows have no split assignment.")

    feature_cols = [c for c in source_df.columns if c not in (IDENTIFIER_COLUMN, TARGET_COLUMN, "split", "country")]
    fg.validate_against_columns(feature_cols)  # fails loudly if feature_groups.py has drifted from the dataset

    return source_df, ph_df, feature_cols


# ============================================================
# Applying a configuration's knobs
# ============================================================

def resolve_feature_cols(all_feature_cols: list, excluded_feature_groups: list) -> list:
    excluded_cols = set()
    for group_name in excluded_feature_groups:
        if group_name in fg.FEATURE_GROUPS:
            excluded_cols |= set(fg.FEATURE_GROUPS[group_name])
        elif group_name in fg.CROSS_CUTTING_GROUPS:
            excluded_cols |= set(fg.CROSS_CUTTING_GROUPS[group_name])
        else:
            raise ValueError(f"Unknown feature group: {group_name}")
    return [c for c in all_feature_cols if c not in excluded_cols]


def apply_country_exclusion(source_df: pd.DataFrame, excluded_countries: list) -> pd.DataFrame:
    if not excluded_countries:
        return source_df
    # Removed from TRAIN, VAL, AND TEST alike -- see module docstring for why.
    return source_df[~source_df["country"].isin(excluded_countries)]


# ============================================================
# Run one configuration (the single shared training/evaluation path)
# ============================================================

def run_configuration(config: dict, source_df: pd.DataFrame, ph_df: pd.DataFrame, all_feature_cols: list,
                       output_dir: Path) -> tuple:
    run_dir = output_dir / config["name"]
    models_dir = run_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    feature_cols = resolve_feature_cols(all_feature_cols, config["excluded_feature_groups"])
    working_source_df = apply_country_exclusion(source_df, config["excluded_countries"])

    train_df = working_source_df[working_source_df["split"] == "train"]
    val_df = working_source_df[working_source_df["split"] == "val"]
    test_df = working_source_df[working_source_df["split"] == "test"]
    ph_train_df = ph_df[ph_df["split"] == "finetune_train"]
    ph_val_df = ph_df[ph_df["split"] == "finetune_val"]
    ph_holdout_df = ph_df[ph_df["split"] == "holdout"]

    print(f"  [{config['name']}] {len(feature_cols)} features, {train_df[IDENTIFIER_COLUMN].nunique()} train cells, "
          f"alpha={config['alpha']}, max_additional_rounds={config['max_additional_rounds']}")

    source_weight = compute_magnitude_weights(train_df[TARGET_COLUMN], config["alpha"])
    source_models = train_source_logspace(train_df[feature_cols], train_df[TARGET_COLUMN], source_weight)
    joblib.dump(source_models["xgboost"], models_dir / "source_xgboost.joblib")
    joblib.dump(source_models["lightgbm"], models_dir / "source_lightgbm.joblib")
    source_models["catboost"].save_model(str(models_dir / "source_catboost.cbm"))

    zero_shot_weights = compute_ensemble_weights(source_models, val_df[feature_cols], val_df[TARGET_COLUMN])
    val_preds = predict_ensemble_real(source_models, zero_shot_weights, val_df[feature_cols])
    test_preds = predict_ensemble_real(source_models, zero_shot_weights, test_df[feature_cols])
    val_metrics = compute_metrics(val_df[TARGET_COLUMN], val_preds["ensemble"])
    test_metrics = compute_metrics(test_df[TARGET_COLUMN], test_preds["ensemble"])

    zero_shot_ph_preds = predict_ensemble_real(source_models, zero_shot_weights, ph_holdout_df[feature_cols])
    zero_shot_ph_metrics = compute_metrics(ph_holdout_df[TARGET_COLUMN], zero_shot_ph_preds["ensemble"])

    ph_weight_train = compute_magnitude_weights(ph_train_df[TARGET_COLUMN], config["alpha"])
    replication_counts = np.maximum(1, np.round(ph_weight_train)).astype(int)
    ph_train_weighted = ph_train_df.loc[ph_train_df.index.repeat(replication_counts)].reset_index(drop=True)

    finetuned_models = finetune_logspace(
        source_models, ph_train_weighted[feature_cols], ph_train_weighted[TARGET_COLUMN],
        ph_val_df[feature_cols], ph_val_df[TARGET_COLUMN],
    ) if config["max_additional_rounds"] == 200 else _finetune_with_custom_cap(
        source_models, ph_train_weighted[feature_cols], ph_train_weighted[TARGET_COLUMN],
        ph_val_df[feature_cols], ph_val_df[TARGET_COLUMN], config["max_additional_rounds"])

    joblib.dump(finetuned_models["xgboost"], models_dir / "finetuned_xgboost.joblib")
    joblib.dump(finetuned_models["lightgbm"], models_dir / "finetuned_lightgbm.joblib")
    finetuned_models["catboost"].save_model(str(models_dir / "finetuned_catboost.cbm"))

    finetuned_weights = compute_ensemble_weights(finetuned_models, ph_val_df[feature_cols], ph_val_df[TARGET_COLUMN])
    holdout_preds = predict_ensemble_real(finetuned_models, finetuned_weights, ph_holdout_df[feature_cols])
    holdout_metrics = compute_metrics(ph_holdout_df[TARGET_COLUMN], holdout_preds["ensemble"])

    n_negative = int((holdout_preds["ensemble"] < 0).sum())
    abs_error = np.abs(holdout_preds["ensemble"] - ph_holdout_df[TARGET_COLUMN].values)

    predictions_df = pd.DataFrame({
        IDENTIFIER_COLUMN: ph_holdout_df[IDENTIFIER_COLUMN].values, "actual_pm25": ph_holdout_df[TARGET_COLUMN].values,
        "zero_shot_pred": zero_shot_ph_preds["ensemble"], "finetuned_pred": holdout_preds["ensemble"],
    })
    predictions_df.to_csv(run_dir / "predictions_holdout.csv", index=False)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))

    result = {
        "configuration": config["name"],
        "alpha": config["alpha"], "max_additional_rounds": config["max_additional_rounds"],
        "excluded_countries": ",".join(config["excluded_countries"]) if config["excluded_countries"] else "none",
        "excluded_feature_groups": ",".join(config["excluded_feature_groups"]) if config["excluded_feature_groups"] else "none",
        "n_features": len(feature_cols), "n_train_cells": train_df[IDENTIFIER_COLUMN].nunique(), "n_train_rows": len(train_df),
        "source_val_rmse": val_metrics["RMSE"], "source_val_r2": val_metrics["R2"],
        "source_test_rmse": test_metrics["RMSE"], "source_test_r2": test_metrics["R2"],
        "zero_shot_ph_rmse": zero_shot_ph_metrics["RMSE"], "zero_shot_ph_mae": zero_shot_ph_metrics["MAE"], "zero_shot_ph_r2": zero_shot_ph_metrics["R2"],
        "finetuned_rmse": holdout_metrics["RMSE"], "finetuned_mae": holdout_metrics["MAE"], "finetuned_r2": holdout_metrics["R2"],
        "n_negative": n_negative, "pct_negative": n_negative / len(predictions_df) * 100,
        "pred_std": float(holdout_preds["ensemble"].std()),
        "p95_abs_error": float(pd.Series(abs_error).quantile(0.95)), "max_abs_error": float(abs_error.max()),
    }
    return result, predictions_df


def _finetune_with_custom_cap(source_models, X_train, y_train, X_val, y_val, max_additional_rounds: int) -> dict:
    """Identical logic to finetune_logspace(), with MAX_ADDITIONAL_ROUNDS
    temporarily overridden -- implemented by reconfiguring the imported
    module's global for the duration of this call, the same reuse pattern
    already used for the Tier 1 and log1p sweeps, rather than duplicating
    finetune_logspace()'s body."""
    import run_logspace_pipeline_experiment as lp
    original_cap = lp.MAX_ADDITIONAL_ROUNDS
    lp.MAX_ADDITIONAL_ROUNDS = max_additional_rounds
    try:
        return finetune_logspace(source_models, X_train, y_train, X_val, y_val)
    finally:
        lp.MAX_ADDITIONAL_ROUNDS = original_cap


# ============================================================
# Cell-level Wilcoxon test vs. baseline (optional, per configuration)
# ============================================================

def wilcoxon_vs_baseline(baseline_predictions: pd.DataFrame, config_predictions: pd.DataFrame) -> dict:
    # Aggregate to per-cell MAE BEFORE combining the two dataframes -- NOT
    # after. grid_id is NOT a unique row key (each holdout cell has one row
    # per hour), so merging the full row-level dataframes directly on
    # grid_id triggers a many-to-many cross-product for every matching
    # cell (confirmed directly: this crashed with a 12.1 GiB allocation
    # attempt on real data -- ~18,000 hourly rows per cell in each
    # dataframe joined to ~18,000 x 18,000 rows for that one cell alone).
    # groupby().mean() is safe because its output is uniquely indexed by
    # grid_id by construction -- combining two SMALL, already-aggregated
    # per-cell series (one row per holdout cell, not per hour) afterward
    # cannot explode.
    baseline_predictions = baseline_predictions.copy()
    config_predictions = config_predictions.copy()
    baseline_predictions["abs_error"] = (baseline_predictions["finetuned_pred"] - baseline_predictions["actual_pm25"]).abs()
    config_predictions["abs_error"] = (config_predictions["finetuned_pred"] - config_predictions["actual_pm25"]).abs()

    baseline_per_cell = baseline_predictions.groupby(IDENTIFIER_COLUMN)["abs_error"].mean().rename("baseline_mae")
    config_per_cell = config_predictions.groupby(IDENTIFIER_COLUMN)["abs_error"].mean().rename("config_mae")
    per_cell = pd.concat([baseline_per_cell, config_per_cell], axis=1).dropna()

    if len(per_cell) < 2 or (per_cell["baseline_mae"] == per_cell["config_mae"]).all():
        return {"wilcoxon_statistic": np.nan, "wilcoxon_pvalue": np.nan, "n_cells": len(per_cell)}
    stat, pval = wilcoxon(per_cell["baseline_mae"], per_cell["config_mae"])
    return {"wilcoxon_statistic": float(stat), "wilcoxon_pvalue": float(pval), "n_cells": len(per_cell)}


# ============================================================
# Experiment suite orchestration (shared by every experiment module)
# ============================================================

def run_experiment_suite(experiment_name: str, configs: list, run_wilcoxon: bool = True) -> pd.DataFrame:
    if configs[0]["name"] != "baseline":
        raise ValueError("The first configuration in every experiment must be named 'baseline' -- "
                          "per the required design, baseline must appear first in every comparison.")

    output_dir = DIAGNOSTICS_OUTPUT_ROOT / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data for experiment '{experiment_name}'...")
    source_df, ph_df, all_feature_cols = load_data()

    results = []
    predictions_by_config = {}
    for config in configs:
        print(f"\n=== {experiment_name} :: {config['name']} ===")
        result, predictions_df = run_configuration(config, source_df, ph_df, all_feature_cols, output_dir)
        results.append(result)
        predictions_by_config[config["name"]] = predictions_df
        print(f"  Zero-shot PH R2={result['zero_shot_ph_r2']:.4f}  Fine-tuned PH RMSE={result['finetuned_rmse']:.4f} R2={result['finetuned_r2']:.4f}")

    comparison_df = pd.DataFrame(results)

    baseline_row = comparison_df.iloc[0]
    for metric in ["finetuned_rmse", "finetuned_mae", "finetuned_r2", "zero_shot_ph_rmse", "zero_shot_ph_r2"]:
        comparison_df[f"delta_{metric}"] = comparison_df[metric] - baseline_row[metric]

    comparison_df["rank_by_finetuned_rmse"] = comparison_df["finetuned_rmse"].rank(method="min").astype(int)

    if run_wilcoxon:
        baseline_preds = predictions_by_config["baseline"]
        wilcoxon_rows = []
        for config_name, preds in predictions_by_config.items():
            if config_name == "baseline":
                wilcoxon_rows.append({"configuration": config_name, "wilcoxon_statistic": np.nan,
                                        "wilcoxon_pvalue": np.nan, "n_cells": np.nan})
                continue
            w = wilcoxon_vs_baseline(baseline_preds, preds)
            wilcoxon_rows.append({"configuration": config_name, **w})
        wilcoxon_df = pd.DataFrame(wilcoxon_rows)
        comparison_df = comparison_df.merge(wilcoxon_df, on="configuration", how="left")

    comparison_df = comparison_df.sort_values("rank_by_finetuned_rmse")
    comparison_df.to_csv(output_dir / "comparison.csv", index=False)

    _write_report(experiment_name, output_dir, comparison_df, baseline_row, run_wilcoxon)
    _write_metadata(experiment_name, output_dir, configs, source_df, ph_df, all_feature_cols)
    _append_to_master_summary(experiment_name, comparison_df)

    print(f"\nWrote comparison.csv, report.txt, experiment_metadata.json to {output_dir}/")
    return comparison_df


def _write_report(experiment_name, output_dir, comparison_df, baseline_row, run_wilcoxon):
    lines = [f"Diagnostics Framework -- {experiment_name}", "=" * (24 + len(experiment_name)), ""]
    lines.append(f"Baseline (current best pipeline): RMSE={baseline_row['finetuned_rmse']:.4f}  "
                  f"MAE={baseline_row['finetuned_mae']:.4f}  R2={baseline_row['finetuned_r2']:.4f}\n")
    lines.append("Ranked by fine-tuned Philippine holdout RMSE (lower is better):\n")

    for _, r in comparison_df.iterrows():
        lines.append(f"#{r['rank_by_finetuned_rmse']}  {r['configuration']}")
        lines.append(f"  Fine-tuned:  RMSE={r['finetuned_rmse']:.4f} (delta={r['delta_finetuned_rmse']:+.4f})  "
                      f"MAE={r['finetuned_mae']:.4f}  R2={r['finetuned_r2']:.4f} (delta={r['delta_finetuned_r2']:+.4f})")
        lines.append(f"  Zero-shot:   RMSE={r['zero_shot_ph_rmse']:.4f} (delta={r['delta_zero_shot_ph_rmse']:+.4f})  "
                      f"R2={r['zero_shot_ph_r2']:.4f} (delta={r['delta_zero_shot_ph_r2']:+.4f})")
        lines.append(f"  Source:      VAL R2={r['source_val_r2']:.4f}  TEST R2={r['source_test_r2']:.4f}")
        lines.append(f"  Negative preds={r['pct_negative']:.2f}%  Pred std={r['pred_std']:.2f}  "
                      f"P95 err={r['p95_abs_error']:.2f}  Max err={r['max_abs_error']:.2f}")
        if run_wilcoxon and not pd.isna(r.get("wilcoxon_pvalue", np.nan)):
            sig = "SIGNIFICANT" if r["wilcoxon_pvalue"] < 0.05 else "not significant"
            lines.append(f"  Wilcoxon vs. baseline: p={r['wilcoxon_pvalue']:.4g} ({sig}, n={int(r['n_cells'])} cells)")
        lines.append("")

    (output_dir / "report.txt").write_text("\n".join(lines))


def _write_metadata(experiment_name, output_dir, configs, source_df, ph_df, all_feature_cols):
    metadata = {
        "experiment_name": experiment_name,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "random_seed": RANDOM_SEED,
        "n_source_rows": len(source_df), "n_source_cells": source_df[IDENTIFIER_COLUMN].nunique(),
        "n_philippines_rows": len(ph_df), "n_philippines_cells": ph_df[IDENTIFIER_COLUMN].nunique(),
        "n_total_features": len(all_feature_cols),
        "n_configurations": len(configs),
        "configuration_names": [c["name"] for c in configs],
    }
    (output_dir / "experiment_metadata.json").write_text(json.dumps(metadata, indent=2))


def _append_to_master_summary(experiment_name, comparison_df):
    summary_rows = comparison_df.copy()
    summary_rows.insert(0, "experiment", experiment_name)
    summary_rows.insert(1, "timestamp_utc", datetime.now(timezone.utc).isoformat())

    DIAGNOSTICS_OUTPUT_ROOT.mkdir(exist_ok=True)
    if MASTER_SUMMARY_PATH.exists():
        existing = pd.read_csv(MASTER_SUMMARY_PATH)
        combined = pd.concat([existing, summary_rows], ignore_index=True)
    else:
        combined = summary_rows
    combined.to_csv(MASTER_SUMMARY_PATH, index=False)
