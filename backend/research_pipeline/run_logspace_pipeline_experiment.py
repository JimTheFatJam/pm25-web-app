"""
Fully Consistent Log-Space Pipeline Experiment
====================================================

Research question: does learning PM2.5 in log-space -- consistently, from
source-domain training through fine-tuning -- combined with source-domain
correction (equal-country reweighting), improve transfer performance to
the Philippines?

Why "fully consistent" this time, and why the earlier attempt failed
------------------------------------------------------------------------------
A prior experiment attempted log1p fine-tuning by CONTINUING boosting on
top of the existing raw-target-trained baseline models. That failed
catastrophically (RMSE ~1000+ instead of ~35) for a specific, confirmed
reason: the baseline's cumulative output was in raw PM2.5 units, so
continued trees were asked to fit a residual between a raw-scale
prediction and a log-scale target -- a nonsensical, unstable quantity.

This experiment avoids that failure mode by construction: source models
are RETRAINED FROM SCRATCH on log1p(pm25) (configurations B and C below),
so there is no pre-existing raw-scale model to create a mismatch against.
Fine-tuning then continues boosting a model whose cumulative output is
ALREADY in log-space, against a log-space fine-tuning target -- internally
consistent at every stage, exactly as required.

Configurations
--------------
  A. Raw-target fine-tuning baseline -- NOT re-run; read directly from the
     existing, already-validated finetune_outputs/ (unweighted source,
     raw target) as the reference point, per "do not modify or re-run
     existing validated results."
  A2. Equal-country raw-target fine-tuning -- similarly read directly from
     the existing finetune_outputs_equalcountry_init/, as a second reference.
  B. Log-space + fine-tuning, UNWEIGHTED source -- source models retrained
     from scratch on log1p(pm25) with sample_weight=1, then fine-tuned on
     Philippine data ALSO in log1p space (train, validate, and early-stop
     all in log-space), with all predictions inverse-transformed via
     expm1() before any evaluation.
  C. Log-space + fine-tuning, EQUAL-COUNTRY source -- identical to B, but
     source models are retrained with the same equal-country sample
     weights used in the source reweighting experiment.

Consistency verified at every stage, not just the target
------------------------------------------------------------------------------
- Source training target: log1p(pm25).
- Source validation (used for the source-domain val/test metrics table):
  predictions inverse-transformed before computing RMSE/MAE/R2 -- these
  metrics are reported in real units for direct comparability with every
  prior experiment's tables, even though the model itself never sees
  real-unit targets during fitting.
- Fine-tuning target: log1p(pm25) on the Philippine finetune_train split.
- Early stopping: monitored on log1p(pm25) of the Philippine finetune_val
  split -- required, since the model's raw output during fitting IS in
  log-space; comparing it against untransformed validation targets would
  compare incompatible scales (the exact issue that broke the prior
  attempt, now avoided at every stage that matters).
- CatBoost's best-iteration correction: reused verbatim from the earlier,
  directly-tested fix -- get_best_iteration() is indexed RELATIVE to the
  continuation session, not the model's true start; the correct absolute
  shrink point is base_tree_count + best_iteration + 1, not
  best_iteration + 1 alone.
- Ensemble reconstruction: inverse-RMSE weights computed from REAL-UNIT
  (expm1-converted) finetune_val performance, consistent with how the
  earlier (failed) log1p attempt's Stage 2 was designed, so ensemble
  weighting reflects genuine prediction quality, not log-space RMSE.
- Numerical safety: log-space predictions are clipped to [-8, 8] before
  expm1() -- confirmed necessary during the earlier attempt, where an
  unclipped, unstable log-space prediction overflowed to infinity and
  crashed downstream metrics.

Explicitly NOT modified or re-implemented: train_baseline_models.py,
run_transfer_learning_experiment.py, and run_source_reweighting_experiment.py
are all left completely untouched -- this script only IMPORTS their
constants and utility functions (hyperparameters, compute_metrics,
compute_country_weights), never their side-effecting stage functions,
since those assume loading an already-saved raw-target model rather than
training fresh in log-space.

Output
------
finetune_outputs_logspace/
  config_B_logspace_unweighted/    -- models/, training_config.json,
                                       source_val_test_metrics.txt,
                                       predictions_source_val_test.csv,
                                       philippines_holdout_metrics.txt,
                                       predictions_philippines_holdout.csv
  config_C_logspace_equalcountry/  -- same structure
  logspace_comparison_report.txt   -- A vs A2 vs B vs C, answering all
                                       five required questions plus a
                                       short interpretation section

Usage
-----
Place this script alongside source_features.csv, source_metadata.csv,
philippines_features.csv, baseline_model_outputs/ (for the source cell
split), finetune_outputs/ (config A reference), and
finetune_outputs_equalcountry_init/ (config A2 reference), and run it.
"""

import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

from train_baseline_models import compute_metrics, XGBOOST_PARAMS, LIGHTGBM_PARAMS, CATBOOST_PARAMS
from run_source_reweighting_experiment import compute_country_weights

# ============================================================
# Configuration -- MACROS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
SOURCE_FEATURES_PATH = SCRIPT_DIR / "source_features.csv"
SOURCE_METADATA_PATH = SCRIPT_DIR / "source_metadata.csv"
PH_FEATURES_PATH = SCRIPT_DIR / "philippines_features.csv"
BASELINE_DIR = SCRIPT_DIR / "baseline_model_outputs"

CONFIG_A_DIR = SCRIPT_DIR / "finetune_outputs"                      # raw-target, unweighted (reference, not re-run)
CONFIG_A2_DIR = SCRIPT_DIR / "finetune_outputs_equalcountry_init"   # raw-target, equal-country (reference, not re-run)
PH_SPLIT_PATH = CONFIG_A_DIR / "philippines_finetune_split_assignment.csv"  # reused unchanged, for a fair comparison

OUTPUT_ROOT = SCRIPT_DIR / "finetune_outputs_logspace"

IDENTIFIER_COLUMN = "grid_id"
TARGET_COLUMN = "pm25"
RANDOM_SEED = 42

MAX_ADDITIONAL_ROUNDS = 200
EARLY_STOPPING_PATIENCE = 20
FINETUNE_LEARNING_RATE = 0.025

LOG_CLIP_BOUNDS = (-8, 8)  # see module docstring: prevents expm1() overflow on an unstable prediction


def to_log(y):
    return np.log1p(y)


def to_real(preds):
    return np.expm1(np.clip(preds, *LOG_CLIP_BOUNDS))


# ============================================================
# Train source models from scratch, in log-space
# ============================================================

def train_source_logspace(X_train, y_train_real, sample_weight):
    y_train_log = to_log(y_train_real)

    print("    Training XGBoost (log-space, from scratch)...")
    xgb_model = XGBRegressor(**XGBOOST_PARAMS)
    xgb_model.fit(X_train, y_train_log, sample_weight=sample_weight)

    print("    Training LightGBM (log-space, from scratch)...")
    lgbm_model = LGBMRegressor(**LIGHTGBM_PARAMS)
    lgbm_model.fit(X_train, y_train_log, sample_weight=sample_weight)

    print("    Training CatBoost (log-space, from scratch)...")
    catboost_model = CatBoostRegressor(**CATBOOST_PARAMS)
    catboost_model.fit(X_train, y_train_log, sample_weight=sample_weight)

    return {"xgboost": xgb_model, "lightgbm": lgbm_model, "catboost": catboost_model}


# ============================================================
# Fine-tune (continued boosting) in log-space, on Philippine data
# ============================================================

def finetune_logspace(source_models: dict, X_train, y_train_real, X_val, y_val_real) -> dict:
    y_train_log = to_log(y_train_real)
    y_val_log = to_log(y_val_real)

    print("    Fine-tuning XGBoost (log-space continuation)...")
    base_xgb = source_models["xgboost"]
    base_xgb_trees = base_xgb.get_booster().num_boosted_rounds()
    ft_xgb = XGBRegressor(n_estimators=MAX_ADDITIONAL_ROUNDS, max_depth=6, learning_rate=FINETUNE_LEARNING_RATE,
                           subsample=0.8, colsample_bytree=0.8, tree_method="hist", n_jobs=-1,
                           random_state=RANDOM_SEED, early_stopping_rounds=EARLY_STOPPING_PATIENCE, eval_metric="rmse")
    ft_xgb.fit(X_train, y_train_log, xgb_model=base_xgb.get_booster(), eval_set=[(X_val, y_val_log)], verbose=False)
    xgb_new_trees = ft_xgb.get_booster().num_boosted_rounds() - base_xgb_trees
    print(f"      base had {base_xgb_trees} trees, added {xgb_new_trees} more.")

    print("    Fine-tuning LightGBM (log-space continuation)...")
    base_lgbm = source_models["lightgbm"]
    base_lgbm_trees = base_lgbm.booster_.num_trees()
    ft_lgbm = LGBMRegressor(n_estimators=MAX_ADDITIONAL_ROUNDS, max_depth=-1, num_leaves=63,
                             learning_rate=FINETUNE_LEARNING_RATE, subsample=0.8, colsample_bytree=0.8,
                             n_jobs=-1, random_state=RANDOM_SEED, verbose=-1)
    ft_lgbm.fit(X_train, y_train_log, init_model=base_lgbm.booster_, eval_X=X_val, eval_y=y_val_log,
                callbacks=[lgb.early_stopping(EARLY_STOPPING_PATIENCE, verbose=False)])
    lgbm_new_trees = ft_lgbm.booster_.num_trees() - base_lgbm_trees
    print(f"      base had {base_lgbm_trees} trees, added {lgbm_new_trees} more.")

    print("    Fine-tuning CatBoost (log-space continuation)...")
    base_catboost = source_models["catboost"]
    base_catboost_trees = base_catboost.tree_count_
    ft_catboost = CatBoostRegressor(iterations=MAX_ADDITIONAL_ROUNDS, depth=6, learning_rate=FINETUNE_LEARNING_RATE,
                                     random_seed=RANDOM_SEED, verbose=False, early_stopping_rounds=EARLY_STOPPING_PATIENCE)
    ft_catboost.fit(X_train, y_train_log, init_model=base_catboost, eval_set=(X_val, y_val_log))
    # Verified correction (see module docstring): get_best_iteration() is
    # relative to the continuation session, not the model's true start.
    best_iter_relative = ft_catboost.get_best_iteration()
    correct_ntree_end = base_catboost_trees + best_iter_relative + 1
    ft_catboost.shrink(ntree_end=correct_ntree_end)
    catboost_new_trees = correct_ntree_end - base_catboost_trees
    print(f"      base had {base_catboost_trees} trees, added {catboost_new_trees} more "
          f"(shrunk to verified absolute tree count {correct_ntree_end}).")

    return {"xgboost": ft_xgb, "lightgbm": ft_lgbm, "catboost": ft_catboost}


# ============================================================
# Ensemble weights from real-unit finetune_val performance
# ============================================================

def compute_ensemble_weights(models: dict, X_val, y_val_real) -> dict:
    val_rmse = {}
    for name, model in models.items():
        preds_real = to_real(model.predict(X_val))
        val_rmse[name] = compute_metrics(y_val_real, preds_real)["RMSE"]
    inv_rmse = {name: 1.0 / r for name, r in val_rmse.items()}
    total_inv = sum(inv_rmse.values())
    return {name: v / total_inv for name, v in inv_rmse.items()}


def predict_ensemble_real(models: dict, weights: dict, X) -> dict:
    preds = {name: to_real(model.predict(X)) for name, model in models.items()}
    preds["ensemble"] = sum(weights[name] * preds[name] for name in models)
    return preds


# ============================================================
# Run one full configuration (B or C)
# ============================================================

def run_configuration(config_name: str, source_df, source_cell_split, ph_df, ph_split,
                       feature_cols, sample_weight_scheme: str):
    run_dir = OUTPUT_ROOT / config_name
    models_dir = run_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # --- Source training (from scratch, log-space) ---
    source_train_df = source_df[source_df["split"] == "train"]
    source_val_df = source_df[source_df["split"] == "val"]
    source_test_df = source_df[source_df["split"] == "test"]

    if sample_weight_scheme == "unweighted":
        sample_weight = np.ones(len(source_train_df))
        country_weights = {c: 1.0 for c in source_train_df["country"].unique()}
    else:
        country_row_counts = source_train_df.groupby("country").size()
        country_weights = compute_country_weights(country_row_counts, "equal_country")
        sample_weight = source_train_df["country"].map(country_weights).values

    print(f"  [{config_name}] Training source models from scratch in log-space ({sample_weight_scheme})...")
    source_models = train_source_logspace(source_train_df[feature_cols], source_train_df[TARGET_COLUMN], sample_weight)

    joblib.dump(source_models["xgboost"], models_dir / "source_xgboost_logspace.joblib")
    joblib.dump(source_models["lightgbm"], models_dir / "source_lightgbm_logspace.joblib")
    source_models["catboost"].save_model(str(models_dir / "source_catboost_logspace.cbm"))

    # Source-domain val/test metrics, reported in REAL units for direct
    # comparability with every prior experiment's tables.
    source_ensemble_weights_pretune = compute_ensemble_weights(source_models, source_val_df[feature_cols], source_val_df[TARGET_COLUMN])
    source_val_preds = predict_ensemble_real(source_models, source_ensemble_weights_pretune, source_val_df[feature_cols])
    source_test_preds = predict_ensemble_real(source_models, source_ensemble_weights_pretune, source_test_df[feature_cols])
    source_val_metrics = {n: compute_metrics(source_val_df[TARGET_COLUMN], p) for n, p in source_val_preds.items()}
    source_test_metrics = {n: compute_metrics(source_test_df[TARGET_COLUMN], p) for n, p in source_test_preds.items()}

    source_metrics_lines = [f"Source-Domain Val/Test Metrics (log-space trained, real-unit evaluated) -- {config_name}", "=" * 90]
    for split_name, metrics in [("VAL", source_val_metrics), ("TEST", source_test_metrics)]:
        source_metrics_lines.append(f"\n{split_name}:")
        for name, m in metrics.items():
            source_metrics_lines.append(f"  {name:<10} RMSE={m['RMSE']:.4f}  MAE={m['MAE']:.4f}  R2={m['R2']:.4f}")
    (run_dir / "source_val_test_metrics.txt").write_text("\n".join(source_metrics_lines))

    # --- Fine-tuning on Philippine data (log-space continuation) ---
    ph_train_df = ph_df[ph_df["split"] == "finetune_train"]
    ph_val_df = ph_df[ph_df["split"] == "finetune_val"]
    ph_holdout_df = ph_df[ph_df["split"] == "holdout"]

    print(f"  [{config_name}] Fine-tuning on Philippine data in log-space...")
    finetuned_models = finetune_logspace(source_models, ph_train_df[feature_cols], ph_train_df[TARGET_COLUMN],
                                          ph_val_df[feature_cols], ph_val_df[TARGET_COLUMN])

    joblib.dump(finetuned_models["xgboost"], models_dir / "finetuned_xgboost_logspace.joblib")
    joblib.dump(finetuned_models["lightgbm"], models_dir / "finetuned_lightgbm_logspace.joblib")
    finetuned_models["catboost"].save_model(str(models_dir / "finetuned_catboost_logspace.cbm"))

    finetuned_ensemble_weights = compute_ensemble_weights(finetuned_models, ph_val_df[feature_cols], ph_val_df[TARGET_COLUMN])
    pd.DataFrame([finetuned_ensemble_weights]).to_csv(run_dir / "ensemble_weights.csv", index=False)

    holdout_preds = predict_ensemble_real(finetuned_models, finetuned_ensemble_weights, ph_holdout_df[feature_cols])
    holdout_metrics = {n: compute_metrics(ph_holdout_df[TARGET_COLUMN], p) for n, p in holdout_preds.items()}

    ph_metrics_lines = [f"Philippines Holdout Metrics (log-space fine-tuned, real-unit evaluated) -- {config_name}", "=" * 90]
    for name, m in holdout_metrics.items():
        ph_metrics_lines.append(f"  {name:<10} RMSE={m['RMSE']:.4f}  MAE={m['MAE']:.4f}  R2={m['R2']:.4f}")
    (run_dir / "philippines_holdout_metrics.txt").write_text("\n".join(ph_metrics_lines))

    predictions_df = pd.DataFrame({
        IDENTIFIER_COLUMN: ph_holdout_df[IDENTIFIER_COLUMN].values, "actual_pm25": ph_holdout_df[TARGET_COLUMN].values,
        **{f"{n}_pred": p for n, p in holdout_preds.items()},
    })
    predictions_df.to_csv(run_dir / "predictions_philippines_holdout.csv", index=False)

    n_negative = int((holdout_preds["ensemble"] < 0).sum())
    training_config = {
        "configuration": config_name, "sample_weight_scheme": sample_weight_scheme, "random_seed": RANDOM_SEED,
        "country_weights": country_weights, "log_clip_bounds": LOG_CLIP_BOUNDS,
        "max_additional_rounds": MAX_ADDITIONAL_ROUNDS, "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "finetune_learning_rate": FINETUNE_LEARNING_RATE,
    }
    (run_dir / "training_config.json").write_text(json.dumps(training_config, indent=2, default=str))

    abs_error = np.abs(holdout_preds["ensemble"] - ph_holdout_df[TARGET_COLUMN].values)
    return {
        "configuration": config_name,
        "holdout_rmse": holdout_metrics["ensemble"]["RMSE"], "holdout_mae": holdout_metrics["ensemble"]["MAE"],
        "holdout_r2": holdout_metrics["ensemble"]["R2"],
        "n_negative": n_negative, "pct_negative": n_negative / len(predictions_df) * 100,
        "pred_min": float(holdout_preds["ensemble"].min()), "pred_max": float(holdout_preds["ensemble"].max()),
        "pred_mean": float(holdout_preds["ensemble"].mean()), "pred_std": float(holdout_preds["ensemble"].std()),
        "p95_abs_error": float(pd.Series(abs_error).quantile(0.95)),
        "max_abs_error": float(abs_error.max()),
    }


# ============================================================
# Load reference results from existing, validated experiments (not re-run)
# ============================================================

def load_reference_config(predictions_path: Path, label: str) -> dict:
    df = pd.read_csv(predictions_path)
    pred_col = "finetuned_ensemble_pred" if "finetuned_ensemble_pred" in df.columns else "ensemble_pred"
    metrics = compute_metrics(df["actual_pm25"], df[pred_col])
    n_negative = int((df[pred_col] < 0).sum())
    abs_err = (df[pred_col] - df["actual_pm25"]).abs()
    return {
        "configuration": label, "holdout_rmse": metrics["RMSE"], "holdout_mae": metrics["MAE"], "holdout_r2": metrics["R2"],
        "n_negative": n_negative, "pct_negative": n_negative / len(df) * 100,
        "pred_min": float(df[pred_col].min()), "pred_max": float(df[pred_col].max()),
        "pred_mean": float(df[pred_col].mean()), "pred_std": float(df[pred_col].std()),
        "p95_abs_error": float(abs_err.quantile(0.95)), "max_abs_error": float(abs_err.max()),
    }


# ============================================================
# Main
# ============================================================

def main():
    OUTPUT_ROOT.mkdir(exist_ok=True)

    print("Loading source features and the existing baseline source cell split...")
    source_df = pd.read_csv(SOURCE_FEATURES_PATH)
    source_meta = pd.read_csv(SOURCE_METADATA_PATH, usecols=[IDENTIFIER_COLUMN, "country"]).drop_duplicates()
    source_cell_split = pd.read_csv(BASELINE_DIR / "baseline_cell_split_assignment.csv")
    source_df = source_df.merge(source_cell_split[[IDENTIFIER_COLUMN, "split"]], on=IDENTIFIER_COLUMN, how="left")
    source_df = source_df.merge(source_meta, on=IDENTIFIER_COLUMN, how="left")

    feature_cols = [c for c in source_df.columns if c not in (IDENTIFIER_COLUMN, TARGET_COLUMN, "split", "country")]

    print("Loading Philippines features and the existing (unmodified) Philippine fine-tune split...")
    ph_df = pd.read_csv(PH_FEATURES_PATH)
    ph_split = pd.read_csv(PH_SPLIT_PATH)
    ph_df = ph_df.merge(ph_split[[IDENTIFIER_COLUMN, "split"]], on=IDENTIFIER_COLUMN, how="left")
    if ph_df["split"].isna().any():
        raise ValueError("Some Philippine rows have no split assignment -- grid_id mismatch with the reused split.")

    print("\n=== Configuration B: log-space + fine-tuning, UNWEIGHTED source ===")
    result_b = run_configuration("config_B_logspace_unweighted", source_df, source_cell_split, ph_df, ph_split,
                                  feature_cols, "unweighted")

    print("\n=== Configuration C: log-space + fine-tuning, EQUAL-COUNTRY source ===")
    result_c = run_configuration("config_C_logspace_equalcountry", source_df, source_cell_split, ph_df, ph_split,
                                  feature_cols, "equal_country")

    print("\nLoading reference results (A, A2) from existing, unmodified experiment outputs...")
    result_a = load_reference_config(CONFIG_A_DIR / "finetune_predictions_holdout.csv", "A_raw_unweighted")
    result_a2 = load_reference_config(CONFIG_A2_DIR / "finetune_predictions_holdout.csv", "A2_raw_equalcountry")

    all_results = [result_a, result_a2, result_b, result_c]
    comparison_df = pd.DataFrame(all_results)
    comparison_df.to_csv(OUTPUT_ROOT / "logspace_comparison.csv", index=False)

    # --- Report ---
    lines = ["Fully Consistent Log-Space Pipeline -- Comparison Report", "=" * 58, ""]
    lines.append("A  = raw-target fine-tuning, unweighted source (existing reference, not re-run)")
    lines.append("A2 = raw-target fine-tuning, equal-country source (existing reference, not re-run)")
    lines.append("B  = log-space source (from scratch) + log-space fine-tuning, unweighted")
    lines.append("C  = log-space source (from scratch) + log-space fine-tuning, equal-country")
    lines.append("All four evaluated on the IDENTICAL Philippine holdout cells.\n")

    for r in all_results:
        lines.append(f"{r['configuration']}:")
        lines.append(f"  RMSE={r['holdout_rmse']:.4f}  MAE={r['holdout_mae']:.4f}  R2={r['holdout_r2']:.4f}")
        lines.append(f"  Negative predictions: {r['n_negative']} ({r['pct_negative']:.2f}%)")
        lines.append(f"  Prediction distribution: min={r['pred_min']:.2f}  max={r['pred_max']:.2f}  "
                      f"mean={r['pred_mean']:.2f}  std={r['pred_std']:.2f}")
        lines.append(f"  95th-pct abs error={r['p95_abs_error']:.2f}  max abs error={r['max_abs_error']:.2f}")
        lines.append("")

    a_rmse, a2_rmse, b_rmse, c_rmse = result_a["holdout_rmse"], result_a2["holdout_rmse"], result_b["holdout_rmse"], result_c["holdout_rmse"]

    lines.append("=" * 58)
    lines.append("REQUIRED QUESTIONS")
    lines.append("=" * 58)

    lines.append(f"\n1. Does log-space ALONE improve transfer performance? (compare B vs. A)")
    lines.append(f"   A (raw, unweighted): RMSE={a_rmse:.4f}   B (log-space, unweighted): RMSE={b_rmse:.4f}")
    lines.append(f"   Delta: {b_rmse - a_rmse:+.4f} ({(b_rmse - a_rmse) / a_rmse * 100:+.2f}%)")

    lines.append(f"\n2. Does equal-country initialization still help IN log-space? (compare C vs. B)")
    lines.append(f"   B (log-space, unweighted): RMSE={b_rmse:.4f}   C (log-space, equal-country): RMSE={c_rmse:.4f}")
    lines.append(f"   Delta: {c_rmse - b_rmse:+.4f} ({(c_rmse - b_rmse) / b_rmse * 100:+.2f}%)")

    lines.append(f"\n3. Does C (equal-country + log-space + fine-tuning) outperform the current best model (A2)?")
    lines.append(f"   A2 (raw, equal-country, current best): RMSE={a2_rmse:.4f}   C: RMSE={c_rmse:.4f}")
    lines.append(f"   Delta: {c_rmse - a2_rmse:+.4f} ({(c_rmse - a2_rmse) / a2_rmse * 100:+.2f}%)")

    lines.append(f"\n4. Does log-space reduce negative predictions?")
    lines.append(f"   A: {result_a['pct_negative']:.2f}%   A2: {result_a2['pct_negative']:.2f}%   "
                  f"B: {result_b['pct_negative']:.2f}%   C: {result_c['pct_negative']:.2f}%")

    lines.append(f"\n5. Does log-space reduce catastrophic/extreme errors (95th-pct and max absolute error)?")
    lines.append(f"   95th-pct abs error -- A: {result_a['p95_abs_error']:.2f}  A2: {result_a2['p95_abs_error']:.2f}  "
                  f"B: {result_b['p95_abs_error']:.2f}  C: {result_c['p95_abs_error']:.2f}")
    lines.append(f"   Max abs error       -- A: {result_a['max_abs_error']:.2f}  A2: {result_a2['max_abs_error']:.2f}  "
                  f"B: {result_b['max_abs_error']:.2f}  C: {result_c['max_abs_error']:.2f}")

    lines.append("\n" + "=" * 58)
    lines.append("INTERPRETATION")
    lines.append("=" * 58)
    best_config = min(all_results, key=lambda r: r["holdout_rmse"])
    lines.append(f"\nBest-performing configuration on this holdout, by RMSE: {best_config['configuration']} "
                  f"(RMSE={best_config['holdout_rmse']:.4f}).")
    lines.append("This is a single run on a single Philippine cell-split seed -- per the same standard already "
                  "applied throughout this project, this ranking should NOT be treated as stable or general "
                  "without a multi-seed check, regardless of which configuration appears best here.")
    lines.append("\nWhether to continue with log-space or return to raw-target modeling should be judged on "
                  "whether log-space specifically improves catastrophic/extreme errors (question 5) and negative "
                  "predictions (question 4) -- the two concrete, previously-identified weaknesses this experiment "
                  "was designed to test -- not solely on whether the headline RMSE is marginally lower, since a "
                  "small RMSE change alone (as with the reweighting experiment) is not strong evidence on its own.")

    report_path = OUTPUT_ROOT / "logspace_comparison_report.txt"
    report_path.write_text("\n".join(lines))
    print(f"\nWrote {report_path}")
    print("\nLog-space pipeline experiment complete.")


if __name__ == "__main__":
    main()
