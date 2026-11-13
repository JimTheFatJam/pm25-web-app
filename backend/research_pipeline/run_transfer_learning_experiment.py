"""
Complete Transfer Learning Experiment Pipeline
====================================================

Runs the full source-to-Philippines transfer learning experiment as one
sequential pipeline, from Philippine cell characterization through final
diagnostics, while preserving clear internal stage boundaries -- each
stage has its own labeled section in this file, its own console logging,
and produces its own clearly named output files, exactly as if it had
been run as an independent script.

STAGE 0: Philippine Cell Characterization and Split
STAGE 1: Continued Training (Fine-Tuning)
STAGE 2: Ensemble Reconstruction
STAGE 3: Evaluation (baseline vs. fine-tuned, on the identical holdout)
STAGE 4: Statistical Testing (cell-level, ensemble only)
STAGE 5: Diagnostics (calibration, category bias, SHAP, distance-to-coast
          error -- baseline vs. fine-tuned, repeated ONLY where the
          underlying quantity can actually change; see Stage 5's own
          docstring for what is deliberately NOT repeated, and why)

The ensemble is the primary experimental subject throughout: XGBoost,
LightGBM, and CatBoost's individual results are reported in every stage
for transparency and interpretation, but hypothesis testing, statistical
significance, and the experiment's success/failure determination all rest
on the ENSEMBLE's holdout performance, not any individual base learner.

Usage
-----
Place this script alongside source_features.csv, source_metadata.csv,
philippines_features.csv, and baseline_model_outputs/ (from the baseline
pipeline), and run it. All outputs are written under finetune_outputs/.

Dependencies: pandas, numpy, scipy, scikit-learn, matplotlib, shap,
              xgboost, lightgbm, catboost, joblib
    pip install pandas numpy scipy scikit-learn matplotlib shap xgboost lightgbm catboost joblib
"""

from pathlib import Path

import joblib
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from scipy.stats import wilcoxon
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

# ============================================================
# Shared configuration -- MACROS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent

PH_FEATURES_PATH = SCRIPT_DIR / "philippines_features.csv"
BASELINE_MODELS_DIR = SCRIPT_DIR / "baseline_model_outputs" / "models"

OUTPUT_DIR = SCRIPT_DIR / "finetune_outputs"
MODELS_DIR = OUTPUT_DIR / "models"
FIGURES_DIR = OUTPUT_DIR / "figures"
SPLIT_ASSIGNMENT_PATH = OUTPUT_DIR / "philippines_finetune_split_assignment.csv"

IDENTIFIER_COLUMN = "grid_id"
TARGET_COLUMN = "pm25"
RANDOM_SEED = 42

# Stage 0
FINETUNE_TRAIN_FRAC = 0.45
FINETUNE_VAL_FRAC = 0.15
CHARACTERIZATION_COLUMNS = ["distance_to_coast_km", "population_local", "elevation_local", TARGET_COLUMN]

# Stage 1
MAX_ADDITIONAL_ROUNDS = 200
EARLY_STOPPING_PATIENCE = 20
FINETUNE_LEARNING_RATE = 0.025

# Stage 4
N_BOOTSTRAP_SAMPLES = 10_000

# Stage 5
SHAP_FEATURES_OF_INTEREST = ["population_local", "population_neighborhood_mean", "distance_to_coast_km"]
SHAP_SAMPLE_SIZE = 5_000
DISTANCE_BINS = [0, 1, 2, 5, 10, 20, np.inf]
DISTANCE_LABELS = ["0-1km", "1-2km", "2-5km", "5-10km", "10-20km", ">20km"]


def log(lines, msg):
    lines.append(msg)
    print(msg)


def section(lines, title):
    lines.append("")
    lines.append("=" * 78)
    lines.append(title)
    lines.append("=" * 78)


def subsection(lines, title):
    lines.append("")
    lines.append(title)
    lines.append("-" * len(title))


def compute_metrics(y_true, y_pred) -> dict:
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


# ============================================================================
# STAGE 0: Philippine Cell Characterization and Split
# ============================================================================
#
# Characterizes all Philippine grid cells (land cover, distance-to-coast,
# population, elevation, pm25) and splits them into finetune_train (~45%,
# where continued boosting happens), finetune_val (~15%, early stopping
# and ensemble reweighting only), and holdout (~40%, touched only at
# final evaluation -- kept deliberately large since this is the number
# the experiment's central claim rests on).
#
# Stratified primarily by land cover class (the clearest category-level
# driver of the baseline's failure); other axes are checked and reported,
# not force-balanced, since jointly stratifying on four correlated axes
# with ~83 cells would create mostly-empty strata. A land cover class with
# only one Philippine cell cannot be meaningfully split or evaluated --
# by explicit rule, it is assigned entirely to finetune_train.
# ============================================================================

def build_cell_level_table(df: pd.DataFrame, lc_cols: list) -> pd.DataFrame:
    agg_dict = {col: "mean" for col in CHARACTERIZATION_COLUMNS if col in df.columns}
    cell_table = df.groupby(IDENTIFIER_COLUMN).agg(agg_dict)
    cell_table["n_rows"] = df.groupby(IDENTIFIER_COLUMN).size()
    lc_per_cell = df.groupby(IDENTIFIER_COLUMN)[lc_cols].first()
    cell_table["landcover_class"] = lc_per_cell.idxmax(axis=1).str.replace("landcover_local_", "")
    return cell_table.reset_index()


def split_cells(cell_table: pd.DataFrame, train_frac: float, val_frac: float, seed: int) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    assignments = []
    for lc_class, group in cell_table.groupby("landcover_class"):
        cells = group[IDENTIFIER_COLUMN].tolist()
        n = len(cells)
        if n == 1:
            assignments.append({IDENTIFIER_COLUMN: cells[0], "landcover_class": lc_class, "split": "finetune_train"})
            continue
        rng.shuffle(cells)
        n_train = min(max(1, round(n * train_frac)), n)
        remaining = n - n_train
        n_val = min(max(1, round(n * val_frac)) if remaining >= 2 else remaining, remaining)
        for c in cells[:n_train]:
            assignments.append({IDENTIFIER_COLUMN: c, "landcover_class": lc_class, "split": "finetune_train"})
        for c in cells[n_train:n_train + n_val]:
            assignments.append({IDENTIFIER_COLUMN: c, "landcover_class": lc_class, "split": "finetune_val"})
        for c in cells[n_train + n_val:]:
            assignments.append({IDENTIFIER_COLUMN: c, "landcover_class": lc_class, "split": "holdout"})
    return pd.DataFrame(assignments)


def run_stage0(pipeline_log):
    section(pipeline_log, "STAGE 0: Philippine Cell Characterization and Split")
    lines = []

    print("  Loading Philippines features...")
    ph_df = pd.read_csv(PH_FEATURES_PATH)
    lc_cols = sorted(c for c in ph_df.columns if c.startswith("landcover_local_"))
    print(f"    {len(ph_df)} rows, {ph_df[IDENTIFIER_COLUMN].nunique()} distinct grid cells.")

    print("  Building cell-level characterization table...")
    cell_table = build_cell_level_table(ph_df, lc_cols)

    print("  Splitting cells (stratified by land cover, singleton classes -> finetune_train)...")
    split_assignment = split_cells(cell_table, FINETUNE_TRAIN_FRAC, FINETUNE_VAL_FRAC, RANDOM_SEED)
    split_assignment.to_csv(SPLIT_ASSIGNMENT_PATH, index=False)
    print(f"    Wrote {SPLIT_ASSIGNMENT_PATH.name}")

    merged = cell_table.merge(split_assignment[[IDENTIFIER_COLUMN, "split"]], on=IDENTIFIER_COLUMN)
    lines.append(f"Total Philippine grid cells: {len(cell_table)}")
    for split_name in ["finetune_train", "finetune_val", "holdout"]:
        n = (merged["split"] == split_name).sum()
        lines.append(f"  {split_name}: {n} cells ({n / len(cell_table) * 100:.1f}%)")

    subsection(lines, "Land cover class representation by split")
    lc_table = pd.crosstab(merged["landcover_class"], merged["split"])
    for lc_class, row in lc_table.iterrows():
        singleton_note = "  <-- SINGLETON CLASS, assigned entirely to finetune_train by rule" if row.sum() == 1 else ""
        lines.append(f"  {lc_class:<35} " + "  ".join(f"{s}={row.get(s, 0)}" for s in ["finetune_train", "finetune_val", "holdout"]) + singleton_note)

    subsection(lines, "Distance-to-coast, population, elevation, pm25 by split (checked, not force-stratified)")
    for col in CHARACTERIZATION_COLUMNS:
        if col not in merged.columns:
            continue
        lines.append(f"\n  {col}:")
        for split_name in ["finetune_train", "finetune_val", "holdout"]:
            vals = merged.loc[merged["split"] == split_name, col]
            lines.append(f"    {split_name:<16} mean={vals.mean():.3f}  median={vals.median():.3f}  "
                          f"std={vals.std():.3f}  min={vals.min():.3f}  max={vals.max():.3f}")

    report_path = OUTPUT_DIR / "philippines_characterization_report.txt"
    report_path.write_text("Stage 0: Philippine Cell Characterization and Split\n" + "=" * 52 + "\n" + "\n".join(lines))
    print(f"    Wrote {report_path.name}")
    pipeline_log.extend(lines)


# ============================================================================
# STAGE 1: Continued Training (Fine-Tuning)
# ============================================================================
#
# Continues training each baseline model on finetune_train, using each
# library's native continued-boosting mechanism, with early stopping
# monitored on finetune_val. The original source-learned trees are never
# modified -- only added to; the real risk this design guards against
# (via a halved learning rate and early stopping) is the new trees'
# contribution growing large enough to DOMINATE the final prediction, not
# "forgetting" in the neural-network sense, since nothing is overwritten.
#
# CatBoost requires an explicit, verified correction: its .predict() does
# NOT auto-restrict to best_iteration (unlike XGBoost and LightGBM, both
# confirmed to do so automatically), and get_best_iteration() is indexed
# RELATIVE TO THE CONTINUATION SESSION, not from the model's true start.
# The correct absolute cut point is base_tree_count + best_iteration + 1;
# using best_iteration alone would keep only the first few trees of the
# ORIGINAL pretrained model and discard nearly all of it. This was
# confirmed directly via a deliberately constructed test using a large,
# distinctive base tree count against a small forced early-stopping point.
# ============================================================================

def run_stage1(pipeline_log):
    section(pipeline_log, "STAGE 1: Continued Training (Fine-Tuning)")
    lines = []

    print("  Loading Philippines features and Stage 0 split assignment...")
    ph_df = pd.read_csv(PH_FEATURES_PATH)
    split_assignment = pd.read_csv(SPLIT_ASSIGNMENT_PATH)
    ph_df = ph_df.merge(split_assignment[[IDENTIFIER_COLUMN, "split"]], on=IDENTIFIER_COLUMN, how="left")
    if ph_df["split"].isna().any():
        raise ValueError("Some Philippine rows have no split assignment -- grid_id mismatch with Stage 0's output.")

    feature_cols = [c for c in ph_df.columns if c not in (IDENTIFIER_COLUMN, TARGET_COLUMN, "split")]
    train_df = ph_df[ph_df["split"] == "finetune_train"]
    val_df = ph_df[ph_df["split"] == "finetune_val"]
    X_train, y_train = train_df[feature_cols], train_df[TARGET_COLUMN]
    X_val, y_val = val_df[feature_cols], val_df[TARGET_COLUMN]
    print(f"    finetune_train: {len(X_train)} rows, finetune_val: {len(X_val)} rows.")

    print("  Loading baseline models...")
    base_xgb = joblib.load(BASELINE_MODELS_DIR / "xgboost_baseline.joblib")
    base_lgbm = joblib.load(BASELINE_MODELS_DIR / "lightgbm_baseline.joblib")
    base_catboost = CatBoostRegressor()
    base_catboost.load_model(str(BASELINE_MODELS_DIR / "catboost_baseline.cbm"))

    print("  Fine-tuning XGBoost...")
    base_xgb_trees = base_xgb.get_booster().num_boosted_rounds()
    ft_xgb = XGBRegressor(n_estimators=MAX_ADDITIONAL_ROUNDS, max_depth=6, learning_rate=FINETUNE_LEARNING_RATE,
                           subsample=0.8, colsample_bytree=0.8, tree_method="hist", n_jobs=-1,
                           random_state=RANDOM_SEED, early_stopping_rounds=EARLY_STOPPING_PATIENCE, eval_metric="rmse")
    ft_xgb.fit(X_train, y_train, xgb_model=base_xgb.get_booster(), eval_set=[(X_val, y_val)], verbose=False)
    xgb_new_trees = ft_xgb.get_booster().num_boosted_rounds() - base_xgb_trees
    xgb_val_rmse = np.sqrt(mean_squared_error(y_val, ft_xgb.predict(X_val)))
    lines.append(f"XGBoost: base had {base_xgb_trees} trees, added {xgb_new_trees} more "
                  f"(early stopping patience={EARLY_STOPPING_PATIENCE}, cap={MAX_ADDITIONAL_ROUNDS})")
    lines.append(f"  finetune_val RMSE={xgb_val_rmse:.4f}")
    joblib.dump(ft_xgb, MODELS_DIR / "xgboost_finetuned.joblib")

    print("  Fine-tuning LightGBM...")
    base_lgbm_trees = base_lgbm.booster_.num_trees()
    ft_lgbm = LGBMRegressor(n_estimators=MAX_ADDITIONAL_ROUNDS, max_depth=-1, num_leaves=63,
                             learning_rate=FINETUNE_LEARNING_RATE, subsample=0.8, colsample_bytree=0.8,
                             n_jobs=-1, random_state=RANDOM_SEED, verbose=-1)
    ft_lgbm.fit(X_train, y_train, init_model=base_lgbm.booster_, eval_X=X_val, eval_y=y_val,
                callbacks=[lgb.early_stopping(EARLY_STOPPING_PATIENCE, verbose=False)])
    lgbm_new_trees = ft_lgbm.booster_.num_trees() - base_lgbm_trees
    lgbm_val_rmse = np.sqrt(mean_squared_error(y_val, ft_lgbm.predict(X_val)))
    lines.append(f"\nLightGBM: base had {base_lgbm_trees} trees, added {lgbm_new_trees} more")
    lines.append(f"  finetune_val RMSE={lgbm_val_rmse:.4f}")
    joblib.dump(ft_lgbm, MODELS_DIR / "lightgbm_finetuned.joblib")

    print("  Fine-tuning CatBoost...")
    base_catboost_trees = base_catboost.tree_count_
    ft_catboost = CatBoostRegressor(iterations=MAX_ADDITIONAL_ROUNDS, depth=6, learning_rate=FINETUNE_LEARNING_RATE,
                                     random_seed=RANDOM_SEED, verbose=False, early_stopping_rounds=EARLY_STOPPING_PATIENCE)
    ft_catboost.fit(X_train, y_train, init_model=base_catboost, eval_set=(X_val, y_val))
    best_iter_relative = ft_catboost.get_best_iteration()
    correct_ntree_end = base_catboost_trees + best_iter_relative + 1
    ft_catboost.shrink(ntree_end=correct_ntree_end)
    catboost_new_trees = correct_ntree_end - base_catboost_trees
    catboost_val_rmse = np.sqrt(mean_squared_error(y_val, ft_catboost.predict(X_val)))
    lines.append(f"\nCatBoost: base had {base_catboost_trees} trees, added {catboost_new_trees} more "
                  f"(shrunk to verified absolute tree count {correct_ntree_end})")
    lines.append(f"  finetune_val RMSE={catboost_val_rmse:.4f}")
    ft_catboost.save_model(str(MODELS_DIR / "catboost_finetuned.cbm"))

    log_path = OUTPUT_DIR / "finetuning_training_log.txt"
    log_path.write_text("Stage 1: Continued Training (Fine-Tuning) Log\n" + "=" * 47 + "\n\n" + "\n".join(lines))
    print(f"    Wrote {log_path.name}")
    pipeline_log.extend(lines)

    return {"xgboost": ft_xgb, "lightgbm": ft_lgbm, "catboost": ft_catboost}


# ============================================================================
# STAGE 2: Ensemble Reconstruction
# ============================================================================
#
# Recomputes ensemble weights from the FINE-TUNED models' performance on
# finetune_val -- not the original source validation set, since
# re-deriving weights from source performance would defeat the entire
# purpose of adapting to the target domain. Same inverse-RMSE weighting
# formula as the baseline, applied to Philippine-adapted validation
# performance instead.
# ============================================================================

def run_stage2(pipeline_log, finetuned_models):
    section(pipeline_log, "STAGE 2: Ensemble Reconstruction")
    lines = []

    print("  Loading finetune_val data...")
    ph_df = pd.read_csv(PH_FEATURES_PATH)
    split_assignment = pd.read_csv(SPLIT_ASSIGNMENT_PATH)
    ph_df = ph_df.merge(split_assignment[[IDENTIFIER_COLUMN, "split"]], on=IDENTIFIER_COLUMN, how="left")
    feature_cols = [c for c in ph_df.columns if c not in (IDENTIFIER_COLUMN, TARGET_COLUMN, "split")]
    val_df = ph_df[ph_df["split"] == "finetune_val"]
    X_val, y_val = val_df[feature_cols], val_df[TARGET_COLUMN]

    print("  Computing inverse-RMSE ensemble weights from finetune_val performance...")
    val_rmse = {}
    for name, model in finetuned_models.items():
        preds = model.predict(X_val)
        val_rmse[name] = float(np.sqrt(mean_squared_error(y_val, preds)))

    inv_rmse = {name: 1.0 / r for name, r in val_rmse.items()}
    total_inv = sum(inv_rmse.values())
    ensemble_weights = {name: v / total_inv for name, v in inv_rmse.items()}

    for name in finetuned_models:
        lines.append(f"  {name:<10} finetune_val RMSE={val_rmse[name]:.4f}  weight={ensemble_weights[name]:.4f}")

    weights_path = OUTPUT_DIR / "finetuned_ensemble_weights.csv"
    pd.DataFrame([ensemble_weights]).to_csv(weights_path, index=False)
    print(f"    Wrote {weights_path.name}: {ensemble_weights}")
    pipeline_log.extend(lines)

    return ensemble_weights


# ============================================================================
# STAGE 3: Evaluation (Baseline vs. Fine-Tuned, on the Identical Holdout)
# ============================================================================
#
# Critical fairness requirement: the baseline ensemble is RE-SCORED on the
# exact same holdout cells used to evaluate the fine-tuned ensemble --
# never reused from the original baseline_metrics.txt, which scored
# against the entire Philippines file. Comparing a fine-tuned model's
# holdout-subset performance against a baseline's full-dataset performance
# would not isolate the effect of fine-tuning from the effect of simply
# evaluating on a different (smaller, differently-composed) set of rows.
# ============================================================================

def run_stage3(pipeline_log, finetuned_models, finetuned_weights):
    section(pipeline_log, "STAGE 3: Evaluation (Baseline vs. Fine-Tuned, Identical Holdout)")
    lines = []

    print("  Loading holdout data...")
    ph_df = pd.read_csv(PH_FEATURES_PATH)
    split_assignment = pd.read_csv(SPLIT_ASSIGNMENT_PATH)
    ph_df = ph_df.merge(split_assignment[[IDENTIFIER_COLUMN, "split"]], on=IDENTIFIER_COLUMN, how="left")
    feature_cols = [c for c in ph_df.columns if c not in (IDENTIFIER_COLUMN, TARGET_COLUMN, "split")]
    holdout_df = ph_df[ph_df["split"] == "holdout"]
    X_holdout, y_holdout = holdout_df[feature_cols], holdout_df[TARGET_COLUMN]
    print(f"    Holdout: {len(X_holdout)} rows across {holdout_df[IDENTIFIER_COLUMN].nunique()} cells.")

    print("  Loading baseline models and weights...")
    baseline_models = {
        "xgboost": joblib.load(BASELINE_MODELS_DIR / "xgboost_baseline.joblib"),
        "lightgbm": joblib.load(BASELINE_MODELS_DIR / "lightgbm_baseline.joblib"),
    }
    baseline_catboost = CatBoostRegressor()
    baseline_catboost.load_model(str(BASELINE_MODELS_DIR / "catboost_baseline.cbm"))
    baseline_models["catboost"] = baseline_catboost
    baseline_weights = pd.read_csv(BASELINE_MODELS_DIR.parent / "baseline_ensemble_weights.csv").iloc[0].to_dict()

    def score_all(models, weights, X, y):
        preds = {name: model.predict(X) for name, model in models.items()}
        preds["ensemble"] = sum(weights[name] * preds[name] for name in models)
        metrics = {name: compute_metrics(y, preds[name]) for name in list(models) + ["ensemble"]}
        return preds, metrics

    print("  Scoring baseline on holdout...")
    baseline_preds, baseline_metrics = score_all(baseline_models, baseline_weights, X_holdout, y_holdout)
    print("  Scoring fine-tuned ensemble on the SAME holdout...")
    finetuned_preds, finetuned_metrics = score_all(finetuned_models, finetuned_weights, X_holdout, y_holdout)

    subsection(lines, "Baseline (re-scored on holdout) vs. Fine-Tuned -- RMSE / MAE / R2")
    for name in ["xgboost", "lightgbm", "catboost", "ensemble"]:
        b, f = baseline_metrics[name], finetuned_metrics[name]
        marker = "  <-- PRIMARY SUBJECT" if name == "ensemble" else ""
        lines.append(f"\n  {name}{marker}")
        lines.append(f"    BASELINE:    RMSE={b['RMSE']:.4f}  MAE={b['MAE']:.4f}  R2={b['R2']:.4f}")
        lines.append(f"    FINE-TUNED:  RMSE={f['RMSE']:.4f}  MAE={f['MAE']:.4f}  R2={f['R2']:.4f}")
        lines.append(f"    Delta:       RMSE={f['RMSE']-b['RMSE']:+.4f}  MAE={f['MAE']-b['MAE']:+.4f}  R2={f['R2']-b['R2']:+.4f}")

    metrics_path = OUTPUT_DIR / "finetune_metrics_comparison.txt"
    metrics_path.write_text("Stage 3: Evaluation -- Baseline vs. Fine-Tuned (Identical Holdout)\n" + "=" * 67 + "\n" + "\n".join(lines))
    print(f"    Wrote {metrics_path.name}")

    predictions_df = pd.DataFrame({
        IDENTIFIER_COLUMN: holdout_df[IDENTIFIER_COLUMN].values,
        "actual_pm25": y_holdout.values,
        "baseline_xgboost_pred": baseline_preds["xgboost"], "baseline_lightgbm_pred": baseline_preds["lightgbm"],
        "baseline_catboost_pred": baseline_preds["catboost"], "baseline_ensemble_pred": baseline_preds["ensemble"],
        "finetuned_xgboost_pred": finetuned_preds["xgboost"], "finetuned_lightgbm_pred": finetuned_preds["lightgbm"],
        "finetuned_catboost_pred": finetuned_preds["catboost"], "finetuned_ensemble_pred": finetuned_preds["ensemble"],
    })
    predictions_path = OUTPUT_DIR / "finetune_predictions_holdout.csv"
    predictions_df.to_csv(predictions_path, index=False)
    print(f"    Wrote {predictions_path.name}")
    pipeline_log.extend(lines)

    return predictions_df


# ============================================================================
# STAGE 4: Statistical Testing (Cell-Level, Ensemble Only)
# ============================================================================
#
# Rows within a Philippine cell are heavily autocorrelated (established
# in the EDA -- effective sample size is closer to the cell count than
# the row count). A row-level paired test would treat thousands of hourly
# rows from one cell as thousands of independent pieces of evidence,
# dramatically inflating apparent significance. Aggregated to per-cell
# mean absolute error FIRST, then tested at the cell level -- N = number
# of holdout cells, not number of holdout rows. Any bootstrap similarly
# resamples entire CELLS with replacement (a block bootstrap), not
# individual rows, for the same reason.
#
# Run ONCE, on the ensemble only -- per the explicit design decision that
# the ensemble is the primary experimental subject; testing each base
# learner separately in addition would be uncorrected multiple testing
# diluting the one comparison the thesis's claim actually rests on.
# ============================================================================

def run_stage4(pipeline_log, predictions_df):
    section(pipeline_log, "STAGE 4: Statistical Testing (Cell-Level, Ensemble Only)")
    lines = []

    print("  Aggregating to per-cell mean absolute error (ensemble only)...")
    predictions_df = predictions_df.copy()
    predictions_df["baseline_abs_error"] = (predictions_df["baseline_ensemble_pred"] - predictions_df["actual_pm25"]).abs()
    predictions_df["finetuned_abs_error"] = (predictions_df["finetuned_ensemble_pred"] - predictions_df["actual_pm25"]).abs()

    per_cell = predictions_df.groupby(IDENTIFIER_COLUMN).agg(
        baseline_mae=("baseline_abs_error", "mean"), finetuned_mae=("finetuned_abs_error", "mean"))
    n_cells = len(per_cell)
    lines.append(f"Holdout cells (N for all tests below): {n_cells}")
    lines.append(f"Per-cell baseline MAE:   mean={per_cell['baseline_mae'].mean():.4f}  median={per_cell['baseline_mae'].median():.4f}")
    lines.append(f"Per-cell fine-tuned MAE: mean={per_cell['finetuned_mae'].mean():.4f}  median={per_cell['finetuned_mae'].median():.4f}")

    subsection(lines, "Wilcoxon signed-rank test (paired, per-cell mean absolute error, ensemble only)")
    stat, pval = wilcoxon(per_cell["baseline_mae"], per_cell["finetuned_mae"])
    lines.append(f"  statistic={stat:.4f}  p-value={pval:.4g}  "
                  f"({'SIGNIFICANT' if pval < 0.05 else 'not significant'} at alpha=0.05)")
    n_improved = (per_cell["finetuned_mae"] < per_cell["baseline_mae"]).sum()
    lines.append(f"  Cells where fine-tuning reduced MAE: {n_improved}/{n_cells} ({n_improved/n_cells*100:.1f}%)")

    subsection(lines, f"Block bootstrap ({N_BOOTSTRAP_SAMPLES} resamples of entire cells, not rows) "
                       f"on the mean MAE difference (baseline - fine-tuned)")
    rng = np.random.RandomState(RANDOM_SEED)
    cell_ids = per_cell.index.values
    diffs = per_cell["baseline_mae"].values - per_cell["finetuned_mae"].values
    bootstrap_means = np.empty(N_BOOTSTRAP_SAMPLES)
    for i in range(N_BOOTSTRAP_SAMPLES):
        sample_idx = rng.randint(0, n_cells, size=n_cells)
        bootstrap_means[i] = diffs[sample_idx].mean()
    ci_low, ci_high = np.percentile(bootstrap_means, [2.5, 97.5])
    lines.append(f"  Mean MAE reduction (baseline - fine-tuned): {diffs.mean():.4f}")
    lines.append(f"  95% block-bootstrap CI: [{ci_low:.4f}, {ci_high:.4f}]  "
                  f"({'excludes zero -- significant improvement' if ci_low > 0 else 'excludes zero -- significant WORSENING' if ci_high < 0 else 'includes zero -- not significant'})")

    results_path = OUTPUT_DIR / "statistical_test_results.txt"
    results_path.write_text("Stage 4: Statistical Testing (Cell-Level, Ensemble Only)\n" + "=" * 57 + "\n\n" + "\n".join(lines))
    print(f"    Wrote {results_path.name}")
    pipeline_log.extend(lines)

    return {
        "n_holdout_cells": n_cells,
        "wilcoxon_statistic": float(stat), "wilcoxon_pvalue": float(pval),
        "pct_cells_improved": n_improved / n_cells * 100,
        "bootstrap_mean_mae_reduction": float(diffs.mean()),
        "bootstrap_ci_low": float(ci_low), "bootstrap_ci_high": float(ci_high),
    }


# ============================================================================
# STAGE 5: Diagnostics (Baseline vs. Fine-Tuned, on the Holdout)
# ============================================================================
#
# Repeats ONLY the diagnostics whose underlying quantity can actually
# change as a result of fine-tuning: calibration, land cover category
# bias, SHAP dependence on the two features directly implicated in the
# domain-shift analysis, and the distance-to-coast error-by-bin table.
#
# Deliberately NOT repeated: PCA, t-SNE, Mahalanobis distance,
# nearest-neighbor distance, and a full all-features SHAP sweep.
# Fine-tuning changes the model's learned mapping, not the input feature
# distributions -- those four methods characterize the DATA, which is
# identical to what the domain-shift analysis already examined in full;
# rerunning them would reproduce the same figures relabeled, exactly the
# redundant-diagnostic pattern this experiment is designed to avoid.
# ============================================================================

def run_stage5(pipeline_log, predictions_df, finetuned_models):
    section(pipeline_log, "STAGE 5: Diagnostics (Baseline vs. Fine-Tuned, Holdout)")
    lines = []
    FIGURES_DIR.mkdir(exist_ok=True)

    ph_df = pd.read_csv(PH_FEATURES_PATH)
    split_assignment = pd.read_csv(SPLIT_ASSIGNMENT_PATH)
    ph_df = ph_df.merge(split_assignment[[IDENTIFIER_COLUMN, "split"]], on=IDENTIFIER_COLUMN, how="left")
    holdout_df = ph_df[ph_df["split"] == "holdout"].reset_index(drop=True)
    combined = pd.concat([predictions_df.reset_index(drop=True), holdout_df.drop(columns=[IDENTIFIER_COLUMN])], axis=1)

    # --- Calibration ---
    print("  Calibration comparison...")
    subsection(lines, "Calibration by predicted-value decile (baseline vs. fine-tuned)")

    def calib_table(pred_col):
        d = combined[[pred_col, "actual_pm25"]].copy()
        d["bin"] = pd.qcut(d[pred_col], 10, duplicates="drop")
        g = d.groupby("bin", observed=True).agg(n=("actual_pm25", "size"), mean_actual=("actual_pm25", "mean"),
                                                    mean_predicted=(pred_col, "mean"))
        g["bias"] = g["mean_predicted"] - g["mean_actual"]
        return g

    baseline_calib = calib_table("baseline_ensemble_pred")
    finetuned_calib = calib_table("finetuned_ensemble_pred")
    lines.append("  BASELINE:")
    for idx, row in baseline_calib.iterrows():
        lines.append(f"    {str(idx):<25} n={int(row['n']):<6} bias={row['bias']:+7.2f}")
    lines.append("  FINE-TUNED:")
    for idx, row in finetuned_calib.iterrows():
        lines.append(f"    {str(idx):<25} n={int(row['n']):<6} bias={row['bias']:+7.2f}")

    fig, ax = plt.subplots(figsize=(7, 7))
    max_val = max(baseline_calib["mean_actual"].max(), baseline_calib["mean_predicted"].max(),
                  finetuned_calib["mean_actual"].max(), finetuned_calib["mean_predicted"].max())
    ax.plot([0, max_val], [0, max_val], "k--", label="Perfect calibration")
    ax.plot(baseline_calib["mean_actual"], baseline_calib["mean_predicted"], "o-", label="Baseline", color="tab:red")
    ax.plot(finetuned_calib["mean_actual"], finetuned_calib["mean_predicted"], "o-", label="Fine-tuned", color="tab:green")
    ax.set_xlabel("Mean actual pm25 (per decile)")
    ax.set_ylabel("Mean predicted pm25")
    ax.set_title("Calibration: Baseline vs. Fine-Tuned (holdout)")
    ax.legend()
    fig.savefig(FIGURES_DIR / "calibration_comparison.png", dpi=100, bbox_inches="tight")
    plt.close(fig)

    # --- Land cover category bias ---
    print("  Land cover category bias comparison...")
    subsection(lines, "Land cover category bias (baseline vs. fine-tuned)")
    lc_cols = sorted(c for c in combined.columns if c.startswith("landcover_local_"))
    category_rows = []
    for col in lc_cols:
        mask = combined[col].astype(bool)
        n = int(mask.sum())
        if n == 0:
            continue
        b_bias = (combined.loc[mask, "baseline_ensemble_pred"] - combined.loc[mask, "actual_pm25"]).mean()
        f_bias = (combined.loc[mask, "finetuned_ensemble_pred"] - combined.loc[mask, "actual_pm25"]).mean()
        label = col.replace("landcover_local_", "")
        category_rows.append({"class": label, "n": n, "baseline_bias": b_bias, "finetuned_bias": f_bias})
        lines.append(f"  {label:<35} n={n:<6} baseline_bias={b_bias:+7.2f}  finetuned_bias={f_bias:+7.2f}  "
                      f"delta={f_bias - b_bias:+7.2f}")
    pd.DataFrame(category_rows).to_csv(OUTPUT_DIR / "category_bias_comparison.csv", index=False)

    # --- SHAP for population and distance-to-coast only ---
    print("  SHAP comparison (population and distance-to-coast only)...")
    subsection(lines, "SHAP comparison: baseline vs. fine-tuned XGBoost, features of interest only")
    feature_cols = [c for c in ph_df.columns if c not in (IDENTIFIER_COLUMN, TARGET_COLUMN, "split")]
    rng = np.random.RandomState(RANDOM_SEED)
    shap_sample = holdout_df[feature_cols].sample(n=min(SHAP_SAMPLE_SIZE, len(holdout_df)), random_state=RANDOM_SEED)

    baseline_xgb = joblib.load(BASELINE_MODELS_DIR / "xgboost_baseline.joblib")
    baseline_explainer = shap.TreeExplainer(baseline_xgb)
    finetuned_explainer = shap.TreeExplainer(finetuned_models["xgboost"])
    baseline_shap = baseline_explainer.shap_values(shap_sample)
    finetuned_shap = finetuned_explainer.shap_values(shap_sample)

    for feat in SHAP_FEATURES_OF_INTEREST:
        if feat not in feature_cols:
            continue
        idx = feature_cols.index(feat)
        b_mean_abs_shap = np.abs(baseline_shap[:, idx]).mean()
        f_mean_abs_shap = np.abs(finetuned_shap[:, idx]).mean()
        lines.append(f"  {feat:<30} baseline mean|SHAP|={b_mean_abs_shap:.4f}  "
                      f"finetuned mean|SHAP|={f_mean_abs_shap:.4f}  ratio={f_mean_abs_shap/b_mean_abs_shap:.2f}")

    # --- Distance-to-coast error by bin ---
    print("  Distance-to-coast error-by-bin comparison...")
    subsection(lines, "Distance-to-coast error by bin (baseline vs. fine-tuned)")
    combined["distance_bin"] = pd.cut(combined["distance_to_coast_km"], bins=DISTANCE_BINS, labels=DISTANCE_LABELS)
    for bin_label, group in combined.groupby("distance_bin", observed=True):
        b_bias = (group["baseline_ensemble_pred"] - group["actual_pm25"]).mean()
        f_bias = (group["finetuned_ensemble_pred"] - group["actual_pm25"]).mean()
        lines.append(f"  {str(bin_label):<12} n={len(group):<6} baseline_bias={b_bias:+7.2f}  "
                      f"finetuned_bias={f_bias:+7.2f}  delta={f_bias - b_bias:+7.2f}")

    # --- Trivial baseline comparison ---
    # R2 is, by construction, already a comparison against predicting the
    # evaluation set's OWN mean (R2=0 corresponds exactly to that trivial
    # predictor) -- but that comparison is often implicit and easy to
    # misread, and a real deployed model wouldn't know the holdout's true
    # mean in advance anyway. Made explicit here with TWO trivial
    # baselines: the holdout's own mean (ties directly to what R2=0 means)
    # and the finetune_train mean (a realistic "naive Philippine
    # climatology, zero machine learning" comparison, using only
    # information actually available at prediction time).
    print("  Trivial baseline comparison...")
    subsection(lines, "Trivial baseline comparison (context for interpreting R2, which by construction "
                       "compares against predicting the holdout's own mean)")
    holdout_mean = combined["actual_pm25"].mean()
    finetune_train_mean = ph_df.loc[ph_df["split"] == "finetune_train", TARGET_COLUMN].mean()

    for label, constant_value in [("Predict holdout's own mean (ties to R2=0 exactly)", holdout_mean),
                                    ("Predict finetune_train mean (realistic 'zero-ML' baseline)", finetune_train_mean)]:
        trivial_metrics = compute_metrics(combined["actual_pm25"], np.full(len(combined), constant_value))
        lines.append(f"\n  {label} (constant={constant_value:.3f}):")
        lines.append(f"    RMSE={trivial_metrics['RMSE']:.4f}  MAE={trivial_metrics['MAE']:.4f}  R2={trivial_metrics['R2']:.4f}")

    baseline_ensemble_metrics = compute_metrics(combined["actual_pm25"], combined["baseline_ensemble_pred"])
    finetuned_ensemble_metrics = compute_metrics(combined["actual_pm25"], combined["finetuned_ensemble_pred"])
    lines.append(f"\n  For direct comparison -- baseline ensemble:   RMSE={baseline_ensemble_metrics['RMSE']:.4f}  "
                  f"MAE={baseline_ensemble_metrics['MAE']:.4f}  R2={baseline_ensemble_metrics['R2']:.4f}")
    lines.append(f"  For direct comparison -- fine-tuned ensemble: RMSE={finetuned_ensemble_metrics['RMSE']:.4f}  "
                  f"MAE={finetuned_ensemble_metrics['MAE']:.4f}  R2={finetuned_ensemble_metrics['R2']:.4f}")
    lines.append(f"\n  Interpretation: a model with R2 < 0 performs WORSE than the corresponding trivial "
                  f"constant-mean predictor above -- if the fine-tuned ensemble's RMSE still exceeds the "
                  f"trivial baselines' RMSE, that is direct evidence the improvement, while statistically "
                  f"significant in relative terms, has not yet reached a genuinely useful absolute "
                  f"performance level.")

    # --- Negative (physically impossible) prediction diagnostic ---
    print("  Negative-prediction diagnostic...")
    subsection(lines, "Negative (physically impossible) prediction diagnostic")
    for label, pred_col in [("Baseline", "baseline_ensemble_pred"), ("Fine-tuned", "finetuned_ensemble_pred")]:
        neg_mask = combined[pred_col] < 0
        n_neg = int(neg_mask.sum())
        pct_neg = n_neg / len(combined) * 100
        lines.append(f"\n  {label} ensemble: {n_neg} rows ({pct_neg:.2f}%) with a negative predicted pm25.")
        if n_neg > 0:
            neg_vals = combined.loc[neg_mask, pred_col]
            lines.append(f"    Most negative: {neg_vals.min():.2f}  mean of negative predictions: {neg_vals.mean():.2f}")

            subsection(lines, f"  Where do {label.lower()}'s negative predictions concentrate?")
            if neg_mask.any():
                neg_by_lc = combined.loc[neg_mask, lc_cols].sum()
                for col, count in neg_by_lc.items():
                    if count > 0:
                        lines.append(f"    land cover {col.replace('landcover_local_', ''):<30} {int(count)} of the negative rows")
                neg_by_dist = combined.loc[neg_mask, "distance_bin"].value_counts()
                for bin_label, count in neg_by_dist.items():
                    if count > 0:
                        lines.append(f"    distance-to-coast bin {str(bin_label):<12} {int(count)} of the negative rows")

    diagnostics_path = OUTPUT_DIR / "finetune_diagnostics_report.txt"
    diagnostics_path.write_text("Stage 5: Diagnostics (Baseline vs. Fine-Tuned, Holdout)\n" + "=" * 56 + "\n\n" + "\n".join(lines))
    print(f"    Wrote {diagnostics_path.name}")
    pipeline_log.extend(lines)


# ============================================================================
# Main
# ============================================================================

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
    pipeline_log = ["Complete Transfer Learning Experiment Pipeline", "=" * 47]

    run_stage0(pipeline_log)
    finetuned_models = run_stage1(pipeline_log)
    finetuned_weights = run_stage2(pipeline_log, finetuned_models)
    predictions_df = run_stage3(pipeline_log, finetuned_models, finetuned_weights)
    run_stage4(pipeline_log, predictions_df)
    run_stage5(pipeline_log, predictions_df, finetuned_models)

    full_log_path = OUTPUT_DIR / "full_pipeline_log.txt"
    full_log_path.write_text("\n".join(pipeline_log))
    print(f"\nAll stages complete. Full consolidated log written to {full_log_path}")


if __name__ == "__main__":
    main()
