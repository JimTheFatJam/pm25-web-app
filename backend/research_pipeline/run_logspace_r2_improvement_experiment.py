"""
Log-Space R2 Improvement Experiment
========================================

Investigates whether the log-space pipeline's modest R2 (~0.15) reflects
genuine under-fitting of high-PM2.5 events -- the model became far more
stable (prediction std ~33 -> ~5) and far better at typical conditions
(P95 error ~65 -> ~22), but its single worst-case error barely moved
(~395 -> ~390), consistent with a model that may now be too conservative,
systematically under-predicting the high end of the distribution that
drives variance and R2.

Scoping decision, stated explicitly
------------------------------------------
"Current baseline" here is config_B_logspace_unweighted specifically (log-
space, UNWEIGHTED source), not config_C (equal-country). The two showed a
negligible difference (0.04% RMSE) in the prior experiment, and stacking
equal-country country-level weighting together with this experiment's
PM2.5-magnitude row-level weighting would confound two different
weighting mechanisms in one result -- inconsistent with the "isolate one
variable" principle already applied throughout this project. Using B
keeps this experiment's own variable (magnitude-based weighting,
calibration) cleanly isolated.

Four sub-experiments
---------------------
  A. Reference -- NOT re-run. Read directly from the existing,
     already-validated finetune_outputs_logspace/config_B_logspace_unweighted/
     outputs.
  B. Weighted log-space training -- source AND fine-tuning targets both
     weighted by weight = 1 + alpha * percentile_rank(pm25), computed
     SEPARATELY within each stage's own training data (source and
     Philippine fine-tune-train have very different pm25 distributions,
     so a single shared percentile rank would not be meaningful for
     both). Tested at alpha = 0.5, 1.0, 2.0. This is a magnitude-based
     SAMPLE weight, not a target transformation -- it changes how much
     each row counts in the loss, never the row's own value, so it
     introduces no target leakage.
  C. Post-hoc calibration -- fit on finetune_val predictions from the
     ALREADY-TRAINED, ALREADY-SAVED config_B models (loaded directly,
     not retrained), NEVER on holdout labels. Two methods: linear
     (pred_calibrated = a*pred + b) and isotonic regression. Applied to
     holdout predictions only after fitting is complete.
  D. Extreme-event diagnostic -- the top 20 largest absolute errors and
     top 5% largest actual PM2.5 observations in the holdout, each
     reported with location, timing, AOD availability, meteorological
     conditions, and neighborhood features, from config_B's existing
     saved holdout predictions (no retraining).

No new pipeline logic is introduced beyond what B/C/D specifically call
for -- no deep learning, no broad hyperparameter search, no target
transformation beyond the already-validated log1p.

Output
------
finetune_outputs_r2_improvement/
  experiment_B_weighted/<alpha_config>/  -- models, predictions, metrics
  experiment_C_calibration/              -- calibrated predictions, metrics
  experiment_D_extreme_events.csv        -- the diagnostic table
  r2_improvement_comparison_report.txt   -- all required questions answered

Usage
-----
Place this script alongside source_features.csv, source_metadata.csv,
philippines_features.csv, philippines_metadata.csv, and
finetune_outputs_logspace/config_B_logspace_unweighted/ (from the prior
log-space experiment), and run it.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.isotonic import IsotonicRegression

from train_baseline_models import compute_metrics
from run_logspace_pipeline_experiment import (
    train_source_logspace, finetune_logspace, compute_ensemble_weights, predict_ensemble_real,
    to_log, to_real, IDENTIFIER_COLUMN, TARGET_COLUMN, RANDOM_SEED,
)

# ============================================================
# Configuration -- MACROS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
SOURCE_FEATURES_PATH = SCRIPT_DIR / "source_features.csv"
SOURCE_METADATA_PATH = SCRIPT_DIR / "source_metadata.csv"
PH_FEATURES_PATH = SCRIPT_DIR / "philippines_features.csv"
PH_METADATA_PATH = SCRIPT_DIR / "philippines_metadata.csv"
BASELINE_DIR = SCRIPT_DIR / "baseline_model_outputs"

CONFIG_B_DIR = SCRIPT_DIR / "finetune_outputs_logspace" / "config_B_logspace_unweighted"  # reference (A), and source of models for C/D
PH_SPLIT_PATH = SCRIPT_DIR / "finetune_outputs" / "philippines_finetune_split_assignment.csv"

OUTPUT_ROOT = SCRIPT_DIR / "finetune_outputs_r2_improvement"

ALPHA_VALUES = [0.5, 1.0, 2.0]
TOP_N_ERRORS = 20
TOP_PCT_PM25 = 0.05

CONTEXT_COLUMNS = ["aod_local", "temperature_2m_local", "wind_speed_10m_local", "boundary_layer_height_local",
                   "precipitation_local", "distance_to_coast_km", "elevation_local",
                   "population_neighborhood_mean", "ndvi_neighborhood_mean"]


# ============================================================
# Experiment B: magnitude-weighted log-space training
# ============================================================

def compute_magnitude_weights(y_real: pd.Series, alpha: float) -> np.ndarray:
    percentile_rank = y_real.rank(pct=True).values
    return 1 + alpha * percentile_rank


def run_experiment_b(source_df, ph_df, feature_cols) -> list:
    results = []
    source_train_df = source_df[source_df["split"] == "train"]
    source_val_df = source_df[source_df["split"] == "val"]
    ph_train_df = ph_df[ph_df["split"] == "finetune_train"]
    ph_val_df = ph_df[ph_df["split"] == "finetune_val"]
    ph_holdout_df = ph_df[ph_df["split"] == "holdout"]

    for alpha in ALPHA_VALUES:
        config_name = f"alpha_{alpha}"
        run_dir = OUTPUT_ROOT / "experiment_B_weighted" / config_name
        models_dir = run_dir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n  --- Experiment B, alpha={alpha} ---")

        source_weight = compute_magnitude_weights(source_train_df[TARGET_COLUMN], alpha)
        print(f"    Training source models (log-space, magnitude-weighted, alpha={alpha})...")
        source_models = train_source_logspace(source_train_df[feature_cols], source_train_df[TARGET_COLUMN], source_weight)

        ph_weight_train = compute_magnitude_weights(ph_train_df[TARGET_COLUMN], alpha)
        # finetune_logspace does not itself accept a fine-tune sample_weight
        # (the validated function's signature is fixed); magnitude weighting
        # for the fine-tuning stage is applied by DUPLICATING high-percentile
        # rows in the fine-tune training set proportionally to their weight,
        # a standard, simple equivalent to sample weighting for tree boosting
        # that does not require modifying the already-validated function.
        replication_counts = np.maximum(1, np.round(ph_weight_train)).astype(int)
        ph_train_weighted = ph_train_df.loc[ph_train_df.index.repeat(replication_counts)].reset_index(drop=True)
        print(f"    Fine-tuning (log-space, magnitude-weighted): {len(ph_train_df)} rows -> "
              f"{len(ph_train_weighted)} rows after weight-proportional replication.")

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
        (run_dir / "config.json").write_text(json.dumps({"alpha": alpha, "random_seed": RANDOM_SEED}, indent=2))

        result = {
            "configuration": f"B_{config_name}", "rmse": metrics["RMSE"], "mae": metrics["MAE"], "r2": metrics["R2"],
            "n_negative": n_negative, "pct_negative": n_negative / len(predictions_df) * 100,
            "pred_std": float(holdout_preds["ensemble"].std()),
            "p95_abs_error": float(pd.Series(abs_error).quantile(0.95)), "max_abs_error": float(abs_error.max()),
        }
        results.append(result)
        print(f"    RMSE={result['rmse']:.4f}  MAE={result['mae']:.4f}  R2={result['r2']:.4f}  "
              f"neg={result['pct_negative']:.2f}%  pred_std={result['pred_std']:.2f}")

    return results


# ============================================================
# Experiment C: post-hoc calibration (fit on val, apply to holdout)
# ============================================================

def run_experiment_c(ph_df, feature_cols) -> list:
    print("\n  Loading config_B's already-trained, already-saved models...")
    xgb_model = joblib.load(CONFIG_B_DIR / "models" / "finetuned_xgboost_logspace.joblib")
    lgbm_model = joblib.load(CONFIG_B_DIR / "models" / "finetuned_lightgbm_logspace.joblib")
    catboost_model = CatBoostRegressor()
    catboost_model.load_model(str(CONFIG_B_DIR / "models" / "finetuned_catboost_logspace.cbm"))
    models = {"xgboost": xgb_model, "lightgbm": lgbm_model, "catboost": catboost_model}

    ph_val_df = ph_df[ph_df["split"] == "finetune_val"]
    ph_holdout_df = ph_df[ph_df["split"] == "holdout"]

    ensemble_weights = compute_ensemble_weights(models, ph_val_df[feature_cols], ph_val_df[TARGET_COLUMN])
    val_preds = predict_ensemble_real(models, ensemble_weights, ph_val_df[feature_cols])
    holdout_preds = predict_ensemble_real(models, ensemble_weights, ph_holdout_df[feature_cols])

    run_dir = OUTPUT_ROOT / "experiment_C_calibration"
    run_dir.mkdir(parents=True, exist_ok=True)
    results = []

    uncalibrated_metrics = compute_metrics(ph_holdout_df[TARGET_COLUMN], holdout_preds["ensemble"])
    results.append({"configuration": "C_uncalibrated_reference", **{"rmse": uncalibrated_metrics["RMSE"],
                     "mae": uncalibrated_metrics["MAE"], "r2": uncalibrated_metrics["R2"]},
                     "n_negative": int((holdout_preds["ensemble"] < 0).sum()),
                     "pct_negative": (holdout_preds["ensemble"] < 0).mean() * 100,
                     "pred_std": float(holdout_preds["ensemble"].std()),
                     "p95_abs_error": float(pd.Series(np.abs(holdout_preds["ensemble"] - ph_holdout_df[TARGET_COLUMN].values)).quantile(0.95)),
                     "max_abs_error": float(np.abs(holdout_preds["ensemble"] - ph_holdout_df[TARGET_COLUMN].values).max())})

    # --- Linear calibration: fit a, b on VAL ONLY ---
    print("  Fitting linear calibration on finetune_val (never holdout)...")
    a, b = np.polyfit(val_preds["ensemble"], ph_val_df[TARGET_COLUMN].values, 1)
    linear_calibrated_holdout = a * holdout_preds["ensemble"] + b
    linear_metrics = compute_metrics(ph_holdout_df[TARGET_COLUMN], linear_calibrated_holdout)
    n_neg_linear = int((linear_calibrated_holdout < 0).sum())
    abs_err_linear = np.abs(linear_calibrated_holdout - ph_holdout_df[TARGET_COLUMN].values)
    results.append({"configuration": "C_linear_calibration", "rmse": linear_metrics["RMSE"], "mae": linear_metrics["MAE"],
                     "r2": linear_metrics["R2"], "n_negative": n_neg_linear,
                     "pct_negative": n_neg_linear / len(holdout_preds["ensemble"]) * 100,
                     "pred_std": float(linear_calibrated_holdout.std()),
                     "p95_abs_error": float(pd.Series(abs_err_linear).quantile(0.95)), "max_abs_error": float(abs_err_linear.max())})
    pd.DataFrame({"a": [a], "b": [b]}).to_csv(run_dir / "linear_calibration_params.csv", index=False)

    # --- Isotonic calibration: fit on VAL ONLY ---
    print("  Fitting isotonic calibration on finetune_val (never holdout)...")
    isotonic = IsotonicRegression(out_of_bounds="clip")
    isotonic.fit(val_preds["ensemble"], ph_val_df[TARGET_COLUMN].values)
    isotonic_calibrated_holdout = isotonic.predict(holdout_preds["ensemble"])
    isotonic_metrics = compute_metrics(ph_holdout_df[TARGET_COLUMN], isotonic_calibrated_holdout)
    n_neg_iso = int((isotonic_calibrated_holdout < 0).sum())
    abs_err_iso = np.abs(isotonic_calibrated_holdout - ph_holdout_df[TARGET_COLUMN].values)
    results.append({"configuration": "C_isotonic_calibration", "rmse": isotonic_metrics["RMSE"], "mae": isotonic_metrics["MAE"],
                     "r2": isotonic_metrics["R2"], "n_negative": n_neg_iso,
                     "pct_negative": n_neg_iso / len(holdout_preds["ensemble"]) * 100,
                     "pred_std": float(isotonic_calibrated_holdout.std()),
                     "p95_abs_error": float(pd.Series(abs_err_iso).quantile(0.95)), "max_abs_error": float(abs_err_iso.max())})
    joblib.dump(isotonic, run_dir / "isotonic_calibration_model.joblib")

    predictions_df = pd.DataFrame({
        IDENTIFIER_COLUMN: ph_holdout_df[IDENTIFIER_COLUMN].values, "actual_pm25": ph_holdout_df[TARGET_COLUMN].values,
        "uncalibrated_pred": holdout_preds["ensemble"], "linear_calibrated_pred": linear_calibrated_holdout,
        "isotonic_calibrated_pred": isotonic_calibrated_holdout,
    })
    predictions_df.to_csv(run_dir / "predictions_holdout_calibrated.csv", index=False)

    for r in results:
        print(f"    {r['configuration']}: RMSE={r['rmse']:.4f}  R2={r['r2']:.4f}  neg={r['pct_negative']:.2f}%")

    return results


# ============================================================
# Experiment D: extreme-event diagnostic (no retraining)
# ============================================================

def run_experiment_d(ph_df_with_meta) -> pd.DataFrame:
    print("\n  Building extreme-event diagnostic table from config_B's saved holdout predictions...")
    preds_df = pd.read_csv(CONFIG_B_DIR / "predictions_philippines_holdout.csv")
    pred_col = "ensemble_pred" if "ensemble_pred" in preds_df.columns else "finetuned_ensemble_pred"

    # ph_df_with_meta already carries datetime_utc/centroid columns as
    # genuine row-aligned columns (attached via position, not a grid_id
    # merge -- grid_id alone is not a unique key, since it repeats once
    # per hour; the established, project-wide convention for combining
    # features and metadata is positional concatenation, not a value
    # merge, exactly as used consistently since the EDA and domain-shift
    # analysis scripts).
    ph_holdout_df = ph_df_with_meta[ph_df_with_meta["split"] == "holdout"].reset_index(drop=True)
    combined = pd.concat([preds_df.reset_index(drop=True), ph_holdout_df.drop(columns=[IDENTIFIER_COLUMN]).reset_index(drop=True)], axis=1)
    combined["abs_error"] = (combined[pred_col] - combined["actual_pm25"]).abs()

    top_errors = combined.nlargest(TOP_N_ERRORS, "abs_error").copy()
    top_errors["category"] = "top_abs_error"

    n_top_pm25 = max(1, int(len(combined) * TOP_PCT_PM25))
    top_pm25 = combined.nlargest(n_top_pm25, "actual_pm25").copy()
    top_pm25["category"] = "top_5pct_actual_pm25"

    diagnostic_df = pd.concat([top_errors, top_pm25], ignore_index=True).drop_duplicates(subset=[IDENTIFIER_COLUMN, "datetime_utc", "category"])

    output_cols = [IDENTIFIER_COLUMN, "category", "centroid_latitude", "centroid_longitude", "datetime_utc",
                   "actual_pm25", pred_col, "abs_error"] + [c for c in CONTEXT_COLUMNS if c in diagnostic_df.columns]
    diagnostic_df = diagnostic_df[output_cols]
    diagnostic_df["aod_available"] = diagnostic_df["aod_local"].notna() if "aod_local" in diagnostic_df.columns else np.nan

    diagnostic_df.to_csv(OUTPUT_ROOT / "experiment_D_extreme_events.csv", index=False)
    print(f"    Wrote experiment_D_extreme_events.csv ({len(diagnostic_df)} rows).")
    return diagnostic_df


# ============================================================
# Main
# ============================================================

def main():
    OUTPUT_ROOT.mkdir(exist_ok=True)

    print("Loading source features and the existing baseline source cell split...")
    source_df = pd.read_csv(SOURCE_FEATURES_PATH)
    source_cell_split = pd.read_csv(BASELINE_DIR / "baseline_cell_split_assignment.csv")
    source_df = source_df.merge(source_cell_split[[IDENTIFIER_COLUMN, "split"]], on=IDENTIFIER_COLUMN, how="left")

    print("Loading Philippines features, metadata, and the existing (unmodified) fine-tune split...")
    ph_df = pd.read_csv(PH_FEATURES_PATH)
    ph_meta = pd.read_csv(PH_METADATA_PATH)
    if len(ph_df) != len(ph_meta):
        raise ValueError(f"Features ({len(ph_df)} rows) and metadata ({len(ph_meta)} rows) row counts differ -- "
                          f"they are no longer row-aligned, which Experiment D's positional metadata attachment depends on.")
    # Positional attachment (not a grid_id merge -- grid_id is not a unique
    # key, since it repeats once per hour), matching the established,
    # project-wide convention for combining features and metadata.
    ph_df_with_meta = pd.concat([ph_df.reset_index(drop=True),
                                  ph_meta[["centroid_latitude", "centroid_longitude", "datetime_utc"]].reset_index(drop=True)], axis=1)
    ph_split = pd.read_csv(PH_SPLIT_PATH)
    ph_df = ph_df.merge(ph_split[[IDENTIFIER_COLUMN, "split"]], on=IDENTIFIER_COLUMN, how="left")
    ph_df_with_meta = ph_df_with_meta.merge(ph_split[[IDENTIFIER_COLUMN, "split"]], on=IDENTIFIER_COLUMN, how="left")
    if ph_df["split"].isna().any():
        raise ValueError("Some Philippine rows have no split assignment.")

    feature_cols = [c for c in source_df.columns if c not in (IDENTIFIER_COLUMN, TARGET_COLUMN, "split", "country")]

    print("\n=== Experiment A: reference (reading existing config_B results, not re-run) ===")
    a_preds = pd.read_csv(CONFIG_B_DIR / "predictions_philippines_holdout.csv")
    a_pred_col = "ensemble_pred" if "ensemble_pred" in a_preds.columns else "finetuned_ensemble_pred"
    a_metrics = compute_metrics(a_preds["actual_pm25"], a_preds[a_pred_col])
    a_abs_err = np.abs(a_preds[a_pred_col] - a_preds["actual_pm25"])
    result_a = {"configuration": "A_reference_unweighted_logspace", "rmse": a_metrics["RMSE"], "mae": a_metrics["MAE"],
                "r2": a_metrics["R2"], "n_negative": int((a_preds[a_pred_col] < 0).sum()),
                "pct_negative": (a_preds[a_pred_col] < 0).mean() * 100, "pred_std": float(a_preds[a_pred_col].std()),
                "p95_abs_error": float(a_abs_err.quantile(0.95)), "max_abs_error": float(a_abs_err.max())}
    print(f"  RMSE={result_a['rmse']:.4f}  MAE={result_a['mae']:.4f}  R2={result_a['r2']:.4f}")

    print("\n=== Experiment B: magnitude-weighted log-space training ===")
    results_b = run_experiment_b(source_df, ph_df, feature_cols)

    print("\n=== Experiment C: post-hoc calibration ===")
    results_c = run_experiment_c(ph_df, feature_cols)

    print("\n=== Experiment D: extreme-event diagnostic ===")
    diagnostic_df = run_experiment_d(ph_df_with_meta)

    all_results = [result_a] + results_b + results_c
    comparison_df = pd.DataFrame(all_results)
    comparison_df.to_csv(OUTPUT_ROOT / "r2_improvement_comparison.csv", index=False)

    # --- Report ---
    lines = ["Log-Space R2 Improvement Experiment -- Comparison Report", "=" * 58, ""]
    lines.append("Reference (A) = config_B_logspace_unweighted, read directly from its existing, "
                  "already-validated outputs -- not re-run.\n")
    for r in all_results:
        lines.append(f"{r['configuration']}:")
        lines.append(f"  RMSE={r['rmse']:.4f}  MAE={r['mae']:.4f}  R2={r['r2']:.4f}")
        lines.append(f"  Negative predictions: {r['n_negative']} ({r['pct_negative']:.2f}%)")
        lines.append(f"  Prediction std={r['pred_std']:.2f}  P95 abs error={r['p95_abs_error']:.2f}  max abs error={r['max_abs_error']:.2f}")
        lines.append("")

    lines.append("=" * 58)
    lines.append("REQUIRED QUESTIONS")
    lines.append("=" * 58)

    best_r2_config = max(all_results, key=lambda r: r["r2"])
    lines.append(f"\n1. Did any method improve R2 over the reference (R2={result_a['r2']:.4f})?")
    for r in all_results[1:]:
        delta = r["r2"] - result_a["r2"]
        lines.append(f"   {r['configuration']}: R2={r['r2']:.4f} (delta={delta:+.4f})")
    lines.append(f"   Best R2: {best_r2_config['configuration']} (R2={best_r2_config['r2']:.4f})")

    lines.append(f"\n2. Did improving R2 increase extreme errors (max abs error)?")
    for r in all_results[1:]:
        max_err_delta = r["max_abs_error"] - result_a["max_abs_error"]
        r2_delta = r["r2"] - result_a["r2"]
        lines.append(f"   {r['configuration']}: R2 delta={r2_delta:+.4f}, max_abs_error delta={max_err_delta:+.2f}")

    lines.append(f"\n3. Is the reference log-space model too conservative? (prediction std comparison)")
    lines.append(f"   Reference pred_std={result_a['pred_std']:.2f}")
    for r in all_results[1:]:
        lines.append(f"   {r['configuration']} pred_std={r['pred_std']:.2f} (delta={r['pred_std']-result_a['pred_std']:+.2f})")

    lines.append(f"\n4. Best tradeoff (RMSE / MAE / R2 / physical validity [0% negative preferred]):")
    for r in all_results:
        lines.append(f"   {r['configuration']}: RMSE={r['rmse']:.2f}  R2={r['r2']:.4f}  neg%={r['pct_negative']:.2f}")

    lines.append(f"\n5. Extreme-event diagnostic summary (see experiment_D_extreme_events.csv for full detail):")
    if "aod_available" in diagnostic_df.columns:
        aod_avail_rate = diagnostic_df["aod_available"].mean() * 100
        lines.append(f"   AOD availability among extreme-error/extreme-PM2.5 rows: {aod_avail_rate:.1f}% "
                      f"(compare against the overall holdout's AOD availability rate to check whether missing "
                      f"AOD is disproportionately represented among failures).")
    lines.append(f"   {len(diagnostic_df)} rows written for manual/further inspection -- categorized as "
                  f"top_abs_error and/or top_5pct_actual_pm25.")

    report_path = OUTPUT_ROOT / "r2_improvement_comparison_report.txt"
    report_path.write_text("\n".join(lines))
    print(f"\nWrote {report_path}")
    print("\nR2 improvement experiment complete.")


if __name__ == "__main__":
    main()
