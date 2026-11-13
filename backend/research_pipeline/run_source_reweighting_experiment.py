"""
Source-Domain Country Reweighting Experiment
==================================================

Primary research question: does correcting source-domain composition
through country-level sample weighting -- equal-country weighting, and
maritime-domain emphasis on top of it -- improve ZERO-SHOT transfer
performance to the Philippines (no fine-tuning involved at all, isolating
this question from the separate fine-tuning technique already studied)?

Deliberately excluded from this experiment, per explicit scope: log-space
training, calibration, clipping. This experiment answers exactly one
question -- does source composition correction help -- and nothing else.

Maritime group: Indonesia, Malaysia, Singapore
------------------------------------------------------
Selected after evaluating every source country against Koppen climate
class, coastal exposure, land-cover similarity, urbanization pattern, and
aerosol transport characteristics relative to the Philippines:
  - Indonesia: Af/Am, archipelagic/coastal, similar tropical land cover.
    Caveat: Indonesia (Sumatra/Kalimantan) is a SOURCE region for the
    regional peat/forest-fire haze, while the Philippines is more a
    downwind recipient -- an imperfect but still far better match than
    any continental country.
  - Malaysia: Af, heavily coastal, similar land cover, and -- unlike
    Indonesia -- primarily a haze RECIPIENT (same transport role as PH).
  - Singapore: climate/coastal match well, but land cover is almost
    entirely Urban_and_Built_up with essentially no rural/agricultural
    share, unlike PH's mixed landscape. Kept anyway: only 12 grid cells,
    so its distorting influence on the pooled group is limited.
  - Vietnam EXCLUDED: spans ~8N-23N, a genuinely wide climate gradient
    (humid subtropical north vs. tropical monsoon south) -- inconsistent
    as a single country-level weight given PH's comparatively uniform
    tropical regime. A southern-VN-only refinement is a legitimate
    FUTURE sub-national experiment, not compatible with keeping this
    design at country-level granularity.
  - Cambodia EXCLUDED: short coastline (~443km), and its population/
    pollution center (Phnom Penh) sits inland along the Mekong/Tonle Sap
    basin, not coastally. Koppen Aw with a pronounced dry season, unlike
    the wetter Af/Am regime PH and the maritime group share.
  - Coastal Thailand NOT carved out: conceptually closer to PH than
    Bangkok/central Thailand, but we lack direct confirmation of how many
    of the 144 Thai cells are actually drawn from that region, and
    carving out a sub-national subset would introduce a third weighting
    granularity beyond the country-level design agreed on. Noted as a
    legitimate future refinement, not implemented here.

Configurations
--------------
  unweighted       -- sample_weight=1 for every row; this is the BASELINE
                       REPRODUCTION CHECKPOINT, not just a comparison row.
                       Verified against the original baseline_model_outputs/
                       before the weighting comparison is treated as valid.
  equal_country    -- weight_i = (total_rows / 7) / rows_in_country_i
  maritime_2x      -- equal_country weights, with ID/MY/SG multiplied by 2
  maritime_4x      -- equal_country weights, with ID/MY/SG multiplied by 4

Fairness safeguards
-------------------
- The EXACT SAME cell-split assignment (baseline_cell_split_assignment.csv,
  the original 70/15/15 grid-cell-blocked, country-stratified split) is
  reused unchanged across all four configurations -- sample_weight is the
  ONLY thing that varies.
- Same three base-learner architectures and hyperparameters as the
  original baseline (untuned, matching train_baseline_models.py exactly).
- Evaluation is always UNWEIGHTED, regardless of training configuration --
  weighting only the loss function during fitting, never the metric,
  since weighted evaluation would circularly measure "did the model
  optimize what we told it to" rather than genuine predictive quality.
- Ensemble weight reconstruction (inverse-RMSE) is also computed from
  UNWEIGHTED validation performance in every configuration.

train_baseline_models.py is NOT modified or imported in a way that alters
its behavior -- this script is fully self-contained except for reusing
its compute_metrics() utility, so the original baseline training script
remains completely untouched.

Output
------
finetune_outputs_reweighting/<configuration_name>/
  models/                        -- xgboost.joblib, lightgbm.joblib, catboost.cbm
  country_weights_used.csv       -- the per-country weight table for this configuration
  training_config.json           -- exact hyperparameters and settings used
  source_val_test_metrics.txt    -- source-domain val/test RMSE/MAE/R2
  philippines_transfer_metrics.txt -- full-Philippines zero-shot transfer metrics
  ensemble_weights.csv
  predictions_source_val_test.csv
  predictions_philippines.csv

finetune_outputs_reweighting/baseline_reproduction_check.txt -- the
  required checkpoint comparing the unweighted retrain against the
  original canonical baseline.
finetune_outputs_reweighting/reweighting_comparison.csv -- one row per
  configuration, the final comparison table.
finetune_outputs_reweighting/reweighting_comparison_report.txt

Usage
-----
Place this script alongside source_features.csv, source_metadata.csv,
philippines_features.csv, and baseline_model_outputs/ (for the
reproduction checkpoint and the original cell split), and run it.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from scipy.stats import pearsonr
from xgboost import XGBRegressor

from train_baseline_models import compute_metrics

# ============================================================
# Configuration -- MACROS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
SOURCE_FEATURES_PATH = SCRIPT_DIR / "source_features.csv"
SOURCE_METADATA_PATH = SCRIPT_DIR / "source_metadata.csv"
PH_FEATURES_PATH = SCRIPT_DIR / "philippines_features.csv"
BASELINE_DIR = SCRIPT_DIR / "baseline_model_outputs"

OUTPUT_ROOT = SCRIPT_DIR / "finetune_outputs_reweighting"

IDENTIFIER_COLUMN = "grid_id"
TARGET_COLUMN = "pm25"
RANDOM_SEED = 42

MARITIME_COUNTRIES = ["ID", "MY", "SG"]
SOURCE_COUNTRIES = ["IN", "TH", "VN", "ID", "KH", "SG", "MY"]

XGBOOST_PARAMS = dict(n_estimators=300, max_depth=6, learning_rate=0.05,
                       subsample=0.8, colsample_bytree=0.8,
                       tree_method="hist", n_jobs=-1, random_state=RANDOM_SEED)
LIGHTGBM_PARAMS = dict(n_estimators=300, max_depth=-1, num_leaves=63, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8,
                        n_jobs=-1, random_state=RANDOM_SEED, verbose=-1)
CATBOOST_PARAMS = dict(iterations=300, depth=6, learning_rate=0.05,
                        random_seed=RANDOM_SEED, verbose=False, thread_count=-1)

CONFIGURATIONS = ["unweighted", "equal_country", "maritime_2x", "maritime_4x"]


# ============================================================
# Country weight computation
# ============================================================

def compute_country_weights(country_row_counts: pd.Series, configuration: str) -> dict:
    total_rows = country_row_counts.sum()
    n_countries = len(country_row_counts)

    if configuration == "unweighted":
        return {c: 1.0 for c in country_row_counts.index}

    equal_weights = {c: (total_rows / n_countries) / country_row_counts[c] for c in country_row_counts.index}
    if configuration == "equal_country":
        return equal_weights

    multiplier = {"maritime_2x": 2.0, "maritime_4x": 4.0}[configuration]
    return {c: (w * multiplier if c in MARITIME_COUNTRIES else w) for c, w in equal_weights.items()}


# ============================================================
# Train and evaluate one configuration
# ============================================================

def run_one_configuration(configuration: str, source_df: pd.DataFrame, cell_split: pd.DataFrame,
                           ph_df: pd.DataFrame, feature_cols: list) -> dict:
    run_dir = OUTPUT_ROOT / configuration
    models_dir = run_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    country_row_counts = source_df.groupby("country").size()
    country_weights = compute_country_weights(country_row_counts, configuration)
    pd.DataFrame([{"country": c, "weight": w} for c, w in country_weights.items()]).to_csv(
        run_dir / "country_weights_used.csv", index=False)

    train_df = source_df[source_df["split"] == "train"]
    val_df = source_df[source_df["split"] == "val"]
    test_df = source_df[source_df["split"] == "test"]

    X_train, y_train = train_df[feature_cols], train_df[TARGET_COLUMN]
    X_val, y_val = val_df[feature_cols], val_df[TARGET_COLUMN]
    X_test, y_test = test_df[feature_cols], test_df[TARGET_COLUMN]
    sample_weight_train = train_df["country"].map(country_weights).values

    print(f"    Training XGBoost ({configuration})...")
    xgb_model = XGBRegressor(**XGBOOST_PARAMS)
    xgb_model.fit(X_train, y_train, sample_weight=sample_weight_train)

    print(f"    Training LightGBM ({configuration})...")
    lgbm_model = LGBMRegressor(**LIGHTGBM_PARAMS)
    lgbm_model.fit(X_train, y_train, sample_weight=sample_weight_train)

    print(f"    Training CatBoost ({configuration})...")
    catboost_model = CatBoostRegressor(**CATBOOST_PARAMS)
    catboost_model.fit(X_train, y_train, sample_weight=sample_weight_train)

    base_models = {"xgboost": xgb_model, "lightgbm": lgbm_model, "catboost": catboost_model}
    joblib.dump(xgb_model, models_dir / "xgboost.joblib")
    joblib.dump(lgbm_model, models_dir / "lightgbm.joblib")
    catboost_model.save_model(str(models_dir / "catboost.cbm"))

    # Ensemble weights: computed from UNWEIGHTED validation RMSE, regardless
    # of training configuration -- keeping every part of the pipeline
    # identical except the one variable under test (sample_weight itself).
    val_preds = {name: model.predict(X_val) for name, model in base_models.items()}
    val_rmse = {name: compute_metrics(y_val, p)["RMSE"] for name, p in val_preds.items()}
    inv_rmse = {name: 1.0 / r for name, r in val_rmse.items()}
    total_inv = sum(inv_rmse.values())
    ensemble_weights = {name: v / total_inv for name, v in inv_rmse.items()}
    pd.DataFrame([ensemble_weights]).to_csv(run_dir / "ensemble_weights.csv", index=False)

    def score_all(X, y):
        preds = {name: model.predict(X) for name, model in base_models.items()}
        preds["ensemble"] = sum(ensemble_weights[name] * preds[name] for name in base_models)
        metrics = {name: compute_metrics(y, preds[name]) for name in list(base_models) + ["ensemble"]}
        return preds, metrics

    val_preds_all, val_metrics = score_all(X_val, y_val)
    test_preds_all, test_metrics = score_all(X_test, y_test)

    source_metrics_lines = [f"Source-Domain Val/Test Metrics -- {configuration}", "=" * 50]
    for split_name, metrics in [("VAL", val_metrics), ("TEST", test_metrics)]:
        source_metrics_lines.append(f"\n{split_name}:")
        for name, m in metrics.items():
            source_metrics_lines.append(f"  {name:<10} RMSE={m['RMSE']:.4f}  MAE={m['MAE']:.4f}  R2={m['R2']:.4f}")
    (run_dir / "source_val_test_metrics.txt").write_text("\n".join(source_metrics_lines))

    test_predictions_df = pd.DataFrame({
        IDENTIFIER_COLUMN: test_df[IDENTIFIER_COLUMN].values, "actual_pm25": y_test.values,
        **{f"{name}_pred": test_preds_all[name] for name in list(base_models) + ["ensemble"]},
    })
    test_predictions_df.to_csv(run_dir / "predictions_source_val_test.csv", index=False)

    print(f"    Scoring zero-shot Philippines transfer ({configuration})...")
    X_ph, y_ph = ph_df[feature_cols], ph_df[TARGET_COLUMN]
    ph_preds_all, ph_metrics = score_all(X_ph, y_ph)

    ph_metrics_lines = [f"Philippines Zero-Shot Transfer Metrics -- {configuration}", "=" * 55]
    for name, m in ph_metrics.items():
        ph_metrics_lines.append(f"  {name:<10} RMSE={m['RMSE']:.4f}  MAE={m['MAE']:.4f}  R2={m['R2']:.4f}")
    (run_dir / "philippines_transfer_metrics.txt").write_text("\n".join(ph_metrics_lines))

    ph_predictions_df = pd.DataFrame({
        IDENTIFIER_COLUMN: ph_df[IDENTIFIER_COLUMN].values, "actual_pm25": y_ph.values,
        **{f"{name}_pred": ph_preds_all[name] for name in list(base_models) + ["ensemble"]},
    })
    ph_predictions_df.to_csv(run_dir / "predictions_philippines.csv", index=False)

    training_config = {
        "configuration": configuration, "random_seed": RANDOM_SEED,
        "country_weights": country_weights,
        "xgboost_params": {k: v for k, v in XGBOOST_PARAMS.items()},
        "lightgbm_params": {k: v for k, v in LIGHTGBM_PARAMS.items()},
        "catboost_params": {k: v for k, v in CATBOOST_PARAMS.items()},
    }
    (run_dir / "training_config.json").write_text(json.dumps(training_config, indent=2, default=str))

    return {
        "configuration": configuration,
        "val_ensemble_rmse": val_metrics["ensemble"]["RMSE"], "val_ensemble_mae": val_metrics["ensemble"]["MAE"],
        "val_ensemble_r2": val_metrics["ensemble"]["R2"],
        "test_ensemble_rmse": test_metrics["ensemble"]["RMSE"], "test_ensemble_mae": test_metrics["ensemble"]["MAE"],
        "test_ensemble_r2": test_metrics["ensemble"]["R2"],
        "ph_ensemble_rmse": ph_metrics["ensemble"]["RMSE"], "ph_ensemble_mae": ph_metrics["ensemble"]["MAE"],
        "ph_ensemble_r2": ph_metrics["ensemble"]["R2"],
        "ph_predictions": ph_preds_all["ensemble"],
    }


# ============================================================
# Baseline reproduction checkpoint (for the unweighted configuration)
# ============================================================

def run_baseline_reproduction_check(unweighted_result: dict, ph_df: pd.DataFrame, feature_cols: list) -> list:
    lines = ["Baseline Reproduction Checkpoint (Unweighted Configuration vs. Original Canonical Baseline)",
             "=" * 92, ""]

    original_ph_metrics_path = BASELINE_DIR / "baseline_metrics.txt"
    lines.append("Original canonical baseline metrics (from baseline_model_outputs/baseline_metrics.txt):")
    if original_ph_metrics_path.exists():
        lines.append(original_ph_metrics_path.read_text())
    else:
        lines.append("  ORIGINAL baseline_metrics.txt not found -- cannot verify reproduction numerically.")

    lines.append("\nRetrained UNWEIGHTED configuration's Philippines transfer metrics:")
    lines.append(f"  ensemble RMSE={unweighted_result['ph_ensemble_rmse']:.4f}  "
                  f"MAE={unweighted_result['ph_ensemble_mae']:.4f}  R2={unweighted_result['ph_ensemble_r2']:.4f}")

    print("  Loading original baseline models for prediction-correlation check...")
    original_xgb = joblib.load(BASELINE_DIR / "models" / "xgboost_baseline.joblib")
    original_lgbm = joblib.load(BASELINE_DIR / "models" / "lightgbm_baseline.joblib")
    original_catboost = CatBoostRegressor()
    original_catboost.load_model(str(BASELINE_DIR / "models" / "catboost_baseline.cbm"))
    original_weights = pd.read_csv(BASELINE_DIR / "baseline_ensemble_weights.csv").iloc[0].to_dict()

    X_ph = ph_df[feature_cols]
    original_preds = {"xgboost": original_xgb.predict(X_ph), "lightgbm": original_lgbm.predict(X_ph),
                       "catboost": original_catboost.predict(X_ph)}
    original_ensemble_pred = sum(original_weights[name] * original_preds[name] for name in original_preds)

    corr, _ = pearsonr(original_ensemble_pred, unweighted_result["ph_predictions"])
    lines.append(f"\nPearson correlation between ORIGINAL baseline ensemble predictions and RETRAINED "
                  f"unweighted ensemble predictions (same Philippines rows): {corr:.6f}")

    rel_rmse_diff = None
    original_ph_predictions_path = BASELINE_DIR / "baseline_predictions_philippines.csv"
    if original_ph_predictions_path.exists():
        # Recomputed directly from the original baseline's own saved
        # predictions, rather than text-parsing baseline_metrics.txt --
        # that file contains an "ensemble RMSE=" line under EACH of
        # TRAIN/VAL/TEST/Philippines, and a naive first-match text search
        # silently grabbed the TRAIN section's RMSE instead of the
        # Philippines section's, producing a spurious 35% "difference"
        # where the real difference (confirmed via this direct
        # recomputation) is zero. Reading the actual saved predictions
        # file is unambiguous and avoids this entire class of bug.
        original_ph_preds_df = pd.read_csv(original_ph_predictions_path)
        original_rmse = compute_metrics(original_ph_preds_df["actual_pm25"], original_ph_preds_df["ensemble_pred"])["RMSE"]
        rel_rmse_diff = abs(unweighted_result["ph_ensemble_rmse"] - original_rmse) / original_rmse * 100
        lines.append(f"\nOriginal baseline's Philippines ensemble RMSE (recomputed directly from its saved "
                      f"predictions file, not text-parsed): {original_rmse:.4f}")
        lines.append(f"Relative RMSE difference (retrained unweighted vs. original baseline): {rel_rmse_diff:.2f}%")

    lines.append("\nInterpretation: correlation > 0.99 and relative RMSE difference < 2% would indicate the "
                  "unweighted retrain successfully reproduces the original baseline (residual differences "
                  "attributable to training nondeterminism/library-version effects rather than a real "
                  "implementation discrepancy). Values outside this range warrant investigation BEFORE "
                  "treating the weighting comparison below as valid.")
    if corr > 0.99 and (rel_rmse_diff is None or rel_rmse_diff < 2.0):
        lines.append("\n>>> REPRODUCTION CHECK PASSED.")
    else:
        lines.append("\n>>> REPRODUCTION CHECK FAILED OR INCONCLUSIVE -- investigate before trusting the "
                      "weighting comparison.")

    return lines


# ============================================================
# Main
# ============================================================

def main():
    OUTPUT_ROOT.mkdir(exist_ok=True)

    print("Loading source features and the EXISTING baseline cell split (unchanged, reused as-is)...")
    source_df = pd.read_csv(SOURCE_FEATURES_PATH)
    source_meta = pd.read_csv(SOURCE_METADATA_PATH, usecols=[IDENTIFIER_COLUMN, "country"]).drop_duplicates()
    cell_split = pd.read_csv(BASELINE_DIR / "baseline_cell_split_assignment.csv")
    source_df = source_df.merge(cell_split[[IDENTIFIER_COLUMN, "split"]], on=IDENTIFIER_COLUMN, how="left")
    source_df = source_df.merge(source_meta, on=IDENTIFIER_COLUMN, how="left")
    if source_df["split"].isna().any() or source_df["country"].isna().any():
        raise ValueError("Some source rows are missing a split or country assignment.")

    feature_cols = [c for c in source_df.columns if c not in (IDENTIFIER_COLUMN, TARGET_COLUMN, "split", "country")]

    print("Loading Philippines features (zero-shot transfer evaluation set)...")
    ph_df = pd.read_csv(PH_FEATURES_PATH)

    results = []
    for configuration in CONFIGURATIONS:
        print(f"\n=== Configuration: {configuration} ===")
        result = run_one_configuration(configuration, source_df, cell_split, ph_df, feature_cols)
        results.append(result)

    unweighted_result = next(r for r in results if r["configuration"] == "unweighted")
    print("\nRunning baseline reproduction checkpoint...")
    reproduction_lines = run_baseline_reproduction_check(unweighted_result, ph_df, feature_cols)
    (OUTPUT_ROOT / "baseline_reproduction_check.txt").write_text("\n".join(reproduction_lines))
    print("  Wrote baseline_reproduction_check.txt")

    comparison_rows = []
    baseline_ph_rmse = unweighted_result["ph_ensemble_rmse"]
    for r in results:
        comparison_rows.append({
            "configuration": r["configuration"],
            "source_val_rmse": r["val_ensemble_rmse"], "source_val_mae": r["val_ensemble_mae"], "source_val_r2": r["val_ensemble_r2"],
            "source_test_rmse": r["test_ensemble_rmse"], "source_test_mae": r["test_ensemble_mae"], "source_test_r2": r["test_ensemble_r2"],
            "philippines_rmse": r["ph_ensemble_rmse"], "philippines_mae": r["ph_ensemble_mae"], "philippines_r2": r["ph_ensemble_r2"],
            "philippines_rmse_improvement_vs_unweighted": baseline_ph_rmse - r["ph_ensemble_rmse"],
            "philippines_rmse_pct_improvement_vs_unweighted": (baseline_ph_rmse - r["ph_ensemble_rmse"]) / baseline_ph_rmse * 100,
        })
    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(OUTPUT_ROOT / "reweighting_comparison.csv", index=False)

    report_lines = ["Source-Domain Country Reweighting -- Comparison Report", "=" * 55, ""]
    for _, row in comparison_df.iterrows():
        report_lines.append(f"{row['configuration']}:")
        report_lines.append(f"  Source VAL:  RMSE={row['source_val_rmse']:.4f}  MAE={row['source_val_mae']:.4f}  R2={row['source_val_r2']:.4f}")
        report_lines.append(f"  Source TEST: RMSE={row['source_test_rmse']:.4f}  MAE={row['source_test_mae']:.4f}  R2={row['source_test_r2']:.4f}")
        report_lines.append(f"  PHILIPPINES (zero-shot transfer): RMSE={row['philippines_rmse']:.4f}  "
                              f"MAE={row['philippines_mae']:.4f}  R2={row['philippines_r2']:.4f}")
        report_lines.append(f"  Improvement vs. unweighted: RMSE {row['philippines_rmse_improvement_vs_unweighted']:+.4f} "
                              f"({row['philippines_rmse_pct_improvement_vs_unweighted']:+.2f}%)")
        report_lines.append("")

    (OUTPUT_ROOT / "reweighting_comparison_report.txt").write_text("\n".join(report_lines))
    print(f"\nWrote reweighting_comparison.csv and reweighting_comparison_report.txt")
    print("\nSource-domain reweighting experiment complete.")


if __name__ == "__main__":
    main()
