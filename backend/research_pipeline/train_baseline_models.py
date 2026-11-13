"""
Baseline Model Development Pipeline (Chapter 4 Methodology)
=================================================================

Implements the baseline ensemble transfer learning model exactly as
specified in the thesis methodology (Section 4.3.3) -- no experimental
modifications (country reweighting, maritime-only subsets, hyperparameter
tuning, feature selection, or new engineered features are deliberately
NOT included here; those are later, separate comparative experiments).

What this script does, in order
------------------------------------
1. Loads source_features.csv (pooled source-domain training data) and
   philippines_features.csv (held-out target-domain evaluation data),
   with a minimal read of each metadata file for country stratification.
2. Splits SOURCE grid cells (not rows) into train/validation/test,
   stratified by country. An entire cell's full row history moves to
   whichever split it's assigned to -- this satisfies the methodology's
   "hold out entire monitoring stations and contiguous time periods"
   validation design in one step: holding out entire cells IS holding out
   entire stations, and since a cell's own hourly history is never
   fragmented across splits, its time series stays contiguous within
   whichever split it lands in, rather than being randomly interspersed.
3. Trains three base learners (XGBoost, LightGBM, CatBoost) on the
   pooled source-domain training split, using fixed, reasonable,
   UNTUNED hyperparameters -- per the thesis's explicit exclusion of
   hyperparameter optimization from this baseline stage.
4. Combines the three base learners into an ensemble via a
   performance-weighted linear combination, with weights set from
   validation RMSE (inverse-RMSE weighting) -- the specific combination
   method the methodology itself cites from Haseeb et al. (2026), not an
   addition made here.
5. Evaluates every base learner and the ensemble at two levels, exactly
   matching the thesis's two-level evaluation design: source-domain
   train/validation/test, and a separate, complete evaluation against the
   full Philippine target-domain file (never pooled into training).
6. Saves predictions, metrics, the cell-split assignment, and the trained
   model artifacts themselves, so later comparative experiments (country
   reweighting, maritime-only baselines, etc.) can be evaluated against
   this exact baseline without needing to retrain it from scratch.

Preprocessing choices, and why nothing more is applied
------------------------------------------------------------
No feature scaling: all three base learners are tree-based gradient
boosting models, which split on raw feature values and are invariant to
monotonic transformations -- scaling would be an unnecessary
transformation with no effect on these specific algorithms.

No imputation of missing values: the thesis's own missing-value handling
design (Section 4.3.2) explicitly retains missing values (after the
local/neighborhood fallback already applied in prepare_features.py) "with
an explicit missing-value indicator, allowing the ensemble learning
framework... to exploit the native capacity of tree-based learners to
handle missing feature values." All three libraries used here handle raw
NaN natively. Imputing now would directly contradict this design.

No log-transformation of the target: the methodology's written text does
not specify a log-transformed target anywhere. A log1p transform has a
real statistical justification (confirmed during EDA -- it substantially
reduces the target's skewness), but was never written into the
methodology itself, and introducing it now would be exactly the kind of
undocumented modification this baseline is meant to avoid. Trained here
on raw pm25.

Usage
-----
Place this script alongside source_features.csv, source_metadata.csv,
philippines_features.csv, and philippines_metadata.csv, and run it.

Dependencies: pandas, numpy, scikit-learn, xgboost, lightgbm, catboost, joblib
    pip install pandas numpy scikit-learn xgboost lightgbm catboost joblib
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

# ============================================================
# Configuration -- MACROS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent

SOURCE_FEATURES_PATH = SCRIPT_DIR / "source_features.csv"
SOURCE_METADATA_PATH = SCRIPT_DIR / "source_metadata.csv"
PHILIPPINES_FEATURES_PATH = SCRIPT_DIR / "philippines_features.csv"

OUTPUT_DIR = SCRIPT_DIR / "baseline_model_outputs"
MODELS_DIR = OUTPUT_DIR / "models"

TARGET_COLUMN = "pm25"
IDENTIFIER_COLUMN = "grid_id"  # kept for output/tracking only, NEVER a model feature

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
# TEST_FRAC is whatever remains (~0.15)
RANDOM_SEED = 42

# Fixed, reasonable, UNTUNED hyperparameters -- no grid/random search is
# performed here, per the thesis's explicit exclusion of hyperparameter
# optimization from this baseline. These are standard, moderate defaults
# for tabular gradient boosting at this data scale (millions of rows),
# not a tuned configuration.
XGBOOST_PARAMS = dict(n_estimators=300, max_depth=6, learning_rate=0.05,
                       subsample=0.8, colsample_bytree=0.8,
                       tree_method="hist", n_jobs=-1, random_state=RANDOM_SEED)
LIGHTGBM_PARAMS = dict(n_estimators=300, max_depth=-1, num_leaves=63, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8,
                        n_jobs=-1, random_state=RANDOM_SEED, verbose=-1)
CATBOOST_PARAMS = dict(iterations=300, depth=6, learning_rate=0.05,
                        random_seed=RANDOM_SEED, verbose=False, thread_count=-1)


# ============================================================
# Stage 1: Grid-cell-blocked, country-stratified train/val/test split
# ============================================================

def split_cells_by_country(cell_country: pd.DataFrame, train_frac: float, val_frac: float,
                            seed: int) -> pd.DataFrame:
    """Assigns each grid cell (not row) to train/validation/test, done
    separately within each country so small countries (e.g. Malaysia,
    Cambodia at 8 cells) are represented in every split rather than
    risking landing entirely in one split by chance."""
    rng = np.random.RandomState(seed)
    assignments = []
    for country, group in cell_country.groupby("country"):
        cells = group[IDENTIFIER_COLUMN].tolist()
        rng.shuffle(cells)
        n = len(cells)
        n_train = max(1, round(n * train_frac))
        n_train = min(n_train, n)
        remaining = n - n_train
        n_val = min(max(1, round(n * val_frac)) if remaining >= 2 else remaining, remaining)
        n_test = remaining - n_val

        for c in cells[:n_train]:
            assignments.append({IDENTIFIER_COLUMN: c, "country": country, "split": "train"})
        for c in cells[n_train:n_train + n_val]:
            assignments.append({IDENTIFIER_COLUMN: c, "country": country, "split": "val"})
        for c in cells[n_train + n_val:]:
            assignments.append({IDENTIFIER_COLUMN: c, "country": country, "split": "test"})

    return pd.DataFrame(assignments)


# ============================================================
# Stage 2: Metrics
# ============================================================

def compute_metrics(y_true, y_pred) -> dict:
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def format_metrics_line(label: str, metrics: dict) -> str:
    return f"  {label:<45} RMSE={metrics['RMSE']:.4f}   MAE={metrics['MAE']:.4f}   R2={metrics['R2']:.4f}"


# ============================================================
# Main
# ============================================================

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
    report_lines = ["Baseline Model Development -- Evaluation Report", "=" * 48, ""]

    # --- Load data ---
    print("Loading source metadata (grid_id, country only) for cell stratification...")
    source_meta_small = pd.read_csv(SOURCE_METADATA_PATH, usecols=[IDENTIFIER_COLUMN, "country"])
    cell_country = source_meta_small.drop_duplicates(subset=[IDENTIFIER_COLUMN])
    del source_meta_small

    print("Splitting source grid cells into train/validation/test (stratified by country)...")
    cell_split = split_cells_by_country(cell_country, TRAIN_FRAC, VAL_FRAC, RANDOM_SEED)
    cell_split_path = OUTPUT_DIR / "baseline_cell_split_assignment.csv"
    cell_split.to_csv(cell_split_path, index=False)
    print(f"  Wrote {cell_split_path.name} ({len(cell_split)} cells: "
          f"{(cell_split['split'] == 'train').sum()} train, "
          f"{(cell_split['split'] == 'val').sum()} val, "
          f"{(cell_split['split'] == 'test').sum()} test)")

    print("Loading full source feature set (this is the large read)...")
    source_df = pd.read_csv(SOURCE_FEATURES_PATH)
    source_df = source_df.merge(cell_split[[IDENTIFIER_COLUMN, "split"]], on=IDENTIFIER_COLUMN, how="left")
    if source_df["split"].isna().any():
        raise ValueError("Some source rows did not receive a split assignment -- "
                          "grid_id mismatch between features and metadata files.")

    feature_cols = [c for c in source_df.columns if c not in (IDENTIFIER_COLUMN, TARGET_COLUMN, "split")]
    print(f"  {len(source_df)} rows, {len(feature_cols)} feature columns "
          f"(excluding {IDENTIFIER_COLUMN} and {TARGET_COLUMN}).")

    train_df = source_df[source_df["split"] == "train"]
    val_df = source_df[source_df["split"] == "val"]
    test_df = source_df[source_df["split"] == "test"]

    X_train, y_train = train_df[feature_cols], train_df[TARGET_COLUMN]
    X_val, y_val = val_df[feature_cols], val_df[TARGET_COLUMN]
    X_test, y_test = test_df[feature_cols], test_df[TARGET_COLUMN]
    del source_df

    print(f"  Train: {len(X_train)} rows, Val: {len(X_val)} rows, Test: {len(X_test)} rows.")

    # --- Train base learners ---
    print("\nTraining XGBoost...")
    xgb_model = XGBRegressor(**XGBOOST_PARAMS)
    xgb_model.fit(X_train, y_train)

    print("Training LightGBM...")
    lgbm_model = LGBMRegressor(**LIGHTGBM_PARAMS)
    lgbm_model.fit(X_train, y_train)

    print("Training CatBoost...")
    catboost_model = CatBoostRegressor(**CATBOOST_PARAMS)
    catboost_model.fit(X_train, y_train)

    base_models = {"xgboost": xgb_model, "lightgbm": lgbm_model, "catboost": catboost_model}

    # --- Save model artifacts for reuse by later comparative experiments ---
    print("\nSaving trained model artifacts...")
    joblib.dump(xgb_model, MODELS_DIR / "xgboost_baseline.joblib")
    joblib.dump(lgbm_model, MODELS_DIR / "lightgbm_baseline.joblib")
    catboost_model.save_model(str(MODELS_DIR / "catboost_baseline.cbm"))

    # --- Predictions on every split ---
    print("Generating predictions on train/val/test...")
    preds = {"train": {}, "val": {}, "test": {}}
    for name, model in base_models.items():
        preds["train"][name] = model.predict(X_train)
        preds["val"][name] = model.predict(X_val)
        preds["test"][name] = model.predict(X_test)

    # --- Ensemble weights from validation RMSE (inverse-RMSE weighting) ---
    val_rmse = {name: compute_metrics(y_val, p)["RMSE"] for name, p in preds["val"].items()}
    inv_rmse = {name: 1.0 / r for name, r in val_rmse.items()}
    total_inv = sum(inv_rmse.values())
    ensemble_weights = {name: v / total_inv for name, v in inv_rmse.items()}
    print(f"\nEnsemble weights (inverse validation-RMSE, normalized): "
          f"{ {k: round(v, 4) for k, v in ensemble_weights.items()} }")

    weights_path = OUTPUT_DIR / "baseline_ensemble_weights.csv"
    pd.DataFrame([ensemble_weights]).to_csv(weights_path, index=False)

    def ensemble_predict(split_preds: dict) -> np.ndarray:
        return sum(ensemble_weights[name] * split_preds[name] for name in base_models)

    for split in preds:
        preds[split]["ensemble"] = ensemble_predict(preds[split])

    # --- Metrics: source domain (train/val/test) ---
    report_lines.append("1. Source-Domain Evaluation (train/validation/test, grid-cell-blocked, "
                          "country-stratified split)")
    report_lines.append("-" * 90)
    y_by_split = {"train": y_train, "val": y_val, "test": y_test}
    all_metrics = {}
    for split, y_true in y_by_split.items():
        report_lines.append(f"\n{split.upper()}:")
        for name in list(base_models.keys()) + ["ensemble"]:
            m = compute_metrics(y_true, preds[split][name])
            all_metrics[(split, name)] = m
            report_lines.append(format_metrics_line(name, m))

    # --- Philippines target-domain evaluation ---
    print("\nLoading Philippines feature set for target-domain evaluation...")
    ph_df = pd.read_csv(PHILIPPINES_FEATURES_PATH)
    ph_feature_cols = [c for c in ph_df.columns if c not in (IDENTIFIER_COLUMN, TARGET_COLUMN)]

    missing_in_ph = set(feature_cols) - set(ph_feature_cols)
    missing_in_source = set(ph_feature_cols) - set(feature_cols)
    if missing_in_ph or missing_in_source:
        raise ValueError(f"Source and Philippines feature columns do not match -- "
                          f"missing in PH: {missing_in_ph}, missing in source: {missing_in_source}. "
                          f"The one-hot land cover encoding scheme is specifically designed to produce "
                          f"identical columns across both files; this mismatch means something upstream "
                          f"changed.")

    X_ph = ph_df[feature_cols]  # explicit column order matching training
    y_ph = ph_df[TARGET_COLUMN]

    print("Generating Philippines predictions...")
    ph_preds = {name: model.predict(X_ph) for name, model in base_models.items()}
    ph_preds["ensemble"] = ensemble_predict(ph_preds)

    report_lines.append("\n\n2. Target-Domain Evaluation (full Philippines file, never pooled into training)")
    report_lines.append("-" * 90)
    for name in list(base_models.keys()) + ["ensemble"]:
        m = compute_metrics(y_ph, ph_preds[name])
        all_metrics[("philippines", name)] = m
        report_lines.append(format_metrics_line(name, m))

    metrics_path = OUTPUT_DIR / "baseline_metrics.txt"
    metrics_path.write_text("\n".join(report_lines))
    print(f"\nWrote {metrics_path.name}")

    # --- Save predictions ---
    print("Saving predictions...")
    source_pred_rows = []
    for split, df_split, y_true in [("train", train_df, y_train), ("val", val_df, y_val), ("test", test_df, y_test)]:
        out = pd.DataFrame({
            IDENTIFIER_COLUMN: df_split[IDENTIFIER_COLUMN].values,
            "split": split,
            "actual_pm25": y_true.values,
            "xgboost_pred": preds[split]["xgboost"],
            "lightgbm_pred": preds[split]["lightgbm"],
            "catboost_pred": preds[split]["catboost"],
            "ensemble_pred": preds[split]["ensemble"],
        })
        source_pred_rows.append(out)
    source_predictions = pd.concat(source_pred_rows, ignore_index=True)
    source_predictions.to_csv(OUTPUT_DIR / "baseline_predictions_source.csv", index=False)

    ph_predictions = pd.DataFrame({
        IDENTIFIER_COLUMN: ph_df[IDENTIFIER_COLUMN].values,
        "actual_pm25": y_ph.values,
        "xgboost_pred": ph_preds["xgboost"],
        "lightgbm_pred": ph_preds["lightgbm"],
        "catboost_pred": ph_preds["catboost"],
        "ensemble_pred": ph_preds["ensemble"],
    })
    ph_predictions.to_csv(OUTPUT_DIR / "baseline_predictions_philippines.csv", index=False)

    print(f"\nDone. All outputs written to {OUTPUT_DIR}/")
    print(f"  - {cell_split_path.name}")
    print(f"  - baseline_ensemble_weights.csv")
    print(f"  - baseline_metrics.txt")
    print(f"  - baseline_predictions_source.csv")
    print(f"  - baseline_predictions_philippines.csv")
    print(f"  - models/xgboost_baseline.joblib, models/lightgbm_baseline.joblib, models/catboost_baseline.cbm")


if __name__ == "__main__":
    main()
