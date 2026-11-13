"""
backend/prediction/inference.py
============================================================================
Runs the three deployed models on an already-assembled, schema-validated
feature matrix, applies the inverse-RMSE ensemble weights, and applies the
log-space inverse transform. NEVER trains, fits, or modifies any model --
this module only ever calls .predict().
"""

import numpy as np


def predict(feature_matrix, loaded_model, log_fn=print):
    """feature_matrix: DataFrame with grid_id, datetime_utc, and exactly
    loaded_model['feature_order']'s columns, in that order (as produced
    by feature_builder.build_feature_matrix).

    Returns feature_matrix's grid_id/datetime_utc columns plus a new
    'pm25_prediction' column.
    """
    feature_order = loaded_model["feature_order"]
    models = loaded_model["models"]
    weights = loaded_model["ensemble_weights"]
    log_clip_bounds = loaded_model["log_clip_bounds"]

    X = feature_matrix[feature_order]

    if X.isna().any().any():
        n_bad = int(X.isna().any(axis=1).sum())
        log_fn(f"  WARNING: {n_bad} row(s) have at least one missing feature value. "
               f"Predictions for these rows may be unreliable or the underlying model "
               f"library may raise its own error -- this is surfaced here rather than "
               f"silently imputed.")

    log_fn(f"  Running XGBoost on {len(X)} row(s)...")
    xgb_pred_log = models["xgboost"].predict(X)
    log_fn(f"  Running LightGBM on {len(X)} row(s)...")
    lgbm_pred_log = models["lightgbm"].predict(X)
    log_fn(f"  Running CatBoost on {len(X)} row(s)...")
    catboost_pred_log = models["catboost"].predict(X)

    # Log-space inverse transform -- confirmed exact from
    # run_logspace_pipeline_experiment.py's own to_real(): np.expm1 with
    # clipping at LOG_CLIP_BOUNDS BEFORE the inverse transform, applied
    # per-model, matching predict_ensemble_real()'s own order of
    # operations (clip-then-expm1 per model, THEN weighted-sum the
    # already-real-space predictions -- not sum-then-transform).
    def to_real(preds_log):
        return np.expm1(np.clip(preds_log, *log_clip_bounds))

    xgb_pred = to_real(xgb_pred_log)
    lgbm_pred = to_real(lgbm_pred_log)
    catboost_pred = to_real(catboost_pred_log)

    ensemble_pred = (
        weights["xgboost"] * xgb_pred +
        weights["lightgbm"] * lgbm_pred +
        weights["catboost"] * catboost_pred
    )

    n_negative = int((ensemble_pred < 0).sum())
    n_nan = int(np.isnan(ensemble_pred).sum())
    if n_negative:
        log_fn(f"  WARNING: {n_negative} negative prediction(s) -- should not happen given "
               f"expm1's structural non-negativity; investigate if this occurs.")
    if n_nan:
        log_fn(f"  WARNING: {n_nan} NaN prediction(s) -- likely caused by missing feature "
               f"values reaching the model; see the missing-value warning above.")

    result = feature_matrix[["grid_id", "datetime_utc"]].copy()
    result["pm25_prediction"] = ensemble_pred
    log_fn(f"  Inference complete: {len(result)} prediction(s). "
           f"Range: [{np.nanmin(ensemble_pred):.2f}, {np.nanmax(ensemble_pred):.2f}] ug/m3.")
    return result
