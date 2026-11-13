"""
backend/prediction/explainer.py
============================================================================
Per-row SHAP explanations, following the thesis's own SHAP methodology
exactly (final_integrated_evaluation.py::run_shap_full and the other
research scripts): shap.TreeExplainer on the Philippine-only XGBoost member
ONLY -- not an ensemble-wide explainer -- with values in the model's own
output space (log1p(pm25)). The thesis reported mean |SHAP| for global
importance; individual explanations here use the SIGNED per-feature
values for one row, which is the same explainer applied to one input.

What this is and is not:
  - It explains the deployed Philippine-only XGBoost artifact's prediction for
    one feature row: positive = pushed THAT model's prediction higher,
    negative = lower, relative to the explainer's base value. It is model
    attribution, not a causal statement about atmospheric PM2.5.
  - The displayed map value is the three-model ensemble; the explained
    member's own prediction is reported alongside so the two are never
    conflated.
  - The explained artifact is the SAME XGBoost member the thesis's
    Philippine-only SHAP analysis used (philippine_only_full_evaluation.py
    saves its Stage A models and runs Stage B's SHAP on those same
    in-memory models). The thesis reports global mean |SHAP| over a
    sample of holdout rows; this panel shows the signed values for one
    cell-hour, so the two agree in model but not in aggregation.

The explainer is built ONCE from the already-loaded model (never refit,
never per request). If shap is unavailable or construction fails, the
app runs exactly as before -- only explanations are unavailable.
Nothing here is used by inference.py; the prediction path is untouched.
"""

import numpy as np
import pandas as pd

EXPLAINED_MODEL_KEY = "xgboost"
EXPLAINED_MODEL_LABEL = "ph_only_xgboost (deployed Philippine-only XGBoost member)"
OUTPUT_SPACE = "log1p(pm25)"

ATTRIBUTION_NOTE = (
    "Signed SHAP values attribute the deployed Philippine-only XGBoost member's prediction (in log1p "
    "space) to its input features: a positive value means the feature pushed the model prediction "
    "higher, a negative value means it pushed it lower, relative to the model's base value. This is "
    "model attribution, not a causal statement about atmospheric PM2.5. The displayed map value is "
    "the three-model ensemble; only the XGBoost member is explained, matching the thesis "
    "methodology. The explained artifact is the same one the thesis's Philippine-only SHAP analysis "
    "used; the thesis reports global mean |SHAP| over many rows, while this shows the signed values "
    "for one cell-hour."
)


ADDITIVITY_TOLERANCE = 1e-4


class Explainer:
    def __init__(self, xgb_model, feature_order):
        import shap  # imported here so a missing shap only disables explanations
        self.model = xgb_model
        self.feature_order = list(feature_order)
        self.explainer = shap.TreeExplainer(xgb_model)
        self.shap_reported_expected_value_log = float(np.ravel(self.explainer.expected_value)[0])

        # BASE VALUE -- derived from the model, not taken from
        # explainer.expected_value. The identity this app displays is
        # base + sum(shap) == model.predict(), so the base is read off that
        # identity on one probe row and then verified on a second,
        # different probe row.
        #
        # Why it is derived rather than read from shap: immediately after
        # construction, explainer.expected_value still reports the booster's
        # CONFIGURED base_score, not its actual bias term; shap only settles
        # it on the first shap_values() call. Measured on both artifacts
        # (shap 0.52.0 / xgboost 3.4.0), expected_value at construction vs
        # after the first call vs the derived base:
        #     Philippine-only (deployed)  2.821584 -> 2.822253 | 2.822252
        #     fine-tuned (superseded)     3.257171 -> 3.161252 | 3.161253
        # So the 0.0959 gap previously documented here was a read-ordering
        # effect, not a property of fine-tuning; it is simply much smaller
        # for this artifact. Deriving the base from the additivity identity
        # is exact regardless of call ordering or which artifact is loaded,
        # which is why it is kept. (shap_reported_expected_value_log below
        # is read at construction, so it is deliberately that pre-settle
        # number, carried in the API payload for diagnostics only.)
        #
        # The per-feature SHAP values from TreeExplainer.shap_values() match
        # XGBoost's own booster.predict(pred_contribs=True) exactly for both
        # artifacts, so the thesis's mean |SHAP| analysis is unaffected by
        # any of this.
        n = len(self.feature_order)
        probe0 = pd.DataFrame(np.zeros((1, n), dtype=np.float32), columns=self.feature_order)
        probe1 = pd.DataFrame(np.full((1, n), 7.0, dtype=np.float32), columns=self.feature_order)
        self.base_value_log = float(np.ravel(self.model.predict(probe0))[0] - np.asarray(self.explainer.shap_values(probe0)).sum())
        check = abs(float(np.ravel(self.model.predict(probe1))[0]) - (self.base_value_log + float(np.asarray(self.explainer.shap_values(probe1)).sum())))
        self.additivity_ok = bool(check < ADDITIVITY_TOLERANCE)
        self.additivity_check_error = float(check)

    def explain_row(self, x_row):
        """x_row: 1-D array of len(feature_order) (float32 or float64).
        Returns dict with base_value_log, shap_log (1-D float array in
        feature_order), xgb_pred_log (from the model itself), and
        n_missing_features."""
        x = np.asarray(x_row, dtype=np.float32).reshape(1, -1)
        if x.shape[1] != len(self.feature_order):
            raise ValueError(f"row has {x.shape[1]} features; explainer expects {len(self.feature_order)}")
        df = pd.DataFrame(x, columns=self.feature_order)
        shap_vals = np.asarray(self.explainer.shap_values(df)).reshape(-1)
        xgb_pred_log = float(np.ravel(self.model.predict(df))[0])
        return {
            "base_value_log": self.base_value_log,
            "shap_log": shap_vals.astype(float),
            "xgb_pred_log": xgb_pred_log,
            "n_missing_features": int(np.isnan(x).sum()),
        }


def build_explainer(loaded_model, log_fn=print):
    """Build once at startup from the loaded model dict (model_loader.py's
    return value). Returns an Explainer or None; never raises."""
    try:
        xgb_model = loaded_model["models"][EXPLAINED_MODEL_KEY]
        ex = Explainer(xgb_model, loaded_model["feature_order"])
        log_fn(f"  OK  SHAP TreeExplainer built on the Philippine-only XGBoost member "
               f"(base value {ex.base_value_log:.4f} in log1p space = {np.expm1(ex.base_value_log):.2f} ug/m3; "
               f"shap-reported expected_value {ex.shap_reported_expected_value_log:.4f}; "
               f"additivity check error {ex.additivity_check_error:.2e}, "
               f"{'OK' if ex.additivity_ok else 'FAILED -- explanations will be flagged'}).")
        return ex
    except Exception as e:  # noqa: BLE001 -- explanations are optional; prediction must not depend on this
        log_fn(f"  Note: SHAP explainer unavailable ({type(e).__name__}: {e}). Predictions are unaffected; "
               f"/api/prediction_explanation will report 503.")
        return None


def to_real(pred_log, log_clip_bounds):
    """Same clip-then-expm1 inverse transform inference.py applies per
    model, reused here only for DISPLAY of the explained member's own
    prediction and of the base value (never fed back into anything)."""
    return float(np.expm1(np.clip(pred_log, *log_clip_bounds)))


def build_explanation_payload(explainer, x_row, feature_order, feature_labels_fn, log_clip_bounds,
                              grid_id, datetime_utc, ensemble_prediction_ugm3=None):
    """Assemble the JSON-ready explanation for one row, with every
    feature's contribution sorted by |shap| (the frontend picks its
    top-N positive/negative)."""
    res = explainer.explain_row(x_row)
    shap_log = res["shap_log"]
    order = np.argsort(-np.abs(shap_log))
    contributions = []
    for i in order:
        name = feature_order[i]
        raw = float(x_row[i])
        contributions.append({
            "feature": name,
            "display_name": feature_labels_fn(name),
            "value": None if np.isnan(raw) else raw,
            "shap_log": float(shap_log[i]),
        })
    return {
        "grid_id": grid_id,
        "datetime_utc": datetime_utc,
        "model_explained": EXPLAINED_MODEL_LABEL,
        "output_space": OUTPUT_SPACE,
        "base_value_log": res["base_value_log"],
        "base_value_ugm3": to_real(res["base_value_log"], log_clip_bounds),
        "xgboost_prediction_log": res["xgb_pred_log"],
        "xgboost_prediction_ugm3": to_real(res["xgb_pred_log"], log_clip_bounds),
        "ensemble_prediction_ugm3": ensemble_prediction_ugm3,
        "n_missing_features": res["n_missing_features"],
        "n_features": len(feature_order),
        "contributions": contributions,
        "additivity_ok": bool(explainer.additivity_ok),
        "shap_reported_expected_value_log": explainer.shap_reported_expected_value_log,
        "attribution_note": ATTRIBUTION_NOTE,
    }
