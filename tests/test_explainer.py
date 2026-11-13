"""
tests/test_explainer.py -- SHAP explainer against the REAL deployed
artifacts (skipped if they or the research pipeline aren't available, as
model_loader requires both): TreeExplainer on the XGBoost member only,
additivity (base + sum(shap) == the member's own prediction), float32 vs
float64 agreement within tolerance, payload shape/sorting, and the
graceful-unavailable path. The prediction pipeline is never invoked.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import MODEL_XGBOOST_PATH, MODEL_FEATURE_SCHEMA_PATH, RESEARCH_PIPELINE_DIR


def _artifacts_available():
    try:
        import shap  # noqa: F401
    except ImportError:
        return False
    return MODEL_XGBOOST_PATH.exists() and MODEL_FEATURE_SCHEMA_PATH.exists() \
        and RESEARCH_PIPELINE_DIR is not None and Path(RESEARCH_PIPELINE_DIR).is_dir()


pytestmark = pytest.mark.skipif(not _artifacts_available(), reason="deployed artifacts / research pipeline / shap not available")


@pytest.fixture(scope="module")
def loaded():
    from backend.prediction.model_loader import load_models
    return load_models()


@pytest.fixture(scope="module")
def explainer(loaded):
    from backend.prediction.explainer import build_explainer
    ex = build_explainer(loaded, log_fn=lambda m: None)
    assert ex is not None
    return ex


def _row(loaded, seed=0):
    rng = np.random.RandomState(seed)
    # Values only need to be finite and plausible-ish; SHAP additivity holds regardless.
    return rng.uniform(0, 1, size=len(loaded["feature_order"])).astype(np.float64) * 10.0


def test_explains_xgboost_member_only(explainer, loaded):
    assert explainer.model is loaded["models"]["xgboost"]
    assert explainer.feature_order == loaded["feature_order"] and len(explainer.feature_order) == 157


def test_additivity_matches_member_prediction(explainer, loaded):
    assert explainer.additivity_ok, f"startup additivity check failed by {explainer.additivity_check_error}"
    for seed in (0, 1, 2):
        x = _row(loaded, seed)
        res = explainer.explain_row(x)
        assert res["shap_log"].shape == (157,)
        assert abs(res["base_value_log"] + res["shap_log"].sum() - res["xgb_pred_log"]) < 1e-4
        # and the member's own prediction is what the sklearn wrapper predicts
        # (i.e. best_iteration-limited, exactly as inference.py's .predict()).
        import pandas as pd
        assert abs(res["xgb_pred_log"] - float(loaded["models"]["xgboost"].predict(
            pd.DataFrame(x.reshape(1, -1), columns=loaded["feature_order"]))[0])) < 1e-6


def test_base_value_equals_xgboost_native_bias_and_shap_matches_native_contribs(explainer, loaded):
    """Independent cross-check against XGBoost's own pred_contribs: the
    derived base value must equal the booster's bias column, and shap's
    per-feature values must equal the native contributions (both limited
    to the same number of trees shap uses)."""
    import pandas as pd
    import xgboost
    x = _row(loaded, seed=11)
    df = pd.DataFrame(x.reshape(1, -1), columns=loaded["feature_order"])
    tree_limit = int(explainer.explainer.model.tree_limit)
    native = loaded["models"]["xgboost"].get_booster().predict(
        xgboost.DMatrix(df), pred_contribs=True, iteration_range=(0, tree_limit))[0]
    assert abs(native[-1] - explainer.base_value_log) < 1e-5
    res = explainer.explain_row(x)
    np.testing.assert_allclose(res["shap_log"], native[:-1], rtol=1e-5, atol=1e-6)
    # shap_reported_expected_value_log is read at explainer construction,
    # before any shap_values() call, so it is the booster's CONFIGURED
    # base_score rather than its settled bias -- off by 0.0959 for the
    # superseded fine-tuned artifact and by 0.0007 for this one. It is the
    # DERIVED base that must satisfy additivity, since that is what the app
    # displays, so that is what is asserted here rather than any
    # artifact-specific agreement or disagreement.
    assert abs(explainer.base_value_log + res["shap_log"].sum() - res["xgb_pred_log"]) < 1e-4
    if abs(explainer.shap_reported_expected_value_log - explainer.base_value_log) > 1e-3:
        # If they disagree, expected_value must NOT satisfy additivity -- i.e.
        # the derivation is doing real work rather than masking a bug.
        assert abs(explainer.shap_reported_expected_value_log + res["shap_log"].sum()
                   - res["xgb_pred_log"]) > 1e-4


def test_float32_and_float64_rows_agree_within_tolerance(explainer, loaded):
    x64 = _row(loaded, seed=3)
    r64 = explainer.explain_row(x64)
    r32 = explainer.explain_row(x64.astype(np.float32))
    np.testing.assert_allclose(r32["shap_log"], r64["shap_log"], rtol=1e-4, atol=1e-5)
    assert abs(r32["xgb_pred_log"] - r64["xgb_pred_log"]) < 1e-5


def test_payload_sorted_by_abs_shap_with_labels_and_real_space(explainer, loaded):
    from backend.prediction.explainer import build_explanation_payload, ATTRIBUTION_NOTE
    from backend.prediction.feature_labels import display_name
    x = _row(loaded, seed=5)
    x[7] = np.nan  # one missing feature: XGBoost handles NaN natively
    p = build_explanation_payload(explainer, x, loaded["feature_order"], display_name, loaded["log_clip_bounds"],
                                  "PHX", "2026-08-24T22:00:00+00:00", ensemble_prediction_ugm3=12.3)
    assert p["n_features"] == 157 and len(p["contributions"]) == 157
    mags = [abs(c["shap_log"]) for c in p["contributions"]]
    assert mags == sorted(mags, reverse=True)
    assert p["n_missing_features"] == 1
    missing = [c for c in p["contributions"] if c["feature"] == loaded["feature_order"][7]][0]
    assert missing["value"] is None
    assert all(c["display_name"] != c["feature"] for c in p["contributions"])
    assert p["base_value_ugm3"] == pytest.approx(np.expm1(p["base_value_log"]))
    assert p["xgboost_prediction_ugm3"] == pytest.approx(np.expm1(np.clip(p["xgboost_prediction_log"], *loaded["log_clip_bounds"])))
    assert p["ensemble_prediction_ugm3"] == 12.3
    assert "not a causal" in p["attribution_note"] and p["attribution_note"] == ATTRIBUTION_NOTE
    assert "XGBoost" in p["model_explained"] and p["output_space"] == "log1p(pm25)"
    # top-5 positive / top-5 negative selection (what the UI does) is well-defined
    pos = [c for c in p["contributions"] if c["shap_log"] > 0][:5]
    neg = [c for c in p["contributions"] if c["shap_log"] < 0][:5]
    assert len(pos) <= 5 and len(neg) <= 5


def test_build_explainer_never_raises_when_model_missing():
    from backend.prediction.explainer import build_explainer
    assert build_explainer({"models": {}, "feature_order": []}, log_fn=lambda m: None) is None


def test_store_row_explains_end_to_end(explainer, loaded):
    """Retained float32 row -> explainer, the exact path the endpoint uses."""
    import pandas as pd
    import config.settings as settings
    from backend.prediction import explanation_store as es
    es._reset_for_tests(); settings.SHAP_FEATURE_RETENTION_MAX_MB = 64
    fo = loaded["feature_order"]
    df = pd.DataFrame([{"grid_id": "PHX", "datetime_utc": pd.Timestamp("2026-08-24 22:00:00+00:00"), **dict(zip(fo, _row(loaded, 9)))}])
    assert es.attach("jobx", df, fo)["retained"]
    row, order = es.get_row("jobx", "PHX", "2026-08-24 22:00:00+00:00")
    res = explainer.explain_row(row)
    assert abs(res["base_value_log"] + res["shap_log"].sum() - res["xgb_pred_log"]) < 1e-4
    es._reset_for_tests()
