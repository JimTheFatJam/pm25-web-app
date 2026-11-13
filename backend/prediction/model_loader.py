"""
backend/prediction/model_loader.py
============================================================================
Loads the real Baseline + Group D deployment artifacts (currently the
Philippine-only, no-transfer ensemble) ONCE at Flask startup. Never trains, fine-tunes, or modifies any model. Fails
loudly and specifically if anything is missing or inconsistent, rather
than silently starting with a partial/mismatched setup.

Also responsible for making the research pipeline's own validated
feature-construction functions importable (config.settings.RESEARCH_PIPELINE_DIR),
since feature_builder.py calls those functions directly rather than
reimplementing them -- see feature_builder.py's own module docstring for
why.
"""

import sys
import json
from pathlib import Path

import joblib

from config.settings import (
    MODEL_DIR, MODEL_XGBOOST_PATH, MODEL_LIGHTGBM_PATH, MODEL_CATBOOST_PATH,
    MODEL_ENSEMBLE_CONFIG_PATH, MODEL_FEATURE_SCHEMA_PATH, MODEL_PREPROCESSING_CONFIG_PATH,
    MODEL_LANDCOVER_SCHEME_PATH, EXPECTED_FEATURE_COUNT, EXPECTED_BASELINE_FEATURE_COUNT,
    EXPECTED_GROUP_D_FEATURE_COUNT, RESEARCH_PIPELINE_DIR,
)


class ModelLoadError(Exception):
    """Raised when the model artifacts are missing, inconsistent, or the
    research pipeline isn't importable. Always carries a specific,
    actionable message -- never a generic failure."""


REQUIRED_ARTIFACT_FILES = {
    "xgboost model": MODEL_XGBOOST_PATH,
    "lightgbm model": MODEL_LIGHTGBM_PATH,
    "catboost model": MODEL_CATBOOST_PATH,
    "ensemble_config.json": MODEL_ENSEMBLE_CONFIG_PATH,
    "feature_schema.json": MODEL_FEATURE_SCHEMA_PATH,
    "preprocessing_config.json": MODEL_PREPROCESSING_CONFIG_PATH,
}

# Required for feature construction (see feature_builder.py) -- these are
# CODE modules (pure dataframe transforms), not data, imported directly so
# the backend reuses the exact validated logic rather than reimplementing
# it. Confirmed by direct inspection of the research pipeline during the
# deployment-planning stage -- see the deployment plan, Sections 1 and 10.
REQUIRED_RESEARCH_MODULES = [
    "prepare_features",                       # add_hourly_rolling_features, add_daily_rolling_features,
                                               # one_hot_encode_landcover, add_cyclical_time_features
    "run_physics_transport_experiment",       # add_physics_features, add_wind_transport_features,
                                               # build_cell_adjacency, PHYSICS_FEATURE_NAMES, TRANSPORT_FEATURE_NAMES
    "build_additional_features",              # build_calendar_features, compute_consecutive_dry_days
    "advanced_feature_engineering_experiment",  # build_group_d, GROUP_D_FEATURES, TENDENCY_VARIABLES
    "extract_era5_hourly_batched",            # derive_additional_features (relative humidity, wind speed)
]


def _check_research_pipeline_importable():
    if RESEARCH_PIPELINE_DIR is None:
        raise ModelLoadError(
            "config.settings.RESEARCH_PIPELINE_DIR is not set. This must point at the "
            "directory containing your research pipeline scripts (prepare_features.py, "
            "run_physics_transport_experiment.py, build_additional_features.py, "
            "advanced_feature_engineering_experiment.py, extract_era5_hourly_batched.py) -- "
            "the same directory the model-training scripts were run from. The backend "
            "reuses those modules' own validated feature-construction logic directly; it "
            "does not reimplement it. Set RESEARCH_PIPELINE_DIR in config/settings.py and "
            "restart."
        )

    research_dir = Path(RESEARCH_PIPELINE_DIR)
    if not research_dir.is_dir():
        raise ModelLoadError(f"RESEARCH_PIPELINE_DIR is set to '{research_dir}', but that directory does not exist.")

    if str(research_dir) not in sys.path:
        sys.path.insert(0, str(research_dir))

    missing = []
    for mod_name in REQUIRED_RESEARCH_MODULES:
        if not (research_dir / f"{mod_name}.py").exists():
            missing.append(f"{mod_name}.py")
    if missing:
        raise ModelLoadError(
            f"RESEARCH_PIPELINE_DIR ('{research_dir}') is missing required research pipeline "
            f"module(s): {missing}. These must be the actual research scripts, not copies or "
            f"stubs -- feature construction directly calls their functions."
        )


def _check_artifact_files_exist():
    missing = [f"{label} ({path})" for label, path in REQUIRED_ARTIFACT_FILES.items() if not path.exists()]
    if missing:
        raise ModelLoadError(
            "Missing required model artifact file(s) in "
            f"{MODEL_DIR}:\n  - " + "\n  - ".join(missing) +
            "\n\nProduce them in your research environment (philippine_only_full_evaluation.py, "
            "Stage A) and copy them into this directory. See the README.txt beside the artifacts."
        )


def load_models():
    """Call once, at Flask startup. Returns a dict with the loaded models,
    ensemble weights, feature schema, and preprocessing config. Raises
    ModelLoadError with a specific message on any failure -- callers
    should let this propagate and stop the app from starting, not catch
    and continue."""
    print("=" * 70)
    print("Loading Baseline + Group D deployment model (Philippine-only ensemble)...")
    print("=" * 70)

    _check_research_pipeline_importable()
    print(f"  OK  Research pipeline importable from: {RESEARCH_PIPELINE_DIR}")

    _check_artifact_files_exist()
    print(f"  OK  All required artifact files present in: {MODEL_DIR}")

    feature_schema = json.loads(MODEL_FEATURE_SCHEMA_PATH.read_text())
    ensemble_config = json.loads(MODEL_ENSEMBLE_CONFIG_PATH.read_text())
    preprocessing_config = json.loads(MODEL_PREPROCESSING_CONFIG_PATH.read_text())

    feature_order = feature_schema["feature_order"]
    if len(feature_order) != EXPECTED_FEATURE_COUNT:
        raise ModelLoadError(
            f"feature_schema.json reports {len(feature_order)} features; expected exactly "
            f"{EXPECTED_FEATURE_COUNT} (Baseline {EXPECTED_BASELINE_FEATURE_COUNT} + Group D "
            f"{EXPECTED_GROUP_D_FEATURE_COUNT}). This artifact does not match the validated "
            f"Baseline + Group D configuration -- refusing to start."
        )
    if feature_schema.get("baseline_feature_count") != EXPECTED_BASELINE_FEATURE_COUNT:
        raise ModelLoadError(
            f"feature_schema.json reports baseline_feature_count="
            f"{feature_schema.get('baseline_feature_count')}; expected {EXPECTED_BASELINE_FEATURE_COUNT}."
        )
    group_d_features = feature_schema.get("group_d_features", [])
    if len(group_d_features) != EXPECTED_GROUP_D_FEATURE_COUNT:
        raise ModelLoadError(
            f"feature_schema.json reports {len(group_d_features)} Group D features; "
            f"expected {EXPECTED_GROUP_D_FEATURE_COUNT}."
        )
    print(f"  OK  feature_schema.json: {len(feature_order)} features "
          f"({feature_schema['baseline_feature_count']} baseline + {len(group_d_features)} Group D)")

    weights = ensemble_config["ensemble_weights"]
    weight_sum = sum(weights.values())
    if abs(weight_sum - 1.0) > 1e-4:
        raise ModelLoadError(f"ensemble_config.json's weights sum to {weight_sum:.6f}, not 1.0 -- refusing to start.")
    print(f"  OK  ensemble_config.json: weights sum to {weight_sum:.6f} -- {weights}")

    print("  Loading XGBoost model...")
    xgb_model = joblib.load(MODEL_XGBOOST_PATH)
    print("  Loading LightGBM model...")
    lgbm_model = joblib.load(MODEL_LIGHTGBM_PATH)
    print("  Loading CatBoost model...")
    from catboost import CatBoostRegressor
    catboost_model = CatBoostRegressor()
    catboost_model.load_model(str(MODEL_CATBOOST_PATH))

    models = {"xgboost": xgb_model, "lightgbm": lgbm_model, "catboost": catboost_model}

    # Confirm each model's OWN recorded feature names match the schema
    # exactly -- catches a stale/mismatched schema file at startup,
    # per the deployment plan's Section 10.5.
    xgb_names = list(xgb_model.get_booster().feature_names or [])
    if xgb_names != feature_order:
        raise ModelLoadError(
            f"XGBoost model's own feature names do not match feature_schema.json. "
            f"Model has {len(xgb_names)} names, schema has {len(feature_order)}. "
            f"This artifact set is internally inconsistent -- refusing to start."
        )
    cb_names = list(catboost_model.feature_names_ or [])
    if cb_names != feature_order:
        raise ModelLoadError(
            f"CatBoost model's own feature names do not match feature_schema.json. "
            f"Model has {len(cb_names)} names, schema has {len(feature_order)}. "
            f"This artifact set is internally inconsistent -- refusing to start."
        )
    print(f"  OK  XGBoost and CatBoost models' own feature names match feature_schema.json exactly")
    print("  Note: LightGBM's sklearn wrapper does not reliably expose fitted feature names for this "
          "check the same way XGBoost/CatBoost do; its schema consistency is covered indirectly by "
          "the shared feature_cols DataFrame construction in inference.py, not re-checked here directly.")

    landcover_scheme = None
    if MODEL_LANDCOVER_SCHEME_PATH.exists():
        landcover_scheme = MODEL_LANDCOVER_SCHEME_PATH.read_text()
        print(f"  OK  landcover_encoding_scheme.txt found")
    else:
        print(f"  WARNING: landcover_encoding_scheme.txt not found in {MODEL_DIR}. Feature construction's "
              f"land-cover one-hot encoding step will fail if it needs this file -- see feature_builder.py.")

    print("=" * 70)
    print("Model loaded successfully. Ready to serve prediction requests.")
    print("=" * 70)

    return {
        "models": models,
        "ensemble_weights": {k: float(v) for k, v in weights.items()},
        "feature_order": feature_order,
        "log_clip_bounds": tuple(ensemble_config.get("log_clip_bounds", [-8, 8])),
        "identifier_column": ensemble_config.get("identifier_column", "grid_id"),
        "target_column": ensemble_config.get("target_column", "pm25"),
        "preprocessing_config": preprocessing_config,
        "landcover_scheme_raw": landcover_scheme,
    }
