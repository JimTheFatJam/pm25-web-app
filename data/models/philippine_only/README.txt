Philippine-only (no-transfer) deployment model artifacts
=========================================================

The deployed ensemble. Trained from scratch on Philippine data only --
no source-domain pretraining, no fine-tuning -- by

    philippine_only_full_evaluation.py   (Stage A)

in the research environment, on the SAME frozen split the previous
fine-tuned artifacts used (fit on the finetune_train cells, early stopping
on finetune_val, evaluated on the 30 held-out cells), with the same 157
Baseline + Group D features, the same log1p target, the same magnitude
weights (alpha 0.75) and the same inverse-RMSE ensemble weighting.

Why it replaced the fine-tuned set (frozen spatial holdout, Group D):

    Philippine-only     RMSE 10.2139   R2 0.4123
    Fine-tuned (deployed before)  RMSE 10.8748   R2 0.3338

    R2 advantage +0.0796, bootstrap 95% CI [0.0488, 0.1114],
    Philippine-only better in 100% of cell resamples.

Note for reporting: under 5-fold cell-blocked CV the Philippine-only model
also beats the fine-tuned model at its DEFAULT settings (R2 0.2650 vs
0.2345, p = 2e-5), but not the fine-tuned model at EXTENDED settings
(0.2650 vs 0.2777, CI crosses zero). The model deployed before this swap
was the default-settings fine-tune. See results.json in the same
diagnostics output folder.

Files
-----

    ph_only_xgboost.joblib          Stage A Philippine-only XGBoost
    ph_only_lightgbm.joblib         Stage A Philippine-only LightGBM
    ph_only_catboost.cbm            Stage A Philippine-only CatBoost
    ensemble_config.json            that run's Philippine-only weights
    feature_schema.json             UNCHANGED from the fine-tuned set --
                                    the 157-feature order is byte-identical
                                    to the run's own feature_cols.json
    preprocessing_config.json       unchanged
    landcover_encoding_scheme.txt   unchanged

Do NOT rename these files, and do not place any other .joblib/.cbm here --
config/settings.py references these exact names.

Note that the run writes feature_cols.json (a plain list); the backend
reads feature_schema.json (feature_count / feature_order /
baseline_feature_count / group_d_features). They carry the same order, so
feature_schema.json was copied across unchanged rather than regenerated.

Startup validation (backend/prediction/model_loader.py) is unchanged:
every file must exist, each model's own feature names must match
feature_schema.json exactly, the schema must report 157 features
(152 baseline + 5 Group D), and the three ensemble weights must sum to 1.

The backend NEVER trains, fine-tunes, or modifies these files. For a new
version, retrain in the research environment and replace them entirely.

Rollback
--------

The superseded fine-tuned artifacts are kept in
data/models/baseline_group_d/. Switching back is a four-line change in
config/settings.py (MODEL_DIR and the three model filenames); nothing else
in the backend depends on which of the two sets is loaded.
