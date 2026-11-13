Baseline + Group D deployment model artifacts
================================================

SUPERSEDED -- NOT LOADED BY THE BACKEND.
config/settings.py now points at data/models/philippine_only/ (the
Philippine-only, no-transfer ensemble: frozen-holdout R2 0.4123 vs 0.3338
for this fine-tuned set). These files are kept for rollback and for the
thesis's transfer-vs-local comparison. To go back, repoint MODEL_DIR and
the three model filenames in config/settings.py.

Place the six files produced by train_final_deployment_model.py's
final_deployment_artifacts/ output directly here (copy, don't move the
originals, so you keep a backup):

    finetuned_xgboost_logspace.joblib
    finetuned_lightgbm_logspace.joblib
    finetuned_catboost_logspace.cbm
    ensemble_config.json
    feature_schema.json
    preprocessing_config.json
    landcover_encoding_scheme.txt   (if it was found and copied by that script)

Do NOT place any other .joblib/.cbm files here, and do not rename these
files -- config/settings.py references these exact filenames.

At Flask startup, backend/prediction/model_loader.py:
  1. Confirms every file above exists (fails clearly, naming exactly
     which file is missing, if not).
  2. Loads the three models.
  3. Confirms each model's own internal feature-name list matches
     feature_schema.json exactly (fails clearly on any mismatch).
  4. Confirms feature_schema.json reports exactly 157 features
     (152 baseline + 5 Group D), per the validated deployment
     configuration -- fails clearly if not.
  5. Confirms ensemble_config.json's three weights sum to 1.0.

If any of these checks fail, the backend refuses to start serving
prediction requests -- it does not fall back to a placeholder model or
skip the check.

The backend NEVER trains, fine-tunes, or modifies these files. If you
need a new model version, retrain with train_final_deployment_model.py
in your research environment and replace these files entirely.
