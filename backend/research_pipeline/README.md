# Vendored research-pipeline modules

**These are verbatim copies. Do not edit them here.**

`backend/prediction/feature_builder.py` reuses the research pipeline's own
validated feature-construction functions rather than reimplementing them, so
the deployed app needs those modules importable. Cloning this repository alone
used to be insufficient -- the modules lived only in the research project --
which meant a server could not start. They are therefore copied in here so the
repository deploys standalone.

Source: `air_quality_thesis_project/OpenAQ_API/`

Five modules are imported directly by the backend:

    prepare_features.py
    run_physics_transport_experiment.py
    build_additional_features.py
    advanced_feature_engineering_experiment.py
    extract_era5_hourly_batched.py

The other 13 files are their transitive imports (resolved automatically; the
full set is 18 modules, ~0.41 MB).

## Keeping them in sync

The research pipeline is frozen, so these should not drift. If you ever do
change a research module, re-copy the whole set rather than editing here --
otherwise the deployed behaviour silently diverges from the thesis:

    python deploy/sync_research_pipeline.py --source <path-to>/OpenAQ_API

`config.settings.RESEARCH_PIPELINE_DIR` defaults to this folder and can be
pointed back at the original with `PM25_RESEARCH_PIPELINE_DIR`.
