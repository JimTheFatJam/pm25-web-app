"""
config/local_settings.example.py
============================================================================
Copy to config/local_settings.py (which is gitignored) and fill in your own
values. Anything defined here overrides config/settings.py.

On the server, prefer real environment variables (see deploy/pm25-web.env
and DEPLOYMENT.md) -- this file is a convenience for local development.
"""

# Your Google Cloud project with the Earth Engine API enabled.
GEE_PROJECT_ID = "your-gcp-project-id"

# Only if you want the backend to import the ORIGINAL research modules
# instead of the vendored copies in backend/research_pipeline/:
# from pathlib import Path
# RESEARCH_PIPELINE_DIR = Path(r"C:\path\to\air_quality_thesis_project\OpenAQ_API")
