"""
tests/test_feature_labels.py -- every column in the deployed
feature_schema.json gets a non-empty, unique, non-raw display label.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.prediction import feature_labels
from config.settings import MODEL_FEATURE_SCHEMA_PATH


def _schema_features():
    return json.loads(MODEL_FEATURE_SCHEMA_PATH.read_text())["feature_order"]


def test_every_schema_feature_has_a_label():
    names = _schema_features()
    assert len(names) == 157
    labels = [feature_labels.display_name(n) for n in names]
    assert all(isinstance(l, str) and l.strip() for l in labels)
    unlabelled = [n for n, l in zip(names, labels) if l == n]
    assert unlabelled == [], f"raw names fell through the rules: {unlabelled}"
    assert len(set(labels)) == len(labels), "display labels must be unique per feature"


def test_specific_rules():
    d = feature_labels.display_name
    assert d("temperature_2m_local") == "2 m air temperature (at cell) [K]"
    assert d("no2_neighborhood_mean_rolling_7d") == "Tropospheric NO₂ column (Sentinel-5P), 7-day rolling mean (5 km neighborhood mean) [mol/m²]"
    assert d("relative_humidity_local_tendency_3h") == "Relative humidity (derived), 3-hour tendency (change, at cell) [%]"
    assert d("aod_local_filled_from_neighborhood").endswith("[flag]")
    assert "Woody Savannas" in d("landcover_local_8_Woody_Savannas")
    assert "other class" in d("landcover_neighborhood_mode_other")
    assert d("transport_so2_local").startswith("Upwind transport of SO")
    assert d("distance_to_coast_km") == "Distance to coast [km]"


def test_unknown_name_falls_back_to_raw():
    assert feature_labels.display_name("something_unexpected_xyz") == "something_unexpected_xyz"
