"""
tests/test_classification.py -- config/pm25_classification.py: scheme
structure/metadata completeness, EPA truncation rule, WHO <= rule, and
missing-value handling. Presentation layer only; nothing here touches the
model or predictions.
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.pm25_classification import (
    PM25_CLASSIFICATION_SCHEMES, classify, get_scheme, schemes_for_api,
)

REQUIRED_META = ("id", "name", "kind", "source", "averaging_period", "units", "description",
                 "threshold_meaning", "lookup_rule", "colors_official", "categories", "basis_notes")


def test_every_scheme_has_complete_metadata_and_notes():
    assert len(PM25_CLASSIFICATION_SCHEMES) >= 2
    ids = set()
    for s in PM25_CLASSIFICATION_SCHEMES:
        for k in REQUIRED_META:
            assert k in s and s[k] not in (None, ""), f"{s.get('id')} missing {k}"
        assert s["id"] not in ids; ids.add(s["id"])
        assert s["kind"] in ("aqi_category", "guideline_levels", "custom")
        assert s["lookup_rule"] in ("truncate_1dp_inclusive", "le_inclusive")
        assert set(s["basis_notes"]) == {"hourly", "daily"}
        assert s["averaging_period"] in s["basis_notes"]["hourly"]
        assert "not" in s["basis_notes"]["daily"].lower()  # explicitly disclaims official determination
        for c in s["categories"]:
            for k in ("label", "lower", "upper", "color", "interpretation"):
                assert k in c, f"{s['id']}/{c.get('label')} missing {k}"
            assert c["interpretation"]


def test_categories_are_ordered_and_non_overlapping():
    for s in PM25_CLASSIFICATION_SCHEMES:
        cats = s["categories"]
        assert cats[0]["lower"] == 0.0
        assert cats[-1]["upper"] is None  # open-ended top band
        for a, b in zip(cats, cats[1:]):
            assert a["upper"] is not None
            assert b["lower"] >= a["upper"], f"{s['id']}: {a['label']} overlaps {b['label']}"
            assert b["lower"] > a["lower"]


def test_epa_scheme_values_and_wording():
    epa = get_scheme("us_epa_aqi_pm25_24h")
    assert epa["source"] == "US EPA, 2024 PM2.5 AQI breakpoint update"
    assert epa["averaging_period"] == "24-hour" and epa["kind"] == "aqi_category"
    bounds = [(c["lower"], c["upper"]) for c in epa["categories"]]
    assert bounds == [(0.0, 9.0), (9.1, 35.4), (35.5, 55.4), (55.5, 125.4), (125.5, 225.4), (225.5, None)]
    assert [c["label"] for c in epa["categories"]] == [
        "Good", "Moderate", "Unhealthy for Sensitive Groups", "Unhealthy", "Very Unhealthy", "Hazardous"]


def test_epa_truncation_rule():
    epa = get_scheme("us_epa_aqi_pm25_24h")
    assert classify(0.0, epa)["label"] == "Good"
    assert classify(9.0, epa)["label"] == "Good"
    assert classify(9.05, epa)["label"] == "Good"       # truncates to 9.0 (not rounded up)
    assert classify(9.1, epa)["label"] == "Moderate"    # float nudge keeps 9.1 -> 9.1
    assert classify(35.45, epa)["label"] == "Moderate"  # 35.4
    assert classify(35.5, epa)["label"] == "Unhealthy for Sensitive Groups"
    assert classify(55.49, epa)["label"] == "Unhealthy for Sensitive Groups"
    assert classify(125.5, epa)["label"] == "Very Unhealthy"
    assert classify(225.5, epa)["label"] == "Hazardous"
    assert classify(1000.0, epa)["label"] == "Hazardous"


def test_who_scheme_is_guideline_levels_not_aqi():
    who = get_scheme("who_2021_pm25_24h")
    assert who["kind"] == "guideline_levels"
    assert [lv["value"] for lv in who["levels"]] == [15.0, 25.0, 37.5, 50.0, 75.0]
    assert "not an aqi" in who["threshold_meaning"].lower()
    assert "application-derived" in who["description"].lower()
    assert who["colors_official"] is False


def test_who_le_inclusive_rule():
    who = get_scheme("who_2021_pm25_24h")
    assert classify(0.0, who)["label"] == "Meets WHO AQG level"
    assert classify(15.0, who)["label"] == "Meets WHO AQG level"            # equal to a level meets it
    assert classify(15.01, who)["label"] == "Above AQG, within Interim Target 4"
    assert classify(25.0, who)["label"] == "Above AQG, within Interim Target 4"
    assert classify(37.5, who)["label"] == "Above IT-4, within Interim Target 3"
    assert classify(50.0, who)["label"] == "Above IT-3, within Interim Target 2"
    assert classify(75.0, who)["label"] == "Above IT-2, within Interim Target 1"
    assert classify(75.1, who)["label"] == "Exceeds Interim Target 1"


def test_missing_or_invalid_values_return_none():
    epa = get_scheme("us_epa_aqi_pm25_24h")
    assert classify(None, epa) is None
    assert classify(float("nan"), epa) is None
    assert classify(-0.5, epa) is None
    assert classify("abc", epa) is None
    assert classify(10.0, None) is None
    assert get_scheme("does-not-exist") is None


def test_api_payload_is_json_serialisable():
    import json
    text = json.dumps(schemes_for_api(), allow_nan=False)
    assert "us_epa_aqi_pm25_24h" in text and "who_2021_pm25_24h" in text
