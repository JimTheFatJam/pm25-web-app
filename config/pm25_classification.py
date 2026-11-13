"""
config/pm25_classification.py -- Configurable PM2.5 concentration
classification schemes (presentation/interpretation layer ONLY).

This module is the single authoritative source of every classification
scheme the web app can display. The frontend receives these definitions
via /api/grid_info and never hardcodes thresholds of its own. Nothing here
touches, or is used by, prediction, feature construction, inference, or
the map's own colour interpolation (frontend/map.js PM25_COLOR_STOPS,
which is unchanged and independent of this file).

The thesis itself did NOT adopt an air-quality classification standard;
which scheme (if any) to display is a product decision made in the UI.
The default selection is "None -- values only".

Every scheme carries its source, averaging period, units, description and
threshold meaning so the UI can label exactly what a category means.

TEMPORAL CAVEAT (surfaced to users via `basis_notes`): the referenced
EPA/WHO thresholds are defined for 24-hour PM2.5. The model produces
hourly predictions; the app's "daily" value is the mean of those modelled
hourly predictions for a Philippine local day, not a separately validated
daily model output. Applying these categories to an hourly value is
therefore indicative only, and applying them to the daily mean is not an
official/regulatory AQI determination.

Adding another documented scheme (e.g. a local standard, if one is ever
formally selected) means appending a dict of the same shape to
PM25_CLASSIFICATION_SCHEMES -- no frontend change is needed.

Scheme dict shape
-----------------
  id, name, kind ("aqi_category" | "guideline_levels" | "custom"),
  source, averaging_period, units, description, threshold_meaning,
  lookup_rule ("truncate_1dp_inclusive" | "le_inclusive"),
  colors_official (bool),
  levels   (guideline_levels only): [{label, value}]
  categories: [{label, lower, upper (None = open-ended), color, interpretation}]
  basis_notes: {"hourly": str, "daily": str}   -- filled in by _finalize()

lookup_rule semantics
---------------------
  truncate_1dp_inclusive: the value is truncated (not rounded) to one
      decimal first -- the US EPA AQI convention for PM2.5 -- and then the
      category with lower <= v <= upper is chosen (e.g. 9.05 -> 9.0 ->
      Good; 35.45 -> 35.4 -> Moderate).
  le_inclusive: bands are "above the previous level, up to and including
      this band's upper" -- a value exactly equal to a WHO level is
      treated as MEETING that level (v <= level).
"""

import math

_HOURLY_NOTE = (
    "Applied to a single modelled hourly prediction. {name} thresholds are defined for "
    "{averaging_period} PM2.5, so this category is indicative only -- not an official "
    "{averaging_period} determination."
)
_DAILY_NOTE = (
    "Applied to the mean of the modelled hourly predictions for this Philippine local day. That "
    "daily value is not a separately validated daily model output, and this category is not an "
    "official or regulatory {averaging_period} determination."
)

PM25_CLASSIFICATION_SCHEMES = [
    {
        "id": "us_epa_aqi_pm25_24h",
        "name": "US EPA PM2.5 AQI (24-hour)",
        "kind": "aqi_category",
        "source": "US EPA, 2024 PM2.5 AQI breakpoint update",
        "averaging_period": "24-hour",
        "units": "µg/m³",
        "description": "The six US EPA Air Quality Index categories for fine particulate matter (PM2.5), "
                       "as defined by the 2024 breakpoint update. A 24-hour PM2.5 AQI classification.",
        "threshold_meaning": "AQI category concentration breakpoints for 24-hour average PM2.5.",
        "lookup_rule": "truncate_1dp_inclusive",
        "colors_official": True,
        "categories": [
            {"label": "Good", "lower": 0.0, "upper": 9.0, "color": "#00e400",
             "interpretation": "Air quality is satisfactory, and air pollution poses little or no risk.",
             "guidance": "No precautions needed. It's a good time to be active outside."},
            {"label": "Moderate", "lower": 9.1, "upper": 35.4, "color": "#ffff00",
             "interpretation": "Air quality is acceptable. However, there may be a risk for some people, "
                               "particularly those who are unusually sensitive to air pollution.",
             "guidance": "Unusually sensitive people: consider reducing prolonged or heavy outdoor exertion; watch for symptoms such as coughing or shortness of breath. Everyone else: no precautions needed."},
            {"label": "Unhealthy for Sensitive Groups", "lower": 35.5, "upper": 55.4, "color": "#ff7e00",
             "interpretation": "Members of sensitive groups may experience health effects. The general "
                               "public is less likely to be affected.",
             "guidance": "People with heart or lung disease, older adults, children, teenagers and outdoor workers: reduce prolonged or heavy outdoor exertion, take more breaks, and watch for symptoms. Everyone else: no precautions needed."},
            {"label": "Unhealthy", "lower": 55.5, "upper": 125.4, "color": "#ff0000",
             "interpretation": "Some members of the general public may experience health effects; members "
                               "of sensitive groups may experience more serious health effects.",
             "guidance": "Sensitive groups: avoid prolonged or heavy outdoor exertion and consider moving activities indoors or rescheduling. Everyone else: reduce prolonged or heavy outdoor exertion and take more breaks."},
            {"label": "Very Unhealthy", "lower": 125.5, "upper": 225.4, "color": "#8f3f97",
             "interpretation": "Health alert: the risk of health effects is increased for everyone.",
             "guidance": "Sensitive groups: avoid all physical activity outdoors; move activities indoors or reschedule. Everyone else: avoid prolonged or heavy outdoor exertion and consider moving activities indoors."},
            {"label": "Hazardous", "lower": 225.5, "upper": None, "color": "#7e0023",
             "interpretation": "Health warning of emergency conditions: everyone is more likely to be affected.",
             "guidance": "Everyone: avoid all physical activity outdoors. Sensitive groups: remain indoors and keep activity levels low. Follow local health advisories."},
        ],
    },
    {
        "id": "who_2021_pm25_24h",
        "name": "WHO 2021 AQG -- PM2.5 24-hour guideline & interim targets",
        "kind": "guideline_levels",
        "source": "WHO Global Air Quality Guidelines (2021), PM2.5 24-hour AQG level and interim targets",
        "averaging_period": "24-hour",
        "units": "µg/m³",
        "description": "Bands between the WHO 2021 24-hour PM2.5 air quality guideline (AQG) level and its four "
                       "interim targets. These bands are APPLICATION-DERIVED interpretations of the WHO levels "
                       "-- WHO does not define an air quality index or named categories.",
        "threshold_meaning": "WHO guideline level (15) and interim targets IT-4 (25), IT-3 (37.5), IT-2 (50), "
                             "IT-1 (75) for 24-hour PM2.5. Not an AQI.",
        "lookup_rule": "le_inclusive",
        "colors_official": False,
        "levels": [
            {"label": "AQG level", "value": 15.0},
            {"label": "Interim target 4", "value": 25.0},
            {"label": "Interim target 3", "value": 37.5},
            {"label": "Interim target 2", "value": 50.0},
            {"label": "Interim target 1", "value": 75.0},
        ],
        "categories": [
            {"label": "Meets WHO AQG level", "lower": 0.0, "upper": 15.0, "color": "#2e7d32",
             "interpretation": "At or below the WHO 2021 24-hour guideline level (15 µg/m³)."},
            {"label": "Above AQG, within Interim Target 4", "lower": 15.0, "upper": 25.0, "color": "#8bc34a",
             "interpretation": "Exceeds the WHO guideline level but is within interim target 4 (25 µg/m³)."},
            {"label": "Above IT-4, within Interim Target 3", "lower": 25.0, "upper": 37.5, "color": "#fdd835",
             "interpretation": "Exceeds interim target 4 but is within interim target 3 (37.5 µg/m³)."},
            {"label": "Above IT-3, within Interim Target 2", "lower": 37.5, "upper": 50.0, "color": "#fb8c00",
             "interpretation": "Exceeds interim target 3 but is within interim target 2 (50 µg/m³)."},
            {"label": "Above IT-2, within Interim Target 1", "lower": 50.0, "upper": 75.0, "color": "#e53935",
             "interpretation": "Exceeds interim target 2 but is within interim target 1 (75 µg/m³)."},
            {"label": "Exceeds Interim Target 1", "lower": 75.0, "upper": None, "color": "#6a1b9a",
             "interpretation": "Exceeds the least stringent WHO interim target (75 µg/m³)."},
        ],
    },
    # "custom": append further documented schemes here using the same
    # shape (kind="custom"). No Philippine national PM2.5 classification is
    # defined here -- none was adopted by the thesis, and none is invented.
]

# Mobile application (SRS REQ-4.3-3..5, REQ-4.3-8): the public-facing app
# shows ONE fixed scheme (the web app lets researchers choose, default
# "None"). The US EPA scheme is used because it is the only scheme here
# with official category names AND established protective guidance --
# its "guidance" texts paraphrase EPA's published PM2.5 cautionary
# statements per category. WHO defines guideline levels, not categories
# or advice, so it carries no guidance field. Presentation only.
MOBILE_CLASSIFICATION_SCHEME_ID = "us_epa_aqi_pm25_24h"


def _finalize(schemes):
    """Attach pre-formatted hourly/daily basis notes to every scheme so
    the frontend displays the same caveat text everywhere."""
    for s in schemes:
        s["basis_notes"] = {
            "hourly": _HOURLY_NOTE.format(name=s["name"], averaging_period=s["averaging_period"]),
            "daily": _DAILY_NOTE.format(averaging_period=s["averaging_period"]),
        }
    return schemes


_finalize(PM25_CLASSIFICATION_SCHEMES)


def get_scheme(scheme_id):
    for s in PM25_CLASSIFICATION_SCHEMES:
        if s["id"] == scheme_id:
            return s
    return None


def _truncate_1dp(value):
    # Truncate toward zero to one decimal (EPA convention), avoiding
    # binary float surprises like 9.1 -> 9.0999 -> 9.0 by nudging first.
    return math.floor(value * 10 + 1e-9) / 10.0


def classify(value, scheme):
    """Returns the matching category dict (label, lower, upper, color,
    interpretation) or None for a missing/NaN/negative value or no
    scheme. Pure lookup; never alters `value`."""
    if scheme is None or value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(v) or v < 0:
        return None

    rule = scheme.get("lookup_rule", "le_inclusive")
    cats = scheme["categories"]
    if rule == "truncate_1dp_inclusive":
        v = _truncate_1dp(v)
        for c in cats:
            if v >= c["lower"] and (c["upper"] is None or v <= c["upper"]):
                return c
        return None
    # le_inclusive
    for i, c in enumerate(cats):
        upper = c["upper"]
        lower_ok = (v >= c["lower"]) if i == 0 else (v > c["lower"])
        if lower_ok and (upper is None or v <= upper):
            return c
    return None


def schemes_for_api():
    """The full list, as served by /api/grid_info. The frontend mirrors
    `lookup_rule` for client-side lookups; the definitions themselves
    live only here."""
    return PM25_CLASSIFICATION_SCHEMES
