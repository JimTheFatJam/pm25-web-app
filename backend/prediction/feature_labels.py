"""
backend/prediction/feature_labels.py
============================================================================
Human-readable display names for the model's 157 feature columns
(the deployed feature_schema.json, config.settings.MODEL_FEATURE_SCHEMA_PATH), for the SHAP
explanation UI only. Purely cosmetic: the raw feature name is always
returned alongside, and nothing here affects feature construction.

Rule-based so every schema column gets a label without a 157-entry
hand-written table: the name is decomposed into
  <base variable> + <scale> + <transform>
and each part is rendered from a small vocabulary confirmed against the
research pipeline's own definitions (scales/rolling windows from
prepare_features.py; RH/wind speed from extract_era5_hourly_batched.py's
derive_additional_features; physics/transport from
run_physics_transport_experiment.py; tendency features from
advanced_feature_engineering_experiment.py; land-cover one-hots from
one_hot_encode_landcover). Unit hints are given only where the source
unit is documented in those scripts (e.g. temperature in Kelvin, RH in %).
Unknown names fall back to the raw name, never an error.
"""

import re

# Base variables (units only where documented in the research extraction scripts).
BASE_VARIABLES = {
    "aod": ("MODIS AOD 470 nm (raw band value)", None),
    "no2": ("Tropospheric NO₂ column (Sentinel-5P)", "mol/m²"),
    "so2": ("SO₂ column (Sentinel-5P)", "mol/m²"),
    "co": ("CO column (Sentinel-5P)", "mol/m²"),
    "o3": ("O₃ column (Sentinel-5P)", "mol/m²"),
    "hcho": ("Tropospheric HCHO column (Sentinel-5P)", "mol/m²"),
    "temperature_2m": ("2 m air temperature", "K"),
    "dewpoint_2m": ("2 m dew point temperature", "K"),
    "surface_pressure": ("Surface pressure", "Pa"),
    "u_wind_10m": ("10 m wind, east–west component (u)", "m/s"),
    "v_wind_10m": ("10 m wind, north–south component (v)", "m/s"),
    "precipitation": ("Hourly precipitation", "m"),
    "boundary_layer_height": ("Boundary layer height", "m"),
    "relative_humidity": ("Relative humidity (derived)", "%"),
    "wind_speed_10m": ("10 m wind speed (derived)", "m/s"),
    "elevation": ("Elevation", "m"),
    "population": ("Population count (WorldPop 1 km raster)", None),
    "ndvi": ("NDVI vegetation index (raw MODIS value)", None),
    "nighttime_lights": ("Nighttime lights radiance (VIIRS)", None),
}

SCALES = {
    "local": "at cell",
    "neighborhood_mean": "5 km neighborhood mean",
}

FIXED_LABELS = {
    "distance_to_coast_km": ("Distance to coast", "km"),
    "road_length_local_m": ("Road length at cell", "m"),
    "road_length_neighborhood_m": ("Road length, 5 km neighborhood", "m"),
    "road_density_neighborhood_m_per_km2": ("Road density, 5 km neighborhood", "m/km²"),
    "landcover_neighborhood_mode_fraction": ("Share of neighborhood in its dominant land-cover class", "fraction"),
    "hour_sin": ("Hour of day (sine encoding)", None),
    "hour_cos": ("Hour of day (cosine encoding)", None),
    "day_of_week_sin": ("Day of week (sine encoding)", None),
    "day_of_week_cos": ("Day of week (cosine encoding)", None),
    "month_sin": ("Month (sine encoding)", None),
    "month_cos": ("Month (cosine encoding)", None),
    "consecutive_dry_days": ("Consecutive dry days", "days"),
    "is_weekend": ("Weekend (flag)", None),
    "is_public_holiday": ("Public holiday (flag)", None),
    "is_new_years_eve": ("New Year's Eve (flag)", None),
    "is_new_years_day": ("New Year's Day (flag)", None),
    "is_christmas_eve": ("Christmas Eve (flag)", None),
    "is_christmas": ("Christmas Day (flag)", None),
    "is_lunar_new_year": ("Lunar New Year (flag)", None),
    "is_day_after_holiday": ("Day after a holiday (flag)", None),
    "aod_over_blh_local": ("AOD / boundary layer height (physics feature, at cell)", None),
    "aod_times_rh_local": ("AOD × relative humidity (physics feature, at cell)", None),
    "aod_over_blh_times_rh_local": ("AOD / BLH × relative humidity (physics feature, at cell)", None),
    "ventilation_coefficient_local": ("Ventilation coefficient (BLH × wind speed, at cell)", None),
    "stagnation_indicator_local": ("Stagnation indicator (at cell)", None),
}

TRANSPORT_RE = re.compile(r"^transport_(aod|no2|so2|co|o3|hcho)_local$")
LANDCOVER_RE = re.compile(r"^landcover_(local|neighborhood_mode)_(?:(\d+)_(.+)|other)$")
FILLED_RE = re.compile(r"^(.+)_local_filled_from_neighborhood$")
ROLLING_RE = re.compile(r"^(.+)_(local|neighborhood_mean)_rolling_(\d+)([hd])$")
TENDENCY_RE = re.compile(r"^(.+)_local_tendency_(\d+)h$")
SCALED_RE = re.compile(r"^(.+)_(local|neighborhood_mean)$")

_OVERRIDES = {}  # raw name -> (label, unit); for any wording the user wants to pin explicitly


def _base(name):
    return BASE_VARIABLES.get(name, (name, None))


def label_and_unit(name):
    """Returns (display_label, unit_or_None) for a raw feature name."""
    if name in _OVERRIDES:
        return _OVERRIDES[name]
    if name in FIXED_LABELS:
        return FIXED_LABELS[name]

    m = TRANSPORT_RE.match(name)
    if m:
        base, _ = _base(m.group(1))
        return (f"Upwind transport of {base} (wind-weighted, within 15 km)", None)

    m = LANDCOVER_RE.match(name)
    if m:
        scope = "at cell" if m.group(1) == "local" else "dominant in 5 km neighborhood"
        if m.group(2) is None:
            return (f"Land cover {scope}: other class", "flag")
        cls = m.group(3).replace("_", " ")
        return (f"Land cover {scope}: {cls} (IGBP {m.group(2)})", "flag")

    m = FILLED_RE.match(name)
    if m:
        base, _ = _base(m.group(1))
        return (f"{base}: cell value was filled from neighborhood", "flag")

    m = ROLLING_RE.match(name)
    if m:
        base, unit = _base(m.group(1))
        window = f"{m.group(3)}-hour" if m.group(4) == "h" else f"{m.group(3)}-day"
        return (f"{base}, {window} rolling mean ({SCALES[m.group(2)]})", unit)

    m = TENDENCY_RE.match(name)
    if m:
        base, unit = _base(m.group(1))
        return (f"{base}, {m.group(2)}-hour tendency (change, at cell)", unit)

    m = SCALED_RE.match(name)
    if m and m.group(1) in BASE_VARIABLES:
        base, unit = _base(m.group(1))
        return (f"{base} ({SCALES[m.group(2)]})", unit)

    return (name, None)  # unknown: raw name, never an error


def display_name(name):
    label, unit = label_and_unit(name)
    return f"{label} [{unit}]" if unit else label
