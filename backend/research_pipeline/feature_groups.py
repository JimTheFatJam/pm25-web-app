"""
Explicit, Version-Controlled Feature Group Definitions
=============================================================

These group definitions were generated ONCE from source_features.csv's
actual columns (128 predictor columns, confirmed complete and
non-overlapping coverage at generation time -- every column assigned to
exactly one primary group, zero unassigned, zero double-counted), then
hardcoded here as the single source of truth. Experiments read from
FEATURE_GROUPS/CROSS_CUTTING_GROUPS directly -- they do NOT re-derive
groups from whatever columns happen to exist at runtime, so results stay
reproducible even if the feature set changes later (a column rename or
addition elsewhere in the pipeline will not silently reshuffle which
group a variable belongs to for an already-published experiment).

validate_against_columns() should be called once per experiment run to
confirm the hardcoded lists still match the actual feature set in use --
it raises a clear, specific error (not a silent mismatch) if the dataset
has drifted from what these groups were generated against.

To regenerate after a genuine feature-set change: rerun
generate_feature_groups() against the new column list, review the diff
against the definitions below, and update this file deliberately --
never let generation silently overwrite what's hardcoded here.
"""

FEATURE_GROUPS = {
    "meteorology": [
        "temperature_2m_local",
        "temperature_2m_neighborhood_mean",
        "dewpoint_2m_local",
        "dewpoint_2m_neighborhood_mean",
        "surface_pressure_local",
        "surface_pressure_neighborhood_mean",
        "u_wind_10m_local",
        "u_wind_10m_neighborhood_mean",
        "v_wind_10m_local",
        "v_wind_10m_neighborhood_mean",
        "precipitation_local",
        "precipitation_neighborhood_mean",
        "boundary_layer_height_local",
        "boundary_layer_height_neighborhood_mean",
        "relative_humidity_local",
        "wind_speed_10m_local",
        "relative_humidity_neighborhood_mean",
        "wind_speed_10m_neighborhood_mean",
        "temperature_2m_local_rolling_3h",
        "temperature_2m_local_rolling_24h",
        "temperature_2m_neighborhood_mean_rolling_3h",
        "temperature_2m_neighborhood_mean_rolling_24h",
        "dewpoint_2m_local_rolling_3h",
        "dewpoint_2m_local_rolling_24h",
        "dewpoint_2m_neighborhood_mean_rolling_3h",
        "dewpoint_2m_neighborhood_mean_rolling_24h",
        "surface_pressure_local_rolling_3h",
        "surface_pressure_local_rolling_24h",
        "surface_pressure_neighborhood_mean_rolling_3h",
        "surface_pressure_neighborhood_mean_rolling_24h",
        "u_wind_10m_local_rolling_3h",
        "u_wind_10m_local_rolling_24h",
        "u_wind_10m_neighborhood_mean_rolling_3h",
        "u_wind_10m_neighborhood_mean_rolling_24h",
        "v_wind_10m_local_rolling_3h",
        "v_wind_10m_local_rolling_24h",
        "v_wind_10m_neighborhood_mean_rolling_3h",
        "v_wind_10m_neighborhood_mean_rolling_24h",
        "precipitation_local_rolling_3h",
        "precipitation_local_rolling_24h",
        "precipitation_neighborhood_mean_rolling_3h",
        "precipitation_neighborhood_mean_rolling_24h",
        "boundary_layer_height_local_rolling_3h",
        "boundary_layer_height_local_rolling_24h",
        "boundary_layer_height_neighborhood_mean_rolling_3h",
        "boundary_layer_height_neighborhood_mean_rolling_24h",
    ],
    "aerosol": [
        "aod_local",
        "aod_neighborhood_mean",
        "aod_local_rolling_3d",
        "aod_local_rolling_7d",
        "aod_neighborhood_mean_rolling_3d",
        "aod_neighborhood_mean_rolling_7d",
    ],
    "trace_gases": [
        "no2_local",
        "no2_neighborhood_mean",
        "so2_local",
        "so2_neighborhood_mean",
        "co_local",
        "co_neighborhood_mean",
        "o3_local",
        "o3_neighborhood_mean",
        "hcho_local",
        "hcho_neighborhood_mean",
        "no2_local_rolling_3d",
        "no2_local_rolling_7d",
        "no2_neighborhood_mean_rolling_3d",
        "no2_neighborhood_mean_rolling_7d",
        "so2_local_rolling_3d",
        "so2_local_rolling_7d",
        "so2_neighborhood_mean_rolling_3d",
        "so2_neighborhood_mean_rolling_7d",
        "co_local_rolling_3d",
        "co_local_rolling_7d",
        "co_neighborhood_mean_rolling_3d",
        "co_neighborhood_mean_rolling_7d",
        "o3_local_rolling_3d",
        "o3_local_rolling_7d",
        "o3_neighborhood_mean_rolling_3d",
        "o3_neighborhood_mean_rolling_7d",
        "hcho_local_rolling_3d",
        "hcho_local_rolling_7d",
        "hcho_neighborhood_mean_rolling_3d",
        "hcho_neighborhood_mean_rolling_7d",
    ],
    "static_geographic": [
        "elevation_local",
        "elevation_neighborhood_mean",
        "distance_to_coast_km",
        "road_length_local_m",
        "road_length_neighborhood_m",
        "road_density_neighborhood_m_per_km2",
        "population_local",
        "population_neighborhood_mean",
    ],
    "vegetation": [
        "ndvi_local",
        "ndvi_neighborhood_mean",
    ],
    "nighttime_lights": [
        "nighttime_lights_local",
        "nighttime_lights_neighborhood_mean",
    ],
    "landcover": [
        "landcover_neighborhood_mode_fraction",
        "landcover_local_8_Woody_Savannas",
        "landcover_local_9_Savannas",
        "landcover_local_10_Grasslands",
        "landcover_local_12_Croplands",
        "landcover_local_13_Urban_and_Built_up",
        "landcover_local_14_Cropland_Natural_Vegetation_Mosaic",
        "landcover_local_17_Water",
        "landcover_local_other",
        "landcover_neighborhood_mode_8_Woody_Savannas",
        "landcover_neighborhood_mode_9_Savannas",
        "landcover_neighborhood_mode_10_Grasslands",
        "landcover_neighborhood_mode_12_Croplands",
        "landcover_neighborhood_mode_13_Urban_and_Built_up",
        "landcover_neighborhood_mode_14_Cropland_Natural_Vegetation_Mosaic",
        "landcover_neighborhood_mode_17_Water",
        "landcover_neighborhood_mode_other",
    ],
    "temporal": [
        "hour_sin",
        "hour_cos",
        "day_of_week_sin",
        "day_of_week_cos",
        "month_sin",
        "month_cos",
    ],
    "missingness_fallback": [
        "temperature_2m_local_filled_from_neighborhood",
        "dewpoint_2m_local_filled_from_neighborhood",
        "surface_pressure_local_filled_from_neighborhood",
        "u_wind_10m_local_filled_from_neighborhood",
        "v_wind_10m_local_filled_from_neighborhood",
        "precipitation_local_filled_from_neighborhood",
        "boundary_layer_height_local_filled_from_neighborhood",
        "aod_local_filled_from_neighborhood",
        "no2_local_filled_from_neighborhood",
        "so2_local_filled_from_neighborhood",
        "co_local_filled_from_neighborhood",
        "o3_local_filled_from_neighborhood",
        "hcho_local_filled_from_neighborhood",
        "ndvi_local_filled_from_neighborhood",
        "population_local_filled_from_neighborhood",
    ],
}

CROSS_CUTTING_GROUPS = {
    "all_neighborhood_scale": [
        "aod_neighborhood_mean",
        "no2_neighborhood_mean",
        "so2_neighborhood_mean",
        "co_neighborhood_mean",
        "o3_neighborhood_mean",
        "hcho_neighborhood_mean",
        "temperature_2m_neighborhood_mean",
        "dewpoint_2m_neighborhood_mean",
        "surface_pressure_neighborhood_mean",
        "u_wind_10m_neighborhood_mean",
        "v_wind_10m_neighborhood_mean",
        "precipitation_neighborhood_mean",
        "boundary_layer_height_neighborhood_mean",
        "relative_humidity_neighborhood_mean",
        "wind_speed_10m_neighborhood_mean",
        "elevation_neighborhood_mean",
        "road_length_neighborhood_m",
        "road_density_neighborhood_m_per_km2",
        "population_neighborhood_mean",
        "landcover_neighborhood_mode_fraction",
        "nighttime_lights_neighborhood_mean",
        "ndvi_neighborhood_mean",
        "temperature_2m_local_filled_from_neighborhood",
        "dewpoint_2m_local_filled_from_neighborhood",
        "surface_pressure_local_filled_from_neighborhood",
        "u_wind_10m_local_filled_from_neighborhood",
        "v_wind_10m_local_filled_from_neighborhood",
        "precipitation_local_filled_from_neighborhood",
        "boundary_layer_height_local_filled_from_neighborhood",
        "aod_local_filled_from_neighborhood",
        "no2_local_filled_from_neighborhood",
        "so2_local_filled_from_neighborhood",
        "co_local_filled_from_neighborhood",
        "o3_local_filled_from_neighborhood",
        "hcho_local_filled_from_neighborhood",
        "ndvi_local_filled_from_neighborhood",
        "population_local_filled_from_neighborhood",
        "temperature_2m_neighborhood_mean_rolling_3h",
        "temperature_2m_neighborhood_mean_rolling_24h",
        "dewpoint_2m_neighborhood_mean_rolling_3h",
        "dewpoint_2m_neighborhood_mean_rolling_24h",
        "surface_pressure_neighborhood_mean_rolling_3h",
        "surface_pressure_neighborhood_mean_rolling_24h",
        "u_wind_10m_neighborhood_mean_rolling_3h",
        "u_wind_10m_neighborhood_mean_rolling_24h",
        "v_wind_10m_neighborhood_mean_rolling_3h",
        "v_wind_10m_neighborhood_mean_rolling_24h",
        "precipitation_neighborhood_mean_rolling_3h",
        "precipitation_neighborhood_mean_rolling_24h",
        "boundary_layer_height_neighborhood_mean_rolling_3h",
        "boundary_layer_height_neighborhood_mean_rolling_24h",
        "aod_neighborhood_mean_rolling_3d",
        "aod_neighborhood_mean_rolling_7d",
        "no2_neighborhood_mean_rolling_3d",
        "no2_neighborhood_mean_rolling_7d",
        "so2_neighborhood_mean_rolling_3d",
        "so2_neighborhood_mean_rolling_7d",
        "co_neighborhood_mean_rolling_3d",
        "co_neighborhood_mean_rolling_7d",
        "o3_neighborhood_mean_rolling_3d",
        "o3_neighborhood_mean_rolling_7d",
        "hcho_neighborhood_mean_rolling_3d",
        "hcho_neighborhood_mean_rolling_7d",
        "landcover_neighborhood_mode_8_Woody_Savannas",
        "landcover_neighborhood_mode_9_Savannas",
        "landcover_neighborhood_mode_10_Grasslands",
        "landcover_neighborhood_mode_12_Croplands",
        "landcover_neighborhood_mode_13_Urban_and_Built_up",
        "landcover_neighborhood_mode_14_Cropland_Natural_Vegetation_Mosaic",
        "landcover_neighborhood_mode_17_Water",
        "landcover_neighborhood_mode_other",
    ],
    "all_rolling_windows": [
        "temperature_2m_local_rolling_3h",
        "temperature_2m_local_rolling_24h",
        "temperature_2m_neighborhood_mean_rolling_3h",
        "temperature_2m_neighborhood_mean_rolling_24h",
        "dewpoint_2m_local_rolling_3h",
        "dewpoint_2m_local_rolling_24h",
        "dewpoint_2m_neighborhood_mean_rolling_3h",
        "dewpoint_2m_neighborhood_mean_rolling_24h",
        "surface_pressure_local_rolling_3h",
        "surface_pressure_local_rolling_24h",
        "surface_pressure_neighborhood_mean_rolling_3h",
        "surface_pressure_neighborhood_mean_rolling_24h",
        "u_wind_10m_local_rolling_3h",
        "u_wind_10m_local_rolling_24h",
        "u_wind_10m_neighborhood_mean_rolling_3h",
        "u_wind_10m_neighborhood_mean_rolling_24h",
        "v_wind_10m_local_rolling_3h",
        "v_wind_10m_local_rolling_24h",
        "v_wind_10m_neighborhood_mean_rolling_3h",
        "v_wind_10m_neighborhood_mean_rolling_24h",
        "precipitation_local_rolling_3h",
        "precipitation_local_rolling_24h",
        "precipitation_neighborhood_mean_rolling_3h",
        "precipitation_neighborhood_mean_rolling_24h",
        "boundary_layer_height_local_rolling_3h",
        "boundary_layer_height_local_rolling_24h",
        "boundary_layer_height_neighborhood_mean_rolling_3h",
        "boundary_layer_height_neighborhood_mean_rolling_24h",
        "aod_local_rolling_3d",
        "aod_local_rolling_7d",
        "aod_neighborhood_mean_rolling_3d",
        "aod_neighborhood_mean_rolling_7d",
        "no2_local_rolling_3d",
        "no2_local_rolling_7d",
        "no2_neighborhood_mean_rolling_3d",
        "no2_neighborhood_mean_rolling_7d",
        "so2_local_rolling_3d",
        "so2_local_rolling_7d",
        "so2_neighborhood_mean_rolling_3d",
        "so2_neighborhood_mean_rolling_7d",
        "co_local_rolling_3d",
        "co_local_rolling_7d",
        "co_neighborhood_mean_rolling_3d",
        "co_neighborhood_mean_rolling_7d",
        "o3_local_rolling_3d",
        "o3_local_rolling_7d",
        "o3_neighborhood_mean_rolling_3d",
        "o3_neighborhood_mean_rolling_7d",
        "hcho_local_rolling_3d",
        "hcho_local_rolling_7d",
        "hcho_neighborhood_mean_rolling_3d",
        "hcho_neighborhood_mean_rolling_7d",
    ],
}


# ============================================================
# Regeneration helper (run manually, review diff, then update the
# hardcoded definitions above deliberately -- never called automatically
# by any experiment)
# ============================================================

def generate_feature_groups(feature_cols: list) -> tuple:
    """Regenerates group definitions from prefix/pattern rules against a
    given column list. Returns (primary_groups, cross_cutting_groups).
    Prints a coverage report (unassigned / overlapping columns) so any
    drift from a genuine feature-set change is caught before the
    hardcoded definitions above are updated."""
    cols = [c for c in feature_cols if c not in ("grid_id", "pm25")]

    def matches_any(col, prefixes):
        return any(col.startswith(p) for p in prefixes)

    fallback_flags = [c for c in cols if c.endswith("_filled_from_neighborhood")]
    non_fallback = [c for c in cols if c not in fallback_flags]

    met_prefixes = ["temperature_2m_", "dewpoint_2m_", "surface_pressure_", "u_wind_10m_", "v_wind_10m_",
                    "precipitation_", "boundary_layer_height_", "relative_humidity_", "wind_speed_10m_"]
    primary_groups = {
        "meteorology": [c for c in non_fallback if matches_any(c, met_prefixes)],
        "aerosol": [c for c in non_fallback if c.startswith("aod_")],
        "trace_gases": [c for c in non_fallback if matches_any(c, ["no2_", "so2_", "co_", "o3_", "hcho_"])],
        "static_geographic": [c for c in non_fallback if matches_any(c, ["elevation_", "distance_to_coast_km",
                                                                            "road_length_", "road_density_", "population_"])],
        "vegetation": [c for c in non_fallback if c.startswith("ndvi_")],
        "nighttime_lights": [c for c in non_fallback if c.startswith("nighttime_lights_")],
        "landcover": [c for c in non_fallback if c.startswith("landcover_")],
        "temporal": [c for c in non_fallback if c in ("hour_sin", "hour_cos", "day_of_week_sin",
                                                          "day_of_week_cos", "month_sin", "month_cos")],
        "missingness_fallback": fallback_flags,
    }
    cross_cutting = {
        "all_neighborhood_scale": [c for c in cols if "neighborhood" in c],
        "all_rolling_windows": [c for c in cols if "_rolling_" in c],
    }

    all_assigned = set()
    for g, members in primary_groups.items():
        overlap = all_assigned & set(members)
        if overlap:
            print(f"WARNING -- overlap in {g}: {overlap}")
        all_assigned |= set(members)
    unassigned = set(cols) - all_assigned
    if unassigned:
        print(f"WARNING -- unassigned columns: {unassigned}")
    print(f"Coverage: {len(all_assigned)}/{len(cols)} columns assigned to a primary group.")

    return primary_groups, cross_cutting


def validate_against_columns(feature_cols: list) -> None:
    """Call once per experiment run. Raises a clear error if the
    hardcoded FEATURE_GROUPS no longer match the actual dataset -- catches
    silent drift rather than letting an experiment run against
    definitions that no longer describe reality."""
    all_group_cols = set()
    for members in FEATURE_GROUPS.values():
        all_group_cols |= set(members)

    available = set(feature_cols)
    missing_from_dataset = all_group_cols - available
    ungrouped_in_dataset = available - all_group_cols

    if missing_from_dataset:
        raise ValueError(
            f"feature_groups.py is out of date: {len(missing_from_dataset)} columns in the hardcoded "
            f"FEATURE_GROUPS no longer exist in the current dataset: {sorted(missing_from_dataset)[:10]}"
            f"{'...' if len(missing_from_dataset) > 10 else ''}. Regenerate with generate_feature_groups() "
            f"and update this file deliberately.")
    if ungrouped_in_dataset:
        raise ValueError(
            f"feature_groups.py is out of date: {len(ungrouped_in_dataset)} columns exist in the current "
            f"dataset but are not assigned to any group: {sorted(ungrouped_in_dataset)[:10]}"
            f"{'...' if len(ungrouped_in_dataset) > 10 else ''}. Regenerate with generate_feature_groups() "
            f"and update this file deliberately.")
