"""
backend/prediction/static_predictors.py
============================================================================
Loads the STATIC and slow-changing predictors for a set of selected cells:
elevation, land cover, NDVI, nighttime lights, population, distance to
coast, and road density -- both "local" (the cell's own footprint) and
"neighborhood_mean" (5x5 cell block, per NEIGHBORHOOD_RADIUS_CELLS=2,
confirmed from extract_daily_predictors_batched.py) scope, matching the
research pipeline's own established local/neighborhood pattern.

Scope decisions for THIS diagnostic-testing stage (see the deployment
plan, Section 4, and this task's explicit "do not implement scheduled
static-layer refreshes" instruction):

  - Elevation, land cover, NDVI, nighttime lights: fetched LIVE from GEE,
    per request, for only the requested cells + their neighborhoods. For
    a small test (a handful of cells) this is fast and simple. The
    deployment plan's longer-term recommendation (precompute once for the
    full 313,845-cell grid, refresh on a schedule) is NOT implemented
    here -- explicitly deferred.
  - Population, distance to coast: read directly from the local files
    (data/local_datasets/), computed for just the requested cells --
    fast, no GEE involved.
  - Road density: NOT recomputed from philippines-latest.osm.pbf per
    request. Parsing the full .pbf file is expensive regardless of how
    few cells are requested, and reimplementing that parsing logic fresh
    (rather than reusing your existing, already-validated
    extract_road_density_streaming.py / extract_road_density_local.py)
    would risk exactly the "inventing alternative feature calculations"
    this task explicitly said not to do. Instead, this REQUIRES a
    precomputed cache file (see ROAD_DENSITY_CACHE_PATH below), which you
    generate ONCE by running your own existing road-density extraction
    script against the full prediction grid (not just the 915 training
    cells) -- see the README for the exact command. If a requested cell
    is missing from that cache, this module fails clearly rather than
    silently using a 0 or an average.
"""

from pathlib import Path

import ee
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterstats

from config.settings import (
    WORLDPOP_DIR, COASTLINE_PATH, DISPLAY_CRS, CELL_SIDE_METERS,
    DATA_DIR,
)

NEIGHBORHOOD_RADIUS_CELLS = 2  # confirmed identical to extract_daily_predictors_batched.py's own constant
ROAD_DENSITY_CACHE_PATH = DATA_DIR / "static_layers" / "road_density_by_cell.csv"

# GEE collection IDs/bands -- confirmed directly from the research
# pipeline's own extraction scripts (extract_elevation.py,
# extract_landcover.py, extract_ndvi_monthly.py,
# extract_nighttime_lights.py), reused exactly, not re-chosen.
ELEVATION_COLLECTION = "USGS/SRTMGL1_003"
ELEVATION_BAND = "elevation"
LANDCOVER_COLLECTION = "MODIS/061/MCD12Q1"
LANDCOVER_BAND = "LC_Type1"
LANDCOVER_NATIVE_SCALE_M = 500  # confirmed from extract_landcover.py's own constant
NDVI_COLLECTION = "MODIS/061/MOD13A3"
NDVI_BAND = "NDVI"
VIIRS_COLLECTION = "NOAA/VIIRS/DNB/MONTHLY_V1/VCMCFG"
VIIRS_BAND = "avg_rad"


class StaticPredictorError(Exception):
    """Raised when a required static predictor cannot be obtained --
    never silently substituted, per explicit instruction."""


def _neighborhood_geometry_wgs84(grid_x, grid_y, grid_crs):
    """Mirrors extract_daily_predictors_batched.py's own
    neighborhood_rectangle() exactly (radius=2 cells), then reprojects to
    WGS84 for GEE (which expects EPSG:4326 geometry input)."""
    import shapely.geometry
    x0 = (grid_x - NEIGHBORHOOD_RADIUS_CELLS) * CELL_SIDE_METERS
    y0 = (grid_y - NEIGHBORHOOD_RADIUS_CELLS) * CELL_SIDE_METERS
    x1 = (grid_x + NEIGHBORHOOD_RADIUS_CELLS + 1) * CELL_SIDE_METERS
    y1 = (grid_y + NEIGHBORHOOD_RADIUS_CELLS + 1) * CELL_SIDE_METERS
    rect = gpd.GeoSeries([shapely.geometry.box(x0, y0, x1, y1)], crs=grid_crs)
    return rect.to_crs(DISPLAY_CRS).iloc[0]


def _local_geometry_wgs84(grid_x, grid_y, grid_crs):
    import shapely.geometry
    x0, y0 = grid_x * CELL_SIDE_METERS, grid_y * CELL_SIDE_METERS
    x1, y1 = x0 + CELL_SIDE_METERS, y0 + CELL_SIDE_METERS
    rect = gpd.GeoSeries([shapely.geometry.box(x0, y0, x1, y1)], crs=grid_crs)
    return rect.to_crs(DISPLAY_CRS).iloc[0]


def fetch_gee_static_layers(cells_df: pd.DataFrame, grid_crs, prediction_year: int, log_fn=print) -> pd.DataFrame:
    """cells_df must have columns: grid_id, grid_x_index, grid_y_index.
    Returns a DataFrame with one row per grid_id and columns:
    elevation_local, elevation_neighborhood_mean, landcover_local (raw
    IGBP code, one-hot encoding happens later in feature_builder.py),
    ndvi_local, ndvi_neighborhood_mean, nighttime_lights_local,
    nighttime_lights_neighborhood_mean.

    Fetched LIVE per request -- see module docstring for why this is
    appropriate scope for a small diagnostic test but not the eventual
    production design."""
    log_fn(f"Fetching static GEE layers (elevation, land cover, NDVI, nighttime lights) for {len(cells_df)} cell(s)...")

    local_geoms = {row.grid_id: _local_geometry_wgs84(row.grid_x_index, row.grid_y_index, grid_crs) for row in cells_df.itertuples()}
    nbhd_geoms = {row.grid_id: _neighborhood_geometry_wgs84(row.grid_x_index, row.grid_y_index, grid_crs) for row in cells_df.itertuples()}

    def to_fc(geom_dict):
        feats = [ee.Feature(ee.Geometry.Rectangle(list(geom.bounds)), {"grid_id": gid}) for gid, geom in geom_dict.items()]
        return ee.FeatureCollection(feats)

    local_fc = to_fc(local_geoms)
    nbhd_fc = to_fc(nbhd_geoms)

    results = {gid: {} for gid in cells_df["grid_id"]}

    # Elevation -- static, single image, no date needed
    elev_img = ee.Image(ELEVATION_COLLECTION).select(ELEVATION_BAND)
    _reduce_and_merge(elev_img, local_fc, results, "elevation_local", log_fn)
    _reduce_and_merge(elev_img, nbhd_fc, results, "elevation_neighborhood_mean", log_fn)

    # Land cover -- annual; use the latest year <= prediction_year with an
    # image actually present (mirrors extract_landcover.py's own
    # resolve_data_year() fallback pattern -- not re-derived from scratch,
    # same logic: clamp to the nearest available year, never silently use
    # an arbitrary year).
    lc_collection = ee.ImageCollection(LANDCOVER_COLLECTION).select(LANDCOVER_BAND)
    lc_year = _resolve_landcover_year(lc_collection, prediction_year, log_fn)
    lc_img = lc_collection.filter(ee.Filter.calendarRange(lc_year, lc_year, "year")).first()
    _reduce_and_merge(lc_img, local_fc, results, "landcover_local", log_fn, reducer="mode")

    # BUG FIX: neighborhood land cover was never computed at all before --
    # only the local mode was. The model expects landcover_neighborhood_mode
    # (the neighborhood's own dominant class -- NOT a mean, since a class
    # code is categorical, not continuous) plus
    # landcover_neighborhood_mode_fraction (what fraction of the
    # neighborhood actually matches that dominant class -- a homogeneity
    # measure). Replicates extract_landcover.py's own combined
    # mode()+frequencyHistogram() reducer exactly, confirmed directly from
    # its source rather than approximated with a simpler reducer.
    _reduce_landcover_neighborhood_mode(lc_img, nbhd_fc, results, log_fn)

    # NDVI -- monthly; use the most recently completed month <= prediction period
    ndvi_collection = ee.ImageCollection(NDVI_COLLECTION).select(NDVI_BAND)
    # For this diagnostic stage, uses the latest available image overall
    # rather than resolving an exact requested month -- acceptable
    # simplification for a first small test; the production version
    # should select the exact month matching the request per the
    # deployment plan's Section 2/3.
    ndvi_img = ndvi_collection.sort("system:time_start", False).first()
    _reduce_and_merge(ndvi_img, local_fc, results, "ndvi_local", log_fn)
    _reduce_and_merge(ndvi_img, nbhd_fc, results, "ndvi_neighborhood_mean", log_fn)

    # Nighttime lights -- monthly; same simplification as NDVI above for this stage
    viirs_collection = ee.ImageCollection(VIIRS_COLLECTION).select(VIIRS_BAND)
    viirs_img = viirs_collection.sort("system:time_start", False).first()
    _reduce_and_merge(viirs_img, local_fc, results, "nighttime_lights_local", log_fn)
    _reduce_and_merge(viirs_img, nbhd_fc, results, "nighttime_lights_neighborhood_mean", log_fn)

    df = pd.DataFrame.from_dict(results, orient="index").reset_index().rename(columns={"index": "grid_id"})
    log_fn(f"Static GEE layers fetched for {len(df)} cell(s).")
    return df


def _reduce_and_merge(image, fc, results, out_col, log_fn, reducer="mean"):
    reducer_obj = ee.Reducer.mode() if reducer == "mode" else ee.Reducer.mean()
    try:
        reduced = image.reduceRegions(collection=fc, reducer=reducer_obj, scale=_scale_for(out_col))
        for feat in reduced.getInfo()["features"]:
            gid = feat["properties"]["grid_id"]
            value = feat["properties"].get(reducer if reducer == "mean" else "mode")
            results[gid][out_col] = value
    except ee.EEException as e:
        log_fn(f"  GEE ERROR fetching {out_col}: {type(e).__name__}: {e}")
        for gid in results:
            results[gid][out_col] = None


def _reduce_landcover_neighborhood_mode(image, fc, results, log_fn):
    """Replicates extract_landcover.py's own run_mode_fraction_reduce_regions()
    exactly: mode (dominant class) plus the fraction of pixels in the
    neighborhood matching that mode, via a combined mode()+
    frequencyHistogram() reducer. Confirmed directly from that script's
    source, including its own noted risk area: the histogram key format
    (a dict of {str(class_id): pixel_count}) -- watch this specifically
    if landcover_neighborhood_mode_fraction comes back null while
    landcover_neighborhood_mode looks correct, per that script's own
    documented caveat."""
    combined_reducer = ee.Reducer.mode().combine(ee.Reducer.frequencyHistogram(), sharedInputs=True)
    try:
        reduced = image.reduceRegions(collection=fc, reducer=combined_reducer, scale=LANDCOVER_NATIVE_SCALE_M)
        for feat in reduced.getInfo()["features"]:
            props = feat["properties"]
            gid = props["grid_id"]
            mode_val = props.get("mode")
            histogram = props.get("histogram", {}) or {}
            total = sum(histogram.values()) if histogram else 0
            mode_count = histogram.get(str(int(mode_val)), 0) if mode_val is not None and total else 0
            fraction = (mode_count / total) if total else None
            results[gid]["landcover_neighborhood_mode"] = mode_val
            results[gid]["landcover_neighborhood_mode_fraction"] = fraction
    except ee.EEException as e:
        log_fn(f"  GEE ERROR fetching landcover_neighborhood_mode(_fraction): {type(e).__name__}: {e}")
        for gid in results:
            results[gid]["landcover_neighborhood_mode"] = None
            results[gid]["landcover_neighborhood_mode_fraction"] = None


def _scale_for(out_col):
    if out_col.startswith("elevation"):
        return 30
    if out_col.startswith("landcover"):
        return 500
    if out_col.startswith("ndvi"):
        return 1000
    if out_col.startswith("nighttime_lights"):
        return 463.83
    return 1000


def _resolve_landcover_year(lc_collection, requested_year, log_fn):
    """Mirrors extract_landcover.py's own resolve_data_year() fallback --
    clamp to the latest year with real data, never silently use an
    unavailable year. NOT re-verified live against the actual current
    collection bounds in this diagnostic stage -- uses a conservative
    hardcoded fallback ceiling consistent with the deployment plan's own
    documented finding (LATEST_AVAILABLE_YEAR was 2024 as of that
    review)."""
    fallback_ceiling = 2024
    year = min(requested_year, fallback_ceiling)
    if requested_year > fallback_ceiling:
        log_fn(f"  Land cover: requested year {requested_year} not yet available; using {year} "
               f"(see the deployment plan, Section 2.1 -- this ceiling needs live re-verification).")
    return year


def load_population(cells_df: pd.DataFrame, grid_crs, prediction_year: int, log_fn=print) -> pd.DataFrame:
    """Local-only, per config.settings.WORLDPOP_DIR. Fails clearly if the
    requested year's raster isn't present -- never substitutes a
    different year silently."""
    import re
    year_pattern = re.compile(r"20(?:1[5-9]|2[0-9]|30)")  # widened per the deployment plan, Section 4A.3 -- NOT the research script's original 2020-2026-only pattern
    matches = []
    if WORLDPOP_DIR.exists():
        for f in WORLDPOP_DIR.glob("*.tif"):
            m = year_pattern.search(f.name)
            if m and int(m.group()) == prediction_year:
                matches.append(f)

    if not matches:
        raise StaticPredictorError(
            f"No WorldPop raster found for year {prediction_year} in {WORLDPOP_DIR}. "
            f"Expected a filename containing '{prediction_year}' (e.g. "
            f"phl_pop_{prediction_year}_CN_1km_R2025A_UA_v1.tif). See "
            f"data/local_datasets/worldpop/README.txt. Refusing to substitute a "
            f"different year."
        )
    raster_path = matches[0]
    log_fn(f"Loading population from {raster_path.name}...")

    # Bug fix, confirmed by direct reproduction: cell geometries are built
    # in the grid's native CRS (ESRI:102028, meters), but WorldPop rasters
    # are distributed in EPSG:4326 (degrees). rasterstats.zonal_stats()
    # does NOT reproject its input geometry to match the raster -- it
    # assumes the geometry is already in the raster's own CRS. Passing
    # meter-scale coordinates straight through was being misread as
    # degree-scale coordinates, producing a nonsensical, enormous
    # bounding window and the MemoryError seen in testing (reproduced
    # directly: a single 1km cell's geometry, passed unprojected,
    # triggered an attempted ~37 GiB array allocation). Fix: read the
    # raster's own CRS and reproject every cell geometry to match it
    # before calling zonal_stats, rather than assuming any particular CRS.
    import rasterio
    with rasterio.open(raster_path) as src:
        raster_crs = src.crs

    local_geoms = [_local_geometry_native(row.grid_x_index, row.grid_y_index, grid_crs) for row in cells_df.itertuples()]
    nbhd_geoms = [_neighborhood_geometry_native(row.grid_x_index, row.grid_y_index, grid_crs) for row in cells_df.itertuples()]
    local_geoms_reprojected = gpd.GeoSeries(local_geoms, crs=grid_crs).to_crs(raster_crs)
    nbhd_geoms_reprojected = gpd.GeoSeries(nbhd_geoms, crs=grid_crs).to_crs(raster_crs)

    # Batched into ONE zonal_stats call per scope (not one call per cell)
    # -- also fixes an efficiency problem in the original code, which
    # reopened the raster file once per cell.
    local_stats = rasterstats.zonal_stats(local_geoms_reprojected, str(raster_path), stats="mean", geojson_out=False)
    nbhd_stats = rasterstats.zonal_stats(nbhd_geoms_reprojected, str(raster_path), stats="mean", geojson_out=False)

    rows = []
    for i, row in enumerate(cells_df.itertuples()):
        rows.append({
            "grid_id": row.grid_id,
            "population_local": local_stats[i]["mean"],
            "population_neighborhood_mean": nbhd_stats[i]["mean"],
        })
    return pd.DataFrame(rows)


def load_coastline_distance(cells_df: pd.DataFrame, grid_crs, log_fn=print) -> pd.DataFrame:
    """Local-only, per config.settings.COASTLINE_PATH.

    BUG FIX, confirmed directly from extract_distance_to_coast.py's own
    output columns (distance_to_coast_km) and unit ("dist_m / 1000.0"):
    this is NOT a local/neighborhood-split feature the way the other
    static predictors are -- the model expects exactly ONE value per
    cell, in kilometers, named "distance_to_coast_km". The previous
    version of this function invented a local/neighborhood split and
    left the value in meters, neither of which matches what the model
    was actually trained on."""
    if not COASTLINE_PATH.exists():
        raise StaticPredictorError(
            f"Coastline file not found at {COASTLINE_PATH}. See data/local_datasets/coastline/README.txt."
        )
    log_fn("Loading distance to coast from ne_10m_coastline.gpkg...")
    coastline = gpd.read_file(COASTLINE_PATH).to_crs(grid_crs)
    coastline_union = coastline.geometry.union_all()

    rows = []
    for row in cells_df.itertuples():
        local_geom = _local_geometry_native(row.grid_x_index, row.grid_y_index, grid_crs)
        dist_m = local_geom.centroid.distance(coastline_union)
        rows.append({"grid_id": row.grid_id, "distance_to_coast_km": dist_m / 1000.0})
    return pd.DataFrame(rows)


def load_road_density(cells_df: pd.DataFrame, log_fn=print) -> pd.DataFrame:
    """Reads from the REQUIRED precomputed cache -- see module docstring
    for why this isn't computed live from the .pbf file per request.
    Fails clearly (naming the missing cells) if any requested cell is
    absent from the cache, rather than substituting 0 or an average.

    BUG FIX, confirmed directly from prepare_features.py's own
    COLUMNS_TO_DROP_FROM_FEATURES_EXACT comment: the model does NOT
    expect "road_density_local" at all -- it's mathematically identical
    to road_length_local_m (a 1km cell's own area is always exactly
    1 km^2, so dividing length by area changes nothing) and was dropped
    from the final feature set during training. The model DOES expect
    road_length_local_m, road_length_neighborhood_m, and
    road_density_neighborhood_m_per_km2 (neighborhood area is 25 km^2,
    so density and length genuinely differ there) as separate features.
    Renamed here at load time (not by regenerating the cache) so a cache
    already produced by precompute_road_density_full_grid.py's earlier
    column names still works without rerunning that script."""
    if not ROAD_DENSITY_CACHE_PATH.exists():
        raise StaticPredictorError(
            f"Road density cache not found at {ROAD_DENSITY_CACHE_PATH}. This must be "
            f"precomputed once using your existing, validated road-density extraction "
            f"script run against the FULL prediction grid -- see the README's "
            f"'Road density precomputation' section for the exact command. This is not "
            f"computed live per request."
        )
    cache = pd.read_csv(ROAD_DENSITY_CACHE_PATH)
    cache = cache.rename(columns={"road_density_neighborhood_mean": "road_density_neighborhood_m_per_km2"})
    required_cols = {"grid_id", "road_length_local_m", "road_length_neighborhood_m", "road_density_neighborhood_m_per_km2"}
    if not required_cols.issubset(cache.columns):
        raise StaticPredictorError(
            f"{ROAD_DENSITY_CACHE_PATH} is missing required column(s): "
            f"{required_cols - set(cache.columns)}."
        )

    result = cells_df[["grid_id"]].merge(cache[list(required_cols)], on="grid_id", how="left")
    missing = result[result["road_length_local_m"].isna()]["grid_id"].tolist()
    if missing:
        raise StaticPredictorError(
            f"Road density cache is missing {len(missing)} requested cell(s): {missing}. "
            f"Re-run the precomputation script to cover the full prediction grid, or "
            f"select different cells. Refusing to substitute a default value."
        )
    log_fn(f"Loaded road density from cache for {len(result)} cell(s).")
    return result


def _local_geometry_native(grid_x, grid_y, grid_crs):
    import shapely.geometry
    x0, y0 = grid_x * CELL_SIDE_METERS, grid_y * CELL_SIDE_METERS
    return shapely.geometry.box(x0, y0, x0 + CELL_SIDE_METERS, y0 + CELL_SIDE_METERS)


def _neighborhood_geometry_native(grid_x, grid_y, grid_crs):
    import shapely.geometry
    x0 = (grid_x - NEIGHBORHOOD_RADIUS_CELLS) * CELL_SIDE_METERS
    y0 = (grid_y - NEIGHBORHOOD_RADIUS_CELLS) * CELL_SIDE_METERS
    x1 = (grid_x + NEIGHBORHOOD_RADIUS_CELLS + 1) * CELL_SIDE_METERS
    y1 = (grid_y + NEIGHBORHOOD_RADIUS_CELLS + 1) * CELL_SIDE_METERS
    return shapely.geometry.box(x0, y0, x1, y1)
