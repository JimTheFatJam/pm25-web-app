"""
backend/cell_geometry.py -- Exact 1km cell corner/centroid geometry in
EPSG:4326 for a set of selected grid rows.

Moved here verbatim from backend/app.py's /api/select_cells_grid route so
the SAME construction serves both that route and the prediction export
(backend/export.py) -- one code path, so exported polygons are always
identical to the ones drawn on the map. Kept out of app.py because that
module runs the full startup sequence (grid load, GEE init, model load)
at import time and therefore cannot be imported by unit tests.

Cell bounds are built directly from grid_x_index/grid_y_index (the same
indices the official grid is defined by -- see grid/grid_provider.py's
provenance notes), not from the cell's own stored geometry, so every cell
is the full, precise 1000m square its indices define. Reprojecting the
true corner points (rather than approximating an axis-aligned lat/lon
rectangle) guarantees adjacent cells still share exact boundaries after
reprojection -- see the original route's docstring in app.py for the
gap/overlap bug this avoids.
"""

import shapely
import geopandas as gpd

from config.settings import CELL_SIDE_METERS, DISPLAY_CRS


def cell_boxes_native(selected: gpd.GeoDataFrame):
    """Full 1km squares in the grid's native CRS, one per row of
    `selected` (which must carry grid_x_index / grid_y_index)."""
    minx = selected["grid_x_index"].to_numpy(dtype=float) * CELL_SIDE_METERS
    miny = selected["grid_y_index"].to_numpy(dtype=float) * CELL_SIDE_METERS
    return shapely.box(minx, miny, minx + CELL_SIDE_METERS, miny + CELL_SIDE_METERS)


def cell_corners_wgs84(selected: gpd.GeoDataFrame, grid_crs):
    """Returns (lons_per_cell, lats_per_cell): two lists, each with one
    4-element list per selected cell (the 4 corners, closing duplicate
    dropped), rounded to 6 decimals -- the exact contract
    /api/select_cells_grid has always returned.

    Vectorized (shapely.get_coordinates over the whole GeoSeries), not a
    per-cell Python loop -- benchmarked at ~1s for a 62,191-cell selection.
    """
    if len(selected) == 0:
        return [], []
    boxes_gs = gpd.GeoSeries(cell_boxes_native(selected), crs=grid_crs)
    boxes_wgs84 = boxes_gs.to_crs(DISPLAY_CRS)

    coords_flat, _ = shapely.get_coordinates(boxes_wgs84.to_numpy(), return_index=True)
    n_cells = len(boxes_wgs84)
    coords_reshaped = coords_flat.reshape(n_cells, 5, 2)  # 4 corners + closing duplicate
    lons_per_cell = coords_reshaped[:, :4, 0].round(6).tolist()
    lats_per_cell = coords_reshaped[:, :4, 1].round(6).tolist()
    return lons_per_cell, lats_per_cell


def cell_centroids_wgs84(selected: gpd.GeoDataFrame, grid_crs):
    """Returns (lats, lons) lists of each full-square cell's centroid,
    computed in the native (equal-area, metric) CRS and then reprojected
    -- never averaged in lat/lon degrees. Rounded to 6 decimals."""
    if len(selected) == 0:
        return [], []
    boxes_gs = gpd.GeoSeries(cell_boxes_native(selected), crs=grid_crs)
    centroids = boxes_gs.centroid.to_crs(DISPLAY_CRS)
    return centroids.y.round(6).tolist(), centroids.x.round(6).tolist()
