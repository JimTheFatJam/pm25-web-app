"""
selection/cell_selector.py -- Polygon -> grid cell intersection.

Selection rule (per explicit specification): a cell is selected if it
is inside the polygon OR intersects/touches the polygon boundary. This
is a genuine geometric intersection test (shapely .intersects(), which
covers overlap, containment, AND boundary-touching), NOT a
centroid-in-polygon test -- .intersects() returns True for a cell that
only shares a boundary edge or corner with the polygon, matching the
explicit requirement that boundary-touched cells must be selected.
"""

import geopandas as gpd
from shapely.geometry import Polygon
from shapely.errors import ShapelyError


class InvalidPolygonError(ValueError):
    pass


def select_cells(polygon: Polygon, grid: gpd.GeoDataFrame, polygon_crs: str) -> gpd.GeoDataFrame:
    """polygon: a shapely Polygon in polygon_crs (typically EPSG:4326,
    since that's what map-drawing tools produce). grid: GeoDataFrame in
    its own analysis CRS. Returns the subset of grid rows that intersect
    the polygon, with all geometric work done in the GRID's CRS (never
    in raw lat/lon degrees) -- the polygon is reprojected here, not the
    grid, so grid alignment is never perturbed by a reprojection
    round-trip."""
    if polygon is None or polygon.is_empty:
        raise InvalidPolygonError("No polygon provided.")
    if not polygon.is_valid:
        raise InvalidPolygonError("The drawn polygon is not a valid geometry (e.g. self-intersecting). Please redraw it.")

    polygon_gdf = gpd.GeoDataFrame(geometry=[polygon], crs=polygon_crs)
    polygon_in_grid_crs = polygon_gdf.to_crs(grid.crs).geometry.iloc[0]

    # Spatial index (grid.sindex, built automatically by GeoPandas)
    # narrows candidates via bounding-box first -- avoids testing full
    # geometry intersection against all 915 cells when only a handful
    # can possibly match, and scales correctly if the grid is later
    # swapped for a larger one.
    candidate_idx = list(grid.sindex.intersection(polygon_in_grid_crs.bounds))
    candidates = grid.iloc[candidate_idx]

    selected_mask = candidates.geometry.intersects(polygon_in_grid_crs)
    selected = candidates[selected_mask].copy()
    return selected


def clip_polygon_to_boundary(polygon: Polygon, boundary: gpd.GeoDataFrame, polygon_crs: str) -> tuple:
    """Returns (clipped_polygon_in_polygon_crs, was_clipped: bool).
    Per the documented Option A behavior (clip rather than reject): any
    portion of the drawn polygon outside the Philippine boundary is
    removed, and the caller is told whether clipping occurred so the UI
    can show the "area outside domain was excluded" message rather than
    silently trimming it."""
    if polygon is None or polygon.is_empty:
        raise InvalidPolygonError("No polygon provided.")

    polygon_gdf = gpd.GeoDataFrame(geometry=[polygon], crs=polygon_crs)
    boundary_in_polygon_crs = boundary.to_crs(polygon_crs)
    boundary_union = boundary_in_polygon_crs.geometry.union_all()

    original_area = polygon.area
    clipped = polygon.intersection(boundary_union)
    if clipped.is_empty:
        raise InvalidPolygonError("The selected area does not overlap the Philippine prediction domain at all.")

    was_clipped = clipped.area < original_area * 0.999999  # small tolerance for floating-point noise
    return clipped, was_clipped


def compute_selection_area_km2(selected_cells: gpd.GeoDataFrame, cell_area_km2: float) -> float:
    """Approximation, documented explicitly per instruction: area =
    (number of selected cells) x (nominal cell area), NOT the drawn
    polygon's own raw area. This is intentional -- the selection is
    defined by which grid cells are chosen, so the reported area should
    reflect the actual prediction domain that would be requested, not
    the freehand polygon's geometric area (which will always differ
    slightly since any drawn shape rarely aligns exactly with cell
    boundaries)."""
    return len(selected_cells) * cell_area_km2
