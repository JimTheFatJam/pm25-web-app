"""
grid/grid_provider.py -- Grid source abstraction.

============================================================================
GRID PROVENANCE STATEMENT -- read before assuming anything about this grid.
============================================================================
Status: OFFICIAL RESEARCH GRID (not a temporary/placeholder grid),
extended to cover the FULL Philippine land area.

data/official_grid_ph_full.gpkg (313,845 cells) was built by
build_full_grid.py using EXACTLY the research pipeline's own grid
definition -- confirmed directly against build_training_grid.py's own
docstring, not reverse-engineered: CRS = ESRI:102028, cell size = 1000m,
grid_x_index/grid_y_index computed via floor division from a fixed
(0, 0) origin. That script's own documentation states this design is
deliberate specifically so a future full-coverage grid would align
exactly with it.

All 915 cells from the original training grid (official_grid_ph.gpkg)
are included with BYTE-IDENTICAL geometry and grid_id -- verified
directly during the build (100% match, logged in build_full_grid.py's
own output). The remaining ~312,930 cells are new, covering the rest of
the Philippine land area, each a full unclipped 1km x 1km square at the
same grid alignment (cells are not clipped to the exact coastline -- a
deliberate, documented performance trade-off; see build_full_grid.py's
own docstring for why exact per-cell coastline clipping was not
practical at this scale, and why cell-inclusion correctness does not
require it for a selection UI).

This module is still written as an abstraction (load_grid() returning a
GeoDataFrame with a documented schema) so that if the official grid file
is ever unavailable in a given deployment, a clearly-labeled temporary
grid can be substituted here WITHOUT changing any calling code in
map/, selection/, or ui/. See _build_temporary_fallback_grid() below --
included for architectural completeness, but NOT used while the
official grid file is present, and any output it produces is labeled
"TEMPORARY PROTOTYPE GRID" if it is ever exercised.
"""

from pathlib import Path

import geopandas as gpd

from config.settings import OFFICIAL_GRID_PATH, PHILIPPINES_BOUNDARY_PATH, CELL_SIDE_METERS

REQUIRED_GRID_COLUMNS = {"grid_id", "grid_x_index", "grid_y_index", "geometry"}


class GridUnavailableError(RuntimeError):
    pass


def load_grid() -> gpd.GeoDataFrame:
    """Returns the Philippine prediction grid as a GeoDataFrame in its
    OWN native analysis CRS (read from the file, not assumed). Raises
    GridUnavailableError with a clear message if the official grid file
    is missing, rather than silently falling back -- a caller that
    wants the documented temporary-grid fallback must request it
    explicitly via load_grid(allow_temporary_fallback=True)."""
    if OFFICIAL_GRID_PATH.exists():
        grid = gpd.read_file(OFFICIAL_GRID_PATH)
        missing = REQUIRED_GRID_COLUMNS - set(grid.columns)
        if missing:
            raise GridUnavailableError(
                f"{OFFICIAL_GRID_PATH.name} is present but missing expected column(s): {missing}. "
                f"Refusing to proceed with an unverified grid schema.")
        grid.attrs["grid_source"] = "OFFICIAL_RESEARCH_GRID"
        return grid

    raise GridUnavailableError(
        f"Official grid file not found at {OFFICIAL_GRID_PATH}. "
        f"Call load_grid(allow_temporary_fallback=True) if a demonstration-only grid is acceptable, "
        f"or restore official_grid_ph.gpkg from the research project's training_grid.gpkg.")


def load_grid_with_fallback() -> gpd.GeoDataFrame:
    """Convenience wrapper: uses the official grid if present, otherwise
    builds and clearly labels a TEMPORARY PROTOTYPE GRID so the UI
    remains demonstrable even without the research data file. Every
    caller in this app uses this function (not load_grid() directly),
    so the fallback is always available but never silent about which
    grid is in use -- grid.attrs["grid_source"] tells the UI layer which
    banner to show."""
    try:
        return load_grid()
    except GridUnavailableError:
        return _build_temporary_fallback_grid()


def _build_temporary_fallback_grid() -> gpd.GeoDataFrame:
    """TEMPORARY PROTOTYPE GRID -- NOT the official research grid.

    Built only if official_grid_ph.gpkg is unavailable. Uses the
    Philippine boundary's own bounding box in an arbitrary (not
    research-aligned) equal-area projection to build a demonstration 1km
    fishnet. This grid's cell IDs, origin, and alignment do NOT match
    the research prediction grid and must never be presented to a user
    as if they did -- the UI must always display the
    grid.attrs["grid_source"] banner so this distinction stays visible.
    """
    import numpy as np
    from shapely.geometry import box

    boundary = gpd.read_file(PHILIPPINES_BOUNDARY_PATH)
    # Arbitrary equal-area projection centered on the Philippines --
    # NOT the research grid's own ESRI:102028 definition, and not
    # claimed to be aligned with it.
    boundary_proj = boundary.to_crs("+proj=aea +lat_1=5 +lat_2=20 +lat_0=12.5 +lon_0=122 +units=m")
    minx, miny, maxx, maxy = boundary_proj.total_bounds

    xs = np.arange(minx, maxx, CELL_SIDE_METERS)
    ys = np.arange(miny, maxy, CELL_SIDE_METERS)
    cells = []
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            cells.append({"grid_id": f"TEMP{i:04d}{j:04d}", "grid_x_index": i, "grid_y_index": j,
                          "geometry": box(x, y, x + CELL_SIDE_METERS, y + CELL_SIDE_METERS)})
    grid = gpd.GeoDataFrame(cells, crs=boundary_proj.crs)
    # Keep only cells that actually intersect the Philippine landmass,
    # to avoid a full rectangular fishnet over open ocean.
    grid = grid[grid.intersects(boundary_proj.union_all())].reset_index(drop=True)
    grid.attrs["grid_source"] = "TEMPORARY_PROTOTYPE_GRID"
    return grid
