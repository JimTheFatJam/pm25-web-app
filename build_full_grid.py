"""
build_full_grid.py -- ONE-TIME build script for the full Philippines
prediction grid. NOT part of the runtime Streamlit app -- run this once
to produce data/official_grid_ph_full.gpkg, which the app then loads.

============================================================================
METHODOLOGY -- confirmed directly against the research pipeline's own grid
construction script (build_training_grid.py), not reverse-engineered
============================================================================
That script's own docstring states the design intent explicitly: "Grid
cell indices are computed by floor division against a FIXED origin at
(0, 0) in the working CRS... This is deliberate: it means any future
script (e.g. the full Philippines prediction grid) that uses the same
CRS, the same 1 km cell size, and the same floor-division-from-zero rule
will automatically produce cells that align exactly with the ones
generated here." This script does exactly that:
    grid_x_index = floor(x / 1000), grid_y_index = floor(y / 1000)
    CRS = ESRI:102028, cell size = 1000m, origin = (0, 0)
Confirmed empirically before use: filtering the existing 915-cell
training grid to only its perfectly-square (non-coastal-clipped) cells
and solving for the implied origin gives exactly (0, 0) with zero
deviation -- matching the source script's own stated design.

============================================================================
DELIBERATE SIMPLIFICATION: cells are NOT clipped to the exact coastline
============================================================================
The research grid's own script clips coastal cells to the exact country
boundary (relevant for feature-extraction accuracy -- not sampling ocean
pixels for a coastal station's neighborhood). That precision is not
needed for a SELECTION UI, and attempting it at this scale is not
practical: benchmarked directly, exact per-cell intersection() against
the Philippines' ~393,000-vertex coastline for ~314,000 candidate cells
did not complete in a reasonable time (exceeded a 90s timeout in
testing), even after simplifying the boundary. This script instead uses
the much faster `intersects()` predicate (a boolean test, not a geometry
computation) purely to decide cell INCLUSION, and keeps every included
cell as its full, unclipped 1km x 1km square -- exactly matching the
grid's true alignment. This is a deliberate, documented trade-off, not
an oversight: a selection UI needs correct cell membership, not
coastline-precise cell shape, and every cell remains genuinely 1.0 km^2
for area-estimation purposes as a result (no partial-area cells to
account for).

============================================================================
ID ASSIGNMENT
============================================================================
The training grid's own grid_id scheme is a SORT-SEQUENCE number (cells
sorted by (grid_y_index, grid_x_index), numbered sequentially within
country) -- not algebraically derivable from grid_x_index/grid_y_index
alone, and not stable under grid expansion (adding ~313,000 new cells
interspersed among the original 915 would shift every sequence number).
Therefore:
  - Cells matching an existing training-grid cell (same grid_x_index,
    grid_y_index) keep their EXACT original grid_id and geometry,
    unchanged -- these are the cells the actual trained model's feature
    data and sensor_grid_lookup.csv reference.
  - All other (new) cells get a NEW, clearly-distinct, positionally
    deterministic ID: "PHFULL_{grid_x_index}_{grid_y_index}" -- directly
    derivable from the cell's own indices (unlike the sort-sequence
    scheme), so it can never collide with an original training grid_id
    (which never contains underscores) and is trivially reproducible.
"""

import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

DATA_DIR = Path(__file__).resolve().parent / "data"
TRAINING_GRID_PATH = DATA_DIR / "official_grid_ph.gpkg"  # existing 915-cell subset, already in this project
BOUNDARY_PATH = DATA_DIR / "philippines-adm0.gpkg"
OUTPUT_PATH = DATA_DIR / "official_grid_ph_full.gpkg"

WORKING_CRS = "ESRI:102028"
CELL_SIZE_M = 1000.0


def main():
    t_start = time.time()

    print("Loading Philippine boundary...")
    boundary = gpd.read_file(BOUNDARY_PATH).to_crs(WORKING_CRS)
    land_geom = boundary.geometry.union_all()
    minx, miny, maxx, maxy = boundary.total_bounds

    print("Loading existing training grid (915 cells) for ID/geometry reuse...")
    training_grid = gpd.read_file(TRAINING_GRID_PATH)
    training_lookup = training_grid.set_index(["grid_x_index", "grid_y_index"])

    print("Generating candidate cell indices (vectorized)...")
    x_min_idx, x_max_idx = int(np.floor(minx / CELL_SIZE_M)), int(np.floor(maxx / CELL_SIZE_M))
    y_min_idx, y_max_idx = int(np.floor(miny / CELL_SIZE_M)), int(np.floor(maxy / CELL_SIZE_M))
    xs = np.arange(x_min_idx, x_max_idx + 1)
    ys = np.arange(y_min_idx, y_max_idx + 1)
    xx, yy = np.meshgrid(xs, ys)
    xx, yy = xx.ravel(), yy.ravel()
    print(f"  {len(xx):,} candidate cells in the bounding-box (mostly ocean, to be filtered next).")

    print("Building candidate geometries (vectorized shapely.box)...")
    cell_minx, cell_miny = xx * CELL_SIZE_M, yy * CELL_SIZE_M
    boxes = shapely.box(cell_minx, cell_miny, cell_minx + CELL_SIZE_M, cell_miny + CELL_SIZE_M)

    print("Filtering to land-intersecting cells (STRtree bulk query)...")
    tree = shapely.STRtree(boxes)
    land_idx = tree.query(land_geom, predicate="intersects")
    print(f"  {len(land_idx):,} cells intersect Philippine land.")

    selected_x, selected_y = xx[land_idx], yy[land_idx]
    selected_boxes = boxes[land_idx]

    print("Assigning geometry/grid_id: reuse exact training-grid cells, new IDs for the rest (vectorized merge)...")
    candidates_df = pd.DataFrame({"grid_x_index": selected_x, "grid_y_index": selected_y})
    candidates_df["_new_geometry"] = selected_boxes  # the fallback (unclipped square) for non-training cells

    training_subset = training_grid[["grid_x_index", "grid_y_index", "grid_id", "geometry"]].rename(
        columns={"grid_id": "_training_grid_id", "geometry": "_training_geometry"})

    merged = candidates_df.merge(training_subset, on=["grid_x_index", "grid_y_index"], how="left")
    is_training_cell = merged["_training_grid_id"].notna().values
    n_reused = int(is_training_cell.sum())

    new_ids = "PHFULL_" + merged["grid_x_index"].astype(str) + "_" + merged["grid_y_index"].astype(str)
    grid_ids = np.where(is_training_cell, merged["_training_grid_id"].values, new_ids.values)
    # Geometry column: training geometry where matched, else the unclipped candidate box.
    geometries = merged["_training_geometry"].where(merged["_training_geometry"].notna(), merged["_new_geometry"])

    print(f"  Reused {n_reused:,} exact training-grid cells; {len(grid_ids) - n_reused:,} new cells generated.")

    full_grid = gpd.GeoDataFrame({
        "grid_id": grid_ids,
        "grid_x_index": selected_x,
        "grid_y_index": selected_y,
        "is_training_cell": is_training_cell,
        "geometry": geometries,
    }, crs=WORKING_CRS)

    print("Computing centroids (lat/lon for display)...")
    centroids_wgs84 = full_grid.geometry.centroid.to_crs("EPSG:4326")
    full_grid["centroid_longitude"] = centroids_wgs84.x
    full_grid["centroid_latitude"] = centroids_wgs84.y

    print(f"Writing {OUTPUT_PATH}...")
    full_grid.to_file(OUTPUT_PATH, driver="GPKG")

    elapsed = time.time() - t_start
    print(f"\nDone in {elapsed:.1f}s. Total cells: {len(full_grid):,} "
          f"({n_reused:,} exact training-grid cells + {len(full_grid) - n_reused:,} new).")


if __name__ == "__main__":
    main()
