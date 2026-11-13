"""
tests/test_cell_selector.py -- Required spatial behavior tests.

Uses a small, synthetic 3x3 grid of exact 1km cells in a simple
projected CRS (not the full 915-cell official grid) so each test is
fast, deterministic, and isolates exactly the geometric behavior being
checked.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import geopandas as gpd
from shapely.geometry import Polygon, box

from selection.cell_selector import select_cells, InvalidPolygonError, compute_selection_area_km2

TEST_CRS = "EPSG:3857"  # any projected CRS in meters is fine for these synthetic geometry tests


def make_test_grid():
    """3x3 grid of 1000m cells, origin at (0,0), grid_id like row-col."""
    cells = []
    for i in range(3):
        for j in range(3):
            cells.append({
                "grid_id": f"TEST{i}{j}",
                "grid_x_index": i, "grid_y_index": j,
                "geometry": box(i * 1000, j * 1000, (i + 1) * 1000, (j + 1) * 1000),
            })
    return gpd.GeoDataFrame(cells, crs=TEST_CRS)


def test_1_polygon_completely_inside_a_cell():
    grid = make_test_grid()
    # Small polygon entirely inside cell (0,0) [bounds 0-1000, 0-1000]
    polygon = Polygon([(100, 100), (200, 100), (200, 200), (100, 200)])
    result = select_cells(polygon, grid, TEST_CRS)
    assert set(result["grid_id"]) == {"TEST00"}, f"Expected only TEST00 selected, got {set(result['grid_id'])}"
    print("Test 1 (polygon inside a cell): PASS")


def test_2_polygon_crosses_cell_edge():
    grid = make_test_grid()
    # Polygon straddling the boundary between cell (0,0) and (1,0) at x=1000
    polygon = Polygon([(900, 100), (1100, 100), (1100, 200), (900, 200)])
    result = select_cells(polygon, grid, TEST_CRS)
    assert {"TEST00", "TEST10"}.issubset(set(result["grid_id"])), f"Expected both cells selected, got {set(result['grid_id'])}"
    print("Test 2 (polygon crosses cell edge): PASS")


def test_3_polygon_touches_cell_boundary_only():
    grid = make_test_grid()
    # A polygon that only touches the shared edge at x=1000 (does not cross into cell (1,0))
    # Line-like touching polygon: a thin sliver whose right edge sits exactly on x=1000.
    polygon = Polygon([(950, 100), (1000, 100), (1000, 200), (950, 200)])
    result = select_cells(polygon, grid, TEST_CRS)
    assert "TEST00" in set(result["grid_id"]), "Expected TEST00 (the cell being touched) selected"
    # The cell on the OTHER side of the touched boundary (TEST10) should also register as
    # touching, since its left edge is exactly the same line at x=1000.
    assert "TEST10" in set(result["grid_id"]), f"Expected TEST10 (touched via shared boundary) also selected, got {set(result['grid_id'])}"
    print("Test 3 (polygon touches cell boundary): PASS")


def test_4_polygon_does_not_intersect_a_cell():
    grid = make_test_grid()
    # Polygon far outside the grid entirely
    polygon = Polygon([(10000, 10000), (10100, 10000), (10100, 10100), (10000, 10100)])
    result = select_cells(polygon, grid, TEST_CRS)
    assert len(result) == 0, f"Expected no cells selected, got {set(result['grid_id'])}"
    print("Test 4 (polygon does not intersect any cell): PASS")


def test_5_polygon_overlaps_several_cells():
    grid = make_test_grid()
    # Polygon covering the full 3x3 grid extent
    polygon = Polygon([(0, 0), (3000, 0), (3000, 3000), (0, 3000)])
    result = select_cells(polygon, grid, TEST_CRS)
    assert len(result) == 9, f"Expected all 9 cells selected, got {len(result)}"
    print("Test 5 (polygon overlaps several cells): PASS")


def test_6_selection_remains_aligned_with_fixed_grid():
    grid = make_test_grid()
    polygon = Polygon([(100, 100), (200, 100), (200, 200), (100, 200)])
    result = select_cells(polygon, grid, TEST_CRS)
    # Selected cell's geometry must be byte-identical to the grid's own
    # stored geometry -- selection must never construct or approximate
    # a new cell shape, only select existing ones by reference.
    original_cell = grid[grid["grid_id"] == "TEST00"].geometry.iloc[0]
    selected_cell = result[result["grid_id"] == "TEST00"].geometry.iloc[0]
    assert original_cell.equals(selected_cell), "Selected cell geometry does not match the fixed grid's own geometry"
    print("Test 6 (selection remains aligned with fixed grid): PASS")


def test_7_invalid_polygon_handling():
    grid = make_test_grid()
    # Self-intersecting "bowtie" polygon
    try:
        bowtie = Polygon([(0, 0), (1000, 1000), (1000, 0), (0, 1000)])
        if bowtie.is_valid:
            print("Test 7 (invalid polygon handling): SKIPPED -- shapely auto-corrected the bowtie to a valid geometry")
            return
        select_cells(bowtie, grid, TEST_CRS)
        assert False, "Expected InvalidPolygonError for a self-intersecting polygon"
    except InvalidPolygonError:
        print("Test 7 (invalid polygon handling): PASS")


def test_area_computation_is_cell_based_not_raw_polygon_area():
    grid = make_test_grid()
    # A polygon with an irregular, non-1000-aligned raw area, selecting exactly 4 cells
    polygon = Polygon([(50, 50), (1950, 50), (1950, 1950), (50, 1950)])
    result = select_cells(polygon, grid, TEST_CRS)
    area = compute_selection_area_km2(result, cell_area_km2=1.0)
    assert area == len(result) * 1.0, "Area must be cell-count based, not the raw polygon area"
    print(f"Area computation test: PASS ({len(result)} cells -> {area} km^2, not the polygon's own irregular area)")


if __name__ == "__main__":
    test_1_polygon_completely_inside_a_cell()
    test_2_polygon_crosses_cell_edge()
    test_3_polygon_touches_cell_boundary_only()
    test_4_polygon_does_not_intersect_a_cell()
    test_5_polygon_overlaps_several_cells()
    test_6_selection_remains_aligned_with_fixed_grid()
    test_7_invalid_polygon_handling()
    test_area_computation_is_cell_based_not_raw_polygon_area()
    print("\nAll tests passed.")
