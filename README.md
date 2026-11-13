# PM2.5 Prediction System -- Web Frontend Prototype (Area Selection)

## 1. What this prototype is

A standalone, independently-runnable web frontend prototype demonstrating
the **geographic area-selection workflow** for the PM2.5 Prediction Web
Application: drawing a polygon over the Philippines, resolving it against
the fixed 1km research prediction grid, and previewing what a backend
prediction request would eventually look like.

**Architecture**: a custom HTML/CSS/JavaScript frontend (matching a
specific wireframe design) backed by a small local Flask API. The heavy
spatial work (313,845-cell grid, polygon intersection) runs server-side
in Python/GeoPandas -- exactly the same code from earlier in this
project's development, just re-exposed over HTTP instead of Streamlit
session state.

## 2. What it does

- Displays the Philippine boundary and the full research prediction grid's
  domain as a bounded, pannable/zoomable vector map -- no tile server, no
  "keeps loading" behavior.
- Lets you draw a polygon directly on the map.
- Selects every grid cell the polygon intersects or touches (not just
  cells whose centroid falls inside it), clipped to the Philippine domain.
- Shows a live selection summary (cell count, approximate area) and a
  full request preview (estimated prediction rows, backend status).
- Provides date/date-range and temporal-resolution controls.
- Matches the specified wireframe: browser-chrome mockup, top nav, AOI
  panel with floating map toolbar, control panel, and step-guide footer.

## 3. What it intentionally does NOT do

- Does not train, load, or run any machine-learning model.
- Does not call Google Earth Engine or OpenAQ.
- Does not perform PM2.5 prediction of any kind.
- Does not implement a real backend prediction API -- only the
  area-selection request contract.
- Does not import, copy, or depend on any research-pipeline Python module.
- Does not implement authentication, accounts, a database, or caching.
- Does not enforce a real maximum-prediction-rows limit yet (the hook
  exists; the actual number does not, since no backend has defined it).

## 4. Installation

```bash
pip install -r requirements.txt
```

## 5. Launching

```bash
python run_app.py
```

This starts the local Flask server and opens your default browser to
`http://127.0.0.1:5050`. No internet connection is required after
installation -- the map is a pre-shipped vector boundary (no tile
server), and D3.js is vendored locally in `frontend/vendor/` (not
loaded from a CDN).

## 6. Why the map is no longer tile-based -- and why it was "always loading"

The previous (Streamlit + Leaflet) version fetched live OpenStreetMap
tiles from the internet on every pan/zoom -- the direct cause of the
"keeps loading" behavior. This version replaces that entirely: the
Philippine boundary is rendered as a single SVG vector shape (D3.js,
vendored locally) from a pre-simplified, locally-shipped GeoJSON file.
Pan/zoom is mathematically constrained to the Philippine extent
(`translateExtent`/`scaleExtent`), so there is nothing to fetch and
nowhere to scroll away to.

## 7. How the grid is defined -- READ THIS CAREFULLY

**Status: OFFICIAL RESEARCH GRID**, covering the **full Philippine land
area** (313,845 cells), not a temporary/placeholder grid.

Built by `build_full_grid.py` using **exactly** the research pipeline's
own grid definition -- confirmed directly against `build_training_grid.py`'s
own docstring (its original author's stated intent), not
reverse-engineered:

- CRS: `ESRI:102028` (Asia South Albers Equal Area Conic), projected, meters.
- Cell size: 1000m x 1000m.
- Grid indices: `grid_x_index = floor(x / 1000)`, `grid_y_index = floor(y / 1000)`,
  from a fixed `(0, 0)` origin -- that script's docstring states this
  design is deliberate specifically so a future full-coverage grid would
  align exactly with it.

**All 915 cells from the original training grid are included with
byte-identical geometry and `grid_id`** -- verified directly during the
build. The remaining ~312,930 cells are new, covering the rest of the
Philippine land area at the same grid alignment, each a full unclipped
1km x 1km square (cells are not clipped to the exact coastline -- exact
per-cell intersection against the ~393,000-vertex coastline was
benchmarked directly and found impractical at this scale; see
`build_full_grid.py`'s own docstring for the full reasoning).

**Performance, benchmarked directly**:
- Full grid load: ~7.6s (once per backend process start).
- Polygon selection query against the full grid: **10-21 milliseconds**
  for realistic 50-150km polygons, via GeoPandas' spatial index.
- Polygon-to-boundary clipping (`clip_polygon_to_boundary`): a real
  performance bug was found and fixed during this version's development
  -- using the FULL-precision boundary (~393,000 vertices) for this
  per-request clip operation did not complete within 60 seconds for a
  realistic drawn polygon. Fixed by using a pre-simplified boundary
  (~500m tolerance, the same used for map display) for this specific
  check, reducing it to ~0.4 seconds. This is a UI-level "is this
  obviously outside the domain" check, not a precision requirement, so
  the small simplification tolerance is an accepted, documented
  trade-off -- see `backend/app.py`'s own comments.

To regenerate the full grid from scratch:

```bash
python build_full_grid.py
```

One-time, ~45-second precomputation step, not part of the running app.

## 8. How polygon-to-cell selection works

A cell is selected if it **intersects** the drawn polygon in any way --
fully inside, partially overlapping, or only touching the polygon's
boundary (Shapely's `.intersects()` predicate, not centroid-containment).
See `selection/cell_selector.py` and the 8 automated tests in
`tests/test_cell_selector.py`.

## 9. CRS decisions

- **Display CRS (`EPSG:4326`)**: used for the boundary and drawn-polygon
  coordinates in the browser.
- **Analysis CRS**: the grid's own native CRS (`ESRI:102028`), read from
  the grid file itself, never hardcoded twice. All polygon intersection
  and area calculations happen here, in meters -- never in raw lat/lon
  degrees.

## 10. How the future backend will connect

The frontend is designed around this eventual request shape (see
`selection/selection_state.py::to_backend_request_preview` and
`backend/app.py`'s `/api/generate_prediction`):

```json
{
    "cell_ids": ["PH000000001", "PH000000002"],
    "start_datetime": "2026-08-01T00:00:00+08:00",
    "end_datetime": "2026-08-01T23:00:00+08:00",
    "temporal_resolution": "hourly"
}
```

## 11. How prediction-row limits can eventually be added

`selection/request_validator.py::validate_prediction_request(...)` is the
single, isolated place this logic lives -- exposed via
`/api/validate_request`. Currently `config.settings.MAX_PREDICTION_ROWS`
is `None`, so the endpoint always returns `is_valid: true` with an
informational message. Once a backend defines a real limit, set
`MAX_PREDICTION_ROWS` in `config/settings.py` -- the enforcement branch
is already written.

## 12. Timezone

All user-facing date/time controls use **Philippine local time**
(`Asia/Manila`, UTC+8), matching the research project's established
convention.

## 13. Domain-boundary behavior

**Clip, not reject**: a drawn polygon extending outside the Philippine
boundary is clipped to the boundary before cell selection, and the UI
explicitly shows "Area outside the Philippine prediction domain was
excluded" -- confirmed working directly, visible in the app when drawing
a polygon that includes ocean.

## 14. Predefined regions

**Honest placeholder, not invented data**: no administrative
boundary dataset was available to this prototype, so the "Select
predefined region" dropdown is present (per the wireframe) but shows a
clear note explaining why it isn't functional yet, rather than being
backed by fabricated boundaries.

## 15. Project structure

```
pm25_web_app/
├── run_app.py                    # launcher -- starts Flask, opens browser
├── backend/
│   └── app.py                    # Flask API, reuses grid/selection modules directly
├── frontend/
│   ├── index.html                # wireframe-matched UI structure
│   ├── style.css
│   ├── map.js                    # Leaflet-based bounded map, drawing tool
│   ├── main.js                   # UI orchestration, backend API calls, theme toggle
│   ├── philippines_boundary_simplified.geojson
│   ├── icons/                    # icon/logo image assets -- see Section 16 below
│   └── vendor/                   # Leaflet, vendored locally (no CDN dependency)
├── grid/
│   └── grid_provider.py          # official grid loading + documented temporary fallback
├── selection/
│   ├── cell_selector.py          # polygon -> cell intersection, boundary clipping
│   ├── selection_state.py
│   └── request_validator.py
├── config/
│   └── settings.py
├── data/
│   ├── philippines-adm0.gpkg
│   ├── official_grid_ph.gpkg      # original 915-cell training subset
│   └── official_grid_ph_full.gpkg # full 313,845-cell grid (the one actually used)
├── build_full_grid.py            # one-time grid-generation script
├── tests/
│   └── test_cell_selector.py
├── requirements.txt
└── README.md
```

## 16. Icons and Logos

All icon and logo images used by the interface live in `frontend/icons/`.
This project no longer uses emoji characters for any UI icon -- every icon
is an `<img>` tag pointing at a file in this folder, so **you can replace
any icon by overwriting its file** without touching any HTML, CSS, or
JavaScript.

**Every asset must be square** (equal width and height -- e.g. 128x128,
256x256, 512x512). The shipped placeholders are 128x128 PNGs; any square
resolution works as a replacement, since the CSS sizes each icon down to
its actual display size regardless of the source file's resolution.

The icons currently shipped are simple placeholder graphics (generated
programmatically for this prototype, not final artwork) so the interface
looks complete out of the box. Replace any or all of them at any time by
overwriting the file at the same path/filename below -- no code changes
needed, and there is nothing else to update.

| Filename | Purpose |
| --- | --- |
| `logo.png` | Application logo, shown in the header |
| `pan_move_map.png` | "Pan / Move Map" toolbar tool |
| `how_to_use.png` | "How to Use" header button |
| `dark_mode.png` | Theme toggle icon, shown while in light mode (click to switch to dark) |
| `light_mode.png` | Theme toggle icon, shown while in dark mode (click to switch to light) |
| `prediction_resolution_fixed.png` | Lock indicator on the "Prediction Resolution (Fixed)" box |
| `draw_boundary.png` | "Draw Boundary" toolbar tool |
| `clear_selection.png` | "Clear Selection" toolbar tool |
| `recenter.png` | Map re-center button |
| `north_arrow.png` | North-arrow map indicator |
| `draw_area.png` | Icon in the "Draw your Area of Interest" callout |
| `info.png` | Icon in the single-day/date-range info callout |
| `warning.png` | Icon on the Philippine-domain warning message |
| `step_draw.png` | Step 1 icon in the How to Use modal |
| `step_calendar.png` | Step 2 icon in the How to Use modal |
| `step_generate.png` | Step 3 icon in the How to Use modal |

If a file listed above is ever missing, the browser will show its
filename as fallback alt text and a broken-image icon in that exact
spot -- deliberately, so a missing asset is obvious and easy to locate
and replace, rather than silently falling back to a different icon or
an emoji.

## 17. Testing

```bash
python tests/test_cell_selector.py
```

All 8 tests (7 required + 1 additional) pass, covering polygon-cell
intersection behavior including the critical boundary-touching case.


## 18. What was tested end-to-end (not just assumed)

Using a real headless browser (Playwright/Chromium) during development:
- Full draw-polygon-to-selection flow: clicking the Draw Boundary tool,
  placing polygon points, closing the shape, and confirming the backend
  correctly returns cell count/area and the UI updates.
- Date input -> request preview -> Generate button -> correct
  "not connected" prototype message.
- Zero failed network requests (confirming the CDN/tile dependency from
  the previous version is fully eliminated).

Two real bugs were found and fixed during this testing, not left for the
user to discover:
1. A GeoJSON ring-winding-order issue causing the Philippine boundary to
   render as a plain rectangle instead of the actual archipelago shape --
   root-caused by testing both winding directions directly and confirmed
   via the render output, not guessed.
2. The polygon-clipping performance bug described in Section 7.

## 19. What remains to be implemented

- The actual backend prediction service and REST API.
- A real `MAX_PREDICTION_ROWS` value, once defined by the backend.
- Predefined-area selection (province/region), pending a real
  administrative boundary dataset.
- The prediction-results screen (heatmap, cell-click PM2.5 detail, SHAP
  explanation) -- out of scope for this area-selection-only prototype.
- Mobile application, authentication, accounts, database, caching,
  production deployment -- all explicitly out of scope.
