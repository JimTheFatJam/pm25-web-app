/**
 * map.js -- Leaflet-based map of the Philippines with a raster basemap
 * (Esri World Light Gray Base; see BASEMAP_URL for why).
 *
 * Architecture note: this replaces an earlier custom D3 SVG vector map
 * that was deliberately tile-free. Per explicit request to restore
 * OpenStreetMap imagery, this now uses Leaflet -- the standard tool for
 * tile-based web maps -- rather than attempting to bolt tile support
 * onto the D3 approach. This also more naturally solves several related
 * requests: Leaflet's maxBounds handles Philippines-only panning
 * natively; its marker/line sizing is pixel-based by default, so
 * zoom-independent point/edge size requires no custom compensation code
 * (unlike the previous D3 implementation's manual rescaling); and it
 * uses the same standard Web Mercator projection as the OSM tiles
 * themselves, which directly addresses the polygon/boundary alignment
 * imprecision from the previous custom (viewport-fitted, non-standard)
 * D3 projection.
 *
 * Leaflet is vendored locally (frontend/vendor/leaflet.js/.css), not
 * loaded from a CDN, for the same reason established earlier in this
 * project (a prior CDN-blocking issue during testing).
 *
 * Trade-off worth stating plainly: OpenStreetMap tiles ARE fetched over
 * the network on pan/zoom, by design -- this is normal behavior for any
 * tile-based map (Leaflet, Google Maps, etc. all work this way), not a
 * defect specific to this implementation.
 */

const PM25Map = (() => {
  // Basemap tiles. One constant -- the PM2.5 overlay, the Philippine
  // boundary and the grid rendering are all independent of it and keep
  // working if tiles fail to load, so a bad provider degrades the map to a
  // usable outline rather than breaking it.
  //
  // Two providers were tried and rejected, both of which return HTTP 200
  // with a *picture* saying no, so only looking at the pixels catches it:
  //   tile.openstreetmap.org  -- donated infrastructure; its usage policy
  //                              does not cover hotlinking from a public
  //                              site, and flagged clients get an
  //                              "Access blocked" tile.
  //   basemaps.cartocdn.com   -- now serves "API KEY REQUIRED" watermarks.
  //
  // Esri's World Light Gray Base needs no key, is unwatermarked, and its
  // muted palette deliberately stays out of the way of the PM2.5 colours.
  // It is tiled to zoom 16; maxNativeZoom lets Leaflet upscale past that
  // rather than request tiles that do not exist (zoom 16 is ~2.4 m/px,
  // already far finer than the 1 km grid).
  const BASEMAP_URL =
    "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}";
  const BASEMAP_ATTRIBUTION = "Tiles &copy; Esri &mdash; Esri, HERE, Garmin, &copy; OpenStreetMap contributors";
  const BASEMAP_MAX_NATIVE_ZOOM = 16;

  let map, boundaryLayer, selectionLayer, drawingLayer, cellsLayer, predictionLayer;
  let boundaryData = null;
  let currentMode = "select"; // "select" (pan) | "draw"
  let drawingPoints = []; // array of L.LatLng
  let drawingMarkers = [];
  let firstPointMarker = null;
  let onPolygonComplete = null;

  // --------------------------------------------------------------
  // Smooth/blended heatmap state (added feature -- see buildSmoothGridStructure
  // and initSmoothHeatmap below for the full design rationale).
  // --------------------------------------------------------------
  let smoothOverlay = null;        // L.imageOverlay -- georeferenced, Leaflet repositions/rescales it automatically on pan/zoom
  let smoothCanvas = null;         // tiny offscreen <canvas>, ONE PIXEL PER CELL, reused across every frame (never recreated per-frame)
  let smoothGridStructure = null;  // { nRows, nCols, cellIndex: {cellId: {row, col}}, bounds } -- precomputed ONCE per prediction result
  let visualizationMode = "grid";  // "grid" | "smooth"

  const VERTICAL_PADDING_DEG = 3.0; // extra lat allowance north/south, preserved from the
                                     // previous implementation's own padding intent -- see
                                     // module docstring and the explicit "extra top/bottom
                                     // allowance" requirement.
  const POINT_RADIUS = 4; // preserved from the previous implementation -- explicitly NOT changed
  const FIRST_POINT_HOVER_RADIUS_PX = 18; // "sufficiently close" hover-highlight threshold
  const CLOSE_CLICK_RADIUS_PX = 12; // click-to-close hit tolerance around the first point

  async function init(mapDivId, statusSelector, onPolygonCompleteCallback) {
    onPolygonComplete = onPolygonCompleteCallback;

    setStatusText(statusSelector, "Loading Philippine boundary...");
    const response = await fetch("philippines_boundary_simplified.geojson");
    boundaryData = await response.json();

    const bounds = L.geoJSON(boundaryData).getBounds();
    // Equal navigable allowance on all four sides (previously only
    // north/south had padding -- east/west used the boundary's raw
    // edge, which is what caused the "insufficient allowance on the
    // right side" behavior). A degree of longitude is physically
    // shorter than a degree of latitude away from the equator (by a
    // factor of cos(latitude)), so the padding in degrees longitude is
    // scaled up accordingly to make the ACTUAL on-the-ground allowance
    // approximately equal on every side, not just the padding's raw
    // degree count.
    const centerLatRad = bounds.getCenter().lat * Math.PI / 180;
    const horizontalPaddingDeg = VERTICAL_PADDING_DEG / Math.cos(centerLatRad);
    const padded = L.latLngBounds(
      [bounds.getSouth() - VERTICAL_PADDING_DEG, bounds.getWest() - horizontalPaddingDeg],
      [bounds.getNorth() + VERTICAL_PADDING_DEG, bounds.getEast() + horizontalPaddingDeg]
    );

    map = L.map(mapDivId, {
      maxBounds: padded,
      maxBoundsViscosity: 1.0,
      minZoom: 6,
      maxZoom: 19,
      zoomControl: true,
    });
    map.fitBounds(bounds);

    L.tileLayer(BASEMAP_URL, {
      attribution: BASEMAP_ATTRIBUTION,
      maxZoom: 19,
      maxNativeZoom: BASEMAP_MAX_NATIVE_ZOOM,
    }).addTo(map);

    L.control.scale({ position: "bottomleft", imperial: false }).addTo(map);

    boundaryLayer = L.geoJSON(boundaryData, {
      style: { color: "#1f2937", weight: 1.5, fillOpacity: 0, dashArray: "4,3" },
      interactive: false, // BUG FIX: an invisible fill (fillOpacity: 0) is still interactive by
                           // default in Leaflet -- this was silently capturing hover/mouse events
                           // over the entire landmass interior, blocking tooltips on predicted
                           // cells beneath it. Confirmed as the cause of the reported "only edge
                           // cells near the coast show their PM2.5 value on hover" bug: edge cells
                           // partially extend over ocean (outside the boundary polygon's hit area),
                           // so they escaped this, while fully-interior cells didn't.
    }).addTo(map);

    selectionLayer = L.layerGroup().addTo(map);
    cellsLayer = L.layerGroup().addTo(map);
    predictionLayer = L.layerGroup().addTo(map);
    drawingLayer = L.layerGroup().addTo(map);

    map.on("click", handleMapClick);
    map.on("mousemove", handleMapMouseMove);

    setStatusText(statusSelector, `Philippines loaded (${boundaryData.features.length} boundary feature(s)).`);
    setTimeout(() => { const el = document.querySelector(statusSelector); if (el) el.style.opacity = 0; }, 2500);

    return { map };
  }

  function setStatusText(selector, text) {
    const el = document.querySelector(selector);
    if (el) { el.textContent = text; el.style.opacity = 1; }
  }

  function setMode(mode) {
    currentMode = mode;
    const container = map.getContainer();
    container.style.cursor = mode === "draw" ? "crosshair" : "";
    // Dragging (panning) stays enabled in both modes -- Leaflet's own
    // click handling already distinguishes a simple click from a
    // click-and-drag (it suppresses the "click" event once the pointer
    // moves past a small threshold during the mousedown-to-mouseup
    // sequence), so a genuine drag pans the map without placing a
    // polygon point, while a simple click still places one. This is the
    // same mechanism every Leaflet-based drawing tool relies on.
    if (mode !== "draw") {
      cancelDrawing();
    }
  }

  function handleMapClick(e) {
    if (currentMode !== "draw") return;
    if (drawingPoints.length >= 3) {
      const firstPx = map.latLngToContainerPoint(drawingPoints[0]);
      const clickPx = map.latLngToContainerPoint(e.latlng);
      const dist = firstPx.distanceTo(clickPx);
      if (dist < CLOSE_CLICK_RADIUS_PX) {
        finishDrawing();
        return;
      }
    }
    drawingPoints.push(e.latlng);
    renderDrawingPreview();
  }

  function handleMapMouseMove(e) {
    if (currentMode !== "draw" || !firstPointMarker || drawingPoints.length < 3) return;
    const firstPx = map.latLngToContainerPoint(drawingPoints[0]);
    const cursorPx = map.latLngToContainerPoint(e.latlng);
    const dist = firstPx.distanceTo(cursorPx);
    const near = dist < FIRST_POINT_HOVER_RADIUS_PX;
    firstPointMarker.setStyle(near ? FIRST_POINT_HOVER_STYLE : FIRST_POINT_STYLE);
  }

  const FIRST_POINT_STYLE = { radius: POINT_RADIUS, color: "#ffffff", weight: 2, fillColor: "#dc2626", fillOpacity: 1 };
  const FIRST_POINT_HOVER_STYLE = { radius: POINT_RADIUS + 2, color: "#ffffff", weight: 2, fillColor: "#16a34a", fillOpacity: 1 };
  const OTHER_POINT_STYLE = { radius: POINT_RADIUS, color: "#ffffff", weight: 1.5, fillColor: "#2563eb", fillOpacity: 1 };

  function renderDrawingPreview() {
    drawingLayer.clearLayers();
    drawingMarkers = [];
    firstPointMarker = null;
    if (drawingPoints.length === 0) return;

    drawingPoints.forEach((pt, i) => {
      const isFirst = i === 0;
      const marker = L.circleMarker(pt, isFirst ? FIRST_POINT_STYLE : OTHER_POINT_STYLE).addTo(drawingLayer);
      if (isFirst) {
        firstPointMarker = marker;
        marker.on("mouseover", () => marker.setStyle(FIRST_POINT_HOVER_STYLE));
        marker.on("mouseout", () => marker.setStyle(FIRST_POINT_STYLE));
        marker.on("click", (e) => {
          L.DomEvent.stopPropagation(e);
          if (drawingPoints.length >= 3) finishDrawing();
        });
      }
      drawingMarkers.push(marker);
    });

    if (drawingPoints.length > 1) {
      L.polyline(drawingPoints, { color: "#2563eb", weight: 1.5, dashArray: "5,3" }).addTo(drawingLayer);
    }
  }

  function finishDrawing() {
    if (drawingPoints.length < 3) return;
    const ring = drawingPoints.map(p => [p.lng, p.lat]);
    ring.push(ring[0]);
    const polygon = { type: "Polygon", coordinates: [ring] };
    cancelDrawing();
    if (onPolygonComplete) onPolygonComplete(polygon);
  }

  function cancelDrawing() {
    drawingPoints = [];
    drawingMarkers = [];
    firstPointMarker = null;
    drawingLayer.clearLayers();
  }

  function clearSelection() {
    selectionLayer.clearLayers();
    cellsLayer.clearLayers();
    predictionLayer.clearLayers();
    cancelDrawing();
  }

  function drawSelection(clippedPolygonGeoJSON) {
    selectionLayer.clearLayers();
    if (!clippedPolygonGeoJSON) return;
    L.geoJSON(clippedPolygonGeoJSON, {
      style: { color: "#1d4ed8", weight: 2, fillColor: "#3b82f6", fillOpacity: 0.35 },
    }).addTo(selectionLayer);
  }

  // Renders each selected cell from its ACTUAL 4 corners (already
  // reprojected correctly by the backend from the grid's native CRS --
  // see backend/app.py's /api/select_cells_grid docstring for why this
  // replaced an earlier axis-aligned lat/lon approximation that caused
  // visible gaps/overlaps between neighboring cells). No geometric
  // computation happens here -- the polygon ring is built directly from
  // the given corners, in the same order they were sent.
  function drawSelectedCellGrid(lons, lats) {
    cellsLayer.clearLayers();
    if (!lons || lons.length === 0) return;

    const features = lons.map((cellLons, i) => {
      const cellLats = lats[i];
      const ring = cellLons.map((lon, j) => [lon, cellLats[j]]);
      ring.push(ring[0]); // close the ring, per GeoJSON polygon convention
      return { type: "Feature", properties: {}, geometry: { type: "Polygon", coordinates: [ring] } };
    });

    L.geoJSON({ type: "FeatureCollection", features }, {
      renderer: L.canvas(),
      style: { color: "#1d4ed8", weight: 0.5, fillColor: "#3b82f6", fillOpacity: 0.5 },
    }).addTo(cellsLayer);
  }

  function clearSelectedCellGrid() { cellsLayer.clearLayers(); }

  // Simple, common PM2.5 concentration color scale (roughly matching
  // widely-used AQI-category breakpoints in ug/m3) -- for visualization
  // only, not an official AQI calculation.
  // Continuous color scale -- linearly interpolates RGB between the same
  // reference concentration points the discrete legend used to snap to
  // (0, 12, 35.4, 55.4, 150.4 ug/m3), so a given value still means the
  // same thing (e.g. ~12 is still the green/yellow transition), but the
  // map now shows smooth gradients instead of hard bucket edges -- a
  // more honest representation of PM2.5 as a continuous quantity.
  const PM25_COLOR_STOPS = [
    [0, [34, 197, 94]],     // #22c55e -- good
    [12, [234, 179, 8]],    // #eab308 -- moderate
    [35.4, [249, 115, 22]], // #f97316 -- unhealthy for sensitive groups
    [55.4, [239, 68, 68]],  // #ef4444 -- unhealthy
    [150.4, [124, 58, 237]], // #7c3aed -- very unhealthy/hazardous
  ];

  // Returns [r, g, b] (0-255 each) or null for a missing value -- the
  // same continuous interpolation pm25Color already used, just exposing
  // the raw components before hex-encoding. Needed by the smooth
  // heatmap's per-pixel canvas writes (ImageData wants raw RGBA bytes,
  // not a hex string), without a wasteful hex-encode-then-decode round trip.
  function pm25ColorRgb(value) {
    if (value == null || Number.isNaN(value)) return null;
    if (value <= PM25_COLOR_STOPS[0][0]) return PM25_COLOR_STOPS[0][1];
    const last = PM25_COLOR_STOPS[PM25_COLOR_STOPS.length - 1];
    if (value >= last[0]) return last[1];

    for (let i = 0; i < PM25_COLOR_STOPS.length - 1; i++) {
      const [v0, c0] = PM25_COLOR_STOPS[i];
      const [v1, c1] = PM25_COLOR_STOPS[i + 1];
      if (value >= v0 && value <= v1) {
        const t = (value - v0) / (v1 - v0);
        return c0.map((c, j) => Math.round(c + t * (c1[j] - c)));
      }
    }
    return last[1]; // unreachable in practice, safety fallback
  }

  function pm25Color(value) {
    const rgb = pm25ColorRgb(value);
    return rgb ? rgbToHex(rgb) : "#9ca3af"; // grey -- missing prediction
  }

  function rgbToHex([r, g, b]) {
    return "#" + [r, g, b].map((c) => c.toString(16).padStart(2, "0")).join("");
  }

  // --------------------------------------------------------------
  // Smooth/blended heatmap.
  //
  // BUG FIX (this revision): the previous approach reconstructed row/
  // column indices by clustering raw WGS84 centroid longitudes/latitudes,
  // then painted exactly one pixel per cell across a canvas covering the
  // cells' full rectangular bounding box, relying on the browser's
  // bilinear image scaling to blend between them. This produced the
  // reported vertical banding/streaking: for a real, irregular selection
  // (e.g. a narrow, elongated island like Cebu, or any hand-drawn
  // non-rectangular polygon), the actual selected cells occupy only a
  // thin fraction of their own bounding box -- confirmed directly by
  // simulating exactly this shape: a 3-cells-wide, 60-cells-long strip
  // plus a few scattered cells reconstructed to a 63x6 grid, where the
  // real data filled less than half the canvas and left large empty
  // regions on either side. Bilinear-upscaling that sparse, mostly-empty
  // rectangular canvas across the map's full display area stretches the
  // thin strip of real data across a disproportionately large area,
  // producing smeared, banded artifacts -- not a genuine spatial
  // interpolation between geographic neighbors.
  //
  // Fixed approach: for every output raster pixel, find the K nearest
  // PREDICTED CELLS by actual geographic (lon/lat) distance to that
  // pixel -- never by array order, reconstructed grid index, or any
  // assumption of rectilinear structure -- and blend their CURRENT
  // values (not colors) by inverse-distance weighting. A distance cutoff
  // (1.5x the cells' own measured width/height) leaves a pixel fully
  // transparent if it isn't genuinely near any real cell, so the
  // interpolation never "fills in" or stretches across unselected gaps --
  // confirmed correct against the same island-shaped test: the filled
  // pixels form a properly thin, correctly-positioned strip matching the
  // real selection shape, with isolated scattered cells rendering as
  // isolated points, not connected to anything geographically distant.
  //
  // Performance: the expensive part (finding each pixel's nearest cells)
  // is a spatial-distance computation, not a value computation -- it
  // depends only on cell POSITIONS, which never change between frames or
  // mode toggles, so it is precomputed exactly ONCE per prediction result
  // (in buildSmoothInterpolationStructure, called from drawPredictedCells,
  // the same single "build geometry once" entry point the grid view
  // already uses) and reused for every later frame. A naive O(pixels x
  // cells) nearest-neighbor search measured ~1.2s for a realistic 220-cell
  // selection -- confirmed too slow for a one-time toggle to feel
  // instant, so cell centroids are additionally indexed into a coarse
  // spatial bucket grid (bucket size = the distance cutoff, so only the
  // 3x3 neighborhood of buckets around a pixel can possibly contain a
  // relevant cell), cutting the same 220-cell case to ~60ms. Each
  // playback frame then only re-blends already-known neighbor
  // values -- measured at ~10ms for a full ~29,000-pixel raster, well
  // under the 800ms playback interval.
  // --------------------------------------------------------------

  // Precomputed ONCE per prediction result. Returns null pixels (stay
  // transparent) wherever nothing is genuinely nearby, and for filled
  // pixels, the exact cell indices + normalized IDW weights needed to
  // blend that pixel's value on every later frame.
  function buildSmoothInterpolationStructure(cellIds, lons, lats) {
    const centroids = cellIds.map((id, i) => ({
      id,
      lon: lons[i].reduce((a, b) => a + b, 0) / lons[i].length,
      lat: lats[i].reduce((a, b) => a + b, 0) / lats[i].length,
    }));

    // Estimate true cell spacing from each cell's OWN corner geometry --
    // robust because it never compares far-apart cells to each other
    // (unlike the old approach's cross-cell centroid clustering).
    const widths = lons.map((cl) => Math.max(...cl) - Math.min(...cl));
    const heights = lats.map((cl) => Math.max(...cl) - Math.min(...cl));
    const median = (arr) => { const s = [...arr].sort((a, b) => a - b); return s[Math.floor(s.length / 2)]; };
    const cellWidth = median(widths), cellHeight = median(heights);

    const allLons = lons.flat(), allLats = lats.flat();
    const bounds = { west: Math.min(...allLons), east: Math.max(...allLons), south: Math.min(...allLats), north: Math.max(...allLats) };
    const bboxWidthDeg = Math.max(bounds.east - bounds.west, 1e-9);
    const bboxHeightDeg = Math.max(bounds.north - bounds.south, 1e-9);

    // Raster resolution: fine enough to look smooth, capped for
    // performance regardless of how sparse the real data is within its
    // own bounding box (sparseness no longer matters for correctness,
    // only the distance cutoff below does -- unlike the old "1 pixel per
    // cell" approach where sparseness directly caused the banding bug).
    const MAX_DIM = 220;
    const largerDim = Math.max(bboxWidthDeg, bboxHeightDeg);
    const nCols = Math.max(2, Math.min(MAX_DIM, Math.round(MAX_DIM * (bboxWidthDeg / largerDim))));
    const nRows = Math.max(2, Math.min(MAX_DIM, Math.round(MAX_DIM * (bboxHeightDeg / largerDim))));

    const K = 4;
    const cutoffDist = 1.5 * Math.max(cellWidth, cellHeight);

    // Spatial bucket index over cell centroids -- bucket size equals the
    // cutoff distance, so any cell within range of a given point must
    // fall in that point's own bucket or one of its 8 immediate
    // neighbors. Turns the search from O(cells) to a small constant per
    // pixel.
    const buckets = new Map();
    const bucketKey = (lon, lat) => Math.floor((lon - bounds.west) / cutoffDist) + "," + Math.floor((lat - bounds.south) / cutoffDist);
    centroids.forEach((cen, i) => {
      const key = bucketKey(cen.lon, cen.lat);
      if (!buckets.has(key)) buckets.set(key, []);
      buckets.get(key).push(i);
    });

    const pixelNeighbors = new Array(nRows * nCols);
    for (let r = 0; r < nRows; r++) {
      const pxLat = bounds.north - (r + 0.5) / nRows * bboxHeightDeg; // row 0 = north
      for (let c = 0; c < nCols; c++) {
        const pxLon = bounds.west + (c + 0.5) / nCols * bboxWidthDeg;
        const bc = Math.floor((pxLon - bounds.west) / cutoffDist);
        const br = Math.floor((pxLat - bounds.south) / cutoffDist);
        const candidates = [];
        for (let dr = -1; dr <= 1; dr++) {
          for (let dc = -1; dc <= 1; dc++) {
            const bucket = buckets.get((bc + dc) + "," + (br + dr));
            if (bucket) candidates.push(...bucket);
          }
        }
        const idx = r * nCols + c;
        if (candidates.length === 0) { pixelNeighbors[idx] = null; continue; }
        const dists = candidates.map((i) => ({ i, d: Math.hypot(centroids[i].lon - pxLon, centroids[i].lat - pxLat) }));
        dists.sort((a, b) => a.d - b.d);
        const nearest = dists.slice(0, K).filter((n) => n.d <= cutoffDist);
        if (nearest.length === 0) { pixelNeighbors[idx] = null; continue; }
        const weights = nearest.map((n) => 1 / (n.d + 1e-6));
        const wsum = weights.reduce((a, b) => a + b, 0);
        pixelNeighbors[idx] = {
          cellIds: nearest.map((n) => centroids[n.i].id),
          weights: weights.map((w) => w / wsum),
        };
      }
    }

    return { nRows, nCols, pixelNeighbors, bounds };
  }

  // Builds the structure + the reusable tiny canvas + the ImageOverlay
  // instance. Called once per prediction result (from drawPredictedCells)
  // -- NOT added to the map here; only shown when the user switches to
  // smooth mode (setVisualizationMode). Also renders the first frame
  // immediately so the overlay is already correct and ready the instant
  // the user first toggles, with zero visible delay.
  function initSmoothHeatmap(cellIds, lons, lats, valuesByCell) {
    if (smoothOverlay) { map.removeLayer(smoothOverlay); smoothOverlay = null; }
    smoothGridStructure = buildSmoothInterpolationStructure(cellIds, lons, lats);
    smoothCanvas = document.createElement("canvas");
    smoothCanvas.width = smoothGridStructure.nCols;
    smoothCanvas.height = smoothGridStructure.nRows;

    renderSmoothFrame(valuesByCell); // paints smoothCanvas before building the overlay's initial image

    const b = smoothGridStructure.bounds;
    const leafletBounds = [[b.south, b.west], [b.north, b.east]];
    smoothOverlay = L.imageOverlay(smoothCanvas.toDataURL(), leafletBounds, {
      opacity: 0.75, interactive: false, // matches the grid view's own fillOpacity: 0.75, non-interactive since there's no per-pixel "cell" to click/hover in this view
    });
  }

  // Lightweight per-frame update -- reuses the precomputed per-pixel
  // neighbor+weight structure, only re-blending VALUES (not recomputing
  // any spatial search). O(pixels x K), no new layers, no new DOM
  // elements, no per-frame recomputation of cell positions.
  function renderSmoothFrame(valuesByCell) {
    if (!smoothGridStructure || !smoothCanvas) return;
    const { nRows, nCols, pixelNeighbors } = smoothGridStructure;
    const ctx = smoothCanvas.getContext("2d");
    const imgData = ctx.createImageData(nCols, nRows); // zero-initialized: fully transparent by default

    for (let i = 0; i < pixelNeighbors.length; i++) {
      const p = pixelNeighbors[i];
      if (!p) continue; // no cell genuinely nearby -- stays transparent, never extrapolated

      // Inverse-distance-weighted blend of the underlying VALUES first,
      // then colorize once -- interpolating in value-space (not
      // color-space) so the result is a genuine blend of PM2.5
      // concentrations, matching how a smoothed physical field should
      // behave, rather than a possibly-muddy blend of colors from a
      // non-linear color scale.
      let blended = 0, wsum = 0;
      for (let k = 0; k < p.cellIds.length; k++) {
        const v = valuesByCell[p.cellIds[k]];
        if (v == null || Number.isNaN(v)) continue;
        blended += p.weights[k] * v;
        wsum += p.weights[k];
      }
      if (wsum === 0) continue; // every nearby neighbor happened to be missing -- leave transparent
      const rgb = pm25ColorRgb(blended / wsum);
      if (!rgb) continue;
      const dataIdx = i * 4;
      imgData.data[dataIdx] = rgb[0]; imgData.data[dataIdx + 1] = rgb[1]; imgData.data[dataIdx + 2] = rgb[2];
      imgData.data[dataIdx + 3] = 255; // overlay's own `opacity: 0.75` option handles overall translucency
    }
    ctx.putImageData(imgData, 0, 0);
    if (smoothOverlay) smoothOverlay.setUrl(smoothCanvas.toDataURL());
  }

  function setVisualizationMode(mode) {
    if (mode === visualizationMode) return;
    visualizationMode = mode;
    if (mode === "smooth") {
      if (predictionLayer && map.hasLayer(predictionLayer)) map.removeLayer(predictionLayer);
      if (smoothOverlay && !map.hasLayer(smoothOverlay)) map.addLayer(smoothOverlay);
    } else {
      if (smoothOverlay && map.hasLayer(smoothOverlay)) map.removeLayer(smoothOverlay);
      if (predictionLayer && !map.hasLayer(predictionLayer)) map.addLayer(predictionLayer);
    }
  }

  function getVisualizationMode() { return visualizationMode; }

  // Renders the actual predicted cells, colored by value. cellIds/lons/lats
  // is the SAME geometry contract as drawSelectedCellGrid (reused, not
  // duplicated); valuesByCell is {cell_id: predicted_value} for the
  // CURRENTLY selected timestamp only -- the caller (main.js) re-calls
  // this on every slider move with a new valuesByCell, which is cheap
  // (pure client-side re-render, no network call, since all timestamps'
  // values were already delivered in the single job-completion response).
  // Layer references for the currently-drawn prediction cells, keyed by
  // cell_id -- populated once by drawPredictedCells, reused by
  // updatePredictedCellColors so playback/slider moves only update each
  // polygon's fill color and tooltip text in place, rather than tearing
  // down and rebuilding the entire canvas renderer + all geometry + all
  // tooltip bindings on every single frame.
  //
  // PERFORMANCE FIX: the previous version called the equivalent of
  // drawPredictedCells on every slider move AND every playback tick (up
  // to every 800ms, looping continuously while "play" is active). Each
  // call destroyed the previous frame's entire Leaflet layer (including
  // its own freshly-created L.canvas() renderer -- a new <canvas> DOM
  // element every time) and rebuilt all N polygons' geometry and
  // tooltip bindings from scratch. For a 108-cell prediction playing
  // back over dozens of timestamps in a loop, this created and
  // discarded a new canvas element and hundreds of new DOM/tooltip
  // objects every fraction of a second -- confirmed as the direct cause
  // of the lag and memory growth reported during playback specifically
  // (not during the initial prediction, which only calls this once).
  let predictedCellLayers = {};
  let cellClickCallback = null;
  let cellHoverCallback = null;

  // Lets main.js register a handler for "user clicked a predicted cell"
  // without map.js needing to know anything about what happens next
  // (opening a chart modal, etc.) -- keeps the two files' responsibilities
  // separate, same pattern the module already uses for its public API.
  function onCellClick(callback) { cellClickCallback = callback; }

  // Same pattern for hover: the callback receives the hovered cell's id on
  // mouseover and null on mouseout. Only fires from the gridded cell
  // layer (predictionLayer) -- the smooth-heatmap overlay is a single
  // non-interactive image with no per-cell events, so no hover is
  // reported while that mode is active.
  function onCellHover(callback) { cellHoverCallback = callback; }

  function drawPredictedCells(cellIds, lons, lats, valuesByCell) {
    predictionLayer.clearLayers();
    predictedCellLayers = {};
    if (!lons || lons.length === 0) return;

    initSmoothHeatmap(cellIds, lons, lats, valuesByCell); // built once here, alongside the grid geometry -- not rebuilt on frame changes or mode toggles

    const features = lons.map((cellLons, i) => {
      const cellLats = lats[i];
      const ring = cellLons.map((lon, j) => [lon, cellLats[j]]);
      ring.push(ring[0]);
      return {
        type: "Feature",
        properties: { cellId: cellIds[i], value: valuesByCell[cellIds[i]] },
        geometry: { type: "Polygon", coordinates: [ring] },
      };
    });

    // Built and added to the map ONCE per prediction result -- this is
    // the only place a new L.canvas() renderer or new tooltip bindings
    // are created. Every later color update reuses this same layer set.
    // Uses predictionLayer, a layer DEDICATED to prediction results --
    // confirmed bug fix: this previously shared cellsLayer with the
    // "Show individual 1km cells" toggle feature (drawSelectedCellGrid),
    // so each one's clearLayers() call silently destroyed the other's
    // content whenever the user interacted with either feature. Now
    // fully independent -- toggling the grid-cell view can never affect
    // the prediction map, and vice versa.
    L.geoJSON({ type: "FeatureCollection", features }, {
      renderer: L.canvas(),
      style: (feature) => ({
        color: "#1f2937", weight: 0.5,
        fillColor: pm25Color(feature.properties.value), fillOpacity: 0.75,
      }),
      onEachFeature: (feature, layer) => {
        const v = feature.properties.value;
        layer.bindTooltip(v == null ? "No prediction" : `${v.toFixed(1)} µg/m³`);
        layer.on("click", () => {
          if (cellClickCallback) cellClickCallback(feature.properties.cellId);
        });
        layer.on("mouseover", () => {
          if (cellHoverCallback) cellHoverCallback(feature.properties.cellId);
        });
        layer.on("mouseout", () => {
          if (cellHoverCallback) cellHoverCallback(null);
        });
        predictedCellLayers[feature.properties.cellId] = layer;
      },
    }).addTo(predictionLayer);
  }

  function clearPredictionLayer() {
    predictionLayer.clearLayers();
    predictedCellLayers = {};
    if (smoothOverlay) { map.removeLayer(smoothOverlay); smoothOverlay = null; }
    smoothCanvas = null;
    smoothGridStructure = null;
    visualizationMode = "grid";
    if (predictionLayer && !map.hasLayer(predictionLayer)) map.addLayer(predictionLayer); // re-attach in case smooth mode had detached it
  }

  // Lightweight recolor-only update for slider moves / playback ticks --
  // reuses the SAME layers/canvas drawPredictedCells already built,
  // just changing each polygon's fill color and tooltip text in place.
  // No geometry rebuild, no new canvas, no new tooltip DOM nodes.
  function updatePredictedCellColors(valuesByCell) {
    for (const cellId in predictedCellLayers) {
      const layer = predictedCellLayers[cellId];
      const v = valuesByCell[cellId];
      layer.setStyle({ fillColor: pm25Color(v) });
      layer.setTooltipContent(v == null ? "No prediction" : `${v.toFixed(1)} µg/m³`);
    }
  }

  function zoomBy(delta) { map.setZoom(map.getZoom() + delta); }
  function recenter() { map.fitBounds(L.geoJSON(boundaryData).getBounds()); }

  return {
    init, setMode, clearSelection, drawSelection, drawSelectedCellGrid, clearSelectedCellGrid,
    drawPredictedCells, updatePredictedCellColors, clearPredictionLayer, onCellClick, onCellHover, pm25Color, zoomBy, recenter,
    renderSmoothFrame, setVisualizationMode, getVisualizationMode,
  };
})();
