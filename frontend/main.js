/**
 * main.js -- Application orchestration: wires the map, toolbar, date
 * controls, and backend API calls together.
 *
 * Data integrity note: this file NEVER computes cell selection, area, or
 * validation itself -- every one of those numbers comes from the Flask
 * backend (backend/app.py), which in turn calls the exact same
 * selection/cell_selector.py and selection/request_validator.py used
 * throughout this project. The frontend only draws what the backend
 * returns.
 */

const state = {
  selectedPolygon: null,
  lastClippedPolygon: null, // the backend-clipped polygon GeoJSON, kept so it can be re-shown when the cell-grid toggle is switched off
  cellCount: 0,
  areaKm2: 0,
  cellIds: [],
  showCellGrid: false, // OFF by default, per explicit requirement -- visualization-only, never affects the actual selection/request
  predictionComplete: false, // tracks whether generate-btn is currently in "Generate" mode or "Reset" mode
};

// Locks every selection-defining control once Generate is pressed, per
// explicit requirement: the user can no longer change what was actually
// requested (date range, resolution, single-day mode) or start drawing a
// new area while a job is running or its results are being shown. Only
// Pan/Move Map and the "Show individual 1km cells" toggle remain
// interactive throughout (the toggle is separately locked later, only
// once the PREDICTION itself completes -- see lockPredictionView()).
function lockSelectionControls() {
  el("single-day-toggle").disabled = true;
  el("start-date").disabled = true;
  el("end-date").disabled = true;
  el("resolution-select").disabled = true;
  el("tool-draw").style.display = "none";
}

function unlockSelectionControls() {
  el("single-day-toggle").disabled = false;
  el("start-date").disabled = false;
  el("end-date").disabled = false;
  el("resolution-select").disabled = false;
  el("tool-draw").style.display = "flex";
}

function el(id) { return document.getElementById(id); }

async function apiPost(path, body) {
  const res = await fetch(path, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || data.message || "Request failed.");
  return data;
}

async function apiGet(path) {
  const res = await fetch(path);
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || data.message || "Request failed.");
  return data;
}

// ---------------------------------------------------------------
// Toolbar
// ---------------------------------------------------------------
function setActiveTool(toolId) {
  document.querySelectorAll(".tool-btn").forEach(b => b.classList.remove("active"));
  el(toolId).classList.add("active");
}

el("tool-select").addEventListener("click", () => {
  setActiveTool("tool-select");
  PM25Map.setMode("select");
  el("drawing-legend").style.display = "none";
});

el("tool-draw").addEventListener("click", () => {
  setActiveTool("tool-draw");
  PM25Map.setMode("draw");
  el("map-status").style.opacity = 1;
  el("map-status").textContent = "Click to place polygon points; click the first (red) point again to close the shape.";
  el("drawing-legend").style.display = "flex";
  unlockPredictionView(); // reverses lockPredictionView() if a prior prediction had locked the view
});

el("tool-clear").addEventListener("click", () => {
  PM25Map.clearSelection();
  clearSelectionState();
  el("drawing-legend").style.display = "none";
});

el("zoom-recenter").addEventListener("click", () => PM25Map.recenter());

// ---------------------------------------------------------------
// Optional 1km cell grid visualization -- OFF by default. Purely a
// display toggle: it never changes state.cellCount, state.areaKm2,
// state.cellIds, or anything sent in the prediction request preview --
// those all continue to come from the exact same /api/select_cells call
// used regardless of this toggle's state. Turning it off skips the
// extra network call and clears the rendered layer entirely, so there
// is no rendering overhead when disabled.
// ---------------------------------------------------------------
function setCellGridStatusBadge(on) {
  const badge = el("cell-grid-status");
  badge.textContent = on ? "On" : "Off";
  badge.classList.toggle("on", on);
}

async function loadCellGridIfEnabled() {
  if (!state.showCellGrid || !state.selectedPolygon) return;
  try {
    const result = await apiPost("/api/select_cells_grid", { polygon: state.selectedPolygon });
    // Per explicit requirement: while cells are shown, the bounding
    // polygon must be hidden completely -- not drawn underneath/behind
    // the cells, actually removed from the map.
    PM25Map.drawSelection(null);
    PM25Map.drawSelectedCellGrid(result.lons, result.lats);
  } catch (err) {
    showValidationMessage(err.message || "Could not load the individual cell visualization.", "error");
  }
}

el("cell-grid-toggle").addEventListener("change", (e) => {
  state.showCellGrid = e.target.checked;
  setCellGridStatusBadge(state.showCellGrid);
  if (state.showCellGrid) {
    loadCellGridIfEnabled();
  } else {
    // Returning to the default behavior: hide the individual cells,
    // restore the bounding-polygon visualization for the current selection.
    PM25Map.clearSelectedCellGrid();
    PM25Map.drawSelection(state.lastClippedPolygon);
  }
});

// ---------------------------------------------------------------
// Selection handling
// ---------------------------------------------------------------
function clearSelectionState() {
  state.selectedPolygon = null;
  state.lastClippedPolygon = null;
  state.cellCount = 0;
  state.areaKm2 = 0;
  state.cellIds = [];
  el("region-value").textContent = "None selected";
  el("area-value").textContent = "\u2014";
  el("clip-warning-row").style.display = "none";
  el("request-preview").style.display = "none";
  el("validation-message").style.display = "none";
  el("date-error").style.display = "none";
  setActiveTool("tool-select");
  PM25Map.setMode("select");
}

async function handlePolygonDrawn(polygonGeoJSON) {
  el("map-status").style.opacity = 1;
  el("map-status").textContent = "Selecting cells...";
  try {
    const result = await apiPost("/api/select_cells", { polygon: polygonGeoJSON });
    state.selectedPolygon = polygonGeoJSON;
    state.lastClippedPolygon = result.clipped_polygon;
    state.cellCount = result.cell_count;
    state.areaKm2 = result.area_km2;
    state.cellIds = result.cell_ids;

    // Show the bounding polygon only when the cell-grid toggle is off --
    // if it's already on, loadCellGridIfEnabled() below draws the cells
    // instead and keeps the polygon hidden.
    if (!state.showCellGrid) {
      PM25Map.drawSelection(result.clipped_polygon);
    }

    el("region-value").textContent = "Custom (drawn polygon)";
    el("area-value").textContent = `Approx. ${result.area_km2.toLocaleString()} km\u00b2 (${result.cell_count.toLocaleString()} cells)`;
    el("clip-warning-row").style.display = result.area_outside_domain_excluded ? "flex" : "none";

    if (result.cell_count === 0) {
      showValidationMessage("No prediction cells were selected. Try drawing a larger area.", "error");
    } else {
      el("validation-message").style.display = "none";
    }

    el("map-status").textContent = `${result.cell_count.toLocaleString()} cell(s) selected.`;
    setTimeout(() => { el("map-status").style.opacity = 0; }, 2000);
    updateRequestPreview();
    loadCellGridIfEnabled();
  } catch (err) {
    showValidationMessage(err.message || "The selected polygon could not be processed. Please redraw it.", "error");
    el("map-status").textContent = "Selection failed.";
  }
  setActiveTool("tool-select");
  PM25Map.setMode("select");
  el("drawing-legend").style.display = "none";
}

function showValidationMessage(msg, kind) {
  const box = el("validation-message");
  box.textContent = msg;
  box.className = `validation-message ${kind}`;
  box.style.display = "block";
}

// ---------------------------------------------------------------
// Date controls -> single-day toggle, validation, request preview
// ---------------------------------------------------------------
function getTemporalResolution() { return el("resolution-select").value; }

function isSingleDayMode() { return el("single-day-toggle").checked; }

// The "effective" end date: in single-day mode, this is always the start
// date itself, regardless of whatever the (hidden) end-date field
// contains -- avoids any stale end-date value leaking into a request
// after switching modes.
function getEffectiveEndDate() {
  const startVal = el("start-date").value;
  if (isSingleDayMode()) return startVal;
  return el("end-date").value || startVal;
}

// Returns an error message string if start/end are invalid, else null.
// Enforced both live (on every date change, for immediate feedback) and
// again at Generate-click time (so it can never be bypassed by a stale UI state).
function validateDateRange() {
  const startVal = el("start-date").value;
  if (!startVal) return null; // no date yet -- not an error state, just incomplete
  if (isSingleDayMode()) return null; // single day: start alone is always valid
  const endVal = el("end-date").value;
  if (!endVal) return null; // end not filled in yet
  if (endVal <= startVal) {
    return "End Date must be strictly after Start Date (a date range of one day should use Single Day Only instead).";
  }
  return null;
}

function updateDateModeUI() {
  const single = isSingleDayMode();
  el("end-date-group").style.display = single ? "none" : "block";
  el("date-info-callout").textContent = single
    ? "Single-day mode: predictions will be generated for the selected date only."
    : "Date range mode: predictions will be generated for every timestamp from Start Date through End Date, inclusive.";
}

function estimateRows() {
  const startVal = el("start-date").value;
  if (!startVal || state.cellCount === 0) return 0;
  const endVal = getEffectiveEndDate();
  const start = new Date(startVal + "T00:00:00");
  const end = new Date(endVal + "T23:00:00");
  const hours = Math.max(0, (end - start) / 3600000);
  const resolution = getTemporalResolution();
  const nTimestamps = resolution === "hourly" ? Math.floor(hours) + 1 : Math.floor(hours / 24) + 1;
  return state.cellCount * nTimestamps;
}

function updateRequestPreview() {
  const dateError = validateDateRange();
  const errorBox = el("date-error");
  if (dateError) {
    errorBox.textContent = dateError;
    errorBox.style.display = "block";
  } else {
    errorBox.style.display = "none";
  }

  const startVal = el("start-date").value;
  if (!startVal || state.cellCount === 0 || dateError) {
    el("request-preview").style.display = "none";
    return;
  }
  el("preview-cells").textContent = state.cellCount.toLocaleString();
  el("preview-rows").textContent = estimateRows().toLocaleString();
  el("request-preview").style.display = "block";
}

el("single-day-toggle").addEventListener("change", () => { updateDateModeUI(); updateRequestPreview(); });
el("start-date").addEventListener("change", updateRequestPreview);
el("end-date").addEventListener("change", updateRequestPreview);
el("resolution-select").addEventListener("change", updateRequestPreview);
updateDateModeUI(); // set initial visibility/text to match the toggle's default (checked = single-day)

// ---------------------------------------------------------------
// Dark mode toggle
// ---------------------------------------------------------------
const THEME_STORAGE_KEY = "pm25-theme";

function applyTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  // Icon shows the mode a click will SWITCH TO (matching the previous
  // emoji-based behavior exactly): light_mode.png while in dark mode
  // (click to go light), dark_mode.png while in light mode (click to go dark).
  el("theme-toggle-icon").src = theme === "dark" ? "icons/light_mode.png" : "icons/dark_mode.png";
  el("theme-toggle-icon").alt = theme === "dark" ? "Switch to light mode" : "Switch to dark mode";
  // Readable category text colours depend on the theme's panel colour --
  // re-render any category text currently shown (never true during
  // initial page load, so this is safe here).
  const inspectionBody = el("inspection-body");
  if (inspectionBody && inspectionBody.style.display === "block") renderInspectionPanel();
  const detailOverlay = el("cell-detail-overlay");
  if (detailOverlay && detailOverlay.style.display === "flex") renderCellDetailClassification();
}

function initTheme() {
  const stored = localStorage.getItem(THEME_STORAGE_KEY);
  const prefersDark = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
  applyTheme(stored || (prefersDark ? "dark" : "light"));
}

el("theme-toggle-btn").addEventListener("click", () => {
  const current = document.documentElement.getAttribute("data-theme");
  const next = current === "dark" ? "light" : "dark";
  applyTheme(next);
  localStorage.setItem(THEME_STORAGE_KEY, next);
});

initTheme();

// ---------------------------------------------------------------
// How to Use modal
// ---------------------------------------------------------------
function openHowToUse() { el("how-to-use-overlay").style.display = "flex"; }
function closeHowToUse() { el("how-to-use-overlay").style.display = "none"; }

el("how-to-use-btn").addEventListener("click", openHowToUse);
el("how-to-use-close").addEventListener("click", closeHowToUse);
el("how-to-use-overlay").addEventListener("click", (e) => {
  if (e.target.id === "how-to-use-overlay") closeHowToUse(); // click outside the modal box closes it
});
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && el("how-to-use-overlay").style.display === "flex") closeHowToUse();
});

// ---------------------------------------------------------------
// Cell detail chart modal -- opened by clicking a predicted cell.
// All data needed is already client-side in predictionViz (delivered
// once, at job completion) -- no new network calls, both the full
// timeline and the hourly-average view are computed here from the
// same already-fetched predictions.
// ---------------------------------------------------------------
let cellDetailState = null; // { mode: "cell" | "area", cellId (mode="cell" only), activeTab: "timeline" | "hourly" }

// Reliable Manila-local hour-of-day extraction (0-23) for a UTC ISO
// timestamp string. Uses Intl.DateTimeFormat's formatToParts rather than
// string-parsing toLocaleString's output directly, since hour12:false
// alone is known to sometimes render midnight as "24" rather than "00"
// in some JS engines -- hourCycle:"h23" avoids that ambiguity explicitly.
const manilaHourFormatter = new Intl.DateTimeFormat("en-US", {
  hour: "2-digit", hourCycle: "h23", timeZone: "Asia/Manila",
});
function getManilaHour(ts) {
  const part = manilaHourFormatter.formatToParts(new Date(ts)).find((p) => p.type === "hour");
  return Number(part.value);
}

function formatTimelineLabel(ts, isDaily) {
  return new Date(ts).toLocaleString(undefined, isDaily
    ? { month: "short", day: "numeric", timeZone: "Asia/Manila" }
    : { month: "short", day: "numeric", hour: "2-digit", timeZone: "Asia/Manila" });
}

// Returns this timestamp's value for the current modal (a single cell's
// own value in "cell" mode, or the mean across every predicted cell that
// has a value at that timestamp in "area" mode) -- null if nothing valid.
function valueAtTimestamp(ts) {
  const cellValues = predictionViz.valuesByTimestamp[ts];
  if (cellDetailState.mode === "cell") {
    const v = cellValues[cellDetailState.cellId];
    return v == null ? null : v;
  }
  const valid = Object.values(cellValues).filter((v) => v != null && !Number.isNaN(v));
  return valid.length ? valid.reduce((a, b) => a + b, 0) / valid.length : null;
}

// Centroid = simple average of the cell's 4 known corners (already
// client-side in predictionViz, from the same geometry fetched once at
// job completion) -- no new network call needed. Good enough for
// display purposes at 1km cell scale; not used for anything computational.
// Returns {lat, lon} or null if the cell isn't in the current result.
// Shared by the cell-detail modal and the hover inspection panel.
function cellCentroid(cellId) {
  if (!predictionViz) return null;
  const idx = predictionViz.cellIds.indexOf(cellId);
  if (idx === -1) return null;
  const lons = predictionViz.lons[idx], lats = predictionViz.lats[idx];
  return {
    lon: lons.reduce((a, b) => a + b, 0) / lons.length,
    lat: lats.reduce((a, b) => a + b, 0) / lats.length,
  };
}

function formatCentroid(c) {
  const latDir = c.lat >= 0 ? "N" : "S";
  const lonDir = c.lon >= 0 ? "E" : "W";
  return `${Math.abs(c.lat).toFixed(5)}\u00b0${latDir}, ${Math.abs(c.lon).toFixed(5)}\u00b0${lonDir}`;
}

function openCellDetail(cellId) {
  if (!predictionViz) return;
  cellDetailState = { mode: "cell", cellId, activeTab: "timeline" };
  el("cell-detail-title").textContent = `Cell ${cellId} -- PM2.5 Prediction`;

  const c = cellCentroid(cellId);
  el("cell-detail-coords").textContent = c ? `Centroid: ${formatCentroid(c)}` : "";

  openCellDetailModal();
}

// The area-wide counterpart to openCellDetail -- same modal, same chart
// component, but every value is the mean across all predicted cells at
// that timestamp (or hour) instead of one cell's own value. Reached via
// the "View Overall Trend" toolbar button, enabled once a prediction
// completes (see lockPredictionView()/resetEverything()).
function openAreaSummary() {
  if (!predictionViz) return;
  cellDetailState = { mode: "area", activeTab: "timeline" };
  const n = predictionViz.cellIds.length;
  el("cell-detail-title").textContent = "Area Average -- PM2.5 Prediction";
  el("cell-detail-coords").textContent = `Averaged across ${n} selected cell${n === 1 ? "" : "s"}`;
  openCellDetailModal();
}

function openCellDetailModal() {
  const isHourly = predictionViz.temporalResolution === "hourly";
  const hourlyTab = el("cell-detail-tab-hourly");
  hourlyTab.style.display = isHourly ? "inline-block" : "none";

  // SHAP explanation tab: per-cell only (an "area" explanation would be a
  // new, un-validated aggregate) and only while a job id exists to ask.
  const explanationTab = el("cell-detail-tab-explanation");
  explanationTab.style.display = (cellDetailState.mode === "cell" && activePredictionJobId) ? "inline-block" : "none";

  el("cell-detail-tab-timeline").classList.add("active");
  hourlyTab.classList.remove("active");
  explanationTab.classList.remove("active");

  el("cell-detail-overlay").style.display = "flex";
  renderCellDetailTab();
}

function closeCellDetail() {
  el("cell-detail-overlay").style.display = "none";
  cellDetailState = null;
}

function renderCellDetailTab() {
  if (!cellDetailState || !predictionViz) return;
  const { activeTab } = cellDetailState;
  const isDaily = predictionViz.temporalResolution === "daily";

  renderCellDetailClassification(); // classification line (if a scheme is selected; cell mode only)

  // The explanation tab replaces the chart/summary area with its own
  // panel; the other two tabs share the chart + summary as before.
  const isExplanation = activeTab === "explanation";
  el("cell-detail-chart").style.display = isExplanation ? "none" : "";
  el("cell-detail-summary").style.display = isExplanation ? "none" : "";
  el("cell-detail-explanation").style.display = isExplanation ? "block" : "none";
  if (isExplanation) {
    renderExplanationTab();
    return;
  }

  if (activeTab === "timeline") {
    const labels = predictionViz.timestamps.map((ts) => formatTimelineLabel(ts, isDaily));
    const values = predictionViz.timestamps.map((ts) => valueAtTimestamp(ts));
    PM25Chart.render(el("cell-detail-chart"), { labels, values, pointColorFn: PM25Map.pm25Color });
    renderCellDetailSummary(values, `${predictionViz.timestamps.length} ${isDaily ? "day" : "hour"}(s), start to end`);
  } else {
    // Hourly average: group every requested timestamp by its Manila
    // local hour-of-day (0-23) and average the relevant value(s) within
    // each hour bucket. In "area" mode this flattens across BOTH cells
    // and days at once (every (cell, timestamp) pair whose local hour
    // matches contributes equally to that hour's average) -- for a
    // single-day, single-cell request this is numerically identical to
    // the timeline tab, just re-expressed as a 00:00-23:00 profile.
    const sums = new Array(24).fill(0);
    const counts = new Array(24).fill(0);
    predictionViz.timestamps.forEach((ts) => {
      const hour = getManilaHour(ts);
      if (cellDetailState.mode === "area") {
        Object.values(predictionViz.valuesByTimestamp[ts]).forEach((v) => {
          if (v == null || Number.isNaN(v)) return;
          sums[hour] += v; counts[hour] += 1;
        });
      } else {
        const v = predictionViz.valuesByTimestamp[ts][cellDetailState.cellId];
        if (v == null) return;
        sums[hour] += v; counts[hour] += 1;
      }
    });
    const labels = Array.from({ length: 24 }, (_, h) => `${String(h).padStart(2, "0")}:00`);
    const values = sums.map((sum, h) => (counts[h] > 0 ? sum / counts[h] : null));
    PM25Chart.render(el("cell-detail-chart"), { labels, values, pointColorFn: PM25Map.pm25Color });
    const nDays = new Set(predictionViz.timestamps.map((ts) => new Date(ts).toLocaleDateString("en-CA", { timeZone: "Asia/Manila" }))).size;
    const scopeLabel = cellDetailState.mode === "area"
      ? `Averaged across ${nDays} day(s) and ${predictionViz.cellIds.length} cell(s), Philippine local hour`
      : `Averaged across ${nDays} day(s), Philippine local hour`;
    renderCellDetailSummary(values, scopeLabel);
  }
}

function renderCellDetailSummary(values, contextLabel) {
  const valid = values.filter((v) => v != null && !Number.isNaN(v));
  const summary = el("cell-detail-summary");
  if (!valid.length) {
    summary.innerHTML = `<span>${contextLabel}</span><span>No valid predictions in this view.</span>`;
    return;
  }
  const min = Math.min(...valid), max = Math.max(...valid);
  const mean = valid.reduce((a, b) => a + b, 0) / valid.length;
  summary.innerHTML = `
    <span>${contextLabel}</span>
    <span>Min: <strong>${min.toFixed(1)}</strong> µg/m³</span>
    <span>Mean: <strong>${mean.toFixed(1)}</strong> µg/m³</span>
    <span>Max: <strong>${max.toFixed(1)}</strong> µg/m³</span>
  `;
}

PM25Map.onCellClick(openCellDetail);
el("tool-area-summary").addEventListener("click", openAreaSummary);

el("tool-heatmap-mode").addEventListener("click", () => {
  if (!predictionViz) return;
  const nextMode = PM25Map.getVisualizationMode() === "smooth" ? "grid" : "smooth";
  PM25Map.setVisualizationMode(nextMode);
  el("tool-heatmap-mode-label").textContent = nextMode === "smooth" ? "Smooth Heatmap: On" : "Smooth Heatmap: Off";
  // Re-render the CURRENT frame into the newly-visible mode immediately --
  // the other mode's layer may be stale if the slider moved while it
  // wasn't displayed, since renderPredictionFrame only updates whichever
  // mode is currently visible (see its own comment). This keeps the
  // switch itself instant and correct without touching the playback
  // timer or the slider position -- playback continues exactly as-is,
  // whether running or stopped, through this same toggle.
  renderPredictionFrame(Number(el("time-slider").value));
});

el("cell-detail-close").addEventListener("click", closeCellDetail);
el("cell-detail-overlay").addEventListener("click", (e) => {
  if (e.target.id === "cell-detail-overlay") closeCellDetail();
});
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && el("cell-detail-overlay").style.display === "flex") closeCellDetail();
});
function activateCellDetailTab(tab) {
  if (!cellDetailState) return;
  cellDetailState.activeTab = tab;
  for (const t of ["timeline", "hourly", "explanation"]) {
    el(`cell-detail-tab-${t}`).classList.toggle("active", t === tab);
  }
  renderCellDetailTab();
}
el("cell-detail-tab-timeline").addEventListener("click", () => activateCellDetailTab("timeline"));
el("cell-detail-tab-hourly").addEventListener("click", () => activateCellDetailTab("hourly"));
el("cell-detail-tab-explanation").addEventListener("click", () => activateCellDetailTab("explanation"));

// ---------------------------------------------------------------
// SHAP model explanation tab (inside the cell-detail modal).
// Fetches /api/prediction_explanation for ONE hourly feature row of the
// completed job -- the deployed Philippine-only XGBoost member explained with
// shap.TreeExplainer in log1p space (thesis methodology). Fetched only
// when the tab is opened or the hour picker changes (never on hover or
// per playback tick), cached per (cell, hour).
//   hourly results: the picker lists the result's own timestamps and
//                   defaults to the frame currently shown on the slider.
//   daily results:  the picker lists the 24 underlying hours of the local
//                   day currently shown -- each explanation is for ONE
//                   hourly row; SHAP values are never averaged over the day.
// ---------------------------------------------------------------
const explanationCache = new Map(); // `${cellId}|${isoHour}` -> payload
const EXPLANATION_TOP_N = 5;

function explanationHourOptions() {
  const ts = currentFrameTimestamp();
  if (ts == null) return [];
  if (predictionViz.temporalResolution === "hourly") {
    return predictionViz.timestamps.map((t) => ({ value: new Date(t).toISOString(), label: formatTimelineLabel(t, false) + " PHT" }));
  }
  // Daily: the representative timestamp is local midnight (UTC instant);
  // the 24 hourly feature rows of that local day follow at +h hours.
  const base = new Date(ts).getTime();
  return Array.from({ length: 24 }, (_, h) => {
    const d = new Date(base + h * 3600 * 1000);
    return { value: d.toISOString(), label: formatTimelineLabel(d.toISOString(), false) + " PHT" };
  });
}

function renderExplanationTab() {
  if (!cellDetailState || cellDetailState.mode !== "cell" || !predictionViz) return;
  const select = el("explanation-hour-select");
  const options = explanationHourOptions();
  const previous = cellDetailState.explanationHour;
  select.innerHTML = "";
  for (const o of options) {
    const opt = document.createElement("option");
    opt.value = o.value; opt.textContent = o.label;
    select.appendChild(opt);
  }
  const isDaily = predictionViz.temporalResolution === "daily";
  let defaultValue;
  if (previous && options.some((o) => o.value === previous)) {
    defaultValue = previous;
  } else if (!isDaily) {
    defaultValue = new Date(currentFrameTimestamp()).toISOString(); // the frame on the slider
  } else {
    defaultValue = options.length ? options[0].value : null;
  }
  if (defaultValue) select.value = defaultValue;
  cellDetailState.explanationHour = select.value || null;

  const dailyNote = el("explanation-daily-note");
  if (isDaily) {
    dailyNote.textContent = "This result is shown at daily resolution (the mean of modelled hourly predictions for each "
      + "Philippine local day). Explanations are only available for individual hourly predictions -- pick one of the "
      + `24 hours of ${formatFrameTimestamp(currentFrameTimestamp())}. SHAP values are not averaged across the day.`;
    dailyNote.style.display = "block";
  } else {
    dailyNote.style.display = "none";
  }
  loadExplanation();
}

async function loadExplanation() {
  if (!cellDetailState || cellDetailState.mode !== "cell" || !activePredictionJobId) return;
  const cellId = cellDetailState.cellId;
  const hour = cellDetailState.explanationHour;
  const status = el("explanation-status");
  const header = el("explanation-header");
  const lists = el("explanation-lists");
  const footer = el("explanation-footer");
  header.innerHTML = ""; lists.innerHTML = ""; footer.textContent = "";
  if (!hour) { status.textContent = "No hour available to explain."; return; }

  const key = `${activePredictionJobId}|${cellId}|${hour}`;
  let payload = explanationCache.get(key);
  if (!payload) {
    status.textContent = "Computing explanation…";
    try {
      payload = await apiGet(`/api/prediction_explanation/${activePredictionJobId}?grid_id=${encodeURIComponent(cellId)}&datetime_utc=${encodeURIComponent(hour)}`);
      explanationCache.set(key, payload);
    } catch (err) {
      status.textContent = `Explanation unavailable: ${err.message}`;
      return;
    }
  }
  // The user may have switched cell/hour while the request was in flight.
  if (!cellDetailState || cellDetailState.cellId !== cellId || cellDetailState.explanationHour !== hour) return;
  status.textContent = "";
  renderExplanation(payload);
}

function renderExplanation(p) {
  const header = el("explanation-header");
  const lists = el("explanation-lists");
  const footer = el("explanation-footer");

  const fmt = (v) => (v == null ? "n/a" : `${v.toFixed(1)} µg/m³`);
  const parts = [];
  if (p.ensemble_prediction_ugm3 != null) parts.push(`<span><span class="info-label">Displayed (ensemble) prediction:</span> <strong>${fmt(p.ensemble_prediction_ugm3)}</strong></span>`);
  parts.push(`<span><span class="info-label">Explained XGBoost member's prediction:</span> <strong>${fmt(p.xgboost_prediction_ugm3)}</strong></span>`);
  parts.push(`<span><span class="info-label">Model base value:</span> ${fmt(p.base_value_ugm3)} <span class="info-label">(${p.base_value_log.toFixed(3)} in log1p space)</span></span>`);
  if (p.n_missing_features > 0) parts.push(`<span class="explanation-warn">${p.n_missing_features} of ${p.n_features} input feature(s) were missing for this row (handled natively by the model).</span>`);
  if (p.additivity_ok === false) parts.push(`<span class="explanation-warn">Additivity check failed for this explainer -- contributions may not sum to the prediction.</span>`);
  header.innerHTML = parts.join("");

  const positive = p.contributions.filter((c) => c.shap_log > 0).slice(0, EXPLANATION_TOP_N);
  const negative = p.contributions.filter((c) => c.shap_log < 0).slice(0, EXPLANATION_TOP_N);
  PM25Chart.renderContributions(lists, { positive, negative });

  footer.textContent = `Top ${EXPLANATION_TOP_N} of ${p.n_features} features in each direction, ranked by |SHAP| `
    + `(values in ${p.output_space}). ${p.attribution_note}`;
}

el("explanation-hour-select").addEventListener("change", (e) => {
  if (!cellDetailState) return;
  cellDetailState.explanationHour = e.target.value;
  loadExplanation();
});

// ---------------------------------------------------------------
// Generate button
// ---------------------------------------------------------------
let activePredictionJobId = null;
let predictionPollTimer = null;

function showPredictionProgress() {
  el("generate-btn").style.display = "none";
  el("prediction-result-banner").style.display = "none";
  el("prediction-progress").style.display = "block";

  // Per explicit requirement: once Generate is pressed, the selection
  // can no longer be cleared/altered until Reset is pressed after
  // completion (see lockPredictionView()/resetEverything() below).
  el("tool-clear").style.display = "none";
  lockSelectionControls();

  // Clear any previous prediction's own visualization -- a stale
  // colored map/slider from an earlier run shouldn't linger while a new
  // job runs. Does NOT touch the current polygon/grid-cell selection
  // view (per explicit requirement, that stays visible and the
  // "Show individual 1km cells" toggle stays interactive throughout
  // data collection and prediction, right up until the job completes).
  if (playbackTimer) { clearInterval(playbackTimer); playbackTimer = null; el("time-slider-play").textContent = "▶"; }
  predictionViz = null;
  el("prediction-viz-controls").style.display = "none";
  PM25Map.clearPredictionLayer();
}

function hidePredictionProgress() {
  el("generate-btn").style.display = "block";
  el("prediction-progress").style.display = "none";
  if (predictionPollTimer) { clearInterval(predictionPollTimer); predictionPollTimer = null; }
}

// ---------------------------------------------------------------
// Rendering predictions on the map, with a time slider for multiple
// timestamps. Cell geometry is fetched ONCE (reusing the existing
// /api/select_cells_grid contract, per the deployment plan's Section
// 9.1); every subsequent slider move is a pure client-side color
// lookup against the single already-delivered predictions matrix --
// no repeated network calls.
// ---------------------------------------------------------------
let predictionViz = null; // { cellIds, lons, lats, timestamps, valuesByTimestamp: {ts: {cellId: value}} }
let playbackTimer = null;

// Locks the map into "prediction results only" view once a prediction
// completes, per explicit requirement: the grid-cell toggle becomes
// non-interactive, any grid cells or bounding polygon currently shown
// are forcibly cleared, and only the colored prediction cells remain
// visible from this point on -- regardless of what the toggle's own
// checked state was right before completion.
function lockPredictionView() {
  state.showCellGrid = false;
  const toggle = el("cell-grid-toggle");
  toggle.checked = false;
  toggle.disabled = true;
  setCellGridStatusBadge(false);
  el("cell-grid-status").textContent = "Locked";
  PM25Map.clearSelectedCellGrid();
  PM25Map.drawSelection(null);

  // Per explicit requirement: once a prediction completes, an "overall
  // trend" summary (averaged across every predicted cell) becomes
  // available -- the same click-to-chart affordance as individual cells,
  // just for the whole selected area at once.
  el("tool-area-summary").style.display = "flex";

  // New requested feature: a toggle between the original gridded-cell
  // heatmap and a smooth/blended one, available once predictions exist.
  // Always starts in "grid" mode (matching drawPredictedCells/
  // clearPredictionLayer's own default) -- the label is reset here too
  // in case a previous prediction had left it toggled to "smooth".
  el("tool-heatmap-mode").style.display = "flex";
  el("tool-heatmap-mode-label").textContent = "Smooth Heatmap: Off";

  // Export links point at the completed job's server-side export (the
  // job id is retained in activePredictionJobId after completion).
  updateExportLinks();

  // Interpretation legend for whichever classification scheme is selected
  // (hidden for the default "None"); depends on the result's resolution
  // for its hourly/daily basis note, so rendered once the result exists.
  renderClassificationLegend();

  // Per explicit requirement: after a prediction completes, everything
  // stays locked (date/resolution controls, Draw Boundary, the grid
  // toggle) and Pan/Move Map + the timestamp slider are the only
  // interactive controls left. Generate becomes Reset -- the only way
  // to start over is resetEverything() below, not re-drawing a new area.
  state.predictionComplete = true;
  const btn = el("generate-btn");
  btn.textContent = "Reset";
  btn.classList.add("reset-mode");
}

// Reverses lockPredictionView() -- called when the user starts a fresh
// selection (Draw Boundary), the only way forward once a prediction has
// completed, per explicit requirement (Clear Selection is hidden and the
// grid toggle is locked throughout the "prediction complete" state).
function unlockPredictionView() {
  el("cell-grid-toggle").disabled = false;
  el("tool-clear").style.display = "flex";
  el("tool-area-summary").style.display = "none";
  el("tool-heatmap-mode").style.display = "none";
  if (el("cell-detail-overlay").style.display !== "none") closeCellDetail();
  if (playbackTimer) { clearInterval(playbackTimer); playbackTimer = null; el("time-slider-play").textContent = "▶"; }
  predictionViz = null;
  hoveredCellId = null;
  renderInspectionPanel();
  renderClassificationLegend();
  explanationCache.clear(); // explanations belong to the job being discarded
  PM25Map.clearPredictionLayer();
  el("prediction-viz-controls").style.display = "none";
  el("prediction-result-banner").style.display = "none";
}

// Full reset, triggered by the Generate button once it has become Reset
// (state.predictionComplete === true), per explicit requirement: clears
// the prediction results AND the user's selection (drawn area, grid
// cells, timestamps), unlocks every control that lockSelectionControls()/
// lockPredictionView() locked, restores the date/resolution controls to
// their original defaults, and puts the app back in the exact state it
// was in before any area was ever drawn.
function resetEverything() {
  unlockPredictionView();
  unlockSelectionControls();

  PM25Map.clearSelection();
  clearSelectionState();

  // Restore date/resolution controls to their original HTML defaults --
  // confirmed directly from index.html: single-day-toggle starts
  // checked, resolution-select's first (default-selected) option is
  // "hourly", both date inputs start empty.
  el("single-day-toggle").checked = true;
  el("start-date").value = "";
  el("end-date").value = "";
  el("resolution-select").value = "hourly";
  updateDateModeUI();
  updateRequestPreview();

  state.predictionComplete = false;
  const btn = el("generate-btn");
  btn.textContent = "Generate PM2.5 Prediction Map";
  btn.classList.remove("reset-mode");
}

async function renderPredictionOnMap(result) {
  if (!state.lastClippedPolygon) return; // nothing to draw geometry for

  let geometry;
  try {
    geometry = await apiPost("/api/select_cells_grid", { polygon: state.lastClippedPolygon });
  } catch (err) {
    showValidationMessage("Predictions completed, but could not load cell geometry to display them: " + err.message, "error");
    return;
  }

  const valuesByTimestamp = {};
  for (const ts of result.timestamps) valuesByTimestamp[ts] = {};
  for (const row of result.predictions) {
    valuesByTimestamp[row.datetime_utc][row.grid_id] = row.pm25_prediction;
  }

  predictionViz = {
    cellIds: geometry.cell_ids, lons: geometry.lons, lats: geometry.lats,
    timestamps: result.timestamps, valuesByTimestamp, temporalResolution: result.temporal_resolution || "hourly",
  };

  // Geometry (all cell polygons, the canvas renderer, tooltip bindings)
  // built exactly ONCE here, for the first timestamp -- renderPredictionFrame
  // (called below, and on every later slider move/playback tick) only
  // recolors these same already-built layers, never rebuilds them.
  PM25Map.drawPredictedCells(geometry.cell_ids, geometry.lons, geometry.lats, valuesByTimestamp[result.timestamps[0]]);
  lockPredictionView();

  el("prediction-viz-controls").style.display = "block";

  const slider = el("time-slider");
  const sliderRow = el("time-slider-row");
  if (result.timestamps.length > 1) {
    sliderRow.style.display = "flex";
    slider.min = 0;
    slider.max = result.timestamps.length - 1;
    slider.value = 0;
  } else {
    sliderRow.style.display = "none";
  }

  renderPredictionFrame(0);
}

function renderPredictionFrame(index) {
  if (!predictionViz) return;
  const ts = predictionViz.timestamps[index];
  const values = predictionViz.valuesByTimestamp[ts];
  // PERFORMANCE FIX: geometry (all N cell polygons + the canvas renderer
  // + tooltip bindings) is built exactly ONCE, in renderPredictionOnMap
  // below, when the prediction first completes. Every frame change
  // after that (slider move, playback tick) only recolors the SAME
  // already-built layers via updatePredictedCellColors -- confirmed as
  // the direct fix for the lag/memory growth reported specifically
  // during playback, where the previous version called the equivalent
  // of a full geometry rebuild on every single frame, up to several
  // times per second while looping.
  //
  // Smooth-heatmap toggle: only the currently-VISIBLE visualization is
  // updated per frame -- the other mode's layer is simply left as-is
  // until the user switches to it (at which point the toggle handler
  // below explicitly re-renders the current frame into it once). This
  // avoids paying the cost of updating an invisible layer on every
  // single playback tick.
  if (PM25Map.getVisualizationMode() === "smooth") {
    PM25Map.renderSmoothFrame(values);
  } else {
    PM25Map.updatePredictedCellColors(values);
  }
  // BUG FIX, confirmed by direct reproduction: without an explicit
  // timeZone, toLocaleString() renders using the VIEWER's own browser/
  // OS system timezone, not Philippine local time -- inconsistent with
  // the rest of this app, which treats every date as Asia/Manila local
  // time throughout (the date picker itself appends "+08:00" when
  // building the request). Confirmed this could show the wrong DATE
  // entirely, not just the wrong hour, for a viewer whose system
  // timezone differs from Manila's (e.g. a UTC timestamp that is
  // 8:00 AM Jan 1 in Manila rendered as "Dec 31, 7:00 PM" for a viewer
  // in US Eastern time). Explicit timeZone forces Asia/Manila
  // regardless of the viewer's own machine/browser settings.
  // Daily-resolution results represent a whole-day average, not a
  // specific hour's measurement -- showing an hour/minute (always
  // midnight, since that's the representative timestamp the backend
  // uses for a daily-averaged row) would misleadingly imply a specific
  // time was measured. Omit the time component for daily results.
  el("time-slider-label").textContent = formatFrameTimestamp(ts);

  // Keep the hover inspection panel in sync with the displayed frame --
  // if the mouse is resting on a cell while the slider moves or playback
  // ticks, the readout follows the timestamp (pure client-side lookup).
  renderInspectionPanel();
  // Same for the cell-detail modal's "at current frame" classification
  // line, if the modal happens to be open during slider/playback changes.
  renderCellDetailClassification();
}

// The slider label's own formatting (see the timezone/daily notes in
// renderPredictionFrame above), factored out so the inspection panel
// shows exactly the same timestamp text as the slider.
function formatFrameTimestamp(ts) {
  const isDaily = predictionViz && predictionViz.temporalResolution === "daily";
  const formatOptions = isDaily
    ? { month: "short", day: "numeric", timeZone: "Asia/Manila" }
    : { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit", timeZone: "Asia/Manila" };
  return new Date(ts).toLocaleString(undefined, formatOptions) + (isDaily ? " (daily avg, PHT)" : " PHT");
}

// ---------------------------------------------------------------
// Grid-cell hover inspection panel. Purely a lookup into the
// already-loaded predictionViz data for (hovered cell, current slider
// frame) -- no backend, GEE, or model request is ever made on hover.
// Only the gridded cell layer reports hover (see PM25Map.onCellHover);
// in smooth-heatmap mode the panel simply stays idle.
// ---------------------------------------------------------------
let hoveredCellId = null;

function currentFrameTimestamp() {
  if (!predictionViz) return null;
  return predictionViz.timestamps[Number(el("time-slider").value) || 0];
}

function renderInspectionPanel() {
  const panel = el("inspection-panel");
  if (!panel) return;
  const idle = el("inspection-idle");
  const body = el("inspection-body");
  const ts = currentFrameTimestamp();
  if (!predictionViz || hoveredCellId == null || ts == null) {
    // Idle: collapse back to the compact single-line state. The
    // sidebar's scrollable height shrinks with it; stabiliseSidebarScroll
    // keeps the controls above from jumping when that shrink would
    // otherwise clamp the scroll position (see its comment).
    const container = document.querySelector(".control-panel");
    const heightBefore = container ? container.scrollHeight : 0;
    const scrollTopBefore = container ? container.scrollTop : 0; // read BEFORE the collapse (the browser clamps it during the next layout)
    idle.style.display = "block";
    body.style.display = "none";
    panel.classList.remove("inspection-panel--active");
    if (container && predictionViz) stabiliseSidebarScroll(container, heightBefore, scrollTopBefore);
    else releaseSidebarScrollStabiliser(); // result cleared (reset): nothing left to stabilise
    return;
  }
  const values = predictionViz.valuesByTimestamp[ts] || {};
  const v = values[hoveredCellId];
  const hasValue = v != null && !Number.isNaN(v);
  el("inspection-value").textContent = hasValue ? `${v.toFixed(1)} µg/m³` : "No prediction";
  el("inspection-value").style.color = hasValue ? PM25Map.pm25Color(v) : "";
  el("inspection-timestamp").textContent = formatFrameTimestamp(ts);
  el("inspection-cell").textContent = hoveredCellId;
  const c = cellCentroid(hoveredCellId);
  el("inspection-centroid").textContent = c ? formatCentroid(c) : "—";
  renderInspectionClassification(hasValue ? v : null);
  idle.style.display = "none";
  body.style.display = "block";
  panel.classList.add("inspection-panel--active");
  releaseSidebarScrollStabiliser(); // the panel is tall again; any compensation is no longer needed

  ensureInspectionVisible();
}

// SCROLL STABILISER for the collapse. When the sidebar is scrolled near
// its bottom (which is exactly where the auto-scroll leaves it) and the
// panel collapses, the container's scrollHeight shrinks and the browser
// clamps scrollTop -- shifting every control (scheme selector, legend,
// slider) down under the cursor. Instead of leaving the panel tall, the
// exact deficit is added as bottom padding on the container so nothing
// above moves; the padding is released as soon as it is no longer
// needed (the panel expands again, the user scrolls up far enough, or
// the result is cleared). No fixed heights anywhere.
let sidebarScrollPad = 0;
function stabiliseSidebarScroll(container, scrollHeightBefore, scrollTopBefore) {
  const shrink = scrollHeightBefore - container.scrollHeight;
  if (shrink <= 0) return;
  const maxAfter = container.scrollHeight - container.clientHeight;
  const deficit = Math.ceil(scrollTopBefore - maxAfter);
  if (deficit <= 0) return; // not near the bottom: nothing would clamp
  const basePad = sidebarScrollPad ? 0 : parseFloat(getComputedStyle(container).paddingBottom) || 0;
  sidebarScrollPad += Math.min(deficit, shrink) + basePad; // keep the stylesheet's own padding
  container.style.paddingBottom = `${sidebarScrollPad}px`;
  container.scrollTop = scrollTopBefore; // undo the clamp now that the height allows it
}
function releaseSidebarScrollStabiliser() {
  if (!sidebarScrollPad) return;
  const container = document.querySelector(".control-panel");
  sidebarScrollPad = 0;
  if (container) container.style.paddingBottom = "";
}
// Release once the user has scrolled up far enough that removing the
// padding cannot clamp the scroll position.
document.querySelector(".control-panel").addEventListener("scroll", (e) => {
  if (!sidebarScrollPad) return;
  const c = e.currentTarget;
  if (c.scrollTop + c.clientHeight <= c.scrollHeight - sidebarScrollPad) releaseSidebarScrollStabiliser();
});

// AUTO-SCROLL: the hover happens on the map, so the browser never scrolls
// the sidebar for us. When the panel is (newly) populated and part of it
// sits outside the control panel's visible area, scroll THAT container
// (.control-panel is the only scrollable element -- the page itself is
// overflow:hidden) by the minimum amount that brings the region
// "scheme selector .. inspection panel .. time slider" into view,
// prioritising the inspection panel itself when the region is taller
// than the viewport. Nothing happens if the panel is already fully
// visible, so moving between cells never re-jumps; window height and
// resizing are handled naturally because the check uses live geometry.
// The scroll is applied instantly (not animated): it happens at most
// once per newly-revealed panel, and an instant adjustment can't be
// interrupted or stack up during rapid hover changes.
function ensureInspectionVisible() {
  const container = document.querySelector(".control-panel");
  const panel = el("inspection-panel");
  if (!container || !panel) return;

  const cRect = container.getBoundingClientRect();
  const toContainerY = (rect) => rect.top - cRect.top + container.scrollTop; // content coordinates
  const pRect = panel.getBoundingClientRect();
  const panelTop = toContainerY(pRect);
  const panelBottom = panelTop + pRect.height;
  const viewTop = container.scrollTop;
  const viewBottom = viewTop + container.clientHeight;
  const MARGIN = 8;

  if (panelTop >= viewTop + MARGIN - 1 && panelBottom <= viewBottom - MARGIN + 1) return; // already visible

  // Preferred region: from the scheme selector down to the slider row.
  const selectorGroup = el("classification-select").closest(".form-group");
  const sliderRow = el("time-slider-row");
  let regionTop = selectorGroup ? toContainerY(selectorGroup.getBoundingClientRect()) : panelTop;
  let regionBottom = (sliderRow && sliderRow.style.display !== "none")
    ? toContainerY(sliderRow.getBoundingClientRect()) + sliderRow.getBoundingClientRect().height
    : panelBottom;
  regionTop = Math.min(regionTop, panelTop);
  regionBottom = Math.max(regionBottom, panelBottom);

  let target;
  if (regionBottom - regionTop + 2 * MARGIN <= container.clientHeight) {
    // Whole region fits: scroll the minimum distance that shows all of it.
    if (regionTop < viewTop + MARGIN) target = regionTop - MARGIN;
    else target = regionBottom + MARGIN - container.clientHeight;
  } else if (panelBottom - panelTop + 2 * MARGIN <= container.clientHeight) {
    // Region too tall: show the whole inspection panel, with the slider
    // (below it) preferred over the selector (above it).
    target = Math.min(panelTop - MARGIN, regionBottom + MARGIN - container.clientHeight);
    target = Math.max(target, panelBottom + MARGIN - container.clientHeight);
  } else {
    target = panelTop - MARGIN; // very short viewport: at least show the top of the panel
  }
  target = Math.max(0, Math.min(target, container.scrollHeight - container.clientHeight));
  if (Math.abs(target - container.scrollTop) < 2) return;
  container.scrollTop = target;
}

// Category/interpretation rows -- populated by the classification layer
// (Step 3) only when a scheme is selected; hidden otherwise. Defined here
// as a no-op hook so the panel works before/without any scheme.
function renderInspectionClassification(value) {
  const row = el("inspection-category-row");
  if (!row) return;
  if (typeof classifyPm25 !== "function") { row.style.display = "none"; return; }
  const cls = classifyPm25(value);
  if (!cls) { row.style.display = "none"; return; }
  const label = el("inspection-category");
  label.textContent = cls.label;
  label.style.color = readableCategoryColor(cls.color, label);
  el("inspection-interpretation").textContent = cls.interpretation || "";
  el("inspection-basis").textContent = cls.basisNote || "";
  row.style.display = "block";
}

// READABLE CATEGORY TEXT: category colours come from the classification
// framework (each scheme's own `color`). Used as-is for legend swatches,
// but as TEXT some of them -- yellow (EPA Moderate, WHO IT-3 band),
// bright greens -- are unreadable on the light-mode panel background.
// The text colour is therefore the category colour with its lightness
// adjusted just enough (same hue and saturation, so the identity stays
// recognisable) to reach the WCAG AA text contrast ratio (4.5:1) against
// the element's CURRENT computed background, by darkening. Applied only
// on LIGHT surfaces (background luminance above LIGHT_SURFACE_LUMINANCE):
// dark mode's current appearance is deliberately left exactly as it is.
// Colours that already meet the ratio are returned unchanged. Generic
// across schemes; the panel background itself is never changed.
const TEXT_CONTRAST_MIN_RATIO = 4.5;
const LIGHT_SURFACE_LUMINANCE = 0.3;
const readableColorCache = new Map();

function readableCategoryColor(categoryColor, contextEl) {
  const cat = parseCssColor(categoryColor);
  if (!cat) return categoryColor || "";
  const bgStr = backgroundColorBehind(contextEl);
  const bg = parseCssColor(bgStr);
  if (!bg) return categoryColor;
  const key = `${categoryColor}|${bgStr}`;
  if (readableColorCache.has(key)) return readableColorCache.get(key);

  let out = categoryColor;
  const onLightSurface = relativeLuminance(bg) > LIGHT_SURFACE_LUMINANCE;
  if (onLightSurface && contrastRatio(cat, bg) < TEXT_CONTRAST_MIN_RATIO) {
    // Darken in steps (never "relative to the colour": a mid-green that
    // is already darker than a white panel still needs to get darker).
    const [h, s, l0] = rgbToHsl(cat);
    let best = null;
    for (let step = 1; step <= 40; step++) {
      const l = l0 - step * 0.02;
      if (l < 0) break;
      const rgb = hslToRgb(h, s, l);
      if (contrastRatio(rgb, bg) >= TEXT_CONTRAST_MIN_RATIO) { best = rgb; break; }
    }
    if (best) out = rgbToHexString(best);
  }
  readableColorCache.set(key, out);
  return out;
}

// The first non-transparent computed background up the ancestor chain.
function backgroundColorBehind(elm) {
  let node = elm;
  while (node && node !== document.documentElement) {
    const bg = getComputedStyle(node).backgroundColor;
    if (bg && bg !== "transparent" && !/rgba\(\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*0\s*\)/.test(bg)) return bg;
    node = node.parentElement;
  }
  return getComputedStyle(document.body).backgroundColor;
}

function rgbToHsl([r, g, b]) {
  r /= 255; g /= 255; b /= 255;
  const max = Math.max(r, g, b), min = Math.min(r, g, b);
  const l = (max + min) / 2;
  if (max === min) return [0, 0, l];
  const d = max - min;
  const s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
  let h;
  if (max === r) h = ((g - b) / d + (g < b ? 6 : 0)) / 6;
  else if (max === g) h = ((b - r) / d + 2) / 6;
  else h = ((r - g) / d + 4) / 6;
  return [h, s, l];
}

function hslToRgb(h, s, l) {
  if (s === 0) { const v = Math.round(l * 255); return [v, v, v]; }
  const hue2rgb = (p, q, t) => {
    if (t < 0) t += 1; if (t > 1) t -= 1;
    if (t < 1 / 6) return p + (q - p) * 6 * t;
    if (t < 1 / 2) return q;
    if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6;
    return p;
  };
  const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
  const p = 2 * l - q;
  return [hue2rgb(p, q, h + 1 / 3), hue2rgb(p, q, h), hue2rgb(p, q, h - 1 / 3)].map((c) => Math.round(c * 255));
}

function rgbToHexString([r, g, b]) {
  return "#" + [r, g, b].map((c) => c.toString(16).padStart(2, "0")).join("");
}

function parseCssColor(str) {
  if (!str) return null;
  const m6 = /^#([0-9a-f]{6})$/i.exec(str.trim());
  if (m6) return [0, 2, 4].map((i) => parseInt(m6[1].slice(i, i + 2), 16));
  const m3 = /^#([0-9a-f]{3})$/i.exec(str.trim());
  if (m3) return m3[1].split("").map((ch) => parseInt(ch + ch, 16));
  const rgb = /^rgba?\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)/i.exec(str.trim());
  if (rgb) return [Number(rgb[1]), Number(rgb[2]), Number(rgb[3])];
  return null;
}

function relativeLuminance([r, g, b]) {
  const lin = (c) => { const s = c / 255; return s <= 0.03928 ? s / 12.92 : Math.pow((s + 0.055) / 1.055, 2.4); };
  return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b);
}

function contrastRatio(rgbA, rgbB) {
  const la = relativeLuminance(rgbA), lb = relativeLuminance(rgbB);
  return (Math.max(la, lb) + 0.05) / (Math.min(la, lb) + 0.05);
}

// Hover -> populate immediately. Mouseout -> collapse after a short
// grace period, cancelled if the pointer lands on another cell first:
// Leaflet fires mouseout/mouseover back-to-back when crossing from one
// cell straight into its neighbour, and collapsing/expanding the panel
// in between would flicker the sidebar on every cell boundary.
const INSPECTION_COLLAPSE_DELAY_MS = 120;
let inspectionCollapseTimer = null;
PM25Map.onCellHover((cellId) => {
  if (inspectionCollapseTimer) { clearTimeout(inspectionCollapseTimer); inspectionCollapseTimer = null; }
  if (cellId != null) {
    hoveredCellId = cellId;
    renderInspectionPanel();
    return;
  }
  inspectionCollapseTimer = setTimeout(() => {
    inspectionCollapseTimer = null;
    hoveredCellId = null; // mouseout -> panel clears and collapses
    renderInspectionPanel();
  }, INSPECTION_COLLAPSE_DELAY_MS);
});

// ---------------------------------------------------------------
// PM2.5 classification (interpretation) layer. Scheme definitions come
// ONLY from the backend (/api/grid_info -> config/pm25_classification.py);
// this code mirrors each scheme's declared lookup_rule and renders the
// result. Presentation only: never changes a prediction, and never
// touches the map's own colour scale (PM25Map.pm25Color / PM25_COLOR_STOPS).
// Default selection is "None -- values only".
// ---------------------------------------------------------------
let classificationSchemes = [];
let selectedSchemeId = "";

function setClassificationSchemes(schemes) {
  classificationSchemes = schemes;
  const select = el("classification-select");
  // Keep the first ("None") option; append one per scheme.
  while (select.options.length > 1) select.remove(1);
  for (const s of schemes) {
    const opt = document.createElement("option");
    opt.value = s.id;
    opt.textContent = s.name;
    select.appendChild(opt);
  }
  select.value = selectedSchemeId;
}

function selectedScheme() {
  return classificationSchemes.find((s) => s.id === selectedSchemeId) || null;
}

// EPA convention: truncate (not round) to one decimal before lookup. The
// tiny nudge avoids binary-float artefacts such as 9.1 -> 9.0999 -> 9.0.
function truncate1dp(v) { return Math.floor(v * 10 + 1e-9) / 10; }

// Returns {label, color, interpretation, lower, upper, basisNote, schemeName}
// or null when no scheme is selected / value is missing. Mirrors
// config/pm25_classification.py::classify exactly.
function classifyPm25(value) {
  const scheme = selectedScheme();
  if (!scheme || value == null || Number.isNaN(value) || value < 0) return null;
  const cats = scheme.categories;
  let match = null;
  if (scheme.lookup_rule === "truncate_1dp_inclusive") {
    const v = truncate1dp(value);
    match = cats.find((c) => v >= c.lower && (c.upper == null || v <= c.upper)) || null;
  } else { // le_inclusive
    for (let i = 0; i < cats.length; i++) {
      const c = cats[i];
      const lowerOk = i === 0 ? value >= c.lower : value > c.lower;
      if (lowerOk && (c.upper == null || value <= c.upper)) { match = c; break; }
    }
  }
  if (!match) return null;
  const isDaily = predictionViz && predictionViz.temporalResolution === "daily";
  return {
    ...match,
    schemeName: scheme.name,
    basisNote: scheme.basis_notes ? scheme.basis_notes[isDaily ? "daily" : "hourly"] : "",
  };
}

function formatRange(c, units) {
  if (c.upper == null) return `${c.lower}+ ${units}`;
  return `${c.lower}–${c.upper} ${units}`;
}

function renderClassificationLegend() {
  const box = el("classification-legend");
  const scheme = selectedScheme();
  if (!scheme || !predictionViz) { box.style.display = "none"; box.innerHTML = ""; return; }
  const isDaily = predictionViz.temporalResolution === "daily";
  const bands = scheme.categories.map((c) =>
    `<div class="classification-band"><span class="classification-swatch" style="background:${c.color}"></span>` +
    `<span>${c.label}</span><span class="classification-range">${formatRange(c, scheme.units)}</span></div>`).join("");
  box.innerHTML =
    `<div class="classification-legend-title">${scheme.name}</div>` +
    `<div class="classification-legend-meta">${scheme.source} · ${scheme.averaging_period} ${scheme.units}` +
    (scheme.colors_official ? "" : " · colours are this app's own, not official") + `</div>` +
    bands +
    `<div class="classification-basis">${scheme.basis_notes ? scheme.basis_notes[isDaily ? "daily" : "hourly"] : ""}</div>`;
  box.style.display = "block";
}

// Cell-detail modal: category of THIS cell's value at the current slider
// frame (the same timestamp the map is showing), with the basis note.
// Hidden in area mode and when no scheme is selected.
function renderCellDetailClassification() {
  const box = el("cell-detail-classification");
  if (!cellDetailState || cellDetailState.mode !== "cell" || !predictionViz) { box.style.display = "none"; return; }
  const ts = currentFrameTimestamp();
  const v = ts != null ? predictionViz.valuesByTimestamp[ts][cellDetailState.cellId] : null;
  const cls = classifyPm25(v == null ? null : v);
  if (!cls) { box.style.display = "none"; return; }
  const labelColor = readableCategoryColor(cls.color, box); // same readability rule as the inspection panel
  box.innerHTML =
    `<div><span class="info-label">At ${formatFrameTimestamp(ts)}:</span> <strong>${v.toFixed(1)} µg/m³</strong> — ` +
    `<strong style="color:${labelColor}">${cls.label}</strong> <span class="info-label">(${cls.schemeName})</span></div>` +
    `<div class="inspection-interpretation">${cls.interpretation}</div>` +
    `<div class="inspection-basis">${cls.basisNote}</div>`;
  box.style.display = "block";
}

el("classification-select").addEventListener("change", (e) => {
  selectedSchemeId = e.target.value;
  renderClassificationLegend();
  renderInspectionPanel();
  renderCellDetailClassification();
});

// ---------------------------------------------------------------
// Prediction export (CSV / GeoJSON) -- plain links to the backend's
// streamed export of the completed job's retained result. The GeoJSON
// link optionally narrows to the slider's current frame; its href is
// recomputed at click time so it always reflects the slider position.
// ---------------------------------------------------------------
function exportBaseUrl() {
  return activePredictionJobId ? `/api/prediction_export/${activePredictionJobId}` : null;
}

function updateExportLinks() {
  const base = exportBaseUrl();
  const row = el("export-row");
  if (!base) { row.style.display = "none"; return; }
  el("export-csv-link").href = `${base}?format=csv`;
  el("export-geojson-link").href = `${base}?format=geojson`;
  row.style.display = "flex";
}

el("export-geojson-link").addEventListener("click", (e) => {
  const base = exportBaseUrl();
  if (!base) { e.preventDefault(); return; }
  const ts = currentFrameTimestamp();
  const frameOnly = el("export-current-frame-only").checked && ts != null;
  e.currentTarget.href = `${base}?format=geojson` + (frameOnly ? `&timestamp=${encodeURIComponent(ts)}` : "");
});

el("time-slider").addEventListener("input", (e) => renderPredictionFrame(Number(e.target.value)));

el("time-slider-play").addEventListener("click", () => {
  const slider = el("time-slider");
  const btn = el("time-slider-play");
  if (playbackTimer) {
    clearInterval(playbackTimer);
    playbackTimer = null;
    btn.textContent = "▶";
    return;
  }
  btn.textContent = "⏸";
  playbackTimer = setInterval(() => {
    let next = Number(slider.value) + 1;
    if (next > Number(slider.max)) next = 0;
    slider.value = next;
    renderPredictionFrame(next);
  }, 800);
});

function showPredictionResult(message, kind) {
  const banner = el("prediction-result-banner");
  banner.textContent = message;
  banner.className = `prediction-result-banner ${kind}`;
  banner.style.display = "block";
}

function renderJobStatus(job) {
  el("prediction-elapsed").textContent = `${job.elapsed_seconds}s`;
  el("prediction-stage-label").textContent = job.stage_name;
  // Real, measurable progress: stage_index / total stages (7) -- NOT a
  // fake incrementing percentage. Each stage boundary is an actual
  // pipeline milestone (see backend/prediction/job_state.py's STAGES).
  const pct = Math.round((job.stage_index / 7) * 100);
  el("prediction-progress-bar").style.width = `${pct}%`;
  el("prediction-log-console").textContent = job.log_lines.join("\n");
  el("prediction-log-console").scrollTop = el("prediction-log-console").scrollHeight;
}

async function pollPredictionJob(jobId) {
  try {
    const job = await apiGet(`/api/prediction_status/${jobId}`);
    renderJobStatus(job);

    if (job.status === "complete") {
      const finalElapsed = job.elapsed_seconds;
      hidePredictionProgress();
      const result = job.result;
      showPredictionResult(
        `Prediction complete in ${finalElapsed}s: ${result.row_count} prediction(s) for ${result.cell_ids.length} cell(s) `
        + `across ${result.timestamps.length} timestamp(s).`
        + (result.rows_with_missing_values > 0 ? ` (${result.rows_with_missing_values} row(s) had missing predictor values.)` : ""),
        "success",
      );
      // Selected cells/dates remain exactly as the user left them --
      // nothing here clears state.selectedPolygon, state.cellIds, or the
      // date controls, per explicit requirement.
      await renderPredictionOnMap(result);
    } else if (job.status === "error") {
      hidePredictionProgress();
      showPredictionResult(`Prediction failed (${job.error.type}): ${job.error.message}`, "error");
    } else if (job.status === "cancelled") {
      hidePredictionProgress();
      showPredictionResult("Prediction cancelled.", "cancelled");
    }
    // else: still running -- next poll tick will check again
  } catch (err) {
    hidePredictionProgress();
    showPredictionResult(err.message || "Lost contact with the prediction job.", "error");
  }
}

el("cancel-prediction-btn").addEventListener("click", async () => {
  if (!activePredictionJobId) return;
  try {
    await apiPost(`/api/cancel_prediction/${activePredictionJobId}`, {});
  } catch (err) {
    // Cancellation request itself failing (e.g. job already finished) is
    // not surfaced as an error -- the next poll tick will show the job's
    // real final state regardless.
  }
});

el("generate-btn").addEventListener("click", async () => {
  if (state.predictionComplete) {
    resetEverything();
    return;
  }
  if (state.cellCount === 0) {
    showValidationMessage("Please draw an area on the map before continuing.", "error");
    return;
  }
  const startVal = el("start-date").value;
  if (!startVal) {
    showValidationMessage("Please select a start date before continuing.", "error");
    return;
  }
  const dateError = validateDateRange();
  if (dateError) {
    showValidationMessage(dateError, "error");
    return;
  }
  const endVal = getEffectiveEndDate();

  const payload = {
    cell_ids: state.cellIds,
    start_datetime: `${startVal}T00:00:00+08:00`,
    end_datetime: `${endVal}T23:00:00+08:00`,
    temporal_resolution: getTemporalResolution(),
  };

  try {
    const validation = await apiPost("/api/validate_request", {
      cell_count: state.cellCount, start_datetime: payload.start_datetime,
      end_datetime: payload.end_datetime, temporal_resolution: payload.temporal_resolution,
    });
    if (!validation.is_valid) {
      showValidationMessage(validation.message, "error");
      return;
    }

    const started = await apiPost("/api/generate_prediction", payload);
    if (started.status === "error") {
      showValidationMessage(started.message, "error");
      return;
    }

    activePredictionJobId = started.job_id;
    showPredictionProgress();
    if (predictionPollTimer) { clearInterval(predictionPollTimer); } // defensive: never let a stale interval reference get overwritten and orphaned
    predictionPollTimer = setInterval(() => pollPredictionJob(activePredictionJobId), 1500);
    pollPredictionJob(activePredictionJobId); // immediate first check, don't wait for the first interval tick
  } catch (err) {
    showValidationMessage(err.message || "Request failed.", "error");
  }
});

// ---------------------------------------------------------------
// Init
// ---------------------------------------------------------------

// Fetches the prediction-date range from the backend's own
// background-checked GEE availability state (see
// backend/prediction/data_availability.py) -- never queries GEE
// directly. Sets the existing native <input type="date"> min/max
// attributes so the browser's own calendar UI enforces the range; no
// new UI is introduced. The backend check runs in a background thread
// and may still be "pending" when the page first loads, so this polls
// a few times with a short, bounded total wait -- NOT indefinitely --
// rather than assuming a single immediate fetch will already have the
// answer. If it's still pending after that, the date inputs are simply
// left without a max for now; the existing backend-side validation in
// pipeline.py still refuses any request it can't verify, so this only
// affects how early the user sees that refusal, not whether an invalid
// request could actually go through.
const AVAILABILITY_POLL_INTERVAL_MS = 2000;
const AVAILABILITY_POLL_MAX_ATTEMPTS = 10; // 10 x 2s = 20s bounded total wait

async function initPredictionDateRange(attempt = 1) {
  let info;
  try {
    info = await apiGet("/api/grid_info");
  } catch (err) {
    console.error("Could not load the prediction date range:", err);
    return;
  }

  if (info.min_prediction_date) {
    el("start-date").min = info.min_prediction_date;
    el("end-date").min = info.min_prediction_date;
  }

  // Classification schemes ride along on the same response (backend is
  // the only place thresholds are defined). Loaded once; the list does
  // not change while the backend runs.
  if (Array.isArray(info.pm25_classification_schemes) && classificationSchemes.length === 0) {
    setClassificationSchemes(info.pm25_classification_schemes);
  }

  if (info.availability_status === "ready" && info.max_prediction_date) {
    el("start-date").max = info.max_prediction_date;
    el("end-date").max = info.max_prediction_date;
    return;
  }

  if (info.availability_status === "failed") {
    showValidationMessage(info.availability_error, "error");
    return;
  }

  // Still "pending" -- the background check is still running. Poll a
  // few more times, bounded, rather than forever.
  if (attempt < AVAILABILITY_POLL_MAX_ATTEMPTS) {
    setTimeout(() => initPredictionDateRange(attempt + 1), AVAILABILITY_POLL_INTERVAL_MS);
  } else {
    console.warn("Prediction date availability check is still pending after " +
                  `${AVAILABILITY_POLL_MAX_ATTEMPTS} attempts; leaving the date picker's upper bound unset for now.`);
  }
}

initPredictionDateRange();
PM25Map.init("map-leaflet", "#map-status", handlePolygonDrawn);
