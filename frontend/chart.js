// ---------------------------------------------------------------
// PM2.5Chart -- a minimal, self-contained SVG line-chart renderer.
// Deliberately not pulling in an external charting library for what's
// just one line chart with a handful of points -- a hand-built SVG
// keeps this dependency-free and small.
// ---------------------------------------------------------------
const PM25Chart = (() => {
  const DEFAULT_WIDTH = 640;
  const MIN_WIDTH = 320;
  const HEIGHT = 280;
  const PAD = { top: 16, right: 16, bottom: 40, left: 44 };
  const svgNS = "http://www.w3.org/2000/svg";

  // RESPONSIVE X-AXIS. The SVG's viewBox width now follows the container's
  // real width (re-rendered on resize via a ResizeObserver), so axis text
  // stays at its true 10px size instead of being scaled down with a
  // fixed 640-unit viewBox in a narrower modal. How many labels fit is
  // then derived from the longest label's estimated rendered width and
  // the plot width, and the stride between labelled points is rounded up
  // to a "nice" hour multiple (1,2,3,4,6,8,12,24h ...) so hourly series
  // get labels at consistent clock times. The first and last points are
  // always labelled; the last label is forced, so the stride label that
  // would otherwise collide with it is dropped rather than overlapping.
  // Exact per-point timestamps remain available via the hover crosshair
  // tooltip and each point's <title>, which are unaffected.
  const AXIS_FONT_PX = 10;
  const AVG_CHAR_PX = 0.58 * AXIS_FONT_PX; // typical sans-serif average glyph width
  const LABEL_GAP_PX = 14;
  const NICE_STRIDES = [1, 2, 3, 4, 6, 8, 12, 24, 48, 72, 96, 120, 168, 240, 336, 504, 720];

  // The first label is anchored at its start and the last at its end
  // (see render), so neither can be clipped at the plot edges; that
  // means each of them occupies a full label width INSIDE the plot,
  // which the spacing rules below account for.
  function chooseLabelIndices(labels, plotW) {
    const n = labels.length;
    if (n <= 1) return new Set(n ? [0] : []);
    const longest = labels.reduce((m, s) => Math.max(m, String(s).length), 1);
    const labelW = longest * AVG_CHAR_PX;
    const slotW = labelW + LABEL_GAP_PX;
    const maxLabels = Math.max(2, Math.floor((plotW - labelW) / slotW) + 1);
    const rawStride = Math.ceil((n - 1) / (maxLabels - 1));
    let stride = NICE_STRIDES.find((s) => s >= rawStride);
    if (!stride) stride = Math.ceil(rawStride / 24) * 24;
    const xStep = plotW / (n - 1);
    const chosen = new Set();
    for (let i = 0; i < n; i += stride) chosen.add(i);
    chosen.add(n - 1);
    // Drop stride labels that would collide with the edge-anchored first
    // or (forced) last label: a centred label next to an edge-anchored
    // one needs 1.5 label widths plus the gap between their ticks.
    const edgeClearance = 1.5 * labelW + LABEL_GAP_PX;
    for (const i of Array.from(chosen)) {
      if (i === 0 || i === n - 1) continue;
      if ((n - 1 - i) * xStep < edgeClearance || i * xStep < edgeClearance) chosen.delete(i);
    }
    return chosen;
  }

  // labels: array of x-axis label strings (one per point) -- also reused
  // verbatim as the hover crosshair's displayed date/time label.
  // values: array of numbers or null (null = missing prediction, renders as a gap)
  // pointColorFn: optional (value) => hex color, defaults to a single accent color
  function render(container, args) {
    const { labels, values, pointColorFn, yUnit = "µg/m³" } = args;
    container.innerHTML = "";
    if (!values || values.length === 0) {
      container.innerHTML = '<div class="chart-empty">No data to display.</div>';
      return;
    }
    observeResize(container, args);

    const WIDTH = Math.max(MIN_WIDTH, Math.round(container.clientWidth) || DEFAULT_WIDTH);
    const plotW = WIDTH - PAD.left - PAD.right;
    const plotH = HEIGHT - PAD.top - PAD.bottom;

    const validValues = values.filter((v) => v != null && !Number.isNaN(v));
    const yMin = validValues.length ? Math.min(0, Math.min(...validValues)) : 0;
    const yMaxRaw = validValues.length ? Math.max(...validValues) : 1;
    const yMax = yMaxRaw <= 0 ? 1 : yMaxRaw * 1.12; // small headroom so the line never touches the top edge

    const xStep = values.length > 1 ? plotW / (values.length - 1) : 0;
    const xAt = (i) => PAD.left + i * xStep;
    const yAt = (v) => PAD.top + plotH - ((v - yMin) / (yMax - yMin)) * plotH;

    const svg = document.createElementNS(svgNS, "svg");
    svg.setAttribute("viewBox", `0 0 ${WIDTH} ${HEIGHT}`);
    svg.setAttribute("class", "pm25-chart-svg");

    // Y-axis gridlines + labels (4 evenly-spaced ticks)
    const nTicks = 4;
    for (let t = 0; t <= nTicks; t++) {
      const v = yMin + ((yMax - yMin) * t) / nTicks;
      const y = yAt(v);
      const line = document.createElementNS(svgNS, "line");
      line.setAttribute("x1", PAD.left); line.setAttribute("x2", WIDTH - PAD.right);
      line.setAttribute("y1", y); line.setAttribute("y2", y);
      line.setAttribute("class", "pm25-chart-gridline");
      svg.appendChild(line);

      const label = document.createElementNS(svgNS, "text");
      label.setAttribute("x", PAD.left - 6); label.setAttribute("y", y + 3);
      label.setAttribute("class", "pm25-chart-axis-label"); label.setAttribute("text-anchor", "end");
      label.textContent = v.toFixed(v < 10 ? 1 : 0);
      svg.appendChild(label);
    }

    // X-axis labels -- subsampled to what actually fits the plot width
    // (see chooseLabelIndices above); every point keeps its exact label
    // in the tooltip.
    const labelIndices = chooseLabelIndices(labels, plotW);
    labels.forEach((lab, i) => {
      if (!labelIndices.has(i)) return;
      const text = document.createElementNS(svgNS, "text");
      text.setAttribute("x", xAt(i)); text.setAttribute("y", HEIGHT - PAD.bottom + 16);
      text.setAttribute("class", "pm25-chart-axis-label");
      // Edge labels are anchored inward so they are never clipped by the
      // SVG's edge (a centred last label sat half outside the plot).
      const anchor = (labels.length > 1 && i === 0) ? "start" : (i === labels.length - 1) ? "end" : "middle";
      text.setAttribute("text-anchor", anchor);
      text.textContent = lab;
      svg.appendChild(text);
    });

    // Line path -- broken into separate segments across any null (missing) values,
    // so a gap in data renders as a visible gap, not a misleading straight line across it.
    let pathD = "";
    let penDown = false;
    values.forEach((v, i) => {
      if (v == null || Number.isNaN(v)) { penDown = false; return; }
      const x = xAt(i), y = yAt(v);
      pathD += (penDown ? " L " : " M ") + x + " " + y;
      penDown = true;
    });
    if (pathD) {
      const path = document.createElementNS(svgNS, "path");
      path.setAttribute("d", pathD);
      path.setAttribute("class", "pm25-chart-line");
      svg.appendChild(path);
    }

    // Point markers, colored to match the map's own PM2.5 color scale
    values.forEach((v, i) => {
      if (v == null || Number.isNaN(v)) return;
      const circle = document.createElementNS(svgNS, "circle");
      circle.setAttribute("cx", xAt(i)); circle.setAttribute("cy", yAt(v)); circle.setAttribute("r", 3.5);
      circle.setAttribute("fill", pointColorFn ? pointColorFn(v) : "#2563eb");
      circle.setAttribute("class", "pm25-chart-point");
      const title = document.createElementNS(svgNS, "title");
      title.textContent = `${labels[i]}: ${v.toFixed(1)} ${yUnit}`;
      circle.appendChild(title);
      svg.appendChild(circle);
    });

    // --------------------------------------------------------------
    // Interactive hover crosshair: a vertical line that follows the
    // mouse, snapping to the nearest data point, showing that point's
    // date/time label and PM2.5 value. Built as its own group, hidden
    // by default, shown/repositioned on mousemove over an invisible
    // capture rect spanning the full plot area.
    // --------------------------------------------------------------
    const crosshair = document.createElementNS(svgNS, "g");
    crosshair.setAttribute("class", "pm25-chart-crosshair");
    crosshair.style.display = "none";

    const crosshairLine = document.createElementNS(svgNS, "line");
    crosshairLine.setAttribute("y1", PAD.top); crosshairLine.setAttribute("y2", HEIGHT - PAD.bottom);
    crosshairLine.setAttribute("class", "pm25-chart-crosshair-line");
    crosshair.appendChild(crosshairLine);

    const crosshairDot = document.createElementNS(svgNS, "circle");
    crosshairDot.setAttribute("r", 5);
    crosshairDot.setAttribute("class", "pm25-chart-crosshair-dot");
    crosshair.appendChild(crosshairDot);

    // Tooltip: background box + two lines of text (date/time, then value),
    // sized to its own text content and clamped so it never renders
    // outside the chart's left/right edges.
    const tooltipGroup = document.createElementNS(svgNS, "g");
    const tooltipBg = document.createElementNS(svgNS, "rect");
    tooltipBg.setAttribute("class", "pm25-chart-tooltip-bg");
    tooltipBg.setAttribute("rx", 4);
    const tooltipLabel = document.createElementNS(svgNS, "text");
    tooltipLabel.setAttribute("class", "pm25-chart-tooltip-label");
    const tooltipValue = document.createElementNS(svgNS, "text");
    tooltipValue.setAttribute("class", "pm25-chart-tooltip-value");
    tooltipGroup.appendChild(tooltipBg);
    tooltipGroup.appendChild(tooltipLabel);
    tooltipGroup.appendChild(tooltipValue);
    crosshair.appendChild(tooltipGroup);

    svg.appendChild(crosshair);

    const overlay = document.createElementNS(svgNS, "rect");
    overlay.setAttribute("x", PAD.left); overlay.setAttribute("y", PAD.top);
    overlay.setAttribute("width", plotW); overlay.setAttribute("height", plotH);
    overlay.setAttribute("fill", "transparent");
    overlay.style.cursor = "crosshair";

    function pointFromEvent(evt) {
      const pt = svg.createSVGPoint();
      pt.x = evt.clientX; pt.y = evt.clientY;
      return pt.matrixTransform(svg.getScreenCTM().inverse());
    }

    overlay.addEventListener("mousemove", (evt) => {
      const svgPoint = pointFromEvent(evt);
      let idx = Math.round((svgPoint.x - PAD.left) / (xStep || 1));
      idx = Math.max(0, Math.min(values.length - 1, idx));
      const v = values[idx];
      const x = xAt(idx);

      crosshairLine.setAttribute("x1", x); crosshairLine.setAttribute("x2", x);

      if (v == null || Number.isNaN(v)) {
        crosshairDot.style.display = "none";
        tooltipLabel.textContent = labels[idx];
        tooltipValue.textContent = "No prediction";
      } else {
        const y = yAt(v);
        crosshairDot.style.display = "";
        crosshairDot.setAttribute("cx", x); crosshairDot.setAttribute("cy", y);
        crosshairDot.setAttribute("fill", pointColorFn ? pointColorFn(v) : "#2563eb");
        tooltipLabel.textContent = labels[idx];
        tooltipValue.textContent = `${v.toFixed(1)} ${yUnit}`;
      }

      // Position the tooltip box above the crosshair, clamped so it
      // never overflows the chart's left/right edges.
      const boxW = Math.max(tooltipLabel.getComputedTextLength ? tooltipLabel.getComputedTextLength() : 60,
                             tooltipValue.getComputedTextLength ? tooltipValue.getComputedTextLength() : 60) + 16;
      const boxH = 38;
      let boxX = x - boxW / 2;
      boxX = Math.max(PAD.left, Math.min(WIDTH - PAD.right - boxW, boxX));
      const boxY = PAD.top + 4;

      tooltipBg.setAttribute("x", boxX); tooltipBg.setAttribute("y", boxY);
      tooltipBg.setAttribute("width", boxW); tooltipBg.setAttribute("height", boxH);
      tooltipLabel.setAttribute("x", boxX + 8); tooltipLabel.setAttribute("y", boxY + 14);
      tooltipValue.setAttribute("x", boxX + 8); tooltipValue.setAttribute("y", boxY + 29);

      crosshair.style.display = "";
    });

    overlay.addEventListener("mouseleave", () => { crosshair.style.display = "none"; });

    svg.appendChild(overlay);
    container.appendChild(svg);
  }

  // Re-render the same data when the container's width changes (modal
  // resized with the window, sidebar width changes). One observer per
  // container; re-renders only on an actual width change, never in a loop.
  const resizeObserver = (typeof ResizeObserver !== "undefined") ? new ResizeObserver((entries) => {
    for (const entry of entries) {
      const c = entry.target;
      const state = c.__pm25Chart;
      if (!state) continue;
      const w = Math.round(c.clientWidth);
      if (w && w !== state.width) { state.width = w; render(c, state.args); }
    }
  }) : null;

  const observedContainers = new Set();

  function observeResize(container, args) {
    const w = Math.round(container.clientWidth);
    if (!container.__pm25Chart) {
      container.__pm25Chart = { args, width: w };
      observedContainers.add(container);
      if (resizeObserver) resizeObserver.observe(container);
    } else {
      container.__pm25Chart.args = args;
      container.__pm25Chart.width = w;
    }
  }

  // Window-resize fallback: covers the same "modal got narrower/wider"
  // case via the plain resize event, so a re-render never depends solely
  // on ResizeObserver delivery. Same width-changed guard, no loops.
  window.addEventListener("resize", () => {
    for (const c of observedContainers) {
      if (!c.isConnected) { observedContainers.delete(c); continue; }
      const state = c.__pm25Chart;
      const w = Math.round(c.clientWidth);
      if (state && w && w !== state.width) { state.width = w; render(c, state.args); }
    }
  });

  // ---------------------------------------------------------------
  // Signed SHAP contribution bars (plain DOM, no SVG needed). Renders two
  // lists -- features that pushed the model prediction HIGHER (positive)
  // and LOWER (negative) -- each row showing display name, the feature's
  // actual value, a bar scaled to |shap| / maxAbs, and the signed SHAP
  // value in the model's log1p output space. Purely presentational.
  // ---------------------------------------------------------------
  function formatFeatureValue(v) {
    if (v == null || Number.isNaN(v)) return "missing";
    const a = Math.abs(v);
    if (a === 0) return "0";
    if (a >= 1e5 || a < 1e-3) return v.toExponential(2);
    if (a >= 100) return v.toFixed(0);
    if (a >= 1) return v.toFixed(2);
    return v.toFixed(4);
  }

  function renderContributions(container, { positive, negative }) {
    container.innerHTML = "";
    const all = [...positive, ...negative];
    if (!all.length) {
      container.innerHTML = '<div class="chart-empty">No contributions to display.</div>';
      return;
    }
    const maxAbs = Math.max(...all.map((c) => Math.abs(c.shap_log))) || 1;

    const makeList = (title, items, cls, sign) => {
      const box = document.createElement("div");
      box.className = `shap-list ${cls}`;
      const h = document.createElement("div");
      h.className = "shap-list-title";
      h.textContent = title;
      box.appendChild(h);
      if (!items.length) {
        const none = document.createElement("div");
        none.className = "shap-none";
        none.textContent = "None";
        box.appendChild(none);
      }
      for (const c of items) {
        const row = document.createElement("div");
        row.className = "shap-row";
        const name = document.createElement("div");
        name.className = "shap-name";
        name.textContent = c.display_name;
        name.title = c.feature;
        const val = document.createElement("div");
        val.className = "shap-value";
        val.textContent = `value: ${formatFeatureValue(c.value)}`;
        const barWrap = document.createElement("div");
        barWrap.className = "shap-bar-wrap";
        const bar = document.createElement("div");
        bar.className = `shap-bar ${cls}`;
        bar.style.width = `${Math.max(2, Math.round(100 * Math.abs(c.shap_log) / maxAbs))}%`;
        barWrap.appendChild(bar);
        const num = document.createElement("div");
        num.className = "shap-num";
        num.textContent = `${sign}${Math.abs(c.shap_log).toFixed(3)}`;
        row.appendChild(name); row.appendChild(val); row.appendChild(barWrap); row.appendChild(num);
        box.appendChild(row);
      }
      return box;
    };

    container.appendChild(makeList("Pushed the model prediction higher", positive, "shap-pos", "+"));
    container.appendChild(makeList("Pushed the model prediction lower", negative, "shap-neg", "−"));
  }

  return { render, renderContributions, _chooseLabelIndices: chooseLabelIndices };
})();
