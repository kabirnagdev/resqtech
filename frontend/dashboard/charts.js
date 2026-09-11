/*
 * Minimal, dependency-free SVG chart renderers.
 *
 * Deliberately hand-rolled instead of pulling in a charting library from a
 * CDN: this dashboard is meant to run in the field, possibly with no
 * internet access, so every visualization has to work fully offline.
 * Exposed as `window.Charts` since these are plain <script> includes, not
 * ES modules.
 */
(() => {
  "use strict";

  const SVG_NS = "http://www.w3.org/2000/svg";

  function svgEl(tag, attrs) {
    const el = document.createElementNS(SVG_NS, tag);
    for (const k in attrs) el.setAttribute(k, attrs[k]);
    return el;
  }

  function clear(node) {
    while (node.firstChild) node.removeChild(node.firstChild);
  }

  function emptyState(container, message) {
    clear(container);
    const div = document.createElement("div");
    div.className = "chart-empty";
    div.textContent = message || "No data yet";
    container.appendChild(div);
  }

  // -- line chart -----------------------------------------------------------
  function renderLineChart(container, labels, values, opts = {}) {
    clear(container);
    if (!values || values.length === 0 || values.every((v) => v === 0)) {
      emptyState(container, opts.emptyMessage || "No activity in this window yet");
      return;
    }

    const W = 400, H = 180, padL = 28, padR = 10, padT = 14, padB = 22;
    const maxV = Math.max(1, ...values);
    const n = values.length;
    const stepX = n > 1 ? (W - padL - padR) / (n - 1) : 0;

    const points = values.map((v, i) => {
      const x = padL + i * stepX;
      const y = H - padB - (v / maxV) * (H - padT - padB);
      return [x, y];
    });

    const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, preserveAspectRatio: "none" });

    // gridlines
    for (let g = 0; g <= 2; g++) {
      const y = padT + (g * (H - padT - padB)) / 2;
      svg.appendChild(svgEl("line", { x1: padL, x2: W - padR, y1: y, y2: y, stroke: "#1c2733", "stroke-width": 1 }));
    }

    // area fill
    const areaPath =
      `M ${points[0][0]} ${H - padB} ` +
      points.map((p) => `L ${p[0]} ${p[1]}`).join(" ") +
      ` L ${points[points.length - 1][0]} ${H - padB} Z`;
    svg.appendChild(svgEl("path", { d: areaPath, fill: "rgba(79,143,247,0.14)", stroke: "none" }));

    // line
    const linePath = `M ${points.map((p) => p.join(",")).join(" L ")}`;
    svg.appendChild(svgEl("path", { d: linePath, fill: "none", stroke: "#4f8ff7", "stroke-width": 2 }));

    // current-value marker
    const last = points[points.length - 1];
    svg.appendChild(svgEl("circle", { cx: last[0], cy: last[1], r: 3.2, fill: "#4f8ff7" }));

    // axis labels: max value, and first/last time labels only (avoid clutter)
    svg.appendChild(labelText(padL, padT - 2, String(Math.round(maxV)), { anchor: "start", size: 9, color: "#8994a6" }));
    if (labels && labels.length) {
      svg.appendChild(labelText(padL, H - 6, labels[0], { anchor: "start", size: 9, color: "#8994a6" }));
      svg.appendChild(labelText(W - padR, H - 6, labels[labels.length - 1], { anchor: "end", size: 9, color: "#8994a6" }));
    }

    container.appendChild(svg);
  }

  // -- donut chart ------------------------------------------------------------
  function renderDonutChart(container, segments, opts = {}) {
    clear(container);
    const total = segments.reduce((s, seg) => s + seg.value, 0);

    const wrap = document.createElement("div");
    wrap.style.display = "flex";
    wrap.style.flexDirection = "column";
    wrap.style.height = "100%";

    const svgWrap = document.createElement("div");
    svgWrap.style.flex = "1";
    svgWrap.style.minHeight = "0";

    const size = 140, r = 46, cx = 60, cy = 60, strokeW = 16;
    const circumference = 2 * Math.PI * r;
    const svg = svgEl("svg", { viewBox: `0 0 120 120` });

    // background ring
    svg.appendChild(
      svgEl("circle", { cx, cy, r, fill: "none", stroke: "#1c2733", "stroke-width": strokeW })
    );

    if (total > 0) {
      let offset = 0;
      for (const seg of segments) {
        if (seg.value <= 0) continue;
        const len = (seg.value / total) * circumference;
        const circle = svgEl("circle", {
          cx, cy, r, fill: "none", stroke: seg.color, "stroke-width": strokeW,
          "stroke-dasharray": `${len} ${circumference - len}`,
          "stroke-dashoffset": String(-offset),
          transform: `rotate(-90 ${cx} ${cy})`,
          "stroke-linecap": "butt",
        });
        svg.appendChild(circle);
        offset += len;
      }
    }

    svg.appendChild(labelText(cx, cy - 2, String(total), { anchor: "middle", size: 20, color: "#e8edf4", weight: 700 }));
    svg.appendChild(labelText(cx, cy + 14, (opts.centerLabel || "TOTAL").toUpperCase(), { anchor: "middle", size: 8, color: "#8994a6" }));

    svgWrap.appendChild(svg);
    wrap.appendChild(svgWrap);

    const legend = document.createElement("div");
    legend.className = "chart-legend";
    for (const seg of segments) {
      const item = document.createElement("span");
      const dot = document.createElement("span");
      dot.className = "dot";
      dot.style.background = seg.color;
      item.appendChild(dot);
      item.appendChild(document.createTextNode(`${seg.label} (${seg.value})`));
      legend.appendChild(item);
    }
    wrap.appendChild(legend);

    container.appendChild(wrap);
  }

  // -- bar chart ----------------------------------------------------------------
  function renderBarChart(container, labels, values, opts = {}) {
    clear(container);
    if (!values || values.length === 0) {
      emptyState(container, opts.emptyMessage || "No data yet");
      return;
    }
    const horizontal = !!opts.horizontal;
    const colorFor = opts.colorFor || (() => "#4f8ff7");
    const maxV = Math.max(1, ...values);

    if (horizontal) {
      const rowH = 26, gap = 8, padL = 4, padR = 40, labelW = 70;
      const H = values.length * (rowH + gap);
      const W = 320;
      const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, preserveAspectRatio: "none" });
      values.forEach((v, i) => {
        const y = i * (rowH + gap);
        const barMaxW = W - labelW - padR;
        const barW = Math.max(2, (v / maxV) * barMaxW);
        svg.appendChild(labelText(labelW - 8, y + rowH / 2 + 3, truncate(labels[i], 10), { anchor: "end", size: 10, color: "#8994a6" }));
        svg.appendChild(svgEl("rect", {
          x: labelW, y, width: barMaxW, height: rowH - 6, rx: 4, fill: "#1c2733",
        }));
        svg.appendChild(svgEl("rect", {
          x: labelW, y, width: barW, height: rowH - 6, rx: 4, fill: colorFor(labels[i], i),
        }));
        svg.appendChild(labelText(labelW + barW + 6, y + rowH / 2 + 3, String(v), { anchor: "start", size: 10, color: "#e8edf4" }));
      });
      container.appendChild(svg);
    } else {
      const W = 360, H = 180, padB = 26, padT = 10, gap = 6;
      const barW = (W - gap * (values.length - 1)) / values.length;
      const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, preserveAspectRatio: "none" });
      values.forEach((v, i) => {
        const x = i * (barW + gap);
        const h = (v / maxV) * (H - padT - padB);
        const y = H - padB - h;
        svg.appendChild(svgEl("rect", {
          x, y, width: barW, height: Math.max(1, h), rx: 2, fill: colorFor(labels[i], i),
        }));
        if (v > 0) {
          svg.appendChild(labelText(x + barW / 2, y - 3, String(v), { anchor: "middle", size: 9, color: "#8994a6" }));
        }
        if (labels && labels[i] !== undefined && (i % 2 === 0 || values.length <= 6)) {
          svg.appendChild(labelText(x + barW / 2, H - 8, labels[i], { anchor: "middle", size: 8, color: "#8994a6" }));
        }
      });
      container.appendChild(svg);
    }
  }

  // -- shared helpers -------------------------------------------------------------
  function labelText(x, y, text, { anchor = "start", size = 10, color = "#8994a6", weight = 400 } = {}) {
    const t = svgEl("text", {
      x, y, "text-anchor": anchor, "font-size": size, fill: color,
      "font-family": "-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif", "font-weight": weight,
    });
    t.textContent = text;
    return t;
  }

  function truncate(s, n) {
    if (!s) return "";
    return s.length > n ? s.slice(0, n - 1) + "…" : s;
  }

  window.Charts = { renderLineChart, renderDonutChart, renderBarChart };
})();
