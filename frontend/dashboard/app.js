/*
 * Dashboard wiring for the Search & Rescue Intelligence Dashboard.
 *
 * Data sources:
 *   - WebSocket /ws: a "snapshot" message every ~0.5s (backend/main.py's
 *     WS_BROADCAST_INTERVAL_SEC) with { overview, cameras, tracks_by_camera },
 *     plus one "event" message per rescue event as it happens.
 *   - REST endpoints under /api/* for camera CRUD, tracking history, and
 *     the analytics charts (polled on a slower interval than the WS feed,
 *     since none of that needs sub-second freshness).
 *
 * Plain script, no build step, no framework -- matches charts.js.
 */
(() => {
  "use strict";

  const PRIORITY_COLORS = {
    CRITICAL: "#e4453b",
    HIGH: "#f2994a",
    MEDIUM: "#f2c94c",
    LOW: "#27ae60",
    UNKNOWN: "#8c8c99",
  };
  const PRIORITY_ORDER = ["CRITICAL", "HIGH", "MEDIUM", "LOW", "UNKNOWN"];
  const ANALYTICS_POLL_MS = 4000;
  const MAX_EVENTS_SHOWN = 120;

  // -- state ------------------------------------------------------------
  let ws = null;
  let wsRetryDelay = 1000;
  let cameras = []; // last snapshot's camera describe() list
  let tracksByCamera = {};
  let selectedCameraId = null;
  let eventCount = 0;
  let probedWebcamIndices = [];

  const $ = (id) => document.getElementById(id);

  // -- clock --------------------------------------------------------------
  function tickClock() {
    $("clock").textContent = new Date().toLocaleTimeString();
  }
  setInterval(tickClock, 1000);
  tickClock();

  // -- websocket ------------------------------------------------------------
  function connectWs() {
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    ws = new WebSocket(`${proto}//${location.host}/ws`);

    ws.onopen = () => {
      wsRetryDelay = 1000;
      setSystemStatus(true);
    };
    ws.onclose = () => {
      setSystemStatus(false);
      setTimeout(connectWs, wsRetryDelay);
      wsRetryDelay = Math.min(wsRetryDelay * 1.5, 10000);
    };
    ws.onerror = () => {
      try { ws.close(); } catch (e) { /* noop */ }
    };
    ws.onmessage = (msg) => {
      let data;
      try { data = JSON.parse(msg.data); } catch (e) { return; }
      if (data.type === "snapshot") handleSnapshot(data);
      else if (data.type === "event") handleEvent(data);
    };
  }

  function setSystemStatus(online) {
    const el = $("system-status");
    el.textContent = online ? "ONLINE" : "RECONNECTING";
    el.classList.toggle("status-online", online);
    el.classList.toggle("status-offline", !online);
  }

  // -- snapshot handling ------------------------------------------------------
  function handleSnapshot(data) {
    cameras = data.cameras || [];
    tracksByCamera = data.tracks_by_camera || {};

    renderKpis(data.overview || {});
    renderCameraSelect();
    renderCamerasTable();
    renderHistoryCameraFilter();

    if (selectedCameraId === null && cameras.length) {
      selectCamera(cameras[0].camera_id);
    } else if (selectedCameraId !== null && !cameras.some((c) => c.camera_id === selectedCameraId)) {
      selectCamera(cameras.length ? cameras[0].camera_id : null);
    } else {
      renderSubjects();
      renderVideoState();
    }

    renderCameraStrip();
  }

  function renderKpis(overview) {
    $("kpi-cameras").textContent = overview.active_cameras ?? 0;
    $("kpi-cameras-sub").textContent = `of ${overview.total_cameras ?? 0} configured`;
    $("kpi-people").textContent = overview.people_detected ?? 0;
    $("kpi-high").textContent = overview.high_priority ?? 0;
    $("kpi-critical").textContent = overview.critical ?? 0;
  }

  // -- camera selection / video panel ------------------------------------------
  function selectCamera(cameraId) {
    selectedCameraId = cameraId;
    const sel = $("camera-select");
    if (sel.value !== (cameraId || "")) sel.value = cameraId || "";
    renderVideoState();
    renderSubjects();
    loadHistory();
  }

  function renderVideoState() {
    const img = $("video");
    const empty = $("video-empty");
    const tag = $("camera-source-tag");
    const cam = cameras.find((c) => c.camera_id === selectedCameraId);

    if (!cam) {
      img.removeAttribute("src");
      empty.hidden = false;
      empty.textContent = cameras.length ? "Select a camera" : "No cameras configured";
      tag.textContent = "--";
      return;
    }
    empty.hidden = true;
    tag.textContent = cam.source || "RGB";
    const wantedSrc = `/video_feed/${cam.camera_id}`;
    if (!img.src.endsWith(wantedSrc)) img.src = wantedSrc;
  }

  function renderCameraSelect() {
    const sel = $("camera-select");
    const prev = sel.value;
    sel.innerHTML = "";
    if (!cameras.length) {
      const opt = document.createElement("option");
      opt.value = "";
      opt.textContent = "No cameras";
      sel.appendChild(opt);
      return;
    }
    for (const cam of cameras) {
      const opt = document.createElement("option");
      opt.value = cam.camera_id;
      opt.textContent = `${cam.camera_id} — ${cam.name}`;
      sel.appendChild(opt);
    }
    if (cameras.some((c) => c.camera_id === prev)) sel.value = prev;
    else if (selectedCameraId && cameras.some((c) => c.camera_id === selectedCameraId)) sel.value = selectedCameraId;
  }
  $("camera-select").addEventListener("change", (e) => selectCamera(e.target.value || null));

  // -- camera strip: a small always-visible thumbnail row above the big
  // view, one tile per camera, so every feed stays visible at a glance
  // (surveillance-wall style) while the big view stays focused on one.
  // Click a thumbnail to focus it below. Rebuilds tiles only when the set
  // of cameras actually changes, so each tile's live MJPEG <img> never
  // gets torn down and reconnected on every snapshot -- in between, only
  // the status dot / selected outline update in place.
  function renderCameraStrip() {
    const strip = $("camera-strip");
    const currentIds = cameras.map((c) => c.camera_id);
    const existingIds = Array.from(strip.children)
      .filter((el) => el.dataset && el.dataset.cameraId)
      .map((el) => el.dataset.cameraId);
    const sameSet = currentIds.length === existingIds.length && currentIds.every((id, i) => id === existingIds[i]);

    if (!sameSet) {
      strip.innerHTML = "";
      for (const cam of cameras) {
        const tile = document.createElement("div");
        tile.className = "strip-tile";
        tile.dataset.cameraId = cam.camera_id;
        tile.title = `${cam.camera_id} — ${cam.name}`;
        tile.innerHTML = `
          <img src="/video_feed/${cam.camera_id}" alt="${cam.name}" />
          <span class="tile-dot"></span>
          <span class="tile-label">${cam.camera_id}</span>
        `;
        tile.addEventListener("click", () => selectCamera(cam.camera_id));
        strip.appendChild(tile);
      }
    }

    for (const cam of cameras) {
      const tile = strip.querySelector(`[data-camera-id="${cam.camera_id}"]`);
      if (!tile) continue;
      tile.classList.toggle("tile-selected", cam.camera_id === selectedCameraId);
      tile.classList.toggle("tile-flagged", !!dominantPriority(cam.priority_counts));
      tile.querySelector(".tile-dot").classList.toggle("online", cam.status === "LIVE");
    }
  }

  // -- subjects list (selected camera's active tracks) ------------------------
  function renderSubjects() {
    const host = $("subjects-list");
    const tracks = tracksByCamera[selectedCameraId] || [];
    if (!tracks.length) {
      host.innerHTML = '<div class="empty-note">No subjects currently in frame</div>';
      return;
    }
    host.innerHTML = "";
    for (const t of tracks) {
      const card = document.createElement("div");
      card.className = `subject-card p-${t.priority}`;
      card.innerHTML = `
        <div class="subject-id">Person #${t.track_id}</div>
        <div class="subject-row"><span>Confidence</span><span>${Math.round(t.confidence * 100)}%</span></div>
        <div class="subject-row"><span>Posture</span><span>${t.posture}</span></div>
        <div class="subject-row"><span>Movement</span><span>${t.movement}</span></div>
        <div class="subject-row"><span>Inactive</span><span>${Math.round(t.inactivity_sec)}s</span></div>
        <div class="subject-row"><span>Priority</span><span class="subject-priority">${t.priority} · ${Math.round(t.priority_score)}/100</span></div>
      `;
      host.appendChild(card);
    }
  }

  // -- cameras management table ------------------------------------------------
  function dominantPriority(counts) {
    if (!counts) return null;
    for (const label of PRIORITY_ORDER) {
      if ((counts[label] || 0) > 0) return label;
    }
    return null;
  }

  function renderCamerasTable() {
    const body = $("cameras-body");
    if (!cameras.length) {
      body.innerHTML = '<tr class="empty-row"><td colspan="8">No cameras configured</td></tr>';
      return;
    }
    body.innerHTML = "";
    for (const cam of cameras) {
      const tr = document.createElement("tr");
      tr.className = "selectable";
      if (cam.camera_id === selectedCameraId) tr.classList.add("row-selected");
      const dom = dominantPriority(cam.priority_counts);
      const priorityCell = dom
        ? `<span class="badge badge-${dom}">${cam.priority_summary}</span>`
        : '<span class="empty-note">—</span>';

      tr.innerHTML = `
        <td><strong>${cam.name}</strong><br><span style="color:var(--text-muted)">${cam.camera_id}</span></td>
        <td><span class="source-tag">${cam.source}</span></td>
        <td><span class="badge badge-${cam.status}">${cam.status}</span></td>
        <td>${cam.people_count}</td>
        <td>${priorityCell}</td>
        <td>${cam.fps}</td>
        <td>${cam.resolution}</td>
        <td></td>
      `;
      tr.addEventListener("click", (e) => {
        if (e.target.closest("button")) return;
        selectCamera(cam.camera_id);
      });
      const removeBtn = document.createElement("button");
      removeBtn.className = "btn-danger";
      removeBtn.type = "button";
      removeBtn.textContent = "Remove";
      removeBtn.addEventListener("click", () => removeCamera(cam.camera_id));
      tr.lastElementChild.appendChild(removeBtn);
      body.appendChild(tr);
    }
  }

  async function removeCamera(cameraId) {
    if (!confirm(`Remove camera ${cameraId}? This stops its feed.`)) return;
    try {
      await fetch(`/api/cameras/${encodeURIComponent(cameraId)}`, { method: "DELETE" });
    } catch (e) { /* next snapshot / poll will reflect actual state either way */ }
  }

  // -- add camera form -----------------------------------------------------------
  const addForm = $("add-camera-form");
  $("add-camera-btn").addEventListener("click", async () => {
    addForm.hidden = !addForm.hidden;
    if (!addForm.hidden) await rescanWebcams();
  });
  $("cam-cancel-btn").addEventListener("click", () => { addForm.hidden = true; });

  $("cam-source-type").addEventListener("change", (e) => {
    const isWebcam = e.target.value === "webcam";
    $("cam-ref-webcam-wrap").hidden = !isWebcam;
    $("cam-ref-file-wrap").hidden = isWebcam;
  });

  $("cam-rescan-btn").addEventListener("click", rescanWebcams);

  async function rescanWebcams() {
    const status = $("add-camera-status");
    status.textContent = "Scanning for webcams…";
    try {
      const res = await fetch("/api/cameras/probe");
      const data = await res.json();
      probedWebcamIndices = data.indices || [];
      const sel = $("cam-ref-webcam");
      sel.innerHTML = "";
      if (!probedWebcamIndices.length) {
        const opt = document.createElement("option");
        opt.value = "0";
        opt.textContent = "None found — try index 0";
        sel.appendChild(opt);
      } else {
        for (const idx of probedWebcamIndices) {
          const opt = document.createElement("option");
          opt.value = String(idx);
          opt.textContent = `Device ${idx}`;
          sel.appendChild(opt);
        }
      }
      status.textContent = probedWebcamIndices.length
        ? `Found ${probedWebcamIndices.length} webcam(s)`
        : "No webcams detected";
    } catch (e) {
      status.textContent = "Could not scan for webcams";
    }
  }

  addForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const status = $("add-camera-status");
    const sourceType = $("cam-source-type").value;
    const sourceRef = sourceType === "webcam" ? $("cam-ref-webcam").value : $("cam-ref-file").value.trim();

    if (!sourceRef) {
      status.textContent = sourceType === "webcam" ? "Rescan and pick a device first" : "Enter a file path";
      return;
    }

    status.textContent = "Adding…";
    try {
      const res = await fetch("/api/cameras", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: $("cam-name").value.trim(),
          source_type: sourceType,
          source_ref: sourceRef,
          source: $("cam-source").value,
        }),
      });
      const data = await res.json();
      if (!res.ok) {
        status.textContent = data.error || "Failed to add camera";
        return;
      }
      status.textContent = `Added ${data.camera_id}`;
      $("cam-name").value = "";
      $("cam-ref-file").value = "";
      addForm.hidden = true;
      selectCamera(data.camera_id);
    } catch (e) {
      status.textContent = "Failed to add camera";
    }
  });

  // -- live rescue event feed --------------------------------------------------
  function eventClass(eventType) {
    if (eventType.includes("ENTERED")) return "event-ENTERED";
    if (eventType.includes("EXITED")) return "event-EXITED";
    if (eventType.includes("PRIORITY")) return "event-PRIORITY";
    if (eventType.includes("COUNT")) return "event-COUNT";
    return "event-COUNT";
  }

  function eventDescription(ev) {
    const who = ev.track_id !== null && ev.track_id !== undefined ? `Person #${ev.track_id}` : "Camera";
    switch (ev.event_type) {
      case "PERSON_ENTERED":
        return `${who} entered frame`;
      case "PERSON_EXITED":
        return `${who} left frame`;
      case "PERSON_COUNT_CHANGED":
        return `Active count changed to ${ev.details && ev.details.count}`;
      case "PRIORITY_CHANGED": {
        const label = ev.details && ev.details.priority;
        const score = ev.details && ev.details.score;
        return `${who} priority → ${label} (${score}/100) — operator verification required`;
      }
      default:
        return `${who} ${ev.event_type}`;
    }
  }

  function addEventToFeed(ev) {
    const list = $("event-list");
    const firstRow = list.querySelector(".empty-row");
    if (firstRow) firstRow.remove();

    const li = document.createElement("li");
    li.innerHTML = `
      <div class="event-top">
        <span class="event-time">${(ev.timestamp || "").split("T")[1] || ev.timestamp || ""}</span>
        <span class="event-camera">${ev.camera_id}</span>
        <span class="event-type ${eventClass(ev.event_type)}">${ev.event_type.replace(/_/g, " ")}</span>
      </div>
      <div>${eventDescription(ev)}</div>
    `;
    list.appendChild(li);
    eventCount += 1;
    while (list.children.length > MAX_EVENTS_SHOWN) {
      list.removeChild(list.firstElementChild);
    }
  }

  function handleEvent(ev) {
    addEventToFeed(ev);
  }

  async function seedEventFeed() {
    try {
      const res = await fetch("/api/events?limit=40");
      const events = await res.json();
      // API returns newest-first; the feed list is CSS column-reverse
      // (visually newest-on-top), so DOM order must be oldest-first.
      for (const ev of events.slice().reverse()) addEventToFeed(ev);
    } catch (e) { /* live feed will still populate going forward */ }
  }

  // -- history / tracking records table -----------------------------------------
  function renderHistoryCameraFilter() {
    const sel = $("history-camera-filter");
    const prev = sel.value;
    const existingIds = new Set(Array.from(sel.options).map((o) => o.value));
    const wantedIds = new Set(cameras.map((c) => c.camera_id));
    if (existingIds.size - 1 === wantedIds.size && [...wantedIds].every((id) => existingIds.has(id))) return;

    sel.innerHTML = '<option value="">All cameras</option>';
    for (const cam of cameras) {
      const opt = document.createElement("option");
      opt.value = cam.camera_id;
      opt.textContent = `${cam.camera_id} — ${cam.name}`;
      sel.appendChild(opt);
    }
    sel.value = existingIds.has(prev) ? prev : "";
  }
  $("history-camera-filter").addEventListener("change", loadHistory);

  async function loadHistory() {
    const camFilter = $("history-camera-filter").value;
    const qs = camFilter ? `?camera_id=${encodeURIComponent(camFilter)}&limit=50` : "?limit=50";
    try {
      const res = await fetch(`/api/tracks${qs}`);
      const rows = await res.json();
      renderHistoryTable(rows);
    } catch (e) { /* leave existing table as-is */ }
  }

  function renderHistoryTable(rows) {
    const body = $("history-body");
    if (!rows.length) {
      body.innerHTML = '<tr class="empty-row"><td colspan="10">Waiting for detections…</td></tr>';
      return;
    }
    body.innerHTML = "";
    for (const r of rows) {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td>#${r.track_id}</td>
        <td>${r.camera_id}</td>
        <td>${Math.round((r.max_confidence || 0) * 100)}%</td>
        <td>${r.posture || "--"}</td>
        <td>${r.movement_state || "--"}</td>
        <td>${(r.first_seen || "").replace("T", " ")}</td>
        <td>${(r.last_seen || "").replace("T", " ")}</td>
        <td>${Math.round(r.duration_sec || 0)}s</td>
        <td><span class="badge badge-${r.max_priority}">${r.max_priority} · ${Math.round(r.max_priority_score || 0)}</span></td>
        <td><span class="badge badge-${r.status}">${r.status}</span></td>
      `;
      body.appendChild(tr);
    }
  }

  // -- analytics charts -----------------------------------------------------------
  async function refreshAnalytics() {
    await Promise.all([
      refreshPeopleOverTime(),
      refreshPriorityDistribution(),
      refreshCameraActivity(),
      refreshConfidence(),
    ]);
  }

  async function refreshPeopleOverTime() {
    try {
      const res = await fetch("/api/analytics/people_over_time");
      const data = await res.json();
      Charts.renderLineChart($("chart-people-time"), data.labels, data.counts, {
        emptyMessage: "No activity in this window yet",
      });
    } catch (e) { /* skip this tick */ }
  }

  async function refreshPriorityDistribution() {
    try {
      const res = await fetch("/api/analytics/priority");
      const counts = await res.json();
      const segments = PRIORITY_ORDER.map((label) => ({
        label: label.charAt(0) + label.slice(1).toLowerCase(),
        value: counts[label] || 0,
        color: PRIORITY_COLORS[label],
      }));
      Charts.renderDonutChart($("chart-priority"), segments, { centerLabel: "In frame" });
    } catch (e) { /* skip this tick */ }
  }

  async function refreshCameraActivity() {
    try {
      const res = await fetch("/api/analytics/camera_activity");
      const data = await res.json();
      Charts.renderBarChart(
        $("chart-activity"),
        data.map((d) => d.name),
        data.map((d) => d.people_count),
        { horizontal: true, emptyMessage: "No cameras configured" }
      );
    } catch (e) { /* skip this tick */ }
  }

  async function refreshConfidence() {
    try {
      const res = await fetch("/api/analytics/confidence");
      const data = await res.json();
      Charts.renderBarChart($("chart-confidence"), data.labels, data.counts, {
        emptyMessage: "No detections recorded yet",
        colorFor: () => "#4f8ff7",
      });
    } catch (e) { /* skip this tick */ }
  }

  // -- sidebar nav (scroll-to-section, active-link highlighting) ------------------
  const sideLinks = Array.from(document.querySelectorAll(".side-link"));
  for (const link of sideLinks) {
    link.addEventListener("click", () => {
      for (const l of sideLinks) l.classList.remove("active");
      link.classList.add("active");
    });
  }

  // -- boot -----------------------------------------------------------------------
  connectWs();
  seedEventFeed();
  refreshAnalytics();
  loadHistory();
  setInterval(refreshAnalytics, ANALYTICS_POLL_MS);
  setInterval(loadHistory, ANALYTICS_POLL_MS);
})();
