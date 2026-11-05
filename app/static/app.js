(function () {
  "use strict";

  const $ = (id) => document.getElementById(id);

  const form = $("upload-form");
  const dropzone = $("dropzone");
  const fileInput = $("file-input");
  const fileName = $("file-name");
  const layoutInput = $("layout-input");
  const maxFrames = $("max-frames");
  const submitButton = $("submit-button");
  const status = $("status");
  const results = $("results");
  const healthPill = $("health-pill");

  const stateClass = {
    OCCUPIED: "state state-occupied",
    EMPTY: "state state-empty",
    UNCERTAIN: "state state-uncertain",
  };

  function setStatus(message, kind) {
    status.textContent = message || "";
    status.className = "status" + (kind ? " " + kind : "");
  }

  function formatNumber(value, digits) {
    if (value === null || value === undefined || Number.isNaN(Number(value))) {
      return "-";
    }
    return Number(value).toLocaleString(undefined, {
      maximumFractionDigits: digits === undefined ? 2 : digits,
    });
  }

  function percent(value) {
    if (value === null || value === undefined || Number.isNaN(Number(value))) {
      return "-";
    }
    return formatNumber(Number(value) * 100, 1) + "%";
  }

  function setText(id, value) {
    $(id).textContent = value === null || value === undefined ? "-" : String(value);
  }

  function clear(node) {
    while (node.firstChild) {
      node.removeChild(node.firstChild);
    }
  }

  function appendDefinition(list, label, value) {
    const dt = document.createElement("dt");
    dt.textContent = label;
    const dd = document.createElement("dd");
    dd.textContent = value === null || value === undefined || value === "" ? "-" : String(value);
    list.appendChild(dt);
    list.appendChild(dd);
  }

  function emptyRow(body, columns, message) {
    const row = document.createElement("tr");
    const cell = document.createElement("td");
    cell.colSpan = columns;
    cell.className = "empty-row";
    cell.textContent = message;
    row.appendChild(cell);
    body.appendChild(row);
  }

  function renderSpaces(spaces) {
    const body = $("spaces-body");
    clear(body);
    $("spaces-count").textContent = String(spaces.length);
    if (!spaces.length) {
      emptyRow(body, 5, "No parking spaces detected.");
      return;
    }
    spaces.forEach((space) => {
      const row = document.createElement("tr");

      const idCell = document.createElement("td");
      idCell.textContent = space.space_id || "-";
      row.appendChild(idCell);

      const occupancy = space.occupancy || {};
      const stateCell = document.createElement("td");
      const badge = document.createElement("span");
      const stateName = occupancy.state || "UNKNOWN";
      badge.className = stateClass[stateName] || "state";
      badge.textContent = stateName;
      stateCell.appendChild(badge);
      row.appendChild(stateCell);

      const confidenceCell = document.createElement("td");
      confidenceCell.textContent = percent(occupancy.confidence);
      row.appendChild(confidenceCell);

      const coverageCell = document.createElement("td");
      coverageCell.textContent = percent(occupancy.space_coverage);
      row.appendChild(coverageCell);

      const methodCell = document.createElement("td");
      methodCell.textContent = space.method || "-";
      row.appendChild(methodCell);

      body.appendChild(row);
    });
  }

  function renderEvents(events) {
    const body = $("events-body");
    clear(body);
    $("events-count").textContent = String(events.length);
    if (!events.length) {
      emptyRow(body, 5, "No occupancy transitions recorded.");
      return;
    }
    events.forEach((event) => {
      const row = document.createElement("tr");

      const timeCell = document.createElement("td");
      timeCell.textContent = formatNumber(event.timestamp, 2) + " s";
      row.appendChild(timeCell);

      const typeCell = document.createElement("td");
      typeCell.textContent = String(event.event_type || "-").replace(/_/g, " ");
      row.appendChild(typeCell);

      const spaceCell = document.createElement("td");
      spaceCell.textContent = event.space_id || "-";
      row.appendChild(spaceCell);

      const transitionCell = document.createElement("td");
      transitionCell.textContent = [event.previous_state, event.new_state]
        .filter(Boolean)
        .join(" -> ") || "-";
      row.appendChild(transitionCell);

      const confidenceCell = document.createElement("td");
      confidenceCell.textContent = percent(event.confidence);
      row.appendChild(confidenceCell);

      body.appendChild(row);
    });
  }

  function renderMedia(result) {
    const holder = $("media-holder");
    clear(holder);
    const annotated = result.urls && result.urls.annotated;

    if (annotated && result.kind === "video") {
      const video = document.createElement("video");
      video.controls = true;
      video.autoplay = true;
      video.loop = true;
      video.muted = true;
      video.playsInline = true;
      video.src = annotated;
      holder.appendChild(video);
    } else if (annotated) {
      const image = document.createElement("img");
      image.src = annotated;
      image.alt = "Annotated parking-lot analysis";
      holder.appendChild(image);
    } else {
      const message = document.createElement("p");
      message.className = "muted";
      message.textContent = "No annotated output was produced.";
      holder.appendChild(message);
    }

    const details = [];
    if (result.width && result.height) {
      details.push(result.width + " x " + result.height);
    }
    if (result.fps) {
      details.push(formatNumber(result.fps, 2) + " fps");
    }
    if (result.frames_processed !== undefined && result.frames_processed !== null) {
      details.push(formatNumber(result.frames_processed, 0) + " frames processed");
    }
    $("media-meta").textContent = details.join(" | ") || "-";

    const openLink = $("open-output");
    if (annotated) {
      openLink.href = annotated;
      openLink.hidden = false;
    } else {
      openLink.hidden = true;
    }

    const jsonLink = $("download-json");
    const jsonUrl = result.urls && result.urls.result_json;
    if (jsonUrl) {
      jsonLink.href = jsonUrl;
      jsonLink.hidden = false;
    } else {
      jsonLink.hidden = true;
    }
  }

  function renderResult(result) {
    setText("run-id", result.run_id);
    setText("kind-badge", result.kind ? result.kind.toUpperCase() : "-");

    const scene = result.scene || {};
    const sceneBadge = $("scene-badge");
    sceneBadge.textContent = scene.status || "UNKNOWN";
    sceneBadge.className = "badge";
    if (scene.status === "NON_PARKING" || scene.status === "INVALID") {
      sceneBadge.classList.add("bad");
    } else if (scene.status === "INSUFFICIENT" || scene.status === "POSSIBLE") {
      sceneBadge.classList.add("warn");
    }

    $("report-notice").hidden = Boolean(result.occupancy_reportable);

    const stats = result.statistics || {};
    setText("stat-total", stats.total_spaces || 0);
    setText("stat-occupied", stats.occupied_spaces || 0);
    setText("stat-empty", stats.empty_spaces || 0);
    setText("stat-uncertain", stats.uncertain_spaces || 0);
    setText(
      "stat-percentage",
      stats.occupancy_percentage === null || stats.occupancy_percentage === undefined
        ? "-"
        : formatNumber(stats.occupancy_percentage, 1) + "%"
    );
    setText("stat-vehicles", stats.detected_vehicles || 0);
    setText("stat-entries", stats.entries || 0);
    setText("stat-exits", stats.exits || 0);

    const sceneDetails = $("scene-details");
    clear(sceneDetails);
    appendDefinition(sceneDetails, "Status", scene.status);
    appendDefinition(sceneDetails, "Confidence", percent(scene.confidence));
    appendDefinition(sceneDetails, "Reason", scene.reason);
    appendDefinition(sceneDetails, "Parking probability", percent(scene.parking_probability));
    appendDefinition(sceneDetails, "Layout spaces", scene.layout_spaces);
    appendDefinition(sceneDetails, "Layout confidence", percent(scene.layout_confidence));

    const timingDetails = $("timing-details");
    clear(timingDetails);
    const timings = result.timings_ms || {};
    const timingKeys = Object.keys(timings);
    if (!timingKeys.length) {
      appendDefinition(timingDetails, "Processing", "Not reported");
    } else {
      timingKeys.forEach((key) => {
        appendDefinition(timingDetails, key.replace(/_/g, " "), formatNumber(timings[key], 1) + " ms");
      });
    }

    renderMedia(result);
    renderSpaces(result.spaces || []);
    renderEvents(result.events || []);

    $("raw-json").textContent = JSON.stringify(result, null, 2);
    results.hidden = false;
    results.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  async function handleSubmit(event) {
    event.preventDefault();

    const file = fileInput.files && fileInput.files[0];
    if (!file) {
      setStatus("Choose an image or video first.", "error");
      return;
    }

    const data = new FormData();
    data.append("file", file);
    if (layoutInput.files && layoutInput.files[0]) {
      data.append("layout", layoutInput.files[0]);
    }
    if (maxFrames.value.trim()) {
      data.append("max_frames", maxFrames.value.trim());
    }

    submitButton.disabled = true;
    setStatus("Processing " + file.name + "...", "busy");

    try {
      const response = await fetch("/api/process", { method: "POST", body: data });
      let payload;
      try {
        payload = await response.json();
      } catch (parseError) {
        throw new Error("Server returned an unreadable response (HTTP " + response.status + ").");
      }
      if (!response.ok || !payload.ok) {
        throw new Error(payload.error || "Processing failed (HTTP " + response.status + ").");
      }
      renderResult(payload.result);
      setStatus("Done in " + formatNumber(payload.result.timings_ms && totalTiming(payload.result.timings_ms), 0) + " ms.", "");
    } catch (error) {
      setStatus(error.message || "Processing failed.", "error");
    } finally {
      submitButton.disabled = false;
    }
  }

  function totalTiming(timings) {
    return Object.keys(timings || {}).reduce((sum, key) => sum + Number(timings[key] || 0), 0);
  }

  function updateFileName() {
    const file = fileInput.files && fileInput.files[0];
    if (!file) {
      fileName.textContent = "No file selected";
      return;
    }
    const sizeMb = file.size / (1024 * 1024);
    fileName.textContent = file.name + " (" + sizeMb.toFixed(2) + " MB)";
  }

  function bindDropzone() {
    fileInput.addEventListener("change", updateFileName);

    ["dragenter", "dragover"].forEach((name) => {
      dropzone.addEventListener(name, (event) => {
        event.preventDefault();
        dropzone.classList.add("dragover");
      });
    });

    ["dragleave", "drop"].forEach((name) => {
      dropzone.addEventListener(name, (event) => {
        event.preventDefault();
        dropzone.classList.remove("dragover");
      });
    });

    dropzone.addEventListener("drop", (event) => {
      const files = event.dataTransfer && event.dataTransfer.files;
      if (!files || !files.length) {
        return;
      }
      fileInput.files = files;
      updateFileName();
    });
  }

  async function checkHealth() {
    try {
      const response = await fetch("/api/health");
      const payload = await response.json();
      if (payload.ok) {
        healthPill.textContent = "API online | " + (payload.device || "cpu");
        healthPill.classList.add("ok");
      } else {
        throw new Error("unhealthy");
      }
    } catch (error) {
      healthPill.textContent = "API unavailable";
      healthPill.classList.add("bad");
    }
  }

  form.addEventListener("submit", handleSubmit);
  bindDropzone();
  checkHealth();
})();
