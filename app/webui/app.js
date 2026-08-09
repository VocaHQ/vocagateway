/* vocaphone WebUI glue: token auth for HTMX, tab styling, mic recorder, toasts. */
(() => {
  "use strict";

  const TOKEN_KEY = "vocaphone.token";
  const THEME_KEY = "vocaphone.theme";
  const EXPOSURE_DISMISS_KEY = "vocaphone.exposure-dismiss-until";
  const EXPOSURE_DISMISS_MS = 24 * 60 * 60 * 1000;
  const overlay = document.getElementById("token-overlay");
  const tokenInput = document.getElementById("token-input");
  const tokenError = document.getElementById("token-error");
  const toast = document.getElementById("toast");
  const themeToggle = document.getElementById("theme-toggle");

  const getToken = () => localStorage.getItem(TOKEN_KEY) || "";

  // ------------------------------------------------ network exposure banner

  function exposureDismissedUntil() {
    const raw = Number(localStorage.getItem(EXPOSURE_DISMISS_KEY) || 0);
    return Number.isFinite(raw) ? raw : 0;
  }

  function applyExposureBanner(root) {
    let banner = document.getElementById("exposure-banner");
    if (root && root.id === "exposure-banner") banner = root;
    if (!banner || banner.dataset.empty === "1") return;
    const until = exposureDismissedUntil();
    const hidden = until > Date.now();
    banner.hidden = hidden;
    banner.setAttribute("aria-hidden", hidden ? "true" : "false");
  }

  function dismissExposureBanner() {
    localStorage.setItem(EXPOSURE_DISMISS_KEY, String(Date.now() + EXPOSURE_DISMISS_MS));
    applyExposureBanner();
  }

  document.body.addEventListener("click", (event) => {
    const btn = event.target.closest && event.target.closest("[data-dismiss-exposure]");
    if (!btn) return;
    event.preventDefault();
    dismissExposureBanner();
  });

  document.body.addEventListener("htmx:afterSwap", (event) => {
    if (event.detail && event.detail.target && event.detail.target.id === "exposure-banner") {
      applyExposureBanner(event.detail.target);
    }
  });

  // --------------------------------------------------------------- theme

  function systemPrefersDark() {
    return window.matchMedia("(prefers-color-scheme: dark)").matches;
  }

  function readThemePreference() {
    const pref = localStorage.getItem(THEME_KEY);
    if (pref === "light" || pref === "dark" || pref === "system") return pref;
    return "system";
  }

  function resolveTheme(preference) {
    if (preference === "light" || preference === "dark") return preference;
    return systemPrefersDark() ? "dark" : "light";
  }

  function themeLabel(preference, resolved) {
    if (preference === "system") {
      return `Theme: system (${resolved}). Click for light.`;
    }
    if (preference === "light") return "Theme: light. Click for dark.";
    return "Theme: dark. Click for system.";
  }

  function setFavicon(resolved) {
    const href = resolved === "dark" ? "/assets/favicon-dark.svg" : "/assets/favicon-light.svg";
    const favicon = document.getElementById("favicon");
    const apple = document.getElementById("apple-touch-icon");
    if (favicon) favicon.setAttribute("href", href);
    if (apple) apple.setAttribute("href", href);
  }

  function applyTheme(preference) {
    const resolved = resolveTheme(preference);
    document.documentElement.setAttribute("data-theme", resolved);
    document.documentElement.setAttribute("data-theme-preference", preference);
    const meta = document.getElementById("meta-theme-color");
    if (meta) {
      meta.setAttribute("content", resolved === "dark" ? "#141614" : "#f7f6f3");
    }
    setFavicon(resolved);
    if (themeToggle) {
      themeToggle.dataset.preference = preference;
      const label = themeLabel(preference, resolved);
      themeToggle.setAttribute("aria-label", label);
      themeToggle.setAttribute("title", label);
    }
  }

  function cycleThemePreference(current) {
    // system → light → dark → system. Default is system so first-time
    // operators match the OS; clicks then pin an explicit choice.
    if (current === "system") return "light";
    if (current === "light") return "dark";
    return "system";
  }

  function initTheme() {
    applyTheme(readThemePreference());
    if (themeToggle) {
      themeToggle.addEventListener("click", () => {
        const next = cycleThemePreference(readThemePreference());
        localStorage.setItem(THEME_KEY, next);
        applyTheme(next);
      });
    }
    const media = window.matchMedia("(prefers-color-scheme: dark)");
    const onSystemChange = () => {
      if (readThemePreference() === "system") applyTheme("system");
    };
    if (typeof media.addEventListener === "function") {
      media.addEventListener("change", onSystemChange);
    } else if (typeof media.addListener === "function") {
      media.addListener(onSystemChange);
    }
  }

  function showToast(message, isError = true) {
    toast.textContent = message;
    toast.classList.toggle("error", isError);
    toast.classList.remove("hidden");
    clearTimeout(showToast.timer);
    showToast.timer = setTimeout(() => toast.classList.add("hidden"), 5000);
  }

  function showOverlay(message = "") {
    tokenError.textContent = message;
    tokenError.classList.toggle("hidden", !message);
    overlay.classList.remove("hidden");
    overlay.setAttribute("aria-hidden", "false");
    tokenInput.focus();
  }

  function hideOverlay() {
    overlay.classList.add("hidden");
    overlay.setAttribute("aria-hidden", "true");
    tokenInput.value = "";
  }

  // ------------------------------------------------------------------ token

  document.getElementById("token-save").addEventListener("click", () => {
    const token = tokenInput.value.trim();
    if (token.length < 32) {
      showOverlay("Token must be at least 32 characters.");
      return;
    }
    localStorage.setItem(TOKEN_KEY, token);
    hideOverlay();
    htmx.ajax("GET", "/ui/partials/overview", { target: "#panel", swap: "innerHTML" });
    htmx.ajax("GET", "/ui/partials/engine-pill", { target: "#engine-pill", swap: "outerHTML" });
    htmx.ajax("GET", "/ui/partials/exposure-banner", {
      target: "#exposure-banner",
      swap: "outerHTML",
    });
  });

  tokenInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter") document.getElementById("token-save").click();
  });

  document.body.addEventListener("click", (event) => {
    if (event.target.id === "forget-token") {
      localStorage.removeItem(TOKEN_KEY);
      showOverlay("Token cleared from this browser.");
    }
  });

  // ------------------------------------------------------- htmx integration

  document.body.addEventListener("htmx:configRequest", (event) => {
    event.detail.headers["Authorization"] = `Bearer ${getToken()}`;
  });

  document.body.addEventListener("htmx:responseError", (event) => {
    if (event.detail.xhr.status === 401) {
      localStorage.removeItem(TOKEN_KEY);
      showOverlay("Token rejected. Paste the current gateway token.");
      return;
    }
    let message = `Request failed (${event.detail.xhr.status}).`;
    try {
      const payload = JSON.parse(event.detail.xhr.responseText);
      if (payload.error && payload.error.message) message = payload.error.message;
    } catch (_) { /* keep default message */ }
    showToast(message);
  });

  // -------------------------------------------------------------------- tabs

  function activateTab(tab, updateLocation = true) {
    if (!tab) return;
    document.querySelectorAll(".tab").forEach((other) => {
      const active = other === tab;
      other.classList.toggle("active", active);
      other.setAttribute("aria-selected", String(active));
      other.tabIndex = active ? 0 : -1;
    });
    const panel = document.getElementById("panel");
    if (panel && tab.id) panel.setAttribute("aria-labelledby", tab.id);
    if (updateLocation) history.replaceState(null, "", `#${tab.dataset.tab}`);
  }

  function openTabByName(name) {
    // #test is kept as an alias for the renamed Pair & test tab.
    const key = name === "test" ? "pair" : name;
    const tab = document.querySelector(`.tab[data-tab="${key}"]`);
    if (!tab) return;
    activateTab(tab);
    const href = tab.getAttribute("hx-get");
    if (href) {
      htmx.ajax("GET", href, { target: "#panel", swap: "innerHTML" });
    }
  }

  document.querySelectorAll(".tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      activateTab(tab);
    });
    tab.addEventListener("keydown", (event) => {
      if (!["ArrowLeft", "ArrowRight"].includes(event.key)) return;
      const tabs = [...document.querySelectorAll(".tab")];
      const offset = event.key === "ArrowRight" ? 1 : -1;
      const next = tabs[(tabs.indexOf(tab) + offset + tabs.length) % tabs.length];
      next.focus();
      next.click();
    });
  });

  document.body.addEventListener("click", (event) => {
    const trigger = event.target.closest("[data-open-tab]");
    if (!trigger) return;
    // Engine pill refreshes via hx-get; without this the tab would open and the
    // pill request would also fire and fight for the swap target.
    event.preventDefault();
    event.stopPropagation();
    openTabByName(trigger.getAttribute("data-open-tab"));
  }, true);

  document.body.addEventListener("htmx:beforeRequest", (event) => {
    if (event.detail.target && event.detail.target.id === "panel") {
      event.detail.target.setAttribute("aria-busy", "true");
    }
  });

  document.body.addEventListener("htmx:afterRequest", (event) => {
    if (event.detail.target && event.detail.target.id === "panel") {
      event.detail.target.setAttribute("aria-busy", "false");
    }
  });

  document.body.addEventListener("htmx:afterSwap", (event) => {
    scheduleModelPoll();
    // Models tab shell (or list refresh) may reintroduce filter controls.
    if (
      event.detail &&
      event.detail.target &&
      (event.detail.target.id === "panel" || event.detail.target.id === "models-list")
    ) {
      initModelFilters();
    }
  });

  // ------------------------------------------------------------ model status

  let modelPollTimer = null;

  function checkedFilterValues(name) {
    return [...document.querySelectorAll(`#models-filter-form input[name="${name}"]:checked`)]
      .map((el) => el.value)
      .filter(Boolean);
  }

  function currentModelFilters() {
    const form = document.getElementById("models-filter-form");
    if (!form) {
      return {
        family: [],
        language: [],
        engine: [],
        max_size: "",
        installed_only: "",
        recommended_only: "",
      };
    }
    const installed = form.querySelector('input[name="installed_only"]');
    const recommended = form.querySelector('input[name="recommended_only"]');
    const maxSize = form.querySelector('select[name="max_size"]');
    return {
      family: checkedFilterValues("family"),
      language: checkedFilterValues("language"),
      engine: checkedFilterValues("engine"),
      max_size: maxSize ? maxSize.value : "",
      installed_only: installed && installed.checked ? "true" : "",
      recommended_only: recommended && recommended.checked ? "true" : "",
    };
  }

  const FILTER_RAIL_KEY = "vocaphone.models-filter-collapsed";

  function isFilterRailCollapsed() {
    try {
      return localStorage.getItem(FILTER_RAIL_KEY) === "1";
    } catch (_) {
      return false;
    }
  }

  function setFilterRailCollapsed(collapsed) {
    const layout = document.getElementById("models-layout");
    const aside = document.getElementById("models-filter");
    const toggle = document.getElementById("filter-rail-toggle");
    const sideCollapse = document.getElementById("filter-side-collapse");
    if (layout) layout.classList.toggle("is-filter-collapsed", collapsed);
    if (toggle) {
      toggle.setAttribute("aria-expanded", collapsed ? "false" : "true");
      toggle.setAttribute("title", collapsed ? "Show filters" : "Hide filters");
      const label = toggle.querySelector(".filter-trigger-label");
      if (label) label.textContent = collapsed ? "Filters" : "Filters";
    }
    if (sideCollapse) {
      sideCollapse.setAttribute("aria-expanded", collapsed ? "false" : "true");
      sideCollapse.setAttribute("aria-label", collapsed ? "Expand filters" : "Collapse filters");
      sideCollapse.setAttribute("title", collapsed ? "Expand filters" : "Collapse filters");
    }
    if (aside) aside.setAttribute("aria-hidden", collapsed ? "true" : "false");
    try {
      localStorage.setItem(FILTER_RAIL_KEY, collapsed ? "1" : "0");
    } catch (_) { /* ignore */ }
  }

  function updateFilterChrome() {
    const root = document.getElementById("models-filter");
    const toggle = document.getElementById("filter-rail-toggle");
    const badge = document.getElementById("filter-active-count");
    if (!badge) return;
    const filters = currentModelFilters();
    let count = 0;
    if (filters.family.length) count += 1;
    if (filters.language.length) count += 1;
    if (filters.engine.length) count += 1;
    if (filters.max_size) count += 1;
    if (filters.installed_only) count += 1;
    if (filters.recommended_only) count += 1;
    if (root) root.classList.toggle("has-active", count > 0);
    if (toggle) toggle.classList.toggle("has-active", count > 0);
    badge.textContent = count ? String(count) : "";
    badge.classList.toggle("hidden", count === 0);
  }

  function familyTiles() {
    return [...document.querySelectorAll("#models-list details.family-tile")];
  }

  function setExpandToggleState(toggle, expanded) {
    if (!toggle) return;
    toggle.dataset.expanded = expanded ? "true" : "false";
    toggle.setAttribute("aria-expanded", expanded ? "true" : "false");
    const label = expanded ? "Collapse all families" : "Expand all families";
    toggle.setAttribute("aria-label", label);
    toggle.setAttribute("title", label);
  }

  function setFamiliesExpanded(expanded) {
    familyTiles().forEach((tile) => {
      tile.open = expanded;
    });
    setExpandToggleState(document.getElementById("families-expand-toggle"), expanded);
  }

  function syncFamiliesExpandToggle() {
    const toggle = document.getElementById("families-expand-toggle");
    const tiles = familyTiles();
    if (!toggle || !tiles.length) return;
    setExpandToggleState(toggle, tiles.every((tile) => tile.open));
  }

  function initModelFilters() {
    const form = document.getElementById("models-filter-form");
    const clear = document.getElementById("filter-clear");
    if (clear && !clear.dataset.bound) {
      clear.dataset.bound = "1";
      clear.addEventListener("click", () => {
        if (form) {
          form.querySelectorAll('input[type="checkbox"]').forEach((el) => {
            el.checked = false;
          });
          form.querySelectorAll("select").forEach((el) => {
            el.value = "";
          });
          form.querySelectorAll(".filter-search").forEach((el) => {
            el.value = "";
            const listId = el.getAttribute("data-filter-search");
            const list = listId ? document.getElementById(listId) : null;
            if (list) {
              list.querySelectorAll(".filter-check").forEach((row) => {
                row.hidden = false;
              });
            }
          });
        }
        updateFilterChrome();
        htmx.ajax("GET", "/ui/partials/models-list", {
          target: "#models-list",
          swap: "innerHTML",
          values: currentModelFilters(),
        });
      });
    }
    if (form && !form.dataset.filterBound) {
      form.dataset.filterBound = "1";
      form.addEventListener("change", updateFilterChrome);
    }
    // Client-side search over long checkbox lists (languages); no server round-trip.
    document.querySelectorAll("[data-filter-search]").forEach((search) => {
      if (search.dataset.bound) return;
      search.dataset.bound = "1";
      search.addEventListener("input", () => {
        const q = search.value.trim().toLowerCase();
        const list = document.getElementById(search.getAttribute("data-filter-search") || "");
        if (!list) return;
        list.querySelectorAll(".filter-check").forEach((row) => {
          const text = (row.textContent || "").toLowerCase();
          row.hidden = Boolean(q) && !text.includes(q);
        });
      });
    });
    const railToggle = document.getElementById("filter-rail-toggle");
    if (railToggle && !railToggle.dataset.bound) {
      railToggle.dataset.bound = "1";
      railToggle.addEventListener("click", () => {
        const layout = document.getElementById("models-layout");
        if (!layout) return;
        setFilterRailCollapsed(!layout.classList.contains("is-filter-collapsed"));
      });
    }
    const sideCollapse = document.getElementById("filter-side-collapse");
    if (sideCollapse && !sideCollapse.dataset.bound) {
      sideCollapse.dataset.bound = "1";
      sideCollapse.addEventListener("click", () => setFilterRailCollapsed(true));
    }
    const expandToggle = document.getElementById("families-expand-toggle");
    if (expandToggle && !expandToggle.dataset.bound) {
      expandToggle.dataset.bound = "1";
      expandToggle.addEventListener("click", () => {
        const expand = expandToggle.dataset.expanded !== "true";
        setFamiliesExpanded(expand);
      });
    }
    // List refreshes return collapsed tiles; keep the expand icon/tooltip in sync.
    const list = document.getElementById("models-list");
    if (list && !list.dataset.expandListen) {
      list.dataset.expandListen = "1";
      list.addEventListener("toggle", (event) => {
        if (event.target && event.target.matches && event.target.matches("details.family-tile")) {
          syncFamiliesExpandToggle();
        }
      }, true);
    }
    // Restore rail open/collapsed; default open on wide layouts.
    if (document.getElementById("models-layout")) {
      setFilterRailCollapsed(isFilterRailCollapsed());
    }
    updateFilterChrome();
    syncFamiliesExpandToggle();
  }

  function formatBytes(size) {
    if (size >= 1_000_000_000) return `${(size / 1_000_000_000).toFixed(1)} GB`;
    if (size >= 1_000_000) return `${Math.round(size / 1_000_000)} MB`;
    if (size >= 1_000) return `${Math.round(size / 1_000)} KB`;
    return `${size} B`;
  }

  function scheduleModelPoll(delay = 1500) {
    clearTimeout(modelPollTimer);
    modelPollTimer = null;
    const downloading = document.querySelector('#models-list [data-state="downloading"]');
    if (!downloading || document.visibilityState !== "visible") return;
    modelPollTimer = setTimeout(pollModelProgress, delay);
  }

  async function pollModelProgress() {
    const cards = [...document.querySelectorAll('#models-list [data-state="downloading"]')];
    if (!cards.length || document.visibilityState !== "visible") return;
    try {
      const response = await fetch("/v1/admin/models", {
        headers: { Authorization: `Bearer ${getToken()}` },
        cache: "no-store",
      });
      if (response.status === 401) {
        localStorage.removeItem(TOKEN_KEY);
        showOverlay("Session expired. Paste the current token.");
        return;
      }
      if (!response.ok) throw new Error(`Model status failed (${response.status}).`);
      const entries = await response.json();
      const byId = new Map(entries.map((entry) => [entry.id, entry]));
      let needsRefresh = false;
      cards.forEach((card) => {
        const entry = byId.get(card.dataset.modelId);
        if (!entry || entry.state !== "downloading") {
          needsRefresh = true;
          return;
        }
        const percent = Math.round((entry.progress || 0) * 100);
        const progress = card.querySelector(".progress");
        const bar = card.querySelector(".bar");
        const copy = card.querySelector(".progress-copy");
        if (progress) progress.setAttribute("aria-valuenow", String(percent));
        if (bar) bar.style.width = `${percent}%`;
        if (copy) {
          copy.textContent = `${percent}% · ${formatBytes(entry.downloaded_bytes || 0)} / ` +
            formatBytes(entry.total_bytes || 0);
        }
      });
      if (needsRefresh && document.getElementById("models-list")) {
        htmx.ajax("GET", "/ui/partials/models-list", {
          target: "#models-list",
          swap: "innerHTML",
          values: currentModelFilters(),
        });
        return;
      }
      scheduleModelPoll();
    } catch (_) {
      scheduleModelPoll(4000);
    }
  }

  document.addEventListener("visibilitychange", () => scheduleModelPoll());

  // ---------------------------------------------------------------- recorder

  let recorder = null;
  let chunks = [];
  let recordingTimer = null;
  let recordingStartedAt = 0;

  function stopRecordingTimer() {
    clearInterval(recordingTimer);
    recordingTimer = null;
  }

  function updateRecordingTimer(timer) {
    const elapsedSeconds = Math.floor((Date.now() - recordingStartedAt) / 1000);
    const minutes = Math.floor(elapsedSeconds / 60);
    timer.textContent = `${minutes}:${String(elapsedSeconds % 60).padStart(2, "0")}`;
  }

  function pickMimeType() {
    const candidates = ["audio/mp4", "audio/webm;codecs=opus", "audio/webm", "audio/ogg"];
    return candidates.find((type) => window.MediaRecorder && MediaRecorder.isTypeSupported(type));
  }

  document.body.addEventListener("click", async (event) => {
    if (event.target.id !== "record-toggle") return;
    const button = event.target;
    const status = document.getElementById("record-status");
    const result = document.getElementById("test-result");
    const errorBox = document.getElementById("test-error");
    const timer = document.getElementById("record-timer");
    const controls = document.getElementById("recorder-controls");
    const maximumSeconds = Number(controls.dataset.maximumSeconds) || 120;

    if (recorder && recorder.state === "recording") {
      recorder.stop();
      return;
    }

    const mimeType = pickMimeType();
    if (!mimeType) {
      errorBox.textContent = "This browser cannot record audio (no MediaRecorder).";
      errorBox.classList.remove("hidden");
      return;
    }

    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch (_) {
      errorBox.textContent = "Microphone permission denied.";
      errorBox.classList.remove("hidden");
      return;
    }

    chunks = [];
    recorder = new MediaRecorder(stream, { mimeType });
    recorder.ondataavailable = (chunk) => chunks.push(chunk.data);
    recorder.onstop = async () => {
      stopRecordingTimer();
      stream.getTracks().forEach((track) => track.stop());
      button.textContent = "Start recording";
      button.classList.remove("recording");
      button.disabled = true;
      timer.classList.add("hidden");
      status.textContent = "Transcribing...";
      const blob = new Blob(chunks, { type: mimeType.split(";")[0] });
      try {
        const language = document.getElementById("test-language").value;
        const runs = Number(document.getElementById("test-runs").value) || 1;
        const payloads = [];
        for (let run = 0; run < runs; run += 1) {
          status.textContent = runs > 1 ? `Benchmarking... run ${run + 1} of ${runs}` : "Transcribing...";
          const response = await fetch(`/v1/admin/test-transcription?language=${language}`, {
            method: "POST",
            headers: {
              Authorization: `Bearer ${getToken()}`,
              "Content-Type": blob.type,
            },
            body: blob,
          });
          const payload = await response.json();
          if (!response.ok) throw new Error(payload.error?.message || "Transcription failed.");
          payloads.push(payload);
        }
        const payload = payloads[payloads.length - 1];
        const measuredPayloads = payloads.length > 1 ? payloads.slice(1) : payloads;
        const average = (field) => measuredPayloads.reduce(
          (sum, item) => sum + (item[field] || 0), 0,
        ) / measuredPayloads.length;
        const formatMs = (value) => value >= 1000 ? `${(value / 1000).toFixed(2)}s` : `${Math.round(value)}ms`;
        document.getElementById("test-transcript").textContent = payload.transcript;
        document.getElementById("test-meta").textContent =
          runs > 1
            ? `${payload.engine} · warm average of runs 2-${runs}; model load is run 1`
            : `${payload.engine} · 1-run result`;
        document.getElementById("benchmark-total").textContent = formatMs(average("duration_ms"));
        document.getElementById("benchmark-normalize").textContent = formatMs(average("normalization_ms"));
        document.getElementById("benchmark-load").textContent = formatMs(payloads[0].model_load_ms);
        document.getElementById("benchmark-inference").textContent = formatMs(average("inference_ms"));
        document.getElementById("benchmark-rtf").textContent =
          payload.real_time_factor == null ? "—" : `${average("real_time_factor").toFixed(2)}×`;
        document.getElementById("benchmark-memory").textContent =
          payload.peak_memory_mb == null ? "—" : `${Math.max(...payloads.map((item) => item.peak_memory_mb || 0)).toFixed(0)} MB`;
        result.classList.remove("hidden");
        errorBox.classList.add("hidden");
        status.textContent = "";
      } catch (error) {
        errorBox.textContent = error.message;
        errorBox.classList.remove("hidden");
        status.textContent = "";
      } finally {
        button.disabled = false;
      }
    };
    recorder.start();
    result.classList.add("hidden");
    errorBox.classList.add("hidden");
    button.textContent = "Stop & transcribe";
    button.classList.add("recording");
    recordingStartedAt = Date.now();
    timer.textContent = "0:00";
    timer.classList.remove("hidden");
    recordingTimer = setInterval(() => {
      updateRecordingTimer(timer);
      if (Date.now() - recordingStartedAt >= maximumSeconds * 1000) recorder.stop();
    }, 250);
    status.textContent = `Recording... max ${maximumSeconds}s.`;
  });

  document.body.addEventListener("click", async (event) => {
    if (event.target.id !== "download-diagnostics") return;
    try {
      const response = await fetch("/v1/admin/diagnostics", {
        headers: { Authorization: `Bearer ${getToken()}` },
      });
      if (response.status === 401) {
        localStorage.removeItem(TOKEN_KEY);
        showOverlay("Session expired. Paste the current token.");
        return;
      }
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error?.message || "Diagnostics failed.");
      const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = `vocaphone-diagnostics-${payload.generated_at.replace(/[:.]/g, "-")}.json`;
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(url);
      showToast("Diagnostics downloaded.", false);
    } catch (error) {
      showToast(error.message || "Could not download diagnostics.");
    }
  });

  document.body.addEventListener("click", async (event) => {
    if (event.target.id !== "copy-new-token") return;
    const value = document.getElementById("new-token-value").textContent;
    try {
      await navigator.clipboard.writeText(value);
      showToast("Token copied.", false);
    } catch (_) {
      showToast("Could not copy. Select the token and copy manually.");
    }
  });

  document.body.addEventListener("click", async (event) => {
    const button = event.target.closest && event.target.closest("#copy-pairing-qr");
    if (!button) return;
    event.preventDefault();
    const url = button.getAttribute("data-url") || "";
    const token = button.getAttribute("data-token") || "";
    const hint = button.querySelector(".pairing-qr-hint");
    const setHint = (text, copied) => {
      if (!hint) return;
      hint.textContent = text;
      button.classList.toggle("is-copied", Boolean(copied));
      button.title = copied ? "Copied" : "Click to copy gateway address and token";
    };
    if (!url || !token) {
      setHint("Nothing to copy", false);
      clearTimeout(button._copyHintTimer);
      button._copyHintTimer = setTimeout(() => setHint("Click to copy", false), 1600);
      return;
    }
    // Same JSON shape the QR encodes (see app.pairing.PairingPayload).
    const value = JSON.stringify({ v: 1, url: url, token: token });
    try {
      if (navigator.clipboard && navigator.clipboard.writeText) {
        await navigator.clipboard.writeText(value);
      } else {
        // Fallback for non-secure contexts / older browsers.
        const area = document.createElement("textarea");
        area.value = value;
        area.setAttribute("readonly", "");
        area.style.position = "fixed";
        area.style.left = "-9999px";
        document.body.appendChild(area);
        area.select();
        document.execCommand("copy");
        area.remove();
      }
      setHint("Copied", true);
      clearTimeout(button._copyHintTimer);
      button._copyHintTimer = setTimeout(() => setHint("Click to copy", false), 1600);
    } catch (_) {
      setHint("Copy failed", false);
      clearTimeout(button._copyHintTimer);
      button._copyHintTimer = setTimeout(() => setHint("Click to copy", false), 1600);
    }
  });

  document.body.addEventListener("click", async (event) => {
    if (event.target.id !== "copy-transcript") return;
    const transcript = document.getElementById("test-transcript").textContent;
    try {
      await navigator.clipboard.writeText(transcript);
      showToast("Transcript copied.", false);
    } catch (_) {
      showToast("Could not copy. Select the transcript and copy manually.");
    }
  });

  // ------------------------------------------------------------------ start

  initTheme();
  applyExposureBanner();

  if (!getToken()) {
    showOverlay();
  }

  const hash = location.hash.slice(1);
  const tabKey = hash === "test" ? "pair" : hash;
  const requestedTab = tabKey
    ? document.querySelector(`.tab[data-tab="${tabKey}"]`)
    : null;
  const initialTab = requestedTab || document.querySelector(".tab.active");
  activateTab(initialTab, false);
  if (requestedTab) {
    document.getElementById("panel").setAttribute("hx-get", requestedTab.getAttribute("hx-get"));
  }
})();
