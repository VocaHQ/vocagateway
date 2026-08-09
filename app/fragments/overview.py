from __future__ import annotations

import re
from html import escape

from app.fragments.shared import _format_bytes, _format_latency, _format_uptime
from app.schemas import (
    AdminStatusResponse,
    DependencyStatus,
    OperationalMetricsStatus,
    ReadinessStatus,
    SystemStatus,
)


def overview_fragment(status: AdminStatusResponse, pairing_html: str = "") -> str:
    # pairing_html is ignored: pairing lives under Pair & test so Overview stays
    # a status dashboard after the one-time setup is done.
    del pairing_html
    is_mac = status.system.os.startswith("Darwin")
    machine_label = "Mac" if is_mac else "server"
    ffmpeg_hint = "brew install ffmpeg" if is_mac else "Install FFmpeg with your package manager"
    engine_hint = (
        "brew install whisper-cpp or whisperkit-cli"
        if is_mac
        else "Install vocaphone-gateway[engines] (sherpa-onnx / faster-whisper) or use Docker"
    )
    ready = (
        status.setup.token_configured
        and status.setup.ffmpeg_available
        and status.setup.engine_ready
    )
    onboarding = _onboarding_steps(status, machine_label, ffmpeg_hint, engine_hint, ready)
    # Network exposure warning lives in the top app banner + Settings panel.
    hero_copy = (
        "Pair a phone or run a quick test when you want."
        if ready
        else "Finish the steps below, then pair your phone."
    )
    return f"""
      <section class="status-hero">
        <div class="headline">
          <span class="dot {"ok" if ready else "warn"}" aria-hidden="true"></span>
          <div>
            <h2>{"Ready for dictation" if ready else "Setup still needed"}</h2>
            <p>{hero_copy}</p>
          </div>
        </div>
      </section>
      {operations_fragment(status.metrics, status.readiness)}
      {_system_panel(status.system, status.version, machine_label)}
      {_dependencies_panel(status.dependencies)}
      {onboarding}
    """


def _system_panel(system: SystemStatus, version: str, machine_label: str) -> str:
    """Capability hero: glance-first hardware for local speech, details on expand."""
    gpus = [item for item in system.accelerators if item != "CPU"]
    primary_gpu = _short_gpu_label(gpus[0]) if gpus else None
    secondary_gpus = [_short_gpu_label(name) for name in gpus[1:]]
    runtime = "Container" if system.containerized else "Host"
    chip = _short_chip_label(system.chip)
    features = system.cpu_features or []
    feature_line = ", ".join(features) if features else "standard"
    os_summary = _os_summary(system.os)

    headline = _capability_headline(system, gpus)
    spec_parts: list[str] = []
    if primary_gpu:
        # Use original accelerator string for vendor detect (short label may drop AMD/NVIDIA).
        gpu_vendor = _hw_vendor(gpus[0] if gpus else primary_gpu)
        spec_parts.append(_spec_chip(_hero_gpu_bit(primary_gpu), vendor=gpu_vendor, kind="gpu"))
    if system.ram_gb:
        spec_parts.append(_spec_chip(f"{system.ram_gb:g} GB RAM", vendor=None, kind="ram"))
    cores = system.effective_cpus if system.effective_cpus else system.logical_cpus
    core_label = ""
    if cores:
        core_label = (
            f"{cores:g}" if isinstance(cores, float) and cores != int(cores) else f"{int(cores)}"
        )
    cpu_name = _hero_cpu_bit(chip)
    cpu_vendor = _hw_vendor(system.chip)
    if cpu_name and core_label:
        cpu_text = f"{cpu_name} · {core_label} cores"
    elif cpu_name:
        cpu_text = cpu_name
    elif core_label:
        cpu_text = f"{core_label} CPU"
    else:
        cpu_text = chip or "CPU only"
    spec_parts.append(_spec_chip(cpu_text, vendor=cpu_vendor, kind="cpu"))
    specs_html = '<span class="sys-spec-sep" aria-hidden="true">·</span>'.join(spec_parts)
    meta_line = f"{runtime} · {os_summary} · gateway {escape(version)}"

    if gpus:
        gpu_detail = _labeled_hw(_short_gpu_label(gpus[0]), gpus[0])
        if secondary_gpus:
            gpu_detail += "".join(
                f'<span class="sys-detail-extra">{_labeled_hw(short, orig)}</span>'
                for short, orig in zip(secondary_gpus, gpus[1:], strict=False)
            )
    else:
        gpu_detail = "None detected"
    cpu_detail = f"{system.effective_cpus:g} effective · {system.logical_cpus} logical"
    details_rows = [
        ("Processor", _labeled_hw(chip, system.chip)),
        ("CPUs", escape(cpu_detail)),
        ("Memory", escape(f"{system.ram_gb:g} GB")),
        ("GPU", gpu_detail),
        ("CPU features", escape(feature_line)),
        ("OS", escape(f"{system.os} ({system.arch})")),
        ("Runtime", escape(runtime)),
        ("Gateway", escape(version)),
    ]
    facts = "".join(f"<dt>{escape(label)}</dt><dd>{value}</dd>" for label, value in details_rows)
    return f"""
      <div class="card system-card">
        <div class="sys-hero">
          <div class="sys-hero-copy">
            <p class="sys-hero-kicker">This {escape(machine_label)}</p>
            <h2 class="sys-hero-headline">{escape(headline)}</h2>
            <p class="sys-hero-specs">{specs_html}</p>
            <p class="sys-hero-meta">{meta_line}</p>
          </div>
        </div>
        <details class="sys-details">
          <summary>Hardware details</summary>
          <dl class="facts sys-details-facts">{facts}</dl>
        </details>
      </div>
    """


def _dependencies_panel(dependencies: list[DependencyStatus]) -> str:
    """Installed CLIs and Python engines — same data as diagnostics, new Overview look."""
    ordered = sorted(dependencies, key=lambda d: (not d.available, d.name.lower()))
    installed = sum(1 for dependency in dependencies if dependency.available)
    total = len(dependencies)
    tiles = "".join(_dependency_tile(dependency) for dependency in ordered)
    return f"""
      <div class="card libraries-card">
        <div class="lib-hero">
          <p class="lib-kicker">Libraries &amp; tools</p>
          <h2 class="lib-headline">{installed} of {total} available</h2>
          <p class="lib-meta">Speech engines, FFmpeg, and companion apps this host can see.</p>
        </div>
        <div class="lib-grid" role="list">{tiles}</div>
      </div>
    """


def _dependency_tile(dependency: DependencyStatus) -> str:
    status = "available" if dependency.available else "missing"
    label = "Installed" if dependency.available else "Missing"
    detail = dependency.path or dependency.install_hint or "—"
    return f"""
      <article class="lib-tile is-{status}" role="listitem">
        <div class="lib-tile-top">
          <span class="dot {"ok" if dependency.available else "bad"}" aria-hidden="true"></span>
          <span class="lib-tile-name">{escape(dependency.name)}</span>
          <span class="lib-tile-status">{label}</span>
        </div>
        <p class="lib-tile-detail" title="{escape(detail, quote=True)}">{escape(detail)}</p>
      </article>
    """


def _capability_headline(system: SystemStatus, gpus: list[str]) -> str:
    """Soft capability line — facts, not a marketing claim."""
    if system.containerized:
        return "Local speech in a container"
    if gpus:
        return "Ready for local speech"
    if system.ram_gb >= 8:
        return "Ready for local speech"
    return "Host capabilities"


def _os_summary(os_line: str) -> str:
    """'Linux 6.17.0-…' → 'Linux'; Darwin stays Darwin."""
    text = (os_line or "").strip()
    if not text:
        return "Unknown OS"
    return text.split()[0]


def _hero_gpu_bit(label: str) -> str:
    """Tighten GPU for the hero line: 'NVIDIA GeForce RTX 5080 (15.9 GB)' → 'RTX 5080 (15.9 GB)'."""
    text = label.strip()
    for prefix in ("NVIDIA GeForce ", "NVIDIA ", "AMD "):
        if text.upper().startswith(prefix.upper()):
            text = text[len(prefix) :].lstrip()
            break
    return text


def _hero_cpu_bit(chip: str) -> str:
    """Tighten CPU for the hero line: 'AMD Ryzen 7 9800X3D 8-Core' → 'Ryzen 7 9800X3D'."""
    text = _short_chip_label(chip)
    if not text:
        return ""
    for prefix in (
        "AMD ",
        "Intel(R) Core(TM) ",
        "Intel(R) ",
        "Intel Core ",
        "Intel ",
        "Apple ",
    ):
        if text.startswith(prefix):
            text = text[len(prefix) :].lstrip()
            break
    # Drop marketing core-count tokens already covered by the cores bit.
    text = re.sub(r"\s+\d+-Core\b", "", text, flags=re.IGNORECASE).strip()
    return text


def _short_chip_label(chip: str) -> str:
    """Drop redundant marketing suffixes so the processor line stays scannable."""
    text = chip.strip()
    for suffix in (" Processor", " CPU", " with Radeon Graphics"):
        if text.endswith(suffix):
            text = text[: -len(suffix)].rstrip()
    return text


def _short_gpu_label(name: str) -> str:
    """Prefer product names over long PCI marketing strings."""
    text = name.strip()
    # "Granite Ridge [Radeon Graphics]" → "Radeon Graphics" when nested in brackets.
    if "[" in text and text.endswith("]"):
        inner = text[text.rfind("[") + 1 : -1].strip()
        if inner:
            vendor = "AMD " if text.upper().startswith("AMD") else ""
            if not inner.upper().startswith(vendor.strip().upper()):
                return f"{vendor}{inner}".strip()
            return inner
    return text


def _hw_vendor(text: str) -> str | None:
    """Best-effort vendor from a chip/GPU string for brand mark icons."""
    u = (text or "").upper()
    if any(k in u for k in ("NVIDIA", "GEFORCE", "RTX ", "GTX ", "QUADRO", "TESLA")):
        return "nvidia"
    if any(k in u for k in ("AMD", "RADEON", "RYZEN", "EPYC", "THREADRIPPER")):
        return "amd"
    if any(k in u for k in ("INTEL", "XEON", "CORE(TM)", " ARC ", " UHD ")):
        return "intel"
    if re.search(r"\bI[3579]-", u) or re.search(r"\bI[3579]\s", u):
        return "intel"
    if any(k in u for k in ("APPLE", "M1 ", "M2 ", "M3 ", "M4 ", "APPLE M")):
        return "apple"
    return None


# Filled brand marks — colored via CSS currentColor (no circular badge).
# NVIDIA / Intel icons: vectorlogo.zone. AMD arrow: file-icons/icons AMD.svg (ISC).
# https://www.vectorlogo.zone/logos/nvidia/nvidia-icon.svg
# https://www.vectorlogo.zone/logos/intel/intel-icon.svg
# https://github.com/file-icons/icons/blob/master/svg/AMD.svg
_VENDOR_MARKS: dict[str, tuple[str, tuple[str, ...]]] = {
    # viewBox, path d's (fill=currentColor so theme CSS sets brand color)
    "nvidia": (
        "0 0 64 64",
        (
            "M23.862 23.46v-3.816l1.13-.047c10.46-.33 17.313 8.998 17.313 8.998s-7.396 "
            "10.27-15.335 10.27a9.73 9.73 0 0 1-3.086-.495v-11.59c4.075.495 4.9 2.285 "
            "7.326 6.36l5.44-4.57s-3.98-5.206-10.67-5.206c-.707-.024-1.413.024-2.12.094"
            "m0-12.626v5.7l1.13-.07c14.534-.495 24.026 11.92 24.026 11.92S38.136 41.622 "
            "26.806 41.622c-.99 0-1.955-.094-2.92-.26v3.533c.8.094 1.625.165 2.426.165 "
            "10.553 0 18.185-5.394 25.58-11.754 1.225.99 6.242 3.368 7.28 4.405-7.02 "
            "5.89-23.39 10.623-32.67 10.623a23.24 23.24 0 0 1-2.591-.141v4.97H64v-42.33"
            "zm0 27.536v3.015C14.1 39.644 11.4 29.49 11.4 29.49s4.688-5.182 12.46-6.03"
            "v3.298h-.024c-4.075-.495-7.28 3.32-7.28 3.32s1.814 6.43 7.302 8.29M6.548 "
            "29.067s5.77-8.527 17.337-9.422v-3.11C11.07 17.572 0 28.408 0 28.408s6.266 "
            "18.138 23.862 19.787v-3.298c-12.908-1.602-17.313-15.83-17.313-15.83z",
        ),
    ),
    "amd": (
        "0 0 512 512",
        (
            "M512,512L369.6281433,369.7356873V142.2643127H142.0852051L0,0h512V512z "
            "M142.0852051,369.7356873V165.0039978L0,307.2261963V512h204.6510773"
            "l142.219223-142.2643127H142.0852051z",
        ),
    ),
    "intel": (
        "0 0 64 64",
        (
            "M63.72 22.866C60.703 8.17 32.275 7.23 13.95 18.427v1.246c18.303-9.447 "
            "44.26-9.403 46.644 4.133.787 4.483-1.706 9.14-6.2 11.83v3.52c5.4-2 "
            "10.934-8.42 9.337-16.29M30.394 48.583C17.755 49.764 4.57 47.905 2.732 "
            "38c-.918-4.876 1.312-10.06 4.264-13.296v-1.728C1.682 27.655-1.204 33.56"
            ".48 40.514 2.6 49.457 13.906 54.53 31.16 52.847c6.845-.656 15.788-2.865 "
            "21.977-6.298v-4.876c-5.642 3.39-14.957 6.19-22.742 6.9z",
            "M52.13 20.723h-3.324V35.55c0 1.75.83 3.258 3.324 3.5M12.638 26.147H9.336"
            "v9.687c0 1.75.83 3.258 3.324 3.5m-3.324-18.15h3.324v3.15H9.336zm23.18 "
            "17.997c-2.7 0-3.827-1.88-3.827-3.717v-12.88h3.28v3.564h2.493v2.668H31.97"
            "v6.43c0 .765.372 1.18 1.137 1.18h1.334v2.755h-1.924m8.703-10.52c-1.115 "
            "0-2 .6-2.34 1.378-.22.48-.284.83-.328 1.42h5.073c-.066-1.443-.722-2.8"
            "-2.405-2.8m-2.668 5.03c0 1.684 1.05 2.93 2.908 2.93 1.465 0 2.187-.415 "
            "3.018-1.246l2.034 1.946c-1.312 1.3-2.668 2.077-5.073 2.077-3.15 0-6.167"
            "-1.728-6.167-6.757 0-4.286 2.624-6.713 6.08-6.713 3.5 0 5.5 2.843 5.5 "
            "6.582v1.18h-8.3m-16.98-4.873c.962 0 1.356.48 1.356 1.246v9.12h3.28v-9.12"
            "c0-1.86-.984-3.914-3.87-3.914h-6.78V39.2h3.28V28.814",
        ),
    ),
}


def _vendor_icon(vendor: str | None) -> str:
    """Bare brand mark next to hardware text (no circular badge chrome)."""
    if not vendor:
        return ""
    title = {"nvidia": "NVIDIA", "amd": "AMD", "intel": "Intel", "apple": "Apple"}.get(
        vendor, vendor
    )
    mark = _VENDOR_MARKS.get(vendor)
    if mark:
        view_box, paths = mark
        glyph = "".join(f'<path fill="currentColor" d="{d}"/>' for d in paths)
        svg_attrs = f'viewBox="{view_box}" width="18" height="18" aria-hidden="true"'
    elif vendor == "apple":
        # Simple outline bite-apple silhouette, monoline.
        glyph = (
            '<path fill="none" stroke="currentColor" stroke-width="1.5" '
            'stroke-linecap="round" stroke-linejoin="round" '
            'd="M14.5 6.5c-.8-1.2-2-2-3.3-2"/>'
            '<path fill="none" stroke="currentColor" stroke-width="1.5" '
            'stroke-linecap="round" stroke-linejoin="round" '
            'd="M12 8.2c-2.8 0-5 2.4-5 5.6 0 3.4 2.2 6.2 5 6.2s5-2.8 5-6.2'
            'c0-3.2-2.2-5.6-5-5.6z"/>'
        )
        svg_attrs = 'viewBox="0 0 24 24" width="18" height="18" aria-hidden="true"'
    else:
        return ""
    return (
        f'<span class="sys-vendor sys-vendor-{escape(vendor)}" title="{escape(title)}" '
        f'aria-label="{escape(title)}">'
        f"<svg {svg_attrs}>{glyph}</svg></span>"
    )


def _spec_chip(text: str, *, vendor: str | None, kind: str) -> str:
    icon = _vendor_icon(vendor)
    return (
        f'<span class="sys-spec sys-spec-{escape(kind)}">'
        f'{icon}<span class="sys-spec-text">{escape(text)}</span></span>'
    )


def _labeled_hw(display: str, source_for_vendor: str) -> str:
    """Detail-row value with optional vendor monogram."""
    icon = _vendor_icon(_hw_vendor(source_for_vendor))
    if not icon:
        return escape(display)
    return f'{icon}<span class="sys-detail-text">{escape(display)}</span>'


def _onboarding_steps(
    status: AdminStatusResponse,
    machine_label: str,
    ffmpeg_hint: str,
    engine_hint: str,
    ready: bool,
) -> str:
    """Short path for first-time operators; collapses once the gateway is ready."""
    if ready:
        return """
      <div class="onboarding-ready card">
        <div class="onboarding-ready-copy">
          <h2>You&rsquo;re set</h2>
          <p class="muted">Pair a phone if you have not yet, or test from this browser.
             Switch models anytime from the header.</p>
        </div>
        <div class="onboarding-actions">
          <button type="button" class="primary" data-open-tab="pair">Pair &amp; test</button>
          <button type="button" class="ghost" data-open-tab="models">Browse models</button>
        </div>
      </div>
        """

    steps: list[str] = []
    if not status.setup.ffmpeg_available:
        steps.append(f"<li><strong>Install FFmpeg</strong>: {escape(ffmpeg_hint)}.</li>")
    if not status.setup.engine_binary_available:
        steps.append(f"<li><strong>Install a speech engine</strong>: {escape(engine_hint)}.</li>")
    if not status.setup.model_installed or not status.setup.engine_ready:
        steps.append(
            f"<li><strong>Download a model</strong> for this {escape(machine_label)}. "
            'Open <button type="button" class="text-link" data-open-tab="models">Models</button> '
            "and pick a recommended one.</li>"
        )
    steps.append(
        "<li><strong>Pair your phone</strong>: "
        'scan the QR under <button type="button" class="text-link" data-open-tab="pair">'
        "Pair &amp; test</button> once.</li>"
    )
    steps.append(
        "<li><strong>Record a short test</strong> on that same tab to confirm audio works.</li>"
    )
    return f"""
      <div class="card onboarding">
        <h2>Get started</h2>
        <p class="muted">Quick path to first dictation. You only scan the pairing QR once.</p>
        <ol class="steps onboarding-steps">{"".join(steps)}</ol>
        <div class="onboarding-actions">
          <button type="button" class="primary" data-open-tab="models">Choose a model</button>
          <button type="button" class="ghost" data-open-tab="pair">Pair &amp; test</button>
        </div>
      </div>
    """


def operations_fragment(metrics: OperationalMetricsStatus, readiness: ReadinessStatus) -> str:
    """Live operations: every main-era fact, KPIs for counters, charts for trends/stages."""
    average_latency = _format_latency(metrics.average_latency_ms)
    last_latency = _format_latency(metrics.last_latency_ms)
    warmup_labels = {
        "pending": ("Pending", "Startup warm-up has not run yet."),
        "warming": ("Warming", "Priming the selected model."),
        "complete": (
            "Warm",
            f"{_format_bytes(readiness.warmed_bytes)} of model data ready.",
        ),
        "unsupported": ("Ready", "This engine does not prefetch models."),
        "unavailable": ("Waiting", "Install or select a model to warm it."),
        "failed": ("Needs retry", "Warm-up failed; transcription can still retry."),
    }
    warmup_label, warmup_detail = warmup_labels[readiness.warmup_state]
    ok = metrics.successful_transcriptions
    bad = metrics.failed_transcriptions
    rejected = metrics.rejected_transcriptions
    failed_detail = (
        f"{rejected} overload rejection" + ("s" if rejected != 1 else "")
        if rejected
        else "Since this process started"
    )
    latency_detail = (
        f"Last {last_latency}" if metrics.last_latency_ms is not None else "Run a test to measure"
    )
    rtf_value = f"{metrics.real_time_factor:.2f}×" if metrics.real_time_factor is not None else "—"
    rtf_bits: list[str] = []
    if metrics.audio_duration_ms is not None:
        rtf_bits.append(f"{_format_latency(metrics.audio_duration_ms)} audio")
    if metrics.peak_memory_mb is not None:
        rtf_bits.append(f"{metrics.peak_memory_mb:g} MB peak")
    rtf_detail = " · ".join(rtf_bits) if rtf_bits else "After first job"
    inference_value = _format_latency(metrics.inference_ms)
    inference_detail = (
        f"Normalize {_format_latency(metrics.normalization_ms)} · "
        f"load {_format_latency(metrics.model_load_ms)}"
        if metrics.inference_ms is not None
        or metrics.normalization_ms is not None
        or metrics.model_load_ms is not None
        else "Stages after first job"
    )
    active = metrics.active_transcriptions
    limit = max(1, metrics.concurrency_limit)
    queued = metrics.queue_depth
    # Capacity strip visualizes the same active/queue numbers as Workload.
    cards = "".join(
        [
            _metric_card("Uptime", _format_uptime(metrics.uptime_seconds), "This process"),
            _metric_card("Succeeded", str(ok), "Completed transcriptions", "success"),
            _metric_card(
                "Failed",
                str(bad),
                failed_detail,
                "failure" if bad else "",
            ),
            _metric_card(
                "Avg latency",
                average_latency if metrics.average_latency_ms is not None else "—",
                latency_detail,
            ),
            _metric_card("Last inference", inference_value, inference_detail),
            _metric_card("Real-time factor", rtf_value, rtf_detail),
            _metric_card("Model cache", warmup_label, warmup_detail, readiness.warmup_state),
            _metric_card(
                "Workload",
                f"{queued} queued",
                f"{active} of {limit} active",
            ),
        ]
    )
    history = metrics.history
    sample_note = (
        f"{len(history)} sample" + ("" if len(history) == 1 else "s")
        if history
        else "Collecting samples"
    )
    return f"""
      <section id="operations" class="operations"
               hx-get="/ui/partials/operations" hx-trigger="every 5s"
               hx-swap="outerHTML" aria-label="Gateway operations">
        <div class="section-heading">
          <h2>Live operations <span class="muted">&middot; since start</span></h2>
          <span class="probe-age">Engine checked {readiness.probe_age_seconds:.1f}s ago ·
            {escape(sample_note)}</span>
        </div>
        <div class="operations-grid operations-grid-kpis">
          {cards}
        </div>
        {_capacity_strip(active, limit, queued)}
        <div class="ops-charts ops-charts-inline" aria-label="Recent activity">
          <div class="ops-charts-grid ops-charts-grid-3">
            {_latency_chart(metrics)}
            {_outcomes_chart(metrics)}
            {_pipeline_chart(metrics, rtf_value, rtf_detail)}
          </div>
        </div>
      </section>
    """


def _capacity_strip(active: int, limit: int, queued: int) -> str:
    active_c = min(active, limit)
    active_pct = 100.0 * active_c / limit
    queue_pct = min(100.0 - active_pct, 100.0 * queued / limit) if queued else 0.0
    free_pct = max(0.0, 100.0 - active_pct - queue_pct)
    return f"""
      <div class="ops-capacity" aria-label="{active_c} of {limit} active, {queued} queued">
        <div class="ops-capacity-meta">
          <span>Capacity</span>
          <span class="muted">{active_c} of {limit} active · {queued} queued</span>
        </div>
        <div class="ops-bar ops-bar-thin" role="presentation">
          <span class="ops-bar-seg ops-bar-active" style="width:{active_pct:.2f}%"></span>
          <span class="ops-bar-seg ops-bar-queue" style="width:{queue_pct:.2f}%"></span>
          <span class="ops-bar-seg ops-bar-free" style="width:{free_pct:.2f}%"></span>
        </div>
      </div>
    """


def _history_series(metrics: OperationalMetricsStatus) -> tuple[list[float | None], list[float]]:
    history = metrics.history
    latency_series: list[float | None] = []
    last_known: float | None = None
    for point in history:
        if point.last_latency_ms is not None:
            last_known = float(point.last_latency_ms)
        latency_series.append(last_known)
    throughput_series: list[float] = []
    prev_done = 0
    for point in history:
        done = point.successful_transcriptions + point.failed_transcriptions
        throughput_series.append(float(max(0, done - prev_done)))
        prev_done = done
    return latency_series, throughput_series


def _latency_chart(metrics: OperationalMetricsStatus) -> str:
    series, _ = _history_series(metrics)
    avg = _format_latency(metrics.average_latency_ms)
    last = _format_latency(metrics.last_latency_ms)
    if metrics.last_latency_ms is None and metrics.average_latency_ms is None:
        sub = "Last ~5 min"
        empty = "Run a transcription to plot latency."
    else:
        sub = f"Avg {avg} · last {last}"
        empty = sub
    return _sparkline_card("Latency", sub, series, empty=empty, unit_suffix="ms")


def _outcomes_chart(metrics: OperationalMetricsStatus) -> str:
    """Succeeded vs failed over the recent sample window (stacked bars)."""
    ok = metrics.successful_transcriptions
    bad = metrics.failed_transcriptions
    sub = f"{ok} succeeded · {bad} failed"
    bars = _outcome_deltas(metrics)
    chart = _stacked_bars_svg(bars)
    recent_ok = sum(a for a, _ in bars)
    recent_bad = sum(b for _, b in bars)
    if recent_ok or recent_bad:
        footer = f"Window {recent_ok} ok · {recent_bad} failed"
    else:
        footer = "Idle — no completed jobs in this window."
    return f"""
      <article class="ops-chart">
        <header class="ops-chart-head">
          <span class="ops-chart-title">Outcomes</span>
          <span class="ops-chart-sub">{escape(sub)}</span>
        </header>
        <div class="ops-chart-body">{chart}</div>
        <p class="ops-chart-footer">{escape(footer)}</p>
      </article>
    """


def _outcome_deltas(metrics: OperationalMetricsStatus) -> list[tuple[int, int]]:
    bars: list[tuple[int, int]] = []
    prev_ok = 0
    prev_bad = 0
    for point in metrics.history:
        ok = max(0, point.successful_transcriptions - prev_ok)
        bad = max(0, point.failed_transcriptions - prev_bad)
        bars.append((ok, bad))
        prev_ok = point.successful_transcriptions
        prev_bad = point.failed_transcriptions
    return bars


def _stacked_bars_svg(
    bars: list[tuple[int, int]],
    *,
    width: int = 220,
    height: int = 48,
) -> str:
    if not bars or not any(ok or bad for ok, bad in bars):
        return (
            f'<svg class="sparkline sparkline-empty" viewBox="0 0 {width} {height}" '
            f'width="100%" height="{height}" preserveAspectRatio="none" aria-hidden="true">'
            f'<line class="sparkline-baseline" x1="0" y1="{height // 2}" '
            f'x2="{width}" y2="{height // 2}"/></svg>'
        )
    n = len(bars)
    gap = 1.5
    bar_w = max(1.0, (width - gap * (n - 1)) / n)
    peak = max(1, max(ok + bad for ok, bad in bars))
    pad_top = 2.0
    usable = height - pad_top
    parts: list[str] = []
    for i, (ok, bad) in enumerate(bars):
        x = i * (bar_w + gap)
        total = ok + bad
        if total <= 0:
            continue
        total_h = usable * (total / peak)
        bad_h = usable * (bad / peak) if bad else 0.0
        ok_h = total_h - bad_h
        y = height - total_h
        if ok_h > 0:
            parts.append(
                f'<rect class="ops-bar-ok" x="{x:.1f}" y="{y:.1f}" '
                f'width="{bar_w:.1f}" height="{ok_h:.1f}" rx="1"/>'
            )
        if bad_h > 0:
            parts.append(
                f'<rect class="ops-bar-bad" x="{x:.1f}" y="{height - bad_h:.1f}" '
                f'width="{bar_w:.1f}" height="{bad_h:.1f}" rx="1"/>'
            )
    return (
        f'<svg class="sparkline ops-bars" viewBox="0 0 {width} {height}" width="100%" '
        f'height="{height}" preserveAspectRatio="none" role="img" '
        f'aria-label="Outcomes over {n} samples">'
        f"{''.join(parts)}</svg>"
    )


def _sparkline_card(
    title: str,
    subtitle: str,
    values: list[float | None] | list[float],
    *,
    empty: str,
    unit_suffix: str = "",
) -> str:
    numeric = [float(v) for v in values if v is not None]
    chart = _sparkline_svg(values)
    if len(numeric) >= 2:
        latest = numeric[-1]
        if unit_suffix == "ms":
            latest_label = _format_latency(int(round(latest)))
        elif latest == int(latest):
            latest_label = str(int(latest))
        else:
            latest_label = f"{latest:.2f}"
        peak = max(numeric)
        peak_label = _format_latency(int(round(peak))) if unit_suffix == "ms" else f"{peak:g}"
        footer = f"Now {escape(latest_label)} · peak {escape(peak_label)}"
    else:
        footer = escape(empty)
    return f"""
      <article class="ops-chart">
        <header class="ops-chart-head">
          <span class="ops-chart-title">{escape(title)}</span>
          <span class="ops-chart-sub">{escape(subtitle)}</span>
        </header>
        <div class="ops-chart-body">{chart}</div>
        <p class="ops-chart-footer">{footer}</p>
      </article>
    """


def _sparkline_svg(
    values: list[float | None] | list[float],
    *,
    width: int = 220,
    height: int = 48,
) -> str:
    """Minimal SVG polyline; empty state is a flat baseline."""
    coords: list[tuple[float, float]] = []
    numeric = [(i, float(v)) for i, v in enumerate(values) if v is not None]
    if len(numeric) < 2:
        return (
            f'<svg class="sparkline sparkline-empty" viewBox="0 0 {width} {height}" '
            f'width="100%" height="{height}" preserveAspectRatio="none" aria-hidden="true">'
            f'<line class="sparkline-baseline" x1="0" y1="{height // 2}" '
            f'x2="{width}" y2="{height // 2}"/></svg>'
        )
    ys = [y for _, y in numeric]
    y_min = min(ys)
    y_max = max(ys)
    if y_max <= y_min:
        y_max = y_min + 1.0
    pad = 3.0
    usable_h = height - pad * 2
    n = len(values)
    span = max(1, n - 1)
    for i, raw in enumerate(values):
        if raw is None:
            continue
        x = (i / span) * width
        y = pad + (1.0 - (float(raw) - y_min) / (y_max - y_min)) * usable_h
        coords.append((x, y))
    points = " ".join(f"{x:.1f},{y:.1f}" for x, y in coords)
    area = (
        f"M{coords[0][0]:.1f},{height} "
        + " ".join(f"L{x:.1f},{y:.1f}" for x, y in coords)
        + f" L{coords[-1][0]:.1f},{height} Z"
    )
    return (
        f'<svg class="sparkline" viewBox="0 0 {width} {height}" width="100%" '
        f'height="{height}" preserveAspectRatio="none" role="img" '
        f'aria-label="Sparkline with {len(coords)} points">'
        f'<path class="sparkline-fill" d="{area}"/>'
        f'<polyline class="sparkline-line" fill="none" points="{points}"/>'
        f"</svg>"
    )


def _pipeline_chart(metrics: OperationalMetricsStatus, rtf_value: str, rtf_detail: str) -> str:
    stages = [
        ("Normalize", metrics.normalization_ms or 0, "ops-bar-normalize"),
        ("Load", metrics.model_load_ms or 0, "ops-bar-load"),
        ("Infer", metrics.inference_ms or 0, "ops-bar-infer"),
    ]
    total = sum(ms for _, ms, _ in stages)
    if total <= 0:
        bar = '<div class="ops-bar ops-bar-empty" aria-hidden="true"></div>'
        footer = "Run a test to measure stages"
        sub = "Normalize · load · infer"
    else:
        segs = "".join(
            f'<span class="ops-bar-seg {css}" style="width:{100.0 * ms / total:.2f}%" '
            f'title="{escape(label)}: {_format_latency(ms)}"></span>'
            for label, ms, css in stages
            if ms > 0
        )
        bar = (
            f'<div class="ops-bar" role="img" '
            f'aria-label="Last pipeline {_format_latency(total)}">'
            f"{segs}</div>"
        )
        footer = " · ".join(f"{label} {_format_latency(ms)}" for label, ms, _ in stages if ms > 0)
        sub = f"RTF {rtf_value}" + (f" · {rtf_detail}" if rtf_detail else "")
    return f"""
      <article class="ops-chart">
        <header class="ops-chart-head">
          <span class="ops-chart-title">Last job stages</span>
          <span class="ops-chart-sub">{escape(sub)}</span>
        </header>
        <div class="ops-chart-body ops-chart-body-bar">{bar}</div>
        <p class="ops-chart-footer">{escape(footer)}</p>
      </article>
    """


def _metric_card(label: str, value: str, detail: str, css: str = "") -> str:
    class_name = f"metric-card {css}".strip()
    return (
        f'<article class="{class_name}"><span class="metric-label">{escape(label)}</span>'
        f'<strong class="metric-value">{escape(value)}</strong>'
        f'<span class="metric-detail">{escape(detail)}</span></article>'
    )
