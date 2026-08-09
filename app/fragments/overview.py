from __future__ import annotations

from html import escape

from app.config import WILDCARD_BIND_HOSTS
from app.fragments.shared import _format_bytes, _format_latency, _format_uptime
from app.schemas import AdminStatusResponse, OperationalMetricsStatus, ReadinessStatus, SystemStatus


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

    exposure_notice = ""
    if status.bind_host in WILDCARD_BIND_HOSTS:
        firewall_hint = "Keep macOS Firewall on" if is_mac else "Keep the host firewall enabled"
        exposure_notice = f"""
          <div class="callout warning">
            <strong>Listening on every network interface</strong>
            <span>The API still needs the bearer token. {firewall_hint}, prefer Tailscale for remote access, and do not publish this port on the public internet.</span>
          </div>
        """
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
      {exposure_notice}
      {onboarding}
    """


def _system_panel(system: SystemStatus, version: str, machine_label: str) -> str:
    """Icon tiles for host facts. Inline SVGs keep the gateway offline-friendly."""
    gpus = [item for item in system.accelerators if item != "CPU"]
    gpu_value = ", ".join(gpus) if gpus else "None detected"
    cpu_value = (
        f"{system.effective_cpus:g} effective / {system.logical_cpus} logical"
    )
    features = system.cpu_features or ["standard"]
    feature_badges = "".join(
        f'<span class="sys-badge">{escape(feature)}</span>' for feature in features
    )
    runtime = "Container" if system.containerized else "Host"
    tiles = [
        _sys_tile("cpu", "Processor", system.chip),
        _sys_tile("memory", "Memory", f"{system.ram_gb:g} GB"),
        _sys_tile("cores", "CPUs", cpu_value),
        _sys_tile("gpu", "GPU", gpu_value),
        _sys_tile("features", "CPU features", feature_badges, raw_value=True),
        _sys_tile("os", "OS", f"{system.os} ({system.arch})"),
        _sys_tile("runtime", "Runtime", runtime),
        _sys_tile("version", "Gateway", version),
    ]
    return f"""
      <div class="card system-card">
        <h2>This {machine_label}</h2>
        <div class="sys-grid">{"".join(tiles)}</div>
      </div>
    """


def _sys_tile(kind: str, label: str, value: str, *, raw_value: bool = False) -> str:
    body = value if raw_value else escape(value)
    return (
        f'<article class="sys-tile sys-tile-{escape(kind)}">'
        f'<span class="sys-icon" aria-hidden="true">{_SYS_ICONS[kind]}</span>'
        f'<span class="sys-label">{escape(label)}</span>'
        f'<span class="sys-value">{body}</span>'
        f"</article>"
    )


# Compact Lucide-style strokes; same teal accent as the rest of the WebUI.
_SYS_ICONS: dict[str, str] = {
    "cpu": (
        '<svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" '
        'stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round">'
        '<rect x="5" y="5" width="14" height="14" rx="2"/>'
        '<path d="M9 1v4M15 1v4M9 19v4M15 19v4M1 9h4M1 15h4M19 9h4M19 15h4"/>'
        '<rect x="9" y="9" width="6" height="6" rx="1"/></svg>'
    ),
    "memory": (
        '<svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" '
        'stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round">'
        '<rect x="2" y="6" width="20" height="12" rx="2"/>'
        '<path d="M6 10v4M10 10v4M14 10v4M18 10v4"/></svg>'
    ),
    "cores": (
        '<svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" '
        'stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round">'
        '<circle cx="12" cy="12" r="3"/>'
        '<path d="M12 2v3M12 19v3M4.9 4.9l2.1 2.1M17 17l2.1 2.1M2 12h3M19 12h3'
        'M4.9 19.1 7 17M17 7l2.1-2.1"/></svg>'
    ),
    "gpu": (
        '<svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" '
        'stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round">'
        '<rect x="3" y="5" width="18" height="12" rx="2"/>'
        '<path d="M7 17v2M12 17v2M17 17v2M7 9h4v4H7zM14 9h3M14 12h3"/></svg>'
    ),
    "features": (
        '<svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" '
        'stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round">'
        '<path d="M12 2 15 8l6 1-4.5 4.2L18 20l-6-3.2L6 20l1.5-6.8L3 9l6-1z"/></svg>'
    ),
    "os": (
        '<svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" '
        'stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round">'
        '<rect x="3" y="4" width="18" height="12" rx="2"/>'
        '<path d="M8 20h8M12 16v4"/></svg>'
    ),
    "runtime": (
        '<svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" '
        'stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round">'
        '<path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/>'
        '<path d="M3.3 7 12 12l8.7-5M12 22V12"/></svg>'
    ),
    "version": (
        '<svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" '
        'stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round">'
        '<path d="M8 6h13M8 12h13M8 18h13M3 6h.01M3 12h.01M3 18h.01"/></svg>'
    ),
}


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
          <p class="muted">Pair a phone if you have not yet, or test from this browser. Switch models anytime from the header.</p>
        </div>
        <div class="onboarding-actions">
          <button type="button" class="primary" data-open-tab="pair">Pair &amp; test</button>
          <button type="button" class="ghost" data-open-tab="models">Browse models</button>
        </div>
      </div>
        """

    steps: list[str] = []
    if not status.setup.ffmpeg_available:
        steps.append(
            f"<li><strong>Install FFmpeg</strong>: {escape(ffmpeg_hint)}.</li>"
        )
    if not status.setup.engine_binary_available:
        steps.append(
            f"<li><strong>Install a speech engine</strong>: {escape(engine_hint)}.</li>"
        )
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
    cards = "".join(
        [
            _metric_card("Uptime", _format_uptime(metrics.uptime_seconds), "This process"),
            _metric_card(
                "Workload",
                f"{metrics.queue_depth} queued",
                f"{metrics.active_transcriptions} of {metrics.concurrency_limit} active",
            ),
            _metric_card(
                "Successful",
                str(metrics.successful_transcriptions),
                "Completed jobs",
                "success",
            ),
            _metric_card(
                "Failed",
                str(metrics.failed_transcriptions),
                f"{metrics.rejected_transcriptions} overload rejections",
                "failure" if metrics.failed_transcriptions else "",
            ),
            _metric_card("Average latency", average_latency, f"Last {last_latency}"),
            _metric_card("Model cache", warmup_label, warmup_detail, readiness.warmup_state),
            _metric_card(
                "Last inference",
                _format_latency(metrics.inference_ms),
                f"Normalize {_format_latency(metrics.normalization_ms)} · "
                f"load {_format_latency(metrics.model_load_ms)}",
            ),
            _metric_card(
                "Real-time factor",
                f"{metrics.real_time_factor:.2f}×" if metrics.real_time_factor is not None else "—",
                (
                    f"{_format_latency(metrics.audio_duration_ms)} audio · "
                    f"{metrics.peak_memory_mb:g} MB peak"
                    if metrics.peak_memory_mb is not None
                    else "Run a test to measure"
                ),
            ),
        ]
    )
    return f"""
      <section id="operations" class="operations"
               hx-get="/ui/partials/operations" hx-trigger="every 5s"
               hx-swap="outerHTML" aria-label="Gateway operations">
        <div class="section-heading">
          <h2>Live operations <span class="muted">&middot; since start</span></h2>
          <span class="probe-age">Engine checked {readiness.probe_age_seconds:.1f}s ago</span>
        </div>
        <div class="operations-grid">
          {cards}
        </div>
      </section>
    """


def _metric_card(label: str, value: str, detail: str, css: str = "") -> str:
    class_name = f"metric-card {css}".strip()
    return (
        f'<article class="{class_name}"><span class="metric-label">{escape(label)}</span>'
        f'<strong class="metric-value">{escape(value)}</strong>'
        f'<span class="metric-detail">{escape(detail)}</span></article>'
    )
