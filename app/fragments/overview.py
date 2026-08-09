from __future__ import annotations

from html import escape

from app.config import WILDCARD_BIND_HOSTS
from app.fragments.shared import _facts, _format_bytes, _format_latency, _format_uptime
from app.schemas import AdminStatusResponse, OperationalMetricsStatus, ReadinessStatus


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
    system_facts = _facts(
        [
            ("Chip", status.system.chip),
            (
                "CPU allocation",
                f"{status.system.effective_cpus:g} effective / "
                f"{status.system.logical_cpus} logical",
            ),
            ("Memory", f"{status.system.ram_gb:g} GB"),
            ("Accelerators", ", ".join(status.system.accelerators)),
            ("CPU features", ", ".join(status.system.cpu_features) or "standard"),
            ("Runtime", "container" if status.system.containerized else "host"),
            ("OS", f"{status.system.os} ({status.system.arch})"),
            ("Version", status.version),
        ]
    )

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
      <div class="card">
        <h2>This {machine_label}</h2>
        <dl class="facts">{system_facts}</dl>
      </div>
      {exposure_notice}
      {onboarding}
    """


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
