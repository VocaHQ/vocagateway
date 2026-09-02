from __future__ import annotations

import re
from html import escape

from markupsafe import Markup

from app.fragments.shared import _format_bytes, _format_latency, _format_uptime
from app.schemas import (
    AdminStatusResponse,
    CommitStatus,
    DependencyStatus,
    OperationalMetricsStatus,
    ReadinessStatus,
    SystemStatus,
)
from app.templating import render

MAXIMUM_COMMIT_SUBJECT_LENGTH = 64


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
        else "Install vocagateway[engines] (sherpa-onnx / faster-whisper) or use Docker"
    )
    ready = (
        status.setup.token_configured
        and status.setup.ffmpeg_available
        and status.setup.engine_ready
    )
    # Network exposure warning lives in the top app banner + Settings panel.
    hero_copy = (
        "Pair a phone or run a quick test when you want."
        if ready
        else "Finish the steps below, then pair your phone."
    )
    return render(
        "overview/page.html",
        ready=ready,
        hero_copy=hero_copy,
        operations_html=operations_fragment(status.metrics, status.readiness),
        system_panel_html=_system_panel(
            status.system, status.version, machine_label, status.commit
        ),
        dependencies_html=_dependencies_panel(status.dependencies),
        onboarding_html=_onboarding_steps(status, machine_label, ffmpeg_hint, engine_hint, ready),
    )


def _system_panel(
    system: SystemStatus,
    version: str,
    machine_label: str,
    commit: CommitStatus | None = None,
) -> str:
    """Capability hero: glance-first hardware for local speech, details on expand."""
    gpus = [accelerator for accelerator in system.accelerators if accelerator != "CPU"]
    runtime = "Container" if system.containerized else "Host"
    chip = _short_chip_label(system.chip)
    features = system.cpu_features or []
    feature_line = ", ".join(features) if features else "standard"
    os_summary = _os_summary(system.os)

    headline = _capability_headline(system, gpus)
    cpu_vendor = _hw_vendor(system.chip)
    cores = system.effective_cpus if system.effective_cpus else system.logical_cpus
    core_label = ""
    if cores:
        core_label = (
            f"{cores:g}" if isinstance(cores, float) and cores != int(cores) else f"{int(cores)}"
        )
    cpu_name = _hero_cpu_bit(chip)
    if cpu_name and core_label:
        cpu_text = f"{cpu_name} · {core_label} cores"
    elif cpu_name:
        cpu_text = cpu_name
    elif core_label:
        cpu_text = f"{core_label} CPU"
    else:
        cpu_text = chip or "CPU only"
    meta_line = f"{runtime} · {os_summary} · gateway {version}"
    if commit is not None:
        meta_line += f" · build {commit.short_sha}"

    specs: list[tuple[str, str | None, str]] = []
    primary_gpu_label = None
    primary_gpu_vendor = None
    if gpus:
        primary_gpu_label = _short_gpu_label(gpus[0])
        # Use the original accelerator string for vendor detection (the short
        # label may drop the AMD/NVIDIA prefix it was detected from).
        primary_gpu_vendor = _hw_vendor(gpus[0])
        specs.append((_hero_gpu_bit(primary_gpu_label), primary_gpu_vendor, "gpu"))
    if system.ram_gb:
        specs.append((f"{system.ram_gb:g} GB RAM", None, "ram"))
    specs.append((cpu_text, cpu_vendor, "cpu"))

    secondary_gpus = [(_short_gpu_label(name), _hw_vendor(name)) for name in gpus[1:]]

    return render(
        "overview/system_panel.html",
        machine_label=machine_label,
        headline=headline,
        specs=specs,
        meta_line=meta_line,
        chip=chip,
        cpu_vendor=cpu_vendor,
        cpu_detail=f"{system.effective_cpus:g} effective · {system.logical_cpus} logical",
        ram_label=f"{system.ram_gb:g} GB",
        has_gpu=bool(gpus),
        primary_gpu_label=primary_gpu_label,
        primary_gpu_vendor=primary_gpu_vendor,
        secondary_gpus=secondary_gpus,
        feature_line=feature_line,
        os_arch=f"{system.os} ({system.arch})",
        runtime=runtime,
        version=version,
        commit=_commit_context(commit) if commit is not None else None,
    )


def _dependencies_panel(dependencies: list[DependencyStatus]) -> str:
    """Installed CLIs and Python engines — same data as diagnostics, new Overview look."""
    ordered = sorted(
        dependencies, key=lambda dependency: (not dependency.available, dependency.name.lower())
    )
    installed = sum(1 for dependency in dependencies if dependency.available)
    return render(
        "overview/dependencies_panel.html",
        installed=installed,
        total=len(dependencies),
        tiles=[
            {
                "name": dependency.name,
                "available": dependency.available,
                "detail": dependency.path or dependency.install_hint or "—",
            }
            for dependency in ordered
        ],
    )


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


def _commit_context(commit: CommitStatus) -> dict[str, str]:
    """'0979263 · Merge pull request #10…', with the full sha on hover."""
    subject = _ellipsize(commit.subject, MAXIMUM_COMMIT_SUBJECT_LENGTH)
    text = f"{commit.short_sha} · {subject}" if subject else commit.short_sha
    if commit.committed_at is not None:
        text += f" ({commit.committed_at:%Y-%m-%d})"
    return {"text": text, "sha": commit.sha}


def _ellipsize(text: str, limit: int) -> str:
    stripped = text.strip()
    return stripped if len(stripped) <= limit else stripped[: limit - 1].rstrip() + "…"


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
    url = (text or "").upper()
    if any(key in url for key in ("NVIDIA", "GEFORCE", "RTX ", "GTX ", "QUADRO", "TESLA")):
        return "nvidia"
    if any(key in url for key in ("AMD", "RADEON", "RYZEN", "EPYC", "THREADRIPPER")):
        return "amd"
    if any(key in url for key in ("INTEL", "XEON", "CORE(TM)", " ARC ", " UHD ")):
        return "intel"
    if re.search(r"\bI[3579]-", url) or re.search(r"\bI[3579]\s", url):
        return "intel"
    if any(key in url for key in ("APPLE", "M1 ", "M2 ", "M3 ", "M4 ", "APPLE M")):
        return "apple"
    return None


def _onboarding_steps(
    status: AdminStatusResponse,
    machine_label: str,
    ffmpeg_hint: str,
    engine_hint: str,
    ready: bool,
) -> str:
    """Short path for first-time operators; collapses once the gateway is ready."""
    if ready:
        return render("overview/onboarding_ready.html")
    return render(
        "overview/onboarding_steps.html",
        show_ffmpeg_step=not status.setup.ffmpeg_available,
        ffmpeg_hint=ffmpeg_hint,
        show_engine_step=not status.setup.engine_binary_available,
        engine_hint=engine_hint,
        show_download_step=not status.setup.model_installed,
        show_select_step=status.setup.model_installed and not status.setup.engine_ready,
        machine_label=machine_label,
    )


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
    metric_cards = [
        {
            "label": "Uptime",
            "value": _format_uptime(metrics.uptime_seconds),
            "detail": "This process",
            "css": "",
        },
        {
            "label": "Succeeded",
            "value": str(ok),
            "detail": "Completed transcriptions",
            "css": "success",
        },
        {
            "label": "Failed",
            "value": str(bad),
            "detail": failed_detail,
            "css": "failure" if bad else "",
        },
        {
            "label": "Avg latency",
            "value": average_latency if metrics.average_latency_ms is not None else "—",
            "detail": latency_detail,
            "css": "",
        },
        {
            "label": "Last inference",
            "value": inference_value,
            "detail": inference_detail,
            "css": "",
        },
        {"label": "Real-time factor", "value": rtf_value, "detail": rtf_detail, "css": ""},
        {
            "label": "Model cache",
            "value": warmup_label,
            "detail": warmup_detail,
            "css": readiness.warmup_state,
        },
        {
            "label": "Workload",
            "value": f"{queued} queued",
            "detail": f"{active} of {limit} active",
            "css": "",
        },
    ]
    history = metrics.history
    sample_note = (
        f"{len(history)} sample" + ("" if len(history) == 1 else "s")
        if history
        else "Collecting samples"
    )
    return render(
        "overview/operations.html",
        probe_age=f"{readiness.probe_age_seconds:.1f}",
        sample_note=sample_note,
        metric_cards=metric_cards,
        capacity=_capacity_context(active, limit, queued),
        latency_chart=_latency_chart(metrics),
        outcomes_chart=_outcomes_chart(metrics),
        pipeline_chart=_pipeline_chart(metrics, rtf_value, rtf_detail),
    )


def _capacity_context(active: int, limit: int, queued: int) -> dict[str, object]:
    active_c = min(active, limit)
    active_pct = 100.0 * active_c / limit
    queue_pct = min(100.0 - active_pct, 100.0 * queued / limit) if queued else 0.0
    free_pct = max(0.0, 100.0 - active_pct - queue_pct)
    return {
        "active": active_c,
        "limit": limit,
        "queued": queued,
        "active_pct": f"{active_pct:.2f}",
        "queue_pct": f"{queue_pct:.2f}",
        "free_pct": f"{free_pct:.2f}",
    }


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


def _latency_chart(metrics: OperationalMetricsStatus) -> dict[str, object]:
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


def _outcomes_chart(metrics: OperationalMetricsStatus) -> dict[str, object]:
    """Succeeded vs failed over the recent sample window (stacked bars)."""
    ok = metrics.successful_transcriptions
    bad = metrics.failed_transcriptions
    sub = f"{ok} succeeded · {bad} failed"
    bars = _outcome_deltas(metrics)
    chart = _stacked_bars_svg(bars)
    recent_ok = sum(first for first, _ in bars)
    recent_bad = sum(second for _, second in bars)
    if recent_ok or recent_bad:
        footer = f"Window {recent_ok} ok · {recent_bad} failed"
    else:
        footer = "Idle — no completed jobs in this window."
    return {"title": "Outcomes", "sub": sub, "body": chart, "footer": footer, "bar_body": False}


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
) -> Markup:
    if not bars or not any(ok or bad for ok, bad in bars):
        return Markup(
            f'<svg class="sparkline sparkline-empty" viewBox="0 0 {width} {height}" '
            f'width="100%" height="{height}" preserveAspectRatio="none" aria-hidden="true">'
            f'<line class="sparkline-baseline" x1="0" y1="{height // 2}" '
            f'x2="{width}" y2="{height // 2}"/></svg>'
        )
    count = len(bars)
    gap = 1.5
    bar_w = max(1.0, (width - gap * (count - 1)) / count)
    peak = max(1, max(ok + bad for ok, bad in bars))
    pad_top = 2.0
    usable = height - pad_top
    parts: list[str] = []
    for index, (ok, bad) in enumerate(bars):
        coordinate_x = index * (bar_w + gap)
        total = ok + bad
        if total <= 0:
            continue
        total_h = usable * (total / peak)
        bad_h = usable * (bad / peak) if bad else 0.0
        ok_h = total_h - bad_h
        coordinate_y = height - total_h
        if ok_h > 0:
            parts.append(
                f'<rect class="ops-bar-ok" x="{coordinate_x:.1f}" y="{coordinate_y:.1f}" '
                f'width="{bar_w:.1f}" height="{ok_h:.1f}" rx="1"/>'
            )
        if bad_h > 0:
            parts.append(
                f'<rect class="ops-bar-bad" x="{coordinate_x:.1f}" y="{height - bad_h:.1f}" '
                f'width="{bar_w:.1f}" height="{bad_h:.1f}" rx="1"/>'
            )
    return Markup(
        f'<svg class="sparkline ops-bars" viewBox="0 0 {width} {height}" width="100%" '
        f'height="{height}" preserveAspectRatio="none" role="img" '
        f'aria-label="Outcomes over {count} samples">'
        f"{''.join(parts)}</svg>"
    )


def _sparkline_card(
    title: str,
    subtitle: str,
    samples: list[float | None] | list[float],
    *,
    empty: str,
    unit_suffix: str = "",
) -> dict[str, object]:
    numeric = [float(entry_value) for entry_value in samples if entry_value is not None]
    chart = _sparkline_svg(samples)
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
        footer = f"Now {latest_label} · peak {peak_label}"
    else:
        footer = empty
    return {"title": title, "sub": subtitle, "body": chart, "footer": footer, "bar_body": False}


def _sparkline_svg(
    samples: list[float | None] | list[float],
    *,
    width: int = 220,
    height: int = 48,
) -> Markup:
    """Minimal SVG polyline; empty state is a flat baseline."""
    coords: list[tuple[float, float]] = []
    numeric = [
        (index, float(entry_value))
        for index, entry_value in enumerate(samples)
        if entry_value is not None
    ]
    if len(numeric) < 2:
        return Markup(
            f'<svg class="sparkline sparkline-empty" viewBox="0 0 {width} {height}" '
            f'width="100%" height="{height}" preserveAspectRatio="none" aria-hidden="true">'
            f'<line class="sparkline-baseline" x1="0" y1="{height // 2}" '
            f'x2="{width}" y2="{height // 2}"/></svg>'
        )
    ys = [coordinate_y for _, coordinate_y in numeric]
    y_min = min(ys)
    y_max = max(ys)
    if y_max <= y_min:
        y_max = y_min + 1.0
    pad = 3.0
    usable_h = height - pad * 2
    count = len(samples)
    span = max(1, count - 1)
    for index, raw in enumerate(samples):
        if raw is None:
            continue
        coordinate_x = (index / span) * width
        coordinate_y = pad + (1.0 - (float(raw) - y_min) / (y_max - y_min)) * usable_h
        coords.append((coordinate_x, coordinate_y))
    points = " ".join(
        f"{coordinate_x:.1f},{coordinate_y:.1f}" for coordinate_x, coordinate_y in coords
    )
    area = (
        f"M{coords[0][0]:.1f},{height} "
        + " ".join(
            f"L{coordinate_x:.1f},{coordinate_y:.1f}" for coordinate_x, coordinate_y in coords
        )
        + f" L{coords[-1][0]:.1f},{height} Z"
    )
    return Markup(
        f'<svg class="sparkline" viewBox="0 0 {width} {height}" width="100%" '
        f'height="{height}" preserveAspectRatio="none" role="img" '
        f'aria-label="Sparkline with {len(coords)} points">'
        f'<path class="sparkline-fill" d="{area}"/>'
        f'<polyline class="sparkline-line" fill="none" points="{points}"/>'
        f"</svg>"
    )


def _pipeline_chart(
    metrics: OperationalMetricsStatus, rtf_value: str, rtf_detail: str
) -> dict[str, object]:
    stages = [
        ("Normalize", metrics.normalization_ms or 0, "ops-bar-normalize"),
        ("Load", metrics.model_load_ms or 0, "ops-bar-load"),
        ("Infer", metrics.inference_ms or 0, "ops-bar-infer"),
    ]
    total = sum(ms for _, ms, _ in stages)
    if total <= 0:
        chart_markup = Markup('<div class="ops-bar ops-bar-empty" aria-hidden="true"></div>')
        footer = "Run a test to measure stages"
        sub = "Normalize · load · infer"
    else:
        segs = "".join(
            f'<span class="ops-bar-seg {css}" style="width:{100.0 * ms / total:.2f}%" '
            f'title="{escape(label)}: {_format_latency(ms)}"></span>'
            for label, ms, css in stages
            if ms > 0
        )
        chart_markup = Markup(
            f'<div class="ops-bar" role="img" '
            f'aria-label="Last pipeline {_format_latency(total)}">'
            f"{segs}</div>"
        )
        footer = " · ".join(f"{label} {_format_latency(ms)}" for label, ms, _ in stages if ms > 0)
        sub = f"RTF {rtf_value}" + (f" · {rtf_detail}" if rtf_detail else "")
    return {
        "title": "Last job stages",
        "sub": sub,
        "body": chart_markup,
        "footer": footer,
        "bar_body": True,
    }
