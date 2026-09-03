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
DETAIL_KEY = "detail"
LABEL_KEY = "label"
VALUE_KEY = "value"
CSS_KEY = "css"
TITLE_KEY = "title"
SUB_KEY = "sub"
BODY_KEY = "body"
FOOTER_KEY = "footer"
BAR_BODY_KEY = "bar_body"
GENERAL_FORMAT = "g"
ONE_DECIMAL = ".1f"
TWO_DECIMALS = ".2f"
SystemSpec = tuple[str, str | None, str]
SystemSpecsContext = tuple[list[SystemSpec], str | None, str | None]
MetricCard = dict[str, str]
Chart = dict[str, object]
Stage = tuple[str, int, str]
Samples = list[float | None] | list[float]


class _HardwareLabels:
    @classmethod
    def headline(cls, system: SystemStatus, gpus: list[str]) -> str:
        if system.containerized:
            return "Local speech in a container"
        if gpus or system.ram_gb >= 8:
            return "Ready for local speech"
        return "Host capabilities"

    @classmethod
    def os_summary(cls, os_line: str) -> str:
        text = (os_line or "").strip()
        if not text:
            return "Unknown OS"
        return text.split()[0]

    @classmethod
    def vendor(cls, text: str) -> str | None:
        url = (text or "").upper()
        vendors = (
            ("nvidia", ("NVIDIA", "GEFORCE", "RTX ", "GTX ", "QUADRO", "TESLA")),
            ("amd", ("AMD", "RADEON", "RYZEN", "EPYC", "THREADRIPPER")),
            ("intel", ("INTEL", "XEON", "CORE(TM)", " ARC ", " UHD ")),
            ("apple", ("APPLE", "M1 ", "M2 ", "M3 ", "M4 ", "APPLE M")),
        )
        for vendor_name, keys in vendors:
            if any(key in url for key in keys):
                return vendor_name
        if re.search(r"\bI[3579]-", url) or re.search(r"\bI[3579]\s", url):
            return "intel"
        return None


class _HardwareCopy:
    @classmethod
    def short_chip(cls, chip: str) -> str:
        text = chip.strip()
        for suffix in (" Processor", " CPU", " with Radeon Graphics"):
            if text.endswith(suffix):
                text = text[: -len(suffix)].rstrip()
        return text

    @classmethod
    def short_gpu(cls, name: str) -> str:
        text = name.strip()
        bracketed = cls._bracketed_gpu(text)
        return bracketed if bracketed else text

    @classmethod
    def hero_gpu(cls, label: str) -> str:
        text = label.strip()
        for prefix in ("NVIDIA GeForce ", "NVIDIA ", "AMD "):
            if text.upper().startswith(prefix.upper()):
                return text[len(prefix) :].lstrip()
        return text

    @classmethod
    def hero_cpu(cls, chip: str) -> str:
        text = cls.short_chip(chip)
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
        return re.sub(r"\s+\d+-Core\b", "", text, flags=re.IGNORECASE).strip()

    @classmethod
    def _bracketed_gpu(cls, text: str) -> str:
        if "[" not in text:
            return ""
        if not text.endswith("]"):
            return ""
        inner = text[text.rfind("[") + 1 : -1].strip()
        if not inner:
            return ""
        return cls._vendor_prefixed_gpu(text, inner)

    @classmethod
    def _vendor_prefixed_gpu(cls, text: str, inner: str) -> str:
        vendor = "AMD " if text.upper().startswith("AMD") else ""
        if inner.upper().startswith(vendor.strip().upper()):
            return inner
        return f"{vendor}{inner}".strip()


class _OverviewPage:
    @classmethod
    def render(cls, status: AdminStatusResponse, _pairing_html: str = "") -> str:
        ready = (
            status.setup.token_configured
            and status.setup.ffmpeg_available
            and status.setup.engine_ready
        )
        onboarding = cls._onboarding_context(status.system.os)
        return render(
            "overview/page.html",
            ready=ready,
            hero_copy=cls._hero_copy(ready),
            operations_html=operations_fragment(status.metrics, status.readiness),
            system_panel_html=_SystemPanel.build(
                status.system, status.version, onboarding["machine_label"], status.commit
            ),
            dependencies_html=cls._dependencies_panel(status.dependencies),
            onboarding_html=cls._onboarding_steps(status, ready=ready, **onboarding),
        )

    @classmethod
    def commit_context(cls, commit: CommitStatus) -> dict[str, str]:
        stripped = commit.subject.strip()
        if len(stripped) <= MAXIMUM_COMMIT_SUBJECT_LENGTH:
            subject = stripped
        else:
            subject_prefix = stripped[: MAXIMUM_COMMIT_SUBJECT_LENGTH - 1].rstrip()
            subject = f"{subject_prefix}…"
        text = f"{commit.short_sha} · {subject}" if subject else commit.short_sha
        if commit.committed_at is not None:
            dated = format(commit.committed_at, "%Y-%m-%d")
            text = f"{text} ({dated})"
        return {"text": text, "sha": commit.sha}

    @classmethod
    def _hero_copy(cls, ready: bool) -> str:
        if ready:
            return "Pair a phone or run a quick test when you want."
        return "Finish the steps below, then pair your phone."

    @classmethod
    def _onboarding_context(cls, os_name: str) -> dict[str, str]:
        is_mac = os_name.startswith("Darwin")
        ffmpeg_hint = (
            "brew install ffmpeg" if is_mac else "Install FFmpeg with your package manager"
        )
        engine_hint = (
            "brew install whisper-cpp or whisperkit-cli"
            if is_mac
            else "Install vocagateway[engines] (sherpa-onnx / faster-whisper) or use Docker"
        )
        return {
            "machine_label": "Mac" if is_mac else "server",
            "ffmpeg_hint": ffmpeg_hint,
            "engine_hint": engine_hint,
        }

    @classmethod
    def _dependencies_panel(cls, dependencies: list[DependencyStatus]) -> str:
        ordered = sorted(
            dependencies, key=lambda dependency: (not dependency.available, dependency.name.lower())
        )
        installed = sum(1 for dependency in dependencies if dependency.available)
        tiles = [
            {
                "name": dependency.name,
                "available": dependency.available,
                DETAIL_KEY: dependency.path or dependency.install_hint or "—",
            }
            for dependency in ordered
        ]
        return render(
            "overview/dependencies_panel.html",
            installed=installed,
            total=len(dependencies),
            tiles=tiles,
        )

    @classmethod
    def _onboarding_steps(
        cls,
        status: AdminStatusResponse,
        machine_label: str,
        ffmpeg_hint: str,
        engine_hint: str,
        ready: bool,
    ) -> str:
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


class _SystemPanel:
    @classmethod
    def build(
        cls,
        system: SystemStatus,
        version: str,
        machine_label: str,
        commit: CommitStatus | None = None,
    ) -> str:
        gpus = [accelerator for accelerator in system.accelerators if accelerator != "CPU"]
        chip = _HardwareCopy.short_chip(system.chip)
        specs, gpu_label, gpu_vendor = cls._specification_context(system, gpus, chip)
        return render(
            "overview/system_panel.html",
            machine_label=machine_label,
            headline=_HardwareLabels.headline(system, gpus),
            specs=specs,
            meta_line=cls._meta_line(
                "Container" if system.containerized else "Host", system, version, commit
            ),
            chip=chip,
            cpu_vendor=_HardwareLabels.vendor(system.chip),
            cpu_detail=cls._cpu_detail(system),
            ram_label=f"{format(system.ram_gb, GENERAL_FORMAT)} GB",
            has_gpu=bool(gpus),
            primary_gpu_label=gpu_label,
            primary_gpu_vendor=gpu_vendor,
            secondary_gpus=[
                (_HardwareCopy.short_gpu(name), _HardwareLabels.vendor(name)) for name in gpus[1:]
            ],
            feature_line=", ".join(system.cpu_features or []) or "standard",
            os_arch=f"{system.os} ({system.arch})",
            runtime="Container" if system.containerized else "Host",
            version=version,
            commit=cls._commit_context(commit),
        )

    @classmethod
    def _cpu_text(cls, system: SystemStatus, chip: str) -> str:
        cores = system.effective_cpus if system.effective_cpus else system.logical_cpus
        core_label = cls._core_label(cores)
        cpu_name = _HardwareCopy.hero_cpu(chip)
        if cpu_name and core_label:
            return f"{cpu_name} · {core_label} cores"
        if cpu_name:
            return cpu_name
        if core_label:
            return f"{core_label} CPU"
        return chip or "CPU only"

    @classmethod
    def _commit_context(cls, commit: CommitStatus | None) -> dict[str, str] | None:
        if commit is None:
            return None
        return _OverviewPage.commit_context(commit)

    @classmethod
    def _cpu_detail(cls, system: SystemStatus) -> str:
        effective = format(system.effective_cpus, GENERAL_FORMAT)
        return f"{effective} effective · {system.logical_cpus} logical"

    @classmethod
    def _core_label(cls, cores: float | int) -> str:
        if not cores:
            return ""
        if isinstance(cores, float) and cores != int(cores):
            return format(cores, GENERAL_FORMAT)
        return str(int(cores))

    @classmethod
    def _meta_line(
        cls, runtime: str, system: SystemStatus, version: str, commit: CommitStatus | None
    ) -> str:
        os_summary = _HardwareLabels.os_summary(system.os)
        meta_line = f"{runtime} · {os_summary} · gateway {version}"
        if commit is None:
            return meta_line
        return f"{meta_line} · build {commit.short_sha}"

    @classmethod
    def _specs(cls, system: SystemStatus, gpus: list[str], cpu_text: str) -> SystemSpecsContext:
        specs: list[SystemSpec] = []
        primary_gpu_label = None
        primary_gpu_vendor = None
        if gpus:
            primary_gpu_label = _HardwareCopy.short_gpu(gpus[0])
            primary_gpu_vendor = _HardwareLabels.vendor(gpus[0])
            specs.append((_HardwareCopy.hero_gpu(primary_gpu_label), primary_gpu_vendor, "gpu"))
        if system.ram_gb:
            ram_text = f"{format(system.ram_gb, GENERAL_FORMAT)} GB RAM"
            specs.append((ram_text, None, "ram"))
        specs.append((cpu_text, _HardwareLabels.vendor(system.chip), "cpu"))
        return specs, primary_gpu_label, primary_gpu_vendor

    @classmethod
    def _specification_context(
        cls, system: SystemStatus, gpus: list[str], chip: str
    ) -> SystemSpecsContext:
        return cls._specs(system, gpus, cls._cpu_text(system, chip))


class _MetricCards:
    @classmethod
    def page(cls, metrics: OperationalMetricsStatus, readiness: ReadinessStatus) -> str:
        metric_cards, rtf_value, rtf_detail = cls.build(metrics, readiness)
        history = metrics.history
        sample_note = "Collecting samples"
        if history:
            plural = "" if len(history) == 1 else "s"
            sample_note = f"{len(history)} sample{plural}"
        return render(
            "overview/operations.html",
            probe_age=format(readiness.probe_age_seconds, ONE_DECIMAL),
            sample_note=sample_note,
            metric_cards=metric_cards,
            capacity=_Charts.capacity(
                metrics.active_transcriptions, metrics.concurrency_limit, metrics.queue_depth
            ),
            latency_chart=_Charts.latency(metrics),
            outcomes_chart=_Charts.outcomes(metrics),
            pipeline_chart=_Charts.pipeline(metrics, rtf_value, rtf_detail),
        )

    @classmethod
    def build(
        cls, metrics: OperationalMetricsStatus, readiness: ReadinessStatus
    ) -> tuple[list[MetricCard], str, str]:
        warmup = cls._warmup(readiness)
        rtf = cls._rtf(metrics)
        return cls._card_rows(metrics, readiness, warmup, rtf), rtf[0], rtf[1]

    @classmethod
    def _card_rows(
        cls,
        metrics: OperationalMetricsStatus,
        readiness: ReadinessStatus,
        warmup: tuple[str, str],
        rtf: tuple[str, str],
    ) -> list[MetricCard]:
        queued = metrics.queue_depth
        active = metrics.active_transcriptions
        limit = max(1, metrics.concurrency_limit)
        failed_css = "failure" if metrics.failed_transcriptions else ""
        avg_text = cls._average_latency(metrics.average_latency_ms)
        latency_detail = cls._latency_detail(metrics.last_latency_ms)
        rejected = metrics.rejected_transcriptions
        failed_detail = "Since this process started"
        if rejected:
            failed_detail = "{} overload rejection{}".format(rejected, "" if rejected == 1 else "s")
        return [
            cls._card("Uptime", _format_uptime(metrics.uptime_seconds), "This process", ""),
            cls._card(
                "Succeeded",
                str(metrics.successful_transcriptions),
                "Completed transcriptions",
                "success",
            ),
            cls._card("Failed", str(metrics.failed_transcriptions), failed_detail, failed_css),
            cls._card("Avg latency", avg_text, latency_detail, ""),
            cls._card(
                "Last inference",
                _format_latency(metrics.inference_ms),
                cls._inference_detail(metrics),
                "",
            ),
            cls._card("Real-time factor", rtf[0], rtf[1], ""),
            cls._card("Model cache", warmup[0], warmup[1], readiness.warmup_state),
            cls._card("Workload", f"{queued} queued", f"{active} of {limit} active", ""),
        ]

    @classmethod
    def _card(cls, label: str, display: str, detail: str, css: str) -> MetricCard:
        return {LABEL_KEY: label, VALUE_KEY: display, DETAIL_KEY: detail, CSS_KEY: css}

    @classmethod
    def _average_latency(cls, milliseconds: int | None) -> str:
        if milliseconds is None:
            return "—"
        return _format_latency(milliseconds)

    @classmethod
    def _latency_detail(cls, milliseconds: int | None) -> str:
        if milliseconds is None:
            return "Run a test to measure"
        return f"Last {_format_latency(milliseconds)}"

    @classmethod
    def _warmup(cls, readiness: ReadinessStatus) -> tuple[str, str]:
        warmup_labels = {
            "pending": ("Pending", "Startup warm-up has not run yet."),
            "warming": ("Warming", "Priming the selected model."),
            "complete": ("Warm", f"{_format_bytes(readiness.warmed_bytes)} of model data ready."),
            "unsupported": ("Ready", "This engine does not prefetch models."),
            "unavailable": ("Waiting", "Install or select a model to warm it."),
            "failed": ("Needs retry", "Warm-up failed; transcription can still retry."),
        }
        return warmup_labels[readiness.warmup_state]

    @classmethod
    def _rtf(cls, metrics: OperationalMetricsStatus) -> tuple[str, str]:
        rtf_value = "—"
        if metrics.real_time_factor is not None:
            rtf_value = f"{format(metrics.real_time_factor, TWO_DECIMALS)}×"
        bits: list[str] = []
        if metrics.audio_duration_ms is not None:
            bits.append(f"{_format_latency(metrics.audio_duration_ms)} audio")
        if metrics.peak_memory_mb is not None:
            bits.append(f"{format(metrics.peak_memory_mb, GENERAL_FORMAT)} MB peak")
        rtf_detail = " · ".join(bits) if bits else "After first job"
        return rtf_value, rtf_detail

    @classmethod
    def _inference_detail(cls, metrics: OperationalMetricsStatus) -> str:
        if (
            metrics.inference_ms is None
            and metrics.normalization_ms is None
            and metrics.model_load_ms is None
        ):
            return "Stages after first job"
        normalize = _format_latency(metrics.normalization_ms)
        load = _format_latency(metrics.model_load_ms)
        return f"Normalize {normalize} · load {load}"


class _Charts:
    @classmethod
    def capacity(cls, active: int, limit: int, queued: int) -> Chart:
        active_c = min(active, limit)
        active_pct = 100 * active_c / limit
        queue_pct = min(100 - active_pct, 100 * queued / limit) if queued else 0
        free_pct = max(0, 100 - active_pct - queue_pct)
        return {
            "active": active_c,
            "limit": limit,
            "queued": queued,
            "active_pct": format(active_pct, TWO_DECIMALS),
            "queue_pct": format(queue_pct, TWO_DECIMALS),
            "free_pct": format(free_pct, TWO_DECIMALS),
        }

    @classmethod
    def latency(cls, metrics: OperationalMetricsStatus) -> Chart:
        series, _throughput = cls._history_series(metrics)
        avg = _format_latency(metrics.average_latency_ms)
        last = _format_latency(metrics.last_latency_ms)
        if metrics.last_latency_ms is None and metrics.average_latency_ms is None:
            sub = "Last ~5 min"
            empty = "Run a transcription to plot latency."
        else:
            sub = f"Avg {avg} · last {last}"
            empty = sub
        return _LineSvg.sparkline_card("Latency", sub, series, empty=empty, unit_suffix="ms")

    @classmethod
    def outcomes(cls, metrics: OperationalMetricsStatus) -> Chart:
        ok = metrics.successful_transcriptions
        bad = metrics.failed_transcriptions
        bars = cls._outcome_deltas(metrics)
        recent_ok = sum(first for first, _ in bars)
        recent_bad = sum(second for _, second in bars)
        footer = "Idle — no completed jobs in this window."
        if recent_ok or recent_bad:
            footer = f"Window {recent_ok} ok · {recent_bad} failed"
        return {
            TITLE_KEY: "Outcomes",
            SUB_KEY: f"{ok} succeeded · {bad} failed",
            BODY_KEY: _BarSvg.stacked_bars(bars),
            FOOTER_KEY: footer,
            BAR_BODY_KEY: False,
        }

    @classmethod
    def pipeline(cls, metrics: OperationalMetricsStatus, rtf_value: str, rtf_detail: str) -> Chart:
        stages: list[Stage] = [
            ("Normalize", metrics.normalization_ms or 0, "ops-bar-normalize"),
            ("Load", metrics.model_load_ms or 0, "ops-bar-load"),
            ("Infer", metrics.inference_ms or 0, "ops-bar-infer"),
        ]
        total = sum(ms for _, ms, _ in stages)
        if total <= 0:
            return {
                TITLE_KEY: "Last job stages",
                SUB_KEY: "Normalize · load · infer",
                BODY_KEY: Markup('<div class="ops-bar ops-bar-empty" aria-hidden="true"></div>'),
                FOOTER_KEY: "Run a test to measure stages",
                BAR_BODY_KEY: True,
            }
        return cls._pipeline_filled(stages, total, rtf_value, rtf_detail)

    @classmethod
    def _pipeline_filled(
        cls,
        stages: list[Stage],
        total: int,
        rtf_value: str,
        rtf_detail: str,
    ) -> Chart:
        segs = "".join(cls._pipeline_segment(stage, total) for stage in stages if stage[1] > 0)
        chart_markup = Markup(cls._pipeline_markup(total, segs))
        footer = " · ".join(f"{label} {_format_latency(ms)}" for label, ms, _ in stages if ms > 0)
        sub = f"RTF {rtf_value}"
        if rtf_detail:
            sub = f"{sub} · {rtf_detail}"
        return {
            TITLE_KEY: "Last job stages",
            SUB_KEY: sub,
            BODY_KEY: chart_markup,
            FOOTER_KEY: footer,
            BAR_BODY_KEY: True,
        }

    @classmethod
    def _pipeline_segment(cls, stage: Stage, total: int) -> str:
        label, milliseconds, css = stage
        width = format(100 * milliseconds / total, TWO_DECIMALS)
        latency = _format_latency(milliseconds)
        return (
            f'<span class="ops-bar-seg {css}" style="width:{width}%" '
            f'title="{escape(label)}: {latency}"></span>'
        )

    @classmethod
    def _pipeline_markup(cls, total: int, segments: str) -> str:
        label = _format_latency(total)
        return (
            f'<div class="ops-bar" role="img" aria-label="Last pipeline {label}">{segments}</div>'
        )

    @classmethod
    def _history_series(
        cls, metrics: OperationalMetricsStatus
    ) -> tuple[list[float | None], list[float]]:
        latency_series: list[float | None] = []
        last_known: float | None = None
        for point in metrics.history:
            if point.last_latency_ms is not None:
                last_known = float(point.last_latency_ms)
            latency_series.append(last_known)
        throughput_series: list[float] = []
        prev_done = 0
        for point in metrics.history:
            done = point.successful_transcriptions + point.failed_transcriptions
            throughput_series.append(float(max(0, done - prev_done)))
            prev_done = done
        return latency_series, throughput_series

    @classmethod
    def _outcome_deltas(cls, metrics: OperationalMetricsStatus) -> list[tuple[int, int]]:
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


class _LineSvg:
    @classmethod
    def sparkline_card(
        cls,
        title: str,
        subtitle: str,
        samples: Samples,
        *,
        empty: str,
        unit_suffix: str = "",
    ) -> Chart:
        numeric = [float(entry_value) for entry_value in samples if entry_value is not None]
        footer = empty
        if len(numeric) >= 2:
            footer = cls._sparkline_footer(numeric, unit_suffix)
        return {
            TITLE_KEY: title,
            SUB_KEY: subtitle,
            BODY_KEY: cls.draw(samples),
            FOOTER_KEY: footer,
            BAR_BODY_KEY: False,
        }

    @classmethod
    def draw(cls, samples: Samples, *, width: int = 220, height: int = 48) -> Markup:
        numeric = [
            (index, float(entry_value))
            for index, entry_value in enumerate(samples)
            if entry_value is not None
        ]
        if len(numeric) < 2:
            return _BarSvg.empty_svg(width, height)
        coords = cls._spark_coords(samples, numeric, width, height)
        points = " ".join(
            f"{format(coordinate_x, ONE_DECIMAL)},{format(coordinate_y, ONE_DECIMAL)}"
            for coordinate_x, coordinate_y in coords
        )
        area = cls._spark_area(coords, height)
        label = f"Sparkline with {len(coords)} points"
        return Markup(
            f'<svg class="sparkline" viewBox="0 0 {width} {height}" width="100%" height="{height}" '
            f'preserveAspectRatio="none" role="img" aria-label="{label}">'
            f'<path class="sparkline-fill" d="{area}"/>'
            f'<polyline class="sparkline-line" fill="none" points="{points}"/>'
            "</svg>"
        )

    @classmethod
    def _sparkline_footer(cls, numeric: list[float], unit_suffix: str) -> str:
        latest = numeric[-1]
        peak = max(numeric)
        if unit_suffix == "ms":
            latest_label = _format_latency(int(round(latest)))
            peak_label = _format_latency(int(round(peak)))
        elif latest == int(latest):
            latest_label = str(int(latest))
            peak_label = format(peak, GENERAL_FORMAT)
        else:
            latest_label = format(latest, TWO_DECIMALS)
            peak_label = format(peak, GENERAL_FORMAT)
        return f"Now {latest_label} · peak {peak_label}"

    @classmethod
    def _spark_coords(
        cls,
        samples: Samples,
        numeric: list[tuple[int, float]],
        width: int,
        height: int,
    ) -> list[tuple[float, float]]:
        ys = [coordinate_y for _, coordinate_y in numeric]
        y_min = min(ys)
        y_max = max(ys)
        if y_max <= y_min:
            y_max = y_min + 1
        pad = 3.0
        usable_h = height - pad * 2
        span = max(1, len(samples) - 1)
        coords: list[tuple[float, float]] = []
        for index, raw in enumerate(samples):
            if raw is None:
                continue
            coordinate_x = (index / span) * width
            coordinate_y = pad + (1 - (float(raw) - y_min) / (y_max - y_min)) * usable_h
            coords.append((coordinate_x, coordinate_y))
        return coords

    @classmethod
    def _spark_area(cls, coords: list[tuple[float, float]], height: int) -> str:
        head_x = format(coords[0][0], ONE_DECIMAL)
        tail_x = format(coords[-1][0], ONE_DECIMAL)
        line = " ".join(
            f"L{format(coordinate_x, ONE_DECIMAL)},{format(coordinate_y, ONE_DECIMAL)}"
            for coordinate_x, coordinate_y in coords
        )
        return f"M{head_x},{height} {line} L{tail_x},{height} Z"


class _BarSvg:
    @classmethod
    def stacked_bars(
        cls, bars: list[tuple[int, int]], *, width: int = 220, height: int = 48
    ) -> Markup:
        if not bars or not any(ok or bad for ok, bad in bars):
            return cls.empty_svg(width, height)
        return cls._filled_bars(bars, width, height)

    @classmethod
    def empty_svg(cls, width: int, height: int) -> Markup:
        mid = height // 2
        return Markup(
            f'<svg class="sparkline sparkline-empty" viewBox="0 0 {width} {height}" '
            f'width="100%" height="{height}" preserveAspectRatio="none" aria-hidden="true">'
            f'<line class="sparkline-baseline" x1="0" y1="{mid}" x2="{width}" y2="{mid}"/></svg>'
        )

    @classmethod
    def _filled_bars(cls, bars: list[tuple[int, int]], width: int, height: int) -> Markup:
        count = len(bars)
        gap = 1.5
        bar_w = max(1.0, (width - gap * (count - 1)) / count)
        peak = max(1, max(ok + bad for ok, bad in bars))
        parts = cls._bar_rects(bars, bar_w, gap, peak, height)
        return Markup(
            '<svg class="sparkline ops-bars" viewBox="0 0 {0} {1}" width="100%" '
            'height="{1}" preserveAspectRatio="none" role="img" '
            'aria-label="Outcomes over {2} samples">{3}</svg>'.format(
                width, height, count, "".join(parts)
            )
        )

    @classmethod
    def _bar_rects(
        cls, bars: list[tuple[int, int]], bar_w: float, gap: float, peak: int, height: int
    ) -> list[str]:
        parts: list[str] = []
        pad_top = 2.0
        usable = height - pad_top
        for index, (ok, bad) in enumerate(bars):
            total = ok + bad
            if total <= 0:
                continue
            parts.extend(cls._bar_pair(index, (ok, bad), (bar_w, gap, peak, usable, height)))
        return parts

    @classmethod
    def _bar_pair(
        cls,
        index: int,
        counts: tuple[int, int],
        layout: tuple[float, float, int, float, int],
    ) -> list[str]:
        ok, bad = counts
        bar_w, gap, peak, usable, height = layout
        coordinate_x = index * (bar_w + gap)
        total_h = usable * ((ok + bad) / peak)
        bad_h = usable * (bad / peak) if bad else 0
        ok_h = total_h - bad_h
        rects: list[str] = []
        if ok_h > 0:
            rects.append(cls._bar_rect("ops-bar-ok", coordinate_x, height - total_h, bar_w, ok_h))
        if bad_h > 0:
            rects.append(cls._bar_rect("ops-bar-bad", coordinate_x, height - bad_h, bar_w, bad_h))
        return rects

    @classmethod
    def _bar_rect(
        cls, css: str, coordinate_x: float, coordinate_y: float, width: float, height: float
    ) -> str:
        attributes = (coordinate_x, coordinate_y, width, height)
        labels = tuple(format(number, ONE_DECIMAL) for number in attributes)
        return '<rect class="{}" x="{}" y="{}" width="{}" height="{}" rx="1"/>'.format(css, *labels)


_commit_context = _OverviewPage.commit_context
overview_fragment = _OverviewPage.render
operations_fragment = _MetricCards.page
