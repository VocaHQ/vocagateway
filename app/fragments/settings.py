from __future__ import annotations

from app.config import format_host_port, local_webui_url
from app.fragments.engine import ENGINE_HINTS, _engine_option_label
from app.runtime_config import IDLE_OFFLOAD_MINUTES
from app.schemas import ConfigResponse
from app.templating import render


def settings_fragment(
    config: ConfigResponse,
    paths: list[tuple[str, str]],
    bind_host: str,
    port: int,
    tokens_html: str,
) -> str:
    return render(
        "settings/page.html",
        config=config,
        engine_options=[
            (engine_id, _engine_option_label(engine_id)) for engine_id in config.available_engines
        ],
        engine_hint=ENGINE_HINTS.get(config.engine, ""),
        paths=paths,
        listener=format_host_port(bind_host, port),
        local_url=local_webui_url(bind_host, port),
        device_options=[("auto", "Auto"), ("cpu", "CPU"), ("cuda", "NVIDIA CUDA")],
        precision_options=[
            ("auto", "Auto (INT8 CPU / FP16 CUDA)"),
            ("int8", "INT8"),
            ("int8_float16", "INT8 + FP16"),
            ("float16", "FP16"),
            ("float32", "FP32"),
        ],
        idle_offload_options=[
            (minutes, _idle_duration_label(minutes)) for minutes in IDLE_OFFLOAD_MINUTES
        ],
        tokens_html=tokens_html,
    )


def _idle_duration_label(minutes: int) -> str:
    if minutes < 60:
        return f"{minutes} minutes"
    hours = minutes // 60
    return f"{hours} hour" if hours == 1 else f"{hours} hours"
