from __future__ import annotations

from html import escape

from app.config import format_host_port, local_webui_url
from app.schemas import EngineStatus
from app.system import engine_requirement

ENGINE_LABELS = {
    "auto": "Auto (recommended)",
    "vocamac": "VocaMac app",
    "handy": "Handy app",
    "whisper.cpp": "whisper.cpp",
    "whisperkit": "WhisperKit",
    "faster-whisper": "faster-whisper",
    "moonshine": "Moonshine",
    "sherpa-onnx": "sherpa-onnx",
    "mlx-audio": "MLX Audio",
}

ENGINE_HINTS = {
    "auto": "Picks the fastest compatible local engine already installed on this machine.",
    "vocamac": (
        "Optional Apple silicon Mac app. Follows VocaMac's selected downloaded "
        "model through its headless interface. No second download needed."
    ),
    "handy": (
        "Optional macOS app. Reuses the Handy app and its downloaded models. No separate download."
    ),
    "whisper.cpp": "Local GGML models via the whisper-cli binary.",
    "whisperkit": "Core ML models via whisperkit-cli on Apple silicon Macs.",
    "faster-whisper": "Keeps a CTranslate2 model loaded. CPU INT8 is the usual Linux default.",
    "moonshine": "Fast language-specific models. Compatible English tiers can stream live.",
    "sherpa-onnx": "Compact INT8 CPU models for fast macOS and Linux transcription.",
    "mlx-audio": "Apple-silicon MLX models with persistent loading.",
}


def _engine_option_label(engine: str) -> str:
    """Name the host an engine needs, so the picker explains its own contents."""
    label = ENGINE_LABELS.get(engine, engine)
    requirement = engine_requirement(engine)
    return f"{label} ({requirement} only)" if requirement else label


def _engine_status(
    engine: EngineStatus,
    *,
    bind_host: str = "0.0.0.0",
    port: int = 8765,
    oob: bool = False,
) -> str:
    """Header control: active model, network details on hover, opens Models on click."""
    classes = "engine-status ready" if engine.ready else "engine-status"
    dot = "ok" if engine.ready else "warn"
    label = escape(engine.name or engine.id)
    listener = escape(format_host_port(bind_host, port))
    local_url = escape(local_webui_url(bind_host, port))
    ready_label = "Ready" if engine.ready else "Not ready"
    swap_oob = ' hx-swap-oob="true"' if oob else ""
    # Network addresses live in the hover card so Overview stays uncluttered.
    return (
        f'<button type="button" id="engine-pill" class="{classes}"{swap_oob}'
        f' data-open-tab="models"'
        f' aria-label="Speech model {label}, {ready_label}. '
        f'Listener {listener}. WebUI {local_url}. Opens Models."'
        f' hx-get="/ui/partials/engine-pill" hx-trigger="every 5s" hx-swap="outerHTML">'
        f'<span class="dot {dot}" aria-hidden="true"></span>'
        f"<span>{label}</span>"
        f'<span class="engine-status-hint" aria-hidden="true">Models</span>'
        f'<span class="engine-popover" role="tooltip">'
        f'<span class="engine-popover-card">'
        f'<span class="engine-popover-title">{ready_label}</span>'
        f'<span class="engine-popover-row"><span>Listener</span>'
        f"<code>{listener}</code></span>"
        f'<span class="engine-popover-row"><span>This host</span>'
        f"<code>{local_url}</code></span>"
        f'<span class="engine-popover-hint">Click to open Models</span>'
        f"</span></span></button>"
    )


def engine_pill_fragment(
    engine: EngineStatus, *, bind_host: str = "0.0.0.0", port: int = 8765
) -> str:
    return _engine_status(engine, bind_host=bind_host, port=port)


def engine_pill_oob(engine: EngineStatus, *, bind_host: str = "0.0.0.0", port: int = 8765) -> str:
    return _engine_status(engine, bind_host=bind_host, port=port, oob=True)


def engine_update_fragment(
    engine: EngineStatus,
    message: str,
    *,
    bind_host: str = "0.0.0.0",
    port: int = 8765,
) -> str:
    css = "ok" if engine.ready else "missing"
    return (
        f'<span class="badge {css}">{escape(engine.name or engine.id)}'
        f"{' ready' if engine.ready else ' not ready'}</span> "
        f"{escape(message)}"
        f"{engine_pill_oob(engine, bind_host=bind_host, port=port)}"
    )
