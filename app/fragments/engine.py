from __future__ import annotations

from html import escape

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
    "auto": "Uses the fastest compatible installed local engine for this machine.",
    "vocamac": (
        "Optional Apple silicon Mac app. Reuses VocaMac's downloaded Core ML "
        "models through whisperkit-cli. No download needed."
    ),
    "handy": (
        "Optional macOS app. Reuses the Handy app and its downloaded models. No download needed."
    ),
    "whisper.cpp": "Runs local GGML models with the whisper-cli binary.",
    "whisperkit": "Runs Core ML models with whisperkit-cli on Apple Silicon Macs.",
    "faster-whisper": "Keeps a CTranslate2 model loaded; CPU INT8 is the Linux default.",
    "moonshine": "Fast, language-specific local models; compatible English tiers stream live.",
    "sherpa-onnx": "Compact INT8 CPU models for fast macOS and Linux transcription.",
    "mlx-audio": "Runs Apple-silicon-native MLX models with persistent loading.",
}


def _engine_option_label(engine: str) -> str:
    """Name the host an engine needs, so the picker explains its own contents."""
    label = ENGINE_LABELS.get(engine, engine)
    requirement = engine_requirement(engine)
    return f"{label} ({requirement} only)" if requirement else label


def _engine_status(engine: EngineStatus, *, oob: bool = False) -> str:
    """Header control: status of the active model; opens Models when clicked."""
    classes = "engine-status ready" if engine.ready else "engine-status"
    dot = "ok" if engine.ready else "warn"
    label = escape(engine.name or engine.id)
    swap_oob = ' hx-swap-oob="true"' if oob else ""
    # A button so keyboard users can open Models; hx-get still refreshes the pill.
    return (
        f'<button type="button" id="engine-pill" class="{classes}"{swap_oob}'
        f' data-open-tab="models"'
        f' title="Open Models to change the speech engine"'
        f' aria-label="Current speech model: {label}. Open Models."'
        f' hx-get="/ui/partials/engine-pill" hx-trigger="every 5s" hx-swap="outerHTML">'
        f'<span class="dot {dot}" aria-hidden="true"></span>'
        f"<span>{label}</span>"
        f'<span class="engine-status-hint" aria-hidden="true">Models</span></button>'
    )


def engine_pill_fragment(engine: EngineStatus) -> str:
    return _engine_status(engine)


def engine_pill_oob(engine: EngineStatus) -> str:
    return _engine_status(engine, oob=True)


def engine_update_fragment(engine: EngineStatus, message: str) -> str:
    css = "ok" if engine.ready else "missing"
    return (
        f'<span class="badge {css}">{escape(engine.name or engine.id)}'
        f"{' ready' if engine.ready else ' not ready'}</span> "
        f"{escape(message)}{engine_pill_oob(engine)}"
    )
