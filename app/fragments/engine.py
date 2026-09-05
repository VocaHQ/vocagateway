from __future__ import annotations

from types import MappingProxyType

from app.config import format_host_port, local_webui_url
from app.schemas import EngineStatus
from app.system import engine_requirement
from app.templating import render

DEFAULT_BIND_HOST = "0.0.0.0"

ENGINE_LABELS = MappingProxyType(
    {
        "auto": "Auto (recommended)",
        "vocamac": "VocaMac app",
        "handy": "Handy app",
        "whisper.cpp": "whisper.cpp",
        "transcribe.cpp": "transcribe.cpp",
        "whisperkit": "WhisperKit",
        "faster-whisper": "faster-whisper",
        "moonshine": "Moonshine",
        "sherpa-onnx": "sherpa-onnx",
        "mlx-audio": "MLX Audio",
    }
)

ENGINE_HINTS = MappingProxyType(
    {
        "auto": "Picks the fastest compatible local engine already installed on this machine.",
        "vocamac": (
            "Optional Apple silicon Mac app. Follows VocaMac's selected downloaded "
            "model through its headless interface. No second download needed."
        ),
        "handy": "Optional macOS app. Reuses the Handy app and its downloaded models. "
        "No separate download.",
        "whisper.cpp": "Local GGML models. Kept resident when the build ships whisper-server.",
        "transcribe.cpp": "GGUF speech models. Requires a separately installed transcribe-cli.",
        "whisperkit": "Core ML models via whisperkit-cli on Apple silicon Macs.",
        "faster-whisper": "Keeps a CTranslate2 model loaded. CPU INT8 is the usual Linux default.",
        "moonshine": "Fast language-specific models. Compatible English tiers can stream live.",
        "sherpa-onnx": "Compact INT8 CPU models for fast macOS and Linux transcription.",
        "mlx-audio": "Apple-silicon MLX models with persistent loading.",
    }
)


def _engine_option_label(engine: str) -> str:
    """Name the host an engine needs, so the picker explains its own contents."""
    label = ENGINE_LABELS.get(engine, engine)
    requirement = engine_requirement(engine)
    return f"{label} ({requirement} only)" if requirement else label


def _engine_status(
    engine: EngineStatus,
    *,
    bind_host: str = DEFAULT_BIND_HOST,
    port: int = 8765,
    oob: bool = False,
) -> str:
    """Header control: active model, network details on hover, opens Models on click."""
    return render(
        "engine/pill.html",
        engine=engine,
        label=engine.name or engine.id,
        listener=format_host_port(bind_host, port),
        local_url=local_webui_url(bind_host, port),
        ready_label="Ready" if engine.ready else "Not ready",
        oob=oob,
    )


def engine_pill_fragment(
    engine: EngineStatus, *, bind_host: str = DEFAULT_BIND_HOST, port: int = 8765
) -> str:
    return _engine_status(engine, bind_host=bind_host, port=port)


def engine_pill_oob(
    engine: EngineStatus, *, bind_host: str = DEFAULT_BIND_HOST, port: int = 8765
) -> str:
    return _engine_status(engine, bind_host=bind_host, port=port, oob=True)


def engine_update_fragment(
    engine: EngineStatus,
    message: str,
    *,
    bind_host: str = DEFAULT_BIND_HOST,
    port: int = 8765,
) -> str:
    return render(
        "engine/update.html",
        engine=engine,
        message=message,
        pill=engine_pill_oob(engine, bind_host=bind_host, port=port),
    )
