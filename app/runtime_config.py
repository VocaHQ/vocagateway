from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

AUTO_ENGINE = "auto"
VALID_ENGINES = (
    AUTO_ENGINE,
    "vocamac",
    "handy",
    "whisper.cpp",
    "transcribe.cpp",
    "whisperkit",
    "faster-whisper",
    "moonshine",
    "sherpa-onnx",
    "mlx-audio",
)
MAXIMUM_CPU_THREADS = 256
ENGINE_FIELD = "engine"
IDLE_OFFLOAD_MINUTES = (10, 15, 30, 60, 120)
DEFAULT_IDLE_OFFLOAD_MINUTES = 15


@dataclass(slots=True)
class RuntimeConfig:
    """User choices made through the WebUI, persisted across restarts."""

    engine: str = AUTO_ENGINE
    whisper_model: str | None = None
    transcribe_model: str | None = None
    whisperkit_model: str | None = None
    faster_whisper_model: str | None = None
    moonshine_model: str = "moonshine:en"
    moonshine_language: str = "en"
    sherpa_model: str | None = None
    mlx_audio_model: str | None = None
    compute_device: str = AUTO_ENGINE
    compute_type: str = AUTO_ENGINE
    cpu_threads: int = 0
    idle_offload_enabled: bool = False
    idle_offload_minutes: int = DEFAULT_IDLE_OFFLOAD_MINUTES
    pairing_url: str | None = None
    pairing_urls: list[str] = field(default_factory=list)

    @classmethod
    def load(cls, path: Path) -> RuntimeConfig:
        payload = _read_payload(path)
        if payload is None:
            return cls()
        fields = _parse_model_fields(payload)
        fields.update(_parse_hardware_fields(payload))
        fields.update(_parse_memory_fields(payload))
        return cls(**fields)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            ENGINE_FIELD: self.engine,
            "whisper_model": self.whisper_model,
            "transcribe_model": self.transcribe_model,
            "whisperkit_model": self.whisperkit_model,
            "faster_whisper_model": self.faster_whisper_model,
            "moonshine_model": self.moonshine_model,
            "moonshine_language": self.moonshine_language,
            "sherpa_model": self.sherpa_model,
            "mlx_audio_model": self.mlx_audio_model,
            "compute_device": self.compute_device,
            "compute_type": self.compute_type,
            "cpu_threads": self.cpu_threads,
            "idle_offload_enabled": self.idle_offload_enabled,
            "idle_offload_minutes": self.idle_offload_minutes,
            "pairing_url": self.pairing_url,
            "pairing_urls": self.pairing_urls,
        }
        descriptor, temporary_name = tempfile.mkstemp(
            dir=path.parent, prefix=".config-", suffix=".tmp"
        )
        _write_temp_config(descriptor, payload)
        try:
            os.replace(temporary_name, path)
        except BaseException:
            Path(temporary_name).unlink(missing_ok=True)
            raise


def _read_payload(path: Path) -> dict[str, Any] | None:
    try:
        raw_payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return raw_payload if isinstance(raw_payload, dict) else None


def _parse_model_fields(payload: dict[str, Any]) -> dict[str, Any]:
    resolved_engine = (
        payload.get(ENGINE_FIELD) if payload.get(ENGINE_FIELD) in VALID_ENGINES else AUTO_ENGINE
    )
    moon_lang = payload.get("moonshine_language")
    resolved_lang = moon_lang if isinstance(moon_lang, str) else "en"
    default_moonshine = f"moonshine:{resolved_lang}"
    moon_model = payload.get("moonshine_model")
    return {
        ENGINE_FIELD: resolved_engine,
        "whisper_model": _optional_str(payload.get("whisper_model")),
        "transcribe_model": _optional_str(payload.get("transcribe_model")),
        "whisperkit_model": _optional_str(payload.get("whisperkit_model")),
        "faster_whisper_model": _optional_str(payload.get("faster_whisper_model")),
        "moonshine_model": moon_model if isinstance(moon_model, str) else default_moonshine,
        "moonshine_language": resolved_lang,
        "sherpa_model": _optional_str(payload.get("sherpa_model")),
        "mlx_audio_model": _optional_str(payload.get("mlx_audio_model")),
    }


def _parse_hardware_fields(payload: dict[str, Any]) -> dict[str, Any]:
    device = payload.get("compute_device")
    comp_type = payload.get("compute_type")
    threads = payload.get("cpu_threads")
    return {
        "compute_device": device if device in {AUTO_ENGINE, "cpu", "cuda"} else AUTO_ENGINE,
        "compute_type": (
            comp_type
            if comp_type in {AUTO_ENGINE, "int8", "int8_float16", "float16", "float32"}
            else AUTO_ENGINE
        ),
        "cpu_threads": (
            threads if isinstance(threads, int) and 0 <= threads <= MAXIMUM_CPU_THREADS else 0
        ),
        "pairing_url": _optional_str(payload.get("pairing_url")),
        "pairing_urls": _clean_urls(payload.get("pairing_urls")),
    }


def _parse_memory_fields(payload: dict[str, Any]) -> dict[str, Any]:
    idle_minutes = payload.get("idle_offload_minutes")
    return {
        "idle_offload_enabled": payload.get("idle_offload_enabled") is True,
        "idle_offload_minutes": (
            idle_minutes
            if isinstance(idle_minutes, int) and idle_minutes in IDLE_OFFLOAD_MINUTES
            else DEFAULT_IDLE_OFFLOAD_MINUTES
        ),
    }


def _clean_urls(raw_urls: Any) -> list[str]:
    if isinstance(raw_urls, list):
        return [url_item for url_item in raw_urls if isinstance(url_item, str)]
    return []


def _optional_str(candidate: Any) -> str | None:
    return candidate if isinstance(candidate, str) else None


def _write_temp_config(descriptor: int, payload: dict[str, Any]) -> None:
    with os.fdopen(descriptor, "w", encoding="utf-8") as config_file:
        json.dump(payload, config_file, indent=2)
        config_file.write("\n")
