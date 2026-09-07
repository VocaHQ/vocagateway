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

# Transcript cleanup. Kept in its own block, saved through its own partial
# update, and never touched by an engine or hardware save: changing whether
# transcripts are corrected must not reset which model transcribes them.
CLEANUP_OFF = "off"
CLEANUP_CONSERVATIVE = "conservative"
CLEANUP_MODES = (CLEANUP_OFF, CLEANUP_CONSERVATIVE)
# Cleanup is part of the shipped deployment rather than a feature to discover:
# the container builds its runtime, and a native install that has one gets the
# same behaviour. On by default costs nothing until a cleanup model is
# downloaded — with no model there is nothing to run, and every transcript takes
# the same path it would with the feature off. Downloading one is the deliberate
# act that starts corrections, and the WebUI toggle turns them off again.
DEFAULT_CLEANUP_ENABLED = True
DEFAULT_CLEANUP_MODE = CLEANUP_CONSERVATIVE
DEFAULT_CLEANUP_TIMEOUT_SECONDS = 5.0
MINIMUM_CLEANUP_TIMEOUT_SECONDS = 1.0
MAXIMUM_CLEANUP_TIMEOUT_SECONDS = 30.0
CLEANUP_IDLE_UNLOAD_MINUTES = (5, 15, 30, 60, 120)
DEFAULT_CLEANUP_IDLE_UNLOAD_MINUTES = 15


@dataclass(slots=True)
class RuntimeConfig:
    """User choices made through the WebUI, persisted across restarts."""

    engine: str = AUTO_ENGINE
    whisper_model: str | None = None
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
    cleanup_enabled: bool = DEFAULT_CLEANUP_ENABLED
    cleanup_mode: str = DEFAULT_CLEANUP_MODE
    cleanup_model: str | None = None
    cleanup_timeout_seconds: float = DEFAULT_CLEANUP_TIMEOUT_SECONDS
    cleanup_idle_unload_enabled: bool = False
    cleanup_idle_unload_minutes: int = DEFAULT_CLEANUP_IDLE_UNLOAD_MINUTES
    # Which language a transcript left on `auto` is corrected as. Empty means
    # "do not guess", which is the default: `auto` then falls back unless the
    # writing system names a language on its own.
    cleanup_auto_language: str = ""
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
        fields.update(_parse_cleanup_fields(payload))
        return cls(**fields)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            ENGINE_FIELD: self.engine,
            "whisper_model": self.whisper_model,
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
            "cleanup_enabled": self.cleanup_enabled,
            "cleanup_mode": self.cleanup_mode,
            "cleanup_model": self.cleanup_model,
            "cleanup_timeout_seconds": self.cleanup_timeout_seconds,
            "cleanup_idle_unload_enabled": self.cleanup_idle_unload_enabled,
            "cleanup_idle_unload_minutes": self.cleanup_idle_unload_minutes,
            "cleanup_auto_language": self.cleanup_auto_language,
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


def _parse_cleanup_fields(payload: dict[str, Any]) -> dict[str, Any]:
    """Read the cleanup block, defaulting anything unrecognised to the shipped value.

    A config file written by a newer build, hand-edited, or truncated falls back
    to what this build ships rather than to whatever happens to be in the file.
    Only a real `false` turns cleanup off, so an operator who unticked it keeps
    it unticked; a missing key is not a decision and inherits the default.
    """
    mode = payload.get("cleanup_mode")
    idle_minutes = payload.get("cleanup_idle_unload_minutes")
    enabled = payload.get("cleanup_enabled")
    return {
        "cleanup_enabled": DEFAULT_CLEANUP_ENABLED if enabled is None else enabled is True,
        "cleanup_mode": mode if mode in CLEANUP_MODES else DEFAULT_CLEANUP_MODE,
        "cleanup_model": _optional_str(payload.get("cleanup_model")),
        "cleanup_timeout_seconds": clamp_cleanup_timeout(payload.get("cleanup_timeout_seconds")),
        "cleanup_auto_language": _optional_str(payload.get("cleanup_auto_language")) or "",
        "cleanup_idle_unload_enabled": payload.get("cleanup_idle_unload_enabled") is True,
        "cleanup_idle_unload_minutes": (
            idle_minutes
            if isinstance(idle_minutes, int) and idle_minutes in CLEANUP_IDLE_UNLOAD_MINUTES
            else DEFAULT_CLEANUP_IDLE_UNLOAD_MINUTES
        ),
    }


def clamp_cleanup_timeout(raw: Any) -> float:
    """Hold the total cleanup deadline inside its supported range."""
    if not isinstance(raw, int | float) or isinstance(raw, bool):
        return DEFAULT_CLEANUP_TIMEOUT_SECONDS
    return float(min(MAXIMUM_CLEANUP_TIMEOUT_SECONDS, max(MINIMUM_CLEANUP_TIMEOUT_SECONDS, raw)))


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
