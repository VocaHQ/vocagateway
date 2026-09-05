from __future__ import annotations

import asyncio
import inspect
import platform
import time
from importlib import import_module
from importlib import util as importlib_util
from pathlib import Path
from typing import Any

from app import scripts
from app.catalog import CatalogModel
from app.errors import (
    EngineUnavailableError,
    LanguageUnsupportedError,
    TranscriptionProcessError,
)
from app.models.base import EngineHealth, EngineTranscription, TranscriptionOptions

TRANSCRIPTION_TIMEOUT_SECONDS = 180
MAXIMUM_ERROR_MESSAGE_LENGTH = 240


class MLXAudioEngine:
    """Persistent Apple-silicon STT engine backed by MLX Audio."""

    def __init__(self, model_root: Path | None, catalog_model: CatalogModel | None) -> None:
        self.model_root = model_root
        self.catalog_model = catalog_model
        self._model: Any | None = None
        self._load_lock = asyncio.Lock()
        self._inference_lock = asyncio.Lock()

    async def health(self) -> EngineHealth:
        package_ready = importlib_util.find_spec("mlx_audio") is not None
        platform_ready = platform.system() == "Darwin" and platform.machine() == "arm64"
        model_ready = (
            self.model_root is not None and (self.model_root / self._marker_file()).is_file()
        )
        model_name = self.model_root.name if self.model_root else "no-model-selected"
        return EngineHealth(
            ready=package_ready and platform_ready and model_ready,
            name=f"mlx-audio:{model_name}",
        )

    async def warmup(self) -> int:
        if not (await self.health()).ready:
            return 0
        await self._ensure_model()
        return _directory_size(self.model_root) if self.model_root else 0

    @property
    def model_is_resident(self) -> bool:
        return self._model is not None

    def unload(self) -> None:
        self._model = None
        try:
            mlx_core = import_module("mlx.core")
        except ImportError:
            return
        mlx_core.clear_cache()

    async def transcribe(
        self, audio_path: Path, options: TranscriptionOptions
    ) -> EngineTranscription:
        await self._validate_request(options.language)
        async with self._inference_lock:
            model, model_load_ms = await self._ensure_model()
            inference_started = time.monotonic()
            try:
                text = await asyncio.wait_for(
                    asyncio.to_thread(
                        _generate_text,
                        model,
                        audio_path,
                        self._decoder_language(options.language),
                        self._transcription_prompt(),
                    ),
                    timeout=TRANSCRIPTION_TIMEOUT_SECONDS,
                )
            except TimeoutError as error:
                raise TranscriptionProcessError("MLX Audio transcription timed out.") from error
            except Exception as error:
                error_suffix = str(error)[-MAXIMUM_ERROR_MESSAGE_LENGTH:]
                raise TranscriptionProcessError(f"MLX Audio failed: {error_suffix}") from error
            if not text:
                raise TranscriptionProcessError("MLX Audio returned an empty transcript.")
            self._validate_output(text)
            return EngineTranscription(
                text=text,
                model_load_ms=model_load_ms,
                inference_ms=_elapsed_ms(inference_started),
            )

    def _marker_file(self) -> str:
        if self.catalog_model and self.catalog_model.marker_file:
            return self.catalog_model.marker_file
        return "model.safetensors"

    def _decoder_language(self, requested: str) -> str:
        if self.catalog_model and self.catalog_model.decoder_language_code:
            return self.catalog_model.decoder_language_code
        return requested

    def _transcription_prompt(self) -> str | None:
        if self.catalog_model and self.catalog_model.key == "granite-speech-4.1-2b":
            # MLX Granite otherwise turns explicit language hints into translation requests.
            return "transcribe the speech with proper punctuation and capitalization."
        return None

    def _validate_output(self, text: str) -> None:
        model = self.catalog_model
        if (
            model
            and model.decoder_language_code
            and len(model.language_codes) == 1
            and not scripts.transcript_matches_language(text, model.language_codes[0])
        ):
            raise LanguageUnsupportedError(
                f"The model did not produce the required {model.language_codes[0]} writing system."
            )

    async def _ensure_model(self) -> tuple[Any, int]:
        if self._model is not None:
            return self._model, 0
        started = time.monotonic()
        async with self._load_lock:
            if self._model is not None:
                return self._model, 0
            self._model = await asyncio.to_thread(self._load_model_sync)
            return self._model, _elapsed_ms(started)

    def _load_model_sync(self) -> Any:
        if self.model_root is None:
            raise EngineUnavailableError("No MLX Audio model is selected.")
        from mlx_audio.stt import utils as mlx_utils

        return mlx_utils.load(self.model_root)

    async def _validate_request(self, language: str) -> None:
        supported = self.catalog_model.language_codes if self.catalog_model else ()
        normalized = _language_code(language)
        if language != "auto" and supported and normalized not in supported:
            choices = ", ".join(supported)
            raise LanguageUnsupportedError(
                f"The selected MLX model does not support {language}. Choose Auto, {choices}, "
                "or another model."
            )
        if not (await self.health()).ready:
            raise EngineUnavailableError(
                "MLX Audio requires an Apple-silicon Mac, the apple engine extra, and a "
                "downloaded MLX model."
            )


def _generate_text(model: Any, audio_path: Path, language: str, prompt: str | None = None) -> str:
    generate_parameters = inspect.signature(model.generate).parameters
    arguments: dict[str, Any] = {}
    if prompt is not None:
        arguments["prompt"] = prompt
    if language != "auto" and "language" in generate_parameters:
        arguments["language"] = language if language == "None" else _language_code(language)
    generation = model.generate(str(audio_path), **arguments)
    return str(generation.text).strip()


def _language_code(language_tag: str) -> str:
    return language_tag.lower().split("-", maxsplit=1)[0]


def _directory_size(path: Path) -> int:
    return sum(
        nested_path.stat().st_size for nested_path in path.rglob("*") if nested_path.is_file()
    )


def _elapsed_ms(started: float) -> int:
    return max(0, int((time.monotonic() - started) * 1000))
