from __future__ import annotations

import asyncio
import json
import time
import wave
from array import array
from importlib import util as importlib_util
from pathlib import Path
from typing import Any

from app import errors
from app.models.base import EngineHealth, EngineTranscription, TranscriptionOptions

MODEL_METADATA = ".vocagateway-model.json"
STREAMING_ARCHITECTURES = frozenset((2, 3, 4, 5))
TRANSCRIPTION_TIMEOUT_SECONDS = 120
MAXIMUM_ERROR_MESSAGE_LENGTH = 240
PCM_SAMPLE_SCALE = 32_768.0
STREAM_UPDATE_INTERVAL_SECONDS = 0.35
LANGUAGE_ALIASES = (
    ("ar", frozenset(("ar", "ar-SA"))),
    ("en", frozenset(("en", "en-US", "en-GB"))),
    ("es", frozenset(("es", "es-ES", "es-MX"))),
    ("ja", frozenset(("ja", "ja-JP"))),
    ("ko", frozenset(("ko", "ko-KR"))),
    ("uk", frozenset(("uk", "uk-UA"))),
    ("vi", frozenset(("vi", "vi-VN"))),
    ("zh", frozenset(("zh", "zh-CN", "zh-TW", "cmn"))),
)
NON_LATIN_LANGUAGES = frozenset(("ar", "ja", "ko", "zh"))


class MoonshineEngine:
    """Persistent Moonshine engine with both batch and incremental APIs."""

    def __init__(self, model_root: Path | None, language: str = "en") -> None:
        self.model_root = model_root
        self.language = language
        self._transcriber: Any | None = None
        self._load_lock = asyncio.Lock()
        self._inference_lock = asyncio.Lock()
        self._metadata = _read_metadata(model_root)
        self.supports_streaming: bool = self._metadata.get("model_arch") in STREAMING_ARCHITECTURES
        # The package owns mutable decoder state, so batch and streaming jobs
        # must never share the persistent transcriber concurrently.
        self.streaming_lock = self._inference_lock

    async def health(self) -> EngineHealth:
        package_ready = importlib_util.find_spec("moonshine_voice") is not None
        metadata_ready = (
            self.model_root is not None and (self.model_root / MODEL_METADATA).is_file()
        )
        model_name = self.model_root.name if self.model_root else "no-model-selected"
        return EngineHealth(
            ready=package_ready and metadata_ready,
            name=f"moonshine:{model_name}",
        )

    async def warmup(self) -> int:
        if not (await self.health()).ready or not self.model_root:
            return 0
        await self._ensure_transcriber()
        total = 0
        for entry in self.model_root.rglob("*"):
            if entry.is_file():
                total += entry.stat().st_size
        return total

    async def transcribe(
        self, audio_path: Path, options: TranscriptionOptions
    ) -> EngineTranscription:
        _check_language(self.language, options.language)
        if not (await self.health()).ready:
            raise errors.EngineUnavailableError(
                "Moonshine or its selected model is unavailable. Install the engines extra and "
                "download a compatible Moonshine model."
            )
        async with self._inference_lock:
            start_time = time.monotonic()
            transcriber, loaded_now = await self._ensure_transcriber()
            load_ms = 0
            if loaded_now:
                load_ms = max(0, int((time.monotonic() - start_time) * 1000))
            inf_time = time.monotonic()
            return EngineTranscription(
                text=await _run_moonshine_inference(transcriber, audio_path),
                model_load_ms=load_ms,
                inference_ms=max(0, int((time.monotonic() - inf_time) * 1000)),
            )

    async def create_stream(self) -> Any:
        transcriber, _ = await self._ensure_transcriber()
        return await asyncio.to_thread(_start_stream, transcriber)

    async def _ensure_transcriber(self) -> tuple[Any, bool]:
        if self._transcriber is not None:
            return self._transcriber, False
        async with self._load_lock:
            if self._transcriber is not None:
                return self._transcriber, False
            self._transcriber = await asyncio.to_thread(self._load_transcriber_sync)
            return self._transcriber, True

    def _load_transcriber_sync(self) -> Any:
        if self.model_root is None:
            raise errors.EngineUnavailableError("No Moonshine model is selected.")
        metadata = self._metadata
        from moonshine_voice import ModelArch, Transcriber

        model_path = self.model_root / str(metadata["model_path"])
        options = {"max_tokens_per_second": "13.0"} if self.language in NON_LATIN_LANGUAGES else {}
        return Transcriber(
            model_path=model_path,
            model_arch=ModelArch(metadata["model_arch"]),
            options=options,
        )


def _check_language(model_lang: str, requested: str) -> None:
    aliases_dict = dict(LANGUAGE_ALIASES)
    aliases = aliases_dict.get(model_lang, frozenset((model_lang,)))
    if requested != "auto" and requested not in aliases:
        raise errors.LanguageUnsupportedError(
            f"The selected Moonshine model supports {model_lang}; "
            f"choose {model_lang}, Auto, or another model."
        )


async def _run_moonshine_inference(transcriber: Any, audio_path: Path) -> str:
    try:
        text = await asyncio.wait_for(
            asyncio.to_thread(_batch_transcribe, transcriber, audio_path),
            timeout=TRANSCRIPTION_TIMEOUT_SECONDS,
        )
    except TimeoutError as error:
        raise errors.TranscriptionProcessError("Moonshine transcription timed out.") from error
    except Exception as error:
        detail = str(error)[-MAXIMUM_ERROR_MESSAGE_LENGTH:]
        raise errors.TranscriptionProcessError(f"Moonshine failed: {detail}") from error
    if not text:
        raise errors.TranscriptionProcessError("Moonshine returned an empty transcript.")
    return text


def _read_metadata(model_root: Path | None) -> dict[str, Any]:
    if model_root is None:
        return {}
    metadata_file = model_root / MODEL_METADATA
    try:
        raw_text = metadata_file.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return {}
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_wave_pcm(audio_path: Path) -> tuple[int, array[int]]:
    with wave.open(str(audio_path), "rb") as source:
        channels = source.getnchannels()
        if source.getsampwidth() != 2:
            raise ValueError("Moonshine expects normalized 16-bit PCM WAV audio.")
        sample_rate = source.getframerate()
        samples = array("h", source.readframes(source.getnframes()))
    if channels > 1:
        samples = array("h", samples[::channels])
    return sample_rate, samples


def _batch_transcribe(transcriber: Any, audio_path: Path) -> str:
    sample_rate, samples = _read_wave_pcm(audio_path)
    floats = [sample / PCM_SAMPLE_SCALE for sample in samples]
    transcript = transcriber.transcribe_without_streaming(floats, sample_rate)
    lines = [line.text.strip() for line in transcript.lines if line.text.strip()]
    return " ".join(lines).strip()


def _start_stream(transcriber: Any) -> Any:
    stream = transcriber.create_stream(update_interval=STREAM_UPDATE_INTERVAL_SECONDS)
    stream.start()
    return stream
