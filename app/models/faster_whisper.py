from __future__ import annotations

import asyncio
import time
from importlib import util as importlib_util
from pathlib import Path
from typing import Any

from app import system
from app.errors import EngineUnavailableError, TranscriptionProcessError
from app.models.base import EngineHealth, EngineTranscription, TranscriptionOptions
from app.runtime_config import AUTO_ENGINE

TRANSCRIPTION_TIMEOUT_SECONDS = 180
MAXIMUM_ERROR_MESSAGE_LENGTH = 240
# Silero VAD costs a few milliseconds and hands the decoder only the speech.
# A dictation clip is mostly the pause before and after the sentence, and the
# decoder is the expensive half on a CPU host, so skipping that silence is the
# largest single saving available here.
VAD_MINIMUM_SILENCE_MS = 500
VAD_SPEECH_PAD_MS = 200


class FasterWhisperEngine:
    """Persistent CTranslate2-backed Whisper engine optimized for server use."""

    def __init__(
        self,
        model_path: Path | None,
        *,
        device: str = AUTO_ENGINE,
        compute_type: str = AUTO_ENGINE,
        cpu_threads: int = 0,
    ) -> None:
        self.model_path = model_path
        self.device = device
        self.compute_type = compute_type
        self.cpu_threads = cpu_threads
        self._model: Any | None = None
        self._load_lock = asyncio.Lock()
        self._inference_lock = asyncio.Lock()

    async def health(self) -> EngineHealth:
        package_ready = importlib_util.find_spec("faster_whisper") is not None
        model_ready = self.model_path is not None and (self.model_path / "model.bin").is_file()
        model_name = self.model_path.name if self.model_path else "no-model-selected"
        return EngineHealth(
            ready=package_ready and model_ready,
            name=f"faster-whisper:{model_name}",
        )

    async def warmup(self) -> int:
        if not (await self.health()).ready or self.model_path is None:
            return 0
        await self._ensure_model()
        return self._directory_size(self.model_path)

    @property
    def model_is_resident(self) -> bool:
        return self._model is not None

    def unload(self) -> None:
        self._model = None

    async def transcribe(
        self, audio_path: Path, options: TranscriptionOptions
    ) -> EngineTranscription:
        if not (await self.health()).ready:
            raise EngineUnavailableError(
                "faster-whisper or its selected model is unavailable. Install the engines "
                "extra and select a downloaded faster-whisper model."
            )
        async with self._inference_lock:
            start_time = time.monotonic()
            model, loaded_now = await self._ensure_model()
            load_ms = _elapsed_ms(start_time) if loaded_now else 0
            inf_time = time.monotonic()
            return EngineTranscription(
                text=await _run_inference(model, audio_path, options),
                model_load_ms=load_ms,
                inference_ms=_elapsed_ms(inf_time),
            )

    async def _ensure_model(self) -> tuple[Any, bool]:
        if self._model is not None:
            return self._model, False
        async with self._load_lock:
            if self._model is not None:
                return self._model, False
            self._model = await asyncio.to_thread(self._load_model_sync)
            return self._model, True

    def _load_model_sync(self) -> Any:
        if self.model_path is None:
            raise EngineUnavailableError("No faster-whisper model is selected.")
        from faster_whisper import WhisperModel

        device = _resolved_device(self.device)
        comp_type = _resolved_compute_type(self.compute_type, device)
        return WhisperModel(
            str(self.model_path),
            device=device,
            compute_type=comp_type,
            cpu_threads=system.inference_thread_count(self.cpu_threads),
            num_workers=1,
            local_files_only=True,
        )

    def _directory_size(self, path: Path) -> int:
        total = 0
        for entry in path.rglob("*"):
            if entry.is_file():
                total += entry.stat().st_size
        return total


async def _run_inference(model: Any, audio_path: Path, options: TranscriptionOptions) -> str:
    try:
        transcript = await asyncio.wait_for(
            asyncio.to_thread(_extract_text, model, audio_path, options),
            timeout=TRANSCRIPTION_TIMEOUT_SECONDS,
        )
    except TimeoutError as error:
        raise TranscriptionProcessError("faster-whisper transcription timed out.") from error
    except Exception as error:
        detail = str(error)[-MAXIMUM_ERROR_MESSAGE_LENGTH:]
        raise TranscriptionProcessError(f"faster-whisper failed: {detail}") from error
    if not transcript:
        raise TranscriptionProcessError("faster-whisper returned an empty transcript.")
    return transcript


def _extract_text(model: Any, audio_path: Path, options: TranscriptionOptions) -> str:
    """Decode the clip, retrying without VAD when VAD found no speech at all.

    The RMS gate in `app.audio` already rejects a silent recording, so an empty
    VAD pass here means quiet-but-real speech that Silero was not confident
    about. Decoding the whole clip is slower than the VAD path but it is the
    difference between a transcript and a failure, so it is worth the retry.
    """
    transcript = _decode(model, audio_path, options, use_vad=True)
    if transcript:
        return transcript
    return _decode(model, audio_path, options, use_vad=False)


def _decode(model: Any, audio_path: Path, options: TranscriptionOptions, *, use_vad: bool) -> str:
    segments, _ = model.transcribe(
        str(audio_path),
        language=None if options.language == AUTO_ENGINE else options.language,
        beam_size=1,
        best_of=1,
        temperature=0,
        condition_on_previous_text=False,
        # Nothing downstream reads segment times — the gateway returns one joined
        # string — so decoding timestamp tokens is work spent on output nobody uses.
        without_timestamps=True,
        vad_filter=use_vad,
        vad_parameters={
            "min_silence_duration_ms": VAD_MINIMUM_SILENCE_MS,
            "speech_pad_ms": VAD_SPEECH_PAD_MS,
        },
    )
    valid_texts = [segment.text.strip() for segment in segments if segment.text.strip()]
    return " ".join(valid_texts).strip()


def _has_cuda() -> bool:
    try:
        import ctranslate2
    except (ImportError, RuntimeError):
        return False
    return bool(ctranslate2.get_cuda_device_count() > 0)


def _resolved_device(configured_device: str) -> str:
    if configured_device in {"cpu", "cuda"}:
        return configured_device
    return "cuda" if _has_cuda() else "cpu"


def _resolved_compute_type(configured_type: str, device: str) -> str:
    if configured_type != AUTO_ENGINE:
        return configured_type
    return "float16" if device == "cuda" else "int8"


def _elapsed_ms(started: float) -> int:
    return max(0, int((time.monotonic() - started) * 1000))
