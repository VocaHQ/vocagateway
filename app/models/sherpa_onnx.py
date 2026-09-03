from __future__ import annotations

import asyncio
import re
import time
import wave
from collections.abc import Callable
from importlib import util as importlib_util
from pathlib import Path
from types import MappingProxyType
from typing import Any

from app import catalog, errors, system
from app.models.base import EngineHealth, EngineTranscription, TranscriptionOptions

MODEL_METADATA = ".vocagateway-model.json"
STREAMING_MODEL_TYPE = "streaming_zipformer"
TRANSCRIPTION_TIMEOUT_SECONDS = 180
MAXIMUM_ERROR_MESSAGE_LENGTH = 240
PCM_SAMPLE_SCALE = 32_768.0
CPU_DEVICE = "cpu"
AUTO_LANGUAGE = "auto"
NEMOTRON_MODEL_KEY = "nemotron-3.5-asr-streaming-0.6b-320ms-int8"
NEMOTRON_LANGUAGE_TAG = re.compile(r"\s*<([a-z]{2,3}(?:-[A-Z]{2})?)>\s*$")
NEMOTRON_LANGUAGE_LOCALES = MappingProxyType(
    {
        "en": "en-US",
        "es": "es-US",
        "fr": "fr-FR",
        "it": "it-IT",
        "pt": "pt-BR",
        "nl": "nl-NL",
        "de": "de-DE",
        "tr": "tr-TR",
        "ru": "ru-RU",
        "ar": "ar-AR",
        "hi": "hi-IN",
        "ja": "ja-JP",
        "ko": "ko-KR",
        "vi": "vi-VN",
        "uk": "uk-UA",
        "pl": "pl-PL",
        "sv": "sv-SE",
        "cs": "cs-CZ",
        "no": "nb-NO",
        "da": "da-DK",
        "bg": "bg-BG",
        "fi": "fi-FI",
        "hr": "hr-HR",
        "sk": "sk-SK",
        "zh": "zh-CN",
        "hu": "hu-HU",
        "ro": "ro-RO",
        "et": "et-EE",
    }
)


class _SherpaRecognizerBuilder:
    def __init__(
        self, model_root: Path | None, catalog_model: catalog.CatalogModel | None, threads: int
    ) -> None:
        self.root = model_root
        self.model = catalog_model
        self.threads = threads
        self.tokens = str(model_root / "tokens.txt") if model_root else ""

    def build(self) -> Any:
        if self.root is None or self.model is None:
            raise errors.EngineUnavailableError("No sherpa-onnx model is selected.")
        import sherpa_onnx

        if self.model.model_type == STREAMING_MODEL_TYPE:
            return self._build_streaming(sherpa_onnx)
        return self._build_offline(sherpa_onnx)

    def validate_language(self, language: str) -> None:
        supported = self.model.language_codes if self.model else ()
        normalized = language.lower().split("-", maxsplit=1)[0]
        if language != AUTO_LANGUAGE and supported and normalized not in supported:
            choices = ", ".join(supported)
            raise errors.LanguageUnsupportedError(
                f"The selected model does not support {language}. Choose Auto, {choices}, or "
                "another model."
            )

    def stream_language(self, language: str) -> str:
        normalized = language.lower().split("-", maxsplit=1)[0]
        if self.model and self.model.key == NEMOTRON_MODEL_KEY:
            return NEMOTRON_LANGUAGE_LOCALES.get(normalized, normalized)
        return language

    def uses_stream_language_locale(self) -> bool:
        return self.model is not None and self.model.key == NEMOTRON_MODEL_KEY

    def strips_stream_language_tags(self) -> bool:
        return self.uses_stream_language_locale()

    def _build_streaming(self, sherpa: Any) -> Any:
        if self.root is None or self.model is None:
            return None
        encoder, decoder, joiner, _ = self.model.required_files
        return sherpa.OnlineRecognizer.from_transducer(
            tokens=self.tokens,
            encoder=str(self.root / encoder),
            decoder=str(self.root / decoder),
            joiner=str(self.root / joiner),
            num_threads=self.threads,
            provider=CPU_DEVICE,
            enable_endpoint_detection=True,
        )

    def _build_offline(self, sherpa: Any) -> Any:
        if self.root is None or self.model is None:
            return None
        mtype = self.model.model_type
        if mtype == "sense_voice":
            return sherpa.OfflineRecognizer.from_sense_voice(
                model=str(self.root / "model.int8.onnx"),
                tokens=self.tokens,
                num_threads=self.threads,
                language=AUTO_LANGUAGE,
                use_itn=True,
                provider=CPU_DEVICE,
            )
        if mtype in {"nemo_transducer", "nemo_ctc", "nemo_canary"}:
            return self._build_nemo(sherpa, mtype)
        if mtype in {"dolphin_ctc", "qwen3_asr"}:
            return self._build_other(sherpa, mtype)
        raise errors.EngineUnavailableError(f"Unsupported sherpa-onnx model type: {mtype}.")

    def _build_nemo(self, sherpa: Any, mtype: str) -> Any:
        if self.root is None or self.model is None:
            return None
        if mtype == "nemo_transducer":
            encoder, decoder, joiner, _ = self.model.required_files
            return sherpa.OfflineRecognizer.from_transducer(
                encoder=str(self.root / encoder),
                decoder=str(self.root / decoder),
                joiner=str(self.root / joiner),
                tokens=self.tokens,
                num_threads=self.threads,
                model_type="nemo_transducer",
                provider=CPU_DEVICE,
            )
        if mtype == "nemo_ctc":
            return sherpa.OfflineRecognizer.from_nemo_ctc(
                model=str(self.root / "model.int8.onnx"),
                tokens=self.tokens,
                num_threads=self.threads,
                provider=CPU_DEVICE,
            )
        return sherpa.OfflineRecognizer.from_nemo_canary(
            encoder=str(self.root / "encoder.int8.onnx"),
            decoder=str(self.root / "decoder.int8.onnx"),
            tokens=self.tokens,
            num_threads=self.threads,
            provider=CPU_DEVICE,
        )

    def _build_other(self, sherpa: Any, mtype: str) -> Any:
        if self.root is None:
            return None
        if mtype == "dolphin_ctc":
            return sherpa.OfflineRecognizer.from_dolphin_ctc(
                model=str(self.root / "model.int8.onnx"),
                tokens=self.tokens,
                num_threads=self.threads,
                provider=CPU_DEVICE,
            )
        return sherpa.OfflineRecognizer.from_qwen3_asr(
            conv_frontend=str(self.root / "conv_frontend.onnx"),
            encoder=str(self.root / "encoder.int8.onnx"),
            decoder=str(self.root / "decoder.int8.onnx"),
            tokenizer=str(self.root / "tokenizer"),
            num_threads=self.threads,
            provider=CPU_DEVICE,
        )


class _SherpaOnnxStreamAdapter:
    def __init__(
        self,
        recognizer: Any = None,
        stream: Any = None,
        line_id: int = 0,
        text: str = "",
        lines: list[_SherpaOnnxStreamAdapter] | None = None,
        language_mapper: Callable[[str], str] | None = None,
    ) -> None:
        self._recognizer = recognizer
        self._stream = stream
        self._listener: Callable[[object], None] | None = None
        self._completed_lines: list[_SherpaOnnxStreamAdapter] = []
        self._next_line_id = 0
        self._last_partial = ""
        self.line_id = line_id
        self.text = text
        self.line = self
        self.lines = lines or []
        self._language_mapper = language_mapper
        self._strip_language_tags = language_mapper is not None
        self.detected_language: str | None = None

    def add_audio(self, samples: list[float], sample_rate: int) -> None:
        self._stream.accept_waveform(sample_rate, samples)
        self._drain()

    def set_language(self, language: str) -> None:
        """Apply a per-stream language option when the export exposes one."""
        mapped = self._language_mapper(language) if self._language_mapper else language
        _set_stream_language(
            self._stream,
            mapped,
            preserve_locale=self._language_mapper is not None,
        )

    def add_listener(self, listener: Callable[[object], None]) -> None:
        self._listener = listener

    def stop(self) -> _SherpaOnnxStreamAdapter:
        self._stream.input_finished()
        self._drain()
        trailing = self._clean_text(str(self._recognizer.get_result(self._stream)))
        if trailing:
            line = _SherpaOnnxStreamAdapter(line_id=self._next_line_id, text=trailing)
            self._completed_lines.append(line)
        completed = _SherpaOnnxStreamAdapter(lines=list(self._completed_lines))
        completed.detected_language = self.detected_language
        return completed

    def _drain(self) -> None:
        while self._recognizer.is_ready(self._stream):
            self._recognizer.decode_stream(self._stream)
        if self._recognizer.is_endpoint(self._stream):
            text = self._clean_text(str(self._recognizer.get_result(self._stream)))
            if text:
                line = _SherpaOnnxStreamAdapter(line_id=self._next_line_id, text=text)
                self._completed_lines.append(line)
                self._notify(line)
            self._next_line_id += 1
            self._last_partial = ""
            self._recognizer.reset(self._stream)
            return
        partial = self._clean_text(str(self._recognizer.get_result(self._stream)))
        if partial and partial != self._last_partial:
            self._last_partial = partial
            self._notify(_SherpaOnnxStreamAdapter(line_id=self._next_line_id, text=partial))

    def _clean_text(self, text: str) -> str:
        cleaned = text.strip()
        if not self._strip_language_tags:
            return cleaned
        match = NEMOTRON_LANGUAGE_TAG.search(cleaned)
        if match:
            self.detected_language = match.group(1)
            cleaned = cleaned[: match.start()].rstrip()
        return cleaned

    def _notify(self, line: _SherpaOnnxStreamAdapter) -> None:
        if self._listener is not None:
            self._listener(line)


class SherpaOnnxEngine:
    """Persistent CPU recognizer for compact sherpa-onnx model exports."""

    def __init__(
        self,
        model_root: Path | None,
        catalog_model: catalog.CatalogModel | None,
        *,
        cpu_threads: int = 0,
    ) -> None:
        self.model_root = model_root
        self.catalog_model = catalog_model
        self.cpu_threads = cpu_threads
        self._recognizer: Any | None = None
        self._load_lock = asyncio.Lock()
        self._inference_lock = asyncio.Lock()
        self.streaming_lock = self._inference_lock
        self.supports_streaming: bool = (
            catalog_model is not None and catalog_model.model_type == STREAMING_MODEL_TYPE
        )
        threads = system.inference_thread_count(cpu_threads)
        self._builder = _SherpaRecognizerBuilder(model_root, catalog_model, threads)

    async def create_stream(self) -> _SherpaOnnxStreamAdapter:
        if not self.supports_streaming:
            raise errors.EngineUnavailableError("The selected sherpa-onnx model does not stream.")
        if not (await self.health()).ready:
            raise errors.EngineUnavailableError(
                "sherpa-onnx or its selected streaming model is unavailable."
            )
        recognizer, _ = await self._ensure_recognizer()
        stream = await asyncio.to_thread(recognizer.create_stream)
        return _SherpaOnnxStreamAdapter(
            recognizer,
            stream,
            language_mapper=(
                self._builder.stream_language
                if self._builder.uses_stream_language_locale()
                else None
            ),
        )

    def configure_stream(self, stream: object, language: str) -> None:
        """Validate and configure a newly-created stream for a request."""
        self._builder.validate_language(language)
        setter = getattr(stream, "set_language", None)
        if callable(setter):
            setter(language)

    async def health(self) -> EngineHealth:
        package_ready = importlib_util.find_spec("sherpa_onnx") is not None
        model_ready = (
            self.model_root is not None
            and self.catalog_model is not None
            and (self.model_root / MODEL_METADATA).is_file()
            and all(
                (self.model_root / name).is_file() for name in self.catalog_model.required_files
            )
        )
        model_name = self.model_root.name if self.model_root else "no-model-selected"
        return EngineHealth(
            ready=package_ready and model_ready,
            name=f"sherpa-onnx:{model_name}",
        )

    async def transcribe(
        self, audio_path: Path, options: TranscriptionOptions
    ) -> EngineTranscription:
        self._builder.validate_language(options.language)
        if not (await self.health()).ready:
            raise errors.EngineUnavailableError(
                "sherpa-onnx or its selected model is unavailable. Install the engines extra "
                "and download a compatible sherpa-onnx model."
            )
        async with self._inference_lock:
            start_time = time.monotonic()
            recognizer, loaded_now = await self._ensure_recognizer()
            load_ms = 0
            if loaded_now:
                load_ms = max(0, int((time.monotonic() - start_time) * 1000))
            start_time = time.monotonic()
            text = await _run_sherpa_inference(
                recognizer,
                audio_path,
                self.supports_streaming,
                self._builder.stream_language(options.language),
                self._builder.uses_stream_language_locale(),
                self._builder.strips_stream_language_tags(),
            )
            if not text:
                if options.language != AUTO_LANGUAGE:
                    raise errors.LanguageUnsupportedError(
                        f"The selected model returned nothing for {options.language}. "
                        "It probably does not cover that language — choose another "
                        "model, or set the language to Automatic."
                    )
                raise errors.TranscriptionProcessError("sherpa-onnx returned an empty transcript.")
            return EngineTranscription(
                text=text,
                model_load_ms=load_ms,
                inference_ms=max(0, int((time.monotonic() - start_time) * 1000)),
            )

    async def warmup(self) -> int:
        if not (await self.health()).ready or not self.model_root:
            return 0
        await self._ensure_recognizer()
        total = 0
        for entry in self.model_root.rglob("*"):
            if entry.is_file():
                total += entry.stat().st_size
        return total

    async def _ensure_recognizer(self) -> tuple[Any, bool]:
        if self._recognizer is not None:
            return self._recognizer, False
        async with self._load_lock:
            if self._recognizer is not None:
                return self._recognizer, False
            self._recognizer = await asyncio.to_thread(self._load_recognizer_sync)
            return self._recognizer, True

    def _load_recognizer_sync(self) -> Any:
        return self._builder.build()


def _read_wave_samples(audio_path: Path) -> tuple[int, Any]:
    """Read a PCM WAV file as the float waveform sherpa-onnx accepts.

    numpy is a declared dependency of the `engines` extra, which is the only
    way this engine is installed, so the vectorized conversion is always
    available on the path that reaches this function. The list comprehension it
    replaces built one Python float per sample, which is roughly half a million
    short-lived objects for a 30-second recording.
    """
    import numpy

    with wave.open(str(audio_path), "rb") as source:
        if source.getsampwidth() != 2:
            raise ValueError("sherpa-onnx expects normalized 16-bit PCM WAV audio.")
        channels = source.getnchannels()
        sample_rate = source.getframerate()
        frames = source.readframes(source.getnframes())
    samples = numpy.frombuffer(frames, dtype=numpy.int16)
    if channels > 1:
        samples = samples[::channels]
    return sample_rate, samples.astype(numpy.float32) / PCM_SAMPLE_SCALE


def _decode_wave(recognizer: Any, audio_path: Path) -> str:
    sample_rate, floats = _read_wave_samples(audio_path)
    stream = recognizer.create_stream()
    stream.accept_waveform(sample_rate, floats)
    recognizer.decode_stream(stream)
    return str(stream.result.text).strip()


def _decode_wave_online(
    recognizer: Any,
    audio_path: Path,
    language: str = AUTO_LANGUAGE,
    preserve_locale: bool = False,
    strip_language_tags: bool = False,
) -> str:
    sample_rate, floats = _read_wave_samples(audio_path)
    stream = recognizer.create_stream()
    _set_stream_language(stream, language, preserve_locale=preserve_locale)
    stream.accept_waveform(sample_rate, floats)
    stream.input_finished()
    while recognizer.is_ready(stream):
        recognizer.decode_stream(stream)
    text = str(recognizer.get_result(stream)).strip()
    return _strip_language_tag(text) if strip_language_tags else text


async def _run_sherpa_inference(
    recognizer: Any,
    audio_path: Path,
    is_streaming: bool,
    language: str = AUTO_LANGUAGE,
    preserve_locale: bool = False,
    strip_language_tags: bool = False,
) -> str:
    try:
        if is_streaming:
            decode_result = asyncio.to_thread(
                _decode_wave_online,
                recognizer,
                audio_path,
                language,
                preserve_locale,
                strip_language_tags,
            )
        else:
            decode_result = asyncio.to_thread(_decode_wave, recognizer, audio_path)
        return await asyncio.wait_for(
            decode_result,
            timeout=TRANSCRIPTION_TIMEOUT_SECONDS,
        )
    except TimeoutError as error:
        raise errors.TranscriptionProcessError("sherpa-onnx transcription timed out.") from error
    except Exception as error:
        detail = str(error)[-MAXIMUM_ERROR_MESSAGE_LENGTH:]
        raise errors.TranscriptionProcessError(f"sherpa-onnx failed: {detail}") from error


def _set_stream_language(stream: Any, language: str, *, preserve_locale: bool = False) -> None:
    """Set sherpa's optional language stream option without breaking fixed exports."""
    setter = getattr(stream, "set_option", None)
    if not callable(setter):
        return
    has_option = getattr(stream, "has_option", None)
    if callable(has_option):
        try:
            has_language = bool(has_option("language"))
        except Exception:  # noqa: BLE001 - older bindings may not probe cleanly
            has_language = True
        if not has_language:
            return
    normalized = (
        language
        if language == AUTO_LANGUAGE or preserve_locale
        else language.lower().split("-", maxsplit=1)[0]
    )
    setter("language", normalized)


def _strip_language_tag(text: str) -> str:
    """Remove Nemotron's automatic ``<locale>`` suffix from clean text."""
    return NEMOTRON_LANGUAGE_TAG.sub("", text).strip()
