# Copyright (c) 2025-present VocaHQ, Inc.
# SPDX-License-Identifier: AGPL-3.0-only
# See the LICENSE file for details.

from __future__ import annotations

import asyncio
import re
import time
import wave
from collections.abc import Callable
from dataclasses import dataclass
from importlib import util as importlib_util
from pathlib import Path
from types import MappingProxyType
from typing import Any

from app import catalog, errors, system
from app.models.base import EngineHealth, EngineTranscription, TranscriptionOptions

MODEL_METADATA = ".vocagateway-model.json"
# The catalog names the model types; dispatching on its constants keeps a
# renamed type from silently falling through to "unsupported" here.
STREAMING_MODEL_TYPE = catalog.STREAMING_TRANSDUCER_TYPE
NEMO_TRANSDUCER_TYPE = catalog.NEMO_TRANSDUCER_TYPE
COHERE_TRANSCRIBE_TYPE = catalog.COHERE_TRANSCRIBE_TYPE
TRANSCRIPTION_TIMEOUT_SECONDS = 180
MAXIMUM_ERROR_MESSAGE_LENGTH = 240
PCM_SAMPLE_SCALE = 32_768.0
CPU_DEVICE = "cpu"
AUTO_LANGUAGE = "auto"
LANGUAGE_TAG_SEPARATOR = "-"
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

    def build(self, language: str = AUTO_LANGUAGE) -> Any:
        if self.root is None or self.model is None:
            raise errors.EngineUnavailableError("No sherpa-onnx model is selected.")
        import sherpa_onnx

        if self.model.model_type == STREAMING_MODEL_TYPE:
            return self._build_streaming(sherpa_onnx)
        return self._build_offline(sherpa_onnx, self.build_language(language))

    def pins_language_at_build(self) -> bool:
        """Whether the recognizer bakes in a language and must be rebuilt to change it.

        Only Cohere Transcribe does. Its decoder takes the language when the
        recognizer is constructed, so a cached recognizer built for one language
        would otherwise keep decoding every later request in that language --
        silently, since a wrong-language decode still returns fluent text.
        """
        return self.model is not None and self.model.model_type == COHERE_TRANSCRIBE_TYPE

    def build_language(self, language: str) -> str:
        """The language to construct a language-pinned recognizer with."""
        supported = self.model.language_codes if self.model else ()
        if language == AUTO_LANGUAGE:
            return supported[0] if supported else AUTO_LANGUAGE
        return language.lower().split(LANGUAGE_TAG_SEPARATOR, maxsplit=1)[0]

    def validate_language(self, language: str) -> None:
        if (
            self.model
            and self.model.model_type == COHERE_TRANSCRIBE_TYPE
            and language == AUTO_LANGUAGE
        ):
            raise errors.LanguageUnsupportedError(
                "Cohere Transcribe requires an explicit spoken language; "
                "choose a language instead of Auto."
            )
        supported = self.model.language_codes if self.model else ()
        normalized = language.lower().split(LANGUAGE_TAG_SEPARATOR, maxsplit=1)[0]
        if language != AUTO_LANGUAGE and supported and normalized not in supported:
            choices = ", ".join(supported)
            raise errors.LanguageUnsupportedError(
                f"The selected model does not support {language}. Choose Auto, {choices}, or "
                "another model."
            )

    def stream_language(self, language: str) -> str:
        normalized = language.lower().split(LANGUAGE_TAG_SEPARATOR, maxsplit=1)[0]
        if self.model and self.model.key == NEMOTRON_MODEL_KEY:
            return NEMOTRON_LANGUAGE_LOCALES.get(normalized, normalized)
        return language

    def language_policy(self, requested: str) -> _LanguagePolicy:
        return _LanguagePolicy(
            language=self.stream_language(requested),
            preserve_locale=self.uses_stream_language_locale(),
            strip_tags=self.strips_stream_language_tags(),
            set_on_stream=self.pins_language_at_build(),
        )

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

    def _build_offline(self, sherpa: Any, language: str = AUTO_LANGUAGE) -> Any:
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
        if mtype in {NEMO_TRANSDUCER_TYPE, "nemo_ctc", "nemo_canary"}:
            return self._build_nemo(sherpa, mtype)
        if mtype == COHERE_TRANSCRIBE_TYPE:
            return sherpa.OfflineRecognizer.from_cohere_transcribe(
                encoder=str(self.root / "encoder.int8.onnx"),
                decoder=str(self.root / "decoder.int8.onnx"),
                tokens=self.tokens,
                num_threads=self.threads,
                language=language,
                provider=CPU_DEVICE,
            )
        if mtype in {"dolphin_ctc", "qwen3_asr"}:
            return self._build_other(sherpa, mtype)
        raise errors.EngineUnavailableError(f"Unsupported sherpa-onnx model type: {mtype}.")

    def _build_nemo(self, sherpa: Any, mtype: str) -> Any:
        if self.root is None or self.model is None:
            return None
        if mtype == NEMO_TRANSDUCER_TYPE:
            encoder, decoder, joiner, _ = self.model.required_files
            return sherpa.OfflineRecognizer.from_transducer(
                encoder=str(self.root / encoder),
                decoder=str(self.root / decoder),
                joiner=str(self.root / joiner),
                tokens=self.tokens,
                num_threads=self.threads,
                model_type=NEMO_TRANSDUCER_TYPE,
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


def _model_files_present(root: Path | None, model: catalog.CatalogModel | None) -> bool:
    if root is None or model is None:
        return False
    return (root / MODEL_METADATA).is_file() and all(
        (root / name).is_file() for name in model.required_files
    )


class _ResidentRecognizer:
    """One cached sherpa recognizer, its builder, and the language it was built for.

    An engine holds one of these per export it can run. A model whose streaming
    export is buffered keeps a second for whole-file work, so the two never
    share a cache slot and neither evicts the other on a language rebuild.
    """

    def __init__(
        self, model_root: Path | None, catalog_model: catalog.CatalogModel | None, threads: int
    ) -> None:
        self.model_root = model_root
        self.catalog_model = catalog_model
        self.builder = _SherpaRecognizerBuilder(model_root, catalog_model, threads)
        self._recognizer: Any | None = None
        self._language: str | None = None
        self._lock = asyncio.Lock()

    @property
    def is_online(self) -> bool:
        """Whether decoding goes through the chunked online API rather than offline."""
        return (
            self.catalog_model is not None and self.catalog_model.model_type == STREAMING_MODEL_TYPE
        )

    @property
    def is_resident(self) -> bool:
        return self._recognizer is not None

    @property
    def files_present(self) -> bool:
        return _model_files_present(self.model_root, self.catalog_model)

    def unload(self) -> None:
        self._recognizer = None
        self._language = None

    async def ensure(self, language: str = AUTO_LANGUAGE) -> tuple[Any, bool]:
        wanted = self.builder.build_language(language)
        if self._recognizer is not None and not self._needs_rebuild(wanted):
            return self._recognizer, False
        async with self._lock:
            if self._recognizer is not None and not self._needs_rebuild(wanted):
                return self._recognizer, False
            self._recognizer = await asyncio.to_thread(self.builder.build, language)
            self._language = wanted
            return self._recognizer, True

    def _needs_rebuild(self, wanted: str) -> bool:
        """A language-pinned recognizer built for another language is the wrong one."""
        return self.builder.pins_language_at_build() and self._language != wanted


class SherpaOnnxEngine:
    """Persistent CPU recognizer for compact sherpa-onnx model exports."""

    def __init__(
        self,
        model_root: Path | None,
        catalog_model: catalog.CatalogModel | None,
        *,
        cpu_threads: int = 0,
        batch_root: Path | None = None,
        batch_model: catalog.CatalogModel | None = None,
    ) -> None:
        self.model_root = model_root
        self.catalog_model = catalog_model
        self.cpu_threads = cpu_threads
        self._inference_lock = asyncio.Lock()
        self.streaming_lock = self._inference_lock
        self.supports_streaming: bool = (
            catalog_model is not None and catalog_model.model_type == STREAMING_MODEL_TYPE
        )
        threads = system.inference_thread_count(cpu_threads)
        self._selected = _ResidentRecognizer(model_root, catalog_model, threads)
        # A buffered streaming export re-encodes its whole context window every
        # step, so transcribing a finished file through it costs many times what
        # the batch export of the same weights costs — and buys nothing, since
        # there are no partials to deliver early. When the twin is installed,
        # send whole-file work there and keep the streaming export for live audio.
        self._twin: _ResidentRecognizer | None = (
            _ResidentRecognizer(batch_root, batch_model, threads)
            if batch_root is not None and batch_model is not None
            else None
        )

    async def create_stream(self) -> _SherpaOnnxStreamAdapter:
        if not self.supports_streaming:
            raise errors.EngineUnavailableError("The selected sherpa-onnx model does not stream.")
        if not (await self.health()).ready:
            raise errors.EngineUnavailableError(
                "sherpa-onnx or its selected streaming model is unavailable."
            )
        recognizer, _ = await self._selected.ensure()
        stream = await asyncio.to_thread(recognizer.create_stream)
        builder = self._selected.builder
        return _SherpaOnnxStreamAdapter(
            recognizer,
            stream,
            language_mapper=(
                builder.stream_language if builder.uses_stream_language_locale() else None
            ),
        )

    def configure_stream(self, stream: object, language: str) -> None:
        """Validate and configure a newly-created stream for a request."""
        self._selected.builder.validate_language(language)
        setter = getattr(stream, "set_language", None)
        if callable(setter):
            setter(language)

    async def health(self) -> EngineHealth:
        package_ready = importlib_util.find_spec("sherpa_onnx") is not None
        model_ready = _model_files_present(self.model_root, self.catalog_model)
        model_name = self.model_root.name if self.model_root else "no-model-selected"
        return EngineHealth(
            ready=package_ready and model_ready,
            name=f"sherpa-onnx:{model_name}",
        )

    async def transcribe(
        self, audio_path: Path, options: TranscriptionOptions
    ) -> EngineTranscription:
        # Validated against the selected model, not whichever export decodes it:
        # the twin shares its weights and its language coverage, and a client
        # asking for something the selection cannot do should hear so either way.
        self._selected.builder.validate_language(options.language)
        if not (await self.health()).ready:
            raise errors.EngineUnavailableError(
                "sherpa-onnx or its selected model is unavailable. Install the engines extra "
                "and download a compatible sherpa-onnx model."
            )
        async with self._inference_lock:
            start_time = time.monotonic()
            batch, recognizer, loaded_now = await self._load_batch(options.language)
            load_ms = 0
            if loaded_now:
                load_ms = max(0, int((time.monotonic() - start_time) * 1000))
            start_time = time.monotonic()
            text = await _run_sherpa_inference(
                recognizer,
                audio_path,
                batch.is_online,
                batch.builder.language_policy(options.language),
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
        """Preload the export each request path will actually reach.

        With a batch twin in play that is two recognizers, because an engine
        that reports itself warm and still pays a cold load on the first
        dictation has warmed the wrong one. The selection is warmed first so a
        twin that cannot load leaves the streaming path ready regardless.
        """
        if not (await self.health()).ready or not self.model_root:
            return 0
        await self._selected.ensure()
        await self._load_batch(AUTO_LANGUAGE)
        return sum(
            _directory_bytes(resident.model_root)
            for resident in self._residents()
            if resident.is_resident
        )

    @property
    def model_is_resident(self) -> bool:
        return any(resident.is_resident for resident in self._residents())

    def unload(self) -> None:
        for resident in self._residents():
            resident.unload()

    @property
    def _batch(self) -> _ResidentRecognizer:
        """The export that decodes a finished file, re-resolved on every use.

        Deliberately not a snapshot taken in `__init__`: models are downloaded
        and deleted from the Models tab without rebuilding the engine, so a
        value fixed there would never notice a twin that arrived afterwards and
        would keep pointing at one that has since been removed.
        """
        twin = self._twin
        return twin if twin is not None and twin.files_present else self._selected

    async def _load_batch(self, language: str) -> tuple[_ResidentRecognizer, Any, bool]:
        """Load the export that decodes a finished file.

        The twin is an optimisation, so it is never allowed to fail a request
        the selection could have served: a download whose files are all in
        place but corrupt, or a machine too small to hold both exports at once,
        falls back instead of taking a working engine down with it.
        """
        batch = self._batch
        try:
            recognizer, loaded_now = await batch.ensure(language)
        except Exception:
            if batch is self._selected:
                raise
            batch = self._selected
            recognizer, loaded_now = await batch.ensure(language)
        return batch, recognizer, loaded_now

    def _residents(self) -> tuple[_ResidentRecognizer, ...]:
        """Every holder this engine owns, whether or not it is currently used.

        A twin deleted from disk keeps its weights in memory until something
        unloads them, so idle offload has to be able to reach it here.
        """
        if self._twin is None:
            return (self._selected,)
        return (self._selected, self._twin)


def _directory_bytes(root: Path | None) -> int:
    if root is None:
        return 0
    return sum(entry.stat().st_size for entry in root.rglob("*") if entry.is_file())


def _read_wave_samples(audio_path: Path) -> tuple[int, Any]:
    """Read a PCM WAV file as the float waveform sherpa-onnx accepts.

    numpy turns this into two vector operations instead of one Python float per
    sample — roughly half a million short-lived objects for a 30-second clip —
    but it arrives with the `engines` extra rather than the core install, and
    the test suite runs without that extra. So the stdlib comprehension stays
    as the fallback, exactly as `app.audio` keeps one for its RMS gate.
    """
    with wave.open(str(audio_path), "rb") as source:
        if source.getsampwidth() != 2:
            raise ValueError("sherpa-onnx expects normalized 16-bit PCM WAV audio.")
        channels = source.getnchannels()
        sample_rate = source.getframerate()
        frames = source.readframes(source.getnframes())
    try:
        import numpy
    except ImportError:
        samples = memoryview(frames).cast("h")[::channels]
        return sample_rate, [sample / PCM_SAMPLE_SCALE for sample in samples]
    block = numpy.frombuffer(frames, dtype=numpy.int16)[::channels]
    return sample_rate, block.astype(numpy.float32) / PCM_SAMPLE_SCALE


@dataclass(frozen=True, slots=True)
class _LanguagePolicy:
    """How one request's language reaches the recognizer.

    The four settings are one decision made per model, so they travel together
    rather than as a row of positional flags through three call layers.
    """

    language: str = AUTO_LANGUAGE
    preserve_locale: bool = False
    strip_tags: bool = False
    set_on_stream: bool = False


def _decode_wave(recognizer: Any, audio_path: Path, policy: _LanguagePolicy) -> str:
    language = policy.language
    set_stream_language = policy.set_on_stream
    sample_rate, floats = _read_wave_samples(audio_path)
    stream = recognizer.create_stream()
    setter = getattr(stream, "set_option", None)
    if set_stream_language and language != AUTO_LANGUAGE and callable(setter):
        # Belt and braces for Cohere only: the recognizer was already built for
        # this language. `_set_stream_language`'s has_option probe is no use
        # here, because offline has_option reports whether a value is *stored*,
        # not whether the option is supported, and a new stream starts empty.
        # Every other offline model decoded without this before, and pinning
        # one that auto-detects would quietly change its transcripts.
        setter("language", language.lower().split(LANGUAGE_TAG_SEPARATOR, maxsplit=1)[0])
    stream.accept_waveform(sample_rate, floats)
    recognizer.decode_stream(stream)
    return str(stream.result.text).strip()


def _decode_wave_online(recognizer: Any, audio_path: Path, policy: _LanguagePolicy) -> str:
    sample_rate, floats = _read_wave_samples(audio_path)
    stream = recognizer.create_stream()
    _set_stream_language(stream, policy.language, preserve_locale=policy.preserve_locale)
    stream.accept_waveform(sample_rate, floats)
    stream.input_finished()
    while recognizer.is_ready(stream):
        recognizer.decode_stream(stream)
    text = str(recognizer.get_result(stream)).strip()
    return _strip_language_tag(text) if policy.strip_tags else text


async def _run_sherpa_inference(
    recognizer: Any,
    audio_path: Path,
    is_streaming: bool,
    policy: _LanguagePolicy,
) -> str:
    try:
        decoder = _decode_wave_online if is_streaming else _decode_wave
        decode_result = asyncio.to_thread(decoder, recognizer, audio_path, policy)
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
        else language.lower().split(LANGUAGE_TAG_SEPARATOR, maxsplit=1)[0]
    )
    setter("language", normalized)


def _strip_language_tag(text: str) -> str:
    """Remove Nemotron's automatic ``<locale>`` suffix from clean text."""
    return NEMOTRON_LANGUAGE_TAG.sub("", text).strip()
