from __future__ import annotations

import sys
import types
import wave
from array import array
from importlib import machinery
from pathlib import Path

import pytest

from app.catalog import DEFAULT_CATALOG, CatalogModel
from app.errors import EngineUnavailableError, LanguageUnsupportedError
from app.models.base import EngineTranscription, TranscriptionOptions
from app.models.sherpa_onnx import (
    SherpaOnnxEngine,
    _LanguagePolicy,
    _set_stream_language,
    _SherpaOnnxStreamAdapter,
    _SherpaRecognizerBuilder,
)

NORMALIZED_SAMPLE_RATE_HZ = 16_000
SENSE_VOICE_MODEL_TYPE = "sense_voice"
SHERPA_MODEL_FILE = "model.int8.onnx"
TOKENS_FILE = "tokens.txt"
ENCODER_INT8_FILE = "encoder.int8.onnx"
DECODER_INT8_FILE = "decoder.int8.onnx"
SHERPA_ONNX_MODULE = "sherpa_onnx"
IMPORTLIB_FIND_SPEC_PATH = "app.models.sherpa_onnx.importlib_util.find_spec"
TOKENS_COMPONENT = "tokens"
ENCODER_COMPONENT = "encoder"
DECODER_COMPONENT = "decoder"
DECODER_FILE = "decoder.onnx"
JOINER_FILE = "joiner.onnx"
ENCODER_FILE = "encoder.onnx"
EXPECTED_TRANSCRIPT = "hello"


def _catalog(
    model_type: str = SENSE_VOICE_MODEL_TYPE, required_files: tuple[str, ...] | None = None
) -> CatalogModel:
    if required_files is None and model_type in (SENSE_VOICE_MODEL_TYPE, "nemo_ctc"):
        files = (SHERPA_MODEL_FILE, TOKENS_FILE)
    elif required_files is None and model_type == "nemo_canary":
        files = (ENCODER_INT8_FILE, DECODER_INT8_FILE, TOKENS_FILE)
    elif required_files is None:
        files = (ENCODER_INT8_FILE, DECODER_INT8_FILE, "joiner.int8.onnx", TOKENS_FILE)
    else:
        files = required_files
    return CatalogModel(
        id=f"sherpa-onnx:{model_type}",
        engine="sherpa-onnx",
        key=model_type,
        label="Test",
        size_bytes=1,
        languages="English only",
        quality="Fast",
        minimum_ram_gb=1,
        marker_file=".vocagateway-model.json",
        required_files=files,
        model_type=model_type,
        language_codes=("en",),
    )


def _model_root(tmp_path: Path, model: CatalogModel) -> Path:
    root = tmp_path / model.key
    root.mkdir()
    (root / ".vocagateway-model.json").write_text("{}")
    for filename in model.required_files:
        # Some families (Qwen3-ASR) list files inside a nested tokenizer directory.
        (root / filename).parent.mkdir(parents=True, exist_ok=True)
        (root / filename).write_bytes(b"model")
    return root


def _wave(path: Path) -> None:
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(NORMALIZED_SAMPLE_RATE_HZ)
        output.writeframes(array("h", [0, 100, -100]).tobytes())


def test_nemotron_stream_language_keeps_locale_prompt() -> None:
    model = next(
        model
        for model in DEFAULT_CATALOG
        if model.id == "sherpa-onnx:nemotron-3.5-asr-streaming-0.6b-320ms-int8"
    )
    builder = _SherpaRecognizerBuilder(None, model, threads=1)

    class Stream:
        def __init__(self) -> None:
            self.options: dict[str, str] = {}

        def set_option(self, name: str, value: str) -> None:
            self.options[name] = value

    stream = Stream()
    mapped = builder.stream_language("de-DE")
    _set_stream_language(stream, mapped, preserve_locale=builder.uses_stream_language_locale())

    assert mapped == "de-DE"
    assert stream.options == {"language": "de-DE"}


async def test_sherpa_keeps_one_recognizer_loaded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    catalog_model = _catalog()
    root = _model_root(tmp_path, catalog_model)
    audio = tmp_path / "audio.wav"
    _wave(audio)
    constructions: list[dict[str, object]] = []

    class TranscriptionResult:
        text = " persistent sherpa result "

    class Stream:
        def accept_waveform(self, sample_rate: int, samples: list[float]) -> None:
            assert sample_rate == NORMALIZED_SAMPLE_RATE_HZ
            assert len(samples) == 3

    Stream.result = TranscriptionResult()

    class Recognizer:
        @classmethod
        def from_sense_voice(cls, **kwargs: object) -> Recognizer:
            constructions.append(kwargs)
            return cls()

        def create_stream(self) -> Stream:
            return Stream()

        def decode_stream(self, stream: Stream) -> None:
            assert isinstance(stream, Stream)

    module = types.ModuleType(SHERPA_ONNX_MODULE)
    module.OfflineRecognizer = Recognizer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, SHERPA_ONNX_MODULE, module)
    monkeypatch.setattr(
        IMPORTLIB_FIND_SPEC_PATH,
        lambda _: machinery.ModuleSpec(SHERPA_ONNX_MODULE, loader=None),
    )

    engine = SherpaOnnxEngine(root, catalog_model, cpu_threads=2)
    first = await engine.transcribe(audio, TranscriptionOptions("en-US", "raw"))
    second = await engine.transcribe(audio, TranscriptionOptions("auto", "raw"))

    assert isinstance(first, EngineTranscription)
    assert first.text == "persistent sherpa result"
    assert second.model_load_ms == 0
    assert len(constructions) == 1
    assert constructions[0]["num_threads"] == 2
    assert constructions[0]["use_itn"] is True


def _fake_recognizer_module(
    factory_name: str,
    constructions: list[dict[str, object]],
    monkeypatch: pytest.MonkeyPatch,
    *,
    attr_name: str = "OfflineRecognizer",
) -> None:
    class Recognizer:
        """Minimal recognizer stub for constructor assertions."""

    def factory(cls: type[Recognizer], **kwargs: object) -> Recognizer:
        constructions.append(kwargs)
        return cls()

    setattr(Recognizer, factory_name, classmethod(factory))
    module = types.ModuleType(SHERPA_ONNX_MODULE)
    setattr(module, attr_name, Recognizer)
    monkeypatch.setitem(sys.modules, SHERPA_ONNX_MODULE, module)


def test_sherpa_nemo_ctc_loads_with_its_own_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    catalog_model = _catalog("nemo_ctc")
    root = _model_root(tmp_path, catalog_model)
    constructions: list[dict[str, object]] = []
    _fake_recognizer_module("from_nemo_ctc", constructions, monkeypatch)
    monkeypatch.setattr(
        IMPORTLIB_FIND_SPEC_PATH,
        lambda _: machinery.ModuleSpec(SHERPA_ONNX_MODULE, loader=None),
    )

    SherpaOnnxEngine(root, catalog_model)._load_recognizer_sync()

    assert len(constructions) == 1
    assert constructions[0]["model"] == str(root / SHERPA_MODEL_FILE)
    assert constructions[0][TOKENS_COMPONENT] == str(root / TOKENS_FILE)


def test_sherpa_nemo_canary_loads_english_o_aa(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    catalog_model = _catalog("nemo_canary")
    root = _model_root(tmp_path, catalog_model)
    constructions: list[dict[str, object]] = []
    _fake_recognizer_module("from_nemo_canary", constructions, monkeypatch)
    monkeypatch.setattr(
        IMPORTLIB_FIND_SPEC_PATH,
        lambda _: machinery.ModuleSpec(SHERPA_ONNX_MODULE, loader=None),
    )

    SherpaOnnxEngine(root, catalog_model)._load_recognizer_sync()

    assert len(constructions) == 1
    assert constructions[0][ENCODER_COMPONENT] == str(root / ENCODER_INT8_FILE)
    assert constructions[0][DECODER_COMPONENT] == str(root / DECODER_INT8_FILE)
    assert constructions[0][TOKENS_COMPONENT] == str(root / TOKENS_FILE)
    # No src_lang/tgt_lang override: sherpa-onnx defaults both to "en", matching the
    # English-only catalog entry this project currently ships for Canary.
    assert "src_lang" not in constructions[0]
    assert "tgt_lang" not in constructions[0]


def test_sherpa_nemo_transducer_uses_each_f_bf636(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """GigaAM's RNNT export ships an INT8 encoder but a non-quantized decoder/joiner."""
    catalog_model = _catalog(
        "nemo_transducer",
        required_files=(ENCODER_INT8_FILE, DECODER_FILE, JOINER_FILE, TOKENS_FILE),
    )
    root = _model_root(tmp_path, catalog_model)
    constructions: list[dict[str, object]] = []
    _fake_recognizer_module("from_transducer", constructions, monkeypatch)
    monkeypatch.setattr(
        IMPORTLIB_FIND_SPEC_PATH,
        lambda _: machinery.ModuleSpec(SHERPA_ONNX_MODULE, loader=None),
    )

    SherpaOnnxEngine(root, catalog_model)._load_recognizer_sync()

    assert constructions[0][ENCODER_COMPONENT] == str(root / ENCODER_INT8_FILE)
    assert constructions[0][DECODER_COMPONENT] == str(root / DECODER_FILE)
    assert constructions[0]["joiner"] == str(root / JOINER_FILE)


def test_sherpa_dolphin_loads_a_single_file_ctc(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for model_type, factory in (("dolphin_ctc", "from_dolphin_ctc"),):
        catalog_model = _catalog(model_type, required_files=(SHERPA_MODEL_FILE, TOKENS_FILE))
        root = _model_root(tmp_path, catalog_model)
        constructions: list[dict[str, object]] = []
        _fake_recognizer_module(factory, constructions, monkeypatch)
        monkeypatch.setattr(
            IMPORTLIB_FIND_SPEC_PATH,
            lambda _: machinery.ModuleSpec(SHERPA_ONNX_MODULE, loader=None),
        )

        SherpaOnnxEngine(root, catalog_model)._load_recognizer_sync()

        assert constructions[0]["model"] == str(root / SHERPA_MODEL_FILE)
        assert constructions[0][TOKENS_COMPONENT] == str(root / TOKENS_FILE)


def test_sherpa_qwen3_asr_loads_a_tokenizer_fe25a(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """This family takes a Hugging Face tokenizer folder, not a `tokens.txt`."""
    catalog_model = _catalog(
        "qwen3_asr",
        required_files=(
            "conv_frontend.onnx",
            ENCODER_INT8_FILE,
            DECODER_INT8_FILE,
            "tokenizer/vocab.json",
            "tokenizer/merges.txt",
            "tokenizer/tokenizer_config.json",
        ),
    )
    root = _model_root(tmp_path, catalog_model)
    constructions: list[dict[str, object]] = []
    _fake_recognizer_module("from_qwen3_asr", constructions, monkeypatch)
    monkeypatch.setattr(
        IMPORTLIB_FIND_SPEC_PATH,
        lambda _: machinery.ModuleSpec(SHERPA_ONNX_MODULE, loader=None),
    )

    SherpaOnnxEngine(root, catalog_model)._load_recognizer_sync()

    assert constructions[0]["conv_frontend"] == str(root / "conv_frontend.onnx")
    assert constructions[0][ENCODER_COMPONENT] == str(root / ENCODER_INT8_FILE)
    assert constructions[0][DECODER_COMPONENT] == str(root / DECODER_INT8_FILE)
    assert constructions[0]["tokenizer"] == str(root / "tokenizer")
    assert TOKENS_COMPONENT not in constructions[0]


def test_streaming_zipformer_loads_via_onli_aaa(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    catalog_model = _catalog(
        "streaming_zipformer",
        required_files=(ENCODER_FILE, DECODER_FILE, JOINER_FILE, TOKENS_FILE),
    )
    root = _model_root(tmp_path, catalog_model)
    constructions: list[dict[str, object]] = []
    _fake_recognizer_module(
        "from_transducer", constructions, monkeypatch, attr_name="OnlineRecognizer"
    )
    monkeypatch.setattr(
        IMPORTLIB_FIND_SPEC_PATH,
        lambda _: machinery.ModuleSpec(SHERPA_ONNX_MODULE, loader=None),
    )

    SherpaOnnxEngine(root, catalog_model)._load_recognizer_sync()

    assert constructions[0][ENCODER_COMPONENT] == str(root / ENCODER_FILE)
    assert constructions[0][DECODER_COMPONENT] == str(root / DECODER_FILE)
    assert constructions[0]["joiner"] == str(root / JOINER_FILE)
    assert constructions[0][TOKENS_COMPONENT] == str(root / TOKENS_FILE)
    assert constructions[0]["enable_endpoint_detection"] is True


async def test_create_stream_wraps_the_recognizer_aaaa(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    catalog_model = _catalog(
        "streaming_zipformer",
        required_files=(ENCODER_FILE, DECODER_FILE, JOINER_FILE, TOKENS_FILE),
    )
    root = _model_root(tmp_path, catalog_model)

    class Recognizer:
        @classmethod
        def from_transducer(cls, **kwargs: object) -> Recognizer:
            return cls()

        def create_stream(self) -> object:
            return object()

    module = types.ModuleType(SHERPA_ONNX_MODULE)
    module.OnlineRecognizer = Recognizer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, SHERPA_ONNX_MODULE, module)
    monkeypatch.setattr(
        IMPORTLIB_FIND_SPEC_PATH,
        lambda _: machinery.ModuleSpec(SHERPA_ONNX_MODULE, loader=None),
    )

    engine = SherpaOnnxEngine(root, catalog_model)
    adapter = await engine.create_stream()

    assert isinstance(adapter, _SherpaOnnxStreamAdapter)
    assert adapter._stream is not None


def test_every_shipped_sherpa_model_type_ha_aaaaa(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A new catalog entry must not reach users before its engine branch exists."""
    from app.catalog import DEFAULT_CATALOG, ENGINE_SHERPA_ONNX

    class Recognizer:
        def __getattr__(self, name: str) -> object:
            raise AttributeError(name)

    def make_module() -> types.ModuleType:
        module = types.ModuleType(SHERPA_ONNX_MODULE)

        class Any_:
            def __getattr__(self, name: str) -> object:
                return lambda **kwargs: Recognizer()

            def __call__(self, **kwargs: object) -> Recognizer:
                return Recognizer()

        module.OfflineRecognizer = Any_()  # type: ignore[attr-defined]
        module.OnlineRecognizer = Any_()  # type: ignore[attr-defined]
        return module

    monkeypatch.setitem(sys.modules, SHERPA_ONNX_MODULE, make_module())
    monkeypatch.setattr(
        IMPORTLIB_FIND_SPEC_PATH,
        lambda _: machinery.ModuleSpec(SHERPA_ONNX_MODULE, loader=None),
    )

    shipped = [model for model in DEFAULT_CATALOG if model.engine == ENGINE_SHERPA_ONNX]
    assert shipped, "expected sherpa-onnx entries in the default catalog"
    for model in shipped:
        root = _model_root(tmp_path, model)
        SherpaOnnxEngine(root, model)._load_recognizer_sync()


def test_sherpa_rejects_unknown_model_type(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    catalog_model = _catalog("not_a_real_engine", required_files=(TOKENS_FILE,))
    root = _model_root(tmp_path, catalog_model)
    monkeypatch.setitem(sys.modules, SHERPA_ONNX_MODULE, types.ModuleType(SHERPA_ONNX_MODULE))

    with pytest.raises(EngineUnavailableError, match="Unsupported sherpa-onnx model type"):
        SherpaOnnxEngine(root, catalog_model)._load_recognizer_sync()


async def test_sherpa_rejects_unsupported_language(tmp_path: Path) -> None:
    catalog_model = _catalog()
    engine = SherpaOnnxEngine(_model_root(tmp_path, catalog_model), catalog_model)

    # The specific subclass, not just the base: the API layer keys the
    # `language_unsupported` code off it, and the clients key Retry off that code.
    with pytest.raises(LanguageUnsupportedError, match="does not support es"):
        await engine.transcribe(
            tmp_path / "unused.wav",
            TranscriptionOptions("es", "casual"),
        )


def test_decode_wave_online_reads_result_fr_a(tmp_path: Path) -> None:
    from app.models.sherpa_onnx import _decode_wave_online

    audio = tmp_path / "audio.wav"
    _wave(audio)

    class FakeStream:
        def __init__(self) -> None:
            self.waveform: tuple[int, list[float]] | None = None
            self.finished = False

        def accept_waveform(self, sample_rate: int, samples: list[float]) -> None:
            self.waveform = (sample_rate, samples)

        def input_finished(self) -> None:
            self.finished = True

    class FakeRecognizer:
        def __init__(self) -> None:
            self.stream = FakeStream()
            self.ready_calls = 0

        def create_stream(self) -> FakeStream:
            return self.stream

        def is_ready(self, stream: FakeStream) -> bool:
            self.ready_calls += 1
            return self.ready_calls == 1

        def decode_stream(self, stream: FakeStream) -> None:
            pass

        def get_result(self, stream: FakeStream) -> str:
            # OnlineRecognizer.get_result returns a plain str, unlike the
            # offline result objects, which expose `.text`.
            return " final streaming text "

    recognizer = FakeRecognizer()
    text = _decode_wave_online(recognizer, audio, _LanguagePolicy())

    assert text == "final streaming text"
    assert recognizer.stream.finished is True
    assert recognizer.stream.waveform is not None


def test_decode_wave_online_sets_optional_language_option(tmp_path: Path) -> None:
    from app.models.sherpa_onnx import _decode_wave_online

    audio = tmp_path / "audio.wav"
    _wave(audio)

    class FakeStream:
        def __init__(self) -> None:
            self.options: dict[str, str] = {}

        def has_option(self, name: str) -> bool:
            return name == "language"

        def set_option(self, name: str, value: str) -> None:
            self.options[name] = value

        def accept_waveform(self, sample_rate: int, samples: list[float]) -> None:
            pass

        def input_finished(self) -> None:
            pass

    class FakeRecognizer:
        def __init__(self) -> None:
            self.stream = FakeStream()

        def create_stream(self) -> FakeStream:
            return self.stream

        def is_ready(self, stream: FakeStream) -> bool:
            return False

        def get_result(self, stream: FakeStream) -> str:
            return "text"

    recognizer = FakeRecognizer()
    assert _decode_wave_online(recognizer, audio, _LanguagePolicy("de-DE")) == "text"
    assert recognizer.stream.options == {"language": "de"}


def test_supports_streaming_only_for_the_st_aa(tmp_path: Path) -> None:
    streaming_model = _catalog(
        "streaming_zipformer",
        required_files=(ENCODER_FILE, DECODER_FILE, JOINER_FILE, TOKENS_FILE),
    )
    batch_model = _catalog(SENSE_VOICE_MODEL_TYPE)

    streaming_engine = SherpaOnnxEngine(_model_root(tmp_path, streaming_model), streaming_model)
    batch_engine = SherpaOnnxEngine(_model_root(tmp_path, batch_model), batch_model)
    no_model_engine = SherpaOnnxEngine(None, None)

    assert streaming_engine.supports_streaming is True
    assert batch_engine.supports_streaming is False
    assert no_model_engine.supports_streaming is False


async def test_create_stream_rejects_a_non_stream_aaa(tmp_path: Path) -> None:
    catalog_model = _catalog(SENSE_VOICE_MODEL_TYPE)
    engine = SherpaOnnxEngine(_model_root(tmp_path, catalog_model), catalog_model)

    with pytest.raises(EngineUnavailableError, match="does not stream"):
        await engine.create_stream()


class _FakeOnlineStream:
    def __init__(self) -> None:
        self.waveforms: list[tuple[int, list[float]]] = []
        self.finished = False

    def accept_waveform(self, sample_rate: int, samples: list[float]) -> None:
        self.waveforms.append((sample_rate, list(samples)))

    def input_finished(self) -> None:
        self.finished = True


class _FakeOnlineRecognizer:
    """Lets a test script exactly what one `is_ready`/`decode_stream` drain sees."""

    def __init__(self) -> None:
        self.ready_remaining = 0
        self.endpoint = False
        self.text = ""
        self.decode_calls = 0
        self.reset_calls = 0

    def is_ready(self, stream: object) -> bool:
        if self.ready_remaining > 0:
            self.ready_remaining -= 1
            return True
        return False

    def decode_stream(self, stream: object) -> None:
        self.decode_calls += 1

    def is_endpoint(self, stream: object) -> bool:
        return self.endpoint

    def get_result(self, stream: object) -> str:
        # OnlineRecognizer.get_result returns a plain str, unlike the offline
        # result objects, which expose `.text`.
        return self.text

    def reset(self, stream: object) -> None:
        self.reset_calls += 1
        self.text = ""
        self.endpoint = False


def test_stream_adapter_reports_partial_the_aaaa() -> None:
    recognizer = _FakeOnlineRecognizer()
    stream = _FakeOnlineStream()
    adapter = _SherpaOnnxStreamAdapter(recognizer, stream)
    events: list[object] = []
    adapter.add_listener(events.append)

    recognizer.ready_remaining = 1
    recognizer.text = EXPECTED_TRANSCRIPT
    adapter.add_audio([0.1, 0.2], NORMALIZED_SAMPLE_RATE_HZ)
    assert len(events) == 1
    assert events[0].line.text == EXPECTED_TRANSCRIPT
    assert events[0].line.line_id == 0

    recognizer.ready_remaining = 1
    recognizer.text = "hello world"
    recognizer.endpoint = True
    adapter.add_audio([0.3, 0.4], NORMALIZED_SAMPLE_RATE_HZ)
    assert len(events) == 2
    assert events[1].line.text == "hello world"
    assert events[1].line.line_id == 0
    assert recognizer.reset_calls == 1

    operation_result = adapter.stop()
    assert [(line.line_id, line.text) for line in operation_result.lines] == [(0, "hello world")]
    assert stream.finished is True


def test_nemotron_stream_adapter_strips_language_tag() -> None:
    recognizer = _FakeOnlineRecognizer()
    stream = _FakeOnlineStream()
    adapter = _SherpaOnnxStreamAdapter(recognizer, stream, language_mapper=lambda text: text)
    events: list[object] = []
    adapter.add_listener(events.append)

    recognizer.text = "Hello world. <en-US>"
    adapter.add_audio([0.1], NORMALIZED_SAMPLE_RATE_HZ)
    result = adapter.stop()

    assert events[0].line.text == "Hello world."
    assert result.lines[0].text == "Hello world."
    assert adapter.detected_language == "en-US"
    assert result.detected_language == "en-US"


def test_stream_adapter_starts_a_new_line_a_aaaaa() -> None:
    recognizer = _FakeOnlineRecognizer()
    stream = _FakeOnlineStream()
    adapter = _SherpaOnnxStreamAdapter(recognizer, stream)

    recognizer.ready_remaining = 1
    recognizer.text = "first segment"
    recognizer.endpoint = True
    adapter.add_audio([0.1], NORMALIZED_SAMPLE_RATE_HZ)

    recognizer.ready_remaining = 1
    recognizer.text = "second segment"
    recognizer.endpoint = True
    adapter.add_audio([0.2], NORMALIZED_SAMPLE_RATE_HZ)

    operation_result = adapter.stop()
    assert [(line.line_id, line.text) for line in operation_result.lines] == [
        (0, "first segment"),
        (1, "second segment"),
    ]


def test_stream_adapter_does_not_repeat_ide_a() -> None:
    recognizer = _FakeOnlineRecognizer()
    stream = _FakeOnlineStream()
    adapter = _SherpaOnnxStreamAdapter(recognizer, stream)
    events: list[object] = []
    adapter.add_listener(events.append)

    recognizer.ready_remaining = 1
    recognizer.text = EXPECTED_TRANSCRIPT
    adapter.add_audio([0.1], NORMALIZED_SAMPLE_RATE_HZ)
    recognizer.ready_remaining = 1
    recognizer.text = EXPECTED_TRANSCRIPT  # unchanged
    adapter.add_audio([0.2], NORMALIZED_SAMPLE_RATE_HZ)

    assert len(events) == 1


@pytest.mark.parametrize("channels", [1, 2])
def test_both_waveform_paths_agree(channels: int, tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """numpy only arrives with the `engines` extra, which CI does not install.

    The vectorized path and the stdlib fallback have to hand sherpa-onnx the
    same waveform, or the engine transcribes different audio depending on how
    the gateway happened to be installed.
    """
    import builtins

    from app.models.sherpa_onnx import _read_wave_samples

    audio = tmp_path / "clip.wav"
    frames = bytes(range(256)) * 8
    with wave.open(str(audio), "wb") as writer:
        writer.setnchannels(channels)
        writer.setsampwidth(2)
        writer.setframerate(16_000)
        writer.writeframes(frames)

    rate, vectorized = _read_wave_samples(audio)
    assert rate == 16_000

    real_import = builtins.__import__

    def without_numpy(name: str, *arguments: object, **keywords: object) -> object:
        if name == "numpy":
            raise ImportError(name)
        return real_import(name, *arguments, **keywords)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", without_numpy)
    _, fallback = _read_wave_samples(audio)

    assert list(fallback) == pytest.approx(list(vectorized))
    assert len(fallback) == len(frames) // 2 // channels


async def test_cohere_requires_language_and_sets_it_for_every_recording(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dataclasses import replace

    from app.errors import LanguageUnsupportedError
    from app.models.sherpa_onnx import _decode_wave

    model = replace(_catalog("cohere_transcribe"), language_codes=("en", "fr", "de"))
    engine = SherpaOnnxEngine(tmp_path, model)
    with pytest.raises(LanguageUnsupportedError, match="explicit spoken language"):
        await engine.transcribe(tmp_path / "audio.wav", TranscriptionOptions("auto", "raw"))
    languages = []

    class Stream:
        result = types.SimpleNamespace(text="hello")

        def has_option(self, key):
            return False

        def set_option(self, key, value):
            assert key == "language"
            languages.append(value)

        def accept_waveform(self, rate, samples):
            assert languages

    class Recognizer:
        def create_stream(self):
            return Stream()

        def decode_stream(self, stream):
            pass

    monkeypatch.setattr("app.models.sherpa_onnx._read_wave_samples", lambda _: (16000, []))
    recognizer = Recognizer()
    policy = _LanguagePolicy("fr-FR", set_on_stream=True)
    assert _decode_wave(recognizer, tmp_path / "audio.wav", policy) == "hello"
    assert (
        _decode_wave(recognizer, tmp_path / "audio.wav", _LanguagePolicy("de", set_on_stream=True))
        == "hello"
    )
    assert languages == ["fr", "de"]

    # Every other offline model decoded without the option before Cohere
    # arrived; pinning one that auto-detects would change its transcripts.
    class QuietStream(Stream):
        def set_option(self, key, value):
            raise AssertionError("only a language-pinned model may set a stream option")

        def accept_waveform(self, rate, samples):
            return None

    monkeypatch.setattr(Recognizer, "create_stream", lambda self: QuietStream())
    assert _decode_wave(recognizer, tmp_path / "audio.wav", _LanguagePolicy("de")) == "hello"


async def test_cohere_recognizer_is_rebuilt_when_the_language_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cached recognizer built for French would decode German as French, and a
    fluent wrong-language transcript never trips the empty-result check."""
    from dataclasses import replace

    model = replace(_catalog("cohere_transcribe"), language_codes=("en", "fr", "de"))
    engine = SherpaOnnxEngine(tmp_path, model)
    built: list[str] = []
    monkeypatch.setattr(
        engine._builder, "build", lambda language=None: built.append(language) or object()
    )

    await engine._ensure_recognizer("fr-FR")
    await engine._ensure_recognizer("fr")
    assert built == ["fr-FR"]
    await engine._ensure_recognizer("de")
    assert built == ["fr-FR", "de"]
    assert engine._builder.build_language("auto") == "en"


def test_cohere_builder_uses_external_data_encoder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.catalog import DEFAULT_CATALOG

    model = next(
        entry for entry in DEFAULT_CATALOG if entry.key == "cohere-transcribe-14-lang-int8"
    )
    root = _model_root(tmp_path, model)
    calls = []
    _fake_recognizer_module("from_cohere_transcribe", calls, monkeypatch)
    SherpaOnnxEngine(root, model)._load_recognizer_sync()
    assert calls[0]["encoder"] == str(root / "encoder.int8.onnx")
    assert (root / "encoder.int8.onnx.data").is_file()
