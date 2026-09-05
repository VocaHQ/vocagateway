from __future__ import annotations

import sys
import types
from importlib import machinery
from pathlib import Path

import pytest

from app.catalog import CatalogModel
from app.errors import LanguageUnsupportedError
from app.models.base import EngineTranscription, TranscriptionOptions
from app.models.mlx_audio import MLXAudioEngine

ENGLISH_LANGUAGE_CODE = "en"


async def test_mlx_audio_keeps_one_model_loaded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "mlx-model"
    root.mkdir()
    (root / "model.safetensors").write_bytes(b"model")
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"audio")
    constructions: list[Path] = []

    class TranscriptionResult:
        text = " persistent mlx result "

    class Model:
        def generate(
            self, audio_path: str, *, language: str = ENGLISH_LANGUAGE_CODE
        ) -> TranscriptionResult:
            assert audio_path == str(audio)
            assert language == ENGLISH_LANGUAGE_CODE
            return TranscriptionResult()

    def load(path: Path) -> Model:
        constructions.append(path)
        return Model()

    module = types.ModuleType("mlx_audio")
    stt_module = types.ModuleType("mlx_audio.stt")
    utils_module = types.ModuleType("mlx_audio.stt.utils")
    utils_module.load = load  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mlx_audio", module)
    monkeypatch.setitem(sys.modules, "mlx_audio.stt", stt_module)
    monkeypatch.setitem(sys.modules, "mlx_audio.stt.utils", utils_module)
    monkeypatch.setattr(
        "app.models.mlx_audio.importlib_util.find_spec",
        lambda _: machinery.ModuleSpec("mlx_audio", loader=None),
    )
    monkeypatch.setattr("app.models.mlx_audio.platform.system", lambda: "Darwin")
    monkeypatch.setattr("app.models.mlx_audio.platform.machine", lambda: "arm64")
    catalog_model = CatalogModel(
        id="mlx-audio:test",
        engine="mlx-audio",
        key="test",
        label="Test",
        size_bytes=1,
        languages="Multilingual",
        quality="Fast",
        minimum_ram_gb=1,
    )

    engine = MLXAudioEngine(root, catalog_model)
    first = await engine.transcribe(audio, TranscriptionOptions("en-US", "raw"))
    second = await engine.transcribe(audio, TranscriptionOptions(ENGLISH_LANGUAGE_CODE, "raw"))

    assert isinstance(first, EngineTranscription)
    assert first.text == "persistent mlx result"
    assert second.model_load_ms == 0
    assert constructions == [root]


async def test_mlx_rejects_a_language_the_model_c_aa(tmp_path: Path) -> None:
    """English-only MLX entries (Parakeet v2, Granite Speech) must refuse other
    languages with the specific error the API turns into `language_unsupported`."""
    root = tmp_path / "mlx-model"
    root.mkdir()
    (root / "model.safetensors").write_bytes(b"model")
    catalog_model = CatalogModel(
        id="mlx-audio:english-only",
        engine="mlx-audio",
        key="english-only",
        label="Test",
        size_bytes=1,
        languages="English only",
        quality="Fast",
        minimum_ram_gb=1,
        language_codes=(ENGLISH_LANGUAGE_CODE,),
    )

    engine = MLXAudioEngine(root, catalog_model)

    with pytest.raises(LanguageUnsupportedError, match="does not support hi"):
        await engine.transcribe(
            tmp_path / "unused.wav",
            TranscriptionOptions(language="hi", style="raw"),
        )


@pytest.mark.parametrize("requested", ["auto", "hinglish_roman"])
async def test_roman_output_uses_hindi_decoder_and_rejects_script_leakage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, requested: str
) -> None:
    from app.catalog import DEFAULT_CATALOG
    from app.models.base import EngineHealth

    model = next(entry for entry in DEFAULT_CATALOG if entry.key == "hinglish-swift")
    engine = MLXAudioEngine(tmp_path, model)
    calls = []

    class FakeModel:
        text = "Aaj office jaana hai"

        def generate(self, audio: str, *, language: str):
            calls.append(language)
            return types.SimpleNamespace(text=self.text)

    fake = FakeModel()

    async def ready():
        return EngineHealth(ready=True, name="test")

    async def loaded():
        return fake, 0

    monkeypatch.setattr(engine, "health", ready)
    monkeypatch.setattr(engine, "_ensure_model", loaded)
    options = TranscriptionOptions(requested, "raw")
    result = await engine.transcribe(tmp_path / "audio.wav", options)
    assert result.text == fake.text
    assert calls == ["hi"]
    fake.text = "Aaj office में jaana hai"
    with pytest.raises(LanguageUnsupportedError, match="writing system"):
        await engine.transcribe(tmp_path / "audio.wav", options)
    with pytest.raises(LanguageUnsupportedError, match="does not support en"):
        await engine.transcribe(tmp_path / "audio.wav", TranscriptionOptions("en", "raw"))


def test_srota_preserves_literal_language_agnostic_prefix(tmp_path: Path) -> None:
    from app.catalog import DEFAULT_CATALOG
    from app.models.mlx_audio import _generate_text

    model = next(entry for entry in DEFAULT_CATALOG if entry.key == "srota-hinglish")
    engine = MLXAudioEngine(tmp_path, model)

    class FakeModel:
        def generate(self, audio: str, *, language: str):
            assert language == "None"
            return types.SimpleNamespace(text="मेरा office")

    assert _generate_text(FakeModel(), tmp_path / "audio.wav", engine._decoder_language("hi"))


@pytest.mark.parametrize("language", ["auto", "fr-FR"])
def test_granite_language_hint_keeps_transcription(tmp_path: Path, language: str) -> None:
    from app.catalog import DEFAULT_CATALOG
    from app.models.mlx_audio import _generate_text

    model = next(entry for entry in DEFAULT_CATALOG if entry.key == "granite-speech-4.1-2b")
    engine = MLXAudioEngine(tmp_path, model)

    class FakeModel:
        def generate(self, audio: str, *, language: str | None = None, prompt: str | None = None):
            assert language in {None, "fr"}
            assert prompt == "transcribe the speech with proper punctuation and capitalization."
            return types.SimpleNamespace(text="Bonjour")

    assert _generate_text(
        FakeModel(), tmp_path / "audio.wav", language, engine._transcription_prompt()
    )
