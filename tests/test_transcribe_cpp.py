from __future__ import annotations

import asyncio
import json
import os
import wave
from pathlib import Path

import pytest

from app.admin_queries import TRANSCRIBE_INSTALL_HINT
from app.catalog import DEFAULT_CATALOG
from app.errors import EngineUnavailableError, LanguageUnsupportedError, TranscriptionProcessError
from app.models.base import TranscriptionOptions
from app.models.transcribe_cpp import TranscribeCppEngine, _execute
from app.schemas import DependencyStatus


def _model(key: str = "canary-qwen-2.5b-Q5_K_M.gguf"):
    return next(model for model in DEFAULT_CATALOG if model.key == key)


def _wave(path: Path, seconds: int = 1) -> bytes:
    pcm = b"\x01\x00" * (16000 * seconds)
    with wave.open(str(path), "wb") as recording:
        recording.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        recording.writeframes(pcm)
    return pcm


async def test_native_cli_uses_file_output_and_cleans_up(tmp_path, monkeypatch):
    model = _model()
    weights = tmp_path / model.key
    weights.write_bytes(b"weights")
    _wave(tmp_path / "audio.wav")
    seen = []
    monkeypatch.setattr("app.models.transcribe_cpp.shutil.which", lambda _: "/bin/transcribe-cli")

    async def execute(arguments):
        output = Path(arguments[arguments.index("-o") + 1])
        seen.append(output)
        assert arguments[arguments.index("-l") + 1] == "en"
        assert arguments[-1] == str(tmp_path / "audio.wav")
        output.write_text("hello world\n")

    monkeypatch.setattr("app.models.transcribe_cpp._execute", execute)
    engine = TranscribeCppEngine("transcribe-cli", weights, model)
    assert (
        await engine.transcribe(tmp_path / "audio.wav", TranscriptionOptions("auto", "raw"))
        == "hello world"
    )
    assert not seen[0].exists()
    with pytest.raises(LanguageUnsupportedError):
        await engine.transcribe(tmp_path / "audio.wav", TranscriptionOptions("hi", "raw"))


async def test_native_cli_missing_runtime_and_explicit_multilingual_language(tmp_path, monkeypatch):
    monkeypatch.setattr("app.models.transcribe_cpp.shutil.which", lambda _: None)
    engine = TranscribeCppEngine("absent-cli", None, _model())
    with pytest.raises(EngineUnavailableError, match="Install transcribe-cli"):
        await engine.transcribe(tmp_path / "audio.wav", TranscriptionOptions("en", "raw"))
    granite = TranscribeCppEngine("absent-cli", None, _model("granite-speech-4.1-2b-Q5_K_M.gguf"))
    with pytest.raises(LanguageUnsupportedError, match="Choose the spoken language"):
        granite._language("auto")
    assert granite._language("fr-FR") == "fr"


@pytest.mark.parametrize("failure", [TimeoutError, asyncio.CancelledError])
async def test_native_process_is_reaped_on_timeout_or_cancellation(monkeypatch, failure):
    calls = []

    class Process:
        returncode = None

        async def wait(self):
            calls.append("wait")
            if len(calls) == 1:
                raise failure
            self.returncode = -9

        def kill(self):
            calls.append("kill")

    async def spawn(*args, **kwargs):
        return Process()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    expected = (
        asyncio.CancelledError if failure is asyncio.CancelledError else TranscriptionProcessError
    )
    with pytest.raises(expected):
        await _execute(["transcribe-cli"])
    assert calls == ["wait", "kill", "wait"]


def test_native_selection_survives_restart_and_deletion(tmp_path):
    from app.config import Settings
    from app.engines import EngineManager
    from app.model_manager import ModelManager
    from app.runtime_config import RuntimeConfig

    model = _model()
    manager = ModelManager(tmp_path / "models")
    weights = manager.model_path(model)
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"model")
    settings = Settings(
        token="x" * 32,
        data_dir=tmp_path,
        whisper_binary=tmp_path / "whisper",
        whisper_model=tmp_path / "old",
    )
    config_file = tmp_path / "config.json"
    engine_manager = EngineManager(settings, RuntimeConfig(), config_file, manager)
    engine_manager.select_model(model.id)
    restored = RuntimeConfig.load(config_file)
    assert restored.engine == "transcribe.cpp"
    assert restored.transcribe_model == str(weights)
    restarted = EngineManager(settings, restored, config_file, manager)
    assert isinstance(restarted.current(), TranscribeCppEngine)
    restarted.forget_if_active(model.id)
    assert restored.transcribe_model is None
    assert restored.engine == "auto"


@pytest.mark.parametrize("failure", [None, "truncated", "missing"])
async def test_long_audio_keeps_samples_uses_one_process_and_checks_each_result(
    tmp_path, monkeypatch, failure
):
    weights = tmp_path / "model.gguf"
    weights.write_bytes(b"weights")
    audio = tmp_path / "audio.wav"
    original = _wave(audio, 45)
    paths = []
    calls = []
    monkeypatch.setattr("app.models.transcribe_cpp.shutil.which", lambda _: "/bin/transcribe-cli")

    async def execute(arguments, *, stdout):
        calls.append(arguments)
        manifest = Path(arguments[arguments.index("--batch") + 1])
        paths.extend(Path(line) for line in manifest.read_text().splitlines())
        collected = []
        records = []
        for index, path in enumerate(paths):
            with wave.open(str(path), "rb") as recording:
                assert recording.getnframes() <= 20 * 16000
                collected.append(recording.readframes(recording.getnframes()))
            records.append({"file": str(path), "text": f"part {index}"})
        assert b"".join(collected) == original
        if failure == "truncated":
            records[-1]["error"] = "output truncated"
        if failure == "missing":
            records.pop()
        header = {"type": "batch_header", "load_ms": 20}
        os.write(
            stdout, "".join(json.dumps(record) + "\n" for record in [header, *records]).encode()
        )

    monkeypatch.setattr("app.models.transcribe_cpp._execute", execute)
    engine = TranscribeCppEngine("transcribe-cli", weights, _model())
    if failure:
        with pytest.raises(TranscriptionProcessError, match="fully transcribe"):
            await engine.transcribe(audio, TranscriptionOptions("en", "raw"))
    else:
        assert await engine.transcribe(audio, TranscriptionOptions("en", "raw")) == (
            "part 0 part 1 part 2"
        )
    assert len(calls) == 1
    assert all(not path.exists() for path in paths)


def test_switching_to_another_model_clears_native_selection(tmp_path):
    from app.config import Settings
    from app.engines import EngineManager
    from app.model_manager import ModelManager
    from app.runtime_config import RuntimeConfig

    manager = ModelManager(tmp_path / "models")
    native = _model()
    whisper = _model("ggml-tiny.bin")
    for model in (native, whisper):
        path = manager.model_path(model)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"model")
    settings = Settings(
        token="x" * 32,
        data_dir=tmp_path,
        whisper_binary=tmp_path / "whisper",
        whisper_model=tmp_path / "old",
    )
    config = RuntimeConfig()
    engine_manager = EngineManager(settings, config, tmp_path / "config.json", manager)
    engine_manager.select_model(native.id)
    engine_manager.select_model(whisper.id)
    assert config.transcribe_model is None
    engine_manager.forget_if_active(native.id)
    assert config.engine == "whisper.cpp"


def test_overview_names_the_missing_runtime_and_how_to_install_it(tmp_path):
    """A GGUF model is useless without transcribe-cli, so the panel has to name it."""
    from app.config import Settings
    from app.fragments.overview import _OverviewPage
    from app.system import detect_system

    installed = tmp_path / "transcribe-cli"
    installed.write_text("#!/bin/sh\n")
    settings = Settings(
        token="x" * 32,
        data_dir=tmp_path,
        whisper_binary=tmp_path / "whisper",
        whisper_model=tmp_path / "unused",
        transcribe_binary="absent-transcribe-cli",
    )

    def probe(binary: str) -> str | None:
        return detect_system(
            whisper_binary=settings.whisper_binary,
            whisperkit_binary=settings.whisperkit_binary,
            handy_binary=settings.handy_binary,
            vocamac_app=settings.vocamac_app,
            transcribe_binary=binary,
        ).transcribe_cli_path

    assert probe(settings.transcribe_binary) is None
    assert probe(str(installed)) == str(installed)

    panel = _OverviewPage._dependencies_panel(
        [
            DependencyStatus(
                name="transcribe.cpp CLI",
                available=False,
                install_hint=TRANSCRIBE_INSTALL_HINT,
            )
        ]
    )
    assert "transcribe.cpp CLI" in panel
    assert "Missing" in panel
    assert "VOCAGATEWAY_TRANSCRIBE_BINARY" in panel
