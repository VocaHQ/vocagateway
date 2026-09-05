from __future__ import annotations

import asyncio
import json
import os
import wave
from pathlib import Path

import pytest

from app.admin_queries import TRANSCRIBE_DOCS_URL, TRANSCRIBE_INSTALL_HINT, _EngineRuntimes
from app.catalog import DEFAULT_CATALOG
from app.config import Settings
from app.errors import EngineUnavailableError, LanguageUnsupportedError, TranscriptionProcessError
from app.fragments.models import models_list_fragment
from app.models.base import TranscriptionOptions
from app.models.transcribe_cpp import TranscribeCppEngine, _execute
from app.runtime_config import AUTO_ENGINE, VALID_ENGINES
from app.schemas import AdminModelEntry, DependencyStatus
from app.system import SystemInfo


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
    monkeypatch.setattr(
        "app.models.transcribe_cpp.system.resolve_binary", lambda _: "/bin/transcribe-cli"
    )

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
    monkeypatch.setattr("app.models.transcribe_cpp.system.resolve_binary", lambda _: None)
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

        async def communicate(self):
            calls.append("communicate")
            raise failure

        async def wait(self):
            calls.append("wait")
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
    assert calls == ["communicate", "kill", "wait"]


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
    monkeypatch.setattr(
        "app.models.transcribe_cpp.system.resolve_binary", lambda _: "/bin/transcribe-cli"
    )

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


def _system(*, transcribe_cli: str | None) -> SystemInfo:
    return SystemInfo(
        os_name="Linux",
        os_version="",
        arch="x86_64",
        chip="",
        ram_gb=16.0,
        is_apple_silicon=False,
        ffmpeg_path="/usr/bin/ffmpeg",
        whisper_cpp_path=None,
        transcribe_cli_path=transcribe_cli,
        whisperkit_cli_path=None,
        handy_installed=False,
        vocamac_installed=False,
        logical_cpus=8,
        effective_cpus=8.0,
        containerized=False,
        accelerators=(),
        cpu_features=(),
    )


def _settings(tmp_path) -> Settings:
    return Settings(
        token="x" * 32,
        data_dir=tmp_path,
        whisper_binary=tmp_path / "whisper",
        whisper_model=tmp_path / "unused",
    )


def test_engine_runtimes_names_the_missing_binary_only_while_it_is_absent(tmp_path):
    absent = _EngineRuntimes(_system(transcribe_cli=None), _settings(tmp_path))
    missing = absent.missing("transcribe.cpp")
    assert missing is not None
    assert missing.name == "transcribe.cpp CLI"
    assert missing.install_hint == TRANSCRIBE_INSTALL_HINT

    # These are the fields a card renders, so the mapping is checked here too.
    assert absent.entry_fields("transcribe.cpp") == {
        "runtime_requirement": "transcribe.cpp CLI",
        "runtime_hint": TRANSCRIBE_INSTALL_HINT,
        "runtime_docs_url": TRANSCRIBE_DOCS_URL,
    }

    present = _EngineRuntimes(_system(transcribe_cli="/opt/transcribe-cli"), _settings(tmp_path))
    assert present.missing("transcribe.cpp") is None
    assert present.entry_fields("transcribe.cpp") == {}
    # The panel and the card read one table, so both agree on the resolved path.
    tile = next(tile for tile in present.tiles() if tile.name == "transcribe.cpp CLI")
    assert tile.available and tile.path == "/opt/transcribe-cli"


def test_every_selectable_engine_has_a_runtime_tile(tmp_path):
    """An engine missing from the table ships a card that can never warn."""
    runtimes = _EngineRuntimes(_system(transcribe_cli=None), _settings(tmp_path))
    assert set(runtimes._tiles) == set(VALID_ENGINES) - {AUTO_ENGINE}
    # One tile per engine, and each names a distinct runtime for the panel.
    names = [tile.name for tile in runtimes.tiles()]
    assert len(names) == len(set(names)) == len(runtimes._tiles)


def test_model_card_warns_before_the_download_but_still_offers_it():
    def card(**runtime) -> str:
        entry = AdminModelEntry(
            id="transcribe.cpp:canary-qwen-2.5b-Q5_K_M.gguf",
            engine="transcribe.cpp",
            label="Canary-Qwen 2.5B Q5",
            size_bytes=1_980_000_000,
            languages="English",
            quality="High",
            family="Canary",
            description="GGUF speech model.",
            source="Handy / transcribe.cpp",
            state="not_installed",
            active=False,
            recommended=False,
            **runtime,
        )
        return models_list_fragment([entry])

    warned = card(
        runtime_requirement="transcribe.cpp CLI",
        runtime_hint=TRANSCRIBE_INSTALL_HINT,
        runtime_docs_url=TRANSCRIBE_DOCS_URL,
    )
    assert "needs transcribe.cpp CLI" in warned
    assert "Not installed yet" in warned
    assert "VOCAGATEWAY_TRANSCRIBE_BINARY" in warned
    assert TRANSCRIBE_DOCS_URL in warned
    assert "Installation instructions" in warned
    # The weights are still correct, so the card must not block the download.
    assert "Download" in warned

    ready = card()
    assert "needs transcribe.cpp CLI" not in ready
    # The old always-on aside warned even once the runtime was installed.
    assert "Installation instructions" not in ready
    assert "Not installed yet" not in ready
    assert "Download" in ready


async def test_missing_runtime_is_reported_as_such_even_with_no_model_selected(
    tmp_path, monkeypatch
):
    """`auto` used to hit the language check first and blame the language."""
    monkeypatch.setattr("app.models.transcribe_cpp.system.resolve_binary", lambda _: None)
    engine = TranscribeCppEngine("absent-cli", None, None)
    with pytest.raises(EngineUnavailableError, match="Install transcribe-cli"):
        await engine.transcribe(tmp_path / "audio.wav", TranscriptionOptions("auto", "raw"))


@pytest.mark.parametrize("setting", ["{home}/bin/transcribe-cli", "~/bin/transcribe-cli"])
async def test_health_accepts_the_same_binaries_the_libraries_panel_does(
    tmp_path, monkeypatch, setting
):
    """The panel said Installed while the engine said not-ready.

    `shutil.which` neither expands `~` nor accepts a file without the execute
    bit, so the two probes disagreed on exactly the binary the setup docs tell
    operators to configure.
    """
    from app.system import resolve_binary

    monkeypatch.setenv("HOME", str(tmp_path))
    binary = tmp_path / "bin" / "transcribe-cli"
    binary.parent.mkdir()
    binary.write_text("#!/bin/sh\n")  # No execute bit, as a fresh copy often has.
    weights = tmp_path / "model.gguf"
    weights.write_bytes(b"weights")

    configured = setting.format(home=tmp_path)
    assert resolve_binary(configured) == str(binary)
    assert (await TranscribeCppEngine(configured, weights, None).health()).ready


async def test_process_failures_carry_their_stderr_and_stay_typed(tmp_path, monkeypatch):
    weights = tmp_path / "model.gguf"
    weights.write_bytes(b"weights")
    audio = tmp_path / "audio.wav"
    _wave(audio)
    monkeypatch.setattr(
        "app.models.transcribe_cpp.system.resolve_binary", lambda _: "/bin/transcribe-cli"
    )
    engine = TranscribeCppEngine("transcribe-cli", weights, _model())
    options = TranscriptionOptions("en", "raw")

    class Failed:
        returncode = 1

        async def communicate(self):
            return b"", b"error: unsupported GGUF version 4\n"

    monkeypatch.setattr(asyncio, "create_subprocess_exec", lambda *a, **k: _async(Failed()))
    with pytest.raises(TranscriptionProcessError, match="unsupported GGUF version 4"):
        await engine.transcribe(audio, options)

    # A binary that vanished between health() and exec is unavailable, not a 500.
    def missing(*args, **kwargs):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", missing)
    with pytest.raises(EngineUnavailableError, match="Could not run"):
        await engine.transcribe(audio, options)

    # A recording the WAV reader cannot parse is an engine failure, not a crash.
    audio.write_bytes(b"not a riff file")
    monkeypatch.setattr(asyncio, "create_subprocess_exec", lambda *a, **k: _async(Failed()))
    with pytest.raises(TranscriptionProcessError, match="could not read the recording"):
        await engine.transcribe(audio, options)


async def _async(value):
    return value


def test_batch_results_survive_canonicalised_paths_and_reordering(tmp_path):
    """macOS hands out /var/... paths that a CLI may echo back as /private/var/..."""
    from app.models.transcribe_chunks import read_batch_output

    recordings = [tmp_path / "chunk-0.wav", tmp_path / "chunk-1.wav"]
    output = tmp_path / "results.jsonl"
    records = [
        {"type": "batch_header", "load_ms": 20},
        {"file": f"/private{recordings[1]}", "text": "second"},
        {"file": f"/private{recordings[0]}", "text": "first"},
        {"type": "batch_footer", "total_ms": 90},
    ]
    output.write_text("".join(json.dumps(record) + "\n" for record in records))
    assert read_batch_output(output, recordings) == "first second"

    # A chunk reported twice is still a failure: one of them is not this chunk.
    output.write_text("".join(json.dumps(records[1]) + "\n" for _ in range(2)))
    with pytest.raises(TranscriptionProcessError, match="fully transcribe"):
        read_batch_output(output, recordings)
