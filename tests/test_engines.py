from __future__ import annotations

import time
from pathlib import Path

import pytest
from pytest import MonkeyPatch

from app import engines as engines_module
from app.catalog import CatalogModel
from app.config import Settings
from app.engines import EngineManager
from app.model_manager import ModelManager
from app.models.base import EngineHealth, TranscriptionOptions
from app.models.mlx_audio import MLXAudioEngine
from app.models.sherpa_onnx import SherpaOnnxEngine
from app.runtime_config import RuntimeConfig
from app.system import engine_requirement, engine_runs_on

MODELS_DIRECTORY = "models"
TEST_TOKEN = "test-token-with-at-least-thirty-two-characters"
WHISPER_BINARY_NAME = "whisper-cli"
WHISPER_MODEL_NAME = "whisper.bin"
MISSING_HANDY_BINARY_NAME = "no-handy"
MISSING_VOCAMAC_APP_NAME = "no-vocamac"
IDLE_AFTER_DEADLINE_SECONDS = 601


class ResidentFakeEngine:
    def __init__(self) -> None:
        self.loaded = True
        self.unload_calls = 0

    @property
    def model_is_resident(self) -> bool:
        return self.loaded

    def unload(self) -> None:
        self.loaded = False
        self.unload_calls += 1

    async def health(self) -> EngineHealth:
        return EngineHealth(ready=True, name="resident-fake")

    async def transcribe(self, audio_path: Path, options: TranscriptionOptions) -> str:
        return "unused"


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        token=TEST_TOKEN,
        data_dir=tmp_path,
        whisper_binary=tmp_path / WHISPER_BINARY_NAME,
        whisper_model=tmp_path / WHISPER_MODEL_NAME,
        handy_binary=tmp_path / MISSING_HANDY_BINARY_NAME,
        vocamac_app=tmp_path / MISSING_VOCAMAC_APP_NAME,
    )


def _moonshine_catalog() -> CatalogModel:
    return CatalogModel(
        id="moonshine:test",
        engine="moonshine",
        key="test",
        label="Moonshine test",
        size_bytes=1,
        languages="English only",
        quality="Fast",
        minimum_ram_gb=1,
        marker_file=".vocagateway-model.json",
        language_code="en",
    )


VOCAMAC_ENGINE = "vocamac"
AUTO_ENGINE = "auto"


@pytest.mark.parametrize(
    ("catalog_model", "expected_type", "config_field"),
    [
        (
            CatalogModel(
                id="sherpa-onnx:test",
                engine="sherpa-onnx",
                key="test",
                label="Sherpa test",
                size_bytes=1,
                languages="English only",
                quality="Fast",
                minimum_ram_gb=1,
                marker_file=".vocagateway-model.json",
                required_files=("model.int8.onnx", "tokens.txt"),
                model_type="sense_voice",
            ),
            SherpaOnnxEngine,
            "sherpa_model",
        ),
        (
            CatalogModel(
                id="mlx-audio:test",
                engine="mlx-audio",
                key="test",
                label="MLX test",
                size_bytes=1,
                languages="Multilingual",
                quality="Fast",
                minimum_ram_gb=1,
                marker_file="model.safetensors",
                apple_silicon_only=True,
            ),
            MLXAudioEngine,
            "mlx_audio_model",
        ),
    ],
)
def test_model_selection_builds_new_engine_aa(
    tmp_path: Path,
    catalog_model: CatalogModel,
    expected_type: type[SherpaOnnxEngine] | type[MLXAudioEngine],
    config_field: str,
) -> None:
    manager = ModelManager(tmp_path / MODELS_DIRECTORY, catalog=(catalog_model,))
    manager.model_path(catalog_model).mkdir(parents=True)
    if catalog_model.marker_file:
        (manager.model_path(catalog_model) / catalog_model.marker_file).write_bytes(b"model")
    runtime = RuntimeConfig()
    settings = _settings(tmp_path)
    config_path = tmp_path / "config.json"
    engines = EngineManager(settings, runtime, config_path, manager)

    engines.select_model(catalog_model.id)

    assert isinstance(engines.current(), expected_type)
    assert runtime.engine == catalog_model.engine
    assert getattr(runtime, config_field) == catalog_model.id
    assert getattr(RuntimeConfig.load(config_path), config_field) == catalog_model.id


async def test_idle_offload_waits_for_active_model_lease(tmp_path: Path) -> None:
    runtime = RuntimeConfig(idle_offload_enabled=True, idle_offload_minutes=10)
    manager = EngineManager(
        _settings(tmp_path),
        runtime,
        tmp_path / "config.json",
        ModelManager(tmp_path / MODELS_DIRECTORY),
    )
    resident = ResidentFakeEngine()
    manager._engine = resident
    after_deadline = time.monotonic() + IDLE_AFTER_DEADLINE_SECONDS

    async with manager.lease():
        assert manager.offload_if_idle(now=after_deadline) is False

    after_lease_deadline = time.monotonic() + IDLE_AFTER_DEADLINE_SECONDS
    assert manager.offload_if_idle(now=after_lease_deadline) is True
    assert resident.unload_calls == 1
    assert manager.model_is_offloaded is True

    async with manager.lease():
        resident.loaded = True

    assert manager.model_is_offloaded is False


def test_whisper_model_selection_keeps_catalog_output_contract(tmp_path: Path) -> None:
    catalog_model = CatalogModel(
        id="whisper.cpp:hinglish",
        engine="whisper.cpp",
        key="hinglish.bin",
        label="Hinglish",
        size_bytes=1,
        languages="Hindi + English, Roman script",
        quality="Experimental",
        minimum_ram_gb=4,
        language_codes=("hinglish_roman",),
        decoder_language_code="hi",
    )
    manager = ModelManager(tmp_path / MODELS_DIRECTORY, catalog=(catalog_model,))
    model_path = manager.model_path(catalog_model)
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(b"model")
    settings = _settings(tmp_path)
    engines = EngineManager(settings, RuntimeConfig(), tmp_path / "config.json", manager)

    engines.select_model(catalog_model.id)

    assert getattr(engines.current(), "catalog_model", None) == catalog_model


@pytest.mark.parametrize(
    ("engine", "linux", "intel_mac", "apple_silicon"),
    [
        (VOCAMAC_ENGINE, False, False, True),
        ("handy", False, True, True),
        ("whisperkit", False, True, True),
        ("mlx-audio", False, False, True),
        ("sherpa-onnx", True, True, True),
        ("whisper.cpp", True, True, True),
        (AUTO_ENGINE, True, True, True),
    ],
)
def test_desktop_and_apple_engines_only_run_aaa(
    engine: str, linux: bool, intel_mac: bool, apple_silicon: bool
) -> None:
    assert engine_runs_on(engine, is_mac=False, is_apple_silicon=False) is linux
    assert engine_runs_on(engine, is_mac=True, is_apple_silicon=False) is intel_mac
    assert engine_runs_on(engine, is_mac=True, is_apple_silicon=True) is apple_silicon


def test_configure_rejects_an_engine_the_ho_db78a(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    settings = Settings(
        token=TEST_TOKEN,
        data_dir=tmp_path,
        whisper_binary=tmp_path / WHISPER_BINARY_NAME,
        whisper_model=tmp_path / WHISPER_MODEL_NAME,
        handy_binary=tmp_path / MISSING_HANDY_BINARY_NAME,
        vocamac_app=tmp_path / MISSING_VOCAMAC_APP_NAME,
    )
    runtime = RuntimeConfig()
    config_path = tmp_path / "config.json"
    engines = EngineManager(
        settings, runtime, config_path, ModelManager(tmp_path / MODELS_DIRECTORY)
    )
    monkeypatch.setattr(engines_module, "engine_runs_here", lambda engine: engine != VOCAMAC_ENGINE)

    with pytest.raises(ValueError, match="runs only on Apple silicon"):
        engines.set_engine(VOCAMAC_ENGINE)

    assert runtime.engine == AUTO_ENGINE
    assert engine_requirement(VOCAMAC_ENGINE) == "Apple silicon"


def test_build_engine_honours_forced_settin_aaaa(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """VOCAGATEWAY_ENGINE must win over a persisted runtime config of 'auto'."""
    from app.engines import build_engine
    from app.models.whisper_cpp import WhisperCppEngine

    settings = Settings(
        token=TEST_TOKEN,
        data_dir=tmp_path,
        engine="whisper.cpp",
        whisper_binary=tmp_path / WHISPER_BINARY_NAME,
        whisper_model=tmp_path / WHISPER_MODEL_NAME,
        handy_binary=tmp_path / MISSING_HANDY_BINARY_NAME,
        vocamac_app=tmp_path / MISSING_VOCAMAC_APP_NAME,
    )
    runtime = RuntimeConfig(engine=AUTO_ENGINE)
    manager = ModelManager(tmp_path / MODELS_DIRECTORY)

    engine = build_engine(settings, runtime, manager)

    assert isinstance(engine, WhisperCppEngine)


def test_forget_if_active_clears_moonshine_aaaaa(tmp_path: Path) -> None:
    """Reproduces the reported bug: deleting the active Moonshine model reset
    `runtime_config.engine` back to AUTO_ENGINE but left `moonshine_model` pointing at
    the now-deleted id — unlike the sherpa-onnx and mlx-audio branches of the same
    method, which null out both fields together."""
    manager = ModelManager(tmp_path / MODELS_DIRECTORY, catalog=(_moonshine_catalog(),))
    manager.model_path(_moonshine_catalog()).mkdir(parents=True)
    (manager.model_path(_moonshine_catalog()) / ".vocagateway-model.json").write_bytes(b"model")
    runtime = RuntimeConfig()
    settings = _settings(tmp_path)
    config_path = tmp_path / "config.json"
    engines = EngineManager(settings, runtime, config_path, manager)

    engines.select_model("moonshine:test")
    assert runtime.engine == "moonshine"
    assert runtime.moonshine_model == "moonshine:test"

    engines.forget_if_active("moonshine:test")

    assert runtime.engine == AUTO_ENGINE
    # moonshine_model is typed `str`, not `str | None` (unlike sherpa_model and
    # mlx_audio_model): moonshine:en is a permanent catalog entry kept exactly as
    # this fallback, so the deleted id must not linger as a dangling reference,
    # in memory or on disk.
    assert runtime.moonshine_model == "moonshine:en"
    assert RuntimeConfig.load(config_path).moonshine_model == "moonshine:en"


def test_select_engine_accepts_sherpa_and_mlx(tmp_path: Path) -> None:
    from app.main import select_engine

    for name in ("sherpa-onnx", "mlx-audio", "moonshine", "faster-whisper"):
        settings = Settings(
            token=TEST_TOKEN,
            data_dir=tmp_path,
            engine=name,
            whisper_binary=tmp_path / WHISPER_BINARY_NAME,
            whisper_model=tmp_path / WHISPER_MODEL_NAME,
            handy_binary=tmp_path / MISSING_HANDY_BINARY_NAME,
            vocamac_app=tmp_path / MISSING_VOCAMAC_APP_NAME,
        )
        # Resolution may fail later if models are missing, but the engine id
        # itself must not be rejected up front the way the old allow-list did.
        try:
            select_engine(settings)
        except RuntimeError as error:
            assert "not a supported engine" not in str(error)
        except Exception:
            # Missing model/binary paths are fine for this test.
            pass
