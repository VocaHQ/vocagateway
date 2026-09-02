from __future__ import annotations

from pathlib import Path

import pytest
from pytest import MonkeyPatch

from app import engines as engines_module
from app.catalog import CatalogModel
from app.config import Settings
from app.engines import EngineManager
from app.model_manager import ModelManager
from app.models.mlx_audio import MLXAudioEngine
from app.models.sherpa_onnx import SherpaOnnxEngine
from app.runtime_config import RuntimeConfig
from app.system import engine_requirement, engine_runs_on


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
def test_model_selection_builds_new_engine__aa(
    tmp_path: Path,
    catalog_model: CatalogModel,
    expected_type: type[SherpaOnnxEngine] | type[MLXAudioEngine],
    config_field: str,
) -> None:
    manager = ModelManager(tmp_path / "models", catalog=(catalog_model,))
    root = manager.model_path(catalog_model)
    root.mkdir(parents=True)
    if catalog_model.marker_file:
        (root / catalog_model.marker_file).write_bytes(b"model")
    runtime = RuntimeConfig()
    settings = Settings(
        token="test-token-with-at-least-thirty-two-characters",
        data_dir=tmp_path,
        whisper_binary=tmp_path / "whisper-cli",
        whisper_model=tmp_path / "whisper.bin",
        handy_binary=tmp_path / "no-handy",
        vocamac_app=tmp_path / "no-vocamac",
    )
    config_path = tmp_path / "config.json"
    engines = EngineManager(settings, runtime, config_path, manager)

    engines.select_model(catalog_model.id)

    assert isinstance(engines.current(), expected_type)
    assert runtime.engine == catalog_model.engine
    assert getattr(runtime, config_field) == catalog_model.id
    assert getattr(RuntimeConfig.load(config_path), config_field) == catalog_model.id


@pytest.mark.parametrize(
    ("engine", "linux", "intel_mac", "apple_silicon"),
    [
        ("vocamac", False, False, True),
        ("handy", False, True, True),
        ("whisperkit", False, True, True),
        ("mlx-audio", False, False, True),
        ("sherpa-onnx", True, True, True),
        ("whisper.cpp", True, True, True),
        ("auto", True, True, True),
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
        token="test-token-with-at-least-thirty-two-characters",
        data_dir=tmp_path,
        whisper_binary=tmp_path / "whisper-cli",
        whisper_model=tmp_path / "whisper.bin",
        handy_binary=tmp_path / "no-handy",
        vocamac_app=tmp_path / "no-vocamac",
    )
    runtime = RuntimeConfig()
    config_path = tmp_path / "config.json"
    engines = EngineManager(settings, runtime, config_path, ModelManager(tmp_path / "models"))
    monkeypatch.setattr(engines_module, "engine_runs_here", lambda engine: engine != "vocamac")

    with pytest.raises(ValueError, match="runs only on Apple silicon"):
        engines.set_engine("vocamac")

    assert runtime.engine == "auto"
    assert engine_requirement("vocamac") == "Apple silicon"


def test_build_engine_honours_forced_settin_aaaa(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """VOCAGATEWAY_ENGINE must win over a persisted runtime config of 'auto'."""
    from app.engines import build_engine
    from app.models.whisper_cpp import WhisperCppEngine

    settings = Settings(
        token="test-token-with-at-least-thirty-two-characters",
        data_dir=tmp_path,
        engine="whisper.cpp",
        whisper_binary=tmp_path / "whisper-cli",
        whisper_model=tmp_path / "whisper.bin",
        handy_binary=tmp_path / "no-handy",
        vocamac_app=tmp_path / "no-vocamac",
    )
    runtime = RuntimeConfig(engine="auto")
    manager = ModelManager(tmp_path / "models")

    engine = build_engine(settings, runtime, manager)

    assert isinstance(engine, WhisperCppEngine)


def test_forget_if_active_clears_moonshine__aaaaa(tmp_path: Path) -> None:
    """Reproduces the reported bug: deleting the active Moonshine model reset
    `runtime_config.engine` back to "auto" but left `moonshine_model` pointing at
    the now-deleted id — unlike the sherpa-onnx and mlx-audio branches of the same
    method, which null out both fields together."""
    catalog_model = CatalogModel(
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
    manager = ModelManager(tmp_path / "models", catalog=(catalog_model,))
    root = manager.model_path(catalog_model)
    root.mkdir(parents=True)
    (root / catalog_model.marker_file).write_bytes(b"model")
    runtime = RuntimeConfig()
    settings = Settings(
        token="test-token-with-at-least-thirty-two-characters",
        data_dir=tmp_path,
        whisper_binary=tmp_path / "whisper-cli",
        whisper_model=tmp_path / "whisper.bin",
        handy_binary=tmp_path / "no-handy",
        vocamac_app=tmp_path / "no-vocamac",
    )
    config_path = tmp_path / "config.json"
    engines = EngineManager(settings, runtime, config_path, manager)

    engines.select_model(catalog_model.id)
    assert runtime.engine == "moonshine"
    assert runtime.moonshine_model == catalog_model.id

    engines.forget_if_active(catalog_model.id)

    assert runtime.engine == "auto"
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
            token="test-token-with-at-least-thirty-two-characters",
            data_dir=tmp_path,
            engine=name,
            whisper_binary=tmp_path / "whisper-cli",
            whisper_model=tmp_path / "whisper.bin",
            handy_binary=tmp_path / "no-handy",
            vocamac_app=tmp_path / "no-vocamac",
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
