from __future__ import annotations

import json
from pathlib import Path

from app.runtime_config import RuntimeConfig


def test_legacy_moonshine_language_migrates(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"engine": "moonshine", "moonshine_language": "es"}))

    config = RuntimeConfig.load(path)

    assert config.moonshine_language == "es"
    assert config.moonshine_model == "moonshine:es"


def test_moonshine_variant_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    original = RuntimeConfig(
        engine="moonshine",
        moonshine_model="moonshine:en-tiny-streaming",
        moonshine_language="en",
    )

    original.save(path)

    assert RuntimeConfig.load(path) == original


def test_new_model_selections_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    original = RuntimeConfig(
        engine="sherpa-onnx",
        sherpa_model="sherpa-onnx:sensevoice-small-int8",
        mlx_audio_model="mlx-audio:whisper-large-v3-turbo-4bit",
    )

    original.save(path)

    assert RuntimeConfig.load(path) == original


def test_idle_offload_policy_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    original = RuntimeConfig(idle_offload_enabled=True, idle_offload_minutes=30)

    original.save(path)

    assert RuntimeConfig.load(path) == original


def test_invalid_idle_offload_policy_uses_safe_defaults(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps({"idle_offload_enabled": "yes", "idle_offload_minutes": 1}),
        encoding="utf-8",
    )

    loaded = RuntimeConfig.load(path)

    assert loaded.idle_offload_enabled is False
    assert loaded.idle_offload_minutes == 15


def test_a_config_written_before_cleanup_existed_stays_off(tmp_path: Path) -> None:
    """A missing key on an already-written config is not consent.

    Every gateway that ran before cleanup has a saved config with no
    `cleanup_enabled` key. Inheriting the shipped default would start
    corrections on upgrade the moment a leftover GGUF was on disk.
    """
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"engine": "moonshine"}), encoding="utf-8")

    assert RuntimeConfig.load(path).cleanup_enabled is False


def test_a_fresh_install_defaults_cleanup_on(tmp_path: Path) -> None:
    """No file at all is a brand-new install, not an upgrade.

    RuntimeConfig.load treats a missing or unreadable path as payload is None
    and returns cls(), which keeps the shipped default on. That is distinct
    from a file that exists and simply never heard of cleanup.
    """
    path = tmp_path / "config.json"
    assert not path.exists()
    assert RuntimeConfig.load(path).cleanup_enabled is True
    assert RuntimeConfig().cleanup_enabled is True


def test_an_operator_who_turned_cleanup_off_keeps_it_off(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"cleanup_enabled": False}), encoding="utf-8")

    assert RuntimeConfig.load(path).cleanup_enabled is False


def test_a_junk_cleanup_flag_falls_back_rather_than_being_believed(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"cleanup_enabled": "yes"}), encoding="utf-8")

    assert RuntimeConfig.load(path).cleanup_enabled is False
