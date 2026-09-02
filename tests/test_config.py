from __future__ import annotations

from pathlib import Path

import pytest
from pytest import MonkeyPatch

from app.config import Settings, format_host_port, local_webui_url

TEST_TOKEN_PADDING_LENGTH = 48
DEFAULT_GATEWAY_PORT = 8765
MINIMUM_TOKEN_LENGTH = 32
CUSTOM_GATEWAY_PORT = 9000


def _isolate_home(monkeypatch: MonkeyPatch, tmp_path: Path) -> Path:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    for name in list(__import__("os").environ):
        if name.startswith("VOCAGATEWAY_"):
            monkeypatch.delenv(name, raising=False)
    return home


def test_environment_defaults_to_all_interf_b6e88(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("VOCAGATEWAY_TOKEN", "test-" + ("x" * TEST_TOKEN_PADDING_LENGTH))
    monkeypatch.delenv("VOCAGATEWAY_BIND_HOST", raising=False)

    settings = Settings.from_env()

    assert settings.bind_host == "0.0.0.0"
    assert format_host_port(settings.bind_host, settings.port) == "0.0.0.0:8765"
    assert local_webui_url(settings.bind_host, settings.port) == "http://127.0.0.1:8765/"


def test_ipv6_listener_and_local_url_are_br_aa() -> None:
    assert format_host_port("::", DEFAULT_GATEWAY_PORT) == "[::]:8765"
    assert local_webui_url("::", DEFAULT_GATEWAY_PORT) == "http://[::1]:8765/"


def test_fresh_install_mints_token_to_xdg_config(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    home = _isolate_home(monkeypatch, tmp_path)

    settings = Settings.from_env()

    token_file = home / ".config" / "vocagateway" / "token"
    assert len(settings.token) >= MINIMUM_TOKEN_LENGTH
    assert token_file.read_text(encoding="utf-8").strip() == settings.token


def test_honours_xdg_config_and_data_dirs(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    _isolate_home(monkeypatch, tmp_path)
    config_home = tmp_path / "xdg-config"
    data_home = tmp_path / "xdg-data"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config_home))
    monkeypatch.setenv("XDG_DATA_HOME", str(data_home))
    monkeypatch.setenv("VOCAGATEWAY_TOKEN", "test-" + ("x" * TEST_TOKEN_PADDING_LENGTH))

    settings = Settings.from_env()

    assert settings.config_path == config_home / "vocagateway" / "config.json"
    assert settings.data_dir == data_home / "vocagateway"
    assert settings.models_dir == data_home / "vocagateway" / "models"


def test_reads_token_from_file(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    home = _isolate_home(monkeypatch, tmp_path)
    token_file = home / ".config" / "vocagateway" / "token"
    token_file.parent.mkdir(parents=True)
    token_file.write_text("file-token-with-at-least-thirty-two-chars\n", encoding="utf-8")

    settings = Settings.from_env()

    assert settings.token == "file-token-with-at-least-thirty-two-chars"


def test_env_token_overrides_file(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    home = _isolate_home(monkeypatch, tmp_path)
    token_file = home / ".config" / "vocagateway" / "token"
    token_file.parent.mkdir(parents=True)
    token_file.write_text("file-token-with-at-least-thirty-two-chars\n", encoding="utf-8")
    monkeypatch.setenv("VOCAGATEWAY_TOKEN", "env-token-with-at-least-thirty-two-chars")

    settings = Settings.from_env()

    assert settings.token == "env-token-with-at-least-thirty-two-chars"


def test_whitespace_only_token_env_is_ignored(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    home = _isolate_home(monkeypatch, tmp_path)
    token_file = home / ".config" / "vocagateway" / "token"
    token_file.parent.mkdir(parents=True)
    token_file.write_text("file-token-with-at-least-thirty-two-chars\n", encoding="utf-8")
    monkeypatch.setenv("VOCAGATEWAY_TOKEN", "   \n\t  ")

    settings = Settings.from_env()

    assert settings.token == "file-token-with-at-least-thirty-two-chars"


def test_cli_token_from_env_strips_whitespace(monkeypatch: MonkeyPatch) -> None:
    from app.cli import _token_from_env

    monkeypatch.setenv("VOCAGATEWAY_TOKEN", "   ")
    assert _token_from_env() is False
    monkeypatch.setenv("VOCAGATEWAY_TOKEN", "real-token-with-at-least-thirty-two-chars")
    assert _token_from_env() is True


def test_env_overrides_bind_host_and_port(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("VOCAGATEWAY_TOKEN", "test-" + ("x" * TEST_TOKEN_PADDING_LENGTH))
    monkeypatch.setenv("VOCAGATEWAY_BIND_HOST", "127.0.0.1")
    monkeypatch.setenv("VOCAGATEWAY_PORT", str(CUSTOM_GATEWAY_PORT))

    settings = Settings.from_env()

    assert settings.bind_host == "127.0.0.1"
    assert settings.port == CUSTOM_GATEWAY_PORT


def test_custom_token_file_env(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    home = _isolate_home(monkeypatch, tmp_path)
    custom_token = home / "custom" / "gateway.token"
    custom_token.parent.mkdir(parents=True)
    custom_token.write_text("custom-token-with-at-least-thirty-two\n", encoding="utf-8")
    monkeypatch.setenv("VOCAGATEWAY_TOKEN_FILE", str(custom_token))

    settings = Settings.from_env()

    assert settings.token == "custom-token-with-at-least-thirty-two"
    assert settings.token_file == custom_token


def test_short_token_raises(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    _isolate_home(monkeypatch, tmp_path)
    monkeypatch.setenv("VOCAGATEWAY_TOKEN", "too-short")

    with pytest.raises(RuntimeError, match="at least 32 characters"):
        Settings.from_env()
