from __future__ import annotations

import os
import secrets
from dataclasses import dataclass
from pathlib import Path

DEFAULT_HANDY_FALLBACK_MODEL = "handy-computer/whisper-base-gguf/whisper-base-Q8_0.gguf"
WILDCARD_BIND_HOSTS = frozenset({"0.0.0.0", "::"})
APP_DIR_NAME = "vocagateway"


def format_host_port(host: str, port: int) -> str:
    """Format a listener address without pretending it is a browsable URL."""
    display_host = f"[{host}]" if ":" in host else host
    return f"{display_host}:{port}"


def local_webui_url(host: str, port: int) -> str:
    """Return the loopback URL that opens a listener from the same host."""
    if host == "0.0.0.0":
        host = "127.0.0.1"
    elif host == "::":
        host = "::1"
    return f"http://{format_host_port(host, port)}/"


def _xdg_config_home() -> Path:
    return Path(os.environ.get("XDG_CONFIG_HOME", "~/.config")).expanduser()


def _xdg_data_home() -> Path:
    return Path(os.environ.get("XDG_DATA_HOME", "~/.local/share")).expanduser()


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, "").strip() or default


def _env_path(name: str, default: Path) -> Path:
    configured_path = _env(name)
    if not configured_path:
        return default
    return Path(configured_path).expanduser()


def _default_token_file() -> Path:
    return _xdg_config_home() / APP_DIR_NAME / "token"


def _default_config_file() -> Path:
    return _xdg_config_home() / APP_DIR_NAME / "config.json"


def _default_data_dir() -> Path:
    return _xdg_data_home() / APP_DIR_NAME


def _display_path(path: Path | str) -> str:
    """Render paths with `~` instead of an absolute home prefix for operators."""
    text = str(path)
    home = str(Path.home())
    if home and (text == home or text.startswith(home + os.sep)):
        return "~" + text[len(home) :]
    return text


@dataclass(frozen=True, slots=True)
class Settings:
    token: str
    data_dir: Path
    whisper_binary: Path
    whisper_model: Path
    engine: str = "auto"
    handy_binary: Path = Path("/Applications/Handy.app/Contents/MacOS/handy")
    handy_model: str | None = None
    handy_fallback_model: str | None = DEFAULT_HANDY_FALLBACK_MODEL
    vocamac_app: Path = Path("/Applications/VocaMac.app")
    vocamac_model: str | None = None
    whisperkit_binary: str = "whisperkit-cli"
    models_dir: Path | None = None
    config_path: Path = Path("~/.config/vocagateway/config.json")
    token_file: Path = Path("~/.config/vocagateway/token")
    bind_host: str = "0.0.0.0"
    port: int = 8765
    maximum_upload_bytes: int = 25 * 1024 * 1024
    maximum_duration_seconds: int = 120
    retention_hours: int = 24
    delete_successful_audio: bool = True
    maximum_concurrent_transcriptions: int = 1
    debug: bool = False

    def resolved_models_dir(self) -> Path:
        return self.models_dir if self.models_dir is not None else self.data_dir / "models"

    @property
    def token_file_display(self) -> str:
        return _display_path(self.token_file)

    @classmethod
    def from_env(cls) -> Settings:
        token_file = _env_path("VOCAGATEWAY_TOKEN_FILE", _default_token_file())
        token = _env("VOCAGATEWAY_TOKEN")
        if not token and token_file.is_file():
            token = token_file.read_text(encoding="utf-8").strip()
        if not token:
            token = _generate_token(token_file)
        if len(token) < 32:
            raise RuntimeError(
                "Set VOCAGATEWAY_TOKEN to at least 32 characters or create "
                f"{_display_path(token_file)} with mode 600."
            )
        data_dir = _env_path("VOCAGATEWAY_DATA_DIR", _default_data_dir())
        models_override = _env("VOCAGATEWAY_MODELS_DIR")
        return cls(
            token=token,
            data_dir=data_dir,
            whisper_binary=Path(
                _env("VOCAGATEWAY_WHISPER_BINARY", "/opt/homebrew/bin/whisper-cli")
            ).expanduser(),
            whisper_model=Path(
                _env(
                    "VOCAGATEWAY_WHISPER_MODEL",
                    "~/.local/share/whisper.cpp/models/ggml-base.en.bin",
                )
            ).expanduser(),
            engine=_env("VOCAGATEWAY_ENGINE", "auto").lower(),
            handy_binary=Path(
                _env(
                    "VOCAGATEWAY_HANDY_BINARY",
                    "/Applications/Handy.app/Contents/MacOS/handy",
                )
            ).expanduser(),
            handy_model=_env("VOCAGATEWAY_HANDY_MODEL") or None,
            handy_fallback_model=_env(
                "VOCAGATEWAY_HANDY_FALLBACK_MODEL",
                DEFAULT_HANDY_FALLBACK_MODEL,
            )
            or None,
            vocamac_app=Path(
                _env("VOCAGATEWAY_VOCAMAC_APP", "/Applications/VocaMac.app")
            ).expanduser(),
            vocamac_model=_env("VOCAGATEWAY_VOCAMAC_MODEL") or None,
            whisperkit_binary=_env("VOCAGATEWAY_WHISPERKIT_BINARY", "whisperkit-cli"),
            models_dir=Path(models_override).expanduser()
            if models_override
            else data_dir / "models",
            config_path=_env_path("VOCAGATEWAY_CONFIG_FILE", _default_config_file()),
            token_file=token_file,
            bind_host=_env("VOCAGATEWAY_BIND_HOST", "0.0.0.0"),
            port=int(_env("VOCAGATEWAY_PORT", "8765")),
            retention_hours=int(_env("VOCAGATEWAY_RETENTION_HOURS", "24")),
            delete_successful_audio=_env("VOCAGATEWAY_DELETE_SUCCESSFUL_AUDIO", "true").lower()
            in {"1", "true", "yes"},
            debug=_env("VOCAGATEWAY_DEBUG", "false").lower() in {"1", "true", "yes"},
        )


def _write_token_file(token_file: Path, token: str) -> None:
    token_file.parent.mkdir(parents=True, exist_ok=True)
    token_file.parent.chmod(0o700)
    descriptor = os.open(token_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as token_handle:
        token_handle.write(token + "\n")


def _generate_token(token_file: Path) -> str:
    """First-run friendly default: create a private token automatically."""
    token = secrets.token_urlsafe(48)
    try:
        _write_token_file(token_file, token)
    except OSError:
        return token
    return token
