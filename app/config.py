from __future__ import annotations

import os
import secrets
from dataclasses import dataclass
from pathlib import Path

DEFAULT_HANDY_FALLBACK_MODEL = "handy-computer/whisper-base-gguf/whisper-base-Q8_0.gguf"
WILDCARD_BIND_HOST = "0.0.0.0"
WILDCARD_BIND_HOSTS = frozenset((WILDCARD_BIND_HOST, "::"))
APP_DIR_NAME = "vocagateway"
DEFAULT_MAXIMUM_UPLOAD_BYTES = 26_214_400
MINIMUM_TOKEN_LENGTH = 32
CONFIGURATION_DIRECTORY_MODE = 0o700
TOKEN_FILE_MODE = 0o600
TOKEN_SECRET_BYTES = 48
FILE_WRITE_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_EXCL


def format_host_port(host: str, port: int) -> str:
    """Format a listener address without pretending it is a browsable URL."""
    display_host = f"[{host}]" if ":" in host else host
    return f"{display_host}:{port}"


def local_webui_url(host: str, port: int) -> str:
    """Return a URL suitable for opening the WebUI on the gateway machine itself."""
    if host == "::":
        host = "::1"
    elif host in WILDCARD_BIND_HOSTS:
        host = "127.0.0.1"
    return f"http://{format_host_port(host, port)}/"


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, "").strip() or default


def _env_path(name: str, default: Path) -> Path:
    configured_path = _env(name)
    if not configured_path:
        return default
    return Path(configured_path).expanduser()


def _optional_path(name: str) -> Path | None:
    configured_path = _env(name)
    return Path(configured_path).expanduser() if configured_path else None


def _default_token_file() -> Path:
    base = Path(os.environ.get("XDG_CONFIG_HOME", "~/.config")).expanduser()
    return base / APP_DIR_NAME / "token"


def _default_config_file() -> Path:
    base = Path(os.environ.get("XDG_CONFIG_HOME", "~/.config")).expanduser()
    return base / APP_DIR_NAME / "config.json"


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
    # Optional override for the `whisper-server` that keeps a whisper.cpp model
    # resident. Unset means "the sibling of whisper_binary, else PATH".
    whisper_server_binary: Path | None = None
    # `quality` (the default) keeps whisper.cpp's narrowed beam search;
    # `fast` decodes greedily, which is cheaper on a CPU-only host and may cost
    # accuracy on accented or noisy audio. See app/models/whisper_cpp.py.
    whisper_decoder_preset: str = "quality"
    models_dir: Path | None = None
    config_path: Path = Path("~/.config/vocagateway/config.json")
    token_file: Path = Path("~/.config/vocagateway/token")
    bind_host: str = "0.0.0.0"
    port: int = 8765
    maximum_upload_bytes: int = DEFAULT_MAXIMUM_UPLOAD_BYTES
    maximum_duration_seconds: int = 120
    retention_hours: int = 24
    delete_successful_audio: bool = True
    maximum_concurrent_transcriptions: int = 1
    debug: bool = False

    def resolved_models_dir(self) -> Path:
        if self.models_dir is None:
            return self.data_dir / "models"
        return self.models_dir

    @classmethod
    def from_env(cls) -> Settings:
        token_file = _env_path("VOCAGATEWAY_TOKEN_FILE", _default_token_file())
        token = _env("VOCAGATEWAY_TOKEN")
        if not token and token_file.is_file():
            token = token_file.read_text(encoding="utf-8").strip()
        if not token:
            token = cls._generate_token(token_file)
        if len(token) < MINIMUM_TOKEN_LENGTH:
            raise RuntimeError(
                "Set VOCAGATEWAY_TOKEN to at least 32 characters or create "
                f"{cls._display_path(token_file)} with mode 600."
            )
        default_data_dir = (
            Path(os.environ.get("XDG_DATA_HOME", "~/.local/share")).expanduser() / APP_DIR_NAME
        )
        data_dir = _env_path("VOCAGATEWAY_DATA_DIR", default_data_dir)
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
            whisper_server_binary=_optional_path("VOCAGATEWAY_WHISPER_SERVER_BINARY"),
            whisper_decoder_preset=_env("VOCAGATEWAY_WHISPER_DECODER_PRESET", "quality").lower(),
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

    @property
    def token_file_display(self) -> str:
        return self._display_path(self.token_file)

    @classmethod
    def _display_path(cls, path: Path | str) -> str:
        """Render paths with `~` instead of an absolute home prefix for operators."""
        text = str(path)
        home = str(Path.home())
        home_prefix = f"{home}{os.sep}"
        if home and (text == home or text.startswith(home_prefix)):
            remainder = text[len(home) :]
            return f"~{remainder}"
        return text

    @classmethod
    def _generate_token(cls, token_file: Path) -> str:
        """First-run friendly default: create a private token automatically."""
        token = secrets.token_urlsafe(TOKEN_SECRET_BYTES)
        try:
            cls._write_token(token_file, token)
        except OSError:
            return token
        return token

    @classmethod
    def _write_token(cls, token_file: Path, token: str) -> None:
        token_file.parent.mkdir(parents=True, exist_ok=True)
        token_file.parent.chmod(CONFIGURATION_DIRECTORY_MODE)
        descriptor = os.open(token_file, FILE_WRITE_FLAGS, TOKEN_FILE_MODE)
        with os.fdopen(descriptor, "w", encoding="utf-8") as token_handle:
            token_handle.write(f"{token}\n")
