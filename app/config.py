from __future__ import annotations

import ipaddress
import os
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
TRUTHY_VALUES = frozenset(("1", "true", "yes", "on"))
FALSY_VALUES = frozenset(("0", "false", "no", "off"))
# Hosts an operator may point transcript cleanup at. The runtime is part of the
# local deployment, not a service to be reached across a network: loopback for a
# native install, a private address or a bare Compose service name for a
# container. Anything routable is refused at startup rather than silently
# turning a "runs on your gateway" promise into a request to somebody else.
LOOPBACK_HOST_NAMES = frozenset(("localhost", "127.0.0.1", "::1"))
MAXIMUM_PORT = 65_535
# Compose used to ship a sidecar service literally named `cleanup`. That
# service is gone; the runtime is in the image. An operator who kept
# VOCAGATEWAY_CLEANUP_ENDPOINT=cleanup:8080 would otherwise hit a dead
# hostname while the in-image worker sat unused.
REMOVED_CLEANUP_SIDECAR_HOST = "cleanup"
CLEANUP_PROFILE_CHOICES = frozenset(("auto", "compact", "full"))


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


def _env_flag(name: str) -> bool | None:
    """A tri-state environment switch: on, off, or "the operator said nothing".

    The difference matters for cleanup, where an unset variable has to leave the
    saved WebUI choice alone rather than override it with a default.
    """
    raw = _env(name).lower()
    if raw in TRUTHY_VALUES:
        return True
    return False if raw in FALSY_VALUES else None


def _env_seconds(name: str) -> float | None:
    raw = _env(name)
    if not raw:
        return None
    try:
        seconds = float(raw)
    except ValueError as error:
        raise RuntimeError(f"{name} must be a number of seconds.") from error
    return seconds


def _env_codes(name: str) -> tuple[str, ...]:
    raw = _env(name)
    codes = (code.strip() for code in raw.replace(";", ",").split(","))
    return tuple(code for code in codes if code)


def _cleanup_profile(raw: str) -> str | None:
    choice = raw.strip().lower()
    if not choice:
        return None
    return choice if choice in CLEANUP_PROFILE_CHOICES else None


def parse_local_endpoint(raw: str, *, name: str) -> tuple[str, int]:
    """Parse `host:port`, refusing anything that is not part of this deployment.

    No scheme, no path, no credentials: a URL would invite a redirect or a proxy
    into a path that is deliberately a raw socket to a fixed address. The old
    Compose sidecar hostname `cleanup` is refused too: that service no longer
    exists, and leaving it set would miss the in-image runtime.
    """
    if "://" in raw or "/" in raw or "@" in raw:
        raise RuntimeError(f"{name} must be host:port, without a scheme or path.")
    host, separator, port_text = raw.rpartition(":")
    if not separator or not host or not port_text.isdigit():
        raise RuntimeError(f"{name} must be host:port.")
    host = host.strip("[]")
    port = int(port_text)
    if not 1 <= port <= MAXIMUM_PORT:
        raise RuntimeError(f"{name} has a port outside 1-{MAXIMUM_PORT}.")
    if host.lower() == REMOVED_CLEANUP_SIDECAR_HOST:
        raise RuntimeError(
            f"{name} names the old Compose sidecar host "
            f"'{REMOVED_CLEANUP_SIDECAR_HOST}', which is no longer part of this "
            "deployment. Unset it to use the in-image runtime, or point it at a "
            "service you run yourself under a different name (for example "
            "my-cleanup:8080)."
        )
    if not _is_local_host(host):
        raise RuntimeError(
            f"{name} must name a loopback address, a private address, or a "
            "container service on this deployment's own network."
        )
    return host, port


def _is_local_host(host: str) -> bool:
    if host.lower() in LOOPBACK_HOST_NAMES:
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        # Not an address at all. A bare label with no dots is a Compose service
        # name on the deployment's own private network; a dotted name is a
        # public DNS name and is refused.
        return "." not in host and bool(host)
    return address.is_loopback or address.is_private


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
    # Transcript cleanup. Every one of these is an *override*: None or empty
    # means "the operator said nothing here", which leaves the saved WebUI
    # choice in charge. A set value wins over the UI and is reported as locked
    # rather than pretending a save changed it.
    cleanup_enabled: bool | None = None
    cleanup_mode: str | None = None
    cleanup_model: str | None = None
    cleanup_timeout_seconds: float | None = None
    cleanup_languages: tuple[str, ...] = ()
    cleanup_auto_language: str | None = None
    # Operator-only. An explicit `llama-server` executable for the gateway to
    # launch, or an address of a server the operator runs themselves. The
    # container image sets the first to the runtime it built; a native install
    # leaves it unset and the gateway looks on PATH. Setting the address gives
    # up gateway-controlled warm-up and idle unloading, because the gateway
    # then does not own the process.
    cleanup_binary: Path | None = None
    cleanup_endpoint: tuple[str, int] | None = None
    cleanup_api_key: str | None = None
    # How the managed llama-server is launched. None or "auto" picks compact or
    # full from this host. "compact" and "full" force one set of flags.
    cleanup_profile: str | None = None

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
            in TRUTHY_VALUES,
            debug=_env("VOCAGATEWAY_DEBUG", "false").lower() in TRUTHY_VALUES,
            **cls._cleanup_env(),
        )

    @property
    def token_file_display(self) -> str:
        return self._display_path(self.token_file)

    @classmethod
    def _cleanup_env(cls) -> dict[str, Any]:
        endpoint = _env("VOCAGATEWAY_CLEANUP_ENDPOINT")
        return {
            "cleanup_enabled": _env_flag("VOCAGATEWAY_CLEANUP_ENABLED"),
            "cleanup_mode": _env("VOCAGATEWAY_CLEANUP_MODE").lower() or None,
            "cleanup_model": _env("VOCAGATEWAY_CLEANUP_MODEL") or None,
            "cleanup_timeout_seconds": _env_seconds("VOCAGATEWAY_CLEANUP_TIMEOUT_SECONDS"),
            "cleanup_languages": _env_codes("VOCAGATEWAY_CLEANUP_LANGUAGES"),
            "cleanup_auto_language": _env("VOCAGATEWAY_CLEANUP_AUTO_LANGUAGE").lower() or None,
            "cleanup_binary": _optional_path("VOCAGATEWAY_CLEANUP_BINARY"),
            "cleanup_endpoint": (
                parse_local_endpoint(endpoint, name="VOCAGATEWAY_CLEANUP_ENDPOINT")
                if endpoint
                else None
            ),
            "cleanup_api_key": _env("VOCAGATEWAY_CLEANUP_API_KEY") or None,
            "cleanup_profile": _cleanup_profile(_env("VOCAGATEWAY_CLEANUP_PROFILE")),
        }

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
