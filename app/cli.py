from __future__ import annotations

import argparse
import json
import os
import sys
from typing import NoReturn, TextIO
from urllib import request as urllib_request

import uvicorn

from app import audio, engines, main, pairing, runtime_config, service, storage
from app.config import (
    WILDCARD_BIND_HOSTS,
    Settings,
    _default_config_file,
    _default_token_file,
    _env_path,
    format_host_port,
    local_webui_url,
)

DEFAULT_GATEWAY_PORT = "8765"


class _Console:
    @classmethod
    def emit(cls, message: str = "", *, stream: TextIO | None = None, end: str = "\n") -> None:
        output = sys.stdout if stream is None else stream
        output.write(f"{message}{end}")

    @classmethod
    def fail(cls, message: str) -> NoReturn:
        cls.emit(message, stream=sys.stderr)
        sys.exit(1)


class _TokenSource:
    @classmethod
    def from_env(cls) -> bool:
        return bool(os.environ.get("VOCAGATEWAY_TOKEN", "").strip())

    @classmethod
    def load_existing(cls) -> str:
        """Return the bootstrap token without minting a new secret.

        Path resolution matches :meth:`Settings.from_env` (blank
        ``VOCAGATEWAY_TOKEN_FILE`` falls back to the default; ``~`` and
        ``XDG_CONFIG_HOME`` are expanded the same way).
        """
        token = os.environ.get("VOCAGATEWAY_TOKEN", "").strip()
        if token:
            return token
        token_file = _env_path("VOCAGATEWAY_TOKEN_FILE", _default_token_file())
        if not token_file.is_file():
            _Console.fail("No token yet — the gateway writes one on first start: just run")
        token = token_file.read_text(encoding="utf-8").strip()
        if not token:
            _Console.fail(f"Token file is empty: {token_file}")
        return token

    @classmethod
    def saved_pairing_url(cls) -> str | None:
        config_path = _env_path("VOCAGATEWAY_CONFIG_FILE", _default_config_file())
        return runtime_config.RuntimeConfig.load(config_path).pairing_url


class _ServerCommand:
    @classmethod
    def run(cls) -> None:
        settings = Settings.from_env()
        host = settings.bind_host
        cls._announce(settings, host)
        app = main.create_app(settings)
        uvicorn.run(app, host=host, port=settings.port, access_log=False)

    @classmethod
    def _announce(cls, settings: Settings, host: str) -> None:
        token_path = settings.token_file_display
        token_source = "(from VOCAGATEWAY_TOKEN)" if _token_from_env() else token_path
        _Console.emit(f"VocaGateway listening on {format_host_port(host, settings.port)}")
        _Console.emit(f"WebUI (this host): {local_webui_url(host, settings.port)}")
        if host in WILDCARD_BIND_HOSTS:
            _Console.emit("Network access: use this host's LAN or Tailscale IP with the same port")
        _Console.emit(f"Token: {token_source}")
        if _token_from_env():
            return
        _Console.emit(f"  (cat {token_path} — enter that value in the phone app)")
        _Console.emit("  or: just token  (prints a terminal QR for headless phone pairing)")


class _TokenCommand:
    @classmethod
    def run(cls) -> None:
        """Print the bootstrap token, and a terminal pairing QR when useful.

        Interactive terminals get the phone-scannable pairing QR (same JSON the
        WebUI encodes). Piped / ``--plain`` output stays a single line so scripts
        can still do ``TOKEN=$(just token --plain)``.
        """
        parser = argparse.ArgumentParser(
            prog="vocagateway-token",
            description="Show the bootstrap bearer token and an optional terminal pairing QR.",
        )
        parser.add_argument(
            "--plain",
            action="store_true",
            help="Print only the token (always used when stdout is not a TTY).",
        )
        args = parser.parse_args()
        secret = _TokenSource.load_existing()
        if args.plain or not sys.stdout.isatty():
            _Console.emit(secret)
            return
        cls._print_pairing(secret)

    @classmethod
    def _print_pairing(cls, secret: str) -> None:
        port = int(os.environ.get("VOCAGATEWAY_PORT", DEFAULT_GATEWAY_PORT))
        gateway_url = default_pairing_url(port, saved_pairing_url=_saved_pairing_url())
        _Console.emit(f"Token: {secret}")
        if gateway_url is None:
            _Console.emit(
                "No phone-reachable gateway address found. Set VOCAGATEWAY_PUBLIC_URL "
                f"or VOCAGATEWAY_PAIRING_URL (for example http://192.168.1.20:{port}), "
                "or pick an address in the WebUI pairing card.",
                stream=sys.stderr,
            )
            return
        cls._print_qr(gateway_url, secret)

    @classmethod
    def _print_qr(cls, gateway_url: str, secret: str) -> None:
        payload = pairing.encode_pairing_payload(gateway_url, secret)
        _Console.emit(f"Gateway: {gateway_url}")
        _Console.emit("Scan with the phone app (Settings → Scan pairing QR / Gateway → Scan QR):")
        _Console.emit()
        _Console.emit(pairing.qr_ascii_for_payload(payload), end="")
        _Console.emit()
        _Console.emit(
            "Override the encoded address with VOCAGATEWAY_PUBLIC_URL, VOCAGATEWAY_PAIRING_URL, "
            "or the WebUI pairing card."
        )


class _OpsCommands:
    @classmethod
    def status(cls) -> None:
        settings = Settings.from_env()
        health_url = f"{local_webui_url(settings.bind_host, settings.port)}health"
        payload = cls._load_json(health_url, "gateway unreachable: {0}", timeout=2)
        _Console.emit(json.dumps(payload, indent=2))

    @classmethod
    def diagnostics(cls) -> None:
        settings = Settings.from_env()
        url = f"{local_webui_url(settings.bind_host, settings.port)}v1/admin/diagnostics"
        headers = {"Authorization": f"Bearer {settings.token}"}
        request = urllib_request.Request(url, headers=headers)
        payload = cls._load_json(request, "gateway unreachable or unauthorized: {0}", timeout=5)
        _Console.emit(json.dumps(payload, indent=2))

    @classmethod
    def cleanup(cls) -> None:
        settings = Settings.from_env()
        repository = storage.SessionRepository(settings.data_dir / "sessions.sqlite3")
        repository.initialize()
        gateway = service.TranscriptionService(
            settings,
            repository,
            engines.StaticEngineProvider(main.select_engine(settings)),
            audio.FFmpegNormalizer(),
        )
        _Console.emit(f"removed {gateway.cleanup_expired()} expired session(s)")

    @classmethod
    def _load_json(
        cls, target: str | urllib_request.Request, error_template: str, *, timeout: int
    ) -> object:
        try:
            response = urllib_request.urlopen(target, timeout=timeout)
        except Exception as error:
            _Console.fail(error_template.format(error))
        with response:
            return json.load(response)


_token_from_env = _TokenSource.from_env
_saved_pairing_url = _TokenSource.saved_pairing_url
serve = _ServerCommand.run
token = _TokenCommand.run
status = _OpsCommands.status
diagnostics = _OpsCommands.diagnostics
cleanup = _OpsCommands.cleanup
# Keep the historical pairing symbol so tests can patch the CLI lookup.
default_pairing_url = pairing.default_pairing_url
