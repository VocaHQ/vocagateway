from __future__ import annotations

import asyncio
import json
import secrets
import shutil
import socket
import subprocess
import time
from pathlib import Path
from urllib.request import Request, urlopen

from app import errors
from app.models.base import EngineHealth, EngineTranscription, TranscriptionOptions

TRANSCRIPTION_TIMEOUT_SECONDS = 180
HTTP_SUCCESS_STATUS = 200
MAXIMUM_CLI_ERROR_LENGTH = 200
SERVER_TOKEN_BYTES = 12
MAXIMUM_SERVER_ERROR_LENGTH = 240


class _WhisperKitServer:
    def __init__(self, process: subprocess.Popen[bytes], port: int) -> None:
        self.process = process
        self.port = port


class _PersistentServerUnavailable(Exception):
    """The installed CLI cannot start or reach its persistent local server."""


class _ServerHttp:
    @classmethod
    def health_url(cls, server: _WhisperKitServer) -> str:
        return f"http://127.0.0.1:{server.port}/health"

    @classmethod
    def transcribe(
        cls,
        server: _WhisperKitServer,
        model_path: Path,
        audio_path: Path,
        language: str,
    ) -> str:
        if server.process.poll() is not None:
            raise _PersistentServerUnavailable("WhisperKit's local server stopped.")
        request = cls._request(server, model_path, audio_path, language)
        payload = cls._load_payload(request)
        transcript = payload.get("text") if isinstance(payload, dict) else None
        if not isinstance(transcript, str):
            raise errors.TranscriptionProcessError(
                "WhisperKit returned an invalid server response."
            )
        return transcript.strip()

    @classmethod
    def _request(
        cls,
        server: _WhisperKitServer,
        model_path: Path,
        audio_path: Path,
        language: str,
    ) -> Request:
        boundary = f"vocaphone-{secrets.token_hex(SERVER_TOKEN_BYTES)}"
        fields = [("model", model_path.name)]
        if language != "auto":
            fields.append(("language", language))
        return Request(
            f"http://127.0.0.1:{server.port}/v1/audio/transcriptions",
            data=cls._multipart_body(boundary, fields, audio_path),
            method="POST",
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )

    @classmethod
    def _load_payload(cls, http_request: Request) -> object:
        try:
            with urlopen(http_request, timeout=TRANSCRIPTION_TIMEOUT_SECONDS) as response:
                return json.load(response)
        except json.JSONDecodeError as error:
            raise errors.TranscriptionProcessError("WhisperKit returned invalid JSON.") from error
        except UnicodeDecodeError as error:
            raise errors.TranscriptionProcessError("WhisperKit returned invalid JSON.") from error
        except OSError as error:
            return cls._reraise_os(error)

    @classmethod
    def _reraise_os(cls, error: OSError) -> object:
        reader = getattr(error, "read", None)
        if callable(reader):
            detail = reader().decode("utf-8", errors="replace")[-MAXIMUM_SERVER_ERROR_LENGTH:]
            reason = detail or getattr(error, "reason", "")
            raise errors.TranscriptionProcessError(
                f"WhisperKit server rejected the audio: {reason}"
            ) from error
        raise _PersistentServerUnavailable("WhisperKit's local server is unreachable.") from error

    @classmethod
    def _multipart_body(
        cls, boundary: str, fields: list[tuple[str, str]], audio_path: Path
    ) -> bytes:
        chunks: list[bytes] = []
        for name, field_value in fields:
            chunks.append(cls._field_chunk(boundary, name, field_value))
        safe_name = audio_path.name.replace('"', "")
        header = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="{safe_name}"\r\n'
            "Content-Type: audio/wav\r\n\r\n"
        )
        chunks.append(header.encode())
        chunks.append(audio_path.read_bytes())
        chunks.append(f"\r\n--{boundary}--\r\n".encode())
        return b"".join(chunks)

    @classmethod
    def _field_chunk(cls, boundary: str, name: str, field_value: str) -> bytes:
        return (
            f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n'
            f"\r\n{field_value}\r\n"
        ).encode()


class _ServerProcess:
    @classmethod
    def start(cls, binary: str, model_path: Path, tokenizer_path: Path | None) -> _WhisperKitServer:
        port = cls._loopback_port()
        arguments = [
            binary,
            "serve",
            "--model-path",
            str(model_path),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ]
        if tokenizer_path is not None:
            arguments.extend(["--download-tokenizer-path", str(tokenizer_path)])
        process = subprocess.Popen(
            arguments,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        server = _WhisperKitServer(process, port)
        cls._wait_until_ready(server)
        return server

    @classmethod
    def stop(cls, process: subprocess.Popen[bytes]) -> None:
        process.terminate()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()

    @classmethod
    def elapsed_ms(cls, started: float) -> int:
        return max(0, int((time.monotonic() - started) * 1000))

    @classmethod
    def _wait_until_ready(cls, server: _WhisperKitServer) -> None:
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            if cls._ready(server):
                return
            time.sleep(0.1)
        cls.stop(server.process)
        raise _PersistentServerUnavailable("WhisperKit's local server did not become ready.")

    @classmethod
    def _ready(cls, server: _WhisperKitServer) -> bool:
        if server.process.poll() is not None:
            raise _PersistentServerUnavailable(
                "The installed WhisperKit CLI does not support persistent serving."
            )
        try:
            with urlopen(_ServerHttp.health_url(server), timeout=1) as response:
                return int(response.status) == HTTP_SUCCESS_STATUS
        except OSError:
            return False

    @classmethod
    def _loopback_port(cls) -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind(("127.0.0.1", 0))
            return int(listener.getsockname()[1])


class _CliFallback:
    @classmethod
    async def transcribe(
        cls,
        resolved: str,
        model_path: Path | None,
        tokenizer_path: Path | None,
        audio_path: Path,
        options: TranscriptionOptions,
    ) -> EngineTranscription:
        inference_started = time.monotonic()
        arguments = cls._arguments(resolved, model_path, tokenizer_path, audio_path, options)
        stdout = await cls._communicate(arguments)
        transcript = cls._extract_transcript(stdout.decode("utf-8", errors="replace"))
        if not transcript:
            raise errors.TranscriptionProcessError("WhisperKit returned an empty transcript.")
        return EngineTranscription(
            text=transcript,
            inference_ms=_ServerProcess.elapsed_ms(inference_started),
        )

    @classmethod
    def resolve_binary(cls, binary: str) -> str | None:
        candidate = Path(binary).expanduser()
        if candidate.is_file():
            return str(candidate)
        return shutil.which(binary)

    @classmethod
    def _arguments(
        cls,
        resolved: str,
        model_path: Path | None,
        tokenizer_path: Path | None,
        audio_path: Path,
        options: TranscriptionOptions,
    ) -> list[str]:
        arguments = [
            resolved,
            "transcribe",
            "--audio-path",
            str(audio_path),
            "--model-path",
            str(model_path),
        ]
        if tokenizer_path is not None:
            arguments.extend(["--download-tokenizer-path", str(tokenizer_path)])
        if options.language != "auto":
            arguments.extend(["--language", options.language])
        return arguments

    @classmethod
    async def _communicate(cls, arguments: list[str]) -> bytes:
        process = await asyncio.create_subprocess_exec(
            *arguments,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=TRANSCRIPTION_TIMEOUT_SECONDS
            )
        except TimeoutError as error:
            process.kill()
            await process.wait()
            raise errors.TranscriptionProcessError("WhisperKit transcription timed out.") from error
        if process.returncode != 0:
            raise errors.TranscriptionProcessError(cls._cli_error(stderr))
        return stdout

    @classmethod
    def _cli_error(cls, stderr: bytes) -> str:
        message = stderr.decode("utf-8", errors="replace").strip().splitlines()
        detail = message[-1][:MAXIMUM_CLI_ERROR_LENGTH] if message else "unknown WhisperKit error"
        return f"WhisperKit exited unsuccessfully: {detail}"

    @classmethod
    def _extract_transcript(cls, output: str) -> str:
        lines = [line.strip() for line in output.strip().splitlines()]
        text_lines = [line for line in lines if line and not line.startswith("[")]
        return " ".join(text_lines).strip()


class WhisperKitEngine:
    """Persistent Apple-silicon WhisperKit adapter with a legacy CLI fallback."""

    def __init__(
        self,
        binary: str,
        model_path: Path | None,
        *,
        tokenizer_path: Path | None = None,
    ) -> None:
        self.binary = binary
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self._server: _WhisperKitServer | None = None
        self._load_lock = asyncio.Lock()
        self._inference_lock = asyncio.Lock()

    async def health(self) -> EngineHealth:
        resolved = self._resolved_binary()
        model_ready = self.model_path is not None and (self.model_path / "config.json").is_file()
        model_name = self.model_path.name if self.model_path else "no-model-selected"
        return EngineHealth(
            ready=resolved is not None and model_ready,
            name=f"whisperkit:{model_name}",
        )

    async def warmup(self) -> int:
        if not (await self.health()).ready or self.model_path is None:
            return 0
        try:
            await self._ensure_server()
        except _PersistentServerUnavailable:
            return 0
        return sum(
            nested.stat().st_size for nested in self.model_path.rglob("*") if nested.is_file()
        )

    async def transcribe(
        self, audio_path: Path, options: TranscriptionOptions
    ) -> EngineTranscription:
        resolved = self._resolved_binary()
        health = await self.health()
        if resolved is None or not health.ready or self.model_path is None:
            raise errors.EngineUnavailableError(
                "The WhisperKit CLI or the selected model is unavailable. "
                "Install it with `brew install whisperkit-cli` and download a model."
            )
        async with self._inference_lock:
            return await _Inference(self, resolved, audio_path, options).run()

    def close(self) -> None:
        server = self._server
        self._server = None
        if server is None or server.process.poll() is not None:
            return
        _ServerProcess.stop(server.process)

    async def _ensure_server(self) -> tuple[_WhisperKitServer, bool]:
        if self._server is not None and self._server.process.poll() is None:
            return self._server, False
        async with self._load_lock:
            if self._server is not None and self._server.process.poll() is None:
                return self._server, False
            self.close()
            resolved = self._resolved_binary()
            if resolved is None or self.model_path is None:
                raise _PersistentServerUnavailable("WhisperKit is unavailable.")
            self._server = await asyncio.to_thread(
                _start_server,
                resolved,
                self.model_path,
                self.tokenizer_path,
            )
            return self._server, True

    def _resolved_binary(self) -> str | None:
        return _CliFallback.resolve_binary(self.binary)


class _Inference:
    def __init__(
        self,
        engine: WhisperKitEngine,
        resolved: str,
        audio_path: Path,
        options: TranscriptionOptions,
    ) -> None:
        self.engine = engine
        self.resolved = resolved
        self.audio_path = audio_path
        self.options = options
        self.model_load_ms = 0

    async def run(self) -> EngineTranscription:
        load_started = time.monotonic()
        try:
            server, loaded_now = await self.engine._ensure_server()
        except _PersistentServerUnavailable:
            return await self._cli()
        if loaded_now:
            self.model_load_ms = _ServerProcess.elapsed_ms(load_started)
        return await self._with_server(server)

    async def _cli(self) -> EngineTranscription:
        return await _CliFallback.transcribe(
            self.resolved,
            self.engine.model_path,
            self.engine.tokenizer_path,
            self.audio_path,
            self.options,
        )

    async def _with_server(self, server: _WhisperKitServer) -> EngineTranscription:
        attempt = 0
        while attempt < 2:
            attempt += 1
            try:
                transcription = await self._attempt(server)
            except _PersistentServerUnavailable:
                return await self._cli()
            if transcription is not None:
                return transcription
            server = self.engine._server or server
        return await self._cli()

    async def _attempt(self, server: _WhisperKitServer) -> EngineTranscription | None:
        model_path = self.engine.model_path
        if model_path is None:
            raise _PersistentServerUnavailable("WhisperKit is unavailable.")
        inference_started = time.monotonic()
        try:
            transcript = await asyncio.wait_for(
                asyncio.to_thread(
                    _server_transcription,
                    server,
                    model_path,
                    self.audio_path,
                    self.options.language,
                ),
                timeout=TRANSCRIPTION_TIMEOUT_SECONDS,
            )
        except TimeoutError as error:
            raise errors.TranscriptionProcessError("WhisperKit transcription timed out.") from error
        except _PersistentServerUnavailable:
            await asyncio.to_thread(self._discard, server)
            await self._reload()
            return None
        if not transcript:
            raise errors.TranscriptionProcessError("WhisperKit returned an empty transcript.")
        return EngineTranscription(
            text=transcript,
            model_load_ms=self.model_load_ms,
            inference_ms=_ServerProcess.elapsed_ms(inference_started),
        )

    async def _reload(self) -> None:
        reload_started = time.monotonic()
        _server, reloaded = await self.engine._ensure_server()
        if reloaded:
            self.model_load_ms += _ServerProcess.elapsed_ms(reload_started)

    def _discard(self, server: _WhisperKitServer) -> None:
        if self.engine._server is server:
            self.engine.close()


_multipart_body = _ServerHttp._multipart_body
_start_server = _ServerProcess.start
_server_transcription = _ServerHttp.transcribe
