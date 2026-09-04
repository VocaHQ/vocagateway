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

LOOPBACK_HOST = "127.0.0.1"
SERVER_BINARY_NAME = "whisper-server"
INFERENCE_PATH = "/inference"
# A large model on a cold page cache can take a while to reach the listen call,
# and the process only binds the port after the model is in memory, so a slow
# start is a slow load rather than a hung worker.
START_TIMEOUT_SECONDS = 120
START_POLL_SECONDS = 0.1
CONNECT_TIMEOUT_SECONDS = 1.0
REQUEST_TIMEOUT_SECONDS = 75
STOP_TIMEOUT_SECONDS = 3
START_ATTEMPTS = 2
BOUNDARY_TOKEN_BYTES = 12
MAXIMUM_SERVER_ERROR_LENGTH = 240
TEXT_RESPONSE_FORMAT = "text"
JSON_PREFIX = "{"


class WhisperServerUnavailable(Exception):
    """The persistent whisper.cpp worker cannot be started or reached."""


def _terminate(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=STOP_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def resolve_server_binary(cli_binary: Path, override: Path | None = None) -> Path | None:
    """Locate the `whisper-server` that pairs with the configured `whisper-cli`.

    An explicit override wins. Otherwise the sibling of the CLI is preferred —
    both binaries come out of the same whisper.cpp build, and a mismatched pair
    from PATH would load the model with different ggml backends.
    """
    if override is not None:
        return override if override.is_file() else None
    # A bare `whisper-cli` has no directory to look beside, and testing one
    # would resolve against the working directory while the launch would still
    # go through PATH. Only an absolute pair can be trusted to match.
    sibling = cli_binary.with_name(SERVER_BINARY_NAME)
    if cli_binary.is_absolute() and sibling.is_file():
        return sibling
    found = shutil.which(SERVER_BINARY_NAME)
    return Path(found) if found else None


class WhisperServerWorker:
    """A resident `whisper-server` process holding one model and its GPU context.

    `whisper-cli` reloads the model and rebuilds the accelerator context on
    every run, which dominates the latency of a short dictation clip. The
    worker pays that once: it listens on an ephemeral loopback port that is
    never published, transcribes over `/inference`, and is torn down by the
    idle-offload monitor or an engine swap.
    """

    def __init__(
        self,
        binary: Path,
        model: Path,
        *,
        cpu_threads: int,
        beam_size: int,
        best_of: int,
    ) -> None:
        self.binary = binary
        self.model = model
        self.cpu_threads = cpu_threads
        self.beam_size = beam_size
        self.best_of = best_of
        self._process: subprocess.Popen[bytes] | None = None
        self._port = 0
        self._start_lock = asyncio.Lock()

    @property
    def is_running(self) -> bool:
        process = self._process
        return process is not None and process.poll() is None

    async def ensure_started(self) -> bool:
        """Start the worker if it is not running. True when this call started it."""
        if self.is_running:
            return False
        async with self._start_lock:
            if self.is_running:
                return False
            self.stop()
            process, port = await asyncio.to_thread(_start_server, self)
            self._process = process
            self._port = port
            return True

    async def transcribe(self, audio_path: Path, language: str) -> str:
        if not self.is_running:
            raise WhisperServerUnavailable("The whisper.cpp worker is not running.")
        port = self._port
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(_server_transcript, port, audio_path, language),
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
        except TimeoutError as error:
            raise errors.TranscriptionProcessError("Transcription timed out.") from error

    def stop(self) -> None:
        process = self._process
        self._process = None
        self._port = 0
        if process is not None:
            _terminate(process)


class _ServerProcess:
    @classmethod
    def start(cls, worker: WhisperServerWorker) -> tuple[subprocess.Popen[bytes], int]:
        """Spawn a ready worker, retrying a port the server could not take.

        The port is chosen by binding one and letting it go, so another process
        can claim it in between; whisper-server then exits instead of serving.
        That is indistinguishable from a build without server support, and
        retiring the worker for it would cost every later request the reload —
        so a refused start is retried on a fresh port before giving up.
        """
        for remaining in reversed(range(START_ATTEMPTS)):
            port = cls._loopback_port()
            process = cls._spawn(worker, port)
            if cls._became_ready(process, port):
                return process, port
            if not remaining:
                break
        raise WhisperServerUnavailable(
            "The installed whisper.cpp build cannot serve the selected model."
        )

    @classmethod
    def elapsed_ms(cls, started: float) -> int:
        return max(0, int((time.monotonic() - started) * 1000))

    @classmethod
    def _spawn(cls, worker: WhisperServerWorker, port: int) -> subprocess.Popen[bytes]:
        return subprocess.Popen(
            cls._arguments(worker, port),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    @classmethod
    def _arguments(cls, worker: WhisperServerWorker, port: int) -> list[str]:
        return [
            str(worker.binary),
            "-m",
            str(worker.model),
            "--host",
            LOOPBACK_HOST,
            "--port",
            str(port),
            "-t",
            str(worker.cpu_threads),
            "-bs",
            str(worker.beam_size),
            "-bo",
            str(worker.best_of),
        ]

    @classmethod
    def _became_ready(cls, process: subprocess.Popen[bytes], port: int) -> bool:
        """Wait for the worker to accept connections. False once it has exited.

        A worker that is merely slow is still loading the model, so waiting it
        out is right; one that never answers is killed and reported, because a
        retry would only spend the same timeout again.
        """
        deadline = time.monotonic() + START_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            if process.poll() is not None:
                return False
            if cls._accepts(port):
                return True
            time.sleep(START_POLL_SECONDS)
        _terminate(process)
        raise WhisperServerUnavailable("The whisper.cpp worker did not become ready.")

    @classmethod
    def _accepts(cls, port: int) -> bool:
        """The port only accepts once the model is loaded, so a connect means ready."""
        try:
            with socket.create_connection((LOOPBACK_HOST, port), timeout=CONNECT_TIMEOUT_SECONDS):
                return True
        except OSError:
            return False

    @classmethod
    def _loopback_port(cls) -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind((LOOPBACK_HOST, 0))
            return int(listener.getsockname()[1])


class _ServerHttp:
    @classmethod
    def transcribe(cls, port: int, audio_path: Path, language: str) -> str:
        transcript = cls._read_transcript(cls._request(port, audio_path, language))
        if not transcript:
            raise errors.TranscriptionProcessError("The transcription result was empty.")
        return transcript

    @classmethod
    def _request(cls, port: int, audio_path: Path, language: str) -> Request:
        boundary = f"vocagateway-{secrets.token_hex(BOUNDARY_TOKEN_BYTES)}"
        fields = [("response_format", TEXT_RESPONSE_FORMAT), ("language", language)]
        return Request(
            f"http://{LOOPBACK_HOST}:{port}{INFERENCE_PATH}",
            data=cls._multipart_body(boundary, fields, audio_path),
            method="POST",
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )

    @classmethod
    def _read_transcript(cls, http_request: Request) -> str:
        try:
            with urlopen(http_request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
                payload = response.read()
        except OSError as error:
            return cls._reraise_os(error)
        return cls._decode(payload)

    @classmethod
    def _decode(cls, payload: bytes) -> str:
        """`response_format=text` answers in plain text; older builds wrap it in JSON."""
        decoded = payload.decode("utf-8", errors="replace").strip()
        if not decoded.startswith(JSON_PREFIX):
            return decoded
        try:
            document = json.loads(decoded)
        except json.JSONDecodeError as error:
            raise errors.TranscriptionProcessError(
                "The whisper.cpp worker returned an invalid response."
            ) from error
        text = document.get("text") if isinstance(document, dict) else None
        if not isinstance(text, str):
            raise errors.TranscriptionProcessError(
                "The whisper.cpp worker returned an invalid response."
            )
        return text.strip()

    @classmethod
    def _reraise_os(cls, error: OSError) -> str:
        """Separate "this clip took too long" from "this worker is gone".

        Only the second is worth falling back to the CLI for; reporting a
        timeout as an unreachable worker would run the same audio a second
        time and double the wait before the caller sees the failure.
        """
        if isinstance(error, TimeoutError):
            raise errors.TranscriptionProcessError("Transcription timed out.") from error
        reader = getattr(error, "read", None)
        if callable(reader):
            detail = reader().decode("utf-8", errors="replace")[-MAXIMUM_SERVER_ERROR_LENGTH:]
            reason = detail or getattr(error, "reason", "")
            raise errors.TranscriptionProcessError(
                f"The whisper.cpp worker rejected the audio: {reason}"
            ) from error
        raise WhisperServerUnavailable("The whisper.cpp worker is unreachable.") from error

    @classmethod
    def _multipart_body(
        cls, boundary: str, fields: list[tuple[str, str]], audio_path: Path
    ) -> bytes:
        chunks = [cls._field_chunk(boundary, name, entry) for name, entry in fields]
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


elapsed_ms = _ServerProcess.elapsed_ms
_start_server = _ServerProcess.start
_server_transcript = _ServerHttp.transcribe
