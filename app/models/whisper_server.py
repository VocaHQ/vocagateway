from __future__ import annotations

import asyncio
import json
import os
import secrets
import shutil
import socket
import subprocess
import tempfile
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import IO

from app import errors

LOOPBACK_HOST = "127.0.0.1"
SERVER_BINARY_NAME = "whisper-server"
INFERENCE_PATH = "/inference"
# A large model on a cold page cache can take a while to reach the listen call,
# and the process only binds the port after the model is in memory, so a slow
# start is a slow load rather than a hung worker.
START_TIMEOUT_SECONDS = 120.0
START_POLL_SECONDS = 0.1
CONNECT_TIMEOUT_SECONDS = 1.0
REQUEST_TIMEOUT_SECONDS = 75.0
STOP_TIMEOUT_SECONDS = 3
# How long a worker gets to notice a closed connection and drop the decode it
# was running. The abort is polled between decoder steps, so a healthy worker
# answers again in milliseconds; one that does not is stuck holding the model.
ABORT_GRACE_SECONDS = 5.0
# A step that starts with its deadline already spent still gets a moment to
# fail properly instead of being handed a negative timeout.
MINIMUM_BUDGET_SECONDS = 1.0
START_ATTEMPTS = 2
BOUNDARY_TOKEN_BYTES = 12
MAXIMUM_SERVER_ERROR_LENGTH = 240
DIAGNOSTIC_TAIL_BYTES = 2048
TEXT_RESPONSE_FORMAT = "text"
JSON_PREFIX = "{"
HTTP_OK = 200
CRLF = "\r\n"
HEADER_TERMINATOR = b"\r\n\r\n"
# One boundary per process is enough: it only has to be absent from the audio,
# and a random token settles that without threading it through every helper.
MULTIPART_BOUNDARY = f"vocagateway-{secrets.token_hex(BOUNDARY_TOKEN_BYTES)}"


class WhisperServerUnavailable(Exception):
    """The persistent whisper.cpp worker cannot be started or reached."""


def elapsed_ms(started: float) -> int:
    return max(0, int((time.monotonic() - started) * 1000))


def time_budget(deadline: float | None, ceiling: float) -> float:
    """Seconds a step may take: its own ceiling, capped by the request deadline.

    One deadline covers the whole transcription, so a worker start, the
    inference, and a CLI fallback share a single budget instead of each
    spending a full timeout while the caller waits.
    """
    if deadline is None:
        return ceiling
    return max(MINIMUM_BUDGET_SECONDS, min(ceiling, deadline - time.monotonic()))


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
        self._stderr: IO[bytes] | None = None
        self._interrupted = False
        self._start_lock = asyncio.Lock()

    @property
    def is_running(self) -> bool:
        process = self._process
        return process is not None and process.poll() is None

    def diagnostics(self) -> str:
        """The tail of the worker's stderr — its backend, thread count, and errors.

        whisper.cpp writes the backend it actually selected and the model load
        there, and nothing else: the transcript only reaches stdout under
        `--print-realtime`, which the gateway never passes.
        """
        stderr = self._stderr
        if stderr is None:
            return ""
        try:
            size = stderr.seek(0, os.SEEK_END)
            stderr.seek(max(0, size - DIAGNOSTIC_TAIL_BYTES))
            captured = stderr.read()
        except OSError:
            return ""
        return captured.decode("utf-8", errors="replace").strip()

    async def ensure_started(self, *, deadline: float | None = None) -> bool:
        """Start the worker if it is not running. True when this call started it."""
        if self.is_running:
            return False
        async with self._start_lock:
            if self.is_running:
                return False
            self.stop()
            await _ServerStart(self).run(deadline)
            return True

    async def reclaim(self) -> None:
        """Drop a worker that never let go of an interrupted decode.

        Closing the socket asks the pinned server to abort, but nothing
        guarantees it obeys. Serving the next clip through a worker still
        holding the model mutex would queue behind the abandoned decode, so an
        unresponsive one is terminated here and reloaded by the caller.
        """
        interrupted = self._interrupted
        self._interrupted = False
        if not interrupted or not self.is_running:
            return
        if not await _ServerHttp.answers(self._port):
            self.stop()

    async def transcribe(
        self, audio_path: Path, language: str, *, deadline: float | None = None
    ) -> str:
        if not self.is_running:
            raise WhisperServerUnavailable("The whisper.cpp worker is not running.")
        interrupted = False
        try:
            return await asyncio.wait_for(
                _ServerHttp.transcribe(self._port, audio_path, language),
                timeout=time_budget(deadline, REQUEST_TIMEOUT_SECONDS),
            )
        except TimeoutError as error:
            interrupted = True
            raise errors.TranscriptionProcessError("Transcription timed out.") from error
        except asyncio.CancelledError:
            # A cut-short request leaves a decode running inside the worker.
            # Nothing is awaited here — the check happens before the next clip.
            interrupted = True
            raise
        finally:
            self._interrupted = interrupted

    def adopt(self, process: subprocess.Popen[bytes], port: int) -> None:
        """Take ownership of a freshly spawned process before anything can await.

        Recording it only once it answers would leak a model-sized process
        whenever the start is cancelled between the spawn and the readiness
        check.
        """
        self._process = process
        self._port = port

    def open_diagnostics(self) -> IO[bytes]:
        # Deliberately not a context manager: the capture lives as long as the
        # worker does, and is read after it fails. `_close_diagnostics` and the
        # unlinked temporary file are what release it.
        self._close_diagnostics()
        self._stderr = tempfile.TemporaryFile()  # noqa: SIM115
        return self._stderr

    def stop(self) -> None:
        process = self._process
        self._process = None
        self._port = 0
        self._interrupted = False
        if process is not None:
            _terminate(process)

    def _close_diagnostics(self) -> None:
        stderr = self._stderr
        self._stderr = None
        if stderr is not None:
            with suppress(OSError):
                stderr.close()


class _ServerStart:
    """Brings one worker up, keeping the spawned process reachable throughout."""

    def __init__(self, worker: WhisperServerWorker) -> None:
        self.worker = worker

    async def run(self, deadline: float | None) -> None:
        """Spawn a ready worker, retrying a port the server could not take.

        The port is chosen by binding one and letting it go, so another process
        can claim it in between; whisper-server then exits instead of serving.
        That is indistinguishable from a build without server support, and
        retiring the worker for it would cost every later request the reload —
        so a refused start is retried on a fresh port before giving up.
        """
        attempts = START_ATTEMPTS
        while attempts:
            attempts -= 1
            if await self._attempt(deadline):
                return
        detail = self.worker.diagnostics()[-MAXIMUM_SERVER_ERROR_LENGTH:]
        raise WhisperServerUnavailable(
            f"The installed whisper.cpp build cannot serve the selected model. {detail}".strip()
        )

    async def _attempt(self, deadline: float | None) -> bool:
        port = _loopback_port()
        try:
            process = self._spawn(port)
        except OSError as error:
            # The binary was there when the engine was built and is not now.
            # Reported as an unavailable worker so the caller falls back to the
            # CLI instead of failing the recording outright.
            raise WhisperServerUnavailable(
                "The whisper.cpp worker could not be launched."
            ) from error
        self.worker.adopt(process, port)
        try:
            became_ready = await self._await_ready(process, port, deadline)
        except BaseException:
            # Includes cancellation: the process is this worker's either way,
            # and leaving it running would hold a whole model in memory.
            self.worker.stop()
            raise
        if not became_ready:
            self.worker.stop()
        return became_ready

    def _spawn(self, port: int) -> subprocess.Popen[bytes]:
        return subprocess.Popen(
            self._arguments(port),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=self.worker.open_diagnostics(),
        )

    def _arguments(self, port: int) -> list[str]:
        worker = self.worker
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
            # The server's own default decodes timestamps and then derives
            # token timestamps from them. A dictation client shows neither, so
            # this is the launch-time half of the per-request opt-out.
            "-nt",
        ]

    async def _await_ready(
        self, process: subprocess.Popen[bytes], port: int, deadline: float | None
    ) -> bool:
        """Wait for the worker to accept connections. False once it has exited.

        A worker that is merely slow is still loading the model, so waiting it
        out is right; one that never answers is reported, because a retry would
        only spend the same timeout again.
        """
        limit = time.monotonic() + time_budget(deadline, START_TIMEOUT_SECONDS)
        while time.monotonic() < limit:
            if process.poll() is not None:
                return False
            if await _accepts(port):
                return True
            await asyncio.sleep(START_POLL_SECONDS)
        raise WhisperServerUnavailable("The whisper.cpp worker did not become ready.")


async def _accepts(port: int) -> bool:
    """The port only accepts once the model is loaded, so a connect means ready."""
    try:
        connection = await asyncio.wait_for(
            asyncio.open_connection(LOOPBACK_HOST, port), timeout=CONNECT_TIMEOUT_SECONDS
        )
    except (OSError, TimeoutError):
        return False
    writer = connection[1]
    writer.close()
    with suppress(OSError, ConnectionError):
        await writer.wait_closed()
    return True


def _loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind((LOOPBACK_HOST, 0))
        return int(listener.getsockname()[1])


@dataclass(frozen=True, slots=True)
class _Reply:
    status: int
    body: bytes


class _ServerHttp:
    """The `/inference` client, written on asyncio streams so it can be cancelled.

    Closing the socket is what stops the worker: the pinned server passes
    `is_connection_closed()` to whisper.cpp's abort callback, so a cancelled or
    timed-out request drops the decode instead of leaving the model busy until
    it finishes work nobody is waiting for. A blocking client in a thread
    cannot do that — the socket stays open until the thread returns.
    """

    @classmethod
    async def transcribe(cls, port: int, audio_path: Path, language: str) -> str:
        audio = await asyncio.to_thread(audio_path.read_bytes)
        reply = await cls._exchange(port, cls._inference_body(language, audio_path.name, audio))
        if reply.status != HTTP_OK:
            detail = reply.body.decode("utf-8", errors="replace")[-MAXIMUM_SERVER_ERROR_LENGTH:]
            raise errors.TranscriptionProcessError(
                f"The whisper.cpp worker rejected the audio: {detail}"
            )
        transcript = cls._decode(reply.body)
        if not transcript:
            raise errors.TranscriptionProcessError("The transcription result was empty.")
        return transcript

    @classmethod
    async def answers(cls, port: int) -> bool:
        """True when `/inference` reaches its handler again.

        The handler takes the model mutex before it looks at the request, so a
        fieldless probe that comes back — with the 400 it deserves — proves the
        previous decode has been released rather than still running.
        """
        probe = cls._body([("response_format", TEXT_RESPONSE_FORMAT)], b"")
        try:
            await asyncio.wait_for(cls._exchange(port, probe), timeout=ABORT_GRACE_SECONDS)
        except (TimeoutError, WhisperServerUnavailable, errors.TranscriptionProcessError):
            return False
        return True

    @classmethod
    async def _exchange(cls, port: int, body: bytes) -> _Reply:
        try:
            reader, writer = await asyncio.open_connection(LOOPBACK_HOST, port)
        except OSError as error:
            raise WhisperServerUnavailable("The whisper.cpp worker is unreachable.") from error
        try:
            writer.write(cls._head(port, len(body)) + body)
            await writer.drain()
            return await cls._read_reply(reader)
        except (OSError, EOFError, asyncio.LimitOverrunError) as error:
            raise WhisperServerUnavailable("The whisper.cpp worker is unreachable.") from error
        finally:
            # Reached on cancellation too, which is the point: the FIN is what
            # tells the worker to abandon the decode.
            writer.close()

    @classmethod
    def _head(cls, port: int, length: int) -> bytes:
        lines = [
            f"POST {INFERENCE_PATH} HTTP/1.1",
            f"Host: {LOOPBACK_HOST}:{port}",
            "Connection: close",
            f"Content-Type: multipart/form-data; boundary={MULTIPART_BOUNDARY}",
            f"Content-Length: {length}",
        ]
        return (CRLF.join(lines) + CRLF + CRLF).encode()

    @classmethod
    async def _read_reply(cls, reader: asyncio.StreamReader) -> _Reply:
        head = await reader.readuntil(HEADER_TERMINATOR)
        lines = head.decode("latin-1").split(CRLF)
        try:
            status = int(lines[0].split(" ")[1])
        except (IndexError, ValueError) as error:
            raise errors.TranscriptionProcessError(
                "The whisper.cpp worker returned an invalid response."
            ) from error
        length = cls._content_length(lines[1:])
        # The pinned server answers from a string, so it always sets
        # Content-Length; reading to EOF is only the safety net, and `Connection:
        # close` is what makes it terminate.
        body = await (reader.read() if length is None else reader.readexactly(length))
        return _Reply(status=status, body=body)

    @classmethod
    def _content_length(cls, header_lines: list[str]) -> int | None:
        for line in header_lines:
            name, _, raw = line.partition(":")
            if name.strip().lower() == "content-length" and raw.strip().isdigit():
                return int(raw.strip())
        return None

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
    def _inference_body(cls, language: str, file_name: str, audio: bytes) -> bytes:
        fields = [
            ("response_format", TEXT_RESPONSE_FORMAT),
            ("language", language),
            # The server's defaults are `no_timestamps=false` and, with
            # `token_timestamps` unset, `token_timestamps = !no_timestamps`.
            # `response_format=text` only hides the timestamps — it still pays
            # for decoding them and for the token-level pass over the signal.
            ("no_timestamps", "true"),
            ("token_timestamps", "false"),
        ]
        return cls._body(fields, audio, file_name=file_name)

    @classmethod
    def _body(cls, fields: list[tuple[str, str]], audio: bytes, *, file_name: str = "") -> bytes:
        chunks = [cls._field_chunk(name, entry) for name, entry in fields]
        if file_name:
            chunks.append(cls._file_header(file_name))
            chunks.append(audio)
            chunks.append(CRLF.encode())
        chunks.append(f"--{MULTIPART_BOUNDARY}--{CRLF}".encode())
        return b"".join(chunks)

    @classmethod
    def _file_header(cls, file_name: str) -> bytes:
        safe_name = file_name.replace('"', "")
        header = (
            f"--{MULTIPART_BOUNDARY}{CRLF}"
            f'Content-Disposition: form-data; name="file"; filename="{safe_name}"{CRLF}'
            f"Content-Type: audio/wav{CRLF}{CRLF}"
        )
        return header.encode()

    @classmethod
    def _field_chunk(cls, name: str, field_value: str) -> bytes:
        return (
            f"--{MULTIPART_BOUNDARY}{CRLF}"
            f'Content-Disposition: form-data; name="{name}"{CRLF}'
            f"{CRLF}{field_value}{CRLF}"
        ).encode()
