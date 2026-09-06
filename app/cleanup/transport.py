"""A minimal, cancellable JSON-over-HTTP client for one fixed local endpoint.

Written on asyncio streams rather than a general HTTP library for three
reasons, all of which the cleanup threat model asks for directly:

* Closing the socket is what stops the backend. A blocking client in a thread
  keeps the connection open until the thread returns, so a timed-out request
  would leave the model generating tokens nobody is waiting for.
* There is no URL to be talked into. The host and port are fixed by the caller,
  redirects are never followed, and no proxy environment variable is consulted.
* The response body is bounded before it is decoded, so a runaway generation
  cannot be read into memory on the strength of its own Content-Length.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any

CRLF = "\r\n"
CRLF_BYTES = b"\r\n"
HEADER_TERMINATOR = b"\r\n\r\n"
HEXADECIMAL = 16
HTTP_OK = 200
# Bounds the whole exchange, separately from any ceiling on the decoded text.
MAXIMUM_BODY_BYTES = 262_144
MAXIMUM_HEADER_BYTES = 16_384
JSON_MEDIA_TYPE = "application/json"
USER_AGENT = "vocagateway-cleanup/1"
# Every way a loopback exchange can fail at the socket layer. Collected so the
# caller sees one "the runtime went away" instead of four transport details.
_STREAM_FAILURES = (OSError, EOFError, asyncio.LimitOverrunError, asyncio.IncompleteReadError)


class TransportError(Exception):
    """The endpoint was unreachable, answered badly, or answered too much."""


@dataclass(frozen=True, slots=True)
class Endpoint:
    host: str
    port: int
    api_key: str | None = None

    @property
    def authority(self) -> str:
        display = f"[{self.host}]" if ":" in self.host else self.host
        return f"{display}:{self.port}"


@dataclass(frozen=True, slots=True)
class Reply:
    status: int
    body: bytes

    def json(self) -> Any:
        try:
            return json.loads(self.body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise TransportError("The runtime returned a malformed JSON body.") from error


async def get_json(endpoint: Endpoint, path: str, *, budget: float) -> Reply:
    return await _run(endpoint, "GET", path, None, budget)


async def post_json(
    endpoint: Endpoint, path: str, payload: dict[str, Any], *, budget: float
) -> Reply:
    encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    return await _run(endpoint, "POST", path, encoded, budget)


async def _run(
    endpoint: Endpoint, method: str, path: str, body: bytes | None, budget: float
) -> Reply:
    try:
        return await asyncio.wait_for(_exchange(endpoint, method, path, body), timeout=budget)
    except TimeoutError as error:
        raise TimeoutError("The cleanup runtime did not answer in time.") from error


async def _exchange(endpoint: Endpoint, method: str, path: str, body: bytes | None) -> Reply:
    try:
        reader, writer = await asyncio.open_connection(endpoint.host, endpoint.port)
    except OSError as error:
        raise TransportError("The cleanup runtime is unreachable.") from error
    try:
        writer.write(_head(endpoint, method, path, body))
        if body is not None:
            writer.write(body)
        await writer.drain()
        return await _read_reply(reader)
    except _STREAM_FAILURES as error:
        raise TransportError("The cleanup runtime closed the connection.") from error
    finally:
        # Reached on cancellation too, and that is the point: the FIN is what
        # asks the backend to abandon the generation.
        writer.close()


def _head(endpoint: Endpoint, method: str, path: str, body: bytes | None) -> bytes:
    length = 0 if body is None else len(body)
    lines = [
        f"{method} {path} HTTP/1.1",
        f"Host: {endpoint.authority}",
        f"User-Agent: {USER_AGENT}",
        "Connection: close",
        "Accept: application/json",
        f"Content-Length: {length}",
    ]
    if body is not None:
        lines.insert(-1, f"Content-Type: {JSON_MEDIA_TYPE}")
    if endpoint.api_key:
        lines.append(f"Authorization: Bearer {endpoint.api_key}")
    return (CRLF.join(lines) + CRLF + CRLF).encode("utf-8")


async def _read_reply(reader: asyncio.StreamReader) -> Reply:
    head = await reader.readuntil(HEADER_TERMINATOR)
    if len(head) > MAXIMUM_HEADER_BYTES:
        raise TransportError("The cleanup runtime sent oversized headers.")
    lines = head.decode("latin-1").split(CRLF)
    return Reply(status=_status(lines[0]), body=await _read_body(reader, _headers(lines[1:])))


def _status(status_line: str) -> int:
    try:
        return int(status_line.split(" ")[1])
    except (IndexError, ValueError) as error:
        raise TransportError("The cleanup runtime sent an invalid status line.") from error


def _headers(header_lines: list[str]) -> dict[str, str]:
    headers: dict[str, str] = {}
    for line in header_lines:
        name, separator, raw = line.partition(":")
        if separator:
            headers[name.strip().lower()] = raw.strip()
    return headers


async def _read_body(reader: asyncio.StreamReader, headers: dict[str, str]) -> bytes:
    if "chunked" in headers.get("transfer-encoding", "").lower():
        return await _read_chunked(reader)
    length = headers.get("content-length", "")
    if length.isdigit():
        return await _read_exactly(reader, int(length))
    return await _read_until_eof(reader)


async def _read_exactly(reader: asyncio.StreamReader, length: int) -> bytes:
    if length > MAXIMUM_BODY_BYTES:
        raise TransportError("The cleanup runtime announced an oversized body.")
    return await reader.readexactly(length)


async def _read_until_eof(reader: asyncio.StreamReader) -> bytes:
    """Drain to EOF, bounded.

    Looped rather than read in one call because `StreamReader.read(n)` returns
    *up to* n bytes: it stops at whatever the first segment happened to carry,
    so a single call truncates any answer that arrives in more than one piece
    and hands the caller a half a JSON document.
    """
    body = bytearray()
    while True:
        chunk = await reader.read(MAXIMUM_BODY_BYTES + 1 - len(body))
        if not chunk:
            return bytes(body)
        body.extend(chunk)
        if len(body) > MAXIMUM_BODY_BYTES:
            raise TransportError("The cleanup runtime sent an oversized body.")


async def _read_chunked(reader: asyncio.StreamReader) -> bytes:
    chunks: list[bytes] = []
    received = 0
    while True:
        header = await reader.readuntil(CRLF_BYTES)
        try:
            size = int(header.split(b";", 1)[0], HEXADECIMAL)
        except ValueError as error:
            raise TransportError("The cleanup runtime sent an invalid chunk header.") from error
        if not size:
            return b"".join(chunks)
        received += size
        if received > MAXIMUM_BODY_BYTES:
            raise TransportError("The cleanup runtime sent an oversized body.")
        chunks.append(await reader.readexactly(size))
        await reader.readexactly(len(CRLF_BYTES))
