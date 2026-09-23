# Copyright (c) 2025-present VocaHQ, Inc.
# SPDX-License-Identifier: AGPL-3.0-only
# See the LICENSE file for details.

from __future__ import annotations

import asyncio
import json
import sys
import threading
import time
from array import array
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, cast

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app import context, scripts, serializers
from app.cleanup.base import MODE_INHERIT, REQUESTABLE_MODES, CleanupOptions, CleanupStatus
from app.models.base import StreamingEngine, TranscriptionEngine

router = APIRouter()

WEBSOCKET_UNAUTHORIZED_CODE = 4401
WEBSOCKET_UNSUPPORTED_ENGINE_CODE = 4409
WEBSOCKET_ENGINE_UNAVAILABLE_CODE = 4410
WEBSOCKET_INTERNAL_ERROR_CODE = 1011
MINIMUM_SAMPLE_RATE_HZ = 8_000
MAXIMUM_SAMPLE_RATE_HZ = 96_000
MAXIMUM_STREAM_ERROR_LENGTH = 200
MESSAGE_TYPE_KEY = "type"
WRITING_STYLES = frozenset(("raw", "clean", "formal", "casual", "very_casual", "excited"))
COMPLETE_MESSAGE = "complete"
# Statuses worth telling a streaming client about. `disabled` is the shape every
# existing client already gets, so the packet stays exactly as it is today.
_REPORTED_STATUSES = frozenset(
    (
        CleanupStatus.SKIPPED,
        CleanupStatus.UNCHANGED,
        CleanupStatus.APPLIED,
        CleanupStatus.FALLBACK,
    )
)


@dataclass(frozen=True, slots=True)
class PendingFinal:
    """A finished recognition, waiting for formatting outside the engine's locks."""

    transcript: str
    style: str
    language: str
    cleanup_mode: str
    started_at: float


class _StreamGate:
    @classmethod
    async def engine_or_close(
        cls, websocket: WebSocket, selected_engine: TranscriptionEngine
    ) -> StreamingEngine | None:
        ctx: context.GatewayContext = websocket.app.state.ctx
        if not ctx.token_is_valid(websocket.headers.get("authorization")):
            await websocket.close(code=WEBSOCKET_UNAUTHORIZED_CODE, reason="Unauthorized")
            return None
        if isinstance(selected_engine, StreamingEngine) and selected_engine.supports_streaming:
            return await cls._ready_or_close(websocket, selected_engine)
        await cls._reject_unsupported(websocket, selected_engine)
        return None

    @classmethod
    async def send_error(cls, websocket: WebSocket, error: Exception) -> None:
        with suppress(RuntimeError):
            await websocket.send_json(
                {
                    MESSAGE_TYPE_KEY: "error",
                    "message": str(error)[-MAXIMUM_STREAM_ERROR_LENGTH:] or "Streaming failed.",
                }
            )
            await websocket.close(code=WEBSOCKET_INTERNAL_ERROR_CODE)

    @classmethod
    async def close_stream(cls, stream: object | None) -> None:
        if stream is None:
            return
        closer = getattr(stream, "close", None)
        if callable(closer):
            with suppress(Exception):
                await asyncio.to_thread(closer)

    @classmethod
    async def _ready_or_close(
        cls, websocket: WebSocket, selected_engine: StreamingEngine
    ) -> StreamingEngine | None:
        selected_health = await selected_engine.health()
        if selected_health.ready:
            await websocket.accept()
            return selected_engine
        await cls._reject_unready(websocket, selected_health.name)
        return None

    @classmethod
    async def _reject_unsupported(
        cls, websocket: WebSocket, selected_engine: TranscriptionEngine
    ) -> None:
        selected_health = await selected_engine.health()
        await websocket.accept()
        await websocket.send_json(
            {
                MESSAGE_TYPE_KEY: "unsupported",
                "reason": "active_engine",
                "engine": selected_health.name,
            }
        )
        await websocket.close(
            code=WEBSOCKET_UNSUPPORTED_ENGINE_CODE,
            reason="Active engine does not support streaming",
        )

    @classmethod
    async def _reject_unready(cls, websocket: WebSocket, engine_name: str) -> None:
        await websocket.accept()
        await websocket.send_json(
            {
                MESSAGE_TYPE_KEY: "unavailable",
                "reason": "engine_not_ready",
                "engine": engine_name,
            }
        )
        await websocket.close(
            code=WEBSOCKET_ENGINE_UNAVAILABLE_CODE,
            reason="Streaming engine is not ready",
        )


class _StreamPackets:
    def __init__(self, session: _StreamSession) -> None:
        self.session = session

    @classmethod
    def configure(cls, session: _StreamSession, start: object) -> None:
        payload = start if isinstance(start, dict) else {}
        if payload.get(MESSAGE_TYPE_KEY) != "start":
            raise ValueError("The first stream message must be start.")
        session._sample_rate = int(payload.get("sample_rate", 0))
        valid_rate = MINIMUM_SAMPLE_RATE_HZ <= session._sample_rate <= MAXIMUM_SAMPLE_RATE_HZ
        if not valid_rate:
            raise ValueError("Sample rate must be between 8000 and 96000 Hz.")
        session._language = str(payload.get("language", "auto"))
        session._style = str(payload.get("style", "casual"))
        if session._style not in WRITING_STYLES:
            raise ValueError("Unsupported writing style.")
        session._cleanup_mode = str(payload.get("cleanup", MODE_INHERIT))
        if session._cleanup_mode not in REQUESTABLE_MODES:
            raise ValueError("Unsupported cleanup mode.")

    async def audio(self, chunk: bytes) -> None:
        if len(chunk) % 4:
            raise ValueError("PCM chunks must contain float32 samples.")
        samples = array("f")
        samples.frombytes(chunk)
        if sys.byteorder != "little":
            samples.byteswap()
        session = self.session
        session._received_samples += len(samples)
        limit = session._sample_rate * session._ctx.settings.maximum_duration_seconds
        if session._received_samples > limit:
            raise ValueError("The stream exceeds the recording duration limit.")
        adder = cast(Any, session._stream).add_audio
        await asyncio.to_thread(adder, samples.tolist(), session._sample_rate)
        with session._lines_lock:
            partial = serializers.joined_stream_lines(session._lines)
        if partial:
            await session._websocket.send_json({MESSAGE_TYPE_KEY: "partial", "transcript": partial})

    async def finish(self) -> None:
        """Close out recognition and hand the raw text back for finalization.

        Formatting deliberately does not happen here. This runs inside the
        streaming lock and the engine lease, and awaiting a text model from
        under them would keep a speech model occupied for the whole correction.
        """
        session = self.session
        stream = cast(Any, session._stream)
        final_result = await asyncio.to_thread(stream.stop)
        transcript = self._recognized(final_result)
        if not transcript:
            raise ValueError("Moonshine returned an empty transcript.")
        await asyncio.to_thread(stream.close)
        session._stream = None
        session.pending = PendingFinal(
            transcript=transcript,
            style=session._style,
            language=session._language,
            cleanup_mode=session._cleanup_mode,
            started_at=session.started_at,
        )

    def _recognized(self, final_result: object) -> str:
        session = self.session
        with session._lines_lock:
            for line in getattr(final_result, "lines", []) or []:
                if getattr(line, "text", ""):
                    session._lines[int(line.line_id)] = str(line.text).strip()
            joined = serializers.joined_stream_lines(session._lines)
        if scripts.transcript_matches_language(joined, session._language):
            return joined
        raise ValueError(
            f"The model transcribed this as a different language than {session._language}."
        )


class _StreamSession:
    def __init__(self, websocket: WebSocket, engine: StreamingEngine) -> None:
        self._websocket = websocket
        self.started_at = time.monotonic()
        self._engine = engine
        self._ctx: context.GatewayContext = websocket.app.state.ctx
        self._stream: object | None = None
        self._lines: dict[int, str] = {}
        self._lines_lock = threading.Lock()
        self._sample_rate = 0
        self._style = "casual"
        self._language = "auto"
        self._cleanup_mode = MODE_INHERIT
        self._received_samples = 0
        # Set once recognition has finished. Read by the caller *after* the
        # engine lease is released, which is what keeps the optional text model
        # off the speech engine's critical path.
        self.pending: PendingFinal | None = None

    async def run(self) -> PendingFinal | None:
        try:
            await self._loop()
        except WebSocketDisconnect:
            return None
        except Exception as error:
            self._ctx.service.metrics.record_result(_elapsed_ms(self.started_at), success=False)
            await _StreamGate.send_error(self._websocket, error)
            return None
        finally:
            await _StreamGate.close_stream(self._stream)
        return self.pending

    async def _loop(self) -> None:
        start = await self._websocket.receive_json()
        _StreamPackets.configure(self, start)
        async with self._engine.streaming_lock:
            self._stream = await self._engine.create_stream()
            configure = getattr(self._engine, "configure_stream", None)
            if callable(configure):
                await asyncio.to_thread(configure, self._stream, self._language)
            await self._listen()

    async def _listen(self) -> None:
        listener = cast(Any, self._stream).add_listener
        await asyncio.to_thread(listener, self._receive_event)
        engine_name = (await self._engine.health()).name.split(":", 1)[0]
        await self._websocket.send_json({MESSAGE_TYPE_KEY: "ready", "engine": engine_name})
        await self._messages()

    def _receive_event(self, event: object) -> None:
        line = getattr(event, "line", None)
        text_value = ""
        line_id = None
        if line is not None:
            text_value = getattr(line, "text", "")
            line_id = getattr(line, "line_id", None)
        if isinstance(line_id, int) and text_value:
            with self._lines_lock:
                self._lines[line_id] = str(text_value).strip()

    async def _messages(self) -> None:
        packets = _StreamPackets(self)
        while True:
            message = await self._websocket.receive()
            if message.get(MESSAGE_TYPE_KEY) == "websocket.disconnect":
                return
            if await self._dispatch(packets, message):
                return

    async def _dispatch(self, packets: _StreamPackets, message: object) -> bool:
        chunk = message.get("bytes") if isinstance(message, dict) else None
        if chunk is not None:
            await packets.audio(chunk)
            return False
        text_message = message.get("text") if isinstance(message, dict) else None
        if not text_message:
            return False
        command = json.loads(text_message)
        if command.get(MESSAGE_TYPE_KEY) != "finish":
            return False
        await packets.finish()
        return True


class _StreamFinalizer:
    """Formats one finished recognition and sends the completion packet.

    Runs with no speech-engine lease and no streaming lock held. Exactly one
    cleanup pass happens per stream, and only here — partials are never
    rewritten, because a corrected partial that changes again a word later is
    worse than an uncorrected one.
    """

    def __init__(
        self, ctx: context.GatewayContext, websocket: WebSocket, pending: PendingFinal
    ) -> None:
        self.ctx = ctx
        self.websocket = websocket
        self.pending = pending

    async def send(self) -> None:
        try:
            payload = await self._payload()
            self.ctx.service.metrics.record_result(
                _elapsed_ms(self.pending.started_at), success=True
            )
        except asyncio.CancelledError:
            raise
        except Exception as error:  # noqa: BLE001 - reported to the client, not raised
            self.ctx.service.metrics.record_result(
                _elapsed_ms(self.pending.started_at), success=False
            )
            await _StreamGate.send_error(self.websocket, error)
            return
        with suppress(RuntimeError, WebSocketDisconnect):
            await self.websocket.send_json(payload)
            await self.websocket.close(code=1000)

    async def _payload(self) -> dict[str, Any]:
        pending = self.pending
        final = await self.ctx.service.cleanup.finalize(
            pending.transcript,
            style=pending.style,
            language=pending.language,
            options=self._options(),
        )
        payload: dict[str, Any] = {
            MESSAGE_TYPE_KEY: COMPLETE_MESSAGE,
            "transcript": final.transcript,
        }
        if final.cleanup.status in _REPORTED_STATUSES:
            payload["original_transcript"] = final.original_transcript
            payload["cleanup"] = serializers.cleanup_result(final.cleanup).model_dump(mode="json")
        return payload

    def _options(self) -> CleanupOptions:
        manager = self.ctx.cleanup
        if manager is None:
            return CleanupOptions()
        return manager.options(self.pending.cleanup_mode)


@router.websocket("/v1/stream")
async def stream_transcription(websocket: WebSocket) -> None:
    """Experimental float32 PCM stream for a streaming-capable engine.

    Any engine exposing `supports_streaming` (True), `streaming_lock`, and
    `create_stream()` returning an object with `add_listener`/`add_audio`/
    `stop` works here — currently Moonshine and the sherpa-onnx streaming
    zipformer model.
    """
    ctx: context.GatewayContext = websocket.app.state.ctx
    active = False
    try:
        async with ctx.engine_provider.lease() as selected_engine:
            engine = await _StreamGate.engine_or_close(websocket, selected_engine)
            if engine is None:
                return
            ctx.service.metrics.started(queued=False)
            active = True
            pending = await _StreamSession(websocket, engine).run()
        # Release speech resources before running cleanup.
        if pending is not None:
            await _StreamFinalizer(ctx, websocket, pending).send()
    finally:
        if active:
            ctx.service.metrics.finished()


def _elapsed_ms(started: float) -> int:
    return max(0, round((time.monotonic() - started) * 1000))
