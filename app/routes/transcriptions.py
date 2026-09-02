from __future__ import annotations

import re
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Annotated, Any
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, Request, Response, UploadFile
from fastapi.routing import APIRoute
from starlette.status import (
    HTTP_400_BAD_REQUEST,
    HTTP_401_UNAUTHORIZED,
    HTTP_411_LENGTH_REQUIRED,
    HTTP_413_CONTENT_TOO_LARGE,
    HTTP_415_UNSUPPORTED_MEDIA_TYPE,
    HTTP_422_UNPROCESSABLE_CONTENT,
)

from app.audio import ALLOWED_AUDIO_TYPES, atomic_upload_path, complete_atomic_upload
from app.context import GatewayContextDependency, get_context, require_token
from app.errors import APIProblem
from app.schemas import OpenAITranscriptionResponse

_LANGUAGE_PATTERN = re.compile(r"^[A-Za-z-]+$|^auto$")
_TRUTHY = frozenset(("true", "1", "yes"))
_MAX_LANGUAGE_LENGTH = 20
_READ_CHUNK_BYTES = 65_536
_MINIMUM_AUDIO_UPLOAD_BYTES = 128
# Multipart wrapping (boundaries, disposition headers) sits on top of the audio
# bytes. Slack lets a file at the cap through the header check; the copy loop
# still enforces maximum_upload_bytes on the file itself.
_MULTIPART_WRAP_SLACK = 65536


def reject_oversized_multipart(request: Request, ctx: GatewayContextDependency) -> None:
    raw = request.headers.get("content-length")
    if raw is None:
        raise APIProblem(HTTP_411_LENGTH_REQUIRED, "length_required", "Content-Length is required.")
    try:
        length = int(raw)
    except ValueError as error:
        raise APIProblem(
            HTTP_400_BAD_REQUEST, "invalid_content_length", "Content-Length is invalid."
        ) from error
    if length < 0:
        raise APIProblem(
            HTTP_400_BAD_REQUEST, "invalid_content_length", "Content-Length is invalid."
        )
    if length > ctx.settings.maximum_upload_bytes + _MULTIPART_WRAP_SLACK:
        raise APIProblem(
            HTTP_413_CONTENT_TOO_LARGE,
            "audio_too_large",
            "The recording exceeds the upload limit.",
        )


class _EarlyUploadLimitRoute(APIRoute):
    """Run auth and the Content-Length cap before FastAPI spools multipart.

    `get_request_handler` calls `request.form()` before it solves router
    dependencies whenever the endpoint has File()/Form() parameters. A 413
    raised from a dependency would still parse the body. Wrapping the handler
    is what fails a missing, invalid, or oversized Content-Length closed
    without reading the body.
    """

    def get_route_handler(self) -> Callable[[Request], Coroutine[Any, Any, Response]]:
        original = super().get_route_handler()

        async def handler(request: Request) -> Response:
            ctx = get_context(request)
            if not ctx.token_is_valid(request.headers.get("authorization")):
                raise APIProblem(
                    HTTP_401_UNAUTHORIZED, "unauthorized", "A valid bearer token is required."
                )
            reject_oversized_multipart(request, ctx)
            return await original(request)

        return handler


router = APIRouter(
    route_class=_EarlyUploadLimitRoute,
    dependencies=[Depends(require_token), Depends(reject_oversized_multipart)],
)


def _audio_suffix(content_type: str | None, filename: str | None) -> str:
    normalized = (content_type or "").split(";", maxsplit=1)[0].strip().lower()
    if not normalized:
        filename_suffix = Path(filename or "").suffix.lower()
        for mime, allowed_suffix in ALLOWED_AUDIO_TYPES.items():
            if allowed_suffix == filename_suffix:
                normalized = mime
                break
    suffix = ALLOWED_AUDIO_TYPES.get(normalized)
    if suffix is None:
        raise APIProblem(
            HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            "unsupported_audio_type",
            "This audio type is not supported.",
        )
    return suffix


def _normalize_language(language: str | None) -> str:
    language_value = (language or "").strip()
    if not language_value or language_value.lower() == "auto":
        return "auto"
    if (
        len(language_value) > _MAX_LANGUAGE_LENGTH
        or _LANGUAGE_PATTERN.fullmatch(language_value) is None
    ):
        raise APIProblem(
            HTTP_422_UNPROCESSABLE_CONTENT,
            "invalid_language",
            "Language must be auto or a language tag.",
        )
    return language_value


def _reject_unsupported_format(response_format: str | None) -> None:
    if response_format is None:
        return
    if response_format.strip().lower() in {"", "json"}:
        return
    raise APIProblem(
        HTTP_400_BAD_REQUEST,
        "unsupported_response_format",
        "Only the json response format is supported.",
    )


def _reject_streaming(stream: str | None) -> None:
    if (stream or "").strip().lower() in _TRUTHY:
        raise APIProblem(
            HTTP_400_BAD_REQUEST,
            "streaming_not_supported",
            "Streaming transcription is not supported on this endpoint.",
        )


@router.post("/v1/audio/transcriptions", response_model=OpenAITranscriptionResponse)
async def create_transcription(
    audio_file: Annotated[UploadFile, File(alias="file")],
    ctx: GatewayContextDependency,
    model: Annotated[str | None, Form()] = None,
    language: Annotated[str | None, Form()] = None,
    response_format: Annotated[str | None, Form()] = None,
    stream: Annotated[str | None, Form()] = None,
) -> OpenAITranscriptionResponse:
    # OpenAI clients send `model`. The engine loaded in the WebUI is what runs.
    try:
        if not audio_file.filename:
            raise APIProblem(HTTP_400_BAD_REQUEST, "missing_file", "An audio file is required.")
        _reject_unsupported_format(response_format)
        _reject_streaming(stream)
        suffix = _audio_suffix(audio_file.content_type, audio_file.filename)
        chosen_language = _normalize_language(language)

        upload_dir = ctx.settings.data_dir / "transcriptions"
        temporary, final = atomic_upload_path(upload_dir, str(uuid4()), suffix)
        received = 0
        maximum_upload_bytes = ctx.settings.maximum_upload_bytes
        try:
            with temporary.open("wb") as output:
                while True:
                    chunk = await audio_file.read(_READ_CHUNK_BYTES)
                    if not chunk:
                        break
                    received += len(chunk)
                    if received > maximum_upload_bytes:
                        raise APIProblem(
                            HTTP_413_CONTENT_TOO_LARGE,
                            "audio_too_large",
                            "The recording exceeds the upload limit.",
                        )
                    output.write(chunk)
            if received < _MINIMUM_AUDIO_UPLOAD_BYTES:
                raise APIProblem(
                    HTTP_422_UNPROCESSABLE_CONTENT, "audio_empty", "The recording is empty."
                )
            complete_atomic_upload(temporary, final)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        try:
            transcription = await ctx.service.transcribe_adhoc(final, chosen_language)
        finally:
            final.unlink(missing_ok=True)
        return OpenAITranscriptionResponse(text=transcription.transcript)
    finally:
        await audio_file.close()
