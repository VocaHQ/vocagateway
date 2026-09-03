from __future__ import annotations

import re
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, BinaryIO
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, Request, Response, UploadFile
from fastapi.routing import APIRoute
from starlette import status

from app import audio, context, errors, schemas

_LANGUAGE_PATTERN = re.compile(r"^[A-Za-z-]+$|^auto$")
_TRUTHY = frozenset(("true", "1", "yes"))
_MAX_LANGUAGE_LENGTH = 20
_READ_CHUNK_BYTES = 65_536
_MINIMUM_AUDIO_UPLOAD_BYTES = 128
# Multipart wrapping (boundaries, disposition headers) sits on top of the audio
# bytes. Slack lets a file at the cap through the header check; the copy loop
# still enforces maximum_upload_bytes on the file itself.
_MULTIPART_WRAP_SLACK = 65536
_JSON_FORMATS = frozenset(("", "json"))


class _UploadLimit:
    @classmethod
    def reject_oversized_multipart(
        cls, request: Request, ctx: context.GatewayContextDependency
    ) -> None:
        length = cls._content_length(request)
        if length > ctx.settings.maximum_upload_bytes + _MULTIPART_WRAP_SLACK:
            raise errors.APIProblem(
                status.HTTP_413_CONTENT_TOO_LARGE,
                "audio_too_large",
                "The recording exceeds the upload limit.",
            )

    @classmethod
    def _content_length(cls, request: Request) -> int:
        raw = request.headers.get("content-length")
        if raw is None:
            raise errors.APIProblem(
                status.HTTP_411_LENGTH_REQUIRED,
                "length_required",
                "Content-Length is required.",
            )
        return cls._parse_length(raw)

    @classmethod
    def _parse_length(cls, raw: str) -> int:
        try:
            length = int(raw)
        except ValueError as error:
            raise errors.APIProblem(
                status.HTTP_400_BAD_REQUEST,
                "invalid_content_length",
                "Content-Length is invalid.",
            ) from error
        if length < 0:
            raise errors.APIProblem(
                status.HTTP_400_BAD_REQUEST,
                "invalid_content_length",
                "Content-Length is invalid.",
            )
        return length


class _GuardedUploadHandler:
    def __init__(self, original: Callable[[Request], Coroutine[object, object, Response]]) -> None:
        self.original = original

    async def __call__(self, request: Request) -> Response:
        ctx = context.get_context(request)
        if not ctx.token_is_valid(request.headers.get("authorization")):
            raise errors.APIProblem(
                status.HTTP_401_UNAUTHORIZED,
                "unauthorized",
                "A valid bearer token is required.",
            )
        _UploadLimit.reject_oversized_multipart(request, ctx)
        return await self.original(request)


class _EarlyUploadLimitRoute(APIRoute):
    """Run auth and the Content-Length cap before FastAPI spools multipart.

    `get_request_handler` calls `request.form()` before it solves router
    dependencies whenever the endpoint has File()/Form() parameters. A 413
    raised from a dependency would still parse the body. Wrapping the handler
    is what fails a missing, invalid, or oversized Content-Length closed
    without reading the body.
    """

    def get_route_handler(self) -> Callable[[Request], Coroutine[object, object, Response]]:
        return _GuardedUploadHandler(super().get_route_handler())


router = APIRouter(
    route_class=_EarlyUploadLimitRoute,
    dependencies=[
        Depends(context.require_token),
        Depends(_UploadLimit.reject_oversized_multipart),
    ],
)


class _AudioForm:
    @classmethod
    def suffix(cls, content_type: str | None, filename: str | None) -> str:
        normalized = cls._normalized_type(content_type, filename)
        suffix = audio.ALLOWED_AUDIO_TYPES.get(normalized)
        if suffix is None:
            raise errors.APIProblem(
                status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                "unsupported_audio_type",
                "This audio type is not supported.",
            )
        return suffix

    @classmethod
    def language(cls, language: str | None) -> str:
        language_value = (language or "").strip()
        if not language_value or language_value.lower() == "auto":
            return "auto"
        if cls._invalid_language(language_value):
            raise errors.APIProblem(
                status.HTTP_422_UNPROCESSABLE_CONTENT,
                "invalid_language",
                "Language must be auto or a language tag.",
            )
        return language_value

    @classmethod
    def reject_unsupported_format(cls, response_format: str | None) -> None:
        if response_format is None:
            return
        if response_format.strip().lower() in _JSON_FORMATS:
            return
        raise errors.APIProblem(
            status.HTTP_400_BAD_REQUEST,
            "unsupported_response_format",
            "Only the json response format is supported.",
        )

    @classmethod
    def reject_streaming(cls, stream: str | None) -> None:
        if (stream or "").strip().lower() in _TRUTHY:
            raise errors.APIProblem(
                status.HTTP_400_BAD_REQUEST,
                "streaming_not_supported",
                "Streaming transcription is not supported on this endpoint.",
            )

    @classmethod
    def _normalized_type(cls, content_type: str | None, filename: str | None) -> str:
        header = (content_type or "").split(";", maxsplit=1)[0]
        normalized = header.strip().lower()
        if normalized:
            return normalized
        filename_suffix = Path(filename or "").suffix.lower()
        for mime, allowed_suffix in audio.ALLOWED_AUDIO_TYPES.items():
            if allowed_suffix == filename_suffix:
                return mime
        return normalized

    @classmethod
    def _invalid_language(cls, language_value: str) -> bool:
        too_long = len(language_value) > _MAX_LANGUAGE_LENGTH
        return too_long or _LANGUAGE_PATTERN.fullmatch(language_value) is None


@dataclass
class _TranscriptionFields:
    model: Annotated[str | None, Form()] = None
    language: Annotated[str | None, Form()] = None
    response_format: Annotated[str | None, Form()] = None
    stream: Annotated[str | None, Form()] = None


class _AdhocUpload:
    def __init__(self, ctx: context.GatewayContext, audio_file: UploadFile, suffix: str) -> None:
        self.ctx = ctx
        self.audio_file = audio_file
        upload_dir = ctx.settings.data_dir / "transcriptions"
        temporary, final = audio.atomic_upload_path(upload_dir, str(uuid4()), suffix)
        self.temporary = temporary
        self.final = final
        self.received = 0

    async def store(self) -> Path:
        try:
            await self._copy_chunks()
        except BaseException:
            self.temporary.unlink(missing_ok=True)
            raise
        audio.complete_atomic_upload(self.temporary, self.final)
        return self.final

    async def _copy_chunks(self) -> None:
        maximum_upload_bytes = self.ctx.settings.maximum_upload_bytes
        with self.temporary.open("wb") as output:
            await self._write_stream(output, maximum_upload_bytes)
        if self.received < _MINIMUM_AUDIO_UPLOAD_BYTES:
            raise errors.APIProblem(
                status.HTTP_422_UNPROCESSABLE_CONTENT,
                "audio_empty",
                "The recording is empty.",
            )

    async def _write_stream(self, output: BinaryIO, maximum_upload_bytes: int) -> None:
        while True:
            chunk = await self.audio_file.read(_READ_CHUNK_BYTES)
            if not chunk:
                return
            self.received += len(chunk)
            if self.received > maximum_upload_bytes:
                raise errors.APIProblem(
                    status.HTTP_413_CONTENT_TOO_LARGE,
                    "audio_too_large",
                    "The recording exceeds the upload limit.",
                )
            output.write(chunk)


class _TranscriptionEndpoint:
    @classmethod
    async def create(
        cls,
        audio_file: Annotated[UploadFile, File(alias="file")],
        ctx: context.GatewayContextDependency,
        fields: Annotated[_TranscriptionFields, Depends()],
    ) -> schemas.OpenAITranscriptionResponse:
        try:
            return await cls._transcribe(audio_file, ctx, fields)
        finally:
            await audio_file.close()

    @classmethod
    async def _transcribe(
        cls,
        audio_file: UploadFile,
        ctx: context.GatewayContext,
        fields: _TranscriptionFields,
    ) -> schemas.OpenAITranscriptionResponse:
        if not audio_file.filename:
            raise errors.APIProblem(
                status.HTTP_400_BAD_REQUEST, "missing_file", "An audio file is required."
            )
        _AudioForm.reject_unsupported_format(fields.response_format)
        _AudioForm.reject_streaming(fields.stream)
        suffix = _AudioForm.suffix(audio_file.content_type, audio_file.filename)
        chosen_language = _AudioForm.language(fields.language)
        stored = await _AdhocUpload(ctx, audio_file, suffix).store()
        return await cls._read_transcript(ctx, stored, chosen_language)

    @classmethod
    async def _read_transcript(
        cls,
        ctx: context.GatewayContext,
        stored: Path,
        chosen_language: str,
    ) -> schemas.OpenAITranscriptionResponse:
        try:
            transcription = await ctx.service.transcribe_adhoc(stored, chosen_language)
        finally:
            stored.unlink(missing_ok=True)
        return schemas.OpenAITranscriptionResponse(text=transcription.transcript)


reject_oversized_multipart = _UploadLimit.reject_oversized_multipart
router.add_api_route(
    "/v1/audio/transcriptions",
    _TranscriptionEndpoint.create,
    methods=["POST"],
    response_model=schemas.OpenAITranscriptionResponse,
)
