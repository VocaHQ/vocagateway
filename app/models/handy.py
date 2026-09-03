from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from app.errors import EngineUnavailableError, TranscriptionProcessError
from app.models.base import EngineHealth, TranscriptionOptions
from app.models.warmup import prefetch_model_paths

TRANSCRIPTION_TIMEOUT_SECONDS = 75
MAXIMUM_ERROR_MESSAGE_LENGTH = 200


class HandyEngine:
    """Adapter for Handy's headless file-transcription interface."""

    def __init__(
        self,
        binary: Path,
        model: str | None = None,
        *,
        fallback_model: str | None = None,
        settings_file: Path | None = None,
        huggingface_cache: Path | None = None,
    ) -> None:
        self.binary = binary
        self.settings_file = (
            settings_file
            or Path("~/Library/Application Support/com.pais.handy/settings_store.json").expanduser()
        )
        self.huggingface_cache = huggingface_cache or Path("~/.cache/huggingface/hub").expanduser()
        # An explicit model pins the adapter. Without one, follow Handy's
        # persisted selection so changing models in the app takes effect without
        # rebuilding or restarting the gateway.
        self.model = model
        self.fallback_model = fallback_model

    async def health(self) -> EngineHealth:
        app_selected_model = self._selected_model()
        available_models = self._downloaded_models()
        selected_model = (
            available_models[0] if available_models else app_selected_model or "no-model-selected"
        )
        is_executable = self.binary.is_file() and os.access(self.binary, os.X_OK)
        ready = is_executable and bool(available_models)
        return EngineHealth(
            ready=ready,
            name=f"handy:{selected_model}",
        )

    async def warmup(self) -> int:
        paths = [
            path
            for model in self._downloaded_models()
            if (path := _model_path(self.huggingface_cache, model))
        ]
        if not paths or not (await self.health()).ready:
            return 0
        return await asyncio.to_thread(prefetch_model_paths, paths)

    async def transcribe(self, audio_path: Path, options: TranscriptionOptions) -> str:
        health = await self.health()
        if not health.ready:
            raise EngineUnavailableError(
                "Handy, its selected model, or the downloaded model file is unavailable."
            )
        models = list(self._downloaded_models())
        last_error: TranscriptionProcessError | None = None
        index = 0
        while index < len(models):
            candidate_model = models[index]
            index += 1
            try:
                return await self._transcribe_with_model(audio_path, candidate_model)
            except TranscriptionProcessError as error:
                last_error = error
        if last_error is not None:
            raise last_error
        raise EngineUnavailableError("No downloaded Handy transcription model is available.")

    async def _transcribe_with_model(self, audio_path: Path, model: str) -> str:
        arguments = [
            str(self.binary),
            "--transcribe-file",
            str(audio_path),
            "--json",
            "--model",
            model,
        ]
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
            raise TranscriptionProcessError("Handy transcription timed out.") from error
        if process.returncode != 0:
            raise TranscriptionProcessError(_format_handy_error(stderr))
        return _parse_handy_output(stdout)

    def _downloaded_models(self) -> list[str]:
        models: list[str] = []
        selected_model = self._selected_model()
        for model in (selected_model, self.fallback_model):
            is_downloaded = _model_path(self.huggingface_cache, model) is not None
            if model and model not in models and is_downloaded:
                models.append(model)
        return models

    def _selected_model(self) -> str | None:
        return self.model or _read_selected_model(self.settings_file)


def _parse_handy_output(stdout: bytes) -> str:
    try:
        payload = json.loads(stdout)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise TranscriptionProcessError(
            "Handy returned an invalid transcription response."
        ) from error
    if not isinstance(payload, dict):
        raise TranscriptionProcessError("Handy returned an invalid transcription response.")
    transcript = payload.get("text")
    if not isinstance(transcript, str) or not transcript.strip():
        raise TranscriptionProcessError("Handy returned an empty transcript.")
    return transcript.strip()


def _format_handy_error(stderr: bytes) -> str:
    lines = stderr.decode("utf-8", errors="replace").strip().splitlines()
    detail = lines[-1][:MAXIMUM_ERROR_MESSAGE_LENGTH] if lines else "unknown Handy error"
    return f"Handy exited unsuccessfully: {detail}"


def _read_selected_model(settings_path: Path) -> str | None:
    try:
        payload = json.loads(settings_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if isinstance(payload, dict):
        settings_dict = payload.get("settings")
        if isinstance(settings_dict, dict):
            selected = settings_dict.get("selected_model")
            if isinstance(selected, str) and selected:
                return selected
    return None


def _model_path(cache_dir: Path, model: str | None) -> Path | None:
    if not model:
        return None
    components = model.split("/")
    if len(components) < 3:
        return None
    repo_slug = "/".join(components[:-1]).replace("/", "--")
    snapshots = cache_dir / f"models--{repo_slug}" / "snapshots"
    filename = components[-1]
    pattern = f"*/{filename}"
    return next((candidate for candidate in snapshots.glob(pattern) if candidate.is_file()), None)
