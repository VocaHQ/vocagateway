from __future__ import annotations

import asyncio
import json
import os
import plistlib
import shutil
import subprocess
import time
from pathlib import Path
from types import MappingProxyType

from app import errors
from app.models import base, whisperkit

DEFAULT_VOCAMAC_APP = Path("/Applications/VocaMac.app")
DEFAULT_SUPPORT_DIR = Path("~/Library/Application Support/VocaMac")
DEFAULT_PREFERENCES_FILE = Path("~/Library/Preferences/com.vocamac.app.plist")
SELECTED_MODEL_KEY = "vocamac.selectedModelSize"
MODEL_REPOSITORY = "argmaxinc/whisperkit-coreml"
FILE_PROBE_CHUNK_BYTES = 65_536
MAXIMUM_HEADLESS_ERROR_LENGTH = 300
REQUIRED_COMPONENTS = (
    "MelSpectrogram.mlmodelc",
    "AudioEncoder.mlmodelc",
    "TextDecoder.mlmodelc",
)
_MODEL_DEFINITIONS = ("model.mil", "model.mlmodel", "coremldata.bin")
_VOCAMAC_EXECUTABLE = Path("Contents/MacOS/VocaMac")
_HEADLESS_MARKER = b"--transcribe-file"
_HEADLESS_TIMEOUT_SECONDS = 300
MODEL_VARIANTS = MappingProxyType(
    {
        "tiny": "openai_whisper-tiny",
        "base": "openai_whisper-base",
        "small": "openai_whisper-small",
        "medium": "openai_whisper-medium",
        "large-v3": "openai_whisper-large-v3",
        "large-v3_turbo": "openai_whisper-large-v3_turbo",
        "large-v3-v20240930": "openai_whisper-large-v3-v20240930",
        "large-v3-v20240930_turbo": "openai_whisper-large-v3-v20240930_turbo",
        "large-v3-v20240930_626MB": "openai_whisper-large-v3-v20240930_626MB",
        "large-v3-v20240930_turbo_632MB": "openai_whisper-large-v3-v20240930_turbo_632MB",
        "distil-large-v3_594MB": "distil-whisper_distil-large-v3_594MB",
        "distil-large-v3_turbo_600MB": "distil-whisper_distil-large-v3_turbo_600MB",
    }
)


class _HeadlessModel:
    def __init__(self, model_id: str, downloaded: bool, supported: bool) -> None:
        self.id = model_id
        self.downloaded = downloaded
        self.supported = supported


class _ModelAssets:
    @classmethod
    def is_usable(cls, directory: Path) -> bool:
        return all(cls.component_is_usable(directory / name) for name in REQUIRED_COMPONENTS)

    @classmethod
    def component_is_usable(cls, component: Path) -> bool:
        metadata = (component / "metadata.json").is_file()
        definition = any((component / name).is_file() for name in _MODEL_DEFINITIONS)
        weights = (component / "weights" / "weight.bin").is_file()
        return metadata and definition and weights

    @classmethod
    def weight_bytes(cls, directory: Path) -> int:
        total = 0
        for name in REQUIRED_COMPONENTS:
            try:
                total += (directory / name / "weights" / "weight.bin").stat().st_size
            except OSError:
                return 0
        return total

    @classmethod
    def downloaded_directories(cls, models_dir: Path) -> list[Path]:
        try:
            return sorted(entry for entry in models_dir.iterdir() if entry.is_dir())
        except OSError:
            return []

    @classmethod
    def ranked_models(cls, models_dir: Path, preferred: Path | None) -> list[Path]:
        models = [
            directory
            for directory in cls.downloaded_directories(models_dir)
            if directory != preferred and cls.is_usable(directory)
        ]
        models.sort(key=cls.weight_bytes, reverse=True)
        if preferred is not None:
            models.insert(0, preferred)
        return models

    @classmethod
    def preference_selection(cls, preferences_file: Path) -> tuple[str | None, str | None]:
        try:
            with preferences_file.open("rb") as preference_file:
                payload = plistlib.load(preference_file)
        except (OSError, plistlib.InvalidFileException):
            return None, None
        try:
            selected = payload[SELECTED_MODEL_KEY]
        except (KeyError, TypeError):
            return None, None
        if not isinstance(selected, str):
            return None, None
        return selected, MODEL_VARIANTS.get(selected)

    @classmethod
    def missing_engine(cls, selected: str | None, variant: str | None) -> Exception:
        if selected is not None and variant is None:
            return errors.EngineUnavailableError(
                f"VocaMac selected '{selected}', which is not a WhisperKit model. "
                "This VocaMac build does not expose its other engines for headless "
                "transcription; select a Whisper model in VocaMac, or select the "
                "matching native engine and model in the gateway."
            )
        return errors.EngineUnavailableError(
            "VocaMac, one of its downloaded models, or the WhisperKit CLI is "
            "unavailable. Install VocaMac, download a model in its Models tab, "
            "and install the CLI with `brew install whisperkit-cli`."
        )


class _HeadlessIO:
    @classmethod
    def environment(cls) -> dict[str, str]:
        environment = os.environ.copy()
        environment["LLVM_PROFILE_FILE"] = os.devnull
        return environment

    @classmethod
    def contains_marker(cls, path: Path, marker: bytes) -> bool:
        try:
            binary_file = path.open("rb")
        except OSError:
            return False
        with binary_file:
            return cls._scan(binary_file, marker)

    @classmethod
    def raise_failure(cls, stderr: bytes) -> None:
        code, message = cls._decode_failure(stderr)
        if code in {"model_not_found", "model_not_downloaded", "model_unsupported"}:
            raise errors.EngineUnavailableError(message)
        raise errors.TranscriptionProcessError(message)

    @classmethod
    def transcription(cls, stdout: bytes, started: float) -> base.EngineTranscription:
        transcript, duration = cls._parse_stdout(stdout)
        total_ms = max(0, round((time.monotonic() - started) * 1000))
        inference_ms = 0
        if isinstance(duration, (int, float)) and not isinstance(duration, bool):
            inference_ms = max(0, round(duration * 1000))
        return base.EngineTranscription(
            text=transcript.strip(),
            model_load_ms=max(0, total_ms - inference_ms),
            inference_ms=inference_ms,
        )

    @classmethod
    def _decode_failure(cls, stderr: bytes) -> tuple[str, str]:
        detail = stderr.decode("utf-8", errors="replace").strip()
        try:
            payload = json.loads(detail)
        except (json.JSONDecodeError, TypeError):
            if not detail:
                return "", "VocaMac transcription failed."
            return "", detail.splitlines()[-1][:MAXIMUM_HEADLESS_ERROR_LENGTH]
        if not isinstance(payload, dict):
            return "", "VocaMac transcription failed."
        raw_code = payload.get("error")
        code = raw_code if isinstance(raw_code, str) else ""
        message = payload.get("message")
        if isinstance(message, str) and message:
            return str(code), message
        return str(code), "VocaMac transcription failed."

    @classmethod
    def _scan(cls, binary_file: object, marker: bytes) -> bool:
        overlap = b""
        while chunk := binary_file.read(FILE_PROBE_CHUNK_BYTES):  # type: ignore[attr-defined]
            combined = overlap + chunk
            if marker in combined:
                return True
            overlap = combined[-max(0, len(marker) - 1) :]
        return False

    @classmethod
    def _parse_stdout(cls, stdout: bytes) -> tuple[str, object]:
        try:
            payload = json.loads(stdout)
        except (json.JSONDecodeError, TypeError) as error:
            raise errors.TranscriptionProcessError(
                "VocaMac returned an invalid transcription response."
            ) from error
        try:
            transcript = payload["text"]
        except (KeyError, TypeError) as error:
            raise errors.TranscriptionProcessError(
                "VocaMac returned an invalid transcription response."
            ) from error
        if not isinstance(transcript, str) or not transcript.strip():
            raise errors.TranscriptionProcessError("VocaMac returned an empty transcript.")
        return transcript, payload.get("duration_seconds")


class _HeadlessCatalog:
    @classmethod
    def configured_model_id(cls, model: str | None) -> str | None:
        if not model:
            return None
        for model_id, folder_name in MODEL_VARIANTS.items():
            if model in {model_id, folder_name}:
                return model_id
        return model

    @classmethod
    def list_models(cls, executable: Path) -> list[object] | None:
        try:
            process = subprocess.run(
                [str(executable), "--list-models", "--json"],
                capture_output=True,
                check=False,
                timeout=5,
                env=_HeadlessIO.environment(),
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return cls._parse_models(process)

    @classmethod
    def match_model(cls, entries: list[object], configured: str | None) -> _HeadlessModel | None:
        for entry in entries:
            matched = cls._entry_model(entry, configured)
            if matched is not None:
                return matched
        return None

    @classmethod
    def _parse_models(cls, process: subprocess.CompletedProcess[bytes]) -> list[object] | None:
        if process.returncode != 0:
            return None
        try:
            payload = json.loads(process.stdout)
        except json.JSONDecodeError:
            return None
        try:
            entries = payload["models"]
        except (KeyError, TypeError):
            return None
        return entries if isinstance(entries, list) else None

    @classmethod
    def _entry_model(cls, entry: object, configured: str | None) -> _HeadlessModel | None:
        if not isinstance(entry, dict):
            return None
        model_id = entry.get("id")
        matches = model_id == configured if configured else entry.get("selected") is True
        if matches and isinstance(model_id, str):
            return _HeadlessModel(
                model_id,
                entry.get("downloaded") is True,
                entry.get("supported") is True,
            )
        return None


class _HeadlessClient:
    def __init__(self, engine: VocaMacEngine) -> None:
        self.engine = engine

    def executable(self) -> Path | None:
        candidate = self.engine.app_path / _VOCAMAC_EXECUTABLE
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
        return None

    def supported(self) -> bool:
        executable = self.executable()
        if executable is None:
            self.engine._headless_signature = None
            self.engine._headless_capable = False
            return False
        try:
            metadata = executable.stat()
        except OSError:
            self.engine._headless_signature = None
            self.engine._headless_capable = False
            return False
        signature = (metadata.st_size, metadata.st_mtime_ns)
        if signature != self.engine._headless_signature:
            self.engine._headless_capable = _HeadlessIO.contains_marker(
                executable, _HEADLESS_MARKER
            )
            self.engine._headless_signature = signature
        return self.engine._headless_capable

    def wanted_model_name(self) -> str:
        configured = _HeadlessCatalog.configured_model_id(self.engine.model)
        if configured:
            return configured
        selected, variant = _LegacyClient(self.engine).selection()
        return selected or variant or "no-model-selected"

    def model(self) -> _HeadlessModel | None:
        executable = self.executable()
        if executable is None:
            return None
        entries = _HeadlessCatalog.list_models(executable)
        if entries is None:
            return None
        configured = _HeadlessCatalog.configured_model_id(self.engine.model)
        return _HeadlessCatalog.match_model(entries, configured)

    async def transcribe(
        self, audio_path: Path, options: base.TranscriptionOptions
    ) -> base.EngineTranscription:
        executable = self.executable()
        if executable is None:
            raise errors.EngineUnavailableError("VocaMac's headless executable is unavailable.")
        started = time.monotonic()
        stdout, stderr, returncode = await self._communicate(executable, audio_path, options)
        if returncode != 0:
            _HeadlessIO.raise_failure(stderr)
        return _HeadlessIO.transcription(stdout, started)

    async def _communicate(
        self,
        executable: Path,
        audio_path: Path,
        options: base.TranscriptionOptions,
    ) -> tuple[bytes, bytes, int]:
        arguments = [str(executable), "--transcribe-file", str(audio_path), "--json"]
        configured_model = _HeadlessCatalog.configured_model_id(self.engine.model)
        if configured_model:
            arguments.extend(["--model", configured_model])
        arguments.extend(["--language", options.language])
        process = await asyncio.create_subprocess_exec(
            *arguments,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=_HeadlessIO.environment(),
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=_HEADLESS_TIMEOUT_SECONDS
            )
        except TimeoutError as error:
            process.kill()
            await process.wait()
            raise errors.TranscriptionProcessError("VocaMac transcription timed out.") from error
        return stdout, stderr, int(process.returncode or 0)


class _LegacyClient:
    def __init__(self, engine: VocaMacEngine) -> None:
        self.engine = engine

    def resolved_binary(self) -> str | None:
        candidate = Path(self.engine.whisperkit_binary).expanduser()
        if candidate.is_file():
            return str(candidate)
        return shutil.which(self.engine.whisperkit_binary)

    def selection(self) -> tuple[str | None, str | None]:
        if self.engine.model:
            return self.engine.model, MODEL_VARIANTS.get(self.engine.model, self.engine.model)
        return _ModelAssets.preference_selection(self.engine.preferences_file)

    def usable_models(self) -> list[Path]:
        selected, variant = self.selection()
        if selected is not None and variant is None:
            return []
        preferred = self.engine.models_dir / variant if variant else None
        if preferred is not None and not _ModelAssets.is_usable(preferred):
            preferred = None
        if self.engine.model:
            return [] if preferred is None else [preferred]
        return _ModelAssets.ranked_models(self.engine.models_dir, preferred)

    def delegate_for(self, model_path: Path) -> whisperkit.WhisperKitEngine:
        engine = self.engine
        if engine._delegate is not None and engine._delegate_path == model_path:
            return engine._delegate
        engine.close()
        engine._delegate = whisperkit.WhisperKitEngine(
            engine.whisperkit_binary,
            model_path,
            tokenizer_path=engine.download_base,
        )
        engine._delegate_path = model_path
        return engine._delegate

    async def health(self) -> base.EngineHealth:
        models = self.usable_models()
        if not models:
            selected, variant = self.selection()
            wanted = variant or selected or "no-model-selected"
            return base.EngineHealth(ready=False, name=f"vocamac:{wanted}")
        delegate = await self.delegate_for(models[0]).health()
        return base.EngineHealth(
            ready=self.engine.app_path.exists() and delegate.ready,
            name=f"vocamac:{models[0].name}",
        )

    async def transcribe(
        self, audio_path: Path, options: base.TranscriptionOptions
    ) -> base.EngineTranscription:
        models = self.usable_models()
        if not models or not (await self.health()).ready:
            raise _ModelAssets.missing_engine(*self.selection())
        last_error: errors.TranscriptionProcessError | None = None
        index = 0
        while index < len(models):
            model_path = models[index]
            index += 1
            try:
                return await self.delegate_for(model_path).transcribe(audio_path, options)
            except errors.TranscriptionProcessError as error:
                last_error = error
        if last_error is not None:
            raise last_error
        raise errors.EngineUnavailableError("No usable VocaMac transcription model is available.")


class VocaMacEngine:
    """Adapter for VocaMac's selected local transcription model.

    VocaMac 0.8.0 and later expose a headless file-transcription command backed
    by the same multi-engine router as the app. Releases through 0.7.2 stay on
    the original WhisperKit-folder adapter. Capability detection inspects the
    executable rather than probing it with flags, because an unknown argument on
    an old build launches the GUI.
    """

    def __init__(
        self,
        whisperkit_binary: str,
        model: str | None = None,
        *,
        app_path: Path | None = None,
        support_dir: Path | None = None,
        preferences_file: Path | None = None,
    ) -> None:
        self.whisperkit_binary = whisperkit_binary
        self.model = model
        self.app_path = app_path or DEFAULT_VOCAMAC_APP
        self.download_base = (support_dir or DEFAULT_SUPPORT_DIR.expanduser()) / "models"
        self.models_dir = self.download_base / "models" / MODEL_REPOSITORY
        self.preferences_file = preferences_file or DEFAULT_PREFERENCES_FILE.expanduser()
        self._delegate: whisperkit.WhisperKitEngine | None = None
        self._delegate_path: Path | None = None
        self._headless_signature: tuple[int, int] | None = None
        self._headless_capable = False

    def is_available(self) -> bool:
        """Cheap synchronous check used while resolving the `auto` engine."""
        headless = _HeadlessClient(self)
        if headless.supported():
            model = headless.model()
            return bool(model is not None and model.downloaded and model.supported)
        legacy = _LegacyClient(self)
        has_binary = legacy.resolved_binary() is not None
        return self.app_path.exists() and has_binary and bool(legacy.usable_models())

    async def health(self) -> base.EngineHealth:
        headless = _HeadlessClient(self)
        if not headless.supported():
            return await _LegacyClient(self).health()
        model = await asyncio.to_thread(headless.model)
        wanted = headless.wanted_model_name() if model is None else model.id
        ready = bool(model is not None and model.downloaded and model.supported)
        return base.EngineHealth(ready=ready, name=f"vocamac:{wanted}")

    async def warmup(self) -> int:
        if _HeadlessClient(self).supported():
            return 0
        models = _LegacyClient(self).usable_models()
        if not models or not (await self.health()).ready:
            return 0
        return await _LegacyClient(self).delegate_for(models[0]).warmup()

    async def transcribe(
        self, audio_path: Path, options: base.TranscriptionOptions
    ) -> base.EngineTranscription:
        if _HeadlessClient(self).supported():
            return await _HeadlessClient(self).transcribe(audio_path, options)
        return await _LegacyClient(self).transcribe(audio_path, options)

    def close(self) -> None:
        delegate = self._delegate
        self._delegate = None
        self._delegate_path = None
        if delegate is not None:
            delegate.close()

    @property
    def model_is_resident(self) -> bool:
        return self._delegate is not None and self._delegate.model_is_resident

    def unload(self) -> None:
        self.close()
