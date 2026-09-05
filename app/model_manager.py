from __future__ import annotations

import asyncio
import hashlib
import json
import re
import shutil
import tarfile
import threading
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from functools import partial
from http import client as http_client
from pathlib import Path, PurePosixPath
from typing import Any, cast
from urllib import error as urllib_error
from urllib import request as urllib_request
from urllib.parse import urlparse

from app.catalog import (
    DEFAULT_CATALOG,
    ENGINE_MOONSHINE,
    ENGINE_SHERPA_ONNX,
    ENGINE_WHISPER_CPP,
    ENGINE_WHISPERKIT,
    RETIRED_CATALOG,
    CatalogModel,
    catalog_by_id,
)

CHUNK_SIZE = 1024 * 1024
MAXIMUM_DOWNLOAD_ERROR_LENGTH = 600
USER_AGENT = "vocagateway-gateway/0.2"
HF_BASE_URL = "https://huggingface.co"
DEFAULT_HF_REVISION = "main"
DOWNLOADING_STATUS = "downloading"
# What an on-disk install looks like, shared by the full scan and the single
# lookup so the two can never disagree about whether a model is installed.
WEIGHT_FILE_SUFFIXES: tuple[str, ...] = (".bin", ".gguf")
WHISPERKIT_CONFIG_FILE = "config.json"
PARTIAL_DIRECTORY_SUFFIX = ".partial"
COMPLETED_STATUS = "completed"
# `?expand=true` is what makes the tree API report each file's SHA-256, but it
# also shrinks the page size from 1000 to 50 and starts sending rel="next".
# Every listing must therefore be paged to completion — a repo like
# argmaxinc/whisperkit-coreml holds ~850 files, so reading only the first page
# would silently download a fraction of a model and call it complete.
_NEXT_LINK = re.compile(r'<([^>]+)>;\s*rel="next"')
_SHA256_PATTERN = re.compile(r"\A[0-9a-f]{64}\Z")

_MAX_NETWORK_ATTEMPTS = 3
_RETRY_BACKOFF_SECONDS = 2.0
MAX_PARALLEL_FILE_DOWNLOADS = 4
_RETRYABLE_NETWORK_ERRORS = (
    TimeoutError,
    ConnectionError,
    urllib_error.URLError,
    http_client.IncompleteRead,
    http_client.HTTPException,
)


def _call_with_retries[ActionResult](
    action: Callable[[], ActionResult],
    *,
    on_retry: Callable[[], None] | None = None,
    attempts: int = _MAX_NETWORK_ATTEMPTS,
    retryable_errors: tuple[type[BaseException], ...] = _RETRYABLE_NETWORK_ERRORS,
    retry_delay: Callable[[BaseException, int], float] | None = None,
) -> ActionResult:
    """Call *action* up to *attempts* times, retrying transport failures.

    A dropped connection or a read timeout can just as easily be a Wi-Fi
    hiccup as a real problem with the source, so a couple of retries with
    backoff are cheap insurance against reporting a corrupted transfer as a
    permanent failure.
    """
    for attempt in range(1, attempts + 1):
        try:
            return action()
        except retryable_errors as error:
            if on_retry is not None:
                on_retry()
            if attempt == attempts:
                raise
            delay = _RETRY_BACKOFF_SECONDS * attempt
            if retry_delay is not None:
                delay = retry_delay(error, attempt)
            time.sleep(delay)
    raise AssertionError("unreachable")  # pragma: no cover


class DownloadCancelled(Exception):
    """A model download was cancelled by the caller."""


class ModelIntegrityError(Exception):
    """A downloaded file did not match its expected SHA-256.

    Raised only after the offending file has been removed, so a failed
    verification can never leave unverified bytes on disk for an engine to
    later load.
    """


def normalize_sha256(digest_text: str) -> str:
    """Validate and canonicalise a user- or catalog-supplied SHA-256."""
    candidate = digest_text.strip().lower()
    if candidate.startswith("sha256:"):
        candidate = candidate[len("sha256:") :]
    if not _SHA256_PATTERN.match(candidate):
        raise ValueError("Expected a 64-character hexadecimal SHA-256 digest.")
    return candidate


class DownloadInProgressError(Exception):
    """A second download was requested while one is active."""


class UnknownModelError(Exception):
    """The requested model is not in the catalog."""


@dataclass(slots=True)
class DownloadState:
    model_id: str
    status: str = DOWNLOADING_STATUS  # downloading | completed | failed | cancelled
    downloaded_bytes: int = 0
    total_bytes: int | None = None
    current_file: str = ""
    error: str | None = None


@dataclass(frozen=True, slots=True)
class InstalledModel:
    id: str
    engine: str
    key: str
    path: Path
    size_bytes: int
    custom: bool = False
    retired: bool = False
    replacement_id: str | None = None


@dataclass(slots=True)
class _DownloadHandle:
    state: DownloadState
    cancel: threading.Event
    task: asyncio.Task[None] | None = field(default=None)


@dataclass(slots=True)
class _DownloadBatch:
    downloads: tuple[Callable[[], Awaitable[None]], ...]
    download_handle: _DownloadHandle
    semaphore: asyncio.Semaphore = field(
        default_factory=lambda: asyncio.Semaphore(MAX_PARALLEL_FILE_DOWNLOADS)
    )

    async def run(self) -> None:
        """Wait for every worker to stop before reporting a folder failure."""
        outcomes = await asyncio.gather(
            *(self._run_one(download) for download in self.downloads),
            return_exceptions=True,
        )
        failure = self._first_failure(outcomes)
        if failure is not None:
            raise failure
        if any(isinstance(outcome, DownloadCancelled) for outcome in outcomes):
            raise DownloadCancelled

    async def _run_one(self, download: Callable[[], Awaitable[None]]) -> None:
        try:
            await self._run_bounded(download)
        except Exception:
            self.download_handle.cancel.set()
            raise

    async def _run_bounded(self, download: Callable[[], Awaitable[None]]) -> None:
        async with self.semaphore:
            if self.download_handle.cancel.is_set():
                raise DownloadCancelled
            await download()

    def _first_failure(self, outcomes: list[None | BaseException]) -> Exception | None:
        return next(
            (
                outcome
                for outcome in outcomes
                if isinstance(outcome, Exception) and not isinstance(outcome, DownloadCancelled)
            ),
            None,
        )


class ModelManager:
    """Downloads, lists, and deletes local speech models."""

    def __init__(
        self,
        models_dir: Path,
        catalog: tuple[CatalogModel, ...] = DEFAULT_CATALOG,
        retired_catalog: tuple[CatalogModel, ...] | None = None,
    ) -> None:
        self.models_dir = models_dir
        self.catalog = catalog
        self.retired_catalog = (
            RETIRED_CATALOG
            if retired_catalog is None and catalog is DEFAULT_CATALOG
            else retired_catalog or ()
        )
        self._by_id = catalog_by_id(catalog)
        self._retired_by_id = catalog_by_id(self.retired_catalog)
        self._downloads: dict[str, _DownloadHandle] = {}

    # ------------------------------------------------------------------ paths

    def model_path(self, model: CatalogModel) -> Path:
        return self.models_dir / model.engine / model.key

    def installed_path(self, model_id: str) -> Path | None:
        """Locate one installed model without sizing every other one.

        `installed()` walks and sums every file under every model directory to
        report sizes. Engine selection calls this on each request path that
        rebuilds an engine and never looks at a size, so it resolves the path
        from the catalog index and checks the same marker `installed()` does.
        """
        path = self._path_for_id(model_id)
        if path is None:
            return None
        model = self.catalog_model(model_id)
        if model is None:
            # A model outside the catalog: a weights file, or a WhisperKit
            # folder, which counts as installed only once its config lands.
            if path.suffix in WEIGHT_FILE_SUFFIXES:
                return path if path.is_file() else None
            return path if (path / WHISPERKIT_CONFIG_FILE).is_file() else None
        marker = path / model.marker_file if model.marker_file else path
        return path if marker.exists() else None

    # --------------------------------------------------------------- listing

    def installed(self) -> list[InstalledModel]:
        installed_models: list[InstalledModel] = []
        catalog_paths: set[Path] = set()
        catalog_models = [(model, False) for model in self.catalog]
        catalog_models.extend((model, True) for model in self.retired_catalog)
        for model, retired in catalog_models:
            path = self.model_path(model)
            marker = path / model.marker_file if model.marker_file else path
            if marker.exists():
                catalog_paths.add(path)
                installed_models.append(
                    InstalledModel(
                        id=model.id,
                        engine=model.engine,
                        key=model.key,
                        path=path,
                        size_bytes=_directory_size(path) if path.is_dir() else path.stat().st_size,
                        retired=retired,
                        replacement_id=model.replacement_id,
                    )
                )
        whisper_cpp_dir = self.models_dir / ENGINE_WHISPER_CPP
        if whisper_cpp_dir.is_dir():
            for model_file in sorted(whisper_cpp_dir.iterdir()):
                if not model_file.is_file() or model_file.suffix not in WEIGHT_FILE_SUFFIXES:
                    continue
                if model_file in catalog_paths:
                    continue
                catalog_model = self._by_key(model_file.name, ENGINE_WHISPER_CPP)
                model_id = catalog_model.id if catalog_model else f"custom:{model_file.name}"
                installed_models.append(
                    InstalledModel(
                        id=model_id,
                        engine=ENGINE_WHISPER_CPP,
                        key=model_file.name,
                        path=model_file,
                        size_bytes=model_file.stat().st_size,
                        custom=catalog_model is None,
                    )
                )
        whisperkit_dir = self.models_dir / ENGINE_WHISPERKIT
        if whisperkit_dir.is_dir():
            for folder in sorted(whisperkit_dir.iterdir()):
                if not folder.is_dir() or not (folder / WHISPERKIT_CONFIG_FILE).is_file():
                    continue
                if folder in catalog_paths:
                    continue
                catalog_model = self._by_key(folder.name, ENGINE_WHISPERKIT)
                model_id = catalog_model.id if catalog_model else f"custom:{folder.name}"
                installed_models.append(
                    InstalledModel(
                        id=model_id,
                        engine=ENGINE_WHISPERKIT,
                        key=folder.name,
                        path=folder,
                        size_bytes=_directory_size(folder),
                        custom=catalog_model is None,
                    )
                )
        return installed_models

    def downloads(self) -> list[DownloadState]:
        return [download_handle.state for download_handle in self._downloads.values()]

    def download_state(self, model_id: str) -> DownloadState | None:
        download_handle = self._downloads.get(model_id)
        return download_handle.state if download_handle else None

    def catalog_model(self, model_id: str) -> CatalogModel | None:
        """Return catalog metadata without exposing the manager's mutable index."""
        return self._by_id.get(model_id) or self._retired_by_id.get(model_id)

    # ------------------------------------------------------------- downloads

    def start_download(self, model_id: str) -> DownloadState:
        model = self._by_id.get(model_id)
        if model is None:
            if model_id in self._retired_by_id:
                raise UnknownModelError(f"Model {model_id} is retired and cannot be downloaded.")
            raise UnknownModelError(model_id)
        return self._start(
            model_id,
            lambda download_handle: self._run_catalog_download(model, download_handle),
        )

    def start_custom_download(self, url: str, sha256: str | None = None) -> DownloadState:
        """Download a user-supplied model URL, optionally pinned to a digest.

        This is the one download path whose source the catalog does not vouch
        for, so the digest is offered rather than required: a user pasting a
        URL from a model card can paste that card's SHA-256 alongside it and
        get the same guarantee the catalog entries have.
        """
        filename = _validate_custom_url(url)
        expected = normalize_sha256(sha256) if sha256 and sha256.strip() else None
        model_id = f"custom:{filename}"
        if self.installed_path(model_id) is not None:
            raise DownloadInProgressError(f"{filename} is already installed.")
        destination = self.models_dir / ENGINE_WHISPER_CPP / filename
        return self._start(
            model_id,
            lambda download_handle: self._run_single_file(
                url, destination, download_handle, expected
            ),
        )

    def cancel_download(self, model_id: str) -> bool:
        download_handle = self._downloads.get(model_id)
        if download_handle is None or download_handle.state.status != DOWNLOADING_STATUS:
            return False
        download_handle.cancel.set()
        return True

    def delete(self, model_id: str) -> bool:
        download_handle = self._downloads.get(model_id)
        if download_handle is not None and download_handle.state.status == DOWNLOADING_STATUS:
            raise DownloadInProgressError(model_id)
        path = self._path_for_id(model_id)
        if path is None or not path.exists():
            return False
        self._downloads.pop(model_id, None)
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        return True

    # ----------------------------------------------------------------- inner

    def _start(
        self,
        model_id: str,
        runner: Any,
    ) -> DownloadState:
        existing = self._downloads.get(model_id)
        if existing is not None and existing.state.status == DOWNLOADING_STATUS:
            raise DownloadInProgressError(model_id)
        download_handle = _DownloadHandle(
            state=DownloadState(model_id=model_id), cancel=threading.Event()
        )
        self._downloads[model_id] = download_handle
        download_handle.task = asyncio.create_task(self._guarded(model_id, runner, download_handle))
        return download_handle.state

    async def _guarded(self, model_id: str, runner: Any, download_handle: _DownloadHandle) -> None:
        try:
            await runner(download_handle)
        except DownloadCancelled:
            download_handle.state.status = "cancelled"
        except Exception as error:  # noqa: BLE001 - surfaced through the API
            download_handle.state.status = "failed"
            # Long enough that a SHA-256 mismatch message (which embeds two
            # 64-character hex digests plus a nested WhisperKit-style path)
            # isn't clipped before the reader can see what to compare.
            download_handle.state.error = str(error)[:MAXIMUM_DOWNLOAD_ERROR_LENGTH]

    async def _run_catalog_download(
        self, model: CatalogModel, download_handle: _DownloadHandle
    ) -> None:
        if model.engine == ENGINE_MOONSHINE:
            await self._run_moonshine_download(model, download_handle)
            return
        if model.archive_url is not None:
            await self._run_archive_download(model, download_handle)
            return
        if model.engine == ENGINE_SHERPA_ONNX and model.huggingface_repo is not None:
            await self._run_sherpa_huggingface_download(model, download_handle)
            return
        if model.huggingface_repo is not None:
            await self._run_huggingface_download(model, download_handle)
            return
        if model.download_url is None:
            raise UnknownModelError(model.id)
        await self._run_single_file(
            model.download_url, self.model_path(model), download_handle, model.sha256
        )

    async def _run_archive_download(
        self, model: CatalogModel, download_handle: _DownloadHandle
    ) -> None:
        if not model.archive_url or not model.archive_root or not model.required_files:
            raise UnknownModelError(f"Archive metadata is incomplete for {model.id}.")
        final_dir = self.model_path(model)
        partial_dir = final_dir.with_name(f"{final_dir.name}{PARTIAL_DIRECTORY_SUFFIX}")
        extraction_dir = final_dir.with_name(f"{final_dir.name}.extracting")
        archive_path = final_dir.with_name(f"{final_dir.name}.download")
        shutil.rmtree(partial_dir, ignore_errors=True)
        _remove_tree(extraction_dir)
        archive_path.unlink(missing_ok=True)
        try:
            # Verified before extraction, not after: `_safe_extract_archive`
            # is the code that parses attacker-influenced bytes, so refusing a
            # bad archive up front keeps it away from the tar parser entirely.
            await asyncio.to_thread(
                _download_file,
                model.archive_url,
                archive_path,
                download_handle.state,
                download_handle.cancel,
                Path(model.archive_url).name,
                model.sha256,
            )
            if download_handle.cancel.is_set():
                raise DownloadCancelled
            await asyncio.to_thread(_safe_extract_archive, archive_path, extraction_dir)
            extracted = extraction_dir / model.archive_root
            if not extracted.is_dir():
                raise RuntimeError(
                    f"Downloaded archive does not contain the expected {model.archive_root} folder."
                )
            missing = [name for name in model.required_files if not (extracted / name).is_file()]
            if missing:
                raise RuntimeError(_missing_model_files_message(missing))
            # Redundant when `model.sha256` pinned the archive, since that one
            # check already covers everything inside it. It matters for an
            # archive with no pinned digest but pinned member digests.
            await asyncio.to_thread(_verify_extracted_files, model, extracted)
            shutil.move(str(extracted), partial_dir)
            metadata = {
                "model_id": model.id,
                "model_type": model.model_type,
                "language_codes": list(model.language_codes),
                "required_files": list(model.required_files),
            }
            (partial_dir / ".vocagateway-model.json").write_text(
                f"{json.dumps(metadata, indent=2)}\n", encoding="utf-8"
            )
            if download_handle.cancel.is_set():
                raise DownloadCancelled
            final_dir.parent.mkdir(parents=True, exist_ok=True)
            _remove_tree(final_dir)
            partial_dir.replace(final_dir)
            download_handle.state.status = COMPLETED_STATUS
            archive_path.unlink(missing_ok=True)
        except BaseException:
            archive_path.unlink(missing_ok=True)
            _remove_tree(extraction_dir)
            _remove_tree(partial_dir)
            raise

    async def _run_sherpa_huggingface_download(
        self, model: CatalogModel, download_handle: _DownloadHandle
    ) -> None:
        """Download exactly `required_files` from a plain Hugging Face model repo.

        Unlike `_run_huggingface_download` (which mirrors an entire folder for
        engines like MLX Audio and writes no marker), this fetches only the
        named files a sherpa-onnx model actually needs and writes the same
        `.vocagateway-model.json` marker `_run_archive_download` does, since
        some model families (GigaAM, Canary) ship as bare Hugging Face repos
        with no pre-packaged `.tar.bz2` release archive.
        """
        if not model.huggingface_repo or not model.required_files:
            raise UnknownModelError(f"Hugging Face metadata is incomplete for {model.id}.")
        final_dir = self.model_path(model)
        partial_dir = final_dir.with_name(final_dir.name + PARTIAL_DIRECTORY_SUFFIX)
        _remove_tree(partial_dir)
        partial_dir.mkdir(parents=True, exist_ok=True)
        try:
            listing = await asyncio.to_thread(
                _list_repo_folder, model.huggingface_repo, "", _revision(model)
            )
            available = {entry.relative_path: entry for entry in listing}
            download_handle.state.total_bytes = sum(
                entry.size_bytes
                for name in model.required_files
                if (entry := available.get(name)) is not None
            )
            downloads = tuple(
                partial(
                    self._download_required_file,
                    model,
                    partial_dir,
                    download_handle,
                    name,
                    available,
                )
                for name in model.required_files
            )
            await _DownloadBatch(downloads, download_handle).run()
            missing = [name for name in model.required_files if not (partial_dir / name).is_file()]
            if missing:
                raise RuntimeError(_missing_model_files_message(missing))
            metadata = {
                "model_id": model.id,
                "model_type": model.model_type,
                "language_codes": list(model.language_codes),
                "required_files": list(model.required_files),
            }
            (partial_dir / ".vocagateway-model.json").write_text(
                f"{json.dumps(metadata, indent=2)}\n", encoding="utf-8"
            )
            if download_handle.cancel.is_set():
                raise DownloadCancelled
            final_dir.parent.mkdir(parents=True, exist_ok=True)
            _remove_tree(final_dir)
            partial_dir.replace(final_dir)
            download_handle.state.status = COMPLETED_STATUS
        except BaseException:
            _remove_tree(partial_dir)
            raise

    async def _run_moonshine_download(
        self, model: CatalogModel, download_handle: _DownloadHandle
    ) -> None:
        final_dir = self.model_path(model)
        partial_dir = final_dir.with_name(f"{final_dir.name}{PARTIAL_DIRECTORY_SUFFIX}")
        _remove_tree(partial_dir)
        partial_dir.mkdir(parents=True, exist_ok=True)
        download_handle.state.total_bytes = model.size_bytes
        download_handle.state.current_file = "Moonshine model assets"
        try:
            model_path, model_arch = await asyncio.to_thread(
                _download_moonshine_model,
                model.language_code or model.key,
                model.model_arch,
                partial_dir,
            )
            if download_handle.cancel.is_set():
                raise DownloadCancelled
            resolved_model_path = Path(model_path).resolve()
            staging_root = partial_dir.resolve()
            if not resolved_model_path.is_relative_to(staging_root):
                raise RuntimeError(
                    "Moonshine downloader returned a path outside its staging directory."
                )
            relative_path = resolved_model_path.relative_to(staging_root)
            if model.required_files:
                missing = [
                    name
                    for name in model.required_files
                    if not (resolved_model_path / name).is_file()
                ]
                if missing:
                    raise RuntimeError(_missing_model_files_message(missing))
            await asyncio.to_thread(_verify_extracted_files, model, resolved_model_path)
            metadata = {
                "model_id": model.id,
                "language": model.language_code or model.key,
                "model_path": str(relative_path),
                "model_arch": int(model_arch),
            }
            (partial_dir / ".vocagateway-model.json").write_text(
                f"{json.dumps(metadata, indent=2)}\n", encoding="utf-8"
            )
            download_handle.state.downloaded_bytes = _directory_size(partial_dir)
            final_dir.parent.mkdir(parents=True, exist_ok=True)
            _remove_tree(final_dir)
            partial_dir.replace(final_dir)
            download_handle.state.status = COMPLETED_STATUS
        except BaseException:
            _remove_tree(partial_dir)
            raise

    async def _run_single_file(
        self,
        url: str,
        destination: Path,
        download_handle: _DownloadHandle,
        expected_sha256: str | None = None,
    ) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_name(f"{destination.name}{PARTIAL_DIRECTORY_SUFFIX}")
        try:
            await asyncio.to_thread(
                _download_file,
                url,
                partial,
                download_handle.state,
                download_handle.cancel,
                "",
                expected_sha256,
            )
            if download_handle.cancel.is_set():
                raise DownloadCancelled
            partial.replace(destination)
            download_handle.state.status = COMPLETED_STATUS
        except BaseException:
            partial.unlink(missing_ok=True)
            raise

    async def _run_huggingface_download(
        self, model: CatalogModel, download_handle: _DownloadHandle
    ) -> None:
        if not model.huggingface_repo or model.huggingface_folder is None:
            raise UnknownModelError(model.id)
        files = await asyncio.to_thread(
            _list_repo_folder, model.huggingface_repo, model.huggingface_folder, _revision(model)
        )
        if model.required_files:
            available = {entry.relative_path for entry in files}
            missing = set(model.required_files) - available
            if missing:
                raise UnknownModelError(f"Required model files are missing: {sorted(missing)}.")
            files = [entry for entry in files if entry.relative_path in model.required_files]
        if not files:
            raise UnknownModelError(f"No model files found in {model.huggingface_repo}.")
        download_handle.state.total_bytes = sum(entry.size_bytes for entry in files)
        final_dir = self.model_path(model)
        partial_dir = final_dir.with_name(f"{final_dir.name}{PARTIAL_DIRECTORY_SUFFIX}")
        _remove_tree(partial_dir)
        partial_dir.mkdir(parents=True, exist_ok=True)
        try:
            downloads = tuple(
                partial(self._download_repo_file, model, partial_dir, download_handle, entry)
                for entry in files
            )
            await _DownloadBatch(downloads, download_handle).run()
            if download_handle.cancel.is_set():
                raise DownloadCancelled
            final_dir.parent.mkdir(parents=True, exist_ok=True)
            _remove_tree(final_dir)
            partial_dir.replace(final_dir)
            download_handle.state.status = COMPLETED_STATUS
        except BaseException:
            _remove_tree(partial_dir)
            raise

    async def _download_required_file(
        self,
        model: CatalogModel,
        partial_dir: Path,
        download_handle: _DownloadHandle,
        name: str,
        available: dict[str, RepoFile],
    ) -> None:
        if download_handle.cancel.is_set():
            raise DownloadCancelled
        listed = available.get(name)
        await asyncio.to_thread(
            _download_file,
            _resolve_url(model, name),
            partial_dir / name,
            download_handle.state,
            download_handle.cancel,
            name,
            _expected_digest(model, name, listed.sha256 if listed else None),
        )

    async def _download_repo_file(
        self,
        model: CatalogModel,
        partial_dir: Path,
        download_handle: _DownloadHandle,
        entry: RepoFile,
    ) -> None:
        relative = entry.relative_path
        if download_handle.cancel.is_set():
            raise DownloadCancelled
        if not is_safe_relative_path(relative):
            raise RuntimeError(f"Repository listing contains an unsafe path: {relative}")
        await asyncio.to_thread(
            _download_file,
            _resolve_url(model, _repo_path(model.huggingface_folder or "", relative)),
            partial_dir / relative,
            download_handle.state,
            download_handle.cancel,
            relative,
            _expected_digest(model, relative, entry.sha256),
        )

    def _by_key(self, key: str, engine: str) -> CatalogModel | None:
        for model in (*self.catalog, *self.retired_catalog):
            if model.key == key and model.engine == engine:
                return model
        return None

    def _path_for_id(self, model_id: str) -> Path | None:
        model = self.catalog_model(model_id)
        if model is not None:
            return self.model_path(model)
        if model_id.startswith("custom:"):
            name = model_id[len("custom:") :]
            if Path(name).name != name or not name:
                return None
            if name.endswith(WEIGHT_FILE_SUFFIXES):
                return self.models_dir / ENGINE_WHISPER_CPP / name
            return self.models_dir / ENGINE_WHISPERKIT / name
        return None


def _validate_custom_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise ValueError("Only HTTPS URLs are supported.")
    filename = Path(parsed.path).name
    if not filename.endswith(WEIGHT_FILE_SUFFIXES):
        raise ValueError("Custom models must be .bin or .gguf files.")
    return filename


@dataclass(frozen=True, slots=True)
class RepoFile:
    """One file in a Hugging Face repo listing."""

    relative_path: str
    size_bytes: int
    # SHA-256 as reported by the tree API. Present for LFS-backed files, which
    # is every file large enough to be a model weight; absent for small plain
    # git blobs, which the API identifies only by git's SHA-1 object id.
    sha256: str | None = None


def is_safe_relative_path(relative: str) -> bool:
    """True when *relative* stays inside the directory it is joined onto.

    Repo listings name the files written to disk, so an entry like
    ``/etc/authorized_keys`` or ``../../x`` would place a download outside the
    model directory — `Path("/a") / "/etc/x"` is `/etc/x`, not `/a/etc/x`.
    Archives are already screened this way in `_safe_extract_archive`; this is
    the same rule for the file-listing paths.
    """
    if not relative or relative.startswith(("/", "\\")):
        return False
    pure = PurePosixPath(relative)
    # An empty `parts` means the entry named no file at all ("" or "."), which
    # would resolve to the model directory itself rather than a file in it.
    if pure.is_absolute() or not pure.parts:
        return False
    return ".." not in pure.parts


def _same_origin(candidate: str, origin: str) -> bool:
    left, right = urlparse(candidate), urlparse(origin)
    return bool(left.scheme) and (left.scheme, left.netloc) == (right.scheme, right.netloc)


def _fetch_repo_page(page_url: str) -> tuple[list[dict[str, Any]], str]:
    request = urllib_request.Request(page_url, headers={"User-Agent": USER_AGENT})
    with urllib_request.urlopen(request, timeout=60) as response:
        return json.load(response), response.headers.get("Link") or ""


def _list_repo_folder(
    repo: str, folder: str, revision: str = DEFAULT_HF_REVISION
) -> list[RepoFile]:
    tree_path = f"/{folder}" if folder else ""
    url = f"{HF_BASE_URL}/api/models/{repo}/tree/{revision}{tree_path}?recursive=true&expand=true"
    origin = url
    files: list[RepoFile] = []
    prefix = f"{folder}/" if folder else ""
    seen: set[str] = set()
    while url:
        payload, link = _call_with_retries(partial(_fetch_repo_page, url))
        for entry in payload:
            if entry.get("type") != "file":
                continue
            path = str(entry.get("path", ""))
            if not path.startswith(prefix):
                continue
            relative = path[len(prefix) :]
            if relative in seen:
                continue
            # Loud, not skipped: real Hugging Face repos never contain a path
            # like this, so one appearing means something is wrong with the
            # listing. Dropping it quietly would install a partial model and
            # report success.
            if not is_safe_relative_path(relative):
                raise RuntimeError(f"Repository listing contains an unsafe path: {relative}")
            seen.add(relative)
            lfs = entry.get("lfs") or {}
            oid = lfs.get("oid") if isinstance(lfs, dict) else None
            size = lfs.get("size") if isinstance(lfs, dict) else None
            files.append(
                RepoFile(
                    relative_path=relative,
                    size_bytes=int(size or entry.get("size") or 0),
                    sha256=str(oid)
                    if isinstance(oid, str) and _SHA256_PATTERN.match(oid)
                    else None,
                )
            )
        match = _NEXT_LINK.search(link)
        # The next page is named by the server being paged. Following it
        # anywhere it points would let a compromised host walk this client onto
        # another origin or scheme — the same host compromise the digests in
        # this module exist to catch, so it cannot be trusted here either.
        following = match.group(1) if match else ""
        url = following if following and _same_origin(following, origin) else ""
    return files


def _repo_path(folder: str, relative: str) -> str:
    return f"{folder}/{relative}" if folder else relative


def _revision(model: CatalogModel) -> str:
    return model.revision or DEFAULT_HF_REVISION


def _resolve_url(model: CatalogModel, path: str) -> str:
    return f"{HF_BASE_URL}/{model.huggingface_repo}/resolve/{_revision(model)}/{path}"


def _expected_digest(model: CatalogModel, relative: str, listed: str | None) -> str | None:
    """Digest to enforce for one file, preferring the catalog over the API.

    A digest pinned in the catalog is reviewed and lives in git, so it still
    holds if the upstream repo is compromised. A digest read from the tree API
    at download time only proves the bytes survived the transfer intact — an
    attacker who can rewrite the file can rewrite its metadata too. Both are
    worth enforcing, but they are not the same guarantee, so the pinned value
    always wins where one exists.
    """
    for name, digest in model.file_digests:
        if name == relative:
            return digest
    return listed


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_stream:
        chunk = file_stream.read(CHUNK_SIZE)
        while chunk:
            digest.update(chunk)
            chunk = file_stream.read(CHUNK_SIZE)
    return digest.hexdigest()


def _verify_extracted_files(model: CatalogModel, root: Path) -> None:
    """Enforce `file_digests` against files that came out of an archive.

    Files inside an archive are never streamed individually, so they cannot be
    hashed during download the way `_download_file` does; they are read back
    once here instead.
    """
    for relative, expected in model.file_digests:
        target = root / relative
        if not target.is_file():
            continue
        actual = _sha256_path(target)
        if actual != expected:
            raise ModelIntegrityError(
                f"{relative} failed SHA-256 verification: expected {expected}, "
                f"got {actual}. The extracted model was discarded."
            )


def _safe_extract_archive(archive_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with tarfile.open(archive_path, mode="r:*") as archive:
        for member in archive.getmembers():
            target = (destination / member.name).resolve()
            if not target.is_relative_to(root):
                raise RuntimeError("Downloaded archive contains an unsafe path.")
            if member.issym() or member.islnk():
                raise RuntimeError("Downloaded archive contains an unsupported link.")
        archive.extractall(destination, filter="data")


def _download_file(
    url: str,
    destination: Path,
    state: DownloadState,
    cancel: threading.Event,
    display_name: str,
    expected_sha256: str | None = None,
) -> str:
    """Stream *url* to *destination* and return the SHA-256 of what arrived.

    The digest is computed from the same chunks that are written, so a
    multi-gigabyte model is never read back a second time to verify it. When
    *expected_sha256* is supplied and does not match, the file is deleted
    before raising, so a rejected download cannot be left behind for an engine
    to load later.

    A dropped connection, short response, or checksum mismatch is retried a
    few times against the exact same expected digest. A persistent mismatch is
    still rejected; retrying can only accept bytes that satisfy the pin.
    """
    download = _DownloadAttempt(url, destination, state, cancel, display_name, expected_sha256)
    return _call_with_retries(
        download.run,
        on_retry=download.rollback,
        retryable_errors=(*_RETRYABLE_NETWORK_ERRORS, ModelIntegrityError),
        retry_delay=download.retry_delay,
    )


@dataclass
class _DownloadAttempt:
    url: str
    destination: Path
    state: DownloadState
    cancel: threading.Event
    display_name: str
    expected_sha256: str | None
    bytes_this_attempt: int = 0
    response_size: int | None = None

    def run(self) -> str:
        self.bytes_this_attempt = 0
        self.response_size = None
        request = urllib_request.Request(self.url, headers={"User-Agent": USER_AGENT})
        digest = hashlib.sha256()
        with urllib_request.urlopen(request, timeout=60) as response:
            self._set_total_bytes(response)
            self.destination.parent.mkdir(parents=True, exist_ok=True)
            self.state.current_file = self.display_name or self.destination.name
            with self.destination.open("wb") as output:
                self._write_response(response, output, digest)
        self._verify_response_size()
        actual = digest.hexdigest()
        self._verify_digest(actual)
        return actual

    def rollback(self) -> None:
        self.destination.unlink(missing_ok=True)
        self.state.downloaded_bytes = max(0, self.state.downloaded_bytes - self.bytes_this_attempt)

    def retry_delay(self, error: BaseException, attempt: int) -> float:
        """Retry a checksum mismatch immediately; back off for network failures."""
        if isinstance(error, ModelIntegrityError):
            return 0
        return _RETRY_BACKOFF_SECONDS * attempt

    def _set_total_bytes(self, response: Any) -> None:
        length = response.headers.get("Content-Length")
        if length and length.isdigit():
            self.response_size = int(length)
            if self.state.total_bytes is None:
                self.state.total_bytes = self.response_size

    def _write_response(self, response: Any, output: Any, digest: Any) -> None:
        chunk = response.read(CHUNK_SIZE)
        while chunk:
            if self.cancel.is_set():
                raise DownloadCancelled
            output.write(chunk)
            digest.update(chunk)
            self.bytes_this_attempt += len(chunk)
            self.state.downloaded_bytes += len(chunk)
            chunk = response.read(CHUNK_SIZE)

    def _verify_response_size(self) -> None:
        if self.response_size is None or self.bytes_this_attempt == self.response_size:
            return
        missing = max(0, self.response_size - self.bytes_this_attempt)
        raise http_client.IncompleteRead(b"", missing)

    def _verify_digest(self, actual: str) -> None:
        if self.expected_sha256 is None or actual == self.expected_sha256:
            return
        self.destination.unlink(missing_ok=True)
        filename = self.display_name or self.destination.name
        raise ModelIntegrityError(_digest_mismatch_message(filename, self.expected_sha256, actual))


def _directory_size(path: Path) -> int:
    return sum(
        nested_path.stat().st_size for nested_path in path.rglob("*") if nested_path.is_file()
    )


def _remove_tree(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)


def _missing_model_files_message(missing: list[str]) -> str:
    names = ", ".join(missing)
    return f"Downloaded model is missing: {names}."


def _digest_mismatch_message(filename: str, expected: str, actual: str) -> str:
    prefix = f"{filename} failed SHA-256 verification"
    return f"{prefix}: expected {expected}, got {actual}. The file was discarded."


def _download_moonshine_model(
    language: str,
    model_arch: int | None,
    cache_root: Path,
) -> tuple[str, Any]:
    try:
        from moonshine_voice import (
            ModelArch,
            get_model_for_language,
        )
    except ImportError as error:
        raise RuntimeError(
            "Moonshine support is not installed. Install vocagateway[engines]."
        ) from error
    architecture = None
    if model_arch is not None:
        architecture = ModelArch(model_arch)
    return cast(
        tuple[str, Any],
        get_model_for_language(language, architecture, cache_root=cache_root),
    )
