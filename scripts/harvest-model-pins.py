#!/usr/bin/env python3
"""Regenerate app/model_pins.json from the upstream model hosts.

The pin file records, per catalog model, the Hugging Face commit to download
from and the SHA-256 of the files that commit contains. Pinning both is what
lets a download be rejected when upstream content changes: TLS already proves
you reached the real host, but not that the host is still serving the bytes
this catalog entry was reviewed against.

Most digests are free to collect because Hugging Face publishes them via
`GET /api/models/{repo}/tree/{rev}?expand=true`, which reports `lfs.oid`, the
SHA-256 of every LFS-backed file (which is every file large enough to be a
model weight). This is also why single-file `download_url` entries are
resolved against the same tree endpoint rather than HEAD-ing the file's
`resolve/` URL: for a repo migrated to Hugging Face's Xet storage backend,
that URL 302s to a Xet CDN host, and the CDN's own `etag` header reports a
Xet content hash, not the SHA-256 -- so a HEAD request that follows the
redirect (as `urlopen` does by default) silently pins the wrong digest.

Two sources publish no usable digest at all -- the sherpa-onnx GitHub release
tarballs and the blob.handy.computer mirror, whose ETag is a multipart S3 hash
rather than a digest of the content. Those can only be pinned by downloading
and hashing the file, which is roughly 3.5 GB in total, so they are skipped
unless --download-unpinnable is passed.

Usage:
    uv run scripts/harvest-model-pins.py                     # pin new models only
    uv run scripts/harvest-model-pins.py --refresh           # refresh every pin
    uv run scripts/harvest-model-pins.py --download-unpinnable
    uv run scripts/harvest-model-pins.py --only sherpa-onnx:  # refresh one family
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import urllib.request
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.catalog import _BASE_CATALOG, PINS_PATH, CatalogModel  # noqa: E402
from app.model_manager import (  # noqa: E402
    _RETRYABLE_NETWORK_ERRORS,
    HF_BASE_URL,
    USER_AGENT,
    _call_with_retries,
)

CHUNK = 1024 * 1024
NEXT_LINK = re.compile(r'<([^>]+)>;\s*rel="next"')
SHA256 = re.compile(r"\A[0-9a-f]{64}\Z")


def _request(url: str, method: str = "GET") -> Any:
    return urllib.request.urlopen(
        urllib.request.Request(url, headers={"User-Agent": USER_AGENT}, method=method),
        timeout=120,
    )


_REVISION_CACHE: dict[str, str | None] = {}
_TREE_CACHE: dict[tuple[str, str], dict[str, str]] = {}


def repo_revision(repo: str) -> str | None:
    """Current commit of a repo's default branch."""
    if repo in _REVISION_CACHE:
        return _REVISION_CACHE[repo]
    _REVISION_CACHE[repo] = _repo_revision_uncached(repo)
    return _REVISION_CACHE[repo]


def _repo_revision_uncached(repo: str) -> str | None:
    def attempt() -> Any:
        with _request(f"{HF_BASE_URL}/api/models/{repo}") as response:
            return json.load(response)

    try:
        payload = _call_with_retries(attempt)
    except (*_RETRYABLE_NETWORK_ERRORS, json.JSONDecodeError) as error:
        print(f"    ! revision lookup failed for {repo}: {error}", file=sys.stderr)
        return None
    sha = payload.get("sha")
    return str(sha) if isinstance(sha, str) else None


def repo_digests(repo: str, folder: str, revision: str) -> dict[str, str]:
    """SHA-256 per file under *folder*, paging the tree API to completion.

    Cached per (repo, revision): the WhisperKit catalog entries are eight
    folders inside one ~850-file repo, which is 18 pages of tree API each time.
    """
    key = (repo, revision)
    if key not in _TREE_CACHE:
        _TREE_CACHE[key] = _fetch_repo_tree(repo, revision)
    full = _TREE_CACHE[key]
    if not folder:
        return full
    prefix = f"{folder}/"
    return {k[len(prefix) :]: v for k, v in full.items() if k.startswith(prefix)}


def _fetch_repo_tree(repo: str, revision: str) -> dict[str, str]:
    """Whole-repo digest map; callers slice the folder they need out of it."""
    url = f"{HF_BASE_URL}/api/models/{repo}/tree/{revision}?recursive=true&expand=true"
    digests: dict[str, str] = {}
    while url:

        def fetch_page(page_url: str = url) -> tuple[list[dict[str, Any]], str]:
            with _request(page_url) as response:
                return json.load(response), response.headers.get("Link") or ""

        payload, link = _call_with_retries(fetch_page)
        for entry in payload:
            if entry.get("type") != "file":
                continue
            path = str(entry.get("path", ""))
            oid = (entry.get("lfs") or {}).get("oid")
            if path and isinstance(oid, str) and SHA256.match(oid):
                digests[path] = oid
        match = NEXT_LINK.search(link)
        url = match.group(1) if match else ""
    return digests


def download_digest(url: str) -> str | None:
    """Last resort: stream the whole file and hash it."""

    def attempt() -> str:
        digest = hashlib.sha256()
        seen = 0
        with _request(url) as response:
            while chunk := response.read(CHUNK):
                digest.update(chunk)
                seen += len(chunk)
                print(f"\r      {seen / 1e6:8.1f} MB", end="", file=sys.stderr)
        return digest.hexdigest()

    try:
        result = _call_with_retries(attempt)
    except _RETRYABLE_NETWORK_ERRORS as error:
        print(f"\n    ! download failed: {error}", file=sys.stderr)
        return None
    print("\r" + " " * 24 + "\r", end="", file=sys.stderr)
    return result


def harvest(model: CatalogModel, *, download_unpinnable: bool) -> dict[str, Any] | None:
    record: dict[str, Any] = {}

    if model.huggingface_repo:
        revision = repo_revision(model.huggingface_repo)
        if revision:
            record["revision"] = revision
            folder = model.huggingface_folder or ""
            digests = repo_digests(model.huggingface_repo, folder, revision)
            if model.required_files:
                digests = {k: v for k, v in digests.items() if k in model.required_files}
            if digests:
                record["file_digests"] = dict(sorted(digests.items()))

    if model.download_url:
        if "huggingface.co/" in model.download_url:
            # Pin the repo commit too, so the URL stops tracking `main`.
            match = re.search(
                r"huggingface\.co/([^/]+/[^/]+)/resolve/[^/]+/(.+)$", model.download_url
            )
            if match:
                repo, path = match.group(1), match.group(2)
                if "revision" not in record:
                    revision = repo_revision(repo)
                    if revision:
                        record["revision"] = revision
                revision = record.get("revision")
                if revision:
                    digest = repo_digests(repo, "", revision).get(path)
                    if digest:
                        record["sha256"] = digest
        elif download_unpinnable:
            digest = download_digest(model.download_url)
            if digest:
                record["sha256"] = digest

    if model.archive_url and download_unpinnable:
        digest = download_digest(model.archive_url)
        if digest:
            record["sha256"] = digest

    return record or None


def _select_models(
    models: list[CatalogModel],
    existing: dict[str, Any],
    *,
    only: str,
    refresh: bool,
    download_unpinnable: bool,
) -> list[CatalogModel]:
    selected = [model for model in models if model.id.startswith(only) and not model.retired]
    if only or refresh:
        return selected
    return [
        model
        for model in selected
        if model.id not in existing
        and (
            download_unpinnable
            or model.huggingface_repo
            or (model.download_url and "huggingface.co/" in model.download_url)
        )
    ]


def _expects_complete_pin(model: CatalogModel, download_unpinnable: bool) -> bool:
    is_huggingface_file = bool(model.download_url and "huggingface.co/" in model.download_url)
    has_external_download = bool(model.download_url or model.archive_url)
    return bool(
        model.huggingface_repo
        or is_huggingface_file
        or (download_unpinnable and has_external_download)
    )


def _pin_is_complete(model: CatalogModel, record: dict[str, Any]) -> bool:
    if model.huggingface_repo and (not record.get("revision") or not record.get("file_digests")):
        return False
    if (
        model.download_url
        and "huggingface.co/" in model.download_url
        and (not record.get("revision") or not record.get("sha256"))
    ):
        return False
    if (model.download_url or model.archive_url) and not (
        model.huggingface_repo or (model.download_url and "huggingface.co/" in model.download_url)
    ):
        return bool(record.get("sha256"))
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--download-unpinnable",
        action="store_true",
        help="Also hash sources that publish no digest (~3.5 GB of transfer).",
    )
    parser.add_argument("--only", default="", help="Only harvest model ids with this prefix.")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Refresh existing pins too. By default only missing models are harvested.",
    )
    parser.add_argument("--output", type=Path, default=PINS_PATH)
    args = parser.parse_args(argv)

    existing: dict[str, Any] = {}
    if args.output.is_file():
        try:
            existing = json.loads(args.output.read_text(encoding="utf-8")).get("models", {})
        except (OSError, json.JSONDecodeError):
            existing = {}

    # Retired entries are refused by `start_download`, so a pin for one can
    # never be checked against a fresh download and only rots in the file.
    models = _select_models(
        list(_BASE_CATALOG),
        existing,
        only=args.only,
        refresh=args.refresh,
        download_unpinnable=args.download_unpinnable,
    )
    print(f"Harvesting pins for {len(models)} model(s)\n")
    live_ids = {model.id for model in _BASE_CATALOG if not model.retired}
    records: dict[str, Any] = {
        model_id: record for model_id, record in existing.items() if model_id in live_ids
    }
    pinned = skipped = failed = 0
    for model in models:
        print(f"  {model.id}")
        try:
            record = harvest(model, download_unpinnable=args.download_unpinnable)
        except _RETRYABLE_NETWORK_ERRORS as error:
            # One repo's exhausted retries shouldn't discard however many
            # models were already harvested this run.
            print(f"    ! harvest failed: {error}", file=sys.stderr)
            failed += 1
            continue
        if record is None:
            # Keep any previously harvested record rather than dropping it.
            if _expects_complete_pin(model, args.download_unpinnable):
                failed += 1
                print("    ! could not produce a complete pin", file=sys.stderr)
            elif model.id not in records:
                skipped += 1
                print("    - nothing pinnable without --download-unpinnable")
            continue
        if not _pin_is_complete(model, record):
            failed += 1
            print("    ! pin is incomplete; previous record preserved", file=sys.stderr)
            continue
        # A revision and its digests are one snapshot. Replacing the whole
        # record prevents a new revision from being paired with a stale digest
        # left over from an earlier model source.
        records[model.id] = record
        pinned += 1
        summary = []
        if record.get("revision"):
            summary.append(f"revision {record['revision'][:12]}")
        if record.get("sha256"):
            summary.append(f"sha256 {record['sha256'][:12]}")
        if record.get("file_digests"):
            summary.append(f"{len(record['file_digests'])} file digests")
        print(f"    + {', '.join(summary) or 'no data'}")

    payload = {
        "_comment": (
            "Generated by scripts/harvest-model-pins.py. Each entry pins the "
            "Hugging Face commit a model is downloaded from and the SHA-256 of "
            "its files. Review digest changes as carefully as code: a changed "
            "digest means the upstream bytes changed."
        ),
        "models": dict(sorted(records.items())),
    }
    rendered = json.dumps(payload, indent=2, sort_keys=False) + "\n"
    current = args.output.read_text(encoding="utf-8") if args.output.is_file() else None
    if rendered != current:
        temporary = args.output.with_name(f".{args.output.name}.tmp")
        try:
            temporary.write_text(rendered, encoding="utf-8")
            temporary.replace(args.output)
        finally:
            temporary.unlink(missing_ok=True)
    print(
        f"\nWrote {args.output} — {pinned} pinned, {skipped} unpinnable, {failed} failed this run"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
