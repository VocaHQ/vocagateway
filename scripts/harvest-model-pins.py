#!/usr/bin/env python3
"""Regenerate app/model_pins.json from the upstream model hosts.

The pin file records, per catalog model, the Hugging Face commit to download
from and the SHA-256 of the files that commit contains. Pinning both is what
lets a download be rejected when upstream content changes: TLS already proves
you reached the real host, but not that the host is still serving the bytes
this catalog entry was reviewed against.

Most digests are free to collect because Hugging Face publishes them:

  * `GET /api/models/{repo}/tree/{rev}?expand=true` reports `lfs.oid`, the
    SHA-256 of every LFS-backed file (which is every file large enough to be
    a model weight).
  * `HEAD /{repo}/resolve/{rev}/{file}` returns `x-linked-etag`, the same
    digest, plus `x-repo-commit` for the current revision.

Two sources publish no usable digest at all -- the sherpa-onnx GitHub release
tarballs and the blob.handy.computer mirror, whose ETag is a multipart S3 hash
rather than a digest of the content. Those can only be pinned by downloading
and hashing the file, which is roughly 3.5 GB in total, so they are skipped
unless --download-unpinnable is passed.

Usage:
    uv run scripts/harvest-model-pins.py                     # free sources only
    uv run scripts/harvest-model-pins.py --download-unpinnable
    uv run scripts/harvest-model-pins.py --only sherpa-onnx:  # id prefix filter
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.catalog import _BASE_CATALOG, PINS_PATH, CatalogModel  # noqa: E402
from app.model_manager import HF_BASE_URL, USER_AGENT  # noqa: E402

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
    try:
        with _request(f"{HF_BASE_URL}/api/models/{repo}") as response:
            payload = json.load(response)
    except (urllib.error.URLError, json.JSONDecodeError, TimeoutError) as error:
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
        with _request(url) as response:
            payload = json.load(response)
            link = response.headers.get("Link") or ""
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


def linked_etag(url: str) -> str | None:
    """SHA-256 a Hugging Face resolve URL advertises without transferring it."""
    try:
        with _request(url, method="HEAD") as response:
            etag = response.headers.get("x-linked-etag") or response.headers.get("etag") or ""
    except (urllib.error.URLError, TimeoutError) as error:
        print(f"    ! HEAD failed: {error}", file=sys.stderr)
        return None
    candidate = etag.strip().strip('"').lower()
    return candidate if SHA256.match(candidate) else None


def download_digest(url: str) -> str | None:
    """Last resort: stream the whole file and hash it."""
    digest = hashlib.sha256()
    seen = 0
    try:
        with _request(url) as response:
            while chunk := response.read(CHUNK):
                digest.update(chunk)
                seen += len(chunk)
                print(f"\r      {seen / 1e6:8.1f} MB", end="", file=sys.stderr)
    except (urllib.error.URLError, TimeoutError) as error:
        print(f"\n    ! download failed: {error}", file=sys.stderr)
        return None
    print("\r" + " " * 24 + "\r", end="", file=sys.stderr)
    return digest.hexdigest()


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
            match = re.search(r"huggingface\.co/([^/]+/[^/]+)/resolve/", model.download_url)
            if match and "revision" not in record:
                revision = repo_revision(match.group(1))
                if revision:
                    record["revision"] = revision
            url = model.download_url
            if record.get("revision"):
                url = url.replace("/resolve/main/", f"/resolve/{record['revision']}/", 1)
            digest = linked_etag(url)
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--download-unpinnable",
        action="store_true",
        help="Also hash sources that publish no digest (~3.5 GB of transfer).",
    )
    parser.add_argument("--only", default="", help="Only harvest model ids with this prefix.")
    parser.add_argument("--output", type=Path, default=PINS_PATH)
    args = parser.parse_args()

    existing: dict[str, Any] = {}
    if args.output.is_file():
        try:
            existing = json.loads(args.output.read_text(encoding="utf-8")).get("models", {})
        except (OSError, json.JSONDecodeError):
            existing = {}

    models = [m for m in _BASE_CATALOG if m.id.startswith(args.only)]
    print(f"Harvesting pins for {len(models)} model(s)\n")
    records: dict[str, Any] = dict(existing)
    pinned = skipped = 0
    for model in models:
        print(f"  {model.id}")
        record = harvest(model, download_unpinnable=args.download_unpinnable)
        if record is None:
            # Keep any previously harvested record rather than dropping it.
            if model.id not in records:
                skipped += 1
                print("    - nothing pinnable without --download-unpinnable")
            continue
        merged = {**records.get(model.id, {}), **record}
        records[model.id] = merged
        pinned += 1
        summary = []
        if merged.get("revision"):
            summary.append(f"revision {merged['revision'][:12]}")
        if merged.get("sha256"):
            summary.append(f"sha256 {merged['sha256'][:12]}")
        if merged.get("file_digests"):
            summary.append(f"{len(merged['file_digests'])} file digests")
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
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    print(f"\nWrote {args.output} — {pinned} pinned, {skipped} unpinnable this run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
