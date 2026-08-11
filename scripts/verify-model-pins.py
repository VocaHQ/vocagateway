#!/usr/bin/env python3
"""Check that app/model_pins.json still matches what Hugging Face serves today.

This is the guard for the bug fixed in harvest-model-pins.py: a wrong digest
in the pin file fails every download of that model with a SHA-256 mismatch,
and nothing short of re-fetching the upstream tree API would have caught it
before a user hit it. Run this after regenerating pins, and periodically
thereafter -- upstream repos can also legitimately re-upload a file under the
same revision, which this catches too.

Digests are re-derived independently from `harvest-model-pins.py`'s own
logic (a fresh fetch of `lfs.oid` off the tree API), not by re-running that
script, so a bug reintroduced there would not also blind this check.

Usage:
    uv run scripts/verify-model-pins.py
    uv run scripts/verify-model-pins.py --only whisper.cpp:
"""

from __future__ import annotations

import argparse
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

_NEXT_LINK = re.compile(r'<([^>]+)>;\s*rel="next"')
_RESOLVE_URL = re.compile(r"huggingface\.co/([^/]+/[^/]+)/resolve/[^/]+/(.+)$")

_TREE_CACHE: dict[tuple[str, str], dict[str, str]] = {}


def _fetch_repo_tree(repo: str, revision: str) -> dict[str, str]:
    """SHA-256 per file at *revision*, straight from the tree API's `lfs.oid`."""
    key = (repo, revision)
    if key in _TREE_CACHE:
        return _TREE_CACHE[key]
    url = f"{HF_BASE_URL}/api/models/{repo}/tree/{revision}?recursive=true&expand=true"
    digests: dict[str, str] = {}
    while url:

        def fetch_page(page_url: str = url) -> tuple[list[dict[str, Any]], str]:
            request = urllib.request.Request(page_url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.load(response), response.headers.get("Link") or ""

        payload, link = _call_with_retries(fetch_page)
        for entry in payload:
            if entry.get("type") != "file":
                continue
            path = str(entry.get("path", ""))
            oid = (entry.get("lfs") or {}).get("oid")
            if path and isinstance(oid, str):
                digests[path] = oid
        match = _NEXT_LINK.search(link)
        url = match.group(1) if match else ""
    _TREE_CACHE[key] = digests
    return digests


def check(model: CatalogModel, record: dict[str, Any]) -> list[str]:
    """Mismatch descriptions for one pinned model; an empty list means clean."""
    problems: list[str] = []
    revision = record.get("revision")
    if not revision:
        return problems

    if record.get("sha256") and model.download_url:
        match = _RESOLVE_URL.search(model.download_url)
        if match:
            repo, path = match.group(1), match.group(2)
            live = _fetch_repo_tree(repo, revision).get(path)
            if live is None:
                problems.append(f"{path}: not listed at revision {revision[:12]}")
            elif live != record["sha256"]:
                problems.append(
                    f"{path}: pinned {record['sha256'][:12]} but upstream reports {live[:12]}"
                )

    file_digests = record.get("file_digests")
    if isinstance(file_digests, dict) and model.huggingface_repo:
        folder = model.huggingface_folder or ""
        prefix = f"{folder}/" if folder else ""
        tree = _fetch_repo_tree(model.huggingface_repo, revision)
        for relative, pinned in file_digests.items():
            live = tree.get(f"{prefix}{relative}")
            if live is None:
                problems.append(f"{relative}: not listed at revision {revision[:12]}")
            elif live != pinned:
                problems.append(
                    f"{relative}: pinned {pinned[:12]} but upstream reports {live[:12]}"
                )

    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", default="", help="Only check model ids with this prefix.")
    parser.add_argument("--pins", type=Path, default=PINS_PATH)
    args = parser.parse_args()

    try:
        pins = json.loads(args.pins.read_text(encoding="utf-8")).get("models", {})
    except (OSError, json.JSONDecodeError) as error:
        print(f"Could not read {args.pins}: {error}", file=sys.stderr)
        return 1

    models = {m.id: m for m in _BASE_CATALOG}
    checked = mismatched = skipped = 0
    for model_id, record in sorted(pins.items()):
        if not model_id.startswith(args.only):
            continue
        model = models.get(model_id)
        if model is None:
            continue
        print(f"  {model_id}")
        try:
            problems = check(model, record)
        except _RETRYABLE_NETWORK_ERRORS as error:
            print(f"    ! check failed: {error}", file=sys.stderr)
            skipped += 1
            continue
        checked += 1
        if problems:
            mismatched += 1
            for problem in problems:
                print(f"    ! {problem}")
        else:
            print("    ok")

    print(f"\n{checked} pin(s) checked, {mismatched} mismatched, {skipped} skipped")
    return 1 if mismatched else 0


if __name__ == "__main__":
    raise SystemExit(main())
