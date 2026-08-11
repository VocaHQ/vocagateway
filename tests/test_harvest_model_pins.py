from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "harvest-model-pins.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("harvest_model_pins", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeResponse:
    def __init__(self, payload: Any, headers: dict[str, str] | None = None) -> None:
        self._payload = payload
        self.headers = headers or {}

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def read(self) -> bytes:
        return json.dumps(self._payload).encode()


def test_single_file_digest_comes_from_the_tree_api_not_a_head_request(monkeypatch) -> None:
    """Reproduces the exact bug this script shipped with: a repo migrated to
    Hugging Face's Xet storage backend makes `HEAD /{repo}/resolve/{rev}/{file}`
    302 to a Xet CDN host, and `urlopen` follows that redirect by default. The
    CDN's own `etag` header reports a Xet content hash, not the file's SHA-256
    -- so a real download hashes to a completely different value and every
    download of that model fails verification forever (this is exactly what
    corrupted `whisper.cpp:ggml-small.bin`'s pin). The fix means harvesting a
    single-file digest must never issue a HEAD request at all; it must come
    from the tree API's `lfs.oid`, which the redirect can't touch."""
    module = _load_module()

    real_sha256 = "1" * 64
    xet_hash_a_head_request_would_wrongly_read = "2" * 64

    def fake_request(url: str, method: str = "GET") -> _FakeResponse:
        if method == "HEAD":
            raise AssertionError(
                "harvest() must not issue a HEAD request for a Hugging Face "
                "resolve URL -- that is the bug this test guards against"
            )
        if url == f"{module.HF_BASE_URL}/api/models/octocat/demo":
            return _FakeResponse({"sha": "deadbeef" * 8})
        if "/tree/" in url:
            return _FakeResponse(
                [{"type": "file", "path": "model.bin", "lfs": {"oid": real_sha256}}]
            )
        raise AssertionError(f"unexpected URL in test: {url}")

    monkeypatch.setattr(module, "_request", fake_request)

    model = module.CatalogModel(
        id="whisper.cpp:model.bin",
        engine="whisper.cpp",
        key="model.bin",
        label="Test Model",
        size_bytes=1,
        languages="Multilingual",
        quality="Balanced",
        minimum_ram_gb=1,
        download_url="https://huggingface.co/octocat/demo/resolve/main/model.bin",
    )

    record = module.harvest(model, download_unpinnable=False)

    assert record is not None
    assert record["sha256"] == real_sha256
    assert record["sha256"] != xet_hash_a_head_request_would_wrongly_read
