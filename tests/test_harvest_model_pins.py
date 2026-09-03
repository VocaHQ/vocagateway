from __future__ import annotations

import json
from importlib import util as importlib_util
from pathlib import Path
from typing import Any

SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "harvest-model-pins.py"
SHA256_DIGEST_LENGTH = 64
WRONG_XET_DIGEST = "2" * SHA256_DIGEST_LENGTH


def _load_module():
    spec = importlib_util.spec_from_file_location("harvest_model_pins", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib_util.module_from_spec(spec)
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


class _SingleFileRequester:
    def __init__(self, hf_base_url: str, real_sha256: str) -> None:
        self._hf_base_url = hf_base_url
        self._real_sha256 = real_sha256

    def __call__(self, url: str, method: str = "GET") -> _FakeResponse:
        if method == "HEAD":
            raise AssertionError(
                "harvest() must not issue a HEAD request for a Hugging Face "
                "resolve URL -- that is the bug this test guards against"
            )
        if url == f"{self._hf_base_url}/api/models/octocat/demo":
            return _FakeResponse({"sha": "deadbeef" * 8})
        if "/tree/" in url:
            return _FakeResponse(
                [{"type": "file", "path": "model.bin", "lfs": {"oid": self._real_sha256}}]
            )
        raise AssertionError(f"unexpected URL in test: {url}")


def test_single_file_digest_comes_from_the_aa(monkeypatch) -> None:
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

    real_sha256 = "1" * SHA256_DIGEST_LENGTH

    fake_request = _SingleFileRequester(module.HF_BASE_URL, real_sha256)
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
    assert record["sha256"] != WRONG_XET_DIGEST


def _single_file_model(module, model_id: str):
    filename = f"{model_id}.bin"
    return module.CatalogModel(
        id=f"whisper.cpp:{filename}",
        engine="whisper.cpp",
        key=filename,
        label=model_id,
        size_bytes=1,
        languages="Multilingual",
        quality="Balanced",
        minimum_ram_gb=1,
        download_url=f"https://huggingface.co/octocat/demo/resolve/main/{filename}",
    )


def _write_pins(path: Path, models: dict[str, Any]) -> None:
    path.write_text(json.dumps({"models": models}), encoding="utf-8")


def test_default_harvest_only_adds_missing_models(tmp_path: Path, monkeypatch) -> None:
    """Adding one model must not silently refresh every existing pin."""
    module = _load_module()
    existing_model = _single_file_model(module, "existing")
    new_model = _single_file_model(module, "new")
    old_record = {"revision": "old-revision", "sha256": "a" * SHA256_DIGEST_LENGTH}
    new_record = {"revision": "new-revision", "sha256": "b" * SHA256_DIGEST_LENGTH}
    output = tmp_path / "pins.json"
    _write_pins(output, {existing_model.id: old_record})
    harvested: list[str] = []

    def fake_harvest(model, *, download_unpinnable: bool):
        harvested.append(model.id)
        return new_record

    monkeypatch.setattr(module, "_BASE_CATALOG", (existing_model, new_model))
    monkeypatch.setattr(module, "harvest", fake_harvest)

    assert module.main(["--output", str(output)]) == 0

    records = json.loads(output.read_text(encoding="utf-8"))["models"]
    assert harvested == [new_model.id]
    assert records[existing_model.id] == old_record
    assert records[new_model.id] == new_record


def test_targeted_refresh_replaces_pin_as_one_snapshot(tmp_path: Path, monkeypatch) -> None:
    """A new revision must never retain stale fields from the prior record."""
    module = _load_module()
    model = _single_file_model(module, "existing")
    output = tmp_path / "pins.json"
    old_record = {
        "revision": "old-revision",
        "sha256": "a" * SHA256_DIGEST_LENGTH,
        "obsolete": "must disappear",
    }
    new_record = {"revision": "new-revision", "sha256": "b" * SHA256_DIGEST_LENGTH}
    _write_pins(output, {model.id: old_record})
    monkeypatch.setattr(module, "_BASE_CATALOG", (model,))
    monkeypatch.setattr(
        module,
        "harvest",
        lambda selected, *, download_unpinnable: new_record,
    )

    assert module.main(["--only", model.id, "--output", str(output)]) == 0

    records = json.loads(output.read_text(encoding="utf-8"))["models"]
    assert records[model.id] == new_record


def test_incomplete_refresh_preserves_previous_pin_and_fails(tmp_path: Path, monkeypatch) -> None:
    """Never create a revision/digest pair assembled from different snapshots."""
    module = _load_module()
    model = _single_file_model(module, "existing")
    output = tmp_path / "pins.json"
    old_record = {"revision": "old-revision", "sha256": "a" * SHA256_DIGEST_LENGTH}
    _write_pins(output, {model.id: old_record})
    monkeypatch.setattr(module, "_BASE_CATALOG", (model,))
    monkeypatch.setattr(
        module,
        "harvest",
        lambda selected, *, download_unpinnable: {"revision": "new-revision"},
    )

    assert module.main(["--only", model.id, "--output", str(output)]) == 1

    records = json.loads(output.read_text(encoding="utf-8"))["models"]
    assert records[model.id] == old_record
