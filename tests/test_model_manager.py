from __future__ import annotations

import asyncio
import dataclasses
import io
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

import pytest

from app import model_manager
from app.catalog import DEFAULT_CATALOG, CatalogModel
from app.model_manager import (
    DownloadInProgressError,
    ModelIntegrityError,
    ModelManager,
    RepoFile,
    UnknownModelError,
    normalize_sha256,
)

TINY_FILE = CatalogModel(
    id="whisper.cpp:ggml-tiny.bin",
    engine="whisper.cpp",
    key="ggml-tiny.bin",
    label="Test Tiny",
    size_bytes=11,
    languages="Multilingual",
    quality="Fastest",
    minimum_ram_gb=4,
    download_url=None,  # filled by fixture
)

TINY_FOLDER = CatalogModel(
    id="whisperkit:openai_whisper-tiny",
    engine="whisperkit",
    key="openai_whisper-tiny",
    label="Test WhisperKit Tiny",
    size_bytes=30,
    languages="Multilingual",
    quality="Fastest",
    minimum_ram_gb=4,
    huggingface_repo="example/repo",
    huggingface_folder="openai_whisper-tiny",
)

TINY_CTRANSLATE = CatalogModel(
    id="faster-whisper:tiny.en",
    engine="faster-whisper",
    key="tiny.en",
    label="Test faster-whisper Tiny EN",
    size_bytes=30,
    languages="English only",
    quality="Fastest",
    minimum_ram_gb=2,
    huggingface_repo="example/faster-repo",
    huggingface_folder="",
    marker_file="model.bin",
)

MOONSHINE_SPANISH = CatalogModel(
    id="moonshine:es",
    engine="moonshine",
    key="es",
    label="Moonshine Spanish",
    size_bytes=65_000_000,
    languages="Spanish only",
    quality="Fast",
    minimum_ram_gb=2,
    marker_file=".vocagateway-model.json",
    language_code="es",
    model_arch=1,
)

SHERPA_TEST = CatalogModel(
    id="sherpa-onnx:test-int8",
    engine="sherpa-onnx",
    key="test-int8",
    label="Test sherpa model",
    size_bytes=30,
    languages="English only",
    quality="Fast",
    minimum_ram_gb=2,
    marker_file=".vocagateway-model.json",
    archive_root="published-model",
    required_files=("model.int8.onnx", "tokens.txt"),
    model_type="sense_voice",
    language_codes=("en",),
)


def test_catalog_includes_standalone_handy_a94aa() -> None:
    entries = {model.key: model for model in DEFAULT_CATALOG}

    assert entries["whisper-medium-q4_1.bin"].engine == "whisper.cpp"
    assert entries["ggml-large-v3-q5_0.bin"].source == "Handy-compatible"
    assert entries["breeze-asr-q5_k.bin"].family == "Breeze ASR"


def test_distilled_faster_whisper_uses_publ_aa() -> None:
    entries = {model.id: model for model in DEFAULT_CATALOG}

    assert (
        entries["faster-whisper:distil-small.en"].huggingface_repo
        == "Systran/faster-distil-whisper-small.en"
    )


def test_catalog_includes_all_moonshine_lan_aaa() -> None:
    entries = {model.id: model for model in DEFAULT_CATALOG}

    assert {entry.language_code for entry in entries.values() if entry.engine == "moonshine"} == {
        "ar",
        "en",
        "es",
        "ja",
        "ko",
        "uk",
        "vi",
        "zh",
    }
    assert entries["moonshine:en"].model_arch == 5
    assert entries["moonshine:en-tiny-streaming"].supports_streaming is True
    assert entries["moonshine:es"].supports_streaming is False
    assert entries["moonshine:es"].commercial_use is False
    assert (
        entries["faster-whisper:distil-medium.en"].huggingface_repo
        == "Systran/faster-distil-whisper-medium.en"
    )


def test_catalog_includes_portable_and_appl_aaaa() -> None:
    entries = {model.id: model for model in DEFAULT_CATALOG}

    sensevoice = entries["sherpa-onnx:sensevoice-small-int8"]
    assert sensevoice.required_files == ("model.int8.onnx", "tokens.txt")
    assert sensevoice.language_codes == ("zh", "yue", "en", "ja", "ko")
    assert sensevoice.apple_silicon_only is False

    parakeet = entries["sherpa-onnx:parakeet-tdt-0.6b-v3-int8"]
    assert parakeet.model_type == "nemo_transducer"
    assert parakeet.license_name == "CC BY 4.0"

    mlx_turbo = entries["mlx-audio:whisper-large-v3-turbo-4bit"]
    assert mlx_turbo.apple_silicon_only is True
    assert mlx_turbo.marker_file == "model.safetensors"


def test_catalog_includes_gigaam_and_canary_aaaaa() -> None:
    entries = {model.id: model for model in DEFAULT_CATALOG}

    gigaam_ctc = entries["sherpa-onnx:gigaam-v3-ctc-russian-int8"]
    assert gigaam_ctc.archive_url is None
    assert (
        gigaam_ctc.huggingface_repo
        == "csukuangfj/sherpa-onnx-nemo-ctc-giga-am-v3-russian-2025-12-16"
    )
    assert gigaam_ctc.required_files == ("model.int8.onnx", "tokens.txt")
    assert gigaam_ctc.model_type == "nemo_ctc"
    assert gigaam_ctc.language_codes == ("ru",)
    assert gigaam_ctc.license_name == "MIT"

    gigaam_rnnt = entries["sherpa-onnx:gigaam-v3-rnnt-russian-int8"]
    assert gigaam_rnnt.huggingface_repo == (
        "csukuangfj/sherpa-onnx-nemo-transducer-giga-am-v3-russian-2025-12-16"
    )
    assert gigaam_rnnt.required_files == (
        "encoder.int8.onnx",
        "decoder.onnx",
        "joiner.onnx",
        "tokens.txt",
    )
    assert gigaam_rnnt.model_type == "nemo_transducer"
    assert gigaam_rnnt.license_name == "MIT"

    canary = entries["sherpa-onnx:canary-180m-flash-en-int8"]
    assert (
        canary.huggingface_repo == "csukuangfj/sherpa-onnx-nemo-canary-180m-flash-en-es-de-fr-int8"
    )
    assert canary.required_files == ("encoder.int8.onnx", "decoder.int8.onnx", "tokens.txt")
    assert canary.model_type == "nemo_canary"
    assert canary.language_codes == ("en",)
    assert canary.license_name == "CC BY 4.0"

    streaming = entries["sherpa-onnx:streaming-zipformer-en-20m-int8"]
    assert (
        streaming.huggingface_repo == "csukuangfj/sherpa-onnx-streaming-zipformer-en-20M-2023-02-17"
    )
    assert streaming.required_files == (
        "encoder-epoch-99-avg-1.int8.onnx",
        "decoder-epoch-99-avg-1.int8.onnx",
        "joiner-epoch-99-avg-1.int8.onnx",
        "tokens.txt",
    )
    assert streaming.model_type == "streaming_zipformer"
    assert streaming.supports_streaming is True
    assert streaming.license_name == "Apache 2.0"


def test_catalog_includes_the_newer_sherpa_cff8b() -> None:
    entries = {model.id: model for model in DEFAULT_CATALOG}

    dolphin = entries["sherpa-onnx:dolphin-small-ctc-int8"]
    assert dolphin.model_type == "dolphin_ctc"
    assert dolphin.required_files == ("model.int8.onnx", "tokens.txt")
    # The only South Asian coverage in the catalog.
    assert {"hi", "bn", "ta", "ur"} <= set(dolphin.language_codes)
    assert entries["sherpa-onnx:dolphin-base-ctc-int8"].language_codes == dolphin.language_codes

    qwen3 = entries["sherpa-onnx:qwen3-asr-0.6b-int8"]
    assert qwen3.model_type == "qwen3_asr"
    assert "tokenizer/vocab.json" in qwen3.required_files
    assert "tokens.txt" not in qwen3.required_files

    parakeet_v2 = entries["sherpa-onnx:parakeet-tdt-0.6b-v2-int8"]
    assert parakeet_v2.model_type == "nemo_transducer"
    assert parakeet_v2.language_codes == ("en",)


def test_catalog_includes_the_newer_apple_s_ed423() -> None:
    entries = {model.id: model for model in DEFAULT_CATALOG}

    for model_id, repository in (
        ("mlx-audio:parakeet-tdt-0.6b-v2", "mlx-community/parakeet-tdt-0.6b-v2"),
        ("mlx-audio:qwen3-asr-0.6b-4bit", "mlx-community/Qwen3-ASR-0.6B-4bit"),
        ("mlx-audio:qwen3-asr-1.7b-4bit", "mlx-community/Qwen3-ASR-1.7B-4bit"),
        ("mlx-audio:granite-speech-4.1-2b-nar", "mlx-community/granite-speech-4.1-2b-nar-mlx-5bit"),
    ):
        model = entries[model_id]
        assert model.huggingface_repo == repository
        assert model.apple_silicon_only is True
        assert model.marker_file == "model.safetensors"
        assert model.huggingface_folder == ""


def test_every_catalog_model_has_a_download_f979c() -> None:
    for model in DEFAULT_CATALOG:
        if model.engine == "moonshine":
            assert model.language_code, f"{model.id} needs a language for the Moonshine downloader"
            continue
        mechanism = (
            model.archive_url is not None
            or model.huggingface_repo is not None
            or model.download_url is not None
        )
        assert mechanism, f"{model.id} has no way to be downloaded"
        if model.archive_url is not None:
            assert model.archive_root, f"{model.id} has an archive_url but no archive_root"
        if model.engine == "sherpa-onnx":
            assert model.required_files, f"{model.id} must name the files it needs"
            assert model.model_type, f"{model.id} must declare a model_type"


def test_sherpa_onnx_helper_requires_a_down_a() -> None:
    from app.catalog import _sherpa_onnx

    with pytest.raises(ValueError, match="archive_url/archive_root or huggingface_repo"):
        _sherpa_onnx(
            "broken",
            "Broken",
            1,
            "English",
            "Fast",
            1,
            required_files=("model.onnx",),
            model_type="nemo_ctc",
            language_codes=("en",),
            family="Broken",
            description="",
            license_name="MIT",
        )


@pytest.fixture
def tiny_file_model(tmp_path: Path) -> CatalogModel:
    source = tmp_path / "source.bin"
    source.write_bytes(b"hello model")
    return dataclasses.replace(TINY_FILE, download_url=source.as_uri())


@pytest.fixture
def manager(tmp_path: Path, tiny_file_model: CatalogModel) -> ModelManager:
    return ModelManager(
        tmp_path / "models", catalog=(tiny_file_model, TINY_FOLDER, TINY_CTRANSLATE)
    )


def test_installed_scans_both_engines(manager: ModelManager) -> None:
    whisper_dir = manager.models_dir / "whisper.cpp"
    whisper_dir.mkdir(parents=True)
    (whisper_dir / "ggml-tiny.bin").write_bytes(b"abc")
    (whisper_dir / "strange.gguf").write_bytes(b"custom-bytes")
    kit_dir = manager.models_dir / "whisperkit" / "openai_whisper-tiny"
    (kit_dir / "AudioEncoder.mlmodelc" / "weights").mkdir(parents=True)
    (kit_dir / "config.json").write_text("{}")
    (kit_dir / "AudioEncoder.mlmodelc" / "weights" / "weight.bin").write_bytes(b"xx")
    (manager.models_dir / "whisperkit" / "incomplete").mkdir()

    installed = {model.id: model for model in manager.installed()}

    assert set(installed) == {
        "whisper.cpp:ggml-tiny.bin",
        "custom:strange.gguf",
        "whisperkit:openai_whisper-tiny",
    }
    assert installed["custom:strange.gguf"].custom is True
    assert installed["whisperkit:openai_whisper-tiny"].size_bytes == 4


async def test_download_installs_single_file(manager: ModelManager) -> None:
    state = manager.start_download("whisper.cpp:ggml-tiny.bin")
    assert state.status == "downloading"
    await asyncio.wait_for(_wait_finished(manager, "whisper.cpp:ggml-tiny.bin"), timeout=5)

    assert state.status == "completed"
    assert state.downloaded_bytes == 11
    installed = manager.installed_path("whisper.cpp:ggml-tiny.bin")
    assert installed is not None
    assert installed.read_bytes() == b"hello model"


async def test_download_unknown_model_raises(manager: ModelManager) -> None:
    with pytest.raises(UnknownModelError):
        manager.start_download("whisper.cpp:nope.bin")


async def test_double_download_rejected(manager: ModelManager) -> None:
    manager.start_download("whisper.cpp:ggml-tiny.bin")
    with pytest.raises(DownloadInProgressError):
        manager.start_download("whisper.cpp:ggml-tiny.bin")
    await asyncio.wait_for(_wait_finished(manager, "whisper.cpp:ggml-tiny.bin"), timeout=5)


async def test_whisperkit_folder_download(
    manager: ModelManager, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mirror = tmp_path / "mirror"
    folder = mirror / "example/repo/resolve/main/openai_whisper-tiny"
    (folder / "AudioEncoder.mlmodelc").mkdir(parents=True)
    (folder / "config.json").write_text("{}")
    (folder / "AudioEncoder.mlmodelc" / "model.mil").write_text("mil")
    monkeypatch.setattr(model_manager, "HF_BASE_URL", mirror.as_uri())
    monkeypatch.setattr(
        model_manager,
        "_list_repo_folder",
        lambda repo, name, revision="main": [
            RepoFile("config.json", 2),
            RepoFile("AudioEncoder.mlmodelc/model.mil", 3),
        ],
    )

    state = manager.start_download("whisperkit:openai_whisper-tiny")
    await asyncio.wait_for(_wait_finished(manager, "whisperkit:openai_whisper-tiny"), timeout=5)

    assert state.status == "completed"
    assert state.total_bytes == 5
    installed = manager.installed_path("whisperkit:openai_whisper-tiny")
    assert installed is not None
    assert (installed / "config.json").is_file()
    assert (installed / "AudioEncoder.mlmodelc" / "model.mil").is_file()
    assert not (manager.models_dir / "whisperkit" / "openai_whisper-tiny.partial").exists()


async def test_root_huggingface_folder_download(
    manager: ModelManager, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mirror = tmp_path / "mirror"
    folder = mirror / "example/faster-repo/resolve/main"
    folder.mkdir(parents=True)
    (folder / "config.json").write_text("{}")
    (folder / "model.bin").write_bytes(b"model")
    monkeypatch.setattr(model_manager, "HF_BASE_URL", mirror.as_uri())
    monkeypatch.setattr(
        model_manager,
        "_list_repo_folder",
        lambda repo, name, revision="main": [
            RepoFile("config.json", 2),
            RepoFile("model.bin", 5),
        ],
    )

    state = manager.start_download("faster-whisper:tiny.en")
    await asyncio.wait_for(_wait_finished(manager, "faster-whisper:tiny.en"), timeout=5)

    assert state.status == "completed"
    installed = manager.installed_path("faster-whisper:tiny.en")
    assert installed is not None
    assert (installed / "model.bin").read_bytes() == b"model"


async def test_moonshine_download_uses_catalog_la_aa(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = ModelManager(tmp_path / "models", catalog=(MOONSHINE_SPANISH,))
    requested: dict[str, object] = {}

    def fake_download(language: str, model_arch: int, cache_root: Path) -> tuple[str, int]:
        requested.update(language=language, model_arch=model_arch)
        model_path = cache_root / "downloaded-model"
        model_path.mkdir()
        (model_path / "weights.bin").write_bytes(b"model")
        return str(model_path), model_arch

    monkeypatch.setattr(model_manager, "_download_moonshine_model", fake_download)

    state = manager.start_download("moonshine:es")
    await asyncio.wait_for(_wait_finished(manager, "moonshine:es"), timeout=5)

    assert state.status == "completed"
    assert requested == {"language": "es", "model_arch": 1}
    installed = manager.installed_path("moonshine:es")
    assert installed is not None
    metadata = (installed / ".vocagateway-model.json").read_text(encoding="utf-8")
    assert '"model_id": "moonshine:es"' in metadata
    assert '"language": "es"' in metadata


async def test_archive_download_extracts_validate_aaa(tmp_path: Path) -> None:
    source = tmp_path / "model.tar.bz2"
    content = tmp_path / "published-model"
    content.mkdir()
    (content / "model.int8.onnx").write_bytes(b"onnx")
    (content / "tokens.txt").write_text("token")
    with tarfile.open(source, "w:bz2") as archive:
        archive.add(content, arcname="published-model")
    catalog_model = dataclasses.replace(SHERPA_TEST, archive_url=source.as_uri())
    manager = ModelManager(tmp_path / "models", catalog=(catalog_model,))

    state = manager.start_download(catalog_model.id)
    await asyncio.wait_for(_wait_finished(manager, catalog_model.id), timeout=5)

    assert state.status == "completed"
    installed = manager.installed_path(catalog_model.id)
    assert installed is not None
    assert (installed / "model.int8.onnx").read_bytes() == b"onnx"
    metadata = (installed / ".vocagateway-model.json").read_text(encoding="utf-8")
    assert '"model_type": "sense_voice"' in metadata
    assert '"language_codes": [' in metadata
    assert not (manager.models_dir / "sherpa-onnx" / "test-int8.download").exists()


async def test_sherpa_huggingface_download_fetche_e896d(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    catalog_model = dataclasses.replace(
        SHERPA_TEST,
        id="sherpa-onnx:gigaam-test",
        key="gigaam-test",
        archive_url=None,
        archive_root=None,
        huggingface_repo="example/gigaam-repo",
        required_files=("model.int8.onnx", "tokens.txt"),
        model_type="nemo_ctc",
    )
    manager = ModelManager(tmp_path / "models", catalog=(catalog_model,))

    mirror = tmp_path / "mirror"
    folder = mirror / "example/gigaam-repo/resolve/main"
    folder.mkdir(parents=True)
    (folder / "model.int8.onnx").write_bytes(b"onnx-bytes")
    (folder / "tokens.txt").write_text("token")
    (folder / "README.md").write_text("not needed")
    monkeypatch.setattr(model_manager, "HF_BASE_URL", mirror.as_uri())
    monkeypatch.setattr(
        model_manager,
        "_list_repo_folder",
        lambda repo, name, revision="main": [
            RepoFile("model.int8.onnx", 10),
            RepoFile("tokens.txt", 5),
            RepoFile("README.md", 999),
        ],
    )

    state = manager.start_download(catalog_model.id)
    await asyncio.wait_for(_wait_finished(manager, catalog_model.id), timeout=5)

    assert state.status == "completed"
    assert state.total_bytes == 15  # only required_files counted, not README.md
    installed = manager.installed_path(catalog_model.id)
    assert installed is not None
    assert (installed / "model.int8.onnx").read_bytes() == b"onnx-bytes"
    assert not (installed / "README.md").exists()
    metadata = (installed / ".vocagateway-model.json").read_text(encoding="utf-8")
    assert '"model_type": "nemo_ctc"' in metadata
    assert not (manager.models_dir / "sherpa-onnx" / "gigaam-test.partial").exists()


def test_archive_extractor_rejects_parent_paths(tmp_path: Path) -> None:
    archive_path = tmp_path / "unsafe.tar.bz2"
    with tarfile.open(archive_path, "w:bz2") as archive:
        member = tarfile.TarInfo("../escaped.txt")
        payload = b"unsafe"
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))

    with pytest.raises(RuntimeError, match="unsafe path"):
        model_manager._safe_extract_archive(archive_path, tmp_path / "extract")

    assert not (tmp_path / "escaped.txt").exists()


async def test_custom_download_validates_url(manager: ModelManager) -> None:
    with pytest.raises(ValueError, match="HTTPS"):
        manager.start_custom_download("http://example.com/model.bin")
    with pytest.raises(ValueError, match=".bin or .gguf"):
        manager.start_custom_download("https://example.com/model.txt")


async def test_custom_download_and_delete(
    manager: ModelManager, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "my-model.gguf"
    source.write_bytes(b"custom")
    # file:// URLs stand in for HTTPS in tests; validation is covered separately.
    monkeypatch.setattr(model_manager, "_validate_custom_url", lambda url: "my-model.gguf")
    state = manager.start_custom_download(source.as_uri())
    assert state.model_id == "custom:my-model.gguf"
    await asyncio.wait_for(_wait_finished(manager, "custom:my-model.gguf"), timeout=5)
    assert manager.installed_path("custom:my-model.gguf") is not None

    assert manager.delete("custom:my-model.gguf") is True
    assert manager.installed_path("custom:my-model.gguf") is None


async def test_delete_removes_folder(manager: ModelManager) -> None:
    kit_dir = manager.models_dir / "whisperkit" / "openai_whisper-tiny"
    kit_dir.mkdir(parents=True)
    (kit_dir / "config.json").write_text("{}")
    assert manager.delete("whisperkit:openai_whisper-tiny") is True
    assert not kit_dir.exists()
    assert manager.delete("whisperkit:openai_whisper-tiny") is False


async def _wait_finished(manager: ModelManager, model_id: str) -> None:
    while True:
        state = manager.download_state(model_id)
        assert state is not None
        if state.status != "downloading":
            return
        await asyncio.sleep(0.01)


# --------------------------------------------------------------- integrity

HELLO_SHA256 = "3b96f0f0e0e34e6d1b8bfe4f8ac71bfad9d0f8dd15b4ae0b8b1e4f4c4a0ac0b1"
"""Deliberately wrong digest for `b"hello model"`, used to force a rejection."""


def _sha256(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()


def test_normalize_sha256_accepts_prefixed_c1c74() -> None:
    digest = _sha256(b"hello model")
    assert normalize_sha256(f"  SHA256:{digest.upper()}  ") == digest


@pytest.mark.parametrize("value", ["", "nope", "abc123", "g" * 64, "a" * 63, "a" * 65])
def test_normalize_sha256_rejects_malformed(value: str) -> None:
    with pytest.raises(ValueError):
        normalize_sha256(value)


async def test_single_file_download_accepts_match_aaaa(
    tmp_path: Path, tiny_file_model: CatalogModel
) -> None:
    model = dataclasses.replace(tiny_file_model, sha256=_sha256(b"hello model"))
    manager = ModelManager(tmp_path / "models", catalog=(model,))

    manager.start_download(model.id)
    await asyncio.wait_for(_wait_finished(manager, model.id), timeout=5)

    state = manager.download_state(model.id)
    assert state is not None and state.status == "completed"
    installed = manager.installed_path(model.id)
    assert installed is not None and installed.read_bytes() == b"hello model"


async def test_single_file_download_rejects_wrong_aaaaa(
    tmp_path: Path, tiny_file_model: CatalogModel
) -> None:
    model = dataclasses.replace(tiny_file_model, sha256=HELLO_SHA256)
    manager = ModelManager(tmp_path / "models", catalog=(model,))

    manager.start_download(model.id)
    await asyncio.wait_for(_wait_finished(manager, model.id), timeout=5)

    state = manager.download_state(model.id)
    assert state is not None
    assert state.status == "failed"
    assert "SHA-256" in (state.error or "")
    # The whole point: a model that failed verification must not be installed,
    # and no partial file may survive for an engine to pick up later.
    assert manager.installed_path(model.id) is None
    assert not (manager.models_dir / "whisper.cpp" / "ggml-tiny.bin").exists()
    assert not (manager.models_dir / "whisper.cpp" / "ggml-tiny.bin.partial").exists()


async def test_folder_download_rejects_a_tampered_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A digest supplied by the repo listing is enforced during the transfer."""
    mirror = tmp_path / "mirror"
    folder = mirror / "example/repo/resolve/main/openai_whisper-tiny"
    folder.mkdir(parents=True)
    (folder / "config.json").write_text("{}")
    (folder / "weights.bin").write_bytes(b"tampered-bytes")
    monkeypatch.setattr(model_manager, "HF_BASE_URL", mirror.as_uri())
    monkeypatch.setattr(
        model_manager,
        "_list_repo_folder",
        lambda repo, name, revision="main": [
            RepoFile("config.json", 2, _sha256(b"{}")),
            RepoFile("weights.bin", 14, _sha256(b"the-bytes-we-expected")),
        ],
    )
    manager = ModelManager(tmp_path / "models", catalog=(TINY_FOLDER,))

    manager.start_download(TINY_FOLDER.id)
    await asyncio.wait_for(_wait_finished(manager, TINY_FOLDER.id), timeout=5)

    state = manager.download_state(TINY_FOLDER.id)
    assert state is not None and state.status == "failed"
    assert manager.installed_path(TINY_FOLDER.id) is None
    assert not (manager.models_dir / "whisperkit" / "openai_whisper-tiny.partial").exists()


async def test_catalog_digest_overrides_the_listi_cdadb(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pinned digest must win over whatever the API reports.

    Otherwise a compromised upstream could serve altered bytes together with
    matching metadata and the verification would happily agree with itself.
    """
    mirror = tmp_path / "mirror"
    folder = mirror / "example/repo/resolve/main/openai_whisper-tiny"
    folder.mkdir(parents=True)
    (folder / "config.json").write_bytes(b"upstream-swapped")
    monkeypatch.setattr(model_manager, "HF_BASE_URL", mirror.as_uri())
    monkeypatch.setattr(
        model_manager,
        "_list_repo_folder",
        # The listing vouches for the swapped bytes; the catalog does not.
        lambda repo, name, revision="main": [
            RepoFile("config.json", 16, _sha256(b"upstream-swapped"))
        ],
    )
    model = dataclasses.replace(
        TINY_FOLDER, file_digests=(("config.json", _sha256(b"the-reviewed-bytes")),)
    )
    manager = ModelManager(tmp_path / "models", catalog=(model,))

    manager.start_download(model.id)
    await asyncio.wait_for(_wait_finished(manager, model.id), timeout=5)

    state = manager.download_state(model.id)
    assert state is not None and state.status == "failed"
    assert manager.installed_path(model.id) is None


async def test_custom_download_rejects_wrong_user_d7ea6(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.gguf"
    source.write_bytes(b"custom-model-bytes")
    manager = ModelManager(tmp_path / "models", catalog=())
    monkeypatch.setattr(model_manager, "_validate_custom_url", lambda url: "my-model.gguf")

    manager.start_custom_download(source.as_uri(), HELLO_SHA256)
    await asyncio.wait_for(_wait_finished(manager, "custom:my-model.gguf"), timeout=5)

    state = manager.download_state("custom:my-model.gguf")
    assert state is not None and state.status == "failed"
    assert not (manager.models_dir / "whisper.cpp" / "my-model.gguf").exists()


async def test_custom_download_accepts_matching_u_a(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.gguf"
    source.write_bytes(b"custom-model-bytes")
    manager = ModelManager(tmp_path / "models", catalog=())
    monkeypatch.setattr(model_manager, "_validate_custom_url", lambda url: "my-model.gguf")

    manager.start_custom_download(source.as_uri(), _sha256(b"custom-model-bytes"))
    await asyncio.wait_for(_wait_finished(manager, "custom:my-model.gguf"), timeout=5)

    state = manager.download_state("custom:my-model.gguf")
    assert state is not None and state.status == "completed"


def test_custom_download_rejects_malformed_b52d7(tmp_path: Path) -> None:
    manager = ModelManager(tmp_path / "models", catalog=())
    with pytest.raises(ValueError):
        manager.start_custom_download("https://example.com/model.gguf", "not-a-digest")


async def test_archive_download_rejects_wrong_arc_a7516(tmp_path: Path) -> None:
    archive = tmp_path / "model.tar.bz2"
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:bz2") as handle:
        payload = b"onnx"
        info = tarfile.TarInfo("model-root/model.onnx")
        info.size = len(payload)
        handle.addfile(info, io.BytesIO(payload))
    archive.write_bytes(buffer.getvalue())
    model = dataclasses.replace(
        SHERPA_TEST,
        archive_url=archive.as_uri(),
        archive_root="model-root",
        required_files=("model.onnx",),
        sha256=HELLO_SHA256,
    )
    manager = ModelManager(tmp_path / "models", catalog=(model,))

    manager.start_download(model.id)
    await asyncio.wait_for(_wait_finished(manager, model.id), timeout=5)

    state = manager.download_state(model.id)
    assert state is not None and state.status == "failed"
    assert "SHA-256" in (state.error or "")
    assert manager.installed_path(model.id) is None


def test_pinned_catalog_revisions_reach_the_aa() -> None:
    """Pins must change the bytes actually fetched, not just be metadata."""
    from app.catalog import pin_download_url

    pinned = [m for m in DEFAULT_CATALOG if m.revision]
    assert pinned, "expected the shipped pin file to cover part of the catalog"
    for model in pinned:
        if model.download_url and "huggingface.co/" in model.download_url:
            assert "/resolve/main/" not in model.download_url
            assert f"/resolve/{model.revision}/" in model.download_url

    assert (
        pin_download_url("https://huggingface.co/o/r/resolve/main/f.bin", "abc123")
        == "https://huggingface.co/o/r/resolve/abc123/f.bin"
    )
    # Non-Hugging-Face URLs are left alone; they have no revision concept.
    assert pin_download_url("https://example.com/f.bin", "abc") == "https://example.com/f.bin"
    assert pin_download_url(None, "abc") is None


def test_shipped_pin_file_is_well_formed() -> None:
    from app.catalog import load_pins

    pins = load_pins()
    # A missing or emptied pin file degrades verification to a no-op without
    # any runtime error, so the floor is asserted here instead: losing pins in
    # packaging or a bad regeneration has to fail CI, not ship quietly.
    assert len(pins) >= 35, f"pin coverage collapsed to {len(pins)} models"
    catalog_ids = {model.id for model in DEFAULT_CATALOG}
    for model_id, record in pins.items():
        assert model_id in catalog_ids, f"pin for unknown model {model_id}"
        if "sha256" in record:
            assert normalize_sha256(record["sha256"]) == record["sha256"]
        for name, digest in record.get("file_digests", {}).items():
            assert normalize_sha256(digest) == digest, f"{model_id}:{name}"


def test_download_file_raises_and_removes_t_aaa(tmp_path: Path) -> None:
    """The low-level guarantee every other integrity test depends on."""
    import threading

    source = tmp_path / "source.bin"
    source.write_bytes(b"real-bytes")
    destination = tmp_path / "out.bin"
    state = model_manager.DownloadState(model_id="t")

    with pytest.raises(ModelIntegrityError) as caught:
        model_manager._download_file(
            source.as_uri(), destination, state, threading.Event(), "out.bin", HELLO_SHA256
        )

    assert "failed SHA-256 verification" in str(caught.value)
    assert not destination.exists()

    # With no expectation supplied the digest is still returned, which is what
    # lets the folder path verify against a listing it fetched separately.
    digest = model_manager._download_file(
        source.as_uri(), destination, state, threading.Event(), "out.bin"
    )
    assert digest == _sha256(b"real-bytes")
    assert destination.read_bytes() == b"real-bytes"


def test_download_file_retries_transient_ne_aaaa(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A dropped connection is retried, not reported as a bad file."""
    import threading

    monkeypatch.setattr(time, "sleep", lambda _seconds: None)

    payload = b"real-bytes"
    attempts = {"count": 0}

    class FakeResponse:
        def __init__(self, data: bytes) -> None:
            self._buffer = io.BytesIO(data)
            self.headers: dict[str, str] = {}

        def __enter__(self) -> FakeResponse:
            return self

        def __exit__(self, *exc: object) -> None:
            return None

        def read(self, size: int) -> bytes:
            return self._buffer.read(size)

    def fake_urlopen(request: object, timeout: int = 60) -> FakeResponse:
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise TimeoutError("simulated read timeout")
        return FakeResponse(payload)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    destination = tmp_path / "out.bin"
    state = model_manager.DownloadState(model_id="t")

    digest = model_manager._download_file(
        "https://example.invalid/model.bin", destination, state, threading.Event(), "out.bin"
    )

    assert attempts["count"] == 3
    assert digest == _sha256(payload)
    assert destination.read_bytes() == payload
    assert state.downloaded_bytes == len(payload)


def test_download_file_does_not_retry_hash_b4fa3(tmp_path: Path) -> None:
    """A mismatch fails immediately -- it is not assumed to be transient."""
    import threading

    source = tmp_path / "source.bin"
    source.write_bytes(b"real-bytes")
    destination = tmp_path / "out.bin"
    state = model_manager.DownloadState(model_id="t")

    with pytest.raises(ModelIntegrityError):
        model_manager._download_file(
            source.as_uri(), destination, state, threading.Event(), "out.bin", HELLO_SHA256
        )

    assert state.downloaded_bytes == len(b"real-bytes")


@pytest.mark.parametrize(
    "relative",
    [
        "/etc/authorized_keys",  # absolute: `Path("/a") / "/etc/x"` is `/etc/x`
        "../../escape.bin",
        "nested/../../escape.bin",
        "\\windows\\path",
        "",
        ".",  # names the model directory itself, not a file in it
    ],
)
def test_unsafe_listing_paths_are_rejected(relative: str) -> None:
    assert not model_manager.is_safe_relative_path(relative)


@pytest.mark.parametrize(
    "relative",
    [
        "model.bin",
        "tokenizer/vocab.json",
        "a/b/c.onnx",
        "a//b",  # POSIX collapses the empty segment; equivalent to a/b
        "a/./b",
    ],
)
def test_ordinary_listing_paths_are_accepted(relative: str) -> None:
    assert model_manager.is_safe_relative_path(relative)


async def test_folder_download_refuses_a_listing_aaaaa(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A repo listing must not be able to place files outside the model dir."""
    mirror = tmp_path / "mirror"
    folder = mirror / "example/repo/resolve/main/openai_whisper-tiny"
    folder.mkdir(parents=True)
    (folder / "config.json").write_text("{}")
    monkeypatch.setattr(model_manager, "HF_BASE_URL", mirror.as_uri())
    monkeypatch.setattr(
        model_manager,
        "_list_repo_folder",
        lambda repo, name, revision="main": [
            RepoFile("config.json", 2),
            RepoFile("../../../../escaped.bin", 4),
        ],
    )
    manager = ModelManager(tmp_path / "models", catalog=(TINY_FOLDER,))

    manager.start_download(TINY_FOLDER.id)
    await asyncio.wait_for(_wait_finished(manager, TINY_FOLDER.id), timeout=5)

    state = manager.download_state(TINY_FOLDER.id)
    assert state is not None and state.status == "failed"
    assert "unsafe path" in (state.error or "")
    assert not (tmp_path / "escaped.bin").exists()
    assert not (tmp_path / "models" / "escaped.bin").exists()


def test_pagination_only_follows_the_same_origin() -> None:
    """A compromised host must not be able to walk the pager off-origin."""
    origin = "https://huggingface.co/api/models/o/r/tree/main?recursive=true"
    assert model_manager._same_origin("https://huggingface.co/api/models/o/r?page=2", origin)
    for hostile in (
        "https://evil.example.com/api/models/o/r?page=2",
        "http://huggingface.co/api/models/o/r?page=2",  # downgraded scheme
        "file:///etc/passwd",
        "/api/models/o/r?page=2",  # scheme-relative, no origin of its own
    ):
        assert not model_manager._same_origin(hostile, origin), hostile


def test_generated_model_docs_match_the_catalog() -> None:
    """docs/models.md is generated; a catalog change must regenerate it.

    Without this the reference silently rots the moment a model is added,
    which is worse than not shipping one.
    """
    import subprocess

    root = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        [sys.executable, str(root / "scripts" / "generate_model_docs.py"), "--check"],
        capture_output=True,
        text=True,
        cwd=root,
    )
    assert result.returncode == 0, (
        f"{result.stdout}{result.stderr}\nRun: uv run scripts/generate_model_docs.py"
    )


# ------------------------------------------------------- language display


def _entry(**overrides: object) -> object:
    from app.schemas import AdminModelEntry

    base = dict(
        id="x:y",
        engine="sherpa-onnx",
        label="Test",
        size_bytes=1,
        languages="Multilingual",
        quality="Fast",
        family="Test",
        description="",
        source="t",
        state="not_installed",
        active=False,
        recommended=False,
    )
    base.update(overrides)
    return AdminModelEntry(**base)  # type: ignore[arg-type]


def _disclosure_html(entry: object) -> str:
    """Render the language-disclosure block through the real list pipeline
    (single-entry family) so these tests exercise the same path production
    uses, not a private helper."""
    from app.fragments.models import models_list_fragment

    return models_list_fragment([entry])  # type: ignore[list-item]


def test_language_preview_names_the_first_f_a() -> None:
    """The point of the change: answerable without opening every card."""
    html = _disclosure_html(
        _entry(language_names=["Bulgarian", "Croatian", "Czech", "Danish", "Dutch", "English"])
    )
    summary = html.split("</summary>")[0]
    for shown in ("Bulgarian", "Croatian", "Czech", "Danish"):
        assert shown in summary
    assert "+2 more" in summary
    # The rest are still present, just below the fold.
    assert "Dutch" in html and "English" in html


def test_short_language_lists_are_shown_wit_aa() -> None:
    html = _disclosure_html(_entry(language_names=["English", "French", "German"]))
    assert "<details" not in html
    for name in ("English", "French", "German"):
        assert name in html


def test_a_single_hidden_language_is_shown_aaa() -> None:
    """Five languages should not cost a click to reveal the fifth."""
    html = _disclosure_html(_entry(language_names=["a", "b", "c", "d", "e"]))
    assert "<details" not in html
    assert "more" not in html


def test_single_language_models_render_no_l_aaaa() -> None:
    html = _disclosure_html(_entry(language_names=["English"]))
    assert "model-languages" not in html


def test_auto_language_note_survives_both_layouts() -> None:
    for names in (["a", "b"], ["a", "b", "c", "d", "e", "f", "g"]):
        html = _disclosure_html(_entry(language_names=names, detects_language_automatically=True))
        assert "picks the language itself" in html


def test_language_names_are_escaped() -> None:
    html = _disclosure_html(_entry(language_names=["<script>", "b", "c", "d", "e", "f"]))
    assert "<script>" not in html
    assert "&lt;script&gt;" in html
