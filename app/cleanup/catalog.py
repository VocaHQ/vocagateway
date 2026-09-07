"""Artifact metadata for the text models that perform transcript cleanup.

A deliberately separate namespace from `app/catalog.py`. A cleanup model is not
a speech engine: it must never appear in `/v1/models`, never be selectable as an
ASR engine, and never inherit an ASR entry's language or audio assumptions. It
also has to record things a speech entry has no field for — which upstream
weights a quantization came from, where the chat template comes from, and which
languages have actually been evaluated for *this* task.

Provenance rule: a community GGUF is a separate artifact from the model whose
card it borrows. The entries below therefore point at quantizations published by
the model's own authors, in the model's own repository, and every download is
gated on a SHA-256 harvested from that repository (see
`app/cleanup_model_pins.json`). An entry with no pinned digest cannot be
installed at all.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

from app.catalog import CatalogModel

PINS_PATH = Path(__file__).resolve().parent.parent / "cleanup_model_pins.json"

RUNTIME_LLAMA_CPP = "llama.cpp"
# The pseudo-engine the shared download machinery files these artifacts under,
# inside the cleanup manager's own models directory. It is never an ASR engine
# id and is never offered to `engines.build_engine`.
DOWNLOAD_ENGINE = "llama.cpp"
APACHE_LICENSE = "Apache 2.0"
HF_BASE_URL = "https://huggingface.co"

# The buckets the model-selection corpus covers. These are *candidates*, not
# claims: `evaluated_languages` stays empty until a model has actually cleared
# the release gates for a language, and the capability endpoint reports the two
# separately so a client can tell a tested language from an offered one.
CANDIDATE_LANGUAGES: tuple[str, ...] = ("en", "hi", "hinglish_roman")
ENGLISH_ONLY: tuple[str, ...] = ("en",)

# Artifact sizes as published by the upstream repository, and the host memory
# each needs beside a resident speech model. Both are display and admission
# figures; the SHA-256 in the pin file is what actually vouches for the bytes.
QWEN3_06B_BYTES = 639_446_688
QWEN3_06B_MINIMUM_RAM_GB = 4.0
QWEN3_06B_Q4_BYTES = 428_970_080
QWEN3_17B_BYTES = 1_834_426_016
QWEN3_17B_MINIMUM_RAM_GB = 8.0


@dataclass(frozen=True, slots=True)
class CleanupModel:
    """One installable cleanup artifact and everything needed to vouch for it."""

    id: str
    label: str
    description: str
    runtime: str
    huggingface_repo: str
    filename: str
    size_bytes: int
    minimum_ram_gb: float
    # Where the quantized file came from, kept apart from the repository that
    # serves it so a community conversion can never borrow an upstream card's
    # authority by sitting in the same field.
    upstream_model: str
    quantization: str
    conversion_source: str
    chat_template_source: str
    license_name: str = APACHE_LICENSE
    license_notice: str = ""
    context_tokens: int = 4_096
    candidate_languages: tuple[str, ...] = CANDIDATE_LANGUAGES
    # Languages this artifact has passed the published release gates for. Empty
    # until a measured evaluation report says otherwise; never inferred from an
    # upstream card's multilingual claim.
    evaluated_languages: tuple[str, ...] = ()
    revision: str | None = None
    sha256: str | None = None

    @property
    def download_url(self) -> str:
        revision = self.revision or "main"
        return f"{HF_BASE_URL}/{self.huggingface_repo}/resolve/{revision}/{self.filename}"

    @property
    def source_url(self) -> str:
        return f"{HF_BASE_URL}/{self.huggingface_repo}"

    @property
    def installable(self) -> bool:
        """Whether this entry may be downloaded at all.

        No pinned revision and digest means the exact bytes this entry was
        reviewed against cannot be recognised, and an unverifiable model file is
        the one thing that must never reach a `llama-server` launch.
        """
        return bool(self.revision and self.sha256)

    def as_catalog_model(self) -> CatalogModel:
        """Adapt to the shared verified-download record.

        The download path only needs a URL, a destination, a size, and a digest.
        Reusing it keeps staging, resumption semantics, cancellation, and atomic
        installation identical to the speech models rather than reimplemented.
        """
        return CatalogModel(
            id=self.id,
            engine=DOWNLOAD_ENGINE,
            key=self.filename,
            label=self.label,
            size_bytes=self.size_bytes,
            languages=", ".join(self.candidate_languages),
            quality=self.quantization,
            minimum_ram_gb=self.minimum_ram_gb,
            download_url=self.download_url,
            huggingface_repo=None,
            family="Transcript cleanup",
            description=self.description,
            source=self.runtime,
            license_name=self.license_name,
            revision=self.revision,
            sha256=self.sha256,
        )


_BASE_CATALOG: tuple[CleanupModel, ...] = (
    CleanupModel(
        id="cleanup:qwen3-0.6b",
        label="Qwen3 0.6B (Q8_0)",
        description=(
            "Higher-precision 0.6B baseline for grammar, punctuation, and casing cleanup."
        ),
        runtime=RUNTIME_LLAMA_CPP,
        huggingface_repo="Qwen/Qwen3-0.6B-GGUF",
        filename="Qwen3-0.6B-Q8_0.gguf",
        size_bytes=QWEN3_06B_BYTES,
        minimum_ram_gb=QWEN3_06B_MINIMUM_RAM_GB,
        upstream_model="Qwen/Qwen3-0.6B",
        # Qwen publishes Q8_0 in its own GGUF repository and no smaller
        # quantization. A Q4_K_M would halve the download, but only from a
        # third-party converter whose bytes the upstream card does not vouch
        # for — so first-party provenance wins over file size here, and a
        # smaller quantization needs its own provenance review.
        quantization="Q8_0",
        conversion_source="Upstream-published quantization (Qwen/Qwen3-0.6B-GGUF)",
        chat_template_source="Embedded in the GGUF by the upstream conversion",
        license_notice="Apache License 2.0, Alibaba Cloud (Qwen).",
    ),
    CleanupModel(
        id="cleanup:qwen3-0.6b-q4",
        label="Qwen3 0.6B Compact (Q4_0)",
        description=(
            "English-only compact candidate. One-third smaller than Q8_0; "
            "choose it for disk footprint, not for an unproven 2x speed claim."
        ),
        runtime=RUNTIME_LLAMA_CPP,
        huggingface_repo="ggml-org/Qwen3-0.6B-GGUF",
        filename="Qwen3-0.6B-Q4_0.gguf",
        size_bytes=QWEN3_06B_Q4_BYTES,
        minimum_ram_gb=QWEN3_06B_MINIMUM_RAM_GB,
        upstream_model="Qwen/Qwen3-0.6B",
        quantization="Q4_0",
        conversion_source="llama.cpp project conversion (ggml-org/Qwen3-0.6B-GGUF)",
        chat_template_source="Embedded in the GGUF by the llama.cpp project conversion",
        license_notice=(
            "Apache License 2.0, Alibaba Cloud (Qwen); GGUF conversion published by ggml-org."
        ),
        candidate_languages=ENGLISH_ONLY,
    ),
    CleanupModel(
        id="cleanup:qwen3-1.7b",
        label="Qwen3 1.7B (Q8_0)",
        description=(
            "Larger cleanup candidate for comparison. Adopt only if it is "
            "measurably better within the latency and memory budget."
        ),
        runtime=RUNTIME_LLAMA_CPP,
        huggingface_repo="Qwen/Qwen3-1.7B-GGUF",
        filename="Qwen3-1.7B-Q8_0.gguf",
        size_bytes=QWEN3_17B_BYTES,
        minimum_ram_gb=QWEN3_17B_MINIMUM_RAM_GB,
        upstream_model="Qwen/Qwen3-1.7B",
        quantization="Q8_0",
        conversion_source="Upstream-published quantization (Qwen/Qwen3-1.7B-GGUF)",
        chat_template_source="Embedded in the GGUF by the upstream conversion",
        license_notice="Apache License 2.0, Alibaba Cloud (Qwen).",
    ),
)


def load_pins(path: Path = PINS_PATH) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    models = payload.get("models") if isinstance(payload, dict) else None
    return models if isinstance(models, dict) else {}


def apply_pins(
    models: tuple[CleanupModel, ...] = _BASE_CATALOG, pins: dict[str, Any] | None = None
) -> tuple[CleanupModel, ...]:
    records = load_pins() if pins is None else pins
    return tuple(_pin(model, records.get(model.id)) for model in models)


def _pin(model: CleanupModel, record: Any) -> CleanupModel:
    """Read one harvested record.

    Both shapes the harvester can write are accepted: a single-file `sha256`
    (what a pinned download URL produces) and a `file_digests` map keyed by
    filename (which is easier to review in a diff).
    """
    if not isinstance(record, dict):
        return model
    digests = record.get("file_digests")
    named = digests.get(model.filename) if isinstance(digests, dict) else None
    return replace(
        model,
        revision=record.get("revision") or model.revision,
        sha256=record.get("sha256") or named or model.sha256,
    )


CLEANUP_CATALOG: tuple[CleanupModel, ...] = apply_pins()
CLEANUP_BY_ID: MappingProxyType[str, CleanupModel] = MappingProxyType(
    {model.id: model for model in CLEANUP_CATALOG}
)
DEFAULT_MODEL_ID = CLEANUP_CATALOG[0].id


def cleanup_model(model_id: str | None) -> CleanupModel | None:
    return CLEANUP_BY_ID.get(model_id or "")


def download_catalog() -> tuple[CatalogModel, ...]:
    """The installable subset, as records the shared download machinery accepts."""
    return tuple(model.as_catalog_model() for model in CLEANUP_CATALOG if model.installable)
