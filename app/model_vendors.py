# Copyright (c) 2025-present VocaHQ, Inc.
# SPDX-License-Identifier: AGPL-3.0-only
# See the LICENSE file for details.

"""Who made each model family, and how to mark it in the model list.

The family cards answer "how many models, how big, which engine" but not the
first thing anyone actually recognises: whose model this is. Parakeet is
NVIDIA's, Whisper is OpenAI's, SenseVoice is Alibaba's — knowing that is most
of how an operator decides what to try, and it was nowhere on the page.

Two things are deliberate here.

*The mapping is by hand.* It cannot be derived from the download URL: almost
every sherpa-onnx entry is served from a converter's Hugging Face account
(`csukuangfj`, `k2-fsa`), and reading the vendor off that would credit the
person who exported the ONNX rather than the lab that trained the weights.

*Not every vendor gets a glyph.* Several — OpenAI and IBM among them — have had
their marks withdrawn from the open icon sets, so there is no icon anyone may
redistribute. Those vendors get a monogram in the same badge, in their own
colour: the same information, the same shape on the page, and no logo file this
repository has no right to ship. `glyph` says which case a vendor is.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

# Slugs, named once: they are the CSS class, the glyph macro, and the key of
# every mapping below, so a typo in one of those places has to be impossible.
NVIDIA = "nvidia"
OPENAI = "openai"
ALIBABA = "alibaba"
QWEN = "qwen"
MEDIATEK = "mediatek"
IBM = "ibm"
COHERE = "cohere"
USEFUL_SENSORS = "useful-sensors"
DATAOCEAN = "dataocean"
SALUTE = "salute"
K2FSA = "k2fsa"

# Vendors whose mark is drawn from the CC0 icon set vendored in
# app/templates/macros/vendor_marks.html. Every other vendor renders its
# monogram instead; nothing here decides *how* a badge looks, only what it says.
# OpenAI is the conspicuous absence, and it is not an oversight: its mark was
# withdrawn from the open icon sets, so there is none to vendor. It reads "OAI".
GLYPH_VENDORS = frozenset((NVIDIA, ALIBABA, QWEN, MEDIATEK))


@dataclass(frozen=True, slots=True)
class Vendor:
    """One organisation, as the model list shows it."""

    slug: str
    name: str
    # Shown when there is no glyph. Two or three characters: it sits in a badge
    # beside a name that spells the vendor out, so it only has to be
    # recognisable, not self-explanatory.
    monogram: str

    @property
    def glyph(self) -> str:
        """The glyph macro to draw, or "" when this vendor uses its monogram."""
        return self.slug if self.slug in GLYPH_VENDORS else ""


_VENDORS: tuple[Vendor, ...] = (
    Vendor(NVIDIA, "NVIDIA", "NV"),
    Vendor(OPENAI, "OpenAI", "OAI"),
    Vendor(ALIBABA, "Alibaba", "AL"),
    Vendor(QWEN, "Qwen", "QW"),
    Vendor(MEDIATEK, "MediaTek", "MTK"),
    Vendor(IBM, "IBM", "IBM"),
    Vendor(COHERE, "Cohere", "CO"),
    Vendor(USEFUL_SENSORS, "Useful Sensors", "US"),
    Vendor(DATAOCEAN, "DataoceanAI", "DO"),
    Vendor(SALUTE, "SaluteDevices", "SD"),
    Vendor(K2FSA, "k2-fsa", "K2"),
)
VENDORS: MappingProxyType[str, Vendor] = MappingProxyType(
    {vendor.slug: vendor for vendor in _VENDORS}
)

# Family label -> vendor slug. Families are the catalog's own `family` strings,
# including the `/ MLX` and `/ CTranslate2` variants: those are re-exports of
# the same weights, so they carry the same vendor as the model they came from.
# A test holds this to the catalog, so a new family cannot ship unmarked.
_FAMILY_VENDORS: MappingProxyType[str, str] = MappingProxyType(
    {
        "Breeze ASR": MEDIATEK,
        "Canary": NVIDIA,
        "Cohere Transcribe": COHERE,
        "Dolphin": DATAOCEAN,
        "GigaAM": SALUTE,
        "Granite Speech / MLX": IBM,
        "Moonshine": USEFUL_SENSORS,
        "Nemotron 3.5 ASR": NVIDIA,
        "Parakeet TDT": NVIDIA,
        "Parakeet TDT / MLX": NVIDIA,
        "Parakeet Unified": NVIDIA,
        "Qwen3-ASR": QWEN,
        "Qwen3-ASR / MLX": QWEN,
        "SenseVoice": ALIBABA,
        "Whisper": OPENAI,
        "Whisper / CTranslate2": OPENAI,
        "Whisper / Hinglish": OPENAI,
        "Whisper / MLX": OPENAI,
        "Zipformer": K2FSA,
    }
)


def vendor_for(family: str) -> Vendor | None:
    """The organisation behind a family, or None when there is no honest answer.

    A user's own imported weights are the None case, and they get no badge:
    inventing an origin for a file someone dropped in themselves would be worse
    than leaving the space empty.
    """
    return VENDORS.get(_FAMILY_VENDORS.get(family, ""))


def mapped_families() -> frozenset[str]:
    """Every family this module can mark. Read by the test that guards it."""
    return frozenset(_FAMILY_VENDORS)
