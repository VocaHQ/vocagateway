"""Relative speed and accuracy ratings, 1-5, for catalog models.

These are **not** measured WER. Nobody benchmarks all 57 models on the operator's
hardware, and printing an invented error rate next to a model name would be
worse than printing nothing. What the catalog does already carry is the author's
own judgement, written into `quality` ("Fastest · cached streaming", "Most
accurate English · punctuation"), plus an objective proxy in `size_bytes`.

So the ratings restate what the catalog already claims on a scale the UI can
compare across families, and the UI says as much wherever it shows them. Change
a model's `quality` wording and its rating moves with it, which is the intent:
one source of truth, not a second set of numbers to keep in step.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.catalog import CatalogModel

MINIMUM_RATING = 1
MAXIMUM_RATING = 5

# Speed comes from size first. Prose is unreliable here: `quality` grades a
# model against its own family ("Fastest" Moonshine, "Balanced" Moonshine), so
# taken literally it rates a 257 MB tier faster than a 136 MB one. Bytes do not
# have that problem, and decode cost tracks them closely within an architecture.
_MEGABYTE = 1024 * 1024
_SPEED_BY_SIZE_MB: tuple[tuple[int, int], ...] = (
    (150, 5),
    (400, 4),
    (800, 3),
    (1600, 2),
)
_SLOWEST_RATING = 1

# Wording still carries real signal, as a nudge either way: a distilled or
# turbo decoder beats its weight class, and an accuracy-first build gives up
# speed to get there.
_FASTER_PHRASES = ("fastest", "low latency", "fast decoder", "distil", "turbo", "smallest")
_SLOWER_PHRASES = ("most accurate", "large-model accuracy")

# Streaming exports decode incrementally instead of re-running on a growing
# buffer, which is the difference an operator feels while dictating.
_STREAMING_SPEED_BONUS = 1

# Accuracy is the other way round: the wording is a deliberate ranking the
# catalog author made across families, which no size heuristic can reproduce.
# Ordered — "most accurate" has to be tested before "accurate".
_ACCURACY_PHRASES: tuple[tuple[str, int], ...] = (
    ("most accurate", 5),
    ("large-model accuracy", 5),
    ("fast and accurate", 4),
    ("accurate", 4),
    ("balanced", 3),
    ("compact", 3),
    ("fast", 3),
    ("fastest", 2),
    ("smallest", 2),
)
# Size breaks ties between tiers the wording grades identically.
_ACCURACY_SIZE_TIERS_MB: tuple[tuple[int, int], ...] = ((120, -1), (1200, 0))
_LARGE_MODEL_ACCURACY_NUDGE = 1


def _phrase_rating(quality: str, phrases: tuple[tuple[str, int], ...], fallback: int) -> int:
    lowered = quality.lower()
    for phrase, rating in phrases:
        if phrase in lowered:
            return rating
    return fallback


def _clamp(rating: int) -> int:
    return max(MINIMUM_RATING, min(MAXIMUM_RATING, rating))


def _speed_by_size(size_bytes: int) -> int:
    megabytes = size_bytes / _MEGABYTE
    for upper, rating in _SPEED_BY_SIZE_MB:
        if megabytes < upper:
            return rating
    return _SLOWEST_RATING


def _accuracy_size_nudge(size_bytes: int) -> int:
    megabytes = size_bytes / _MEGABYTE
    for upper, nudge in _ACCURACY_SIZE_TIERS_MB:
        if megabytes < upper:
            return nudge
    return _LARGE_MODEL_ACCURACY_NUDGE


# Parakeet TDT on a host with no GPU. The ratings are otherwise host-agnostic,
# which understates this one badly: a transducer pairs a full encoder with a
# tiny decoder, so on CPU it costs a fraction of what Whisper's 32-layer
# autoregressive decoder costs at the same download size. Kept narrow on
# purpose — it is a claim about this architecture, not about sherpa-onnx or
# quantisation in general — and it moves speed only. Parakeet is not more
# accurate for being on a CPU box, so "Best accuracy" is unaffected.
_CPU_FAVOURED_MARKERS = ("parakeet",)
_CPU_ONLY_SPEED_BONUS = 1


def _is_cpu_favoured(model: CatalogModel) -> bool:
    return any(marker in model.id.lower() for marker in _CPU_FAVOURED_MARKERS)


def _speed_word_nudge(model: CatalogModel) -> int:
    # Label as well as quality: "Turbo" and "Distil" are named in the model
    # name, and a turbo decoder is the clearest case of a build outrunning its
    # weight class.
    lowered = f"{model.label} {model.quality}".lower()
    nudge = 0
    if any(phrase in lowered for phrase in _FASTER_PHRASES):
        nudge += 1
    if any(phrase in lowered for phrase in _SLOWER_PHRASES):
        nudge -= 1
    return nudge


def speed_rating(model: CatalogModel, *, cpu_only: bool = False) -> int:
    """How quickly this model turns speech into text, 1 (slowest) to 5.

    `cpu_only` describes the host, not the model: pass it when the machine
    reports no GPU accelerator, so a Parakeet transducer is not rated against
    Whisper as though both were running on a graphics card.
    """
    bonus = _STREAMING_SPEED_BONUS if model.supports_streaming else 0
    if cpu_only and _is_cpu_favoured(model):
        bonus += _CPU_ONLY_SPEED_BONUS
    return _clamp(_speed_by_size(model.size_bytes) + _speed_word_nudge(model) + bonus)


def accuracy_rating(model: CatalogModel) -> int:
    """How faithful the transcript is, 1 (roughest) to 5."""
    base = _phrase_rating(model.quality, _ACCURACY_PHRASES, fallback=3)
    return _clamp(base + _accuracy_size_nudge(model.size_bytes))
