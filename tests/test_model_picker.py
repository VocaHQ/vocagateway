# Copyright (c) 2025-present VocaHQ, Inc.
# SPDX-License-Identifier: AGPL-3.0-only
# See the LICENSE file for details.

from __future__ import annotations

from app.fragments.models import model_picker_fragment
from app.schemas import AdminModelEntry

MEGABYTE = 1024 * 1024


def _entry(**overrides: object) -> AdminModelEntry:
    base: dict[str, object] = {
        "id": "sherpa-onnx:example",
        "engine": "sherpa-onnx",
        "label": "Example",
        "size_bytes": 200 * MEGABYTE,
        "languages": "English only",
        "quality": "Balanced",
        "family": "Example",
        "description": "An example model.",
        "source": "sherpa-onnx",
        "state": "not_installed",
        "active": False,
        "recommended": False,
        "speed_rating": 3,
        "accuracy_rating": 3,
        "language_codes": ["en"],
    }
    base.update(overrides)
    return AdminModelEntry(**base)  # type: ignore[arg-type]


def _spread() -> list[AdminModelEntry]:
    """One clear winner per intent, so each card has an obvious right answer."""
    return [
        _entry(id="a:quick", label="Quick", speed_rating=5, accuracy_rating=2),
        _entry(id="a:exact", label="Exact", speed_rating=1, accuracy_rating=5),
        _entry(id="a:middle", label="Middle", speed_rating=4, accuracy_rating=4),
    ]


def test_picker_names_one_model_per_intent() -> None:
    html = model_picker_fragment(_spread())
    for intent in ("balanced", "accurate", "fastest"):
        assert f'data-intent="{intent}"' in html
    assert "Middle" in html and "Exact" in html and "Quick" in html


def test_picker_never_repeats_a_model() -> None:
    """Three cards showing the same name would read as a broken page.

    With only two candidates the balanced and accurate picks would otherwise
    collide, so each intent skips what earlier ones already took.
    """
    entries = [
        _entry(id="a:best", label="Best", speed_rating=5, accuracy_rating=5),
        _entry(id="a:next", label="Next", speed_rating=4, accuracy_rating=4),
        _entry(id="a:third", label="Third", speed_rating=3, accuracy_rating=3),
    ]
    html = model_picker_fragment(entries)
    for label in ("Best", "Next", "Third"):
        assert html.count(f">{label}</h4>") == 1


def test_picker_leads_with_the_balanced_pick() -> None:
    """It is the "start here" advice, so it is drawn first and marked."""
    html = model_picker_fragment(_spread())
    assert html.index('data-intent="balanced"') < html.index('data-intent="accurate"')
    assert 'data-intent="balanced" data-lead="1"' in html


def test_fastest_prefers_a_model_that_streams_live() -> None:
    """The card exists for live dictation, where incremental decoding is the
    difference an operator feels — not one rating point."""
    entries = [
        _entry(id="a:batch", label="Batch", speed_rating=5, supports_streaming=False),
        _entry(id="a:live", label="Live", speed_rating=4, supports_streaming=True),
        _entry(id="a:other", label="Other", speed_rating=2, accuracy_rating=5),
        _entry(id="a:filler", label="Filler", speed_rating=3, accuracy_rating=3),
    ]
    html = model_picker_fragment(entries)
    fastest = html.split('data-intent="fastest"')[1]
    assert "Live" in fastest.split("</article>")[0]


def test_picker_skips_models_this_host_cannot_run() -> None:
    """Recommending a model the machine cannot load is worse than recommending
    nothing, so a missing runtime takes it out of the running entirely."""
    entries = [
        *_spread(),
        _entry(
            id="a:unrunnable",
            label="Unrunnable",
            speed_rating=5,
            accuracy_rating=5,
            runtime_requirement="MLX Audio",
        ),
    ]
    assert "Unrunnable" not in model_picker_fragment(entries)


def test_picker_skips_retired_and_unrated_models() -> None:
    entries = [
        *_spread(),
        _entry(id="a:old", label="Retired", speed_rating=5, accuracy_rating=5, retired=True),
        # A user-supplied custom model carries no rating, so it cannot be ranked.
        _entry(id="custom:mine", label="Custom", speed_rating=0, accuracy_rating=0),
    ]
    html = model_picker_fragment(entries)
    assert "Retired" not in html
    assert "Custom" not in html


def test_picker_respects_the_chosen_language() -> None:
    entries = [
        _entry(id="a:en", label="EnglishOnly", language_codes=["en"]),
        _entry(id="a:de", label="GermanOnly", language_codes=["de"]),
        # No codes at all means the model takes any language.
        _entry(id="a:any", label="AnyLanguage", language_codes=[]),
    ]
    german = model_picker_fragment(entries, "de")
    assert "GermanOnly" in german
    assert "AnyLanguage" in german
    assert "EnglishOnly" not in german


def test_picker_says_so_when_nothing_covers_the_language() -> None:
    html = model_picker_fragment([_entry(language_codes=["en"])], "ja")
    assert "empty-state" in html
    assert "Japanese" in html


def test_the_cpu_preference_never_wins_best_accuracy() -> None:
    """Parakeet's edge on CPU is throughput, not transcript quality. If a
    genuinely more accurate model is available it must still take that card,
    or the panel would be recommending the wrong thing for the stated intent.
    """
    entries = [
        # What a CPU-only host's ratings look like: Parakeet carries the speed
        # bonus, Whisper large-v3 keeps the higher accuracy.
        _entry(id="a:parakeet", label="Parakeet", speed_rating=4, accuracy_rating=4),
        _entry(id="a:whisper", label="WhisperLarge", speed_rating=2, accuracy_rating=5),
        _entry(id="a:tiny", label="Tiny", speed_rating=5, accuracy_rating=2),
    ]
    html = model_picker_fragment(entries)
    accurate = html.split('data-intent="accurate"')[1].split("</article>")[0]
    assert "WhisperLarge" in accurate
    balanced = html.split('data-intent="balanced"')[1].split("</article>")[0]
    assert "Parakeet" in balanced
