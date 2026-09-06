from __future__ import annotations

import pytest

from app.catalog import DEFAULT_CATALOG, CatalogModel
from app.model_ratings import MAXIMUM_RATING, MINIMUM_RATING, accuracy_rating, speed_rating

MEGABYTE = 1024 * 1024


def _model(**overrides: object) -> CatalogModel:
    base: dict[str, object] = {
        "id": "sherpa-onnx:example",
        "engine": "sherpa-onnx",
        "key": "example",
        "label": "Example",
        "size_bytes": 200 * MEGABYTE,
        "languages": "English only",
        "quality": "Balanced",
        "minimum_ram_gb": 4.0,
    }
    base.update(overrides)
    return CatalogModel(**base)  # type: ignore[arg-type]


def test_every_catalog_model_is_rated_in_range() -> None:
    """The UI draws exactly five pips, so a 0 or a 6 would render wrong."""
    for model in DEFAULT_CATALOG:
        speed = speed_rating(model)
        accuracy = accuracy_rating(model)
        assert MINIMUM_RATING <= speed <= MAXIMUM_RATING, model.id
        assert MINIMUM_RATING <= accuracy <= MAXIMUM_RATING, model.id


def test_ratings_use_the_whole_scale() -> None:
    """A rating everything scores 3 on tells the reader nothing."""
    speeds = {speed_rating(model) for model in DEFAULT_CATALOG}
    accuracies = {accuracy_rating(model) for model in DEFAULT_CATALOG}
    assert speeds == set(range(MINIMUM_RATING, MAXIMUM_RATING + 1))
    assert accuracies == set(range(MINIMUM_RATING, MAXIMUM_RATING + 1))


@pytest.mark.parametrize(
    ("megabytes", "heavier_megabytes"),
    [(40, 300), (300, 900), (900, 2000)],
)
def test_a_heavier_model_is_never_rated_faster(megabytes: int, heavier_megabytes: int) -> None:
    """Speed is derived from size precisely so this holds.

    The catalog's own wording does not: it grades each model against its family,
    which rated a 257 MB Moonshine tier "Fastest" and a 136 MB one "Balanced".
    """
    light = _model(size_bytes=megabytes * MEGABYTE)
    heavy = _model(size_bytes=heavier_megabytes * MEGABYTE)
    assert speed_rating(light) > speed_rating(heavy)


def test_streaming_models_rate_faster_than_their_batch_twin() -> None:
    batch = _model(size_bytes=300 * MEGABYTE, supports_streaming=False)
    streaming = _model(size_bytes=300 * MEGABYTE, supports_streaming=True)
    assert speed_rating(streaming) > speed_rating(batch)


def test_accuracy_follows_the_catalog_wording() -> None:
    """Accuracy is the one the wording ranks well: it is a deliberate ordering
    across families that no size heuristic reproduces."""
    ladder = ["Smallest · batch", "Fastest", "Balanced", "Accurate", "Most accurate"]
    ratings = [accuracy_rating(_model(quality=quality)) for quality in ladder]
    assert ratings == sorted(ratings)
    assert ratings[0] < ratings[-1]


def test_accuracy_wording_beats_a_substring_of_itself() -> None:
    """ "Most accurate" must not be read as the weaker "accurate"."""
    assert accuracy_rating(_model(quality="Most accurate")) > accuracy_rating(
        _model(quality="Accurate")
    )


def test_an_accuracy_first_build_gives_up_speed() -> None:
    same_size = 900 * MEGABYTE
    assert speed_rating(_model(size_bytes=same_size, quality="Most accurate")) < speed_rating(
        _model(size_bytes=same_size, quality="Balanced")
    )


def test_a_turbo_decoder_outruns_its_weight_class() -> None:
    """ "Turbo" is in the label, not in `quality`, so the scan has to read both."""
    plain = _model(size_bytes=1500 * MEGABYTE, label="Whisper Large v3")
    turbo = _model(size_bytes=1500 * MEGABYTE, label="Whisper Large v3 Turbo")
    assert speed_rating(turbo) > speed_rating(plain)
