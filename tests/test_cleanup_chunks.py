from __future__ import annotations

from app.cleanup.base import DEFAULT_TOKEN_BUDGET
from app.cleanup.chunks import character_limit, pack

LIMIT = DEFAULT_TOKEN_BUDGET.input_tokens


def test_a_short_transcript_is_one_piece() -> None:
    assert pack("hello there", limit=LIMIT) == ["hello there"]


def test_empty_text_packs_to_nothing() -> None:
    assert pack("", limit=LIMIT) == []


def test_pieces_always_join_back_to_the_original() -> None:
    text = "First sentence. Second sentence.\n\nNew paragraph. And another."
    pieces = pack(text, limit=20)
    assert "".join(pieces) == text
    assert all(len(piece) <= 20 or " " not in piece.strip() for piece in pieces)


def test_blank_lines_are_preserved() -> None:
    text = "One.\n\nTwo."
    assert "".join(pack(text, limit=4)) == text


def test_a_long_url_is_not_split() -> None:
    url = "https://example.com/" + ("a" * 80)
    text = f"See {url} please."
    pieces = pack(text, limit=40)
    assert url in "".join(pieces)
    assert any(url in piece for piece in pieces)


def test_a_transcript_longer_than_the_limit_is_split() -> None:
    text = "Hello world. " * 400
    pieces = pack(text, limit=LIMIT)
    assert "".join(pieces) == text
    assert all(len(piece) <= LIMIT or " " not in piece.strip() for piece in pieces)
    assert len(pieces) > 1


def test_the_character_limit_follows_the_measured_density() -> None:
    """English costs about five characters a token; Devanagari about one."""
    english = character_limit(characters=5_000, tokens=1_000, budget_tokens=1_536)
    dense = character_limit(characters=1_000, tokens=1_000, budget_tokens=1_536)
    assert english > 1_536 > dense
    # A script the tokenizer has no entries for costs more than a token a
    # character, and the limit has to shrink below the budget for it.
    byte_fallback = character_limit(characters=1_000, tokens=2_000, budget_tokens=1_536)
    assert byte_fallback < dense


def test_the_character_limit_never_returns_nothing_to_pack_with() -> None:
    assert character_limit(characters=0, tokens=0, budget_tokens=1_536) == 1_536
    assert character_limit(characters=1, tokens=10_000, budget_tokens=8) == 1
