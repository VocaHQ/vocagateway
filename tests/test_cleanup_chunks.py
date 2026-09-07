from __future__ import annotations

from app.cleanup.base import DEFAULT_TOKEN_BUDGET, JSON_WRAPPER_TOKENS
from app.cleanup.chunks import character_limit, pack

LIMIT = DEFAULT_TOKEN_BUDGET.input_tokens
PACKED = DEFAULT_TOKEN_BUDGET.packed_tokens


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


def test_an_initial_is_kept_with_the_following_oversize_token() -> None:
    token = "B" * 250
    text = "A. " + token
    pieces = pack(text, limit=200)
    assert "".join(pieces) == text
    assert token in pieces[0]
    assert "A. " not in pieces


def test_titles_and_initialisms_are_kept_with_the_following_oversize_token() -> None:
    token = "B" * 250
    for prefix in ("Mr. ", "Dr. ", "U.S. ", "Prof. ", "vs. "):
        text = prefix + token
        pieces = pack(text, limit=200)
        assert "".join(pieces) == text
        assert pieces[0].startswith(prefix)
        assert token in pieces[0]
        assert prefix not in pieces


def test_real_sentences_still_split_when_over_the_limit() -> None:
    text = "Hello. World."
    pieces = pack(text, limit=8)
    assert "".join(pieces) == text
    assert pieces == ["Hello. ", "World."]


def test_abbreviation_pieces_join_back_to_the_original() -> None:
    token = "B" * 250
    text = f"A. {token} Mr. {token} Hello. World. {token}"
    pieces = pack(text, limit=200)
    assert "".join(pieces) == text
    assert "A. " not in pieces
    assert "Mr. " not in pieces


def test_an_abbreviation_does_not_push_a_limit_piece_over_the_output_budget() -> None:
    filler = ("hello " * 400)[:PACKED]
    assert len(filler) == PACKED
    for prefix in ("A. ", "Mr. Dr. "):
        text = prefix + filler
        pieces = pack(text, limit=PACKED)
        assert "".join(pieces) == text
        for piece in pieces:
            if len(piece) > PACKED and " " not in piece.strip():
                continue
            assert len(piece) + JSON_WRAPPER_TOKENS <= DEFAULT_TOKEN_BUDGET.output_tokens


def test_abbreviation_only_reassembly_stays_within_the_output_budget() -> None:
    text = "A. " * 633
    pieces = pack(text, limit=PACKED)
    assert "".join(pieces) == text
    assert len(pieces) > 1
    assert all(
        len(piece) + JSON_WRAPPER_TOKENS <= DEFAULT_TOKEN_BUDGET.output_tokens for piece in pieces
    )
