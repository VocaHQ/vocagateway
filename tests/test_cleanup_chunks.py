from __future__ import annotations

from app.cleanup.base import CHUNK_CHAR_LIMIT
from app.cleanup.chunks import pack


def test_a_short_transcript_is_one_piece() -> None:
    assert pack("hello there") == ["hello there"]


def test_empty_text_packs_to_nothing() -> None:
    assert pack("") == []


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


def test_the_default_limit_matches_the_runtime_budget() -> None:
    text = "Hello world. " * 400
    pieces = pack(text)
    assert "".join(pieces) == text
    assert all(len(piece) <= CHUNK_CHAR_LIMIT or " " not in piece.strip() for piece in pieces)
    assert len(pieces) > 1
