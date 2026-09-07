"""Split a long transcript into pieces that fit one cleanup inference.

A 120 s dictation is usually one piece. When it is not, refusing the whole
thing leaves grammar uncorrected for no reason other than the KV window.
Pieces are packed on paragraph, then sentence, then whitespace boundaries so
a URL or path is not cut in half. Joining them in order recovers the original
bytes exactly; the runtime stitches model output the same way.

Splitting is a last resort, not a default: a piece is corrected without the
sentences on either side of it, so the fewer pieces a transcript needs, the
better the correction. `character_limit` therefore sizes them from the
transcript's own measured token density rather than from a worst case.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from app.cleanup.base import TOKEN_ESTIMATE_MARGIN

_PARAGRAPH = re.compile(r"\n\n+")
_SENTENCE = re.compile(r"[.!?。！？।…][\"'”’)\]]*\s+")
_WHITESPACE = re.compile(r"\s+")


def character_limit(characters: int, tokens: int, budget_tokens: int) -> int:
    """How many characters of *this* text fit one inference's token budget.

    The ratio is measured from the transcript in hand instead of assumed.
    English dictation runs about five characters to the token, Devanagari
    close to one, and a script the tokenizer has no entries for costs more
    than one token per character - so a single fixed number is either wrong
    for most languages or four times too small for the common one.
    """
    if characters <= 0 or tokens <= 0:
        return max(1, budget_tokens)
    return max(1, int(budget_tokens * TOKEN_ESTIMATE_MARGIN * characters / tokens))


def pack(text: str, *, limit: int) -> list[str]:
    """Split *text* into pieces of at most *limit* characters.

    A single whitespace-delimited token longer than *limit* is left whole so a
    URL or path is not broken; the runtime then either accepts it or keeps that
    slice uncorrected. ``"".join(pack(text)) == text``.
    """
    if limit < 1:
        raise ValueError("pack limit must be at least 1")
    if len(text) <= limit:
        return [text] if text else []
    packed: list[str] = []
    current = ""
    for unit in _units(text, limit):
        if not current:
            current = unit
            continue
        if len(current) + len(unit) <= limit:
            current += unit
            continue
        packed.append(current)
        current = unit
    if current:
        packed.append(current)
    return packed


def _units(text: str, limit: int) -> list[str]:
    paragraphs = _split_after(text, _PARAGRAPH)
    sentences = _break_oversized(paragraphs, limit, lambda part: _split_after(part, _SENTENCE))
    return _break_oversized(sentences, limit, lambda part: _split_words(part, limit))


def _break_oversized(
    units: list[str], limit: int, splitter: Callable[[str], list[str]]
) -> list[str]:
    broken: list[str] = []
    for unit in units:
        if len(unit) <= limit:
            broken.append(unit)
            continue
        broken.extend(splitter(unit))
    return broken


def _split_after(text: str, pattern: re.Pattern[str]) -> list[str]:
    pieces: list[str] = []
    start = 0
    for match in pattern.finditer(text):
        end = match.end()
        if end > start:
            pieces.append(text[start:end])
            start = end
    if start < len(text):
        pieces.append(text[start:])
    return pieces or [text]


def _split_words(text: str, limit: int) -> list[str]:
    """Split on whitespace, keeping an oversize token as its own piece."""
    pieces: list[str] = []
    start = 0
    for match in _WHITESPACE.finditer(text):
        word = text[start : match.start()]
        space = match.group(0)
        start = match.end()
        if word:
            pieces.append(word)
        if space:
            if pieces and len(pieces[-1]) + len(space) <= limit:
                pieces[-1] += space
            else:
                pieces.append(space)
    if start < len(text):
        pieces.append(text[start:])
    return pieces or [text]
