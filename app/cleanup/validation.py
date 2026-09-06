"""Risk-reduction checks between a cleanup candidate and the original transcript.

None of this proves the two mean the same thing — no text-only check can. What
it does is make the failure modes that matter cheap to catch: a changed number,
a dropped negation, a rewritten URL, a translated sentence, a leaked reasoning
block, or an edit so large it stopped being a correction. Anything a check
cannot vouch for is rejected, and the caller returns the deterministic result
it computed independently.

Rejections are deliberately cheap. Returning the original text unchanged is a
success; inserting a plausible-looking rewrite at someone's cursor is not.
"""

from __future__ import annotations

import difflib
import math
import re
import unicodedata
from types import MappingProxyType

import tldextract

from app import scripts
from app.cleanup.base import MAXIMUM_INPUT_BYTES, CleanupReason

# Bounds. Calibrated to be permissive enough for real grammar repair (which can
# legitimately reshape a clause) and tight enough that a rewrite, a translation,
# or a hallucinated continuation lands outside them. Held-out calibration is
# part of the model-selection evaluation, not of this module.
MINIMUM_SIMILARITY = 0.72
# The edit budget, as a share of the original's length. Expressed as a count
# rather than a ratio because a ratio is brutal on short text: turning "i have
# not seen it" into "I haven't seen it" changes two of five words, which is a
# perfectly ordinary contraction fix and a similarity of 0.67.
MAXIMUM_EDIT_SHARE = 0.28
MINIMUM_ALLOWED_EDITS = 1
MINIMUM_LENGTH_RATIO = 0.7
MAXIMUM_LENGTH_RATIO = 1.6
SHORT_TEXT_SLACK_CHARACTERS = 48
# Beyond this many comparison units the quadratic matcher is skipped for its
# linear upper bound. A long dictation is exactly where the exact ratio costs
# the most and matters the least, because a wholesale rewrite is still far
# outside the bound.
EXACT_MATCH_UNIT_LIMIT = 3_000

# Scripts written without spaces, where whitespace tokens are not a unit of
# comparison and characters are.
_UNSPACED_SCRIPTS = frozenset(("CJK", "HIRAGANA", "KATAKANA", "THAI", "LAO", "MYANMAR", "KHMER"))
_UNSPACED_LANGUAGES = frozenset(("ja", "zh", "yue", "th", "lo", "my", "km", "bo"))

# Text a model emits when it stopped answering and started narrating. Each is
# only a rejection when the original did not contain it: a speaker may well
# dictate "here is the thing", and that is content, not a wrapper.
LEAK_MARKERS = (
    "<think>",
    "</think>",
    "<|",
    "|>",
    "```",
    "[/INST]",
    "<s>",
    "assistant:",
    "here is the corrected",
    "here's the corrected",
    "corrected transcript:",
    "as an ai",
)
# Refusals, matched on an apostrophe-stripped form so `can't` and `cant` are the
# same word here as they are for polarity. Spelled out as whole phrases rather
# than a bare "i cannot", which rejected the ordinary contraction repair this
# feature exists for: a dictated "i cant come" corrected to "I cannot come" is
# not a model declining to answer.
REFUSAL_MARKERS = tuple(
    f"i {verb} {action}"
    for verb in ("cannot", "cant", "wont", "am not able to", "am unable to")
    for action in ("help", "assist", "provide", "do that", "comply", "fulfill", "fulfil")
)

_SUFFIXES = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None)

_TRAILING = r"[^\s.,;:!?'\"“”)\]]"
_SCHEME_URL = rf"[A-Za-z][A-Za-z0-9+.\-]*://\S*{_TRAILING}"
_EMAIL = r"[\w.+\-]+@(?:[\w\-]+\.)+[A-Za-z]{2,}"
_DOMAIN = rf"(?:[\w\-]+\.)+[A-Za-z]{{2,24}}(?:/\S*{_TRAILING})?"
_FILE_PATH = r"(?:~|\.{1,2})?/[\w.\-]+(?:/[\w.\-]+)+"
_SNAKE_CASE = r"\w+(?:_\w+)+"
_NAMESPACED = r"\w+(?:::\w+)+"

_PROTECTED = re.compile(
    rf"""(
        (?P<url>{_SCHEME_URL})
      | (?P<email>{_EMAIL})
      | (?P<path>{_FILE_PATH})
      | (?P<snake>{_SNAKE_CASE})
      | (?P<namespaced>{_NAMESPACED})
      | (?P<domain>{_DOMAIN})
    )""",
    re.VERBOSE,
)
# Every run of digits, with any internal separators a written number keeps.
# Compared as an ordered list, so "16 files" turning into "60 files" is caught
# even though nothing else about the sentence moved.
_DIGIT_RUN = re.compile(r"\d+(?:[.,:/٫٬]\d+)*")
_WORD = re.compile(r"[^\W_]+", re.UNICODE)

# Names a correction pass has no business changing, and which nothing else here
# would catch: they are ordinary words, not digits or protected spans, and
# swapping one for another reads as fluent English. Compared case-insensitively,
# because capitalising a dictated "friday" is exactly the kind of fix this
# feature exists for. A wider entity check belongs with the evaluation corpus.
_CALENDAR_NAMES = (
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
    "today",
    "tomorrow",
    "yesterday",
)

# Apostrophes are removed before polarity is counted, which is what lets a
# transcript's `cant` and a correction's `can't` be recognised as the same word.
# Without it, adding the apostrophe a speech model dropped would read as adding
# a negation — and, far worse, `can't` becoming `can` would read as no change
# at all, because a leading word boundary can never match inside `don't`.
_APOSTROPHES = str.maketrans("", "", "'\u2019\u2018`\u00b4")

# Words that carry polarity. Counted, not parsed: an equal count is weak
# evidence, but an unequal one is strong evidence that something flipped.
# Contracted forms are listed apostrophe-free, to match the normalisation above.
_NEGATIONS: MappingProxyType[str, tuple[str, ...]] = MappingProxyType(
    {
        "latin": (
            "not",
            "no",
            "none",
            "never",
            "nobody",
            "nothing",
            "nowhere",
            "neither",
            "nor",
            "cannot",
            "without",
            "unable",
            "dont",
            "doesnt",
            "didnt",
            "cant",
            "wont",
            "wouldnt",
            "shouldnt",
            "couldnt",
            "isnt",
            "arent",
            "wasnt",
            "werent",
            "hasnt",
            "havent",
            "hadnt",
            "aint",
            "mustnt",
            "neednt",
            "shant",
        ),
        "devanagari": (
            "नहीं",
            "नही",
            "ना",
            "न",
            "मत",
            "बिना",
            "कभी",
            "कोई",
        ),
    }
)


def _negation_pattern(tokens: tuple[str, ...]) -> re.Pattern[str]:
    alternatives = "|".join(re.escape(token) for token in tokens)
    return re.compile(rf"(?<!\w)(?:{alternatives})(?!\w)", re.IGNORECASE)


_NEGATION_PATTERNS = tuple(_negation_pattern(tokens) for tokens in _NEGATIONS.values())
_CALENDAR_SET = frozenset(_CALENDAR_NAMES)


def _casefolded_words(text: str) -> list[str]:
    return [match.group(0).casefold() for match in _WORD.finditer(text)]


class SpanGuard:
    """Extracts the substrings a correction pass is never allowed to touch."""

    @classmethod
    def spans(cls, text: str) -> list[str]:
        found: list[str] = []
        for match in _PROTECTED.finditer(text):
            span = match.group(0)
            if match.lastgroup == "domain" and not cls._is_real_domain(span):
                continue
            found.append(span)
        return found

    @classmethod
    def digits(cls, text: str) -> list[str]:
        return [match.group(0) for match in _DIGIT_RUN.finditer(text)]

    @classmethod
    def calendar_names(cls, text: str) -> list[str]:
        return [word for word in _casefolded_words(text) if word in _CALENDAR_SET]

    @classmethod
    def negations(cls, text: str) -> int:
        """How many polarity words the text carries, apostrophes ignored."""
        normalized = text.translate(_APOSTROPHES)
        return sum(len(pattern.findall(normalized)) for pattern in _NEGATION_PATTERNS)

    @classmethod
    def _is_real_domain(cls, candidate: str) -> bool:
        host = candidate.split("/", 1)[0]
        suffix = host.rsplit(".", 1)[-1]
        if not suffix.islower():
            return False
        return bool(_SUFFIXES(host).suffix)


def exceeds_input_ceiling(text: str) -> bool:
    """Whether a transcript is too long to correct in one bounded pass.

    Measured in encoded bytes, which is what the ceiling is written in. The real
    limit is the runtime's own token count, checked later against its tokenizer;
    this is the cheap guard that keeps an enormous transcript from being sent at
    all. Over the ceiling the answer is the untouched ASR result — never a
    truncated one, and never a transcript split and stitched back together.
    """
    return len(text.encode("utf-8")) > MAXIMUM_INPUT_BYTES


def comparison_units(text: str, language: str) -> list[str]:
    """Split into the units an edit should be measured in for this language.

    Words for languages written with spaces; characters for the ones that are
    not, where a whitespace split would report one enormous token and hide every
    edit inside it.
    """
    if _is_unspaced(text, language):
        return [character for character in text if not character.isspace()]
    return [match.group(0).casefold() for match in _WORD.finditer(text)]


def _is_unspaced(text: str, language: str) -> bool:
    code = language.lower().split("-", maxsplit=1)[0]
    if code in _UNSPACED_LANGUAGES:
        return True
    if code not in {"auto", ""}:
        return False
    letters = [character for character in text if character.isalpha()]
    if not letters:
        return False
    unspaced = sum(1 for character in letters if _script_of(character) in _UNSPACED_SCRIPTS)
    return unspaced * 2 > len(letters)


def _script_of(character: str) -> str:
    try:
        return unicodedata.name(character).split(" ", maxsplit=1)[0]
    except ValueError:
        return ""


def excessive_edit(original: str, candidate: str, language: str) -> bool:
    """Whether more units changed than a correction is allowed to change.

    Counted in units rather than scored as a ratio, and with a floor, so a short
    transcript is not held to an impossible standard while a wholesale rewrite is
    still far outside the budget. The share is a starting point: calibrating it
    against held-out examples is part of model selection, not of this module.
    """
    left = comparison_units(original, language)
    right = comparison_units(candidate, language)
    if not left and not right:
        return False
    if max(len(left), len(right)) > EXACT_MATCH_UNIT_LIMIT:
        # The exact matcher is quadratic, and a very long dictation is where it
        # costs most and decides least: a rewrite is nowhere near the bound.
        return similarity(original, candidate, language) < MINIMUM_SIMILARITY
    matcher = difflib.SequenceMatcher(None, left, right, autojunk=False)
    matched = sum(block.size for block in matcher.get_matching_blocks())
    changed = max(len(left), len(right)) - matched
    allowed = max(MINIMUM_ALLOWED_EDITS, math.ceil(len(left) * MAXIMUM_EDIT_SHARE))
    return changed > allowed


def similarity(original: str, candidate: str, language: str) -> float:
    """How much of the original survives, in language-appropriate units."""
    left = comparison_units(original, language)
    right = comparison_units(candidate, language)
    if not left and not right:
        return 1.0
    matcher = difflib.SequenceMatcher(None, left, right, autojunk=False)
    if max(len(left), len(right)) > EXACT_MATCH_UNIT_LIMIT:
        return matcher.quick_ratio()
    return matcher.ratio()


class OutputChecks:
    """Structural checks on the raw string a runtime handed back."""

    @classmethod
    def reject(cls, original: str, candidate: str) -> CleanupReason | None:
        if not candidate.strip():
            return CleanupReason.INVALID_OUTPUT
        if cls._leaked(original, candidate):
            return CleanupReason.INVALID_OUTPUT
        if "�" in candidate and "�" not in original:
            return CleanupReason.INVALID_OUTPUT
        return None

    @classmethod
    def _leaked(cls, original: str, candidate: str) -> bool:
        if cls._appeared(original.casefold(), candidate.casefold(), LEAK_MARKERS):
            return True
        return cls._appeared(
            original.casefold().translate(_APOSTROPHES),
            candidate.casefold().translate(_APOSTROPHES),
            REFUSAL_MARKERS,
        )

    @classmethod
    def _appeared(cls, original: str, candidate: str, markers: tuple[str, ...]) -> bool:
        """Whether a marker is in the answer and was not in the transcript.

        A speaker may well dictate any of these, and that is content to
        preserve. Only a marker the model added on its own is a rejection.
        """
        return any(marker in candidate and marker not in original for marker in markers)


class PreservationChecks:
    """Semantic-risk checks between the original transcript and a candidate."""

    @classmethod
    def reject(cls, original: str, candidate: str, language: str) -> CleanupReason | None:
        for check in (cls._spans, cls._digits, cls._negation, cls._calendar, cls._script):
            rejection = check(original, candidate, language)
            if rejection is not None:
                return rejection
        return cls._size(original, candidate, language)

    @classmethod
    def _spans(cls, original: str, candidate: str, _language: str) -> CleanupReason | None:
        # Exactly, and in order: a URL that survived but moved past a number is
        # no longer the sentence that was dictated.
        if SpanGuard.spans(original) != SpanGuard.spans(candidate):
            return CleanupReason.UNSAFE_EDIT
        return None

    @classmethod
    def _digits(cls, original: str, candidate: str, _language: str) -> CleanupReason | None:
        if SpanGuard.digits(original) != SpanGuard.digits(candidate):
            return CleanupReason.UNSAFE_EDIT
        return None

    @classmethod
    def _negation(cls, original: str, candidate: str, _language: str) -> CleanupReason | None:
        if SpanGuard.negations(original) != SpanGuard.negations(candidate):
            return CleanupReason.UNSAFE_EDIT
        return None

    @classmethod
    def _calendar(cls, original: str, candidate: str, _language: str) -> CleanupReason | None:
        if SpanGuard.calendar_names(original) != SpanGuard.calendar_names(candidate):
            return CleanupReason.UNSAFE_EDIT
        return None

    @classmethod
    def _script(cls, original: str, candidate: str, language: str) -> CleanupReason | None:
        # Script equality cannot separate two languages sharing an alphabet, so
        # this catches transliteration and outright translation into another
        # writing system — not a French answer to English dictation.
        if not scripts.transcript_matches_language(candidate, language):
            return CleanupReason.UNSAFE_EDIT
        if _script_profile(original) != _script_profile(candidate):
            return CleanupReason.UNSAFE_EDIT
        return None

    @classmethod
    def _size(cls, original: str, candidate: str, language: str) -> CleanupReason | None:
        stripped = original.strip()
        slack = SHORT_TEXT_SLACK_CHARACTERS
        lower_bound = len(stripped) * MINIMUM_LENGTH_RATIO - slack
        upper_bound = len(stripped) * MAXIMUM_LENGTH_RATIO + slack
        if not lower_bound <= len(candidate.strip()) <= upper_bound:
            return CleanupReason.UNSAFE_EDIT
        if excessive_edit(stripped, candidate.strip(), language):
            return CleanupReason.UNSAFE_EDIT
        return None


def _script_profile(text: str) -> frozenset[str]:
    """The set of writing systems present, so a new one cannot appear silently."""
    return frozenset(
        _script_of(character) for character in text if character.isalpha()
    ) - frozenset(("",))


def rejection(original: str, candidate: str, language: str) -> CleanupReason | None:
    """The single reason to refuse this candidate, or None to accept it."""
    structural = OutputChecks.reject(original, candidate)
    if structural is not None:
        return structural
    return PreservationChecks.reject(original, candidate, language)
