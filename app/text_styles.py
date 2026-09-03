from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from types import MappingProxyType

import pysbd
import tldextract
from pysbd.languages import LANGUAGE_CODES

SUPPORTED_WRITING_STYLES = frozenset(("raw", "clean", "formal", "casual", "very_casual", "excited"))

_UNIVERSAL_TERMINATORS = ".!?\u3002\uff01\uff1f\u0964\u06d4\u104b\u17d4\u0f0d\u061f"
EXCLAMATION_MARK = "!"
QUESTION_MARK = "?"


@dataclass(frozen=True)
class Punctuation:
    terminator: str
    separator: str
    exclamation: str
    question: str
    terminators: str
    segmentation_language: str
    join: str

    def __post_init__(self) -> None:
        merged = self.terminators + _UNIVERSAL_TERMINATORS
        object.__setattr__(self, "terminators", "".join(dict.fromkeys(merged)))

    def with_segmentation_language(self, code: str) -> Punctuation:
        if code == self.segmentation_language:
            return self
        return Punctuation(
            self.terminator,
            self.separator,
            self.exclamation,
            self.question,
            self.terminators,
            code,
            self.join,
        )

    def ensure_terminator(self, text: str) -> str:
        if text and text[-1] not in self.terminators:
            return f"{text}{self.terminator}"
        return text


_LATIN = Punctuation(".", ",", EXCLAMATION_MARK, QUESTION_MARK, ".!?", "en", " ")
_CJK = Punctuation("。", "、", "！", "？", "。！？.!?", "ja", "")
_ARABIC = Punctuation(".", "،", EXCLAMATION_MARK, "؟", ".!?؟", "ar", " ")
_URDU = Punctuation("۔", "،", EXCLAMATION_MARK, "؟", "۔.!?؟", "ur", " ")
_DANDA = Punctuation("।", ",", EXCLAMATION_MARK, QUESTION_MARK, "।.!?", "hi", " ")
_INDIC_LATIN = Punctuation(".", ",", EXCLAMATION_MARK, QUESTION_MARK, "।.!?", "en", " ")
_UNTERMINATED = Punctuation("", " ", EXCLAMATION_MARK, QUESTION_MARK, "!?", "th", " ")
_BURMESE = Punctuation("။", "၊", EXCLAMATION_MARK, QUESTION_MARK, "။.!?", "my", " ")
_KHMER = Punctuation("។", ",", EXCLAMATION_MARK, QUESTION_MARK, "។.!?", "km", " ")
_TIBETAN = Punctuation("།", "།", EXCLAMATION_MARK, QUESTION_MARK, "།.!?", "bo", " ")

_PUNCTUATION_BY_LANGUAGE = MappingProxyType(
    {
        "ja": _CJK,
        "zh": _CJK,
        "yue": _CJK,
        "ar": _ARABIC,
        "fa": _ARABIC,
        "ps": _ARABIC,
        "ur": _URDU,
        "hi": _DANDA,
        "mr": _DANDA.with_segmentation_language("mr"),
        "ne": _DANDA.with_segmentation_language("ne"),
        "sa": _DANDA.with_segmentation_language("sa"),
        "bn": _DANDA.with_segmentation_language("bn"),
        "as": _DANDA.with_segmentation_language("as"),
        "pa": _DANDA.with_segmentation_language("pa"),
        "or": _DANDA.with_segmentation_language("or"),
        "kok": _DANDA.with_segmentation_language("kok"),
        "mai": _DANDA.with_segmentation_language("mai"),
        "brx": _DANDA.with_segmentation_language("brx"),
        "doi": _DANDA.with_segmentation_language("doi"),
        "ta": _INDIC_LATIN.with_segmentation_language("ta"),
        "te": _INDIC_LATIN.with_segmentation_language("te"),
        "kn": _INDIC_LATIN.with_segmentation_language("kn"),
        "ml": _INDIC_LATIN.with_segmentation_language("ml"),
        "gu": _INDIC_LATIN.with_segmentation_language("gu"),
        "si": _INDIC_LATIN.with_segmentation_language("si"),
        "sd": _URDU.with_segmentation_language("sd"),
        "ks": _URDU.with_segmentation_language("ks"),
        "ug": _ARABIC.with_segmentation_language("ug"),
        "th": _UNTERMINATED,
        "lo": _UNTERMINATED.with_segmentation_language("lo"),
        "my": _BURMESE,
        "km": _KHMER,
        "bo": _TIBETAN,
    }
)

_CJK_MARKS = "。、！？"
_ARABIC_MARKS = "،؟"


class _PunctuationRegistry:
    @classmethod
    def resolve(cls, language: str, text: str) -> Punctuation:
        code = language.lower().split("-")[0]
        known = _PUNCTUATION_BY_LANGUAGE.get(code)
        if known is not None:
            return known
        if code not in ("auto", ""):
            return _LATIN.with_segmentation_language(code)
        return cls._infer_from_text(text)

    @classmethod
    def _infer_from_text(cls, text: str) -> Punctuation:
        if any(mark in text for mark in _CJK_MARKS):
            return _CJK
        if any(mark in text for mark in _ARABIC_MARKS):
            return _ARABIC
        if "۔" in text:
            return _URDU
        return _DANDA if "।" in text else _LATIN


_PLACEHOLDER_START = ""
_PLACEHOLDER_END = ""
_PLACEHOLDER = re.compile(rf"{_PLACEHOLDER_START}(\d+){_PLACEHOLDER_END}")

_TRAILING = r"[^\s.,;:!?'\"“”)\]]"
_SCHEME_URL = rf"[A-Za-z][A-Za-z0-9+.\-]*://[^\s]*{_TRAILING}"
_EMAIL = r"[\w.+\-]+@(?:[\w\-]+\.)+[A-Za-z]{2,}"
_DOMAIN = rf"(?:[\w\-]+\.)+[A-Za-z]{{2,24}}(?:/[^\s]*{_TRAILING})?"

_PROTECTED = re.compile(
    rf"""(
        (?P<url>{_SCHEME_URL})
      | (?P<email>{_EMAIL})
      | (?P<domain>{_DOMAIN})
      | (?P<number>\d+(?:[.,:/]\d+)+)
      | (?P<ordinal>\d+(?:st|nd|rd|th)\b)
      | (?P<acronym>(?:[A-Za-z]\.){{2,}})
      | (?P<contraction>\w+['’]\w+)
    )""",
    re.VERBOSE,
)

_ADDRESS = re.compile(rf"^(?:{_SCHEME_URL}|{_EMAIL})$")
_SUFFIXES = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None)


class _SpanProtector:
    @classmethod
    def is_real_domain(cls, candidate: str) -> bool:
        host = candidate.split("/", 1)[0]
        suffix = host.rsplit(".", 1)[-1]
        if not suffix.islower():
            return False
        return bool(_SUFFIXES(host).suffix)

    @classmethod
    def protect(cls, text: str) -> tuple[str, list[str]]:
        tokens: list[str] = []
        masked_text = _PROTECTED.sub(lambda match: cls._capture_match(match, tokens), text)
        return masked_text, tokens

    @classmethod
    def restore(cls, text: str, tokens: list[str]) -> str:
        if not tokens:
            return text
        return _PLACEHOLDER.sub(lambda match: cls._restore_match(match, tokens), text)

    @classmethod
    def lower_outside_placeholders(cls, text: str) -> str:
        pieces: list[str] = []
        position = 0
        for match in _PLACEHOLDER.finditer(text):
            pieces.append(text[position : match.start()].lower())
            pieces.append(match.group(0))
            position = match.end()
        pieces.append(text[position:].lower())
        return "".join(pieces)

    @classmethod
    def sanitize_tokens(cls, tokens: list[str], style: str) -> list[str]:
        if style != "very_casual":
            return tokens
        return [token if _ADDRESS.match(token) else token.lower() for token in tokens]

    @classmethod
    def _capture_match(cls, match: re.Match[str], tokens: list[str]) -> str:
        span = match.group(0)
        if match.lastgroup == "domain" and not cls.is_real_domain(span):
            return span
        tokens.append(span)
        token_index = len(tokens) - 1
        return f"{_PLACEHOLDER_START}{token_index}{_PLACEHOLDER_END}"

    @classmethod
    def _restore_match(cls, match: re.Match[str], tokens: list[str]) -> str:
        token_index = int(match.group(1))
        return tokens[token_index]


_SEGMENTERS = threading.local()


class _SegmenterEngine:
    _break_cache: dict[str, re.Pattern[str]] = {}

    @classmethod
    def internal_break_pattern(cls, punctuation: Punctuation) -> re.Pattern[str]:
        key = f"{punctuation.terminators}|{punctuation.join}"
        cached = cls._break_cache.get(key)
        if cached is None:
            marks = re.escape(punctuation.terminators)
            gap = r"\s*" if punctuation.join == "" else r"\s+"
            cached = re.compile(rf"[{marks}]{gap}\S")
            cls._break_cache[key] = cached
        return cached

    @classmethod
    def segmenter(cls, code: str) -> pysbd.Segmenter:
        cache = getattr(_SEGMENTERS, "cache", None)
        if cache is None:
            cache = {}
            _SEGMENTERS.cache = cache
        cached_seg = cache.get(code)
        if cached_seg is None:
            cached_seg = pysbd.Segmenter(language=code, clean=False)
            cache[code] = cached_seg
        return cached_seg

    @classmethod
    def split_on_terminators(cls, text: str, punctuation: Punctuation) -> list[str]:
        boundary = re.compile(rf"[{re.escape(punctuation.terminators)}]+\s+")
        sentences: list[str] = []
        start = 0
        for match in boundary.finditer(text):
            sentences.append(text[start : match.end()])
            start = match.end()
        if start < len(text):
            sentences.append(text[start:])
        return sentences or [text]

    @classmethod
    def segment(cls, text: str, punctuation: Punctuation) -> list[str]:
        if not cls.internal_break_pattern(punctuation).search(text):
            return [text]
        code = punctuation.segmentation_language
        if code in LANGUAGE_CODES:
            sentences = [str(sentence) for sentence in cls.segmenter(code).segment(text)]
            if sentences and "".join(sentences) == text:
                return sentences
        return cls.split_on_terminators(text, punctuation)


_FIRST_LETTER = re.compile(r"^([\"'“‘(\[]*)([^\W\d_])")


class _SentenceCaser:
    _sentence_start_cache: dict[str, re.Pattern[str]] = {}

    @classmethod
    def capitalize(cls, text: str) -> str:
        return _FIRST_LETTER.sub(cls._capitalize_first_match, text, count=1)

    @classmethod
    def sentence_start_pattern(cls, punctuation: Punctuation) -> re.Pattern[str]:
        cached = cls._sentence_start_cache.get(punctuation.terminators)
        if cached is None:
            marks = re.escape(punctuation.terminators)
            cached = re.compile(rf"(^|[{marks}]\s*)([\"'“‘(\[]*)([^\W\d_])")
            cls._sentence_start_cache[punctuation.terminators] = cached
        return cached

    @classmethod
    def capitalize_sentence_starts(cls, text: str, punctuation: Punctuation) -> str:
        return cls.sentence_start_pattern(punctuation).sub(
            cls._capitalize_match,
            text,
        )

    @classmethod
    def split_terminator(cls, sentence: str, punctuation: Punctuation) -> tuple[str, str]:
        body = sentence.strip()
        has_terminator = bool(body and body[-1] in punctuation.terminators)
        if has_terminator and not body.endswith(".."):
            return body[:-1], body[-1]
        return body, ""

    @classmethod
    def normalize_spacing(cls, text: str, punctuation: Punctuation) -> str:
        normalized_text = re.sub(r"\s+", " ", text).strip()
        marks = re.escape(f"{punctuation.terminators}{punctuation.separator};:")
        return re.sub(rf"\s+([{marks}])", r"\1", normalized_text)

    @classmethod
    def _capitalize_first_match(cls, match: re.Match[str]) -> str:
        prefix = match.group(1)
        letter = match.group(2).upper()
        return f"{prefix}{letter}"

    @classmethod
    def _capitalize_match(cls, match: re.Match[str]) -> str:
        lead, bracket, char = match.groups()
        return f"{lead}{bracket}{char.upper()}"


class _StyleApplier:
    @classmethod
    def formal(cls, text: str, punctuation: Punctuation) -> str:
        capitalized = _SentenceCaser.capitalize_sentence_starts(text, punctuation)
        return punctuation.ensure_terminator(capitalized)

    @classmethod
    def casual(cls, text: str, punctuation: Punctuation) -> str:
        formatted = _SentenceCaser.capitalize_sentence_starts(text, punctuation)
        drop_term = bool(
            punctuation.terminator
            and formatted.endswith(punctuation.terminator)
            and not formatted.endswith("..")
        )
        if drop_term:
            return formatted[: -len(punctuation.terminator)]
        return formatted

    @classmethod
    def very_casual(cls, sentences: list[str], punctuation: Punctuation) -> str:
        last_index = len(sentences) - 1
        parts = [
            cls._casual_segment(sent, punctuation, idx == last_index)
            for idx, sent in enumerate(sentences)
        ]
        return _SpanProtector.lower_outside_placeholders(
            punctuation.join.join(part for part in parts if part)
        )

    @classmethod
    def excited(cls, sentences: list[str], punctuation: Punctuation) -> str:
        parts = [cls._excited_segment(sent, punctuation) for sent in sentences]
        return punctuation.join.join(part for part in parts if part)

    @classmethod
    def format_style(cls, style: str, text: str, punctuation: Punctuation) -> str:
        normalized = _SentenceCaser.normalize_spacing(text, punctuation)
        if style == "clean":
            return punctuation.ensure_terminator(normalized)
        if style == "formal":
            return cls.formal(normalized, punctuation)
        if style == "casual":
            return cls.casual(normalized, punctuation)
        sentences = _SegmenterEngine.segment(normalized, punctuation)
        if style == "very_casual":
            return cls.very_casual(sentences, punctuation)
        return cls.excited(sentences, punctuation)

    @classmethod
    def _casual_segment(cls, sentence: str, punct: Punctuation, is_last: bool) -> str:
        body, term = _SentenceCaser.split_terminator(sentence, punct)
        if not body:
            return ""
        if is_last:
            return body
        if term in ("", punct.terminator):
            return f"{body}{punct.separator}"
        return f"{body}{term}"

    @classmethod
    def _excited_segment(cls, sentence: str, punct: Punctuation) -> str:
        body, term = _SentenceCaser.split_terminator(sentence, punct)
        if not body:
            return ""
        mark = term if term == punct.question else punct.exclamation
        return f"{_SentenceCaser.capitalize(body)}{mark}"


def apply_writing_style(text: str, style: str, language: str = "auto") -> str:
    if style == "raw":
        return text.strip()
    if style not in SUPPORTED_WRITING_STYLES:
        raise ValueError(f"Unsupported writing style: {style}")

    protected = _SpanProtector.protect(text)
    return _SpanProtector.restore(
        _StyleApplier.format_style(
            style, protected[0], _PunctuationRegistry.resolve(language, text)
        ),
        _SpanProtector.sanitize_tokens(protected[1], style),
    )
