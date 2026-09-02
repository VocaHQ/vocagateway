from __future__ import annotations

import re

import pytest

from app.text_styles import (
    _PUNCTUATION_BY_LANGUAGE,
    SUPPORTED_WRITING_STYLES,
    apply_writing_style,
)

VERY_CASUAL_STYLE = "very_casual"
CLEAN_STYLE = "clean"
FORMAL_STYLE = "formal"
CASUAL_STYLE = "casual"
EXCITED_STYLE = "excited"
JAPANESE_LANGUAGE = "ja"
HINDI_LANGUAGE = "hi"
THAI_LANGUAGE = "th"

SAMPLES = [
    "I don't think it's ready yet.",
    "Meet me at 3:30, it's about 2.5 miles away.",
    "Email me at john@example.com or see example.com/docs.",
    "Wow. That went really well. I'm so happy.",
    "It cost $1,200 and that's a lot.",
    "I went home. John called me later.",
    "Check https://example.com/a/b?q=1 for details.",
    "Wait... I'm not sure.",
    "Can you send it? I'll wait.",
    "We shipped on the 3rd, e.g. before the U.S. holiday.",
]


def _words(text: str) -> list[str]:
    return re.findall(r"[^\W_]+", text.lower())


@pytest.mark.parametrize("style", sorted(SUPPORTED_WRITING_STYLES))
@pytest.mark.parametrize("sample", SAMPLES)
def test_styles_never_change_the_words(sample: str, style: str) -> None:
    """The documented guarantee: styling is presentation only."""
    assert _words(apply_writing_style(sample, style)) == _words(sample)


@pytest.mark.parametrize("style", sorted(SUPPORTED_WRITING_STYLES))
@pytest.mark.parametrize(
    "fragment",
    [
        "$1,200",
        "2.5",
        "3:30",
        "john@example.com",
        "https://example.com/a/b?q=1",
        "don't",
        "it's",
        "I'll",
    ],
)
def test_meaningful_punctuation_survives_ev_cb4fe(fragment: str, style: str) -> None:
    """Punctuation inside prices, times, decimals, addresses and contractions
    carries meaning. Removing it silently corrupted the transcript."""
    styled = apply_writing_style(f"Here is {fragment} for you.", style)
    expected = fragment.lower() if style == VERY_CASUAL_STYLE else fragment
    assert expected in styled


def test_raw_returns_the_model_output_untouched() -> None:
    assert apply_writing_style("  hello   there  ", "raw") == "hello   there"


def test_clean_normalizes_spacing_without_a_aa() -> None:
    assert apply_writing_style("hello   there ,ok", CLEAN_STYLE) == "hello there,ok."


def test_formal_capitalizes_and_terminates() -> None:
    assert apply_writing_style("hello there. how are you", FORMAL_STYLE) == (
        "Hello there. How are you."
    )


def test_casual_drops_only_the_closing_full_stop() -> None:
    assert apply_writing_style("Hello there. It is fine.", CASUAL_STYLE) == (
        "Hello there. It is fine"
    )
    # A question mark carries meaning and is not a decoration to strip.
    assert apply_writing_style("Are we ready?", CASUAL_STYLE) == "Are we ready?"


def test_casual_never_demotes_a_name() -> None:
    """Casual keeps sentence structure, so there is no lowercasing decision
    to get wrong and no English stop-word list to maintain."""
    assert apply_writing_style("I went home. John called.", CASUAL_STYLE) == (
        "I went home. John called"
    )


def test_very_casual_lowercases_prose_but_n_ef338() -> None:
    styled = apply_writing_style("Email John@Example.com now. Thanks.", VERY_CASUAL_STYLE)
    assert "John@Example.com" in styled
    assert styled.startswith("email")


def test_excited_exclaims_every_statement_a_aaa() -> None:
    assert apply_writing_style("Wow. That worked. I'm happy.", EXCITED_STYLE) == (
        "Wow! That worked! I'm happy!"
    )
    assert apply_writing_style("Can you send it? I'll wait.", EXCITED_STYLE) == (
        "Can you send it? I'll wait!"
    )


@pytest.mark.parametrize("style", sorted(SUPPORTED_WRITING_STYLES))
def test_an_ellipsis_is_never_mistaken_for_aaaa(style: str) -> None:
    styled = apply_writing_style("Wait... I am not sure.", style)
    assert "..." in styled


def test_a_trailing_address_does_not_gain_a_adc01() -> None:
    assert apply_writing_style("See example.com/docs.", FORMAL_STYLE) == ("See example.com/docs.")


def test_empty_input_stays_empty() -> None:
    for style in SUPPORTED_WRITING_STYLES:
        assert apply_writing_style("", style) == ""


def test_unsupported_style_is_rejected() -> None:
    with pytest.raises(ValueError):
        apply_writing_style("hello", "shouty")


# ---------------------------------------------------------------------------
# Languages other than English
#
# Every rule used to assume a Latin full stop and comma, so Japanese came back
# unstyled with a stray "!" appended and Arabic gained Latin commas.
# ---------------------------------------------------------------------------

JAPANESE = "家に帰りました。ジョンが電話してきました。"
ARABIC = "ذهبت إلى المنزل. اتصل بي جون لاحقاً."
KOREAN = "집에 갔어요. 존이 나중에 전화했어요."


def test_japanese_uses_its_own_sentence_marks() -> None:
    assert apply_writing_style(JAPANESE, EXCITED_STYLE, JAPANESE_LANGUAGE) == (
        "家に帰りました！ジョンが電話してきました！"
    )
    assert apply_writing_style(JAPANESE, CASUAL_STYLE, JAPANESE_LANGUAGE) == (
        "家に帰りました。ジョンが電話してきました"
    )
    # No space is inserted between CJK sentences.
    assert " " not in apply_writing_style(JAPANESE, VERY_CASUAL_STYLE, JAPANESE_LANGUAGE)


def test_arabic_uses_an_arabic_clause_separator() -> None:
    styled = apply_writing_style(ARABIC, VERY_CASUAL_STYLE, "ar")
    assert "،" in styled
    assert "," not in styled


HINDI = "मैं घर गया। जॉन ने बाद में फोन किया।"


def test_hindi_is_terminated_with_a_danda_n_aaaaa() -> None:
    """Indic scripts end a sentence with "।". Appending "." both looked wrong and,
    because the danda was not recognised as a terminator, produced a second one on
    text the model had already punctuated."""
    assert apply_writing_style(HINDI, CLEAN_STYLE, HINDI_LANGUAGE) == HINDI
    assert (
        apply_writing_style("मैं कल बाजार जाऊंगा", FORMAL_STYLE, HINDI_LANGUAGE) == "मैं कल बाजार जाऊंगा।"
    )
    # Casual drops the closing danda the way it drops a closing full stop.
    assert (
        apply_writing_style(HINDI, CASUAL_STYLE, HINDI_LANGUAGE) == "मैं घर गया। जॉन ने बाद में फोन किया"
    )


def test_hindi_sentences_are_actually_segmented() -> None:
    """Without the danda in `terminators` the whole transcript read as one
    sentence, so very_casual and excited silently did nothing."""
    assert (
        apply_writing_style(HINDI, EXCITED_STYLE, HINDI_LANGUAGE)
        == "मैं घर गया! जॉन ने बाद में फोन किया!"
    )
    assert (
        apply_writing_style(HINDI, VERY_CASUAL_STYLE, HINDI_LANGUAGE)
        == "मैं घर गया, जॉन ने बाद में फोन किया"
    )


def test_a_danda_is_detected_without_an_exp_a() -> None:
    """Several models detect the language themselves, so Hindi commonly arrives
    with the request language still set to Automatic."""
    assert apply_writing_style(HINDI, EXCITED_STYLE, "auto") == "मैं घर गया! जॉन ने बाद में फोन किया!"


def test_bengali_uses_the_danda_but_tamil_u_d9a86() -> None:
    """Not every Indic script writes the danda: the Dravidian ones use "." in
    modern usage, while still needing to recognise a danda a model may emit."""
    bengali = "আমি ভালো আছি। তুমি কেমন আছো"
    assert apply_writing_style(bengali, CLEAN_STYLE, "bn") == "আমি ভালো আছি। তুমি কেমন আছো।"
    tamil = "நான் நன்றாக இருக்கிறேன்"
    assert apply_writing_style(tamil, CLEAN_STYLE, "ta") == "நான் நன்றாக இருக்கிறேன்."


def test_thai_and_lao_end_sentences_with_no_aa() -> None:
    """These scripts have no sentence-ending mark, so no style may add one — and
    dropping a zero-length terminator must not truncate the transcript, since
    `"x".endswith("")` is True and `"x"[:-0]` is the empty string."""
    thai = "ผมสบายดี"
    for style in (CLEAN_STYLE, FORMAL_STYLE, CASUAL_STYLE, VERY_CASUAL_STYLE, EXCITED_STYLE):
        styled = apply_writing_style(thai, style, THAI_LANGUAGE)
        assert thai in styled, f"{style} lost the Thai transcript: {styled!r}"
    assert apply_writing_style(thai, CASUAL_STYLE, THAI_LANGUAGE) == thai
    assert apply_writing_style(thai, CLEAN_STYLE, THAI_LANGUAGE) == thai
    assert apply_writing_style("ຂ້ອຍສະບາຍດີ", CASUAL_STYLE, "lo") == "ຂ້ອຍສະບາຍດີ"


def test_scripts_with_their_own_sentence_marks() -> None:
    assert apply_writing_style("ကျွန်တော် အိမ်ပြန်သွားတယ်", CLEAN_STYLE, "my").endswith("။")
    assert apply_writing_style("ខ្ញុំសុខសប្បាយជាទេ", CLEAN_STYLE, "km").endswith("។")
    assert apply_writing_style("ང་བདེ་པོ་ཡིན", CLEAN_STYLE, "bo").endswith("།")
    # Perso-Arabic Indic languages follow Urdu's "۔", not the Arabic full stop.
    assert apply_writing_style("بہٕ چھُس ٹھیک", CLEAN_STYLE, "ks").endswith("۔")
    assert apply_writing_style("مان ٺيڪ آهيان", CLEAN_STYLE, "sd").endswith("۔")


def test_no_style_ever_erases_a_transcript_aaa() -> None:
    """The broad invariant behind the per-language tables: whatever the language,
    styling is presentation only and can never leave the user with nothing."""
    samples = ["hello", "मैं ठीक हूँ", "私は元気です", "ผมสบายดี", "مان ٺيڪ آهيان", "hi।", "ok..."]
    languages = sorted(set(_PUNCTUATION_BY_LANGUAGE) | {"en", "auto", "ko", THAI_LANGUAGE})
    for language in languages:
        for style in SUPPORTED_WRITING_STYLES:
            for sample in samples:
                styled = apply_writing_style(sample, style, language)
                assert styled.strip(), f"{language}/{style} erased {sample!r}"


def test_a_language_pysbd_lacks_still_gets_styled() -> None:
    """Korean has no pysbd profile, so it falls back to terminator splitting."""
    assert apply_writing_style(KOREAN, EXCITED_STYLE, "ko") == (
        "집에 갔어요! 존이 나중에 전화했어요!"
    )


def test_german_abbreviations_come_from_the_aaaa() -> None:
    styled = apply_writing_style(
        "Ich ging nach Hause. Herr z.B. Schmidt rief an.", EXCITED_STYLE, "de"
    )
    assert styled == "Ich ging nach Hause! Herr z.B. Schmidt rief an!"


def test_auto_infers_the_script_from_the_tr_aaaaa() -> None:
    """ "auto" is the default, so the script has to be inferred from the text."""
    assert apply_writing_style(JAPANESE, EXCITED_STYLE, "auto") == (
        "家に帰りました！ジョンが電話してきました！"
    )


def test_a_country_code_word_is_not_mistake_a() -> None:
    """ ".it" is a real suffix, so "home.It" must not be read as an address."""
    styled = apply_writing_style("I went home.It was fine.", FORMAL_STYLE, "en")
    assert styled == "I went home.It was fine."


def test_a_foreign_terminator_is_recognised_aa() -> None:
    """A model that picks its own language leaks that language's punctuation:
    Dolphin ends a Hindi sentence with the CJK "。". Appending a danda to text
    that already ended produced "。।" at the cursor."""
    leaked = "आज मैं ऑफिस जा रहा हूँ。"
    assert apply_writing_style(leaked, CLEAN_STYLE, HINDI_LANGUAGE) == leaked
    assert apply_writing_style(leaked, FORMAL_STYLE, HINDI_LANGUAGE) == leaked
    # Styles that rewrite terminators normalise it to the right language's mark.
    assert apply_writing_style(leaked, EXCITED_STYLE, HINDI_LANGUAGE) == "आज मैं ऑफिस जा रहा हूँ!"
    # The reverse direction too: a danda leaking into Japanese.
    assert apply_writing_style("私は元気です।", CLEAN_STYLE, JAPANESE_LANGUAGE) == "私は元気です।"


def test_recognising_foreign_marks_does_not_aaa() -> None:
    for language, text in (
        (HINDI_LANGUAGE, "मैं घर गया। जॉन ने फोन किया।"),
        ("en", "I went home. John called."),
        (JAPANESE_LANGUAGE, "家に帰りました。ジョンが電話してきました。"),
        ("ar", "ذهبت إلى المنزل. اتصل بي جون."),
    ):
        assert apply_writing_style(text, CLEAN_STYLE, language) == text
