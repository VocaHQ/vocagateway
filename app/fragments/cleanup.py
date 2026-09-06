from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

from app.cleanup.base import MAXIMUM_TIMEOUT_SECONDS, MINIMUM_TIMEOUT_SECONDS
from app.runtime_config import CLEANUP_IDLE_UNLOAD_MINUTES
from app.schemas import CleanupConfigResponse, CleanupModelEntry
from app.templating import render

# Deliberately short and specific. An operator deciding whether to turn this on
# needs to know what it does, where it runs, and that it is not perfect — in
# that order, in one sentence each.
CLEANUP_SUMMARY = (
    "Fix grammar and punctuation while keeping your meaning. Runs on your "
    "gateway. May make mistakes."
)
CLEANUP_CAVEAT = (
    "It only sees the recognised text, never the audio, so a word the speech "
    "model heard wrongly can still read as a sentence and will be left alone."
)
MUTED_TONE = "muted"
TIMEOUT_CHOICES = (2.0, 3.0, 5.0, 8.0, 12.0)
STATE_LABELS: MappingProxyType[str, tuple[str, str]] = MappingProxyType(
    {
        "disabled": ("Off", "muted"),
        "unavailable": ("Not available", "warn"),
        "loading": ("Loading", "warn"),
        "ready": ("Ready", "ok"),
        "offloaded": ("Offloaded", "muted"),
        "error": ("Error", "error"),
    }
)
# What the state means for the transcripts going through right now, in one
# sentence. The pill says what the runtime is doing; this says what an operator
# actually gets, which is not always the same thing.
STATE_SENTENCES: MappingProxyType[str, str] = MappingProxyType(
    {
        "disabled": "Cleanup is disabled. Your usual writing style still applies.",
        "unavailable": "Nothing is being corrected: there is no usable model on this host yet.",
        "loading": "The model is loading. Dictations are returned uncorrected until it is ready.",
        "ready": "The model is ready for supported languages. Raw text bypasses cleanup.",
        "offloaded": "Ready, but unloaded to save memory. The next correction loads it again.",
        "error": "The cleanup runtime failed to start, so transcripts are returned uncorrected.",
    }
)
# The example the try-it box offers. Deliberately a sentence a speech model
# really does produce — no capitals, no punctuation, a contraction with the
# apostrophe dropped — so what comes back is a fair demonstration rather than a
# rigged one.
SAMPLE_TEXT = (
    "so i told the team we cant ship on friday because the api isnt ready "
    "and we still need to review the migration"
)


def cleanup_page(
    config: CleanupConfigResponse,
    models: list[CleanupModelEntry],
    message: str = "",
) -> str:
    """The whole section: status, a live demonstration, settings, and a library.

    See app/templates/cleanup/*.html for the markup.
    """
    return render(
        "cleanup/page.html",
        config=config,
        models=models,
        status_html=cleanup_status(config, message),
        settings_html=cleanup_settings(config),
    )


def cleanup_pill(config: CleanupConfigResponse) -> str:
    label, tone = STATE_LABELS.get(config.state, ("Unknown", MUTED_TONE))
    return render("cleanup/pill.html", config=config, state_label=label, state_tone=tone)


def cleanup_library(models: list[CleanupModelEntry]) -> str:
    return render("cleanup/library.html", models=models)


def cleanup_status(
    config: CleanupConfigResponse, message: str = "", *, out_of_band: bool = False
) -> str:
    """The one-glance answer, plus the steps still standing between here and it."""
    label, tone = STATE_LABELS.get(config.state, ("Unknown", MUTED_TONE))
    return render(
        "cleanup/status_card.html",
        config=config,
        message=message,
        out_of_band=out_of_band,
        state_label=label,
        state_tone=tone,
        state_sentence=STATE_SENTENCES.get(config.state, ""),
        steps=setup_steps(config),
    )


def cleanup_settings(config: CleanupConfigResponse) -> str:
    return render(
        "cleanup/settings_card.html",
        config=config,
        mode_options=[("conservative", "Conservative (recommended)"), ("off", "Off")],
        timeout_options=[
            (choice, _timeout_label(choice))
            for choice in TIMEOUT_CHOICES
            if MINIMUM_TIMEOUT_SECONDS <= choice <= MAXIMUM_TIMEOUT_SECONDS
        ],
        idle_options=[(minutes, _idle_label(minutes)) for minutes in CLEANUP_IDLE_UNLOAD_MINUTES],
        auto_language_options=_auto_language_options(config.languages),
    )


@dataclass(frozen=True, slots=True)
class SetupStep:
    """One thing that has to be true before a transcript gets corrected."""

    title: str
    detail: str
    done: bool


def setup_steps(config: CleanupConfigResponse) -> list[SetupStep]:
    """Every precondition, shown together and in order.

    Four separate things have to line up, and when one is missing the symptom is
    identical to the other three: the transcript comes back unchanged. Listing
    them turns "it does nothing" into "step three is not done".
    """
    return [
        SetupStep(
            "A runtime to run it",
            "llama-server found on this host"
            if config.runtime_available
            else "No llama-server found. Install llama.cpp, or set VOCAGATEWAY_CLEANUP_BINARY",
            config.runtime_available,
        ),
        SetupStep(
            "A model downloaded",
            f"Using {config.model_label}"
            if config.model_installed and config.model_label
            else "Pick one from the library below and download it",
            config.model_installed,
        ),
        SetupStep(
            "Corrections turned on",
            "On by default for clients that do not ask for something else"
            if config.enabled and config.mode == "conservative"
            else "Enable cleanup and choose Conservative in Settings below",
            config.enabled and config.mode == "conservative",
        ),
        SetupStep(
            "A language to correct in",
            f"Clients sending 'detect language' are corrected as {config.auto_language}"
            if config.auto_language
            else "Clients that send 'detect language' come back uncorrected. Set one below",
            bool(config.auto_language),
        ),
    ]


def _auto_language_options(languages: list[str]) -> list[tuple[str, str]]:
    """The choices for a transcript whose language is `auto`.

    "Do not guess" leads and is the default, because it is the honest answer
    when nothing knows the language: the speech engines do not report a
    detected one, and Latin script does not name one.
    """
    return [
        ("", "Do not guess (leave uncorrected)"),
        *(
            (
                code,
                {"en": "English", "hi": "Hindi", "hinglish_roman": "Hinglish (Roman)"}.get(
                    code, code
                ),
            )
            for code in languages
        ),
    ]


def _timeout_label(seconds: float) -> str:
    return f"{seconds:.0f} seconds"


def _idle_label(minutes: int) -> str:
    if minutes < 60:
        return f"{minutes} minutes"
    hours = minutes // 60
    return f"{hours} hour" if hours == 1 else f"{hours} hours"
