from __future__ import annotations

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
TIMEOUT_CHOICES = (2.0, 3.0, 5.0, 8.0, 12.0)
STATE_LABELS: MappingProxyType[str, tuple[str, str]] = MappingProxyType(
    {
        "disabled": ("Off", "muted"),
        "unavailable": ("Not available", "warn"),
        "ready": ("Ready", "ok"),
        "offloaded": ("Offloaded", "muted"),
        "error": ("Error", "error"),
    }
)


def cleanup_card(
    config: CleanupConfigResponse,
    models: list[CleanupModelEntry],
    message: str = "",
) -> str:
    """See app/templates/settings/cleanup_card.html for the markup.

    Both transcripts anywhere in this panel are rendered as escaped text by the
    autoescaping template environment — never as HTML, and never as Markdown a
    browser would execute.
    """
    label, tone = STATE_LABELS.get(config.state, ("Unknown", "muted"))
    return render(
        "settings/cleanup_card.html",
        config=config,
        models=models,
        message=message,
        state_label=label,
        state_tone=tone,
        summary=CLEANUP_SUMMARY,
        caveat=CLEANUP_CAVEAT,
        mode_options=[("conservative", "Conservative"), ("off", "Off")],
        timeout_options=[
            (choice, _timeout_label(choice))
            for choice in TIMEOUT_CHOICES
            if MINIMUM_TIMEOUT_SECONDS <= choice <= MAXIMUM_TIMEOUT_SECONDS
        ],
        idle_options=[(minutes, _idle_label(minutes)) for minutes in CLEANUP_IDLE_UNLOAD_MINUTES],
    )


def _timeout_label(seconds: float) -> str:
    return f"{seconds:.0f} seconds"


def _idle_label(minutes: int) -> str:
    if minutes < 60:
        return f"{minutes} minutes"
    hours = minutes // 60
    return f"{hours} hour" if hours == 1 else f"{hours} hours"
