from __future__ import annotations

from dataclasses import dataclass

from markupsafe import Markup

from app.templating import render


@dataclass(frozen=True, slots=True)
class PairingFragmentData:
    selected_url: str | None
    candidates: list[str]
    token_redacted: str
    token_plaintext: str
    qr_svg: str
    saved_urls: list[str]
    token_options: list[tuple[str, str]]
    selected_token_id: str
    token_status: str
    requested_token_id: str
    requested_token_label: str | None


def redact_token(token: str) -> str:
    token_len = len(token)
    if token_len <= 8:
        return "•" * token_len
    prefix = token[:4]
    suffix = token[-4:]
    return f"{prefix}…{suffix} ({token_len} characters)"


def pairing_fragment(pairing_data: PairingFragmentData) -> str:
    """See app/templates/pairing/card.html for the markup and behavior notes."""
    if not pairing_data.selected_url:
        return render("pairing/empty.html")
    selected_token_label = next(
        (
            label
            for token_id, label in pairing_data.token_options
            if token_id == pairing_data.selected_token_id
        ),
        "Bootstrap token",
    )
    # Strip the XML declaration for clean inline embedding.
    inline_svg = pairing_data.qr_svg
    if inline_svg.lstrip().startswith("<?xml"):
        inline_svg = inline_svg.split("?>", 1)[-1].lstrip()
    return render(
        "pairing/card.html",
        selected_url=pairing_data.selected_url,
        candidates=pairing_data.candidates,
        token_redacted=pairing_data.token_redacted,
        token_plaintext=pairing_data.token_plaintext,
        qr_svg_inline=Markup(inline_svg),
        saved_urls=pairing_data.saved_urls,
        token_options=pairing_data.token_options,
        selected_token_id=pairing_data.selected_token_id,
        selected_token_label=selected_token_label,
        token_status=pairing_data.token_status,
        requested_token_id=pairing_data.requested_token_id,
        requested_token_label=pairing_data.requested_token_label,
    )
