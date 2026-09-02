from __future__ import annotations

from markupsafe import Markup

from app.templating import render


def pairing_fragment(
    *,
    selected_url: str | None,
    candidates: list[str],
    token_redacted: str,
    token_plaintext: str = "",
    qr_svg: str = "",
    saved_urls: list[str],
    token_options: list[tuple[str, str]],
    selected_token_id: str,
    token_status: str = "ok",
    requested_token_id: str = "",
    requested_token_label: str | None = None,
    bind_host: str = "0.0.0.0",
    is_mac: bool = False,
) -> str:
    """See app/templates/pairing/card.html for the markup and behavior notes."""
    del bind_host, is_mac  # reserved for callers; exposure lives on the page shell
    if not selected_url:
        return render("pairing/empty.html")
    selected_token_label = next(
        (label for token_id, label in token_options if token_id == selected_token_id),
        "Bootstrap token",
    )
    # Strip the XML declaration for clean inline embedding.
    inline_svg = qr_svg
    if inline_svg.lstrip().startswith("<?xml"):
        inline_svg = inline_svg.split("?>", 1)[-1].lstrip()
    return render(
        "pairing/card.html",
        selected_url=selected_url,
        candidates=candidates,
        token_redacted=token_redacted,
        token_plaintext=token_plaintext,
        qr_svg_inline=Markup(inline_svg),
        saved_urls=saved_urls,
        token_options=token_options,
        selected_token_id=selected_token_id,
        selected_token_label=selected_token_label,
        token_status=token_status,
        requested_token_id=requested_token_id,
        requested_token_label=requested_token_label,
    )
