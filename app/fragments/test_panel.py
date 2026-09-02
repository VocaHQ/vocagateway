from __future__ import annotations

from app.config import WILDCARD_BIND_HOSTS
from app.fragments.exposure import exposure_network_panel
from app.templating import render


def pair_and_test_fragment(
    pairing_html: str,
    maximum_duration_seconds: int,
    *,
    bind_host: str = "0.0.0.0",
    is_mac: bool = False,
) -> str:
    """See app/templates/test_panel/page.html for the markup and behavior notes."""
    network = exposure_network_panel(is_mac=is_mac) if bind_host in WILDCARD_BIND_HOSTS else ""
    return render(
        "test_panel/page.html",
        pairing_html=pairing_html,
        test_card_html=test_fragment(maximum_duration_seconds),
        network_html=network,
    )


def test_fragment(maximum_duration_seconds: int) -> str:
    return render("test_panel/test_card.html", maximum_duration_seconds=maximum_duration_seconds)
