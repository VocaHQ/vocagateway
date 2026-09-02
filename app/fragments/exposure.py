from __future__ import annotations

from app.config import WILDCARD_BIND_HOSTS
from app.templating import render


def exposure_banner_fragment(bind_host: str, *, is_mac: bool = False) -> str:
    """See app/templates/exposure/banner.html for the markup and behavior notes."""
    firewall = "Keep macOS Firewall on" if is_mac else "Keep the host firewall enabled"
    return render(
        "exposure/banner.html",
        show=bind_host in WILDCARD_BIND_HOSTS,
        firewall=firewall,
    )


def exposure_network_panel(*, is_mac: bool = False) -> str:
    """See app/templates/exposure/panel.html for the markup and behavior notes."""
    firewall = "Keep macOS Firewall on" if is_mac else "Keep the host firewall enabled"
    return render("exposure/panel.html", firewall=firewall)
