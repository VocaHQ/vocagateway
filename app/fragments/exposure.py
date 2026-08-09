from __future__ import annotations

from app.config import WILDCARD_BIND_HOSTS


def exposure_banner_fragment(bind_host: str, *, is_mac: bool = False) -> str:
    """Full-width one-line top strip when the gateway binds every interface.

    Client JS may hide this for 24h after the × is clicked; Pair & test keeps
    a permanent network panel.
    """
    if bind_host not in WILDCARD_BIND_HOSTS:
        return (
            '<div id="exposure-banner" class="exposure-banner" hidden '
            'data-empty="1" aria-hidden="true"></div>'
        )
    firewall = "Keep macOS Firewall on" if is_mac else "Keep the host firewall enabled"
    return f"""
      <div id="exposure-banner" class="exposure-banner" role="status"
           data-dismiss-key="vocaphone.exposure-dismiss-until">
        <div class="exposure-banner-inner">
          <span class="exposure-banner-mark" aria-hidden="true">!</span>
          <p class="exposure-banner-copy">
            <strong>Listening on every network interface.</strong>
            The API still needs the bearer token. {firewall}, prefer Tailscale
            for remote access, and do not publish this port on the public internet.
          </p>
          <button type="button" class="exposure-banner-dismiss"
                  data-dismiss-exposure
                  title="Dismiss for 24 hours"
                  aria-label="Dismiss for 24 hours">
            <span aria-hidden="true">&times;</span>
          </button>
        </div>
      </div>
    """


def exposure_network_panel(*, is_mac: bool = False) -> str:
    """Persistent network callout for Pair & test — not dismissible.

    Mounted once at the end of the Pair & test page shell (not inside
    `#pairing-card`), so HTMX swaps of the pairing card never duplicate it.
    """
    firewall = "Keep macOS Firewall on" if is_mac else "Keep the host firewall enabled"
    return f"""
      <div id="pairing-exposure-panel" class="exposure-panel" role="note">
        <div class="exposure-panel-head">
          <span class="exposure-banner-mark" aria-hidden="true">!</span>
          <strong>Listening on every network interface</strong>
        </div>
        <p>Any device that can reach this host can open the gateway port.
           Transcription and admin still need the bearer token.</p>
        <ul class="exposure-panel-list">
          <li>{firewall}</li>
          <li>Prefer Tailscale (or a similar mesh) for remote access</li>
          <li>Do not publish this port on the public internet</li>
        </ul>
        <p class="muted small">Bind to a single address with
           <code>VOCAPHONE_BIND_HOST</code> if you want to stop listening on every interface.</p>
      </div>
    """
