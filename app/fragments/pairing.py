from __future__ import annotations

from html import escape
from urllib.parse import quote


def pairing_fragment(
    *,
    selected_url: str | None,
    candidates: list[str],
    token_redacted: str,
    qr_svg: str = "",
    saved_urls: list[str],
    token_options: list[tuple[str, str]],
    selected_token_id: str,
    token_status: str = "ok",
    requested_token_id: str = "",
    requested_token_label: str | None = None,
) -> str:
    """Authenticated phone-pairing card with QR for the selected gateway URL.

    The SVG is inlined so the browser does not need a second request (img tags
    cannot attach the WebUI bearer header). `token_options` lists the bootstrap
    token plus every device token, marking ones this process cannot currently
    display for QR purposes — only cached plaintexts can be encoded. That
    device's own pairing is completely unaffected either way: the phone keeps
    authenticating with its already-scanned secret regardless of whether this
    process can still show it. `token_status` explains why the shown token
    differs from what was asked for: `stale` means the requested device token
    still works fine but this process cannot redisplay its secret (typically
    after a restart) — rotating is optional and only needed to get a fresh,
    viewable QR, for example to re-pair a lost phone. `unknown` means it no
    longer exists at all (typically already revoked).
    """
    if not selected_url:
        return """
      <div class="card" id="pairing-card">
        <h2>Pair phone app</h2>
        <p class="muted">No address the phone can reach was found. Set <code>VOCAPHONE_PUBLIC_URL</code> to the URL the phone should use (for example <code>http://192.168.1.20:8765</code>), then reload.</p>
      </div>
        """
    options = "".join(
        f'<option value="{escape(url)}"{" selected" if url == selected_url else ""}>'
        f"{escape(url)}</option>"
        for url in candidates
    )
    token_option_html = "".join(
        f'<option value="{escape(token_id)}"{" selected" if token_id == selected_token_id else ""}>'
        f"{escape(label)}</option>"
        for token_id, label in token_options
    )
    selected_token_label = next(
        (label for token_id, label in token_options if token_id == selected_token_id),
        "Bootstrap token",
    )
    if token_status == "stale":
        stale_label = escape(requested_token_label or "That device token")
        unavailable_notice = f"""
          <div class="callout compact">
            <span><strong>{stale_label} is still paired and working.</strong> After a server restart this session cannot show its secret again (we never store it for recovery). Showing the bootstrap token instead. Only rotate if you need a new QR, for example to re-pair a lost phone.</span>
            <div class="callout-actions">
              <form
                hx-post="/ui/partials/pairing/tokens/{quote(requested_token_id, safe="")}/rotate"
                hx-target="#pairing-card" hx-swap="outerHTML">
                <input type="hidden" name="url" value="{escape(selected_url)}" />
                <button type="submit" class="primary small">Rotate &amp; show a new QR</button>
              </form>
              <span class="muted small">Rotate only if you need it: the current secret stops working immediately and the device must pair again.</span>
            </div>
          </div>
        """
    elif token_status == "unknown":
        unavailable_notice = """
          <div class="callout warning compact">
            <span>That device token is gone (it may have been revoked). Showing the bootstrap token instead. Create a new device token below for a fresh QR.</span>
          </div>
        """
    else:
        unavailable_notice = ""
    saved_rows = "".join(
        f"<li><code>{escape(url)}</code>"
        f'<button type="button" class="ghost small danger"'
        f' hx-delete="/ui/partials/pairing?url={quote(url, safe="")}"'
        f' hx-target="#pairing-card" hx-swap="outerHTML"'
        f' hx-confirm="Remove {escape(url)} from saved addresses?">Remove</button></li>'
        for url in saved_urls
    )
    saved_section = (
        f"""
            <div class="pairing-saved">
              <span>Saved addresses</span>
              <ul>{saved_rows}</ul>
            </div>
        """
        if saved_urls
        else ""
    )
    # Strip XML declaration for clean inline embedding.
    inline_svg = qr_svg
    if inline_svg.lstrip().startswith("<?xml"):
        inline_svg = inline_svg.split("?>", 1)[-1].lstrip()
    return f"""
      <div class="card" id="pairing-card">
        <h2>Pair phone app</h2>
        <p class="muted">Scan in VocaPhone on iPhone or Android to fill the gateway address and token. The QR includes the live token, so keep this page private.</p>
        <div class="pairing-layout">
          <div class="pairing-qr" role="img"
               aria-label="Pairing QR code for {escape(selected_url)}">{inline_svg}</div>
          <div class="pairing-meta">
            {unavailable_notice}
            <div class="pairing-fields">
              <label><span>Address encoded in the QR</span>
                <select name="url"
                        hx-get="/ui/partials/pairing"
                        hx-target="#pairing-card"
                        hx-swap="outerHTML"
                        hx-trigger="change"
                        hx-include="this,[name='token_id']">
                  {options}
                </select>
              </label>
              <label><span>Token to encode</span>
                <select name="token_id"
                        hx-get="/ui/partials/pairing"
                        hx-target="#pairing-card"
                        hx-swap="outerHTML"
                        hx-trigger="change"
                        hx-include="this,[name='url']">
                  {token_option_html}
                </select>
              </label>
            </div>
            <dl class="facts">
              <dt>Gateway URL</dt><dd>{escape(selected_url)}</dd>
              <dt>Token</dt>
              <dd>{escape(token_redacted)} &middot; {escape(selected_token_label)}</dd>
            </dl>
          </div>
        </div>
        <div class="pairing-fields pairing-extra">
          <form hx-get="/ui/partials/pairing" hx-target="#pairing-card" hx-swap="outerHTML">
            <input type="hidden" name="token_id" value="{escape(selected_token_id)}" />
            <label><span>Or type your own address (Tailscale IP, etc.)</span>
              <div class="row">
                <input name="url" type="text"
                       placeholder="100.101.102.103 or phone.tailnet-name.ts.net" />
                <button type="submit" class="ghost small">Use this</button>
              </div>
            </label>
          </form>
          <form hx-post="/ui/partials/pairing/tokens" hx-target="#pairing-card"
                hx-swap="outerHTML">
            <input type="hidden" name="url" value="{escape(selected_url)}" />
            <label><span>Or create a token for a new device</span>
              <div class="row">
                <input name="label" type="text" required maxlength="100"
                       placeholder="e.g. Kanishk&#39;s iPhone" />
                <button type="submit" class="primary small">Create &amp; show QR</button>
              </div>
            </label>
          </form>
          {saved_section}
        </div>
        <p class="pairing-note">Prefer the Wi-Fi LAN address when the phone is on the same network, or a Tailscale MagicDNS name when both devices are on the tailnet. <code>VOCAPHONE_PUBLIC_URL</code> overrides discovery. Device tokens you create here also show under Settings, where you can revoke them.</p>
      </div>
    """
