from __future__ import annotations

from datetime import datetime
from html import escape
from urllib.parse import quote

from app.schemas import DeviceTokenEntry


def tokens_fragment(
    entries: list[DeviceTokenEntry], *, new_token: tuple[str, str] | None = None
) -> str:
    reveal = ""
    if new_token is not None:
        label, plaintext = new_token
        reveal = f"""
          <div class="callout success">
            <strong>New secret for {escape(label)}</strong>
            <span>Copy this token now &mdash; it will not be shown again. Paste it with the
              gateway address into that device's manual "Gateway address and token" fields.</span>
            <div class="row">
              <code id="new-token-value">{escape(plaintext)}</code>
              <button type="button" class="ghost small" id="copy-new-token">Copy</button>
            </div>
          </div>
        """
    rows = "".join(
        "<tr>"
        f"<td>{escape(entry.label)}</td>"
        f"<td>{escape(_format_created(entry.created_at))}</td>"
        '<td class="align-right">'
        + (
            '<button type="button" class="ghost small"'
            f' hx-post="/ui/partials/tokens/{quote(entry.id, safe="")}/rotate"'
            ' hx-target="#tokens-card" hx-swap="outerHTML"'
            f' hx-confirm="Give {escape(entry.label)} a new secret? Its previous token stops '
            'working immediately.">Regenerate</button>'
            '<button type="button" class="ghost small danger"'
            f' hx-delete="/ui/partials/tokens/{quote(entry.id, safe="")}"'
            ' hx-target="#tokens-card" hx-swap="outerHTML"'
            f' hx-confirm="Revoke {escape(entry.label)}? Any device using only this token '
            'loses access immediately.">Revoke</button>'
            if entry.revocable
            else '<span class="muted small">Not revocable here</span>'
        )
        + "</td></tr>"
        for entry in entries
    )
    return f"""
      <div class="card" id="tokens-card">
        <h2>Paired device tokens</h2>
        <p class="muted">One token per phone, so losing a device means revoking its token
          rather than rotating everyone else's. Devices paired with the bootstrap token keep
          working until you re-pair them.</p>
        {reveal}
        <div class="table-scroll">
          <table class="table">
            <thead><tr><th>Label</th><th>Created</th><th class="align-right"></th></tr></thead>
            <tbody>{rows}</tbody>
          </table>
        </div>
        <form hx-post="/ui/partials/tokens" hx-target="#tokens-card" hx-swap="outerHTML">
          <div class="row">
            <input name="label" type="text" required maxlength="100"
                   placeholder="e.g. Kanishk&#39;s iPhone" />
            <button type="submit" class="primary">Create token</button>
          </div>
        </form>
      </div>
    """


def _format_created(value: datetime | None) -> str:
    return "—" if value is None else value.strftime("%Y-%m-%d %H:%M UTC")
