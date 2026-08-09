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
            <span>Copy this token now. It will not be shown again. Paste it with
              the gateway address into that device&rsquo;s manual gateway fields.</span>
            <div class="row">
              <code id="new-token-value">{escape(plaintext)}</code>
              <button type="button" class="ghost small" id="copy-new-token">Copy</button>
            </div>
          </div>
        """
    rows = "".join(_token_row(entry) for entry in entries)
    if not rows:
        body = '<p class="empty-state tokens-empty">No tokens listed yet.</p>'
    else:
        body = f"""
        <div class="table-scroll">
          <table class="table">
            <thead>
              <tr>
                <th>Label</th>
                <th>Created</th>
                <th class="align-right">Actions</th>
              </tr>
            </thead>
            <tbody>{rows}</tbody>
          </table>
        </div>
        """
    return f"""
      <div class="card" id="tokens-card">
        <div class="section-heading">
          <h2>Device tokens</h2>
        </div>
        <p class="muted">One token per phone so a lost device only needs that token
           revoked. Phones on the bootstrap token keep working until you re-pair them.
           You can also create tokens while pairing.</p>
        {reveal}
        {body}
        <form class="tokens-create" hx-post="/ui/partials/tokens"
              hx-target="#tokens-card" hx-swap="outerHTML">
          <label class="settings-field">
            <span>New device</span>
            <span class="field-hint">Shown in the pairing QR picker and this list</span>
            <div class="row">
              <input name="label" type="text" required maxlength="100"
                     placeholder="e.g. Kanishk&#39;s iPhone" autocomplete="off" />
              <button type="submit" class="primary">Create token</button>
            </div>
          </label>
        </form>
      </div>
    """


def _token_row(entry: DeviceTokenEntry) -> str:
    if entry.revocable:
        actions = (
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
        )
    else:
        actions = '<span class="muted small">Bootstrap · not revocable here</span>'
    return (
        "<tr>"
        f"<td>{escape(entry.label)}</td>"
        f"<td>{escape(_format_created(entry.created_at))}</td>"
        f'<td class="align-right token-actions">{actions}</td>'
        "</tr>"
    )


def _format_created(value: datetime | None) -> str:
    return "—" if value is None else value.strftime("%Y-%m-%d %H:%M UTC")
