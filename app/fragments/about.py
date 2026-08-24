"""About tab: this build, the Voca family, and how to talk to us."""

from __future__ import annotations

from html import escape
from pathlib import Path

from app.schemas import CommitStatus

# Official marks from VocaHQ/.github. Do not redraw.
# Talk-to-us: brand/vocahq/social. Host hero: brand/vocagateway/vocagateway-1u.svg.
_SOCIAL_DIR = Path(__file__).resolve().parent.parent / "webui" / "brand" / "social"
_MARK_1U = "/assets/brand/vocagateway/vocagateway-1u.svg"
_ISSUES_URL = "https://github.com/VocaHQ/vocagateway/issues"
_LICENSE_URL = "https://github.com/VocaHQ/vocagateway/blob/main/LICENSE"
_SITE_HOST = "vocagateway.vocahq.com"
_SITE_URL = f"https://{_SITE_HOST}/"

_FAMILY_LINKS: tuple[tuple[str, str], ...] = (
    ("https://vocahq.com", "vocahq.com"),
    ("https://vocalinux.com", "vocalinux.com"),
    ("https://vocamac.com", "vocamac.com"),
    ("https://vocawin.com", "vocawin.com"),
    ("https://vocaphone.vocahq.com", "vocaphone.vocahq.com"),
    ("https://vocagateway.vocahq.com", "vocagateway.vocahq.com"),
)

_CONTACT_LINKS: tuple[tuple[str, str, str], ...] = (
    ("discord", "https://discord.gg/UMJduhcqn", "Discord"),
    ("x", "https://x.com/vocahq", "X"),
    ("mail", "mailto:hello@vocahq.com", "Email"),
)


def _icon(name: str) -> str:
    return (_SOCIAL_DIR / f"{name}.svg").read_text(encoding="utf-8").strip()


def _link_target(url: str) -> str:
    if url.startswith("mailto:"):
        return ""
    return ' target="_blank" rel="noopener noreferrer"'


def about_fragment(version: str, commit: CommitStatus | None = None) -> str:
    family_links = "".join(
        f'<a href="{escape(url, quote=True)}" target="_blank" '
        f'rel="noopener noreferrer">{escape(label)}</a>'
        for url, label in _FAMILY_LINKS
    )
    contacts = "".join(
        f'<a class="ghost" href="{escape(url, quote=True)}"'
        f"{_link_target(url)}>"
        f'<span class="about-icon">{_icon(name)}</span>{escape(label)}</a>'
        for name, url, label in _CONTACT_LINKS
    )
    facts = [
        (
            "<dt>License</dt>",
            f'<dd><a href="{_LICENSE_URL}" target="_blank" '
            f'rel="noopener noreferrer">AGPL-3.0</a></dd>',
        ),
        ("<dt>Version</dt>", f"<dd>{escape(version)}</dd>"),
    ]
    if commit is not None:
        facts.append(("<dt>Build</dt>", f"<dd>{_build_fact(commit)}</dd>"))
    facts_html = "".join(label + value for label, value in facts)
    return f"""
      <div class="page-head">
        <div>
          <h2>VocaGateway</h2>
          <p>Optional self-hosted compute for other Voca clients. Never on-device.
             <a href="{_SITE_URL}" target="_blank" rel="noopener noreferrer">{_SITE_HOST}</a></p>
        </div>
      </div>

      <div class="card system-card" id="about-host">
        <div class="sys-hero">
          <img class="about-1u" src="{_MARK_1U}" width="52" height="52" alt="VocaGateway" />
          <div class="sys-hero-copy">
            <p class="sys-hero-kicker">Early</p>
            <h2 class="sys-hero-headline">The host you run</h2>
            <p class="sys-hero-meta">Audio you send here is transcribed on this machine.</p>
          </div>
        </div>
      </div>

      <div class="card" id="about-this-build">
        <h2>This build</h2>
        <p>VocaGateway is Early optional self-hosted compute for other Voca clients.
           This is infrastructure, not a dictation client.</p>
        <p>Audio leaves the client. This host transcribes it. There is no Voca
           account and no Voca cloud.</p>
        <div class="callout">
          <strong>Keep this host private.</strong>
          <span>Trusted LAN, Tailscale, or HTTPS. Never expose port 8765
                to the public internet.</span>
        </div>
        <dl class="facts">{facts_html}</dl>
      </div>

      <div class="card" id="about-family">
        <h2>Part of VocaHQ</h2>
        <p>VocaGateway is part of the VocaHQ family. Companion apps: VocaLinux,
           VocaMac, VocaWin, and VocaPhone. VocaGateway is optional self-hosted
           compute for other Voca clients.</p>
        <p class="about-family-links">{family_links}</p>
      </div>

      <div class="card" id="about-talk">
        <h2>Talk to us</h2>
        <p>Bugs and feature ideas go on GitHub issues.</p>
        <div class="onboarding-actions">
          <a class="primary" href="{_ISSUES_URL}" target="_blank" rel="noopener noreferrer">
            <span class="about-icon">{_icon("github")}</span>Report a bug or idea
          </a>
          {contacts}
        </div>
      </div>
    """


def _build_fact(commit: CommitStatus) -> str:
    return f'<span title="{escape(commit.sha)}">{escape(commit.short_sha)}</span>'
