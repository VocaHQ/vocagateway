"""About tab: this build, the Voca family, and how to talk to us."""

from __future__ import annotations

from html import escape
from pathlib import Path

from app.schemas import CommitStatus

# Official marks from VocaHQ/.github. Do not redraw.
# Talk-to-us: brand/vocahq/social.
# Family platforms: brand/promo/cards/platform.
# Host hero: brand/vocagateway/vocagateway-1u.svg.
_BRAND_DIR = Path(__file__).resolve().parent.parent / "webui" / "brand"
_MARK_1U = "/assets/brand/vocagateway/vocagateway-1u.svg"
_MARK_HQ = "/assets/voca-logo.svg"
_ISSUES_URL = "https://github.com/VocaHQ/vocagateway/issues"
_LICENSE_URL = "https://github.com/VocaHQ/vocagateway/blob/main/LICENSE"

# icon, href, label. Platform names are files in webui/brand/platform/.
_FAMILY_LINKS: tuple[tuple[str, str, str], ...] = (
    ("hq", "https://vocahq.com", "VocaHQ"),
    ("linux", "https://vocalinux.com", "VocaLinux"),
    ("apple", "https://vocamac.com", "VocaMac"),
    ("windows", "https://vocawin.com", "VocaWin"),
    ("android", "https://vocaphone.vocahq.com", "VocaPhone"),
    ("gateway", "https://vocagateway.vocahq.com", "VocaGateway"),
)

_CONTACT_LINKS: tuple[tuple[str, str, str], ...] = (
    ("discord", "https://discord.gg/t6muquAJbm", "Discord"),
    ("x", "https://x.com/vocahq", "X"),
    ("mail", "mailto:hello@vocahq.com", "Email"),
)


def _icon(name: str, folder: str = "social") -> str:
    return (_BRAND_DIR / folder / f"{name}.svg").read_text(encoding="utf-8").strip()


def _link_target(url: str) -> str:
    if url.startswith("mailto:"):
        return ""
    return ' target="_blank" rel="noopener noreferrer"'


def _mark_img(src: str) -> str:
    return (
        f'<span class="about-icon">'
        f'<img src="{escape(src, quote=True)}" width="16" height="16" alt="" />'
        f"</span>"
    )


def _family_icon(name: str) -> str:
    if name == "hq":
        return _mark_img(_MARK_HQ)
    if name == "gateway":
        return _mark_img(_MARK_1U)
    return f'<span class="about-icon">{_icon(name, folder="platform")}</span>'


def about_fragment(version: str, commit: CommitStatus | None = None) -> str:
    family = "".join(
        f'<a class="ghost" href="{escape(url, quote=True)}"'
        f"{_link_target(url)}>"
        f"{_family_icon(name)}{escape(label)}</a>"
        for name, url, label in _FAMILY_LINKS
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
          <h2>About</h2>
          <p>Optional self-hosted compute for other Voca clients. Never on-device.</p>
        </div>
      </div>

      <div class="card system-card" id="about-host">
        <div class="sys-hero">
          <img class="about-1u" src="{_MARK_1U}" width="52" height="52" alt="VocaGateway" />
          <div class="sys-hero-copy">
            <p class="sys-hero-kicker">Beta</p>
            <h2 class="sys-hero-headline">The host you run</h2>
            <p class="sys-hero-meta">Audio you send here is transcribed on this machine.</p>
          </div>
        </div>
      </div>

      <div class="card" id="about-this-build">
        <h2>This build</h2>
        <p>This is infrastructure, not a dictation client. Audio leaves the client.
           This host transcribes it. There is no Voca account and no Voca cloud.</p>
        <div class="callout">
          <strong>Keep this host private.</strong>
          <span>Trusted LAN, Tailscale, or HTTPS. Never expose port 8765
                to the public internet.</span>
        </div>
        <dl class="facts">{facts_html}</dl>
      </div>

      <div class="card" id="about-family">
        <h2>Part of VocaHQ</h2>
        <p>Companion apps: VocaLinux, VocaMac, VocaWin, and VocaPhone.</p>
        <div class="onboarding-actions">{family}</div>
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
