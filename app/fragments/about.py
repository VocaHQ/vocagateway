"""About tab: this build, the Voca family, and how to talk to us."""

from __future__ import annotations

from html import escape
from pathlib import Path

from app.schemas import CommitStatus

# Official marks from VocaHQ/.github. Do not redraw.
# Talk-to-us: brand/vocahq/social.
# Family row: brand/promo/cards/platform, vocahq/voca-mark, vocagateway-1u chassis.
# Host hero: brand/vocagateway/vocagateway-tower.svg.
_BRAND_DIR = Path(__file__).resolve().parent.parent / "webui" / "brand"
_MARK_TOWER = "/assets/brand/vocagateway/vocagateway-tower.svg"
_ISSUES_URL = "https://github.com/VocaHQ/vocagateway/issues"
_LICENSE_URL = "https://github.com/VocaHQ/vocagateway/blob/main/LICENSE"
_PLATFORM_BRAND_DIR = "platform"

# folder, file stem, href, label.
_FAMILY_LINKS: tuple[tuple[str, str, str, str], ...] = (
    ("vocahq", "voca-mark", "https://vocahq.com", "VocaHQ"),
    (_PLATFORM_BRAND_DIR, "linux", "https://vocalinux.com", "VocaLinux"),
    (_PLATFORM_BRAND_DIR, "apple", "https://vocamac.com", "VocaMac"),
    (_PLATFORM_BRAND_DIR, "windows", "https://vocawin.com", "VocaWin"),
    (_PLATFORM_BRAND_DIR, "android", "https://vocaphone.vocahq.com", "VocaPhone"),
    ("vocagateway", "vocagateway-1u-mark", "https://vocagateway.vocahq.com", "VocaGateway"),
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


def about_fragment(version: str, commit: CommitStatus | None = None) -> str:
    family = _family_links_html()
    contacts = _contact_links_html()
    facts_html = _facts_html(version, commit)
    return f"""
      <div class="page-head">
        <div>
          <h2>About</h2>
          <p>Optional self-hosted compute for other Voca clients. Never on-device.</p>
        </div>
      </div>

      <div class="card system-card" id="about-host">
        <div class="sys-hero">
          <img class="about-host-mark" src="{_MARK_TOWER}"
               width="52" height="52" alt="VocaGateway" />
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
        <div class="about-info" role="note">
          <span class="about-info-mark" aria-hidden="true">i</span>
          <div>
            <strong>Keep this host private.</strong>
            <span>Trusted LAN, Tailscale, or HTTPS. Never expose port 8765
                  to the public internet.</span>
          </div>
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


def _facts_html(version: str, commit: CommitStatus | None) -> str:
    facts: tuple[tuple[str, str], ...] = (
        (
            "<dt>License</dt>",
            f'<dd><a href="{_LICENSE_URL}" target="_blank" '
            f'rel="noopener noreferrer">AGPL-3.0</a></dd>',
        ),
        ("<dt>Version</dt>", f"<dd>{escape(version)}</dd>"),
    )
    if commit is not None:
        facts += (("<dt>Build</dt>", f"<dd>{_build_fact(commit)}</dd>"),)
    return "".join(label + detail for label, detail in facts)


def _family_links_html() -> str:
    return "".join(
        f'<a class="ghost" href="{escape(url, quote=True)}"'
        f"{_link_target(url)}>"
        f'<span class="about-icon">{_icon(stem, folder=folder)}</span>'
        f"{escape(label)}</a>"
        for folder, stem, url, label in _FAMILY_LINKS
    )


def _contact_links_html() -> str:
    return "".join(
        f'<a class="ghost" href="{escape(url, quote=True)}"'
        f"{_link_target(url)}>"
        f'<span class="about-icon">{_icon(name)}</span>{escape(label)}</a>'
        for name, url, label in _CONTACT_LINKS
    )
