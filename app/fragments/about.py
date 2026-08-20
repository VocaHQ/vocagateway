"""About tab: this build, the Voca family, and how to talk to us."""

from __future__ import annotations

from html import escape
from pathlib import Path

from app.schemas import CommitStatus

# Official Talk-to-us marks from VocaHQ/.github (brand/vocahq/social @ 61c8eee).
# Do not redraw. Swap the files in app/webui/brand/social/ when Design updates them.
_SOCIAL_DIR = Path(__file__).resolve().parent.parent / "webui" / "brand" / "social"
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
        f'<li><a href="{escape(url, quote=True)}" target="_blank" '
        f'rel="noopener noreferrer">{escape(label)}</a></li>'
        for url, label in _FAMILY_LINKS
    )
    contacts = "".join(
        f'<a class="about-contact" href="{escape(url, quote=True)}"'
        f"{_link_target(url)}>"
        f'<span class="about-icon">{_icon(name)}</span>{escape(label)}</a>'
        for name, url, label in _CONTACT_LINKS
    )
    facts = [("<dt>Version</dt>", f"<dd>{escape(version)}</dd>")]
    if commit is not None:
        facts.append(("<dt>Build</dt>", f"<dd>{_build_fact(commit)}</dd>"))
    facts_html = "".join(label + value for label, value in facts)
    return f"""
      <section class="about-page" aria-labelledby="about-title">
        <header class="about-hero">
          <p class="about-kicker">About</p>
          <h2 id="about-title">VocaGateway <span class="about-status">early.</span></h2>
          <p>Optional self-hosted compute for other Voca clients.</p>
          <p><a href="{_SITE_URL}" target="_blank" rel="noopener noreferrer">{_SITE_HOST}</a></p>
        </header>

        <article class="card about-card" id="about-this-build">
          <h3 class="about-card-kicker">This build</h3>
          <p>VocaGateway is Early (alpha). Optional self-hosted compute for other
             Voca clients. This is infrastructure, not a dictation client.</p>
          <p>Audio leaves the client. This host transcribes it. There is no Voca
             account and no Voca cloud.</p>
          <p>Keep it on a trusted LAN, Tailscale, or HTTPS. Never expose port
             8765 to the public internet.</p>
          <p>License <a href="{_LICENSE_URL}" target="_blank"
             rel="noopener noreferrer">AGPL-3.0</a>.</p>
          <dl class="facts about-facts">{facts_html}</dl>
        </article>

        <article class="card about-card" id="about-family">
          <h3 class="about-card-kicker">Part of VocaHQ</h3>
          <p>VocaGateway is part of the VocaHQ family. Companion apps: VocaLinux,
             VocaMac, VocaWin, and VocaPhone.</p>
          <p>VocaGateway is optional self-hosted compute for other Voca clients.</p>
          <ul class="about-family-links">{family_links}</ul>
        </article>

        <article class="card about-card" id="about-talk">
          <h3 class="about-card-kicker">Talk to us</h3>
          <p>Bugs and feature ideas go on GitHub issues.</p>
          <p>
            <a class="about-cta" href="{_ISSUES_URL}" target="_blank"
               rel="noopener noreferrer">
              <span class="about-icon">{_icon("github")}</span>Report a bug or idea
            </a>
          </p>
          <div class="about-contacts">{contacts}</div>
        </article>
      </section>
    """


def _build_fact(commit: CommitStatus) -> str:
    text = commit.short_sha
    if commit.subject.strip():
        text += f" · {commit.subject.strip()}"
    return f'<span title="{escape(commit.sha)}">{escape(text)}</span>'
