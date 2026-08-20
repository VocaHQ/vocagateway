from __future__ import annotations

from pathlib import Path

from app.context import VERSION
from app.fragments.about import about_fragment
from app.schemas import CommitStatus
from tests.test_admin import _assert_about_surface

SOCIAL = Path(__file__).resolve().parents[1] / "app" / "webui" / "brand" / "social"
SHA = "0979263b31465a19a6c5fa375ccdd0f2af250ca5"


def test_about_fragment_has_the_family_surface() -> None:
    html = about_fragment(VERSION)
    _assert_about_surface(html)
    assert VERSION in html
    assert "<dt>Version</dt>" in html
    assert "<dt>Build</dt>" not in html


def test_about_fragment_shows_commit_when_given() -> None:
    html = about_fragment(
        VERSION,
        CommitStatus(sha=SHA, short_sha="0979263", subject="Add About tab"),
    )
    assert "<dt>Build</dt>" in html
    assert "0979263" in html
    assert f'title="{SHA}"' in html


def test_social_marks_are_the_official_files() -> None:
    discord = (SOCIAL / "discord.svg").read_text(encoding="utf-8")
    x_mark = (SOCIAL / "x.svg").read_text(encoding="utf-8")
    github = (SOCIAL / "github.svg").read_text(encoding="utf-8")
    mail = (SOCIAL / "mail.svg").read_text(encoding="utf-8")
    for svg in (discord, x_mark, github, mail):
        assert 'viewBox="0 0 24 24"' in svg
        assert 'fill="currentColor"' in svg
        assert "5865F2" not in svg
        assert "#000" not in svg
    assert "M20.317 4.3698" in discord
    assert "M18.901 1.153" in x_mark
    assert "M12 .297" in github
    assert "M3 5h18a2 2 0 0 1 2 2v10" in mail


def test_talk_to_us_styles_use_design_currentcolor_ink() -> None:
    css = (Path(__file__).resolve().parents[1] / "app" / "webui" / "styles.css").read_text()
    about = css.split("/* Talk-to-us marks", 1)[1]
    assert "#14231c" in about
    assert "#0f6b57" in about
    assert "#f2f6f2" in about
    assert "currentColor" in about or "currentcolor" in about.lower()
    assert "5865F2" not in about
    assert "#000" not in about or "--about" in about
