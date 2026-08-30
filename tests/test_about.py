from __future__ import annotations

from pathlib import Path

from app.context import VERSION
from app.fragments.about import about_fragment
from app.schemas import CommitStatus
from tests.test_admin import _assert_about_surface

BRAND = Path(__file__).resolve().parents[1] / "app" / "webui" / "brand"
SOCIAL = BRAND / "social"
PLATFORM = BRAND / "platform"
SHA = "0979263b31465a19a6c5fa375ccdd0f2af250ca5"


def test_about_fragment_has_the_family_surface() -> None:
    html = about_fragment(VERSION)
    _assert_about_surface(html)
    assert "Beta" in html
    assert "Early" not in html
    assert "https://discord.gg/t6muquAJbm" in html
    assert VERSION in html
    assert "<h2>About</h2>" in html
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
    assert "Add About tab" not in html


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


def test_platform_marks_are_the_official_files() -> None:
    linux = (PLATFORM / "linux.svg").read_text(encoding="utf-8")
    apple = (PLATFORM / "apple.svg").read_text(encoding="utf-8")
    windows = (PLATFORM / "windows.svg").read_text(encoding="utf-8")
    android = (PLATFORM / "android.svg").read_text(encoding="utf-8")
    for svg in (linux, apple, windows, android):
        assert 'viewBox="0 0 24 24"' in svg
        assert 'fill="currentColor"' in svg
        assert "5865F2" not in svg
        assert "#000" not in svg
        assert "#0F6B57" not in svg
    assert "M12.504 0c-.155" in linux
    assert "M12.152 6.896" in apple
    assert "M0,0H11.377" in windows
    assert "M18.4395 5.5586" in android


def test_official_1u_mark_is_vendored() -> None:
    mark = (
        Path(__file__).resolve().parents[1]
        / "app"
        / "webui"
        / "brand"
        / "vocagateway"
        / "vocagateway-1u.svg"
    ).read_text(encoding="utf-8")
    assert 'viewBox="0 0 1024 1024"' in mark
    assert "#0F6B57" in mark
    assert "#F2F6F2" in mark
    assert 'aria-label="VocaGateway"' in mark


def test_about_styles_reuse_webui_chrome() -> None:
    css = (Path(__file__).resolve().parents[1] / "app" / "webui" / "styles.css").read_text()
    assert "text-transform: uppercase" not in css.split("/* About reuses")[1]
    assert "about-kicker" not in css
    assert "about-card-kicker" not in css
    assert ".about-icon" in css
    assert "about-family-links" not in css
    assert "5865F2" not in css.split("/* About reuses")[1]
    about_css = css.split("/* About reuses")[1]
    assert "linear-gradient" not in about_css
