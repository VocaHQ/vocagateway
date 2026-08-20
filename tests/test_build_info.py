from __future__ import annotations

import dataclasses
import subprocess
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest
from conftest import FakeEngine, FakeNormalizer

from app import build_info
from app.build_info import DATE_ENV, SHA_ENV, SUBJECT_ENV, CommitInfo, current_commit
from app.config import Settings
from app.fragments.overview import _commit_detail
from app.main import create_app
from app.schemas import CommitStatus

SHA = "0979263b31465a19a6c5fa375ccdd0f2af250ca5"


@pytest.fixture(autouse=True)
def _clear_commit_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    # current_commit() caches for the life of the process; every test here
    # changes what it should resolve to.
    for name in (SHA_ENV, SUBJECT_ENV, DATE_ENV):
        monkeypatch.delenv(name, raising=False)
    current_commit.cache_clear()


@pytest.fixture
async def debug_client(
    settings: Settings, fake_engine: FakeEngine
) -> AsyncIterator[httpx.AsyncClient]:
    """The shared `client` runs with debug off, which is exactly what hides the commit."""
    app = create_app(
        dataclasses.replace(settings, debug=True),
        engine=fake_engine,
        normalizer=FakeNormalizer(),
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as test_client:
        yield test_client


def test_env_overrides_win_over_git(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(SHA_ENV, SHA)
    monkeypatch.setenv(SUBJECT_ENV, "Merge pull request #10")
    monkeypatch.setenv(DATE_ENV, "2026-08-11T15:51:01+05:30")
    commit = current_commit()
    assert commit == CommitInfo(
        sha=SHA,
        short_sha="0979263",
        subject="Merge pull request #10",
        committed_at="2026-08-11T15:51:01+05:30",
    )


def test_env_sha_alone_is_enough(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(SHA_ENV, SHA)
    commit = current_commit()
    assert commit is not None
    assert commit.subject == ""
    assert commit.committed_at is None


def test_reads_the_latest_commit_from_git(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _git_repo(tmp_path, subject="feat: add commit info")
    monkeypatch.setattr(build_info, "REPO_ROOT", tmp_path)
    commit = current_commit()
    assert commit is not None
    assert commit.subject == "feat: add commit info"
    assert len(commit.sha) == 40
    assert commit.short_sha == commit.sha[:7]
    assert commit.committed_at is not None


def test_no_git_directory_reports_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # An installed wheel or a container built without the build args: the
    # gateway still answers, it just cannot name a commit.
    monkeypatch.setattr(build_info, "REPO_ROOT", tmp_path)
    assert current_commit() is None


def test_git_failure_is_not_fatal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / ".git").mkdir()  # looks like a repo, but `git log` will fail
    monkeypatch.setattr(build_info, "REPO_ROOT", tmp_path)
    assert current_commit() is None


def test_commit_detail_shortens_the_subject() -> None:
    detail = _commit_detail(
        CommitStatus(
            sha=SHA,
            short_sha="0979263",
            subject="x" * 100,
            committed_at="2026-08-11T15:51:01+05:30",
        )
    )
    assert f'title="{SHA}"' in detail
    assert "0979263" in detail
    assert "…" in detail
    assert "2026-08-11" in detail


async def test_status_and_diagnostics_carry_the_commit(
    debug_client: httpx.AsyncClient, authorization: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(SHA_ENV, SHA)
    monkeypatch.setenv(SUBJECT_ENV, "Merge pull request #10")
    current_commit.cache_clear()
    status = await debug_client.get("/v1/admin/status", headers=authorization)
    assert status.status_code == 200
    assert status.json()["commit"] == {
        "sha": SHA,
        "short_sha": "0979263",
        "subject": "Merge pull request #10",
        "committed_at": None,
    }
    diagnostics = await debug_client.get("/v1/admin/diagnostics", headers=authorization)
    assert diagnostics.status_code == 200
    assert diagnostics.json()["commit"]["sha"] == SHA

    overview = await debug_client.get("/ui/partials/overview", headers=authorization)
    assert overview.status_code == 200
    assert "build 0979263" in overview.text
    assert "<dt>Build</dt>" in overview.text

    about = await debug_client.get("/ui/partials/about", headers=authorization)
    assert about.status_code == 200
    assert "<dt>Build</dt>" in about.text
    assert "0979263" in about.text


async def test_commit_is_hidden_without_debug(
    client: httpx.AsyncClient, authorization: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Debug defaults to off, so a plain deployment names no revision anywhere."""
    monkeypatch.setenv(SHA_ENV, SHA)
    monkeypatch.setenv(SUBJECT_ENV, "Merge pull request #10")
    current_commit.cache_clear()
    status = await client.get("/v1/admin/status", headers=authorization)
    assert status.status_code == 200
    assert status.json()["commit"] is None

    diagnostics = await client.get("/v1/admin/diagnostics", headers=authorization)
    assert diagnostics.status_code == 200
    assert diagnostics.json()["commit"] is None

    overview = await client.get("/ui/partials/overview", headers=authorization)
    assert overview.status_code == 200
    # Neither the hero meta line nor a "Build" row in the hardware details, and
    # no stray sha anywhere in the markup.
    assert "build 0979263" not in overview.text
    assert "<dt>Build</dt>" not in overview.text
    assert SHA not in overview.text

    about = await client.get("/ui/partials/about", headers=authorization)
    assert about.status_code == 200
    assert "<dt>Build</dt>" not in about.text
    assert "0979263" not in about.text
    assert SHA not in about.text


def _git_repo(root: Path, *, subject: str) -> None:
    (root / "file.txt").write_text("hello\n")
    env = {
        "GIT_AUTHOR_NAME": "Test",
        "GIT_AUTHOR_EMAIL": "test@example.com",
        "GIT_COMMITTER_NAME": "Test",
        "GIT_COMMITTER_EMAIL": "test@example.com",
        "PATH": "/usr/bin:/bin:/usr/local/bin",
    }
    for args in (
        ["init", "--quiet"],
        ["add", "file.txt"],
        ["commit", "--quiet", "--no-gpg-sign", "--message", subject],
    ):
        subprocess.run(["git", *args], cwd=root, env=env, check=True, capture_output=True)
