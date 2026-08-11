from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Build-time overrides. Container images ship `app/` without `.git`, so the
# commit has to be baked in at build time (`--build-arg VOCAGATEWAY_GIT_COMMIT=...`)
# for anything but a source checkout to report one.
SHA_ENV = "VOCAGATEWAY_GIT_COMMIT"
SUBJECT_ENV = "VOCAGATEWAY_GIT_COMMIT_SUBJECT"
DATE_ENV = "VOCAGATEWAY_GIT_COMMIT_DATE"

_SEPARATOR = "\x1f"
_GIT_TIMEOUT_SECONDS = 5


@dataclass(frozen=True, slots=True)
class CommitInfo:
    sha: str
    short_sha: str
    subject: str
    committed_at: str | None


@lru_cache(maxsize=1)
def current_commit() -> CommitInfo | None:
    """The commit this gateway was built from, or None when it is unknowable.

    Cached: the answer cannot change while the process runs. Tests that patch
    the environment call `current_commit.cache_clear()`.
    """
    return _commit_from_env() or _commit_from_git()


def _commit_from_env() -> CommitInfo | None:
    sha = os.environ.get(SHA_ENV, "").strip()
    if not sha:
        return None
    return CommitInfo(
        sha=sha,
        short_sha=sha[:7],
        subject=os.environ.get(SUBJECT_ENV, "").strip(),
        committed_at=os.environ.get(DATE_ENV, "").strip() or None,
    )


def _commit_from_git() -> CommitInfo | None:
    # `.git` is a directory in a normal clone and a file in a worktree; both
    # are fine. Checking it first keeps `git` from walking up into an unrelated
    # parent repository when the gateway is vendored inside another checkout.
    if not (REPO_ROOT / ".git").exists():
        return None
    try:
        completed = subprocess.run(
            ["git", "log", "-1", f"--format=%H{_SEPARATOR}%s{_SEPARATOR}%cI"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_SECONDS,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        # No git binary, an empty repo with no commits, a slow filesystem —
        # commit info is a nicety, never a reason to fail a status request.
        return None
    parts = completed.stdout.strip().split(_SEPARATOR)
    if len(parts) != 3 or not parts[0]:
        return None
    sha, subject, committed_at = parts
    return CommitInfo(
        sha=sha,
        short_sha=sha[:7],
        subject=subject,
        committed_at=committed_at or None,
    )
