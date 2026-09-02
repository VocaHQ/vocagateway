"""Console-script entry points: primary names and deprecated aliases."""

from __future__ import annotations

import tomllib
from importlib.metadata import entry_points
from pathlib import Path

PRIMARY_SCRIPTS = {
    "vocagateway": "app.cli:serve",
    "vocagateway-status": "app.cli:status",
    "vocagateway-diagnostics": "app.cli:diagnostics",
    "vocagateway-cleanup": "app.cli:cleanup",
    "vocagateway-token": "app.cli:token",
}

DEPRECATED_ALIASES = {
    "vocaphone-server": "vocagateway",
    "vocaphone-status": "vocagateway-status",
    "vocaphone-diagnostics": "vocagateway-diagnostics",
    "vocaphone-cleanup": "vocagateway-cleanup",
    "vocaphone-token": "vocagateway-token",
}


def _pyproject_scripts() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    payload = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    scripts = payload["project"]["scripts"]
    assert isinstance(scripts, dict)
    return {str(name): str(target) for name, target in scripts.items()}


def test_pyproject_declares_primary_and_dep_aa() -> None:
    scripts = _pyproject_scripts()
    for name, target in PRIMARY_SCRIPTS.items():
        assert scripts[name] == target
    for alias, primary in DEPRECATED_ALIASES.items():
        assert scripts[alias] == scripts[primary]


def test_installed_entry_points_resolve_pri_dca7a() -> None:
    """Require the editable/venv install so `uv run` / console scripts work."""
    console = entry_points().select(group="console_scripts")
    by_name = {ep.name: ep for ep in console}
    expected = dict(PRIMARY_SCRIPTS)
    for alias, primary in DEPRECATED_ALIASES.items():
        expected[alias] = PRIMARY_SCRIPTS[primary]
    for name, target in expected.items():
        assert name in by_name, f"missing console script: {name}"
        assert f"{by_name[name].module}:{by_name[name].attr}" == target
