"""The deployment file and the contributor's file must describe one gateway.

`compose.prod.yaml` is deliberately self-contained: after a release has
published an image, an operator downloads it on its own, without a checkout,
and runs that image. The cost of that is a second copy of the service
definition, and a second copy is a thing that drifts — a hardening flag
tightened in one file and not the other, an environment variable forwarded
to contributors and not to deployments.

These tests are what makes the duplication safe. They read both files and hold
everything but the image to be identical, so the divergence has to be
deliberate and visible in a diff.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
COMPOSE_FILE = REPOSITORY_ROOT / "compose.yaml"
PRODUCTION_COMPOSE_FILE = REPOSITORY_ROOT / "compose.prod.yaml"
GATEWAY_SERVICE = "gateway"
PUBLISHED_IMAGE = "docker.io/vocahq/vocagateway"
# Everything about the service that is not "where do the bytes come from".
SHARED_KEYS = (
    "restart",
    "init",
    "network_mode",
    "ports",
    "environment",
    "secrets",
    "volumes",
    "security_opt",
    "cap_drop",
    "tmpfs",
    "logging",
)


def _load(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def development() -> dict[str, Any]:
    return _load(COMPOSE_FILE)


@pytest.fixture(scope="module")
def production() -> dict[str, Any]:
    return _load(PRODUCTION_COMPOSE_FILE)


@pytest.mark.parametrize("key", SHARED_KEYS)
def test_the_two_gateway_services_agree(
    key: str, development: dict[str, Any], production: dict[str, Any]
) -> None:
    """One difference is allowed — the image — and it is asserted separately."""
    assert (
        production["services"][GATEWAY_SERVICE][key]
        == (development["services"][GATEWAY_SERVICE][key])
    )


def test_the_production_file_never_builds(production: dict[str, Any]) -> None:
    """A deployment that compiles whisper.cpp is the thing this file exists to avoid."""
    assert "build" not in production["services"][GATEWAY_SERVICE]


def test_the_production_file_defaults_to_the_published_image(
    production: dict[str, Any],
) -> None:
    image = production["services"][GATEWAY_SERVICE]["image"]
    assert image.startswith("${VOCAGATEWAY_IMAGE:-")
    assert PUBLISHED_IMAGE in image


def test_the_production_file_carries_only_the_published_service(
    production: dict[str, Any],
) -> None:
    """The cuda and vulkan images are not published, so they are not offered here.

    Naming them would promise a tag the release workflow never pushes, and
    `docker compose --profile cuda up` would fail on a pull rather than on
    something an operator can read.
    """
    assert list(production["services"]) == [GATEWAY_SERVICE]


def test_both_files_take_the_token_the_same_way(
    development: dict[str, Any], production: dict[str, Any]
) -> None:
    """The token is the one piece of setup an operator cannot skip."""
    assert production["secrets"] == development["secrets"]
    assert production["name"] == development["name"]


def test_the_data_volume_is_named_the_same_in_both(
    development: dict[str, Any], production: dict[str, Any]
) -> None:
    """A renamed volume is a silently empty gateway: no models, no config, no tokens."""
    assert list(production["volumes"]) == list(development["volumes"])


def _section(text: str, heading: str) -> str:
    start = text.index(heading)
    next_heading = text.find("\n## ", start + 1)
    return text[start:] if next_heading == -1 else text[start:next_heading]


def test_readme_compose_quick_start_builds_from_source_first() -> None:
    """Hub is not live yet; the first command a clone runs has to be a build."""
    readme = (REPOSITORY_ROOT / "README.md").read_text(encoding="utf-8")
    section = _section(readme, "## Docker Compose quick start")
    build_at = section.index("docker compose up --detach --build")
    prod_at = section.index("docker compose -f compose.prod.yaml up --detach")
    assert build_at < prod_at


def test_deployment_docs_build_from_source_before_published_pull() -> None:
    docs = (REPOSITORY_ROOT / "docs" / "deployment.md").read_text(encoding="utf-8")
    section = _section(docs, "## Docker Compose deployment")
    assert section.index("### Building from the checkout") < section.index(
        "### Running a published image"
    )
    assert section.index("docker compose up --detach --build") < section.index(
        "docker compose -f compose.prod.yaml up --detach"
    )


def test_release_workflow_smokes_digest_before_promoting_tags() -> None:
    """Public tags must not move until the amd64 digest has been run."""
    workflow = (REPOSITORY_ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8")
    smoke = workflow.index("name: Smoke-test the amd64 image")
    assemble = workflow.index("name: Assemble the manifest list")
    copy = workflow.index("name: Copy the manifest to GHCR")
    assert smoke < assemble < copy
    smoke_run = workflow[smoke:assemble]
    assert "@sha256:" in smoke_run
    assert "imagetools create" not in smoke_run
    assert "enable=${{ steps.gate.outputs.move_minor == 'true' }}" in workflow
    assert "steps.gate.outputs.move_latest == 'true'" in workflow
    assert "pattern={{version}}" in workflow
    assert "git tag --list" in workflow
