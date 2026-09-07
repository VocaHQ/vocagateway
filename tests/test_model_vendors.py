"""The vendor badge is a factual claim about who made a model, so it is tested.

Three things can go wrong with it and none of them are visible in review: a new
family ships with no vendor and quietly loses its badge, a vendor is added
without a colour and renders in the muted default, or a vendor with no glyph
ends up with no monogram either and shows an empty box. Each has a test.
"""

from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path

import httpx
import pytest
from conftest import TOKEN

from app import admin_queries
from app.catalog import DEFAULT_CATALOG, RETIRED_CATALOG
from app.config import Settings
from app.main import create_app
from app.model_vendors import GLYPH_VENDORS, VENDORS, Vendor, mapped_families, vendor_for

STYLESHEET = Path(__file__).resolve().parent.parent / "app" / "webui" / "styles.css"
MACROS = (
    Path(__file__).resolve().parent.parent / "app" / "templates" / "macros" / "vendor_marks.html"
)
# Three characters is what the badge holds at a legible size.
MAXIMUM_MONOGRAM_LENGTH = 3
CATALOG_FAMILIES = frozenset(
    model.family for model in (*DEFAULT_CATALOG, *RETIRED_CATALOG) if model.family
)


def test_every_catalog_family_names_its_vendor() -> None:
    """A new family must not reach the model list unmarked.

    The mapping is by hand — the download URL names whoever exported the ONNX,
    not whoever trained the weights — so nothing but this test notices when a
    family is added and the mapping is not.
    """
    unmapped = sorted(family for family in CATALOG_FAMILIES if vendor_for(family) is None)

    assert not unmapped, f"families with no vendor: {unmapped}"


def test_the_mapping_carries_no_families_the_catalog_dropped() -> None:
    """Stale entries are how a rename turns into a family with no badge."""
    assert mapped_families() <= CATALOG_FAMILIES


def test_imported_weights_get_no_vendor() -> None:
    """A file the operator supplied has no publisher to name, and none is invented."""
    assert vendor_for("Custom Whisper") is None
    assert vendor_for("") is None


@pytest.mark.parametrize("vendor", VENDORS.values(), ids=lambda vendor: vendor.slug)
def test_every_vendor_can_be_drawn(vendor: Vendor) -> None:
    """Glyph or monogram, a badge always has something in it.

    The monogram is required even for a vendor that has a glyph: it is what the
    badge falls back to if the mark is ever withdrawn, which is exactly how
    OpenAI and IBM came to need one.
    """
    assert vendor.monogram.strip()
    assert len(vendor.monogram) <= MAXIMUM_MONOGRAM_LENGTH
    assert vendor.glyph in ("", vendor.slug)


def test_each_glyph_vendor_has_a_macro_to_draw() -> None:
    """A slug in GLYPH_VENDORS with no macro renders an empty badge."""
    macros = MACROS.read_text(encoding="utf-8")

    for slug in GLYPH_VENDORS:
        assert f"{{% macro {slug}() %}}" in macros
        assert f'vendor.glyph == "{slug}"' in macros


def test_each_vendor_has_a_brand_colour() -> None:
    """Without a rule the badge inherits the muted default and reads as disabled."""
    stylesheet = STYLESHEET.read_text(encoding="utf-8")
    coloured = set(re.findall(r"\.vendor-([\w-]+) \{ --vendor-mark:", stylesheet))

    assert set(VENDORS) <= coloured


def test_the_glyphs_are_credited_where_they_came_from() -> None:
    """CC0 icon files, trademarks that are not ours: both have to be said once."""
    macros = MACROS.read_text(encoding="utf-8")

    assert "simple-icons" in macros.lower()
    assert "CC0" in macros
    assert "trademark" in macros.lower()


@pytest.mark.parametrize("apple_silicon", [True, False], ids=["apple-host", "linux-host"])
async def test_the_shipped_catalog_renders_a_badge_on_every_family(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, apple_silicon: bool
) -> None:
    """Every family the list shows carries a badge, on either kind of host.

    Both hosts, because they show different lists: a Linux gateway hides the
    Apple-only MLX families, so a count taken on a Mac is four too high there.
    What holds everywhere is the property — no tile without a mark — and both
    badge shapes appearing, which exercises the two halves of the macro.
    """
    probe = admin_queries.detect_system
    monkeypatch.setattr(
        admin_queries,
        "detect_system",
        lambda **kwargs: replace(probe(**kwargs), is_apple_silicon=apple_silicon),
    )
    app = create_app(settings)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        page = (
            await client.get("/ui/partials/models", headers={"Authorization": f"Bearer {TOKEN}"})
        ).text

    tiles = re.findall(r'<div class="family-tile".*?</button>', page, re.S)
    rendered = {re.search(r'data-family="([^"]+)"', tile).group(1) for tile in tiles}  # type: ignore[union-attr]
    assert rendered <= mapped_families()
    assert rendered
    for tile in tiles:
        assert "vendor-badge" in tile
        assert "vendor-glyph" in tile or "vendor-monogram" in tile

    assert 'class="vendor-glyph"' in page
    assert "vendor-monogram" in page
