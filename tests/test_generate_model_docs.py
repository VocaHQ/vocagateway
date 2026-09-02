from __future__ import annotations

import importlib.util
import re
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "generate_model_docs.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("generate_model_docs", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_set_anchor_differs_for_same_length_aa() -> None:
    """Reproduces the reported bug: the anchor used to be `language-set-{len(codes)}`,
    so any two distinct language sets of the same length collided into one HTML id
    and one link target — silently pointing readers at the wrong section."""
    module = _load_module()
    a = tuple(f"lang-{i}" for i in range(12))
    b = tuple(f"other-{i}" for i in range(12))
    assert len(a) == len(b)
    assert module.set_anchor(a) != module.set_anchor(b)


def test_set_anchor_is_stable_for_the_same_codes() -> None:
    module = _load_module()
    codes = ("en", "fr", "de")
    assert module.set_anchor(codes) == module.set_anchor(codes)


def test_rendered_language_set_anchors_are_be2db() -> None:
    """Guards the live catalog too: every `<a id="language-set-...">` emitted by
    render() must be distinct, and every link must point at an anchor that exists."""
    module = _load_module()
    rendered = module.render()
    anchor_ids = re.findall(r'<a id="(language-set-[^"]+)"></a>', rendered)
    assert len(anchor_ids) == len(set(anchor_ids))

    linked_targets = re.findall(r"\]\(#(language-set-[^)]+)\)", rendered)
    assert set(linked_targets) <= set(anchor_ids)
