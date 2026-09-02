from __future__ import annotations

import re
from html import escape
from urllib.parse import quote

from app.catalog import DEFAULT_CATALOG, LANGUAGE_NAMES
from app.fragments.engine import ENGINE_LABELS
from app.fragments.shared import _format_bytes
from app.schemas import AdminModelEntry
from app.templating import render

# Download-size caps shown in the filter panel (keys match admin_queries).
_SIZE_FILTER_OPTIONS: tuple[tuple[str, str], ...] = (
    ("", "Any size"),
    ("100mb", "Under 100 MB"),
    ("300mb", "Under 300 MB"),
    ("800mb", "Under 800 MB"),
    ("1500mb", "Under 1.5 GB"),
)

# How many languages to name on the closed card. Enough to recognise whether a
# model is worth opening, short enough not to push the buttons off screen.
LANGUAGE_PREVIEW_COUNT = 4


def models_fragment(entries: list[AdminModelEntry]) -> str:
    installed = sum(entry.state == "installed" for entry in entries)
    families = len({entry.family for entry in entries if entry.family})
    return render(
        "models/page.html",
        installed=installed,
        total=len(entries),
        families=families,
        family_options=_family_filter_options(entries),
        language_options=_language_filter_options(),
        engine_options=_engine_filter_options(entries),
        size_options=_SIZE_FILTER_OPTIONS,
        list_html=models_list_fragment(entries),
        request_model_issue_url=_request_model_issue_url(),
    )


def _request_model_issue_url() -> str:
    """Pre-filled GitHub issue to ask for a new catalog entry on vocagateway."""
    title = "Request: add a speech model to the catalog"
    body = """## Model request

**Name / id:**
**Engine** (sherpa-onnx, faster-whisper, whisper.cpp, moonshine, mlx-audio, other):
**Download URL** (direct file or official release):
**Languages:**
**Why it should ship in the catalog:**

Thanks.
"""
    return (
        f"https://github.com/VocaHQ/vocagateway/issues/new?title={quote(title)}&body={quote(body)}"
    )


def _language_filter_options() -> list[tuple[str, str]]:
    """Only languages some model actually covers, so the filter never has a
    dead option. Whisper's set dominates, but Dolphin contributes the South and
    Southeast Asian languages Whisper lacks."""
    covered = {code for model in DEFAULT_CATALOG for code in model.language_codes}
    named = sorted((LANGUAGE_NAMES.get(code, code), code) for code in covered)
    return [(code, name) for name, code in named]


def _family_filter_options(entries: list[AdminModelEntry]) -> list[tuple[str, str]]:
    """Families present in the live catalogue view (includes custom when listed)."""
    names = sorted({entry.family for entry in entries if entry.family}, key=str.lower)
    return [(name, name) for name in names]


def _engine_filter_options(entries: list[AdminModelEntry]) -> list[tuple[str, str]]:
    """Engines present in the live catalogue view."""
    engines = sorted({entry.engine for entry in entries if entry.engine})
    return [(eng, ENGINE_LABELS.get(eng, eng)) for eng in engines]


def models_list_fragment(
    entries: list[AdminModelEntry],
    installed_only: bool = False,
    language: str | list[str] = "",
    family: str | list[str] = "",
    languages: list[str] | None = None,
    families: list[str] | None = None,
    engines: list[str] | None = None,
    max_size: str = "",
    recommended_only: bool = False,
) -> str:
    """Family tiles: always collapsed for density; open one to browse its models.

    `language` / `family` still accept a single string for older call sites; prefer
    the plural list kwargs.
    """
    lang_list = list(languages) if languages is not None else _as_list(language)
    fam_list = list(families) if families is not None else _as_list(family)
    eng_list = list(engines or [])

    groups: dict[str, list[AdminModelEntry]] = {}
    for entry in entries:
        name = entry.family or ENGINE_LABELS.get(entry.engine, entry.engine)
        groups.setdefault(name, []).append(entry)

    def family_sort_key(
        item: tuple[str, list[AdminModelEntry]],
    ) -> tuple[bool, bool, int, str]:
        name, items = item
        has_active = any(entry.active for entry in items)
        has_recommended = any(entry.recommended for entry in items)
        installed_count = sum(entry.state == "installed" for entry in items)
        # Active family first, then recommended, then most installed, then name.
        return (not has_active, not has_recommended, -installed_count, name.lower())

    family_contexts = [
        _family_context(name, items) for name, items in sorted(groups.items(), key=family_sort_key)
    ]
    return render(
        "models/list.html",
        families=family_contexts,
        language_hint=_dictation_language_hint(entries, lang_list),
        empty_message=None
        if family_contexts
        else _empty_filter_message(
            installed_only=installed_only,
            languages=lang_list,
            families=fam_list,
            engines=eng_list,
            max_size=max_size,
            recommended_only=recommended_only,
        ),
    )


def _as_list(value: str | list[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    return [item for item in value if item]


def _empty_filter_message(
    *,
    installed_only: bool,
    languages: list[str],
    families: list[str],
    engines: list[str] | None = None,
    max_size: str = "",
    recommended_only: bool = False,
) -> str | None:
    bits: list[str] = []
    if families:
        bits.append("families " + ", ".join(families))
    if languages:
        bits.append(", ".join(LANGUAGE_NAMES.get(code, code) for code in languages))
    if engines:
        bits.append("engines " + ", ".join(ENGINE_LABELS.get(eng, eng) for eng in engines))
    if max_size:
        label = next((lab for key, lab in _SIZE_FILTER_OPTIONS if key == max_size), max_size)
        bits.append(label.lower())
    if recommended_only:
        bits.append("fits this machine")
    if installed_only:
        bits.append("installed only")
    return ", ".join(bits) if bits else None


def _family_dom_id(family: str) -> str:
    """Stable id for aria-controls / grid sibling panel."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", family).strip("-").lower() or "family"
    return f"family-models-{slug}"


def _family_context(family: str, items: list[AdminModelEntry]) -> dict[str, object]:
    installed = sum(entry.state == "installed" for entry in items)
    active = next((entry for entry in items if entry.active), None)
    recommended = next((entry for entry in items if entry.recommended), None)
    engines = sorted({ENGINE_LABELS.get(entry.engine, entry.engine) for entry in items})
    count = f"{len(items)} model" + ("" if len(items) == 1 else "s")
    return {
        "name": family,
        "dom_id": _family_dom_id(family),
        "count_label": count,
        "installed_label": f"{installed} installed" if installed else "none installed",
        "engine_note": ", ".join(engines),
        # Family-level status once. Active uses a check label (not the same pill
        # as recommended); nested cards skip repeating "recommended".
        "is_active": active is not None,
        "is_recommended": recommended is not None,
        "cards": [
            _model_card_context(
                item,
                nested_in_family=True,
                suppress_recommended=recommended is not None,
            )
            for item in items
        ],
    }


def _dictation_language_hint(entries: list[AdminModelEntry], language: str | list[str]) -> str:
    """Warn when a filtered list mixes pinnable and auto-detecting models.

    Only fires for a single selected language (multi-select is too broad); see
    app/templates/models/language_hint.html for the full rationale.
    """
    codes = _as_list(language)
    if len(codes) != 1:
        return ""
    code = codes[0]
    automatic = [entry for entry in entries if entry.detects_language_automatically]
    pinnable = [entry for entry in entries if not entry.detects_language_automatically]
    if not automatic or not pinnable:
        return ""
    return render(
        "models/language_hint.html",
        language_name=LANGUAGE_NAMES.get(code, code),
        pinnable_count=len(pinnable),
    )


def _display_label(entry: AdminModelEntry, *, nested_in_family: bool) -> str:
    """Drop a leading family name when the card already sits under that family.

    Catalog labels often start with the family ("Parakeet TDT 0.6B v3 INT8");
    under an open family shell that prefix is noise.

    Only strip on a clean boundary (space / separator). Do not peel letters out of a
    longer token — "Whisper" must not turn "whisper.cpp Tiny" into ".cpp Tiny" or
    "WhisperKit Base" into "Kit Base".
    """
    label = entry.label
    if not nested_in_family or not entry.family:
        return label
    fam = entry.family.strip()
    if not fam or not label.lower().startswith(fam.lower()):
        return label
    rest = label[len(fam) :]
    if not rest:
        return label
    # Boundary: whitespace or a short list of separators — not '.' or more letters.
    if rest[0].isspace() or rest[0] in "/·-–—":
        stripped = rest.lstrip(" /·-–—")
        if stripped:
            return stripped
    return label


def _language_disclosure_context(entry: AdminModelEntry) -> dict[str, object] | None:
    names = entry.language_names
    if len(names) < 2:
        return None
    show_all = len(names) <= LANGUAGE_PREVIEW_COUNT + 1
    return {
        "show_all": show_all,
        "names": names,
        "preview": None if show_all else names[:LANGUAGE_PREVIEW_COUNT],
        "remaining_count": None if show_all else len(names) - LANGUAGE_PREVIEW_COUNT,
        "auto": entry.detects_language_automatically,
    }


def _model_card_context(
    entry: AdminModelEntry,
    *,
    nested_in_family: bool = False,
    suppress_recommended: bool = False,
) -> dict[str, object]:
    if entry.state == "downloading":
        download = {
            "percent": round((entry.progress or 0) * 100),
            "downloaded": _format_bytes(entry.downloaded_bytes or 0),
            "total": _format_bytes(entry.total_bytes or 0),
        }
    else:
        download = None
    # This id only needs to be a valid, stable DOM id — not HTML-safe text — so
    # it is sanitized (not escaped-for-display) once here rather than in the
    # template.
    info_id = f"model-info-{escape(entry.id, quote=True).replace('%', '').replace(':', '-')}"
    return {
        "entry": entry,
        "encoded_id": quote(entry.id, safe=""),
        "display_label": _display_label(entry, nested_in_family=nested_in_family),
        "suppress_recommended": suppress_recommended,
        "size_label": _format_bytes(entry.size_bytes),
        "license_name": (
            entry.license_name
            if entry.license_name and entry.license_name != "See model source"
            else None
        ),
        "personal_use_only": not entry.commercial_use,
        "info_id": info_id,
        "language_disclosure": _language_disclosure_context(entry),
        "download": download,
    }
