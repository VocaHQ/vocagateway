from __future__ import annotations

import re
from html import escape
from urllib.parse import quote

from app.catalog import DEFAULT_CATALOG, LANGUAGE_NAMES
from app.fragments.engine import ENGINE_LABELS
from app.fragments.shared import _format_bytes
from app.schemas import AdminModelEntry
from app.templating import render

LIST_SEPARATOR = ", "

_SIZE_FILTER_OPTIONS: tuple[tuple[str, str], ...] = (
    ("", "Any size"),
    ("100mb", "Under 100 MB"),
    ("300mb", "Under 300 MB"),
    ("800mb", "Under 800 MB"),
    ("1500mb", "Under 1.5 GB"),
)

LANGUAGE_PREVIEW_COUNT = 4


class _FilterOptions:
    @classmethod
    def language_options(cls) -> list[tuple[str, str]]:
        covered = {code for model in DEFAULT_CATALOG for code in model.language_codes}
        named = sorted((LANGUAGE_NAMES.get(code, code), code) for code in covered)
        return [(code, name) for name, code in named]

    @classmethod
    def family_options(cls, entries: list[AdminModelEntry]) -> list[tuple[str, str]]:
        names = sorted({entry.family for entry in entries if entry.family}, key=str.lower)
        return [(name, name) for name in names]

    @classmethod
    def engine_options(cls, entries: list[AdminModelEntry]) -> list[tuple[str, str]]:
        engines = sorted({entry.engine for entry in entries if entry.engine})
        return [(eng, ENGINE_LABELS.get(eng, eng)) for eng in engines]


_language_filter_options = _FilterOptions.language_options


class _ModelCardView:
    @classmethod
    def display_label(cls, entry: AdminModelEntry, nested_in_family: bool) -> str:
        label = entry.label
        if not nested_in_family or not entry.family:
            return label
        fam = entry.family.strip()
        if not fam or not label.lower().startswith(fam.lower()):
            return label
        rest = label[len(fam) :]
        if not rest:
            return label
        if rest[0].isspace() or rest[0] in "/·-–—":
            stripped = rest.lstrip(" /·-–—")
            if stripped:
                return stripped
        return label

    @classmethod
    def info_id(cls, model_id: str) -> str:
        escaped = escape(model_id, quote=True).replace("%", "")
        sanitized = escaped.replace(":", "-")
        return f"model-info-{sanitized}"

    @classmethod
    def family_dom_id(cls, family: str) -> str:
        cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", family).strip("-")
        slug = cleaned.lower() or "family"
        return f"family-models-{slug}"

    @classmethod
    def language_disclosure(cls, entry: AdminModelEntry) -> dict[str, object] | None:
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

    @classmethod
    def model_card_context(
        cls,
        entry: AdminModelEntry,
        nested_in_family: bool = False,
        suppress_recommended: bool = False,
    ) -> dict[str, object]:
        if entry.state == "downloading":
            download: dict[str, object] | None = {
                "percent": round((entry.progress or 0) * 100),
                "downloaded": _format_bytes(entry.downloaded_bytes or 0),
                "total": _format_bytes(entry.total_bytes or 0),
            }
        else:
            download = None
        has_custom_lic = entry.license_name and entry.license_name != "See model source"
        return {
            "entry": entry,
            "encoded_id": quote(entry.id, safe=""),
            "display_label": cls.display_label(entry, nested_in_family=nested_in_family),
            "suppress_recommended": suppress_recommended,
            "size_label": _format_bytes(entry.size_bytes),
            "license_name": entry.license_name if has_custom_lic else None,
            "personal_use_only": not entry.commercial_use,
            "info_id": cls.info_id(entry.id),
            "language_disclosure": cls.language_disclosure(entry),
            "download": download,
        }

    @classmethod
    def family_context(cls, family: str, models: list[AdminModelEntry]) -> dict[str, object]:
        installed = sum(entry.state == "installed" for entry in models)
        active = next((entry for entry in models if entry.active), None)
        recommended = next((entry for entry in models if entry.recommended), None)
        engines = sorted({ENGINE_LABELS.get(entry.engine, entry.engine) for entry in models})
        return {
            "name": family,
            "dom_id": cls.family_dom_id(family),
            "count_label": cls._count_label(len(models)),
            "installed_label": f"{installed} installed" if installed else "none installed",
            "engine_note": LIST_SEPARATOR.join(engines),
            "is_active": active is not None,
            "is_recommended": recommended is not None,
            "cards": [
                cls.model_card_context(
                    model,
                    nested_in_family=True,
                    suppress_recommended=recommended is not None,
                )
                for model in models
            ],
        }

    @classmethod
    def _count_label(cls, count: int) -> str:
        return f"{count} model" if count == 1 else f"{count} models"


class _ModelListBuilder:
    def __init__(self, entries: list[AdminModelEntry], filter_args: dict[str, object]) -> None:
        self._entries = entries
        self._installed_only = bool(filter_args.get("installed_only", False))
        self._recommended_only = bool(filter_args.get("recommended_only", False))
        self._max_size = str(filter_args.get("max_size", ""))
        self._languages = self._resolve_list(
            filter_args.get("languages"), filter_args.get("language", "")
        )
        self._families = self._resolve_list(
            filter_args.get("families"), filter_args.get("family", "")
        )
        engines = filter_args.get("engines")
        self._engines = (
            [str(element) for element in engines] if isinstance(engines, (list, tuple)) else []
        )

    def render(self) -> str:
        groups: dict[str, list[AdminModelEntry]] = {}
        for entry in self._entries:
            name = entry.family or ENGINE_LABELS.get(entry.engine, entry.engine)
            groups.setdefault(name, []).append(entry)

        family_contexts = [
            _ModelCardView.family_context(name, family_models)
            for name, family_models in sorted(groups.items(), key=self._sort_key)
        ]
        return render(
            "models/list.html",
            families=family_contexts,
            language_hint=self._dictation_language_hint(),
            empty_message=None if family_contexts else self._empty_message(),
        )

    @classmethod
    def _resolve_list(cls, plural: object, singular: object) -> list[str]:
        if isinstance(plural, (list, tuple)):
            return [str(element) for element in plural]
        return _as_list(singular)

    @classmethod
    def _size_label(cls, max_size: str) -> str:
        for key, label in _SIZE_FILTER_OPTIONS:
            if key == max_size:
                return label.lower()
        return max_size.lower()

    def _dictation_language_hint(self) -> str:
        if len(self._languages) != 1:
            return ""
        code = self._languages[0]
        automatic = [entry for entry in self._entries if entry.detects_language_automatically]
        pinnable = [entry for entry in self._entries if not entry.detects_language_automatically]
        if not automatic or not pinnable:
            return ""
        return render(
            "models/language_hint.html",
            language_name=LANGUAGE_NAMES.get(code, code),
            pinnable_count=len(pinnable),
        )

    def _empty_message(self) -> str | None:
        bits: list[str] = []
        if self._families:
            bits.append(f"families {LIST_SEPARATOR.join(self._families)}")
        if self._languages:
            names = (LANGUAGE_NAMES.get(code, code) for code in self._languages)
            bits.append(LIST_SEPARATOR.join(names))
        if self._engines:
            labels = (ENGINE_LABELS.get(eng, eng) for eng in self._engines)
            bits.append(f"engines {LIST_SEPARATOR.join(labels)}")
        if self._max_size:
            bits.append(self._size_label(self._max_size))
        if self._recommended_only:
            bits.append("fits this machine")
        if self._installed_only:
            bits.append("installed only")
        return LIST_SEPARATOR.join(bits) if bits else None

    @classmethod
    def _sort_key(
        cls,
        family_entries: tuple[str, list[AdminModelEntry]],
    ) -> tuple[bool, bool, int, str]:
        name, family_models = family_entries
        has_active = any(entry.active for entry in family_models)
        has_recommended = any(entry.recommended for entry in family_models)
        installed_count = sum(entry.state == "installed" for entry in family_models)
        return (not has_active, not has_recommended, -installed_count, name.lower())


def _as_list(selected_values: object) -> list[str]:
    if selected_values is None:
        return []
    if isinstance(selected_values, str):
        return [selected_values] if selected_values else []
    if isinstance(selected_values, (list, tuple)):
        return [str(entry) for entry in selected_values if entry]
    return []


def _request_model_issue_url() -> str:
    title = "Request: add a speech model to the catalog"
    body = (
        "## Model request\n\n"
        "**Name / id:**\n"
        "**Engine** (sherpa-onnx, faster-whisper, whisper.cpp, moonshine, mlx-audio, other):\n"
        "**Download URL** (direct file or official release):\n"
        "**Languages:**\n"
        "**Why it should ship in the catalog:**\n\n"
        "Thanks.\n"
    )
    return (
        f"https://github.com/VocaHQ/vocagateway/issues/new?title={quote(title)}&body={quote(body)}"
    )


def models_fragment(entries: list[AdminModelEntry]) -> str:
    installed = sum(entry.state == "installed" for entry in entries)
    families = len({entry.family for entry in entries if entry.family})
    return render(
        "models/page.html",
        installed=installed,
        total=len(entries),
        families=families,
        family_options=_FilterOptions.family_options(entries),
        language_options=_FilterOptions.language_options(),
        engine_options=_FilterOptions.engine_options(entries),
        size_options=_SIZE_FILTER_OPTIONS,
        list_html=models_list_fragment(entries),
        request_model_issue_url=_request_model_issue_url(),
    )


def models_list_fragment(
    entries: list[AdminModelEntry],
    **filter_args: object,
) -> str:
    builder = _ModelListBuilder(entries, filter_args)
    return builder.render()
