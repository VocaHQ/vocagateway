from __future__ import annotations

import re
from html import escape
from urllib.parse import quote

from app.catalog import DEFAULT_CATALOG, LANGUAGE_NAMES
from app.fragments.engine import ENGINE_LABELS
from app.fragments.shared import _format_bytes
from app.schemas import AdminModelEntry

# Every action that re-renders the model list sends all filters, so downloading
# or selecting a model never silently resets the view. The filter form holds
# multi-select checkboxes (family / language / engine) plus size and toggles.
MODEL_FILTER_INPUTS = "#models-filter-form"

# Download-size caps shown in the filter panel (keys match admin_queries).
_SIZE_FILTER_OPTIONS: tuple[tuple[str, str], ...] = (
    ("", "Any size"),
    ("100mb", "Under 100 MB"),
    ("300mb", "Under 300 MB"),
    ("800mb", "Under 800 MB"),
    ("1500mb", "Under 1.5 GB"),
)

_FILTER_ICON = (
    '<svg class="filter-icon" viewBox="0 0 24 24" width="16" height="16" fill="none" '
    'stroke="currentColor" stroke-width="1.75" stroke-linecap="round" '
    'stroke-linejoin="round" aria-hidden="true">'
    '<path d="M4 5h16l-6 7.5V19l-4 2v-8.5L4 5z"/></svg>'
)
_EXPAND_ICON = (
    '<svg class="toolbar-icon icon-expand" viewBox="0 0 24 24" width="18" height="18" fill="none" '
    'stroke="currentColor" stroke-width="1.75" stroke-linecap="round" '
    'stroke-linejoin="round" aria-hidden="true">'
    '<path d="m7 15 5 5 5-5M7 9l5-5 5 5"/></svg>'
)
_COLLAPSE_ICON = (
    '<svg class="toolbar-icon icon-collapse" viewBox="0 0 24 24" '
    'width="18" height="18" fill="none" '
    'stroke="currentColor" stroke-width="1.75" stroke-linecap="round" '
    'stroke-linejoin="round" aria-hidden="true">'
    '<path d="m7 20 5-5 5 5M7 4l5 5 5-5"/></svg>'
)
_REFRESH_ICON = (
    '<svg class="toolbar-icon" viewBox="0 0 24 24" width="18" height="18" fill="none" '
    'stroke="currentColor" stroke-width="1.75" stroke-linecap="round" '
    'stroke-linejoin="round" aria-hidden="true">'
    '<path d="M21 12a9 9 0 0 0-9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/>'
    '<path d="M3 3v5h5"/>'
    '<path d="M3 12a9 9 0 0 0 9 9 9.75 9.75 0 0 0 6.74-2.74L21 16"/>'
    '<path d="M16 16h5v5"/></svg>'
)


def models_fragment(entries: list[AdminModelEntry]) -> str:
    installed = sum(entry.state == "installed" for entry in entries)
    families = len({entry.family for entry in entries if entry.family})
    return f"""
      <div class="page-head">
        <div>
          <h2>Models</h2>
          <p>{installed} of {len(entries)} downloaded across {families} families,
             all stored on this machine. Families start collapsed; open one to pick a model.</p>
        </div>
        <div class="models-toolbar-actions">
          <button type="button" class="ghost filter-trigger" id="filter-rail-toggle"
                  aria-controls="models-filter" aria-expanded="true"
                  title="Hide filters">
            {_FILTER_ICON}
            <span class="filter-trigger-label">Filters</span>
            <span id="filter-active-count" class="filter-count hidden" aria-live="polite"></span>
          </button>
          <button type="button" class="ghost icon-btn families-expand-toggle"
                  id="families-expand-toggle"
                  data-expanded="false"
                  aria-controls="models-list"
                  aria-expanded="false"
                  aria-label="Expand all families"
                  title="Expand all families">
            {_EXPAND_ICON}{_COLLAPSE_ICON}
          </button>
          <button type="button" class="ghost icon-btn models-refresh"
                  aria-label="Refresh models"
                  title="Refresh models"
                  hx-get="/ui/partials/models-list" hx-include="{MODEL_FILTER_INPUTS}"
                  hx-target="#models-list" hx-swap="innerHTML">{_REFRESH_ICON}</button>
        </div>
      </div>
      <div class="models-layout" id="models-layout">
        <aside class="models-filter" id="models-filter" aria-label="Model filters">
          <div class="filter-side-head">
            <h3 class="filter-side-title">Filters</h3>
            <button type="button" class="ghost icon-btn filter-side-collapse"
                    id="filter-side-collapse"
                    aria-controls="models-filter"
                    aria-expanded="true"
                    aria-label="Collapse filters"
                    title="Collapse filters">
              <svg class="toolbar-icon" viewBox="0 0 24 24" width="18" height="18" fill="none"
                   stroke="currentColor" stroke-width="1.75" stroke-linecap="round"
                   stroke-linejoin="round" aria-hidden="true">
                <path d="M15 6 9 12l6 6"/>
              </svg>
            </button>
          </div>
          <form id="models-filter-form" class="filter-panel" role="group"
                aria-label="Model filter options"
                hx-get="/ui/partials/models-list"
                hx-trigger="change from:input[name], change from:select[name]"
                hx-target="#models-list" hx-swap="innerHTML"
                hx-include="this">
            <fieldset class="filter-field">
              <legend>Family</legend>
              <div class="filter-check-list" id="family-filter-list">
                {_family_filter_options(entries)}
              </div>
            </fieldset>
            <fieldset class="filter-field">
              <legend>Language</legend>
              <input type="search" class="filter-search" data-filter-search="language-filter-list"
                     placeholder="Search languages" autocomplete="off" spellcheck="false"
                     aria-label="Search languages" />
              <div class="filter-check-list filter-check-list-tall" id="language-filter-list">
                {_language_filter_options()}
              </div>
            </fieldset>
            <fieldset class="filter-field">
              <legend>Engine</legend>
              <div class="filter-check-list filter-check-list-engines" id="engine-filter-list">
                {_engine_filter_options(entries)}
              </div>
            </fieldset>
            <label class="filter-field">
              <span>Max download size</span>
              <select id="max-size-filter" name="max_size" class="max-size-filter">
                {_size_filter_options()}
              </select>
            </label>
            <div class="filter-toggles">
              <label class="filter-toggle" for="installed-only-toggle">
                <input type="checkbox" id="installed-only-toggle" name="installed_only"
                       value="true" class="sr-only" />
                Installed only
              </label>
              <label class="filter-toggle" for="recommended-only-toggle"
                     title="Models picked for this machine (RAM and platform)">
                <input type="checkbox" id="recommended-only-toggle" name="recommended_only"
                       value="true" class="sr-only" />
                Fits this machine
              </label>
            </div>
            <button type="button" class="ghost small filter-clear" id="filter-clear">
              Clear filters
            </button>
          </form>
        </aside>
        <div class="models-main">
          <div id="models-list" aria-live="polite">
            {models_list_fragment(entries)}
          </div>
          <div class="card">
            <h2>Bring your own Whisper model</h2>
            <p class="muted">Paste a direct HTTPS link to a <code>.bin</code> or
               <code>.gguf</code> file. It runs through the standalone engine.
               Use a <code>/resolve/</code> URL on Hugging Face, not the repo home page.</p>
            <form hx-post="/ui/partials/models/custom" hx-include="{MODEL_FILTER_INPUTS}"
                  hx-target="#models-list" hx-swap="innerHTML">
              <div class="row">
                <input name="url" type="url" required
                       placeholder="https://huggingface.co/&hellip;/resolve/main/model.gguf" />
                <button type="submit" class="primary">Download</button>
              </div>
              <div class="row">
                <input name="sha256" type="text" spellcheck="false"
                       pattern="\\s*(?:[Ss][Hh][Aa]256:)?[0-9a-fA-F]{{64}}\\s*"
                       title="64 hexadecimal characters, optionally prefixed with sha256:"
                       placeholder="Optional SHA-256 from the model card (verified on download)" />
              </div>
            </form>
            <p class="muted small">Catalog models are verified automatically. For
               your own URL, paste the digest the model card publishes and the
               download is discarded unless it matches.</p>
            <p class="muted small model-custom-links">
              <a href="https://huggingface.co/models?pipeline_tag=automatic-speech-recognition&amp;library=gguf&amp;sort=trending"
                 target="_blank" rel="noopener noreferrer">Browse GGUF speech models</a>
              <span aria-hidden="true">·</span>
              <a href="{_request_model_issue_url()}" target="_blank"
                 rel="noopener noreferrer">Request a catalog model</a>
            </p>
          </div>
        </div>
      </div>
    """


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


def _filter_check(name: str, value: str, label: str) -> str:
    return (
        f'<label class="filter-check">'
        f'<input type="checkbox" name="{escape(name, quote=True)}" '
        f'value="{escape(value, quote=True)}" />'
        f"<span>{escape(label)}</span></label>"
    )


def _language_filter_options() -> str:
    """Only languages some model actually covers, so the filter never has a
    dead option. Whisper's set dominates, but Dolphin contributes the South and
    Southeast Asian languages Whisper lacks."""
    covered = {code for model in DEFAULT_CATALOG for code in model.language_codes}
    named = sorted((LANGUAGE_NAMES.get(code, code), code) for code in covered)
    return "".join(_filter_check("language", code, name) for name, code in named)


def _family_filter_options(entries: list[AdminModelEntry]) -> str:
    """Families present in the live catalogue view (includes custom when listed)."""
    names = sorted({entry.family for entry in entries if entry.family}, key=str.lower)
    return "".join(_filter_check("family", name, name) for name in names)


def _engine_filter_options(entries: list[AdminModelEntry]) -> str:
    """Engines present in the live catalogue view."""
    engines = sorted({entry.engine for entry in entries if entry.engine})
    return "".join(_filter_check("engine", eng, ENGINE_LABELS.get(eng, eng)) for eng in engines)


def _size_filter_options() -> str:
    return "".join(
        f'<option value="{escape(value, quote=True)}">{escape(label)}</option>'
        for value, label in _SIZE_FILTER_OPTIONS
    )


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

    parts: list[str] = []
    for name, items in sorted(groups.items(), key=family_sort_key):
        parts.append(_family_tile(name, items))
    if parts:
        return (
            _dictation_language_hint(entries, lang_list)
            + f'<div class="family-grid">{"".join(parts)}</div>'
        )
    return _empty_filter_state(
        installed_only=installed_only,
        languages=lang_list,
        families=fam_list,
        engines=eng_list,
        max_size=max_size,
        recommended_only=recommended_only,
    )


def _as_list(value: str | list[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    return [item for item in value if item]


def _empty_filter_state(
    *,
    installed_only: bool,
    languages: list[str],
    families: list[str],
    engines: list[str] | None = None,
    max_size: str = "",
    recommended_only: bool = False,
) -> str:
    bits: list[str] = []
    if families:
        bits.append("families " + ", ".join(escape(f) for f in families))
    if languages:
        bits.append(", ".join(escape(LANGUAGE_NAMES.get(code, code)) for code in languages))
    if engines:
        bits.append("engines " + ", ".join(escape(ENGINE_LABELS.get(eng, eng)) for eng in engines))
    if max_size:
        label = next((lab for key, lab in _SIZE_FILTER_OPTIONS if key == max_size), max_size)
        bits.append(escape(label.lower()))
    if recommended_only:
        bits.append("fits this machine")
    if installed_only:
        bits.append("installed only")
    if bits:
        joined = ", ".join(bits)
        return (
            f'<p class="empty-state">No models match {joined}. '
            "Clear or widen the filters to see more.</p>"
        )
    return '<p class="empty-state">No models available.</p>'


def _family_dom_id(family: str) -> str:
    """Stable id for aria-controls / grid sibling panel."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", family).strip("-").lower() or "family"
    return f"family-models-{slug}"


def _family_tile(family: str, items: list[AdminModelEntry]) -> str:
    installed = sum(entry.state == "installed" for entry in items)
    active = next((entry for entry in items if entry.active), None)
    recommended = next((entry for entry in items if entry.recommended), None)
    count = f"{len(items)} model" + ("" if len(items) == 1 else "s")
    installed_label = f"{installed} installed" if installed else "none installed"
    engines = sorted({ENGINE_LABELS.get(entry.engine, entry.engine) for entry in items})
    engine_note = escape(", ".join(engines))
    # Family-level status once. Active uses a check label (not the same pill as
    # recommended); nested cards skip repeating "recommended".
    tags = ""
    if active:
        tags = _active_label()
    elif recommended:
        tags = '<span class="badge recommended">recommended</span>'
    cards = "".join(
        _model_card(
            item,
            nested_in_family=True,
            suppress_recommended=bool(recommended),
        )
        for item in items
    )
    active_attr = ' data-active="1"' if active else ""
    models_id = _family_dom_id(family)
    chevron = (
        '<svg class="family-expand-icon" viewBox="0 0 24 24" width="18" height="18" '
        'fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" '
        'stroke-linejoin="round" aria-hidden="true">'
        '<path d="m6 9 6 6 6-6"/></svg>'
    )
    # Tile + models are *siblings* in the family-grid so the open models panel
    # can take grid-column: 1 / -1 on the next row while the tile stays put.
    # (Nested <details> + display:contents is unreliable across browsers.)
    return f"""
      <div class="family-tile"{active_attr} data-family="{escape(family, quote=True)}">
        <button type="button" class="family-summary"
                aria-expanded="false" aria-controls="{escape(models_id, quote=True)}">
          <span class="family-summary-body">
            <span class="family-summary-main">
              <span class="family-name" title="{escape(family, quote=True)}">{escape(family)}</span>
              <span class="family-tags">{tags}</span>
            </span>
            <span class="family-meta">
              <span>{count}</span>
              <span aria-hidden="true">·</span>
              <span>{installed_label}</span>
              <span class="family-engines" title="Engines in this family">{engine_note}</span>
            </span>
          </span>
          {chevron}
        </button>
      </div>
      <div class="family-models" id="{escape(models_id, quote=True)}" role="list"
           data-family="{escape(family, quote=True)}" hidden>
        <div class="family-models-head">
          <span class="family-models-label">{escape(family)}</span>
          <span class="family-models-count muted small">{count}</span>
        </div>
        <div class="family-models-grid">{cards}</div>
      </div>
    """


def _active_label() -> str:
    """Check + Active — solid treatment, not the outline recommended pill."""
    return (
        '<span class="model-active-label" title="Currently selected model">'
        '<svg viewBox="0 0 24 24" width="14" height="14" fill="none" '
        'stroke="currentColor" stroke-width="2.25" stroke-linecap="round" '
        'stroke-linejoin="round" aria-hidden="true">'
        '<path d="M20 6 9 17l-5-5"/></svg>'
        "Active"
        "</span>"
    )


def _dictation_language_hint(entries: list[AdminModelEntry], language: str | list[str]) -> str:
    """Warn when a filtered list mixes pinnable and auto-detecting models.

    Testing showed every auto-detecting model returns the wrong writing system on
    a short phrase — "ठीक है" came back as Chinese from both Dolphin and
    Qwen3-ASR. Pinning the language fixes it outright, so someone filtering for
    their language needs to know which half of the list to trust for dictation,
    which is mostly short phrases.

    Only fires for a single selected language (multi-select is too broad).
    """
    codes = _as_list(language)
    if len(codes) != 1:
        return ""
    code = codes[0]
    automatic = [entry for entry in entries if entry.detects_language_automatically]
    pinnable = [entry for entry in entries if not entry.detects_language_automatically]
    if not automatic or not pinnable:
        return ""
    name = escape(LANGUAGE_NAMES.get(code, code))
    return (
        '<div class="callout warning models-language-hint">'
        f"<strong>Picking a model for {name}</strong>"
        "<span>Models tagged <em>auto language</em> pick the language themselves. "
        "They do well on full sentences but often guess the wrong script on short "
        "phrases (most of dictation). Prefer one of the other "
        f"{len(pinnable)} models here, which are forced to use {name}.</span>"
        "</div>"
    )


# How many languages to name on the closed card. Enough to recognise whether a
# model is worth opening, short enough not to push the buttons off screen.
LANGUAGE_PREVIEW_COUNT = 4


def _language_chips(names: list[str]) -> str:
    return "".join(f'<span class="model-language-chip">{escape(name)}</span>' for name in names)


def _language_disclosure(entry: AdminModelEntry) -> str:
    """Name the languages behind a summary like "25 European languages".

    The first few are shown on the closed card so the common question — "does
    this speak my language?" — is answerable without clicking every model in
    the list. The rest stay behind the toggle, as chips in a scrollable wrap
    rather than a tall comma list that blows up the card.
    """
    names = entry.language_names
    if len(names) < 2:
        return ""
    note = (
        '<p class="model-language-note muted small">'
        "This model picks the language itself; these are languages it handles well, "
        "not options you can force.</p>"
        if entry.detects_language_automatically
        else ""
    )
    if len(names) <= LANGUAGE_PREVIEW_COUNT + 1:
        # Show them all rather than offer "+1 more": a toggle that hides one
        # chip costs a click and saves nothing.
        return f"""
        <div class="model-languages">
          {note}
          <div class="model-language-list">{_language_chips(names)}</div>
        </div>
    """
    preview, remaining = names[:LANGUAGE_PREVIEW_COUNT], names[LANGUAGE_PREVIEW_COUNT:]
    return f"""
        <details class="model-languages">
          <summary>
            <span class="model-language-preview">{_language_chips(preview)}</span>
            <span class="model-language-more">+{len(remaining)} more</span>
            <span class="model-language-less">Show fewer</span>
          </summary>
          {note}
          <div class="model-language-list">{_language_chips(names)}</div>
        </details>
    """


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


def _model_card(
    entry: AdminModelEntry,
    *,
    nested_in_family: bool = False,
    suppress_recommended: bool = False,
) -> str:
    encoded_id = quote(entry.id, safe="")
    # Active uses a check label (card wash too). Recommended keeps the outline
    # pill. Family already shows recommended when nested, so skip repeating it.
    badges = ""
    if entry.active:
        badges += _active_label()
    elif entry.recommended and not suppress_recommended:
        badges += '<span class="badge recommended">recommended</span>'
    if entry.detects_language_automatically:
        badges += (
            '<span class="badge auto-language" title="This model decides the language itself.'
            ' The language chosen in the app does not constrain it.">auto language</span>'
        )
    # Short scan line on the card; family lives as a toolbar pill (not here).
    meta: list[str] = [
        _format_bytes(entry.size_bytes),
        escape(entry.languages),
        escape(entry.quality),
    ]
    display_label = _display_label(entry, nested_in_family=nested_in_family)
    family_pill = (
        f'<span class="model-family-pill" title="Family">{escape(entry.family)}</span>'
        if entry.family
        else ""
    )
    # Source is a real link when we know a project/HF page. The default license
    # string "See model source" is a placeholder, not a link — omit it then.
    if entry.source_url:
        source_html = (
            f'<a class="model-source-link" href="{escape(entry.source_url, quote=True)}" '
            f'target="_blank" rel="noopener noreferrer">{escape(entry.source)}</a>'
        )
    else:
        source_html = escape(entry.source)
    detail_bits: list[str] = [source_html]
    if entry.license_name and entry.license_name != "See model source":
        detail_bits.append(escape(entry.license_name))
    if entry.supports_streaming:
        detail_bits.append("streams live")
    if not entry.commercial_use:
        detail_bits.append("personal use only")
    detail_meta = " · ".join(detail_bits)

    if entry.state == "downloading":
        percent = round((entry.progress or 0) * 100)
        downloaded = _format_bytes(entry.downloaded_bytes or 0)
        total = _format_bytes(entry.total_bytes or 0)
        action = (
            f'<div class="model-download-status">'
            f'<div class="progress" role="progressbar" aria-valuemin="0" aria-valuemax="100"'
            f' aria-valuenow="{percent}"><div class="bar" style="width:{percent}%"></div></div>'
            f'<div class="model-download-meta">'
            f'<span class="muted small progress-copy">'
            f"{percent}% · {downloaded} / {total}</span>"
            f'<button class="ghost small" hx-post="/ui/partials/models/{encoded_id}/cancel"'
            f' hx-include="{MODEL_FILTER_INPUTS}"'
            f' hx-target="#models-list" hx-swap="innerHTML">Cancel</button>'
            f"</div></div>"
        )
    elif entry.state == "installed":
        select_button = (
            ""
            if entry.active
            else (
                f'<button class="primary small"'
                f' hx-post="/ui/partials/models/{encoded_id}/select"'
                f' hx-include="{MODEL_FILTER_INPUTS}"'
                f' hx-target="#models-list" hx-swap="innerHTML">Select</button>'
            )
        )
        action = (
            f"{select_button}"
            f'<button class="ghost small danger"'
            f' hx-delete="/ui/partials/models/{encoded_id}"'
            f' hx-include="{MODEL_FILTER_INPUTS}"'
            f' hx-confirm="Delete {escape(entry.label)} from this server?"'
            f' hx-target="#models-list" hx-swap="innerHTML">Delete</button>'
        )
    else:
        note = (
            f'<p class="error small model-action-error">Download failed: {escape(entry.error)}</p>'
            if entry.error
            else ""
        )
        # Ghost, not primary: green is reserved for Select.
        action = (
            f"{note}"
            f'<button class="ghost small"'
            f' hx-post="/ui/partials/models/{encoded_id}/download"'
            f' hx-include="{MODEL_FILTER_INPUTS}"'
            f' hx-target="#models-list" hx-swap="innerHTML">Download</button>'
        )

    # Info icon (left) + actions (right). Hover/focus shows a popover — no
    # in-card accordion that stretches tile height.
    info_id = f"model-info-{escape(entry.id, quote=True).replace('%', '').replace(':', '-')}"
    info_icon = (
        '<svg viewBox="0 0 24 24" width="16" height="16" fill="none" '
        'stroke="currentColor" stroke-width="1.75" stroke-linecap="round" '
        'stroke-linejoin="round" aria-hidden="true">'
        '<circle cx="12" cy="12" r="9"/>'
        '<path d="M12 11v5M12 8h.01"/></svg>'
    )
    active_attr = ' data-active="1"' if entry.active else ""
    return f"""
      <article class="model-card" role="listitem"
               data-model-id="{escape(entry.id, quote=True)}"
               data-state="{entry.state}"{active_attr}>
        <div class="model-main">
          <div class="model-title">
            <h4 title="{escape(entry.label, quote=True)}">{escape(display_label)}</h4>
            <span class="model-tags">{badges}</span>
          </div>
          <p class="model-meta">{" &middot; ".join(meta)}</p>
          <p class="model-blurb" title="{escape(entry.description, quote=True)}">
            {escape(entry.description)}
          </p>
        </div>
        <div class="model-toolbar">
          <div class="model-toolbar-start">
            <div class="model-info">
              <button type="button" class="model-info-btn" aria-label="More model details"
                      aria-describedby="{info_id}" title="Source, license, languages">
                {info_icon}
              </button>
              <div class="model-info-pop" id="{info_id}" role="tooltip">
                <p class="model-detail-meta">{detail_meta}</p>
                {_language_disclosure(entry)}
              </div>
            </div>
            {family_pill}
          </div>
          <div class="model-actions">{action}</div>
        </div>
      </article>
    """
