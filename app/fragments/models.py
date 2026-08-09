from __future__ import annotations

from html import escape
from urllib.parse import quote

from app.catalog import DEFAULT_CATALOG, LANGUAGE_NAMES
from app.fragments.engine import ENGINE_LABELS
from app.fragments.shared import _format_bytes
from app.schemas import AdminModelEntry

# Every action that re-renders the model list sends all filters, so downloading
# or selecting a model never silently resets the view.
MODEL_FILTER_INPUTS = "#family-filter, #language-filter, #installed-only-toggle"

_FILTER_ICON = (
    '<svg class="filter-icon" viewBox="0 0 24 24" width="16" height="16" fill="none" '
    'stroke="currentColor" stroke-width="1.75" stroke-linecap="round" '
    'stroke-linejoin="round" aria-hidden="true">'
    '<path d="M4 5h16l-6 7.5V19l-4 2v-8.5L4 5z"/></svg>'
)


def models_fragment(entries: list[AdminModelEntry]) -> str:
    installed = sum(entry.state == "installed" for entry in entries)
    families = len({entry.family for entry in entries if entry.family})
    return f"""
      <div class="page-head">
        <div>
          <h2>Models</h2>
          <p>{installed} of {len(entries)} downloaded across {families} families, stored only on
            this machine. Families start collapsed — open one to pick a model.</p>
        </div>
        <div class="models-toolbar-actions">
          <details class="models-filter" id="models-filter">
            <summary class="filter-trigger">
              {_FILTER_ICON}
              <span>Filter</span>
              <span id="filter-active-count" class="filter-count hidden" aria-live="polite"></span>
            </summary>
            <div class="filter-panel" role="group" aria-label="Model filters">
              <label class="filter-field">
                <span>Family</span>
                <select id="family-filter" name="family" class="family-filter"
                        hx-get="/ui/partials/models-list" hx-include="{MODEL_FILTER_INPUTS}"
                        hx-target="#models-list" hx-swap="innerHTML" hx-trigger="change">
                  <option value="">All families</option>
                  {_family_filter_options(entries)}
                </select>
              </label>
              <label class="filter-field">
                <span>Language</span>
                <select id="language-filter" name="language" class="language-filter"
                        hx-get="/ui/partials/models-list" hx-include="{MODEL_FILTER_INPUTS}"
                        hx-target="#models-list" hx-swap="innerHTML" hx-trigger="change">
                  <option value="">All languages</option>
                  {_language_filter_options()}
                </select>
              </label>
              <label class="filter-toggle" for="installed-only-toggle">
                <input type="checkbox" id="installed-only-toggle" name="installed_only"
                       value="true" class="sr-only"
                       hx-get="/ui/partials/models-list" hx-include="{MODEL_FILTER_INPUTS}"
                       hx-target="#models-list" hx-swap="innerHTML" hx-trigger="change" />
                Installed only
              </label>
              <button type="button" class="ghost small filter-clear" id="filter-clear">
                Clear filters
              </button>
            </div>
          </details>
          <button type="button" class="ghost small"
                  hx-get="/ui/partials/models-list" hx-include="{MODEL_FILTER_INPUTS}"
                  hx-target="#models-list" hx-swap="innerHTML">Refresh</button>
        </div>
      </div>
      <div id="models-list" aria-live="polite">
        {models_list_fragment(entries)}
      </div>
      <div class="card">
        <h2>Bring your own Whisper model</h2>
        <p class="muted">A direct HTTPS link to a <code>.bin</code> or <code>.gguf</code>
          file, which runs through the standalone engine.</p>
        <form hx-post="/ui/partials/models/custom" hx-include="{MODEL_FILTER_INPUTS}"
              hx-target="#models-list" hx-swap="innerHTML">
          <div class="row">
            <input name="url" type="url" required
                   placeholder="https://huggingface.co/&hellip;/resolve/main/model.gguf" />
            <button type="submit" class="primary">Download</button>
          </div>
        </form>
      </div>
    """


def _language_filter_options() -> str:
    """Only languages some model actually covers, so the filter never has a
    dead option. Whisper's set dominates, but Dolphin contributes the South and
    Southeast Asian languages Whisper lacks."""
    covered = {code for model in DEFAULT_CATALOG for code in model.language_codes}
    named = sorted((LANGUAGE_NAMES.get(code, code), code) for code in covered)
    return "".join(
        f'<option value="{escape(code, quote=True)}">{escape(name)}</option>'
        for name, code in named
    )


def _family_filter_options(entries: list[AdminModelEntry]) -> str:
    """Families present in the live catalogue view (includes custom when listed)."""
    names = sorted({entry.family for entry in entries if entry.family}, key=str.lower)
    return "".join(
        f'<option value="{escape(name, quote=True)}">{escape(name)}</option>' for name in names
    )


def models_list_fragment(
    entries: list[AdminModelEntry],
    installed_only: bool = False,
    language: str = "",
    family: str = "",
) -> str:
    """Family tiles: always collapsed for density; open one to browse its models."""
    groups: dict[str, list[AdminModelEntry]] = {}
    for entry in entries:
        name = entry.family or ENGINE_LABELS.get(entry.engine, entry.engine)
        groups.setdefault(name, []).append(entry)

    def family_sort_key(item: tuple[str, list[AdminModelEntry]]) -> tuple:
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
            _dictation_language_hint(entries, language)
            + f'<div class="family-grid">{"".join(parts)}</div>'
        )
    return _empty_filter_state(installed_only=installed_only, language=language, family=family)


def _empty_filter_state(*, installed_only: bool, language: str, family: str) -> str:
    bits: list[str] = []
    if family:
        bits.append(f"family {escape(family)}")
    if language:
        bits.append(escape(LANGUAGE_NAMES.get(language, language)))
    if installed_only:
        bits.append("installed only")
    if bits:
        joined = ", ".join(bits)
        return (
            f'<p class="empty-state">No models match {joined}. '
            "Clear or widen the filters to see more of the catalog.</p>"
        )
    return '<p class="empty-state">No models are available.</p>'


def _family_tile(family: str, items: list[AdminModelEntry]) -> str:
    installed = sum(entry.state == "installed" for entry in items)
    active = next((entry for entry in items if entry.active), None)
    recommended = next((entry for entry in items if entry.recommended), None)
    count = f"{len(items)} model" + ("" if len(items) == 1 else "s")
    installed_label = f"{installed} installed" if installed else "none installed"
    engines = sorted({ENGINE_LABELS.get(entry.engine, entry.engine) for entry in items})
    engine_note = escape(", ".join(engines))
    badges = ""
    if active:
        badges += '<span class="badge active">active</span>'
    elif recommended:
        badges += '<span class="badge recommended">recommended</span>'
    cards = "".join(_model_card(item) for item in items)
    # Always collapsed: open a tile deliberately for density across 50+ models.
    return f"""
      <details class="family-tile">
        <summary class="family-summary">
          <span class="family-summary-main">
            <span class="family-name">{escape(family)}</span>
            <span class="family-tags">{badges}</span>
          </span>
          <span class="family-meta">
            <span>{count}</span>
            <span aria-hidden="true">·</span>
            <span>{installed_label}</span>
            <span class="family-engines" title="Engines in this family">{engine_note}</span>
          </span>
        </summary>
        <div class="model-grid family-models">{cards}</div>
      </details>
    """

def _dictation_language_hint(entries: list[AdminModelEntry], language: str) -> str:
    """Warn when a filtered list mixes pinnable and auto-detecting models.

    Testing showed every auto-detecting model returns the wrong writing system on
    a short phrase — "ठीक है" came back as Chinese from both Dolphin and
    Qwen3-ASR. Pinning the language fixes it outright, so someone filtering for
    their language needs to know which half of the list to trust for dictation,
    which is mostly short phrases.
    """
    if not language:
        return ""
    automatic = [entry for entry in entries if entry.detects_language_automatically]
    pinnable = [entry for entry in entries if not entry.detects_language_automatically]
    if not automatic or not pinnable:
        return ""
    name = escape(LANGUAGE_NAMES.get(language, language))
    return (
        '<div class="callout warning models-language-hint">'
        f"<strong>Choosing a model for {name}</strong>"
        "<span>The models badged <em>auto language</em> decide the language themselves. "
        "They are strong on full sentences but return the wrong alphabet entirely on a "
        "short phrase, which is most of dictation. Prefer one of the other "
        f"{len(pinnable)} models here, which are told to transcribe {name}.</span>"
        "</div>"
    )


def _language_disclosure(entry: AdminModelEntry) -> str:
    """Name the languages behind a summary like "25 European languages".

    A `<details>` rather than an always-open list: the summary line is what people
    scan, and forty language names on every card would bury everything else. It
    needs no JavaScript, stays keyboard-accessible, and is searchable by the
    browser's own find-in-page once opened.
    """
    # A single-language model needs no disclosure: the "English only" summary
    # beside it already says everything the list would.
    if len(entry.language_names) < 2:
        return ""
    names = ", ".join(escape(name) for name in entry.language_names)
    note = (
        " This model chooses the language itself, so these are the languages it "
        "transcribes well, not a list you can pick from."
        if entry.detects_language_automatically
        else ""
    )
    return f"""
        <details class="model-languages">
          <summary>{len(entry.language_names)} languages</summary>
          <p class="model-language-list">{names}.{escape(note)}</p>
        </details>
    """


def _model_card(entry: AdminModelEntry) -> str:
    encoded_id = quote(entry.id, safe="")
    # Only state carries a tag. Everything else that used to be a pill is now part
    # of the metadata line, because five pills per model told nobody anything.
    badges = ""
    if entry.active:
        badges += '<span class="badge active">active</span>'
    elif entry.recommended:
        badges += '<span class="badge recommended">recommended</span>'
    if entry.detects_language_automatically:
        badges += (
            '<span class="badge auto-language" title="This model decides the language itself.'
            ' The language chosen in the app does not constrain it.">auto language</span>'
        )
    meta = [
        escape(entry.family),
        _format_bytes(entry.size_bytes),
        escape(entry.languages),
        escape(entry.quality),
    ]
    if entry.supports_streaming:
        meta.append("streams live")
    if not entry.commercial_use:
        meta.append("personal use only")
    meta.append(f"{escape(entry.source)} · {escape(entry.license_name)}")

    if entry.state == "downloading":
        percent = round((entry.progress or 0) * 100)
        downloaded = _format_bytes(entry.downloaded_bytes or 0)
        total = _format_bytes(entry.total_bytes or 0)
        action = (
            f'<div class="progress" role="progressbar" aria-valuemin="0" aria-valuemax="100"'
            f' aria-valuenow="{percent}"><div class="bar" style="width:{percent}%"></div></div>'
            f'<span class="muted small progress-copy">'
            f"{percent}% · {downloaded} / {total}</span>"
            f'<button class="ghost small" hx-post="/ui/partials/models/{encoded_id}/cancel"'
            f' hx-include="{MODEL_FILTER_INPUTS}"'
            f' hx-target="#models-list" hx-swap="innerHTML">Cancel</button>'
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
            f'<div class="row-actions">{select_button}'
            f'<button class="ghost small danger"'
            f' hx-delete="/ui/partials/models/{encoded_id}"'
            f' hx-include="{MODEL_FILTER_INPUTS}"'
            f' hx-confirm="Delete {escape(entry.label)} from this server?"'
            f' hx-target="#models-list" hx-swap="innerHTML">Delete</button></div>'
        )
    else:
        note = (
            f'<p class="error small">Download failed: {escape(entry.error)}</p>'
            if entry.error
            else ""
        )
        # Ghost, not primary: with 58 catalogue entries, 58 filled buttons is a wall
        # of colour. Green is reserved for the decision that changes the gateway,
        # which is selecting a model.
        action = (
            f"{note}"
            f'<button class="ghost small"'
            f' hx-post="/ui/partials/models/{encoded_id}/download"'
            f' hx-include="{MODEL_FILTER_INPUTS}"'
            f' hx-target="#models-list" hx-swap="innerHTML">Download</button>'
        )

    return f"""
      <article class="model-card" data-model-id="{escape(entry.id, quote=True)}"
               data-state="{entry.state}">
        <div class="model-main">
          <div class="model-title">
            <h4>{escape(entry.label)}</h4>
            <span class="model-tags">{badges}</span>
          </div>
          <p class="model-meta">{" &middot; ".join(meta)}</p>
          <details class="model-more">
            <summary>Details</summary>
            <p class="model-description">{escape(entry.description)}</p>
            {_language_disclosure(entry)}
          </details>
        </div>
        <div class="actions">{action}</div>
      </article>
    """