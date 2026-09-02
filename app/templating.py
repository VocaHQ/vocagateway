from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from urllib.parse import quote

from jinja2 import Environment, FileSystemLoader
from markupsafe import Markup

from app.fragments.shared import _format_bytes, _format_latency, _format_uptime

TEMPLATES_DIR = Path(__file__).parent / "templates"


@lru_cache(maxsize=1)
def _environment() -> Environment:
    env = Environment(
        loader=FileSystemLoader(TEMPLATES_DIR),
        autoescape=True,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.filters["format_bytes"] = _format_bytes
    env.filters["format_latency"] = _format_latency
    env.filters["format_uptime"] = _format_uptime
    env.filters["urlpath"] = lambda path_segment: quote(path_segment, safe="")
    return env


def render(name: str, /, **context: object) -> Markup:
    """Render a template to a `Markup`-safe string.

    `Markup` (not plain `str`) matters here: fragment functions commonly embed
    one rendered piece inside another, e.g. the models page template does
    `{{ list_html }}` with `list_html` itself a `render()` result. Autoescape
    treats a plain `str` there as untrusted and escapes the `<div>`s away —
    `Markup` tells Jinja this HTML was already built safely and can be
    inserted as-is.
    """
    html = _environment().get_template(name).render(**context)
    return Markup(html)
