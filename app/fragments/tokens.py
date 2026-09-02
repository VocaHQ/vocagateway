from __future__ import annotations

from datetime import datetime

from app.schemas import DeviceTokenEntry
from app.templating import render


def tokens_fragment(
    entries: list[DeviceTokenEntry], *, new_token: tuple[str, str] | None = None
) -> str:
    return render(
        "tokens/card.html",
        entries=[_token_row_context(entry) for entry in entries],
        new_token={"label": new_token[0], "plaintext": new_token[1]} if new_token else None,
    )


def _token_row_context(entry: DeviceTokenEntry) -> dict[str, object]:
    return {
        "id": entry.id,
        "label": entry.label,
        "created": _format_created(entry.created_at),
        "revocable": entry.revocable,
    }


def _format_created(value: datetime | None) -> str:
    return "—" if value is None else value.strftime("%Y-%m-%d %H:%M UTC")
