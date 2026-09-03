from __future__ import annotations

from datetime import datetime

from app.admin_queries import token_entries
from app.context import GatewayContext
from app.schemas import DeviceTokenEntry
from app.templating import render


def tokens_fragment_str(ctx: GatewayContext, *, new_token: tuple[str, str] | None = None) -> str:
    return tokens_fragment(token_entries(ctx), new_token=new_token)


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


def _format_created(timestamp: datetime | None) -> str:
    return "—" if timestamp is None else timestamp.strftime("%Y-%m-%d %H:%M UTC")
