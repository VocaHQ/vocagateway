from __future__ import annotations

BYTES_PER_GIGABYTE = 1_000_000_000
BYTES_PER_MEGABYTE = 1_000_000
BYTES_PER_KILOBYTE = 1_000
SECONDS_PER_DAY = 86_400
SECONDS_PER_HOUR = 3_600
MILLISECONDS_PER_SECOND = 1_000


def _format_bytes(size: int) -> str:
    if size >= BYTES_PER_GIGABYTE:
        return f"{format(size / BYTES_PER_GIGABYTE, '.1f')} GB"
    if size >= BYTES_PER_MEGABYTE:
        return f"{format(size / BYTES_PER_MEGABYTE, '.0f')} MB"
    if size >= BYTES_PER_KILOBYTE:
        return f"{format(size / BYTES_PER_KILOBYTE, '.0f')} KB"
    return f"{size} B"


def _format_uptime(seconds: int) -> str:
    days, remainder = divmod(max(0, seconds), SECONDS_PER_DAY)
    hours, remainder = divmod(remainder, SECONDS_PER_HOUR)
    minutes, remaining_seconds = divmod(remainder, 60)
    if days:
        return f"{days}d {hours}h"
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {remaining_seconds}s"
    return f"{remaining_seconds}s"


def _format_latency(milliseconds: int | None) -> str:
    if milliseconds is None:
        return "—"
    if milliseconds >= MILLISECONDS_PER_SECOND:
        return f"{format(milliseconds / MILLISECONDS_PER_SECOND, '.1f')}s"
    return f"{milliseconds}ms"
