"""Launch flags for the cleanup worker, chosen from this machine.

A 0.6B cleanup model is cheap on a GPU or a 16 GB host and expensive on a
CPU-only 8 GB box sitting next to a speech engine. One set of llama-server
flags cannot serve both. The gateway therefore has two profiles:

* ``compact`` — 4096 context, 8-bit KV, small batches. Default on CPU-only
  hosts with under 16 GB RAM.
* ``full`` — 8192 context, 16-bit KV, default llama.cpp batches. Default when
  a GPU (Metal, CUDA, or AMD) is present, or when the host has 16 GB RAM or
  more.

``auto`` (the shipped default) picks between them. ``VOCAGATEWAY_CLEANUP_PROFILE``
forces one. An operator-run server is not launched by us, so these flags are
documented for that case rather than applied to it; the runtime still sizes
its sentence packing from the window that server reports.
"""

from __future__ import annotations

from dataclasses import dataclass

from app import system
from app.cleanup.base import CHUNK_CHAR_LIMIT, MINIMUM_CONTEXT_TOKENS

PROFILE_AUTO = "auto"
PROFILE_COMPACT = "compact"
PROFILE_FULL = "full"
PROFILE_CHOICES = (PROFILE_AUTO, PROFILE_COMPACT, PROFILE_FULL)
FULL_RAM_GB = 16.0
COMPACT_BATCH_TOKENS = 512
COMPACT_UBATCH_TOKENS = 256
FULL_CONTEXT_TOKENS = 8_192
FULL_BATCH_TOKENS = 2_048
FULL_UBATCH_TOKENS = 512
# Tokens reserved in n_ctx for the system instruction, chat template, and
# JSON output, so the packed transcript does not crowd generation off the end.
PROMPT_AND_OUTPUT_RESERVE = 2_500


@dataclass(frozen=True, slots=True)
class CleanupLaunchProfile:
    """The argv knobs one managed `llama-server` is launched with."""

    id: str
    context_tokens: int
    kv_cache_type: str
    batch_tokens: int
    ubatch_tokens: int
    chunk_char_limit: int
    summary: str


COMPACT_PROFILE = CleanupLaunchProfile(
    id=PROFILE_COMPACT,
    context_tokens=MINIMUM_CONTEXT_TOKENS,
    kv_cache_type="q8_0",
    batch_tokens=COMPACT_BATCH_TOKENS,
    ubatch_tokens=COMPACT_UBATCH_TOKENS,
    chunk_char_limit=CHUNK_CHAR_LIMIT,
    summary=(
        "Low-end profile: 4096 context and 8-bit KV cache, for CPU-only machines under 16 GB RAM."
    ),
)

FULL_PROFILE = CleanupLaunchProfile(
    id=PROFILE_FULL,
    context_tokens=FULL_CONTEXT_TOKENS,
    kv_cache_type="f16",
    batch_tokens=FULL_BATCH_TOKENS,
    ubatch_tokens=FULL_UBATCH_TOKENS,
    chunk_char_limit=FULL_CONTEXT_TOKENS - PROMPT_AND_OUTPUT_RESERVE,
    summary=(
        "High-end profile: 8192 context and 16-bit KV cache, for GPUs and "
        "machines with 16 GB RAM or more."
    ),
)


def resolve_profile(
    requested: str | None, host: system.SystemInfo | None = None
) -> CleanupLaunchProfile:
    """The profile to launch with, honouring an explicit override first."""
    choice = (requested or PROFILE_AUTO).strip().lower()
    if choice == PROFILE_COMPACT:
        return COMPACT_PROFILE
    if choice == PROFILE_FULL:
        return FULL_PROFILE
    if host is None:
        return COMPACT_PROFILE
    if system.is_cpu_only(host) and host.ram_gb < FULL_RAM_GB:
        return COMPACT_PROFILE
    return FULL_PROFILE


def describe_profile(profile: CleanupLaunchProfile, requested: str | None) -> str:
    """One sentence for the Cleanup tab, including whether auto or forced."""
    choice = (requested or PROFILE_AUTO).strip().lower()
    if choice in {PROFILE_COMPACT, PROFILE_FULL}:
        return f"{profile.summary} Forced by VOCAGATEWAY_CLEANUP_PROFILE={choice}."
    return f"{profile.summary} Chosen automatically for this host."


def chunk_limit_for_window(n_ctx: int) -> int:
    """How many characters one inference can take, given a server's ``n_ctx``."""
    if n_ctx <= 0:
        return CHUNK_CHAR_LIMIT
    return max(CHUNK_CHAR_LIMIT, n_ctx - PROMPT_AND_OUTPUT_RESERVE)
