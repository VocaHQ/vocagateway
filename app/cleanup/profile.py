# Copyright (c) 2025-present VocaHQ, Inc.
# SPDX-License-Identifier: AGPL-3.0-only
# See the LICENSE file for details.

"""Launch flags for the cleanup worker, chosen from this machine.

A 0.6B cleanup model is cheap on a GPU or a 16 GB host and expensive on a
CPU-only 8 GB box sitting next to a speech engine. One set of llama-server
flags cannot serve both. The gateway therefore has two profiles:

* ``compact`` — 4096 context, 8-bit KV, small batches. Default on CPU-only
  hosts with under 16 GB RAM.
* ``full`` — 8192 context, 16-bit KV, default llama.cpp batches. Default when
  a GPU (Metal, CUDA, or AMD) is present, or when the host has 16 GB RAM or
  more.

The window is not only a memory figure: it is what `TokenBudget` divides
between the transcript and its correction, so the full profile's larger
`--ctx-size` buys longer one-pass corrections rather than an idle KV cache.

``auto`` (the shipped default) picks between them. ``VOCAGATEWAY_CLEANUP_PROFILE``
forces one. An operator-run server is not launched by us, so these flags are
documented for that case rather than applied to it; the runtime still sizes
its budget from the window that server reports.
"""

from __future__ import annotations

from dataclasses import dataclass

from app import system
from app.cleanup.base import MINIMUM_CONTEXT_TOKENS, TokenBudget, budget_for_context

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
# The unquantized KV cache type, and the only one `llama-server` will build a
# context for without flash attention.
KV_CACHE_F16 = "f16"
KV_CACHE_Q8 = "q8_0"


@dataclass(frozen=True, slots=True)
class CleanupLaunchProfile:
    """The argv knobs one managed `llama-server` is launched with."""

    id: str
    context_tokens: int
    kv_cache_type: str
    batch_tokens: int
    ubatch_tokens: int
    summary: str

    @property
    def budget(self) -> TokenBudget:
        """How this profile's window splits between transcript and correction."""
        return budget_for_context(self.context_tokens)

    @property
    def quantized_value_cache(self) -> bool:
        """Whether the V cache this profile asks for needs flash attention.

        `llama-server` refuses to create a context with a quantized V cache and
        flash attention off — "quantized V cache requires flash_attn" — and
        `--flash-attn auto` resolves to off on a backend that cannot do it. The
        worker therefore asks for flash attention explicitly here, and falls
        back to an f16 V cache on a build too old to be asked.
        """
        return self.kv_cache_type != KV_CACHE_F16


COMPACT_PROFILE = CleanupLaunchProfile(
    id=PROFILE_COMPACT,
    context_tokens=MINIMUM_CONTEXT_TOKENS,
    kv_cache_type=KV_CACHE_Q8,
    batch_tokens=COMPACT_BATCH_TOKENS,
    ubatch_tokens=COMPACT_UBATCH_TOKENS,
    summary=(
        "Low-end profile: 4096 context and 8-bit KV cache, for CPU-only machines under 16 GB RAM."
    ),
)

FULL_PROFILE = CleanupLaunchProfile(
    id=PROFILE_FULL,
    context_tokens=FULL_CONTEXT_TOKENS,
    kv_cache_type=KV_CACHE_F16,
    batch_tokens=FULL_BATCH_TOKENS,
    ubatch_tokens=FULL_UBATCH_TOKENS,
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


def budget_for_window(n_ctx: int) -> TokenBudget:
    """The token split for a window the gateway did not choose.

    An operator-run server that reports nothing is treated as the smallest
    window the gateway will work with, which is the conservative reading: too
    small a budget costs an extra piece, too large a one silently drops the
    system instruction.
    """
    return budget_for_context(n_ctx if n_ctx > 0 else MINIMUM_CONTEXT_TOKENS)
