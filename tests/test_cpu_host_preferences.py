from __future__ import annotations

import types
from typing import Any

from app.catalog import (
    PARAKEET_ENGLISH_ID,
    PARAKEET_MINIMUM_RAM_GB,
    PARAKEET_MULTILINGUAL_ID,
    recommended_ids,
)
from app.system import is_cpu_only

CUDA_ACCELERATORS = ("CPU", "NVIDIA CUDA")
CPU_ONLY_ACCELERATORS = ("CPU",)
PARAKEET_IDS = {PARAKEET_MULTILINGUAL_ID, PARAKEET_ENGLISH_ID}


def _host(ram: float, accelerators: tuple[str, ...], *, apple: bool = False) -> Any:
    return types.SimpleNamespace(ram_gb=ram, is_apple_silicon=apple, accelerators=accelerators)


def test_is_cpu_only_reads_the_accelerator_list() -> None:
    """`accelerators` always leads with CPU and gains an entry per device, so a
    lone "CPU" is the whole signal."""
    assert is_cpu_only(_host(16, CPU_ONLY_ACCELERATORS)) is True
    assert is_cpu_only(_host(16, CUDA_ACCELERATORS)) is False
    assert is_cpu_only(_host(16, ("CPU", "AMD ROCm"))) is False
    assert is_cpu_only(_host(16, ("CPU", "Metal / Core ML"), apple=True)) is False


def test_a_cpu_only_host_is_offered_parakeet_at_every_tier_that_fits() -> None:
    """Including the sub-8 GB rung, which offered nothing from this family even
    though Parakeet INT8 asks only 4 GB."""
    for ram in (16.0, 8.0, PARAKEET_MINIMUM_RAM_GB):
        assert recommended_ids(_host(ram, CPU_ONLY_ACCELERATORS)) >= PARAKEET_IDS, ram


def test_parakeet_is_not_forced_onto_a_machine_too_small_for_it() -> None:
    picks = recommended_ids(_host(PARAKEET_MINIMUM_RAM_GB - 1, CPU_ONLY_ACCELERATORS))
    assert not (PARAKEET_IDS & picks)


def test_a_gpu_host_keeps_its_own_ram_tier_picks() -> None:
    """The preference is about CPU inference, so a box with a card is untouched:
    at 4 GB it still gets the small-model rung rather than Parakeet."""
    assert not (PARAKEET_IDS & recommended_ids(_host(4.0, CUDA_ACCELERATORS)))


def test_the_preference_only_adds_and_never_removes() -> None:
    """A CPU-only host should still see everything its RAM tier offered."""
    for ram in (16.0, 8.0, 4.0):
        gpu = recommended_ids(_host(ram, CUDA_ACCELERATORS))
        cpu = recommended_ids(_host(ram, CPU_ONLY_ACCELERATORS))
        assert gpu <= cpu, ram
