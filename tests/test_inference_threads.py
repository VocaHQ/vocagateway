"""Thread-count policy shared by every CPU inference engine."""

from __future__ import annotations

import pytest
from pytest import MonkeyPatch

from app import system

CPUINFO_TWO_CORES_FOUR_THREADS = """\
processor\t: 0
physical id\t: 0
core id\t\t: 0
processor\t: 1
physical id\t: 0
core id\t\t: 1
processor\t: 2
physical id\t: 0
core id\t\t: 0
processor\t: 3
physical id\t: 0
core id\t\t: 1
"""


def test_an_explicit_operator_choice_wins() -> None:
    assert system.inference_thread_count(3) == 3


def test_threads_are_capped_so_synchronisation_does_not_dominate(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setattr(system.os, "cpu_count", lambda: 128)
    monkeypatch.setattr(system._CpuSets, "physical_cpu_count", classmethod(lambda cls, _: 64))
    monkeypatch.setattr(system._CpuSets, "effective_cpu_count", classmethod(lambda cls, _: 64.0))

    assert system.inference_thread_count() == system.MAXIMUM_INFERENCE_THREADS


def test_a_container_quota_holds_the_count_down(monkeypatch: MonkeyPatch) -> None:
    """`os.cpu_count()` reports the host's cores, not the cgroup's share."""
    monkeypatch.setattr(system.os, "cpu_count", lambda: 64)
    monkeypatch.setattr(system._CpuSets, "physical_cpu_count", classmethod(lambda cls, _: 32))
    monkeypatch.setattr(system._CpuSets, "effective_cpu_count", classmethod(lambda cls, _: 2.0))

    assert system.inference_thread_count() == 2


def test_at_least_one_thread_is_always_requested(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(system.os, "cpu_count", lambda: 1)
    monkeypatch.setattr(system._CpuSets, "physical_cpu_count", classmethod(lambda cls, _: 0))
    monkeypatch.setattr(system._CpuSets, "effective_cpu_count", classmethod(lambda cls, _: 0.4))

    assert system.inference_thread_count() == 1


def test_hyperthread_siblings_are_not_counted_as_cores() -> None:
    cores = system._CpuSets._cpuinfo_cores(CPUINFO_TWO_CORES_FOUR_THREADS)

    assert len(cores) == 2


@pytest.mark.parametrize("cpuinfo", ["", "processor\t: 0\n"])
def test_a_cpuinfo_without_core_ids_falls_back_to_the_logical_count(
    cpuinfo: str, monkeypatch: MonkeyPatch
) -> None:
    """A kernel that reports no topology must not leave the engine single-threaded."""
    assert system._CpuSets._cpuinfo_cores(cpuinfo) == set()

    monkeypatch.setattr(
        system.Path,
        "read_text",
        lambda self, **keywords: cpuinfo,  # noqa: ARG005
    )
    assert system._CpuSets._linux_physical_cpus(6) == 6


def test_an_unreadable_cpuinfo_falls_back_to_the_logical_count(
    monkeypatch: MonkeyPatch,
) -> None:
    def unreadable(self: object, **keywords: object) -> str:
        raise OSError("no /proc")

    monkeypatch.setattr(system.Path, "read_text", unreadable)
    assert system._CpuSets._linux_physical_cpus(6) == 6
