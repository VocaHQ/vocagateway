"""Lifecycle races around the managed llama-server child.

The request path never waits on these; the tests still have to prove that a
start which *does* wait cannot leak a process when stop() races spawn, and
that a port another process stole is retried rather than retired.
"""

from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path

import pytest

from app.cleanup.base import CleanupUnavailable
from app.cleanup.worker import LlamaServerWorker


class FakeProcess:
    def __init__(self, *, running: bool = True) -> None:
        self._code: int | None = None if running else 1
        self.terminated = False
        self.killed = False

    def poll(self) -> int | None:
        return self._code

    def terminate(self) -> None:
        self.terminated = True
        self._code = 1

    def kill(self) -> None:
        self.killed = True
        self._code = 1

    def wait(self, timeout: float | None = None) -> int:
        if self._code is None:
            if timeout is not None:
                raise TimeoutError
            self._code = 0
        return self._code


def worker_for(tmp_path: Path) -> LlamaServerWorker:
    binary = tmp_path / "llama-server"
    binary.write_text("#!/bin/sh\n")
    binary.chmod(0o755)
    model = tmp_path / "model.gguf"
    model.write_bytes(b"gguf")
    return LlamaServerWorker(binary, model, context_tokens=8192, cpu_threads=1)


def test_stop_during_spawn_does_not_orphan_the_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """stop() must serialize with spawn/adopt, or llama-server leaks."""
    worker = worker_for(tmp_path)
    in_spawn = threading.Event()
    release_spawn = threading.Event()
    process = FakeProcess()

    def spawn(port: int) -> FakeProcess:
        in_spawn.set()
        assert release_spawn.wait(timeout=2)
        return process

    async def never_ready(_endpoint: object) -> bool:
        return False

    worker._spawn = spawn  # type: ignore[method-assign]
    monkeypatch.setattr("app.cleanup.worker._health_ok", never_ready)
    errors: list[BaseException] = []

    def start() -> None:
        try:
            asyncio.run(worker.ensure_started(budget=2))
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=start)
    thread.start()
    assert in_spawn.wait(timeout=2)

    def delayed_release() -> None:
        time.sleep(0.05)
        release_spawn.set()

    threading.Thread(target=delayed_release).start()
    worker.stop()
    thread.join(timeout=2)
    assert not thread.is_alive()
    assert process.terminated or process.killed
    assert worker.is_running is False
    assert worker.is_ready is False
    assert errors and isinstance(errors[0], CleanupUnavailable)


async def test_stop_during_health_wait_terminates_the_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker = worker_for(tmp_path)
    process = FakeProcess()
    worker._spawn = lambda port: process  # type: ignore[method-assign]
    polling = asyncio.Event()
    stopped = asyncio.Event()
    original_stop = worker.stop

    def stop(*, force: bool = False) -> None:
        original_stop(force=force)
        stopped.set()

    worker.stop = stop  # type: ignore[method-assign]

    async def hang(_endpoint: object) -> bool:
        polling.set()
        await stopped.wait()
        return False

    monkeypatch.setattr("app.cleanup.worker._health_ok", hang)
    task = asyncio.create_task(worker.ensure_started(budget=30))
    await polling.wait()
    worker.stop()
    with pytest.raises(CleanupUnavailable):
        await task
    assert process.terminated or process.killed
    assert worker.is_ready is False


async def test_a_refused_port_is_retried_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker = worker_for(tmp_path)
    first = FakeProcess(running=False)
    second = FakeProcess()
    spawned = iter([first, second])
    worker._spawn = lambda port: next(spawned)  # type: ignore[method-assign]

    async def health(_endpoint: object) -> bool:
        return second.poll() is None

    monkeypatch.setattr("app.cleanup.worker._health_ok", health)
    assert await worker.ensure_started(budget=1) is True
    assert worker.is_ready is True
    assert worker.is_running is True
