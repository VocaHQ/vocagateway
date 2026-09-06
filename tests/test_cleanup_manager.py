"""Preference precedence, runtime ownership, and the idle unload.

The manager is where "what did the operator ask for" meets "what is actually
running". Both halves have failure modes that only show up here: an environment
variable that silently loses to a saved choice, a worker that keeps running
after the model was switched, an idle unload that fires mid-request.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from conftest import CLEANUP_MODEL_ID, FakeCleanupRuntime, FakeWorkerHost

from app.cleanup import host as host_module
from app.cleanup import manager as manager_module
from app.cleanup.base import (
    MINIMUM_CONTEXT_TOKENS,
    MODE_CONSERVATIVE,
    MODE_INHERIT,
    MODE_OFF,
    CleanupReason,
)
from app.cleanup.manager import CleanupUpdate, build_manager
from app.cleanup.transport import Endpoint
from app.cleanup.worker import LlamaServerWorker, resolve_binary
from app.config import Settings
from app.runtime_config import RuntimeConfig

OTHER_CLEANUP_MODEL_ID = "cleanup:qwen3-1.7b"


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        token="manager-" + ("x" * 48),
        data_dir=tmp_path,
        whisper_binary=tmp_path / "whisper-cli",
        whisper_model=tmp_path / "model.bin",
        config_path=tmp_path / "config.json",
    )


def manager_for(settings: Settings, **config: object) -> object:
    run_config = RuntimeConfig(**config)  # type: ignore[arg-type]
    return build_manager(settings, run_config, settings.config_path)


def test_the_gateway_default_is_off(settings: Settings) -> None:
    manager = manager_for(settings)
    assert manager.enabled is False
    assert manager.options(MODE_INHERIT).mode == MODE_OFF
    assert manager.options(MODE_CONSERVATIVE).mode == MODE_OFF


def test_an_explicit_opt_in_cannot_override_an_operator_who_said_no(
    settings: Settings,
) -> None:
    """Asking for cleanup does not install a model or reopen a disabled runtime."""
    manager = manager_for(settings, cleanup_enabled=False, cleanup_mode=MODE_CONSERVATIVE)
    assert manager.options(MODE_CONSERVATIVE).mode == MODE_OFF


def test_inherit_follows_the_gateway_default(settings: Settings) -> None:
    manager = manager_for(settings, cleanup_enabled=True, cleanup_mode=MODE_CONSERVATIVE)
    assert manager.options(MODE_INHERIT).mode == MODE_CONSERVATIVE
    assert manager.options(None).mode == MODE_CONSERVATIVE
    # An explicit opt-out always wins over the default.
    assert manager.options(MODE_OFF).mode == MODE_OFF


def test_an_environment_override_beats_the_saved_choice(settings: Settings) -> None:
    overridden = replace(settings, cleanup_enabled=False)
    manager = manager_for(overridden, cleanup_enabled=True, cleanup_mode=MODE_CONSERVATIVE)
    assert manager.enabled is False
    assert manager.status().locked_settings == ("enabled",)


def test_an_unset_override_leaves_the_saved_choice_alone(settings: Settings) -> None:
    """The tri-state matters: unset must not overwrite with a default."""
    manager = manager_for(settings, cleanup_enabled=True, cleanup_mode=MODE_CONSERVATIVE)
    assert settings.cleanup_enabled is None
    assert manager.enabled is True
    assert manager.status().locked_settings == ()


def test_a_partial_update_is_persisted_atomically(settings: Settings) -> None:
    manager = manager_for(settings, engine="moonshine", cpu_threads=4)
    manager.configure(CleanupUpdate(enabled=True, mode=MODE_CONSERVATIVE))
    reloaded = RuntimeConfig.load(settings.config_path)
    assert reloaded.cleanup_enabled is True
    assert reloaded.cleanup_mode == MODE_CONSERVATIVE
    # The engine block came along untouched rather than being reset.
    assert reloaded.engine == "moonshine"
    assert reloaded.cpu_threads == 4


def test_an_out_of_range_timeout_is_clamped_not_rejected(settings: Settings) -> None:
    manager = manager_for(settings)
    manager.configure(CleanupUpdate(timeout_seconds=900))
    assert manager.preferences.timeout_seconds == 30.0
    manager.configure(CleanupUpdate(timeout_seconds=0.01))
    assert manager.preferences.timeout_seconds == 1.0


def test_switching_the_model_stops_the_worker(settings: Settings) -> None:
    manager = manager_for(settings, cleanup_enabled=True, cleanup_model=CLEANUP_MODEL_ID)
    host = FakeWorkerHost(FakeCleanupRuntime())
    manager.host = host
    manager.configure(CleanupUpdate(model_id=OTHER_CLEANUP_MODEL_ID))
    assert host.stops == 1


def test_disabling_stops_the_worker(settings: Settings) -> None:
    manager = manager_for(settings, cleanup_enabled=True, cleanup_model=CLEANUP_MODEL_ID)
    host = FakeWorkerHost(FakeCleanupRuntime())
    manager.host = host
    manager.configure(CleanupUpdate(enabled=False))
    assert host.stops == 1


async def test_the_admission_bound_yields_none_rather_than_queueing(
    settings: Settings,
) -> None:
    manager = manager_for(settings, cleanup_enabled=True, cleanup_model=CLEANUP_MODEL_ID)
    manager.host = FakeWorkerHost(FakeCleanupRuntime())
    manager.model_path = lambda: Path("model.gguf")  # type: ignore[method-assign]
    async with manager.lease() as first:
        assert first.runtime is not None
        async with manager.lease() as second:
            assert second.runtime is None
            assert second.reason is CleanupReason.BUSY


async def test_a_cold_request_falls_back_without_waiting_for_start(settings: Settings) -> None:
    """Transcription must not sit on a 300s model load. The load-hold runs alone."""
    manager = manager_for(settings, cleanup_enabled=True, cleanup_model=CLEANUP_MODEL_ID)
    host = _HangingHost(FakeCleanupRuntime("ok"))
    manager.host = host
    manager.model_path = lambda: Path("model.gguf")  # type: ignore[method-assign]

    async with manager.lease() as slot:
        assert slot.runtime is None
        assert slot.reason is CleanupReason.MODEL_LOADING
    assert manager._active_leases >= 1
    manager.configure(CleanupUpdate(model_id=OTHER_CLEANUP_MODEL_ID))
    assert host.stops == 0
    host.release.set()
    await _settle_load_hold(manager)
    assert host.stops == 1


async def test_warmup_holds_a_lease_so_configure_cannot_stop_the_start(
    settings: Settings,
) -> None:
    manager = manager_for(settings, cleanup_enabled=True, cleanup_model=CLEANUP_MODEL_ID)
    host = _HangingHost(FakeCleanupRuntime("ok"))
    manager.host = host
    manager.model_path = lambda: Path("model.gguf")  # type: ignore[method-assign]

    assert await asyncio.wait_for(manager.warmup(), timeout=1.0) is False
    assert manager._active_leases >= 1
    manager.configure(CleanupUpdate(enabled=False))
    assert host.stops == 0
    host.release.set()
    await _settle_load_hold(manager)
    assert host.stops == 1
    assert host.is_running is False


async def test_disable_while_leased_stops_the_worker_once_the_lease_drains(
    settings: Settings,
) -> None:
    manager = manager_for(settings, cleanup_enabled=True, cleanup_model=CLEANUP_MODEL_ID)
    host = FakeWorkerHost(FakeCleanupRuntime("ok"))
    manager.host = host
    manager.model_path = lambda: Path("model.gguf")  # type: ignore[method-assign]

    async with manager.lease():
        manager.configure(CleanupUpdate(enabled=False))
        assert host.stops == 0
        assert host.is_running is True
    assert host.stops == 1
    assert host.is_running is False


async def test_disable_during_warmup_stops_the_worker_once_the_lease_drains(
    settings: Settings,
) -> None:
    """configure cannot kill a start in flight; it must stop the worker after."""
    manager = manager_for(settings, cleanup_enabled=True, cleanup_model=CLEANUP_MODEL_ID)
    host = _HangingHost(FakeCleanupRuntime("ok"))
    manager.host = host
    manager.model_path = lambda: Path("model.gguf")  # type: ignore[method-assign]

    assert await manager.warmup() is False
    manager.configure(CleanupUpdate(enabled=False))
    assert host.stops == 0
    host.release.set()
    await _settle_load_hold(manager)
    assert host.stops == 1
    assert host.is_running is False


async def test_reenable_during_warmup_keeps_the_worker_once_the_lease_drains(
    settings: Settings,
) -> None:
    """A deferred disable must not stick after the operator turns cleanup back on."""
    manager = manager_for(settings, cleanup_enabled=True, cleanup_model=CLEANUP_MODEL_ID)
    host = _HangingHost(FakeCleanupRuntime("ok"))
    manager.host = host
    manager.model_path = lambda: Path("model.gguf")  # type: ignore[method-assign]

    assert await manager.warmup() is False
    manager.configure(CleanupUpdate(enabled=False))
    assert host.stops == 0
    manager.configure(CleanupUpdate(enabled=True))
    assert host.stops == 0
    host.release.set()
    await _settle_load_hold(manager)
    assert host.stops == 0
    assert host.is_running is True
    assert manager.enabled is True


async def test_reenable_while_leased_keeps_the_worker_once_the_lease_drains(
    settings: Settings,
) -> None:
    manager = manager_for(settings, cleanup_enabled=True, cleanup_model=CLEANUP_MODEL_ID)
    host = FakeWorkerHost(FakeCleanupRuntime("ok"))
    manager.host = host
    manager.model_path = lambda: Path("model.gguf")  # type: ignore[method-assign]

    async with manager.lease():
        manager.configure(CleanupUpdate(enabled=False))
        assert host.stops == 0
        manager.configure(CleanupUpdate(enabled=True))
        assert host.stops == 0
        assert manager.enabled is True
    assert host.stops == 0
    assert host.is_running is True


async def test_idle_unload_never_fires_while_a_lease_is_out(settings: Settings) -> None:
    manager = manager_for(
        settings,
        cleanup_enabled=True,
        cleanup_model=CLEANUP_MODEL_ID,
        cleanup_idle_unload_enabled=True,
        cleanup_idle_unload_minutes=5,
    )
    host = FakeWorkerHost(FakeCleanupRuntime())
    manager.host = host
    manager.model_path = lambda: Path("model.gguf")  # type: ignore[method-assign]
    long_past = time.monotonic() + 10_000
    async with manager.lease():
        assert manager.offload_if_idle(now=long_past) is False
    assert manager.offload_if_idle(now=long_past) is True
    assert host.offloaded is True


def test_idle_unload_is_off_for_a_server_the_gateway_does_not_own(
    settings: Settings,
) -> None:
    """Stopping somebody else's process is not this object's business."""
    external = replace(settings, cleanup_endpoint=("127.0.0.1", 8080))
    manager = manager_for(
        external,
        cleanup_enabled=True,
        cleanup_idle_unload_enabled=True,
        cleanup_idle_unload_minutes=5,
    )
    manager.host = FakeWorkerHost(FakeCleanupRuntime())
    assert manager.preferences.managed is False
    assert manager.offload_if_idle(now=time.monotonic() + 10_000) is False


def test_status_reports_unavailable_without_an_installed_model(settings: Settings) -> None:
    manager = manager_for(settings, cleanup_enabled=True, cleanup_model=CLEANUP_MODEL_ID)
    manager.host = FakeWorkerHost(None)
    report = manager.status()
    assert report.enabled is True
    assert report.model_installed is False
    assert report.state == "unavailable"


def test_the_worker_is_launched_with_argv_and_a_private_credential(
    tmp_path: Path,
) -> None:
    binary = tmp_path / "llama-server"
    binary.write_text("#!/bin/sh\n")
    binary.chmod(0o755)
    model = tmp_path / "model.gguf"
    model.write_bytes(b"gguf")
    worker = LlamaServerWorker(binary, model, context_tokens=8192, cpu_threads=2)

    arguments = worker._arguments(9999)

    assert arguments[0] == str(binary)
    assert "--host" in arguments and "127.0.0.1" in arguments
    assert "--jinja" in arguments
    assert "--no-webui" in arguments
    # A credential of the worker's own: the client's bearer token never travels.
    assert worker.api_key in arguments
    assert len(worker.api_key) >= 24
    # argv, never a shell string.
    assert all(isinstance(argument, str) for argument in arguments)


def test_a_binary_override_that_is_not_executable_is_not_resolved(tmp_path: Path) -> None:
    plain = tmp_path / "not-executable"
    plain.write_text("")
    assert resolve_binary(plain) is None


def test_the_worker_environment_drops_proxy_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.cleanup.worker import _worker_environment

    monkeypatch.setenv("HTTP_PROXY", "http://evil.example.com")
    monkeypatch.setenv("VOCAGATEWAY_TOKEN", "secret")
    environment = _worker_environment()
    assert "HTTP_PROXY" not in environment
    assert "VOCAGATEWAY_TOKEN" not in environment
    assert "PATH" in environment


ENVIRONMENT_CASES = (
    ("VOCAGATEWAY_CLEANUP_ENABLED", "true", "cleanup_enabled", True),
    ("VOCAGATEWAY_CLEANUP_ENABLED", "off", "cleanup_enabled", False),
    ("VOCAGATEWAY_CLEANUP_MODE", "conservative", "cleanup_mode", "conservative"),
    ("VOCAGATEWAY_CLEANUP_MODEL", CLEANUP_MODEL_ID, "cleanup_model", CLEANUP_MODEL_ID),
    ("VOCAGATEWAY_CLEANUP_TIMEOUT_SECONDS", "7.5", "cleanup_timeout_seconds", 7.5),
    ("VOCAGATEWAY_CLEANUP_LANGUAGES", "en, hi", "cleanup_languages", ("en", "hi")),
    ("VOCAGATEWAY_CLEANUP_ENDPOINT", "cleanup:8080", "cleanup_endpoint", ("cleanup", 8080)),
)


@pytest.mark.parametrize(("name", "raw", "field", "expected"), ENVIRONMENT_CASES)
def test_cleanup_environment_settings(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    name: str,
    raw: str,
    field: str,
    expected: object,
) -> None:
    monkeypatch.setenv("VOCAGATEWAY_TOKEN", "env-" + ("x" * 48))
    monkeypatch.setenv("VOCAGATEWAY_DATA_DIR", str(tmp_path))
    monkeypatch.setenv(name, raw)
    assert getattr(Settings.from_env(), field) == expected


def test_an_unset_cleanup_switch_is_neither_on_nor_off(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("VOCAGATEWAY_TOKEN", "env-" + ("x" * 48))
    monkeypatch.setenv("VOCAGATEWAY_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("VOCAGATEWAY_CLEANUP_ENABLED", raising=False)
    assert Settings.from_env().cleanup_enabled is None


ROUTABLE_ENDPOINTS = ("https://api.example.com/v1", "example.com:8080", "8.8.8.8:80", "nope")


@pytest.mark.parametrize("endpoint", ROUTABLE_ENDPOINTS)
def test_a_routable_cleanup_endpoint_is_refused_at_startup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, endpoint: str
) -> None:
    """Cleanup runs on the gateway. A remote address would quietly make that false."""
    monkeypatch.setenv("VOCAGATEWAY_TOKEN", "env-" + ("x" * 48))
    monkeypatch.setenv("VOCAGATEWAY_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("VOCAGATEWAY_CLEANUP_ENDPOINT", endpoint)
    with pytest.raises(RuntimeError):
        Settings.from_env()


class _NeverReadyWorker:
    """A worker whose model load never finishes, like a cold multi-gigabyte GGUF.

    `is_running` flips as soon as the child would have been spawned; `is_ready`
    stays false until `/health` would have answered. That gap is the mid-load
    window a concurrent request must not treat as a usable runtime.
    """

    def __init__(self, *_: object, **__: object) -> None:
        self.is_running = False
        self.is_ready = False
        self.starts = 0
        self.stop_calls = 0
        self.endpoint = Endpoint("127.0.0.1", 1)
        self.released = asyncio.Event()

    async def ensure_started(self, *, budget: float = 0.0) -> bool:
        self.starts += 1
        self.is_running = True
        await self.released.wait()
        self.is_ready = True
        return True

    def stop(self, *, force: bool = False) -> None:
        self.stop_calls += 1
        self.is_running = False
        self.is_ready = False


class _WorkerRecorder:
    """Builds a new scripted worker for each WorkerHost generation."""

    def __init__(self) -> None:
        self.workers: list[_NeverReadyWorker] = []

    def __call__(self, *_: object, **__: object) -> _NeverReadyWorker:
        built = _NeverReadyWorker()
        self.workers.append(built)
        return built


def _bind_model_files(manager: object, tmp_path: Path) -> None:
    files = {
        CLEANUP_MODEL_ID: tmp_path / "qwen3-0.6b.gguf",
        OTHER_CLEANUP_MODEL_ID: tmp_path / "qwen3-1.7b.gguf",
    }
    for path in files.values():
        path.write_bytes(b"gguf")

    def model_path() -> Path | None:
        return files.get(manager.model_id)  # type: ignore[attr-defined]

    manager.model_path = model_path  # type: ignore[method-assign]


class _HangingHost(FakeWorkerHost):
    """A host whose load never finishes until `release` is set."""

    def __init__(self, runtime: FakeCleanupRuntime) -> None:
        super().__init__(runtime)
        self.is_running = False
        self.is_ready = False
        self.is_loading = True
        self.release = asyncio.Event()

    async def runtime(self, model_id: str, model_file: Path) -> FakeCleanupRuntime | None:
        if not self.is_ready:
            return None
        self.loaded_model_id = model_id
        return self.runtime_value

    async def wait_for_load(self) -> None:
        await self.release.wait()
        self.is_loading = False
        self.is_running = True
        self.is_ready = True
        self.loaded_model_id = CLEANUP_MODEL_ID


async def _settle_load_hold(manager: object) -> None:
    task = getattr(manager, "_load_hold_task", None)
    if task is not None:
        await asyncio.wait({task})


@pytest.fixture
def cold_worker(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> _NeverReadyWorker:
    built = _NeverReadyWorker()
    monkeypatch.setattr(host_module.worker, "LlamaServerWorker", lambda *a, **k: built)
    monkeypatch.setattr(host_module.worker, "resolve_binary", lambda _=None: tmp_path / "llama")
    return built


@pytest.fixture
def scripted_workers(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> _WorkerRecorder:
    recorder = _WorkerRecorder()
    monkeypatch.setattr(host_module.worker, "LlamaServerWorker", recorder)
    monkeypatch.setattr(host_module.worker, "resolve_binary", lambda _=None: tmp_path / "llama")
    return recorder


async def _wait_for_stop(worker: _NeverReadyWorker) -> None:
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        if worker.stop_calls:
            return
        await asyncio.sleep(0.01)
    raise AssertionError("worker was not stopped")


async def _load_until_ready(manager: object, recorder: _WorkerRecorder) -> _NeverReadyWorker:
    assert await manager.warmup() is False  # type: ignore[union-attr]
    worker = recorder.workers[0]
    worker.released.set()
    await _settle_load_hold(manager)
    assert manager.host.is_ready  # type: ignore[union-attr]
    return worker


async def test_a_cold_load_never_makes_a_request_wait_for_it(
    settings: Settings, cold_worker: _NeverReadyWorker, tmp_path: Path
) -> None:
    """A request's budget is seconds; loading a model is minutes.

    Waiting for the load inside the lease put the whole start timeout — five
    minutes — in front of a transcription that had already succeeded.
    """
    worker_host = host_module.WorkerHost(settings, RuntimeConfig())
    leased = await asyncio.wait_for(
        worker_host.runtime(CLEANUP_MODEL_ID, tmp_path / "model.gguf"), timeout=1.0
    )
    assert leased is None
    await asyncio.sleep(0)  # let the background load actually begin
    assert cold_worker.starts == 1
    assert worker_host.is_running is True
    assert worker_host.is_ready is False
    worker_host.stop()


async def test_a_cold_load_is_reported_as_loading_not_as_a_full_runtime(
    settings: Settings, cold_worker: _NeverReadyWorker, tmp_path: Path
) -> None:
    """`busy` sends an operator looking for a queue that is not there."""
    manager = manager_for(settings, cleanup_enabled=True, cleanup_model=CLEANUP_MODEL_ID)
    manager.model_path = lambda: tmp_path / "model.gguf"  # type: ignore[method-assign]
    async with manager.lease() as slot:
        assert slot.runtime is None
        assert slot.reason is CleanupReason.MODEL_LOADING
    manager.host.stop()
    await _settle_load_hold(manager)


async def test_an_operator_warm_up_starts_the_load_and_reports_it(
    settings: Settings, cold_worker: _NeverReadyWorker, tmp_path: Path
) -> None:
    """Pressing "Load model now" returns at once; the card polls until ready.

    Holding the request open would mean an HTTP call that outlives its own
    timeout for a load running in the background regardless.
    """
    manager = manager_for(settings, cleanup_enabled=True, cleanup_model=CLEANUP_MODEL_ID)
    manager.model_path = lambda: tmp_path / "model.gguf"  # type: ignore[method-assign]
    assert await asyncio.wait_for(manager.warmup(), timeout=1.0) is False
    await asyncio.sleep(0)
    assert manager.loading is True
    assert manager.status().state == "loading"
    cold_worker.released.set()
    await _settle_load_hold(manager)
    assert manager.loading is False
    assert manager.status().state == "ready"
    manager.host.stop()


async def test_concurrent_requests_share_one_load_rather_than_starting_several(
    settings: Settings, cold_worker: _NeverReadyWorker, tmp_path: Path
) -> None:
    worker_host = host_module.WorkerHost(settings, RuntimeConfig())
    model = tmp_path / "model.gguf"
    for _ in range(3):
        assert await worker_host.runtime(CLEANUP_MODEL_ID, model) is None
        await asyncio.sleep(0)
    assert cold_worker.starts == 1
    worker_host.stop()


async def test_a_running_but_unready_worker_is_not_handed_out(
    settings: Settings, cold_worker: _NeverReadyWorker, tmp_path: Path
) -> None:
    """is_running is not ready: a later request must not get a half-loaded runtime."""
    worker_host = host_module.WorkerHost(settings, RuntimeConfig())
    model = tmp_path / "model.gguf"
    assert await worker_host.runtime(CLEANUP_MODEL_ID, model) is None
    await asyncio.sleep(0)
    assert cold_worker.is_running is True
    assert cold_worker.is_ready is False
    assert worker_host.is_running is True
    assert worker_host.is_ready is False
    assert await worker_host.runtime(CLEANUP_MODEL_ID, model) is None
    worker_host.stop()


async def test_a_mid_load_lease_reports_loading_not_a_runtime(
    settings: Settings, cold_worker: _NeverReadyWorker, tmp_path: Path
) -> None:
    manager = manager_for(settings, cleanup_enabled=True, cleanup_model=CLEANUP_MODEL_ID)
    manager.model_path = lambda: tmp_path / "model.gguf"  # type: ignore[method-assign]
    async with manager.lease() as first:
        assert first.runtime is None
        assert first.reason is CleanupReason.MODEL_LOADING
    await asyncio.sleep(0)
    assert cold_worker.is_running is True
    assert cold_worker.is_ready is False
    async with manager.lease() as second:
        assert second.runtime is None
        assert second.reason is CleanupReason.MODEL_LOADING
    manager.host.stop()
    await _settle_load_hold(manager)


async def test_stopping_mid_load_abandons_it_rather_than_adopting_it_later(
    settings: Settings, cold_worker: _NeverReadyWorker, tmp_path: Path
) -> None:
    """A model switch during a load must not leave the old one arriving after it."""
    worker_host = host_module.WorkerHost(settings, RuntimeConfig())
    assert await worker_host.runtime(CLEANUP_MODEL_ID, tmp_path / "model.gguf") is None
    await asyncio.sleep(0)
    worker_host.stop()
    cold_worker.released.set()
    await asyncio.sleep(0)
    assert worker_host.is_running is False


async def test_a_pinned_worker_is_not_replaced_on_a_key_change(
    settings: Settings, scripted_workers: _WorkerRecorder, tmp_path: Path
) -> None:
    """_ensure must refuse to reap a generation the manager still covers."""
    worker_host = host_module.WorkerHost(settings, RuntimeConfig())
    first = tmp_path / "a.gguf"
    second = tmp_path / "b.gguf"
    assert await worker_host.runtime(CLEANUP_MODEL_ID, first) is None
    await asyncio.sleep(0)
    assert len(scripted_workers.workers) == 1
    worker_host.pin()
    assert await worker_host.runtime(OTHER_CLEANUP_MODEL_ID, second) is None
    assert worker_host.loaded_model_id == CLEANUP_MODEL_ID
    assert scripted_workers.workers[0].stop_calls == 0
    assert len(scripted_workers.workers) == 1
    worker_host.unpin()
    assert await worker_host.runtime(OTHER_CLEANUP_MODEL_ID, second) is None
    await asyncio.sleep(0)
    assert worker_host.loaded_model_id == OTHER_CLEANUP_MODEL_ID
    assert len(scripted_workers.workers) == 2
    worker_host.stop()


async def test_warmup_does_not_reap_a_leased_worker_on_model_switch(
    settings: Settings, scripted_workers: _WorkerRecorder, tmp_path: Path
) -> None:
    """B1/C1: configure defers stop, but warmup/_ensure must not reap the lease."""
    manager = manager_for(settings, cleanup_enabled=True, cleanup_model=CLEANUP_MODEL_ID)
    _bind_model_files(manager, tmp_path)
    worker_a = await _load_until_ready(manager, scripted_workers)

    async with manager.lease() as slot:
        assert slot.runtime is not None
        manager.configure(CleanupUpdate(model_id=OTHER_CLEANUP_MODEL_ID))
        assert await manager.warmup() is False
        assert manager.host._worker is worker_a
        assert manager.host.loaded_model_id == CLEANUP_MODEL_ID
        assert worker_a.stop_calls == 0
        assert len(scripted_workers.workers) == 1

    assert manager.host.loaded_model_id is None
    await _wait_for_stop(worker_a)


async def test_load_hold_survives_a_concurrent_admit_and_model_switch(
    settings: Settings, scripted_workers: _WorkerRecorder, tmp_path: Path
) -> None:
    """B2/C2: load-hold does not take _slot, but it must still pin the worker."""
    manager = manager_for(settings, cleanup_enabled=True, cleanup_model=CLEANUP_MODEL_ID)
    _bind_model_files(manager, tmp_path)
    assert await manager.warmup() is False
    await asyncio.sleep(0)
    worker_a = scripted_workers.workers[0]
    assert manager._active_leases >= 1
    assert manager.host.is_loading

    manager.configure(CleanupUpdate(model_id=OTHER_CLEANUP_MODEL_ID))
    async with manager.lease() as slot:
        assert slot.runtime is None
        assert manager.host._worker is worker_a
        assert manager.host.loaded_model_id == CLEANUP_MODEL_ID
        assert worker_a.stop_calls == 0
        assert len(scripted_workers.workers) == 1

    assert manager.host._worker is worker_a
    worker_a.released.set()
    await _settle_load_hold(manager)
    assert manager.host.loaded_model_id is None
    await _wait_for_stop(worker_a)


async def test_a_replacement_load_is_not_started_under_another_models_hold(
    settings: Settings, scripted_workers: _WorkerRecorder, tmp_path: Path
) -> None:
    """C3: swapping A→B under A's hold would leave B uncovered when A settles."""
    manager = manager_for(settings, cleanup_enabled=True, cleanup_model=CLEANUP_MODEL_ID)
    _bind_model_files(manager, tmp_path)
    assert await manager.warmup() is False
    await asyncio.sleep(0)
    worker_a = scripted_workers.workers[0]

    manager.configure(CleanupUpdate(model_id=OTHER_CLEANUP_MODEL_ID))
    assert await manager.warmup() is False
    await asyncio.sleep(0)
    assert len(scripted_workers.workers) == 1
    assert manager.host._worker is worker_a

    worker_a.released.set()
    await _settle_load_hold(manager)
    assert manager.host._worker is None
    await _wait_for_stop(worker_a)

    assert await manager.warmup() is False
    await asyncio.sleep(0)
    assert len(scripted_workers.workers) == 2
    worker_b = scripted_workers.workers[1]
    assert manager._active_leases >= 1
    manager.configure(CleanupUpdate(enabled=False))
    assert manager.host._worker is worker_b
    worker_b.released.set()
    await _settle_load_hold(manager)
    assert manager.host._worker is None
    await _wait_for_stop(worker_b)


class _ContextRuntime:
    """An operator-run server that reports a context window and counts the asks."""

    def __init__(self, tokens: int) -> None:
        self.tokens = tokens
        self.asks = 0
        self.model_id = CLEANUP_MODEL_ID

    async def context_is_sufficient(self) -> bool:
        self.asks += 1
        return not self.tokens or self.tokens >= MINIMUM_CONTEXT_TOKENS

    async def available(self) -> bool:
        return True


@pytest.fixture
def external(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> Callable[[int], tuple[Any, _ContextRuntime]]:
    """A manager pointed at an operator-run endpoint reporting a given window."""

    def build(tokens: int) -> tuple[Any, _ContextRuntime]:
        runtime = _ContextRuntime(tokens)
        monkeypatch.setattr(manager_module, "LlamaServerRuntime", lambda *a, **k: runtime)
        manager = manager_for(
            replace(settings, cleanup_endpoint=("127.0.0.1", 9)),
            cleanup_enabled=True,
            cleanup_model=CLEANUP_MODEL_ID,
        )
        manager.host = FakeWorkerHost(None)
        return manager, runtime

    return build


async def test_an_external_endpoint_with_too_small_a_window_is_refused(
    external: Callable[[int], tuple[Any, _ContextRuntime]],
) -> None:
    """A window too small to hold the prompt drops the system instruction silently.

    The gateway never chose this server's `--ctx-size`, so unlike a managed
    worker it has to ask.
    """
    manager, runtime = external(512)
    async with manager.lease() as slot:
        assert slot.runtime is None
        assert slot.reason is CleanupReason.CONTEXT_TOO_SMALL
    assert runtime.asks == 1


async def test_a_large_enough_window_is_measured_once_not_per_request(
    external: Callable[[int], tuple[Any, _ContextRuntime]],
) -> None:
    manager, runtime = external(32_768)
    for _ in range(3):
        async with manager.lease() as slot:
            assert slot.runtime is not None
    assert runtime.asks == 1


async def test_a_server_that_reports_no_window_is_used_as_before(
    external: Callable[[int], tuple[Any, _ContextRuntime]],
) -> None:
    """An unknown window is not evidence of a bad one."""
    manager, _ = external(0)
    async with manager.lease() as slot:
        assert slot.runtime is not None


async def test_switching_the_endpoint_re_measures_the_window(
    external: Callable[[int], tuple[Any, _ContextRuntime]],
) -> None:
    manager, runtime = external(32_768)
    async with manager.lease() as slot:
        assert slot.runtime is not None
    manager.configure(CleanupUpdate(timeout_seconds=8.0))
    async with manager.lease() as slot:
        assert slot.runtime is not None
    assert runtime.asks == 2
