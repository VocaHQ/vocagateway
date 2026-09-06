"""Preference precedence, runtime ownership, and the idle unload.

The manager is where "what did the operator ask for" meets "what is actually
running". Both halves have failure modes that only show up here: an environment
variable that silently loses to a saved choice, a worker that keeps running
after the model was switched, an idle unload that fires mid-request.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import replace
from pathlib import Path

import pytest
from conftest import CLEANUP_MODEL_ID, FakeCleanupRuntime, FakeWorkerHost

from app.cleanup import host as host_module
from app.cleanup.base import MODE_CONSERVATIVE, MODE_INHERIT, MODE_OFF, CleanupReason
from app.cleanup.manager import CleanupUpdate, build_manager
from app.cleanup.transport import Endpoint
from app.cleanup.worker import LlamaServerWorker, resolve_binary
from app.config import Settings
from app.runtime_config import RuntimeConfig


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
    manager.configure(CleanupUpdate(model_id="cleanup:qwen3-1.7b"))
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
    """A worker whose model load never finishes, like a cold multi-gigabyte GGUF."""

    def __init__(self, *_: object, **__: object) -> None:
        self.is_running = False
        self.starts = 0
        self.endpoint = Endpoint("127.0.0.1", 1)
        self.released = asyncio.Event()

    async def ensure_started(self, *, budget: float = 0.0) -> bool:
        self.starts += 1
        await self.released.wait()
        self.is_running = True
        return True

    def stop(self, *, force: bool = False) -> None:
        self.is_running = False


@pytest.fixture
def cold_worker(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> _NeverReadyWorker:
    built = _NeverReadyWorker()
    monkeypatch.setattr(host_module.worker, "LlamaServerWorker", lambda *a, **k: built)
    monkeypatch.setattr(host_module.worker, "resolve_binary", lambda _=None: tmp_path / "llama")
    return built


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


async def test_an_operator_warm_up_does_wait_for_the_load(
    settings: Settings, cold_worker: _NeverReadyWorker, tmp_path: Path
) -> None:
    """The one caller that asked to pay the cold-load cost deliberately."""
    worker_host = host_module.WorkerHost(settings, RuntimeConfig())
    warming = asyncio.create_task(
        worker_host.runtime(CLEANUP_MODEL_ID, tmp_path / "model.gguf", wait=True)
    )
    await asyncio.sleep(0)
    assert not warming.done()
    cold_worker.released.set()
    assert await asyncio.wait_for(warming, timeout=1.0) is not None
    worker_host.stop()


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
