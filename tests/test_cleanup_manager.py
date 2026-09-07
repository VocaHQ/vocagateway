"""Preference precedence, runtime ownership, and the idle unload.

The manager is where "what did the operator ask for" meets "what is actually
running". Both halves have failure modes that only show up here: an environment
variable that silently loses to a saved choice, a worker that keeps running
after the model was switched, an idle unload that fires mid-request.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from conftest import CLEANUP_MODEL_ID, FakeCleanupRuntime, FakeWorkerHost

from app import system
from app.cleanup import catalog as cleanup_catalog
from app.cleanup import host as host_module
from app.cleanup import manager as manager_module
from app.cleanup.base import (
    MINIMUM_CONTEXT_TOKENS,
    MODE_CONSERVATIVE,
    MODE_INHERIT,
    MODE_OFF,
    CleanupReason,
)
from app.cleanup.manager import CleanupUpdate, build_manager, preserve_implicit_model_selection
from app.cleanup.profile import COMPACT_PROFILE, FULL_PROFILE
from app.cleanup.transport import Endpoint
from app.cleanup.worker import LlamaServerWorker, RuntimeFlags, probe_flags, resolve_binary
from app.config import Settings
from app.runtime_config import RuntimeConfig

_TOKEN_FILL = "x" * 48


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        token=f"manager-{_TOKEN_FILL}",
        data_dir=tmp_path,
        whisper_binary=tmp_path / "whisper-cli",
        whisper_model=tmp_path / "model.bin",
        config_path=tmp_path / "config.json",
    )


def manager_for(settings: Settings, **config: object) -> object:
    run_config = RuntimeConfig(**config)  # type: ignore[arg-type]
    return build_manager(settings, run_config, settings.config_path)


def install_model(manager: Any, model_id: str = CLEANUP_MODEL_ID) -> Path:
    """Put a cleanup artifact where the manager's own model manager finds it.

    The real path, not a patched one: whether a downloaded model is picked up
    without a separate selection click is exactly what several tests below are
    about, and a stubbed `model_path` would answer that question for them.
    """
    model = cleanup_catalog.cleanup_model(model_id)
    assert model is not None
    installed = manager.models.models_dir / "llama.cpp" / model.filename
    installed.parent.mkdir(parents=True, exist_ok=True)
    installed.write_bytes(b"gguf")
    return installed


def usable_manager(settings: Settings, **config: object) -> Any:
    """A manager with both halves of a working deployment: a runtime and a model."""
    manager = manager_for(settings, **config)
    manager.host = FakeWorkerHost(FakeCleanupRuntime())
    install_model(manager)
    return manager


def test_the_gateway_default_is_on_and_inert_until_a_model_is_installed(
    settings: Settings,
) -> None:
    """Shipped on, but a gateway with nothing to run it on corrects nothing.

    `inherit` resolving to `off` here is what keeps the default free: a
    deployment that never installs a cleanup model returns exactly the packets
    it returned before the feature existed.
    """
    manager = manager_for(settings)
    assert manager.enabled is True
    assert manager.usable is False
    assert manager.options(MODE_INHERIT).mode == MODE_OFF
    # An explicit request is not silently downgraded: it runs, finds no model,
    # and is answered with a reason rather than with silence.
    assert manager.options(MODE_CONSERVATIVE).mode == MODE_CONSERVATIVE


def test_a_pre_cleanup_config_stays_inert_even_with_a_leftover_model(
    settings: Settings,
) -> None:
    """A missing key on an already-written config is not consent to correct.

    A leftover GGUF from an earlier cleanup experiment would otherwise start
    editing transcripts on upgrade, because the only-installed-model fallback
    would pick it up the moment enabled inherited on.
    """
    settings.config_path.write_text(json.dumps({"engine": "moonshine"}), encoding="utf-8")
    manager = build_manager(
        settings, RuntimeConfig.load(settings.config_path), settings.config_path
    )
    manager.host = FakeWorkerHost(FakeCleanupRuntime())
    install_model(manager)
    assert manager.enabled is False
    assert manager.usable is False
    assert manager.options(MODE_INHERIT).mode == MODE_OFF


def test_a_downloaded_model_needs_no_second_selection(settings: Settings) -> None:
    manager = manager_for(settings)
    manager.host = FakeWorkerHost(FakeCleanupRuntime())
    assert manager.model_id is None
    install_model(manager)
    assert manager.model_id == CLEANUP_MODEL_ID
    assert manager.usable is True
    assert manager.options(MODE_INHERIT).mode == MODE_CONSERVATIVE


def test_a_second_installed_model_restores_the_choice(settings: Settings) -> None:
    """Two candidates and no saved choice is a decision the operator has to make."""
    manager = manager_for(settings)
    install_model(manager)
    install_model(manager, "cleanup:qwen3-1.7b")
    assert manager.model_id is None
    manager.configure(CleanupUpdate(model_id="cleanup:qwen3-1.7b"))
    assert manager.model_id == "cleanup:qwen3-1.7b"


def test_starting_a_second_download_preserves_the_sole_model_choice(settings: Settings) -> None:
    manager = manager_for(settings)
    install_model(manager)
    assert manager.runtime_config.cleanup_model is None

    preserve_implicit_model_selection(manager)
    install_model(manager, "cleanup:qwen3-1.7b")

    assert manager.runtime_config.cleanup_model == CLEANUP_MODEL_ID
    assert manager.model_id == CLEANUP_MODEL_ID


def test_an_environment_model_is_not_copied_into_saved_preferences(settings: Settings) -> None:
    manager = manager_for(replace(settings, cleanup_model=CLEANUP_MODEL_ID))
    install_model(manager)

    preserve_implicit_model_selection(manager)

    assert manager.runtime_config.cleanup_model is None


def test_an_explicit_opt_in_cannot_override_an_operator_who_said_no(
    settings: Settings,
) -> None:
    """Asking for cleanup does not install a model or reopen a disabled runtime."""
    manager = manager_for(settings, cleanup_enabled=False, cleanup_mode=MODE_CONSERVATIVE)
    assert manager.options(MODE_CONSERVATIVE).mode == MODE_OFF


def test_inherit_follows_the_gateway_default(settings: Settings) -> None:
    manager = usable_manager(settings, cleanup_enabled=True, cleanup_mode=MODE_CONSERVATIVE)
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
    assert manager.preferences.timeout_seconds == 30
    manager.configure(CleanupUpdate(timeout_seconds=0.01))
    assert manager.preferences.timeout_seconds == 1


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
    worker = LlamaServerWorker(binary, model, context_tokens=MINIMUM_CONTEXT_TOKENS, cpu_threads=2)

    arguments = worker._arguments(9999)

    assert arguments[0] == str(binary)
    assert "--host" in arguments and "127.0.0.1" in arguments
    assert "--jinja" in arguments
    assert "--no-webui" in arguments
    assert arguments[arguments.index("--ctx-size") + 1] == str(MINIMUM_CONTEXT_TOKENS)
    assert arguments[arguments.index("--batch-size") + 1] == str(COMPACT_PROFILE.batch_tokens)
    assert arguments[arguments.index("--ubatch-size") + 1] == str(COMPACT_PROFILE.ubatch_tokens)
    assert arguments[arguments.index("--cache-type-k") + 1] == COMPACT_PROFILE.kv_cache_type
    # No flash attention was advertised by this stub binary, so the V cache
    # falls back rather than launching a server that refuses to start.
    assert arguments[arguments.index("--cache-type-v") + 1] == "f16"
    assert arguments[arguments.index("--threads") + 1] == "2"
    # A credential of the worker's own: the client's bearer token never travels.
    assert worker.api_key in arguments
    assert len(worker.api_key) >= 24
    # argv, never a shell string.
    assert all(isinstance(argument, str) for argument in arguments)


def test_the_worker_caps_threads_the_same_way_speech_engines_do(tmp_path: Path) -> None:
    binary = tmp_path / "llama-server"
    binary.write_text("#!/bin/sh\n")
    binary.chmod(0o755)
    model = tmp_path / "model.gguf"
    model.write_bytes(b"gguf")
    worker = LlamaServerWorker(binary, model, context_tokens=MINIMUM_CONTEXT_TOKENS, cpu_threads=0)

    arguments = worker._arguments(9999)

    assert arguments[arguments.index("--threads") + 1] == str(system.inference_thread_count(0))


def test_the_full_profile_launches_with_a_larger_window_and_f16_cache(tmp_path: Path) -> None:
    binary = tmp_path / "llama-server"
    binary.write_text("#!/bin/sh\n")
    binary.chmod(0o755)
    model = tmp_path / "model.gguf"
    model.write_bytes(b"gguf")
    worker = LlamaServerWorker(binary, model, profile=FULL_PROFILE, cpu_threads=2)

    arguments = worker._arguments(9999)

    assert arguments[arguments.index("--ctx-size") + 1] == str(FULL_PROFILE.context_tokens)
    assert arguments[arguments.index("--cache-type-k") + 1] == "f16"
    assert arguments[arguments.index("--batch-size") + 1] == str(FULL_PROFILE.batch_tokens)


def test_catalog_models_declare_a_window_ceiling_the_profiles_fit_inside() -> None:
    """The catalog records what a model supports; the profile picks what to use."""
    assert cleanup_catalog.CLEANUP_CATALOG
    assert all(
        model.maximum_context_tokens >= FULL_PROFILE.context_tokens
        for model in cleanup_catalog.CLEANUP_CATALOG
    )


def test_a_model_ceiling_never_raises_the_compact_window() -> None:
    """A 40k trained window must not undo the low-end profile on a 4 GB host."""
    assert host_module._context_tokens("cleanup:qwen3-0.6b", COMPACT_PROFILE) == (
        MINIMUM_CONTEXT_TOKENS
    )
    assert host_module._context_tokens("cleanup:qwen3-0.6b", FULL_PROFILE) == (
        FULL_PROFILE.context_tokens
    )
    # An entry with no recorded ceiling gets the floor, whatever the profile asks.
    assert host_module._context_tokens("cleanup:unknown", FULL_PROFILE) == MINIMUM_CONTEXT_TOKENS


def test_the_probed_flags_enable_a_quantized_value_cache(tmp_path: Path) -> None:
    """A build that advertises the value form gets `--flash-attn on` and q8_0 V."""
    binary = tmp_path / "llama-server"
    binary.write_text("#!/bin/sh\n")
    binary.chmod(0o755)
    model = tmp_path / "model.gguf"
    model.write_bytes(b"gguf")
    worker = LlamaServerWorker(binary, model, cpu_threads=2)
    worker._flags = RuntimeFlags(
        flash_attention=("--flash-attn", "on"),
        no_context_shift=True,
        reasoning_budget=True,
    )

    arguments = worker._arguments(9999)

    assert arguments[arguments.index("--cache-type-v") + 1] == COMPACT_PROFILE.kv_cache_type
    assert arguments[arguments.index("--flash-attn") + 1] == "on"
    assert "--no-context-shift" in arguments
    assert arguments[arguments.index("--reasoning-budget") + 1] == "0"


def test_an_older_binary_is_offered_only_the_flags_it_advertises(tmp_path: Path) -> None:
    """`--flash-attn` used to be a bare switch, and the rest did not exist."""
    binary = tmp_path / "llama-server"
    binary.write_text("#!/bin/sh\n")
    binary.chmod(0o755)
    model = tmp_path / "model.gguf"
    model.write_bytes(b"gguf")
    worker = LlamaServerWorker(binary, model, cpu_threads=2)
    worker._flags = RuntimeFlags(flash_attention=("--flash-attn",))

    arguments = worker._arguments(9999)

    # Last in argv, so the bare switch has nothing behind it to swallow.
    assert arguments[-1] == "--flash-attn"
    assert "on" not in arguments
    assert "--no-context-shift" not in arguments
    assert "--reasoning-budget" not in arguments


def test_flags_are_read_from_the_binary_help_text(tmp_path: Path) -> None:
    """The gateway launches whatever `llama-server` is on PATH, so it asks."""
    modern = tmp_path / "modern"
    modern.write_text(
        "#!/bin/sh\n"
        "echo '-fa,   --flash-attn [on|off|auto]  set Flash Attention use'\n"
        "echo '--context-shift, --no-context-shift  whether to use context shift'\n"
        "echo '--reasoning-budget N  token budget for thinking'\n"
    )
    modern.chmod(0o755)
    assert probe_flags(modern) == RuntimeFlags(
        flash_attention=("--flash-attn", "on"),
        no_context_shift=True,
        reasoning_budget=True,
    )

    ancient = tmp_path / "ancient"
    ancient.write_text("#!/bin/sh\necho '-fa, --flash-attn  enable Flash Attention'\n")
    ancient.chmod(0o755)
    assert probe_flags(ancient) == RuntimeFlags(flash_attention=("--flash-attn",))

    missing = tmp_path / "absent"
    assert probe_flags(missing) == RuntimeFlags()


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
    ("VOCAGATEWAY_CLEANUP_ENDPOINT", "my-cleanup:8080", "cleanup_endpoint", ("my-cleanup", 8080)),
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
    monkeypatch.setenv("VOCAGATEWAY_TOKEN", f"env-{_TOKEN_FILL}")
    monkeypatch.setenv("VOCAGATEWAY_DATA_DIR", str(tmp_path))
    monkeypatch.setenv(name, raw)
    assert getattr(Settings.from_env(), field) == expected


def test_an_unset_cleanup_switch_is_neither_on_nor_off(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("VOCAGATEWAY_TOKEN", f"env-{_TOKEN_FILL}")
    monkeypatch.setenv("VOCAGATEWAY_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("VOCAGATEWAY_CLEANUP_ENABLED", raising=False)
    assert Settings.from_env().cleanup_enabled is None


ROUTABLE_ENDPOINTS = ("https://api.example.com/v1", "example.com:8080", "8.8.8.8:80", "nope")
REMOVED_SIDECAR_ENDPOINTS = ("cleanup:8080", "cleanup:1234", "CLEANUP:8080")


@pytest.mark.parametrize("endpoint", ROUTABLE_ENDPOINTS)
def test_a_routable_cleanup_endpoint_is_refused_at_startup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, endpoint: str
) -> None:
    """Cleanup runs on the gateway. A remote address would quietly make that false."""
    monkeypatch.setenv("VOCAGATEWAY_TOKEN", f"env-{_TOKEN_FILL}")
    monkeypatch.setenv("VOCAGATEWAY_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("VOCAGATEWAY_CLEANUP_ENDPOINT", endpoint)
    with pytest.raises(RuntimeError):
        Settings.from_env()


@pytest.mark.parametrize("endpoint", REMOVED_SIDECAR_ENDPOINTS)
def test_the_old_cleanup_sidecar_endpoint_is_refused_at_startup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, endpoint: str
) -> None:
    """The Compose service named cleanup is gone; keep using it and the
    in-image runtime sits unused while every request hits a dead host.
    """
    monkeypatch.setenv("VOCAGATEWAY_TOKEN", f"env-{_TOKEN_FILL}")
    monkeypatch.setenv("VOCAGATEWAY_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("VOCAGATEWAY_CLEANUP_ENDPOINT", endpoint)
    with pytest.raises(RuntimeError, match="sidecar"):
        Settings.from_env()


class _NeverReadyWorker:
    """A worker whose model load never finishes, like a cold multi-gigabyte GGUF."""

    def __init__(self, *_: object, **_unused: object) -> None:
        self.is_running = False
        self.starts = 0
        self.endpoint = Endpoint("127.0.0.1", 1)
        self.released = asyncio.Event()

    async def ensure_started(self, *, budget: float = 0) -> bool:
        self.starts += 1
        await self.released.wait()
        self.is_running = True
        return True

    def stop(self, *, force: bool = False) -> None:
        self.is_running = False


@pytest.fixture
def cold_worker(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> _NeverReadyWorker:
    built = _NeverReadyWorker()
    monkeypatch.setattr(host_module.worker, "LlamaServerWorker", lambda *attempt, **kind: built)
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
    await asyncio.sleep(0)
    assert manager.loading is False
    assert manager.status().state == "ready"
    manager.host.stop()


async def _assert_runtime_unavailable(worker_host: Any, model: Path) -> None:
    assert await worker_host.runtime(CLEANUP_MODEL_ID, model) is None
    await asyncio.sleep(0)


async def test_concurrent_requests_share_one_load_rather_than_starting_several(
    settings: Settings, cold_worker: _NeverReadyWorker, tmp_path: Path
) -> None:
    worker_host = host_module.WorkerHost(settings, RuntimeConfig())
    model = tmp_path / "model.gguf"
    await _assert_runtime_unavailable(worker_host, model)
    await _assert_runtime_unavailable(worker_host, model)
    await _assert_runtime_unavailable(worker_host, model)
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


class _ContextRuntime:
    """An operator-run server that reports a context window and counts the asks."""

    def __init__(self, tokens: int) -> None:
        self.tokens = tokens
        self.asks = 0
        self.model_id = CLEANUP_MODEL_ID

    async def context_is_sufficient(self) -> bool:
        self.asks += 1
        return not self.tokens or self.tokens >= MINIMUM_CONTEXT_TOKENS

    async def context_tokens(self) -> int:
        return self.tokens

    async def available(self) -> bool:
        return True


@pytest.fixture
def external(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> Callable[[int], tuple[Any, _ContextRuntime]]:
    """A manager pointed at an operator-run endpoint reporting a given window."""

    def build(tokens: int) -> tuple[Any, _ContextRuntime]:
        runtime = _ContextRuntime(tokens)
        monkeypatch.setattr(manager_module, "LlamaServerRuntime", lambda *attempt, **kind: runtime)
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


async def test_the_minimum_context_window_is_accepted(
    external: Callable[[int], tuple[Any, _ContextRuntime]],
) -> None:
    manager, runtime = external(MINIMUM_CONTEXT_TOKENS)
    async with manager.lease() as slot:
        assert slot.runtime is not None
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
