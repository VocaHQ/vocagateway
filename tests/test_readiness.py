from __future__ import annotations

from pathlib import Path

from app.engines import StaticEngineProvider
from app.models.base import EngineHealth, TranscriptionOptions
from app.readiness import ReadinessMonitor

READINESS_CACHE_SECONDS = 30


class WarmableFakeEngine:
    def __init__(self) -> None:
        self.health_calls = 0
        self.warmup_calls = 0

    async def health(self) -> EngineHealth:
        self.health_calls += 1
        return EngineHealth(ready=True, name="warmable-fake")

    async def warmup(self) -> int:
        self.warmup_calls += 1
        return 1024

    async def transcribe(self, audio_path: Path, options: TranscriptionOptions) -> str:
        return "unused"


class IdleOffloadFakeProvider(StaticEngineProvider):
    def __init__(self, engine: WarmableFakeEngine) -> None:
        super().__init__(engine)
        self.model_is_offloaded = False

    def offload_if_idle(self, *, now: float | None = None) -> bool:
        self.model_is_offloaded = True
        return True


async def test_readiness_caches_probes_and_tracks_f8ea4() -> None:
    engine = WarmableFakeEngine()
    monitor = ReadinessMonitor(StaticEngineProvider(engine), ttl_seconds=READINESS_CACHE_SECONDS)

    first = await monitor.probe()
    second = await monitor.probe()
    await monitor.warmup()
    details = await monitor.details()

    assert first.ready is True
    assert second == first
    assert (engine.health_calls, engine.warmup_calls) == (3, 1)
    assert details.warmup_state == "complete"
    assert details.warmed_bytes == 1024


async def test_readiness_reports_offload_and_reload_state() -> None:
    engine = WarmableFakeEngine()
    provider = IdleOffloadFakeProvider(engine)
    monitor = ReadinessMonitor(provider)
    await monitor.warmup()

    monitor._check_idle_offload(provider)
    offloaded = await monitor.details()

    assert offloaded.warmup_state == "offloaded"
    assert offloaded.warmed_bytes == 0

    provider.model_is_offloaded = False
    reloaded = await monitor.details()

    assert reloaded.warmup_state == "complete"
    assert reloaded.warmed_bytes == 1024
