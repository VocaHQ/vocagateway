"""The rest of the surface: capabilities, one-shot, streaming, settings, privacy.

The through-line is that a client which knows nothing about cleanup must not be
able to tell it exists — same request, same response, same behaviour — while a
client that does ask gets a bounded, honest answer about what happened.
"""

from __future__ import annotations

import json
from array import array
from collections.abc import AsyncIterator
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from conftest import (
    CLEANUP_MODEL_ID,
    TOKEN,
    FakeCleanupRuntime,
    FakeEngine,
    FakeNormalizer,
    enable_cleanup,
)
from fastapi.testclient import TestClient
from pytest import MonkeyPatch
from starlette.status import (
    HTTP_200_OK,
    HTTP_401_UNAUTHORIZED,
    HTTP_404_NOT_FOUND,
    HTTP_422_UNPROCESSABLE_CONTENT,
)

from app import admin_queries
from app.cleanup import catalog as cleanup_catalog
from app.config import Settings
from app.main import create_app
from app.models.base import EngineHealth
from app.models.moonshine import MoonshineEngine

SPOKEN = "we was going to leave early but the train was late"
CORRECTED = "We were going to leave early, but the train was late."
AUTH = {"Authorization": f"Bearer {TOKEN}"}
TRANSCRIPTIONS = "/v1/audio/transcriptions"
CAPABILITIES = "/v1/capabilities"
CLEANUP_CONFIG = "/v1/admin/cleanup"
UPLOAD = {"file": ("clip.wav", b"x" * 200, "audio/wav")}


@pytest.fixture
async def gateway(settings: Settings) -> AsyncIterator[tuple[httpx.AsyncClient, Any]]:
    app = create_app(settings, engine=FakeEngine(SPOKEN), normalizer=FakeNormalizer())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        yield client, app


async def test_capabilities_needs_a_token(gateway: Any) -> None:
    client, _ = gateway
    assert (await client.get(CAPABILITIES)).status_code == HTTP_401_UNAUTHORIZED


async def test_capabilities_reports_cleanup_off_until_it_is_usable(gateway: Any) -> None:
    client, _ = gateway
    payload = (await client.get(CAPABILITIES, headers=AUTH)).json()["cleanup"]
    assert payload["supported"] is False
    assert payload["enabled"] is False
    assert payload["default_mode"] == "off"
    assert payload["modes"] == ["off"]


async def test_capabilities_separates_offered_from_tested_languages(gateway: Any) -> None:
    """An offered language is not a tested one until an evaluation says so."""
    client, app = gateway
    enable_cleanup(app, FakeCleanupRuntime(CORRECTED))
    payload = (await client.get(CAPABILITIES, headers=AUTH)).json()["cleanup"]
    assert payload["supported"] is True
    assert payload["enabled"] is True
    assert payload["modes"] == ["off", "conservative"]
    assert payload["languages"] == ["en", "hi", "hinglish_roman"]
    assert payload["evaluated_languages"] == []
    assert payload["prompt_version"] == "cleanup-v1"


async def test_capabilities_never_leaks_the_runtime_address(gateway: Any) -> None:
    client, app = gateway
    enable_cleanup(app, FakeCleanupRuntime(CORRECTED))
    body = (await client.get(CAPABILITIES, headers=AUTH)).text
    for secret in ("127.0.0.1", "llama-server", "/tmp", "endpoint", "binary"):
        assert secret not in body


async def test_one_shot_default_is_unchanged_by_the_feature(gateway: Any) -> None:
    """The OpenAI-compatible shape and the raw text it returns must not move."""
    client, app = gateway
    runtime = FakeCleanupRuntime(CORRECTED)
    enable_cleanup(app, runtime)
    response = await client.post(TRANSCRIPTIONS, files=UPLOAD, headers=AUTH)
    assert response.json() == {"text": SPOKEN}
    assert runtime.calls == []
    assert "X-Voca-Cleanup-Status" not in response.headers


async def test_one_shot_opt_in_corrects_and_reports_in_headers(gateway: Any) -> None:
    client, app = gateway
    enable_cleanup(app, FakeCleanupRuntime(CORRECTED))
    response = await client.post(
        TRANSCRIPTIONS,
        files=UPLOAD,
        data={"cleanup": "conservative", "language": "en"},
        headers=AUTH,
    )
    # The body stays exactly `{"text": ...}`; status travels beside it.
    assert list(response.json()) == ["text"]
    assert response.json()["text"] == CORRECTED
    assert response.headers["X-Voca-Cleanup-Status"] == "applied"
    assert response.headers["X-Voca-Cleanup-Model"] == CLEANUP_MODEL_ID


async def test_one_shot_opt_in_still_answers_when_cleanup_fails(gateway: Any) -> None:
    client, app = gateway
    enable_cleanup(app, FakeCleanupRuntime(TimeoutError()))
    response = await client.post(
        TRANSCRIPTIONS,
        files=UPLOAD,
        data={"cleanup": "conservative", "language": "en"},
        headers=AUTH,
    )
    assert response.status_code == HTTP_200_OK
    assert response.headers["X-Voca-Cleanup-Status"] == "fallback"
    assert response.headers["X-Voca-Cleanup-Reason"] == "timeout"


async def test_one_shot_rejects_an_unknown_cleanup_value(gateway: Any) -> None:
    client, _ = gateway
    response = await client.post(
        TRANSCRIPTIONS, files=UPLOAD, data={"cleanup": "creative"}, headers=AUTH
    )
    assert response.status_code == HTTP_422_UNPROCESSABLE_CONTENT
    assert response.json()["error"]["code"] == "invalid_cleanup"


async def test_the_mic_test_shows_both_texts(gateway: Any) -> None:
    client, app = gateway
    enable_cleanup(app, FakeCleanupRuntime(CORRECTED))
    response = await client.post(
        "/v1/admin/test-transcription?language=en&cleanup_mode=conservative",
        content=b"x" * 200,
        headers={**AUTH, "Content-Type": "audio/wav"},
    )
    payload = response.json()
    assert payload["original_transcript"] == SPOKEN
    assert payload["transcript"] == CORRECTED
    assert payload["cleanup"]["status"] == "applied"


async def test_the_mic_test_stays_raw_by_default(gateway: Any) -> None:
    """A benchmark that quietly corrected its output would measure the wrong thing."""
    client, app = gateway
    runtime = FakeCleanupRuntime(CORRECTED)
    enable_cleanup(app, runtime)
    payload = (
        await client.post(
            "/v1/admin/test-transcription?language=en",
            content=b"x" * 200,
            headers={**AUTH, "Content-Type": "audio/wav"},
        )
    ).json()
    assert payload["transcript"] == SPOKEN
    assert payload["cleanup"] is None
    assert runtime.calls == []


async def test_cleanup_config_updates_do_not_reset_the_engine(gateway: Any) -> None:
    client, app = gateway
    run_config = app.state.ctx.cleanup.runtime_config
    run_config.engine = "moonshine"
    run_config.cpu_threads = 6
    run_config.idle_offload_enabled = True

    response = await client.put(
        CLEANUP_CONFIG, json={"enabled": True, "mode": "conservative"}, headers=AUTH
    )

    assert response.json()["enabled"] is True
    assert run_config.engine == "moonshine"
    assert run_config.cpu_threads == 6
    assert run_config.idle_offload_enabled is True


async def test_a_partial_update_leaves_unmentioned_fields_alone(gateway: Any) -> None:
    client, app = gateway
    run_config = app.state.ctx.cleanup.runtime_config
    await client.put(
        CLEANUP_CONFIG,
        json={"enabled": True, "model_id": CLEANUP_MODEL_ID, "timeout_seconds": 8},
        headers=AUTH,
    )
    await client.put(CLEANUP_CONFIG, json={"timeout_seconds": 3}, headers=AUTH)
    assert run_config.cleanup_enabled is True
    assert run_config.cleanup_model == CLEANUP_MODEL_ID
    assert run_config.cleanup_timeout_seconds == 3


async def test_an_unknown_model_is_refused(gateway: Any) -> None:
    client, _ = gateway
    response = await client.put(CLEANUP_CONFIG, json={"model_id": "cleanup:nope"}, headers=AUTH)
    assert response.status_code == HTTP_422_UNPROCESSABLE_CONTENT


async def test_an_unknown_model_cannot_be_downloaded(gateway: Any) -> None:
    client, _ = gateway
    response = await client.post("/v1/admin/cleanup/models/cleanup:nope/download", headers=AUTH)
    assert response.status_code == HTTP_404_NOT_FOUND


async def test_starting_a_second_download_keeps_the_implicit_selection(
    gateway: Any, monkeypatch: MonkeyPatch
) -> None:
    client, app = gateway
    manager = app.state.ctx.cleanup
    installed = cleanup_catalog.cleanup_model(CLEANUP_MODEL_ID)
    assert installed is not None
    model_path = manager.models.models_dir / "llama.cpp" / installed.filename
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_bytes(b"gguf")
    monkeypatch.setattr(manager.models, "start_download", lambda _model_id: None)

    response = await client.post(
        "/v1/admin/cleanup/models/cleanup:qwen3-0.6b-q4/download", headers=AUTH
    )

    assert response.status_code == HTTP_200_OK
    assert manager.runtime_config.cleanup_model == CLEANUP_MODEL_ID


async def test_the_cleanup_catalog_lists_provenance(gateway: Any) -> None:
    client, _ = gateway
    entries = (await client.get("/v1/admin/cleanup/models", headers=AUTH)).json()
    assert [entry["id"] for entry in entries] == [
        "cleanup:qwen3-0.6b",
        "cleanup:qwen3-0.6b-q4",
        "cleanup:qwen3-1.7b",
        "cleanup:gemma-3-270m",
        "cleanup:gemma-3-1b",
    ]
    for entry in entries:
        # A model with no pinned digest must not be installable at all.
        assert entry["installable"] is (entry["sha256"] is not None)
        assert entry["evaluated_languages"] == []
    qwen = [entry for entry in entries if entry["id"].startswith("cleanup:qwen3-")]
    assert qwen
    assert all(entry["upstream_model"].startswith("Qwen/") for entry in qwen)
    compact = next(entry for entry in entries if entry["id"] == "cleanup:qwen3-0.6b-q4")
    assert compact["languages"] == ["en"]
    assert compact["conversion_source"].startswith("llama.cpp project")
    gemma_small = next(entry for entry in entries if entry["id"] == "cleanup:gemma-3-270m")
    assert gemma_small["languages"] == ["en"]
    assert gemma_small["license_name"] == "Gemma"
    assert gemma_small["upstream_model"] == "google/gemma-3-270m-it"
    gemma_1b = next(entry for entry in entries if entry["id"] == "cleanup:gemma-3-1b")
    assert gemma_1b["languages"] == ["en"]
    assert gemma_1b["license_name"] == "Gemma"
    assert gemma_1b["upstream_model"] == "google/gemma-3-1b-it"


async def test_a_cleanup_model_is_never_offered_as_a_speech_engine(gateway: Any) -> None:
    client, _ = gateway
    models = (await client.get("/v1/models", headers=AUTH)).json()
    assert all("cleanup:" not in entry["id"] for entry in models)
    admin_models = (await client.get("/v1/admin/models", headers=AUTH)).json()
    assert all("cleanup:" not in entry["id"] for entry in admin_models)


async def test_environment_overrides_are_shown_as_locked(settings: Settings) -> None:
    locked = replace_settings(settings, cleanup_enabled=True, cleanup_mode="conservative")
    app = create_app(locked, engine=FakeEngine(SPOKEN), normalizer=FakeNormalizer())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        payload = (await client.get(CLEANUP_CONFIG, headers=AUTH)).json()
        # A save must not pretend it changed something the environment owns.
        await client.put(CLEANUP_CONFIG, json={"enabled": False}, headers=AUTH)
        after = (await client.get(CLEANUP_CONFIG, headers=AUTH)).json()
    assert set(payload["locked_settings"]) == {"enabled", "mode"}
    assert after["enabled"] is True


def replace_settings(settings: Settings, **changes: Any) -> Settings:
    from dataclasses import replace

    return replace(settings, **changes)


async def test_the_settings_page_renders_transcripts_as_text(gateway: Any) -> None:
    client, app = gateway
    enable_cleanup(app, FakeCleanupRuntime(CORRECTED))
    body = (await client.get("/ui/partials/cleanup", headers=AUTH)).text
    assert "Transcript cleanup" in body
    assert "fixes grammar and punctuation after speech" in " ".join(body.split())
    # The caveat about recognition errors is not optional wording.
    assert "never sees your audio" in " ".join(body.split())


@pytest.mark.parametrize("host_os", ["Darwin", "Linux"])
async def test_diagnostics_carry_the_cleanup_block_and_no_text(
    gateway: Any, monkeypatch: MonkeyPatch, host_os: str
) -> None:
    """Run for both platforms: the install hint differs, and the rule must not.

    Naming a host explicitly is the point. The advice for a Mac and the advice
    for a Linux host are different sentences, and a check written against
    whichever one the developer happened to be on is a check that fails in CI.
    """
    client, app = gateway
    monkeypatch.setattr(admin_queries.platform, "system", lambda: host_os)
    enable_cleanup(app, FakeCleanupRuntime(CORRECTED))
    await client.post(
        TRANSCRIPTIONS,
        files=UPLOAD,
        data={"cleanup": "conservative", "language": "en"},
        headers=AUTH,
    )
    response = await client.get("/v1/admin/diagnostics", headers=AUTH)
    bundle, body = response.json(), response.text
    assert '"cleanup"' in body
    assert SPOKEN not in body
    assert CORRECTED not in body
    # The cleanup block itself still describes no topology: no executable path,
    # no endpoint, no credential. One field names the runtime — `runtime_hint`,
    # which is install advice and identical on every host of a platform — and
    # nothing else in the block mentions it at all.
    cleanup_block = bundle["config"]["cleanup"]
    assert not {"binary", "endpoint", "api_key", "path"} & set(cleanup_block)
    described = {key: text for key, text in cleanup_block.items() if key != "runtime_hint"}
    assert "llama-server" not in json.dumps(described)
    # The runtime appears with a path only as a dependency tile — the same
    # shape, and the same redaction, as whisper.cpp and FFmpeg — so an operator
    # reading a shared bundle can tell a missing runtime from a missing model.
    tile = next(item for item in bundle["dependencies"] if item["name"] == "llama.cpp server")
    assert tile["path"] is None or "llama-server" in tile["path"]


async def test_metrics_count_cleanup_outcomes_without_text(gateway: Any) -> None:
    client, app = gateway
    enable_cleanup(app, FakeCleanupRuntime(CORRECTED, TimeoutError()))
    for _ in range(2):
        await client.post(
            TRANSCRIPTIONS,
            files=UPLOAD,
            data={"cleanup": "conservative", "language": "en"},
            headers=AUTH,
        )
    metrics = (await client.get("/v1/admin/status", headers=AUTH)).json()["metrics"]
    assert metrics["cleanup_applied"] == 1
    assert metrics["cleanup_fallback"] == 1
    assert metrics["cleanup_reasons"] == {"timeout": 1}


# ------------------------------------------------------------------ streaming

STREAM_TOKEN = "stream-" + ("x" * 48)


class FakeStream:
    def __init__(self) -> None:
        self.listener: Any = None
        self.closed = False

    def add_listener(self, listener: Any) -> None:
        self.listener = listener

    def add_audio(self, samples: list[float], sample_rate: int) -> None:
        self.listener(SimpleNamespace(line=SimpleNamespace(text=SPOKEN, line_id=1)))

    def stop(self) -> Any:
        return SimpleNamespace(lines=[SimpleNamespace(text=SPOKEN, line_id=1)])

    def close(self) -> None:
        self.closed = True


def streaming_app(tmp_path: Path, monkeypatch: MonkeyPatch, runtime: FakeCleanupRuntime) -> Any:
    settings = Settings(
        token=STREAM_TOKEN,
        data_dir=tmp_path,
        whisper_binary=tmp_path / "whisper-cli",
        whisper_model=tmp_path / "model.bin",
        config_path=tmp_path / "config.json",
    )
    model_root = tmp_path / "moonshine"
    model_root.mkdir()
    (model_root / ".vocagateway-model.json").write_text(
        '{"language":"en","model_path":"model","model_arch":5}', encoding="utf-8"
    )
    engine = MoonshineEngine(model_root)

    async def create_stream() -> FakeStream:
        return FakeStream()

    async def health() -> EngineHealth:
        return EngineHealth(ready=True, name="moonshine:en")

    monkeypatch.setattr(engine, "create_stream", create_stream)
    monkeypatch.setattr(engine, "health", health)
    app = create_app(settings, engine=engine)
    enable_cleanup(app, runtime)
    return app


def test_streaming_partials_are_never_rewritten(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """A corrected partial that changes again a word later is worse than none."""
    runtime = FakeCleanupRuntime(CORRECTED)
    app = streaming_app(tmp_path, monkeypatch, runtime)
    with (
        TestClient(app) as client,
        client.websocket_connect(
            "/v1/stream", headers={"Authorization": f"Bearer {STREAM_TOKEN}"}
        ) as websocket,
    ):
        websocket.send_json(
            {"type": "start", "sample_rate": 16_000, "style": "formal", "language": "en"}
        )
        assert websocket.receive_json()["type"] == "ready"
        websocket.send_bytes(array("f", [0.1, -0.1]).tobytes())
        partial = websocket.receive_json()
        assert partial == {"type": "partial", "transcript": SPOKEN}
        assert runtime.calls == []

        websocket.send_json({"type": "finish"})
        complete = websocket.receive_json()

    assert complete["type"] == "complete"
    assert complete["transcript"] == CORRECTED
    assert complete["original_transcript"] == SPOKEN
    assert complete["cleanup"]["status"] == "applied"
    # Exactly one cleanup pass per stream, on the final text only.
    assert len(runtime.calls) == 1


def test_streaming_opt_out_leaves_the_packet_as_it_was(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    runtime = FakeCleanupRuntime(CORRECTED)
    app = streaming_app(tmp_path, monkeypatch, runtime)
    with (
        TestClient(app) as client,
        client.websocket_connect(
            "/v1/stream", headers={"Authorization": f"Bearer {STREAM_TOKEN}"}
        ) as websocket,
    ):
        websocket.send_json(
            {
                "type": "start",
                "sample_rate": 16_000,
                "style": "formal",
                "language": "en",
                "cleanup": "off",
            }
        )
        websocket.receive_json()
        websocket.send_json({"type": "finish"})
        complete = websocket.receive_json()

    assert complete == {
        "type": "complete",
        "transcript": "We was going to leave early but the train was late.",
    }
    assert runtime.calls == []


def test_streaming_rejects_an_unknown_cleanup_mode(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    app = streaming_app(tmp_path, monkeypatch, FakeCleanupRuntime(CORRECTED))
    with (
        TestClient(app) as client,
        client.websocket_connect(
            "/v1/stream", headers={"Authorization": f"Bearer {STREAM_TOKEN}"}
        ) as websocket,
    ):
        websocket.send_json({"type": "start", "sample_rate": 16_000, "cleanup": "creative"})
        assert websocket.receive_json()["type"] == "error"


def test_streaming_releases_the_engine_before_correcting(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """The lock and the lease must both be gone before the text model runs.

    Awaiting an LLM from under them would keep a speech model occupied for the
    whole correction, which is the reason finalization moved out of `finish()`.
    """
    seen: list[bool] = []

    class Watching(FakeCleanupRuntime):
        async def clean(self, transcript: str, language: str, *, budget_seconds: float) -> str:
            seen.append(engine.streaming_lock.locked())
            return await super().clean(transcript, language, budget_seconds=budget_seconds)

    runtime = Watching(CORRECTED)
    app = streaming_app(tmp_path, monkeypatch, runtime)
    engine = app.state.ctx.engine_provider.current()
    with (
        TestClient(app) as client,
        client.websocket_connect(
            "/v1/stream", headers={"Authorization": f"Bearer {STREAM_TOKEN}"}
        ) as websocket,
    ):
        websocket.send_json(
            {"type": "start", "sample_rate": 16_000, "style": "formal", "language": "en"}
        )
        websocket.receive_json()
        websocket.send_json({"type": "finish"})
        websocket.receive_json()

    assert seen == [False]


# ------------------------------------------------------------------- privacy

SENTINEL = "zebra-quartz-lantern-sentinel"


async def test_health_and_readiness_never_depend_on_cleanup(gateway: Any) -> None:
    """An optional corrector must not be able to take the gateway down."""
    client, app = gateway
    enable_cleanup(app, FakeCleanupRuntime(TimeoutError()), installed=False)
    live = await client.get("/health/live")
    ready = await client.get("/health/ready")
    assert live.status_code == HTTP_200_OK
    assert ready.json()["engine_ready"] is True
    assert "cleanup" not in ready.text


async def test_no_transcript_reaches_the_logs(
    gateway: Any, caplog: pytest.LogCaptureFixture
) -> None:
    client, app = gateway
    app.state.ctx.engine_provider.current().transcript = SENTINEL
    enable_cleanup(app, FakeCleanupRuntime(f"{SENTINEL} corrected."))
    with caplog.at_level("DEBUG"):
        await client.post(
            TRANSCRIPTIONS,
            files=UPLOAD,
            data={"cleanup": "conservative", "language": "en"},
            headers=AUTH,
        )
    assert SENTINEL not in caplog.text


async def test_a_deleted_session_takes_both_texts_with_it(gateway: Any) -> None:
    """The original is stored under the same retention as the transcript."""
    client, app = gateway
    enable_cleanup(app, FakeCleanupRuntime(CORRECTED))
    created = (
        await client.post(
            "/v1/sessions",
            json={
                "client_session_id": "11111111-2222-3333-4444-555555555555",
                "language": "en",
                "style": "casual",
            },
            headers=AUTH,
        )
    ).json()
    session_id = created["session_id"]
    await client.put(
        f"/v1/sessions/{session_id}/audio",
        content=b"x" * 200,
        headers={**AUTH, "Content-Type": "audio/wav"},
    )
    finished = (await client.post(f"/v1/sessions/{session_id}/finish", headers=AUTH)).json()
    assert finished["original_transcript"] == SPOKEN

    await client.delete(f"/v1/sessions/{session_id}", headers=AUTH)
    gone = await client.get(f"/v1/sessions/{session_id}", headers=AUTH)
    assert gone.status_code == HTTP_404_NOT_FOUND


async def test_the_writing_style_is_applied_exactly_once(gateway: Any) -> None:
    """Double processing would capitalise, then capitalise the capitalisation."""
    client, app = gateway
    # The runtime hands back text that is already sentence-cased and terminated;
    # a second styling pass must leave it exactly as it is.
    enable_cleanup(app, FakeCleanupRuntime(CORRECTED))
    payload = (
        await client.post(
            "/v1/admin/test-transcription?language=en&cleanup_mode=conservative",
            content=b"x" * 200,
            headers={**AUTH, "Content-Type": "audio/wav"},
        )
    ).json()
    assert payload["transcript"] == CORRECTED


async def test_the_auto_language_choice_survives_a_save_and_is_reported(gateway: Any) -> None:
    """It travels the whole way: form, schema, saved config, status, capability."""
    client, app = gateway
    enable_cleanup(app, FakeCleanupRuntime(CORRECTED))
    saved = await client.put(
        CLEANUP_CONFIG, json={"model_id": CLEANUP_MODEL_ID, "auto_language": "en"}, headers=AUTH
    )
    assert saved.json()["auto_language"] == "en"
    assert app.state.ctx.cleanup.runtime_config.cleanup_auto_language == "en"
    capability = (await client.get("/v1/capabilities", headers=AUTH)).json()["cleanup"]
    assert capability["auto_language"] == "en"


async def test_an_auto_language_off_the_allowlist_is_reported_as_unset(gateway: Any) -> None:
    """Saved as asked, but never reported as something cleanup would honour."""
    client, app = gateway
    enable_cleanup(app, FakeCleanupRuntime(CORRECTED))
    saved = await client.put(
        CLEANUP_CONFIG, json={"model_id": CLEANUP_MODEL_ID, "auto_language": "fr"}, headers=AUTH
    )
    assert saved.json()["auto_language"] == ""


async def test_clearing_the_auto_language_is_distinct_from_leaving_it_alone(
    gateway: Any,
) -> None:
    client, app = gateway
    enable_cleanup(app, FakeCleanupRuntime(CORRECTED))
    await client.put(CLEANUP_CONFIG, json={"auto_language": "en"}, headers=AUTH)
    # An absent field leaves it alone...
    await client.put(CLEANUP_CONFIG, json={"timeout_seconds": 8}, headers=AUTH)
    assert (await client.get(CLEANUP_CONFIG, headers=AUTH)).json()["auto_language"] == "en"
    # ...while an empty one is the operator saying "stop guessing".
    await client.put(CLEANUP_CONFIG, json={"auto_language": ""}, headers=AUTH)
    assert (await client.get(CLEANUP_CONFIG, headers=AUTH)).json()["auto_language"] == ""


async def test_the_cleanup_section_explains_what_each_control_does(gateway: Any) -> None:
    """A control named "Mode" with options "Conservative" and "Off" explains nothing."""
    client, app = gateway
    enable_cleanup(app, FakeCleanupRuntime(CORRECTED))
    body = _flattened(await client.get("/ui/partials/cleanup", headers=AUTH))
    assert "Conservative (recommended)" in body
    for explanation in (
        "Conservative fixes grammar and punctuation.",
        "If cleanup takes longer, keep the standard transcript.",
        "The next correction reloads the model.",
        "Choose a text model to run alongside your speech model",
    ):
        assert explanation in body


async def test_the_setup_checklist_names_the_step_that_is_not_done(gateway: Any) -> None:
    """Four preconditions share one symptom: the transcript comes back unchanged.

    Listing them turns "it does nothing" into "step four is not done".
    """
    client, app = gateway
    enable_cleanup(app, FakeCleanupRuntime(CORRECTED))
    stranded = _flattened(await client.get("/ui/partials/cleanup", headers=AUTH))
    assert "A runtime to run it" in stranded
    assert "come back uncorrected. Set one below" in stranded
    await client.put(CLEANUP_CONFIG, json={"auto_language": "en"}, headers=AUTH)
    covered = _flattened(await client.get("/ui/partials/cleanup", headers=AUTH))
    assert "come back uncorrected. Set one below" not in covered
    assert "corrected as en" in covered


async def test_a_missing_runtime_is_reported_on_the_dashboard_and_the_checklist(
    settings: Settings,
) -> None:
    """The runtime is a host requirement, so it is reported where those are.

    Before, a missing `llama-server` was named only inside the Cleanup tab —
    the one page an operator has no reason to open until they already suspect
    cleanup. It is now a Libraries tile beside FFmpeg and whisper.cpp, and both
    places quote the same install line, so the two cannot give different advice.
    """
    absent = replace(settings, cleanup_binary=Path("/nonexistent/llama-server"))
    app = create_app(absent, engine=FakeEngine(SPOKEN), normalizer=FakeNormalizer())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        tile = await _runtime_tile(client)
        assert tile["available"] is False
        assert tile["path"] is None
        assert "VOCAGATEWAY_CLEANUP_BINARY" in tile["install_hint"]

        page = _flattened(await client.get("/ui/partials/cleanup", headers=AUTH))
        assert f"No llama-server found. {tile['install_hint']}" in page

        # Where the gateway looked is the operator's business and nobody
        # else's: the configured path reaches the dashboard tile and stops
        # there, never the cleanup block a diagnostics bundle carries.
        cleanup = (await client.get(CLEANUP_CONFIG, headers=AUTH)).json()
        assert "/nonexistent" not in json.dumps(cleanup)


async def test_an_installed_runtime_is_reported_with_the_path_it_was_found_at(
    settings: Settings, tmp_path: Path
) -> None:
    """The tile answers from the manager, so it cannot disagree with the worker."""
    binary = tmp_path / "llama-server"
    binary.write_text("#!/bin/sh\n")
    binary.chmod(0o755)
    app = create_app(
        replace(settings, cleanup_binary=binary),
        engine=FakeEngine(SPOKEN),
        normalizer=FakeNormalizer(),
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        tile = await _runtime_tile(client)
        assert tile["available"] is True
        assert tile["path"] == str(binary)
        assert app.state.ctx.cleanup.host.runtime_available() is True


async def test_an_operator_run_endpoint_is_not_asked_for_a_local_runtime(
    settings: Settings,
) -> None:
    """A deployment that points at its own server does not need a binary here.

    Reporting the missing executable as a step to fix would be advice for a
    problem that deployment does not have — the gateway is not launching
    anything on this host.
    """
    external = replace(
        settings,
        cleanup_binary=Path("/nonexistent/llama-server"),
        cleanup_endpoint=("127.0.0.1", 8080),
    )
    app = create_app(external, engine=FakeEngine(SPOKEN), normalizer=FakeNormalizer())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        tile = await _runtime_tile(client)
        assert tile["available"] is False
        assert "VOCAGATEWAY_CLEANUP_ENDPOINT" in tile["install_hint"]

        page = _flattened(await client.get("/ui/partials/cleanup", headers=AUTH))
        assert "Using the cleanup server you configured" in page
        assert "No llama-server found" not in page
        # The pill has to agree with the checklist beside it: a card reading
        # "Not available" above four ticked steps helps nobody.
        assert "Not available" not in page
        assert (await client.get(CLEANUP_CONFIG, headers=AUTH)).json()["state"] == "ready"


async def _runtime_tile(client: httpx.AsyncClient) -> Any:
    payload = (await client.get("/v1/admin/status", headers=AUTH)).json()
    return next(item for item in payload["dependencies"] if item["name"] == "llama.cpp server")


def _flattened(response: Any) -> str:
    """Body with runs of whitespace collapsed, so a wrapped sentence still matches."""
    return " ".join(response.text.split())


async def test_the_mic_test_offers_a_before_and_after_with_a_legend(gateway: Any) -> None:
    client, _ = gateway
    body = (await client.get("/ui/partials/test", headers=AUTH)).text
    assert 'id="test-original-block"' in body
    assert 'class="diff-legend"' in body
    assert 'id="test-cleanup-warning"' in body


async def test_the_preview_corrects_supplied_text_without_a_recording(gateway: Any) -> None:
    """The shortest answer to "is this doing anything": type a sentence, see it fixed."""
    client, app = gateway
    enable_cleanup(app, FakeCleanupRuntime(CORRECTED))
    response = await client.post(
        "/v1/admin/cleanup/preview", json={"text": SPOKEN, "language": "en"}, headers=AUTH
    )
    assert response.status_code == HTTP_200_OK
    body = response.json()
    assert body["original"] == SPOKEN
    assert body["transcript"] == CORRECTED
    # The baseline the panel compares against: what this text becomes with the
    # feature off, so the marks credit the model only for what it added.
    assert body["without_cleanup"] == "We was going to leave early but the train was late."
    assert body["cleanup"]["status"] == "applied"


async def test_a_preview_that_cannot_run_still_answers_with_the_original(gateway: Any) -> None:
    """Same contract as a dictation: never an error the caller has to handle."""
    client, app = gateway
    enable_cleanup(app, None, installed=False)
    body = (
        await client.post(
            "/v1/admin/cleanup/preview", json={"text": SPOKEN, "language": "en"}, headers=AUTH
        )
    ).json()
    # Identical to the baseline, so the panel shows no marks and says why.
    assert body["transcript"] == body["without_cleanup"]
    assert body["cleanup"]["reason"] == "model_unavailable"


async def test_the_preview_needs_the_admin_token(gateway: Any) -> None:
    client, _ = gateway
    response = await client.post("/v1/admin/cleanup/preview", json={"text": SPOKEN})
    assert response.status_code == HTTP_401_UNAUTHORIZED


async def test_the_preview_refuses_an_empty_or_oversized_body(gateway: Any) -> None:
    client, app = gateway
    enable_cleanup(app, FakeCleanupRuntime(CORRECTED))
    for text in ("", "x" * 2_001):
        response = await client.post("/v1/admin/cleanup/preview", json={"text": text}, headers=AUTH)
        assert response.status_code == HTTP_422_UNPROCESSABLE_CONTENT


async def test_selecting_a_model_from_the_library_does_not_reset_other_settings(
    gateway: Any,
) -> None:
    """The card's button and the settings form write to the same partial update."""
    client, app = gateway
    manager = enable_cleanup(app, FakeCleanupRuntime(CORRECTED))
    await client.put(
        CLEANUP_CONFIG, json={"timeout_seconds": 8, "auto_language": "en"}, headers=AUTH
    )
    await client.put(f"/ui/partials/cleanup/select/{CLEANUP_MODEL_ID}", headers=AUTH)
    assert manager.runtime_config.cleanup_model == CLEANUP_MODEL_ID
    assert manager.runtime_config.cleanup_timeout_seconds == 8
    assert manager.runtime_config.cleanup_auto_language == "en"


async def test_saving_settings_does_not_clear_the_selected_model(gateway: Any) -> None:
    """The form no longer carries the model, so it must not blank it either."""
    client, app = gateway
    manager = enable_cleanup(app, FakeCleanupRuntime(CORRECTED))
    await client.put(
        "/ui/partials/cleanup/settings",
        data={"mode": "conservative", "timeout_seconds": "5", "idle_unload_minutes": "15"},
        headers=AUTH,
    )
    assert manager.runtime_config.cleanup_model == CLEANUP_MODEL_ID


async def test_cleanup_library_refresh_is_authenticated_and_isolated(gateway: Any) -> None:
    client, _ = gateway
    url = "/ui/partials/cleanup/library"
    assert (await client.get(url)).status_code == HTTP_401_UNAUTHORIZED
    response = await client.get(url, headers=AUTH)
    assert response.status_code == HTTP_200_OK
    assert 'id="cleanup-library"' in response.text
    assert 'hx-swap-oob="outerHTML"' in response.text
    assert 'id="cleanup-try-input"' not in response.text
    assert 'id="cleanup-settings"' not in response.text


async def test_cleanup_library_polls_only_while_downloading(
    gateway: Any, monkeypatch: MonkeyPatch
) -> None:
    from app import admin_queries

    client, app = gateway
    entries = admin_queries.cleanup_model_entries(app.state.ctx)
    monkeypatch.setattr(admin_queries, "cleanup_model_entries", lambda ctx: entries)
    entries[0].state = "downloading"
    entries[0].progress = 0.42
    response = await client.get("/ui/partials/cleanup/library", headers=AUTH)
    assert 'hx-trigger="every 2s"' in response.text
    assert 'aria-valuenow="42"' in response.text
    entries[0].state = "installed"
    response = await client.get("/ui/partials/cleanup/library", headers=AUTH)
    assert 'hx-trigger="every 2s"' not in response.text
    assert "Use this model" in response.text
    assert 'hx-select="#cleanup-page"' in response.text


async def test_preview_remains_available_when_cleanup_default_is_off(gateway: Any) -> None:
    client, app = gateway
    manager = enable_cleanup(app, FakeCleanupRuntime(CORRECTED))
    manager.runtime_config.cleanup_mode = "off"
    response = await client.get("/ui/partials/cleanup", headers=AUTH)
    assert 'id="cleanup-try-input"' in response.text
    preview = await client.post(
        "/v1/admin/cleanup/preview", json={"text": SPOKEN, "language": "en"}, headers=AUTH
    )
    assert preview.json()["cleanup"]["status"] == "applied"
    assert manager.runtime_config.cleanup_mode == "off"


async def test_header_cleanup_status_and_activity(gateway: Any) -> None:
    client, app = gateway
    assert (await client.get("/ui/partials/cleanup/pill")).status_code == HTTP_401_UNAUTHORIZED
    enable_cleanup(app, FakeCleanupRuntime(CORRECTED))
    pill = await client.get("/ui/partials/cleanup/pill", headers=AUTH)
    assert 'data-open-tab="cleanup"' in pill.text
    assert "Ready" in pill.text
    await client.post(
        "/v1/admin/cleanup/preview", json={"text": SPOKEN, "language": "en"}, headers=AUTH
    )
    response = await client.get("/ui/partials/operations", headers=AUTH)
    assert 'aria-label="Cleanup statistics"' in response.text
    assert "Cleaned up" in response.text
    status = (await client.get("/v1/admin/status", headers=AUTH)).json()
    assert status["metrics"]["cleanup_applied"] == 1
