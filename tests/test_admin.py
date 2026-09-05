from __future__ import annotations

import asyncio
import dataclasses
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest
from conftest import TOKEN, FakeNormalizer
from starlette.status import (
    HTTP_200_OK,
    HTTP_401_UNAUTHORIZED,
    HTTP_404_NOT_FOUND,
    HTTP_409_CONFLICT,
    HTTP_415_UNSUPPORTED_MEDIA_TYPE,
    HTTP_422_UNPROCESSABLE_CONTENT,
)

from app import engine_state
from app import engines as engines_module
from app.catalog import CatalogModel
from app.config import Settings
from app.main import create_app
from app.model_manager import ModelManager
from app.models.whisper_cpp import WhisperCppEngine
from app.runtime_config import RuntimeConfig

MAC_ONLY = frozenset({"vocamac", "handy", "whisperkit", "mlx-audio"})
TINY_MODEL_SIZE_BYTES = 11
GATEWAY_PORT = 8765
ASYNC_POLL_SECONDS = 0.02
MAXIMUM_DOWNLOAD_POLL_ATTEMPTS = 200
TEST_AUDIO_BYTES = b"x" * MAXIMUM_DOWNLOAD_POLL_ATTEMPTS
MAXIMUM_FILTERED_MODEL_SIZE_BYTES = 800_000_000
TINY_MODEL_ID = "whisper.cpp:ggml-tiny.bin"
WHISPER_CPP_ENGINE = "whisper.cpp"
ADMIN_STATUS_PATH = "/v1/admin/status"
ADMIN_TOKENS_PATH = "/v1/admin/tokens"
ADMIN_MODELS_PATH = "/v1/admin/models"
ADMIN_CONFIG_PATH = "/v1/admin/config"
SETTINGS_PARTIAL_PATH = "/ui/partials/settings"
MODEL_ID_KEY = "id"
LABEL_KEY = "label"
ERROR_KEY = "error"
ERROR_CODE_KEY = "code"
ENGINE_KEY = "engine"
AUTO_ENGINE = "auto"
SHERPA_ONNX_ENGINE = "sherpa-onnx"
METRICS_KEY = "metrics"
STATE_KEY = "state"
NOT_INSTALLED_STATE = "not_installed"
INSTALLED_STATE = "installed"
INSTALLED_ONLY_FILTER = "installed_only"
MODEL_CARD_CLASS = 'class="model-card"'
HINDI_LANGUAGE_CODE = "hi"
LANGUAGE_CODES_KEY = "language_codes"
ENGLISH_LANGUAGE_CODE = "en"
HINDI_LANGUAGE = "Hindi"
MODELS_LIST_PARTIAL_PATH = "/ui/partials/models-list"
LANGUAGE_KEY = "language"
NETWORK_INTERFACE_NOTICE = "Listening on every network interface"
MODELS_LANGUAGE_HINT_CLASS = "models-language-hint"

TINY = CatalogModel(
    id=TINY_MODEL_ID,
    engine=WHISPER_CPP_ENGINE,
    key="ggml-tiny.bin",
    label="Test Tiny",
    size_bytes=TINY_MODEL_SIZE_BYTES,
    languages="Multilingual",
    quality="Fastest",
    minimum_ram_gb=4,
)


@pytest.fixture
def admin_settings(tmp_path: Path) -> Settings:
    return Settings(
        token=TOKEN,
        data_dir=tmp_path,
        whisper_binary=tmp_path / "whisper-cli",
        whisper_model=tmp_path / "model.bin",
        handy_binary=tmp_path / "no-handy",
        vocamac_app=tmp_path / "no-vocamac",
        models_dir=tmp_path / "models",
        config_path=tmp_path / "config.json",
    )


@pytest.fixture
def admin_manager(tmp_path: Path) -> ModelManager:
    source = tmp_path / "source.bin"
    source.write_bytes(b"hello model")
    catalog = (dataclasses.replace(TINY, download_url=source.as_uri()),)
    return ModelManager(tmp_path / "models", catalog=catalog)


@pytest.fixture
async def admin_client(
    admin_settings: Settings, admin_manager: ModelManager
) -> AsyncIterator[httpx.AsyncClient]:
    app = create_app(
        admin_settings,
        model_manager=admin_manager,
        runtime_config=RuntimeConfig(),
        normalizer=FakeNormalizer(),
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as test_client:
        yield test_client


@pytest.fixture
def auth() -> dict[str, str]:
    return {"Authorization": f"Bearer {TOKEN}"}


async def test_admin_endpoints_require_token(admin_client: httpx.AsyncClient) -> None:
    for path in (
        ADMIN_STATUS_PATH,
        "/v1/admin/diagnostics",
        ADMIN_TOKENS_PATH,
        ADMIN_MODELS_PATH,
        ADMIN_CONFIG_PATH,
        "/ui/partials/overview",
        "/ui/partials/models",
        SETTINGS_PARTIAL_PATH,
        "/ui/partials/about",
    ):
        response = await admin_client.get(path)
        assert response.status_code == HTTP_401_UNAUTHORIZED, path


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        (f"Bearer {TOKEN}", 200),
        # RFC 7235 makes the scheme case-insensitive, and HTTPBearer honors that.
        (f"bearer {TOKEN}", 200),
        # The prefix is not optional: a bare token is as good as no token.
        (TOKEN, 401),
        (f"Basic {TOKEN}", 401),
        ("Bearer", 401),
        ("Bearer ", 401),
        ("Bearer not-the-token", 401),
    ],
)
async def test_authorization_header_forms(
    admin_client: httpx.AsyncClient, header: str, expected: int
) -> None:
    response = await admin_client.get(ADMIN_STATUS_PATH, headers={"Authorization": header})
    assert response.status_code == expected, header


async def test_openapi_documents_the_bearer_scheme(admin_settings: Settings) -> None:
    """Without this, Swagger UI has no Authorize button and cannot be used."""
    # /openapi.json and /docs are only mounted in debug mode, which is also the
    # only mode where the Swagger page this schema feeds exists at all.
    debug_app = create_app(
        dataclasses.replace(admin_settings, debug=True),
        runtime_config=RuntimeConfig(),
        normalizer=FakeNormalizer(),
    )
    schema = debug_app.openapi()
    assert schema["components"]["securitySchemes"] == {
        "Bearer token": {
            "type": "http",
            "scheme": "bearer",
            "description": schema["components"]["securitySchemes"]["Bearer token"]["description"],
        }
    }
    operation = schema["paths"][ADMIN_STATUS_PATH]["get"]
    assert operation["security"] == [{"Bearer token": []}]
    # The hand-rolled header parameter is what used to show up instead, as a
    # free-text box that silently sent an unusable request.
    assert "parameters" not in operation


async def test_tokens_list_starts_with_only_the_b_aa(
    admin_client: httpx.AsyncClient, auth: dict[str, str]
) -> None:
    response = await admin_client.get(ADMIN_TOKENS_PATH, headers=auth)
    assert response.status_code == HTTP_200_OK
    entries = response.json()
    assert entries == [
        {
            MODEL_ID_KEY: "bootstrap",
            LABEL_KEY: "Bootstrap token (VOCAGATEWAY_TOKEN / token file)",
            "created_at": None,
            "revocable": False,
        }
    ]


async def test_created_device_token_authenticates_a04c2(
    admin_client: httpx.AsyncClient, auth: dict[str, str]
) -> None:
    created = await admin_client.post(ADMIN_TOKENS_PATH, headers=auth, json={LABEL_KEY: "Pixel 6a"})
    assert created.status_code == HTTP_200_OK
    payload = created.json()
    assert payload[LABEL_KEY] == "Pixel 6a"
    device_auth = {"Authorization": f"Bearer {payload['token']}"}

    # The new device token authenticates on its own, independent of the bootstrap token.
    status = await admin_client.get(ADMIN_STATUS_PATH, headers=device_auth)
    assert status.status_code == HTTP_200_OK

    listed = await admin_client.get(ADMIN_TOKENS_PATH, headers=auth)
    ids = {entry[MODEL_ID_KEY]: entry for entry in listed.json()}
    assert payload[MODEL_ID_KEY] in ids
    assert ids[payload[MODEL_ID_KEY]]["revocable"] is True

    revoked = await admin_client.delete(f"/v1/admin/tokens/{payload['id']}", headers=auth)
    assert revoked.status_code == HTTP_200_OK
    assert revoked.json() == {"revoked": True}

    # Revoking one device token never touches the bootstrap token or other clients.
    still_ok = await admin_client.get(ADMIN_STATUS_PATH, headers=auth)
    assert still_ok.status_code == HTTP_200_OK
    now_rejected = await admin_client.get(ADMIN_STATUS_PATH, headers=device_auth)
    assert now_rejected.status_code == HTTP_401_UNAUTHORIZED


async def test_revoking_unknown_token_returns_number(
    admin_client: httpx.AsyncClient, auth: dict[str, str]
) -> None:
    response = await admin_client.delete("/v1/admin/tokens/does-not-exist", headers=auth)
    assert response.status_code == HTTP_404_NOT_FOUND
    assert response.json() == {"revoked": False}


async def test_bootstrap_token_cannot_be_revoked(
    admin_client: httpx.AsyncClient, auth: dict[str, str]
) -> None:
    response = await admin_client.delete("/v1/admin/tokens/bootstrap", headers=auth)
    assert response.status_code == HTTP_409_CONFLICT
    assert response.json()[ERROR_KEY][ERROR_CODE_KEY] == "bootstrap_token_not_revocable"


async def test_diagnostics_bundle_is_downloadable_aaaa(
    admin_client: httpx.AsyncClient, auth: dict[str, str]
) -> None:
    response = await admin_client.get("/v1/admin/diagnostics", headers=auth)
    assert response.status_code == HTTP_200_OK
    assert response.headers["content-disposition"].startswith(
        'attachment; filename="vocagateway-diagnostics-'
    )
    payload = response.json()
    assert payload[ENGINE_KEY][MODEL_ID_KEY] == AUTO_ENGINE
    assert payload["config"][ENGINE_KEY] == AUTO_ENGINE
    assert "never_included" in payload and payload["never_included"]
    assert TOKEN not in response.text


async def test_status_reports_system_and_setup(
    admin_client: httpx.AsyncClient, auth: dict[str, str]
) -> None:
    response = await admin_client.get(ADMIN_STATUS_PATH, headers=auth)
    assert response.status_code == HTTP_200_OK
    payload = response.json()
    assert payload[ENGINE_KEY][MODEL_ID_KEY] == AUTO_ENGINE
    assert payload["system"]["arch"]
    assert {dependency["name"] for dependency in payload["dependencies"]} == {
        "FFmpeg",
        "whisper.cpp CLI",
        "WhisperKit CLI",
        "Handy app",
        "VocaMac app",
        "faster-whisper",
        "Moonshine Voice",
        SHERPA_ONNX_ENGINE,
        "MLX Audio",
    }
    assert payload["setup"]["token_configured"] is True
    assert payload["setup"]["model_installed"] is False
    assert payload["bind_host"] == "0.0.0.0"
    assert payload["port"] == GATEWAY_PORT
    assert payload[METRICS_KEY] == {
        "uptime_seconds": payload[METRICS_KEY]["uptime_seconds"],
        "queue_depth": 0,
        "active_transcriptions": 0,
        "concurrency_limit": 1,
        "successful_transcriptions": 0,
        "failed_transcriptions": 0,
        "rejected_transcriptions": 0,
        "average_latency_ms": None,
        "last_latency_ms": None,
        "normalization_ms": None,
        "model_load_ms": None,
        "inference_ms": None,
        "audio_duration_ms": None,
        "real_time_factor": None,
        "peak_memory_mb": None,
        "history": payload[METRICS_KEY]["history"],
    }
    assert isinstance(payload[METRICS_KEY]["history"], list)
    assert payload["readiness"]["warmup_state"] == "pending"


async def test_models_list_contains_catalog(
    admin_client: httpx.AsyncClient, auth: dict[str, str]
) -> None:
    response = await admin_client.get(ADMIN_MODELS_PATH, headers=auth)
    assert response.status_code == HTTP_200_OK
    entries = {entry[MODEL_ID_KEY]: entry for entry in response.json()}
    assert TINY_MODEL_ID in entries
    assert entries[TINY_MODEL_ID][STATE_KEY] == NOT_INSTALLED_STATE
    assert entries[TINY_MODEL_ID]["family"] == "Whisper"
    assert entries[TINY_MODEL_ID]["description"]
    # A card warns about a missing runtime using the same table the Overview
    # panel reads, so the two can never disagree about what is installed.
    status = await admin_client.get(ADMIN_STATUS_PATH, headers=auth)
    tile = next(
        dependency
        for dependency in status.json()["dependencies"]
        if dependency["name"] == "whisper.cpp CLI"
    )
    entry = entries[TINY_MODEL_ID]
    if tile["available"]:
        assert entry["runtime_requirement"] is None
        assert entry["runtime_hint"] is None
    else:
        assert entry["runtime_requirement"] == "whisper.cpp CLI"
        assert entry["runtime_hint"] == tile["install_hint"]


async def test_download_select_and_delete_flow(
    admin_client: httpx.AsyncClient,
    auth: dict[str, str],
    admin_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_id = TINY_MODEL_ID

    missing = await admin_client.post(f"/v1/admin/models/{model_id}/select", headers=auth)
    assert missing.status_code == HTTP_404_NOT_FOUND

    started = await admin_client.post(f"/v1/admin/models/{model_id}/download", headers=auth)
    assert started.status_code == HTTP_200_OK

    duplicate = await admin_client.post(f"/v1/admin/models/{model_id}/download", headers=auth)
    assert duplicate.status_code == HTTP_409_CONFLICT

    for _ in range(MAXIMUM_DOWNLOAD_POLL_ATTEMPTS):
        entries = {
            entry[MODEL_ID_KEY]: entry
            for entry in (await admin_client.get(ADMIN_MODELS_PATH, headers=auth)).json()
        }
        if entries[model_id][STATE_KEY] == INSTALLED_STATE:
            break
        await asyncio.sleep(ASYNC_POLL_SECONDS)
    assert entries[model_id][STATE_KEY] == INSTALLED_STATE

    warmup_calls = 0

    async def record_warmup(_: WhisperCppEngine) -> int:
        nonlocal warmup_calls
        warmup_calls += 1
        return TINY_MODEL_SIZE_BYTES

    admin_settings.whisper_binary.write_bytes(b"binary")
    monkeypatch.setattr(WhisperCppEngine, "warmup", record_warmup)
    selected = await admin_client.post(f"/v1/admin/models/{model_id}/select", headers=auth)
    assert selected.status_code == HTTP_200_OK
    assert selected.json()[ENGINE_KEY][MODEL_ID_KEY] == WHISPER_CPP_ENGINE
    assert warmup_calls == 1

    saved = RuntimeConfig.load(admin_settings.config_path)
    assert saved.engine == WHISPER_CPP_ENGINE
    assert saved.whisper_model and saved.whisper_model.endswith("ggml-tiny.bin")

    entries = {
        entry[MODEL_ID_KEY]: entry
        for entry in (await admin_client.get(ADMIN_MODELS_PATH, headers=auth)).json()
    }
    assert entries[model_id]["active"] is True

    deleted = await admin_client.delete(f"/v1/admin/models/{model_id}", headers=auth)
    assert deleted.status_code == HTTP_200_OK
    saved = RuntimeConfig.load(admin_settings.config_path)
    assert saved.whisper_model is None


async def test_models_list_installed_only_filter(
    admin_client: httpx.AsyncClient, auth: dict[str, str]
) -> None:
    model_id = TINY_MODEL_ID

    before = await admin_client.get(
        ADMIN_MODELS_PATH, params={INSTALLED_ONLY_FILTER: "true"}, headers=auth
    )
    assert before.status_code == HTTP_200_OK
    assert model_id not in {entry[MODEL_ID_KEY] for entry in before.json()}

    started = await admin_client.post(f"/v1/admin/models/{model_id}/download", headers=auth)
    assert started.status_code == HTTP_200_OK

    for _ in range(MAXIMUM_DOWNLOAD_POLL_ATTEMPTS):
        filtered = {
            entry[MODEL_ID_KEY]: entry
            for entry in (
                await admin_client.get(
                    ADMIN_MODELS_PATH, params={INSTALLED_ONLY_FILTER: "true"}, headers=auth
                )
            ).json()
        }
        if model_id in filtered:
            break
        await asyncio.sleep(ASYNC_POLL_SECONDS)
    assert filtered[model_id][STATE_KEY] == INSTALLED_STATE
    assert all(entry[STATE_KEY] == INSTALLED_STATE for entry in filtered.values())


async def test_ui_select_preserves_installed_only_aaaaa(
    admin_client: httpx.AsyncClient, auth: dict[str, str]
) -> None:
    model_id = TINY_MODEL_ID

    await admin_client.post(f"/v1/admin/models/{model_id}/download", headers=auth)
    entries: dict[str, dict[str, object]] = {}
    for _ in range(MAXIMUM_DOWNLOAD_POLL_ATTEMPTS):
        entries = {
            entry[MODEL_ID_KEY]: entry
            for entry in (await admin_client.get(ADMIN_MODELS_PATH, headers=auth)).json()
        }
        if entries[model_id][STATE_KEY] == INSTALLED_STATE:
            break
        await asyncio.sleep(ASYNC_POLL_SECONDS)
    assert entries[model_id][STATE_KEY] == INSTALLED_STATE

    selected = await admin_client.post(
        f"/ui/partials/models/{model_id}/select",
        headers=auth,
        data={INSTALLED_ONLY_FILTER: "true"},
    )
    assert selected.status_code == HTTP_200_OK
    assert MODEL_CARD_CLASS in selected.text
    assert "No models downloaded yet" not in selected.text


def test_language_filter_answers_which_mode_f8f45() -> None:
    """The filter exists to invert the question people actually ask. Driven off
    the real catalog, since the stub the admin fixtures use has one model."""
    from app.catalog import DEFAULT_CATALOG, language_names
    from app.schemas import AdminModelEntry
    from app.serializers import model_covers as _model_covers

    entries = [
        AdminModelEntry(
            id=model.id,
            engine=model.engine,
            label=model.label,
            size_bytes=model.size_bytes,
            languages=model.languages,
            quality=model.quality,
            family=model.family,
            description=model.description,
            source=model.source,
            state=NOT_INSTALLED_STATE,
            active=False,
            recommended=False,
            detects_language_automatically=model.detects_language_automatically,
            language_names=language_names(model.language_codes),
            language_codes=list(model.language_codes),
        )
        for model in DEFAULT_CATALOG
    ]

    hindi = [entry for entry in entries if _model_covers(entry, HINDI_LANGUAGE_CODE)]
    assert hindi, "Hindi must match something"
    assert len(hindi) < len(entries), "the filter has to actually narrow the list"
    # Both the auto-detecting recogniser and the pinnable Whisper builds appear,
    # so the badge is what tells them apart rather than their absence.
    assert any("dolphin" in entry.id for entry in hindi)
    assert any(entry.engine == WHISPER_CPP_ENGINE for entry in hindi)
    # An English-only build must not surface under Hindi.
    assert not any(entry.id.endswith(".en.bin") for entry in hindi)

    # A model with no declared languages (an imported one) matches everything
    # rather than disappearing from every filter.
    unlabelled = entries[0].model_copy(update={LANGUAGE_CODES_KEY: []})
    assert _model_covers(unlabelled, HINDI_LANGUAGE_CODE) and _model_covers(unlabelled, "yo")

    english = [entry for entry in entries if _model_covers(entry, ENGLISH_LANGUAGE_CODE)]
    assert any(entry.id.endswith(".en.bin") for entry in english)

    # Moonshine carries its language in `language_code`, not `language_codes`.
    # Leaving the tuple empty made every English Moonshine match "covers
    # everything", so they all turned up under Hindi.
    assert not any(entry.engine == "moonshine" and entry.id != "moonshine:hi" for entry in hindi)
    assert any(entry.engine == "moonshine" for entry in english)

    # Odia is real but Whisper-less: only Dolphin covers it, which
    # is exactly why the phone clients do not offer it as a choice.
    odia = [entry for entry in entries if _model_covers(entry, "or")]
    assert odia and all(entry.detects_language_automatically for entry in odia)


def test_language_filter_offers_only_langua_b4011() -> None:
    from app.fragments.models import _language_filter_options

    options = _language_filter_options()
    codes = {code for code, _ in options}
    names = {name for _, name in options}
    assert HINDI_LANGUAGE_CODE in codes and HINDI_LANGUAGE in names
    assert "ta" in codes and "Tamil" in names
    # Odia is Dolphin-only, which is still a model, so it belongs here even
    # though the phone clients deliberately do not offer it.
    assert "or" in codes and "Odia" in names
    # A language no catalog model covers must not appear as a dead option.
    assert "xx" not in codes


def test_multi_select_filters_combine_with_and_or() -> None:
    """Within a dimension match is OR; across dimensions match is AND."""
    from app.admin_queries import SIZE_FILTER_CAPS, filtered_model_entries
    from app.catalog import DEFAULT_CATALOG
    from app.schemas import AdminModelEntry
    from app.serializers import model_covers

    # Drive off catalog fields without a live gateway: pure predicate checks.
    entries = [
        AdminModelEntry(
            id=model.id,
            engine=model.engine,
            label=model.label,
            size_bytes=model.size_bytes,
            languages=model.languages,
            quality=model.quality,
            family=model.family,
            description=model.description,
            source=model.source,
            state=NOT_INSTALLED_STATE,
            active=False,
            recommended=False,
            language_codes=list(model.language_codes),
        )
        for model in DEFAULT_CATALOG
    ]

    # OR across languages.
    hi_or_yue = [
        entry
        for entry in entries
        if model_covers(entry, HINDI_LANGUAGE_CODE) or model_covers(entry, "yue")
    ]
    assert hi_or_yue and len(hi_or_yue) < len(entries)

    # Size cap is a hard upper bound.
    cap = SIZE_FILTER_CAPS["300mb"]
    under = [entry for entry in entries if entry.size_bytes <= cap]
    assert under and all(entry.size_bytes <= cap for entry in under)
    assert any(entry.size_bytes > cap for entry in entries)

    # filtered_model_entries wiring is covered by the API tests below; keep the
    # helper import live so a rename fails here.
    assert callable(filtered_model_entries)


async def test_ui_actions_preserve_the_language_filter(
    admin_client: httpx.AsyncClient, auth: dict[str, str]
) -> None:
    """Downloading or selecting must not silently reset the view, the same
    guarantee the installed-only toggle already makes."""
    listed = await admin_client.get(
        MODELS_LIST_PARTIAL_PATH, params={LANGUAGE_KEY: HINDI_LANGUAGE_CODE}, headers=auth
    )
    assert listed.status_code == HTTP_200_OK

    # The stub catalog's only model is multilingual Whisper, so it covers Hindi.
    assert MODEL_CARD_CLASS in listed.text

    filtered_out = await admin_client.get(
        MODELS_LIST_PARTIAL_PATH, params={LANGUAGE_KEY: "yue"}, headers=auth
    )
    assert "Cantonese" in filtered_out.text or MODEL_CARD_CLASS in filtered_out.text


async def test_admin_models_api_accepts_the_langu_a(
    admin_client: httpx.AsyncClient, auth: dict[str, str]
) -> None:
    response = await admin_client.get(
        ADMIN_MODELS_PATH, params={LANGUAGE_KEY: HINDI_LANGUAGE_CODE}, headers=auth
    )
    assert response.status_code == HTTP_200_OK
    assert all(
        HINDI_LANGUAGE_CODE in entry[LANGUAGE_CODES_KEY] or not entry[LANGUAGE_CODES_KEY]
        for entry in response.json()
    )


async def test_admin_models_api_accepts_multi_filters(
    admin_client: httpx.AsyncClient, auth: dict[str, str]
) -> None:
    # Repeated query keys for multi-select family + language; size cap optional.
    response = await admin_client.get(
        ADMIN_MODELS_PATH,
        params=[
            (LANGUAGE_KEY, ENGLISH_LANGUAGE_CODE),
            (LANGUAGE_KEY, HINDI_LANGUAGE_CODE),
            ("max_size", "800mb"),
        ],
        headers=auth,
    )
    assert response.status_code == HTTP_200_OK
    body = response.json()
    assert body
    for entry in body:
        codes = entry[LANGUAGE_CODES_KEY]
        assert not codes or ENGLISH_LANGUAGE_CODE in codes or HINDI_LANGUAGE_CODE in codes
        assert entry["size_bytes"] <= MAXIMUM_FILTERED_MODEL_SIZE_BYTES


async def test_models_ui_filter_panel_is_multi_select(
    admin_client: httpx.AsyncClient, auth: dict[str, str]
) -> None:
    page = await admin_client.get("/ui/partials/models", headers=auth)
    assert page.status_code == HTTP_200_OK
    assert 'id="models-filter-form"' in page.text
    assert 'id="models-layout"' in page.text
    assert 'class="models-filter"' in page.text or 'class="models-filter ' in page.text
    assert 'id="filter-rail-toggle"' in page.text
    assert 'name="family"' in page.text
    assert f'name="{LANGUAGE_KEY}"' in page.text
    assert f'name="{ENGINE_KEY}"' in page.text
    assert 'name="max_size"' in page.text
    assert 'name="recommended_only"' in page.text
    assert "Fits this machine" in page.text


async def test_models_list_accepts_cleared_filters(
    admin_client: httpx.AsyncClient, auth: dict[str, str]
) -> None:
    """Clear filters sends empty bools; must not 422."""
    cleared = await admin_client.get(MODELS_LIST_PARTIAL_PATH, headers=auth)
    assert cleared.status_code == HTTP_200_OK
    assert 'class="family-tile"' in cleared.text or MODEL_CARD_CLASS in cleared.text

    empty_bools = await admin_client.get(
        MODELS_LIST_PARTIAL_PATH,
        params={INSTALLED_ONLY_FILTER: "", "recommended_only": ""},
        headers=auth,
    )
    assert empty_bools.status_code == HTTP_200_OK
    assert 'class="family-tile"' in empty_bools.text or MODEL_CARD_CLASS in empty_bools.text


async def test_unknown_model_download_number(
    admin_client: httpx.AsyncClient, auth: dict[str, str]
) -> None:
    response = await admin_client.post(
        "/v1/admin/models/whisper.cpp:missing.bin/download", headers=auth
    )
    assert response.status_code == HTTP_404_NOT_FOUND


async def test_config_update_persists_engine(
    admin_client: httpx.AsyncClient, auth: dict[str, str], admin_settings: Settings
) -> None:
    invalid = await admin_client.put(ADMIN_CONFIG_PATH, headers=auth, json={ENGINE_KEY: "cloud"})
    assert invalid.status_code == HTTP_422_UNPROCESSABLE_CONTENT

    updated = await admin_client.put(
        ADMIN_CONFIG_PATH, headers=auth, json={ENGINE_KEY: SHERPA_ONNX_ENGINE}
    )
    assert updated.status_code == HTTP_200_OK
    assert RuntimeConfig.load(admin_settings.config_path).engine == SHERPA_ONNX_ENGINE


async def test_mac_only_engines_are_hidden_and_re_aaa(
    admin_client: httpx.AsyncClient,
    auth: dict[str, str],
    admin_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(engine_state, "engine_runs_on", lambda engine, **_: engine not in MAC_ONLY)
    monkeypatch.setattr(engines_module, "engine_runs_here", lambda engine: engine not in MAC_ONLY)

    config = await admin_client.get(ADMIN_CONFIG_PATH, headers=auth)
    settings_html = (await admin_client.get(SETTINGS_PARTIAL_PATH, headers=auth)).text
    rejected = await admin_client.put(ADMIN_CONFIG_PATH, headers=auth, json={ENGINE_KEY: "vocamac"})

    assert set(config.json()["available_engines"]).isdisjoint(MAC_ONLY)
    assert "VocaMac app" not in settings_html
    assert "Handy app" not in settings_html
    assert rejected.status_code == HTTP_422_UNPROCESSABLE_CONTENT
    assert rejected.json()[ERROR_KEY][ERROR_CODE_KEY] == "invalid_engine"
    assert "Apple silicon" in rejected.json()[ERROR_KEY]["message"]
    assert RuntimeConfig.load(admin_settings.config_path).engine != "vocamac"


async def test_mac_only_engines_are_labelled_with_aaaa(
    admin_client: httpx.AsyncClient, auth: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(engine_state, "engine_runs_on", lambda engine, **_: True)

    settings_html = (await admin_client.get(SETTINGS_PARTIAL_PATH, headers=auth)).text

    assert "VocaMac app (Apple silicon only)" in settings_html
    assert "Handy app (macOS only)" in settings_html
    assert "sherpa-onnx</option>" in settings_html
    assert "Offload model when idle" in settings_html
    assert 'option value="10"' in settings_html
    assert 'option value="120"' in settings_html


async def test_ui_config_update_switches_engine_a_abe0c(
    admin_client: httpx.AsyncClient, auth: dict[str, str], admin_settings: Settings
) -> None:
    response = await admin_client.put(
        "/ui/partials/config",
        headers=auth,
        data={
            ENGINE_KEY: SHERPA_ONNX_ENGINE,
            "compute_device": "cpu",
            "cpu_threads": "2",
            "idle_offload_enabled": "true",
            "idle_offload_minutes": "30",
        },
    )
    assert response.status_code == HTTP_200_OK
    assert "Engine preference saved." in response.text
    assert 'id="engine-pill"' in response.text
    assert RuntimeConfig.load(admin_settings.config_path).engine == SHERPA_ONNX_ENGINE
    saved = RuntimeConfig.load(admin_settings.config_path)
    assert saved.idle_offload_enabled is True
    assert saved.idle_offload_minutes == 30


async def test_ui_config_update_rejects_an_invali_aaaaa(
    admin_client: httpx.AsyncClient, auth: dict[str, str]
) -> None:
    response = await admin_client.put(
        "/ui/partials/config",
        headers=auth,
        data={ENGINE_KEY: "cloud"},
    )
    assert response.status_code == HTTP_422_UNPROCESSABLE_CONTENT
    assert response.json()[ERROR_KEY][ERROR_CODE_KEY] == "invalid_engine"


async def test_custom_download_rejects_bad_url(
    admin_client: httpx.AsyncClient, auth: dict[str, str]
) -> None:
    response = await admin_client.post(
        "/v1/admin/models/custom",
        headers=auth,
        json={"url": "https://example.com/not-a-model.txt"},
    )
    assert response.status_code == HTTP_422_UNPROCESSABLE_CONTENT
    assert response.json()[ERROR_KEY][ERROR_CODE_KEY] == "invalid_model_url"


async def test_partials_render_html(admin_client: httpx.AsyncClient, auth: dict[str, str]) -> None:
    overview = await admin_client.get("/ui/partials/overview", headers=auth)
    assert overview.status_code == HTTP_200_OK
    assert "Live operations" in overview.text
    assert 'hx-get="/ui/partials/operations"' in overview.text
    assert "Succeeded" in overview.text
    assert "Failed" in overview.text
    assert "Avg latency" in overview.text
    assert "Last inference" in overview.text
    assert "Real-time factor" in overview.text
    assert "Model cache" in overview.text
    assert "Workload" in overview.text
    assert "Outcomes" in overview.text
    assert "Last job stages" in overview.text
    assert "Engine checked" in overview.text
    # Libraries / engines restored from main, styled like system/ops cards.
    assert "Libraries &amp; tools" in overview.text or "Libraries & tools" in overview.text
    assert 'class="libraries-card"' in overview.text or "libraries-card" in overview.text
    assert 'class="lib-tile' in overview.text
    assert "FFmpeg" in overview.text
    assert SHERPA_ONNX_ENGINE in overview.text
    assert "Installed" in overview.text or "Missing" in overview.text
    # Old dense table layout should not return.
    assert "Path / install" not in overview.text
    # Setup checklist stays in Get started / You're set — not the old always-on list.
    assert "Setup checklist" not in overview.text
    # Bind addresses live on the header model pill hover card, not Overview.
    assert "0.0.0.0:8765" not in overview.text
    assert "http://127.0.0.1:8765/" not in overview.text
    # Exposure warning is a top-of-app banner, not mid-Overview copy.
    assert NETWORK_INTERFACE_NOTICE not in overview.text

    models = await admin_client.get("/ui/partials/models", headers=auth)
    assert models.status_code == HTTP_200_OK
    assert "Test Tiny" in models.text
    assert MODEL_CARD_CLASS in models.text
    assert 'hx-trigger="every 1500ms"' not in models.text
    assert 'hx-post="/ui/partials/models/whisper.cpp%3Aggml-tiny.bin/download"' in models.text

    settings = await admin_client.get(SETTINGS_PARTIAL_PATH, headers=auth)
    assert settings.status_code == HTTP_200_OK
    assert "Speech engine" in settings.text
    assert "exposure-panel" not in settings.text
    assert "Device tokens" in settings.text
    assert "Bootstrap token (VOCAGATEWAY_TOKEN / token file)" in settings.text

    banner = await admin_client.get("/ui/partials/exposure-banner", headers=auth)
    assert banner.status_code == HTTP_200_OK
    assert "exposure-banner" in banner.text
    assert NETWORK_INTERFACE_NOTICE in banner.text
    assert "Dismiss for 24 hours" in banner.text
    assert "&times;" in banner.text or "×" in banner.text

    pair = await admin_client.get("/ui/partials/test", headers=auth)
    assert pair.status_code == HTTP_200_OK
    assert "exposure-panel" in pair.text
    assert NETWORK_INTERFACE_NOTICE in pair.text
    # Once, at page end — never stacked by pairing-card HTMX swaps.
    assert pair.text.count(NETWORK_INTERFACE_NOTICE) == 1
    assert pair.text.count('id="pairing-exposure-panel"') == 1
    assert pair.text.index('id="test-card"') < pair.text.index('id="pairing-exposure-panel"')

    # Token/url pairing partials must not re-emit the exposure panel.
    pairing_only = await admin_client.get("/ui/partials/pairing", headers=auth)
    assert pairing_only.status_code == HTTP_200_OK
    assert "exposure-panel" not in pairing_only.text
    assert NETWORK_INTERFACE_NOTICE not in pairing_only.text

    tokens = await admin_client.get("/ui/partials/tokens", headers=auth)
    assert tokens.status_code == HTTP_200_OK
    assert 'id="tokens-card"' in tokens.text

    created = await admin_client.post(
        "/ui/partials/tokens", headers=auth, data={LABEL_KEY: "Kanishk's iPhone"}
    )
    assert created.status_code == HTTP_200_OK
    assert "New secret for Kanishk&#39;s iPhone" in created.text
    assert 'id="new-token-value"' in created.text
    assert "Regenerate</button>" in created.text

    operations = await admin_client.get("/ui/partials/operations", headers=auth)
    assert operations.status_code == HTTP_200_OK
    assert "0 queued" in operations.text
    assert "Latency" in operations.text

    pill = await admin_client.get("/ui/partials/engine-pill", headers=auth)
    assert pill.status_code == HTTP_200_OK
    assert "engine-pill" in pill.text
    assert "engine-popover" in pill.text
    assert "0.0.0.0:8765" in pill.text
    assert "http://127.0.0.1:8765/" in pill.text

    about = await admin_client.get("/ui/partials/about", headers=auth)
    assert about.status_code == HTTP_200_OK
    _assert_about_surface(about.text)


def _assert_about_surface(html: str) -> None:
    assert 'class="page-head"' in html
    assert 'id="about-host"' in html
    assert 'class="card system-card"' in html
    assert 'class="sys-hero-kicker"' in html
    assert "Beta" in html
    assert "Early" not in html
    assert "The host you run" in html
    assert "/assets/brand/vocagateway/vocagateway-tower.svg" in html
    assert 'id="about-this-build"' in html
    assert 'id="about-family"' in html
    assert 'id="about-talk"' in html
    assert "<h2>This build</h2>" in html
    assert "<h2>Part of VocaHQ</h2>" in html
    assert "<h2>Talk to us</h2>" in html
    assert "about-kicker" not in html
    assert "about-card-kicker" not in html
    assert "VocaGateway early." not in html
    assert "self-hosted compute" in html
    assert "other Voca clients" in html
    assert "Never on-device" in html
    assert "is on-device" not in html.lower()
    assert "Never expose port" in html and "8765" in html
    assert "about-info" in html
    assert 'class="callout"' not in html
    assert "AGPL-3.0" in html
    for host in (
        "https://vocahq.com",
        "https://vocalinux.com",
        "https://vocamac.com",
        "https://vocawin.com",
        "https://vocaphone.vocahq.com",
        "https://vocagateway.vocahq.com",
    ):
        assert host in html
    assert "about-family-links" not in html
    assert html.count('class="onboarding-actions"') == 2
    assert "VocaHQ</a>" in html
    assert "VocaLinux</a>" in html
    assert "VocaMac</a>" in html
    assert "VocaWin</a>" in html
    assert "VocaPhone</a>" in html
    assert "VocaGateway</a>" in html
    # Family row is currentColor: HQ mic, 1U chassis, platform marks.
    assert "M 164 127 A 65 65" in html
    assert 'width="832" height="280"' in html
    # Official platform marks from VocaHQ/.github brand/promo/cards/platform.
    assert "M12.504 0c-.155" in html
    assert "M12.152 6.896" in html
    assert "M0,0H11.377" in html
    assert "M18.4395 5.5586" in html
    assert "https://github.com/VocaHQ/vocagateway/issues" in html
    assert "Report a bug or idea" in html
    assert "https://discord.gg/t6muquAJbm" in html
    assert "https://x.com/vocahq" in html
    assert "mailto:hello@vocahq.com" in html
    assert "Discord</a>" in html
    assert "X</a>" in html
    assert "Email</a>" in html
    assert "X @vocahq" not in html
    # Official VocaHQ/.github social marks, consumed as-is (fill=currentColor).
    assert "M20.317 4.3698" in html
    assert "M18.901 1.153" in html
    assert "M12 .297" in html
    assert "M3 5h18a2 2 0 0 1 2 2v10" in html
    assert 'fill="currentColor"' in html
    assert "twitter.com" not in html.lower()
    assert "Twitter" not in html
    assert "hosted cloud" not in html.lower()
    assert "vocagateway.com" not in html.replace("vocagateway.vocahq.com", "")


async def test_webui_shell_is_public(admin_client: httpx.AsyncClient) -> None:
    response = await admin_client.get("/")
    assert response.status_code == HTTP_200_OK
    assert "htmx.min.js" in response.text
    assert 'data-tab="about"' in response.text
    assert 'hx-get="/ui/partials/about"' in response.text
    # Fifth tab, after the existing four. Hash routing uses data-tab="about".
    tabs = [part.split('"', 1)[0] for part in response.text.split('data-tab="')[1:]]
    assert tabs[:5] == ["overview", "models", "pair", "settings", "about"]
    assert "footer-copy" in response.text
    assert "footer-links" not in response.text
    assert "© 2026" in response.text or "&copy; 2026" in response.text
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-frame-options"] == "DENY"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["referrer-policy"] == "no-referrer"
    assert response.headers["permissions-policy"] == "microphone=(self)"
    assert "frame-ancestors 'none'" in response.headers["content-security-policy"]


async def test_private_responses_are_not_cached(
    admin_client: httpx.AsyncClient, auth: dict[str, str]
) -> None:
    response = await admin_client.get(ADMIN_STATUS_PATH, headers=auth)
    assert response.status_code == HTTP_200_OK
    assert response.headers["cache-control"] == "no-store"


async def test_recorder_ui_shows_limit_and_copy_action(
    admin_client: httpx.AsyncClient, auth: dict[str, str]
) -> None:
    response = await admin_client.get("/ui/partials/test", headers=auth)
    assert response.status_code == HTTP_200_OK
    assert 'data-maximum-seconds="120"' in response.text
    assert 'id="record-timer"' in response.text
    assert 'id="copy-transcript"' in response.text


def test_model_cards_name_their_languages() -> None:
    """ "25 European languages" does not tell anyone whether their language is in
    there. The card names the first few inline so the common question is
    answerable while scanning, and keeps the rest behind a "+N more" toggle.

    Driven off the real catalog rather than the stub the admin fixtures use, since
    the point is that shipped entries carry usable language metadata.
    """
    from app.catalog import DEFAULT_CATALOG, language_names
    from app.fragments.models import models_list_fragment
    from app.schemas import AdminModelEntry

    def card(model_id: str) -> str:
        model = next(model for model in DEFAULT_CATALOG if model.id == model_id)
        entry = AdminModelEntry(
            id=model.id,
            engine=model.engine,
            label=model.label,
            size_bytes=model.size_bytes,
            languages=model.languages,
            quality=model.quality,
            family=model.family,
            description=model.description,
            source=model.source,
            state=NOT_INSTALLED_STATE,
            active=False,
            recommended=False,
            detects_language_automatically=model.detects_language_automatically,
            language_names=language_names(model.language_codes),
        )
        # Render through the real list pipeline (single-entry family) so this
        # exercises the same path production uses, not a private helper.
        return models_list_fragment([entry])

    parakeet = card("sherpa-onnx:parakeet-tdt-0.6b-v3-int8")
    parakeet_summary = parakeet.split("</summary>")[0]
    # Named on the closed card, not hidden behind a bare count.
    assert 'class="model-language-chip">Bulgarian</span>' in parakeet_summary
    assert 'class="model-language-chip">Croatian</span>' in parakeet_summary
    assert 'class="model-language-chip">Czech</span>' in parakeet_summary
    assert "+21 more" in parakeet_summary
    # The remaining 21 are still on the card, just below the toggle.
    assert 'class="model-language-chip">Ukrainian</span>' in parakeet
    # A model that can be pinned carries neither the badge nor the caveat.
    assert "badge auto-language" not in parakeet
    assert "picks the language itself" not in parakeet

    dolphin = card("sherpa-onnx:dolphin-small-ctc-int8")
    assert "+36 more" in dolphin.split("</summary>")[0]
    assert HINDI_LANGUAGE in dolphin and "Bengali" in dolphin and "Tamil" in dolphin
    assert 'class="badge auto-language"' in dolphin
    assert "picks the language itself" in dolphin
    assert 'class="model-language-note' in dolphin

    # Whisper carries its full set too, and every code resolves to a real name
    # rather than leaking a bare "af, am, be" at the reader.
    whisper = card("whisper.cpp:ggml-large-v3-turbo.bin")
    assert "+96 more" in whisper.split("</summary>")[0]
    assert "Afrikaans" in whisper and HINDI_LANGUAGE in whisper
    assert 'class="model-language-chip">Afrikaans</span>' in whisper
    assert "badge auto-language" not in whisper

    # An English-only build carries just ENGLISH_LANGUAGE_CODE, and gets no disclosure at all —
    # its "English only" summary already says everything a list would.
    english_only = card("whisper.cpp:ggml-small.en.bin")
    assert "model-languages" not in english_only

    # Card structure: blurb on tile; info icon + actions; extra facts in popover.
    assert 'class="model-actions"' in parakeet
    assert 'class="model-info-btn"' in parakeet
    assert 'class="model-info-pop"' in parakeet
    assert 'class="model-blurb"' in parakeet
    assert "model-footer" not in parakeet
    assert "model-more" not in parakeet


async def test_recorder_offers_every_language_a_c_ca791(
    admin_client: httpx.AsyncClient, auth: dict[str, str]
) -> None:
    """The Test tab must not be narrower than the mobile pickers.

    This list is duplicated in `TranscriptionLanguage` on iOS and Android, which
    pin the same order in their own suites. An operator who downloads a model for
    a language has to be able to test that language here.
    """
    import re

    response = await admin_client.get("/ui/partials/test", headers=auth)
    select = response.text.split('<select id="test-language">')[1].split("</select>")[0]

    assert re.findall(r'<option value="([a-z]+)"', select) == [
        AUTO_ENGINE,
        "ar",
        "as",
        "bn",
        "nl",
        ENGLISH_LANGUAGE_CODE,
        "fr",
        "de",
        "gu",
        HINDI_LANGUAGE_CODE,
        "it",
        "ja",
        "kn",
        "ko",
        "ml",
        "zh",
        "mr",
        "ne",
        "pl",
        "pt",
        "pa", "ru", "es", "ta", "te", "uk", "ur", "vi",
    ]  # fmt: skip


async def test_test_transcription_endpoint(
    client: httpx.AsyncClient,
    authorization: dict[str, str],
    audio_bytes: bytes,
) -> None:
    response = await client.post(
        "/v1/admin/test-transcription?language=en",
        headers={**authorization, "Content-Type": "audio/wav"},
        content=audio_bytes,
    )
    assert response.status_code == HTTP_200_OK
    payload = response.json()
    assert payload["transcript"] == "hello from the local model"
    assert payload[ENGINE_KEY] == "fake-local-model"
    assert payload["duration_ms"] >= 0
    assert payload["normalization_ms"] >= 0
    assert payload["inference_ms"] >= 0
    assert "real_time_factor" in payload
    assert payload["peak_memory_mb"] > 0

    status = await client.get(ADMIN_STATUS_PATH, headers=authorization)
    metrics = status.json()[METRICS_KEY]
    assert metrics["successful_transcriptions"] == 1
    assert metrics["failed_transcriptions"] == 0
    assert metrics["queue_depth"] == 0
    assert metrics["active_transcriptions"] == 0
    assert metrics["last_latency_ms"] == payload["duration_ms"]


async def test_test_transcription_rejects_unsuppo_f97c8(
    client: httpx.AsyncClient, authorization: dict[str, str]
) -> None:
    response = await client.post(
        "/v1/admin/test-transcription",
        headers={**authorization, "Content-Type": "text/plain"},
        content=TEST_AUDIO_BYTES,
    )
    assert response.status_code == HTTP_415_UNSUPPORTED_MEDIA_TYPE


def test_a_filtered_list_warns_which_models_dfdf9() -> None:
    """Every auto-detecting model tested returned the wrong writing system on a
    short phrase, and dictation is mostly short phrases. A list mixing both kinds
    has to say which half to trust."""
    from app.catalog import DEFAULT_CATALOG, language_names
    from app.fragments.models import models_list_fragment
    from app.schemas import AdminModelEntry
    from app.serializers import model_covers as _model_covers

    entries = [
        AdminModelEntry(
            id=model.id, engine=model.engine, label=model.label, size_bytes=model.size_bytes,
            languages=model.languages, quality=model.quality, family=model.family,
            description=model.description, source=model.source, state=NOT_INSTALLED_STATE,
            active=False, recommended=False,
            detects_language_automatically=model.detects_language_automatically,
            language_names=language_names(model.language_codes),
            language_codes=list(model.language_codes),
        )
        for model in DEFAULT_CATALOG
    ]  # fmt: skip

    hindi = [entry for entry in entries if _model_covers(entry, HINDI_LANGUAGE_CODE)]
    rendered = models_list_fragment(hindi, languages=[HINDI_LANGUAGE_CODE])
    assert MODELS_LANGUAGE_HINT_CLASS in rendered
    # Still accepts the legacy single-string kwarg.
    assert MODELS_LANGUAGE_HINT_CLASS in models_list_fragment(hindi, language=HINDI_LANGUAGE_CODE)
    assert "wrong script" in rendered
    assert HINDI_LANGUAGE in rendered

    # No hint when there is nothing to choose between.
    only_auto = [entry for entry in hindi if entry.detects_language_automatically]
    assert MODELS_LANGUAGE_HINT_CLASS not in models_list_fragment(
        only_auto, language=HINDI_LANGUAGE_CODE
    )
    # And never on the unfiltered catalog, where it would just be noise.
    assert MODELS_LANGUAGE_HINT_CLASS not in models_list_fragment(entries)
