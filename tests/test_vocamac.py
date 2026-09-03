from __future__ import annotations

import json
import plistlib
from pathlib import Path

import pytest

from app.errors import EngineUnavailableError
from app.models.base import EngineTranscription, TranscriptionOptions
from app.models.vocamac import REQUIRED_COMPONENTS, VocaMacEngine

EXECUTABLE_FILE_MODE = 0o700
DEFAULT_TEST_WEIGHT_BYTES = 16
LARGE_TEST_WEIGHT_BYTES = 512
EXPECTED_HEADLESS_INFERENCE_MS = 125
UTF8_ENCODING = "utf-8"
SUPPORT_DIRECTORY_NAME = "support"
MODELS_DIRECTORY_NAME = "models"
PARAKEET_MODEL = "parakeet-tdt-0.6b-v2"
SMALL_MODEL = "small"
AUDIO_FILE_NAME = "audio.wav"
AUDIO_FORMAT = "audio"
TEST_AUDIO_BYTES = b"audio"
RAW_STYLE = "raw"
WHISPER_SMALL_MODEL = "openai_whisper-small"
AUTO_LANGUAGE = "auto"
WHISPER_TINY_MODEL = "openai_whisper-tiny"

# Records the arguments of the last call so the one-shot CLI path stays inspectable.
FAKE_CLI = """#!/bin/sh
printf '%s\\n' "$@" > "$0.args"
case "$1" in
  serve) exit 1 ;;
esac
printf '%s\\n' 'private local result'
"""


def _write_model(
    models_dir: Path, variant: str, *, weight_bytes: int = DEFAULT_TEST_WEIGHT_BYTES
) -> Path:
    """Create a complete VocaMac Core ML model folder."""
    directory = models_dir / variant
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text("{}", encoding=UTF8_ENCODING)
    for component in REQUIRED_COMPONENTS:
        weights = directory / component / "weights"
        weights.mkdir(parents=True)
        (weights / "weight.bin").write_bytes(b"w" * weight_bytes)
        (directory / component / "metadata.json").write_text("[]", encoding=UTF8_ENCODING)
        (directory / component / "coremldata.bin").write_bytes(b"model")
    return directory


def _write_partial_model(models_dir: Path, variant: str) -> Path:
    """Create the folder an interrupted VocaMac download leaves behind."""
    directory = models_dir / variant
    for component in REQUIRED_COMPONENTS:
        (directory / component / "weights").mkdir(parents=True)
    return directory


def _write_preferences(tmp_path: Path, selected: str) -> None:
    with (tmp_path / "com.vocamac.app.plist").open("wb") as file_handle:
        plistlib.dump({"vocamac.selectedModelSize": selected}, file_handle)


def _write_audio(tmp_path: Path) -> Path:
    audio = tmp_path / AUDIO_FILE_NAME
    audio.write_bytes(TEST_AUDIO_BYTES)
    return audio


def _headless_script(
    models: list[dict[str, object]],
    transcription: dict[str, object] | None,
    failure: dict[str, str] | None,
) -> str:
    model_payload = json.dumps({MODELS_DIRECTORY_NAME: models}, separators=(",", ":"))
    transcription_payload = json.dumps(
        transcription
        or {
            "text": "private local result",
            "model": PARAKEET_MODEL,
            "engine": "parakeet",
            "detected_language": "en",
            "duration_seconds": 0.125,
            "audio_length_seconds": 1.0,
        },
        separators=(",", ":"),
    )
    failure_payload = json.dumps(failure, separators=(",", ":")) if failure else ""
    transcribe_line = (
        f"  --transcribe-file) printf '%s\\n' '{failure_payload}' >&2; exit 4 ;;\n"
        if failure
        else f"  --transcribe-file) printf '%s\\n' '{transcription_payload}' ;;\n"
    )
    return (
        "#!/bin/sh\n"
        "# --transcribe-file capability marker\n"
        'printf \'%s\\n\' "$@" > "$0.args"\n'
        'case "$1" in\n'
        f"  --list-models) printf '%s\\n' '{model_payload}' ;;\n"
        + transcribe_line
        + "  *) exit 2 ;;\n"
        "esac\n"
    )


def _build_engine(tmp_path: Path, *, model: str | None = None) -> tuple[VocaMacEngine, Path]:
    app_path = tmp_path / "VocaMac.app"
    app_path.mkdir(exist_ok=True)
    binary = tmp_path / "whisperkit-cli"
    binary.write_text(FAKE_CLI, encoding=UTF8_ENCODING)
    binary.chmod(EXECUTABLE_FILE_MODE)
    models_dir = (
        tmp_path
        / SUPPORT_DIRECTORY_NAME
        / MODELS_DIRECTORY_NAME
        / MODELS_DIRECTORY_NAME
        / "argmaxinc"
        / "whisperkit-coreml"
    )
    models_dir.mkdir(parents=True, exist_ok=True)
    engine = VocaMacEngine(
        str(binary),
        model,
        app_path=app_path,
        support_dir=tmp_path / SUPPORT_DIRECTORY_NAME,
        preferences_file=tmp_path / "com.vocamac.app.plist",
    )
    return engine, models_dir


def _build_headless_engine(
    tmp_path: Path,
    models: list[dict[str, object]],
    *,
    model: str | None = None,
    transcription: dict[str, object] | None = None,
    failure: dict[str, str] | None = None,
) -> tuple[VocaMacEngine, Path]:
    app_path = tmp_path / "VocaMac.app"
    executable = app_path / "Contents" / "MacOS" / "VocaMac"
    executable.parent.mkdir(parents=True)
    executable.write_text(_headless_script(models, transcription, failure), encoding=UTF8_ENCODING)
    executable.chmod(EXECUTABLE_FILE_MODE)
    engine = VocaMacEngine(
        str(tmp_path / "missing-whisperkit-cli"),
        model,
        app_path=app_path,
        support_dir=tmp_path / SUPPORT_DIRECTORY_NAME,
        preferences_file=tmp_path / "com.vocamac.app.plist",
    )
    return engine, executable


def _headless_model(
    model_id: str,
    *,
    selected: bool,
    downloaded: bool = True,
    supported: bool = True,
) -> dict[str, object]:
    return {
        "id": model_id,
        "name": model_id,
        "engine": "parakeet" if model_id.startswith("parakeet") else "whisperkit",
        "selected": selected,
        "downloaded": downloaded,
        "supported": supported,
        "system_managed": False,
    }


async def test_vocamac_headless_cli_uses_the_apps_aa(
    tmp_path: Path,
) -> None:
    engine, executable = _build_headless_engine(
        tmp_path,
        [
            _headless_model(SMALL_MODEL, selected=False),
            _headless_model(PARAKEET_MODEL, selected=True),
        ],
    )
    audio = _write_audio(tmp_path)

    assert engine.is_available() is True
    assert (await engine.health()).name == "vocamac:parakeet-tdt-0.6b-v2"
    operation_result = await engine.transcribe(audio, TranscriptionOptions("en", RAW_STYLE))

    assert operation_result.text == "private local result"
    assert operation_result.inference_ms == EXPECTED_HEADLESS_INFERENCE_MS
    arguments = Path(f"{executable}.args").read_text(encoding=UTF8_ENCODING).splitlines()
    assert arguments[:3] == ["--transcribe-file", str(audio), "--json"]
    assert arguments[arguments.index("--language") + 1] == "en"
    assert "--model" not in arguments


async def test_vocamac_headless_cli_honours_an_ex_aaa(
    tmp_path: Path,
) -> None:
    engine, executable = _build_headless_engine(
        tmp_path,
        [
            _headless_model(SMALL_MODEL, selected=False),
            _headless_model(PARAKEET_MODEL, selected=True),
        ],
        model=WHISPER_SMALL_MODEL,
    )
    audio = _write_audio(tmp_path)

    assert engine.is_available() is True
    assert (await engine.health()).name == "vocamac:small"
    await engine.transcribe(audio, TranscriptionOptions(AUTO_LANGUAGE, RAW_STYLE))

    arguments = Path(f"{executable}.args").read_text(encoding=UTF8_ENCODING).splitlines()
    assert arguments[arguments.index("--model") + 1] == SMALL_MODEL
    assert arguments[arguments.index("--language") + 1] == AUTO_LANGUAGE


async def test_vocamac_headless_cli_reports_an_un_aaaa(
    tmp_path: Path,
) -> None:
    engine, _ = _build_headless_engine(
        tmp_path,
        [_headless_model(PARAKEET_MODEL, selected=True, downloaded=False)],
    )

    health = await engine.health()

    assert engine.is_available() is False
    assert health.ready is False
    assert health.name == "vocamac:parakeet-tdt-0.6b-v2"


async def test_vocamac_headless_cli_surfaces_mode_aaaaa(
    tmp_path: Path,
) -> None:
    engine, _ = _build_headless_engine(
        tmp_path,
        [_headless_model(PARAKEET_MODEL, selected=True)],
        failure={
            "error": "model_not_downloaded",
            "message": "Model is not downloaded: parakeet-tdt-0.6b-v2",
        },
    )
    audio = _write_audio(tmp_path)

    with pytest.raises(EngineUnavailableError, match="Model is not downloaded"):
        await engine.transcribe(audio, TranscriptionOptions(AUTO_LANGUAGE, RAW_STYLE))


async def test_vocamac_headless_warmup_does_not_l_a(
    tmp_path: Path,
) -> None:
    engine, executable = _build_headless_engine(
        tmp_path,
        [_headless_model(PARAKEET_MODEL, selected=True)],
    )

    assert await engine.warmup() == 0
    assert not Path(f"{executable}.args").exists()


async def test_vocamac_without_headless_cli_keeps_aa(tmp_path: Path) -> None:
    engine, models_dir = _build_engine(tmp_path)
    executable = engine.app_path / "Contents" / "MacOS" / "VocaMac"
    executable.parent.mkdir(parents=True)
    executable.write_text(
        '#!/bin/sh\nprintf \'%s\\n\' "$@" > "$0.args"\nexit 0\n',
        encoding=UTF8_ENCODING,
    )
    executable.chmod(EXECUTABLE_FILE_MODE)
    _write_model(models_dir, WHISPER_SMALL_MODEL)
    _write_preferences(tmp_path, SMALL_MODEL)
    audio = _write_audio(tmp_path)

    await engine.transcribe(audio, TranscriptionOptions(AUTO_LANGUAGE, RAW_STYLE))

    assert (await engine.health()).ready is True
    assert (await engine.health()).name == "vocamac:openai_whisper-small"
    assert not Path(f"{executable}.args").exists()
    arguments = (tmp_path / "whisperkit-cli.args").read_text(encoding=UTF8_ENCODING).splitlines()
    assert arguments[arguments.index("--model-path") + 1] == str(models_dir / WHISPER_SMALL_MODEL)


async def test_vocamac_runs_the_model_selected_in_aaa(tmp_path: Path) -> None:
    engine, models_dir = _build_engine(tmp_path)
    _write_model(models_dir, WHISPER_TINY_MODEL, weight_bytes=LARGE_TEST_WEIGHT_BYTES)
    _write_model(models_dir, WHISPER_SMALL_MODEL)
    _write_preferences(tmp_path, SMALL_MODEL)
    audio = _write_audio(tmp_path)

    transcript = await engine.transcribe(audio, TranscriptionOptions(AUTO_LANGUAGE, RAW_STYLE))
    arguments = (tmp_path / "whisperkit-cli.args").read_text(encoding=UTF8_ENCODING).splitlines()

    assert (await engine.health()).ready is True
    assert (await engine.health()).name == "vocamac:openai_whisper-small"
    assert isinstance(transcript, EngineTranscription)
    assert transcript.text == "private local result"
    assert arguments[arguments.index("--model-path") + 1] == str(models_dir / WHISPER_SMALL_MODEL)
    # VocaMac already downloaded the matching tokenizer, so reuse it.
    assert arguments[arguments.index("--download-tokenizer-path") + 1] == str(
        tmp_path / SUPPORT_DIRECTORY_NAME / MODELS_DIRECTORY_NAME
    )


async def test_vocamac_skips_an_interrupted_download(tmp_path: Path) -> None:
    engine, models_dir = _build_engine(tmp_path)
    _write_partial_model(models_dir, WHISPER_TINY_MODEL)
    _write_model(models_dir, WHISPER_SMALL_MODEL)
    _write_preferences(tmp_path, "tiny")

    health = await engine.health()

    assert health.ready is True
    assert health.name == "vocamac:openai_whisper-small"


async def test_vocamac_does_not_replace_a_selecte_aaaa(
    tmp_path: Path,
) -> None:
    engine, models_dir = _build_engine(tmp_path)
    _write_model(models_dir, WHISPER_SMALL_MODEL)
    _write_preferences(tmp_path, PARAKEET_MODEL)

    health = await engine.health()

    assert engine.is_available() is False
    assert health.ready is False
    assert health.name == "vocamac:parakeet-tdt-0.6b-v2"

    audio = tmp_path / AUDIO_FILE_NAME
    audio.write_bytes(TEST_AUDIO_BYTES)
    with pytest.raises(EngineUnavailableError, match="not a WhisperKit model"):
        await engine.transcribe(audio, TranscriptionOptions(AUTO_LANGUAGE, RAW_STYLE))


async def test_vocamac_does_not_replace_an_unknow_aaaaa(
    tmp_path: Path,
) -> None:
    engine, models_dir = _build_engine(tmp_path)
    _write_model(models_dir, WHISPER_SMALL_MODEL)
    _write_preferences(tmp_path, "future-vocamac-engine-model")

    health = await engine.health()

    assert engine.is_available() is False
    assert health.ready is False
    assert health.name == "vocamac:future-vocamac-engine-model"


async def test_vocamac_prefers_the_largest_model_a(tmp_path: Path) -> None:
    engine, models_dir = _build_engine(tmp_path)
    _write_model(models_dir, WHISPER_TINY_MODEL, weight_bytes=DEFAULT_TEST_WEIGHT_BYTES)
    largest = _write_model(models_dir, WHISPER_SMALL_MODEL, weight_bytes=LARGE_TEST_WEIGHT_BYTES)

    assert (await engine.health()).name == f"vocamac:{largest.name}"


async def test_vocamac_honours_an_explicit_model_aa(tmp_path: Path) -> None:
    engine, models_dir = _build_engine(tmp_path, model="tiny")
    pinned = _write_model(models_dir, WHISPER_TINY_MODEL)
    _write_model(models_dir, WHISPER_SMALL_MODEL, weight_bytes=LARGE_TEST_WEIGHT_BYTES)
    _write_preferences(tmp_path, SMALL_MODEL)

    assert (await engine.health()).name == f"vocamac:{pinned.name}"


async def test_vocamac_never_substitutes_a_config_f22a7(tmp_path: Path) -> None:
    engine, models_dir = _build_engine(tmp_path, model="large-v3")
    _write_model(models_dir, WHISPER_SMALL_MODEL)

    health = await engine.health()

    assert engine.is_available() is False
    assert health.ready is False
    assert health.name == "vocamac:openai_whisper-large-v3"


async def test_vocamac_is_unavailable_without_the_aaa(tmp_path: Path) -> None:
    engine, models_dir = _build_engine(tmp_path)

    assert engine.is_available() is False
    assert (await engine.health()).name == "vocamac:no-model-selected"

    _write_model(models_dir, WHISPER_SMALL_MODEL)
    assert engine.is_available() is True

    engine.app_path.rmdir()
    assert engine.is_available() is False
    assert (await engine.health()).ready is False


async def test_vocamac_is_unavailable_without_whi_aaaa(tmp_path: Path) -> None:
    engine, models_dir = _build_engine(tmp_path)
    _write_model(models_dir, WHISPER_SMALL_MODEL)
    Path(engine.whisperkit_binary).unlink()

    assert engine.is_available() is False
    assert (await engine.health()).ready is False
