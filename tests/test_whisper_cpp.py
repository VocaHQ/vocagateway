from __future__ import annotations

from pathlib import Path

import pytest

from app.catalog import CatalogModel
from app.errors import EngineUnavailableError, LanguageUnsupportedError, TranscriptionProcessError
from app.models.base import TranscriptionOptions
from app.models.whisper_cpp import (
    DECODER_BEAM_SIZE,
    DECODER_BEST_OF,
    WhisperCppEngine,
    _build_arguments,
)

EXECUTABLE_FILE_MODE = 0o700
WHISPER_BINARY_NAME = "whisper-cli"
MODEL_FILE_NAME = "model.bin"
MODEL_BYTES = b"model"
AUDIO_FILE_NAME = "audio.wav"
AUDIO_BYTES = b"audio"
RAW_STYLE = "raw"
AUTO_LANGUAGE = "auto"


def _write_binary(path: Path, script: str) -> None:
    path.write_text(script, encoding="utf-8")
    path.chmod(EXECUTABLE_FILE_MODE)


async def test_health_requires_both_the_binary_an_aa(tmp_path: Path) -> None:
    binary = tmp_path / WHISPER_BINARY_NAME
    model = tmp_path / MODEL_FILE_NAME

    engine = WhisperCppEngine(binary, model)
    assert (await engine.health()).ready is False

    _write_binary(binary, "#!/bin/sh\nexit 0\n")
    assert (await engine.health()).ready is False

    model.write_bytes(MODEL_BYTES)
    health = await engine.health()
    assert health.ready is True
    assert health.name == f"whisper.cpp:{model.name}"


@pytest.mark.parametrize(
    ("language", "expected_text", "expect_lang_flag"),
    [
        ("en", "private local result", True),
        (AUTO_LANGUAGE, "auto detected", False),
    ],
)
async def test_transcribe_writes_the_output_stem_aaa(
    tmp_path: Path, language: str, expected_text: str, expect_lang_flag: bool
) -> None:
    binary = tmp_path / WHISPER_BINARY_NAME
    _write_binary(
        binary,
        r"""#!/bin/sh
printf '%s\n' "$@" > "$0.args"
of=""
lang=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    -of) of="$2"; shift 2 ;;
    -l) lang="$2"; shift 2 ;;
    *) shift ;;
  esac
done
if [ -n "$lang" ]; then
  printf '%s' "private local result" > "$of.txt"
else
  printf '%s' "auto detected" > "$of.txt"
fi
""",
    )
    model = tmp_path / MODEL_FILE_NAME
    model.write_bytes(MODEL_BYTES)
    audio = tmp_path / AUDIO_FILE_NAME
    audio.write_bytes(AUDIO_BYTES)

    transcript = await WhisperCppEngine(binary, model).transcribe(
        audio, TranscriptionOptions(language, RAW_STYLE)
    )

    assert transcript == expected_text
    recorded = (tmp_path / "whisper-cli.args").read_text(encoding="utf-8").splitlines()
    if expect_lang_flag:
        assert recorded[recorded.index("-l") + 1] == language
        assert recorded[recorded.index("-f") + 1] == str(audio)
    else:
        assert "-l" not in recorded


async def test_transcribe_uses_catalog_decoder_language_for_output_contract(
    tmp_path: Path,
) -> None:
    binary = tmp_path / WHISPER_BINARY_NAME
    _write_binary(
        binary,
        r"""#!/bin/sh
printf '%s\n' "$@" > "$0.args"
of=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    -of) of="$2"; shift 2 ;;
    *) shift ;;
  esac
done
printf '%s' "aaj office hai" > "$of.txt"
""",
    )
    model = tmp_path / MODEL_FILE_NAME
    model.write_bytes(MODEL_BYTES)
    audio = tmp_path / AUDIO_FILE_NAME
    audio.write_bytes(AUDIO_BYTES)
    catalog_model = CatalogModel(
        id="whisper.cpp:hinglish",
        engine="whisper.cpp",
        key=MODEL_FILE_NAME,
        label="Hinglish",
        size_bytes=1,
        languages="Hindi + English, Roman script",
        quality="Experimental",
        minimum_ram_gb=4,
        language_codes=("hinglish_roman",),
        decoder_language_code="hi",
    )

    engine = WhisperCppEngine(binary, model, catalog_model)
    await engine.transcribe(audio, TranscriptionOptions("hinglish_roman", RAW_STYLE))

    recorded = (tmp_path / "whisper-cli.args").read_text(encoding="utf-8").splitlines()
    assert recorded[recorded.index("-l") + 1] == "hi"

    with pytest.raises(LanguageUnsupportedError, match="only hinglish_roman"):
        await engine.transcribe(audio, TranscriptionOptions("en", RAW_STYLE))


async def test_fixed_output_contract_rejects_devanagari_leakage(tmp_path: Path) -> None:
    binary = tmp_path / WHISPER_BINARY_NAME
    _write_binary(
        binary,
        r"""#!/bin/sh
of=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    -of) of="$2"; shift 2 ;;
    *) shift ;;
  esac
done
printf '%s' "Aaj office में meeting hai" > "$of.txt"
""",
    )
    model = tmp_path / MODEL_FILE_NAME
    model.write_bytes(MODEL_BYTES)
    audio = tmp_path / AUDIO_FILE_NAME
    audio.write_bytes(AUDIO_BYTES)
    catalog_model = CatalogModel(
        id="whisper.cpp:hinglish",
        engine="whisper.cpp",
        key=MODEL_FILE_NAME,
        label="Hinglish",
        size_bytes=1,
        languages="Hindi + English, Roman script",
        quality="Experimental",
        minimum_ram_gb=4,
        language_codes=("hinglish_roman",),
        decoder_language_code="hi",
    )

    with pytest.raises(LanguageUnsupportedError, match="required hinglish_roman"):
        await WhisperCppEngine(binary, model, catalog_model).transcribe(
            audio, TranscriptionOptions("auto", RAW_STYLE)
        )


async def test_transcribe_raises_when_the_engine_aaaa(tmp_path: Path) -> None:
    engine = WhisperCppEngine(tmp_path / "missing-cli", tmp_path / "missing-model.bin")

    with pytest.raises(EngineUnavailableError):
        await engine.transcribe(
            tmp_path / AUDIO_FILE_NAME, TranscriptionOptions(AUTO_LANGUAGE, RAW_STYLE)
        )


async def test_transcribe_raises_on_a_nonzero_exit_code(tmp_path: Path) -> None:
    binary = tmp_path / WHISPER_BINARY_NAME
    _write_binary(binary, "#!/bin/sh\necho 'boom' 1>&2\nexit 1\n")
    model = tmp_path / MODEL_FILE_NAME
    model.write_bytes(MODEL_BYTES)
    audio = tmp_path / AUDIO_FILE_NAME
    audio.write_bytes(AUDIO_BYTES)

    engine = WhisperCppEngine(binary, model)

    with pytest.raises(TranscriptionProcessError, match="boom"):
        await engine.transcribe(audio, TranscriptionOptions(AUTO_LANGUAGE, RAW_STYLE))


async def test_transcribe_raises_when_the_transcr_c5efb(tmp_path: Path) -> None:
    binary = tmp_path / WHISPER_BINARY_NAME
    _write_binary(
        binary,
        """#!/bin/sh
of=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    -of) of="$2"; shift 2 ;;
    *) shift ;;
  esac
done
printf '' > "$of.txt"
""",
    )
    model = tmp_path / MODEL_FILE_NAME
    model.write_bytes(MODEL_BYTES)
    audio = tmp_path / AUDIO_FILE_NAME
    audio.write_bytes(AUDIO_BYTES)

    engine = WhisperCppEngine(binary, model)

    with pytest.raises(TranscriptionProcessError, match="empty"):
        await engine.transcribe(audio, TranscriptionOptions(AUTO_LANGUAGE, RAW_STYLE))


async def test_warmup_prefetches_the_model_when_ready(tmp_path: Path) -> None:
    missing_engine = WhisperCppEngine(tmp_path / "missing-cli", tmp_path / "missing-model.bin")
    assert await missing_engine.warmup() == 0

    binary = tmp_path / WHISPER_BINARY_NAME
    _write_binary(binary, "#!/bin/sh\nexit 0\n")
    model = tmp_path / MODEL_FILE_NAME
    model.write_bytes(b"x" * 1024)

    ready_engine = WhisperCppEngine(binary, model)
    advised = await ready_engine.warmup()

    assert advised > 0


def test_decoding_flags_narrow_the_search_and_use_the_whole_cpu(tmp_path: Path) -> None:
    """whisper-cli's own defaults (4 threads, beam 5, best-of 5) are batch settings.

    A dictation clip is short and the decoder dominates on a CPU-only host, so
    the gateway asks for the machine's cores and a narrower beam instead.
    """
    arguments = _build_arguments(
        tmp_path / WHISPER_BINARY_NAME,
        tmp_path / MODEL_FILE_NAME,
        tmp_path / AUDIO_FILE_NAME,
        tmp_path / "result",
        AUTO_LANGUAGE,
        6,
    )

    assert arguments[arguments.index("-t") + 1] == "6"
    assert arguments[arguments.index("-bs") + 1] == str(DECODER_BEAM_SIZE)
    assert arguments[arguments.index("-bo") + 1] == str(DECODER_BEST_OF)
    # Temperature fallback stays on: it is what rescues a repetition loop.
    assert "-nf" not in arguments


async def test_the_operator_thread_override_reaches_the_command_line(tmp_path: Path) -> None:
    """The WebUI's CPU threads box has to survive the whole way to argv.

    Before this engine took `cpu_threads` it silently ran on whisper-cli's own
    default of 4 no matter what the operator had chosen.
    """
    binary = tmp_path / WHISPER_BINARY_NAME
    _write_binary(
        binary,
        r"""#!/bin/sh
printf '%s\n' "$@" > "$0.args"
of=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    -of) of="$2"; shift 2 ;;
    *) shift ;;
  esac
done
printf '%s' "threaded result" > "$of.txt"
""",
    )
    model = tmp_path / MODEL_FILE_NAME
    model.write_bytes(MODEL_BYTES)
    audio = tmp_path / AUDIO_FILE_NAME
    audio.write_bytes(AUDIO_BYTES)

    await WhisperCppEngine(binary, model, cpu_threads=3).transcribe(
        audio, TranscriptionOptions(AUTO_LANGUAGE, RAW_STYLE)
    )

    recorded = (tmp_path / "whisper-cli.args").read_text(encoding="utf-8").splitlines()
    assert recorded[recorded.index("-t") + 1] == "3"
