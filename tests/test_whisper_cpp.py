from __future__ import annotations

from pathlib import Path

import pytest

from app.errors import EngineUnavailableError, TranscriptionProcessError
from app.models.base import TranscriptionOptions
from app.models.whisper_cpp import WhisperCppEngine

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
