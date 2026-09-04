from __future__ import annotations

from pathlib import Path

import pytest

from app.catalog import CatalogModel
from app.errors import EngineUnavailableError, LanguageUnsupportedError, TranscriptionProcessError
from app.models import whisper_server
from app.models.base import MemoryResidentEngine, TranscriptionOptions
from app.models.whisper_cpp import (
    DECODER_BEAM_SIZE,
    DECODER_BEST_OF,
    WhisperCppEngine,
    _build_arguments,
)
from app.models.whisper_server import resolve_server_binary

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


def _cli_engine(
    binary: Path,
    model: Path,
    catalog_model: CatalogModel | None = None,
    *,
    cpu_threads: int = 0,
) -> WhisperCppEngine:
    """An engine pinned to the one-shot CLI path.

    The override names a file that does not exist, which is how a host without
    `whisper-server` is reported, so these cases stay on the fallback even when
    the developer's PATH happens to carry a real server binary.
    """
    return WhisperCppEngine(
        binary,
        model,
        catalog_model,
        cpu_threads=cpu_threads,
        server_binary=binary.parent / "absent-whisper-server",
    )


async def test_health_requires_both_the_binary_an_aa(tmp_path: Path) -> None:
    binary = tmp_path / WHISPER_BINARY_NAME
    model = tmp_path / MODEL_FILE_NAME

    engine = _cli_engine(binary, model)
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

    outcome = await _cli_engine(binary, model).transcribe(
        audio, TranscriptionOptions(language, RAW_STYLE)
    )

    assert outcome.text == expected_text
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

    engine = _cli_engine(binary, model, catalog_model)
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
        await _cli_engine(binary, model, catalog_model).transcribe(
            audio, TranscriptionOptions("auto", RAW_STYLE)
        )


async def test_transcribe_raises_when_the_engine_aaaa(tmp_path: Path) -> None:
    engine = _cli_engine(tmp_path / "missing-cli", tmp_path / "missing-model.bin")

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

    engine = _cli_engine(binary, model)

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

    engine = _cli_engine(binary, model)

    with pytest.raises(TranscriptionProcessError, match="empty"):
        await engine.transcribe(audio, TranscriptionOptions(AUTO_LANGUAGE, RAW_STYLE))


async def test_warmup_prefetches_the_model_when_ready(tmp_path: Path) -> None:
    missing_engine = _cli_engine(tmp_path / "missing-cli", tmp_path / "missing-model.bin")
    assert await missing_engine.warmup() == 0

    binary = tmp_path / WHISPER_BINARY_NAME
    _write_binary(binary, "#!/bin/sh\nexit 0\n")
    model = tmp_path / MODEL_FILE_NAME
    model.write_bytes(b"x" * 1024)

    ready_engine = _cli_engine(binary, model)
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

    await _cli_engine(binary, model, cpu_threads=3).transcribe(
        audio, TranscriptionOptions(AUTO_LANGUAGE, RAW_STYLE)
    )

    recorded = (tmp_path / "whisper-cli.args").read_text(encoding="utf-8").splitlines()
    assert recorded[recorded.index("-t") + 1] == "3"


SERVER_BINARY_NAME = "whisper-server"
RESIDENT_TRANSCRIPT = "resident transcript"
# A stand-in for whisper-server: it records how often it was launched and with
# which arguments, then answers /inference the way the real server does with
# `response_format=text`.
FAKE_SERVER_SCRIPT = r"""#!/usr/bin/env python3
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ARGUMENTS = sys.argv[1:]
PORT = int(ARGUMENTS[ARGUMENTS.index("--port") + 1])
STARTS = Path(sys.argv[0] + ".starts")
LAUNCH = int(STARTS.read_text()) + 1 if STARTS.exists() else 1
STARTS.write_text(str(LAUNCH))
Path(sys.argv[0] + ".args").write_text("\n".join(ARGUMENTS))
REFUSALS = Path(sys.argv[0] + ".refusals")
if REFUSALS.exists() and LAUNCH <= int(REFUSALS.read_text()):
    raise SystemExit(1)


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        Path(sys.argv[0] + ".body").write_bytes(body)
        payload = b"resident transcript"
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *unused):
        return None


HTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
"""
COUNTING_CLI_SCRIPT = r"""#!/bin/sh
of=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    -of) of="$2"; shift 2 ;;
    *) shift ;;
  esac
done
printf 'x' >> "$0.runs"
printf '%s' "cli transcript" > "$of.txt"
"""


def _resident_engine(tmp_path: Path, server_script: str) -> WhisperCppEngine:
    binary = tmp_path / WHISPER_BINARY_NAME
    _write_binary(binary, COUNTING_CLI_SCRIPT)
    _write_binary(tmp_path / SERVER_BINARY_NAME, server_script)
    (tmp_path / MODEL_FILE_NAME).write_bytes(MODEL_BYTES)
    (tmp_path / AUDIO_FILE_NAME).write_bytes(AUDIO_BYTES)
    return WhisperCppEngine(binary, tmp_path / MODEL_FILE_NAME)


def _start_count(tmp_path: Path) -> int:
    record = tmp_path / f"{SERVER_BINARY_NAME}.starts"
    return int(record.read_text(encoding="utf-8")) if record.is_file() else 0


async def test_the_resident_worker_serves_repeat_requests_from_one_load(
    tmp_path: Path,
) -> None:
    """The point of the worker: the model is loaded once, not once per clip."""
    engine = _resident_engine(tmp_path, FAKE_SERVER_SCRIPT)
    audio = tmp_path / AUDIO_FILE_NAME
    try:
        first = await engine.transcribe(audio, TranscriptionOptions("en", RAW_STYLE))
        second = await engine.transcribe(audio, TranscriptionOptions("en", RAW_STYLE))

        assert first.text == RESIDENT_TRANSCRIPT
        assert second.text == RESIDENT_TRANSCRIPT
        assert first.model_load_ms > 0
        assert _start_count(tmp_path) == 1
        assert engine.model_is_resident is True
        # Only the request that loaded the model reports a load cost.
        assert second.model_load_ms == 0
        assert not (tmp_path / f"{WHISPER_BINARY_NAME}.runs").exists()
        # Without this field the server decodes as English whatever the
        # audio is, so the selected output language has to travel per request.
        body = (tmp_path / f"{SERVER_BINARY_NAME}.body").read_bytes()
        assert b'name="language"\r\n\r\nen\r\n' in body
        assert b'name="response_format"\r\n\r\ntext\r\n' in body
    finally:
        engine.unload()


async def test_the_worker_reloads_after_an_idle_offload(tmp_path: Path) -> None:
    engine = _resident_engine(tmp_path, FAKE_SERVER_SCRIPT)
    audio = tmp_path / AUDIO_FILE_NAME
    try:
        await engine.transcribe(audio, TranscriptionOptions(AUTO_LANGUAGE, RAW_STYLE))
        engine.unload()

        assert engine.model_is_resident is False

        reloaded = await engine.transcribe(audio, TranscriptionOptions(AUTO_LANGUAGE, RAW_STYLE))

        assert reloaded.text == RESIDENT_TRANSCRIPT
        assert reloaded.model_load_ms > 0
        assert _start_count(tmp_path) == 2
        assert engine.model_is_resident is True
    finally:
        engine.unload()


async def test_the_worker_passes_the_decoding_flags_and_thread_count(tmp_path: Path) -> None:
    engine = _resident_engine(tmp_path, FAKE_SERVER_SCRIPT)
    try:
        await engine.warmup()

        recorded = (tmp_path / f"{SERVER_BINARY_NAME}.args").read_text(encoding="utf-8").split("\n")
        assert recorded[recorded.index("-m") + 1] == str(tmp_path / MODEL_FILE_NAME)
        assert recorded[recorded.index("--host") + 1] == "127.0.0.1"
        assert recorded[recorded.index("-bs") + 1] == str(DECODER_BEAM_SIZE)
        assert recorded[recorded.index("-bo") + 1] == str(DECODER_BEST_OF)
        assert engine.model_is_resident is True
    finally:
        engine.unload()


async def test_a_server_that_cannot_start_falls_back_to_the_cli(tmp_path: Path) -> None:
    """An older or partial whisper.cpp build must not cost the operator transcripts."""
    engine = _resident_engine(tmp_path, "#!/bin/sh\nexit 1\n")
    audio = tmp_path / AUDIO_FILE_NAME
    try:
        outcome = await engine.transcribe(audio, TranscriptionOptions(AUTO_LANGUAGE, RAW_STYLE))
        await engine.transcribe(audio, TranscriptionOptions(AUTO_LANGUAGE, RAW_STYLE))

        assert outcome.text == "cli transcript"
        assert engine.model_is_resident is False
        # The failed worker is retired, so the second request goes straight to
        # the CLI instead of paying the start attempt again.
        assert (tmp_path / f"{WHISPER_BINARY_NAME}.runs").read_text(encoding="utf-8") == "xx"
    finally:
        engine.unload()


HANGING_SERVER_SCRIPT = r"""#!/usr/bin/env python3
import socket
import sys
import time

ARGUMENTS = sys.argv[1:]
PORT = int(ARGUMENTS[ARGUMENTS.index("--port") + 1])
LISTENER = socket.socket()
LISTENER.bind(("127.0.0.1", PORT))
LISTENER.listen(8)
while True:
    LISTENER.accept()
    time.sleep(30)
"""


async def test_a_slow_clip_times_out_instead_of_running_twice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A timeout is the transcription's fault, not the worker's.

    Treating it as an unreachable worker would hand the same audio to the CLI
    and make the caller wait out a second timeout before hearing about it.
    """
    monkeypatch.setattr(whisper_server, "REQUEST_TIMEOUT_SECONDS", 1)
    engine = _resident_engine(tmp_path, HANGING_SERVER_SCRIPT)
    try:
        with pytest.raises(TranscriptionProcessError, match="timed out"):
            await engine.transcribe(
                tmp_path / AUDIO_FILE_NAME, TranscriptionOptions(AUTO_LANGUAGE, RAW_STYLE)
            )

        assert not (tmp_path / f"{WHISPER_BINARY_NAME}.runs").exists()
        assert engine.model_is_resident is True
    finally:
        engine.unload()


async def test_a_refused_port_is_retried_before_the_cli_takes_over(tmp_path: Path) -> None:
    """Losing the port between picking it and binding it must not cost the worker.

    The gateway reserves a port, closes it, and lets whisper-server bind it, so
    another process can take it first — which looks exactly like a build that
    cannot serve. Retiring the worker for that would silently return the host
    to a model reload per request.
    """
    engine = _resident_engine(tmp_path, FAKE_SERVER_SCRIPT)
    (tmp_path / f"{SERVER_BINARY_NAME}.refusals").write_text("1", encoding="utf-8")
    try:
        outcome = await engine.transcribe(
            tmp_path / AUDIO_FILE_NAME, TranscriptionOptions(AUTO_LANGUAGE, RAW_STYLE)
        )

        assert outcome.text == RESIDENT_TRANSCRIPT
        assert _start_count(tmp_path) == 2
        assert engine.model_is_resident is True
        assert not (tmp_path / f"{WHISPER_BINARY_NAME}.runs").exists()
    finally:
        engine.unload()


def test_the_server_binary_is_taken_from_the_cli_s_own_build(tmp_path: Path) -> None:
    """Both binaries come out of one build, so the sibling is the matching pair."""
    binary = tmp_path / WHISPER_BINARY_NAME
    _write_binary(binary, "#!/bin/sh\nexit 0\n")

    assert resolve_server_binary(binary, tmp_path / "absent") is None

    sibling = tmp_path / SERVER_BINARY_NAME
    _write_binary(sibling, "#!/bin/sh\nexit 0\n")

    assert resolve_server_binary(binary) == sibling
    assert resolve_server_binary(binary, sibling) == sibling
    # A bare name is launched through PATH, so a same-named file in the working
    # directory is not the pair that would actually run.
    assert resolve_server_binary(Path(WHISPER_BINARY_NAME)) != sibling


def test_the_engine_answers_the_idle_offload_protocol(tmp_path: Path) -> None:
    """`EngineManager.offload_if_idle` only reaches engines matching this shape."""
    engine = _cli_engine(tmp_path / WHISPER_BINARY_NAME, tmp_path / MODEL_FILE_NAME)

    assert isinstance(engine, MemoryResidentEngine)
    assert engine.model_is_resident is False
