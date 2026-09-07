from __future__ import annotations

from app.cleanup.base import CHUNK_CHAR_LIMIT, MINIMUM_CONTEXT_TOKENS
from app.cleanup.profile import (
    COMPACT_PROFILE,
    FULL_PROFILE,
    FULL_RAM_GB,
    chunk_limit_for_window,
    describe_profile,
    resolve_profile,
)
from app.system import CPU_ACCELERATOR, SystemInfo


def _info(*, ram_gb: float, accelerators: tuple[str, ...]) -> SystemInfo:
    return SystemInfo(
        os_name="Linux",
        os_version="test",
        arch="x86_64",
        chip="test",
        ram_gb=ram_gb,
        is_apple_silicon=False,
        ffmpeg_path=None,
        whisper_cpp_path=None,
        llama_server_path=None,
        whisperkit_cli_path=None,
        handy_installed=False,
        vocamac_installed=False,
        logical_cpus=4,
        effective_cpus=4.0,
        containerized=False,
        accelerators=accelerators,
        cpu_features=(),
    )


def test_cpu_only_small_ram_selects_compact() -> None:
    profile = resolve_profile("auto", _info(ram_gb=8.0, accelerators=(CPU_ACCELERATOR,)))
    assert profile is COMPACT_PROFILE
    assert profile.context_tokens == MINIMUM_CONTEXT_TOKENS
    assert profile.kv_cache_type == "q8_0"


def test_gpu_selects_full_even_on_small_ram() -> None:
    profile = resolve_profile(
        "auto", _info(ram_gb=8.0, accelerators=(CPU_ACCELERATOR, "NVIDIA RTX"))
    )
    assert profile is FULL_PROFILE
    assert profile.context_tokens == 8_192
    assert profile.kv_cache_type == "f16"


def test_cpu_only_with_enough_ram_selects_full() -> None:
    profile = resolve_profile("auto", _info(ram_gb=FULL_RAM_GB, accelerators=(CPU_ACCELERATOR,)))
    assert profile is FULL_PROFILE


def test_explicit_compact_wins_on_a_gpu_host() -> None:
    profile = resolve_profile(
        "compact", _info(ram_gb=32.0, accelerators=(CPU_ACCELERATOR, "Metal / Core ML"))
    )
    assert profile is COMPACT_PROFILE


def test_explicit_full_wins_on_a_small_cpu_host() -> None:
    assert (
        resolve_profile("full", _info(ram_gb=4.0, accelerators=(CPU_ACCELERATOR,))) is FULL_PROFILE
    )


def test_unknown_request_falls_back_to_auto() -> None:
    profile = resolve_profile("turbo", _info(ram_gb=8.0, accelerators=(CPU_ACCELERATOR,)))
    assert profile is COMPACT_PROFILE


def test_missing_host_probe_defaults_to_compact() -> None:
    assert resolve_profile("auto") is COMPACT_PROFILE


def test_describe_profile_says_when_it_was_forced() -> None:
    text = describe_profile(COMPACT_PROFILE, "compact")
    assert "VOCAGATEWAY_CLEANUP_PROFILE=compact" in text
    automatic = describe_profile(FULL_PROFILE, None)
    assert "automatically" in automatic
    assert "VOCAGATEWAY_CLEANUP_PROFILE" not in automatic


def test_chunk_limit_grows_with_a_larger_server_window() -> None:
    assert chunk_limit_for_window(0) == CHUNK_CHAR_LIMIT
    assert chunk_limit_for_window(4_096) >= CHUNK_CHAR_LIMIT
    assert chunk_limit_for_window(8_192) > chunk_limit_for_window(4_096)
