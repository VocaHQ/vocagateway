from __future__ import annotations

from pathlib import Path

from app.models.warmup import prefetch_model_paths

LARGE_MODEL_BYTES = 4_096
MODEL_WEIGHT_BYTES = 1_024
TOKENIZER_BYTES = 256
PREFETCH_BUDGET_BYTES = 2_048


def test_prefetch_a_single_file_advises_its_aa(tmp_path: Path) -> None:
    path = tmp_path / "model.bin"
    path.write_bytes(b"x" * LARGE_MODEL_BYTES)

    assert prefetch_model_paths([path]) == LARGE_MODEL_BYTES


def test_prefetch_walks_directories_for_files(tmp_path: Path) -> None:
    directory = tmp_path / "model"
    directory.mkdir()
    (directory / "weights.bin").write_bytes(b"x" * MODEL_WEIGHT_BYTES)
    (directory / "tokenizer.json").write_bytes(b"y" * TOKENIZER_BYTES)

    assert prefetch_model_paths([directory]) == MODEL_WEIGHT_BYTES + TOKENIZER_BYTES


def test_prefetch_skips_paths_that_do_not_exist(tmp_path: Path) -> None:
    assert prefetch_model_paths([tmp_path / "does-not-exist"]) == 0


def test_prefetch_stops_once_the_byte_budge_aaa(tmp_path: Path) -> None:
    large = tmp_path / "large.bin"
    large.write_bytes(b"x" * LARGE_MODEL_BYTES)
    small = tmp_path / "small.bin"
    small.write_bytes(b"y" * MODEL_WEIGHT_BYTES)

    advised = prefetch_model_paths([large, small], maximum_bytes=PREFETCH_BUDGET_BYTES)

    assert advised == PREFETCH_BUDGET_BYTES


def test_prefetch_with_no_candidates_advise_cf6cf(tmp_path: Path) -> None:
    assert prefetch_model_paths([]) == 0
