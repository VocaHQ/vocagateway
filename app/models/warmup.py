from __future__ import annotations

import mmap
import os
from collections.abc import Iterable
from pathlib import Path

DEFAULT_WARMUP_BYTES = 268_435_456


class ModelPrefetcher:
    def prefetch(self, paths: Iterable[Path], maximum_bytes: int = DEFAULT_WARMUP_BYTES) -> int:
        candidates = self._collect_candidates(paths)
        advised = 0
        for size, path in candidates:
            if advised >= maximum_bytes:
                break
            requested = min(size, maximum_bytes - advised)
            if requested > 0:
                advised += self._prefetch_file(path, requested)
        return advised

    def _collect_candidates(self, paths: Iterable[Path]) -> list[tuple[int, Path]]:
        candidates: list[tuple[int, Path]] = []
        for path in paths:
            if path.is_file():
                candidates.append((path.stat().st_size, path))
            elif path.is_dir():
                candidates.extend(
                    (entry.stat().st_size, entry) for entry in path.rglob("*") if entry.is_file()
                )
        return sorted(candidates, reverse=True)

    def _prefetch_file(self, path: Path, length: int) -> int:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            return self._advise_descriptor(descriptor, length)
        except Exception:
            return 0
        finally:
            os.close(descriptor)

    def _advise_descriptor(self, descriptor: int, length: int) -> int:
        if self._fadvise(descriptor, length):
            return length
        return self._madvise(descriptor, length)

    def _fadvise(self, descriptor: int, length: int) -> bool:
        posix_fadvise = getattr(os, "posix_fadvise", None)
        will_need = getattr(os, "POSIX_FADV_WILLNEED", None)
        if callable(posix_fadvise) and isinstance(will_need, int):
            try:
                posix_fadvise(descriptor, 0, length, will_need)
            except OSError:
                return False
            return True
        return False

    def _madvise(self, descriptor: int, length: int) -> int:
        if not hasattr(mmap, "MADV_WILLNEED"):
            return 0
        try:
            return self._apply_madvise(descriptor, length)
        except (OSError, ValueError):
            return 0

    def _apply_madvise(self, descriptor: int, length: int) -> int:
        with mmap.mmap(descriptor, length, access=mmap.ACCESS_READ) as mapped:
            mapped.madvise(mmap.MADV_WILLNEED)
        return length


def prefetch_model_paths(paths: Iterable[Path], maximum_bytes: int = DEFAULT_WARMUP_BYTES) -> int:
    """Ask the OS to prefetch model pages without retaining transcripts or model data."""
    return ModelPrefetcher().prefetch(paths, maximum_bytes)
