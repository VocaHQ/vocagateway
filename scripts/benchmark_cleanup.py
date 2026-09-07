#!/usr/bin/env python3
"""Benchmark a local cleanup server with the production prompt and validators.

The input is JSON Lines. Every row needs ``id``, ``category``, ``input``, and
``expected`` strings. Results contain no machine or user data beyond that
explicit synthetic corpus.

Example:
    uv run scripts/benchmark_cleanup.py \
      --endpoint 127.0.0.1:8080 --api-key local-key \
      --model-id cleanup:qwen3-0.6b-q4 \
      --corpus tests/fixtures/cleanup/english_core.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import random
import statistics
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.cleanup import validation  # noqa: E402
from app.cleanup.llama_server import LlamaServerRuntime  # noqa: E402
from app.cleanup.transport import Endpoint  # noqa: E402


@dataclass(frozen=True, slots=True)
class Case:
    id: str
    category: str
    input: str
    expected: str


@dataclass(frozen=True, slots=True)
class Result:
    id: str
    category: str
    elapsed_ms: int
    output: str | None
    expected: str
    accepted: bool
    exact: bool
    error: str | None
    rejection: str | None


def load_cases(path: Path) -> list[Case]:
    cases: list[Case] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
            cases.append(Case(**value))
        except (json.JSONDecodeError, TypeError) as error:
            raise ValueError(f"{path}:{line_number}: invalid benchmark case") from error
    if not cases:
        raise ValueError(f"{path}: no benchmark cases")
    if len({case.id for case in cases}) != len(cases):
        raise ValueError(f"{path}: duplicate case id")
    return cases


def percentile(values: list[int], fraction: float) -> int:
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


async def run_case(runtime: LlamaServerRuntime, case: Case, budget_seconds: float) -> Result:
    started = time.monotonic()
    try:
        output = await runtime.clean(case.input, "en", budget_seconds=budget_seconds)
        rejected = validation.rejection(case.input, output, "en")
        return Result(
            id=case.id,
            category=case.category,
            elapsed_ms=round((time.monotonic() - started) * 1000),
            output=output,
            expected=case.expected,
            accepted=rejected is None,
            exact=output == case.expected,
            error=None,
            rejection=None if rejected is None else str(rejected),
        )
    except Exception as error:  # noqa: BLE001 - a benchmark must record every runtime failure
        return Result(
            id=case.id,
            category=case.category,
            elapsed_ms=round((time.monotonic() - started) * 1000),
            output=None,
            expected=case.expected,
            accepted=False,
            exact=False,
            error=type(error).__name__,
            rejection=None,
        )


def summarize(results: list[Result]) -> dict[str, Any]:
    elapsed = [result.elapsed_ms for result in results]
    return {
        "cases": len(results),
        "accepted": sum(result.accepted for result in results),
        "exact": sum(result.exact for result in results),
        "errors": dict(sorted(Counter(r.error for r in results if r.error).items())),
        "rejections": dict(sorted(Counter(r.rejection for r in results if r.rejection).items())),
        "latency_ms": {
            "mean": round(statistics.fmean(elapsed)),
            "p50": percentile(elapsed, 0.5),
            "p95": percentile(elapsed, 0.95),
        },
        "categories": {
            category: {
                "cases": len(group),
                "accepted": sum(result.accepted for result in group),
                "exact": sum(result.exact for result in group),
            }
            for category in sorted({result.category for result in results})
            if (group := [result for result in results if result.category == category])
        },
    }


async def benchmark(args: argparse.Namespace) -> dict[str, Any]:
    host, separator, port_text = args.endpoint.rpartition(":")
    if not separator or not host:
        raise ValueError("--endpoint must be host:port")
    runtime = LlamaServerRuntime(
        Endpoint(host, int(port_text), args.api_key),
        model_id=args.model_id,
    )
    if not await runtime.available():
        raise RuntimeError("cleanup server is unavailable")
    cases = load_cases(args.corpus)
    for case in cases[: args.warmups]:
        await run_case(runtime, case, args.timeout)
    results: list[Result] = []
    randomizer = random.Random(args.seed)
    for _ in range(args.repeats):
        pass_cases = cases.copy()
        randomizer.shuffle(pass_cases)
        for case in pass_cases:
            results.append(await run_case(runtime, case, args.timeout))
    return {
        "metadata": {
            "model_id": args.model_id,
            "endpoint": args.endpoint,
            "corpus": str(args.corpus),
            "repeats": args.repeats,
            "warmups": args.warmups,
            "timeout_seconds": args.timeout,
            "random_seed": args.seed,
            "artifact_revision": args.artifact_revision,
            "artifact_sha256": args.artifact_sha256,
            "runtime_version": args.runtime_version,
            "host_note": args.host_note,
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "summary": summarize(results),
        "results": [asdict(result) for result in results],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--api-key", default="")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--artifact-revision", default="")
    parser.add_argument("--artifact-sha256", default="")
    parser.add_argument("--runtime-version", default="")
    parser.add_argument("--host-note", default="")
    args = parser.parse_args()
    report = asyncio.run(benchmark(args))
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
