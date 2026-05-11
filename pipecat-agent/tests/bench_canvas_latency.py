"""Latency benchmark — measures end-of-speech to canvas action time across
representative scenarios. Run manually after S64c lands; compare against
pre-S64c subjective baseline.

This is a STRUCTURAL benchmark — it does not run a full Pipecat session.
It feeds controlled, mock Anthropic streaming chunks through the eager
dispatch hook and measures (a) how soon the hook fires the canvas
command (eager path) and (b) how long until the full stream completes
(stop_reason path). The delta approximates the in-the-wild latency
savings from the eager-dispatch optimization for arg-less verbs.

For real end-to-end latency, instrument the agent with structured
timing logs (T0..T9 from the session guide) and run manual voice tests
against a live room, capturing timings from production logs.

Run:
    .venv/bin/python tests/bench_canvas_latency.py --iterations 50
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from typing import List

# Bootstrap sys.path so imports work whether invoked from the project root
# (recommended) or directly via `python tests/bench_canvas_latency.py`.
# pyproject.toml's pythonpath = ["."] applies to pytest only; running the
# file as a script doesn't see it.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


async def mock_anthropic_stream(verb: str) -> List[dict]:
    """Yield Anthropic-shape chunks for a canvas_control(verb=X) call.

    The name field uses the post-rename underscore form. OpenAI and
    Anthropic both reject tool names containing '.' (regex
    ``^[a-zA-Z0-9_-]+$``), so the S64c surface ships ``canvas_control``,
    ``canvas_highlight``, etc. — see tools/canvas_protocol_tools.py.
    The eager hook's verb-completion check matches on the underscore
    form (services/eager_dispatch/__init__.py:maybe_fire_eager), so the
    benchmark must use it too or the eager path never fires.
    """
    return [
        {"type": "content_block_start", "index": 0,
         "content_block": {"type": "tool_use", "name": "canvas_control", "id": "t1", "input": {}}},
        {"type": "content_block_delta", "index": 0,
         "delta": {"type": "input_json_delta", "partial_json": '{"verb"'}},
        {"type": "content_block_delta", "index": 0,
         "delta": {"type": "input_json_delta", "partial_json": ":"}},
        {"type": "content_block_delta", "index": 0,
         "delta": {"type": "input_json_delta", "partial_json": f'"{verb}"'}},
        {"type": "content_block_delta", "index": 0,
         "delta": {"type": "input_json_delta", "partial_json": "}"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "tool_use"}},
        {"type": "message_stop"},
    ]


async def measure_eager_savings(verb: str, iterations: int = 50) -> dict:
    from services.eager_dispatch.anthropic_adapter import AnthropicEagerHook
    from tools.canvas_protocol_tools import PendingCommandRegistry

    eager_times: List[float] = []
    nonEager_times: List[float] = []

    for _ in range(iterations):
        chunks = await mock_anthropic_stream(verb)
        pending = PendingCommandRegistry()

        # Time path A: eager hook fires when verb token closes.
        sent_time: List[float | None] = [None]

        async def _send_eager(payload):
            sent_time[0] = time.perf_counter()

        hook = AnthropicEagerHook(pending, _send_eager)

        t_start = time.perf_counter()
        # Add small per-chunk inter-arrival delay simulating network/LLM speed.
        for chunk in chunks:
            await hook.on_stream_event(chunk)
            await asyncio.sleep(0.020)  # 20ms between chunks (typical streaming cadence)
        t_end = time.perf_counter()

        if sent_time[0] is not None:
            eager_times.append((sent_time[0] - t_start) * 1000)
        nonEager_times.append((t_end - t_start) * 1000)

    return {
        "verb": verb,
        "iterations": iterations,
        "eager_dispatch_ms_p50": statistics.median(eager_times) if eager_times else None,
        "stop_reason_ms_p50": statistics.median(nonEager_times),
        "savings_ms_p50": (
            statistics.median(nonEager_times) - statistics.median(eager_times)
            if eager_times else 0
        ),
    }


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--verbs", nargs="+", default=["next_scene", "clear", "play"])
    args = parser.parse_args()

    print("Eager dispatch latency benchmark (Anthropic streaming, mock chunks)")
    print(f"Iterations per verb: {args.iterations}")
    print()
    for verb in args.verbs:
        result = await measure_eager_savings(verb, args.iterations)
        print(json.dumps(result, indent=2))
        print()


if __name__ == "__main__":
    asyncio.run(main())
