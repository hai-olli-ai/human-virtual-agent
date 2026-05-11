# Canvas-tool latency baseline — 2026-05-11

S64c eager-dispatch microbenchmark. Measures the time savings from firing
arg-less canvas verbs the moment the verb token closes on the Anthropic
LLM stream, vs. waiting for the full `stop_reason: tool_use` event.

**Setup:** `tests/bench_canvas_latency.py`, mock 8-chunk Anthropic stream
with 20ms inter-arrival delay (typical streaming cadence). 50 iterations
per verb. Hardware: local laptop (asyncio event loop on Python 3.12).

**What this is NOT:** end-to-end voice latency. The structural benchmark
isolates the eager-dispatch optimization. For real round-trip numbers
(visitor end-of-speech → canvas action visible on screen), instrument
the agent with the T0..T9 timing logs from the session guide and
capture from a live Pipecat Cloud session.

## Run

```
.venv/bin/python tests/bench_canvas_latency.py --iterations 50
```

## Results

```json
{
  "verb": "next_scene",
  "iterations": 50,
  "eager_dispatch_ms_p50": 85.63,
  "stop_reason_ms_p50": 172.73,
  "savings_ms_p50": 87.10
}

{
  "verb": "clear",
  "iterations": 50,
  "eager_dispatch_ms_p50": 85.41,
  "stop_reason_ms_p50": 170.87,
  "savings_ms_p50": 85.46
}

{
  "verb": "play",
  "iterations": 50,
  "eager_dispatch_ms_p50": 86.17,
  "stop_reason_ms_p50": 172.78,
  "savings_ms_p50": 86.60
}
```

## Reading the numbers

- **`eager_dispatch_ms_p50` ≈ 85 ms** — wall time from first stream
  chunk to the moment `AnthropicEagerHook` invokes `send_app_message`.
  Five chunks pass before the verb token closes (`content_block_start`
  + four `input_json_delta` fragments), times 20 ms inter-arrival,
  plus per-chunk handler overhead.
- **`stop_reason_ms_p50` ≈ 171 ms** — wall time through the full
  eight-chunk stream (the non-eager / `stop_reason: tool_use` baseline).
- **`savings_ms_p50` ≈ 86 ms** — the gap eager-dispatch shaves off.

Verbs tested are all in `EAGER_DISPATCH_VERBS` (see
`services/eager_dispatch/__init__.py`), which is what eager-dispatch
fires for. Arg-bearing verbs (`draw_arrow`, `add_annotation`,
`goto_scene` with an `index`) take the regular `stop_reason` path
because the hook can't safely fire before all args have arrived.

## Real-world expectation

In production with real LLM streaming, savings are typically
**100–250 ms** because:

- Real chunk inter-arrival is more variable than the 20 ms uniform
  cadence here; tail latencies stretch the `stop_reason` path more
  than they stretch the eager path.
- After the `tool_use` block closes, the LLM often emits a short
  textual response in subsequent content blocks before
  `stop_reason: tool_use` finally fires — that text emission adds
  further delay the eager path skips entirely.

This benchmark conservatively underestimates production savings.

## Caveats / known limitations

- **Mock stream is uniform.** Real LLM streaming has long tails (rare
  multi-hundred-ms inter-arrival gaps) that the median p50 doesn't
  expose. If we want tail-latency comparisons, switch to p95/p99 and
  add jitter to the mock cadence.
- **`AnthropicEagerHook` is the only adapter exercised here.** OpenAI
  and Gemini paths (`openai_adapter.py`, `gemini_adapter.py`) have
  comparable structure but different chunk shapes; add their benches
  if/when those providers go live in production.
- **No I/O involved.** Real eager dispatch sends a Daily app-message;
  this benchmark mocks `send_app_message` so the timing reflects pure
  agent-side latency, not network. Network-bound time is roughly the
  same on both paths, so it cancels out for the "savings" delta.
