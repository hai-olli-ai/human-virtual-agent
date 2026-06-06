# Canvas Vision latency baseline (S67b, V8)

**Date:** 2026-06-04
**Goal:** visitor-utterance-end → first **vision-grounded** spoken token, on a
**warm stream** (after the first `getDisplayMedia` grant), within a ~1.5–2 s budget.

---

## ⚠️ Status: methodology + budget, NOT a live measurement

A true end-to-end warm-stream number cannot be taken yet — three prerequisites
are not in place in this environment:

1. **`GOOGLE_AI_API_KEY` is unset** → the Gemini vision call (the dominant stage)
   can't be timed; `VisionClient` stubs to `VISION_UNAVAILABLE`.
2. **Frontend B1 is not built** → no `request_canvas_capture` handler, so the
   capture round-trip times out instead of returning real grab/encode/upload time.
3. **Backend Redis ingest is not built** → `get_vision_capture` 404s; no backend
   GET to measure.

This document is therefore the **measurement plan + budget** with the two stages
that *are* measurable agent-side filled in. Replace the `BUDGET (est.)` column
with measured values once the slice is live (see "Taking the real measurement").

---

## The real latency path (longer than the naive stage list)

The requested breakdown — grab → encode → upload → Daily RT → backend GET →
Gemini → total — covers the **capture + reason** portion. But "first vision-grounded
*spoken* token" also includes two stages the naive list omits, because of how the
agent is wired (B12, Design B):

```
utterance-end
  │
  ├─ A. STT-final → LLM tool-call decision (LLM picks canvas_analyze)      [conversational LLM]
  │
  ├─ run_vision_query (services/vision_query.py):
  │     ├─ B. request_canvas_capture  ── Daily ──▶ shell                   [agent ⇄ shell]
  │     │      the shell reply (canvas_capture_result) is sent only AFTER
  │     │      it has grabbed + encoded + uploaded, so B ENCLOSES B1/B2/B3:
  │     │        B1. grant→first grab   (warm: track already live)         [shell]
  │     │        B2. grab + downscale + JPEG encode                        [shell]
  │     │        B3. upload JPEG → backend Redis ingest                    [shell → backend]
  │     │      (for 'assess', the scene-context snapshot fetch runs in
  │     │       PARALLEL with B — A-AG-5 — so it's hidden under B)
  │     ├─ C. get_vision_capture  ── HTTPS ──▶ backend Redis (delete-on-read)  [agent → backend]
  │     └─ D. vision_client.analyze_image  ── HTTPS ──▶ Gemini 3.5-flash   [agent → Gemini]
  │            NON-streamed: returns the FULL answer text
  │
  ├─ E. answer injected as a `[vision: …]` developer message; handle_analyze
  │     also dispatches the frontend `analyze` (semantic state)  ── Daily ──▶ shell  [agent ⇄ shell]
  │
  └─ F. conversational LLM re-runs with the vision message + tool result →
        first spoken token → TTS                                          [conversational LLM]
```

**Key structural finding:** because `analyze_image` is **non-streamed** and its
answer is injected for the **conversational** LLM to then speak (F), the path has
**two sequential LLM calls** (Gemini vision D + conversational TTS-LLM F) plus the
capture round-trip. The naive stage list (which stops at D) undercounts the
"first *spoken* token" latency by stages A + E + F.

---

## Stage budget (targets) + measured agent-side values

| Stage | Who measures | BUDGET (est.) | Notes |
|---|---|---|---|
| A. LLM tool-call decision | agent (TTFT to tool call) | 150–400 ms | conversational-LLM latency to emit `canvas_analyze` |
| B1. grant→first grab (warm) | **shell** | ≤ 50 ms | warm = track live, `ImageCapture.grabFrame`; no permission prompt |
| B2. grab + downscale + JPEG | **shell** | ≤ 80 ms | `canvas.toBlob`, q≈0.7, maxDim 1280 |
| B3. upload → backend ingest | **shell→backend** | 30–250 ms | network-bound; see payload sizes below |
| B. capture round-trip (encloses B1–B3) | **agent** (`wait_for`) | 250–450 ms | what the agent actually observes; ceiling `VISION_CAPTURE_TIMEOUT_MS`=4000 |
| C. backend GET (Redis, delete-on-read) | agent | 10–50 ms | small JPEG from Redis |
| D. Gemini `analyze_image` (3.5-flash, `thinking_level=low`, non-streamed) | agent | **500–1000 ms** | **dominant stage**; full short answer, not first-token |
| E. frontend `analyze` dispatch (semantic state) | agent | 80–200 ms | second Daily round-trip in `handle_analyze` |
| F. conversational LLM → first spoken token | agent (TTFT) | 250–600 ms | speaks the injected vision answer |
| **agent-side orchestration glue** | **agent (measured)** | **0.0018 ms** | `run_vision_query` minus all I/O, 2000-iter avg — negligible |
| **TOTAL (utterance-end → first spoken token)** | — | **~1.3–2.7 s** | tight against the 1.5–2 s budget; see assessment |

> The budget rows are **engineering estimates**, not measurements (no live slice /
> no Gemini key here). Only the glue overhead and the payload sizes below are measured.

### Measured: JPEG payload size (drives the B3 upload budget)

PIL proxy at `VISION_MAX_DIM=1280`, **synthetic high-frequency noise = worst case**:

| Frame | q=0.7 | q=0.6 |
|---|---|---|
| 1280×720 | 321 KB | 277 KB |
| 1280×800 | 354 KB | 307 KB |
| 960×540 | 181 KB | 157 KB |

**Real screen content compresses far better** (UI chrome, text, flat backgrounds) —
typically **~30–100 KB** at 1280 / q0.7. The worst-case figures are the upper bound
for image-heavy scenes. Either way this **confirms the backend-hop is mandatory**:
even a 30 KB JPEG is ≫ Daily's 4 KB `sendAppMessage` limit, so the bytes cannot ride
the data channel (B11 rationale).

---

## Budget assessment

- **Best case** (warm stream, fast network, low conversational-LLM latency):
  A 150 + B 250 + C 20 + D 550 + E 80 + F 280 ≈ **1.33 s** — within budget.
- **Typical/worst** (image-heavy scene, slower network, higher LLM TTFT):
  A 350 + B 450 + C 40 + D 900 + E 180 + F 500 ≈ **2.42 s** — over the 2 s budget.

The two **LLM** stages (D Gemini + F conversational) are ~60–70% of the budget and
are the hardest to cut. The capture round-trip (B) is the next lever and is
shell/network-bound.

---

## Tuning knobs (if over budget)

Agent-side, all advisory to the shell except where noted:

1. **`VISION_MAX_DIM`** (default 1280) → drop to **960**: ~halves the JPEG payload
   (321 KB → 181 KB worst-case) and the B3 upload time, at some loss of fine
   detail. Sent as `maxDim` in `request_canvas_capture`; the shell must honor it.
2. **JPEG quality** (shell-owned, advisory `VISION_JPEG_QUALITY`≈0.7) → 0.6:
   ~15% smaller (321 → 277 KB) with minor quality loss.
3. **`thinking_level`** is already `"low"` (the tuned low-latency setting on
   3.5-flash). `"minimal"` exists if D must be cut further, at some reasoning cost.
4. **Architectural (biggest lever, not yet done):** stream the Gemini answer
   directly to TTS and **skip the second conversational-LLM turn (F)** for pure
   visual questions. Removes ~250–600 ms but loses the conversational LLM's
   persona/voice consistency over the answer — a real tradeoff to weigh in a V8
   follow-up. Today's design favors persona consistency (D→inject→F).

---

## Taking the real measurement (when the slice is live)

1. **Prereqs:** frontend B1 deployed, backend Redis ingest deployed,
   `GOOGLE_AI_API_KEY` set, a real Daily room, and a **warm** screen-share
   (do one throwaway capture first to clear the grant prompt).
2. **Agent-observable stages (B, C, D, glue)** — wrap timers in
   `run_vision_query` / `request_canvas_capture` (e.g. `time.perf_counter()`
   around the `wait_for`, the `get_vision_capture`, and the `analyze_image`),
   and log a `[perf-vision]` line mirroring the existing `[perf-refresh]` (S66).
3. **Shell sub-stages (B1, B2, B3)** — the agent can't see these; have the shell
   stamp `grabMs` / `encodeMs` / `uploadMs` into the `canvas_capture_result`
   payload (it's already a non-canvas message; add three numeric fields) so the
   agent can log the full breakdown.
4. **D in isolation** (no full slice needed, just a key) — time a warm
   `gemini-3.5-flash` call directly:
   ```bash
   GOOGLE_AI_API_KEY=… .venv/bin/python -c "
   import asyncio, time
   from services.vision_client import VisionClient
   vc = VisionClient()
   jpeg = open('sample_capture.jpg','rb').read()
   async def m():
       await vc.analyze_image(jpeg, 'describe')          # warm the client
       t = time.perf_counter()
       await vc.analyze_image(jpeg, 'point', 'ctx')
       print('gemini_analyze_ms:', round((time.perf_counter()-t)*1000))
   asyncio.run(m())"
   ```
5. Record the measured table here, confirm warm total ≤ ~2 s, and if not, apply
   the tuning knobs above (drop `VISION_MAX_DIM` first, then weigh the
   skip-second-LLM-turn option).

---

## Gate (this session)

`ruff check . && uv run pytest -q` → **All checks passed! · 200 passed**.
(B16 added a `[tool.ruff]` per-file-ignore for bot.py's intentional `E402`
load-dotenv-before-config pattern, and a one-time cleanup of 7 pre-existing
unused imports while adopting the linter.)
