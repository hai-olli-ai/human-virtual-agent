# pipecat-agent — CLAUDE.md

> Voice agent for **Human Virtual** (hv.ai), built on the **Pipecat Framework**.
> **Status:** S64c–e + S65 + S65b + S65c + S66 shipped; **S67a shipped frontend-only (NO agent change); S67b (agent vision of the annotated canvas) complete agent-side — `services/vision_client.py` (dedicated Gemini, decoupled from `LLM_CANVAS_PROVIDER`) + `services/vision_query.py` (live-capture-first — Pillow only for a *repeated* `describe` while screen-share is off) + the `request_canvas_capture`/`canvas_capture_result` round-trip + permission fast-path + two-stage `describe` nudge + the deterministic `vision_state` indicator + PLAYBOOK/tool-description broadening. End-to-end still needs three cross-repo pieces: the frontend capture handler (B1), the frontend `vision_state` indicator, and the backend Redis ingest.** The classic TTS service is a `CachedFirstTTSService` (cache hit → raw PCM in the identical `TTSStartedFrame → TTSAudioRawFrame → TTSStoppedFrame` envelope so `BotStoppedSpeakingFrame` still fires; miss → live Cartesia). `generate_quiz_from_knowledge`'s core is the factored `run_quiz_generation` coroutine (shared by the LLM tool + the manual `request_quiz` trigger); inbound `request_narrate` / `request_quiz` branches sit alongside the canvas dispatch in `on_app_message`; `script_complete` carries `trigger: 'auto' | 'manual'`. **S66** made scene-change refresh fast: **lazy vision** (`VISION_REFRESH_MODE=lazy` default — render the Pillow frame only on demand), **flow-knowledge reuse** across scene changes, and **by-`sceneId` snapshot fetch** off `canvas.sceneChanged` (cursor-independent; cursor fallback retained).
> **Repo:** `pipecat-agent/` (alongside `human-virtual-backend/` and `human-virtual-frontend/`).

---

## What this service does

The Pipecat agent runs the avatar's **conversation pipeline**: STT → LLM → TTS → audio out, with optional vision and canvas tool calls. It connects to a Live Room over WebRTC and:

1. Listens to the visitor's microphone (Deepgram STT).
2. Calls an LLM (OpenAI / Anthropic / Gemini, selected at boot via `LLM_CANVAS_PROVIDER`) with a system prompt assembled from the Live Room's persona, knowledge (aggregated across the whole flow), current scene instruction, current scene's elements (referenced by short aliases), and the active Canvas Page's manifest.
3. Emits TTS audio (Cartesia, now via `CachedFirstTTSService`) plus avatar lip-sync data (SoulX-Flashtalk for "talking" display mode).
4. Calls **canvas tools** via the generic 5-tool protocol surface (`canvas_analyze`, `canvas_highlight`, `canvas_control`, `canvas_action`, `canvas_set_page` — **underscored, not dotted**).
5. (S46+) Calls **vision** by adding a snapshot image to the LLM's context.
6. (S49 + S65) Plays the scene's script on entry. **S65** added per-segment script-avatar voice switching, a localized post-script invitation, and `{type:'script_complete'}` carrying `sceneIndex`/`hadScript`. **S65b** added the cached-audio fast path through the same narrator without changing the control flow.

---

## Stack

| Layer | Choice |
|---|---|
| Framework | Pipecat (Python, pinned at 0.0.108 in `pyproject.toml`) |
| Transport (prod) | DailyTransport (Daily.co WebRTC) |
| Transport (local) | SmallWebRTCTransport |
| Deployment (prod) | Pipecat Cloud |
| LLM default | OpenAI GPT (via `LLM_CANVAS_PROVIDER=openai`) |
| LLM alt | Anthropic Claude, Google Gemini (extras must be installed) |
| LLM selection | `LLM_CANVAS_PROVIDER` env var, validated at boot |
| STT | Deepgram |
| TTS (classic) | **`CachedFirstTTSService`** *(S65b)* — composes `CartesiaTTSService`; cache-first, live-fallback |
| Lip-sync (talking) | SoulX-Flashtalk (S48 — relay pipeline; **S65b cache does NOT apply** — relay renders its own audio) |

---

## Project structure

```
pipecat-agent/
  bot.py                            # Main entrypoint — pipeline assembly, transport, LLM provider branching;
                                    #   constructs CachedFirstTTSService (S65b) and wires it into both pipelines
  config.py                         # Settings, provider validation, model defaults;
                                    #   S65b NARRATION_* paired constants (must match backend)
  persona.py                        # build_system_prompt — Strategy 1 path + S65 SCRIPT prompt directive
  scene_context.py                  # Section helpers, knowledge formatting, alias generator
  api_client.py                     # HTTP client for the backend (snapshot, persona-prompt, navigate, scene image, generate-quiz, …)
  narration.py                      # S65 — narrate_scene_script: per-segment voice switch + invitation + script_complete
                                    #   S65b — _prefetch_cached_audio + per-segment prime_cached(...) before each speak
  context/
    prompt_builder.py               # render_canvas_page_section
    canvas_manifest.py              # CanvasManifestRegistry
  tools/
    canvas_protocol_tools.py        # 5 generic protocol tools + dispatch_canvas_command; SCENE_NAV_VERBS exemption
    quiz_generation.py              # S64e — generate_quiz_from_knowledge tool + bundled set_page
  services/
    cached_first_tts.py             # S65b — CachedFirstTTSService
    vision_client.py                # S67b — dedicated Gemini multimodal client + derive_vision_mode (decoupled from LLM_CANVAS_PROVIDER)
    vision_query.py                 # S67b — capture-first/Pillow-fallback orchestration core (run_vision_query, injected deps; testable)
    eager_dispatch/                 # Per-provider streaming hooks (S64c) — constructed, not wired into the streaming loop
      __init__.py
      anthropic_adapter.py
      openai_adapter.py
      gemini_adapter.py
  tests/
    test_canvas_highlight_validation.py
    test_eager_dispatch.py
    test_link_narration_directive.py
    test_quiz_generation.py
    test_scene_context_knowledge.py
    test_scene_context_s61.py
    test_scene_narration.py         # S65
    test_cached_first_tts.py        # S65b
    test_submit_answer_bundling.py  # S64e
    bench_canvas_latency.py
  docs/
    benchmarks/
      canvas_latency_2026-05-11.md
      narration_cache_<date>.md     # S65b baseline / cache-hit comparison (when measured)
  README.md
  Dockerfile                        # Explicit-allowlist COPY; new top-level dirs MUST be added
```

The Dockerfile uses explicit `COPY <file> .` lines per top-level file/directory. Adding new code directories requires a matching `COPY <dir> <dir>` line, or the Pipecat Cloud container hits `ModuleNotFoundError` at session start. The new `services/cached_first_tts.py` is reached via the existing `COPY services services` line (no new top-level dir).

---

## Pipecat pipeline (high level)

**Classic pipeline** (display_mode ∈ {normal, invisible, 3dgs}):

```
DailyTransport.input  →  Deepgram STT
                      →  TranscriptForwarder
                      →  LLMContextAggregatorPair (user side)
                      →  LLM (the 5 canvas_* tools registered)
                      →  ThinkingNotifier / TranscriptForwarder / SpeakingStateNotifier
                      →  CachedFirstTTSService (was CartesiaTTSService)   ← S65b
                      →  DailyTransport.output  ← emits BotStartedSpeakingFrame / BotStoppedSpeakingFrame
                      →  LLMContextAggregatorPair (assistant side)
```

**Relay pipeline** (display_mode == 'talking') — no local TTS; text is forwarded to the SoulX-Flashtalk avatar bot via Daily app-messages on the `avatar-relay.v1` protocol. **S65b caching does NOT apply here** — SoulX renders its own audio, the `CachedFirstTTSService` isn't in the relay pipeline. Narration still happens (text forwarded; `script_complete` emitted identically), per S65 D4.

Pipeline selection is automatic based on the snapshot's `avatar_display_mode`. Falls back to `CLOUD_OUTPUT_MODE` env var when the snapshot can't be fetched.

When the LLM emits a tool call, Pipecat's function-call aggregator invokes the registered handler in `tools/canvas_protocol_tools.py`. The handler builds a wire-format `canvas.command` payload, sends it as a Daily app-message, and awaits the matching `canvas.commandResult` via an `asyncio.Future` (managed by `PendingCommandRegistry`).

---

## System prompt — assembly path

The current live prompt is assembled as a **concatenation**:

```
system_prompt = build_system_prompt(room_id, …)             # persona.py — Strategy 1
              + "\n\n"
              + render_canvas_page_section(manifest)        # context/prompt_builder.py
```

Rebuilt at three points: **session start**, **scene change** (via `refresh_agent_for_current_scene`), and **manifest registration** (`canvas.register` branch in `on_app_message`).

### S65 — SCRIPT section directive

S65 added a short SCRIPT directive telling the LLM: *the scene's script is narrated automatically by the system in the script avatar's voice; do not read or paraphrase it; stay silent until the visitor speaks.* The invitation owns the conversational hand-off.

Strategy 1 vs Strategy 2 (in `persona.build_system_prompt`):

- **Strategy 1** (live path): when `room_id` is set, the agent fetches the backend's `/persona-prompt` endpoint and appends audience / knowledge / link narration / canvas tools section / **SCRIPT directive (S65)** / scripts.
- **Strategy 2** (fallback / tests): builds locally from avatar + scene via `build_scene_description`.

LANGUAGE on both ends. PERSONA, AUDIENCE, KNOWLEDGE between. The S64c CANVAS PAGE section sits **outside** the language sandwich (post-hoc append). The S65 SCRIPT directive sits inside the body, before the closing LANGUAGE reminder.

---

## Tool naming — underscored, not dotted

The 5 generic protocol tools register with the LLM as:

| LLM-facing name | Wire-format `tool` field |
|---|---|
| `canvas_analyze` | `analyze` |
| `canvas_highlight` | `highlight` |
| `canvas_control` | `control` |
| `canvas_action` | `action` |
| `canvas_set_page` | `set_page` |

OpenAI and Anthropic both validate tool names against `^[a-zA-Z0-9_-]+$` — `.` is rejected. Daily wire-format message `type` fields keep their dots (`canvas.command`, `canvas.sceneChanged`, etc.) — those don't go to the LLM.

---

## Canvas tools — the 5 generic protocol tools + `generate_quiz_from_knowledge`

`tools/canvas_protocol_tools.py` registers exactly 5 generic canvas tools. `tools/quiz_generation.py` adds a sixth non-canvas tool — `generate_quiz_from_knowledge` — bundled with `set_page` internally.

Each canvas handler logs entry, translates aliases to UUIDs, validates against the active Page's manifest (with the `SCENE_NAV_VERBS` exemption), builds the wire payload with a fresh `commandId`, registers a pending future, sends the Daily app-message, awaits the future (6 s default timeout), and returns to the LLM. Errors are returned as tool results (`{error, message, details}`), not raised — a single failed call doesn't break the turn.

**`canvas_action` verb arg shapes**, **`canvas_set_page` allowlist** (`{composition, youtube, quiz}`), the **all-or-nothing override semantics** (empty `pageInit` = restore snapshot for both pageType and init), and the **scene-nav exemption** (`next_scene`/`previous_scene`/`goto_scene` bypass the per-Page manifest validator because they're shell-level) are all unchanged from S64e. See the Quiz section below for the bundling rationale.

---

## Element alias layer (post-S64c)

Backend snapshots return element ids as UUID7 strings. UUID7s share long prefixes (timestamp-ordered) and LLMs fail to copy them verbatim mid-stream.

The agent generates short deterministic aliases per snapshot (`text_1`, `avatar_1`, `emoji_1`, `button_1`, …). The LLM uses aliases in tool calls; `tools/canvas_protocol_tools.py`'s `_resolve_element_id` translates back to the real UUID. Unknown values pass through so a typo fails loudly at the frontend, not silently.

`build_scene_description` and `_summarize_element` render `emoji_character` in element lines so the LLM has semantic content for emojis.

---

## Quiz Canvas Page (S64e)

`tools/quiz_generation.py` on the agent + `public/canvas-pages/quiz/` on the frontend. End-to-end shape from two hard-learned principles:

1. **LLMs don't reliably copy large structured blobs between tool calls.** Fix: bundle the side-effect dispatch into the data-producing tool. `generate_quiz_from_knowledge` generates the blob AND dispatches `canvas.set_page(quiz, blob)` in one handler — the LLM only ever calls one tool.
2. **LLMs aren't good at pacing UI animations.** Fix: the Quiz Page owns post-answer pacing (cancellable `REVEAL_EXPLANATION_DELAY_MS` / `AUTO_ADVANCE_DELAY_MS` timers); the agent's `submit_answer` / `skip_question` are pass-through.

`SessionContext` carries the live-room slug + current scene id. `bot.py` populates slug from the runner-args body at session start and refreshes `current_scene_id` on every `canvas.sceneChanged`.

The agent's PLAYBOOK forbids the LLM from calling `canvas_set_page(pageType='quiz', …)` directly — quiz activation only via `generate_quiz_from_knowledge`. `skip_question` ("I don't know") is a separate action verb; `submit_answer` reply is `{ choice, correct, completed }`, `skip_question` reply is `{ skipped, correct: false, completed }`.

**S65c factoring:** the quiz-generation *core* was extracted into a plain coroutine `run_quiz_generation(api_client, session_context, canvas_ctx, count, language, on_state)` in `tools/quiz_generation.py`. Two callers share it: the LLM tool handler (`make_handle_generate_quiz` wraps it and returns the blob as a tool result — **return shape unchanged from S64e**, guarded by `test_quiz_factor.py`) and the manual `request_quiz` button trigger (passes an `on_state` hook that emits `quiz_generation_state` Daily messages). The bundled `canvas.set_page(quiz, blob)` dispatch (Option D) lives inside the core, so both entry points behave identically.

---

## Live Room Narration (S65)

S65 turns a scripted flow into a self-driving guided walk-through. `narration.py` owns the playback path:

- **`narrate_scene_script(...)`** runs on scene entry (session start + after every `canvas.sceneChanged` refresh). Idempotent per scene entry — guarded against re-narrating on `canvas.register`-only prompt rebuilds. **(S65c)** the entrypoint takes `force: bool = False` and `trigger: str = "auto"`; `force=True` bypasses the once-per-entry guard (used by the manual Script button so a replay always re-narrates), and `trigger` is passed through to the `script_complete` payload.
- **Classic pipeline:** for each segment in `current_scene.scripts[*]`, set the Cartesia voice to the segment's resolved `voice_id` (fallback to primary on `is_fallback`), speak via the TTS service, await TTS-stopped; reset to primary after the last segment; speak the localized `invitation_line` in the primary voice (suppressed on non-final auto-advance scenes per D8); emit `{type:'script_complete', sceneIndex, hadScript, trigger}`.
- **Relay pipeline:** forward script text via the `avatar-relay.v1` path (primary SoulX voice; per-script-avatar voice is a v0.2 punt per D4); emit the same `script_complete`.

The SCRIPT prompt directive (above) keeps the LLM silent during narration.

`refresh_agent_for_current_scene` reads the snapshot's new `current_scene.scripts`, `current_scene.has_script`, `current_scene.narration.*`, and `live_room.auto_advance` to drive the narrator. The agent does **NOT** call `api_navigate` from narration — auto-advance is shell-owned (S64e lesson). The agent's only signal is `script_complete`.

### Auto-advance signalling (D8)

If `live_room.auto_advance && scene_index < total - 1`: optionally speak `transition_cue`, **skip** the invitation, emit `script_complete`. Else: speak `invitation_line`, emit `script_complete`. No-script scenes: neither branch (the existing conversational greeting stands; the shell knows `hadScript=false` and won't schedule an advance). **(S65c)** entry narration emits `trigger:'auto'`; a manual Script-button replay emits `trigger:'manual'` and the shell ignores `'manual'` for auto-advance — so a replay never moves the flow.

### Tests

`test_scene_narration.py`: voice resolution picks clone vs fallback; invitation suppressed on non-final auto-advance scene, present on final / non-auto-advance; narration runs once per entry; relay path emits `script_complete`.

---

## Cached narration audio (S65b) — `CachedFirstTTSService`

S65b makes scene-start narration **instant and free** for clone-voiced segments. The agent fetches pre-rendered PCM from R2 (public CDN URL via `media.hv.ai`) and feeds it through the existing TTS frame envelope. Live synthesis is the automatic fallback on miss, pending, error, fallback-voiced segments, or fetch failure.

### `services/cached_first_tts.py`

```python
class CachedFirstTTSService(CartesiaTTSService):
    """If a cached segment has been primed for the NEXT run_tts call, play it from bytes
    in the canonical TTS frame envelope; otherwise fall through to live Cartesia synthesis."""

    def prime_cached(self, segment: Optional[CachedSegment]) -> None: ...
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        seg = self._consume()
        if seg is None:
            async for frame in super().run_tts(text):    # live path: voice already set by narrator
                yield frame
            return
        # cache hit — emit IDENTICAL envelope CartesiaTTSService would
        yield TTSStartedFrame()
        for chunk in _chunk_pcm(seg.pcm, seg.sample_rate, seg.num_channels, ms=20):
            yield TTSAudioRawFrame(audio=chunk, sample_rate=seg.sample_rate, num_channels=seg.num_channels)
        yield TTSStoppedFrame()
```

Wired into `bot.py`'s classic pipeline in place of `CartesiaTTSService`. Same constructor kwargs; the cache-first behavior is opt-in via `prime_cached(...)` — when nothing is primed, the service is indistinguishable from the parent.

### Narrator prefetch + prime

`narration.py` at scene-narration start fetches every `ready` segment's PCM bytes in one `aiohttp` gather, stashes them by segment id, then **primes the service immediately before each `speak_segment` call** with no awaits in between:

```python
cached = await _prefetch_cached(current_scene["scripts"])     # {segment_id: CachedSegment | None}
for seg in current_scene["scripts"]:
    hit = cached.get(seg["id"])
    if hit is None:
        set_cartesia_voice(tts, seg["voice_id"])              # MISS keeps S65's per-segment voice switch
    tts.prime_cached(hit)                                     # None on miss
    await speak_segment(tts, seg["text"])
# reset to primary voice, speak invitation (live), emit script_complete — unchanged from S65
```

> **On a HIT the voice is baked into the bytes — do not call `set_cartesia_voice` on the hit path.** Doing so wouldn't break anything (the cached bytes win), but it's wasted Cartesia control traffic and noise in logs.

### Frame-envelope discipline (the gate-fix anchor)

`CachedFirstTTSService` emits the **exact** `TTSStartedFrame → TTSAudioRawFrame(s) → TTSStoppedFrame` sequence that `CartesiaTTSService` emits. **Never `OutputAudioRawFrame`.** Why this matters:

- The output transport's speaking-state machine drives `BotStartedSpeakingFrame` / `BotStoppedSpeakingFrame` off the TTS frame stream, regardless of who produced it. Yielding the canonical envelope means cached playback emits `BotStoppedSpeakingFrame` on **playback** completion, exactly as live playback does.
- Direct `OutputAudioRawFrame` emission (rejected alternative) **bypasses** that machine — `BotStoppedSpeakingFrame` would never fire, and the in-flight `NarrationCompletionGate` fix (which re-keys on `BotStoppedSpeakingFrame` to stop clipping audio during auto-advance) would break for cached scenes.
- `test_cached_first_tts.py::test_hit_never_emits_output_audio_raw_frame` guards this invariant. **Do not relax that test.**

### Cross-repo audio invariant (READ BEFORE TOUCHING)

The agent's `config.py` and the backend's `app/config.py` carry **paired** constants. Mismatch = garbled cached playback that "works" locally but breaks where configs drift. Paired keys:

- `NARRATION_TTS_MODEL_ID` (e.g. `sonic-2`).
- `NARRATION_AUDIO_ENCODING` (`pcm_s16le`).
- `NARRATION_AUDIO_SAMPLE_RATE` (e.g. `24000` Hz).
- `NARRATION_AUDIO_NUM_CHANNELS` (`1`).

The `CartesiaTTSService` constructor in `bot.py` is configured with `sample_rate=NARRATION_AUDIO_SAMPLE_RATE` and `model=NARRATION_TTS_MODEL_ID`. The backend's Celery `generate_narration_audio` renders with those exact values. Change one repo without the other and cached `TTSAudioRawFrame`s will play at the wrong rate (chipmunk / slow drone).

There's no automated test that asserts the two repos agree — the gate is convention + the agent's `test_cached_first_tts.py` assertion that `TTSAudioRawFrame.sample_rate` equals `settings.NARRATION_AUDIO_SAMPLE_RATE`.

### Tests

`test_cached_first_tts.py`: HIT emits started/audio/stopped envelope; HIT **never** emits `OutputAudioRawFrame`; MISS delegates to `super().run_tts`; `prime_cached` is consumed once (a stray second `run_tts` after one prime is a miss); chunking aligns to sample boundaries (no clicks); cached PCM carries the configured sample rate.

### What's NOT changed in S65b on the agent side

- Voice resolution logic, S65's per-segment voice switching, the invitation/`transition_cue` flow, and `script_complete` payload shape are all unchanged.
- The relay (`talking`) pipeline. SoulX renders its own audio — caching doesn't apply.
- The LLM provider branching, eager-dispatch infrastructure, and canvas tools.
- Every existing S65 test continues to pass — the cache path is invisible to its assertions (miss path = identical behavior).

---

## LLM provider selection

`LLM_CANVAS_PROVIDER` env var selects the main LLM at boot. `bot.py:_build_llm_and_eager_hook` branches with **lazy imports** so only the selected provider's SDK needs to be installed. Read once at boot and fixed for the session — no mid-session switching.

---

## Eager streaming dispatch — instantiated, not yet wired

Per-provider adapters in `services/eager_dispatch/` peek at LLM streaming events. When the LLM finishes streaming the verb token for an arg-less verb, the adapter would fire the canvas command without waiting for `stop_reason: tool_use`.

```python
EAGER_DISPATCH_VERBS = frozenset({
    "next_scene", "previous_scene", "clear",
    "next_question", "previous_question", "restart",
    "play", "pause",
})
```

**Current limitation:** the hook objects are constructed in `_build_llm_and_eager_hook` but **not yet called by Pipecat's streaming loop**. Result: 0 ms savings in production today. The infrastructure is ready when someone wires `await eager_hook.on_stream_event(chunk)` into the Pipecat LLM service subclass. (Tracked for S74.)

**Double-dispatch safety:** `PendingCommandRegistry.is_eager(commandId)`. When `stop_reason` arrives and the regular handler runs, it checks this flag and awaits the existing future without re-sending.

---

## Scene-change refresh — single path (post-S64c, S65-extended, S66-optimized)

Both voice-initiated and visitor rail-click scene navigation flow through ONE refresh path:

1. Frontend's `navigateToIndex` is the canonical scene-change function.
2. After `setSnapshot(snap)`, `navigateToIndex` emits `{type:'canvas.sceneChanged', sceneIndex, sceneId}` via the relay's `broadcastSceneChanged` — **`sceneId` added in S66**.
3. Agent's `on_app_message` `canvas.sceneChanged` branch awaits `refresh_agent_for_current_scene()`:
   - **(S66) Fetches the snapshot by `sceneId`** off the cursor-independent endpoint (`get_scene_snapshot(room_id, scene_id=sceneId)`) — no cursor race, and the backend serves it from the Redis cache. **Falls back to the cursor-based fetch when `sceneId` is absent** — correctness never depends on the pushed field.
   - `build_system_prompt(…)` rebuilds the base + SCRIPT directive (S65), but **(S66) reuses the cached flow-knowledge section** when the snapshot's knowledge version is unchanged (only the per-scene bits — instruction, elements, aliases, link, scripts — are re-stitched).
   - Re-appends `render_canvas_page_section(canvas_manifest.current())`.
   - Sets `llm._settings.system_instruction`.
   - **(S66) Does NOT render the vision frame eagerly** in `VISION_REFRESH_MODE=lazy` (default) — marks it stale; `_ensure_vision_frame()` fetches on demand (visual question / first `canvas_analyze`). `eager` restores per-change rendering.
   - **Runs `narrate_scene_script` for the new scene** (S65) — idempotent per entry; cached segments served via `CachedFirstTTSService` prime (S65b); fallback to live for misses.

**The agent does NOT call `api_navigate` from `refresh_agent_for_current_scene`.** The shell already advanced the backend cursor; calling it again double-steps. (The S66 by-id fetch is also cursor-independent, so it can't accidentally move the cursor either.)

**Agent-side `SCENE_NAV_VERBS` exemption.** `{"next_scene", "previous_scene", "goto_scene"}` — `handle_control` skips the per-Page manifest verb check for these (kept in sync with the relay-side constant by convention).

**Timing note for voice nav:** `canvas.sceneChanged` is emitted BEFORE `canvas.commandResult` (same JS frame) so the agent's prompt refresh completes before the tool future resolves.

---

## Daily app-messages

**Outgoing (agent → frontend):**

- `{type: 'canvas.command', commandId, tool, verb?, args}` — canvas tool dispatch.
- `{type: 'transcript', speaker, text}` — STT or avatar text.
- `{type: 'speaking_state', isSpeaking}`, `{type: 'llm_thinking', thinking}` — UI cues.
- **`{type: 'script_complete', sceneIndex, hadScript, trigger}`** *(S65; `trigger` added S65c)* — emitted after the scene's scripts finish; the shell's auto-advance handler advances only when `trigger === 'auto'`.
- **`{type: 'quiz_generation_state', state, error?}`** *(S65c)* — `state ∈ {generating, ready, error}`. Emitted by the manual `request_quiz` path (via `run_quiz_generation`'s `on_state` hook) so the shell's Quiz button can show a spinner / error. The LLM-tool quiz path does **not** emit these (no button to update).
- **`{type: 'request_canvas_capture', captureId, hint, maxDim}`** *(S67b)* — asks the visitor's shell to screenshot the live canvas (scene + S67a annotations). `maxDim` is advisory (`VISION_MAX_DIM`); the shell owns the encode. The image does **not** ride Daily (4 KB limit) — the shell uploads JPEG bytes to the backend Redis ingest and replies with `canvas_capture_result` carrying only metadata.
- **`{type: 'vision_state', state}`** *(S67b)* — `state ∈ {analyzing, idle}`. **Deterministic** signal bracketing the Gemini `analyze_image` call (emitted via `run_vision_query`'s `on_vision_state` hook), so the shell can show a "looking at your screen…" indicator **only while the model is actually running** — `analyzing` before, `idle` after (try/finally, always clears). The fast nudge / retry paths (no Gemini call) **never** emit it, so the indicator can't flash without a real analyze. Session-level (handle in the shell's general handler, like `quiz_generation_state` — not `DailyRelay`).
- `{type: 'avatar_relay.*', …}` (relay pipeline only) — text/turn protocol for SoulX.

**Incoming (frontend → agent):**

- `{type: 'canvas.register', pageType, version, capabilities, semanticState}` → `CanvasManifestRegistry.set_manifest()`; rebuilds CANVAS PAGE prompt section.
- `{type: 'canvas.stateChange', semanticState}` → cached in `CanvasManifestRegistry` for the next `analyze()`.
- `{type: 'canvas.sceneChanged', sceneIndex, sceneId}` → `refresh_agent_for_current_scene()`. **(S66)** `sceneId` drives the cursor-independent by-id snapshot fetch; cursor fallback if absent.
- `{type: 'canvas.commandResult', commandId, result}` / `{type: 'canvas.commandError', commandId, error}` → complete the awaiting Future.
- **`{type: 'request_narrate'}`** *(S65c)* → `narrate_scene_script(force=True, trigger='manual')` — re-narrates the current scene's script without auto-advancing.
- **`{type: 'request_quiz', count?, language?}`** *(S65c)* → `run_quiz_generation(...)` with the `quiz_generation_state` emitter. Silently ignored if the agent isn't ready yet (no `session_context`).
- **`{type: 'canvas_capture_result', captureId, status, w?, h?, error?}`** *(S67b)* → resolves the `asyncio.Future` opened by `request_canvas_capture`, keyed by `captureId` in the sibling `_pending_captures` dict (NOT a canvas commandId). `status ∈ {ready, error, permission_denied}`. Carries metadata only — the JPEG bytes are fetched separately via `api_client.get_vision_capture(slug, captureId)`.

**Routing rule (S65c + S67b, important):** the `request_narrate` / `request_quiz` / `canvas_capture_result` branches are **early-return branches in `on_app_message`, alongside but BEFORE the `canvas.*` dispatch** — never inside the canvas/relay path. They're session-level messages, not canvas commands (mirrors the frontend keeping them out of `DailyRelay`). They piggyback on the existing defensive `json.loads` (below).

Defensive JSON parsing in `on_app_message`: if Daily delivers the payload as a string (varies by SDK version), parse before the `isinstance(dict)` check. *(S64d hardening — still in place.)*

---

## Live Room snapshot consumer

The agent fetches `GET /live-rooms/{room_id}/scene-snapshot` on session start and on every scene navigation. **`api_client.get_scene_snapshot` always passes `?include_all_scene_knowledge=true`** — gives the agent a stable flow-knowledge block across navigations. **(S66)** on scene change it also passes **`?scene_id={sceneId}`** (from the `canvas.sceneChanged` payload) so the fetch is cursor-independent and served from the backend's Redis cache; it falls back to the cursor-based fetch when `sceneId` is absent.

The snapshot includes:

- `live_room`: language, persona, recipient_prompt, **`auto_advance`** *(S65)*.
- `current_scene`: name, instruction, display_mode, background_url, **elements (with `id`)**, link_url, link_source, `canvas_page_type`, **`has_script`** *(S65)*, **`narration.{invitation_line, transition_cue}`** *(S65)*, **`scripts[*]`** *(S65 — with resolved per-segment voice + S65b `audio` sub-object)*, **`actions`** *(S65c — informational for the agent; the shell consumes it)*.
- `flow_state`: scene_index, total_scenes, scene_ids array.
- `knowledge`: text content (flow scope = aggregated all-scene set) **+ a version token (S66)** the agent uses to skip re-stitching the flow-knowledge prompt section when unchanged.
- `faqs`: per-scope array.

**(S66) Flow-knowledge reuse.** The flow-knowledge block is invariant across scenes in a flow, so the agent holds the assembled prompt section in-session keyed by the snapshot's knowledge version and re-stitches only when it changes. On scene change, only the per-scene bits (instruction, elements, aliases, link, scripts) are rebuilt.

**Don't duplicate snapshot logic locally.** Extend the snapshot endpoint in the backend rather than fetching multiple endpoints. `LiveRoomService.get_scene_snapshot()` is the single source of truth, shared with composition / youtube / quiz Pages.

---

## Vision (S46 → S67b)

When the visitor asks about the screen or refers to something they've drawn / pointed at / written, `canvas_analyze` routes through `_ensure_vision_for_active_scene(question)` → `services/vision_query.run_vision_query(...)` (S67b). It reasons over what the visitor actually sees with the **dedicated Gemini `vision_client`** (decoupled from `LLM_CANVAS_PROVIDER`) and injects the result as a `[vision: …]` developer message. The conversational LLM no longer does the pixel reasoning — it receives `vision_client`'s **text**. Mode (`point | assess | describe`) is derived from the utterance; `assess` grounds on the scene's expected answer, fetched in parallel with the capture (A-AG-5).

**Live-capture-first — Pillow is NEVER used while screen-share is on.** The capture `status` *is* the per-request permission signal (`ready` = screen-share on; a permission error = off), so `run_vision_query` decides without any session flag:

- **`ready` + bytes** → reason over the **live capture** (annotations and all). The only path while screen-share is on; Pillow is never substituted. A fast transient miss (`ready` but bytes gone, or a non-permission error) **retries the capture once**; a timeout (already waited the full budget) does not.
- **Live capture fails with screen-share on** (timeout, or both attempts missed) → a **`_CAPTURE_RETRY_NOTE`** ("couldn't grab your screen, try again") — **never** the annotation-less Pillow scene.
- **Permission denied** (screen-share off) is the **only** place Pillow lives. `point`/`assess` → `_SHARE_SCREEN_NOTE` (ask to click the **"Let the Assistant see your screen"** button; never Pillow). `describe` is **two-stage**: the **first** describe-while-off → `_DESCRIBE_SHARE_NUDGE_NOTE` (ask to click the same button, **no Pillow yet**); a **repeated** describe while still off → the base-scene **Pillow PNG** (`_fetch_pillow`) → `vision_client` + blind-spot note. The describe-nudge flag lives on `SessionContext.describe_share_nudged` (set on the first nudge) and **resets whenever a capture comes back `ready`** (screen-share back on), so each off-period asks once before falling back. `run_vision_query` takes `session_context` (and derives slug + scene_id from it).

`get_vision_image` (the old "live-or-Pillow" combiner) was split into `_fetch_live_bytes` (live only, never falls back) and `_fetch_pillow` (base scene only).

**`VISION_REFRESH_MODE=lazy|eager` (default `lazy`, S66).** In `lazy` the Pillow frame isn't rendered on every scene change — `refresh_agent_for_current_scene` marks it stale. **Exception:** `eager` still pre-loads a Pillow image into the *main* LLM context on scene change (vestigial after S67b; the default `lazy` does not, and S67b's per-question vision path makes it redundant).

> **The S66 `VisionFrameTracker` is now write-only.** `refresh_agent_for_current_scene` still writes it (eager pre-load / lazy invalidate), but the vision path no longer reads it — S67b captures fresh per question because annotations change *within* a scene (A-AG-3). `vision_refresh.ensure_vision_frame_for_scene` has no runtime caller (tests only). Both the tracker and that function are removable in a future cleanup.

---

## Local development

```bash
python -m venv .venv && source .venv/bin/activate && uv sync
uv run bot.py                                   # WebRTC server in browser
LLM_CANVAS_PROVIDER=anthropic uv run bot.py     # extras must be installed first
uv run pytest -q
.venv/bin/python tests/bench_canvas_latency.py --iterations 50
```

`.env`:

```
OPENAI_API_KEY=...                # main LLM (default)
ANTHROPIC_API_KEY=...             # main LLM when LLM_CANVAS_PROVIDER=anthropic
GOOGLE_AI_API_KEY=...             # main LLM when LLM_CANVAS_PROVIDER=gemini; ALSO S67b vision (always — unset ⇒ vision stubs)
DEEPGRAM_API_KEY=...
CARTESIA_API_KEY=...
DAILY_API_KEY=...
HV_API_URL=http://localhost:3001/api/v1   # backend; passed through runner_args.body in prod
LLM_CANVAS_PROVIDER=openai

# --- S65b — must match backend exactly ---
NARRATION_CACHE_ENABLED=true
NARRATION_TTS_MODEL_ID=sonic-2            # MUST equal backend
NARRATION_AUDIO_ENCODING=pcm_s16le
NARRATION_AUDIO_SAMPLE_RATE=24000         # MUST equal backend
NARRATION_AUDIO_NUM_CHANNELS=1

# --- S66 ---
VISION_REFRESH_MODE=lazy                  # lazy (default) | eager

# --- S67b — agent vision of the annotated canvas ---
VISION_MODEL=gemini-3.5-flash             # dedicated vision model; decoupled from LLM_CANVAS_PROVIDER
VISION_CAPTURE_TIMEOUT_MS=4000            # capture round-trip ceiling
VISION_MAX_DIM=1280                       # advisory maxDim sent to the shell (shell owns encode)
```

For Pipecat Cloud testing where the container can't reach `localhost`, point `HV_API_URL` at a publicly-accessible URL.

---

## Deployment — Pipecat Cloud

Production runs on Pipecat Cloud. The Dockerfile is the build manifest. Backend's live-room start endpoint creates a Daily room and registers the Pipecat Cloud agent; the agent boots, reads room metadata, fetches the snapshot, joins via DailyTransport.

`LLM_CANVAS_PROVIDER` and the **`NARRATION_*` constants** are set as Pipecat Cloud env vars. Changing the `NARRATION_*` values requires coordinated redeploy with the backend.

**Dockerfile gotcha:** the build uses `pip install --no-cache-dir .` against `pyproject.toml`, but `packages = []` / `py-modules = []` — install pulls only dependencies. Agent code reaches `/app/` exclusively via explicit `COPY` lines. Adding a new top-level directory requires a matching `COPY` line. `services/cached_first_tts.py` lives under the existing `COPY services services`.

---

## Conventions Claude Code should follow

- **Match the installed pipecat-ai version's import paths.** Grep `bot.py` first; module layout shifts between versions.
- **Tool names must satisfy `^[a-zA-Z0-9_-]+$`.** Don't reintroduce dots. Daily wire-format `type` may use dots.
- **Don't add eager-dispatch entries for verbs with required args.**
- **Don't modify `tools/canvas_protocol_tools.py` for new Page types.** The 5 generic tools are page-agnostic; new Page types are accommodated through their manifest.
- **Don't add new Daily message types without coordinating with the frontend.** `DailyRelay` is the sole consumer/producer of `canvas.*` messages; non-canvas messages (`script_complete`, `request_*` triggers, S67b `request_canvas_capture` / `canvas_capture_result`) belong in the shell's general handler.
- **Don't call `api_navigate` from agent-side scene-change handlers.** The frontend's `navigateToIndex` already advances the cursor.
- **Don't add `__init__.py` to `tools/`, `context/`, `services/`** unless you also update the Dockerfile (Python 3.12 implicit namespace packages).
- **Don't ask the LLM to copy large structured blobs between tool calls.** Bundle the side effect (`generate_quiz_from_knowledge` is canonical).
- **Don't ask the LLM to pace UI animations.** Own pacing in the Page (cancellable timers); keep the agent a pass-through. (S65 auto-advance applies the same lesson — pacing lives in the shell.)
- **`canvas_set_page` is all-or-nothing.** Empty `pageInit` ⇒ snapshot wins for BOTH pageType and init.
- **`CachedFirstTTSService` must emit the canonical TTS frame envelope on hits.** Never `OutputAudioRawFrame`. The guard test (`test_hit_never_emits_output_audio_raw_frame`) is non-negotiable — the `NarrationCompletionGate` fix depends on it.
- **`NARRATION_*` constants are paired with the backend.** Don't change one without the other. Use `NARRATION_CACHE_SCHEMA_VERSION` (bumped on the backend) if you need a one-way invalidation.
- **On a cache hit, don't call `set_cartesia_voice`.** The voice is baked into the bytes; the call is wasted control traffic.
- **Vision is a dedicated Gemini path, decoupled from `LLM_CANVAS_PROVIDER` (S67b).** Capture-first AND Pillow-fallback bytes both go to `services/vision_client.py`; the conversational LLM gets `vision_client`'s **text**, not an image. Don't route vision back through the main LLM, and don't couple `VISION_MODEL` to the conversational provider.
- **`vision_client` uses `thinking_level`, not `thinking_budget`.** gemini-3.5-flash is a 3.x model; `ThinkingConfig(thinking_level="low")`. Setting both errors. Use the async surface `client.aio.models.generate_content(...)` — the sync call blocks the pipeline loop.
- **The capture image never rides Daily.** 4 KB `sendAppMessage` limit; a JPEG is 30–300+ KB. The shell uploads to the backend Redis ingest; the agent fetches via `get_vision_capture`. `request_canvas_capture`/`canvas_capture_result` carry metadata only, keyed by `captureId` in a sibling registry (never collide with canvas commandIds).
- **Never let the agent claim to see annotations on the Pillow fallback (S67b V7).** `run_vision_query` appends a `[vision note: …]` blind-spot/unavailable note and the PLAYBOOK forbids fabricating drawings; keep both.

---

## Historical: S66 complete (Flow Scene-Switching Performance — agent hot-path cuts)

S66 cut agent-side scene-change latency (`T_agent`) so transitions land under 1 s. **No Canvas Protocol contract change.** Detail in "Scene-change refresh", "Live Room snapshot consumer", and "Vision" above. Summary:

- **5a — Lazy vision** (`VISION_REFRESH_MODE=lazy` default): `refresh_agent_for_current_scene` no longer renders the Pillow PNG on every scene change; `_ensure_vision_frame()` fetches on demand (visual question / first `canvas_analyze`) and caches per scene id. Biggest single `T_agent` win. `eager` is the escape hatch.
- **5b — Flow-knowledge reuse:** the invariant all-scene knowledge block is held in-session keyed by knowledge version; only per-scene bits are re-stitched on scene change. The backend's S66 flow-knowledge cache makes the fetch cheap; the agent avoids re-parsing.
- **5c — By-`sceneId` fetch:** `canvas.sceneChanged` carries `sceneId`; the agent fetches the snapshot by id off the cursor-independent endpoint (cursor-race-free, cache-served), with the cursor-based fetch retained as fallback. Snapshot-payload piggyback (option b) was deferred.

### Tests

+5 in `test_scene_change_perf.py`: `lazy` → no image fetch on scene change; `_ensure_vision_frame` fetches once per scene then caches; `eager` restores per-change render; flow-knowledge reused when version unchanged / re-stitched when changed; `sceneId` present → by-id fetch, absent → cursor fallback. Existing S65/S65b/S65c tests pass unchanged.

### Lessons (READ BEFORE TOUCHING)

1. **Lazy vision is correctness-neutral, latency-positive.** Vision quality on a real visual question is unchanged — the frame is just fetched when needed instead of speculatively. Don't "optimize" by pre-rendering in the background; that re-introduces the hot-path cost S66 removed.
2. **Never depend on the pushed `sceneId`.** It's an optimization. The cursor-based fetch must remain a working fallback or a dropped Daily message strands the agent on the wrong scene.
3. **Flow-knowledge reuse is keyed by version, not "same flow."** A knowledge edit mid-session must bump the version and force a re-stitch — otherwise the agent answers from stale knowledge. The version comes from the snapshot (backend computes it).
4. **The agent still does no navigation.** S66 changed *how* the agent fetches the new scene, not *who* decides the scene. The shell owns navigation + auto-advance (S64e/S65). By-id fetch is cursor-independent precisely so it can't accidentally move anything.

---

## Historical: S65c complete (Live Room Action Buttons — manual triggers)

S65c gave visitors three explicit controls (Script · Survey · Quiz) in the full-bleed shell. Agent-side, it factored the quiz core for reuse and added two inbound session-level triggers — without touching the canvas protocol or the relay.

### What landed

- **`run_quiz_generation(...)` core** (`tools/quiz_generation.py`) — the quiz-generation logic (backend `generate_quiz` call + bundled `canvas.set_page(quiz, blob)` dispatch) extracted into a plain coroutine with an optional `on_state(state, error)` hook. Two callers: the LLM tool handler (wraps it; **return shape unchanged** — `test_quiz_factor.py` snapshots before/after) and the manual `request_quiz` trigger (passes the `on_state` emitter).
- **Inbound `{type:'request_narrate'}`** → `narrate_scene_script(force=True, trigger='manual')`. `force=True` bypasses the once-per-entry guard; `trigger='manual'` rides through to `script_complete` so the shell does **not** auto-advance on a replay.
- **Inbound `{type:'request_quiz', count?, language?}`** → `run_quiz_generation(...)` with an `on_state` hook emitting `{type:'quiz_generation_state', state}`. Silently ignored when the agent isn't ready (`session_context` absent) — the S64d lazy-spawn lesson.
- **Both `request_*` branches are early-return branches in `on_app_message`, alongside but BEFORE the canvas dispatch** — never inside the canvas/relay path (mirrors the frontend keeping these off `DailyRelay`).
- **`script_complete` payload gained `trigger: 'auto' | 'manual'`** (default `'auto'` — back-compat with the S65 emit; one line in `narration.py`).
- **Outbound `{type:'quiz_generation_state', state, error?}`** — only emitted by the manual path (the LLM-tool quiz has no button to update).

### Tests

+6 across `test_request_narrate.py` (manual trigger → `force=True`, `trigger='manual'`, agent does NOT navigate), `test_request_quiz.py` (runs the core, emits `generating→ready`, emits `error` on backend failure, ignored when not ready), `test_quiz_factor.py` (**`test_llm_tool_return_shape_unchanged`** — the regression guard for the refactor; button + LLM paths produce the same blob shape), and `test_script_complete_trigger.py` (default `'auto'`, manual passes through, `force` bypasses the guard). Existing S64e quiz tests and S65 narration tests pass unchanged.

### Lessons (READ BEFORE TOUCHING)

1. **The factor must not change the LLM tool's return shape.** S64e's voice-initiated quiz ships today; a careless extraction breaks it silently. `test_llm_tool_return_shape_unchanged` is the guard — keep it.
2. **`request_*` are session-level, not canvas commands.** They live alongside (before) the canvas dispatch in `on_app_message`, never inside it / never in the relay. Same single-responsibility discipline as the frontend's `DailyRelay`.
3. **Manual narration never auto-advances.** The whole point of the `trigger` tag. The agent does no navigation either way (the shell owns advance) — `trigger` just tells the shell which `script_complete`s are advance-eligible.
4. **`request_quiz` is silently ignored before readiness.** A button click that arrives before `session_context` is populated is dropped, not errored — the agent can't have joined-but-uninitialized produce a half-quiz.

---

## Historical: S65b complete (`CachedFirstTTSService` + narrator prefetch/prime)

S65b shipped the agent-side cache consumer for the backend's pre-rendered narration audio. The full detail is in **"Cached narration audio (S65b)"** above. Summary:

- `services/cached_first_tts.py` — composes `CartesiaTTSService`; cache-first on `prime_cached(...)`, live-fallback otherwise; emits the canonical TTS frame envelope so `BotStoppedSpeakingFrame` still fires on playback-done (gate-fix composition).
- `bot.py` instantiates `CachedFirstTTSService` in place of `CartesiaTTSService` in the classic pipeline; relay pipeline untouched.
- `narration.py` adds `_prefetch_cached_audio(...)` (one `aiohttp` gather over all `ready` segments at scene-narration start) and primes the service per-segment immediately before each speak call. Miss path keeps S65's voice switch; hit path skips it (voice baked in).
- New `config.py` `NARRATION_*` constants paired with backend.
- `tests/test_cached_first_tts.py` — +6 tests. Total agent test count up; existing S65 + S64e tests pass unchanged.

**Lessons (READ BEFORE TOUCHING):**

1. **The frame envelope is the load-bearing detail.** Emitting `OutputAudioRawFrame` instead of `TTSAudioRawFrame` bypasses the speaking-state machine and silently breaks `BotStoppedSpeakingFrame` for cached scenes — which is what the `NarrationCompletionGate` fix relies on. Guarded by `test_hit_never_emits_output_audio_raw_frame`.
2. **`prime_cached` is consumed once.** A stray second `run_tts` after one prime is a miss, not a replay. If S65 ever batches multiple segments into one TTS call, split it (one speak per segment) so hits/misses can differ within a scene.
3. **On a hit, voice switching is a no-op.** The bytes already carry the voice. Calling `set_cartesia_voice` is harmless but noisy — skip on the hit path.
4. **The relay pipeline does NOT cache.** SoulX renders its own audio. `CachedFirstTTSService` only lives in the classic pipeline.
5. **Cross-repo invariant has no automated cross-repo test.** Sample-rate / model / encoding mismatch produces audible breakage at session start. The gate is convention plus `test_cached_first_tts.py`'s sample-rate assertion against `settings.NARRATION_AUDIO_SAMPLE_RATE`.

---

## Historical: S65 complete (Live Room Script Narration + Flow Auto-Advance)

The narration foundation S65b builds on. Full detail in **"Live Room Narration (S65)"** above. Agent-side summary:

- `narration.py` with `narrate_scene_script` — per-segment Cartesia voice switching, post-script localized invitation, `script_complete` emit with `sceneIndex`/`hadScript`.
- Auto-advance signalling (D8): on non-final + `auto_advance` scenes, **suppress** the invitation; on final scenes or non-auto-advance flows, speak the invitation.
- Relay pipeline narrates in the primary SoulX voice and still emits `script_complete` (per-script-avatar voice is a v0.2 punt — D4).
- SCRIPT prompt directive added to `persona.build_system_prompt` (Strategy 1) telling the LLM not to read or paraphrase the script.
- `refresh_agent_for_current_scene` extended to call `narrate_scene_script` after each scene-change snapshot fetch; idempotent per entry (no double-narration on `canvas.register`-only rebuilds).
- Tests in `test_scene_narration.py`: voice clone-vs-fallback, invitation suppress/show by auto-advance + last-scene, once-per-entry, relay-path `script_complete`.

**Key lesson:** **the agent emits `script_complete`; the shell decides whether to advance.** Never let the agent navigate scenes — that's the S64e "don't let the LLM pace UI" rule applied to flow control.

---

## Historical: S64e complete (Quiz Page + end-to-end `canvas.set_page`)

`canvas.set_page` end-to-end (Quiz is the first scenario where the agent switches Pages mid-scene). Agent-side: `tools/quiz_generation.py` with `generate_quiz_from_knowledge` (Option D — backend call + bundled `set_page` dispatch); `dispatch_canvas_command` promoted to module-level helper; `SessionContext` carries slug + current scene id; PLAYBOOK forbids the LLM from calling `canvas_set_page(pageType='quiz')` directly; `skip_question` action verb ("I don't know"); `submit_answer` is pass-through (`test_submit_answer_bundling.py` guards "advancement is the Page's job"); `canvas_set_page` all-or-nothing override semantics (empty `pageInit` = restore snapshot).

---

## Historical: S64d complete (YouTube Canvas Page) — zero agent code

Frontend shipped the YouTube Canvas Page; **Pipecat required zero changes.** When a visitor navigates to a YouTube scene: frontend mounts the YouTube iframe → Page registers its manifest (`play/pause/seek/set_speed/clear`, highlight `['box']`) → relay forwards `canvas.register` → agent's `on_app_message` routes to `CanvasManifestRegistry.set_manifest` and rebuilds the prompt → LLM uses `canvas_control` with the new verbs. `play`/`pause`/`clear` are in `EAGER_DISPATCH_VERBS` (will eager-fire once the streaming loop is wired); `seek`/`set_speed` wait for full streaming (required args).

---

## Recent: S67a (Live Room Annotation Tools) — zero agent code

S67a added pen/highlight/text/eraser annotation tools to the live-room **frontend** shell (a shell-owned overlay above the iframe + a right-edge rail). Annotations are **temporary, session-only, never persisted**, and **never sent to the agent** — no Daily message, no canvas-protocol traffic, no snapshot field. The agent was **not touched**. Recorded here only for phase completeness.

> The annotations matter to the agent **only via S67b**, and the mechanism is deliberately **pixels, not vector data**: the agent never learns about strokes/text boxes as structured data — it asks the shell for a screenshot (which already has the annotations burned in) and reasons over the image.

---

## Recent: S67b complete (agent-side — Agent Vision of the Annotated Canvas)

S67b is the **big agent session of Phase 9j** and the consumer of S67a. It gives the agent eyes on **what the visitor actually sees** — the live scene plus the S67a annotations in real pixels — for visual Q&A, "what am I pointing at?", and fill-in-the-blank answer assessment. **Agent-side complete** — B9–B16 + PLAYBOOK, plus the live-testing refinements (permission fast-path, live-capture-first, the two-stage `describe` nudge, and the deterministic `vision_state` indicator). **The end-to-end slice still needs three cross-repo pieces: the frontend capture handler (B1), the frontend `vision_state` indicator, and the backend Redis ingest.** The `## Vision (S46 → S67b)` and `## Daily app-messages` sections above describe the runtime behavior; this section is the design + as-built record.

**Core architecture (read first):** the agent is a server-side Pipecat process with **no DOM**, so it **cannot call `getDisplayMedia` itself**. Capture happens in the **visitor's browser** (the shell), triggered by the agent over a **non-canvas Daily round-trip** — the only way to get the visitor's true client-side pixels. **The image does NOT travel over Daily** — Daily's `sendAppMessage` enforces a **4 KB limit** and a JPEG screenshot is tens of KB, so the shell uploads the bytes to a tiny **backend Redis ingest** and the agent fetches them from there (the Daily messages carry only `{captureId, status, w, h}`):

```
agent: request_canvas_capture(hint)  ──Daily(<4KB)──▶  {type:'request_canvas_capture', captureId, hint}
        │  (general non-canvas handler in the shell — NOT DailyRelay)
        │  shell runs Screen Capture API → grabFrame → downscale → JPEG (Blob)
        │  shell ──HTTPS POST raw bytes──▶ backend Redis  vision:capture:{slug}:{captureId}  (TTL 60s)
        ▼
shell: {type:'canvas_capture_result', captureId, status:'ready'|'error', w, h}  ──Daily(<4KB)──▶  agent
        │  resolve the awaiting asyncio.Future by captureId (sibling of PendingCommandRegistry)
        ▼
agent: api_client.get_vision_capture(slug, captureId)  ──HTTPS GET──▶ bytes (delete-on-read)
        ▼
agent: vision_client.analyze_image(image_bytes, mode, scene_context) → reasoning
```

> **Why the backend hop:** the 4 KB Daily limit forbids sending the screenshot over the data channel, so it transits a short-TTL no-store backend relay (two endpoints; Redis only; no DB, no R2). This is the one change from the V2.17 sketch, which had assumed the image could ride Daily / the backend was optional. The image still goes **agent-direct to Gemini** — the backend only relays bytes.

**Design (locked decisions — all implemented agent-side; per-item as-built notes in "As built" below):**

- **`services/vision_client.py`** — a **dedicated Gemini multimodal client** (lazy Google-SDK import, mirroring `_build_llm_and_eager_hook`). `VISION_MODEL` env, default **`gemini-3.5-flash`** (stable, multimodal image input, 1M-token context). **Decoupled from `LLM_CANVAS_PROVIDER`** — vision is fast Gemini regardless of the conversational LLM. **Stub/degrade gracefully when `GOOGLE_AI_API_KEY` is unset** (every external service degrades — project principle).
- **Capture round-trip** — `request_canvas_capture(hint)` emits the non-canvas Daily request with a fresh `captureId`, registers an `asyncio.Future` (sibling of `PendingCommandRegistry`, keyed by `captureId`), awaits `canvas_capture_result` (timeout `VISION_CAPTURE_TIMEOUT_MS` ~4000). The result carries `status`/`w`/`h` only — **not the image**. Handle it in an **early-return `on_app_message` branch alongside `request_narrate`/`request_quiz`, BEFORE the canvas dispatch** — never via the relay (same discipline that keeps `script_complete`/`request_*` off `DailyRelay`). Rides the existing defensive `json.loads`.
- **`api_client.get_vision_capture(slug, captureId)`** — fetches the JPEG bytes the shell uploaded to the backend ingest (HTTPS GET, public base URL like `get_scene_snapshot_image`; backend deletes on read).
- **Vision entry-point rework** — `_ensure_vision_frame()` (or a new `get_vision_image()`) becomes: → `request_canvas_capture()`; on `status:'ready'` → `api_client.get_vision_capture()` → bytes → `vision_client`; *on timeout/error/permission-denied* → **fall back to the existing `get_scene_snapshot_image()` Pillow PNG** and set a flag so the reply notes the annotation blind spot ("I can describe the scene but can't see your drawings unless you let me view your screen"). The visual-question / first-`canvas_analyze` path routes through this. **(Evolved in live testing — see "As built": the final rule is live-capture-first; Pillow is used ONLY for a *repeated* `describe` while screen-share is off, and timeouts/errors retry once then a try-again note rather than Pillow.)**
- **Reasoning modes (V6)** — derive `mode` (`point | assess | describe`) from the utterance; for `assess`, pass the scene's expected answer/knowledge (already in the snapshot — `scripts`/`knowledge`) as `scene_context` so Gemini grounds the correctness judgment.
- **PLAYBOOK directive** — when a visitor refers to something they've drawn/pointed at/written, invoke the vision path; on the Pillow fallback, never claim to see annotations.
- **Config** — `VISION_MODEL` (default `gemini-3.5-flash`), `VISION_CAPTURE_TIMEOUT_MS` (~4000), advisory `VISION_MAX_DIM`/`VISION_JPEG_QUALITY` (the shell owns encode; the agent may request a maxDim in the `hint`).
- **Dockerfile** — `services/vision_client.py` is reached via the existing `COPY services services` (no new top-level dir). Ensure the Google SDK is a dependency even when `LLM_CANVAS_PROVIDER` isn't Gemini.
- **Speed (V8)** — kick the capture request off in parallel with prompt/context assembly; persistent client stream means no per-question permission prompt; downscale-before-encode + JPEG q≈0.7 keep the payload tiny; `gemini-3.5-flash` is the low-latency model. Target end-to-end < ~1.5–2 s on a warm stream; baseline into `docs/benchmarks/canvas_vision_<date>.md`.
- **Tests (~6–8)** in `tests/test_canvas_vision.py`: round-trip resolves the future by `captureId`; timeout → Pillow fallback + blind-spot flag; `permission_denied` → fallback; `assess` passes scene answer into the prompt; `vision_client` stub path with the key unset; `canvas_capture_result` handled in the non-canvas branch (never via the relay). Existing S65/S65b/S65c/S66 tests pass unchanged.

### As built (final — including live-testing refinements)

**Deviations from the V2.17 sketch.**
- **Orchestration extracted to `services/vision_query.py`** (the sketch kept it inline). `run_vision_query(...)` is a dependency-injected core (mirrors `run_quiz_generation`); the bot.py `_ensure_vision_for_active_scene(question)` closure is a 1-line adapter. Extracted so the capture / fallback / assess / nudge logic is unit-testable (bot.py needs the full pipecat stack to import).
- **The vision result is injected as a `[vision: …]` developer message** (same `context.add_message` mechanism the old Pillow path used); the frontend `analyze` dispatch (semantic state) still runs. `ensure_vision`'s `CanvasToolContext` type is `Callable[[str], …]` (passes the visitor's `question` for mode derivation).
- **`thinking_level`, not `thinking_budget`** — gemini-3.5-flash is 3.x. `vision_client` uses the **async** `client.aio.models.generate_content(...)` with `Part.from_bytes` + `ThinkingConfig(thinking_level="low")`, validated against the installed **google-genai 1.75.0**. Exports `derive_vision_mode(utterance)`.
- **`google-genai>=1.68.0,<2` is a BASE dep** (not `pipecat-ai[google]`); resolved to 1.75.0. `vision_query.py` + `vision_client.py` ride the existing `COPY services services` — no Dockerfile change.

**Refinements after the initial close-out (driven by live testing).** The entry-point / fallback logic evolved well past the sketch's "fall back to Pillow on any failure":
- **Live-capture-first.** When screen-share is on (`status: ready`) the agent ALWAYS reasons over the live capture; **Pillow is never substituted**. `get_vision_image` was split into `_fetch_live_bytes` (live only) + `_fetch_pillow` (base scene only). A fast transient miss (ready-but-no-bytes / non-permission error) **retries the capture once**; a timeout does not (already waited the budget) → `_CAPTURE_RETRY_NOTE`.
- **Permission fast-path.** `point`/`assess` + permission denial → `_SHARE_SCREEN_NOTE` (ask to click the button; no Pillow, no Gemini).
- **Two-stage `describe` nudge.** First `describe` while screen-share is OFF → `_DESCRIBE_SHARE_NUDGE_NOTE` (ask to click **"Let the Assistant see your screen"** — `_SHARE_SCREEN_BUTTON`, named verbatim; **no Pillow yet**); a *repeated* `describe` while still off → base-scene Pillow + blind-spot note. Tracked by `SessionContext.describe_share_nudged`, **reset whenever a capture comes back `ready`** (one nudge per off-period). `run_vision_query` takes `session_context` (derives slug + scene_id + the nudge flag).
- **Prompt broadening so describe questions actually reach vision.** Live testing showed the LLM answered "what's on screen?" from its prompt context and **never called `canvas_analyze`** → the vision path (and the nudge) never ran. Fix: the `canvas_analyze` tool description dropped the "cannot determine from existing context" escape hatch, and the PLAYBOOK directive now leads with "the live screen is NOT in your text context" + plain-describe examples. (Prompt-level ⇒ probabilistic, not a hard guarantee.)
- **`vision_state` indicator (V8 — latency UX).** A **deterministic** `{type:'vision_state', state:'analyzing'|'idle'}` Daily message brackets the Gemini `analyze_image` call (via `run_vision_query`'s `on_vision_state` hook), so the shell can show a "looking at your screen…" cue ONLY while the model runs — never on the fast nudge/retry paths. `try/finally` guarantees `idle`. Chosen over a verbal acknowledgement ("too much verbal").
- **Observability.** `request_canvas_capture` and the `canvas_capture_result` branch log capture id / status / future-found, so the round-trip is debuggable in Pipecat Cloud logs (added after a live test couldn't tell whether the reply reached `on_app_message`).

**Tests** — `tests/test_canvas_vision.py` is now **26 tests** (**211 total**, + 1 deselected `live` real-Gemini test): round-trip resolve / timeout / stale; `_fetch_live_bytes` & `_fetch_pillow`; live-first never-touches-Pillow, ready-no-bytes-retries, timeout-no-retry, error-retries; assess scene-context; client-degrade → unavailable; permission fast-path (point); two-stage describe (first-nudge / repeat-Pillow / **reset-on-ready**); `_is_permission_error`; **`vision_state` brackets-Gemini / no-flash-on-fast-path / clears-on-degrade**; `VisionClient` stub + no-SDK-touch; `canvas_capture_result` routed-before-dispatch (real-source guard + a contract-mirror, since the bot.py closures aren't importable). `ruff check . && pytest` green (B16 added a `[tool.ruff]` per-file-ignore for bot.py's intentional `E402`).

**Latency (V8)** — `docs/benchmarks/canvas_vision_2026-06-04.md`. The path has **two sequential LLM calls** (Gemini vision + the conversational TTS-LLM that speaks the answer, since `analyze_image` is non-streamed) + the capture round-trip → ~1.3 s best / ~2.4 s worst; the < 1.5–2 s target is **tight, not guaranteed**. The `vision_state` indicator masks the Gemini window in the UI; the deferred hard win is streaming the vision answer straight to TTS to drop the 2nd LLM turn.

**Lessons (READ BEFORE TOUCHING).**
1. **Vision is dedicated Gemini, decoupled from `LLM_CANVAS_PROVIDER`** — text out, not image. `eager` mode still pre-loads a Pillow image into the main context — vestigial.
2. **Pillow is used in exactly ONE place:** a `describe` question's *repeat* ask while screen-share is OFF. Screen-share on ⇒ never Pillow. Don't reintroduce a blanket fallback.
3. **`GOOGLE_AI_API_KEY` must be set on the DEPLOYED agent**, not just locally — else every capture stubs to `VISION_UNAVAILABLE` and the agent silently degrades. (Cost a live debugging session.)
4. **The LLM must choose to call `canvas_analyze`** for the vision path (and the nudge) to run at all — a prompt concern, inherently probabilistic. If it regresses, the deterministic fallback is to detect screen/visual utterances and force the path.
5. **The feature depends on the frontend actually granting `getDisplayMedia`.** Until it does, captures return `permission_required` and the agent can only nudge / show the base scene.
6. The S66 `VisionFrameTracker` is write-only post-S67b; `vision_refresh.ensure_vision_frame_for_scene` has no runtime caller. Removable later.
7. Capture bytes never ride Daily (4 KB) — backend Redis hop is mandatory. `request_canvas_capture` / `canvas_capture_result` / `vision_state` are session-level (early-return before the canvas dispatch / general handler), keyed by `captureId` in a sibling `_pending_captures` registry; session-end drains it.
8. **The agent names the "Let the Assistant see your screen" button verbatim** (`_SHARE_SCREEN_BUTTON` in `vision_query.py`). If the shell's label changes, update that constant or the agent points at a button that doesn't exist.

**Cross-repo remainder (NOT agent-side — required for the slice to work end-to-end).**
- **Frontend B1 — capture handler.** The shell's `request_canvas_capture` handler (general non-`DailyRelay`): Screen Capture API → grab → downscale (honor `maxDim`) → JPEG → POST to the backend ingest → reply `canvas_capture_result {status, w, h}`. **Must actually obtain + hold the `getDisplayMedia` grant**, and send a permission error (something containing `permission`) on denial — the agent keys the nudge / fast-path off that.
- **Frontend — `vision_state` indicator.** Show "looking at your screen…" on `state:'analyzing'`, hide on `'idle'`, with a ~8 s safety auto-hide (Daily is best-effort).
- **Backend — Redis ingest.** Two endpoints: POST upload (`vision:capture:{slug}:{captureId}`, TTL ~60 s) + GET delete-on-read at `/live-rooms/by-slug/{slug}/vision-capture/{captureId}`.

After S67b: **S68 (Knowledge-Aware Generate Scene)** — the former S67, unchanged in scope; agent involvement minimal (generation is a studio/backend feature). Then MCP (S69–70), video export (S71), deployment + launch (S72–76).

---

## Out of scope

- Mid-session provider switching (`LLM_CANVAS_PROVIDER` fixed at boot).
- Natural-language highlight targets.
- Persistent iframe shell (per-scene unmount + keyed remount is current; S66's optional prewarm double-buffer was **deferred to S74** — Blocks 1–5 hit the < 1 s target without it).
- A/B testing infrastructure for comparing providers in production.
- **Streaming the vision answer directly to TTS** to skip the 2nd conversational-LLM turn for pure visual questions (the S67b latency lever — deferred; today's design routes the vision text back through the conversational LLM for persona/voice consistency). *(Vision provider separation itself is now DONE — S67b's `vision_client` is dedicated Gemini, decoupled from `LLM_CANVAS_PROVIDER`.)*
- Eager-dispatch-to-Pipecat-streaming-loop wiring (hooks constructed; never invoked; tracked for S74).
- **Caching narration audio for the relay (`talking`) pipeline.** SoulX renders its own audio — `CachedFirstTTSService` is only in the classic pipeline. Per-script-avatar voice in relay is the same v0.2 punt.
- **Caching fallback (room-primary-voiced) segments.** Room-dependent → requires room-scoped keys → v2.
- **Cache warming on publish, edge/multi-region warming, Opus/OGG encoding to cut R2 size.** All v2 considerations.
