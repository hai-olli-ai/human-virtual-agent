# pipecat-agent — CLAUDE.md

> **S80 (2026-08-16/17) — Avatar Mode Split: ZERO agent changes; the relay pipeline is REACHABLE again (image stays `0.16`).** The backend split `display_mode` into per-scene `avatar_interactivity` × `avatar_visibility` and the snapshot's `avatar_display_mode` is now DERIVED: `'talking'` ⇔ interactivity `talking` (visibility `invisible` wins; only `'3dgs'` is retired). **P-1′ law: Talking = the live SoulX face** — `_resolve_output_mode` selects `relay_avatar` for sessions whose CURRENT scene at join is a talking scene, and the S79-field-wave hybrid already aboard (`bot.py` relay `NarrationCueController`: clip-cued lines vs live-SoulX lines) is now the PRODUCT behavior: pre-paid clips narrate when available, live SoulX renders clip-less lines and all chat. Recorded limitation: pipeline selection stays per-SESSION (a session entering on a static scene keeps the portrait for later talking scenes; the shell suppresses the live layer on static scenes in relay sessions). **GO preflight (S80 §E): verify the prod secret set carries `SOULX_WS_URL` before any prod scene flips Talking** — a talking scene selecting relay with no renderer degrades to voice-only. Record: backend `guidelines/SESSION_80_COMPLETION.md`.

> **S76 (2026-08-04→08) — V2 arc CLOSED · hv.ai LIVE 2026-08-06T07:21:31Z · announce 2026-09-04.** Prod `pcc-deploy.toml` `min_agents 0→1` (PR #11, same tag 0.9, deployment 22af6789) — warm join **< 5 s** vs the 19 s cold baseline; standing cost ~$43.80/mo (agent-2x reserved $0.0010/min); `max_agents` 10 platform-confirmed (the capacity census's binding ceiling — watch concurrent sessions at the announce). **Staging toml stays 0.** Rollback = flip to 0 + the same `pcc deploy` line. No production branch — image-tag promotion unchanged. Arc record: backend `guidelines/SESSION_76_COMPLETION.md`.

> **S75 (2026-08-01→04):** hygiene only — the 34-file `ruff format` backlog cleared and `format --check` now enforced INSIDE the `lint` job (job name unchanged — protection contract); `.DS_Store` untracked with the repo's first ignore entry. No behavioral change, no image redeploy (tag-promotion model). The prod launch baseline recorded T_agent p95 434 ms (classic pipeline, lazy vision) — backend runbook §12.


> Voice agent for **Human Virtual** (hv.ai), built on the **Pipecat Framework**.
> **Status:** S64c–e + S65 + S65b + S65c + S66 + **S67b + S67c** shipped · **S68 (External Embeds) shipped — zero agent changes** · **S69a (Generation Engine) shipped — zero agent changes** · **P3 latency pass (2026-07-13) shipped** · **Auto Play Phase A (2026-07-16) shipped — truthful playout completion + interruption awareness (see *Recent: Auto Play Phase A*); Phase B is frontend-side** · **next agent-relevant work: none until MCP E2E (S71).** **S67b** added agent vision of the live canvas: `services/vision_client.py` (dedicated `gemini-3.5-flash` multimodal client, decoupled from `LLM_CANVAS_PROVIDER`, graceful stub when `GOOGLE_AI_API_KEY` is unset), the `request_canvas_capture` round-trip (sibling `asyncio.Future` registry; `canvas_capture_result` handled in the non-canvas `on_app_message` branch), `api_client.get_vision_capture()` (fetch the shell-uploaded JPEG from the backend `vision-capture` ingest — the 4 KB Daily limit forbids the image on the data channel), and the `get_vision_image()` rework (capture-first → Pillow fallback with a blind-spot flag); the vision reasoning is returned **in-band as the `canvas_analyze` tool result** (not an out-of-band developer message). **S67c** unified canvas-interaction: **`canvas_highlight` is RETIRED**, replaced by **`canvas_annotate`** which draws on the **same S67a shell overlay** (non-canvas `agent_annotate` round-trip, not the Canvas Protocol), the `vision_client` gained a **`locate`** mode (normalized bbox for described targets), and the in-iframe highlight path is gone (Canvas-Protocol manifest reduced to **v0.2**). The classic TTS service is a `CachedFirstTTSService` (cache hit → raw PCM in the identical `TTSStartedFrame → TTSAudioRawFrame → TTSStoppedFrame` envelope so `BotStoppedSpeakingFrame` still fires; miss → live Cartesia). `generate_quiz_from_knowledge`'s core is the factored `run_quiz_generation` coroutine; inbound `request_narrate` / `request_quiz` / `canvas_capture_result` / `agent_annotate_result` branches sit alongside the canvas dispatch in `on_app_message`; `script_complete` carries `trigger: 'auto' | 'manual'`. **S66** made scene-change refresh fast: **lazy vision** (`VISION_REFRESH_MODE=lazy` default), **flow-knowledge reuse**, and **by-`sceneId` snapshot fetch** off `canvas.sceneChanged` (cursor-independent; cursor fallback retained). **LLM provider:** the conversational LLM now **defaults to Groq** (`GroqLLMService`, OpenAI-compatible, model `openai/gpt-oss-120b`); `openai`/`anthropic`/`gemini` stay selectable via `LLM_CANVAS_PROVIDER`. ⚠️ `gpt-oss-120b` is **text-only**, so the S46 scene-image-in-main-LLM-context injection is **gated off** for Groq (`MAIN_LLM_SUPPORTS_VISION`); visual Q&A still runs via the decoupled S67b Gemini path (see *LLM provider selection*).
> **Repo:** `pipecat-agent/` (alongside `human-virtual-backend/` and `human-virtual-frontend/`).

---

## What this service does

The Pipecat agent runs the avatar's **conversation pipeline**: STT → LLM → TTS → audio out, with optional vision and canvas tool calls. It connects to a Live Room over WebRTC and:

1. Listens to the visitor's microphone (Deepgram STT).
2. Calls an LLM (**Groq** (default) / OpenAI / Anthropic / Gemini, selected at boot via `LLM_CANVAS_PROVIDER`) with a system prompt assembled from the Live Room's persona, knowledge (aggregated across the whole flow), current scene instruction, current scene's elements (referenced by short aliases), and the active Canvas Page's manifest.
3. Emits TTS audio (Cartesia, now via `CachedFirstTTSService`) plus avatar lip-sync data (SoulX-Flashtalk for "talking" display mode).
4. Calls **canvas tools** via the generic 5-tool surface (`canvas_analyze`, `canvas_annotate`, `canvas_control`, `canvas_action`, `canvas_set_page` — **underscored, not dotted**). *(S67c: `canvas_annotate` replaced the retired `canvas_highlight`; unlike the others it does NOT emit a `canvas.*` command — it drives the shell annotation overlay via the non-canvas `agent_annotate` round-trip.)*
5. (S46+) Calls **vision** by adding a snapshot image to the LLM's context.
6. (S49 + S65) Plays the scene's script on entry. **S65** added per-segment script-avatar voice switching, a localized post-script invitation, and `{type:'script_complete'}` carrying `sceneIndex`/`hadScript`. **S65b** added the cached-audio fast path through the same narrator without changing the control flow. **Auto Play Phase A** made `script_complete` truthful (emitted only after real playout drain, never from an interrupted/cancelled run) and added the inbound `autoplay_control` stop/resume controls — every narration entry point now runs in a single-slot background task.

---

## Stack

| Layer | Choice |
|---|---|
| Framework | Pipecat (Python, pinned at 0.0.108 in `pyproject.toml`) |
| Transport (prod) | DailyTransport (Daily.co WebRTC) |
| Transport (local) | SmallWebRTCTransport |
| Deployment (prod) | Pipecat Cloud |
| LLM default | **Groq** `GroqLLMService` — OpenAI-compatible; model `openai/gpt-oss-120b` (via `GROQ_MODEL`). ⚠️ text-only — in-context image injection gated off (`MAIN_LLM_SUPPORTS_VISION`) |
| LLM alt | OpenAI GPT (extra already installed), Anthropic Claude / Google Gemini (extras must be installed) |
| LLM selection | `LLM_CANVAS_PROVIDER` env var — `groq`/`openai`/`anthropic`/`gemini`, validated at boot |
| STT | Deepgram |
| TTS (classic) | **`CachedFirstTTSService`** *(S65b)* — composes `CartesiaTTSService`; cache-first, live-fallback; runs **`MarkdownTextFilter`** to strip stray Markdown before synthesis (#2 — plain-speech, audio-side) |
| Lip-sync (talking) | SoulX-Flashtalk (S48 — relay pipeline; **S65b cache does NOT apply** — relay renders its own audio) |

---

## Project structure

```
pipecat-agent/
  bot.py                            # Main entrypoint — pipeline assembly, transport, LLM provider branching;
                                    #   constructs CachedFirstTTSService (S65b) and wires it into both pipelines;
                                    #   Phase A — single-slot narration runs (_narrate_and_complete /
                                    #   _start_narration_task), _flush_bot_audio handshake, autoplay_control branches
  config.py                         # Settings, provider validation, model defaults;
                                    #   S65b NARRATION_* paired constants (must match backend)
  persona.py                        # build_system_prompt — Strategy 1 path + S65 SCRIPT prompt directive
  scene_context.py                  # Section helpers, knowledge formatting, alias generator
  api_client.py                     # HTTP client for the backend (snapshot, persona-prompt, navigate, scene image, generate-quiz, …)
  narration.py                      # S65 — narrate_scene_script: per-segment voice switch + invitation + script_complete
                                    #   S65b — _prefetch_cached_audio + per-segment prime_cached(...) before each speak
                                    #   Phase A — NarrationCompletionGate drain/interruption (per-utterance counters,
                                    #   NARRATION_INTERRUPTED sentinel, begin_run), NarrationInterrupted,
                                    #   compute_playout_drain_timeout + PLAYOUT_DRAIN_* constants (agent-only)
  context/
    prompt_builder.py               # render_canvas_page_section + render_agent_playbook_section + render_voice_output_style_section
    canvas_manifest.py              # CanvasManifestRegistry
  tools/
    canvas_protocol_tools.py        # 5 generic tools (canvas_annotate replaced canvas_highlight in S67c) + dispatch_canvas_command; SCENE_NAV_VERBS exemption
    quiz_generation.py              # S64e — generate_quiz_from_knowledge tool + bundled set_page
  services/
    cached_first_tts.py             # S65b — CachedFirstTTSService; Phase A — drops an armed prime on InterruptionFrame
    eager_dispatch/                 # Per-provider streaming hooks (S64c) — constructed, not wired into the streaming loop
      __init__.py
      anthropic_adapter.py
      openai_adapter.py             # also reused by the groq branch (Groq is OpenAI-compatible)
      gemini_adapter.py
  tests/
    test_canvas_annotate.py
    test_canvas_vision.py
    test_eager_dispatch.py
    test_link_narration_directive.py
    test_quiz_generation.py
    test_scene_context_knowledge.py
    test_scene_context_s61.py
    test_scene_narration.py         # S65
    test_cached_first_tts.py        # S65b
    test_autoplay_phase_a.py        # Phase A — gate drain/interruption, drain-timeout budget, slot compositions
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
                      →  NarrationCompletionGate                          ← S65 G3; Phase A made it load-bearing:
                      →  DailyTransport.output  ← emits BotStartedSpeakingFrame / BotStoppedSpeakingFrame
                      →  LLMContextAggregatorPair (assistant side)
```

The `NarrationCompletionGate` between TTS and output observes `TTSStoppedFrame` (per-segment sequencing, S65) and — since **Phase A** — mirrors the transport's upstream `BotStartedSpeakingFrame`/`BotStoppedSpeakingFrame` broadcast and the pipeline's `InterruptionFrame`, which is what gates `script_complete` on true playout drain and makes narration barge-in-aware. **No pipeline elements were added or moved for Phase A** — the gate was already in place; it just grew the drain/interruption bookkeeping.

**Relay pipeline** (display_mode == 'talking') — no local TTS; text is forwarded to the SoulX-Flashtalk avatar bot via Daily app-messages on the `avatar-relay.v1` protocol. **S65b caching does NOT apply here** — SoulX renders its own audio, the `CachedFirstTTSService` isn't in the relay pipeline. Narration still happens (text forwarded; `script_complete` emitted identically — at text-forwarded, no drain-wait: the Phase A v1 punt), per S65 D4.

Pipeline selection is automatic based on the snapshot's `avatar_display_mode`. Falls back to `CLOUD_OUTPUT_MODE` env var when the snapshot can't be fetched.

When the LLM emits a tool call, Pipecat's function-call aggregator invokes the registered handler in `tools/canvas_protocol_tools.py`. The handler builds a wire-format `canvas.command` payload, sends it as a Daily app-message, and awaits the matching `canvas.commandResult` via an `asyncio.Future` (managed by `PendingCommandRegistry`).

---

## System prompt — assembly path

The current live prompt is assembled as a **concatenation**:

```
system_prompt = _assemble_full_prompt(base, manifest)    # bot.py — "\n\n".join of:
  build_system_prompt(room_id, …)            # persona.py — Strategy 1 (base + S65 SCRIPT directive)
  render_canvas_page_section(manifest)       # context/prompt_builder.py — active Page verbs
  render_agent_playbook_section()            # quiz + vision/annotation sequences
  render_voice_output_style_section()        # VOICE OUTPUT — STRICT (plain speech; appended last)
```

Rebuilt at three points: **session start**, **scene change** (via `refresh_agent_for_current_scene`), and **manifest registration** (`canvas.register` branch in `on_app_message`).

### S65 — SCRIPT section directive

S65 added a short SCRIPT directive telling the LLM: *the scene's script is narrated automatically by the system in the script avatar's voice; do not read or paraphrase it; stay silent until the visitor speaks.* The invitation owns the conversational hand-off.

Strategy 1 vs Strategy 2 (in `persona.build_system_prompt`):

- **Strategy 1** (live path): when `room_id` is set, the agent fetches the backend's `/persona-prompt` endpoint and appends audience / knowledge / link narration / canvas tools section / **SCRIPT directive (S65)** / scripts.
- **Strategy 2** (fallback / tests): builds locally from avatar + scene via `build_scene_description`.

LANGUAGE on both ends. PERSONA, AUDIENCE, KNOWLEDGE between. The S64c CANVAS PAGE section sits **outside** the language sandwich (post-hoc append). The S65 SCRIPT directive sits inside the body, before the closing LANGUAGE reminder.

### Voice output style directive (plain speech — Groq/gpt-oss)

`render_voice_output_style_section()` (`context/prompt_builder.py`) appends a blunt **`## VOICE OUTPUT — STRICT`** block as the **last** section of every assembled prompt (recency), via `_assemble_full_prompt` — so it covers **both** pipelines at all three rebuild points. The rule: plain spoken words only — **no Markdown** (`**`/`*`/backticks/`#`/bullets/numbered lists), **no emojis, no ellipses** (`…`/`...`), complete sentences.

**Why:** `gpt-oss-120b` (the default Groq model) is markdown-happy and, on chatty turns, can trail into `…` filler — which surfaced as garbled captions in the wild (`**… **… Oops! Looks …`). This is **mitigation #1**: it curbs both at the **source**, which is what cleans the **caption** — the transcript is forwarded **upstream** of the TTS-side filter, so a TTS filter alone would *not* fix the caption. It pairs with **mitigation #2**, the `MarkdownTextFilter` on the classic TTS (audio-only safety net — see Stack). The relay (`talking`) pipeline has no TTS filter, so there it relies on this directive alone. Guarded by `test_voice_output_style.py`.

---

## Tool naming — underscored, not dotted

The 5 generic tools register with the LLM as:

| LLM-facing name | Wire-format `tool` field |
|---|---|
| `canvas_analyze` | `analyze` |
| `canvas_control` | `control` |
| `canvas_action` | `action` |
| `canvas_set_page` | `set_page` |
| `canvas_annotate` | *(none — non-canvas `agent_annotate`)* |

*(S67c: `canvas_highlight`/`highlight` was retired. `canvas_annotate` is the fifth tool but is **not** a Canvas-Protocol verb — it emits a non-canvas `agent_annotate` message to the shell overlay, off `DailyRelay`. The four real protocol tools still map to dotted wire verbs.)*

OpenAI and Anthropic both validate tool names against `^[a-zA-Z0-9_-]+$` — `.` is rejected. Daily wire-format message `type` fields keep their dots (`canvas.command`, `canvas.sceneChanged`, etc.) — those don't go to the LLM.

---

## Canvas tools — 4 protocol tools + `canvas_annotate` + `generate_quiz_from_knowledge`

`tools/canvas_protocol_tools.py` registers 5 generic canvas tools. **Four** of them (`canvas_analyze`, `canvas_control`, `canvas_action`, `canvas_set_page`) are true Canvas-Protocol tools: each translates aliases → UUIDs, validates against the active Page's manifest, builds a `canvas.command` with a fresh `commandId`, awaits the `commandResult` future over `DailyRelay`. **The fifth, `canvas_annotate` (S67c), is different** — it does NOT touch the protocol/relay; it resolves a target to normalized coords and emits a non-canvas `agent_annotate` message that the shell applies to the S67a annotation overlay (see the S67c section below). `tools/quiz_generation.py` adds the non-canvas `generate_quiz_from_knowledge` (bundled with `set_page` internally).

Each *protocol* canvas handler logs entry, translates aliases to UUIDs, validates against the active Page's manifest (with the `SCENE_NAV_VERBS` exemption), builds the wire payload with a fresh `commandId`, registers a pending future, sends the Daily app-message, awaits the future (6 s default timeout), and returns to the LLM. Errors are returned as tool results (`{error, message, details}`), not raised — a single failed call doesn't break the turn.

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

**(Phase A)** `script_complete` emission rules hardened (frozen wire contract v1 — see *Recent: Auto Play Phase A*): classic-pipeline emissions fire only after **true playout drain** (`BotStoppedSpeakingFrame`, not synthesis-complete); cancelled or **interrupted** runs never emit; ALL narration entry points (session start, scene change, manual replay, autoplay resume) run through the **single-slot** background task.

### Auto-advance signalling (D8)

If `live_room.auto_advance && scene_index < total - 1`: **speak NOTHING** (S77 — the spoken `transition_cue` is gone at its source; a legacy cue in an old snapshot is ignored), hold `SCENE_TRANSITION_PAUSE_MS = 600` of silence after playout drain, then emit `script_complete`. Else: speak `invitation_line`, emit `script_complete` (immediate — no pause on manual/last/script-less paths, frozen wire rule 4). No-script scenes: neither branch (the existing conversational greeting stands; the shell knows `hadScript=false` and won't schedule an advance). **(S65c)** entry narration emits `trigger:'auto'`; a manual Script-button replay emits `trigger:'manual'` and the shell ignores `'manual'` for auto-advance — so a replay never moves the flow.

### Tests

`test_scene_narration.py`: voice resolution picks clone vs fallback; invitation suppressed on non-final auto-advance scene, present on final / non-auto-advance; narration runs once per entry; relay path emits `script_complete`.

---

## Per-line narration language + silent transitions (S77)

Q8's law: **same voice, per-line language.** The backend serves, per script line, `narration_text` (a room-language translation when one is fresh, else the authored base text) + `narration_language`, alongside additive `base_language`/`adaptable`.

- **Parsing (`plan_narration_segments`):** the SPOKEN text is `narration_text`; **legacy fallback** to `text` + room language when the fields are absent (a pre-B4 backend during rollout — remove-by-default is NOT allowed). Segments carry `language`.
- **Cached-audio drop rule (load-bearing):** the S65b cache holds BASE-text renders (its key has no language input) — `audio` is dropped whenever `narration_text != text`, so translated lines synthesize live in the right language; base-text lines keep their cache. Guarded by `test_plan_drops_cached_audio_for_translated_lines`.
- **Switching:** the Cartesia service boots at the ROOM language (`Settings(language=_cartesia_language(snapshot_language))`); a line whose `language` differs gets a `TTSUpdateSettingsFrame(delta=Settings(language=…))` via `_classic_set_language` (the `_classic_set_voice` twin — pipecat 0.0.108 applies it before the next `TTSSpeakFrame`, no reconnect); skipped on cache hits; `_reset_to_primary` restores the room language on EVERY exit path (normal / interrupted / cancelled) so conversational turns never inherit a script line's language. Voice and language are independent `Settings` fields — a language delta never touches the voice. Relay pipeline: no Cartesia TTS ⇒ no switching.
- **Silent transitions (Q10/B6):** the auto-advance followup branch returns `None` ALWAYS; `SCENE_TRANSITION_PAUSE_MS = 600` (narration.py) runs after final-line playout drain, BEFORE `script_complete` — the pause IS the inter-scene gap the visitor hears (the shell's advance follows the emission). Never re-add spoken filler; the transcript-probe test (`test_scene_transitions.py`) pins ZERO transition tokens.
- Tests: `test_narration_language.py` + `test_scene_transitions.py` (285 total).

## Cached narration audio (S65b) — `CachedFirstTTSService`

S65b makes scene-start narration **instant and free** for clone-voiced segments. The agent fetches pre-rendered PCM from R2 (public CDN URL via `media.hv.ai`) and feeds it through the existing TTS frame envelope. Live synthesis is the automatic fallback on miss, pending, error, fallback-voiced segments, or fetch failure.

### `services/cached_first_tts.py`

```python
class CachedFirstTTSService(CartesiaTTSService):
    """If a cached segment has been primed for the NEXT run_tts call, play it from bytes
    in the canonical TTS frame envelope; otherwise fall through to live Cartesia synthesis."""

    def prime_cached(self, segment: Optional[CachedSegment]) -> None: ...
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        seg = self._consume()
        if seg is None:
            async for frame in super().run_tts(text, context_id):    # live path: voice already set by narrator
                yield frame
            return
        # cache hit — emit IDENTICAL envelope CartesiaTTSService would
        yield TTSStartedFrame(context_id=context_id)
        for chunk in _chunk_pcm(seg.pcm, seg.sample_rate, seg.num_channels, ms=20):
            yield TTSAudioRawFrame(audio=chunk, sample_rate=seg.sample_rate, num_channels=seg.num_channels, context_id=context_id)
        # Block 15 — sleep the PCM's playback duration before TTSStoppedFrame
        yield TTSStoppedFrame(context_id=context_id)
```

Wired into `bot.py`'s classic pipeline in place of `CartesiaTTSService`. Same constructor kwargs; the cache-first behavior is opt-in via `prime_cached(...)` — when nothing is primed, the service is indistinguishable from the parent. **(Phase A)** `process_frame` drops an armed prime on `InterruptionFrame`: the narrator primes immediately before each speak, so a barge-in aborting the run between prime and `run_tts` would otherwise leave the prime for the NEXT `run_tts` — i.e. the LLM's reply to the barge-in would play the scene-script PCM instead of the reply. Guarded by `test_interruption_drops_armed_cache_prime` (in `test_autoplay_phase_a.py`).

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
- Direct `OutputAudioRawFrame` emission (rejected alternative) **bypasses** that machine — `BotStoppedSpeakingFrame` would never fire, and the Phase A `NarrationCompletionGate` drain-wait (`expect_playout_drain` resolves on `BotStoppedSpeakingFrame` — the fix that stops auto-advance clipping audio, **shipped 2026-07-16**; earlier CLAUDE.md revisions described it as in-flight before it was built) breaks for cached scenes.
- `test_cached_first_tts.py::test_hit_never_emits_output_audio_raw_frame` guards this invariant. **Do not relax that test.**

### Cross-repo audio invariant (READ BEFORE TOUCHING)

The agent's `config.py` and the backend's `app/config.py` carry **paired** constants. Mismatch = garbled cached playback that "works" locally but breaks where configs drift. Paired keys:

- `NARRATION_TTS_MODEL_ID` (e.g. `sonic-3`).
- `NARRATION_AUDIO_ENCODING` (`pcm_s16le`).
- `NARRATION_AUDIO_SAMPLE_RATE` (e.g. `24000` Hz).
- `NARRATION_AUDIO_NUM_CHANNELS` (`1`).

The `CartesiaTTSService` constructor in `bot.py` is configured with `sample_rate=NARRATION_AUDIO_SAMPLE_RATE` and `model=NARRATION_TTS_MODEL_ID`. The backend's Celery `generate_narration_audio` renders with those exact values. Change one repo without the other and cached `TTSAudioRawFrame`s will play at the wrong rate (chipmunk / slow drone).

There's no automated test that asserts the two repos agree — the gate is convention + the agent's `test_cached_first_tts.py` assertion that `TTSAudioRawFrame.sample_rate` equals `settings.NARRATION_AUDIO_SAMPLE_RATE`.

### Tests

`test_cached_first_tts.py`: HIT emits started/audio/stopped envelope; HIT **never** emits `OutputAudioRawFrame`; MISS delegates to `super().run_tts`; `prime_cached` is consumed once (a stray second `run_tts` after one prime is a miss); chunking aligns to sample boundaries (no clicks); cached PCM carries the configured sample rate.

### What's NOT changed in S65b on the agent side

- Voice resolution logic, S65's per-segment voice switching, the invitation flow, and `script_complete` payload shape are all unchanged. *(S77 later removed the `transition_cue` half of the followup and added per-line language switching — see the S77 section.)*
- The relay (`talking`) pipeline. SoulX renders its own audio — caching doesn't apply.
- The LLM provider branching, eager-dispatch infrastructure, and canvas tools.
- Every existing S65 test continues to pass — the cache path is invisible to its assertions (miss path = identical behavior).

---

## LLM provider selection

`LLM_CANVAS_PROVIDER` env var selects the main LLM at boot — one of `{groq, openai, anthropic, gemini}`, validated in `config.py`. `bot.py:_build_llm_and_eager_hook` branches with **lazy imports** so only the selected provider's SDK needs to be installed. Read once at boot and fixed for the session — no mid-session switching.

**Default is `groq`** — `GroqLLMService(model=GROQ_MODEL)`, `GROQ_MODEL` default `openai/gpt-oss-120b`. Groq subclasses `OpenAILLMService` and speaks the OpenAI-compatible wire format, so:

- It takes the system prompt via `GroqLLMService.Settings(model=…, system_instruction=…)` — same shape as the OpenAI branch.
- It **reuses `OpenAIEagerHook`** (no `groq_adapter.py`) — the OpenAI chunk-shape parser handles Groq's streamed tool calls verbatim.
- Import is `from pipecat.services.groq.llm import GroqLLMService` (the `.llm` submodule; the package `__init__` is a deprecation proxy). **Extra gotcha:** importing it runs the package `__init__`, which eagerly pulls `groq/tts.py` → needs the native `groq` SDK. That SDK ships via the **`pipecat-ai[groq]`** extra in `pyproject.toml`. Both `groq` and `openai` extras are installed, so flipping `groq`↔`openai` needs no reinstall; `anthropic`/`gemini` still need their own extras.

Per-provider model env vars: `GROQ_MODEL`, `ANTHROPIC_MODEL`, `GEMINI_MODEL` (OpenAI uses `LLM_MODEL`).

> **Model-id note:** Groq's catalog lists the 120B GPT-OSS model as **`openai/gpt-oss-120b`** (the bare `gpt-oss-120b` 404s). Override via `GROQ_MODEL`.

> **⚠️ `gpt-oss-120b` is text-only.** Verified: Groq returns `400 — messages[].content must be a string` when sent an OpenAI `image_url` content block. The agent's **S46 vision path injects a scene image into the main-LLM context at session start** (`build_vision_message` → `initial_messages` in `run_bot_classic` / `run_bot_relay`; also on every scene change in `VISION_REFRESH_MODE=eager`), so with the default Groq provider that **first turn 400s whenever a scene image is present**. The decoupled **S67b** vision path is unaffected (it returns Gemini's *text* reasoning as the `canvas_analyze` tool result, never a raw image). **This injection is gated by `MAIN_LLM_SUPPORTS_VISION`** (`config._resolve_main_llm_vision`): default **False** for `groq` — so `build_vision_message` is skipped at session start and in `eager` refresh (both `run_bot_classic` and `run_bot_relay` log the skip) — and **True** for openai/anthropic/gemini. To run Groq *with* in-context vision, point `GROQ_MODEL` at a multimodal Groq model and set `MAIN_LLM_SUPPORTS_VISION=true`. See **Vision** below.

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

**Current limitation:** the hook objects are constructed in `_build_llm_and_eager_hook` but **not yet called by Pipecat's streaming loop**. Result: 0 ms savings in production today. The infrastructure is ready when someone wires `await eager_hook.on_stream_event(chunk)` into the Pipecat LLM service subclass. (Tracked for S75.)

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
- **`{type: 'script_complete', sceneIndex, hadScript, trigger}`** *(S65; `trigger` added S65c)* — emitted after the scene's scripts finish; the shell's auto-advance handler advances only when `trigger === 'auto'`. **(Phase A)** wire shape UNCHANGED, emission rules hardened: classic emits only after **true playout drain** (`BotStoppedSpeakingFrame`); cancelled/interrupted runs never emit; a `resume`-initiated run emits `trigger:'auto'`; script-less scenes still emit `hadScript:false` immediately. Relay still emits at text-forwarded (v1 punt).
- **`{type: 'quiz_generation_state', state, error?}`** *(S65c)* — `state ∈ {generating, ready, error}`. Emitted by the manual `request_quiz` path (via `run_quiz_generation`'s `on_state` hook) so the shell's Quiz button can show a spinner / error. The LLM-tool quiz path does **not** emit these (no button to update).
- `{type: 'avatar_relay.*', …}` (relay pipeline only) — text/turn protocol for SoulX.

**Incoming (frontend → agent):**

- `{type: 'canvas.register', pageType, version, capabilities, semanticState}` → `CanvasManifestRegistry.set_manifest()`; rebuilds CANVAS PAGE prompt section.
- `{type: 'canvas.stateChange', semanticState}` → cached in `CanvasManifestRegistry` for the next `analyze()`.
- `{type: 'canvas.sceneChanged', sceneIndex, sceneId}` → `refresh_agent_for_current_scene()`. **(S66)** `sceneId` drives the cursor-independent by-id snapshot fetch; cursor fallback if absent.
- `{type: 'canvas.commandResult', commandId, result}` / `{type: 'canvas.commandError', commandId, error}` → complete the awaiting Future.
- **`{type: 'request_narrate'}`** *(S65c)* → force re-narration of the current scene with `trigger='manual'` — never auto-advances. **(Phase A)** now runs as the **single-slot background task** instead of inline (the inline await held app-message dispatch hostage for the whole narration — the P3 lesson; worse with the A1 drain-wait), so the emit follows the frozen-contract rules (drain-gated; suppressed on interruption/supersede).
- **`{type: 'request_quiz', count?, language?}`** *(S65c)* → `run_quiz_generation(...)` with the `quiz_generation_state` emitter. Silently ignored if the agent isn't ready yet (no `session_context`).
- **`{type: 'autoplay_control', action: 'stop' | 'resume'}`** *(Phase A — frozen wire contract v1)* — session-level `request_*`-style early-return branch in BOTH pipelines. `stop`: cancel the single-slot narration run + `narration_gate.cancel_all` + **flush queued bot audio** (classic: `InterruptionTaskFrame` → pipeline-wide interruption, confirmed via `expect_interruption`; relay: best-effort `RELAY_INTERRUPT` of the open narration turn). A stopped run never emits `script_complete`. `resume`: fresh **cursor** snapshot (the `request_narrate` template) → force re-narration of the current scene from segment 0 in the single slot → emits `script_complete {trigger:'auto'}` on true completion. Unknown actions are logged and ignored.

**S65c routing rule (important):** the `request_*` / `autoplay_control` branches are **early-return branches in `on_app_message`, alongside but BEFORE the `canvas.*` dispatch** — never inside the canvas/relay path. They're session-level requests, not canvas commands (mirrors the frontend keeping them out of `DailyRelay`). They piggyback on the existing defensive `json.loads` (below).

Defensive JSON parsing in `on_app_message`: if Daily delivers the payload as a string (varies by SDK version), parse before the `isinstance(dict)` check. *(S64d hardening — still in place.)*

---

## Live Room snapshot consumer

The agent fetches `GET /live-rooms/{room_id}/scene-snapshot` on session start and on every scene navigation. **`api_client.get_scene_snapshot` always passes `?include_all_scene_knowledge=true`** — gives the agent a stable flow-knowledge block across navigations. **(S66)** on scene change it also passes **`?scene_id={sceneId}`** (from the `canvas.sceneChanged` payload) so the fetch is cursor-independent and served from the backend's Redis cache; it falls back to the cursor-based fetch when `sceneId` is absent.

The snapshot includes:

- `live_room`: language, persona, recipient_prompt, **`auto_advance`** *(S65)*.
- `current_scene`: name, instruction, display_mode, background_url, **elements (with `id`)**, link_url, link_source, `canvas_page_type`, **`has_script`** *(S65)*, **`narration.{invitation_line}`** *(S65; **S77 removed `transition_cue`**)*, **`scripts[*]`** *(S65 voice + S65b `audio` + **S77 `base_language · adaptable · narration_text · narration_language`** — 12 snake_case keys; read defensively, tolerate absence)*, **`actions`** *(S65c — informational for the agent; the shell consumes it)*.
- `flow_state`: scene_index, total_scenes, scene_ids array.
- `knowledge`: text content (flow scope = aggregated all-scene set) **+ a version token (S66)** the agent uses to skip re-stitching the flow-knowledge prompt section when unchanged.
- `faqs`: per-scope array.

**(S66) Flow-knowledge reuse.** The flow-knowledge block is invariant across scenes in a flow, so the agent holds the assembled prompt section in-session keyed by the snapshot's knowledge version and re-stitches only when it changes. On scene change, only the per-scene bits (instruction, elements, aliases, link, scripts) are rebuilt.

**Don't duplicate snapshot logic locally.** Extend the snapshot endpoint in the backend rather than fetching multiple endpoints. `LiveRoomService.get_scene_snapshot()` is the single source of truth, shared with composition / youtube / quiz Pages.

---

## Vision (S46, S66-lazy)

When the visitor asks "what's on screen?", the agent fetches `/scene-snapshot/image` (a Pillow-rendered base64 PNG) and adds it as a user message. The main LLM does the visual reasoning.

**(S66) `VISION_REFRESH_MODE=lazy|eager` (default `lazy`).** In `lazy` mode the Pillow frame is **not** rendered on every scene change — `refresh_agent_for_current_scene` marks it stale, and `_ensure_vision_frame()` fetches it on demand (a visual question / the first `canvas_analyze` of a scene), caching per scene id. This removes a backend image-render from the hot path of every transition — the biggest agent-side `T_agent` win. `eager` restores per-change rendering (escape hatch).

V2.14 documented Vision as a separately-hardpinned OpenAI service for quality invariance under `LLM_CANVAS_PROVIDER` swaps, but **that separation isn't implemented yet** (the S46 in-context-image path uses the main LLM).

> **⚠️ Text-only main LLM (default Groq).** The S46 path adds the scene image to the **main LLM's** context via `build_vision_message` (session start + `eager` mode). `gpt-oss-120b` is text-only and **400s on image content** (`content must be a string`), so this injection is **gated off** for text-only main LLMs via `MAIN_LLM_SUPPORTS_VISION` (default False for `groq`) — see *LLM provider selection*. The **S67b** capture→Gemini path is unaffected: it returns Gemini's *text* reasoning as the `canvas_analyze` tool result, never a raw image, so it works under any main LLM.

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
GROQ_API_KEY=...                  # main LLM — DEFAULT provider (GroqLLMService)
GROQ_MODEL=openai/gpt-oss-120b            # Groq catalog id (bare gpt-oss-120b 404s); text-only
OPENAI_API_KEY=...                # main LLM when LLM_CANVAS_PROVIDER=openai
ANTHROPIC_API_KEY=...             # main LLM when LLM_CANVAS_PROVIDER=anthropic
GOOGLE_AI_API_KEY=...             # S67b vision (always, decoupled); main LLM when LLM_CANVAS_PROVIDER=gemini
DEEPGRAM_API_KEY=...
CARTESIA_API_KEY=...
DAILY_API_KEY=...
HV_API_URL=http://localhost:3001/api/v1   # backend; passed through runner_args.body in prod
LLM_CANVAS_PROVIDER=groq                  # groq (default) | openai | anthropic | gemini
# MAIN_LLM_SUPPORTS_VISION=true            # opt in only if GROQ_MODEL is multimodal (default: off for groq)

# --- S65b — must match backend exactly ---
NARRATION_CACHE_ENABLED=true
NARRATION_TTS_MODEL_ID=sonic-3            # MUST equal backend (R5 resolved S73: sonic-3 everywhere)
NARRATION_AUDIO_ENCODING=pcm_s16le
NARRATION_AUDIO_SAMPLE_RATE=24000         # MUST equal backend
NARRATION_AUDIO_NUM_CHANNELS=1

# --- S66 ---
VISION_REFRESH_MODE=lazy                  # lazy (default) | eager
```

For Pipecat Cloud testing where the container can't reach `localhost`, point `HV_API_URL` at a publicly-accessible URL.

---

## Deployment — Pipecat Cloud

Production runs on Pipecat Cloud. The Dockerfile is the build manifest. Backend's live-room start endpoint creates a Daily room and registers the Pipecat Cloud agent; the agent boots, reads room metadata, fetches the snapshot, joins via DailyTransport.

`LLM_CANVAS_PROVIDER` (default `groq`), the selected provider's key (**`GROQ_API_KEY`** for the default), **`GROQ_MODEL`**, and the **`NARRATION_*` constants** are set as Pipecat Cloud env vars. Changing the `NARRATION_*` values requires coordinated redeploy with the backend.

**Dockerfile gotcha:** the build uses `pip install --no-cache-dir .` against `pyproject.toml`, but `packages = []` / `py-modules = []` — install pulls only dependencies. Agent code reaches `/app/` exclusively via explicit `COPY` lines. Adding a new top-level directory requires a matching `COPY` line. `services/cached_first_tts.py` lives under the existing `COPY services services`.

---

## Conventions Claude Code should follow

- **Match the installed pipecat-ai version's import paths.** Grep `bot.py` first; module layout shifts between versions.
- **Tool names must satisfy `^[a-zA-Z0-9_-]+$`.** Don't reintroduce dots. Daily wire-format `type` may use dots.
- **Don't add eager-dispatch entries for verbs with required args.**
- **Don't modify `tools/canvas_protocol_tools.py` for new Page types.** The 5 generic tools are page-agnostic; new Page types are accommodated through their manifest.
- **Don't add new Daily message types without coordinating with the frontend.** `DailyRelay` is the sole consumer/producer of `canvas.*` messages; non-canvas messages (`script_complete`, future S65c `request_*` triggers) belong in the shell's general handler.
- **Don't call `api_navigate` from agent-side scene-change handlers.** The frontend's `navigateToIndex` already advances the cursor.
- **Don't add `__init__.py` to `tools/`, `context/`, `services/`** unless you also update the Dockerfile (Python 3.12 implicit namespace packages).
- **Don't ask the LLM to copy large structured blobs between tool calls.** Bundle the side effect (`generate_quiz_from_knowledge` is canonical).
- **Don't ask the LLM to pace UI animations.** Own pacing in the Page (cancellable timers); keep the agent a pass-through. (S65 auto-advance applies the same lesson — pacing lives in the shell.)
- **`canvas_set_page` is all-or-nothing.** Empty `pageInit` ⇒ snapshot wins for BOTH pageType and init.
- **`CachedFirstTTSService` must emit the canonical TTS frame envelope on hits.** Never `OutputAudioRawFrame`. The guard test (`test_hit_never_emits_output_audio_raw_frame`) is non-negotiable — the `NarrationCompletionGate` fix depends on it.
- **`script_complete` emission rules are the frozen wire contract (Phase A).** Emit only after true playout drain (classic); NEVER emit from a cancelled or interrupted run; script-less scenes emit `hadScript:false` immediately. Don't add emit sites outside `_narrate_and_complete` / `_session_start_narration_run` / `_queue_greeting`.
- **Every classic narration run must go through the single slot and call `narration_gate.begin_run()` first.** `begin_run` clears the interruption latch + stale futures; skipping it either lets a stale barge-in kill the new run instantly (no `script_complete`, auto-play stalls) or lets a stale future misalign the TTS-stop FIFO.
- **Flush before starting a new run, and confirm via `expect_interruption()`.** `_flush_bot_audio` awaits the gate observing the `InterruptionFrame` it caused; starting a narration run before that confirmation can get its first segment killed by the in-flight flush.
- **`NARRATION_*` constants are paired with the backend.** Don't change one without the other. Use `NARRATION_CACHE_SCHEMA_VERSION` (bumped on the backend) if you need a one-way invalidation.
- **On a cache hit, don't call `set_cartesia_voice`.** The voice is baked into the bytes; the call is wasted control traffic.
- **Keep agent output plain-spoken (Groq/gpt-oss).** The reply is TTS'd *and* shown as a live caption. The `VOICE OUTPUT — STRICT` prompt directive (`render_voice_output_style_section`, appended by `_assemble_full_prompt`) forbids Markdown / emojis / ellipses; the classic TTS also runs `MarkdownTextFilter`. **The directive — not the filter — is what keeps the caption clean**, because the transcript forwarder sits *upstream* of the TTS filter. The relay (`talking`) pipeline has no TTS filter and relies on the directive alone.

---

## Recent: Auto Play Phase A ✅ (2026-07-16 — truthful playout completion + interruption awareness)

Agent half of the Auto Play work (Phase B — the shell's playback UI — is a separate frontend session; the **frozen wire contract v1** below is shared verbatim between both briefs). Zero backend changes. Fixes two bugs:

**Bug 1 — `script_complete` fired at synthesis-complete, not playout-complete.** Live Cartesia renders several× faster than realtime, so the gate's `TTSStoppedFrame`-keyed future released while seconds of audio were still queued in the output transport — the shell advanced over the tail of the narration. **Bug 2 — narration wasn't interruption-aware.** Visitor speech flushed the audio (VAD → `InterruptionFrame` → `MediaSender.handle_interruptions`) but Cartesia drops the cancelled context's `done`, orphaning the gate future: `_classic_speak` stalled the full 30 s, then **continued to the next segment** — narration resumed mid-scene over the conversation and finally emitted an advance-eligible `script_complete`.

### What landed

- **A1 — final playout drain.** `NarrationCompletionGate` now mirrors the transport's speaking state (`BotStartedSpeakingFrame`/`BotStoppedSpeakingFrame` traverse the gate on the upstream broadcast) and exposes `expect_playout_drain()`. ⚠️ **Per-utterance accounting is load-bearing:** the transport emits `BotStoppedSpeakingFrame` at EVERY utterance boundary (`MediaSender._handle_frame` fires per dequeued `TTSStoppedFrame`), and per-segment gating releases at *synthesis*-complete, so a multi-segment run stacks utterances in the transport queue — the gate counts synthesized utterances (`TTSStoppedFrame` downstream, only when audio frames were seen, mirroring the transport's `_tts_audio_received` guard) against played ones (`BotStoppedSpeakingFrame` upstream, only on a true speaking→quiet transition — the post-flush stray is ignored) and releases the drain only when played ≥ synthesized. Resolving on the *first* `BotStoppedSpeakingFrame` would re-open Bug 1 for every multi-utterance scene (caught in adversarial review before ship). Immediate resolve when already drained covers the cached-playback race. `run_scene_narration` gained `wait_playout` (classic passes `_classic_wait_playout`; relay passes nothing — v1 punt); the per-segment `TTSStoppedFrame` gating is UNCHANGED (it's what overlaps segment N's playout with N+1's synthesis). Timeout budget: `compute_playout_drain_timeout` — sum of known `audio.duration_ms` + margin; **`duration_ms` of 0/null = unknown** (backend dedup edge) → fixed cap. Constants `PLAYOUT_DRAIN_MARGIN_S`/`PLAYOUT_DRAIN_FALLBACK_S` in `narration.py` (agent-only, NOT backend-paired).
- **A2 — interruption awareness.** The gate observes `InterruptionFrame`: resolves ALL pending stop/drain futures with the `NARRATION_INTERRUPTED` sentinel and latches `_interrupted_since_run_start` (an interruption in the between-segments window — no future registered — kills the run's NEXT expect call). `begin_run()` clears the latch + stale futures + utterance counters at the start of every run. Speak/drain callables raise `NarrationInterrupted`; `SceneNarrator.narrate` aborts the loop but **resets the voice to primary on BOTH abort paths — `NarrationInterrupted` AND `asyncio.CancelledError`** (else the LLM's reply renders in the script avatar's clone; the cancel path matters because autoplay stop has no follow-up run to realign the voice); all callers suppress `script_complete`. **`CachedFirstTTSService` drops an armed prime on `InterruptionFrame`** (and `_classic_speak` clears it on the pre-queue abort) — a stale prime would make the LLM's reply to the barge-in play scene-script PCM. The 30 s per-segment `wait_for` stays as a lost-frame backstop only — deliberately NOT reduced, since a long *cached* segment legitimately holds its `TTSStoppedFrame` for the full playback duration (Block 15 sleep).
- **A3 — flush on scene change.** When a scene change cancels an active narration run, `_flush_bot_audio` queues an `InterruptionTaskFrame` (task source → pipeline-wide `InterruptionFrame`; pipecat exposes no narrower bot-audio flush — the deprecated `BotInterruptionFrame` alias was avoided). The flush is **unconditional when a narration run was active** — gating on `bot_is_speaking` missed the Cartesia-TTFB and inter-utterance windows where old-scene TTS is in flight but not yet audible (old audio then played over the new scene AND its stale `TTSStoppedFrame` could misalign the new run's FIFO). It is **confirmed** via `expect_interruption()` (a fresh waiter, immune to the stale latch; tolerates its waiter being cancelled by a concurrent `cancel_all`) before the new run starts. Relay half: best-effort `RELAY_INTERRUPT` of the open narration turn — **ordering rule: cancel the superseded run's task FIRST, then interrupt** (state cleared synchronously so the cancelled task's shielded `close_turn` no-ops instead of closing the NEW run's turn; interrupting before cancelling would let the still-live old run reopen a fresh turn).
- **A4 — `autoplay_control` inbound.** See *Daily app-messages*. Both pipelines, early-return branch. `stop` flushes unconditionally (the pause control means "stop the bot's voice now" — an in-flight LLM reply dies too, accepted for v1). `resume` fetches the snapshot **by the session's tracked scene id** (cursor fallback) so the P2 background-cursor window can't serve the previous scene.
- **A5 — session start in the single slot.** Classic `on_client_connected` narration and the relay `_queue_greeting` now run in `scene_narration_task` with `replace=False` (never stomp a run that raced ahead), so a scene change during the opening narration cancels it. Because that makes session-start cancellable, the **one-time context seeding (presentation-done note vs `GREETING_TRIGGER_PROMPT` + `LLMRunFrame` wake) moved to `_seed_session_context_once`**, called by whichever run completes first — a superseding scene-change run seeds with its own outcome, so a script-less flow still wakes the LLM (an interrupted run marks it done instead: the visitor's own speech drives the LLM, and a belated greeting mid-conversation would be worse). Classic `request_narrate` also moved into the slot (was inline — held app-message dispatch hostage; a replay is now supersedable and drain-gated, `trigger:'manual'` unchanged on the wire).

### Frozen wire contract v1 (shared with the Phase B brief — do not deviate unilaterally)

1. Inbound `{type:'autoplay_control', action:'stop'|'resume'}` (session-level, never rides DailyRelay). 2. `script_complete` shape UNCHANGED; rules: (a) emitted only after true playout drain, (b) cancelled/interrupted runs NEVER emit, (c) resume runs emit `trigger:'auto'`, (d) the shell drops `script_complete` whose `sceneIndex` mismatches its current scene. 3. No new outbound agent→shell messages in v1. 4. Script-less scenes unchanged (`hadScript:false`, immediate).

### Known v1 relay limitations (recorded, not bugs)

No drain-wait (`script_complete` fires at text-forwarded — SoulX owns its playout); narration turns bypass `AvatarRelayProcessor`, so visitor speech does NOT interrupt SoulX narration (only LLM turns); `stop` is best-effort cancel + `RELAY_INTERRUPT`.

### Tests

+27 in `tests/test_autoplay_phase_a.py`: gate drain semantics (pends through `TTSStoppedFrame`, **waits for EVERY synthesized utterance's boundary** — the multi-utterance case, inter-utterance-gap registration, post-flush stray boundary ignored, immediate when truly drained), interruption sentinel + between-segments latch + `begin_run` hygiene + `expect_interruption` freshness, drain-timeout budget (0/null/junk → fallback; blanks skipped), `wait_playout` ordering/skip rules, interrupted-run abort (no emit, no stall, voice reset) + **cancelled-run voice reset** + **prime-drop on interruption**, and slot compositions for stop/resume/supersede — including the real-gate ordering test the pre-Phase-A suite deliberately skipped (see the note at the bottom of `test_cached_first_tts.py`). Existing 248 tests pass unchanged (275 total).

### Lessons (READ BEFORE TOUCHING)

1. **`TTSStoppedFrame` means synthesis done, not audio heard.** Only `BotStoppedSpeakingFrame` (or the cached path's Block-15 sleep) means the visitor finished hearing it. Never re-key `script_complete` on anything upstream of the transport's drain signal.
2. **One `BotStoppedSpeakingFrame` ≠ drained.** The transport fires it at EVERY utterance boundary; since per-segment gating releases at synthesis-complete, later segments + the followup are still queued when the first boundary lands. The gate's synthesized-vs-played counters are the fix — any future "simplification" back to first-frame resolution re-opens Bug 1 for multi-segment scenes.
3. **Interruption must resolve, not cancel, the gate futures.** Cancelling looks identical to a scene-change supersede; the sentinel lets `_classic_speak` distinguish "abort quietly, suppress emit" from "task torn down". But BOTH abort paths must reset the voice and clear the cache prime — stop/supersede cancellation has no follow-up run to clean up after it.
4. **The latch is per-run state.** Forgetting `begin_run()` on a new entry point makes a stale barge-in kill the run instantly → no `script_complete` → auto-play stalls. Conversely, resolving the latch lazily (only at expect-time) is what closes the between-segments window without a lock.
5. **The flush handshake is ordering-critical.** Queue `InterruptionTaskFrame`, await `expect_interruption()`, THEN start the next run — the interruption travels source→sink→source→pipeline and would otherwise race the new run's first `TTSSpeakFrame`. Relay mirror: cancel the superseded task, THEN `_relay_interrupt_narration_turn()` (zeroes turn state synchronously), THEN start — any other order lets the old run's shield-deferred `close_turn` eat the new run's turn, or lets the still-live old run reopen a fresh one.
6. **Known residual race (accepted, v1):** a barge-in whose `InterruptionFrame` is mid-pipeline when `_classic_speak` registers its future can let exactly ONE more segment queue behind the flush and play briefly. Pre-Phase-A the same race replayed ALL remaining segments after a 30 s stall — strictly better now; fully closing it needs transactional frame queueing pipecat doesn't expose.

---

## Recent: P3 latency pass (2026-07-13 — cross-repo scene-switch optimization)

Agent half of the P0–P4 latency pass (backend P0/P1, frontend P2, media P4 — see the backend repo's `docs/benchmarks/scene_switch_2026-07-13.md`). Four changes, no contract change:

1. **Shared httpx client** (`api_client.get_shared_client()`): every helper used to build a fresh `AsyncClient` per call, paying TCP+TLS per backend request (4–6 requests per scene change). One lazy module-level client with keep-alive now serves all calls. Tests that patch `httpx.AsyncClient` must also reset `api_client._shared_client` (see `_patched_client()` in `test_sceneid_by_id_fetch.py`).
2. **Redundant snapshot fetch deleted:** `build_system_prompt` now accepts a pre-fetched `snapshot=` — the refresh closure fetches the post-nav snapshot ONCE and threads it in (it used to be fetched twice per scene change: once inside build_system_prompt, once for session_context). Session start similarly threads its snapshot in (was fetched twice there too).
3. **Session-start fetches gathered:** snapshot + avatar-config + scene-image run under one `asyncio.gather` instead of serially; inside `build_system_prompt`, snapshot + persona-prompt are also gathered when no snapshot is passed.
4. **Narration off the hot path:** the scene-change narration (`run_scene_narration` + `script_complete` emission) runs as a single-slot background `asyncio.Task` instead of being awaited inline in the sceneChanged handler — a long script no longer holds app-message processing hostage. A newer scene change cancels the previous narration task; a cancelled task does NOT emit `script_complete` (the superseding nav already moved the shell). Relay: the `RELAY_TURN` close is `asyncio.shield`ed so cancellation can't strand a dangling turn. Both disconnect handlers cancel the slot.

**Deliberately NOT built:** agent-side adjacent-scene prefetch. The backend's warm-on-navigate (P1) pre-builds the k1 snapshot variant + persona for target and target+1, so the agent's by-id fetch is already a Redis hit — an agent prefetch would only add load.

---

## Recent: S66 complete (Flow Scene-Switching Performance — agent hot-path cuts)

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

## Recent: S67b ✅ — Agent Vision of the Annotated Canvas

The agent has eyes on **what the visitor actually sees** — the live scene plus the S67a annotations in real pixels — for visual Q&A, "what am I pointing at?", and fill-in-the-blank assessment. The agent has **no DOM**, so it **cannot call `getDisplayMedia`**; capture happens in the **visitor's browser** (the shell), triggered over a **non-canvas Daily round-trip**. **The image does NOT travel over Daily** — `sendAppMessage` caps at **4 KB**, so the shell uploads the JPEG to a tiny **backend Redis ingest** and the agent fetches it (the Daily messages carry only `{captureId, status, w, h}`):

```
agent: request_canvas_capture(hint)  ──Daily(<4KB)──▶  {type:'request_canvas_capture', captureId, hint}
        │  (general non-canvas handler in the shell — NOT DailyRelay)
        │  shell: Screen Capture API → grabFrame → downscale → JPEG (Blob)
        │  shell ──HTTPS POST raw bytes──▶ backend Redis  vision:capture:{slug}:{captureId}  (TTL 60s)
        ▼
shell: {type:'canvas_capture_result', captureId, status:'ready'|'error', w, h}  ──Daily(<4KB)──▶  agent
        │  resolve the awaiting asyncio.Future by captureId (sibling registry _pending_captures)
        ▼
agent: api_client.get_vision_capture(slug, captureId)  ──HTTPS GET──▶ bytes (delete-on-read)
        ▼
agent: vision_client.analyze_image(image_bytes, mode, scene_context) → reasoning
        ▼
agent: handle_analyze RETURNS the reasoning as the canvas_analyze TOOL RESULT (in-band):
        {answer:"[vision: …] …", page_state:<iframe analyze state>}   ← NOT an out-of-band developer message
```

- **`services/vision_client.py`** — dedicated Gemini multimodal client (lazy Google-SDK import). `VISION_MODEL` env, default **`gemini-3.5-flash`**; **decoupled from `LLM_CANVAS_PROVIDER`**; stubs gracefully when `GOOGLE_AI_API_KEY` is unset. `analyze_image(image_bytes, mode, scene_context)` with modes `point | assess | describe`; **`locate` added in S67c** (below).
- **Capture round-trip** — `request_canvas_capture(hint)` emits the non-canvas request with a fresh `captureId`, registers an `asyncio.Future` in `_pending_captures`, awaits `canvas_capture_result` (timeout `VISION_CAPTURE_TIMEOUT_MS` ~4000); the result carries `status`/`w`/`h` only — **not the image**. Handled in an early-return `on_app_message` branch (before the canvas dispatch), never via the relay.
- **`api_client.get_vision_capture(slug, captureId)`** — HTTPS GET of the shell-uploaded JPEG from the backend ingest (backend deletes on read).
- **Vision delivery (in-band)** — the orchestration (`run_vision_query`, wired through `ensure_vision`) does capture-first → on `ready` fetch bytes → `vision_client`; on timeout/error/permission-denied → **Pillow base-scene fallback** + a blind-spot flag so the reply admits it can't see drawings. It **RETURNS** the reasoning text; `handle_analyze` folds it into the **`canvas_analyze` TOOL RESULT** (`{answer, page_state}`) — **NOT** an out-of-band `developer` message (the old `context.add_message` path raced the function-call re-run; see the in-band lesson below).
- **Config** — `VISION_MODEL`, `VISION_CAPTURE_TIMEOUT_MS`, advisory `VISION_MAX_DIM`/`VISION_JPEG_QUALITY`. **Dockerfile** unchanged (`COPY services services`); the Google SDK is a base dep regardless of `LLM_CANVAS_PROVIDER`.

---

## Recent: S67c ✅ — Canvas-Interaction unified onto the New Annotation Tools

The agent's canvas-interaction **no longer renders inside the iframe**. The Old `canvas_highlight` tool is **retired**; the agent now draws on the **same S67a shell overlay** the visitor uses and S67b sees — one annotation system, not two.

**`canvas_annotate` (replaces `canvas_highlight`).** LLM-facing params: `op ∈ {circle, arrow, shape, highlight, text, erase}`; `target ∈ {element:<alias>} | {region:{x,y,w,h}} | {describe:<text>}` (required unless `op=erase`); `shape` (when `op=shape`); `text` (when `op=text`). The handler resolves the target → **normalized 0..1 coords** (C2), builds op(s), and emits a non-canvas `agent_annotate` — it does **not** go through the Canvas Protocol / `DailyRelay`.

```
canvas_annotate(op, target, …)
   ▼  target resolution (agent-side):
     {element:'title'} → _element_box_from_snapshot(alias)   # snapshot 1280×720 geometry ÷ (1280,720)
     {describe:'the actor'} → get_vision_image('locate') bytes → vision_client.locate() → {x,y,w,h}
     {region:{…}} → passthrough
   ▼  send_non_canvas({type:'agent_annotate', annotateId, ops:[{op, box, …}]})   # general handler, NOT relay
   ▼  await agent_annotate_result {annotateId, ok} in _pending_annotates  (timeout AGENT_ANNOTATE_TIMEOUT_MS ~2000 → best-effort ok)
   ▼  tool result: "Drew circle on the canvas." / "Couldn't locate that on screen…"
```

- **`vision_client.locate(image_bytes, description)`** — returns a normalized `{x,y,w,h}` for a described target (Gemini object localization; parses `[ymin,xmin,ymax,xmax]`/1000 → 0..1), or `None` on a miss → the handler asks the visitor to point/clarify. Stub-degrades when the key is unset. This is what makes "circle the actor" work on a YouTube/arbitrary scene with no element geometry.
- **`_pending_annotates`** — sibling `asyncio.Future` registry (keyed by `annotateId`), mirroring `_pending_captures`. **`agent_annotate_result`** is resolved in an early-return `on_app_message` branch (alongside `canvas_capture_result`, before the canvas dispatch).
- **Element targets** resolve agent-side from the snapshot's 1280×720 element geometry (composition); the shell is a pure renderer of normalized ops.
- **Removed:** the `canvas_highlight` tool + handler, its PLAYBOOK guidance, the highlight verb from the manifest expectations, and the highlight validation tests. **Canvas-Protocol manifest reduced to v0.2** (envelope/handshake unchanged).
- **Tests** in `tests/test_canvas_annotate.py`: element target → resolved bbox → `agent_annotate` emitted; `describe` → `vision_client.locate` → coords; `region` passthrough; ack resolves / timeout → best-effort ok; `erase`; `locate` stub when key unset; `agent_annotate_result` handled in the non-canvas branch; **regression: `canvas_highlight` is no longer registered**.

> **Lessons / invariants (S67b + S67c):**
> 1. **Vision is decoupled from the main LLM.** `gemini-3.5-flash` runs vision (`analyze_image` + `locate`) regardless of `LLM_CANVAS_PROVIDER`; the Google SDK is a base dep.
> 2. **The image never rides Daily** (4 KB). Capture transits the backend `vision-capture` ingest; only `{captureId,status,w,h}` goes over the data channel.
> 3. **`canvas_annotate` is NOT a Canvas-Protocol tool.** It resolves coords agent-side and drives the shell overlay via non-canvas `agent_annotate` — `DailyRelay` stays `canvas.*`-only. Don't route it through the relay.
> 4. **Pixels are ground truth; geometry is convenience.** Element targets use snapshot geometry (deterministic, composition); described targets use vision `locate` (works anywhere). Same op stream either way; agent marks land on the same overlay and are re-seen by the next capture.
> 5. **Best-effort ack.** A missing `agent_annotate_result` within the timeout is treated as success; only an explicit `{ok:false}` reports a problem to the LLM.
> 6. **Vision is delivered IN-BAND as the `canvas_analyze` tool result** (`{answer, page_state}`), not an out-of-band `developer` message. The old out-of-band `context.add_message` injection raced the function-call re-run (pipecat seeds an `IN_PROGRESS` tool placeholder, and the spoken reply was sometimes generated before the developer message landed) → intermittent "I can't see" replies despite an accurate `[vision: …]` in the logs. Returning the reasoning as the tool result guarantees it's in the re-run context: `ensure_vision` returns the text and `handle_analyze` folds it via `_merge_analyze_result` (vision leads as `answer`; the iframe page state rides along as `page_state`). Guarded by `test_canvas_analyze_vision_result.py`.

---

## CI (S72)

`.github/workflows/ci.yml` **at the repo root** (the project lives in the `pipecat-agent/` subdirectory — the workflow sets `defaults.run.working-directory`). Jobs **`lint` · `test`** — ⚠️ **the branch-protection contract (C8)**: main requires both; renaming one leaves the old required context stuck on "Expected" and jams every PR until protection is updated in lockstep. (The C5 revisit ADDED lint — a passing ruff config existed.)

- `lint`: `uvx ruff@0.15.7 check .` — check-only (ruff is not a dev dependency; `ruff format --check` is dirty today → S75 hygiene).
- `test`: `uv sync --frozen` → `uv run pytest -q` — 275 tests, `live` marker excluded by pyproject addopts, python 3.12 pinned in the workflow (no committed interpreter pin).
- **This repo is PUBLIC — C9 hard rule: the workflow carries ZERO env and ZERO secrets** (the suite passes under `env -i`). Nothing here may ever name a provider key.

Local parity (from `pipecat-agent/`): `uvx ruff@0.15.7 check .` · `uv run pytest -q`.

## Production (S73 — live since 2026-07-23)

- **Pipecat Cloud** (us-west, org `varying-tiger-jade-219`): agent **`human-virtual-agent`**, image **`haiolli/human-virtual-agent:0.9`** built at repo SHA `23166735af18` (A9 pin), profile `agent-2x`, scaling 0–10 (`min_agents=0` ⇒ ~19 s cold join measured at B9; **S74 E9 decision: keep `min_agents=0` through S74–S75, flip to 1 at S76 launch**).
- **Secret set `human-virtual-agent-secrets` = 17 names** (S73 surgery): rotated provider keys + `HV_API_URL=https://api.hv.ai/api/v1` (env FALLBACK — the runner-body `hv_api_url` from the backend's start-session remains primary) + explicit `NARRATION_TTS_MODEL_ID=sonic-3` / `NARRATION_AUDIO_SAMPLE_RATE=24000` / `NARRATION_AUDIO_NUM_CHANNELS=1` + `VISION_REFRESH_MODE=lazy`. **Removed as stale:** `HV_API_TOKEN` (expired JWT), `DAILY_API_KEY` (dead code on cloud — creds arrive via runner args), `PIPECAT_CLOUD_API_KEY` (never read here), `DEFAULT_ROOM_ID`/`DEFAULT_SCENE_ID`.
- **R5 resolved: `sonic-3`** across backend env, this repo's code default, and the cloud secrets — prod cache byte-verified (209,252 B ≈ 4.359 s × 24 kHz × 16-bit mono). This doc's old `sonic-2` lines were the drift (fixed in PR #4).
- **Deploy procedure (as executed at B9):** `docker build --platform=linux/arm64 -t haiolli/human-virtual-agent:<tag> .` → `docker push` → bump `pcc-deploy.toml` → `pcc deploy --force`. Explicit-COPY Dockerfile is the manifest.
- Ops: `human-virtual-backend/docs/deploy/PRODUCTION_RUNBOOK.md` · evidence: `human-virtual-backend/guidelines/SESSION_73_COMPLETION.md`.

## Coming next

**S68 (External Embeds)** and **S69a (Generation Engine)** both shipped with **zero agent changes** — embeds ride the snapshot's `link` block, and generated flows are indistinguishable from hand-built ones (narration S65, fast switching S66, and the visual-interaction stack S67a/b/c all key off the snapshot, not how the scene was authored). The **P3 latency pass (2026-07-13)** shipped (shared httpx client, narration off the hot path). **Auto Play Phase A (2026-07-16)** shipped the agent half of the Auto Play work; **Phase B (the shell's playback UI) is frontend-only — zero agent changes expected**, and both sides build against the frozen wire contract v1 (see *Recent: Auto Play Phase A*; don't change the contract unilaterally). **Next agent-relevant work: none until MCP E2E (S71).** Roadmap: **S69b (/hv Prompt Orchestrator + Create-with-AI studio UI** — also zero agent changes; a `/hv`-created room is indistinguishable from a modal-created one), then MCP (S70–71). **S72 CI/CD ✅ (2026-07-20** — agent CI live, see the CI section**)** · **S73 Production Deployment ✅ (2026-07-22/23)** — this agent is live on Pipecat Cloud at image 0.9 with the rotated secret set (see Production above); credential rotation executed (ledger in the backend completion file). **S74 Monitoring ✅ (2026-07-25)** — zero agent code (the Sentry ×3 is backend/mcp/frontend); **E9 decision: keep `min_agents=0`, flip to 1 at S76 launch.** Next: S75 Hardening, Hygiene & Performance · S76 Launch. Real video export stays post-launch (P1).

---

## Out of scope

- Mid-session provider switching (`LLM_CANVAS_PROVIDER` fixed at boot).
- Persistent iframe shell (per-scene unmount + keyed remount is current; S66's optional prewarm double-buffer was **deferred to S75** — Blocks 1–5 hit the < 1 s target without it).
- A/B testing infrastructure for comparing providers in production.
- Eager-dispatch-to-Pipecat-streaming-loop wiring (hooks constructed; never invoked; tracked for S75).
- **Caching narration audio for the relay (`talking`) pipeline.** SoulX renders its own audio — `CachedFirstTTSService` is only in the classic pipeline. Per-script-avatar voice in relay is the same v0.2 punt.
- **Caching fallback (room-primary-voiced) segments.** Room-dependent → requires room-scoped keys → v2.
- **Cache warming on publish, edge/multi-region warming, Opus/OGG encoding to cut R2 size.** All v2 considerations.

## Staging & Promotion (S74b — live since 2026-07-30)

- **Staging agent:** `human-virtual-agent-staging` (Pipecat Cloud us-west), deployed from **`pcc-deploy.staging.toml`** — always pinned to **the SAME image tag production runs**; secret set `human-virtual-agent-staging-secrets` = the prod set verbatim with ONE delta (`HV_API_URL=https://api.staging.hv.ai/api/v1`). The §7/NARRATION pairing law spans tiers — the sonic-3 block is byte-identical in both sets.
- **Promotion model — the agent has NO `production` branch.** It promotes by image tag: build/push a new tag → deploy to STAGING first (`pcc deploy --config-file pcc-deploy.staging.toml`) → verify via the staging smoke → then `pcc deploy` with the prod toml at that same tag (the backend `scripts/promote_production.sh` tail prints this reminder).
- The staging backend summons this agent via `PIPECAT_AGENT_NAME=human-virtual-agent-staging` (settings-backed on the backend since S69-era config; the staging Railway matrix carries the value).
- Ops detail: backend `docs/deploy/PRODUCTION_RUNBOOK.md` §11; evidence: backend `guidelines/SESSION_74B_COMPLETION.md`.

## S78 (2026-08-13/14): zero changes

Live-room access control + greet-by-name shipped entirely backend/frontend/mcp-side. The greet block rides the backend persona prompt (`## Audience`, appended last — adjacent to this repo's own AUDIENCE section in the composite; verified LLM-generated greeting, no hardcoded literal, at the S78 A5 gate). The access gate covers only the by-slug room lookup and start-session — neither of which this agent calls — so it cannot be stranded; the by-uuid and slug-scoped surfaces it consumes stay public. Record: backend `guidelines/SESSION_78_COMPLETION.md`.

## The combined S77+S78 GO (2026-08-14/15): image 0.10 — the S77 agent code's first deploy

The GO's discrepancy check found BOTH tiers still on `0.9` (staging untouched since 2026-07-29) — the S77 agent code was merged but **never built into an image**; S77's staging narration "verification" had heard the base-text fallback, not translations. Remediation at the GO: `haiolli/human-virtual-agent:0.10` built from main `87b8421` (arm64, digest `sha256:144bc7…`) → staging deploy → **the real translation ear-check (PASS — room-language narration heard for the first time)** → prod deploy. Both tomls now pin `0.10`; prod scaling min 1 / max 10 (the S76 warm floor) survived. Record: backend `guidelines/SESSION_78_COMPLETION.md` §H.

## S79 (2026-08-15/16): the animated-narration cue path — images 0.13→0.16, STAGING-deployed (prod at the S79 GO — HOLD)

S79 gives narration lines pre-rendered SoulX MP4 clips (backend-rendered, R2-cached); this repo cues them onto the shell tile instead of speaking them. PR #18 (the build, image **0.13**) + the field wave #19/#20/#21 (**0.14/0.15/0.16** — one FRESH tag per build, the same-tag law: rebuilt tags never reach running pods). Both tomls pin **0.16**; prod runs **0.10** until the S79 GO (blocked by backend D-3). Record: backend `guidelines/SESSION_79_COMPLETION.md`.

- **`services/narration_cue.py` — `NarrationCueController`**: `cue()` sends `narration_segment {sceneId, lineIndex, url, durationSeconds}` (snake type, camel fields — the v0.3 wire law) and waits for the shell's `script_complete` (stale `lineIndex` ignored; absent is permissive). Timeout (duration + margin) ⇒ `narration_cancel` + TTS fallback — **never-block**. **Barge-in ⇒ `narration_cancel` + `NarrationInterrupted`** — the run ABORTS (rule 2b: interrupted runs never emit `script_complete`), the line is deliberately NOT TTS'd, and there is NO auto-resume (§2.6 amended 2026-08-16; the pause/resume machinery was deleted in #19 — `narration_pause`/`resume` stay dormant wire verbs).
- **`narration.py`**: `NarrationSegment.animation` rides the plan (degenerate shapes drop to None); `SceneNarrator` routes animated lines to `cue`, voice lines to `speak` — **no cue wired ⇒ byte-alike pre-S79 behavior** (diff-zero lock). `plan_continue_action(narration_completed, scene_index, total_scenes)` is the spoken-continue rule: unfinished ⇒ `resume` (restart from segment 0), finished+next ⇒ `next_scene`, else `end` — vector-locked.
- **`bot.py` — `continue_presentation`** (both pipelines): arms `nav_guard_until` (+3 s) FIRST; resume sends `autoplay_state {mode:"playing"}` (lifts the shell's transcript-triggered suspension so post-resume auto-advance proceeds — Hai's ruling B) then restarts narration; resume/advance results carry `FunctionCallResultProperties(run_llm=False)` (silent — the double-audio fix); end/error keep the LLM turn. Eager dispatch suppresses SCENE_NAV verbs when the turn contains `continue_presentation`; a guarded `canvas_control` nav gets the corrective `NAV_SUPPRESSED` result (the double-action fix). Classic-pipeline supersede is INLINE (`_cancel_active_narration_run` is relay-scoped — F821 if referenced in classic scope).
- **`services/soulx_audio.py`**: `expect_interruption()` futures (the relay barge-in signal, resolved on `InterruptionFrame`); `wait_playout_completed` (truth → sent-audio estimate `PLAYOUT_EST_MARGIN_S=2.0` → budget) — the drain-wait that predated the session.
- **Tests**: `test_narration_cue.py` (cue wire casing, stale/permissive completion, timeout fallback, barge-in abort law, diff-zero) + `test_continue_nav_guard.py` (eager suppression matrix, guard window) — the `tests/` convention (no pytest-asyncio; `asyncio.run` per test).

## S83 (Canvas Actions) — two inbound types, image 0.17

PR-6's whole footprint: **`cta_completed {}`** and **`handoff_state {state}`** handled in BOTH routers (early-return before `canvas.*`), plus the pure helpers in `narration.py` (`take_cta_ack` · `apply_handoff_state` · `narration_allowed` · the two resolvers — the `request_quiz_ready` extraction precedent; contract-mirror tests in `tests/test_cta_handoff_messages.py`).

- **Snapshot-only speech law:** both payloads carry NO speakable field. `cta_completed` speaks root `target_action.completed_ack_line`; handoff-open speaks `live_room.handoff.ack_line` — a visitor cannot puppet the twin. Both acks are TTS-DIRECT (classic: bare `TTSSpeakFrame`; relay: `_narration_speak` + `_relay_close_turn`) — the LLM-ack route is the documented double-audio bug.
- **Once per session, literal:** each ack's guard flips on FIRST receipt, line or not.
- **The handoff hold (h6-B/C/D):** `handoff_quiet` is declared beside `scene_narration_task` in both scopes and gates `_start_narration_task` itself — one check covers scene-entry, session-start, manual replay, and autoplay resume. Open also cancels the active run (classic: cancel + `narration_gate.cancel_all` + `_flush_bot_audio`; relay: `_cancel_active_narration_run` + `_relay_interrupt_narration_turn`). 'closed' lifts the hold and NEVER auto-resumes (the shell's Play control owns resumption).
- `canvas_page_type='cta'` needs no tolerance work: `scene_context.py`'s neutral unknown-page-type fallback (post-S64d) already covers it, and `api_client`'s setdefaults tolerate the additive snapshot keys.
- Image **0.17** pinned in BOTH tomls; deploy = build + `pcc deploy` staging-first (★ operator).

### PR-13 addendum — `user_text` (image 0.18)

The panel's typed question (P-11's deferred half, lifted by Hai): `user_text {text}` in BOTH routers queues the VAD trio (`UserStartedSpeaking → Transcription → UserStoppedSpeaking`) — byte-equivalent to speech: same aggregator→LLM→TTS turn, same barge-in interruption, and the existing `TranscriptForwarder` echoes the bubble to the shell (no second echo path). Validation in `tools/user_text.py` (strip · 500-char cap · silent ignore on blank/non-string); tests `tests/test_user_text.py`.

### S83 close (2026-08-23) — image 0.19 on staging; prod stays 0.16

Hai built/pushed/deployed the #25 code as tag **0.19** (0.18 was pinned
but never pushed — the fresh-tag law absorbed the skip; PR #26 trued
both tomls to 0.19). The S83 agent footprint stayed exactly PR-6 +
PR-13: two inbound message types + `user_text` — the rest of the Canvas
Actions arc (types retired D-15, verify dismissed D-16, link-only
checkout D-19, Stage+Dock D-20) landed with ZERO further agent changes.
**Prod runs 0.16 until the S83 GO** (image promotion 0.16 → 0.19 rides
it; platform-verify BOTH tiers post-deploy — the S78 D-10 lesson).
Session record: backend `guidelines/SESSION_83_COMPLETION.md`.
