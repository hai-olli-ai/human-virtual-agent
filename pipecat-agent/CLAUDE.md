# pipecat-agent — CLAUDE.md

> Voice agent for **Human Virtual** (hv.ai), built on the **Pipecat Framework**.
> **Status:** S64c + S64d + S64e shipped, plus a round of post-S64e hardenings driven by real-session bug reports — `generate_quiz_from_knowledge` bundles the canvas Page swap (Option D); the Quiz Page owns its post-answer reveal/auto-advance timing instead of the agent racing it; a `skip_question` action verb handles "I don't know"; `canvas.set_page` override is all-or-nothing (empty `pageInit` means "restore snapshot — both pageType and init"); scene-nav verbs (`next_scene` / `previous_scene` / `goto_scene`) are exempt from the per-Page manifest validator because they're shell-level. **S65 (Live Room Script Narration + Flow Auto-Advance) is next.**
> **Repo:** `pipecat-agent/` (lives alongside `human-virtual-backend/` and `human-virtual-frontend/`).

---

## What this service does

The Pipecat agent runs the avatar's **conversation pipeline**: STT → LLM → TTS → audio out, with optional vision and canvas tool calls. It connects to a Live Room over WebRTC and:

1. Listens to the visitor's microphone (Deepgram STT).
2. Calls an LLM (OpenAI / Anthropic / Gemini, selected at boot via `LLM_CANVAS_PROVIDER`) with a system prompt assembled from the Live Room's persona, knowledge (aggregated across the whole flow), current scene instruction, current scene's elements (referenced by short aliases the LLM can actually copy), and the active Canvas Page's manifest.
3. Emits TTS audio (Cartesia) plus avatar lip-sync data (SoulX-Flashtalk for "talking" display mode).
4. Calls **canvas tools** via the generic 5-tool protocol surface (`canvas_analyze`, `canvas_highlight`, `canvas_control`, `canvas_action`, `canvas_set_page` — **underscored, not dotted**; see "Tool naming" below).
5. (S46+) Calls **vision** by adding a snapshot image to the LLM's context. Currently the main LLM does the visual reasoning — a separate hard-pinned vision service is described in V2.14 but not yet implemented in the agent.
6. (S49) Plays the scene's script on entry and emits `{type: 'script_complete'}` when narration finishes. **S65 extends this with script-avatar voice switching + a post-script invitation + flow auto-advance signalling.**

---

## Stack

| Layer | Choice |
|---|---|
| Framework | Pipecat (Python, pinned at 0.0.108 in `pyproject.toml`) |
| Transport (prod) | DailyTransport (Daily.co WebRTC) |
| Transport (local) | SmallWebRTCTransport |
| Deployment (prod) | Pipecat Cloud |
| LLM default | OpenAI GPT (via `LLM_CANVAS_PROVIDER=openai`) |
| LLM alt | Anthropic Claude (requires `pipecat-ai[anthropic]` extra), Google Gemini (requires `pipecat-ai[google]`) |
| LLM selection | `LLM_CANVAS_PROVIDER` env var, validated against `{anthropic, openai, gemini}` at boot |
| STT | Deepgram |
| TTS | Cartesia |
| Lip-sync | SoulX-Flashtalk (S48 — relay pipeline, `display_mode == 'talking'`) |

**LLM default note:** V2.14 documents Anthropic as the target default, but the installed `pyproject.toml` ships only `pipecat-ai[…openai…]`. `config.py` defaults `LLM_CANVAS_PROVIDER=openai` because importing `pipecat.services.anthropic.llm` without the anthropic extra crashes at boot. To flip the default to Anthropic, add `anthropic` to the pipecat extras in `pyproject.toml`, `uv sync`, then change the `os.getenv("LLM_CANVAS_PROVIDER", "openai")` fallback in `config.py`.

---

## Project structure

```
pipecat-agent/
  bot.py                            # Main entrypoint — pipeline assembly, transport setup, LLM provider branching
  config.py                         # Settings (env vars), provider validation, model defaults
  persona.py                        # build_system_prompt — assembles the live system prompt (Strategy 1 = backend persona-prompt endpoint + agent additions)
  scene_context.py                  # Section helpers, knowledge formatting, build_canvas_tools_section, alias generator
  api_client.py                     # HTTP client for the backend (snapshot, persona-prompt, navigate, scene image, generate-quiz, …)
  context/
    prompt_builder.py               # Alternative split-prompt builder (build_system_prompt_split) + render_canvas_page_section (the bit that IS wired into the live prompt)
    canvas_manifest.py              # CanvasManifestRegistry — in-memory holder for the active Page's manifest
  tools/
    canvas_protocol_tools.py        # 5 generic protocol tools (S64c) + dispatch_canvas_command (module-level since S64e); alias translation; SCENE_NAV_VERBS exemption from manifest validation
    quiz_generation.py              # S64e — generate_quiz_from_knowledge tool. Bundled set_page dispatch (Option D); SessionContext (slug + current_scene_id) for backend lookup
  services/
    eager_dispatch/                 # Per-provider streaming hooks (S64c) — instantiated but NOT wired into the LLM streaming loop yet
      __init__.py                   # EAGER_DISPATCH_VERBS, PendingCommandRegistry, verb-detection regex
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
    test_submit_answer_bundling.py  # S64e — guards "submit_answer is pass-through; advancement is the Page's job"
    bench_canvas_latency.py         # Manual structural benchmark (not pytest-collected)
  docs/
    benchmarks/
      canvas_latency_2026-05-11.md  # S64c eager-dispatch baseline
  README.md
  Dockerfile                        # Explicit-allowlist COPY; new top-level dirs MUST be added or Pipecat Cloud builds break silently
```

The Dockerfile uses explicit `COPY <file> .` lines per top-level file/directory. Adding new code directories (e.g. a future `services/billing/`) requires a matching `COPY <dir> <dir>` line, or the cloud container will hit `ModuleNotFoundError` at session start. See the `.dockerignore` for what's intentionally excluded.

---

## Pipecat pipeline (high level)

**Classic pipeline** (display_mode ∈ {normal, invisible, 3dgs}):

```
DailyTransport.input  →  Deepgram STT
                      →  TranscriptForwarder
                      →  LLMContextAggregatorPair (user side)
                      →  LLM (the 5 canvas_* tools registered)
                      →  ThinkingNotifier / TranscriptForwarder / SpeakingStateNotifier
                      →  Cartesia TTS
                      →  DailyTransport.output
                      →  LLMContextAggregatorPair (assistant side)
```

**Relay pipeline** (display_mode == 'talking') — no local TTS; text is forwarded to the SoulX-Flashtalk avatar bot via Daily app-messages on the `avatar-relay.v1` protocol. SoulX renders speech + avatar video into the same Daily room.

Pipeline selection is automatic based on the snapshot's `avatar_display_mode`. Falls back to `CLOUD_OUTPUT_MODE` env var when the snapshot can't be fetched.

When the LLM emits a tool call, Pipecat's function-call aggregator invokes the registered handler in `tools/canvas_protocol_tools.py`. The handler builds a wire-format `canvas.command` payload, sends it as a Daily app-message, and awaits the matching `canvas.commandResult` via an `asyncio.Future` (managed by `PendingCommandRegistry`).

---

## System prompt — assembly path (post-S64c+manifest wiring)

The current live prompt is assembled as a **concatenation**, not via `build_system_prompt_split`:

```
system_prompt = build_system_prompt(room_id, …)             # persona.py — Strategy 1
              + "\n\n"
              + render_canvas_page_section(manifest)        # context/prompt_builder.py
```

`build_system_prompt` (in `persona.py`) returns the base — LANGUAGE + persona + audience + knowledge + link narration + canvas tools section + scripts + LANGUAGE close. `render_canvas_page_section` then appends the manifest-driven verb list for the active Page (or a "no page registered yet" stub at session start before the iframe registers).

The prompt is rebuilt at three points:

1. **Session start** (`bot.py:run_bot_classic` and `run_bot_relay`, before `_build_llm_and_eager_hook`). Manifest is empty here, so the appended section is the stub.
2. **Scene change** — `refresh_agent_for_current_scene()` awaits `build_system_prompt(…)` to capture the new scene's elements/ids/knowledge, then re-appends the current manifest's CANVAS PAGE section. Both `base_system_prompt` and the LLM service's `system_instruction` are reassigned.
3. **Manifest registration** (`on_app_message` `canvas.register` branch). The base is unchanged; only the appended manifest section is recomputed from the freshly-registered manifest.

`build_system_prompt_split` exists in `context/prompt_builder.py` (returns `(stable_prefix, dynamic_suffix)` for explicit Anthropic `cache_control`) but is NOT used by the live path. Treat it as documentation of the eventual cache-aware structure; the current concatenation works and ships.

### Strategy 1 vs Strategy 2 (in `persona.build_system_prompt`)

- **Strategy 1** (live path): when `room_id` is set, the agent fetches the backend's `/persona-prompt` endpoint and uses its return as the persona prompt body. The agent appends audience / knowledge / link narration / canvas tools section / scripts. Strategy 1 does NOT call `build_scene_description` — element descriptions are inside `build_canvas_tools_section`'s "Available canvas elements" listing (alias-based).
- **Strategy 2** (fallback / tests): builds locally from avatar + scene via `build_scene_description`. Used when there's no `room_id` or the persona-prompt endpoint fails.

### Sandwich pattern (locked)

LANGUAGE on both ends. PERSONA, AUDIENCE, KNOWLEDGE between. The S64c CANVAS PAGE section sits OUTSIDE the language sandwich (appended after the closing LANGUAGE reminder) — a deliberate post-hoc append rather than a section reordering, so the existing prompt-builder doesn't need restructuring to support it.

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

The underscore form is required because **OpenAI and Anthropic both validate tool names against `^[a-zA-Z0-9_-]+$`** — `.` is rejected. The dotted form (`canvas.X`) appears throughout V2.14 design docs but does NOT work as an actual tool name; it would 400 the first LLM request.

**Daily wire-format message types keep their dots:** `canvas.command`, `canvas.commandResult`, `canvas.commandError`, `canvas.register`, `canvas.stateChange`, `canvas.sceneChanged`. These travel between Pipecat ↔ frontend (Daily app-messages), never to the LLM provider, so the regex doesn't apply.

---

## Canvas tools — the 5 generic protocol tools (plus `generate_quiz_from_knowledge`)

`tools/canvas_protocol_tools.py` registers exactly 5 generic canvas tools (`canvas_analyze`, `canvas_highlight`, `canvas_control`, `canvas_action`, `canvas_set_page`). `tools/quiz_generation.py` adds a sixth, non-canvas tool — `generate_quiz_from_knowledge` — which is bundled with `set_page` internally; see the "Quiz Canvas Page (S64e)" section below for why.

Each canvas handler:

1. Logs entry with a `[CANVAS_<VERB>] called: …` line so cloud logs are greppable.
2. Translates element_id aliases (see "Element alias layer" below) to real UUIDs before dispatch.
3. Validates verb against the active Page's manifest when present — **with one exemption**: scene-nav verbs (`next_scene` / `previous_scene` / `goto_scene`) bypass this check. See "Scene-nav verbs are shell-level" below.
4. Builds the wire-format payload with a fresh `commandId` (uuid).
5. Registers the pending command with `PendingCommandRegistry`, getting an `asyncio.Future`.
6. Sends the Daily app-message via `dispatch_canvas_command(ctx, tool, args, command_id=None)` (module-level helper since S64e — same dispatch path reused by `tools/quiz_generation.py` for the bundled set_page). Logs `[CANVAS DISPATCH] tool=… commandId=… payload=…`.
7. Awaits the Future (6s default timeout).
8. Logs `[CANVAS RESULT] tool=… commandId=… result=…`, returns to LLM.
9. On `CanvasCommandError`: logs `[CANVAS_<VERB>] error code=… message=…`, returns the error to the LLM as a tool result (`{error, message, details}`) — NOT a raise — so a single failed call doesn't break the conversation turn.

**`canvas_action` verb-specific arg shape** (LLM gets this in both system prompt + FunctionSchema description):

- `draw_arrow`: `args = {"from": "<element_id>", "to": "<element_id>"}` — both must be element ids from the "Available canvas elements" listing.
- `add_annotation`: `args = {"text": "<string>", "x": <number>, "y": <number>}` — x/y in 1280×720 design space.
- `submit_answer` (quiz): `args = {"choice": "A"|"B"|"C"|"D"}` — handler dispatches and returns the frontend reply unchanged. **Advancement is NOT bundled**; the Quiz Page schedules its own reveal-then-advance timer.
- `skip_question` (quiz): `args = {}`. The "I don't know" path — Page reveals the correct answer without recording a user pick, then auto-advances on the same timer. Returns `{skipped: true, correct: false, completed: bool}`.

The redundancy (prompt + schema) is intentional — early LLM tool calls had a flatten bug where it'd pass `from`/`to` at the top level alongside `verb`. Documenting the nesting in both surfaces fixed it.

**`canvas_set_page` allowlist:** handler validates `pageType` against `{"composition", "youtube", "quiz"}`. The `youtube` and `quiz` entries were pre-allowed in S64c so the allowlist doesn't change as new Page types ship; end-to-end `set_page` execution requires the corresponding Page bundle on the frontend.

**`canvas_set_page` is all-or-nothing on the frontend.** The override layer in `app/(live)/live/[slug]/page.tsx` derives both `effectivePageType` and `effectivePageInit` from a single `useOverride` flag: true only when the LLM supplies BOTH a non-null pageType AND a non-empty pageInit. When pageInit is empty (or missing), BOTH fields fall back to the snapshot. This is what makes "exit overlay back to the scene" work on every scene type: the LLM calls `canvas_set_page(pageType='composition', pageInit={})`, the override is treated as "restore snapshot", and the iframe lands on whatever Page the scene's `canvas_page_type` declares.

The agent's PLAYBOOK forbids the LLM from calling `canvas_set_page(pageType='quiz', …)` directly — the blob-copy failure mode is too reliable. Quiz activation goes through `generate_quiz_from_knowledge` instead, which builds the blob AND dispatches set_page in one tool call.

**Scene-nav verbs are shell-level.** `next_scene`, `previous_scene`, and `goto_scene` are routed by the frontend's `DailyRelay` to the live-room shell's `navigateToIndex` BEFORE reaching the iframe (see `lib/canvas-protocol/daily-relay.ts SCENE_NAV_VERBS`). Consequently:

- These verbs are NOT (and should not be) listed in any iframe-side manifest. The composition Page used to declare them and worked by accident — the relay short-circuited before the iframe got a chance to reject. YouTube and Quiz correctly omit them, which tripped the agent's manifest validator on `next_scene` from those scenes.
- The agent's `handle_control` skips the manifest verb check when `verb in SCENE_NAV_VERBS` (`tools/canvas_protocol_tools.py`). The dispatch goes out, the relay intercepts it, the shell navigates, the agent learns about the change via inbound `canvas.sceneChanged`.
- The CANVAS PAGE prompt section always appends a closing paragraph telling the LLM that scene-nav verbs are available regardless of the per-Page verb listing above. Without this, the LLM would conclude "next_scene isn't in YouTube's verb list, so I can't use it".

---

## Element alias layer (post-S64c)

Backend snapshots return element ids as UUID7 strings (e.g. `019e0e07-0215-7af4-9cc4-3f130fc91d9b`). UUID7s share long prefixes (timestamp-ordered), and LLMs fail to copy them verbatim mid-stream — the model picks the right element conceptually but mis-types the id.

**Solution:** the agent generates short, deterministic aliases per snapshot and uses those in the prompt. The LLM only ever has to copy 5–10 characters.

- `scene_context.compute_element_aliases(elements)` returns `{alias: element_id}`. Format: `<type>_<ordinal>` (`text_1`, `avatar_1`, `emoji_1`, `button_1`, …). Deterministic per snapshot.
- `persona.build_system_prompt(…, aliases_out=...)` computes and populates the map.
- `bot.py` passes `canvas_ctx.element_alias_map` as `aliases_out`; the dict is cleared + repopulated in place on every prompt rebuild.
- `tools/canvas_protocol_tools.py` `_resolve_element_id` translates back: when `target.element_id` (or `args.from` / `args.to`) is a known alias, it's swapped for the real UUID before dispatch. Unknown values pass through (so a typo fails loudly at the frontend, not silently).
- `scene_context.build_canvas_tools_section` renders the listing as alias-with-content:
  ```
  - `avatar_1` — avatar
  - `text_1`   — text "Welcome to QUIZ"
  - `emoji_1`  — emoji 🥰
  ```

**Emoji rendering:** `build_scene_description` and `_summarize_element` render `emoji_character` in element lines so the LLM has semantic content for emojis.

---

## Quiz Canvas Page (S64e)

Ships in `tools/quiz_generation.py` on the agent and `public/canvas-pages/quiz/` on the frontend. End-to-end, the quiz flow is shaped by two principles learned the hard way:

1. **LLMs don't reliably copy large structured blobs between tool calls.** Fix pattern (S64e Option D): bundle the side-effect dispatch into the data-producing tool.
2. **LLMs aren't good at pacing UI animations.** Fix pattern: move the pacing into the frontend (cancellable timers in the Page), keep the agent as a pass-through.

### `generate_quiz_from_knowledge` (bundled set_page)

The tool generates the quiz blob from the backend AND dispatches `canvas.set_page(quiz, blob)` itself, in one handler. The LLM only ever calls one tool.

Wiring (`tools/quiz_generation.py:make_handle_generate_quiz`):

1. `await backend_client.generate_quiz(slug, scene_id, count, language)` → blob (backend `POST /live-rooms/by-slug/{slug}/scenes/{scene_id}/generate-quiz`, Anthropic-backed, scoped by slug).
2. `await dispatch_canvas_command(canvas_ctx, "set_page", {"pageType": "quiz", "pageInit": blob})` — same dispatch path the canvas tools use; same `PendingCommandRegistry` futures; same 6s timeout.
3. Return blob to the LLM as the tool result.

The factory takes `canvas_ctx` (third positional arg). When `canvas_ctx.send_app_message` is None (tests, degraded sessions), the bundled dispatch is skipped and the blob is returned anyway.

`bot.py` wires both classic and relay pipelines:

```python
llm.register_function(
    "generate_quiz_from_knowledge",
    make_handle_generate_quiz(api_client, session_context, canvas_ctx),
)
```

`SessionContext` (same file) carries the live-room slug and current scene id. `bot.py` populates slug from the runner-args body at session start and refreshes `current_scene_id` on every `canvas.sceneChanged`.

### Quiz Page lifecycle (frontend pacing)

The Quiz Page owns the post-answer pacing. The agent's `submit_answer` / `skip_question` calls are dispatched, the frontend records the answer and replies IMMEDIATELY, then two timers fire:

| Constant (`main.js`) | Value | Effect |
|---|---|---|
| `REVEAL_EXPLANATION_DELAY_MS` | `1000` | Explanation banner appears 1 s after the answer. |
| `AUTO_ADVANCE_DELAY_MS` | `6500` | Auto-advance to the next question 6.5 s after the answer. Skipped on the last question. |

Both timers are cancellable — `cancelPostSubmitTimers()` runs at the top of every `next_question` / `previous_question` / `restart` / `clear` handler so an explicit nav doesn't race the auto-advance.

`submit_answer` reply: `{ choice, correct: bool, completed: bool }`. `skip_question` reply: `{ skipped: true, correct: false, completed: bool }`. `completed: true` on the last question is the agent's signal to narrate a wrap-up. Once a quiz finishes, the Page sits on the result view until the agent calls `canvas_set_page(pageType='composition', pageInit={})` to exit.

### `skip_question` — "I don't know"

LLM calls `canvas_action(verb='skip_question', args={})` instead of `submit_answer`. The Page calls `revealAnswer(state)` (`selectedChoice: null`, `correct: false`, `revealed: true`), skips audio/confetti, schedules the same reveal/advance timers, and shows "Here's the answer." `deriveSemanticState` surfaces `revealed`; `completed` treats `selectedChoice !== null || revealed` as "engaged".

### Validation: `buildQuizInit`

`lib/canvas-protocol/quiz-init.ts` validates the LLM-supplied blob before handing it to the iframe; on failure `effectivePageInit` is null and the dispatch times out cleanly. The handler catches `CanvasCommandError` and returns `{"ok": false, "error": "..."}` so the LLM doesn't narrate to a stalled iframe.

### Things that did NOT make the cut (intentionally)

- **In-place reinit for same-pageType `set_page`.** Reverted — no live use case + it required loosening the SDK double-register guard.
- **Bundled `next_question` after `submit_answer` on the agent.** The iframe ripped past confetti/explanation in ~50 ms; replaced by the frontend-owned timer pattern.

---

## LLM provider selection

`LLM_CANVAS_PROVIDER` env var selects the main LLM at boot. `bot.py:_build_llm_and_eager_hook` branches with **lazy imports** so only the selected provider's SDK needs to be installed:

```python
if provider == "openai":
    from pipecat.services.openai.llm import OpenAILLMService
    from services.eager_dispatch.openai_adapter import OpenAIEagerHook
elif provider == "anthropic":
    from pipecat.services.anthropic.llm import AnthropicLLMService
    from services.eager_dispatch.anthropic_adapter import AnthropicEagerHook
elif provider == "gemini":
    from pipecat.services.google.llm import GoogleLLMService
    from services.eager_dispatch.gemini_adapter import GeminiEagerHook
```

`LLM_CANVAS_PROVIDER` is read once at boot and fixed for the session. No mid-session switching.

---

## Eager streaming dispatch — instantiated, not yet wired

Per-provider adapters in `services/eager_dispatch/` peek at LLM streaming events. When the LLM finishes streaming the verb token for an arg-less verb, the adapter would fire the canvas command immediately, without waiting for `stop_reason: tool_use`.

```python
EAGER_DISPATCH_VERBS = frozenset({
    "next_scene", "previous_scene", "clear",
    "next_question", "previous_question", "restart",
    "play", "pause",
})
```

**Current limitation:** the hook objects are constructed in `_build_llm_and_eager_hook` but **not yet called by Pipecat's streaming loop**. There's no Pipecat 0.0.108 integration point exposed for "observe every chunk before it reaches the function-call aggregator", and we deferred building a custom processor for that. Result: in production today, eager dispatch is 0 ms savings. The infrastructure is ready when someone wires `await eager_hook.on_stream_event(chunk)` into the Pipecat LLM service subclass. (Tracked for S74.)

`bench_canvas_latency.py` measures the savings the wiring WOULD unlock. Verbs with required args (`seek`, `set_speed`, `goto_scene`, `draw_arrow`, `add_annotation`) always take the regular `stop_reason` path.

**Double-dispatch safety:** `PendingCommandRegistry.is_eager(commandId)`. When `stop_reason` arrives and the regular handler runs, it checks this flag and awaits the existing future without re-sending.

---

## Scene-change refresh — single path (post-S64c)

Both voice-initiated and visitor rail-click scene navigation flow through ONE refresh path:

1. Frontend's `navigateToIndex` is the canonical scene-change function. Called from the rail click and from the Daily relay's `onSceneNavigation` hook (voice `canvas_control(verb=next_scene)`).
2. After `setSnapshot(snap)`, `navigateToIndex` emits `{type: 'canvas.sceneChanged', sceneIndex}` via the relay's `broadcastSceneChanged(sceneIndex)`.
3. Agent's `on_app_message` `canvas.sceneChanged` branch awaits `refresh_agent_for_current_scene()`:
   - `build_system_prompt(…)` refetches `/scene-snapshot` (post-nav cursor) and rebuilds the base.
   - Re-appends `render_canvas_page_section(canvas_manifest.current())`.
   - Sets `llm._settings.system_instruction`.
   - Refreshes the vision frame via `get_scene_image_base64`.

**The agent does NOT call `api_navigate` from `refresh_agent_for_current_scene`.** The shell's `navigateToIndex` already advanced the backend cursor; calling it again double-steps. (Real bug pre-unification.)

The old per-tool `on_scene_change` callback is gone; `CanvasToolContext.on_scene_change` and the `NAVIGATION_VERBS` constant were removed. Scene-change refresh is exclusively driven by inbound `canvas.sceneChanged`.

**Agent-side `SCENE_NAV_VERBS` exemption.** `SCENE_NAV_VERBS = {"next_scene", "previous_scene", "goto_scene"}` is consulted by `handle_control` to skip the per-Page manifest verb check. Kept in sync with the relay-side constant by convention.

**Timing note for voice nav:** `canvas.sceneChanged` is emitted BEFORE `canvas.commandResult` (same JS frame), so the agent's prompt refresh completes before the tool future resolves. Small Daily-message-reorder race accepted.

> **S65 hook:** `refresh_agent_for_current_scene` is also where the new script narration runs on scene entry (idempotent per scene). It reads the snapshot's new `current_scene.scripts` (with resolved voice), `current_scene.narration.*`, and `live_room.auto_advance` to narrate, then invite-or-suppress, then emit `script_complete`. See "Coming next — S65".

---

## Daily app-messages

**Outgoing (agent → frontend):**

- `{type: 'canvas.command', commandId, tool, verb?, args}` — canvas tool dispatch.
- `{type: 'transcript', speaker, text}` — STT or avatar text.
- `{type: 'speaking_state', isSpeaking}`, `{type: 'llm_thinking', thinking}` — UI cues.
- `{type: 'script_complete'}` — emitted after the scene's scripts finish playing (S49). **S65 adds `sceneIndex` + `hadScript` fields and uses it as the auto-advance trigger.**
- `{type: 'avatar_relay.*', …}` (relay pipeline only) — text/turn protocol for SoulX.

**Incoming (frontend → agent):**

- `{type: 'canvas.register', pageType, version, capabilities, semanticState}` — routed to `CanvasManifestRegistry.set_manifest()`; the CANVAS PAGE prompt section rebuilds immediately.
- `{type: 'canvas.stateChange', semanticState}` — cached in `CanvasManifestRegistry` for the next `analyze()`.
- `{type: 'canvas.sceneChanged', sceneIndex}` — triggers `refresh_agent_for_current_scene()`.
- `{type: 'canvas.commandResult', commandId, result}` / `{type: 'canvas.commandError', commandId, error}` — complete the awaiting Future.

Defensive JSON parsing in `on_app_message`: if Daily delivers the payload as a string (varies by SDK version), parse before the `isinstance(dict)` check.

---

## Live Room snapshot consumer

The agent fetches `GET /live-rooms/{room_id}/scene-snapshot` on session start and on every scene navigation. **`api_client.get_scene_snapshot` always passes `?include_all_scene_knowledge=true`** — the backend aggregates every sibling scene's scene-scope knowledge into `knowledge.flow`, giving the agent a stable flow-knowledge block across navigations.

The snapshot includes:

- `live_room`: language, persona, recipient_prompt **(+ `auto_advance` from S65)**.
- `current_scene`: name, instruction, display_mode, background_url, **elements (with `id`)**, link_url, link_source, `canvas_page_type` **(+ S65 `scripts[*]` with resolved voice, `has_script`, `narration.{invitation_line, transition_cue}`)**.
- `flow_state`: scene_index, total_scenes, scene_ids array.
- `knowledge`: text content (flow scope = aggregated all-scene set).
- `faqs`: array of `{question, answer}` per scope.

**Don't duplicate snapshot logic locally.** Extend the snapshot endpoint in the backend rather than fetching multiple endpoints. `LiveRoomService.get_scene_snapshot()` is the single source of truth, shared across the agent + composition + youtube + quiz Pages. Backend regression tests in `tests/test_scenes_canvas_page_type.py` and `tests/services/test_live_room_snapshot_buttons.py` catch drift.

---

## Vision (S46)

When the visitor asks "what's on screen?", the agent fetches `/scene-snapshot/image` (a Pillow-rendered base64 PNG) and adds it as a user message. The main LLM does the visual reasoning.

V2.14 documents Vision as a separately-hardpinned OpenAI service so quality is invariant under `LLM_CANVAS_PROVIDER` swaps, but **that separation isn't implemented yet**. (S66 will make the per-scene vision-frame refresh **lazy** to cut scene-switch latency — see "Coming after — S66".)

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
OPENAI_API_KEY=...                # main LLM (default) and vision
ANTHROPIC_API_KEY=...             # main LLM when LLM_CANVAS_PROVIDER=anthropic
GOOGLE_AI_API_KEY=...             # main LLM when LLM_CANVAS_PROVIDER=gemini
DEEPGRAM_API_KEY=...
CARTESIA_API_KEY=...
DAILY_API_KEY=...
HV_API_URL=http://localhost:3001/api/v1   # backend; passed through runner_args.body in prod
LLM_CANVAS_PROVIDER=openai        # default; override to anthropic or gemini
```

For Pipecat Cloud testing where the container can't reach `localhost`, point `HV_API_URL` at a publicly-accessible URL.

---

## Deployment — Pipecat Cloud

Production runs on Pipecat Cloud. The Dockerfile is the build manifest. Backend's live-room start endpoint creates a Daily room and registers the Pipecat Cloud agent; the agent boots, reads room metadata, fetches the snapshot (with `include_all_scene_knowledge`), joins via DailyTransport.

`LLM_CANVAS_PROVIDER` is set as a Pipecat Cloud env var. Changing it requires a redeploy.

**Dockerfile gotcha:** the build uses `pip install --no-cache-dir .` against `pyproject.toml`, but `pyproject.toml` has `packages = []` and `py-modules = []` — so install pulls **only dependencies**, not the agent's Python code. Agent code reaches `/app/` exclusively via explicit `COPY` lines. Adding a new top-level directory requires a matching `COPY` line or the cloud container hits `ModuleNotFoundError` at session start (local dev keeps working).

---

## Conventions Claude Code should follow

- **Match the installed pipecat-ai version's import paths.** Grep `bot.py` first; module layout shifts between versions.
- **Tool names must satisfy `^[a-zA-Z0-9_-]+$`.** Don't reintroduce dots. The Daily wire-format `type` field is separate and may use dots.
- **Don't add eager-dispatch entries for verbs that have required args.** A verb goes in `EAGER_DISPATCH_VERBS` only if its handler is provably correct with `args={}`.
- **Don't modify `tools/canvas_protocol_tools.py` for new Page types.** The 5 generic tools are page-agnostic; new Page types are accommodated through their manifest. A 6th canvas tool is a major change — discuss first.
- **Don't add new Daily message types without coordinating with the frontend.** `DailyRelay` is the sole consumer/producer of `canvas.*` messages.
- **Don't call `api_navigate` from agent-side scene-change handlers.** The frontend's `navigateToIndex` already advances the cursor.
- **Don't unify vision + main LLM in a way that loses the separability path.**
- **Don't add `__init__.py` to `tools/`, `context/`, `services/`** unless you also update the Dockerfile (Python 3.12 implicit namespace packages).
- **When extending tool definitions:** prefer the `FunctionSchema` shape; the `description` field is what the LLM sees. For args the LLM keeps mis-shaping, document the JSON shape in BOTH the schema description AND the system-prompt section.
- **When adding a new top-level directory:** add `COPY <dir> <dir>` to `Dockerfile`.
- **Don't ask the LLM to copy large structured blobs between tool calls.** It drops the blob ~half the time. Bundle the side effect into the producing handler (`generate_quiz_from_knowledge` is canonical).
- **Don't ask the LLM to pace UI animations.** Own the pacing in the Page (cancellable timers); keep the agent a pass-through. (S65's auto-advance applies the same lesson — it lives in the *shell*, not the agent.)
- **`canvas_set_page` is all-or-nothing.** Empty `pageInit` ⇒ snapshot wins for BOTH pageType and init. This is what makes "exit to scene" work regardless of the underlying page type.

---

## Recent: S64e complete (Quiz Page + end-to-end `canvas.set_page` + post-cutover hardenings)

S64e wired `canvas.set_page` end-to-end (the Quiz Page is the first scenario where the agent switches Pages mid-scene). The mechanics + lessons are in the **"Quiz Canvas Page (S64e)"** section above. Agent-side summary:

- `tools/quiz_generation.py` with `generate_quiz_from_knowledge` (bundled set_page — Option D), `SessionContext`, `make_handle_generate_quiz`.
- `dispatch_canvas_command` promoted to a module-level helper (shared by canvas tools + quiz generation).
- PLAYBOOK forbids the LLM from calling `canvas_set_page(pageType='quiz')` directly; quiz activation only via `generate_quiz_from_knowledge`.
- `skip_question` action verb ("I don't know"); `submit_answer` is pass-through (advancement is the Page's job — `test_submit_answer_bundling.py` guards this).
- `canvas_set_page` all-or-nothing override semantics (empty `pageInit` = restore snapshot).
- `next_question`/`previous_question`/`restart` were already in `EAGER_DISPATCH_VERBS` (pre-allowlisted in S64c), so no eager-dispatch changes were needed.

---

## Historical: S64d (YouTube Canvas Page) — complete

The frontend shipped the YouTube Canvas Page; **Pipecat required zero code changes.** When a visitor navigates to a YouTube scene: the frontend mounts the YouTube iframe → the Page registers its manifest (`play / pause / seek / set_speed / clear`, highlight `['box']`) → the relay forwards `canvas.register` → the agent's `on_app_message` routes to `CanvasManifestRegistry.set_manifest` and rebuilds the prompt → the LLM uses `canvas_control` with the new verbs.

Runtime notes: `play`/`pause`/`clear` are in `EAGER_DISPATCH_VERBS` (will eager-fire once wired); `seek`/`set_speed` wait for full streaming (required args). `canvas_highlight` on YouTube takes `target = {box: [x,y,w,h]}` (box-only manifest). New arg-less verbs from future Pages should be added to `EAGER_DISPATCH_VERBS`; otherwise no code changes.

---

## Coming next — S65 (Live Room Script Narration + Flow Auto-Advance)

A live-experience session (no Canvas Protocol contract change, no new canvas tools). Agent deliverables:

- **Discovery first:** locate the existing S49 narration path (`grep` `bot.py`/`persona.py`/`scene_context.py` for `script`, `script_complete`, `TTSSpeakFrame`, `.say(`, `narrate`). Confirm it speaks scripts deterministically (expected — `script_complete` already exists). Confirm the version-correct Cartesia voice-update API.
- **Per-segment voice switching (classic pipeline):** narrate each `current_scene.scripts[*].text` in its resolved `voice_id` (script avatar's Cartesia clone → primary fallback), reset to primary after the last segment. **Relay (`talking`) pipeline narrates in the primary SoulX voice — per-script-avatar voice is a v0.2 punt** — but still emits `script_complete`.
- **Post-script invitation + auto-advance branch:** read `live_room.auto_advance` + `flow_state`. If `auto_advance` and not last scene → optional localized `transition_cue`, **suppress** the invitation, emit `script_complete` (the shell advances). Else → speak the localized `narration.invitation_line`, emit `script_complete`. No-script scenes: neither branch (existing conversational greeting stands; `script_complete` carries `hadScript=false`).
- **`script_complete` payload:** add `{sceneIndex, hadScript}`; emit it LAST (after the invitation/cue).
- **Prompt directive:** tell the LLM the script is narrated automatically — do NOT read/paraphrase it; stay silent until the visitor speaks.
- **Idempotent** per scene entry (don't re-narrate on `canvas.register`-only prompt rebuilds).
- **Do NOT** add agent-side navigation — auto-advance is shell-owned (frontend) and triggers off `script_complete` (the S64e "don't let the LLM pace transitions" lesson).
- **Tests:** ~6–8 in `test_scene_narration.py` (voice resolution clone-vs-fallback, invitation suppress/show by auto_advance + last-scene, `script_complete` once-per-entry with payload, idempotency). Factor the decision logic into a pure helper for testability; keep frame I/O thin.

---

## Coming after — S66 (Flow Scene-Switching Performance)

Targets < 1 s scene transitions. Agent-side cuts: **lazy vision-frame refresh** (`VISION_REFRESH_MODE=lazy|eager`, default lazy — don't render the Pillow PNG on every scene change; fetch only on a visual question / first `analyze`), **reuse cached flow-knowledge** across scene changes (the flow-knowledge block is invariant), and **fetch the snapshot by `scene_id`** (cursor-independent endpoint from S66 backend) or accept a snapshot pushed on `canvas.sceneChanged` — with a robust fallback fetch. No Canvas Protocol contract change.

---

## Out of scope

- Mid-session provider switching (`LLM_CANVAS_PROVIDER` fixed at boot).
- Natural-language highlight targets.
- Persistent iframe shell (per-scene unmount + keyed remount is current; the S66 prewarm is a constrained, flag-gated slice).
- A/B testing infrastructure for comparing providers in production.
- Vision provider separation (today's vision uses the main LLM).
- Eager-dispatch-to-Pipecat-streaming-loop wiring (hooks constructed; never invoked by Pipecat — the latency win is structural, not realized; tracked for S74).
