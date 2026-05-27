# pipecat-agent — CLAUDE.md

> Voice agent for **Human Virtual** (hv.ai), built on the **Pipecat Framework**.
> **Status:** S64c + S64d + S64e shipped, plus a round of post-S64e hardenings driven by real-session bug reports — `generate_quiz_from_knowledge` bundles the canvas Page swap (Option D); the Quiz Page owns its post-answer reveal/auto-advance timing instead of the agent racing it; a `skip_question` action verb handles "I don't know"; `canvas.set_page` override is all-or-nothing (empty `pageInit` means "restore snapshot — both pageType and init"); scene-nav verbs (`next_scene` / `previous_scene` / `goto_scene`) are exempt from the per-Page manifest validator because they're shell-level.
> **Repo:** `pipecat-agent/` (lives alongside `human-virtual-backend/` and `human-virtual-frontend/`).

---

## What this service does

The Pipecat agent runs the avatar's **conversation pipeline**: STT → LLM → TTS → audio out, with optional vision and canvas tool calls. It connects to a Live Room over WebRTC and:

1. Listens to the visitor's microphone (Deepgram STT).
2. Calls an LLM (OpenAI / Anthropic / Gemini, selected at boot via `LLM_CANVAS_PROVIDER`) with a system prompt assembled from the Live Room's persona, knowledge (aggregated across the whole flow), current scene instruction, current scene's elements (referenced by short aliases the LLM can actually copy), and the active Canvas Page's manifest.
3. Emits TTS audio (Cartesia) plus avatar lip-sync data (SoulX-Flashtalk for "talking" display mode).
4. Calls **canvas tools** via the generic 5-tool protocol surface (`canvas_analyze`, `canvas_highlight`, `canvas_control`, `canvas_action`, `canvas_set_page` — **underscored, not dotted**; see "Tool naming" below).
5. (S46+) Calls **vision** by adding a snapshot image to the LLM's context. Currently the main LLM does the visual reasoning — a separate hard-pinned vision service is described in V2.14 but not yet implemented in the agent.

---

## Stack

| Layer | Choice |
|---|---|
| Framework | Pipecat (Python, pinned at 0.0.108 in `pyproject.toml`) |
| Transport (prod) | DailyTransport (Daily.co WebRTC) |
| Transport (local) | SmallWebRTCTransport |
| Deployment (prod) | Pipecat Cloud |
| LLM default | OpenAI GPT (via `LLM_CANVAS_PROVIDER=openai`) |
| LLM alt | Anthropic Claude (requires `pipecat-ai[anthropic]` extra installed), Google Gemini (requires `pipecat-ai[google]`) |
| LLM selection | `LLM_CANVAS_PROVIDER` env var, validated against `{anthropic, openai, gemini}` at boot |
| STT | Deepgram |
| TTS | Cartesia |
| Lip-sync | SoulX-Flashtalk (S48 — relay pipeline, `display_mode == 'talking'`) |

**LLM default note:** V2.14 documents Anthropic as the target default, but the installed `pyproject.toml` ships only `pipecat-ai[…openai…]`. `config.py` defaults `LLM_CANVAS_PROVIDER=openai` because attempting to import `pipecat.services.anthropic.llm` without the anthropic extra crashes at boot. To flip the default to Anthropic, add `anthropic` to the pipecat extras in `pyproject.toml`, `uv sync`, then change the `os.getenv("LLM_CANVAS_PROVIDER", "openai")` fallback in `config.py`.

---

## Project structure

```
pipecat-agent/
  bot.py                            # Main entrypoint — pipeline assembly, transport setup, LLM provider branching
  config.py                         # Settings (env vars), provider validation, model defaults
  persona.py                        # build_system_prompt — assembles the live system prompt (Strategy 1 = backend persona-prompt endpoint + agent additions)
  scene_context.py                  # Section helpers, knowledge formatting, build_canvas_tools_section, alias generator
  api_client.py                     # HTTP client for the backend (snapshot, persona-prompt, navigate, scene image, …)
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

**Daily wire-format message types keep their dots:**

- `canvas.command`, `canvas.commandResult`, `canvas.commandError`, `canvas.register`, `canvas.stateChange`, `canvas.sceneChanged`.

These travel between Pipecat ↔ frontend (Daily app-messages), never to the LLM provider, so the regex doesn't apply.

---

## Canvas tools — the 5 generic protocol tools (plus `generate_quiz_from_knowledge`)

`tools/canvas_protocol_tools.py` registers exactly 5 generic canvas tools (`canvas_analyze`, `canvas_highlight`, `canvas_control`, `canvas_action`, `canvas_set_page`). `tools/quiz_generation.py` adds a sixth, non-canvas tool — `generate_quiz_from_knowledge` — which is bundled with `set_page` internally; see the "Quiz Canvas Page (S64e)" section below for why.

Each canvas handler:

1. Logs entry with a `[CANVAS_<VERB>] called: …` line so cloud logs are greppable.
2. Translates element_id aliases (see "Element alias layer" below) to real UUIDs before dispatch.
3. Validates verb against the active Page's manifest when present — **with one exemption**: scene-nav verbs (`next_scene` / `previous_scene` / `goto_scene`) bypass this check. See "Scene-nav verbs are shell-level" below.
4. Builds the wire-format payload with a fresh `commandId` (uuid).
5. Registers the pending command with `PendingCommandRegistry`, getting an `asyncio.Future`.
6. Sends the Daily app-message via `dispatch_canvas_command(ctx, tool, args, command_id=None)` (module-level helper, since S64e — same dispatch path is reused by `tools/quiz_generation.py` for the bundled set_page). Logs `[CANVAS DISPATCH] tool=… commandId=… payload=…`.
7. Awaits the Future (6s default timeout).
8. Logs `[CANVAS RESULT] tool=… commandId=… result=…`, returns to LLM.
9. On `CanvasCommandError`: logs `[CANVAS_<VERB>] error code=… message=…`, returns the error to the LLM as a tool result (`{error, message, details}`) — NOT a raise — so a single failed call doesn't break the conversation turn. Mirrors V2.13's always-result_callback resilience.

**`canvas_action` verb-specific arg shape** (LLM gets this in both system prompt + FunctionSchema description):

- `draw_arrow`: `args = {"from": "<element_id>", "to": "<element_id>"}` — both must be element ids from the "Available canvas elements" listing.
- `add_annotation`: `args = {"text": "<string>", "x": <number>, "y": <number>}` — x/y in 1280×720 design space.
- `submit_answer` (quiz): `args = {"choice": "A"|"B"|"C"|"D"}` — handler dispatches and returns the frontend reply unchanged. **Advancement is NOT bundled**; the Quiz Page schedules its own reveal-then-advance timer. See the Quiz section below.
- `skip_question` (quiz): `args = {}`. The "I don't know" path — Page reveals the correct answer without recording a user pick, then auto-advances on the same timer. Returns `{skipped: true, correct: false, completed: bool}`.

The redundancy (prompt + schema) is intentional — early LLM tool calls had a flatten bug where it'd pass `from`/`to` at the top level alongside `verb`. Documenting the nesting in both surfaces fixed it.

**`canvas_set_page` allowlist:** handler validates `pageType` against `{"composition", "youtube", "quiz"}`. The `youtube` and `quiz` entries were pre-allowed in S64c so the allowlist doesn't need to change as new Page types ship; end-to-end `set_page` execution requires the corresponding Page bundle on the frontend.

**`canvas_set_page` is all-or-nothing on the frontend.** The override layer in `app/(live)/live/[slug]/page.tsx` derives both `effectivePageType` and `effectivePageInit` from a single `useOverride` flag: true only when the LLM supplies BOTH a non-null pageType AND a non-empty pageInit. When pageInit is empty (or missing), BOTH fields fall back to the snapshot. This is what makes "exit overlay back to the scene" work on every scene type (composition or YouTube): the LLM calls `canvas_set_page(pageType='composition', pageInit={})`, the override is treated as "restore snapshot", and the iframe lands on whatever Page the scene's `canvas_page_type` declares. The PLAYBOOK can therefore continue to say "exit to composition" even though the actual destination is "exit to scene".

The agent's PLAYBOOK forbids the LLM from calling `canvas_set_page(pageType='quiz', …)` directly — the blob-copy failure mode is too reliable. Quiz activation goes through `generate_quiz_from_knowledge` instead, which builds the blob AND dispatches set_page in one tool call.

**Scene-nav verbs are shell-level.** `next_scene`, `previous_scene`, and `goto_scene` are routed by the frontend's `DailyRelay` to the live-room shell's `navigateToIndex` BEFORE reaching the iframe (see `lib/canvas-protocol/daily-relay.ts SCENE_NAV_VERBS`). Consequently:

- These verbs are NOT (and should not be) listed in any iframe-side manifest. Listing them implies the iframe handles them; the iframe doesn't. The composition Page used to declare them and worked by accident — the relay short-circuited before the iframe got a chance to reject. YouTube and Quiz correctly omit them, which tripped the agent's manifest validator on `next_scene` from a YouTube scene.
- The agent's `handle_control` skips the manifest verb check when `verb in SCENE_NAV_VERBS` (`tools/canvas_protocol_tools.py`). The dispatch goes out, the relay intercepts it, the shell navigates, the agent learns about the change via inbound `canvas.sceneChanged`.
- The CANVAS PAGE prompt section always appends a closing paragraph telling the LLM that scene-nav verbs are available regardless of the per-Page verb listing above. Without this, the LLM would conclude "next_scene isn't in YouTube's verb list, so I can't use it".

---

## Element alias layer (post-S64c)

Backend snapshots return element ids as UUID7 strings (e.g. `019e0e07-0215-7af4-9cc4-3f130fc91d9b`). UUID7s share long prefixes (timestamp-ordered), and LLMs fail to copy them verbatim mid-stream — the model picks the right element conceptually but mis-types the id, which surfaces as "highlighted the wrong element".

**Solution:** the agent generates short, deterministic aliases per snapshot and uses those in the prompt. The LLM only ever has to copy 5–10 characters.

- `scene_context.compute_element_aliases(elements)` returns `{alias: element_id}`. Format: `<type>_<ordinal>` (`text_1`, `avatar_1`, `emoji_1`, `button_1`, `emoji_2`, …). Deterministic per snapshot.
- `persona.build_system_prompt(…, aliases_out=...)` computes and populates the map.
- `bot.py` passes `canvas_ctx.element_alias_map` as `aliases_out`; the dict is cleared + repopulated in place on every prompt rebuild.
- `tools/canvas_protocol_tools.py` `_resolve_element_id` translates back: when `target.element_id` (or `args.from` / `args.to` for `draw_arrow`) is a known alias, it's swapped for the real UUID before the Daily message goes out. Unknown values pass through (so a typo or stale alias fails loudly at the frontend, not silently in translation).
- `scene_context.build_canvas_tools_section` renders the listing as alias-with-content:
  ```
  - `avatar_1` — avatar
  - `text_1`   — text "Welcome to QUIZ"
  - `emoji_1`  — emoji 🥰
  ```

**Emoji rendering:** `build_scene_description` and `_summarize_element` render `emoji_character` in element lines so the LLM has semantic content for emojis (without it, the LLM occasionally confused emoji elements for "the title").

---

## Quiz Canvas Page (S64e)

Ships in `tools/quiz_generation.py` on the agent and `public/canvas-pages/quiz/` on the frontend. End-to-end, the quiz flow is shaped by two principles learned the hard way:

1. **LLMs don't reliably copy large structured blobs between tool calls.** Anywhere we asked the model to "take the blob you just got and pass it into the next tool call", it silently dropped the blob about half the time. Fix pattern (S64e Option D): bundle the side-effect dispatch into the data-producing tool.
2. **LLMs aren't good at pacing UI animations.** Anywhere we let the model dispatch the next visual transition immediately after a result, it raced ahead of confetti / explanation reveals. Fix pattern: move the pacing into the frontend (cancellable timers in the Page), keep the agent as a pass-through.

### `generate_quiz_from_knowledge` (bundled set_page)

The tool generates the quiz blob from the backend AND dispatches `canvas.set_page(quiz, blob)` itself, in one handler. The LLM only ever calls one tool; the iframe is already showing the new quiz by the time the result returns.

Wiring (`tools/quiz_generation.py:make_handle_generate_quiz`):

1. `await backend_client.generate_quiz(slug, scene_id, count, language)` → blob.
2. `await dispatch_canvas_command(canvas_ctx, "set_page", {"pageType": "quiz", "pageInit": blob})` — same dispatch path the canvas tools use; same `PendingCommandRegistry` futures; same 6s timeout.
3. Return blob to the LLM as the tool result.

The factory takes `canvas_ctx` (third positional arg) — the same `CanvasToolContext` threaded into `make_handlers` in `bot.py`. When `canvas_ctx.send_app_message` is None (tests, degraded sessions), the bundled dispatch is skipped and the blob is returned anyway.

`bot.py` wires both classic and relay pipelines:

```python
llm.register_function(
    "generate_quiz_from_knowledge",
    make_handle_generate_quiz(api_client, session_context, canvas_ctx),
)
```

`SessionContext` (in the same file) carries the live-room slug and current scene id. `bot.py` populates slug from the runner-args body at session start and refreshes `current_scene_id` on every `canvas.sceneChanged`. The backend `POST /live-rooms/by-slug/{slug}/scenes/{scene_id}/generate-quiz` endpoint is scoped by slug.

### Quiz Page lifecycle (frontend pacing)

The Quiz Page (`public/canvas-pages/quiz/`) owns the post-answer pacing. The agent's `submit_answer` / `skip_question` calls are dispatched, the frontend records the answer and replies IMMEDIATELY, and then two timers fire:

| Constant (`main.js`) | Value | Effect |
|---|---|---|
| `REVEAL_EXPLANATION_DELAY_MS` | `1000` | At 1 s after the answer, the explanation banner appears (`showingExplanation: true`, render, emit stateChange). |
| `AUTO_ADVANCE_DELAY_MS` | `6500` | At 6.5 s after the answer, `transitionCard(nextQuestion(state))` runs, the iframe scrolls to the next question, highlights clear, emit stateChange. Skipped entirely on the last question. |

Both timers are cancellable — `cancelPostSubmitTimers()` is called at the top of every `next_question` / `previous_question` / `restart` / `clear` handler so an explicit nav from the agent doesn't race the auto-advance and double-step. Inside each timer, a final sanity check (`state.selectedChoice === null` for reveal, plus index bounds for advance) guards against state changes between scheduling and firing.

`submit_answer` reply shape: `{ choice, correct: bool, completed: bool }`.
`skip_question` reply shape: `{ skipped: true, correct: false, completed: bool }`.

`completed: true` on the last question is the agent's signal to narrate a final wrap-up (the LLM keeps its own running score tally; the Page doesn't compute one). Once a quiz finishes, the Page sits on the result view and waits for the agent to call `canvas_set_page(pageType='composition', pageInit={})` to exit back to the scene.

**Tuning knobs.** If the agent's narration feels rushed inside the 1 s → 6.5 s window, bump `AUTO_ADVANCE_DELAY_MS` further. The reveal delay is tuned to overlap with confetti / wrong-flag cue duration; tweak if those audio cues change.

### `skip_question` — "I don't know"

When the visitor says "I don't know" / "I'm not sure" / "skip" / etc., the LLM calls `canvas_action(verb='skip_question', args={})` INSTEAD of `submit_answer`. The Page:

- Calls `revealAnswer(state)` (in `quiz-state.js`) → `selectedChoice: null`, `correct: false`, `revealed: true`. The `revealed` flag is a NEW state field added in S64e specifically for this path so the renderer can highlight the correct choice (`data-correct`) WITHOUT the red user-pick stripe (`data-incorrect`) a wrong answer would draw.
- Skips audio cue + confetti (the visitor didn't attempt; a "wrong" sound would feel scolding).
- Schedules the same reveal/advance timers as `submit_answer`.
- Feedback banner shows "Here's the answer." instead of "Not quite — see the explanation." (a softer phrasing for non-attempts).

`deriveSemanticState` surfaces `revealed` for the agent's analyze path. The `completed` flag treats `selectedChoice !== null || revealed` as "engaged" so a last-question skip correctly reports `completed: true`.

### Validation: `buildQuizInit`

`lib/canvas-protocol/quiz-init.ts` validates the LLM-supplied blob before handing it to the iframe. If validation fails, `effectivePageInit` returns `null`, the per-iframe effect skips the bridge attach, and the dispatch times out at 10 s with a clean `TIMEOUT` error the agent surfaces to the LLM. The handler in `tools/quiz_generation.py` catches `CanvasCommandError` and returns `{"ok": false, "error": "..."}` instead of the blob, so the LLM doesn't narrate questions to a stalled iframe.

### Things that did NOT make the cut (intentionally)

- **In-place reinit for same-pageType `set_page`.** Tried during the S64e debug cycle: `service.handleSetPage` would, for same-pageType swaps with fresh pageInit, send `canvas.init` through the existing bridge instead of unmounting the iframe. Reverted because (a) there's no live use case (quiz→quiz mid-session auto-load isn't a product flow), and (b) it required making the SDK's `register()` idempotent, which loosens a guard against buggy Pages double-registering. If we ever want multi-quiz auto-load, this is the path to revisit.
- **Bundled `next_question` after `submit_answer` on the agent.** Tried as S64e "Option 2" — the agent fired `canvas_control(next_question)` immediately after `submit_answer`. Symptom: the visitor never saw confetti / explanation because the iframe ripped past the result view in ~50 ms. The frontend-owned timer pattern replaced it.

---

## LLM provider selection

`LLM_CANVAS_PROVIDER` env var selects the main LLM at boot. `bot.py:_build_llm_and_eager_hook` branches with **lazy imports** so only the selected provider's SDK needs to be installed:

```python
if provider == "openai":
    from pipecat.services.openai.llm import OpenAILLMService
    from services.eager_dispatch.openai_adapter import OpenAIEagerHook
    …
elif provider == "anthropic":
    from pipecat.services.anthropic.llm import AnthropicLLMService
    from services.eager_dispatch.anthropic_adapter import AnthropicEagerHook
    …
elif provider == "gemini":
    from pipecat.services.google.llm import GoogleLLMService
    from services.eager_dispatch.gemini_adapter import GeminiEagerHook
    …
```

`LLM_CANVAS_PROVIDER` is read once at boot and fixed for the session. No mid-session switching.

---

## Eager streaming dispatch — instantiated, not yet wired

Per-provider adapters in `services/eager_dispatch/` peek at LLM streaming events. When the LLM finishes streaming the verb token for an arg-less verb, the adapter fires the canvas command immediately, without waiting for `stop_reason: tool_use`. Saves a structural 100–250ms per arg-less call (~85ms confirmed in the local microbenchmark; real-world expected higher due to chunk-arrival jitter and the post-tool-use text the LLM often emits before `stop_reason`).

```python
EAGER_DISPATCH_VERBS = frozenset({
    "next_scene", "previous_scene", "clear",
    "next_question", "previous_question", "restart",
    "play", "pause",
})
```

**Current limitation:** the hook objects are constructed in `_build_llm_and_eager_hook` but **not yet called by Pipecat's streaming loop**. There's no Pipecat 0.0.108 integration point exposed for "observe every chunk before it reaches the function-call aggregator", and we deferred building a custom processor for that purpose. Result: in production today, eager dispatch is 0ms savings. The infrastructure is ready when someone wires `await eager_hook.on_stream_event(chunk)` into the Pipecat LLM service subclass.

The `bench_canvas_latency.py` script measures the savings the wiring WOULD unlock. Verbs not in `EAGER_DISPATCH_VERBS` (e.g. `seek`, `set_speed`, `goto_scene`, `draw_arrow`, `add_annotation`) take the regular `stop_reason` path always — they have required args that can't be eager-fired without losing.

**Double-dispatch safety:** `PendingCommandRegistry.is_eager(commandId)`. When `stop_reason` arrives and the regular handler runs, it checks this flag first; if true, the handler awaits the existing future without re-sending the Daily message.

---

## Scene-change refresh — single path (post-S64c)

Both voice-initiated scene navigation and visitor rail-click scene navigation now flow through ONE refresh path on the agent side:

1. Frontend's `navigateToIndex` (in `human-virtual-frontend/app/(live)/live/[slug]/page.tsx`) is the canonical scene-change function. It's called from the rail click directly, and from the Daily relay's `onSceneNavigation` hook when the voice agent invokes `canvas_control(verb=next_scene)`.
2. After `setSnapshot(snap)`, `navigateToIndex` emits `{type: 'canvas.sceneChanged', sceneIndex}` via the Daily relay's `broadcastSceneChanged(sceneIndex)` method.
3. Agent's `on_app_message` `canvas.sceneChanged` branch awaits `refresh_agent_for_current_scene()`:
   - `build_system_prompt(…)` refetches `/scene-snapshot` (which now reflects the post-nav cursor) and rebuilds the base.
   - Re-appends `render_canvas_page_section(canvas_manifest.current())`.
   - Sets `llm._settings.system_instruction`.
   - Refreshes the vision frame via `get_scene_image_base64`.

**The agent does NOT call `api_navigate` from `refresh_agent_for_current_scene`.** The shell's `navigateToIndex` already advanced the backend cursor; calling it again would double-step. (We had this bug pre-unification — it produced 2-scene jumps in user testing.)

The old per-tool `on_scene_change` callback that fired inside `handle_control` is gone — `CanvasToolContext.on_scene_change` was removed entirely. The `NAVIGATION_VERBS` constant is also gone. Scene-change refresh is exclusively driven by inbound `canvas.sceneChanged` messages.

**Agent-side `SCENE_NAV_VERBS` exemption.** In `tools/canvas_protocol_tools.py`, the frozenset `SCENE_NAV_VERBS = {"next_scene", "previous_scene", "goto_scene"}` is consulted by `handle_control` to skip the per-Page manifest verb check. Without this, the LLM would hit `UNSUPPORTED_VERB` whenever it tried scene nav from a YouTube or Quiz scene (those iframes correctly don't list scene-nav in their manifests). Kept in sync with the relay-side `SCENE_NAV_VERBS` constant in `lib/canvas-protocol/daily-relay.ts` by convention — two-line drift risk, acceptable; these verbs are stable.

**Timing note for voice nav:** `canvas.sceneChanged` is emitted BEFORE `canvas.commandResult` (from the same JS frame), so the agent's prompt refresh completes before `_dispatch`'s future resolves and the LLM is told the tool succeeded. In practice this means the LLM speaks with the fresh prompt. There's a small Daily-message-reorder race that we accept; if it ever surfaces as observable staleness, the fix is having `_dispatch` for nav verbs wait for a "sceneChanged consumed" signal.

---

## Daily app-messages

**Outgoing (agent → frontend):**

- `{type: 'canvas.command', commandId, tool, verb?, args}` — canvas tool dispatch.
- `{type: 'transcript', speaker, text}` — STT or avatar text.
- `{type: 'speaking_state', isSpeaking}`, `{type: 'llm_thinking', thinking}` — UI cues.
- `{type: 'script_complete'}` — emitted after the scene's scripts finish playing.
- `{type: 'avatar_relay.*', …}` (relay pipeline only) — text/turn protocol for SoulX.

**Incoming (frontend → agent):**

- `{type: 'canvas.register', pageType, version, capabilities, semanticState}` — Page declared its capabilities. Routed to `CanvasManifestRegistry.set_manifest()`; the system prompt's CANVAS PAGE section rebuilds immediately so the LLM sees the new verb list on the very next turn (not just on next scene change).
- `{type: 'canvas.stateChange', semanticState}` — Page's semantic state changed. Cached in `CanvasManifestRegistry` for the next `analyze()` call.
- `{type: 'canvas.sceneChanged', sceneIndex}` — fired from the frontend's `navigateToIndex` after every successful nav. Triggers `refresh_agent_for_current_scene()`.
- `{type: 'canvas.commandResult', commandId, result}` — completes the awaiting tool handler's Future with success.
- `{type: 'canvas.commandError', commandId, error}` — completes the Future with `CanvasCommandError` (then converted to a tool-result error for the LLM).

There's defensive JSON parsing in `on_app_message`: if Daily delivers the payload as a string (varies by SDK version), the handler parses it before the `isinstance(dict)` check. Without this, `canvas.register` was silently dropping on some Pipecat Cloud versions.

---

## Live Room snapshot consumer

The agent fetches `GET /live-rooms/{room_id}/scene-snapshot` from the backend on session start and on every scene navigation. **`api_client.get_scene_snapshot` always passes `?include_all_scene_knowledge=true`** — the backend then aggregates every sibling scene's scene-scope knowledge into the snapshot's `knowledge.flow` payload, in addition to whatever knowledge is attached at the flow level. This gives the agent a stable "flow knowledge" block in its prompt that persists across scene navigations; a creator can attach knowledge to any scene and the agent can answer from it regardless of which scene is currently active.

The snapshot includes:

- `live_room`: language, persona, recipient_prompt.
- `current_scene`: name, instruction, display_mode, background_url, **elements (with `id` field — required for the alias layer)**, link_url, link_source, `canvas_page_type`.
- `flow_state`: scene_index, total_scenes, scene_ids array.
- `knowledge`: text content from extracted sources (flow scope contains the aggregated all-scene set after the flag).
- `faqs`: array of `{question, answer}` per scope.

The element `id` field was added to the backend snapshot specifically to support the alias layer — earlier versions of the snapshot didn't surface element ids, so the frontend Composition Page was generating synthetic `el_<idx>` ids. After the backend change, both the agent's alias resolver and the frontend's element lookup use the real DB ids, and they match.

**Don't duplicate snapshot logic locally.** If the agent needs additional context, extend the snapshot endpoint in the backend rather than fetching multiple endpoints. The backend's `LiveRoomService.get_scene_snapshot()` is the single source of truth.

The snapshot's per-scene shape is shared between three consumers: the **Composition Canvas Page** (frontend), the **YouTube Canvas Page** (frontend, S64d), and this Pipecat agent. Don't change it without coordinating across all three. The backend has regression tests in `tests/test_scenes_canvas_page_type.py` and `tests/services/test_live_room_snapshot_buttons.py` to catch drift.

---

## Vision (S46)

When the visitor asks "what's on screen?", the agent fetches `/scene-snapshot/image` (a Pillow-rendered base64 PNG) and adds it as a user message in the LLMContext. The main LLM does the visual reasoning.

V2.14 documents Vision as a separately-hardpinned OpenAI GPT-4.1 service alongside the main LLM (so vision quality is invariant under `LLM_CANVAS_PROVIDER` swaps), but **that separation hasn't been implemented in the agent yet**. Today's behavior: vision goes through whichever LLM `LLM_CANVAS_PROVIDER` selected. Worth landing if/when we have a multi-provider production deployment that needs the visual-reasoning quality lock-in.

---

## Local development

```bash
# Setup
python -m venv .venv
source .venv/bin/activate
uv sync

# Run agent locally with WebRTC server in browser
uv run bot.py

# Test with a different provider (extras must be installed first):
LLM_CANVAS_PROVIDER=anthropic uv run bot.py
LLM_CANVAS_PROVIDER=gemini    uv run bot.py

# Run tests
uv run pytest -q

# Run the eager-dispatch latency benchmark (manual; not pytest-collected)
.venv/bin/python tests/bench_canvas_latency.py --iterations 50
```

`.env` should include:

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

For Pipecat Cloud testing where the agent's container can't reach `localhost`, point `HV_API_URL` (passed via the runner-args body from the backend) at a publicly-accessible URL — production backend, or your local tunneled via ngrok/cloudflared.

---

## Deployment — Pipecat Cloud

Production runs on Pipecat Cloud. The Dockerfile in this repo is the build manifest. Backend's live-room start endpoint creates a Daily room and registers the Pipecat Cloud agent for that room; the agent boots, reads room metadata, fetches the snapshot (with the `include_all_scene_knowledge` flag), joins via DailyTransport.

`LLM_CANVAS_PROVIDER` is set as a Pipecat Cloud env var. Changing it requires a redeploy.

**Dockerfile gotcha:** the build uses `pip install --no-cache-dir .` against `pyproject.toml`, but `pyproject.toml` has `packages = []` and `py-modules = []` — so the install step pulls **only dependencies**, not the agent's Python code. Agent code reaches `/app/` exclusively via the explicit `COPY` lines. Adding a new top-level directory (e.g. `services/billing/`) requires a matching `COPY billing billing` line, or the cloud container will hit `ModuleNotFoundError` at session start while local dev keeps working fine. The `.dockerignore` already excludes `__pycache__/`, `*.pyc`, `.venv/`, `.env*`, `.git/`, `README.md`, `env.example`.

---

## Conventions Claude Code should follow

- **Match the installed pipecat-ai version's import paths.** Pipecat's module layout shifts between versions. Don't assume `from pipecat.services.openai.llm import OpenAILLMService` works — older versions had `from pipecat.services.openai import …`. Grep `bot.py` first.
- **Tool names must satisfy `^[a-zA-Z0-9_-]+$`.** Don't reintroduce dots. The Daily wire-format `type` field is separate and may use dots.
- **Don't add eager-dispatch entries for verbs that have required args.** `seek`, `set_speed`, `goto_scene`, `draw_arrow`, `add_annotation` all take args; firing eagerly on just the verb loses them. The hard rule: a verb goes in `EAGER_DISPATCH_VERBS` only if its handler is provably correct with `args={}`.
- **Don't modify `tools/canvas_protocol_tools.py` for new Page types.** The 5 generic tools are page-agnostic. New Page types are accommodated through their manifest (delivered via `canvas.register`), not through new tools. Adding a 6th canvas tool is a major architectural change — discuss before implementing.
- **Don't add new Daily message types without coordinating with the frontend.** `DailyRelay` in `lib/canvas-protocol/daily-relay.ts` is the sole consumer and producer of `canvas.*` messages; adding a new type requires a frontend change.
- **Don't call `api_navigate` from agent-side scene-change handlers.** The frontend's `navigateToIndex` already advances the cursor. Calling it again from the agent double-steps. (This was a real bug pre-unification.)
- **Don't unify vision + main LLM in a way that loses the separability path.** V2.14 wants vision to be hard-pinned. Today's combined behavior is acceptable but the eventual split should remain feasible.
- **Don't add `__init__.py` to `tools/`, `context/`, `services/`** unless you also update the Dockerfile (and check that namespace-package behavior isn't being relied on elsewhere). The current setup uses Python 3.12 implicit namespace packages.
- **When extending tool definitions:** prefer the `FunctionSchema` shape used by Pipecat's tool registration. The schema's `description` field is what the LLM sees — write it for the LLM. For args that the LLM keeps mis-shaping, document the expected JSON shape in BOTH the FunctionSchema description AND the system-prompt section; the LLM weighs both.
- **When adding a new top-level directory:** add `COPY <dir> <dir>` to `Dockerfile`. Catching this in CI requires booting the actual built image; local pytest doesn't catch missing `COPY` lines.
- **Don't ask the LLM to copy large structured blobs between tool calls.** It will drop the blob ~half the time and proceed without it. When a tool produces data that needs to drive a side effect, bundle the side effect into the same handler. `generate_quiz_from_knowledge` is the canonical example: it both produces the quiz blob AND dispatches `set_page(quiz, blob)` in one call, because asking the LLM to follow up with `canvas_set_page(pageType='quiz', pageInit=<blob>)` reliably failed.
- **Don't ask the LLM to pace UI animations.** It will race ahead of any reveal / transition the moment it gets a tool result. When a Page needs a result-then-transition sequence, own the pacing in the Page (cancellable timers), keep the agent as a pass-through that returns the immediate result. See `public/canvas-pages/quiz/main.js`'s `REVEAL_EXPLANATION_DELAY_MS` / `AUTO_ADVANCE_DELAY_MS`.
- **`canvas_set_page` is all-or-nothing.** When the LLM passes an empty `pageInit`, the frontend's `useOverride` flag is false and BOTH `effectivePageType` and `effectivePageInit` fall back to the snapshot. Don't assume the LLM's requested pageType "wins" when pageInit is empty — the snapshot wins. This is what makes "exit to scene" work regardless of whether the underlying scene is composition or YouTube. If you add a new Canvas Page type that needs an overlay-exit flow, the same rule applies — exit by calling `canvas_set_page(pageType=<anything>, pageInit={})`.
- **Don't declare scene-nav verbs (`next_scene` / `previous_scene` / `goto_scene`) in any Canvas Page's manifest.** They're routed by the shell's `DailyRelay.onSceneNavigation` BEFORE reaching the iframe, so listing them in the iframe's manifest is misleading (it implies iframe ownership). The agent's `handle_control` exempts them from manifest validation via `SCENE_NAV_VERBS`. If a new Page handles a verb at the iframe level, then it goes in the manifest; if it's a shell-level concern, keep it out.
- **Don't bundle `next_question` (or any inter-question advancement) into `submit_answer` on the agent.** We tried this; the iframe ripped past confetti / explanation before the visitor saw any feedback. The Quiz Page now schedules its own reveal-then-advance timer. Same lesson generalizes: result reveals belong in the Page, not in the agent's handler.

---

## Recent: S64c complete + post-cutover hardenings

Heavy iteration over many sessions. The list below is what someone reading the code should expect to find:

### Tool surface

The 5 V2.13 tools (`highlight_element`, `arrow_between`, `add_annotation`, `navigate_scene`, `clear_overlays` — actual names were `draw_arrow`, `place_annotation`, `clear_annotations`) are **deleted**. Replaced by the 5 generic protocol tools with underscored names (`canvas_analyze`, `canvas_highlight`, `canvas_control`, `canvas_action`, `canvas_set_page`). The DEPRECATED markers that were briefly on the V2.13 schemas (to steer the LLM during the transition window) are gone with the tools.

### Multi-provider LLM selection

`LLM_CANVAS_PROVIDER` lazy-imports the right SDK. Default is `openai` (only installed extra). Anthropic / Gemini paths require adding the respective pipecat extra.

### System prompt: manifest-driven CANVAS PAGE section

Wired in `bot.py`: `system_prompt = base + "\n\n" + render_canvas_page_section(manifest)`. Rebuilds at session start (manifest empty → stub), on `canvas.register` (full manifest), and on `canvas.sceneChanged` (rebuild base + re-append section).

### Element alias layer

UUID7 ids → short `<type>_<ordinal>` aliases. LLM uses aliases; handlers translate to UUIDs on dispatch. `compute_element_aliases`, `CanvasToolContext.element_alias_map`, `_resolve_element_id`.

### Scene-change refresh: unified path

Single inbound `canvas.sceneChanged` from the frontend's `navigateToIndex` drives `refresh_agent_for_current_scene`. No `on_scene_change` callback; no `api_navigate` from the agent. Covers voice nav AND visitor rail-click nav with one mechanism.

### `canvas_action` strict arg nesting

System prompt and FunctionSchema both explicitly document `args = {"from": "<element_id>", "to": "<element_id>"}` shape for `draw_arrow` and `args = {"text", "x", "y"}` for `add_annotation`. Closed a flatten-bug class where the LLM put nested keys at the top level alongside `verb`.

### Graceful handler errors

Handlers catch `CanvasCommandError` and return `{error, message, details}` to the LLM as a tool result instead of raising. A failed canvas call doesn't break the conversation turn; the LLM reads the error and self-corrects (e.g. "I can't highlight that — let me try a different way").

### Knowledge aggregation (backend + agent)

Backend `build_knowledge_snapshot(scene_id, flow_id, include_all_scene_knowledge=False)` accepts a flag; when on, folds every sibling scene's scene-scope knowledge into the flow scope. Exposed as `?include_all_scene_knowledge=true` on `/scene-snapshot`. Agent always passes the flag. Net effect: the agent's flow-knowledge prompt block is stable across scenes and contains every scene's attached knowledge.

### Frontend collaborations the agent depends on

- Composition Page uses `el.id` (real DB id) from the snapshot — matches the agent's alias resolution.
- Composition Page's `canvas.reply` payloads no longer leak internal overlay ids back to the LLM (closed a class of "LLM tries to highlight an `ovl_3` it saw in a prior tool result").
- Daily relay intercepts scene-nav verbs (`next_scene` / `previous_scene` / `goto_scene`) at the shell level via `onSceneNavigation` — the iframe never sees them, which dodges the iframe's single-scene-init rejection.
- Daily relay emits `canvas.sceneChanged` after every successful nav.
- Keyed iframe (`key={snapshot.scene_id}`) so the Composition Page reboots fresh on scene change. Without this, the page's `canvas.onInit` only fires on first mount and subsequent prepareForPageSwap goes nowhere.

### Dockerfile fix (in this round)

`COPY tools tools`, `COPY context context`, `COPY services services` added — earlier cutover deleted `canvas_actions.py` but didn't include the new directories, so cloud builds hit `ModuleNotFoundError: 'tools'` until that landed.

### Block 7 cutover completeness

`canvas_actions.py` is deleted. Both `run_bot_classic` and `run_bot_relay` use the new tools-assembly block. The Strategy 1 / Strategy 2 paths in `persona.build_system_prompt` are both alias-aware.

### Latency benchmark

`tests/bench_canvas_latency.py` runs the eager-dispatch microbench; `docs/benchmarks/canvas_latency_2026-05-11.md` is the recorded baseline (~85ms structural savings on the mock stream, expected higher in real production once the hook is actually wired into Pipecat's streaming loop).

---

## Coming next — S64d (YouTube Canvas Page)

The frontend has shipped the YouTube Canvas Page. **Pipecat continues to require zero code changes.** When a visitor navigates to a YouTube scene:

1. Frontend unmounts the composition iframe and mounts the YouTube iframe.
2. YouTube Page registers its manifest (`pageType: 'youtube'`, verbs `play / pause / seek / set_speed / clear`, highlight targets `['box']`).
3. Daily relay forwards `canvas.register` to Pipecat.
4. Agent's `on_app_message` routes to `CanvasManifestRegistry.set_manifest`; the prompt rebuild fires immediately, so the LLM sees `play / pause / seek / set_speed / clear` on its next turn.
5. The LLM uses `canvas_control` with the new verbs; the agent's tool handler is unchanged.

**Expected runtime characteristics:**

- `play`, `pause`, `clear` are already in `EAGER_DISPATCH_VERBS` and will eager-fire (once eager dispatch is wired). `seek` and `set_speed` wait for full streaming because of their required args.
- `canvas_highlight` on a YouTube scene takes `target = {box: [x, y, w, h]}` (the YouTube Page declares box-only targets in its manifest). The composition path is element_id-only.
- `canvas.set_page` is registered but currently unused for transitions; Page switches happen via scene navigation.

If a new Page type introduces verbs the agent should recognize as arg-less for eager dispatch, add them to `EAGER_DISPATCH_VERBS`. Otherwise no code changes.

---

## Coming after — S64e (Quiz Canvas Page + end-to-end `canvas.set_page`)

S64e adds the Quiz Page and wires `canvas.set_page` for explicit Page transitions (today's transitions are via scene navigation). The Quiz Page's manifest will declare verbs like `next_question`, `previous_question`, `restart`, `submit_answer`. `next_question`, `previous_question`, `restart` are already in `EAGER_DISPATCH_VERBS` — they were pre-allowlisted in S64c specifically so S64e doesn't need code changes here. `submit_answer` takes required args (the answer choice) and will take the regular `stop_reason` path.

`canvas_set_page` execution wiring is the new mechanic — currently the handler validates `pageType` against the allowlist and dispatches the Daily message, but the frontend doesn't yet respond to `canvas.set_page` commands as "swap the iframe". S64e wires that.

---

## Out of scope

- Mid-session provider switching. `LLM_CANVAS_PROVIDER` is fixed at boot.
- Natural-language highlight targets. `canvas_highlight` takes `element_id` (composition) or `box` (YouTube); resolving "the red button" in prose belongs to a future analyze-then-highlight composition.
- Persistent iframe shell. Per-scene unmount + keyed-iframe-remount is the current model; persistent shell with internal page swapping is a future optimization.
- A/B testing infrastructure for comparing providers in production.
- Vision provider separation. Today's vision goes through the same LLM as the main path; V2.14's split-vision design is feasible but unimplemented.
- Eager-dispatch-to-Pipecat-streaming-loop wiring. Hooks are constructed; their `on_stream_event` / `on_chunk` methods are never invoked by Pipecat. The latency win is structural, not yet realized in production.
