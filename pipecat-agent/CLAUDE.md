# pipecat-agent — CLAUDE.md

> Voice agent for **Human Virtual** (hv.ai), built on the **Pipecat Framework**.
> **Status:** Sessions through 64b complete · **S64c (generic 5-tool surface + multi-provider eager streaming dispatch) is next**.
> **Repo:** `pipecat-agent/` (lives alongside `human-virtual-backend/` and `human-virtual-frontend/`).

---

## What this service does

The Pipecat agent runs the avatar's **conversation pipeline**: STT → LLM → TTS → audio out, with optional vision and canvas tool calls. It connects to a Live Room over WebRTC and:

1. Listens to the visitor's microphone (Deepgram STT).
2. Calls an LLM (currently Anthropic Claude Sonnet 4 by default) with a system prompt assembled from the Live Room's persona, knowledge, current scene instruction, and canvas state.
3. Emits TTS audio (Cartesia) plus avatar lip-sync data (SoulX-Flashtalk for "talking" display mode).
4. Calls **canvas tools** when the visitor asks the avatar to highlight an element, draw an arrow, navigate scenes, etc.
5. (S46+) Calls **vision** (OpenAI GPT-4.1) when the visitor asks about screen contents.

---

## Stack

| Layer | Choice |
|---|---|
| Framework | Pipecat (Python) |
| Transport (prod) | DailyTransport (Daily.co WebRTC) |
| Transport (local) | SmallWebRTCTransport |
| Deployment (prod) | Pipecat Cloud |
| LLM (default) | Anthropic Claude Sonnet 4 |
| LLM (alt) | OpenAI GPT-4o, Google Gemini 2.0 Flash |
| Vision | OpenAI GPT-4.1 (hardcoded for canvas snapshot understanding — S46) |
| STT | Deepgram |
| TTS | Cartesia |
| Lip-sync | SoulX-Flashtalk (S48 — for "talking" display mode) |

---

## Project structure

```
pipecat-agent/
  agent.py                   # Main entrypoint — pipeline assembly, transport setup
  config.py                  # Settings (env vars)
  context/
    prompt_builder.py        # System prompt assembly (sandwich pattern)
    knowledge.py             # Knowledge / FAQ rendering
  tools/
    canvas_actions.py        # V2.13 hardcoded canvas tools (REMOVED in S64c)
    canvas_protocol_tools.py # NEW in S64c — 5 generic protocol tools
  services/
    eager_dispatch/          # NEW in S64c — per-provider streaming hooks
      __init__.py
      anthropic_adapter.py
      openai_adapter.py
      gemini_adapter.py
  tests/
    test_eager_dispatch.py   # NEW in S64c
    test_canvas_protocol_tools.py
  README.md
```

---

## Pipecat pipeline (high level)

```
DailyTransport.input  ->  Deepgram STT
                      ->  user_context_aggregator
                      ->  LLM (Anthropic / OpenAI / Gemini, with tools)
                      ->  assistant_context_aggregator
                      ->  Cartesia TTS
                      ->  SoulX (when display_mode=='talking')
                      ->  DailyTransport.output
```

The LLM service has **canvas tools registered** as function-calling tools. When the LLM emits a tool call, Pipecat's function-call aggregator invokes the registered handler. The handler emits a Daily app-message to the frontend (the live-room shell handles it and dispatches to the Canvas Service).

---

## System prompt — the sandwich pattern

`context/prompt_builder.py` assembles the system prompt in a fixed order. The pattern locks LANGUAGE on both ends ("sandwich") to keep multilingual output consistent:

**Current order (post-S64b):**

```
1. LANGUAGE         (open)
2. PERSONA
3. AUDIENCE
4. KNOWLEDGE
5. SCENE INSTRUCTION
6. DISPLAY MODE
7. CANVAS ELEMENTS
8. CANVAS ACTIONS   (V2.13 — describes 5 hardcoded tools)
9. LANGUAGE         (close)
```

**Coming in S64c order:**

```
1.  LANGUAGE          (open)
2.  PERSONA
3.  AUDIENCE
4.  KNOWLEDGE
5.  SCENE INSTRUCTION
5b. CANVAS PAGE       (NEW — describes active Page's manifest verbs)
6.  DISPLAY MODE
7.  CANVAS ELEMENTS
8.  CANVAS ACTIONS    (REPLACED — describes 5 generic protocol tools)
9.  LANGUAGE          (close)
```

The `CANVAS PAGE` section is positioned right after `SCENE INSTRUCTION` because Page type can change with scene navigation (S64d/e via `set_page`). Having it adjacent to scene context makes the LLM's verb selection more reliable.

**For Anthropic specifically (S64c):** the prompt prefix LANGUAGE → CANVAS PAGE will be marked with `cache_control: {"type": "ephemeral"}` for Anthropic prompt caching. Sections 6–9 are the dynamic suffix. `build_system_prompt_split()` returns the (stable_prefix, dynamic_suffix) tuple.

OpenAI and Gemini benefit from automatic prefix caching — same structural prefix gives them implicit cache hits even without explicit markers.

---

## Canvas tools — current (V2.13, post-S64b)

The agent currently has **5 hardcoded tools** that ship V2.13-shape Daily app-messages:

| Tool | Daily message shape |
|---|---|
| `highlight_element(id)` | `{type: 'canvas_tool', name: 'highlight_element', args: {id}}` |
| `arrow_between(from, to)` | `{type: 'canvas_tool', name: 'arrow_between', args: {from, to}}` |
| `add_annotation(text, x, y)` | `{type: 'canvas_tool', name: 'add_annotation', args: {text, x, y}}` |
| `navigate_scene(direction)` | `{type: 'canvas_tool', name: 'navigate_scene', args: {direction: 'next' \| 'previous'}}` |
| `clear_overlays()` | `{type: 'canvas_tool', name: 'clear_overlays', args: {}}` |

These live in `tools/canvas_actions.py`. Each handler:

1. Validates args.
2. Sends the Daily app-message via the transport.
3. Returns "ok" to the LLM (fire-and-forget — the V2.13 path doesn't wait for canvas confirmation).

**The frontend's translation shim** (S64b — `src/lib/canvas-protocol/translation-shim.ts`) catches these V2.13-shape messages and converts them to Canvas Protocol dispatches. The agent has no idea the protocol exists; it just emits V2.13 messages. The shim makes everything work.

**This is replaced in S64c.** The agent will emit protocol-shape messages directly. The shim will be removed. See "Coming in S64c" below.

---

## LLM provider — current (post-S64b)

Currently Anthropic Claude Sonnet 4 is hardcoded as the LLM service in `agent.py`:

```python
from pipecat.services.anthropic.llm import AnthropicLLMService
llm = AnthropicLLMService(
    api_key=os.environ["ANTHROPIC_API_KEY"],
    model="claude-sonnet-4-20250514",
)
```

There's no provider abstraction yet. Vision (S46) is a separate service (OpenAI GPT-4.1) that runs alongside.

S64c introduces `LLM_CANVAS_PROVIDER` env var (anthropic|openai|gemini) and three adapter classes — see "Coming in S64c".

---

## Vision (S46) — separate, hardcoded

`POST /live-rooms/by-slug/{slug}/scene-snapshot/image` (backend) returns a Pillow-rendered base64 PNG of the current scene. When the visitor asks "what's on screen?", the agent calls the OpenAI GPT-4.1 vision API with the image and the question.

**Vision provider does NOT change with `LLM_CANVAS_PROVIDER`.** Vision is hardcoded on OpenAI GPT-4.1 because (a) GPT-4.1 vision is currently best-in-class for canvas understanding and (b) the volume is low enough that the separate OpenAI key is justified. Keep `OPENAI_API_KEY` set even when running with `LLM_CANVAS_PROVIDER=anthropic` or `LLM_CANVAS_PROVIDER=gemini`.

---

## Live Room snapshot consumer

The agent fetches `GET /live-rooms/by-slug/{slug}/scene-snapshot` from backend on session start (and on scene navigation). The snapshot includes:

- `live_room`: language, persona, recipient_prompt
- `current_scene`: name, instruction, display_mode, background_url, elements, link_url, link_source
- `flow_state`: scene_index, total_scenes, scene_ids array
- `knowledge`: text content from extracted sources
- `faqs`: array of {question, answer}

The agent passes these into `prompt_builder.build_system_prompt(...)` to assemble the per-turn system prompt.

**Do not duplicate snapshot logic locally.** If the agent needs additional context, extend the snapshot endpoint in the backend rather than fetching multiple endpoints. The backend's `LiveRoomService.get_scene_snapshot()` is the single source of truth.

The snapshot's per-scene shape is shared with the **composition Canvas Page** (S64b — frontend). Don't change the snapshot's per-scene shape without coordinating with the composition Page bundle (`public/canvas-pages/composition/main.js` in human-virtual-frontend) and the Pipecat consumer (this agent). The backend has regression tests in `tests/test_scenes_canvas_page_type.py` to catch drift.

---

## Daily app-messages — current (post-S64b)

**Outgoing (agent → frontend):**

- `{type: 'canvas_tool', name, args}` — V2.13 canvas tool call (intercepted by frontend shim).
- `{type: 'caption', text, language}` — caption strip update.
- `{type: 'lip_sync', frames}` — SoulX lip-sync frame data when display_mode == 'talking'.
- `{type: 'session_event', event, payload}` — session lifecycle events.

**Incoming (frontend → agent):**

- `{type: 'navigate', direction}` — visitor clicks scene rail (frontend tells agent to advance scene context).
- `{type: 'session_event', event, payload}` — visitor session lifecycle.

**S64c will add:** new outgoing message type `{type: 'canvas.command', commandId, tool, verb?, args}` (replaces V2.13 `canvas_tool`). New incoming types: `canvas.register`, `canvas.stateChange`, `canvas.commandResult`, `canvas.commandError` — replied by the frontend Canvas Service.

---

## Local development

```bash
# Set up
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run agent locally with WebRTC server in browser
python agent.py --transport=local

# Run against a live room slug (production-like)
DAILY_API_KEY=... python agent.py --transport=daily --slug=<live-room-slug>

# Run tests
pytest -q
```

`.env.local` should include:

```
ANTHROPIC_API_KEY=...
OPENAI_API_KEY=...                # for vision
GOOGLE_AI_API_KEY=...             # for Gemini (optional in current state, required for LLM_CANVAS_PROVIDER=gemini in S64c)
DEEPGRAM_API_KEY=...
CARTESIA_API_KEY=...
DAILY_API_KEY=...
HV_API_BASE=http://localhost:8000  # backend
HV_LIVE_ROOM_SLUG=...              # for local testing against a specific room
```

---

## Deployment — Pipecat Cloud

Production runs on Pipecat Cloud. The deployment manifest (in `deploy.toml` or equivalent) bundles `agent.py` and dependencies. Pipecat Cloud handles per-session container provisioning when a Daily room is created.

Backend's live-room start endpoint creates a Daily room and registers the Pipecat Cloud agent for that room. The agent boots, reads room metadata, fetches the snapshot, joins via DailyTransport.

---

## Conventions Claude Code should follow

- **Match the installed pipecat-ai version's import paths.** Pipecat's module layout has shifted between versions. Don't assume `from pipecat.services.openai.llm import OpenAILLMService` works — older versions had `from pipecat.services.openai import OpenAILLMService`. Match what the existing code in `agent.py` uses.
- **Don't introduce a provider abstraction prematurely.** S64c handles multi-provider via direct branching in `agent.py` (if/elif on `LLM_CANVAS_PROVIDER`). A general LLM service factory can come later if the codebase needs it; v0.1 doesn't.
- **Don't add async sessions or asyncio primitives in tool handlers without considering Pipecat's pipeline lifecycle.** Pipecat manages its own event loop; tool handlers run inside it. Use `asyncio.Future` for awaitable results (S64c does this for canvas command results).
- **Don't unify vision + main LLM.** They're separate by design. Vision is hardcoded on OpenAI GPT-4.1 and stays that way regardless of `LLM_CANVAS_PROVIDER`.
- **Don't add new Daily message types without coordinating with the frontend.** The frontend's `LiveRoomCanvas.tsx` has a Daily app-message handler that routes by `type`. Adding a new type requires a frontend code change to handle it.
- When extending tool definitions: prefer the `FunctionSchema` shape used by Pipecat's tool registration. The schema's `description` field is what the LLM sees — write it for the LLM, not for human readers.

---

## Recent: S64a + S64b complete (no Pipecat changes)

S64a and S64b were **frontend + backend only.** Pipecat's tool surface and system prompt are unchanged from V2.13.

The frontend gained the Canvas Protocol substrate (S64a) and rendered the live-room visitor view through a sandboxed iframe (S64b). The frontend's translation shim catches Pipecat's V2.13-shape Daily messages and converts them to protocol dispatches — the agent has no idea any of this happened.

End-user behavior is pixel-identical to V2.13. The agent's tool call success rate, latency, and error patterns should all match pre-S64b baselines.

---

## Coming in S64c (next session — heavy refactor)

S64c is **the big Pipecat session.** The full tool surface flips. New files, new system prompt structure, new per-provider eager-dispatch adapters. Major changes:

### Tool surface

The 5 V2.13 tools (`highlight_element`, `arrow_between`, `add_annotation`, `navigate_scene`, `clear_overlays`) are **deleted**. Replaced by 5 generic protocol tools:

```
canvas.analyze(question, options={})
canvas.highlight(target, options={})
canvas.control(verb, args={})
canvas.action(verb, args={})
canvas.set_page(pageType, pageInit={})
```

Where `verb` is enum'd per the active Canvas Page's manifest. For composition (the only Page in v0.1):

- control verbs: `next_scene` | `previous_scene` | `goto_scene` | `clear`
- action verbs: `draw_arrow` | `add_annotation`

Tool handlers in `tools/canvas_protocol_tools.py` build a wire-format `canvas.command` payload, send it as a Daily app-message, and await the matching `canvas.commandResult` reply via an `asyncio.Future` (managed by `PendingCommandRegistry`).

### System prompt restructure

`context/prompt_builder.py` adds a `CANVAS PAGE` section between SCENE INSTRUCTION and DISPLAY MODE. The section renders dynamically from the active Page's manifest (held in `context/canvas_manifest.py` — `CanvasManifestRegistry`).

The `CANVAS ACTIONS` section is replaced — instead of describing 5 hardcoded tools, it describes the 5 generic protocol tools with usage notes.

For Anthropic, `build_system_prompt_split()` returns `(stable_prefix, dynamic_suffix)` so the Anthropic LLM service can apply `cache_control: {"type": "ephemeral"}` on the stable prefix.

### Multi-provider LLM selection

A new env var `LLM_CANVAS_PROVIDER` (anthropic|openai|gemini, default anthropic) selects the main LLM at boot. `agent.py` instantiates the appropriate Pipecat LLM service:

```python
if provider == "anthropic":
    llm = AnthropicLLMService(...)
    eager_hook = AnthropicEagerHook(...)
elif provider == "openai":
    llm = OpenAILLMService(...)
    eager_hook = OpenAIEagerHook(...)
elif provider == "gemini":
    llm = GoogleLLMService(...)
    eager_hook = GeminiEagerHook(...)
```

### Eager streaming dispatch

Three per-provider adapters (`services/eager_dispatch/{anthropic,openai,gemini}_adapter.py`) peek at the LLM's streaming events. When the LLM finishes streaming the verb token for an arg-less verb, the adapter fires the canvas command immediately — without waiting for `stop_reason: tool_use`. Saves 100–250ms per arg-less call.

Arg-less verb registry (hardcoded constant in `services/eager_dispatch/__init__.py`):

```python
EAGER_DISPATCH_VERBS = frozenset({
    "next_scene", "previous_scene", "clear",
    "next_question", "previous_question", "restart",
    "play", "pause",
})
```

Verbs not in this set wait for full tool-call completion as before. The per-provider adapter implementations:

- **Anthropic:** reads `input_json_delta.partial_json` chunks; runs a regex (`"verb"\s*:\s*"X"\s*(,|})`) against accumulated text to detect verb completion.
- **OpenAI:** reads `delta.tool_calls[].function.arguments` (string) chunks; same regex detection.
- **Gemini:** reads `candidates[].content.parts[].function_call`; whole-args path (dict) gets immediate detection, partial-args path falls back to the shared regex. Eager benefit on ~60% of Gemini calls.

Double-dispatch safety: when eager fires, the `PendingCommandRegistry` stores the future with `eager_dispatched=True`. When `stop_reason` arrives and the regular tool handler runs, it checks `pending.is_eager(commandId)` first; if true, it just awaits the existing future without re-sending the Daily message.

### Daily message handlers (new)

`agent.py` adds an `on_app_message` handler routing four new incoming message types:

- `canvas.register` → `CanvasManifestRegistry.set_manifest(message)` — Page declared its capabilities; system prompt's CANVAS PAGE section will reflect this on next turn.
- `canvas.stateChange` → `CanvasManifestRegistry.update_state(...)` — Page's semantic state changed; cached for next analyze() call.
- `canvas.commandResult` → `PendingCommandRegistry.resolve(commandId, result)` — completes the awaiting tool handler's Future with success.
- `canvas.commandError` → `PendingCommandRegistry.reject(commandId, error)` — completes the Future with `CanvasCommandError`.

### Cleanup

After end-to-end verification gates pass:

- `tools/canvas_actions.py` — DELETED.
- `llm.register_function("highlight_element", ...)` and the other 4 V2.13 registrations — DELETED.
- The frontend's translation shim and feature flag are also removed (in the frontend repo).

After S64c: the agent speaks the Canvas Protocol natively. Same end-user behavior as V2.13, but with a 100–250ms latency improvement on common arg-less verb calls and a stable abstraction that supports YouTube (S64d) and Quiz (S64e) Pages without further tool-surface changes.

---

## Out of scope for S64c (don't try to do these)

- `canvas.set_page` execution path. The tool is registered and the agent can call it, but the only allowed pageType is `'composition'` in v0.1; the dispatch validates against the allowlist and rejects anything else. S64e wires `set_page` end-to-end with Quiz Page.
- YouTube Canvas Page (S64d). Quiz Canvas Page (S64e).
- `analyze()` with a vision provider — semanticState only in v0.1.
- Natural-language highlight targets (e.g., "the second button on the left"). v0.1 highlight accepts only `element_id` or `box`.
- Mid-session provider switching. `LLM_CANVAS_PROVIDER` is set at boot and fixed for the session.
- Persistent iframe shell (still per-scene unmount in v0.1).
- A/B testing infrastructure for comparing providers in production.
