# CLAUDE.md — `pipecat-agent`

> Last updated: Session 63 (Knowledge extraction at Live Room generation; agent picks up extracted link content for free via S56 RAG. Block 7 narration-mode directive applied — LINK NARRATION section now sits between KNOWLEDGE and SCENE INSTRUCTION). Canonical agent state: Session 63 (Block 7 applied).

This file is the operating manual for Claude Code working in the Pipecat voice-agent repo. Read before making changes.

---

## What this repo is

The **Pipecat voice agent** for **Human Virtual** (hv.ai). A standalone Python service that orchestrates a real-time conversation between a visitor and an AI avatar inside a Live Room.

The agent's job:

1. **Connect to Daily** (production) or SmallWebRTC (local dev) for WebRTC audio.
2. **Fetch a scene-snapshot** from the backend (`/api/v1/live-rooms/{slug}/scene-snapshot`) — single source of truth for everything: persona, knowledge (S56 — now includes link-extracted content from S63), instruction, display mode, canvas elements, scripts, flow state, language, recipient prompt, link.
3. **Configure Deepgram STT** per-room based on `language` (S61), with a `"multi"` multilingual fallback.
4. **Build a system prompt** in the agreed sandwich order (§5) and pass it to the LLM.
5. **Stream** transcripts back to the live-room frontend over a side channel and emit canvas-action function calls (S47) when the LLM decides to use them.

It does NOT own: HTTP endpoints, browser UI, persistent storage. It's a stateless transcript producer + canvas-action function caller.

Domain: Pipecat Cloud as `<agent-name>` per environment.

---

## Tech stack

| Layer | Choice |
|---|---|
| Framework | Pipecat |
| Transport (prod) | Daily WebRTC |
| Transport (dev) | SmallWebRTCTransport |
| STT | Deepgram (`nova-2`, language-configurable per S61) |
| TTS | Cartesia (`sonic-2`, multilingual best-effort) |
| LLM | OpenAI (gpt-4.1) |
| Vision | OpenAI gpt-4.1 multimodal (S46 — Pillow-rendered scene snapshot) |
| Logging | structlog |
| Errors | Sentry |
| HTTP | httpx (async) |
| Settings | pydantic-settings |
| Package mgr | `uv` |

Local dev: `uv run python bot.py` from `pipecat-agent/`. Connects to local backend at `http://localhost:3001`.

---

## Repository layout

```
pipecat-agent/
  bot.py                          # Entry point — wires the pipeline
  config.py                       # pydantic-settings; DEEPGRAM_LANGUAGE_MAP (S61)
  scene_context.py                # Snapshot → system-prompt builder
                                  #   S61: LANGUAGE_NAMES, build_language_directive,
                                  #         build_language_reminder, RECIPIENT_PREAMBLE,
                                  #         build_recipient_context
                                  #   S63 (Block 7, OPTIONAL): NARRATION_MODE_DIRECTIVES,
                                  #         build_link_narration_directive
  scene_image.py                  # Fetches the Pillow-rendered scene image (S46)
  canvas_actions.py               # S47 — function-call schemas + dispatcher
  transcript_forwarder.py         # S45 — streams transcripts to frontend
  speaking_state_notifier.py      # S45 — emits avatar speaking/listening events
  hv_api.py                       # httpx wrapper for backend calls
tests/
  test_scene_context.py
  test_canvas_actions.py
  test_language_handling.py
  test_link_narration_directive.py  # S63 Block 7 (applied)
```

---

## Critical conventions — read before editing

### 1. The snapshot is the agent's input. Period.

- On session start, the agent calls `GET /api/v1/live-rooms/{slug}/scene-snapshot` once.
- **Snapshot is loaded once per session.** Mid-session changes don't propagate. Visitors must reconnect.
- Newer snapshots may have fields older agents don't know about. **Always treat optional fields as optional** (`.get("field")` on dicts, `getattr(obj, "field", None)` on Pydantic models).
- **The snapshot's `link` field** (S62A) carries `{url, source, embed_url, narration_mode}`. The agent does **not** read this directly to inject link content — link content arrives via the `knowledge` field after S63 extraction completes (see §11). The agent may use `link.narration_mode` if Block 7 was applied (see §11).

### 2. System prompt sandwich pattern

The system prompt is assembled in a fixed order. Don't reorder without understanding why each section sits where it does:

```
1. LANGUAGE directive       ← top — establishes conversation language
2. PERSONA                  ← who the avatar is (persona_prompt from backend)
3. AUDIENCE                 ← S61 — recipient_prompt verbatim (RECIPIENT_PREAMBLE)
4. KNOWLEDGE                ← S56 — RAG-retrieved knowledge chunks; S63: includes link extractions
5. LINK NARRATION (optional)← S63 Block 7 — how to use the linked content (per narration_mode)
6. SCENE INSTRUCTION        ← S40 — creator-authored per-scene instruction
7. DISPLAY MODE             ← S39 — normal / invisible / 3dgs / talking
8. CANVAS ELEMENTS          ← S46 — describes what's visible
9. CANVAS ACTIONS           ← S47 — function-call schemas (highlight, arrow, …)
10. LANGUAGE reminder       ← bottom — restates the language directive
```

**The "sandwich" naming refers to LANGUAGE bracketing the prompt at top + bottom.** Empirically the most reliable way to keep the LLM speaking the configured language across long conversations.

**S63 LINK NARRATION section is OPTIONAL — only present if Block 7 of the S63 session guide was applied.** If absent, the agent uses link knowledge implicitly via the KNOWLEDGE section without an explicit directive. This is fine — the LLM does a reasonable default, just less stylistically consistent across narration-mode choices.

### 3. Language handling (S61)

- `config.py`: `DEEPGRAM_LANGUAGE_MAP: dict[str, str]` maps our 9 codes (`en`, `es`, `fr`, `de`, `pt`, `ja`, `ko`, `vi`, `zh`) to Deepgram model codes. `resolve_deepgram_language(code)` falls back to `"multi"` for unmapped codes.
- `scene_context.LANGUAGE_NAMES: dict[str, str]` maps codes to English names. Used by `build_language_directive` for natural phrasing.
- **Adding a 10th language: 4 places.** `LIVE_ROOM_LANGUAGES` (frontend); `LiveRoomLanguage` Literal + `ck_live_rooms_language` migration (backend); `DEEPGRAM_LANGUAGE_MAP` + `LANGUAGE_NAMES` (this repo). Forgetting one = silent or visible failure.
- **Cartesia TTS multilingual is best-effort.** Cloned voices in non-source languages sound accented. Documented caveat.

### 4. Recipient prompt is creator-authored steering — pass it through, don't transform it

- `recipient_prompt` is creator-authored, max 2000 chars (backend enforced).
- AUDIENCE section built by `build_recipient_context(prompt)` — prepends `RECIPIENT_PREAMBLE`, inserts the prompt verbatim.
- **Don't summarize, sanitize, or reword `recipient_prompt`.** Creators expect their text to reach the LLM as-is.
- Empty `recipient_prompt` → omit the AUDIENCE section entirely.

### 5. Canvas actions are function calls, not free text

- 5 function calls (S47): `highlight_element`, `add_arrow`, `add_annotation`, `navigate`, `clear_canvas`.
- Dispatched via `transcript_forwarder` to the live-room frontend.
- **Never render canvas actions as text.** "I'm pointing at the Sign Up button" is a UX failure — emit `highlight_element({id: "btn-signup"})` and let the frontend draw the SVG overlay.

### 6. The Pillow-rendered scene image is the agent's vision input

- Backend renders 1280×720 PNG via `scene_renderer.py`, exposed at `/api/v1/live-rooms/{slug}/scene-image`. Public endpoint.
- Agent fetches once on session start (and on every scene navigation in a flow). Passed to gpt-4.1 multimodal alongside the text system prompt.
- The image carries: background, all canvas elements (text, image, shape, button) in their actual positions.
- **The renderer does NOT yet show the link layer (S62A) — `link_url` is ignored by Pillow.** S64+ may extend the renderer to draw a link placeholder; for now the agent learns about the link only via the text snapshot + the S56 KNOWLEDGE section.

### 7. Display mode shapes the prompt + the rendering pipeline

- `display_mode`: `normal` / `invisible` / `3dgs` / `talking`.
- `talking` (S48) uses SoulX-Flashtalk on the frontend.
- `invisible` mode tells the LLM "you're a disembodied voice."
- `build_display_mode_directive(mode)` — terse, one sentence.

### 8. Snapshot fields the agent intentionally ignores

The "data ships now / runtime ships later" pattern from the backend is mirrored here. Some snapshot fields exist but are silently dropped today:

- **Survey** (S58) — visitor-facing client-side UX; agent doesn't reference. Permanent.
- **`link.url`, `link.source`, `link.embed_url` (S62A)** — the agent doesn't read these directly. Link CONTENT is consumed via the KNOWLEDGE section after S63 extraction populates a `source_type='link'` knowledge_source row. The agent code stays unaware of *where* the knowledge came from — it just sees text in KNOWLEDGE.
- **`link.narration_mode` (S62A)** — consumed only if S63 Block 7 was applied (LINK NARRATION section). If not, ignored.
- **Future fields** — silently ignore unknown snapshot fields. Forward compat wins.

### 9. The agent is stateless across sessions

- Each session = fresh process or fresh `BotPipeline`. No cross-session memory.
- Daily tracks session IDs; the agent doesn't manage session state.
- LLM has no recollection of previous conversations. Persistent memory = creator-side feature (knowledge, FAQ).

### 10. HV_API_URL points to the backend

- `config.HV_API_URL`: backend base URL. Env var.
- All HTTP calls go through `hv_api.py`.
- Public endpoints only — same as the live-room frontend.

### 11. Link knowledge integration (S63)

This is the load-bearing fact about S63 from the agent's perspective:

> **The agent has zero code change in S63 for link knowledge consumption.**

When the creator generates a Live Room with a scene that has a link, the backend's `extract_link_knowledge` Celery task pulls content from the link (Gemini for YouTube, FireCrawl for the rest) and persists it as a `knowledge_source` row with `source_type='link'`. The existing S56 `knowledge_snapshot` already retrieves all `knowledge_source` rows for the scene and assembles them into the snapshot's `knowledge` field. The agent already reads `knowledge` and injects it as the KNOWLEDGE section of the system prompt.

So: extracted link content "just works" through the existing pipe.

**Timing caveat:** snapshot is loaded once per session. If the visitor connects before extraction completes, KNOWLEDGE will lack the link content. The visitor must reconnect after extraction is `ready` (frontend status indicator surfaces this for the creator). Documented v1 behavior.

**Block 7 (LINK NARRATION directive — OPTIONAL).** If applied:

- `scene_context.NARRATION_MODE_DIRECTIVES` maps each of the 4 modes (`walk_through` / `summarize` / `answer_questions` / `reference_as_needed`) to a one-paragraph directive.
- `build_link_narration_directive(link)` returns a `LINK NARRATION:` system-prompt section when `link` is present, or empty string otherwise.
- The section sits AFTER KNOWLEDGE and BEFORE SCENE INSTRUCTION.
- Returns empty string for unknown narration modes — defensive against stale snapshots referencing modes added in a future session.

If Block 7 is **not** applied, the agent still works fine; the LLM uses link knowledge via KNOWLEDGE without a narration-style directive. Block 7 is about controlling *style*, not *whether* the knowledge is used.

---

## Cross-repo contracts

- **Snapshot shape.** Backend `app/schemas/scene.py::SceneSnapshotResponse` is canonical. Frontend (`src/types/scene.ts`) and agent (Pydantic models in `scene_context.py`) consume it. New fields are `Optional`.
- **Language enum.** 9 codes; must match across backend, frontend, agent.
- **Link source enum (S62A).** 7 IDs; backend authoritative. Agent doesn't currently consume `link.source` to inject the source's display name into a directive (Block 7 references it informally if applied).
- **Link narration mode enum (S62A).** 4 modes; consumed by Block 7 directive (if applied).
- **Knowledge field shape (S56 + S63).** Each `knowledge_source` row contributes a chunk with title + content. The agent doesn't differentiate `source_type='link'` from `source_type='file'` etc. in its prompt — it just concatenates and trusts S56's existing assembly logic. **The shape never reveals the URL or source ID to the LLM** — only the extracted content. (If you want the LLM to know "this came from YouTube," that's Block 7's LINK NARRATION section's job.)
- **Canvas action function-call schemas (S47).** Shared between agent (LLM-facing) and frontend (event-handler). Arg names must match exactly.

---

## Testing

```bash
uv run pytest               # full suite
uv run pytest -k language   # language-specific tests
uv run pytest -k context    # snapshot → prompt builder
uv run pytest -k narration  # S63 Block 7
```

- Unit-level only — no real Daily/Deepgram calls. Fixtures provide synthetic snapshots; tests assert on the assembled system-prompt string.
- New tests in S63: 6 narration-directive tests in `tests/test_link_narration_directive.py` (Block 7 applied).

---

## Common gotchas

1. **Mid-session snapshot changes don't propagate.** Restart the agent or have the visitor reconnect. By-design but trips up live debugging.
2. **Cartesia multilingual is best-effort.** Cloned voice in non-source language → accented/robotic. Document; don't fix agent-side.
3. **Deepgram unknown-language fallback.** Always falls back to `"multi"`. Never crash. New language → add to BOTH `DEEPGRAM_LANGUAGE_MAP` AND `LANGUAGE_NAMES`.
4. **System-prompt section ordering.** The sandwich is real. Reordering breaks language adherence. Add new sections inside the sandwich, not at the boundaries.
5. **Free-text canvas actions.** The LLM occasionally tries to "describe" a highlight in text instead of emitting the function call. Fight in the system prompt with explicit examples.
6. **`scene_image` 404s.** Older live rooms or scenes deleted mid-session. Fall back to text-only prompting; don't crash.
7. **Daily participant tracking.** Agent participant identified by stable participant ID (S45b). Don't add identity logic.
8. **Visitor connects before link extraction completes.** Snapshot's `knowledge` field lacks link content. Documented v1 behavior. The visitor reconnects to pick up the new KNOWLEDGE section.
9. **Treating link content differently from other knowledge.** Don't. The agent doesn't know (or care) which `knowledge_source` row came from a link versus a file upload — it's all text in KNOWLEDGE. Block 7 is the only place the *source* of link knowledge is acknowledged in the prompt.
10. **Hand-rolling Gemini context caching for YouTube.** Don't. Backend uses Redis to cache extraction results by URL across sessions (S63). Gemini's first-class CachedContent API solves a different problem (within-session prompt caching) and isn't relevant here.

---

## Session history (agent-relevant only)

| # | Title | Status |
|---|---|---|
| 37 | Pipecat scaffold | ✅ |
| 38 | Pipecat Cloud deployment | ✅ |
| 45 | Pipecat ↔ live-room integration | ✅ |
| 45b | E2E hardening | ✅ |
| 46 | Scene understanding (vision input) | ✅ |
| 47 | Canvas actions function-call schemas | ✅ |
| 48 | Talking display-mode prompt directive | ✅ |
| 49 | Auto-speak script on scene join | ✅ |
| 56 | Knowledge RAG injection in system prompt | ✅ |
| 58 | Survey fields ignored intentionally | ✅ |
| 60 | Snapshot carries language + recipient_prompt; agent ignores | ✅ |
| 61 | Multi-language Deepgram + sandwich + recipient context | ✅ |
| 62A | Snapshot carries `link`; agent ignores `link.url/source/embed_url` | ✅ |
| 62B | Frontend ship; no agent changes | ✅ |
| **63** | **Backend extracts link → knowledge_source(source_type='link') → S56 RAG → KNOWLEDGE section. Block 7 LINK NARRATION directive applied (+30 LOC `scene_context.py` + 6 tests + wired into `persona.build_system_prompt`).** | **✅ Current** |
| 64 | TBD | ⬜ Planned |

**S63 status:**
- **Knowledge consumption: zero agent code changes.** Link content arrives transparently via S56's existing pipe.
- **Block 7 (LINK NARRATION directive): APPLIED.** `NARRATION_MODE_DIRECTIVES` + `build_link_narration_directive(link)` in `scene_context.py`. Wired into both the sync `scene_context.build_system_prompt` and the async runtime `persona.build_system_prompt` (Strategies 1 and 2). Section sits between KNOWLEDGE and SCENE INSTRUCTION. 6 tests in `tests/test_link_narration_directive.py`. Returns empty string for unknown narration modes — defensive against stale snapshots referencing modes added in a future session.

---

## Environment variables

```
HV_API_URL=http://localhost:3001                  # local backend (or https://api.hv.ai)
DEEPGRAM_API_KEY=...
CARTESIA_API_KEY=...
CARTESIA_MODEL_ID=sonic-2
OPENAI_API_KEY=...
DAILY_API_KEY=...                                  # production transport
DAILY_DOMAIN=...
PIPECAT_TRANSPORT=daily                            # or "smallwebrtc" for local dev
SENTRY_DSN=...
LOG_LEVEL=INFO
```

**S63 added no new env vars.** All extraction work happens backend-side; the agent reads the result via the existing knowledge channel.

---

## When in doubt

- **New snapshot field that the agent needs to consume.** Add Pydantic field as `Optional`. Builder in `scene_context.py` returns empty string when missing. Slot the section into the sandwich at the appropriate position. Don't reorder existing sections.
- **New language.** Four-places change — see §3. Run a smoke test in the new language end-to-end.
- **New canvas action.** Update `canvas_actions.py` schema; coordinate with frontend. System-prompt examples for when to use it.
- **New external service for the agent.** Wrapper module. Settings in `config.py`. Mock in tests. Never block the pipeline event loop with sync I/O.
- **Field that should ship as data but not be consumed yet.** Document in §8. Forward-compat is non-negotiable.
- **A piece of knowledge the agent should reference differently based on metadata.** That's a Block 7-style directive — small system-prompt section that tells the LLM how to *use* something, separate from the content itself. Don't try to encode it into the content.
