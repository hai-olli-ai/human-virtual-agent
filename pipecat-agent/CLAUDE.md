# CLAUDE.md — `pipecat-agent/`

> Last updated: Session 61 (Live Room i18n + Voice Agent Multi-Language Integration). Version: V2.10.

This is the operating manual for Claude Code working in the voice agent. Read before making changes.

---

## What this repo is

The Pipecat-powered voice agent that drives real-time conversation in **Human Virtual** Live Rooms. It connects via WebRTC to the visitor's browser, listens, thinks, talks, and looks at the scene — all under one orchestrated pipeline.

It is a **separate service** from the FastAPI backend. It runs locally (for development) via `SmallWebRTCTransport` and in production via `DailyTransport` on **Pipecat Cloud**.

It is the **only** component that talks to the visitor in real time. The backend serves data; the frontend renders UI; this service IS the avatar's voice and eyes.

---

## Pipeline (Pipecat)

```
Visitor mic
    │
    ▼
SmallWebRTCTransport (local)  /  DailyTransport (prod)
    │
    ▼
Silero VAD            (S37)
    │
    ▼
Deepgram STT          (S45 — multilingual config in S61)
    │                    language=resolve_deepgram_language(snapshot["language"])
    ▼                    fallback to "multi" for unknown values
LLM (OpenAI/Anthropic via OpenAI-compatible API)   (S45)
    │                    system prompt assembled by scene_context.build_system_prompt
    │                    LANGUAGE directive (top) + AUDIENCE + ... + LANGUAGE reminder (bottom) — S61
    │
    ├──► CanvasActionsObserver        (S47) — emits highlight / arrow / annotation / navigate / clear
    ├──► VisionInjector               (S46) — fetches scene snapshot PNG, injects as image_url
    └──► KnowledgeContext             (S56) — fetched once at session start
    │
    ▼
Cartesia TTS          (S45, S51a — multilingual at output level via sonic-2)
    │
    ▼
Visitor speaker
```

Other observers / services running alongside:

- **TranscriptForwarder** (S45) — streams STT/LLM transcripts to the backend for chat-panel display.
- **SpeakingStateNotifier** (S45) — pushes `speaking_started` / `speaking_stopped` events so the frontend can drive the talking-avatar (S48 SoulX) animation.
- **Script auto-speak** (S49) — fires the scene's first script line on agent join.

---

## Tech stack

| Layer | Choice |
|---|---|
| Framework | Pipecat 0.0.x |
| Transport (local) | SmallWebRTCTransport |
| Transport (prod) | DailyTransport |
| VAD | Silero |
| STT | Deepgram (`nova-2`; per-language config + `multi` fallback — S61) |
| TTS | Cartesia (`sonic-2` — multilingual at LLM-output level, no per-language model) |
| LLM | OpenAI-compatible (GPT-4.1 vision for scene understanding) |
| HTTP | httpx (async) |
| Logger | structlog |
| Package mgr | `uv` |
| Deploy | Pipecat Cloud |

Local dev: `uv run python bot.py` from `pipecat-agent/`.

---

## Repository layout

```
pipecat-agent/
  bot.py                     # entrypoint — pipeline assembly, Deepgram language config (S61), transport selection
  scene_context.py           # builds system prompt from snapshot
                             #   - build_language_directive (S61)
                             #   - build_language_reminder (S61)
                             #   - build_recipient_context (S61)
                             #   - build_system_prompt (LANGUAGE-sandwiched, AUDIENCE between persona and knowledge — S61)
  hv_client.py               # async client for backend's public endpoints
  config.py                  # env vars + DEEPGRAM_LANGUAGE_MAP + resolve_deepgram_language (S61)
  observers/
    canvas_actions.py        # S47 — function-call observer
    transcript_forwarder.py  # S45
    speaking_state.py        # S45
    script_auto_speak.py     # S49
  services/
    vision_injector.py       # S46 — pulls snapshot PNG as image_url
    knowledge_context.py     # S56 — formats knowledge into system prompt
  tests/
    test_scene_context.py
    test_scene_context_s61.py  # S61 — language directive, audience, prompt assembly
  pyproject.toml
  README.md
  .env.example
```

---

## Critical conventions — read before editing

### 1. The agent doesn't own data — the snapshot does

- On session start, the agent calls `GET /api/v1/live-rooms/{slug}/scene-snapshot` (public, unauthenticated).
- That snapshot is the **single source of truth** for: persona, knowledge, survey (NOTE: ignored — see #4), display mode, scene elements (incl. buttons from S59), instruction, scripts, flow state, **language and recipient_prompt (consumed in S61 — Deepgram + system prompt)**.
- **Don't add new endpoint calls to the agent** for data that should be in the snapshot. Add to the snapshot instead — backend repo, `LiveRoomService.get_scene_snapshot`.

### 2. System prompt assembly lives in `scene_context.py` (post-S61)

The agent's system prompt is composed from snapshot data. Section order (post-S61):

1. **LANGUAGE directive** (top — strong steering) — S61
2. PERSONA
3. **AUDIENCE** (only when `recipient_prompt` is non-empty) — S61
4. KNOWLEDGE — S56
5. SCENE INSTRUCTION
6. DISPLAY MODE CONTEXT
7. SCENE ELEMENTS SUMMARY (incl. buttons — S59)
8. CANVAS ACTION TOOL GUIDANCE
9. **LANGUAGE reminder** (bottom — sandwich pattern) — S61

If you add a section, put it in `build_system_prompt()` in `scene_context.py`. Don't fork prompt-building elsewhere.

**Sandwich pattern rationale.** LLMs weight the first and last sections of a system prompt most heavily. A LANGUAGE directive at top + a LANGUAGE reminder at bottom materially reduces drift mid-conversation. Don't drop either — they're belt-and-suspenders by design.

### 3. Canvas tools — five and only five

The agent has these LLM function-callable tools (S47):

- `highlight_element(element_id_or_title)`
- `draw_arrow(from_x, from_y, to_x, to_y, label?)`
- `add_annotation(x, y, text)`
- `navigate_to_scene(scene_id_or_title)`
- `clear_canvas()`

**The agent has NO tool to click buttons.** Buttons (S59) are visitor-clickable only. The agent can describe them by title and use `highlight_element` to point at them, but it must NOT attempt to "click" anything — there is no such affordance. See the in-line comment in `scene_context.py` (S59).

### 4. Snapshot fields the agent treats specially

- **`survey` (S58):** ignored permanently. Survey UX is 100% client-side. The agent must not echo question text. If the agent is referencing survey content unprompted, debug on the BACKEND side (snapshot assembly), not in the agent.
- **`language` (S60 → consumed in S61):** drives Deepgram language config (`config.py::resolve_deepgram_language`) and the LANGUAGE directive at top + bottom of the system prompt.
- **`recipient_prompt` (S60 → consumed in S61):** drives the AUDIENCE section in the system prompt. Non-empty values appear after persona, before knowledge.

### 5. Knowledge is loaded once per session, not per turn

S56 wires Knowledge into the system prompt at session start. **Don't make the agent call the knowledge endpoint per turn** — it's expensive and the snapshot already carries it. If knowledge needs to update mid-session, that's a future feature (live snapshot polling), not part of S56's contract.

### 6. Display mode awareness

Snapshot includes `display_mode` ∈ `normal | invisible | 3dgs | talking`. The agent's system prompt mentions this so the LLM knows whether the visitor sees a static avatar, a 3D one, or a talking-mouth video (SoulX, S48). The agent does NOT change voice or behavior based on display mode — it just gets context.

### 7. Auto-speak the first script on join

S49 fires the scene's first script line via TTS as soon as the agent connects. **This is in addition to the LLM** — the script is a static line, not LLM-generated, so the visitor gets an immediate greeting that matches the creator's authoring. Subsequent turns are LLM-driven.

**Note (S61):** the script TEXT is creator-authored — it's spoken in whatever language the creator wrote it. If the creator set `language="es"` but wrote scripts in English, the visitor hears English text rendered in a Spanish-multilingual voice. Acceptable per the "no auto-translation" decision.

### 8. Transport selection by environment

```python
# bot.py (sketch)
if os.getenv("PIPECAT_CLOUD") == "1":
    transport = DailyTransport(...)
else:
    transport = SmallWebRTCTransport(...)
```

Pipecat Cloud sets `PIPECAT_CLOUD=1`. Local dev defaults to SmallWebRTC. Don't hardcode either; respect the env var.

### 9. Multi-language wiring (S61, shipped)

- **Deepgram STT.** `config.py::resolve_deepgram_language(snapshot.get("language"))` returns one of the 9 explicit codes (`en`, `es`, `fr`, `de`, `pt`, `ja`, `ko`, `vi`, `zh`) or falls back to `"multi"` for unknown values, or `"en"` for None/empty. The result is passed to `DeepgramSTTService(language=...)` in `bot.py`.
- **Deepgram model.** `nova-2` supports all 9 languages directly. For the `"multi"` fallback, use `nova-2-general` if your Pipecat version requires explicit model selection for multilingual mode.
- **Cartesia TTS.** `sonic-2` is multilingual at the LLM-output level. The LLM emits text in the target language, Cartesia synthesizes it. **No per-language model selection needed.** Voice ID matters: voice clones from English source produce English-accented multilingual output (known caveat — documented for creators, not handled in code).
- **LLM.** Hard "respond in {Language}" directive at top AND bottom of the system prompt. Sandwich pattern.

The 9 supported language codes match the backend's `LiveRoomLanguage` Literal (S60) and the frontend's `LIVE_ROOM_LANGUAGES` array. The DB CHECK constraint guarantees the agent never sees an out-of-set code; the `"multi"` fallback is forward compat for languages added later.

### 10. Recipient prompt (S61, shipped)

A creator-authored "audience" description (e.g. "Sales team at Acme Corp evaluating renewal — focus on ROI"). Inserted into the system prompt as a `# AUDIENCE` section between persona and knowledge. Empty / whitespace-only prompts produce no section.

The field is capped at 2000 chars by the backend (validated at write time). The agent reads as-is.

**Recipient prompt is steering, not knowledge.** It's about WHO the avatar is talking to; knowledge is about WHAT the avatar knows. Different sections, different purposes. Don't conflate them.

### 11. Adding a new language

Five places to update:

1. Backend `LiveRoomLanguage` Literal in `app/schemas/live_room.py`.
2. Backend Alembic migration: drop `ck_live_rooms_language`, recreate with the new value.
3. Frontend `LIVE_ROOM_LANGUAGES` array in `src/types/live-room.ts`.
4. Frontend new locale file in `src/lib/i18n/<code>.json` (if not already present from the studio side).
5. **In this repo:** `DEEPGRAM_LANGUAGE_MAP` in `config.py` and `LANGUAGE_NAMES` in `scene_context.py`.

If you forget the agent maps, the agent gets the new code in the snapshot but falls back to `"multi"` Deepgram and English LLM directive — degraded but functional.

---

## Cross-repo contracts

- **Backend's snapshot endpoint is public.** Don't add auth headers to the agent's snapshot fetch — it's a public contract by design (S43).
- **Snapshot field additions must be Optional.** Older agents shouldn't crash on newer snapshots, and vice versa. The agent must read snapshot fields with `.get()` defaults.
- **Canvas action events flow agent → frontend** through the Pipecat data channel (S47). Frontend listens for the events and renders SVG overlays. Don't try to render canvas actions from the agent side; the agent only emits events.
- **Snapshot is loaded once per session.** Mid-session PATCH to `language`, `recipient_prompt`, or any other field doesn't propagate until reconnect. Visitors see consistent agent behavior for the duration of a session.

---

## Testing

```bash
uv run pytest                                 # unit tests
uv run pytest tests/test_scene_context.py -v  # original assembly tests
uv run pytest tests/test_scene_context_s61.py -v  # S61 — language + audience tests
```

S61 tests cover:
- `build_recipient_context(None | "" | "  ")` → `""`
- `build_recipient_context("text")` → contains `# AUDIENCE` and the text
- `build_language_directive(code)` → mentions language name ≥2 times for all 9 codes
- `build_language_directive("xx" | None | "")` → falls back to English
- `build_language_reminder` → short, mentions the language
- `build_system_prompt` order: LANGUAGE top → PERSONA → AUDIENCE (if non-empty) → KNOWLEDGE → SCENE INSTRUCTION → DISPLAY MODE → ELEMENTS → CANVAS ACTIONS → LANGUAGE reminder bottom
- Snapshot with empty/null `recipient_prompt` → no AUDIENCE section in prompt
- Snapshot with missing `language` → defaults to English
- All 9 language codes produce a prompt with the language name appearing ≥2 times
- `resolve_deepgram_language` for all 9 codes, unknown → `"multi"`, None/empty → `"en"`

Manual smoke: run `uv run python bot.py` against a local backend with a Spanish or Vietnamese live room. Speak — listen for response in the configured language.

---

## Common gotchas

1. **Pipecat versions move fast.** Pin in `pyproject.toml`. When updating, re-test the entire pipeline; minor versions break observer APIs.
2. **Cartesia voice IDs are multilingual-capable but quality varies.** Cloned voices from English source sound English-accented in other languages. Document for creators; don't try to fix in the agent.
3. **Vision injection is expensive.** S46 injects the scene snapshot PNG every N turns (configurable). Don't inject every turn — token cost balloons.
4. **Don't `await` cold paths inside hot ones.** The pipeline is real-time; latency adds. Cache the snapshot at session start; don't re-fetch on every turn.
5. **Pipecat Cloud has cold starts.** If a session takes >5s to connect, that's a cold start, not a bug. Pre-warming is in the S69 polish backlog.
6. **Daily room URLs expire.** Backend creates them on session-start; the agent joins via the room URL passed in the start-session payload. Don't cache room URLs on the agent side.
7. **Deepgram supports 9 languages directly.** Beyond that, use `language="multi"` (auto-detect). Multi-detect is slower and lower-accuracy for short utterances; document as a caveat.
8. **`OPENAI_API_KEY` is the LLM key, not necessarily OpenAI.** The agent uses an OpenAI-compatible API; could be Anthropic via a proxy. Treat the env var as opaque "LLM key".
9. **Snapshot schema changes require an agent restart.** When the backend ships a new snapshot field (S60 added two), redeploy / restart the agent service so any cached schema validation picks up the new shape. Not a code change in this repo — just a deployment ordering note.
10. **LLM drift mid-conversation.** The sandwich pattern (LANGUAGE top + bottom) reduces drift but doesn't eliminate it on long sessions. If a session drifts, it's a tuning issue with the prompt, not a bug. Don't add complexity (translation post-processors, mid-session re-injection) without measurement.
11. **Empty `recipient_prompt` should produce NO `# AUDIENCE` section.** Don't emit an empty section header — that's a different signal to the LLM ("you have an audience but I won't tell you about it"). The absence of the section is the signal "general visitors".

---

## Session history (agent-relevant)

| # | Title | Status | Agent change |
|---|---|---|---|
| 37–38 | Pipecat scaffold + Cloud deployment | ✅ | Initial bot.py, transport selection |
| 45 | Live room voice agent integration | ✅ | TranscriptForwarder, SpeakingStateNotifier |
| 45b | Daily WebRTC hardening | ✅ | DailyTransport, agent participant tracking |
| 46 | Scene understanding (vision) | ✅ | VisionInjector — base64 PNG into LLM context |
| 47 | Canvas actions | ✅ | CanvasActionsObserver + 5 LLM tools |
| 48 | Talking display mode (SoulX) | ✅ | speaking_state events frontend-side |
| 49 | Script auto-speak | ✅ | ScriptAutoSpeak observer |
| 50 | Live room polish + analytics | ✅ | session timing telemetry |
| 51a | Cartesia voice + speech | ✅ | Cartesia TTS replaced earlier provider |
| 56 | Knowledge RAG | ✅ | KnowledgeContext loaded once per session |
| 57 | Survey backend + editor | ✅ | (no agent change — visitor UX) |
| 58 | Survey live + responses | ✅ | (no agent change — survey field ignored) |
| 59 | Interactive Button Elements | ✅ | Comment-only update in scene_context.py |
| 60 | Live Room Language + Recipient Prompt — backend + studio modal | ✅ | (no agent change — fields shipped in snapshot, ignored in S60) |
| **61** | **Live Room i18n + Voice Agent Multi-Language Integration** | **✅ Current** | **Deepgram language config + system prompt LANGUAGE/AUDIENCE sections + ~25 new tests** |
| 62 | Generate scene (knowledge-aware, language-aware) | ⬜ Next | (TBD — likely no agent change) |

S61 in this repo:

- `config.py`: added `DEEPGRAM_LANGUAGE_MAP`, `DEEPGRAM_FALLBACK_LANGUAGE`, `resolve_deepgram_language()`.
- `bot.py`: passes `language=resolve_deepgram_language(snapshot.get("language"))` to Deepgram service init; added structured log line `Deepgram language configured`.
- `scene_context.py`: added `LANGUAGE_NAMES` dict, `build_language_directive()`, `build_language_reminder()`, `RECIPIENT_PREAMBLE`, `build_recipient_context()`. Refactored `build_system_prompt()` to assemble in the new order (LANGUAGE → PERSONA → AUDIENCE → KNOWLEDGE → SCENE → DISPLAY MODE → ELEMENTS → CANVAS ACTIONS → LANGUAGE reminder).
- `tests/test_scene_context_s61.py`: ~25 new unit tests.

---

## Environment variables

```
HV_API_URL=http://localhost:3001          # backend (S45b)
LIVE_ROOM_SLUG=...                        # set per-session by backend on agent spawn
PIPECAT_CLOUD=0                           # 1 in production
PIPECAT_CLOUD_API_KEY=...                 # prod only

DEEPGRAM_API_KEY=...
CARTESIA_API_KEY=...
OPENAI_API_KEY=...                        # LLM (could be proxy to Anthropic)

DAILY_API_KEY=...                         # prod only
DAILY_DOMAIN=...                          # prod only
```

**S61 added no new env vars** — Deepgram and Cartesia keys are already configured; multi-language uses the same keys.

---

## When in doubt

- **New snapshot field in the backend?** Read it via `snapshot.get("new_field", default)` in `scene_context.py`. Older snapshots without the field must not crash. **If the field's session is "data ships now, runtime ships next" (S58 survey, S60 language/recipient_prompt → S61), do not consume it yet — wait for the runtime session.**
- **Adding to the system prompt?** Put it in `scene_context.py::build_system_prompt`. Document order in this file. New sections go between existing ones, not at the very top or very bottom — those slots are reserved for the LANGUAGE sandwich.
- **New LLM tool?** Add to the canvas-actions observer (S47) only if it's a canvas-side effect (highlight, arrow, etc.). Side effects in the visitor's browser go through the Pipecat data channel. Server-side mutations go via httpx to the backend.
- **Survey-related agent behavior?** **Stop.** Survey UX is 100% client-side. The agent's job is to ignore the `survey` snapshot field. If you're tempted to read it, debug the issue you're trying to fix elsewhere first.
- **Canvas button click handling?** The agent has no click tool. If you find yourself adding one, that's out of scope for S59 and likely all of v2 — buttons are visitor-only.
- **Adding a 10th language?** Two changes here: `DEEPGRAM_LANGUAGE_MAP` in `config.py` (verify Deepgram supports the code on `nova-2`), and `LANGUAGE_NAMES` in `scene_context.py`. Plus the four backend/frontend places — see "Adding a new language" above.
- **Mid-session language switching?** Not supported. Snapshot is loaded once per session. If a creator wants to switch language mid-conversation, the visitor must reconnect. Don't try to hot-swap Deepgram or LLM mid-stream — the complexity isn't worth the use case.
- **Auto-translation of creator content?** Not in v1. Scripts, instructions, knowledge, survey questions, recipient prompts, CTA labels are all passed through verbatim. Don't add translation services without an explicit decision.
