# CLAUDE.md — `pipecat-agent/`

> Last updated: Session 59 (Interactive Button Elements). Version: V2.10.

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
    │
    ▼
LLM (OpenAI/Anthropic via OpenAI-compatible API)   (S45)
    │
    ├──► CanvasActionsObserver        (S47) — emits highlight / arrow / annotation / navigate / clear
    ├──► VisionInjector               (S46) — fetches scene snapshot PNG, injects as image_url
    ├──► KnowledgeContext             (S56) — fetched once at session start
    └──► (S60+) LanguageDirective + RecipientPrompt — prepended to system prompt
    │
    ▼
Cartesia TTS          (S45, S51a)
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
| STT | Deepgram (`nova-2`; `language=multi` fallback in S61) |
| TTS | Cartesia (`sonic-2`) |
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
  bot.py                     # entrypoint — pipeline assembly, transport selection
  scene_context.py           # builds system prompt from snapshot
  hv_client.py               # async client for backend's public endpoints
  observers/
    canvas_actions.py        # S47 — function-call observer
    transcript_forwarder.py  # S45
    speaking_state.py        # S45
    script_auto_speak.py     # S49
  services/
    vision_injector.py       # S46 — pulls snapshot PNG as image_url
    knowledge_context.py     # S56 — formats knowledge into system prompt
  config.py                  # env vars, Deepgram language map (post-S61)
  pyproject.toml
  README.md
  .env.example
```

---

## Critical conventions — read before editing

### 1. The agent doesn't own data — the snapshot does

- On session start, the agent calls `GET /api/v1/live-rooms/{slug}/scene-snapshot` (public, unauthenticated).
- That snapshot is the **single source of truth** for: persona, knowledge, survey (NOTE: ignored — see #4), display mode, scene elements (incl. buttons from S59), instruction, scripts, flow state, and (post-S60) language + recipient prompt.
- **Don't add new endpoint calls to the agent** for data that should be in the snapshot. Add to the snapshot instead — backend repo, `LiveRoomService.get_scene_snapshot`.

### 2. System prompt assembly lives in `scene_context.py`

The agent's system prompt is composed from snapshot data. The order (post-S61) will be:

1. Language directive (top — strong steering) — S61
2. Persona
3. Audience / recipient prompt (if non-empty) — S60
4. Knowledge — S56
5. Scene instruction
6. Display mode context
7. Scene elements summary (incl. buttons — S59)
8. Canvas action tool usage guidance
9. Language reminder (bottom — reinforcement) — S61

If you add a section, put it in `build_system_prompt()` in `scene_context.py`. Don't fork prompt-building elsewhere.

### 3. Canvas tools — five and only five

The agent has these LLM function-callable tools (S47):

- `highlight_element(element_id_or_title)`
- `draw_arrow(from_x, from_y, to_x, to_y, label?)`
- `add_annotation(x, y, text)`
- `navigate_to_scene(scene_id_or_title)`
- `clear_canvas()`

**The agent has NO tool to click buttons.** Buttons (S59) are visitor-clickable only. The agent can describe them by title and use `highlight_element` to point at them, but it must NOT attempt to "click" anything — there is no such affordance. See the in-line comment in `scene_context.py` (S59).

### 4. Survey snapshot field is intentionally ignored

The `survey` field in the snapshot (added in S58) is **for the frontend, not the agent**. The agent must not echo survey question text to the visitor; doing so creates a confusing dual-experience (avatar reading questions out loud while the modal also shows them). If the agent is referencing survey content unprompted, debug on the BACKEND side (snapshot assembly) — don't band-aid in the agent.

### 5. Knowledge is loaded once per session, not per turn

S56 wires Knowledge into the system prompt at session start. **Don't make the agent call the knowledge endpoint per turn** — it's expensive and the snapshot already carries it. If knowledge needs to update mid-session, that's a future feature (live snapshot polling), not part of S56's contract.

### 6. Display mode awareness

Snapshot includes `display_mode` ∈ `normal | invisible | 3dgs | talking`. The agent's system prompt mentions this so the LLM knows whether the visitor sees a static avatar, a 3D one, or a talking-mouth video (SoulX, S48). The agent does NOT change voice or behavior based on display mode — it just gets context.

### 7. Auto-speak the first script on join

S49 fires the scene's first script line via TTS as soon as the agent connects. **This is in addition to the LLM** — the script is a static line, not LLM-generated, so the visitor gets an immediate greeting that matches the creator's authoring. Subsequent turns are LLM-driven.

### 8. Transport selection by environment

```python
# bot.py (sketch)
if os.getenv("PIPECAT_CLOUD") == "1":
    transport = DailyTransport(...)
else:
    transport = SmallWebRTCTransport(...)
```

Pipecat Cloud sets `PIPECAT_CLOUD=1`. Local dev defaults to SmallWebRTC. Don't hardcode either; respect the env var.

### 9. Multi-language prep (S61, not yet shipped)

After S60–61:
- Read `language` from snapshot.
- Map to Deepgram's language code (`en`, `es`, `fr`, `de`, `pt`, `ja`, `ko`, `vi`, `zh`).
- Pass `language=<code>` to Deepgram service init. Fall back to `language=multi` if a specific model isn't available.
- Cartesia's `sonic-2` is multilingual at the LLM-output level — no per-language model selection. Voice ID matters: voice clones from English source produce English-accented multilingual output (known caveat).
- LLM gets a hard "respond in {Language}" directive at top AND bottom of the system prompt (LLMs weight first/last most heavily — belt-and-suspenders).

### 10. Recipient prompt (S60, not yet shipped)

A creator-authored "audience" description (e.g. "Sales team at Acme Corp evaluating renewal — focus on ROI"). Inserted into the system prompt as a `# AUDIENCE` section between persona and knowledge. Empty / whitespace-only prompts produce no section.

---

## Cross-repo contracts

- **Backend's snapshot endpoint is public.** Don't add auth headers to the agent's snapshot fetch — it's a public contract by design (S43).
- **Snapshot field additions must be Optional.** Older agents shouldn't crash on newer snapshots, and vice versa. The agent must read snapshot fields with `.get()` defaults.
- **Canvas action events flow agent → frontend** through the Pipecat data channel (S47). Frontend listens for the events and renders SVG overlays. Don't try to render canvas actions from the agent side; the agent only emits events.

---

## Testing

```bash
uv run pytest                          # unit tests
uv run pytest tests/test_scene_context.py -v
```

Manual smoke: run `uv run python bot.py` against a local backend with a real Live Room. Speak through your laptop mic (after `pipecat-agent/.env` is configured); listen for response.

For S60+ multi-language verification, the test is end-to-end: change `language` on the live_room row in the DB, restart the agent, speak English, hear the configured language.

---

## Common gotchas

1. **Pipecat versions move fast.** Pin in `pyproject.toml`. When updating, re-test the entire pipeline; minor versions break observer APIs.
2. **Cartesia voice IDs are multilingual-capable but quality varies.** Cloned voices from English source sound English-accented in other languages. Document for creators; don't try to fix in the agent.
3. **Vision injection is expensive.** S46 injects the scene snapshot PNG every N turns (configurable). Don't inject every turn — token cost balloons.
4. **Don't `await` cold paths inside hot ones.** The pipeline is real-time; latency adds. Cache the snapshot at session start; don't re-fetch on every turn.
5. **Pipecat Cloud has cold starts.** If a session takes >5s to connect, that's a cold start, not a bug. Pre-warming is in the S69 polish backlog.
6. **Daily room URLs expire.** Backend creates them on session-start; the agent joins via the room URL passed in the start-session payload. Don't cache room URLs on the agent side.
7. **Deepgram supports 9 languages directly.** Beyond that, use `language=multi` (auto-detect). Multi-detect is slower and lower-accuracy for short utterances; document as a caveat.
8. **`OPENAI_API_KEY` is the LLM key, not necessarily OpenAI.** The agent uses an OpenAI-compatible API; could be Anthropic via a proxy. Treat the env var as opaque "LLM key".

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
| **59** | **Interactive Button Elements** | **✅ Current** | **Comment-only update in scene_context.py** |
| 60 | Live Room language + recipient prompt | ⬜ Next | (no agent change — backend + studio) |
| 61 | Live Room i18n + voice agent multi-language | ⬜ Planned | **Deepgram language, system prompt LANGUAGE + AUDIENCE sections** |

S59 in this repo is documentation-only:
- A multi-line in-line comment in `scene_context.py` clarifying that buttons are visitor-clickable; the agent describes (not clicks) them and may use `highlight_element` to point at one.
- (Optional) A `button` branch in the elements summary formatter for nicer prose.

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

**S59 added no new env vars.** S61 will not add any either — Deepgram and Cartesia are already configured.

---

## When in doubt

- **New snapshot field in the backend?** Read it via `snapshot.get("new_field", default)` in `scene_context.py`. Older snapshots without the field must not crash.
- **Adding to the system prompt?** Put it in `scene_context.py::build_system_prompt`. Document order in this file.
- **New LLM tool?** Add to the canvas-actions observer (S47) only if it's a canvas-side effect (highlight, arrow, etc.). Side effects in the visitor's browser go through the Pipecat data channel. Server-side mutations go via httpx to the backend.
- **Survey-related agent behavior?** **Stop.** Survey UX is 100% client-side. The agent's job is to ignore the `survey` snapshot field. If you're tempted to read it, debug the issue you're trying to fix elsewhere first.
- **Canvas button click handling?** The agent has no click tool. If you find yourself adding one, that's out of scope for S59 and likely all of v2 — buttons are visitor-only.
