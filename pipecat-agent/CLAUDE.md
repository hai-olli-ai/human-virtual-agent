# CLAUDE.md — pipecat-agent

> **For Claude Code.** Ground truth for this repo. Read top-to-bottom before editing. Last synced after **Session 56 (Knowledge RAG — agent half)**. Next agent-touching session is **S58 or later** — S57 may or may not touch this repo depending on whether survey display requires agent coordination.

---

## What this repo is

The Pipecat voice agent service for **Human Virtual** (hv.ai). Runs as a Pipecat pipeline (VAD → STT → LLM → TTS → WebRTC transport) and connects visitors on `/live/[slug]` to an avatar that can hold a real-time voice conversation, see the scene canvas, trigger canvas actions, and — as of S56 — answer grounded in Scene/Flow knowledge.

Deployed to **Pipecat Cloud**. Local dev uses `SmallWebRTCTransport`; production uses `DailyTransport`.

---

## Tech stack

| Component | Stack |
|---|---|
| Language | Python 3.11+ |
| Framework | Pipecat |
| VAD | Silero |
| STT | Deepgram streaming |
| LLM | OpenAI (GPT-4o family — see `OPENAI_LLM_MODEL`) |
| TTS | Cartesia (streaming) |
| Transport (prod) | DailyTransport → Pipecat Cloud → Daily.co WebRTC |
| Transport (local) | SmallWebRTCTransport for loopback testing |
| Package manager | uv |
| Logging | Python `logging` (Pipecat logger hierarchy) |

---

## Directory structure

```
pipecat-agent/
  bot.py                          # pipeline entry point — assembles VAD/STT/LLM/TTS/transport
  scene_context.py                # builds the LLM system prompt from scene snapshot
                                  # (S56: now includes build_knowledge_context + KNOWLEDGE_PREAMBLE)
  canvas_actions.py               # LLM function tools (highlight/arrow/annotation/navigate/clear)
  hv_api.py                       # thin client for human-virtual-backend
  config.py                       # env vars
  Dockerfile
  pcc-deploy.toml                 # Pipecat Cloud deployment config
  pyproject.toml
  tests/
    test_scene_context_knowledge.py    # S56 — pure-function tests for build_knowledge_context
  README.md, DEPLOY.md
```

---

## Pipeline architecture

```
Visitor browser (WebRTC)
        │
        ▼
[Daily.co / Pipecat Cloud]  ← or SmallWebRTCTransport locally
        │
        ▼
┌─────────────────────────────────────────┐
│  Pipecat Pipeline (bot.py)               │
│                                           │
│  Audio IN                                 │
│    → Silero VAD                           │
│    → Deepgram STT (streaming)             │
│    → LLM Service (OpenAI)                 │
│        + tools (canvas actions)           │
│        + system prompt                    │
│           1. Persona                      │
│           2. Knowledge (NEW in S56) ─┐    │
│           3. Scene instruction       │    │
│           4. Display mode context    │    │
│           5. Scene elements summary  │    │
│           6. Canvas action guidance  │    │
│           + base64 scene PNG          │    │
│    → Cartesia TTS (streaming)             │
│  Audio OUT                                │
│                                           │
│  Data channel events:                     │
│    speaking_state / transcription /       │
│    thinking_state / canvas_action         │
└─────────────────────────────────────────┘
        │
        ▼
[Human Virtual Frontend]
  /live/[slug] page
  - renders avatar (display mode)
  - shows transcription
  - listens for canvas_action on data channel
```

---

## Current state (post-S56)

**Wired and working:**
- Real-time voice conversation (STT → LLM → TTS) on Pipecat Cloud
- Persona prompt fetched from backend on session start
- Scene instruction injected into system prompt
- Scene canvas rendered to PNG via Pillow (backend), base64 sent to LLM (vision)
- 5 canvas action tools with deterministic keyword-based element lookup
- Script auto-speak on connect (pre-generated Cartesia audio chunks streamed on start)
- Thinking state indicator via data channel
- Transcription forwarded via Daily data channel
- Connection quality + reconnection handled by transport
- **Knowledge RAG (S56):** `build_knowledge_context` in `scene_context.py` formats scene + flow knowledge (FAQ → docs → URLs) into a markdown block. Injected with `KNOWLEDGE_PREAMBLE` after persona, before scene instruction. Flow scope prepended before Scene scope.

**Not yet:** Survey integration (TBD in S58 — may need agent awareness for "on exit" trigger, or may be handled entirely client-side via beforeunload + session-end events).

---

## Non-negotiable conventions

### Transport differences — CRITICAL
| Transport | Message method | Payload |
|---|---|---|
| `DailyTransport` (prod) | `send_app_message(dict)` | dict, auto-JSONified |
| `SmallWebRTCTransport` (local) | `send_message(str)` | string, JSON-stringify yourself |

Frontend handler is tolerant to both. Adding new data-channel events: test both transports.

### Canvas action tools
- `run_llm=False` for all EXCEPT `navigate_scene` (needs `run_llm=True` so the LLM speaks about the transition)
- Fire-and-forget: tool emits data-channel event, returns quickly, LLM continues talking
- Coordinates from **keyword matching** in `canvas_actions.py` — not AI inference

### Scene context assembly (S56-updated order)
`scene_context.py::build_system_prompt(snapshot)` assembles the full system prompt. Section order (top → bottom):
1. **Persona** (avatar's prompt)
2. **Knowledge** (NEW in S56) — prefixed with `KNOWLEDGE_PREAMBLE`, then FLOW KNOWLEDGE section, then SCENE KNOWLEDGE section. Inside each scope: FAQ → Documents → URLs.
3. **Scene instruction**
4. **Display mode context**
5. **Scene elements summary**
6. **Canvas action tool usage guidance**

The reasoning: persona defines character first, knowledge primes the model with domain context second, instruction steers tone/behavior third, then technical scene details and tool specifics fill in the rest.

### Knowledge conventions (S56)
- `build_knowledge_context(knowledge)` is a pure function — pass the `knowledge` dict from the snapshot or `None`.
- Returns `""` for None / empty / all-items-empty (graceful no-op; no preamble appears in prompt).
- Format:
  ```
  {KNOWLEDGE_PREAMBLE}

  # FLOW KNOWLEDGE

  ## FAQ
  Q: ...
  A: ...

  ---

  ## Document: file_name.pdf
  [extracted_text]

  ---

  ## Web Page: Title
  [markdown_content]

  # SCENE KNOWLEDGE
  [same structure]
  ```
- Defensive `.get(key) or default` everywhere — old snapshots lacking the `knowledge` key must not crash.

### Backend API
- Single unauthenticated endpoint: `GET /api/v1/live-rooms/{id}/scene-snapshot` returns the full agent context in one call.
- **After S56, response includes `knowledge: {scene, flow, budget_exceeded, total_chars}`.**
- Public intentionally — Pipecat agent cannot hold a user token; the slug's existence is the authorization.
- Agent does NOT call knowledge endpoints directly. It reads only the snapshot.

### Error handling
- Snapshot fetch fails → fallback greeting with minimal context (don't crash).
- Canvas action tool fails to find keyword → graceful verbal acknowledgment ("I can't seem to point that out right now").
- Knowledge present but visitor asks off-topic → agent should say it's outside the provided context (per preamble guidance).

### Data channel event format
```python
{
  "type": "speaking_state" | "transcription" | "thinking_state" | "canvas_action",
  "payload": { ... }
}
```

---

## Environment variables

```bash
HV_API_URL=http://localhost:3001           # or https://api.hv.ai in prod
DEEPGRAM_API_KEY=...
OPENAI_API_KEY=...
OPENAI_LLM_MODEL=gpt-4o                    # or another GPT-4 class
CARTESIA_API_KEY=...                        # shared with backend
PIPECAT_CLOUD_API_KEY=...                   # only when deploying
DAILY_API_KEY=...                           # only in prod transport path
```

**No new env vars in S56.** Knowledge comes over HTTP from the snapshot; no new service credentials.

---

## Dev commands

```bash
uv sync

# Local (SmallWebRTC loopback)
uv run python bot.py

# Tests
uv run pytest tests/ -v
uv run pytest tests/test_scene_context_knowledge.py -v  # S56

# Deploy to Pipecat Cloud
pcc deploy       # uses pcc-deploy.toml
pcc logs -f      # tail
```

---

## Session history (agent-touching sessions)

- S37–38: Pipecat scaffold + Cloud deployment
- S45: Wired to backend for persona prompts; real-time transcription + speaking state via Daily data channel
- S45b: End-to-end WebRTC hardening, agent participant tracking, cold-start UX
- S46: Multimodal vision — consumes scene canvas PNG, injects base64 into LLM context
- S47: LLM function calling for 5 canvas action tools with keyword-based element lookup
- S49: Script auto-speak on session start
- S50: Polish — thinking state indicator, connection quality reporting, graceful reconnect
- (S51a–S55 were all backend or frontend; agent untouched)
- **S56: Knowledge RAG — `build_knowledge_context` in `scene_context.py`, `KNOWLEDGE_PREAMBLE` constant, injected into `build_system_prompt` between persona and scene instruction. FLOW scope prepended before SCENE scope. FAQ → Docs → URLs within each scope.**

---

## Gotchas

1. **Pipecat Cloud cold start.** First session after idle takes 10–20s. Don't try to fix in the agent — it's a Pipecat Cloud property.

2. **Transport message shape.** Local works but Cloud breaks (or vice-versa) → almost always this.

3. **Tool call bloat.** Every tool adds prompt tokens. Canvas actions are deliberately 5.

4. **Don't call the DB from the agent.** All DB access via the backend's snapshot endpoint. Keep the agent stateless.

5. **Don't reintroduce BuddyOS.** Voice-agent pivot to Pipecat replaced it. Enum value `buddyos` exists for legacy records, but agent uses Cartesia.

6. **Image size matters.** Canvas PNG is base64'd into LLM context. If prompt bloat appears, talk to the backend — don't resize in the agent.

7. **System prompt order matters.** LLM weights earlier sections more. Don't rearrange without a good reason. Current order (post-S56): Persona → Knowledge → Scene instruction → Display mode → Scene elements → Tools guidance.

8. **Knowledge is snapshot at connect, not live (S56).** Visitor sees knowledge as it was when their session started. Creator edits mid-session require visitor reconnect. Intentional — alternative would invalidate the LLM's accumulated conversation state.

9. **Preamble wording is tuned.** The `KNOWLEDGE_PREAMBLE` string wording matters — it steers the LLM to prefer grounded answers AND to flag when answering off-knowledge. Don't tweak casually.

10. **Defensive key access.** `snapshot.get("knowledge")` may be missing on older backends or in tests; `build_knowledge_context` handles this. Don't assume keys exist.

11. **No RAG/embeddings/vector search.** S56 is pass-all-text-as-context. If you're reaching for a vector store, stop — that's deferred until someone proves the current approach has real user-visible problems.

---

## When starting a new session in this repo

1. Confirm the session actually touches the agent (check the V2.9 guide).
2. Sanity-check locally with SmallWebRTC before deploying.
3. Verify both transport paths work — add print/log lines if needed.
4. Update this file afterward — bump the "last synced after" line and add to session history.

---

## For the next agent session (S58 or later)

Session 57 adds the Survey system (backend + frontend editor). It probably does NOT touch this repo — survey display logic lives in the live-room frontend (on-exit beforeunload handler + CTA button + modal).

Session 58 may touch this repo IF survey collection needs to coordinate with the agent lifecycle (e.g. "don't fire on-exit survey while the agent is still speaking" or "the agent should acknowledge when the user opens the survey"). Reading S58 ahead of time: the design is primarily client-side (beforeunload + Pipecat session-end events). If this repo stays untouched, great — more time for S59 scene generation.

Session 59 (knowledge-aware scene generation) is backend-only — does not touch this repo.

**Most likely next agent work: S62 or later.** The current LLM prompt + RAG setup should hold for a while; major agent changes are not planned until post-launch tuning.
