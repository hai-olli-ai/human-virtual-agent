# pipecat-agent — CLAUDE.md

> Voice agent for **Human Virtual** (hv.ai) — drives the avatar's voice + canvas behavior in Live Rooms.
> **Status:** V2 in progress · Sessions through 63b complete · **S64a (Canvas Protocol Foundation) is next; S64c rebuilds this repo's tool surface**.
> **Stack:** Pipecat Framework · Pipecat Cloud · Daily.co (prod) · Cartesia + Deepgram + OpenAI/Anthropic/Gemini.

---

## Stack

| Layer | Choice |
|---|---|
| Framework | Pipecat (Daily.co's voice-agent framework) |
| Transport (local dev) | `SmallWebRTCTransport` |
| Transport (production) | `DailyTransport` via Daily.co |
| Hosting (production) | Pipecat Cloud |
| STT | Deepgram (multilingual, 9 languages) |
| TTS | Cartesia (with voice clones from S51b) |
| LLM (canvas tool routing) | OpenAI / Anthropic / Gemini — selectable via `LLM_CANVAS_PROVIDER` (introduced S64c) |
| LLM (vision) | OpenAI GPT-4.1 (S46 scene understanding) |
| Avatar display | SoulX-Flashtalk (S48 talking mode) |

---

## What this agent does

For each Live Room session, the agent:

1. Fetches the per-session context from the backend's snapshot endpoint.
2. Constructs a system prompt using the **sandwich pattern** (see below).
3. Runs the conversation loop: user speaks → STT → LLM → tool call (canvas action) + speech response → TTS → speaks back.
4. Optionally uses GPT-4.1 vision (S46) to understand the rendered canvas when a question requires visual grounding.
5. Retrieves Knowledge via RAG (S56) when the question relates to creator-provided documents/URLs/FAQs.

**The agent is stateless across sessions.** Each Live Room session boots a fresh agent with a fresh prompt.

---

## Project structure

```
pipecat-agent/
  agent.py                   # Entry point — bootstraps the pipeline
  config.py                  # Settings (env vars, transport, model IDs)
  context/
    snapshot_client.py       # Calls backend HV_API_URL endpoints
    prompt_builder.py        # Sandwich-pattern system prompt assembly
    knowledge_rag.py         # S56 RAG retrieval
  tools/                     # Function-calling tools
    canvas_actions.py        # Currently S47's 5 hardcoded tools — REPLACED in S64c
    vision.py                # S46 GPT-4.1 vision for canvas understanding
    knowledge.py             # RAG-backed knowledge lookup
  transports/
    local.py                 # SmallWebRTCTransport setup
    daily.py                 # DailyTransport setup
  services/
    cartesia_tts.py          # Cartesia TTS wrapper with voice-clone support
    deepgram_stt.py          # Multilingual STT
  utils/
    language.py              # Locale → STT/TTS config mapping (S60–61)
  Dockerfile                 # For Pipecat Cloud deployment
  requirements.txt
```

---

## Local dev vs production

| Aspect | Local dev | Production |
|---|---|---|
| Transport | `SmallWebRTCTransport` | `DailyTransport` |
| Run command | `python agent.py --transport=local` | Pipecat Cloud manages |
| WebRTC signaling | Built-in to Pipecat dev server | Daily.co room API |
| Backend URL | `http://localhost:8000` (`HV_API_URL`) | `https://api.hv.ai` |
| Voice room | Direct browser ↔ agent | Browser ↔ Daily room ↔ agent |

The transport is selected at startup based on the `--transport` CLI flag or `TRANSPORT` env var. **Don't unify** the two paths — Daily-specific session metadata (room URLs, app messages) doesn't have a clean local equivalent.

---

## Backend integration — snapshot endpoint

The agent's source of truth is the backend's per-session snapshot:

```
GET /live-rooms/by-slug/{slug}/scene-snapshot
```

This endpoint is **public** (no auth) — it's hit by both the Pipecat agent and the live-room frontend visitor view. The response shape:

```python
{
  "scenes": [
    {
      "id": "...",
      "name": "...",
      "display_mode": "normal" | "invisible" | "3dgs" | "talking",
      "instruction": "...",
      "elements": [...],            # JSONB array of canvas elements
      "scripts": [...],
      "knowledge_text": "...",      # assembled from Scene Knowledge + extracted link content
      "faqs": [...],
      "link_url": "..." | null,
      "link_source": "..." | null,
      "canvas_page_type": "composition"  # NEW in S64a — Pipecat ignores until S64c
    },
    ...
  ],
  "live_room": {
    "language": "en",               # S60–61 multi-language
    "recipient_prompt": "...",      # S60–61
    "persona": {
      "avatar_name": "...",
      "voice_model_provider": "cartesia",
      "voice_id": "...",
      ...
    }
  }
}
```

Snapshot is fetched **once per session** at session start. The agent caches it in memory for the session's duration. If the user navigates between scenes mid-session, the agent reads the new scene from the cached snapshot — no refetch.

**Adding a new field to the snapshot is non-breaking** — Pipecat parses what it needs and ignores the rest. There is no strict Pydantic model with `extra="forbid"`. Don't add one.

---

## System prompt — sandwich pattern

The system prompt is assembled in `context/prompt_builder.py` in this order:

```
1. LANGUAGE                     # "Respond in {language}. ..."
2. PERSONA                      # avatar identity, tone, role
3. AUDIENCE                     # recipient_prompt from S60–61
4. KNOWLEDGE                    # RAG-retrieved + scene knowledge text
5. SCENE INSTRUCTION            # creator-authored instruction
6. DISPLAY MODE                 # what the avatar looks like (talking, invisible, etc.)
7. CANVAS ELEMENTS              # serialized JSON of current scene's elements
8. CANVAS ACTIONS               # tool descriptions — REPLACED in S64c
9. LANGUAGE                     # repeat the language instruction (sandwich close)
```

The double-LANGUAGE wrap (1 + 9) is intentional — multilingual reliability requires the language directive to be both at the start (for instruction-following) and at the end (closest to the next-token prediction). Don't remove either anchor.

**The sandwich must remain stable across sessions.** Reordering the sections invalidates LLM prompt caches across all three providers (Anthropic explicit `cache_control`, OpenAI/Gemini structural prefix). When extending the prompt, **append within an existing section** rather than inserting a new section between existing ones.

### Coming in S64c

Section 8 (CANVAS ACTIONS) is being restructured:

- The 5 hardcoded tool descriptions (`highlight_element`, `arrow_between`, `add_annotation`, `navigate_scene`, `clear_overlays`) are removed.
- A new compact section called `CANVAS PAGE` describes the active Page type and supported verbs:
  ```
  Active page: composition.
  control: next_scene | previous_scene | clear.
  highlight: element_id targets.
  analyze: supported (semanticState provider).
  ```
- 5 generic tools (`canvas.analyze`, `canvas.highlight`, `canvas.control`, `canvas.action`, `canvas.set_page`) replace the hardcoded surface.

This is the most delicate change in Phase 9g — the agent's tool surface fundamentally shifts. Before/after latency benchmarking is mandatory in S64c.

---

## Voice tech

### STT — Deepgram (multilingual)

`services/deepgram_stt.py` configures Deepgram per the live-room language. Locale mapping in `utils/language.py`:

```python
LOCALE_TO_DEEPGRAM_MODEL = {
    "en": "nova-3",
    "es": "nova-2",
    "fr": "nova-2",
    # ... 9 total
}
```

Deepgram returns interim transcripts; the agent waits for `is_final=True` before sending to LLM. The pipeline uses Pipecat's `STTService` interface — replacing Deepgram is a single-class swap.

### TTS — Cartesia (with voice clones)

`services/cartesia_tts.py` uses Cartesia's WebSocket TTS API. Voice selection logic:

1. If `voice_model_provider == "cartesia"` and a `voice_id` is set on the avatar, use that voice (cloned from S51b).
2. Else use the default Cartesia voice for the session's language (mapped in `utils/language.py`).

Cartesia's auto-clone (S51b) creates a voice from the 6–10s recording the user makes during avatar creation. The backend manages clone creation; this repo only consumes the resulting `voice_id`.

### Display modes (S39, S48)

- `normal` — static avatar image
- `invisible` — no avatar shown (audio-only)
- `3dgs` — 3D Gaussian Splatting (rendered frontend-side)
- `talking` — SoulX-Flashtalk lip-sync (rendered frontend-side, this repo only relays the TTS audio)

The agent doesn't render the avatar. It emits TTS audio frames; the frontend renders the visual based on `display_mode` from the snapshot.

---

## Vision — S46 GPT-4.1

When a user question requires visual grounding ("what color is the box on the left?"), the agent calls `tools/vision.py`:

1. Fetches the canvas snapshot image from `GET /live-rooms/by-slug/{slug}/scene-snapshot/image` (Pillow-rendered PNG of the current scene).
2. Sends to GPT-4.1 with the user's question.
3. Returns the answer, which the LLM incorporates into its response.

Vision is **opt-in per turn** — the agent's main LLM decides when to call the vision tool. Don't always-call vision; it adds 1–3s latency.

**Coming in S64c+:** with the Canvas Protocol, vision becomes one of multiple `analyze()` providers. v0.1 ships only the semanticState provider (microsecond-fast, answers most questions). Vision is queued for v0.2.

---

## Knowledge RAG — S56

`context/knowledge_rag.py` retrieves relevant Knowledge chunks for a user question:

1. Knowledge text was assembled into `scene.knowledge_text` at snapshot time (backend-side).
2. The agent's LLM has the full knowledge text in the prompt's KNOWLEDGE section.
3. For very large knowledge bodies, the agent calls a RAG tool (`tools/knowledge.py`) to retrieve specific chunks via the backend's `POST /knowledge/search` endpoint.

The simple path (knowledge fits in prompt) handles most cases. RAG retrieval is reserved for >~10k token knowledge bodies. Backend handles vector storage + retrieval; this repo just calls the search endpoint.

---

## Canvas tools — current state (S47), changing in S64c

### Current (post-S63b)

5 hardcoded tools in `tools/canvas_actions.py`:

- `highlight_element(element_id: str)` — render a highlight box on a canvas element
- `arrow_between(from_element_id: str, to_element_id: str)` — render an arrow
- `add_annotation(text: str, x: int, y: int)` — render a text bubble at a coordinate
- `navigate_scene(direction: "next" | "previous")` — flow navigation
- `clear_overlays()` — remove all highlights/arrows/annotations

Tools are emitted as Daily app-messages (in production) or local WebRTC data-channel messages (in dev). The frontend listens and renders.

### Coming in S64c (Canvas Protocol generic surface)

The 5 hardcoded tools are **removed**. Replaced by 5 generic verbs:

- `canvas.analyze(question, options)` — Q&A about the current Page state
- `canvas.highlight(target, options)` — target = `{element_id}` or `{box: [x,y,w,h]}`
- `canvas.control(verb, args)` — Page-specific verbs (composition: `next_scene`, `previous_scene`, `clear`)
- `canvas.action(verb, args)` — Page-specific action verbs (quiz: `submit_answer`, `request_hint`)
- `canvas.set_page(pageType, pageInit)` — switch the active Canvas Page (composition / youtube / quiz)

The 5 generic verbs work uniformly across all Page types. The active Page's manifest (received via Daily app-message from the frontend's Canvas Service) tells the agent which `control` and `action` verbs are valid in the current context.

**Multi-provider eager streaming dispatch** also lands in S64c — three per-provider Pipecat adapters (Anthropic, OpenAI, Gemini) implement eager dispatch on tool-call streaming so arg-less verbs (`pause`, `play`, `clear`, `restart`, `next_question`) fire 100–250ms earlier than a wait-for-stop_reason path.

---

## Multi-language (S60–61)

Live Rooms support 9 languages: `en`, `es`, `fr`, `de`, `pt`, `it`, `ja`, `ko`, `zh-CN`.

The `language` field on the snapshot drives:

1. **STT model selection** — `LOCALE_TO_DEEPGRAM_MODEL[language]`
2. **TTS voice selection** — language-specific Cartesia voice (or the cloned voice if it supports the language)
3. **Prompt LANGUAGE sections (1 + 9)** — instruct the LLM to respond in the target language
4. **Recipient prompt** — creator-authored audience description, optionally translated

The voice agent does not auto-detect the language. The creator picks the language at Live Room creation time (frontend studio); the agent operates in that language exclusively.

When adding a new language: extend `LOCALE_TO_DEEPGRAM_MODEL`, extend the Cartesia voice mapping, extend `utils/language.py`. The frontend's i18n locale list is independent — UI language ≠ Live Room language.

---

## Pipecat Cloud deployment

Production runs on Pipecat Cloud. Deployment via the Pipecat CLI:

```bash
pcc deploy
pcc logs --tail
```

The `Dockerfile` is the deployment artifact. `pipecat-cloud.yaml` (or equivalent) configures the agent name, transport, scaling.

`PIPECAT_AGENT_NAME` is set in the backend so it can dispatch agent boot requests to the right deployment.

---

## Environment variables

```
TRANSPORT=local | daily             # local for dev, daily for prod
HV_API_URL=http://localhost:8000    # backend snapshot fetch
DAILY_API_KEY=...                    # production only
DEEPGRAM_API_KEY=...
CARTESIA_API_KEY=...
OPENAI_API_KEY=...                   # main LLM + vision
ANTHROPIC_API_KEY=...                # alt LLM (S64c canvas tool routing — optional)
GOOGLE_AI_API_KEY=...                # alt LLM
LLM_CANVAS_PROVIDER=anthropic | openai | gemini   # NEW in S64c — picks Pipecat LLM service for canvas tools
```

---

## Common commands

```bash
# Local dev (SmallWebRTC transport)
python agent.py --transport=local

# Production deploy
pcc deploy

# Tail production logs
pcc logs --tail

# Test snapshot fetch locally
curl http://localhost:8000/live-rooms/by-slug/{slug}/scene-snapshot | jq .
```

---

## Conventions Claude Code should follow

- **Always** treat the snapshot as immutable for the session. Don't mutate the cached snapshot; if state needs to evolve, store it separately.
- **Always** keep the system prompt's section order stable. Append within existing sections; don't reorder, don't insert new sections between existing ones.
- **Always** test prompt changes against all 9 supported languages before merging — localized failure modes are easy to miss.
- **Always** route external API calls (STT/TTS/LLM) through Pipecat's service interfaces. Don't hardcode SDK calls outside `services/`.
- **Never** add auth requirements to backend snapshot calls. The snapshot endpoint is public on purpose.
- **Never** introduce a strict Pydantic model with `extra="forbid"` for the snapshot. New fields must be non-breaking.
- **Never** unify local + Daily transports. They have genuinely different session lifecycles.
- **Never** add fallback logic that auto-detects language. The creator-set language is authoritative.
- When adding a new tool: register it in the LLM service's tool list, write the handler in `tools/`, document it in the system prompt's CANVAS ACTIONS section. **(After S64c, this changes — new tools are added by extending the active Page's manifest, not by adding to Pipecat's tool surface.)**

---

## Coming in S64a (next session — minimal direct change here)

S64a touches **zero** code in this repo. The new `canvas_page_type` field in the snapshot is forward-compat ignored. Verify after S64a lands that:

- The agent boots normally against a snapshot that includes `canvas_page_type: "composition"` per scene.
- No errors logged about unexpected snapshot fields.
- All existing tools (`highlight_element`, etc.) continue to work end-to-end.

---

## Coming in S64c (the big change for this repo)

S64c is the heaviest session in Phase 9g for this repo:

1. **Remove S47 tools.** Delete `highlight_element`, `arrow_between`, `add_annotation`, `navigate_scene`, `clear_overlays` from `tools/canvas_actions.py`.
2. **Add 5 generic tools.** `canvas.analyze`, `canvas.highlight`, `canvas.control`, `canvas.action`, `canvas.set_page`.
3. **Restructure the system prompt's CANVAS ACTIONS section.** Add the new `CANVAS PAGE` section that describes the active Page's manifest in compact form.
4. **Implement multi-provider eager streaming dispatch.** Three Pipecat adapters (~100 LOC each) for Anthropic, OpenAI, Gemini.
5. **Add Daily app-message relay.** Pipecat ↔ frontend's Canvas Service. Agent emits `canvas.command` payloads as Daily app-messages; receives `canvas.commandResult` / `canvas.stateChange` back.
6. **Add `LLM_CANVAS_PROVIDER` env var** and provider selection in agent boot.

After S64c: this file should be updated to remove the "currently using S47 tools" framing and document the production state of the 5 generic tools + multi-provider streaming.
