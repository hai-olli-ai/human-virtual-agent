# CLAUDE.md — pipecat-agent

> Last updated: Session 47 (canvas action tools — highlight, arrow, annotation, navigate, clear)

## Overview

Voice agent for Human Virtual's Avatar Live URL. Deployed to **Pipecat Cloud**. Pipeline: VAD → STT → LLM → TTS → Daily WebRTC. Multimodal vision (sees canvas). **LLM function calling** for canvas actions.

## Commands

```bash
uv sync && uv run bot.py    # Install + run locally → http://localhost:7860/client
```

## Deploy

```bash
docker build --platform=linux/arm64 -t human-virtual-agent:0.4 .
docker push YOUR_USER/human-virtual-agent:0.4
pcc deploy
```

## Structure

```
pipecat-agent/
├── bot.py              # Pipeline + vision + tools + transcript forwarding
├── canvas_actions.py   # 5 LLM tools + handlers + element resolution — Session 47
├── config.py           # Env vars
├── persona.py          # Prompt builder (persona-prompt endpoint)
├── scene_context.py    # Scene snapshot → text + vision message + tools section
├── api_client.py       # HTTP client (public + auth + scene image)
├── pcc-deploy.toml
└── Dockerfile
```

## Key Decisions

### Canvas Action Tools (Session 47)
- **5 tools** registered via `FunctionSchema` + `ToolsSchema` + `llm.register_function()`:
  - `highlight_element` — pulsing highlight box on an element (`run_llm=False`)
  - `draw_arrow` — animated arrow between two elements (`run_llm=False`)
  - `place_annotation` — pill text label near an element (`run_llm=False`)
  - `navigate_scene` — go to next/previous scene, re-fetches vision (`run_llm=True`)
  - `clear_annotations` — remove all overlays (`run_llm=False`)
- **Element resolution:** `resolve_element_region()` maps LLM descriptions to canvas coordinates via keyword matching against element type/text/label/title. Falls back to canvas center.
- **Dispatch:** `transport.send_app_message({ type: "canvas_action", action: { name, params } })`
- **Colors:** orange=#C15F3C, green=#4A7C59, blue=#4A6FA5, red=#C1443C
- **Duration:** Tool args in seconds, converted to milliseconds for frontend

### Vision (Session 46)
- Canvas image fetched as base64 via `GET /scene-snapshot/image?format=base64`
- Injected as first `LLMContext` message with `detail: "high"`
- Model: `gpt-4.1` (supports vision natively)

### Pipeline
- input → STT → TranscriptForwarder → UserAggregator → LLM → SpeakingNotifier → TTS → output → AssistantAggregator
- `LLMContext(messages=[vision_message], tools=canvas_tools)`
- `llm.register_function()` for each of 5 canvas action handlers

### Rules
- NEVER imports from backend `app/` — separate service
- All API calls try/except — never crashes
- `run_llm=False` on fire-and-forget tools (highlight, arrow, annotation, clear)
- `run_llm=True` on `navigate_scene` so LLM describes the new scene

## Environment Variables (Pipecat Cloud secret set)

```bash
DEEPGRAM_API_KEY, OPENAI_API_KEY, CARTESIA_API_KEY,
HV_API_URL=https://api.hv.ai/api/v1,
CARTESIA_VOICE_ID, LLM_MODEL=gpt-4.1
```
