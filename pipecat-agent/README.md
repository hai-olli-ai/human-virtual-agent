# Human Virtual — Pipecat Voice Agent

Voice agent for Human Virtual's Avatar Live URL feature.
Powers real-time conversations between visitors and AI avatars.

## Local Development

```bash
# Install dependencies
uv sync

# Configure API keys
cp env.example .env
# Edit .env with your keys

# Run locally (opens http://localhost:7860/client)
uv run bot.py
```

## Architecture

```
Visitor → Mic → WebRTC → Pipecat Pipeline:
  VAD (Silero) → STT (Deepgram) → LLM (OpenAI) → TTS (Cartesia)
  → WebRTC → Speaker → Visitor
```

## Production Deployment

See Session 38 for Pipecat Cloud deployment instructions.

## Canvas Protocol (S64c+)

Canvas tool routing uses 5 generic tools: `canvas_{analyze,highlight,control,action,set_page}`
(underscores, not dots — OpenAI/Anthropic tool-name validation rejects `.`. Daily wire-format
message types like `canvas.command` keep their dots; only the LLM-facing function names changed.)
Provider selection via `LLM_CANVAS_PROVIDER` env var (`anthropic` | `openai` | `gemini`,
default `openai` — only the openai extra is currently in `pyproject.toml`; install the
matching pipecat extra (`pipecat-ai[anthropic]` or `pipecat-ai[google]`) before switching).

Eager streaming dispatch fires arg-less verbs early. See
`services/eager_dispatch/__init__.py` for the verb registry and the per-provider hook
adapters under the same package.

Daily app-message types received from the frontend Canvas Service:

- `canvas.register` → sets active Page manifest
- `canvas.stateChange` → updates semantic state
- `canvas.commandResult` / `canvas.commandError` → resolves pending commands

Daily app-message types sent to the frontend:

- `canvas.command` → outbound canvas tool call
