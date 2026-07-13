"""Configuration for the Human Virtual Pipecat agent."""
import os
from dotenv import load_dotenv

load_dotenv(override=True)

# AI Service keys
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
GOOGLE_AI_API_KEY = os.getenv("GOOGLE_AI_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
CARTESIA_API_KEY = os.getenv("CARTESIA_API_KEY", "")

# Human Virtual API
HV_API_URL = os.getenv("HV_API_URL", "http://localhost:3001/api/v1")

# Daily (managed by Pipecat Cloud, but useful for local Daily testing)
DAILY_API_KEY = os.getenv("DAILY_API_KEY", "")

# Default IDs for testing
DEFAULT_AVATAR_ID = os.getenv("DEFAULT_AVATAR_ID", "")
DEFAULT_SCENE_ID = os.getenv("DEFAULT_SCENE_ID", "")
DEFAULT_ROOM_ID = os.getenv("DEFAULT_ROOM_ID", "")

# TTS voice configuration
CARTESIA_VOICE_ID = os.getenv("CARTESIA_VOICE_ID", "71a7ad14-091c-4e8e-a314-022ece01c121")
# Default: "British Reading Lady" — will be customizable per avatar later

# ──────────────────────────────────────────────────────────────────────
# Narration audio cache (Block 1 — must equal the backend's renderer)
# ──────────────────────────────────────────────────────────────────────
# Cartesia returns 16-bit signed little-endian PCM regardless. These
# three values gate cache compatibility: a pre-rendered segment is only
# safe to replay if it was synthesized with the same model at the same
# sample_rate and channel count the live service is configured for.
# A mismatch would either (a) miss cleanly (safe — falls back to live)
# or (b) play mangled audio if the cache claims a match but the PCM was
# rendered against different settings. Keep these in lockstep with the
# backend's narration renderer.
NARRATION_TTS_MODEL_ID = os.getenv("NARRATION_TTS_MODEL_ID", "sonic-3")
NARRATION_AUDIO_SAMPLE_RATE = int(os.getenv("NARRATION_AUDIO_SAMPLE_RATE", "24000"))
NARRATION_AUDIO_NUM_CHANNELS = int(os.getenv("NARRATION_AUDIO_NUM_CHANNELS", "1"))


# ──────────────────────────────────────────────────────────────────────
# Vision-frame refresh policy (S66 Block 5a)
# ──────────────────────────────────────────────────────────────────────
# Pre-5a, every scene change re-fetched the Pillow-rendered scene image
# (a backend round-trip in the hot path of every navigation). Lazy mode
# (default) skips the per-scene fetch and defers it to the first
# canvas_analyze of the scene — the session-start bootstrap fetch is
# unchanged. Set to "eager" to restore the pre-5a behavior if a quality
# regression surfaces.
VISION_REFRESH_MODE = os.getenv("VISION_REFRESH_MODE", "lazy").strip().lower()

_VALID_VISION_REFRESH_MODES = {"lazy", "eager"}
if VISION_REFRESH_MODE not in _VALID_VISION_REFRESH_MODES:
    raise ValueError(
        f"VISION_REFRESH_MODE must be one of {sorted(_VALID_VISION_REFRESH_MODES)}, "
        f"got '{VISION_REFRESH_MODE}'"
    )


# ──────────────────────────────────────────────────────────────────────
# Vision capture round-trip (S67b)
# ──────────────────────────────────────────────────────────────────────
# Decoupled from LLM_CANVAS_PROVIDER: vision always runs on a fast Gemini
# model regardless of the conversational LLM. VISION_MODEL is read by
# services/vision_client.py; the timeout + max-dim drive the agent→shell
# capture round-trip in bot.py (request_canvas_capture). VISION_MAX_DIM is
# advisory — the shell owns the actual screenshot encode.
VISION_MODEL = os.getenv("VISION_MODEL", "gemini-3.5-flash")
VISION_CAPTURE_TIMEOUT_MS = int(os.getenv("VISION_CAPTURE_TIMEOUT_MS", "4000"))
VISION_MAX_DIM = int(os.getenv("VISION_MAX_DIM", "1280"))  # advisory; shell owns encode

# Block 8 — agent_annotate ack round-trip. Shorter than the capture timeout: the
# overlay draw is best-effort (a timeout is treated as rendered), so the LLM turn
# isn't held long waiting for a cosmetic ack.
AGENT_ANNOTATE_TIMEOUT_MS = int(os.getenv("AGENT_ANNOTATE_TIMEOUT_MS", "2000"))


# LLM model — must support vision for scene understanding (Session 46)
# gpt-4.1 and gpt-4o both support vision
#LLM_MODEL = os.getenv("LLM_MODEL", "gpt-5.4")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4.1")


# ──────────────────────────────────────────────────────────────────────
# Canvas-protocol LLM provider selection (S64c)
# ──────────────────────────────────────────────────────────────────────
#
# Selects which LLM service handles the Canvas Protocol tool surface.
# Vision (S46) stays hardcoded on OpenAI GPT-4.1 regardless — they're
# separate services. See CLAUDE.md "Vision (S46) — separate, hardcoded".
#
# Default is "groq" — the agent runs on Groq's OpenAI-compatible
# GroqLLMService (LPU inference; model = GROQ_MODEL). The groq extra ships
# in pyproject.toml (pipecat-ai[groq]); the openai extra stays installed
# too, so swapping back to "openai" needs no new install. "anthropic" /
# "gemini" still require their pipecat extras (pipecat-ai[anthropic] /
# pipecat-ai[google]) before selecting them. In production this value is
# set explicitly as a Pipecat Cloud env var.
LLM_CANVAS_PROVIDER = os.getenv("LLM_CANVAS_PROVIDER", "groq").strip().lower()

_VALID_CANVAS_PROVIDERS = {"anthropic", "openai", "gemini", "groq"}
if LLM_CANVAS_PROVIDER not in _VALID_CANVAS_PROVIDERS:
    raise ValueError(
        f"LLM_CANVAS_PROVIDER must be one of {sorted(_VALID_CANVAS_PROVIDERS)}, "
        f"got '{LLM_CANVAS_PROVIDER}'"
    )

# Per-provider model overrides (defaults match CLAUDE.md's documented choices).
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
# Groq serves OpenAI's open-weight GPT-OSS models. NOTE: Groq's catalog
# lists the 120B model as "openai/gpt-oss-120b"; the bare id below matches
# the request as given — set GROQ_MODEL=openai/gpt-oss-120b if the API
# returns model_not_found.
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")


# ──────────────────────────────────────────────────────────────────────
# Main-LLM in-context vision capability
# ──────────────────────────────────────────────────────────────────────
# The S46 vision path injects the scene image straight into the MAIN LLM's
# context (build_vision_message → an OpenAI `image_url` content block). That
# only works if the selected conversational model accepts image input. Groq's
# default gpt-oss-120b is TEXT-ONLY and returns 400 ("messages[].content must
# be a string") on image content, which would break the very first turn
# whenever a scene image is present. So that injection is gated on this flag.
#
# Default: False for the groq provider (gpt-oss text-only), True for
# openai/anthropic/gemini. Set MAIN_LLM_SUPPORTS_VISION=true to opt back in if
# you point GROQ_MODEL at a multimodal Groq model (or =false to force it off).
# The decoupled S67b Gemini vision path (run_vision_query) is UNAFFECTED — it
# injects text reasoning, never a raw image, so visual Q&A works under Groq.
def _resolve_main_llm_vision(provider: str, override: str | None) -> bool:
    """Whether the main LLM accepts images in its context.

    ``override`` is the raw ``MAIN_LLM_SUPPORTS_VISION`` env value (or None
    when unset). When set it wins; otherwise we derive from the provider —
    only ``groq`` (text-only gpt-oss default) is False.
    """
    if override is not None:
        return override.strip().lower() in {"1", "true", "yes", "on"}
    return provider != "groq"


MAIN_LLM_SUPPORTS_VISION = _resolve_main_llm_vision(
    LLM_CANVAS_PROVIDER, os.getenv("MAIN_LLM_SUPPORTS_VISION")
)


# ──────────────────────────────────────────────────────────────────────
# Deepgram language mapping (Session 61 — voice agent multi-language)
# ──────────────────────────────────────────────────────────────────────

# Maps live-room language codes (the backend's LiveRoomLanguage Literal)
# to the Deepgram `language` parameter. The 9 codes here mirror the
# nine-language enum the backend enforces via CHECK constraint, and
# Deepgram's nova-2 / nova-3 models support all of them directly.
DEEPGRAM_LANGUAGE_MAP: dict[str, str] = {
    "en": "en",
    "es": "es",
    "fr": "fr",
    "de": "de",
    "pt": "pt",
    "ja": "ja",
    "ko": "ko",
    "vi": "vi",
    "zh": "zh",
}

# Forward-compat: if the backend ever ships a code we haven't mapped
# (e.g. a 10th language added before this file is updated), fall back
# to Deepgram's multilingual auto-detect rather than crashing.
DEEPGRAM_FALLBACK_LANGUAGE: str = "multi"


def resolve_deepgram_language(snapshot_language: str | None) -> str:
    """Map a scene-snapshot language code to a Deepgram language parameter.

    - None / empty → "en" (matches the backend's default).
    - Mapped code  → its Deepgram value.
    - Unknown code → "multi" (auto-detect, slower but always works).
    """
    if not snapshot_language:
        return "en"
    return DEEPGRAM_LANGUAGE_MAP.get(snapshot_language, DEEPGRAM_FALLBACK_LANGUAGE)
