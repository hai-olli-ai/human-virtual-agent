"""Dedicated multimodal vision client for the voice agent (S67b).

The agent has no DOM, so it cannot screenshot the visitor's canvas itself —
the shell captures the annotated scene in the visitor's browser and uploads
the JPEG to a short-TTL backend ingest; the agent fetches those bytes and
hands them to *this* client for reasoning (see the S67b capture round-trip).

Design decisions (from CLAUDE.md "Coming next — S67b"):

* **Dedicated fast model.** Vision runs on ``gemini-3.5-flash`` (low-latency,
  multimodal image input), **decoupled from ``LLM_CANVAS_PROVIDER``** — the
  conversational LLM can be OpenAI/Anthropic while vision is always Gemini.
* **Lazy SDK import.** ``from google import genai`` happens only when a real
  call is made (mirrors ``bot.py:_build_llm_and_eager_hook``'s per-provider
  lazy imports). ``google-genai`` is a declared base dependency
  (``pyproject.toml``), but deferring the import is still useful: it keeps the
  module — and the stub path — loadable even if the SDK is missing, so when the
  key is unset the stub returns before the import fires.
* **Graceful degrade.** When ``GOOGLE_AI_API_KEY`` is unset (or any error
  occurs), ``analyze_image`` returns the ``VISION_UNAVAILABLE`` sentinel and
  the caller falls back to the existing Pillow-PNG path with a blind-spot
  flag ("I can describe the scene but can't see your drawings…"). Every
  external service in this project has a mock/stub fallback.

SDK call shape confirmed in A-AG-4 against the pinned ``google-genai``
(``pipecat-ai 0.0.108``'s ``google`` extra: ``google-genai<2,>=1.68.0``):

    client = genai.Client(api_key=...)
    resp = await client.aio.models.generate_content(            # async, not client.models.*
        model="gemini-3.5-flash",
        contents=[types.Part.from_bytes(data=jpeg, mime_type="image/jpeg"), prompt],
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_level="low"),  # 3.x — not thinking_budget
        ),
    )

``thinking_level`` ("minimal" | "low" | "medium" | "high") replaces the 2.5-era
numeric ``thinking_budget``; "low" is a tuned low-latency setting on 3.5 Flash.
"""

from __future__ import annotations

import json
import os

from loguru import logger

# Sentinel returned whenever vision cannot answer (no key, SDK missing, empty
# image, or upstream error). The caller degrades to the Pillow fallback on it.
VISION_UNAVAILABLE = "VISION_UNAVAILABLE"

# NOTE: read straight from the environment to keep this module self-contained
# and importable regardless of config-block ordering. ``GOOGLE_AI_API_KEY``
# already lives in config.py; ``VISION_MODEL`` is the S67b addition (A-AG-6).
# Either may be imported from ``config`` once that constant lands — the default
# below matches it.
_DEFAULT_VISION_MODEL = "gemini-3.5-flash"


def _build_prompt(mode: str, scene_context: str) -> str:
    """Compose the Gemini text part for the requested reasoning mode (V6).

    ``mode`` is derived from the visitor's utterance by the caller:
    ``point`` ("what am I pointing at?"), ``assess`` (grade a written answer
    against the scene's expected answer), or ``describe`` (general "what's on
    screen?"). Unknown modes fall through to ``describe``.
    """
    base = (
        "You are the vision component of a voice agent in a live interactive room. The attached "
        "image is a screenshot of exactly what the visitor currently sees, including any pen marks, "
        "highlights, or text the visitor has drawn on top of the scene. "
    )
    if mode == "point":
        task = (
            "The visitor is pointing at or circling something with their annotation. Identify the "
            "specific object, person, or region the annotation indicates and describe it concisely."
        )
    elif mode == "assess":
        task = (
            "The visitor has written an answer on the scene. Using the expected answer in the scene "
            "context, state whether the visitor's answer is correct, partially correct, or incorrect, "
            "with a one-line reason."
        )
    else:  # describe (default / fallback)
        task = "Describe what is currently on screen, concisely."
    ctx = f"\nScene context: {scene_context}" if scene_context else ""
    return (
        base
        + task
        + ctx
        + "\nRespond in 1-3 short sentences for the voice agent to speak."
    )


def derive_vision_mode(utterance: str) -> str:
    """Pick a reasoning mode (V6) from the visitor's utterance.

    Keyword heuristic — imperfect by design; refine as real transcripts come
    in. Order matters: 'assess' (grading a written answer) is checked before
    'point' (referring to a drawn pointer/circle); everything else falls
    through to 'describe'.
    """
    u = (utterance or "").lower()
    if any(
        k in u
        for k in (
            "is this right",
            "is this correct",
            "is that right",
            "is that correct",
            "am i right",
            "am i correct",
            "did i get",
            "is my answer",
            "correct?",
            "right answer",
            "wrong",
        )
    ):
        return "assess"
    if any(
        k in u
        for k in (
            "pointing at",
            "point at",
            "what am i",
            "what's this",
            "what is this",
            "what did i",
            "circled",
            "circle",
            "drew",
            "drawing",
            "marked",
            "highlighted",
            "underlined",
            "this here",
            "right here",
        )
    ):
        return "point"
    return "describe"


class VisionClient:
    """Thin async wrapper over ``google-genai`` for one-shot image reasoning.

    Stateless across calls except for a cached ``genai.Client`` (connection
    reuse). Safe to construct unconditionally at session start — when no API
    key is present it simply reports ``enabled == False`` and every call
    returns the sentinel without touching the SDK.
    """

    def __init__(self, model: str | None = None, api_key: str | None = None):
        # Params default to the environment but are injectable for tests
        # (e.g. ``VisionClient(api_key="")`` exercises the stub path with no
        # monkeypatching and no SDK installed).
        self.model = model or os.getenv("VISION_MODEL", _DEFAULT_VISION_MODEL)
        self._api_key = (
            os.getenv("GOOGLE_AI_API_KEY", "") if api_key is None else api_key
        )
        self._client = None

    @property
    def enabled(self) -> bool:
        return bool(self._api_key)

    def _ensure_client(self):
        """Lazily construct + cache the genai client. Import is deferred here
        so the module loads even when ``google-genai`` isn't installed."""
        if self._client is None:
            from google import (
                genai,
            )  # google-genai SDK (A-AG-4: google-genai<2,>=1.68.0)

            self._client = genai.Client(api_key=self._api_key)
        return self._client

    async def analyze_image(
        self,
        image_bytes: bytes,
        mode: str,
        scene_context: str = "",
        mime_type: str = "image/jpeg",
    ) -> str:
        """Reason over a captured frame; return spoken-ready text or the sentinel.

        Never raises — a vision failure must not break the in-flight turn;
        the caller checks for ``VISION_UNAVAILABLE`` and falls back.
        """
        if not self.enabled:
            logger.info("[VISION] GOOGLE_AI_API_KEY unset — vision stubbed (sentinel)")
            return VISION_UNAVAILABLE
        if not image_bytes:
            logger.warning("[VISION] analyze_image called with empty image bytes")
            return VISION_UNAVAILABLE

        prompt = _build_prompt((mode or "describe").strip().lower(), scene_context)
        try:
            client = self._ensure_client()
            from google.genai import types

            # Native async surface (client.aio.*) — the SDK's sync
            # generate_content would block the Pipecat event loop (audio, STT,
            # Daily I/O all share it). See A-AG-4's async gotcha.
            resp = await client.aio.models.generate_content(
                model=self.model,
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                    prompt,
                ],
                config=types.GenerateContentConfig(
                    # gemini-3.5-flash uses thinking_level (3.x); "low" is the
                    # tuned low-latency setting. For 2.5 models this would be
                    # thinking_budget=<int> instead — never set both (3.x errors).
                    thinking_config=types.ThinkingConfig(thinking_level="low"),
                ),
            )
            text = (getattr(resp, "text", "") or "").strip()
            if not text:
                logger.warning("[VISION] model returned empty text mode={!r}", mode)
                return VISION_UNAVAILABLE
            logger.info("[VISION] analyze_image ok mode={!r} chars={}", mode, len(text))
            return text
        except Exception as exc:
            # Covers a missing SDK (key set but google-genai not yet
            # installed), bad key, model-not-found, quota, network — all
            # degrade to the Pillow fallback rather than breaking the turn.
            logger.warning("[VISION] analyze_image failed mode={!r}: {!r}", mode, exc)
            return VISION_UNAVAILABLE

    async def locate(self, image_bytes: bytes, description: str) -> dict | None:
        """Locate a described target in the image; return a normalized box or None.

        Block 7 (S67b agent-annotate path) — ask Gemini *where* a described
        element is so the shell can draw an annotation there. Returns
        ``{"x", "y", "w", "h"}`` as fractions in 0-1 (top-left origin) — NOT
        1000-space and NOT pixels — so the caller maps the fraction into
        whichever target space the annotate command wants (design px / percent;
        see A-AG-3's coordinate-space note). Returns None on stub / not-found /
        degenerate box / error. Never raises — a vision failure must not break
        the in-flight turn (mirrors analyze_image).
        """
        if not self.enabled:
            logger.info("[VISION] GOOGLE_AI_API_KEY unset — locate stubbed")
            return None
        if not image_bytes:
            logger.warning("[VISION] locate called with empty image bytes")
            return None

        # gemini-3.5-flash returns box_2d as [ymin, xmin, ymax, xmax] normalized
        # 0-1000 with a top-left origin (A-AG-5, confirmed against the official
        # image-understanding docs). response_mime_type forces parseable JSON.
        prompt = (
            "Return ONLY JSON locating this target in the image: " + description + ". "
            'Format: {"box_2d": [ymin, xmin, ymax, xmax]} with integers normalized '
            "0-1000 (top-left origin). If the target is not visible, return "
            '{"box_2d": null}.'
        )
        try:
            client = self._ensure_client()
            from google.genai import types

            # Native async surface (client.aio.*) — same discipline as
            # analyze_image; the SDK's sync generate_content would block the
            # Pipecat event loop. (The B7 sketch's asyncio.to_thread + sync call
            # is the part adapted to A-AG-5's confirmed aio shape.)
            resp = await client.aio.models.generate_content(
                model=self.model,
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                    prompt,
                ],
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(thinking_level="low"),
                    response_mime_type="application/json",
                ),
            )
            data = json.loads((getattr(resp, "text", "") or "").strip())
            box = data.get("box_2d") if isinstance(data, dict) else None
            if not box or len(box) != 4:
                logger.info(
                    "[VISION] locate found nothing description={!r}", description
                )
                return None
            # Normalize 0-1000 → 0-1 fractions. box_2d order is [ymin, xmin, ymax, xmax].
            ymin, xmin, ymax, xmax = (float(v) / 1000.0 for v in box)
            x, y = max(0.0, xmin), max(0.0, ymin)
            w, h = max(0.0, xmax - xmin), max(0.0, ymax - ymin)
            if w <= 0 or h <= 0:
                logger.info(
                    "[VISION] locate degenerate box description={!r} box_2d={}",
                    description,
                    box,
                )
                return None
            logger.info(
                "[VISION] locate ok description={!r} box_2d={}", description, box
            )
            return {"x": x, "y": y, "w": w, "h": h}
        except Exception as exc:
            logger.warning(
                "[VISION] locate failed description={!r}: {!r}", description, exc
            )
            return None
