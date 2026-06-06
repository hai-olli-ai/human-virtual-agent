"""S67b — vision query orchestration (capture-first, Pillow fallback).

Mirrors the extracted-core pattern of ``tools.quiz_generation.run_quiz_generation``:
a plain, dependency-injected coroutine that bot.py's per-pipeline
``_ensure_vision_for_active_scene`` closure wraps in one line. Keeping the
orchestration here (not inline in bot.py) makes it unit-testable — bot.py
needs the full pipecat stack to import; this module needs only its injected
deps (``request_capture`` closure, the vision client, and the api_client
module as ``backend_client``).

Flow (Design B — A-AG-3):
  1. derive the reasoning mode from the utterance (V6).
  2. request a live capture of the visitor's annotated canvas; in parallel,
     for 'assess', fetch the scene's expected-answer context (A-AG-5).
  3. ready capture → fetch JPEG bytes → reason via the dedicated Gemini
     vision client; timeout/error/not-ready → fall back to the base-scene
     Pillow PNG and flag the annotation blind spot (V7 honesty).
  4. return a developer message (or None) for the caller to add to context.

Rides on services.vision_client (the thin Gemini wrapper) and is decoupled
from the conversational LLM provider exactly as that client is.
"""

from __future__ import annotations

import asyncio
import base64

from loguru import logger

from scene_context import build_scene_knowledge_section
from services.vision_client import VISION_UNAVAILABLE, derive_vision_mode

# V7 — honesty notes injected into the LLM context.
_BLIND_SPOT_NOTE = (
    " [vision note: this is the base scene only — the visitor's own pen marks, "
    "highlights, and handwriting are NOT visible to you. Do not claim to see "
    "circles, drawings, or written answers; invite the visitor to share their "
    "screen so you can see what they drew.]"
)
_UNAVAILABLE_NOTE = (
    "[vision note: you cannot see the canvas this turn (capture unavailable). "
    "Answer from your knowledge/context if you can, and do not claim to see the "
    "screen or the visitor's drawings.]"
)
# The exact live-room shell button the visitor clicks to enable screen-share.
# Named verbatim in the notes so the agent points at the real control.
_SHARE_SCREEN_BUTTON = '"Let the Assistant see your screen"'

_SHARE_SCREEN_NOTE = (
    "[vision note: the visitor is asking about something they drew, circled, or "
    "pointed at, but screen-share is not on, so you cannot see their screen or "
    "annotations. Do not guess or describe the base scene as if it were their "
    f"drawing. Warmly and briefly ask them to click the {_SHARE_SCREEN_BUTTON} "
    "button so you can see what they've marked, then invite them to ask again once "
    "they have.]"
)
_DESCRIBE_SHARE_NUDGE_NOTE = (
    "[vision note: the visitor asked what's on screen, but screen-share is not on "
    "yet, so you can only see the base scene — not their live view. Before "
    "answering, warmly and briefly ask them to click the "
    f"{_SHARE_SCREEN_BUTTON} button so you can see their actual screen and give a "
    "better answer. Do not describe the scene this turn — just invite them to "
    "enable it.]"
)
_CAPTURE_RETRY_NOTE = (
    "[vision note: you could not grab the visitor's screen this turn — a transient "
    "capture glitch, NOT a permission problem (screen-share is on). Do not answer from "
    "the base scene or guess at their drawing. Briefly tell the visitor you couldn't see "
    "their screen just now and ask them to try again in a moment.]"
)


def _is_permission_error(capture_result) -> bool:
    """True when the shell declined the capture for a permission reason — the
    visitor hasn't granted screen-share, so their annotations are unviewable.

    Tolerant of both shapes the shell may send: ``status: 'permission_denied'``
    and ``status: 'error', error: 'permission_required'`` (any field containing
    'permission').
    """
    if not capture_result:
        return False
    blob = f"{capture_result.get('status', '')} {capture_result.get('error', '')}".lower()
    return "permission" in blob


async def _fetch_live_bytes(
    capture_id, capture_result, backend_client, slug, api_url
) -> bytes | None:
    """Live JPEG bytes IFF the capture is ready and the backend still has them.

    Never falls back to anything — when screen-share is on this is the only
    acceptable image source (it carries the visitor's actual annotations).
    Returns None on not-ready / missing bytes so the caller can retry or report
    a transient failure rather than substitute the annotation-less base scene.
    """
    if capture_result and capture_result.get("status") == "ready" and slug:
        img = await backend_client.get_vision_capture(slug, capture_id, api_url)
        if img:
            return img
        logger.warning(
            "[VISION] capture {} ready but byte fetch returned nothing", capture_id
        )
    return None


async def _fetch_pillow(backend_client, room_id, api_url) -> bytes | None:
    """Base-scene Pillow PNG bytes (annotation-less, S46/S66).

    Used in exactly ONE place — a 'describe' question when screen-share is OFF
    (a permission denial). It is NEVER a substitute for a live annotated
    capture; when screen-share is on, the live path is authoritative.
    """
    png_b64 = await backend_client.get_scene_image_base64(room_id, api_url)
    if png_b64:
        try:
            return base64.b64decode(png_b64)
        except Exception as exc:
            logger.warning("[VISION] failed to decode Pillow PNG: {!r}", exc)
    return None


async def _fetch_assess_scene_context(
    *,
    backend_client,
    scene_id: str | None,
    room_id: str | None,
    api_url: str | None,
) -> str:
    """Assemble the scene's expected-answer grounding for 'assess' (A-AG-5).

    Pulls instruction + scene-scope knowledge + script text from a by-id
    snapshot fetch (cheap, backend-cached). Empty string on any miss.
    """
    snap = await backend_client.get_scene_snapshot(room_id, api_url, scene_id=scene_id)
    if not snap:
        return ""
    cs = snap.get("current_scene") or {}
    parts: list[str] = []
    if cs.get("instruction"):
        parts.append(str(cs["instruction"]))
    scene_knowledge = build_scene_knowledge_section(snap.get("knowledge"))
    if scene_knowledge:
        parts.append(scene_knowledge)
    parts.extend(s.get("text", "") for s in (cs.get("scripts") or []) if s.get("text"))
    return "\n\n".join(p for p in parts if p)


async def run_vision_query(
    question: str,
    *,
    request_capture,
    vision_client,
    backend_client,
    session_context,
    room_id: str | None,
    api_url: str | None,
    on_vision_state=None,
) -> dict | None:
    """Run the Design-B vision flow; return a developer message or None.

    None ⇒ nothing to inject (no room_id). Otherwise a ``{role, content}`` dict
    the caller adds to the LLM context. **Live-capture-first:** when screen-share
    is on (the capture comes back ``ready``) the agent ALWAYS reasons over the
    real annotated screen via Gemini — Pillow is never substituted. When
    screen-share is OFF (a permission denial): point/assess → a share-your-screen
    note (never Pillow); 'describe' → a two-stage nudge — the FIRST describe asks
    the visitor to click the screen-share button (no Pillow), a REPEATED describe
    while still off falls back to the base-scene Pillow PNG + blind-spot note. The
    describe nudge resets whenever a capture comes back ``ready``. Other outcomes:
    a transient capture miss with screen-share on (retried once) → a try-again
    note; a Gemini degrade / no image → an unavailable note. Reads slug, scene_id
    and the describe-nudge flag from ``session_context``. Never raises —
    handle_analyze wraps the caller in try/except, but a vision failure should
    still degrade to a note rather than break the turn. An optional
    ``on_vision_state`` async hook is invoked with "analyzing" then "idle"
    bracketing the Gemini call — a deterministic signal for a shell-side
    "looking at your screen…" indicator. Fast nudge/retry paths never call it.
    """
    if not room_id:
        return None
    slug = session_context.get_slug()
    scene_id = session_context.get_current_scene_id()
    mode = derive_vision_mode(question)

    # A-AG-5 — run the capture round-trip and (for 'assess') the expected-answer
    # snapshot fetch in PARALLEL, so the backend-cached snapshot GET hides under
    # the capture latency.
    if mode == "assess":
        (capture_id, capture_result), scene_context = await asyncio.gather(
            request_capture(question),
            _fetch_assess_scene_context(
                backend_client=backend_client, scene_id=scene_id,
                room_id=room_id, api_url=api_url,
            ),
        )
    else:
        capture_id, capture_result = await request_capture(question)
        scene_context = ""

    # Resolve the live capture, with one retry on a FAST transient miss (the
    # shell replied but bytes were missing / a non-permission error). A permission
    # denial is a stable state (no retry); a timeout (capture_result is None)
    # already waited the full budget, so don't double the dead-air — fall through
    # to the retry note instead.
    img = await _fetch_live_bytes(capture_id, capture_result, backend_client, slug, api_url)
    if img is None and capture_result is not None and not _is_permission_error(capture_result):
        logger.info("[VISION] live capture transient miss → retrying once")
        capture_id, capture_result = await request_capture(question)
        img = await _fetch_live_bytes(capture_id, capture_result, backend_client, slug, api_url)

    # Screen-share is confirmed ON whenever a capture comes back 'ready' — reset
    # the describe nudge so the NEXT off-period asks once more before Pillow
    # (decision 1: reset-on-ready).
    if (capture_result or {}).get("status") == "ready":
        session_context.set_describe_share_nudged(False)

    # Bracket the (slow) Gemini call with a deterministic vision_state signal so
    # the shell shows "looking at your screen…" only while the model is actually
    # running — not on the fast nudge/retry paths. "analyzing" before, "idle"
    # after (try/finally, so it always clears even if analyze degrades).
    async def _analyze(image_bytes, mime):
        if on_vision_state is not None:
            await on_vision_state("analyzing")
        try:
            return await vision_client.analyze_image(image_bytes, mode, scene_context, mime_type=mime)
        finally:
            if on_vision_state is not None:
                await on_vision_state("idle")

    # 1. Live capture succeeded → reason over the REAL screen (annotations and
    #    all). This is the ONLY path used when screen-share is on — Pillow is
    #    never substituted here.
    if img is not None:
        answer = await _analyze(img, "image/jpeg")
        if answer == VISION_UNAVAILABLE:
            return {"role": "developer", "content": _UNAVAILABLE_NOTE}
        return {"role": "developer", "content": f"[vision: {mode}] {answer}"}

    # 2. Screen-share is OFF (permission denied) — the ONLY place Pillow lives.
    if _is_permission_error(capture_result):
        if mode in ("point", "assess"):
            # Annotation question with no screen-share → ask for the share; the
            # base scene can't show what they drew anyway.
            logger.info(
                "[VISION] capture {} permission-denied + mode={} → screen-share fast-path",
                capture_id, mode,
            )
            return {"role": "developer", "content": _SHARE_SCREEN_NOTE}
        # 'describe' — two-stage (decision 2: describe-specific). The FIRST
        # describe while screen-share is off nudges the visitor to click the
        # share button (no Pillow yet); only a REPEATED describe while still off
        # falls back to the base-scene Pillow render.
        if not session_context.get_describe_share_nudged():
            session_context.set_describe_share_nudged(True)
            logger.info("[VISION] describe + screen-share off (first) → nudge to enable screen-share")
            return {"role": "developer", "content": _DESCRIBE_SHARE_NUDGE_NOTE}
        logger.info("[VISION] describe + screen-share off (repeat) → Pillow base scene")
        png = await _fetch_pillow(backend_client, room_id, api_url)
        if png is None:
            return {"role": "developer", "content": _UNAVAILABLE_NOTE}
        answer = await _analyze(png, "image/png")
        if answer == VISION_UNAVAILABLE:
            return {"role": "developer", "content": _UNAVAILABLE_NOTE}
        return {"role": "developer", "content": f"[vision: {mode}] {answer}{_BLIND_SPOT_NOTE}"}

    # 3. Transient failure with screen-share ON (timeout, or both attempts missed)
    #    — never substitute the annotation-less Pillow scene; ask for a retry.
    logger.info("[VISION] capture {} transient failure → retry note", capture_id)
    return {"role": "developer", "content": _CAPTURE_RETRY_NOTE}
