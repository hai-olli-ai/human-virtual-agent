"""Lazy vision-frame tracker for the voice agent (S66 Block 5a).

The agent adds a vision frame (base64-encoded scene PNG) to the LLM
context so the model can answer "what's on screen?" questions. Pre-Block
5a, that frame was fetched and re-added on EVERY scene change — a
backend Pillow render in the hot path of every navigation, the single
biggest contributor to ``T_agent``.

This module owns the in-session state that decides when a fresh fetch is
needed. The runtime path:

1. Session start fetches once (eager bootstrap; unchanged by 5a).
   ``VisionFrameTracker.mark_loaded(initial_scene_id)`` records that.
2. Scene change in ``VISION_REFRESH_MODE=lazy`` (default) calls
   ``invalidate()`` — no fetch, just mark the cached frame stale.
3. Next ``canvas_analyze`` calls ``ensure_loaded(...)`` (wired via
   :class:`CanvasToolContext.ensure_vision`) which fetches + adds the
   frame to context if the tracker doesn't already cover the current
   scene.

Eager mode keeps the pre-5a behavior: every scene change fetches +
``mark_loaded(...)``s, and ``canvas_analyze``'s ensure() is a no-op
because the tracker already covers the current scene.
"""

from __future__ import annotations

from typing import Awaitable, Callable, Optional

from loguru import logger


class VisionFrameTracker:
    """Tracks which scene_id's vision frame is currently in the LLM context.

    A scene_id of ``None`` is treated as "not loaded" — we can't tell on
    the next ensure() whether an unknown-id frame is still valid, so we
    refuse to mark unknown ids as loaded. The edge case (no flow, no
    scene_id ever) ends up refetching on every ensure(), which is
    correct-but-wasteful; flow-based rooms always have ids.
    """

    def __init__(self) -> None:
        self._loaded_scene_id: Optional[str] = None

    @property
    def loaded_scene_id(self) -> Optional[str]:
        return self._loaded_scene_id

    def is_loaded_for(self, scene_id: Optional[str]) -> bool:
        if self._loaded_scene_id is None or scene_id is None:
            return False
        return self._loaded_scene_id == scene_id

    def mark_loaded(self, scene_id: Optional[str]) -> None:
        if scene_id:
            self._loaded_scene_id = scene_id

    def invalidate(self) -> None:
        self._loaded_scene_id = None


# Type aliases for the injection points.
VisionFetcher = Callable[[], Awaitable[Optional[str]]]
VisionConsumer = Callable[[str], None]


async def ensure_vision_frame_for_scene(
    tracker: VisionFrameTracker,
    current_scene_id: Optional[str],
    fetch_image_base64: VisionFetcher,
    add_vision_message: VisionConsumer,
) -> bool:
    """Fetch + inject a vision frame when the tracker doesn't cover the scene.

    Returns ``True`` iff a fetch was attempted and produced a frame that
    was added to the LLM context. Returns ``False`` when the tracker
    already covers ``current_scene_id`` (no-op), or when the fetch
    returned nothing (transient backend failure; the next call retries).

    Failures from ``fetch_image_base64`` propagate to the caller — the
    canvas_analyze handler wraps the call in try/except so a vision
    failure doesn't break the in-flight tool call.
    """
    if tracker.is_loaded_for(current_scene_id):
        return False

    img = await fetch_image_base64()
    if not img:
        logger.warning(
            "[VISION] lazy fetch returned no image for scene_id={!r}",
            current_scene_id,
        )
        return False

    add_vision_message(img)
    tracker.mark_loaded(current_scene_id)
    logger.info(
        "[VISION] lazy-loaded scene image for scene_id={!r} ({} chars base64)",
        current_scene_id,
        len(img),
    )
    return True
