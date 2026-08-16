"""S79 — the animated-narration cue path (Canvas Protocol v0.3 additions).

For a script line whose snapshot entry carries ``animation.url`` the agent does
NOT synthesize speech — it cues the SHELL to play the pre-rendered MP4 (which
carries its own audio; the double-audio case is the cardinal bug) and waits for
the shell's ``script_complete`` app-message on the video's ``ended``. The
never-block law shapes every exit:

  * completion  → the shell reported ``ended`` — proceed to the next line;
  * timeout     → ``duration + margin`` elapsed with no completion (shell too
                  old, URL 404, tab hidden…) — send ``narration_cancel`` and
                  SPEAK the line through the pipeline's own TTS;
  * barge-in    → ``narration_cancel`` + :class:`NarrationInterrupted` — the
                  SAME abort law voice narration follows (field spec
                  2026-08-16, superseding §2.6's pause/resume sketch): the
                  run dies, ``script_complete`` is suppressed, and resumption
                  is EXPLICIT — the Play button (``autoplay_control resume``)
                  or the spoken-continue intent, both re-narrating the scene
                  from segment 0.

Wire casing (the A1 census law): ``type`` values are snake_case, payload
FIELDS are camelCase — an underscore type routes to the shell's general
handler; camel fields match every other Daily app-message.

Pipeline-agnostic by injection: bot.py wires one controller per pipeline with
its own send/speak/interruption/quiet primitives. A room with no animated
lines never invokes any of this — the off-path is diff-zero (lock-tested).
"""

from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable, Optional

from loguru import logger

from narration import NarrationInterrupted

# Timer margin over the clip's own duration before the TTS fallback fires.
CUE_TIMEOUT_MARGIN_S = 10.0

SendFn = Callable[[dict], Awaitable[None]]
SpeakFn = Callable[[str], Awaitable[None]]
ExpectInterruptionFn = Callable[[], "asyncio.Future[Any]"]


class NarrationCueController:
    """One per pipeline instance; one active cue at a time (narration is
    strictly sequential). ``begin_run`` re-arms per narration run."""

    def __init__(
        self,
        *,
        send_message: SendFn,
        speak_fallback: SpeakFn,
        expect_interruption: ExpectInterruptionFn,
        timeout_margin_s: float = CUE_TIMEOUT_MARGIN_S,
    ):
        self._send = send_message
        self._speak_fallback = speak_fallback
        self._expect_interruption = expect_interruption
        self._timeout_margin_s = timeout_margin_s
        self._scene_id: Optional[str] = None
        self._active_line: Optional[int] = None
        self._completion: Optional[asyncio.Future] = None

    def begin_run(self, scene_id: str | None) -> None:
        """Per-run arming (call where narration_gate.begin_run / the sink
        watch are armed). Clears any stale in-flight cue state."""
        self._scene_id = scene_id
        self._drop_active()

    def _drop_active(self) -> None:
        if self._completion is not None and not self._completion.done():
            self._completion.cancel()
        self._active_line = None
        self._completion = None

    def on_script_complete(self, payload: dict) -> bool:
        """Inbound shell→agent ``script_complete`` (the video's ``ended``).

        Returns True when it resolved the active cue — the caller then stops
        routing the message (it is NOT one of the agent's own run-level
        emissions echoing back; Daily doesn't loop those to self anyway).
        A ``lineIndex`` mismatch is a stale completion from a superseded
        cue — ignored. Payloads without lineIndex (older shells) resolve
        the active cue permissively.
        """
        if self._completion is None or self._completion.done():
            return False
        line_index = payload.get("lineIndex")
        if line_index is not None and line_index != self._active_line:
            logger.debug(
                "[CUE] stale script_complete lineIndex={} (active={}) — ignored",
                line_index,
                self._active_line,
            )
            return False
        self._completion.set_result(True)
        return True

    async def cue(
        self, *, line_index: int, url: str, duration_seconds: float, text: str
    ) -> None:
        """Play one animated line through the shell; never raises upward —
        every failure lands on the TTS fallback (never-block)."""
        loop = asyncio.get_running_loop()
        self._active_line = line_index
        self._completion = loop.create_future()
        await self._send(
            {
                "type": "narration_segment",
                "sceneId": self._scene_id,
                "lineIndex": line_index,
                "url": url,
                "durationSeconds": duration_seconds,
            }
        )
        try:
            completed = await self._await_playback(duration_seconds)
        finally:
            self._drop_active()
        if not completed:
            logger.warning(
                "[CUE] line {} never completed (timeout) — narration_cancel + TTS fallback",
                line_index,
            )
            await self._send({"type": "narration_cancel"})
            await self._speak_fallback(text)

    async def _await_playback(self, duration_seconds: float) -> bool:
        """Wait out one clip: completion wins; barge-in cancels the clip and
        raises (the run aborts — field spec 2026-08-16); the timer loses to
        the TTS fallback."""
        timeout = max(1.0, float(duration_seconds or 0.0)) + self._timeout_margin_s
        interruption = self._expect_interruption()
        done, _ = await asyncio.wait(
            {self._completion, interruption},
            timeout=timeout,
            return_when=asyncio.FIRST_COMPLETED,
        )
        if self._completion in done and not self._completion.cancelled():
            if not interruption.done():
                interruption.cancel()
            return True
        if interruption in done and not interruption.cancelled():
            # Clear the clip shell-side (crossfade to live/portrait — the
            # live SoulX track then carries the ANSWER in talking mode),
            # then abort the run like any interrupted narration.
            await self._send({"type": "narration_cancel"})
            raise NarrationInterrupted()
        # timeout
        if not interruption.done():
            interruption.cancel()
        return False
