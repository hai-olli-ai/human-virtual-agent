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
  * barge-in    → ``narration_pause``, let the conversation answer on the live
                  path, then ``narration_resume`` at position (§2.6 — pause,
                  not the abort semantics TTS narration keeps).

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

# Timer margin over the clip's own duration before the TTS fallback fires.
CUE_TIMEOUT_MARGIN_S = 10.0
# After a barge-in: how long we're willing to hold the paused clip while the
# conversation answers, before resuming regardless (a stuck detector, not a
# feature — ☐12 exercises the real path).
CUE_MAX_PAUSE_S = 180.0
# The answer must START within this window after a barge-in or we resume
# (visitor noise with no actual question produces no bot turn at all).
CUE_ANSWER_START_S = 20.0
# Settle after the bot's answer finishes before the video resumes.
CUE_RESUME_SETTLE_S = 2.0
_POLL_S = 0.25

SendFn = Callable[[dict], Awaitable[None]]
SpeakFn = Callable[[str], Awaitable[None]]
ExpectInterruptionFn = Callable[[], "asyncio.Future[Any]"]
BoolFn = Callable[[], bool]


class NarrationCueController:
    """One per pipeline instance; one active cue at a time (narration is
    strictly sequential). ``begin_run`` re-arms per narration run."""

    def __init__(
        self,
        *,
        send_message: SendFn,
        speak_fallback: SpeakFn,
        expect_interruption: ExpectInterruptionFn,
        bot_is_speaking: BoolFn,
        timeout_margin_s: float = CUE_TIMEOUT_MARGIN_S,
    ):
        self._send = send_message
        self._speak_fallback = speak_fallback
        self._expect_interruption = expect_interruption
        self._bot_is_speaking = bot_is_speaking
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
        """Wait out one clip: completion wins; barge-in pauses and resumes;
        the (re-armed) timer loses to the fallback."""
        while True:
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
                await self._pause_for_answer()
                # Loop: fresh interruption future + a fresh full timer (the
                # shell resumed at position; full-duration + margin is a safe
                # cap, not a schedule).
                continue
            # timeout
            if not interruption.done():
                interruption.cancel()
            return False

    async def _pause_for_answer(self) -> None:
        """§2.6 barge-in: pause the clip, let the live path answer, resume."""
        await self._send({"type": "narration_pause"})
        loop = asyncio.get_running_loop()
        deadline = loop.time() + CUE_MAX_PAUSE_S
        answer_start_deadline = loop.time() + CUE_ANSWER_START_S
        # Phase 1: wait for the answer to START (bot speaking) — a barge-in
        # that produces no bot turn at all resumes after a short window.
        while loop.time() < answer_start_deadline:
            if self._bot_is_speaking():
                break
            if self._completion is not None and self._completion.done():
                return  # shell finished anyway (races are legal)
            await asyncio.sleep(_POLL_S)
        # Phase 2: wait for the answer to FINISH (quiet + settle), capped.
        quiet_since: float | None = None
        while loop.time() < deadline:
            if self._completion is not None and self._completion.done():
                return
            if self._bot_is_speaking():
                quiet_since = None
            elif quiet_since is None:
                quiet_since = loop.time()
            elif loop.time() - quiet_since >= CUE_RESUME_SETTLE_S:
                break
            await asyncio.sleep(_POLL_S)
        await self._send({"type": "narration_resume"})
