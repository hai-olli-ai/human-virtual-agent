"""S79 — the animated-narration cue path (Canvas Protocol v0.3).

Layers under test:

  * :func:`narration.plan_narration_segments` — the per-line ``animation``
    pointer rides the plan (both pipelines), degenerate shapes drop to None.
  * :class:`narration.SceneNarrator` — animated lines route to ``cue``,
    everything else to ``speak``; **no cue wired ⇒ byte-alike pre-S79
    behavior** (the off-path diff-zero lock).
  * :class:`services.narration_cue.NarrationCueController` — the cue state
    machine: completion via the shell's ``script_complete`` (stale lineIndex
    ignored), timeout ⇒ ``narration_cancel`` + TTS fallback (never-block),
    barge-in ⇒ ``narration_cancel`` + NarrationInterrupted (the abort law).
  * Wire casing — snake ``type`` values, camelCase payload fields (the A1
    census law: an underscore type routes to the shell's general handler).
  * :meth:`SoulXAudioSink.expect_interruption` — the relay barge-in future.

Follows the existing tests/ convention: no pytest-asyncio, so each async
test body goes through ``asyncio.run``.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

from pipecat.frames.frames import InterruptionFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from narration import (
    NarrationInterrupted,
    SceneNarrator,
    plan_continue_action,
    plan_narration_segments,
)
from services.narration_cue import NarrationCueController
from services.soulx_audio import SoulXAudioClient, SoulXAudioSink


def _run(coro):
    return asyncio.run(coro)


# ──────────────────────────────────────────────────────────────────────
# plan_narration_segments — the animation pointer
# ──────────────────────────────────────────────────────────────────────


def _snapshot(lines):
    return {
        "live_room": {"language": "en"},
        "current_scene": {"scene_id": "sc-1", "scripts": lines},
    }


def test_plan_carries_animation_on_both_pipelines():
    lines = [
        {
            "text": "Hello",
            "animation": {
                "url": "https://media.maika.ai/animations/video/a.mp4",
                "duration_seconds": 4.2,
            },
        }
    ]
    for is_relay in (True, False):
        plan = plan_narration_segments(
            _snapshot(lines), primary_voice_id="v1", is_relay=is_relay
        )
        assert plan[0].animation == {
            "url": "https://media.maika.ai/animations/video/a.mp4",
            "duration_seconds": 4.2,
        }


def test_plan_drops_degenerate_animation_shapes():
    lines = [
        {"text": "no field"},
        {"text": "null", "animation": None},
        {"text": "junk", "animation": "not-a-dict"},
        {"text": "urlless", "animation": {"duration_seconds": 3}},
    ]
    plan = plan_narration_segments(
        _snapshot(lines), primary_voice_id="v1", is_relay=True
    )
    assert [seg.animation for seg in plan] == [None, None, None, None]


# ──────────────────────────────────────────────────────────────────────
# SceneNarrator routing + the diff-zero lock
# ──────────────────────────────────────────────────────────────────────


def _animated_snapshot():
    return _snapshot(
        [
            {
                "text": "Animated line",
                "animation": {"url": "https://x/a.mp4", "duration_seconds": 2},
            },
            {"text": "Voice line"},
        ]
    )


def test_narrator_routes_animated_to_cue_and_voice_to_speak():
    async def body():
        spoken, cued = [], []

        async def speak(text):
            spoken.append(text)

        async def cue(idx, seg):
            cued.append((idx, seg.text))

        narrator = SceneNarrator(
            primary_voice_id=None, set_voice=None, speak=speak, cue=cue
        )
        assert await narrator.narrate(_animated_snapshot()) is True
        assert cued == [(0, "Animated line")]
        assert spoken == ["Voice line"]

    _run(body())


def test_narrator_without_cue_speaks_everything_diff_zero():
    async def body():
        spoken = []

        async def speak(text):
            spoken.append(text)

        narrator = SceneNarrator(primary_voice_id=None, set_voice=None, speak=speak)
        await narrator.narrate(_animated_snapshot())
        assert spoken == ["Animated line", "Voice line"]

    _run(body())


# ──────────────────────────────────────────────────────────────────────
# NarrationCueController
# ──────────────────────────────────────────────────────────────────────


class _Harness:
    def __init__(self, *, timeout_margin_s=0.4):
        self.sent: list[dict] = []
        self.spoken: list[str] = []
        self._interruption_waiters: list[asyncio.Future] = []
        self.controller = NarrationCueController(
            send_message=self._send,
            speak_fallback=self._speak,
            expect_interruption=self._expect_interruption,
            timeout_margin_s=timeout_margin_s,
        )
        self.controller.begin_run("sc-1")

    async def _send(self, payload):
        self.sent.append(payload)

    async def _speak(self, text):
        self.spoken.append(text)

    def _expect_interruption(self):
        fut = asyncio.get_running_loop().create_future()
        self._interruption_waiters.append(fut)
        return fut

    def barge_in(self):
        for fut in self._interruption_waiters:
            if not fut.done():
                fut.set_result(True)
        self._interruption_waiters.clear()

    def types_sent(self):
        return [p["type"] for p in self.sent]


def test_cue_sends_camel_payload_and_resolves_on_script_complete():
    async def body():
        h = _Harness()
        task = asyncio.create_task(
            h.controller.cue(
                line_index=0, url="https://x/a.mp4", duration_seconds=2.0, text="Line"
            )
        )
        await asyncio.sleep(0.05)
        assert h.sent[0] == {
            "type": "narration_segment",
            "sceneId": "sc-1",
            "lineIndex": 0,
            "url": "https://x/a.mp4",
            "durationSeconds": 2.0,
        }
        assert (
            h.controller.on_script_complete({"type": "script_complete", "lineIndex": 0})
            is True
        )
        await asyncio.wait_for(task, timeout=1.0)
        assert h.spoken == []  # completion — no TTS fallback
        assert h.types_sent() == ["narration_segment"]

    _run(body())


def test_stale_line_index_completion_is_ignored():
    async def body():
        h = _Harness()
        task = asyncio.create_task(
            h.controller.cue(
                line_index=2, url="https://x/a.mp4", duration_seconds=0.1, text="Line"
            )
        )
        await asyncio.sleep(0.05)
        assert h.controller.on_script_complete({"lineIndex": 1}) is False
        assert not task.done()
        assert h.controller.on_script_complete({"lineIndex": 2}) is True
        await asyncio.wait_for(task, timeout=1.0)

    _run(body())


def test_completion_without_line_index_is_permissive():
    async def body():
        h = _Harness()
        task = asyncio.create_task(
            h.controller.cue(
                line_index=3, url="https://x/a.mp4", duration_seconds=1.0, text="Line"
            )
        )
        await asyncio.sleep(0.05)
        assert h.controller.on_script_complete({"type": "script_complete"}) is True
        await asyncio.wait_for(task, timeout=1.0)

    _run(body())


def test_timeout_sends_cancel_and_falls_back_to_tts():
    async def body():
        h = _Harness(timeout_margin_s=0.2)
        await asyncio.wait_for(
            h.controller.cue(
                line_index=0, url="https://x/a.mp4", duration_seconds=0.0, text="Say me"
            ),
            timeout=5.0,
        )
        assert h.types_sent() == ["narration_segment", "narration_cancel"]
        assert h.spoken == ["Say me"]  # never-block

    _run(body())


def test_barge_in_cancels_clip_and_raises_never_speaks():
    """Field spec 2026-08-16: a barge-in ABORTS the run (the voice-path law).
    The clip is cancelled shell-side (crossfade to live/portrait — the live
    SoulX track then carries the answer), NarrationInterrupted propagates so
    script_complete is suppressed, and the line is deliberately NOT TTS'd —
    resumption is explicit (Play button / the continue intent)."""

    async def body():
        h = _Harness(timeout_margin_s=5.0)
        task = asyncio.create_task(
            h.controller.cue(
                line_index=0, url="https://x/a.mp4", duration_seconds=30.0, text="Line"
            )
        )
        await asyncio.sleep(0.05)
        h.barge_in()
        try:
            await asyncio.wait_for(task, timeout=2.0)
            raise AssertionError("expected NarrationInterrupted")
        except NarrationInterrupted:
            pass
        assert h.types_sent() == ["narration_segment", "narration_cancel"]
        assert "narration_pause" not in h.types_sent()
        assert "narration_resume" not in h.types_sent()
        assert h.spoken == []  # never TTS the interrupted line

    _run(body())


def test_begin_run_drops_stale_cue_state():
    async def body():
        h = _Harness()
        task = asyncio.create_task(
            h.controller.cue(
                line_index=0, url="https://x/a.mp4", duration_seconds=5.0, text="Line"
            )
        )
        await asyncio.sleep(0.05)
        h.controller.begin_run("sc-2")  # superseding run
        assert h.controller.on_script_complete({"lineIndex": 0}) is False
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):  # noqa: BLE001
            pass

    _run(body())


# ──────────────────────────────────────────────────────────────────────
# SoulXAudioSink.expect_interruption
# ──────────────────────────────────────────────────────────────────────


def test_sink_expect_interruption_resolves_on_interruption_frame():
    async def body():
        client = SoulXAudioClient(
            ws_url="wss://test.invalid/ws",
            auth_token="t",
            room_url="r",
            room_token="rt",
        )
        client._healthy = True

        class _WS:
            async def send(self, data):
                pass

        client._ws = _WS()
        sink = SoulXAudioSink(client)
        waiter = sink.expect_interruption()
        with (
            patch.object(FrameProcessor, "process_frame", new=AsyncMock()),
            patch.object(sink, "push_frame", new=AsyncMock()),
        ):
            await sink.process_frame(InterruptionFrame(), FrameDirection.DOWNSTREAM)
        assert waiter.done() and waiter.result() is True

    _run(body())


# ──────────────────────────────────────────────────────────────────────
# plan_continue_action — the spoken-continue intent's one rule
# ──────────────────────────────────────────────────────────────────────


def test_continue_action_vectors():
    # Unfinished narration (interrupted or never run) → resume from segment 0.
    assert (
        plan_continue_action(narration_completed=False, scene_index=0, total_scenes=5)
        == "resume"
    )
    assert (
        plan_continue_action(narration_completed=False, scene_index=4, total_scenes=5)
        == "resume"
    )
    # Finished + a next scene exists → advance.
    assert (
        plan_continue_action(narration_completed=True, scene_index=0, total_scenes=5)
        == "next_scene"
    )
    assert (
        plan_continue_action(narration_completed=True, scene_index=3, total_scenes=5)
        == "next_scene"
    )
    # Finished at the last scene (and degenerate rooms) → end.
    assert (
        plan_continue_action(narration_completed=True, scene_index=4, total_scenes=5)
        == "end"
    )
    assert (
        plan_continue_action(narration_completed=True, scene_index=0, total_scenes=1)
        == "end"
    )
    assert (
        plan_continue_action(narration_completed=True, scene_index=0, total_scenes=0)
        == "end"
    )
