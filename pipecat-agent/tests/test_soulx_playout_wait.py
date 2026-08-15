"""soulx-audio.v1 narration drain-wait — playout_completed + the estimate fallback.

Layers under test:

  * :meth:`SoulXAudioClient.wait_playout_completed` — resolution order:
    the renderer's ``playout_completed`` (truth), the sent-audio estimate
    frozen at ``turn_end`` (version-skew fallback for renderer builds that
    predate the emission), then the caller's timeout budget. ``turn_failed``
    and ``interrupt`` also release waiters — narration must never hang on a
    render failure or a barge-in.
  * :meth:`SoulXAudioSink.wait_group_playout` — the relay pipeline's
    ``wait_playout`` hook. On this path narration audio plays out on the
    RENDERER (the sink swallows it), so the classic NarrationCompletionGate
    has nothing to observe; this wait is what makes ``script_complete`` mean
    "the visitor finished hearing it" instead of "the frames were queued" —
    the queue-time emission is what made auto-advance skip every scripted
    scene in SoulX mode.
  * :meth:`SoulXAudioSink.begin_narration_watch` — arming hygiene: a stale
    conversational turn id must not satisfy a new narration run's wait.

Follows the existing tests/ convention: no pytest-asyncio, so each async
test body goes through ``asyncio.run``; sink internals are poked directly
where wiring a full pipeline would add nothing (the ``_primed`` precedent
in ``test_cached_first_tts.py``).
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, patch

from pipecat.frames.frames import (
    InterruptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

import services.soulx_audio as soulx_audio
from narration import NarrationInterrupted
from services.soulx_audio import SoulXAudioClient, SoulXAudioSink


def _run(coro):
    return asyncio.run(coro)


class _FakeWS:
    """Collects sends; iterable for the read-loop test."""

    def __init__(self, incoming: list[str] | None = None):
        self.sent: list = []
        self._incoming = list(incoming or [])

    async def send(self, data):
        self.sent.append(data)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._incoming:
            raise StopAsyncIteration
        return self._incoming.pop(0)


def _make_client(healthy: bool = True) -> SoulXAudioClient:
    client = SoulXAudioClient(
        ws_url="wss://test.invalid/ws",
        auth_token="t",
        room_url="https://d.daily.co/room",
        room_token="rt",
    )
    client._ws = _FakeWS()
    client._healthy = healthy
    return client


# ──────────────────────────────────────────────────────────────────────
# SoulXAudioClient.wait_playout_completed
# ──────────────────────────────────────────────────────────────────────


def test_wait_returns_immediately_when_already_completed():
    async def body():
        client = _make_client()
        await client._mark_playout_done("turn-1")
        t0 = time.monotonic()
        assert await client.wait_playout_completed("turn-1", timeout_s=5.0) is True
        assert time.monotonic() - t0 < 0.5

    _run(body())


def test_wait_resolves_when_playout_completed_arrives_mid_wait():
    async def body():
        client = _make_client()
        waiter = asyncio.create_task(client.wait_playout_completed("turn-1", 5.0))
        await asyncio.sleep(0.05)
        assert not waiter.done()
        await client._mark_playout_done("turn-1")
        assert await asyncio.wait_for(waiter, timeout=1.0) is True

    _run(body())


def test_turn_failed_releases_waiter_via_read_loop():
    async def body():
        client = _make_client()
        client._ws = _FakeWS(
            incoming=['{"type": "turn_failed", "turn_id": "turn-9", "error": "boom"}']
        )
        waiter = asyncio.create_task(client.wait_playout_completed("turn-9", 5.0))
        await asyncio.sleep(0.05)
        await client._read_loop()
        # The read loop ending also flips healthy False; the completion set
        # must win — the turn IS resolved, not timed out.
        assert await asyncio.wait_for(waiter, timeout=1.0) is True

    _run(body())


def test_estimate_fallback_when_renderer_never_signals(monkeypatch):
    async def body():
        monkeypatch.setattr(soulx_audio, "PLAYOUT_EST_MARGIN_S", 0.2)
        client = _make_client()
        await client.turn_start("turn-1")
        # 0.1s of s16le mono @ 24kHz
        await client.send_pcm(b"\x00" * int(24000 * 2 * 0.1))
        await client.turn_end("turn-1")
        t0 = time.monotonic()
        assert await client.wait_playout_completed("turn-1", timeout_s=10.0) is True
        elapsed = time.monotonic() - t0
        # Resolved by the estimate (~0.3s from first send), NOT the 10s budget.
        assert elapsed < 2.0

    _run(body())


def test_wait_times_out_when_turn_never_ends():
    async def body():
        client = _make_client()
        await client.turn_start("turn-1")
        await client.send_pcm(b"\x00" * 4800)
        # No turn_end => no estimate; no playout_completed => budget exhausts.
        t0 = time.monotonic()
        assert await client.wait_playout_completed("turn-1", timeout_s=0.3) is False
        assert time.monotonic() - t0 >= 0.3

    _run(body())


def test_wait_returns_false_fast_when_unhealthy():
    async def body():
        client = _make_client(healthy=False)
        t0 = time.monotonic()
        assert await client.wait_playout_completed("turn-1", timeout_s=5.0) is False
        assert time.monotonic() - t0 < 0.5

    _run(body())


def test_interrupt_releases_waiter():
    async def body():
        client = _make_client()
        waiter = asyncio.create_task(client.wait_playout_completed("turn-3", 5.0))
        await asyncio.sleep(0.05)
        await client.interrupt("turn-3")
        assert await asyncio.wait_for(waiter, timeout=1.0) is True

    _run(body())


# ──────────────────────────────────────────────────────────────────────
# SoulXAudioSink.wait_group_playout
# ──────────────────────────────────────────────────────────────────────


def _drive(sink: SoulXAudioSink, frame) -> asyncio.Task:
    """process_frame with the FrameProcessor base + push_frame no-op'd.

    The sink's own turn bookkeeping is under test, not pipecat's pipeline
    plumbing — same isolation style as test_autoplay_phase_a.py's gate tests.
    """
    return asyncio.ensure_future(sink.process_frame(frame, FrameDirection.DOWNSTREAM))


def test_sink_interruption_latches_and_wait_raises():
    async def body():
        client = _make_client()
        sink = SoulXAudioSink(client)
        sink.begin_narration_watch()
        sink.begin_turn_group()
        with (
            patch.object(FrameProcessor, "process_frame", new=AsyncMock()),
            patch.object(sink, "push_frame", new=AsyncMock()),
        ):
            await _drive(sink, TTSStartedFrame())
            waiter = asyncio.create_task(sink.wait_group_playout(5.0))
            await asyncio.sleep(0.05)
            assert not waiter.done()
            await _drive(sink, InterruptionFrame())
            try:
                await asyncio.wait_for(waiter, timeout=2.0)
                raise AssertionError("expected NarrationInterrupted")
            except NarrationInterrupted:
                pass

    _run(body())


def test_sink_wait_returns_promptly_in_fallback():
    async def body():
        client = _make_client(healthy=False)
        sink = SoulXAudioSink(client)
        sink.begin_narration_watch()
        t0 = time.monotonic()
        await sink.wait_group_playout(5.0)  # must not raise, must not hang
        assert time.monotonic() - t0 < 0.5

    _run(body())


def test_sink_grouped_run_waits_for_renderer_playout(monkeypatch):
    async def body():
        monkeypatch.setattr(soulx_audio, "TURN_GROUP_DEBOUNCE_S", 0.05)
        client = _make_client()
        sink = SoulXAudioSink(client)
        sink.begin_narration_watch()
        sink.begin_turn_group()
        with (
            patch.object(FrameProcessor, "process_frame", new=AsyncMock()),
            patch.object(sink, "push_frame", new=AsyncMock()),
        ):
            # Two segments, one grouped renderer turn.
            await _drive(sink, TTSStartedFrame())
            await _drive(
                sink,
                TTSAudioRawFrame(
                    audio=b"\x00" * 4800, sample_rate=24000, num_channels=1
                ),
            )
            await _drive(sink, TTSStoppedFrame())
            await _drive(sink, TTSStartedFrame())
            await _drive(sink, TTSStoppedFrame())
            assert sink._await_turn_id == "turn-1"

            waiter = asyncio.create_task(sink.wait_group_playout(5.0))
            # Let the debounced group close send turn_end.
            await asyncio.sleep(0.2)
            assert sink._turn_id is None
            assert not waiter.done()  # playout hasn't been signalled yet

            await client._mark_playout_done("turn-1")
            await asyncio.wait_for(waiter, timeout=2.0)

    _run(body())


def test_sink_watch_ignores_stale_completed_turn():
    async def body():
        client = _make_client()
        sink = SoulXAudioSink(client)
        # A prior conversational turn completed long ago...
        sink._turn = 1
        sink._await_turn_id = "turn-1"
        await client._mark_playout_done("turn-1")
        # ...then a new narration run arms the watch with no turn open.
        sink.begin_narration_watch()
        assert sink._await_turn_id is None
        with (
            patch.object(FrameProcessor, "process_frame", new=AsyncMock()),
            patch.object(sink, "push_frame", new=AsyncMock()),
        ):
            waiter = asyncio.create_task(sink.wait_group_playout(5.0))
            await asyncio.sleep(0.15)
            # Phase 1 must still be polling — the stale turn-1 completion
            # must not have satisfied the new run's wait.
            assert not waiter.done()
            await _drive(sink, TTSStartedFrame())
            await _drive(sink, TTSStoppedFrame())  # ungrouped: closes turn-2 now
            await client._mark_playout_done("turn-2")
            await asyncio.wait_for(waiter, timeout=2.0)

    _run(body())
