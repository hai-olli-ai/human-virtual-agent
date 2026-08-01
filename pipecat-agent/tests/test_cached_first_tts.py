"""Tests for ``services/cached_first_tts.py`` (Block 12).

Layers under test:

  * :meth:`CachedFirstTTSService.run_tts` — hit path emits the
    TTSStartedFrame → TTSAudioRawFrame(s) → TTSStoppedFrame envelope
    with ``context_id`` propagated; miss path delegates to the live
    Cartesia ``run_tts`` (mocked here to avoid websocket I/O).
  * :meth:`CachedFirstTTSService.prime_cached` — single-shot
    consumption semantics: one prime feeds exactly one ``run_tts`` call.
  * Chunk alignment — every emitted ``TTSAudioRawFrame.audio`` payload
    is a whole number of sample frames (bytes_per_sample × num_channels)
    so the receiving transport sees a well-formed PCM stream.

Frame-envelope choice matters for the ``NarrationCompletionGate``
composition (see ``narration.py``): the gate observes ``TTSStoppedFrame``,
which is only emitted by the TTS lifecycle. If the cache regresses to
bare ``OutputAudioRawFrame``, the gate's ``expect_next_stop()`` future
never resolves and per-segment narration stalls.
``test_hit_never_emits_bare_output_audio_raw_frame`` guards this.

Follows the existing tests/ convention: no pytest-asyncio (not in the
dependency closure), so each async test body goes through ``asyncio.run``
via the ``_run`` helper. The internal ``_primed`` attribute is poked
directly in a couple of tests — a small SLF001 cost in exchange for
verifying single-shot consumption without introducing observability
hooks just for tests.
"""

from __future__ import annotations

import asyncio
from typing import AsyncGenerator
from unittest.mock import patch

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    OutputAudioRawFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.cartesia.tts import CartesiaTTSService

from services.cached_first_tts import CachedFirstTTSService, CachedSegment


# ──────────────────────────────────────────────────────────────────────
# Test helpers
# ──────────────────────────────────────────────────────────────────────


def _run(coro):
    return asyncio.run(coro)


def _make_service() -> CachedFirstTTSService:
    """Construct the service without opening any websocket.

    Cartesia's ``_connect`` is lazy (only called from the live ``run_tts``
    path), so constructing here is network-free. The api_key is a sentinel
    string — never used because every test that touches the live path
    patches ``CartesiaTTSService.run_tts`` first.
    """
    return CachedFirstTTSService(
        api_key="test-key",
        sample_rate=24000,
        settings=CachedFirstTTSService.Settings(voice="test-voice", model="sonic-3"),
    )


def _make_segment(
    *,
    num_samples: int = 480,
    sample_rate: int = 24000,
    num_channels: int = 1,
) -> CachedSegment:
    """Generate a CachedSegment of zero-PCM at the requested shape.

    ``num_samples`` is per-channel; total bytes = num_samples × 2 (pcm_s16le)
    × num_channels. Default = 480 samples × 2 bytes = 960 bytes = exactly
    one 20 ms chunk at 24 kHz mono — convenient single-chunk shape.
    """
    sample_frame = 2 * num_channels  # pcm_s16le
    return CachedSegment(
        pcm=b"\x00" * (num_samples * sample_frame),
        sample_rate=sample_rate,
        num_channels=num_channels,
    )


async def _collect(agen: AsyncGenerator[Frame, None]) -> list[Frame]:
    return [f async for f in agen]


# ──────────────────────────────────────────────────────────────────────
# Hit path — frame envelope + context_id propagation
# ──────────────────────────────────────────────────────────────────────


def test_hit_emits_started_audio_stopped_envelope():
    """Primed call yields TTSStartedFrame → TTSAudioRawFrame(s) → TTSStoppedFrame
    with ``context_id`` propagated through every frame, and the audio
    frames' sample_rate / num_channels match the primed segment."""

    async def body():
        svc = _make_service()
        svc.prime_cached(_make_segment(num_samples=480))  # 1 chunk @ 24kHz mono
        return await _collect(svc.run_tts("hello", "ctx-abc"))

    frames = _run(body())

    assert isinstance(frames[0], TTSStartedFrame)
    assert frames[0].context_id == "ctx-abc"

    assert isinstance(frames[-1], TTSStoppedFrame)
    assert frames[-1].context_id == "ctx-abc"

    audio_frames = frames[1:-1]
    assert audio_frames, "expected at least one TTSAudioRawFrame between Start/Stop"
    for f in audio_frames:
        assert isinstance(f, TTSAudioRawFrame), (
            f"expected TTSAudioRawFrame, got {type(f).__name__}"
        )
        assert f.context_id == "ctx-abc"
        assert f.sample_rate == 24000
        assert f.num_channels == 1


def test_hit_never_emits_bare_output_audio_raw_frame():
    """Guards the NarrationCompletionGate composition: every audio frame on
    the hit path must be the TTSAudioRawFrame subtype, NOT bare
    OutputAudioRawFrame. If we regress, the output transport won't fire
    the TTSStoppedFrame lifecycle for cached chunks and the gate's
    expect_next_stop() future hangs forever — narration would stall
    after the first segment."""

    async def body():
        svc = _make_service()
        svc.prime_cached(_make_segment(num_samples=480 * 3))  # 3 chunks
        return await _collect(svc.run_tts("hi", "ctx"))

    frames = _run(body())

    for f in frames:
        if isinstance(f, OutputAudioRawFrame):
            assert isinstance(f, TTSAudioRawFrame), (
                f"Cache emitted bare {type(f).__name__}; must be "
                "TTSAudioRawFrame so the output transport drives "
                "TTSStoppedFrame and the gate unblocks."
            )


# ──────────────────────────────────────────────────────────────────────
# Miss path — delegates to live Cartesia run_tts
# ──────────────────────────────────────────────────────────────────────


def test_miss_delegates_to_super_run_tts():
    """No prime ⇒ delegate to ``CartesiaTTSService.run_tts``. We patch the
    parent class's run_tts so the test doesn't try to open a Cartesia
    websocket; the sentinel frame in the patched output proves the
    miss path actually called through to super()."""
    sentinel = ErrorFrame(error="fake-live-frame")
    captured: dict = {}

    async def fake_run_tts(self, text, context_id):
        captured["text"] = text
        captured["context_id"] = context_id
        yield sentinel

    async def body():
        svc = _make_service()
        assert svc._primed is None  # noqa: SLF001 — verifying internal state
        with patch.object(CartesiaTTSService, "run_tts", new=fake_run_tts):
            return await _collect(svc.run_tts("hello", "ctx-miss"))

    frames = _run(body())

    assert frames == [sentinel]
    assert captured == {"text": "hello", "context_id": "ctx-miss"}


# ──────────────────────────────────────────────────────────────────────
# Prime consumption — single-shot
# ──────────────────────────────────────────────────────────────────────


def test_prime_consumed_once():
    """After a single run_tts call, the prime is cleared and a second call
    with no re-prime falls through to live. Otherwise a stale prime from
    segment N could play instead of segment N+1's text — a correctness
    bug the narrator's per-segment prime/speak invariant relies on
    not happening."""

    async def fake_run_tts(self, text, context_id):
        yield TTSStoppedFrame(context_id=context_id)

    async def body():
        svc = _make_service()
        svc.prime_cached(_make_segment(num_samples=240))

        # First call — cache hit. Drain frames; assertion is on _primed
        # post-call, not on frame contents (covered by envelope tests).
        await _collect(svc.run_tts("hi-1", "ctx-1"))
        assert svc._primed is None, "prime must be consumed by run_tts"  # noqa: SLF001

        # Second call — no re-prime ⇒ must delegate. We mock the live
        # path so its sentinel frame proves we took the miss branch.
        with patch.object(CartesiaTTSService, "run_tts", new=fake_run_tts):
            return await _collect(svc.run_tts("hi-2", "ctx-2"))

    frames = _run(body())

    assert len(frames) == 1
    assert isinstance(frames[0], TTSStoppedFrame)
    assert frames[0].context_id == "ctx-2"


def test_prime_cached_with_none_clears():
    """``prime_cached(None)`` explicitly clears any stashed segment so a
    caller that wants to abort a planned hit (e.g. an in-flight
    interruption) doesn't have to reach into ``_primed`` directly."""

    async def fake_run_tts(self, text, context_id):
        yield TTSStoppedFrame(context_id=context_id)

    async def body():
        svc = _make_service()
        svc.prime_cached(_make_segment(num_samples=100))
        svc.prime_cached(None)
        assert svc._primed is None  # noqa: SLF001
        with patch.object(CartesiaTTSService, "run_tts", new=fake_run_tts):
            return await _collect(svc.run_tts("x", "ctx"))

    frames = _run(body())

    assert isinstance(frames[0], TTSStoppedFrame)


# ──────────────────────────────────────────────────────────────────────
# Chunk alignment
# ──────────────────────────────────────────────────────────────────────


def _assert_aligned_replay(svc: CachedFirstTTSService, seg: CachedSegment) -> None:
    """Drive ``svc`` with ``seg`` primed; verify every audio chunk is a
    multiple of ``bytes_per_sample * num_channels`` and the cache replays
    every byte of the primed PCM (no truncation, no padding)."""
    sample_frame_bytes = 2 * seg.num_channels  # pcm_s16le

    async def body():
        svc.prime_cached(seg)
        return await _collect(svc.run_tts("x", "ctx-chunk"))

    frames = _run(body())
    audio_frames = [f for f in frames if isinstance(f, TTSAudioRawFrame)]

    if seg.pcm:
        assert audio_frames, "non-empty PCM must produce at least one audio frame"

    total = 0
    for f in audio_frames:
        assert len(f.audio) % sample_frame_bytes == 0, (
            f"chunk size {len(f.audio)} not aligned to sample_frame="
            f"{sample_frame_bytes} (sr={seg.sample_rate}, nch={seg.num_channels})"
        )
        total += len(f.audio)
    assert total == len(seg.pcm), (
        f"cache must replay every byte of primed PCM: replayed={total} "
        f"input_len={len(seg.pcm)}"
    )


def test_chunking_aligns_mono_24khz():
    """Default narration shape: mono pcm_s16le @ 24 kHz. sample_frame=2,
    200 ms of audio → multiple 20 ms chunks of 960 bytes each."""
    _assert_aligned_replay(
        _make_service(),
        _make_segment(num_samples=int(24000 * 0.2)),
    )


def test_chunking_aligns_stereo_24khz():
    """Stereo: sample_frame=4 (2 ch × 2 bytes). 20 ms chunk would be 24000×4×
    0.020 = 1920 bytes; alignment-to-4 leaves it at 1920 (already aligned)."""
    _assert_aligned_replay(
        _make_service(),
        _make_segment(num_samples=int(24000 * 0.2), num_channels=2),
    )


def test_chunking_aligns_mono_44100():
    """Cartesia also supports 44.1 kHz. 20 ms = 44100×2×0.020 = 1764 bytes
    which is even ⇒ aligned-to-2 stays at 1764."""
    _assert_aligned_replay(
        _make_service(),
        _make_segment(num_samples=int(44100 * 0.2), sample_rate=44100),
    )


def test_chunking_aligns_mono_16000():
    """16 kHz (narrow-band). 20 ms = 16000×2×0.020 = 640 bytes."""
    _assert_aligned_replay(
        _make_service(),
        _make_segment(num_samples=int(16000 * 0.2), sample_rate=16000),
    )


def test_chunking_handles_non_chunk_multiple_pcm():
    """PCM length not a multiple of one chunk_bytes — the final slice is
    shorter than the standard chunk but still sample-aligned.

    720 samples mono @ 24 kHz = 1440 bytes. chunk_bytes = 960. Iterating
    range(0, 1440, 960) yields offsets [0, 960]; the second slice covers
    bytes [960:1440] = 480 bytes (also a multiple of sample_frame=2)."""
    _assert_aligned_replay(
        _make_service(),
        _make_segment(num_samples=720),
    )


# ──────────────────────────────────────────────────────────────────────
# Edge case — empty PCM still emits the envelope
# ──────────────────────────────────────────────────────────────────────


def test_empty_pcm_emits_envelope_no_audio():
    """Zero-byte segment still emits TTSStartedFrame + TTSStoppedFrame so the
    gate's future resolves and the next segment can run. Otherwise a
    corrupt/empty cache entry (e.g. a 0-byte response from CDN that
    the prefetch didn't catch) would stall the per-segment narration
    loop indefinitely."""

    async def body():
        svc = _make_service()
        svc.prime_cached(CachedSegment(pcm=b"", sample_rate=24000, num_channels=1))
        return await _collect(svc.run_tts("silent", "ctx-empty"))

    frames = _run(body())

    assert len(frames) == 2
    assert isinstance(frames[0], TTSStartedFrame)
    assert isinstance(frames[1], TTSStoppedFrame)
    assert frames[0].context_id == "ctx-empty"
    assert frames[1].context_id == "ctx-empty"


# ──────────────────────────────────────────────────────────────────────
# Block 15 — TTSStoppedFrame defer-by-playback-duration
# ──────────────────────────────────────────────────────────────────────


def test_hit_defers_tts_stopped_by_playback_duration():
    """Block 15 regression guard: ``run_tts`` on a cache hit must NOT
    return before the PCM's playback duration has elapsed.

    Without this sleep, ``TTSStoppedFrame`` would fire microseconds
    after the bytes-handoff. The transport (``BaseTransportOutput``)
    routes ``TTSStoppedFrame`` → ``BotStoppedSpeakingFrame``
    synchronously, ``NarrationCompletionGate`` releases its pending
    future, ``run_scene_narration`` returns, the agent emits
    ``script_complete``, and the shell auto-advances mid-narration.
    This test pins the wall-clock floor so a future "optimization"
    that removes the sleep is caught locally instead of producing
    severe audio clipping in a live room.

    Uses an exaggeratedly-short segment (10 ms) so the test is fast.
    No upper bound asserted — asyncio.sleep granularity + padding make
    it flaky at this scale; the relevant invariant is the lower bound.
    """
    import time

    sample_rate = 24000
    num_samples = int(sample_rate * 0.010)  # 10 ms
    pcm_duration_s = num_samples / sample_rate

    async def body():
        svc = _make_service()
        svc.prime_cached(
            _make_segment(num_samples=num_samples, sample_rate=sample_rate)
        )
        t0 = time.monotonic()
        frames = await _collect(svc.run_tts("hello", "ctx"))
        return frames, time.monotonic() - t0

    frames, elapsed = _run(body())

    # Envelope intact (regression guard against accidentally yielding
    # TTSStoppedFrame before the audio frames).
    assert isinstance(frames[0], TTSStartedFrame)
    assert isinstance(frames[-1], TTSStoppedFrame)
    assert any(isinstance(f, TTSAudioRawFrame) for f in frames)

    assert elapsed >= pcm_duration_s, (
        f"run_tts returned in {elapsed * 1000:.1f} ms but PCM is "
        f"{pcm_duration_s * 1000:.1f} ms long — TTSStoppedFrame fired "
        f"before audio playback duration elapsed. The Block 15 sleep "
        f"in CachedFirstTTSService.run_tts is missing or wrong."
    )


# ──────────────────────────────────────────────────────────────────────
# BotStoppedSpeakingFrame — integration scope, intentionally skipped
# ──────────────────────────────────────────────────────────────────────
#
# ``BotStoppedSpeakingFrame`` is emitted by ``BaseTransportOutput`` when
# its audio buffer drains after a ``TTSStoppedFrame`` — NOT by the TTS
# service. Verifying it fires after cached playback requires standing up
# a real (or mock-with-clock) output transport with its audio scheduler,
# which is integration-test scope.
#
# The relevant invariant for Block 12 is "the cache emits the SAME frame
# types ``CartesiaTTSService`` would, in the same order, with the same
# context_id" — covered by ``test_hit_emits_started_audio_stopped_envelope``
# and ``test_hit_never_emits_bare_output_audio_raw_frame`` above. If the
# transport drives the live path correctly today (it does), it will
# drive the cached path correctly because the frame stream is byte- and
# type-identical.
