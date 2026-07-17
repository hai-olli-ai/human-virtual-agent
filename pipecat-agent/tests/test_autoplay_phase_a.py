"""Auto Play Phase A — truthful playout completion + interruption awareness.

Layers under test:

  * :class:`narration.NarrationCompletionGate` — the Phase A extensions
    (playout-drain futures, interruption sentinel + latch, ``begin_run``
    hygiene, ``expect_interruption`` flush waiters, bot-speaking mirror).
    Phase A made the gate load-bearing (it now decides WHEN
    ``script_complete`` may be emitted), so unlike the S65 suites it is
    unit-tested directly here. ``push_frame`` is mocked out — the gate's
    pass-through behavior isn't under test, its bookkeeping is.
  * :func:`narration.compute_playout_drain_timeout` — pure budget logic.
  * :func:`narration.run_scene_narration` — the ``wait_playout`` hook
    (called last, only when something was spoken, with the computed
    budget; ``NarrationInterrupted`` propagation).
  * :class:`narration.SceneNarrator` — abort-on-interrupt + the
    reset-to-primary-voice guarantee on the abort path.
  * Composition mirrors of bot.py's ``_narrate_and_complete`` /
    ``_start_narration_task`` (the handlers are closures inside
    ``run_bot_*`` and can't be imported — same convention as
    ``test_request_narrate.py``): emission suppressed on interruption
    and on cancellation (stop / supersede), exactly one emission per
    surviving run, ``trigger`` passthrough for autoplay resume.

Frozen wire contract v1 rules pinned here:

  2a. ``script_complete`` is emitted ONLY after true playout drain
      (``BotStoppedSpeakingFrame`` after the final ``TTSStoppedFrame``) —
      the scope deliberately skipped at the bottom of
      ``test_cached_first_tts.py`` pre-Phase-A.
  2b. Cancelled or interrupted runs NEVER emit.
  2c. A resume-initiated run emits ``trigger:'auto'``.
   4. Script-less scenes still emit ``hadScript:false`` immediately (no
      drain wait).

Follows the existing tests/ convention: no pytest-asyncio, so each async
test goes through ``asyncio.run``.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    InterruptionFrame,
    TTSAudioRawFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection

from narration import (
    NARRATION_INTERRUPTED,
    NarrationCompletionGate,
    NarrationInterrupted,
    PLAYOUT_DRAIN_FALLBACK_S,
    PLAYOUT_DRAIN_MARGIN_S,
    SceneNarrator,
    build_script_complete_payload,
    compute_playout_drain_timeout,
    run_scene_narration,
)


def _run(coro):
    return asyncio.run(coro)


def _snap(
    *,
    scene_id: str | None = "scene-1",
    scripts: list | None = None,
    narration: dict | None = None,
    auto_advance: bool = False,
    scene_index: int = 0,
    total_scenes: int = 1,
) -> dict:
    return {
        "live_room": {"auto_advance": auto_advance, "language": "en"},
        "flow_state": {"scene_index": scene_index, "total_scenes": total_scenes},
        "current_scene": {
            "scene_id": scene_id,
            "scripts": scripts if scripts is not None else [],
            "narration": narration
            if narration is not None
            else {"invitation_line": "Any questions?", "transition_cue": "Onward."},
        },
    }


def _make_gate() -> NarrationCompletionGate:
    gate = NarrationCompletionGate()
    # The gate's pass-through push isn't under test; mocking it also
    # keeps the base FrameProcessor from needing a linked pipeline. The
    # base's _start_interruption (task-manager bookkeeping on
    # InterruptionFrame) is likewise stubbed — it needs a running
    # pipeline and isn't what these tests exercise.
    gate.push_frame = AsyncMock()
    gate._start_interruption = AsyncMock()
    return gate


async def _feed(gate, frame, direction=FrameDirection.DOWNSTREAM):
    await gate.process_frame(frame, direction)


async def _feed_synthesized_utterance(gate, ctx="ctx"):
    """One utterance's downstream synthesis frames as the gate sees them:
    audio chunk(s), then TTSStoppedFrame (synthesis-complete)."""
    await _feed(gate, TTSAudioRawFrame(audio=b"\x00\x00", sample_rate=24000,
                                       num_channels=1, context_id=ctx))
    await _feed(gate, TTSStoppedFrame(context_id=ctx))


async def _feed_playout_boundary(gate):
    """One utterance's playout boundary as the transport broadcasts it:
    BotStartedSpeaking … BotStoppedSpeaking (upstream siblings)."""
    await _feed(gate, BotStartedSpeakingFrame(), FrameDirection.UPSTREAM)
    await _feed(gate, BotStoppedSpeakingFrame(), FrameDirection.UPSTREAM)


# ──────────────────────────────────────────────────────────────────────
# NarrationCompletionGate — playout drain (A1)
# ──────────────────────────────────────────────────────────────────────


def test_gate_stop_future_still_resolves_fifo_on_tts_stopped():
    """Baseline guard: the pre-Phase-A per-segment contract is intact."""

    async def body():
        gate = _make_gate()
        first = gate.expect_next_stop()
        second = gate.expect_next_stop()
        await _feed(gate, TTSStoppedFrame(context_id="ctx-1"))
        assert first.done() and first.result() == "ctx-1"
        assert not second.done()
        await _feed(gate, TTSStoppedFrame(context_id="ctx-2"))
        assert second.done() and second.result() == "ctx-2"

    _run(body())


def test_gate_drain_pends_while_speaking_and_resolves_on_bot_stopped():
    """A1 core: the drain future ignores TTSStoppedFrame (synthesis
    complete) and resolves only on BotStoppedSpeakingFrame (true
    transport queue drain, broadcast upstream through the gate)."""

    async def body():
        gate = _make_gate()
        gate.begin_run()
        await _feed(gate, BotStartedSpeakingFrame(), FrameDirection.UPSTREAM)
        assert gate.bot_is_speaking is True

        drain = gate.expect_playout_drain()
        assert not drain.done()

        # Synthesis-complete does NOT release the drain — this is the
        # exact early-firing Bug 1 was about.
        await _feed_synthesized_utterance(gate)
        assert not drain.done()

        await _feed(gate, BotStoppedSpeakingFrame(), FrameDirection.UPSTREAM)
        assert drain.done() and drain.result() is None
        assert gate.bot_is_speaking is False

    _run(body())


def test_gate_drain_waits_for_every_synthesized_utterance():
    """The critical multi-utterance case: the transport emits
    BotStoppedSpeakingFrame at EVERY utterance boundary (MediaSender
    fires _bot_stopped_speaking per dequeued TTSStoppedFrame), and
    per-segment gating releases at synthesis-complete, so segments
    2..N + the followup are still queued when segment 1's boundary
    lands. Resolving the drain on the FIRST boundary would emit
    script_complete seconds early — the gate must count synthesized
    vs played utterances and release only when they match."""

    async def body():
        gate = _make_gate()
        gate.begin_run()

        # Three utterances synthesized back-to-back (fast synthesis)
        # while playout of the first is still in progress.
        await _feed(gate, BotStartedSpeakingFrame(), FrameDirection.UPSTREAM)
        for i in range(3):
            await _feed_synthesized_utterance(gate, ctx=f"ctx-{i}")

        drain = gate.expect_playout_drain()
        assert not drain.done()

        # Boundary 1: segment 1 finished playing; 2 + followup queued.
        await _feed(gate, BotStoppedSpeakingFrame(), FrameDirection.UPSTREAM)
        assert not drain.done(), (
            "drain resolved on the FIRST utterance boundary — "
            "script_complete would fire while later segments still play"
        )
        await _feed(gate, BotStartedSpeakingFrame(), FrameDirection.UPSTREAM)

        # Boundary 2.
        await _feed(gate, BotStoppedSpeakingFrame(), FrameDirection.UPSTREAM)
        assert not drain.done()
        await _feed(gate, BotStartedSpeakingFrame(), FrameDirection.UPSTREAM)

        # Boundary 3 — the LAST synthesized utterance drained.
        await _feed(gate, BotStoppedSpeakingFrame(), FrameDirection.UPSTREAM)
        assert drain.done() and drain.result() is None

    _run(body())


def test_gate_drain_registration_in_inter_utterance_gap_still_pends():
    """Registering the drain in the microsecond gap BETWEEN utterance
    boundaries (bot momentarily quiet, later utterances still queued)
    must pend — 'not speaking' alone is not 'drained'."""

    async def body():
        gate = _make_gate()
        gate.begin_run()
        await _feed(gate, BotStartedSpeakingFrame(), FrameDirection.UPSTREAM)
        await _feed_synthesized_utterance(gate, ctx="a")
        await _feed_synthesized_utterance(gate, ctx="b")

        # Utterance A's boundary passed; bot momentarily quiet.
        await _feed(gate, BotStoppedSpeakingFrame(), FrameDirection.UPSTREAM)
        assert gate.bot_is_speaking is False

        drain = gate.expect_playout_drain()
        assert not drain.done(), (
            "drain resolved in the inter-utterance gap — utterance B "
            "is still queued in the transport"
        )

        await _feed_playout_boundary(gate)  # utterance B plays out
        assert drain.done() and drain.result() is None

    _run(body())


def test_gate_post_flush_stray_bot_stopped_does_not_skew_next_run():
    """MediaSender.handle_interruptions pushes one extra
    BotStoppedSpeakingFrame after a flush; it reaches the gate when the
    gate already marked itself quiet (InterruptionFrame branch). It must
    NOT count as a played utterance for the next run, or that run's
    drain would release one utterance early."""

    async def body():
        gate = _make_gate()
        gate.begin_run()
        await _feed(gate, BotStartedSpeakingFrame(), FrameDirection.UPSTREAM)
        await _feed(gate, InterruptionFrame())
        # The flush's trailing BotStoppedSpeakingFrame (bot already quiet).
        await _feed(gate, BotStoppedSpeakingFrame(), FrameDirection.UPSTREAM)

        gate.begin_run()  # next run
        await _feed(gate, BotStartedSpeakingFrame(), FrameDirection.UPSTREAM)
        await _feed_synthesized_utterance(gate, ctx="a")
        await _feed_synthesized_utterance(gate, ctx="b")
        drain = gate.expect_playout_drain()

        await _feed(gate, BotStoppedSpeakingFrame(), FrameDirection.UPSTREAM)
        assert not drain.done(), "stray post-flush boundary skewed the counters"
        await _feed_playout_boundary(gate)
        assert drain.done()

    _run(body())


def test_gate_drain_resolves_immediately_when_bot_not_speaking():
    """Covers the cached-playback race (drain landed before registration)
    and the no-audio edge — the caller must not stall on either."""

    async def body():
        gate = _make_gate()
        gate.begin_run()
        drain = gate.expect_playout_drain()
        assert drain.done() and drain.result() is None

        # Same after a fully-drained utterance cycle.
        await _feed(gate, BotStartedSpeakingFrame(), FrameDirection.UPSTREAM)
        await _feed_synthesized_utterance(gate)
        await _feed(gate, BotStoppedSpeakingFrame(), FrameDirection.UPSTREAM)
        drain2 = gate.expect_playout_drain()
        assert drain2.done() and drain2.result() is None

    _run(body())


# ──────────────────────────────────────────────────────────────────────
# NarrationCompletionGate — interruption awareness (A2)
# ──────────────────────────────────────────────────────────────────────


def test_gate_interruption_resolves_stop_and_drain_with_sentinel():
    async def body():
        gate = _make_gate()
        await _feed(gate, BotStartedSpeakingFrame(), FrameDirection.UPSTREAM)
        stop = gate.expect_next_stop()
        drain = gate.expect_playout_drain()

        await _feed(gate, InterruptionFrame())

        assert stop.done() and stop.result() is NARRATION_INTERRUPTED
        assert drain.done() and drain.result() is NARRATION_INTERRUPTED
        assert gate.bot_is_speaking is False

    _run(body())


def test_gate_interruption_latch_covers_between_segment_window():
    """An InterruptionFrame landing while NO future is registered (the
    between-segments window) must still kill the run's next expect call
    — otherwise narration would resume mid-scene over the visitor's
    conversation. begin_run clears the latch for the next run."""

    async def body():
        gate = _make_gate()
        await _feed(gate, InterruptionFrame())

        stop = gate.expect_next_stop()
        assert stop.done() and stop.result() is NARRATION_INTERRUPTED
        drain = gate.expect_playout_drain()
        assert drain.done() and drain.result() is NARRATION_INTERRUPTED

        gate.begin_run()
        fresh = gate.expect_next_stop()
        assert not fresh.done()

    _run(body())


def test_gate_expect_interruption_ignores_latch_resolves_on_next_frame():
    """The flush waiter must observe ITS OWN InterruptionFrame — a stale
    latch from an earlier barge-in must not fake-confirm the flush (the
    next narration run's first segment would race the real in-flight
    interruption and be killed by it)."""

    async def body():
        gate = _make_gate()
        await _feed(gate, InterruptionFrame())  # stale latch

        waiter = gate.expect_interruption()
        assert not waiter.done()

        await _feed(gate, InterruptionFrame())
        assert waiter.done()

    _run(body())


def test_gate_begin_run_drops_stale_futures_preserving_fifo():
    """A stale (e.g. timed-out) future left at the head of the FIFO must
    not consume the new run's first TTSStoppedFrame."""

    async def body():
        gate = _make_gate()
        stale = gate.expect_next_stop()

        gate.begin_run()
        assert stale.cancelled()

        fresh = gate.expect_next_stop()
        await _feed(gate, TTSStoppedFrame(context_id="ctx-new"))
        assert fresh.done() and fresh.result() == "ctx-new"

    _run(body())


def test_gate_cancel_all_covers_all_three_registries():
    async def body():
        gate = _make_gate()
        await _feed(gate, BotStartedSpeakingFrame(), FrameDirection.UPSTREAM)
        stop = gate.expect_next_stop()
        drain = gate.expect_playout_drain()
        waiter = gate.expect_interruption()
        gate.cancel_all("session_end")
        assert stop.cancelled() and drain.cancelled() and waiter.cancelled()

    _run(body())


# ──────────────────────────────────────────────────────────────────────
# compute_playout_drain_timeout — budget logic (A1)
# ──────────────────────────────────────────────────────────────────────


def test_drain_timeout_sums_known_durations_plus_margin():
    snap = _snap(scripts=[
        {"text": "a", "audio": {"url": "u1", "duration_ms": 1500}},
        {"text": "b", "audio": {"url": "u2", "duration_ms": 2500}},
    ])
    assert compute_playout_drain_timeout(snap) == 4.0 + PLAYOUT_DRAIN_MARGIN_S


def test_drain_timeout_unknown_duration_falls_back():
    """Missing audio block, missing duration, zero (the backend dedup
    edge — a legit 0 means UNKNOWN, not instant), and junk values all
    force the fixed cap."""
    for bad_audio in (None, {}, {"duration_ms": 0}, {"duration_ms": None},
                      {"duration_ms": "junk"}):
        scripts = [
            {"text": "known", "audio": {"url": "u", "duration_ms": 1000}},
            {"text": "unknown"},
        ]
        if bad_audio is not None:
            scripts[1]["audio"] = bad_audio
        snap = _snap(scripts=scripts)
        assert compute_playout_drain_timeout(snap) == PLAYOUT_DRAIN_FALLBACK_S, (
            f"audio={bad_audio!r} should be treated as unknown"
        )


def test_drain_timeout_blank_segments_neither_add_nor_force_fallback():
    """Blank segments are never narrated — they must not drag the total
    to the fallback just because they carry no audio."""
    snap = _snap(scripts=[
        {"text": "  ", "order": 0},  # blank, no audio — skipped
        {"text": "spoken", "audio": {"url": "u", "duration_ms": 3000}},
    ])
    assert compute_playout_drain_timeout(snap) == 3.0 + PLAYOUT_DRAIN_MARGIN_S


def test_drain_timeout_no_narratable_segments_falls_back():
    assert compute_playout_drain_timeout(None) == PLAYOUT_DRAIN_FALLBACK_S
    assert compute_playout_drain_timeout({}) == PLAYOUT_DRAIN_FALLBACK_S
    assert (
        compute_playout_drain_timeout(_snap(scripts=[]))
        == PLAYOUT_DRAIN_FALLBACK_S
    )


# ──────────────────────────────────────────────────────────────────────
# run_scene_narration — the wait_playout hook (A1)
# ──────────────────────────────────────────────────────────────────────


def _make_event_narrator(events: list, *, speak_side_effect=None):
    set_voice = AsyncMock(side_effect=lambda v: events.append(("set_voice", v)))

    async def _speak(text: str) -> None:
        if speak_side_effect is not None:
            speak_side_effect(text)
        events.append(("speak", text))

    narrator = SceneNarrator(
        primary_voice_id="primary",
        set_voice=set_voice,
        speak=_speak,
    )
    return narrator, set_voice, _speak


def test_wait_playout_called_last_with_computed_budget():
    events: list[tuple] = []
    narrator, _, speak = _make_event_narrator(events)

    async def wait_playout(timeout_s: float) -> None:
        events.append(("drain", timeout_s))

    snap = _snap(
        scripts=[{"text": "hello", "audio": {"url": "u", "duration_ms": 2000}}],
        auto_advance=False,
    )
    spoke = _run(run_scene_narration(
        snap, narrator=narrator, speak_followup=speak, wait_playout=wait_playout,
    ))
    assert spoke is True
    # Order: script speak → invitation speak → drain wait LAST, with the
    # budget computed from the snapshot's cached durations.
    assert events[-1] == ("drain", 2.0 + PLAYOUT_DRAIN_MARGIN_S)
    speak_indices = [i for i, e in enumerate(events) if e[0] == "speak"]
    assert all(i < len(events) - 1 for i in speak_indices)


def test_wait_playout_skipped_when_nothing_spoken():
    """Wire rule 4: hadScript:false scenes must not pay a drain wait —
    their script_complete stays immediate."""
    events: list[tuple] = []
    narrator, _, speak = _make_event_narrator(events)
    drain = AsyncMock()

    spoke = _run(run_scene_narration(
        _snap(scripts=[]), narrator=narrator, speak_followup=speak,
        wait_playout=drain,
    ))
    assert spoke is False
    drain.assert_not_awaited()


# ──────────────────────────────────────────────────────────────────────
# Interruption abort (A2) — loop abort, voice reset, no emit, no stall
# ──────────────────────────────────────────────────────────────────────


async def _drive_run(
    snapshot,
    narrator,
    speak,
    send,
    *,
    wait_playout=None,
    force=False,
    trigger="auto",
):
    """Mirror bot.py's ``_narrate_and_complete`` emission rules: emit
    script_complete after the run UNLESS it was interrupted (returns
    None) — cancellation propagates to the caller like the real task."""
    try:
        spoke = await run_scene_narration(
            snapshot,
            narrator=narrator,
            speak_followup=speak,
            force=force,
            wait_playout=wait_playout,
        )
    except NarrationInterrupted:
        return None
    await send(build_script_complete_payload(
        snapshot, spoke_script=spoke, trigger=trigger
    ))
    return spoke


def test_interrupted_segment_aborts_loop_resets_voice_no_emit_no_stall():
    """A2: an interrupted speak must abort the remaining segments AND the
    followup, reset the voice to primary (the LLM's reply to the
    barge-in must not render in the script avatar's clone), suppress
    script_complete, and return promptly (no 30 s orphaned-future
    stall)."""
    events: list[tuple] = []

    calls = {"n": 0}

    def _boom_on_second(text: str) -> None:
        calls["n"] += 1
        if calls["n"] == 2:
            raise NarrationInterrupted()

    narrator, set_voice, speak = _make_event_narrator(
        events, speak_side_effect=_boom_on_second
    )
    send = AsyncMock(side_effect=lambda m: events.append(("send", m)))

    snap = _snap(scripts=[
        {"text": "one", "voice_id": "clone-A"},
        {"text": "two", "voice_id": "clone-A"},
        {"text": "three", "voice_id": "clone-A"},
    ])

    t0 = time.monotonic()
    result = _run(_drive_run(snap, narrator, speak, send))
    elapsed = time.monotonic() - t0

    assert result is None  # interrupted → suppressed
    send.assert_not_awaited()
    spoken = [t for k, t in events if k == "speak"]
    assert spoken == ["one"]  # segment 2 raised before appending; 3 never ran
    assert "Any questions?" not in spoken  # followup suppressed too
    # Voice was switched to the clone, then reset to primary on abort.
    assert set_voice.await_args_list[-1].args == ("primary",)
    assert narrator.current_voice == "primary"
    assert elapsed < 2.0, f"interrupted run stalled for {elapsed:.1f}s"


def test_cancelled_run_still_resets_voice_to_primary():
    """Review fix: a CANCELLED run (autoplay stop / supersede /
    disconnect) needs the same voice reset as an interrupted one —
    stop has no follow-up narration to realign the voice, so without
    the reset every subsequent LLM reply renders in the script
    avatar's clone indefinitely."""
    events: list[tuple] = []

    calls = {"n": 0}

    def _cancel_on_second(text: str) -> None:
        calls["n"] += 1
        if calls["n"] == 2:
            raise asyncio.CancelledError()

    narrator, set_voice, speak = _make_event_narrator(
        events, speak_side_effect=_cancel_on_second
    )
    snap = _snap(scripts=[
        {"text": "one", "voice_id": "clone-A"},
        {"text": "two", "voice_id": "clone-A"},
    ])

    async def body():
        try:
            await narrator.narrate(snap)
        except asyncio.CancelledError:
            return True
        return False

    propagated = _run(body())
    assert propagated is True  # cancellation still propagates
    assert set_voice.await_args_list[-1].args == ("primary",)
    assert narrator.current_voice == "primary"


def test_interruption_drops_armed_cache_prime():
    """Review fix: an InterruptionFrame must clear an armed single-shot
    prime, or the LLM's reply to the barge-in consumes the stale prime
    and plays the scene-script PCM instead of the reply.

    The parent chain's InterruptionFrame handling needs a live pipeline
    task manager, which is out of scope here — the override clears the
    prime BEFORE delegating, so the parent's process_frame is stubbed
    and asserted-delegated-to instead.
    """
    from unittest.mock import patch

    from pipecat.services.cartesia.tts import CartesiaTTSService

    from services.cached_first_tts import CachedFirstTTSService, CachedSegment

    async def body():
        svc = CachedFirstTTSService(
            api_key="test-key",
            sample_rate=24000,
            settings=CachedFirstTTSService.Settings(
                voice="test-voice", model="sonic-3"
            ),
        )
        svc.prime_cached(CachedSegment(pcm=b"\x00" * 960, sample_rate=24000))
        assert svc._primed is not None
        frame = InterruptionFrame()
        with patch.object(
            CartesiaTTSService, "process_frame", new=AsyncMock()
        ) as parent_pf:
            await svc.process_frame(frame, FrameDirection.DOWNSTREAM)
            parent_pf.assert_awaited_once_with(frame, FrameDirection.DOWNSTREAM)
        assert svc._primed is None, (
            "stale prime survived the interruption — the next run_tts "
            "(the LLM's reply) would play the scene-script PCM"
        )

    _run(body())


def test_interruption_during_drain_wait_suppresses_emit():
    """Rule 2b applies to the playout tail too: everything was spoken and
    synthesized, but the visitor barged in while queued audio was still
    draining — the run must not emit."""
    events: list[tuple] = []
    narrator, _, speak = _make_event_narrator(events)
    send = AsyncMock()

    async def wait_playout(timeout_s: float) -> None:
        raise NarrationInterrupted()

    snap = _snap(scripts=[{"text": "hello"}])
    result = _run(_drive_run(snap, narrator, speak, send, wait_playout=wait_playout))
    assert result is None
    send.assert_not_awaited()


# ──────────────────────────────────────────────────────────────────────
# End-to-end against the REAL gate — the A1 ordering the pre-Phase-A
# suite deliberately skipped (see the note at the bottom of
# test_cached_first_tts.py): script_complete only after
# BotStoppedSpeakingFrame follows the final TTSStoppedFrame.
# ──────────────────────────────────────────────────────────────────────


def _gate_wired_callables(gate: NarrationCompletionGate, events: list):
    """Speak + wait_playout closures shaped exactly like bot.py's
    ``_classic_speak`` / ``_classic_wait_playout``, driven by frames fed
    into the real gate. Each speak models fast synthesis (audio chunk +
    TTSStoppedFrame reach the gate immediately) while playout lags —
    the per-utterance boundary frames are fed separately by the test."""

    async def speak(text: str) -> None:
        fut = gate.expect_next_stop()
        if fut.done() and fut.result() is NARRATION_INTERRUPTED:
            raise NarrationInterrupted()
        events.append(("speak", text))
        # Simulate the transport: audio starts flowing, then synthesis
        # completes (fast — several× realtime) while playout continues.
        await gate.process_frame(
            BotStartedSpeakingFrame(), FrameDirection.UPSTREAM
        )
        await gate.process_frame(
            TTSAudioRawFrame(audio=b"\x00\x00", sample_rate=24000,
                             num_channels=1, context_id="ctx"),
            FrameDirection.DOWNSTREAM,
        )
        await gate.process_frame(
            TTSStoppedFrame(context_id="ctx"), FrameDirection.DOWNSTREAM
        )
        result = await fut
        if result is NARRATION_INTERRUPTED:
            raise NarrationInterrupted()

    async def wait_playout(timeout_s: float) -> None:
        fut = gate.expect_playout_drain()
        result = await asyncio.wait_for(fut, timeout=timeout_s)
        if result is NARRATION_INTERRUPTED:
            raise NarrationInterrupted()

    return speak, wait_playout


def test_script_complete_waits_for_last_utterance_playout():
    """End-to-end against the REAL gate: 2 script segments + invitation =
    3 synthesized utterances. script_complete must survive BOTH the
    synthesis-complete point (Bug 1's original form) AND the
    intermediate per-utterance boundaries (the transport emits
    BotStoppedSpeakingFrame after EVERY utterance) — only the LAST
    boundary releases it."""

    async def body():
        gate = _make_gate()
        events: list[tuple] = []
        speak, wait_playout = _gate_wired_callables(gate, events)
        send = AsyncMock(side_effect=lambda m: events.append(("send", m)))
        narrator = SceneNarrator(
            primary_voice_id="primary", set_voice=AsyncMock(), speak=speak
        )
        snap = _snap(
            scripts=[{"text": "hello"}, {"text": "world"}], auto_advance=False
        )

        gate.begin_run()
        run = asyncio.create_task(
            _drive_run(snap, narrator, speak, send, wait_playout=wait_playout)
        )
        # Let the run reach the drain wait: every utterance synthesized
        # (2 segments + invitation), playout still in progress.
        for _ in range(50):
            await asyncio.sleep(0)
        assert not any(k == "send" for k, _ in events), (
            "script_complete emitted at synthesis-complete — Bug 1 regressed"
        )
        assert gate.bot_is_speaking is True

        # Intermediate boundaries: utterances 1 and 2 finish playing
        # while the next one starts — still no emission.
        for _ in range(2):
            await gate.process_frame(
                BotStoppedSpeakingFrame(), FrameDirection.UPSTREAM
            )
            for _ in range(10):
                await asyncio.sleep(0)
            assert not any(k == "send" for k, _ in events), (
                "script_complete emitted at an intermediate utterance "
                "boundary — later segments still queued in the transport"
            )
            await gate.process_frame(
                BotStartedSpeakingFrame(), FrameDirection.UPSTREAM
            )

        # Final boundary — the invitation's audio drained.
        await gate.process_frame(
            BotStoppedSpeakingFrame(), FrameDirection.UPSTREAM
        )
        spoke = await asyncio.wait_for(run, timeout=5.0)
        assert spoke is True
        sends = [m for k, m in events if k == "send"]
        assert len(sends) == 1
        assert sends[0]["hadScript"] is True
        assert sends[0]["trigger"] == "auto"
        # And the send happened strictly after all speaks.
        assert events[-1][0] == "send"

    _run(body())


def test_interruption_mid_playout_tail_no_emit_via_real_gate():
    async def body():
        gate = _make_gate()
        events: list[tuple] = []
        speak, wait_playout = _gate_wired_callables(gate, events)
        send = AsyncMock(side_effect=lambda m: events.append(("send", m)))
        narrator = SceneNarrator(
            primary_voice_id="primary", set_voice=AsyncMock(), speak=speak
        )
        snap = _snap(scripts=[{"text": "hello"}], auto_advance=False)

        gate.begin_run()
        run = asyncio.create_task(
            _drive_run(snap, narrator, speak, send, wait_playout=wait_playout)
        )
        for _ in range(50):
            await asyncio.sleep(0)
        assert not any(k == "send" for k, _ in events)

        # Visitor barges in during the tail.
        await gate.process_frame(InterruptionFrame(), FrameDirection.DOWNSTREAM)
        result = await asyncio.wait_for(run, timeout=5.0)
        assert result is None
        send.assert_not_awaited()

    _run(body())


# ──────────────────────────────────────────────────────────────────────
# Single-slot semantics — stop / resume / scene-change supersede (A3/A4)
# Composition mirrors of bot.py's _start_narration_task + autoplay
# branches (closures — not importable; same convention as
# test_request_narrate.py).
# ──────────────────────────────────────────────────────────────────────


class _Slot:
    """Mirror of bot.py's scene_narration_task single-slot semantics."""

    def __init__(self):
        self.task: asyncio.Task | None = None

    def start(self, coro, *, replace: bool = True) -> bool:
        if self.task is not None and not self.task.done():
            if not replace:
                coro.close()
                return False
            self.task.cancel()
        self.task = asyncio.create_task(coro)
        return True

    def cancel(self) -> None:
        if self.task is not None and not self.task.done():
            self.task.cancel()


def test_autoplay_stop_cancels_run_and_suppresses_emit():
    """A4 stop: the active run is cancelled → no script_complete, ever
    (frozen rule: a stopped run never emits)."""

    async def body():
        events: list[tuple] = []
        block = asyncio.Event()

        async def speak(text: str) -> None:
            events.append(("speak", text))
            await block.wait()  # narration audio "playing"

        send = AsyncMock(side_effect=lambda m: events.append(("send", m)))
        narrator = SceneNarrator(
            primary_voice_id="primary", set_voice=AsyncMock(), speak=speak
        )
        snap = _snap(scripts=[{"text": "hello"}])

        slot = _Slot()
        slot.start(_drive_run(snap, narrator, speak, send))
        for _ in range(10):
            await asyncio.sleep(0)
        assert ("speak", "hello") in events  # mid-narration

        slot.cancel()  # autoplay_control stop
        try:
            await slot.task
        except asyncio.CancelledError:
            pass
        send.assert_not_awaited()

    _run(body())


def test_autoplay_resume_force_renarrates_and_emits_trigger_auto():
    """A4 resume: force=True bypasses the once-per-entry guard the
    stopped run already set; exactly one script_complete with
    trigger='auto' (rule 2c — advance-eligible)."""

    async def body():
        events: list[tuple] = []

        async def speak(text: str) -> None:
            events.append(("speak", text))

        send = AsyncMock(side_effect=lambda m: events.append(("send", m)))
        narrator = SceneNarrator(
            primary_voice_id="primary", set_voice=AsyncMock(), speak=speak
        )
        snap = _snap(scripts=[{"text": "hello"}], scene_index=1, total_scenes=3)

        # The stopped run marked the scene before it was cancelled.
        marked = await narrator.narrate(snap)
        assert marked is True
        events.clear()

        slot = _Slot()
        slot.start(_drive_run(snap, narrator, speak, send, force=True))
        spoke = await slot.task
        assert spoke is True  # force bypassed the guard
        sends = [m for k, m in events if k == "send"]
        assert len(sends) == 1
        assert sends[0] == {
            "type": "script_complete",
            "sceneIndex": 1,
            "hadScript": True,
            "trigger": "auto",
        }

    _run(body())


def test_scene_change_supersede_yields_exactly_one_emission():
    """A3: scene change mid-narration cancels the old run (no emission)
    and the new scene's run emits exactly one script_complete."""

    async def body():
        events: list[tuple] = []
        block = asyncio.Event()

        async def slow_speak(text: str) -> None:
            events.append(("speak", text))
            await block.wait()

        async def fast_speak(text: str) -> None:
            events.append(("speak", text))

        send = AsyncMock(side_effect=lambda m: events.append(("send", m)))
        narrator1 = SceneNarrator(
            primary_voice_id="primary", set_voice=AsyncMock(), speak=slow_speak
        )
        narrator2 = SceneNarrator(
            primary_voice_id="primary", set_voice=AsyncMock(), speak=fast_speak
        )
        snap1 = _snap(scene_id="s1", scripts=[{"text": "scene one"}],
                      scene_index=0, total_scenes=2)
        snap2 = _snap(scene_id="s2", scripts=[{"text": "scene two"}],
                      scene_index=1, total_scenes=2)

        slot = _Slot()
        slot.start(_drive_run(snap1, narrator1, slow_speak, send))
        for _ in range(10):
            await asyncio.sleep(0)
        assert ("speak", "scene one") in events

        # Scene change: newest wins.
        slot.start(_drive_run(snap2, narrator2, fast_speak, send))
        spoke2 = await slot.task
        assert spoke2 is True

        sends = [m for k, m in events if k == "send"]
        assert len(sends) == 1
        assert sends[0]["sceneIndex"] == 1

    _run(body())


def test_slot_replace_false_yields_to_active_run():
    """A5: session-start narration must NOT stomp a run that already owns
    the slot (a scene change that raced ahead of on_client_connected)."""

    async def body():
        block = asyncio.Event()
        ran = {"second": False}

        async def first():
            await block.wait()

        async def second():
            ran["second"] = True

        slot = _Slot()
        assert slot.start(first()) is True
        assert slot.start(second(), replace=False) is False
        block.set()
        await slot.task
        assert ran["second"] is False

    _run(body())
