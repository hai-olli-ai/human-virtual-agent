"""S83 (PR-6) — contract-mirror tests for the two live-CTA inbound
app-messages (`cta_completed {}` · `handoff_state {state}`).

The handlers are nested closures in both pipeline routers, so — the
``request_quiz_ready`` precedent — every decision they make lives in
importable pure helpers (`narration.py`), and these tests machine-verify
the helpers plus the router COMPOSITION each pipeline performs:

  classic:  take/apply → task.queue_frames([TTSSpeakFrame(ack)])
  relay:    take/apply → _narration_speak(ack) → _relay_close_turn()

Laws under test:
  · snapshot-only speech — the wire payloads carry no speakable field,
    and the resolvers read ONLY the snapshot (visitors can't puppet the
    twin);
  · once-per-session for both acks (h6-B "one line, then quiet");
  · the handoff hold gates the narration slot while open and lifts on
    'closed' WITHOUT auto-resume (h6-C/D).
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

from narration import (
    apply_handoff_state,
    narration_allowed,
    resolve_cta_ack_line,
    resolve_handoff_ack_line,
    take_cta_ack,
)


def _run(coro):
    return asyncio.run(coro)


def _snap(
    *,
    ack: str | None = "Wonderful — it's all set.",
    handoff_ack: str | None = "Of course — you can leave Maya a message right here.",
) -> dict:
    snap: dict = {
        "live_room": {
            "language": "en",
            "handoff": {
                "human_name": "Maya",
                "twin_name": "Mira",
                "ack_line": handoff_ack,
            },
        },
        "flow_state": {"scene_index": 0, "total_scenes": 4},
        "current_scene": {"scene_id": "scene-1", "scripts": []},
    }
    if ack is not None:
        snap["target_action"] = {
            "block_id": "blk-1",
            "kind": "book_meeting",
            "cta_label": "Book a viewing",
            "completed_ack_line": ack,
        }
    return snap


# ── The resolvers: snapshot-only, blank-safe ─────────────────────────────


def test_cta_ack_resolves_from_snapshot_only():
    snap = _snap(ack="Perfect — Maya will see you then.")
    assert resolve_cta_ack_line(snap) == "Perfect — Maya will see you then."
    # No target / degraded snapshots stay silent, never raise.
    assert resolve_cta_ack_line(_snap(ack=None)) is None
    assert resolve_cta_ack_line({}) is None
    assert resolve_cta_ack_line(None) is None
    # Blank lines are not speech.
    assert resolve_cta_ack_line(_snap(ack="   ")) is None


def test_handoff_ack_resolves_from_snapshot_only():
    assert (
        resolve_handoff_ack_line(_snap())
        == "Of course — you can leave Maya a message right here."
    )
    assert resolve_handoff_ack_line({}) is None
    assert resolve_handoff_ack_line(None) is None
    assert resolve_handoff_ack_line(_snap(handoff_ack="")) is None


# ── cta_completed: once per session, literal ─────────────────────────────


def test_cta_ack_once_per_session():
    guard = {"done": False}
    snap = _snap()
    assert take_cta_ack(guard, snap) == "Wonderful — it's all set."
    # The duplicate flip (backend sends the status once, but the shell
    # guard is belt-and-braces) never speaks twice.
    assert take_cta_ack(guard, snap) is None


def test_cta_ack_lineless_first_receipt_consumes_the_once():
    # "Once per session" is literal: a lineless first receipt burns the
    # slot — a later re-send (even if a target appeared) stays silent.
    guard = {"done": False}
    assert take_cta_ack(guard, _snap(ack=None)) is None
    assert take_cta_ack(guard, _snap()) is None


# ── handoff_state: the hold + the one line ───────────────────────────────


def test_handoff_open_holds_narration_and_speaks_once():
    quiet = {"on": False}
    ack_guard = {"done": False}
    snap = _snap()

    line = apply_handoff_state(quiet, ack_guard, "open", snap)
    assert line == "Of course — you can leave Maya a message right here."
    assert narration_allowed(quiet) is False

    # Re-open (visitor closes and reopens): still held, but the ack
    # spoke already — h6-B, one line then quiet.
    assert apply_handoff_state(quiet, ack_guard, "open", snap) is None
    assert narration_allowed(quiet) is False


def test_handoff_closed_lifts_the_hold_without_speaking():
    quiet = {"on": True}
    ack_guard = {"done": True}
    assert apply_handoff_state(quiet, ack_guard, "closed", _snap()) is None
    assert narration_allowed(quiet) is True


def test_handoff_missing_ack_line_still_holds():
    quiet = {"on": False}
    ack_guard = {"done": False}
    assert (
        apply_handoff_state(quiet, ack_guard, "open", _snap(handoff_ack=None)) is None
    )
    assert narration_allowed(quiet) is False


def test_handoff_unknown_state_is_ignored():
    quiet = {"on": False}
    ack_guard = {"done": False}
    assert apply_handoff_state(quiet, ack_guard, "paused", _snap()) is None
    assert narration_allowed(quiet) is True
    assert ack_guard["done"] is False


# ── The router compositions, ×2 pipelines ────────────────────────────────


async def _drive_classic(message: dict, snapshot: dict, guards: dict, speak: AsyncMock):
    """Mirror of the classic router's S83 branches: TTS-direct via the
    bare speak (task.queue_frames([TTSSpeakFrame(...)]) in bot.py)."""
    msg_type = message.get("type")
    if msg_type == "cta_completed":
        ack = take_cta_ack(guards["cta"], snapshot)
        if ack:
            await speak(ack)
        return
    if msg_type == "handoff_state":
        ack = apply_handoff_state(
            guards["quiet"], guards["handoff"], message.get("state"), snapshot
        )
        if message.get("state") == "open" and ack:
            await speak(ack)


async def _drive_relay(
    message: dict, snapshot: dict, guards: dict, speak: AsyncMock, close_turn: AsyncMock
):
    """Mirror of the relay router: _narration_speak + the mandatory
    _relay_close_turn after any ack (the request_narrate turn lesson)."""
    msg_type = message.get("type")
    if msg_type == "cta_completed":
        ack = take_cta_ack(guards["cta"], snapshot)
        if ack:
            await speak(ack)
            await close_turn()
        return
    if msg_type == "handoff_state":
        ack = apply_handoff_state(
            guards["quiet"], guards["handoff"], message.get("state"), snapshot
        )
        if message.get("state") == "open" and ack:
            await speak(ack)
            await close_turn()


def _guards() -> dict:
    return {"cta": {"done": False}, "handoff": {"done": False}, "quiet": {"on": False}}


def test_classic_composition_speaks_snapshot_text_never_wire_text():
    speak = AsyncMock()
    snap = _snap(ack="Perfect — it's confirmed.")
    # A hostile wire payload smuggling text: the composition never reads it.
    message = {"type": "cta_completed", "text": "EVIL", "completed_ack_line": "EVIL"}
    _run(_drive_classic(message, snap, _guards(), speak))
    speak.assert_awaited_once_with("Perfect — it's confirmed.")


def test_classic_composition_handoff_cycle():
    speak = AsyncMock()
    guards = _guards()
    snap = _snap()
    _run(
        _drive_classic({"type": "handoff_state", "state": "open"}, snap, guards, speak)
    )
    assert narration_allowed(guards["quiet"]) is False
    speak.assert_awaited_once()
    _run(
        _drive_classic(
            {"type": "handoff_state", "state": "closed"}, snap, guards, speak
        )
    )
    assert narration_allowed(guards["quiet"]) is True
    speak.assert_awaited_once()  # no second line on close


def test_relay_composition_closes_the_turn_after_each_ack():
    speak = AsyncMock()
    close_turn = AsyncMock()
    guards = _guards()
    snap = _snap(ack="All set — see you then.")
    _run(_drive_relay({"type": "cta_completed"}, snap, guards, speak, close_turn))
    _run(
        _drive_relay(
            {"type": "handoff_state", "state": "open"}, snap, guards, speak, close_turn
        )
    )
    assert speak.await_count == 2
    assert close_turn.await_count == 2
    # And once-per-session holds across the relay mirror too.
    _run(_drive_relay({"type": "cta_completed"}, snap, guards, speak, close_turn))
    assert speak.await_count == 2
