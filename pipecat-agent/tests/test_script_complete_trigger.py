"""Tests for the ``trigger`` field on ``script_complete`` (S65c Block 4).

Three concerns:

  * Auto path: ``trigger`` defaults to ``"auto"`` and ALWAYS appears on
    the wire — every ``script_complete`` is self-describing. The S65
    payload-shape tests in ``tests/test_scene_narration.py`` were
    updated in the same change to include ``"trigger": "auto"`` in
    their strict-equality assertions.
  * Manual replay surfaces the explicit value: passing
    ``trigger="manual"`` produces ``"trigger": "manual"`` on the wire so
    the shell can short-circuit auto-advance on visitor-initiated
    replays. Future trigger values pass through verbatim.
  * ``SceneNarrator.narrate(..., force=True)`` bypasses the
    once-per-entry guard so the Script-button can replay a scene whose
    ``scene_id`` was already marked by the entry narration.

Follows the project test convention (no pytest-asyncio — wrap async
work in ``asyncio.run`` via the ``_run`` helper).
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

from narration import (
    SceneNarrator,
    build_script_complete_payload,
)


def _run(coro):
    return asyncio.run(coro)


def _snap(*, scene_id: str = "scene-1", scripts: list | None = None) -> dict:
    return {
        "live_room": {"auto_advance": False, "language": "en"},
        "flow_state": {"scene_index": 0, "total_scenes": 1},
        "current_scene": {
            "scene_id": scene_id,
            "scripts": scripts if scripts is not None else [],
            "narration": {},
        },
    }


# ──────────────────────────────────────────────────────────────────────
# build_script_complete_payload — trigger field wire-shape contract
# ──────────────────────────────────────────────────────────────────────


def test_default_trigger_is_auto_on_payload():
    """Default ``trigger="auto"`` ⇒ the payload INCLUDES ``"trigger": "auto"``.

    The wire shape is self-describing: every ``script_complete`` carries
    its own trigger explicitly so the shell never has to default-fill.
    """
    snapshot = _snap()
    payload = build_script_complete_payload(snapshot, spoke_script=True)
    assert payload == {
        "type": "script_complete",
        "sceneIndex": 0,
        "hadScript": True,
        "trigger": "auto",
    }


def test_manual_trigger_includes_field_in_payload():
    """``trigger="manual"`` ⇒ the payload includes ``"trigger": "manual"``.

    The shell's auto-advance handler short-circuits on this so a Script
    button click never queues a scene jump, even on auto-advance rooms.
    """
    snapshot = _snap()
    payload = build_script_complete_payload(
        snapshot, spoke_script=True, trigger="manual"
    )
    assert payload == {
        "type": "script_complete",
        "sceneIndex": 0,
        "hadScript": True,
        "trigger": "manual",
    }


def test_unrecognized_trigger_value_passes_through_to_payload():
    """Any non-"auto" trigger surfaces verbatim — the validation lives
    on the shell side. Documents the wire-shape contract: the agent
    doesn't gate on a fixed enum, so future trigger values (e.g.
    ``"keyboard_shortcut"``) work without an agent change."""
    snapshot = _snap()
    payload = build_script_complete_payload(
        snapshot, spoke_script=False, trigger="future-value"
    )
    assert payload["trigger"] == "future-value"


# ──────────────────────────────────────────────────────────────────────
# SceneNarrator.narrate(force=True) — once-per-entry guard bypass
# ──────────────────────────────────────────────────────────────────────


def test_force_bypasses_already_narrated_guard():
    """A scene whose ``scene_id`` is already marked is re-narrated when
    ``force=True``. Without force the second call no-ops (the standard
    S65 idempotency); with force the speak callable fires again."""
    set_voice = AsyncMock()
    speak = AsyncMock()
    narrator = SceneNarrator(
        primary_voice_id="primary",
        set_voice=set_voice,
        speak=speak,
    )
    snapshot = _snap(
        scene_id="s1",
        scripts=[{"text": "hello", "voice_id": "primary"}],
    )

    # First entry: narrates and marks scene_id.
    spoke1 = _run(narrator.narrate(snapshot))
    assert spoke1 is True
    assert speak.await_count == 1

    # Second entry (auto): the idempotency guard fires; speak does NOT
    # run again. This is the S65 behavior the Script-button bypasses.
    spoke2 = _run(narrator.narrate(snapshot))
    assert spoke2 is False
    assert speak.await_count == 1

    # Third entry (forced): the guard is bypassed; speak DOES run again.
    spoke3 = _run(narrator.narrate(snapshot, force=True))
    assert spoke3 is True
    assert speak.await_count == 2


def test_force_still_marks_scene_id_after_replay():
    """A forced replay still updates the idempotency mark so a
    subsequent UNFORCED auto-trigger (e.g. canvas.register prompt
    rebuild) stays a no-op as designed. Without this, every
    canvas.register after a forced replay would re-narrate."""
    speak = AsyncMock()
    narrator = SceneNarrator(
        primary_voice_id="primary",
        set_voice=AsyncMock(),
        speak=speak,
    )
    snapshot = _snap(
        scene_id="s1",
        scripts=[{"text": "hello", "voice_id": "primary"}],
    )

    # Forced entry on a never-narrated scene.
    spoke_forced = _run(narrator.narrate(snapshot, force=True))
    assert spoke_forced is True
    assert narrator.narrated_scene_id == "s1"

    # Subsequent auto-trigger (no force) for the SAME scene_id ⇒ no-op.
    # Mirrors the canvas.register rebuild flow.
    spoke_auto = _run(narrator.narrate(snapshot))
    assert spoke_auto is False
    assert speak.await_count == 1
