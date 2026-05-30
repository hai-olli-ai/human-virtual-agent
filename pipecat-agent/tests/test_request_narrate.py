"""Tests for the ``request_narrate`` inbound-message handler (S65c Block 5).

The handler lives inside ``run_bot_classic`` / ``run_bot_relay`` as a
nested closure over the transport, narrator, snapshot fetcher, and
emit channel. Direct import isn't possible, so this test exercises
the same composition the handler performs:

  1. Run ``run_scene_narration(force=True)`` so a scene whose entry
     narration already marked it gets re-narrated.
  2. Emit ``build_script_complete_payload(..., trigger="manual")`` so
     the shell short-circuits auto-advance.

The orchestration shape is identical to the existing
``_drive_classic_scene_entry`` helper in ``test_scene_narration.py`` —
this test just adds the manual-replay deltas.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

from narration import (
    SceneNarrator,
    build_script_complete_payload,
    run_scene_narration,
)


def _run(coro):
    return asyncio.run(coro)


def _snap(
    *,
    scene_id: str = "scene-1",
    scripts: list | None = None,
    scene_index: int = 1,
    total_scenes: int = 3,
    auto_advance: bool = True,
) -> dict:
    return {
        "live_room": {"auto_advance": auto_advance, "language": "en"},
        "flow_state": {"scene_index": scene_index, "total_scenes": total_scenes},
        "current_scene": {
            "scene_id": scene_id,
            "scripts": scripts if scripts is not None else [],
            "narration": {
                "invitation_line": "Any questions?",
                "transition_cue": "moving on",
            },
        },
    }


async def _drive_manual_replay(
    snapshot: dict,
    narrator: SceneNarrator,
    speak: AsyncMock,
    send: AsyncMock,
) -> bool:
    """Mirror the orchestration in bot.py's ``request_narrate`` branch
    (both pipelines) — narrate(force=True) + emit(trigger="manual")."""
    spoke_script = await run_scene_narration(
        snapshot,
        narrator=narrator,
        speak_followup=speak,
        force=True,
    )
    await send(
        build_script_complete_payload(
            snapshot, spoke_script=spoke_script, trigger="manual"
        )
    )
    return spoke_script


def test_request_narrate_replays_already_narrated_scene_with_manual_trigger():
    """The Block 5 ``request_narrate`` composition re-narrates a scene
    whose ``scene_id`` is already marked, then emits ``script_complete``
    with ``trigger="manual"``.

    This is the load-bearing scenario: a visitor clicked Script AFTER
    the scene's entry narration already played. Without the force
    bypass + manual-trigger emit, the click would no-op AND would emit
    a default auto trigger that could mis-route through the shell's
    auto-advance handler.
    """
    set_voice = AsyncMock()
    events: list[tuple] = []
    speak = AsyncMock(side_effect=lambda t: events.append(("speak", t)))
    send = AsyncMock(side_effect=lambda m: events.append(("send", m)))
    narrator = SceneNarrator(
        primary_voice_id="primary",
        set_voice=set_voice,
        speak=speak,
    )
    snapshot = _snap(
        scene_id="s2",
        scripts=[{"text": "scene two content", "voice_id": "primary"}],
        scene_index=1,
        total_scenes=3,
        auto_advance=True,
    )

    # Prime the narrator: simulate the entry narration that ran when the
    # visitor first reached this scene. After this call, scene_id is
    # marked and an unforced replay would no-op.
    entry_spoke = _run(narrator.narrate(snapshot))
    assert entry_spoke is True
    assert narrator.narrated_scene_id == "s2"
    assert speak.await_count == 1

    # Manual replay: drive the Block 5 composition. force=True bypasses
    # the guard; trigger="manual" surfaces on the emit.
    replay_spoke = _run(_drive_manual_replay(snapshot, narrator, speak, send))
    assert replay_spoke is True

    # Verify (a) the script was actually re-spoken (proving force
    # bypassed the guard), and (b) the wire emit carries trigger:manual.
    spoken = [t for kind, t in events if kind == "speak"]
    assert "scene two content" in spoken

    send_events = [m for kind, m in events if kind == "send"]
    assert len(send_events) == 1
    assert send_events[0] == {
        "type": "script_complete",
        "sceneIndex": 1,
        "hadScript": True,
        "trigger": "manual",
    }


def test_request_narrate_emits_manual_trigger_even_when_no_script():
    """A no-script scene still emits ``script_complete`` with
    ``trigger="manual"`` and ``hadScript=False``. The shell's
    auto-advance handler ignores manual triggers regardless, but the
    emit ordering keeps the shell's other state machines consistent
    (e.g. clearing any pending spinner that the button mounted)."""
    set_voice = AsyncMock()
    events: list[tuple] = []
    speak = AsyncMock(side_effect=lambda t: events.append(("speak", t)))
    send = AsyncMock(side_effect=lambda m: events.append(("send", m)))
    narrator = SceneNarrator(
        primary_voice_id="primary",
        set_voice=set_voice,
        speak=speak,
    )
    snapshot = _snap(
        scene_id="s-empty",
        scripts=[],
        scene_index=0,
        total_scenes=2,
        auto_advance=False,
    )

    spoke = _run(_drive_manual_replay(snapshot, narrator, speak, send))
    assert spoke is False

    # No script ⇒ no speak. Single emit with hadScript=False + trigger.
    assert [kind for kind, _ in events] == ["send"]
    send_events = [m for kind, m in events if kind == "send"]
    assert send_events[0] == {
        "type": "script_complete",
        "sceneIndex": 0,
        "hadScript": False,
        "trigger": "manual",
    }
