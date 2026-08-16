"""S79 — the continue-navigation guard (field fix 2026-08-16).

The double-action bug: "continue" made the LLM call continue_presentation
AND canvas_control(next_scene) — narration restarted while the eager
next_scene fire advanced the room. Two layers under test:

  * :func:`services.eager_dispatch.maybe_fire_eager` — a stream turn that
    contains a continue_presentation call suppresses eager fires for
    SCENE_NAV verbs (the tracker is the only component that sees the whole
    turn mid-stream). Non-nav eager verbs stay eager; turns without
    continue_presentation are untouched.
  * :func:`tools.canvas_protocol_tools.make_handlers`'s handle_control —
    a scene-nav verb arriving while ``ctx.nav_guard_until`` is in the
    future returns the corrective NAV_SUPPRESSED result and dispatches
    nothing; an expired guard dispatches normally.

Follows the tests/ convention: no pytest-asyncio, asyncio.run per test.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, patch

from services.eager_dispatch import EagerToolCallTracker, maybe_fire_eager
from tools.canvas_protocol_tools import (
    CanvasToolContext,
    CanvasManifestRegistry,
    PendingCommandRegistry,
    make_handlers,
)


def _run(coro):
    return asyncio.run(coro)


def _tracker_with(calls: list[tuple[str, str]]) -> EagerToolCallTracker:
    """calls = [(tool_name, accumulated_args_json), ...] in stream order."""
    tracker = EagerToolCallTracker()
    for idx, (tool, args) in enumerate(calls):
        tracker.begin_tool_call(idx, tool)
        tracker.append_args(idx, args)
    return tracker


# ── Eager-side suppression ────────────────────────────────────────────


def test_eager_nav_suppressed_when_continue_in_turn():
    async def body():
        sent = []

        async def send(payload):
            sent.append(payload)

        tracker = _tracker_with(
            [
                ("continue_presentation", "{}"),
                ("canvas_control", '{"verb": "next_scene"}'),
            ]
        )
        await maybe_fire_eager(tracker, 1, PendingCommandRegistry(), send)
        assert sent == []  # suppressed — continue owns navigation
        assert tracker.get_call(1)["eager_fired"] is False

    _run(body())


def test_eager_nav_fires_without_continue():
    async def body():
        sent = []

        async def send(payload):
            sent.append(payload)

        tracker = _tracker_with([("canvas_control", '{"verb": "next_scene"}')])
        await maybe_fire_eager(tracker, 0, PendingCommandRegistry(), send)
        assert len(sent) == 1  # normal eager behavior untouched

    _run(body())


def test_eager_non_nav_verbs_stay_eager_beside_continue():
    async def body():
        sent = []

        async def send(payload):
            sent.append(payload)

        tracker = _tracker_with(
            [
                ("continue_presentation", "{}"),
                ("canvas_control", '{"verb": "pause"}'),
            ]
        )
        await maybe_fire_eager(tracker, 1, PendingCommandRegistry(), send)
        assert len(sent) == 1  # pause is not scene-nav — eager stands

    _run(body())


# ── Handler-side guard ────────────────────────────────────────────────


class _Params:
    def __init__(self):
        self.arguments = {"verb": "next_scene"}
        self.results = []

    async def result_callback(self, result):
        self.results.append(result)


def _ctx() -> CanvasToolContext:
    return CanvasToolContext(
        manifest_registry=CanvasManifestRegistry(),
        pending=PendingCommandRegistry(),
        send_app_message=AsyncMock(),
    )


def test_handle_control_nav_suppressed_inside_guard_window():
    async def body():
        ctx = _ctx()
        ctx.nav_guard_until = time.monotonic() + 5.0
        handlers = make_handlers(ctx)
        params = _Params()
        with patch(
            "tools.canvas_protocol_tools.dispatch_canvas_command", new=AsyncMock()
        ) as dispatch:
            await handlers["canvas_control"](params)
            dispatch.assert_not_awaited()
        assert params.results[0]["error"] == "NAV_SUPPRESSED"

    _run(body())


def test_handle_control_nav_dispatches_after_guard_expires():
    async def body():
        ctx = _ctx()
        ctx.nav_guard_until = time.monotonic() - 1.0
        handlers = make_handlers(ctx)
        params = _Params()
        with patch(
            "tools.canvas_protocol_tools.dispatch_canvas_command",
            new=AsyncMock(return_value={"ok": True}),
        ) as dispatch:
            await handlers["canvas_control"](params)
            dispatch.assert_awaited_once()
        assert params.results[0] == {"ok": True}

    _run(body())
