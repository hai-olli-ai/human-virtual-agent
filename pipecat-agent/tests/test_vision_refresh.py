"""Tests for ``vision_refresh.py`` (S66 Block 5a).

Covers:

  * :class:`VisionFrameTracker` — load/invalidate semantics; refusal to
    mark unknown ids as loaded (so a None scene_id never gets cached and
    silently masks a stale frame).
  * :func:`ensure_vision_frame_for_scene` — hit (no fetch), miss (fetch
    + add + mark), empty-fetch path (no add, no mark — next call
    retries).
  * Integration: when ``CanvasToolContext.ensure_vision`` is wired,
    ``handle_analyze`` invokes it before dispatching; when unwired,
    behaviour is the pre-5a baseline.

Follows the existing tests/ convention: no pytest-asyncio (not in the
dependency closure), so each async test goes through ``asyncio.run``.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from vision_refresh import (
    VisionFrameTracker,
    ensure_vision_frame_for_scene,
)


def _run(coro):
    return asyncio.run(coro)


# ──────────────────────────────────────────────────────────────────────
# VisionFrameTracker
# ──────────────────────────────────────────────────────────────────────


def test_tracker_starts_unloaded():
    t = VisionFrameTracker()
    assert t.loaded_scene_id is None
    assert not t.is_loaded_for("scene-1")
    assert not t.is_loaded_for(None)


def test_tracker_mark_then_match():
    t = VisionFrameTracker()
    t.mark_loaded("scene-1")
    assert t.loaded_scene_id == "scene-1"
    assert t.is_loaded_for("scene-1")
    assert not t.is_loaded_for("scene-2")


def test_tracker_invalidate_clears():
    t = VisionFrameTracker()
    t.mark_loaded("scene-1")
    t.invalidate()
    assert t.loaded_scene_id is None
    assert not t.is_loaded_for("scene-1")


def test_tracker_none_scene_id_never_loaded():
    """Refusing to mark unknown ids keeps the next ensure() honest —
    otherwise an initial-fetch race that finishes without a scene_id
    would pin the tracker at None forever and skip refetches."""
    t = VisionFrameTracker()
    t.mark_loaded(None)
    assert t.loaded_scene_id is None
    assert not t.is_loaded_for(None)
    assert not t.is_loaded_for("scene-1")


def test_tracker_relabel_updates_id():
    t = VisionFrameTracker()
    t.mark_loaded("scene-1")
    t.mark_loaded("scene-2")
    assert t.loaded_scene_id == "scene-2"
    assert t.is_loaded_for("scene-2")
    assert not t.is_loaded_for("scene-1")


# ──────────────────────────────────────────────────────────────────────
# ensure_vision_frame_for_scene
# ──────────────────────────────────────────────────────────────────────


def test_ensure_hits_when_tracker_covers_scene():
    """Tracker already covers the target scene → no fetch, no add."""
    tracker = VisionFrameTracker()
    tracker.mark_loaded("scene-1")
    fetch = AsyncMock(return_value="ignored")
    add = MagicMock()
    did_load = _run(
        ensure_vision_frame_for_scene(
            tracker=tracker,
            current_scene_id="scene-1",
            fetch_image_base64=fetch,
            add_vision_message=add,
        )
    )
    assert did_load is False
    fetch.assert_not_called()
    add.assert_not_called()


def test_ensure_fetches_on_miss():
    tracker = VisionFrameTracker()
    fetch = AsyncMock(return_value="base64data")
    add = MagicMock()
    did_load = _run(
        ensure_vision_frame_for_scene(
            tracker=tracker,
            current_scene_id="scene-1",
            fetch_image_base64=fetch,
            add_vision_message=add,
        )
    )
    assert did_load is True
    fetch.assert_awaited_once_with()
    add.assert_called_once_with("base64data")
    assert tracker.loaded_scene_id == "scene-1"


def test_ensure_refetches_on_scene_change():
    tracker = VisionFrameTracker()
    tracker.mark_loaded("scene-1")
    fetch = AsyncMock(return_value="scene2-data")
    add = MagicMock()
    did_load = _run(
        ensure_vision_frame_for_scene(
            tracker=tracker,
            current_scene_id="scene-2",
            fetch_image_base64=fetch,
            add_vision_message=add,
        )
    )
    assert did_load is True
    fetch.assert_awaited_once()
    add.assert_called_once_with("scene2-data")
    assert tracker.loaded_scene_id == "scene-2"


def test_ensure_no_fetch_returned_keeps_tracker_unset():
    """Backend transient failure → no add, no mark; the next call retries."""
    tracker = VisionFrameTracker()
    fetch = AsyncMock(return_value=None)
    add = MagicMock()
    did_load = _run(
        ensure_vision_frame_for_scene(
            tracker=tracker,
            current_scene_id="scene-1",
            fetch_image_base64=fetch,
            add_vision_message=add,
        )
    )
    assert did_load is False
    fetch.assert_awaited_once()
    add.assert_not_called()
    assert tracker.loaded_scene_id is None

    fetch.return_value = "actual-data"
    did_load = _run(
        ensure_vision_frame_for_scene(
            tracker=tracker,
            current_scene_id="scene-1",
            fetch_image_base64=fetch,
            add_vision_message=add,
        )
    )
    assert did_load is True
    assert tracker.loaded_scene_id == "scene-1"


def test_ensure_no_scene_id_refetches_every_time():
    """Edge case: unknown scene_id → tracker never marks loaded → fetch
    fires every ensure(). Wasteful but correct (flow-based rooms always
    have ids; this is degraded-snapshot fallback territory)."""
    tracker = VisionFrameTracker()
    fetch = AsyncMock(return_value="data")
    add = MagicMock()
    for _ in range(3):
        _run(
            ensure_vision_frame_for_scene(
                tracker=tracker,
                current_scene_id=None,
                fetch_image_base64=fetch,
                add_vision_message=add,
            )
        )
    assert fetch.await_count == 3
    assert add.call_count == 3


# ──────────────────────────────────────────────────────────────────────
# CanvasToolContext.ensure_vision wiring → handle_analyze
# ──────────────────────────────────────────────────────────────────────


def _make_canvas_ctx(ensure_vision=None):
    """Build a minimal CanvasToolContext for handler tests."""
    from tools.canvas_protocol_tools import (
        CanvasToolContext,
        PendingCommandRegistry,
    )
    from context.canvas_manifest import CanvasManifestRegistry

    sent: list[dict] = []

    async def _send(payload):
        sent.append(payload)

    ctx = CanvasToolContext(
        manifest_registry=CanvasManifestRegistry(),
        pending=PendingCommandRegistry(),
        send_app_message=_send,
        ensure_vision=ensure_vision,
    )
    return ctx, sent


def test_handle_analyze_invokes_ensure_vision_when_wired():
    from tools.canvas_protocol_tools import make_handlers

    called = []

    async def _ensure(question=""):
        called.append("ensure")

    ctx, sent = _make_canvas_ctx(ensure_vision=_ensure)

    # Resolve the canvas command future so handle_analyze returns
    # promptly. We schedule a separate task that drains `sent`, looks
    # at the dispatched commandId, and resolves the registry future
    # with a dummy result before the timeout.
    async def _drive():
        handlers = make_handlers(ctx)

        async def _result_callback(r):
            called.append(("result", r))

        params = MagicMock()
        params.arguments = {"question": "what is on screen?"}
        params.result_callback = _result_callback

        task = asyncio.create_task(handlers["canvas_analyze"](params))
        # Wait briefly for the dispatch to register and send.
        for _ in range(20):
            if sent:
                break
            await asyncio.sleep(0.005)
        assert sent, "canvas_analyze did not dispatch"
        cmd_id = sent[0]["commandId"]
        ctx.pending.resolve(cmd_id, {"ok": True})
        await task

    _run(_drive())

    # ensure() ran AND the analyze dispatch completed.
    assert "ensure" in called
    assert any(isinstance(c, tuple) and c[0] == "result" for c in called)


def test_handle_analyze_continues_when_ensure_raises():
    """A vision-frame failure must not break the analyze tool call —
    the iframe's semantic state alone still answers most questions."""
    from tools.canvas_protocol_tools import make_handlers

    async def _ensure_fail(question=""):
        raise RuntimeError("backend down")

    ctx, sent = _make_canvas_ctx(ensure_vision=_ensure_fail)

    results: list = []

    async def _drive():
        handlers = make_handlers(ctx)

        async def _result_callback(r):
            results.append(r)

        params = MagicMock()
        params.arguments = {"question": "what is on screen?"}
        params.result_callback = _result_callback

        task = asyncio.create_task(handlers["canvas_analyze"](params))
        for _ in range(20):
            if sent:
                break
            await asyncio.sleep(0.005)
        assert sent, "ensure failure should not block dispatch"
        ctx.pending.resolve(sent[0]["commandId"], {"ok": True})
        await task

    _run(_drive())
    assert results == [{"ok": True}]


def test_handle_analyze_skips_ensure_when_unwired():
    """Pre-5a path: ensure_vision=None ⇒ handler dispatches directly."""
    from tools.canvas_protocol_tools import make_handlers

    ctx, sent = _make_canvas_ctx(ensure_vision=None)
    results: list = []

    async def _drive():
        handlers = make_handlers(ctx)

        async def _result_callback(r):
            results.append(r)

        params = MagicMock()
        params.arguments = {"question": "q?"}
        params.result_callback = _result_callback

        task = asyncio.create_task(handlers["canvas_analyze"](params))
        for _ in range(20):
            if sent:
                break
            await asyncio.sleep(0.005)
        ctx.pending.resolve(sent[0]["commandId"], {"ok": True})
        await task

    _run(_drive())
    assert results == [{"ok": True}]
