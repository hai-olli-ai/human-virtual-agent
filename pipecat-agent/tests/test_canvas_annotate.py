"""Tests for canvas_annotate (Block 8) — the agent overlay-annotation tool.

Covers the pure element-box math (the A-AG-3 ÷100 landmine: stored geometry is
PERCENT 0-100, not 1280x720 pixels) and the handler's C2 target resolution +
best-effort emit, exercised with injected fakes (no pipecat stack needed). Tests
are sync defs that drive coroutines through asyncio.run, matching the repo
convention (see test_canvas_vision.py).
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

from services.vision_client import VisionClient
from tools.canvas_annotate import (
    AGENT_ANNOTATE_OPS,
    AGENT_ANNOTATE_SCHEMA,
    element_box_from_snapshot,
    make_handle_canvas_annotate,
)


def _run(coro):
    return asyncio.run(coro)


# ── Fakes ────────────────────────────────────────────────────────────────────


class FakeParams:
    """Minimal FunctionCallParams stand-in: carries arguments, captures the result."""

    def __init__(self, arguments: dict):
        self.arguments = arguments
        self.result = None

    async def result_callback(self, result):
        self.result = result


class FakeSession:
    def __init__(self, slug="s", scene_id="scene-1"):
        self._slug, self._scene = slug, scene_id

    def get_slug(self):
        return self._slug

    def get_current_scene_id(self):
        return self._scene


class FakeVision:
    def __init__(self, box=None):
        self._box = box

    async def locate(self, image_bytes, description):
        return self._box


class FakeBackend:
    def __init__(self, elements=None):
        self._elements = elements or []

    async def get_scene_snapshot(self, room_id, api_url, scene_id=None):
        return {"current_scene": {"elements": self._elements}}


def _make(
    *,
    elements=None,
    locate_box=None,
    ack=True,
    ack_ok=True,
    timeout_s=2.0,
    capture=("cid", {"status": "ready"}),
    live_bytes=b"jpeg",
    alias_map=None,
    vision_client=None,
):
    """Build a handler + the list it records sent agent_annotate payloads into."""
    sent: list[dict] = []
    pending: dict = {}

    async def send_message(payload):
        sent.append(payload)
        if ack:  # simulate the shell replying agent_annotate_result
            fut = pending.get(payload.get("annotateId"))
            if fut is not None and not fut.done():
                fut.set_result({"ok": ack_ok})

    async def request_capture(hint):
        return capture

    async def fetch_live_bytes(cid, res, backend, slug, api_url):
        return live_bytes

    handler = make_handle_canvas_annotate(
        send_message=send_message,
        pending=pending,
        vision_client=vision_client
        if vision_client is not None
        else FakeVision(locate_box),
        request_capture=request_capture,
        fetch_live_bytes=fetch_live_bytes,
        backend_client=FakeBackend(elements),
        session_context=FakeSession(),
        element_alias_map=alias_map
        if alias_map is not None
        else {"button_1": "uuid-b1"},
        room_id="room-1",
        api_url=None,
        timeout_s=timeout_s,
    )
    return handler, sent


def _el(eid, x, y, w, h, etype="button"):
    return {
        "id": eid,
        "type": etype,
        "position": {"x": x, "y": y},
        "size": {"width": w, "height": h},
    }


# ── Schema ───────────────────────────────────────────────────────────────────


def test_schema_shape():
    assert AGENT_ANNOTATE_SCHEMA.name == "canvas_annotate"
    assert AGENT_ANNOTATE_SCHEMA.required == ["op"]
    enum = AGENT_ANNOTATE_SCHEMA.properties["op"]["enum"]
    assert set(enum) == set(AGENT_ANNOTATE_OPS)
    assert "erase" in enum and "circle" in enum


# ── element_box_from_snapshot — the A-AG-3 ÷100 contract ──────────────────────


def test_element_box_percent_to_fraction():
    elements = [_el("uuid-b1", 25, 50, 10, 20)]
    box = element_box_from_snapshot("button_1", elements, {"button_1": "uuid-b1"})
    # PERCENT (25,50,10,20) / 100 → fractions, NOT /1280 or /720.
    assert box == {"x": 0.25, "y": 0.5, "w": 0.1, "h": 0.2}


def test_element_box_raw_uuid_passthrough():
    # alias not in map but matches an element id directly → resolves (pass-through).
    elements = [_el("raw-uuid", 0, 0, 100, 100)]
    box = element_box_from_snapshot("raw-uuid", elements, {"button_1": "uuid-b1"})
    assert box == {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0}


def test_element_box_unknown_alias_returns_none():
    assert (
        element_box_from_snapshot(
            "text_9", [_el("uuid-b1", 1, 1, 1, 1)], {"button_1": "uuid-b1"}
        )
        is None
    )


def test_element_box_missing_geometry_returns_none():
    el = {"id": "uuid-b1", "type": "button"}  # no position/size
    assert element_box_from_snapshot("button_1", [el], {"button_1": "uuid-b1"}) is None


def test_element_box_degenerate_returns_none():
    elements = [_el("uuid-b1", 10, 10, 0, 20)]  # zero width
    assert (
        element_box_from_snapshot("button_1", elements, {"button_1": "uuid-b1"}) is None
    )


def test_element_box_empty_alias_returns_none():
    assert element_box_from_snapshot("", [], {}) is None


# ── handler: erase / region / element / describe / failure / emit ─────────────


def test_erase_emits_erase_op():
    handler, sent = _make()
    p = FakeParams({"op": "erase"})
    _run(handler(p))
    assert sent == [sent[0]] and sent[0]["type"] == "agent_annotate"
    assert sent[0]["ops"] == [{"op": "erase"}]
    assert p.result["ok"] is True


def test_region_target_passthrough():
    handler, sent = _make()
    region = {"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4}
    p = FakeParams({"op": "circle", "target": {"region": region}})
    _run(handler(p))
    assert sent[0]["ops"] == [{"op": "circle", "box": region}]
    assert p.result["ok"] is True


def test_element_target_from_snapshot():
    handler, sent = _make(elements=[_el("uuid-b1", 50, 50, 20, 10)])
    p = FakeParams({"op": "highlight", "target": {"element": "button_1"}})
    _run(handler(p))
    assert sent[0]["ops"] == [
        {"op": "highlight", "box": {"x": 0.5, "y": 0.5, "w": 0.2, "h": 0.1}}
    ]


def test_describe_target_uses_locate():
    box = {"x": 0.3, "y": 0.3, "w": 0.2, "h": 0.2}
    handler, sent = _make(locate_box=box)
    p = FakeParams({"op": "circle", "target": {"describe": "the red car"}})
    _run(handler(p))
    assert sent[0]["ops"] == [{"op": "circle", "box": box}]


def test_describe_target_locate_miss_returns_error_no_emit():
    handler, sent = _make(locate_box=None)  # vision found nothing
    p = FakeParams({"op": "circle", "target": {"describe": "a unicorn"}})
    _run(handler(p))
    assert sent == []  # nothing emitted
    assert p.result["ok"] is False and "locate" in p.result["message"]


def test_unresolvable_element_returns_error_no_emit():
    handler, sent = _make(elements=[])  # alias resolves to uuid, but no such element
    p = FakeParams({"op": "circle", "target": {"element": "button_1"}})
    _run(handler(p))
    assert sent == []
    assert p.result["ok"] is False


def test_shape_op_includes_shape_field():
    handler, sent = _make()
    p = FakeParams(
        {
            "op": "shape",
            "shape": "heart",
            "target": {"region": {"x": 0, "y": 0, "w": 0.1, "h": 0.1}},
        }
    )
    _run(handler(p))
    assert sent[0]["ops"][0]["shape"] == "heart"


def test_text_op_includes_text_field():
    handler, sent = _make()
    p = FakeParams(
        {
            "op": "text",
            "text": "Look here",
            "target": {"region": {"x": 0, "y": 0, "w": 0.1, "h": 0.1}},
        }
    )
    _run(handler(p))
    assert sent[0]["ops"][0]["text"] == "Look here"


def test_arrow_from_resolves_source_box():
    handler, sent = _make(elements=[_el("uuid-b1", 10, 10, 10, 10)])
    p = FakeParams(
        {
            "op": "arrow",
            "from": "button_1",
            "target": {"region": {"x": 0.5, "y": 0.5, "w": 0.1, "h": 0.1}},
        }
    )
    _run(handler(p))
    op = sent[0]["ops"][0]
    assert op["box"] == {"x": 0.5, "y": 0.5, "w": 0.1, "h": 0.1}
    assert op["from"] == {"x": 0.1, "y": 0.1, "w": 0.1, "h": 0.1}


def test_emit_timeout_is_best_effort_ok_true():
    # No ack from the shell + a tiny timeout → _emit times out but reports rendered.
    handler, sent = _make(ack=False, timeout_s=0.01)
    p = FakeParams(
        {"op": "circle", "target": {"region": {"x": 0, "y": 0, "w": 0.1, "h": 0.1}}}
    )
    _run(handler(p))
    assert len(sent) == 1
    assert p.result["ok"] is True


def test_emit_ack_failure_reports_not_ok():
    handler, sent = _make(ack=True, ack_ok=False)
    p = FakeParams(
        {"op": "circle", "target": {"region": {"x": 0, "y": 0, "w": 0.1, "h": 0.1}}}
    )
    _run(handler(p))
    assert p.result["ok"] is False


def test_unknown_op_rejected_no_emit():
    handler, sent = _make()
    p = FakeParams(
        {"op": "frobnicate", "target": {"region": {"x": 0, "y": 0, "w": 0.1, "h": 0.1}}}
    )
    _run(handler(p))
    assert sent == []
    assert p.result["ok"] is False


# ── B10 #1/#4 — element target → box → emitted + success copy ─────────────────


def test_circle_element_target_emits_box_and_success_copy():
    handler, sent = _make(
        elements=[_el("uuid-title", 10, 20, 30, 40)],
        alias_map={"title": "uuid-title"},
    )
    p = FakeParams({"op": "circle", "target": {"element": "title"}})
    _run(handler(p))
    assert sent[0]["type"] == "agent_annotate"
    assert sent[0]["ops"] == [
        {"op": "circle", "box": {"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4}}
    ]
    assert p.result["ok"] is True and "circle" in p.result["message"]


# ── B10 #7 — real VisionClient stub (GOOGLE_AI_API_KEY unset) → locate None ────


def test_describe_with_vision_stub_asks_to_clarify():
    # A real VisionClient with no key is the deployed stub path: locate() returns
    # None, so the describe branch can't resolve a box → clarify, nothing emitted.
    vc = VisionClient(api_key="")
    assert vc.enabled is False
    handler, sent = _make(vision_client=vc)
    p = FakeParams({"op": "circle", "target": {"describe": "the actor"}})
    _run(handler(p))
    assert sent == []
    assert p.result["ok"] is False and "locate" in p.result["message"]


# ── B10 #8 — agent_annotate_result routes in the non-canvas branch, BEFORE the
#    canvas dispatch, in BOTH handlers. bot.py's closures aren't importable
#    (full pipecat stack), so we guard the real source structurally AND mirror
#    the routing contract — same approach test_canvas_vision uses for captures.


def test_agent_annotate_result_routed_before_canvas_dispatch():
    src = (Path(__file__).resolve().parent.parent / "bot.py").read_text()
    # Present in both pipelines (classic + relay).
    assert src.count('msg_type == "agent_annotate_result"') == 2
    # Each branch sits before canvas.register and early-returns, so the canvas
    # dispatch never sees the ack.
    blocks = re.findall(
        r'msg_type == "agent_annotate_result"(.*?)canvas\.register', src, re.S
    )
    assert len(blocks) == 2, (
        "annotate-ack branch must precede canvas.register in both handlers"
    )
    for b in blocks:
        assert "return" in b, (
            "annotate-ack branch must early-return before the canvas dispatch"
        )


def test_agent_annotate_result_short_circuits_before_dispatch():
    """Contract-mirror of on_app_message: an agent_annotate_result resolves its
    pending Future and returns BEFORE the canvas/relay dispatch (which must never
    run for it). Rides the same defensive json.loads as the real handler."""

    def _handle_annotate_result_mirror(message, *, pending):
        aid = message.get("annotateId")
        fut = pending.get(aid)
        if fut is not None and not fut.done():
            fut.set_result(message)

    async def _route(message, *, pending, canvas_dispatch):
        if isinstance(message, str):
            try:
                message = json.loads(message)
            except (json.JSONDecodeError, ValueError):
                pass
        if not isinstance(message, dict):
            return "ignored"
        if message.get("type") == "agent_annotate_result":
            _handle_annotate_result_mirror(message, pending=pending)
            return "annotate"
        await canvas_dispatch(message)
        return "dispatch"

    async def _drive():
        loop = asyncio.get_running_loop()
        pending = {"a1": loop.create_future()}
        dispatched: list = []

        async def _canvas_dispatch(m):
            dispatched.append(m)

        # delivered as a JSON STRING (defensive json.loads) → routed to the
        # annotate branch, dispatch never called, future resolved.
        r1 = await _route(
            json.dumps(
                {"type": "agent_annotate_result", "annotateId": "a1", "ok": True}
            ),
            pending=pending,
            canvas_dispatch=_canvas_dispatch,
        )
        # control: a canvas.* message DOES reach the dispatch.
        r2 = await _route(
            {"type": "canvas.register", "pageType": "quiz"},
            pending=pending,
            canvas_dispatch=_canvas_dispatch,
        )
        return r1, r2, dispatched, pending["a1"]

    r1, r2, dispatched, fut = _run(_drive())
    assert r1 == "annotate"
    assert r2 == "dispatch"
    assert dispatched == [{"type": "canvas.register", "pageType": "quiz"}]
    assert fut.done() and fut.result()["ok"] is True
