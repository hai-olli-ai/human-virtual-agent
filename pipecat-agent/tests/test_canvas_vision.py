"""Tests for the S67b agent-side vision path.

What's directly testable vs not (same convention as test_request_quiz.py):

  * ``request_canvas_capture`` and the ``canvas_capture_result`` branch are
    nested closures inside bot.py's run functions — NOT importable. Their
    *contract* is exercised two ways:
      - a faithful behavioural mirror of the future-by-captureId round-trip
        (``_request_capture_mirror`` / ``_handle_capture_result_mirror``),
        kept byte-for-byte aligned with bot.py;
      - a real-source structural guard asserting the branch is an
        early-return BEFORE the canvas dispatch in both handlers.
  * The orchestration core IS importable and gets real-code coverage:
    ``services.vision_query.{run_vision_query, _fetch_live_bytes, _fetch_pillow}`` and
    ``services.vision_client.{VisionClient, derive_vision_mode}``.

Convention: no pytest-asyncio (not in the dependency closure) — async work
goes through ``asyncio.run`` via ``_run``.
"""

from __future__ import annotations

import asyncio
import json
import re
import uuid
from pathlib import Path

from services.vision_client import VisionClient, VISION_UNAVAILABLE, derive_vision_mode
from services.vision_query import _fetch_live_bytes, _fetch_pillow, run_vision_query
from tools.quiz_generation import SessionContext


def _run(coro):
    return asyncio.run(coro)


# ──────────────────────────────────────────────────────────────────────
# Fakes — the injected deps run_vision_query / the _fetch_* helpers expect.
# ──────────────────────────────────────────────────────────────────────


class FakeVisionClient:
    """Records analyze_image calls; returns a canned answer."""

    def __init__(self, answer: str = "an answer"):
        self.answer = answer
        self.calls: list[tuple] = []

    async def analyze_image(self, image_bytes, mode, scene_context="", mime_type="image/jpeg"):
        self.calls.append((mode, scene_context, mime_type, image_bytes))
        return self.answer


class FakeBackend:
    """Stands in for the api_client module (the ``backend_client`` dep)."""

    def __init__(self, capture_bytes=None, png_b64=None, snapshot=None):
        self._capture_bytes = capture_bytes
        self._png_b64 = png_b64
        self._snapshot = snapshot
        self.get_vision_capture_calls: list[tuple] = []
        self.scene_image_calls = 0

    async def get_vision_capture(self, slug, capture_id, api_url=None):
        self.get_vision_capture_calls.append((slug, capture_id, api_url))
        if isinstance(self._capture_bytes, list):  # sequenced (retry tests)
            return self._capture_bytes.pop(0) if self._capture_bytes else None
        return self._capture_bytes

    async def get_scene_image_base64(self, room_id, api_url=None):
        self.scene_image_calls += 1
        return self._png_b64

    async def get_scene_snapshot(self, room_id, api_url=None, scene_id=None):
        return self._snapshot


def _capture_returning(result):
    """Build a fake request_capture that yields ``(capture_id, result)`` and
    counts calls (``.calls``) so retry behaviour can be asserted."""

    async def _request(hint):
        _request.calls += 1
        return "cap-fixed-id", result

    _request.calls = 0
    return _request


# ──────────────────────────────────────────────────────────────────────
# (1) Round-trip contract — behavioural mirror of bot.py's closures.
#     Kept aligned with bot.py:request_canvas_capture + the
#     canvas_capture_result on_app_message branch.
# ──────────────────────────────────────────────────────────────────────


async def _request_capture_mirror(hint, *, send, pending, timeout_ms=4000):
    capture_id = str(uuid.uuid4())
    loop = asyncio.get_running_loop()
    fut = loop.create_future()
    pending[capture_id] = fut
    try:
        await send({"type": "request_canvas_capture", "captureId": capture_id, "hint": hint})
        return capture_id, await asyncio.wait_for(fut, timeout=timeout_ms / 1000)
    except asyncio.TimeoutError:
        return capture_id, None
    finally:
        pending.pop(capture_id, None)


def _handle_capture_result_mirror(message, *, pending):
    cid = message.get("captureId")
    fut = pending.get(cid)
    if fut is not None and not fut.done():
        fut.set_result(message)


def test_capture_round_trip_resolves_by_id():
    """A matching captureId resolves the awaiting future with the result dict."""

    async def _drive():
        pending: dict = {}
        sent: list = []

        async def _send(payload):
            sent.append(payload)

        task = asyncio.create_task(
            _request_capture_mirror("point", send=_send, pending=pending)
        )
        for _ in range(20):  # let the request register + send
            if sent:
                break
            await asyncio.sleep(0.005)
        assert sent and sent[0]["type"] == "request_canvas_capture"
        cid = sent[0]["captureId"]
        _handle_capture_result_mirror(
            {"type": "canvas_capture_result", "captureId": cid, "status": "ready", "w": 800, "h": 600},
            pending=pending,
        )
        return await task

    capture_id, result = _run(_drive())
    assert result == {"type": "canvas_capture_result", "captureId": capture_id, "status": "ready", "w": 800, "h": 600}


def test_capture_round_trip_timeout_returns_none():
    """No reply within the timeout → (capture_id, None); the pending entry is cleaned up."""

    async def _drive():
        pending: dict = {}

        async def _send(payload):
            pass

        cid, result = await _request_capture_mirror(
            "point", send=_send, pending=pending, timeout_ms=30
        )
        return cid, result, pending

    cid, result, pending = _run(_drive())
    assert result is None
    assert pending == {}  # finally-pop ran


def test_stale_capture_result_is_ignored():
    """A canvas_capture_result for an unknown captureId is a harmless no-op."""
    pending: dict = {}
    # Must not raise even though nothing is registered.
    _handle_capture_result_mirror(
        {"type": "canvas_capture_result", "captureId": "never-sent", "status": "ready"},
        pending=pending,
    )
    assert pending == {}


# ──────────────────────────────────────────────────────────────────────
# _fetch_live_bytes / _fetch_pillow — the two image sources.
# ──────────────────────────────────────────────────────────────────────


def test_fetch_live_bytes_ready_returns_jpeg():
    """ready + bytes in backend → the live JPEG, fetched by (slug, captureId)."""
    backend = FakeBackend(capture_bytes=b"JPEGBYTES")
    img = _run(_fetch_live_bytes("cap-1", {"status": "ready", "w": 800}, backend, "demo-slug", None))
    assert img == b"JPEGBYTES"
    assert backend.get_vision_capture_calls == [("demo-slug", "cap-1", None)]


def test_fetch_live_bytes_not_ready_returns_none_without_fetching():
    """Non-ready capture → None, and no backend fetch attempted."""
    backend = FakeBackend(capture_bytes=b"unused")
    assert _run(_fetch_live_bytes("cap-1", None, backend, "demo-slug", None)) is None
    assert _run(_fetch_live_bytes("cap-1", {"status": "error"}, backend, "demo-slug", None)) is None
    assert backend.get_vision_capture_calls == []  # never fetched on a non-ready result


def test_fetch_live_bytes_ready_but_missing_bytes_returns_none():
    """ready but the backend has no bytes (404/expired) → None (caller retries)."""
    backend = FakeBackend(capture_bytes=None)
    assert _run(_fetch_live_bytes("cap-1", {"status": "ready"}, backend, "demo-slug", None)) is None
    assert backend.get_vision_capture_calls == [("demo-slug", "cap-1", None)]  # it DID try


def test_fetch_pillow_decodes_png():
    import base64

    backend = FakeBackend(png_b64=base64.b64encode(b"\x89PNG").decode())
    assert _run(_fetch_pillow(backend, "room-1", None)) == b"\x89PNG"
    assert backend.scene_image_calls == 1


def test_fetch_pillow_none_when_no_render():
    backend = FakeBackend(png_b64=None)
    assert _run(_fetch_pillow(backend, "room-1", None)) is None


# ──────────────────────────────────────────────────────────────────────
# run_vision_query — live-capture-first: Pillow is NEVER used while
# screen-share is on (the capture comes back ready / transient).
# ──────────────────────────────────────────────────────────────────────


def test_run_vision_query_live_capture_never_touches_pillow():
    """ready + bytes → live Gemini (jpeg, no blind-spot); Pillow not fetched, no retry."""
    vc = FakeVisionClient(answer="You circled the lead actor.")
    backend = FakeBackend(capture_bytes=b"JPEG", png_b64="must-not-be-used")
    rc = _capture_returning({"status": "ready", "w": 1280, "h": 720})
    msg = _run(
        run_vision_query(
            "what am I pointing at?",
            request_capture=rc, vision_client=vc, backend_client=backend,
            session_context=SessionContext(slug="demo-slug", current_scene_id="s1"), room_id="room-1", api_url=None,
        )
    )
    assert vc.calls[0][2] == "image/jpeg"          # reasoned over the live capture
    assert "circled the lead actor" in msg["content"]
    assert "NOT visible" not in msg["content"]     # no blind-spot note on a live capture
    assert backend.scene_image_calls == 0          # Pillow NEVER fetched
    assert rc.calls == 1                            # no retry needed


def test_run_vision_query_ready_no_bytes_retries_then_succeeds():
    """ready but bytes missing → retry the capture once → live Gemini (still no Pillow)."""
    vc = FakeVisionClient(answer="A highlighted paragraph.")
    backend = FakeBackend(capture_bytes=[None, b"JPEG"], png_b64="must-not-be-used")  # miss then hit
    rc = _capture_returning({"status": "ready"})
    _run(
        run_vision_query(
            "what did I highlight?",
            request_capture=rc, vision_client=vc, backend_client=backend,
            session_context=SessionContext(slug="demo-slug", current_scene_id="s1"), room_id="room-1", api_url=None,
        )
    )
    assert rc.calls == 2                            # retried once
    assert len(backend.get_vision_capture_calls) == 2
    assert vc.calls[0][2] == "image/jpeg"          # live capture on the retry
    assert backend.scene_image_calls == 0          # still no Pillow


def test_run_vision_query_timeout_no_retry_returns_retry_note():
    """Timeout (capture_result None) → NO retry (already waited the budget), NO Pillow."""
    vc = FakeVisionClient(answer="must not be called")
    backend = FakeBackend(png_b64="must-not-be-used")
    rc = _capture_returning(None)
    msg = _run(
        run_vision_query(
            "what am I pointing at?",
            request_capture=rc, vision_client=vc, backend_client=backend,
            session_context=SessionContext(slug="demo-slug", current_scene_id="s1"), room_id="room-1", api_url=None,
        )
    )
    assert rc.calls == 1                            # a timeout is not retried
    assert vc.calls == [] and backend.scene_image_calls == 0
    assert "try again" in msg["content"].lower()


def test_run_vision_query_error_retries_then_retry_note():
    """A non-permission error retries once; if it misses again → retry note, no Pillow."""
    vc = FakeVisionClient(answer="must not be called")
    backend = FakeBackend(png_b64="must-not-be-used")
    rc = _capture_returning({"status": "error", "error": "capture_failed"})
    msg = _run(
        run_vision_query(
            "what did I draw?",
            request_capture=rc, vision_client=vc, backend_client=backend,
            session_context=SessionContext(slug="demo-slug", current_scene_id="s1"), room_id="room-1", api_url=None,
        )
    )
    assert rc.calls == 2                            # retried the fast error once
    assert vc.calls == [] and backend.scene_image_calls == 0
    assert "try again" in msg["content"].lower()


# ──────────────────────────────────────────────────────────────────────
# (5) run_vision_query — assess mode threads scene answer into the prompt.
# ──────────────────────────────────────────────────────────────────────


def test_run_vision_query_assess_passes_scene_context():
    """'assess' → scene instruction + script text reach analyze_image's scene_context."""
    snapshot = {
        "current_scene": {
            "instruction": "The expected answer is mitochondria.",
            "scripts": [{"text": "It is the powerhouse of the cell."}],
        },
        "knowledge": None,
    }
    vc = FakeVisionClient(answer="Correct.")
    backend = FakeBackend(capture_bytes=b"JPEG", snapshot=snapshot)
    msg = _run(
        run_vision_query(
            "is my answer correct?",
            request_capture=_capture_returning({"status": "ready"}),
            vision_client=vc,
            backend_client=backend,
            session_context=SessionContext(slug="demo-slug", current_scene_id="scene-1"),
            room_id="room-1",
            api_url=None,
        )
    )
    mode, scene_context, mime, _img = vc.calls[0]
    assert mode == "assess"
    assert "mitochondria" in scene_context
    assert "powerhouse of the cell" in scene_context
    # answer surfaced as a developer message, no blind-spot note (annotated)
    assert msg["role"] == "developer"
    assert "Correct." in msg["content"]
    assert "NOT visible" not in msg["content"]


def test_run_vision_query_client_degrade_yields_unavailable_note():
    """vision_client returns the sentinel → an honest 'cannot see' note, not the sentinel."""
    vc = FakeVisionClient(answer=VISION_UNAVAILABLE)
    backend = FakeBackend(capture_bytes=b"JPEG")
    msg = _run(
        run_vision_query(
            "what's this?",
            request_capture=_capture_returning({"status": "ready"}),
            vision_client=vc,
            backend_client=backend,
            session_context=SessionContext(slug="demo-slug", current_scene_id="scene-1"),
            room_id="room-1",
            api_url=None,
        )
    )
    assert "cannot see the canvas" in msg["content"]


def test_run_vision_query_permission_point_fast_path():
    """point + permission-denied capture → ask for screen-share, skipping BOTH
    the Pillow fetch and the Gemini call (fast + honest)."""
    vc = FakeVisionClient(answer="must not be called")
    backend = FakeBackend(png_b64="must-not-be-fetched")
    msg = _run(
        run_vision_query(
            "what am I pointing at?",
            request_capture=_capture_returning({"status": "error", "error": "permission_required"}),
            vision_client=vc,
            backend_client=backend,
            session_context=SessionContext(slug="demo-slug", current_scene_id="scene-1"),
            room_id="room-1",
            api_url=None,
        )
    )
    assert vc.calls == []                          # Gemini skipped
    assert backend.get_vision_capture_calls == []  # no live-bytes fetch
    assert backend.scene_image_calls == 0          # no Pillow fetch either
    assert "share" in msg["content"].lower() and "screen" in msg["content"].lower()


def test_run_vision_query_describe_off_first_nudges_button():
    """describe + screen-share OFF, FIRST time → nudge to click the share button;
    NO Pillow, NO Gemini, and the describe-nudge flag gets set."""
    vc = FakeVisionClient(answer="must not be called")
    backend = FakeBackend(png_b64="must-not-be-fetched")
    sc = SessionContext(slug="demo-slug", current_scene_id="scene-1")
    msg = _run(
        run_vision_query(
            "what is on screen right now?",
            request_capture=_capture_returning({"status": "error", "error": "permission_required"}),
            vision_client=vc, backend_client=backend,
            session_context=sc, room_id="room-1", api_url=None,
        )
    )
    assert vc.calls == [] and backend.scene_image_calls == 0       # no Pillow, no Gemini
    assert "Let the Assistant see your screen" in msg["content"]   # names the button verbatim
    assert sc.get_describe_share_nudged() is True                 # flag set


def test_run_vision_query_describe_off_repeat_uses_pillow():
    """describe + screen-share OFF, REPEAT (already nudged) → base-scene Pillow + blind-spot."""
    import base64

    vc = FakeVisionClient(answer="The screen shows a video player.")
    backend = FakeBackend(png_b64=base64.b64encode(b"png").decode())
    sc = SessionContext(slug="demo-slug", current_scene_id="scene-1")
    sc.set_describe_share_nudged(True)  # already nudged this off-period
    msg = _run(
        run_vision_query(
            "what is on screen right now?",
            request_capture=_capture_returning({"status": "error", "error": "permission_required"}),
            vision_client=vc, backend_client=backend,
            session_context=sc, room_id="room-1", api_url=None,
        )
    )
    assert len(vc.calls) == 1 and vc.calls[0][2] == "image/png"   # Gemini on the base scene
    assert backend.scene_image_calls == 1                         # Pillow fetched
    assert "NOT visible" in msg["content"]                        # blind-spot note


def test_run_vision_query_describe_nudge_resets_on_ready():
    """A ready capture (screen-share back on) clears the nudge, so the NEXT
    off-period describe nudges again instead of going straight to Pillow."""
    sc = SessionContext(slug="demo-slug", current_scene_id="scene-1")
    perm = _capture_returning({"status": "error", "error": "permission_required"})

    # 1st describe while off → nudge (flag set).
    _run(run_vision_query("what is on screen?", request_capture=perm,
                          vision_client=FakeVisionClient(), backend_client=FakeBackend(png_b64="x"),
                          session_context=sc, room_id="room-1", api_url=None))
    assert sc.get_describe_share_nudged() is True

    # Screen-share back on → a ready capture resets the flag.
    _run(run_vision_query("what is on screen?", request_capture=_capture_returning({"status": "ready"}),
                          vision_client=FakeVisionClient(), backend_client=FakeBackend(capture_bytes=b"JPEG"),
                          session_context=sc, room_id="room-1", api_url=None))
    assert sc.get_describe_share_nudged() is False

    # Off again → first describe of the NEW off-period nudges again (no Pillow).
    be = FakeBackend(png_b64="must-not-be-fetched")
    msg = _run(run_vision_query("what is on screen?", request_capture=perm,
                                vision_client=FakeVisionClient(answer="x"), backend_client=be,
                                session_context=sc, room_id="room-1", api_url=None))
    assert be.scene_image_calls == 0 and "Let the Assistant see your screen" in msg["content"]


def test_is_permission_error_tolerates_both_shapes():
    """The detector catches status-level and error-level permission shapes."""
    from services.vision_query import _is_permission_error

    assert _is_permission_error({"status": "error", "error": "permission_required"})
    assert _is_permission_error({"status": "permission_denied"})
    assert not _is_permission_error({"status": "ready"})
    assert not _is_permission_error(None)


# ──────────────────────────────────────────────────────────────────────
# vision_state — deterministic visual-indicator signal around the Gemini call.
# ──────────────────────────────────────────────────────────────────────


def _state_recorder():
    """An on_vision_state hook that records the sequence of emitted states."""
    states: list[str] = []

    async def _rec(state):
        states.append(state)

    _rec.states = states
    return _rec


def test_run_vision_query_brackets_gemini_with_vision_state():
    """Live capture → on_vision_state fires 'analyzing' then 'idle' around Gemini."""
    vc = FakeVisionClient(answer="a video player")
    rec = _state_recorder()
    _run(
        run_vision_query(
            "what do you see?",
            request_capture=_capture_returning({"status": "ready"}),
            vision_client=vc, backend_client=FakeBackend(capture_bytes=b"JPEG"),
            session_context=SessionContext(slug="s", current_scene_id="x"),
            room_id="room-1", api_url=None, on_vision_state=rec,
        )
    )
    assert rec.states == ["analyzing", "idle"]   # bracketed the analyze call
    assert len(vc.calls) == 1                     # and the Gemini call happened


def test_run_vision_query_no_vision_state_on_fast_path():
    """Fast paths (here: point + permission denial → ask-share, no Gemini) never
    fire the indicator — so it can't flash without a real analyze."""
    rec = _state_recorder()
    _run(
        run_vision_query(
            "what am I pointing at?",
            request_capture=_capture_returning({"status": "error", "error": "permission_required"}),
            vision_client=FakeVisionClient(), backend_client=FakeBackend(),
            session_context=SessionContext(slug="s", current_scene_id="x"),
            room_id="room-1", api_url=None, on_vision_state=rec,
        )
    )
    assert rec.states == []   # no analyze → no indicator


def test_run_vision_query_vision_state_clears_on_degrade():
    """Even if Gemini degrades (sentinel), 'idle' still fires (try/finally)."""
    rec = _state_recorder()
    _run(
        run_vision_query(
            "what do you see?",
            request_capture=_capture_returning({"status": "ready"}),
            vision_client=FakeVisionClient(answer=VISION_UNAVAILABLE),
            backend_client=FakeBackend(capture_bytes=b"JPEG"),
            session_context=SessionContext(slug="s", current_scene_id="x"),
            room_id="room-1", api_url=None, on_vision_state=rec,
        )
    )
    assert rec.states == ["analyzing", "idle"]   # cleared despite the degrade


# ──────────────────────────────────────────────────────────────────────
# (6) VisionClient stub — no key → sentinel, and the SDK is never touched.
# ──────────────────────────────────────────────────────────────────────


def test_vision_client_stub_without_key_no_sdk_import():
    vc = VisionClient(api_key="")
    assert vc.enabled is False

    # If the stub path short-circuits correctly it returns BEFORE _ensure_client
    # (which performs `from google import genai`). Record any invocation: a
    # regression that fell through to the SDK would populate `touched`.
    touched: list = []
    vc._ensure_client = lambda: touched.append(1)

    out = _run(vc.analyze_image(b"\xff\xd8\xff", "describe"))
    assert out == VISION_UNAVAILABLE
    assert touched == []  # SDK never reached on the stub path


# ──────────────────────────────────────────────────────────────────────
# (7) Routing invariant — canvas_capture_result is an early-return BEFORE
#     the canvas dispatch in BOTH on_app_message handlers (real source).
# ──────────────────────────────────────────────────────────────────────


def test_canvas_capture_result_routed_before_canvas_dispatch():
    src = (Path(__file__).resolve().parent.parent / "bot.py").read_text()
    # Present in both pipelines (classic + relay).
    assert src.count('msg_type == "canvas_capture_result"') == 2
    # In each handler the branch sits before the canvas.register dispatch and
    # early-returns — so the canvas/relay dispatch never sees the message.
    # Anchor on the branch condition (not the bare string, which also appears
    # in request_canvas_capture's docstring).
    blocks = re.findall(r'msg_type == "canvas_capture_result"(.*?)canvas\.register', src, re.S)
    assert len(blocks) == 2, "capture branch must precede canvas.register in both handlers"
    for b in blocks:
        assert "return" in b, "capture branch must early-return before the canvas dispatch"


def test_canvas_capture_result_not_in_relay_payload_path():
    """Mirror of the on_app_message ordering: a capture_result short-circuits
    before the canvas/relay dispatch (which must never run for it)."""

    async def _route(message, *, pending, canvas_dispatch):
        if isinstance(message, str):
            try:
                message = json.loads(message)
            except (json.JSONDecodeError, ValueError):
                pass
        if not isinstance(message, dict):
            return "ignored"
        mt = message.get("type")
        if mt == "canvas_capture_result":
            _handle_capture_result_mirror(message, pending=pending)
            return "capture"
        await canvas_dispatch(message)
        return "dispatch"

    async def _drive():
        pending: dict = {}
        dispatched: list = []

        async def _canvas_dispatch(m):
            dispatched.append(m)

        # capture_result delivered as a JSON STRING (defensive json.loads) →
        # routed to the capture branch, dispatch never called.
        r1 = await _route(
            json.dumps({"type": "canvas_capture_result", "captureId": "x"}),
            pending=pending,
            canvas_dispatch=_canvas_dispatch,
        )
        # control: a canvas.* message DOES reach the dispatch.
        r2 = await _route(
            {"type": "canvas.register", "pageType": "quiz"},
            pending=pending,
            canvas_dispatch=_canvas_dispatch,
        )
        return r1, r2, dispatched

    r1, r2, dispatched = _run(_drive())
    assert r1 == "capture"
    assert r2 == "dispatch"
    assert dispatched == [{"type": "canvas.register", "pageType": "quiz"}]


# ──────────────────────────────────────────────────────────────────────
# (bonus) derive_vision_mode — utterance → reasoning mode (V6).
# ──────────────────────────────────────────────────────────────────────


def test_derive_vision_mode():
    assert derive_vision_mode("Is this correct?") == "assess"
    assert derive_vision_mode("did I get the answer right") == "assess"
    assert derive_vision_mode("what am I pointing at?") == "point"
    assert derive_vision_mode("what did I circle here") == "point"
    assert derive_vision_mode("tell me about this room") == "describe"
    assert derive_vision_mode("") == "describe"
