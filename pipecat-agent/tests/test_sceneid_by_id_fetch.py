"""Tests for S66 Block 5c — sceneId broadcast + by-id snapshot fetch.

Layers under test:

  * :func:`api_client.get_scene_snapshot` — optional ``scene_id`` query
    param is forwarded when supplied; omitted (along with anything other
    than the always-on ``include_all_scene_knowledge=true``) when not.
  * :func:`persona.build_system_prompt` — Strategy 1 (live path) threads
    its ``snapshot_scene_id`` argument into the snapshot fetch so
    callers that received a ``sceneId`` on ``canvas.sceneChanged`` end
    up asking for that specific scene. The legacy ``scene_id`` arg is
    Strategy-2-only (avatar+scene fetch when no room_id) and must NOT
    drive the snapshot fetch — it carries Pipecat's runner-args body
    scene_id, which can be stale for flow rooms.

Follows the existing tests/ convention: no pytest-asyncio (not in the
dependency closure), so each async test goes through ``asyncio.run``.
"""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from unittest.mock import AsyncMock, patch

import api_client


def _run(coro):
    return asyncio.run(coro)


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _FakeAsyncClient:
    """Minimal httpx.AsyncClient stand-in that records ``.get`` calls.

    P3 (2026-07-13) — api_client now routes every call through a shared
    module-level client (``get_shared_client``), which checks
    ``is_closed`` before reuse; the fake mirrors that attribute and the
    harness below resets the module global so each test gets a fresh
    fake (and no fake leaks into other test files' real calls).
    """

    last_calls: list[tuple] = []
    is_closed = False

    def __init__(self, *args, **kwargs):
        self.init_kwargs = kwargs

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return None

    async def get(self, url, params=None):
        _FakeAsyncClient.last_calls.append((url, params))
        return _FakeResponse(
            {
                "live_room": {"language": "en"},
                "flow_state": {"scene_index": 0, "total_scenes": 1},
                "current_scene": {"scene_id": "scene-x"},
                "knowledge": None,
            }
        )


@contextmanager
def _patched_client():
    """Patch httpx.AsyncClient AND reset the shared-client global, both
    before (so the fake is actually constructed) and after (so the fake
    instance can't leak into other tests' real backend calls)."""
    _FakeAsyncClient.last_calls = []
    api_client._shared_client = None
    with patch.object(api_client.httpx, "AsyncClient", _FakeAsyncClient):
        try:
            yield
        finally:
            api_client._shared_client = None


# ──────────────────────────────────────────────────────────────────────
# api_client.get_scene_snapshot — query param threading
# ──────────────────────────────────────────────────────────────────────


def test_snapshot_without_scene_id_omits_param():
    with _patched_client():
        _run(api_client.get_scene_snapshot("room-1", api_url="http://x/api"))
    assert _FakeAsyncClient.last_calls, "get was not called"
    _, params = _FakeAsyncClient.last_calls[-1]
    assert params == {"include_all_scene_knowledge": "true"}


def test_snapshot_with_scene_id_includes_param():
    with _patched_client():
        _run(
            api_client.get_scene_snapshot(
                "room-1", api_url="http://x/api", scene_id="scene-abc"
            )
        )
    _, params = _FakeAsyncClient.last_calls[-1]
    assert params == {
        "include_all_scene_knowledge": "true",
        "scene_id": "scene-abc",
    }


def test_snapshot_with_none_scene_id_treated_as_omitted():
    """Explicit ``scene_id=None`` matches the legacy no-param call shape
    — the bot.py call sites pass ``scene_id=target_scene_id or None``
    so a falsy broadcast scene_id must not surface as an empty-string
    query param."""
    with _patched_client():
        _run(
            api_client.get_scene_snapshot(
                "room-1", api_url="http://x/api", scene_id=None
            )
        )
    _, params = _FakeAsyncClient.last_calls[-1]
    assert "scene_id" not in params


def test_snapshot_with_empty_string_scene_id_treated_as_omitted():
    with _patched_client():
        _run(
            api_client.get_scene_snapshot(
                "room-1", api_url="http://x/api", scene_id=""
            )
        )
    _, params = _FakeAsyncClient.last_calls[-1]
    assert "scene_id" not in params


# ──────────────────────────────────────────────────────────────────────
# persona.build_system_prompt — Strategy 1 threading
# ──────────────────────────────────────────────────────────────────────


def _mock_snapshot_fetch():
    """Build a stub get_scene_snapshot that records its call args."""
    return AsyncMock(
        return_value={
            "live_room": {"language": "en"},
            "flow_state": {"scene_index": 0, "total_scenes": 1},
            "current_scene": {"scene_id": "scene-y"},
            "knowledge": None,
        }
    )


def _snapshot_scene_id_seen(call) -> str | None:
    """Pull the scene_id forwarded to get_scene_snapshot from a call_args."""
    kwargs = call.kwargs
    args = call.args
    if "scene_id" in kwargs:
        return kwargs["scene_id"]
    # Positional fallback: get_scene_snapshot(room_id, api_url, scene_id)
    if len(args) >= 3:
        return args[2]
    return None


def test_build_system_prompt_threads_snapshot_scene_id_to_fetch():
    """``snapshot_scene_id`` is the by-id driver — it flows from
    canvas.sceneChanged's broadcast sceneId through refresh into the
    snapshot fetch."""
    import persona

    mock_get_snapshot = _mock_snapshot_fetch()
    mock_get_persona = AsyncMock(return_value=None)

    with patch.object(persona, "get_scene_snapshot", mock_get_snapshot), \
         patch.object(persona, "get_persona_prompt", mock_get_persona):
        _run(
            persona.build_system_prompt(
                room_id="room-1",
                avatar_id="",
                api_url="http://x/api",
                snapshot_scene_id="scene-target",
            )
        )

    mock_get_snapshot.assert_awaited_once()
    assert _snapshot_scene_id_seen(mock_get_snapshot.call_args) == "scene-target"


def test_build_system_prompt_legacy_scene_id_does_NOT_drive_snapshot_fetch():
    """The legacy ``scene_id`` arg carries Pipecat's runner-args body
    scene_id, which can be stale for flow rooms — forwarding it to the
    snapshot fetch produced 404s in prod (S66 Block 5c regression).
    The fix: ``scene_id`` is Strategy-2-only and never reaches the
    snapshot query param. This test is the regression guard."""
    import persona

    mock_get_snapshot = _mock_snapshot_fetch()
    mock_get_persona = AsyncMock(return_value=None)

    with patch.object(persona, "get_scene_snapshot", mock_get_snapshot), \
         patch.object(persona, "get_persona_prompt", mock_get_persona):
        _run(
            persona.build_system_prompt(
                room_id="room-1",
                avatar_id="",
                scene_id="body-scene-id-stale",  # body hint, not authoritative
                api_url="http://x/api",
                # snapshot_scene_id intentionally omitted
            )
        )

    mock_get_snapshot.assert_awaited_once()
    seen = _snapshot_scene_id_seen(mock_get_snapshot.call_args)
    assert seen in (None, ""), (
        f"legacy scene_id leaked into snapshot fetch — got {seen!r}; "
        "this is the prod 404 bug."
    )


def test_build_system_prompt_without_any_scene_id_passes_none():
    """No scene_id at all → cursor-based fetch (no query param)."""
    import persona

    mock_get_snapshot = _mock_snapshot_fetch()
    mock_get_persona = AsyncMock(return_value=None)

    with patch.object(persona, "get_scene_snapshot", mock_get_snapshot), \
         patch.object(persona, "get_persona_prompt", mock_get_persona):
        _run(
            persona.build_system_prompt(
                room_id="room-1",
                avatar_id="",
                api_url="http://x/api",
            )
        )

    mock_get_snapshot.assert_awaited_once()
    assert _snapshot_scene_id_seen(mock_get_snapshot.call_args) in (None, "")


def test_build_system_prompt_both_args_snapshot_wins():
    """When both legacy scene_id (body) and snapshot_scene_id
    (broadcast) are supplied, snapshot_scene_id wins for the snapshot
    fetch. This is the refresh path's call shape — bot.py passes both
    so Strategy 2 still works in tests/dev (no room_id) while the
    snapshot fetch uses the authoritative broadcast id."""
    import persona

    mock_get_snapshot = _mock_snapshot_fetch()
    mock_get_persona = AsyncMock(return_value=None)

    with patch.object(persona, "get_scene_snapshot", mock_get_snapshot), \
         patch.object(persona, "get_persona_prompt", mock_get_persona):
        _run(
            persona.build_system_prompt(
                room_id="room-1",
                avatar_id="",
                scene_id="body-scene-id",
                snapshot_scene_id="broadcast-scene-id",
                api_url="http://x/api",
            )
        )

    mock_get_snapshot.assert_awaited_once()
    assert _snapshot_scene_id_seen(mock_get_snapshot.call_args) == "broadcast-scene-id"
