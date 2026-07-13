"""HTTP client for fetching data from api.hv.ai."""
import httpx
from loguru import logger

from config import HV_API_URL

# ── Shared connection pool (P3 2026-07-13) ──
# Every helper used to construct a fresh AsyncClient per call, paying a
# new TCP + TLS handshake each time (2-3 extra round trips per backend
# call — the agent makes 4-6 calls per scene change). One module-level
# client reuses keep-alive connections across all calls. httpx clients
# are asyncio-safe for concurrent requests; the pool is lazily created
# so import stays side-effect free.
_shared_client: httpx.AsyncClient | None = None


def get_shared_client() -> httpx.AsyncClient:
    global _shared_client
    if _shared_client is None or _shared_client.is_closed:
        _shared_client = httpx.AsyncClient(
            timeout=30.0,
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
        )
    return _shared_client


async def close_shared_client() -> None:
    """Optional teardown for tests / graceful shutdown."""
    global _shared_client
    if _shared_client is not None and not _shared_client.is_closed:
        await _shared_client.aclose()
    _shared_client = None


class BackendError(Exception):
    """Raised when a backend HTTP call returns a non-2xx status.

    Used by callers that must distinguish success from failure (e.g. tool
    handlers that surface the error to the LLM). The existing
    fetch-or-None helpers above intentionally swallow errors because they
    feed into prompt assembly where a missing snapshot just means "no
    enrichment"; this exception is for endpoints where the caller has to
    react to the failure.
    """


# ── Public endpoints (Session 43 — no auth required) ──

async def get_avatar_config(room_id: str, api_url: str | None = None) -> dict | None:
    """Fetch structured avatar configuration for a live room.

    Uses GET /live-rooms/{room_id}/avatar-config (no auth).
    Returns dict with avatarId, name, thumbnailUrl, profilePhotoUrl,
    voiceRecordUrl, voiceModelId — or None on failure.
    """
    base_url = api_url or HV_API_URL
    try:
        response = await get_shared_client().get(
            f"{base_url}/live-rooms/{room_id}/avatar-config",
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.warning(f"Failed to fetch avatar config for room {room_id}: {e!r}")
        return None


async def get_persona_prompt(
    room_id: str,
    api_url: str | None = None,
    scene_id: str | None = None,
) -> str | None:
    """Fetch the assembled persona system prompt for a live room.

    Uses GET /live-rooms/{room_id}/persona-prompt (no auth).
    Returns the full prompt string, or None on failure.

    P3 (2026-07-13) — ``scene_id`` targets the prompt at a specific scene
    (backend selector added alongside the S66 snapshot ones; cursor-safe).
    REQUIRED for the scene-change path: the shell advances the server
    cursor with a background 'goto' POST it does not await, so a
    cursor-relative persona fetch fired on canvas.sceneChanged can race
    the cursor commit and build the prompt from the OLD scene. No
    ``scene_id`` → legacy cursor-relative read (session start).
    """
    base_url = api_url or HV_API_URL
    params: dict[str, str] = {}
    if scene_id:
        params["scene_id"] = scene_id
    try:
        response = await get_shared_client().get(
            f"{base_url}/live-rooms/{room_id}/persona-prompt",
            params=params,
        )
        response.raise_for_status()
        data = response.json()
        return data.get("prompt")
    except Exception as e:
        logger.warning(f"Failed to fetch persona prompt for room {room_id}: {e!r}")
        return None


async def get_scene_snapshot(
    room_id: str,
    api_url: str | None = None,
    scene_id: str | None = None,
    scene_index: int | None = None,
) -> dict | None:
    """Fetch the current scene snapshot for a live room.

    Uses GET /live-rooms/{room_id}/scene-snapshot (no auth).
    Returns scene data with elements, instruction, display mode.

    Always passes ``include_all_scene_knowledge=true`` (S64c) so the
    snapshot's ``knowledge.flow`` payload aggregates every sibling
    scene's scene-scope knowledge alongside whatever is attached at
    the flow level. This keeps the agent's flow-knowledge prompt
    block stable across scene navigations and lets the agent answer
    from any scene's knowledge regardless of which scene is current.

    S66 Block 5c — when ``scene_id`` is provided, it's passed as
    ``?scene_id=…`` so the backend returns the snapshot for THAT scene
    specifically rather than the room's cursor-current scene. This
    eliminates the cursor race on canvas.sceneChanged. P2 (2026-07-13)
    made this LOAD-BEARING, not just an optimization: the shell now
    advances the cursor with a background 'goto' POST it does not
    await, so at broadcast time the cursor may still point at the OLD
    scene. A cursor-relative fallback (no scene_id) remains correct
    only in the eventual-consistency sense — prefer passing the id.
    """
    base_url = api_url or HV_API_URL
    params: dict[str, str] = {"include_all_scene_knowledge": "true"}
    if scene_id:
        params["scene_id"] = scene_id
    elif scene_index is not None:
        # P3 (2026-07-13) — by-index selector (S66), cursor-safe like
        # scene_id. The two are mutually exclusive on the backend;
        # scene_id wins when both are passed.
        params["scene_index"] = str(scene_index)
    try:
        response = await get_shared_client().get(
            f"{base_url}/live-rooms/{room_id}/scene-snapshot",
            params=params,
        )
        response.raise_for_status()
        data = response.json()
        # S65 (Option B) — snapshot nested under {live_room,
        # flow_state, current_scene, knowledge, survey}. Defensive
        # setdefaults so a pre-S65 backend (or a degraded response)
        # still yields a dict every consumer's `(snapshot.get("X")
        # or {}).get("Y")` chain can read without KeyError.
        data.setdefault("live_room", {})
        data.setdefault("flow_state", {})
        data.setdefault("current_scene", {})
        data["current_scene"].setdefault("scripts", [])
        return data
    except Exception as e:
        logger.warning(f"Failed to fetch scene snapshot for room {room_id}: {e!r}")
        return None


async def get_scene_image_base64(
    room_id: str,
    api_url: str | None = None,
    scene_id: str | None = None,
) -> str | None:
    """Fetch the rendered scene canvas as a base64-encoded PNG.

    Uses GET /live-rooms/{room_id}/scene-snapshot/image?format=base64 (no auth).
    Returns the base64 string (no data: prefix), or None on failure.

    P3 (2026-07-13) — ``scene_id`` renders THAT scene (S66 selector,
    cursor-safe). Pass it on scene-change refreshes: the shell's cursor
    advance is a background POST now, so a cursor-relative render can
    race it and return the OLD scene's image.
    """
    base_url = api_url or HV_API_URL
    params: dict[str, str] = {"format": "base64"}
    if scene_id:
        params["scene_id"] = scene_id
    try:
        response = await get_shared_client().get(
            f"{base_url}/live-rooms/{room_id}/scene-snapshot/image",
            params=params,
        )
        response.raise_for_status()
        data = response.json()
        return data.get("image")
    except Exception as e:
        logger.warning(f"Failed to fetch scene image for room {room_id}: {e!r}")
        return None


async def get_vision_capture(
    slug: str, capture_id: str, api_url: str | None = None
) -> bytes | None:
    """Fetch the JPEG bytes the shell uploaded for a vision capture (S67b).

    Uses GET /live-rooms/by-slug/{slug}/vision-capture/{capture_id} (no auth) —
    same public, no-auth style as get_scene_image_base64. The backend serves
    the raw image bytes from its short-TTL (~60s) Redis ingest and deletes on
    read, so a given capture_id is fetchable exactly once. Returns the raw
    JPEG bytes, or None on any failure (expired / already-read capture,
    backend down, network) — the caller degrades to the Pillow fallback +
    blind-spot flag on None.
    """
    base_url = api_url or HV_API_URL
    try:
        response = await get_shared_client().get(
            f"{base_url}/live-rooms/by-slug/{slug}/vision-capture/{capture_id}",
        )
        response.raise_for_status()
        return response.content
    except Exception as e:
        logger.warning(
            f"Failed to fetch vision capture {capture_id} for slug {slug}: {e!r}"
        )
        return None


async def generate_quiz(
    slug: str,
    scene_id: str,
    count: int = 3,
    language: str = "en",
    api_url: str | None = None,
) -> dict:
    """POST /live-rooms/by-slug/{slug}/scenes/{scene_id}/generate-quiz (S64e).

    Calls the backend's quiz-generation endpoint, which assembles the
    scene's knowledge_text and prompts Anthropic for a structured
    multiple-choice quiz blob. Returns the quiz blob dict on 200.

    Raises BackendError on non-2xx so the tool handler can hand the
    error back to the LLM. Timeout is 30s — the upstream LLM call
    typically takes 1-2s but allow headroom for cold starts and longer
    scenes.

    Public endpoint: the slug acts as the access token (matches the
    pattern the visitor frontend already uses for survey submission).
    """
    base_url = api_url or HV_API_URL
    url = f"{base_url}/live-rooms/by-slug/{slug}/scenes/{scene_id}/generate-quiz"
    payload = {"count": count, "language": language}

    response = await get_shared_client().post(url, json=payload)
    if response.status_code != 200:
        detail = response.text[:300]
        raise BackendError(
            f"generate_quiz failed: HTTP {response.status_code} {detail}"
        )
    return response.json()


async def navigate_scene(
    room_id: str,
    direction: str,
    target_index: int | None = None,
    api_url: str | None = None,
) -> dict | None:
    """Navigate to a different scene in the flow.

    Uses POST /live-rooms/{room_id}/navigate (no auth).
    Direction: "next", "previous", or "goto" (with target_index).
    """
    base_url = api_url or HV_API_URL
    body: dict = {"direction": direction}
    if target_index is not None:
        body["target_index"] = target_index

    try:
        response = await get_shared_client().post(
            f"{base_url}/live-rooms/{room_id}/navigate",
            json=body,
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.warning(f"Failed to navigate room {room_id} ({direction}): {e!r}")
        return None
