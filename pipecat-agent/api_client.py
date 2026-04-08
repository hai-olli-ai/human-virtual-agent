"""HTTP client for fetching data from api.hv.ai."""
import httpx
from loguru import logger

from config import HV_API_URL, HV_API_TOKEN


# ── Authenticated endpoints (existing — for direct avatar/scene fetch) ──

async def get_avatar(avatar_id: str) -> dict | None:
    """Fetch avatar with persona and knowledge from the API."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                f"{HV_API_URL}/avatars/{avatar_id}",
                headers={"Authorization": f"Bearer {HV_API_TOKEN}"},
            )
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logger.warning(f"Failed to fetch avatar {avatar_id}: {e}")
        return None


async def get_scene(scene_id: str) -> dict | None:
    """Fetch scene with elements from the API."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                f"{HV_API_URL}/scenes/{scene_id}",
                headers={"Authorization": f"Bearer {HV_API_TOKEN}"},
            )
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logger.warning(f"Failed to fetch scene {scene_id}: {e}")
        return None


# ── Public endpoints (Session 43 — no auth required) ──

async def get_avatar_config(room_id: str, api_url: str | None = None) -> dict | None:
    """Fetch structured avatar configuration for a live room.

    Uses GET /live-rooms/{room_id}/avatar-config (no auth).
    Returns dict with avatarId, name, thumbnailUrl, profilePhotoUrl,
    voiceRecordUrl, voiceModelId — or None on failure.
    """
    base_url = api_url or HV_API_URL
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                f"{base_url}/live-rooms/{room_id}/avatar-config",
            )
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logger.warning(f"Failed to fetch avatar config for room {room_id}: {e}")
        return None


async def get_persona_prompt(room_id: str, api_url: str | None = None) -> str | None:
    """Fetch the assembled persona system prompt for a live room.

    Uses GET /live-rooms/{room_id}/persona-prompt (no auth).
    Returns the full prompt string, or None on failure.
    """
    base_url = api_url or HV_API_URL
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(
                f"{base_url}/live-rooms/{room_id}/persona-prompt",
            )
            response.raise_for_status()
            data = response.json()
            return data.get("prompt")
    except Exception as e:
        logger.warning(f"Failed to fetch persona prompt for room {room_id}: {e}")
        return None


async def get_scene_snapshot(room_id: str, api_url: str | None = None) -> dict | None:
    """Fetch the current scene snapshot for a live room.

    Uses GET /live-rooms/{room_id}/scene-snapshot (no auth).
    Returns scene data with elements, instruction, display mode.
    """
    base_url = api_url or HV_API_URL
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                f"{base_url}/live-rooms/{room_id}/scene-snapshot",
            )
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logger.warning(f"Failed to fetch scene snapshot for room {room_id}: {e}")
        return None


async def get_scene_image_base64(room_id: str, api_url: str | None = None) -> str | None:
    """Fetch the rendered scene canvas as a base64-encoded PNG.

    Uses GET /live-rooms/{room_id}/scene-snapshot/image?format=base64 (no auth).
    Returns the base64 string (no data: prefix), or None on failure.
    """
    base_url = api_url or HV_API_URL
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(
                f"{base_url}/live-rooms/{room_id}/scene-snapshot/image",
                params={"format": "base64"},
            )
            response.raise_for_status()
            data = response.json()
            return data.get("image")
    except Exception as e:
        logger.warning(f"Failed to fetch scene image for room {room_id}: {e}")
        return None


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
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                f"{base_url}/live-rooms/{room_id}/navigate",
                json=body,
            )
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logger.warning(f"Failed to navigate room {room_id} ({direction}): {e}")
        return None
