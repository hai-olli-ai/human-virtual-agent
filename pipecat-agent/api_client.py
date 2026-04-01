"""HTTP client for fetching avatar/scene data from api.hv.ai."""
import httpx
from loguru import logger

from config import HV_API_URL, HV_API_TOKEN


async def get_avatar(avatar_id: str) -> dict:
    """Fetch avatar with persona and knowledge from the API."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(
            f"{HV_API_URL}/avatars/{avatar_id}",
            headers={"Authorization": f"Bearer {HV_API_TOKEN}"},
        )
        response.raise_for_status()
        return response.json()


async def get_scene(scene_id: str) -> dict:
    """Fetch scene with elements and scripts from the API."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(
            f"{HV_API_URL}/scenes/{scene_id}",
            headers={"Authorization": f"Bearer {HV_API_TOKEN}"},
        )
        response.raise_for_status()
        return response.json()


async def get_avatar_safe(avatar_id: str) -> dict | None:
    """Fetch avatar data, return None on failure (don't crash the agent)."""
    if not avatar_id:
        return None
    try:
        return await get_avatar(avatar_id)
    except Exception as e:
        logger.warning(f"Failed to fetch avatar {avatar_id}: {e}")
        return None


async def get_scene_safe(scene_id: str) -> dict | None:
    """Fetch scene data, return None on failure."""
    if not scene_id:
        return None
    try:
        return await get_scene(scene_id)
    except Exception as e:
        logger.warning(f"Failed to fetch scene {scene_id}: {e}")
        return None
