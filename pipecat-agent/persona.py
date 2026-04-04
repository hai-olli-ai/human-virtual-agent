"""Build system prompts for the Pipecat voice agent.

Session 45: Uses Session 43's persona-prompt endpoint for the base prompt,
then enriches with scene snapshot data for instruction + display mode awareness.
"""
from loguru import logger

from api_client import (
    get_avatar,
    get_persona_prompt,
    get_scene,
    get_scene_snapshot,
)
from scene_context import (
    build_canvas_tools_section,
    build_instruction_section,
    build_scene_description,
)

# Fallback prompt when API is unavailable
DEFAULT_PROMPT = """## Your Identity
You are a friendly AI assistant for Human Virtual.

## Guidelines
- Speak naturally and conversationally
- Keep responses concise — this is a voice conversation, not a text chat
- Be warm and engaging — you're presenting content to a real person
- If asked something you don't know, say so honestly"""


async def build_system_prompt(
    room_id: str = "",
    avatar_id: str = "",
    scene_id: str = "",
    api_url: str | None = None,
) -> str:
    """Build the full system prompt for the voice agent.

    Strategy:
    1. If room_id is available, use the persona-prompt endpoint (includes everything)
    2. Supplement with scene snapshot data for instruction + display mode
    3. If no room_id, fall back to building from avatar + scene directly
    4. If everything fails, use a sensible default
    """
    prompt_parts = []

    # ── Strategy 1: Use persona-prompt endpoint (Session 43) ──
    if room_id:
        persona_prompt = await get_persona_prompt(room_id, api_url)
        if persona_prompt:
            logger.info(f"Loaded persona prompt from backend for room {room_id}")
            prompt_parts.append(persona_prompt)

            # The persona-prompt endpoint already includes scene context + instruction,
            # but we can still fetch the snapshot for enrichment
            snapshot = await get_scene_snapshot(room_id, api_url)
            if snapshot:
                # Add canvas tools section (for Session 47)
                tools = build_canvas_tools_section(snapshot)
                if tools:
                    prompt_parts.append(tools)

            return "\n\n".join(prompt_parts)

    # ── Strategy 2: Build locally from avatar + scene ──
    logger.info("Building prompt locally (no room_id or persona-prompt unavailable)")

    # Avatar persona
    if avatar_id:
        avatar = await get_avatar(avatar_id)
        if avatar:
            parts = [f"## Your Identity\nYou are {avatar.get('name', 'an AI assistant')}."]
            if avatar.get("persona"):
                parts.append(avatar["persona"])
            if avatar.get("gender"):
                parts.append(f"Gender: {avatar['gender']}")
            prompt_parts.append("\n".join(parts))

            if avatar.get("knowledge"):
                prompt_parts.append(f"## Your Knowledge\n{avatar['knowledge']}")

    # Scene context
    if room_id:
        snapshot = await get_scene_snapshot(room_id, api_url)
        if snapshot:
            prompt_parts.append(build_scene_description(snapshot))

            instruction = build_instruction_section(snapshot)
            if instruction:
                prompt_parts.append(instruction)

            tools = build_canvas_tools_section(snapshot)
            if tools:
                prompt_parts.append(tools)
    elif scene_id:
        scene = await get_scene(scene_id)
        if scene:
            prompt_parts.append(f"## Current Scene: {scene.get('title', 'Untitled')}")
            if scene.get("instruction"):
                prompt_parts.append(f"## Scene Instruction\n{scene['instruction']}")

    # Guidelines (always included)
    prompt_parts.append("""## Guidelines
- Speak naturally and conversationally
- Keep responses concise — this is a voice conversation, not a text chat
- Reference elements visible on the canvas when relevant
- If the visitor asks about something on the canvas, describe it
- If you're in a multi-scene flow, you can navigate between scenes when appropriate
- Be warm and engaging — you're presenting this content to a real person""")

    if not prompt_parts:
        return DEFAULT_PROMPT

    return "\n\n".join(prompt_parts)
