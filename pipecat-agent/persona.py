"""Build LLM system prompt from avatar persona and knowledge."""
from loguru import logger

from api_client import get_avatar_safe, get_scene_safe


async def build_system_prompt(
    avatar_id: str | None = None,
    scene_id: str | None = None,
) -> str:
    """Build a system prompt that gives the LLM the avatar's personality and scene context."""

    # Fetch avatar data
    avatar = await get_avatar_safe(avatar_id) if avatar_id else None
    scene = await get_scene_safe(scene_id) if scene_id else None

    # Build persona section
    persona_section = _build_persona_section(avatar)
    knowledge_section = _build_knowledge_section(avatar)
    scene_section = _build_scene_section(scene)

    prompt = f"""You are an AI avatar presenting content to a visitor in a live interactive session.
Your responses will be spoken aloud via text-to-speech, so keep them conversational and natural.
Avoid special characters, markdown formatting, or overly long responses.

{persona_section}

{knowledge_section}

{scene_section}

## Conversation Guidelines
- Speak naturally as if in a real conversation
- Keep responses concise (1-3 sentences for simple questions, longer for complex ones)
- Stay in character with the persona described above
- If asked about something outside your knowledge, acknowledge it honestly
- Be warm, engaging, and helpful
- When discussing the scene, reference specific elements you can see
"""

    logger.info(f"Built system prompt ({len(prompt)} chars) for avatar={avatar_id}, scene={scene_id}")
    return prompt


def _build_persona_section(avatar: dict | None) -> str:
    """Build the persona portion of the system prompt."""
    if not avatar or not avatar.get("persona"):
        return "## Your Identity\nYou are a friendly AI assistant."

    persona = avatar["persona"]
    parts = ["## Your Identity"]

    name = persona.get("avatarName") or avatar.get("name", "AI Assistant")
    parts.append(f"Your name is {name}.")

    if persona.get("gender"):
        parts.append(f"Gender: {persona['gender']}.")

    if persona.get("speakingLanguage"):
        parts.append(f"You speak in {persona['speakingLanguage']}.")

    if persona.get("shortDescription"):
        parts.append(f"About you: {persona['shortDescription']}")

    if persona.get("toneOfVoice"):
        tones = persona["toneOfVoice"]
        if isinstance(tones, list):
            parts.append(f"Your tone is: {', '.join(tones)}.")
        else:
            parts.append(f"Your tone is: {tones}.")

    return "\n".join(parts)


def _build_knowledge_section(avatar: dict | None) -> str:
    """Build the knowledge portion of the system prompt."""
    if not avatar or not avatar.get("knowledge"):
        return ""

    knowledge = avatar["knowledge"]
    parts = ["## Your Knowledge"]

    if knowledge.get("background"):
        parts.append(f"Background: {knowledge['background']}")

    if knowledge.get("areasOfExpertise"):
        expertise = knowledge["areasOfExpertise"]
        if isinstance(expertise, list):
            parts.append(f"Areas of expertise: {', '.join(expertise)}")
        else:
            parts.append(f"Areas of expertise: {expertise}")

    if knowledge.get("customInstructions"):
        parts.append(f"Special instructions: {knowledge['customInstructions']}")

    return "\n".join(parts) if len(parts) > 1 else ""


def _build_scene_section(scene: dict | None) -> str:
    """Build the scene context portion of the system prompt."""
    if not scene:
        return ""

    parts = [f"## Current Scene: {scene.get('title', 'Untitled')}"]

    elements = scene.get("elements", [])
    if elements:
        parts.append("Elements on the canvas:")
        for el in elements:
            el_type = el.get("type", "unknown")
            props = el.get("properties", {})
            content = props.get("content", "")
            desc = f"- {el_type}"
            if content:
                desc += f': "{content}"'
            parts.append(desc)

    scripts = scene.get("scripts", [])
    if scripts:
        parts.append("\nScripts (dialogue):")
        for s in scripts:
            text = s.get("text", "")
            if text:
                parts.append(f'- "{text[:100]}{"..." if len(text) > 100 else ""}"')

    return "\n".join(parts)
