"""Build LLM system prompt from avatar persona and knowledge."""
from loguru import logger


def build_system_prompt(
    avatar: dict | None = None,
    scene: dict | None = None,
) -> str:
    """Build a system prompt from avatar persona/knowledge and scene context."""

    persona_section = _build_persona_section(avatar)
    knowledge_section = _build_knowledge_section(avatar)
    scene_section = _build_scene_section(scene)

    prompt = f"""You are an AI avatar presenting content to a visitor in a live interactive session.
Your responses will be spoken aloud via text-to-speech, so keep them conversational and natural.
Avoid special characters, markdown formatting, or overly long responses.

{persona_section}

{knowledge_section}

{scene_section}

## Vision
- You have been shown the scene's background image. You can reference what you see in it.
- You have a tool called `capture_what_i_see` that captures a live video frame.
- Use this tool when a visitor asks you to look at or describe what's currently on screen.

## Conversation Guidelines
- Speak naturally as if in a real conversation
- Keep responses concise (1-3 sentences for simple questions, longer for complex ones)
- Stay in character with the persona described above
- If asked about something outside your knowledge, acknowledge it honestly
- Be warm, engaging, and helpful
"""

    avatar_id = avatar.get("id") if avatar else None
    scene_id = scene.get("id") if scene else None
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

    if persona.get("purposes"):
        purposes = persona["purposes"]
        if isinstance(purposes, list):
            parts.append(f"Your purpose: {', '.join(purposes)}.")
        else:
            parts.append(f"Your purpose: {purposes}.")

    if persona.get("signaturePhrases"):
        phrases = persona["signaturePhrases"]
        if isinstance(phrases, list):
            parts.append(f"Your signature phrases (use them naturally): {', '.join(f'"{p}"' for p in phrases)}.")
        else:
            parts.append(f"Your signature phrase: \"{phrases}\".")

    if persona.get("relationshipToUser"):
        parts.append(f"Your relationship to the creator: {persona['relationshipToUser']}.")

    if persona.get("values"):
        values = persona["values"]
        if isinstance(values, list) and values:
            parts.append(f"Your values: {', '.join(values)}.")

    return "\n".join(parts)


def _build_knowledge_section(avatar: dict | None) -> str:
    """Build the knowledge portion of the system prompt."""
    if not avatar or not avatar.get("knowledge"):
        return ""

    knowledge = avatar["knowledge"]
    parts = ["## Your Knowledge"]

    if knowledge.get("background"):
        parts.append(f"Background: {knowledge['background']}")

    if knowledge.get("expertise"):
        expertise = knowledge["expertise"]
        if isinstance(expertise, list) and expertise:
            parts.append(f"Areas of expertise: {', '.join(expertise)}")
        elif expertise:
            parts.append(f"Areas of expertise: {expertise}")

    if knowledge.get("importantPeople"):
        people = knowledge["importantPeople"]
        if isinstance(people, list) and people:
            people_strs = [f"{p['name']} ({p.get('relationship', 'unknown')})" for p in people if p.get("name")]
            if people_strs:
                parts.append(f"Important people in your life: {', '.join(people_strs)}.")

    if knowledge.get("preferences"):
        parts.append(f"Preferences: {knowledge['preferences']}")

    if knowledge.get("memories"):
        memories = knowledge["memories"]
        if isinstance(memories, list) and memories:
            parts.append("Personal memories:")
            for m in memories:
                if isinstance(m, str):
                    parts.append(f"- {m}")
                elif isinstance(m, dict) and m.get("text"):
                    parts.append(f"- {m['text']}")

    if knowledge.get("notes"):
        notes = knowledge["notes"]
        if isinstance(notes, list) and notes:
            parts.append("Notes:")
            for n in notes:
                if isinstance(n, str):
                    parts.append(f"- {n}")
                elif isinstance(n, dict) and n.get("text"):
                    parts.append(f"- {n['text']}")

    return "\n".join(parts) if len(parts) > 1 else ""


def _build_scene_section(scene: dict | None) -> str:
    """Build the scene context portion of the system prompt."""
    if not scene:
        return ""

    parts = [f"## Current Scene: {scene.get('title', 'Untitled')}"]

    # Background info
    canvas = scene.get("canvasState", {})
    bg = canvas.get("background", {})
    if bg.get("label"):
        parts.append(f"Background: {bg['label']}")

    # Elements are nested inside canvasState
    elements = canvas.get("elements", [])
    if elements:
        parts.append("Elements on the canvas:")
        for el in elements:
            el_type = el.get("type", "unknown")
            display_mode = el.get("displayMode", "normal")

            if el_type == "avatar":
                # Tell the agent how the avatar is being displayed
                mode_desc = {
                    "normal": "shown as profile photo",
                    "invisible": "hidden from canvas (voice only)",
                    "3dgs": "displayed as 3D model",
                    "talking": "displayed as talking head with lip sync",
                }.get(display_mode, "shown as profile photo")
                parts.append(f"- Avatar ({mode_desc})")
            elif display_mode == "invisible":
                continue  # Skip invisible non-avatar elements
            else:
                props = el.get("properties", {})
                content = props.get("content", "")
                desc = f"- {el_type}"
                if content:
                    desc += f': "{content}"'
                parts.append(desc)

    # Scripts are top-level
    scripts = scene.get("scripts", [])
    if scripts:
        parts.append("\nScripts (dialogue):")
        for s in scripts:
            text = s.get("text", "")
            if text:
                parts.append(f'- "{text[:100]}{"..." if len(text) > 100 else ""}"')

    return "\n".join(parts)
