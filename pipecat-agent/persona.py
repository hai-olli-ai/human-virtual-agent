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
    KNOWLEDGE_PREAMBLE,
    build_canvas_tools_section,
    build_instruction_section,
    build_knowledge_context,
    build_language_directive,
    build_language_reminder,
    build_recipient_context,
    build_scene_description,
    build_scripts_section,
)

# Fallback prompt when API is unavailable
DEFAULT_PROMPT = """## Your Identity
You are a friendly AI assistant for Human Virtual.

## Guidelines
- Speak naturally and conversationally
- Keep responses concise — this is a voice conversation, not a text chat
- Be warm and engaging — you're presenting content to a real person
- If asked something you don't know, say so honestly"""


def _build_knowledge_block(snapshot: dict) -> str:
    """Build the knowledge section for the system prompt.

    Returns preamble + formatted knowledge, or "" when the snapshot has no
    usable knowledge. Emits one log line with snapshot metadata when content
    is injected — helps debug why the avatar does/doesn't know something.
    """
    knowledge = snapshot.get("knowledge")
    knowledge_context = build_knowledge_context(knowledge)
    if not knowledge_context:
        return ""

    logger.info(
        "Knowledge injected into system prompt: total_chars={tc}, budget_exceeded={be}, scene_sources={ss}, flow_sources={fs}",
        tc=knowledge.get("total_chars", 0),
        be=knowledge.get("budget_exceeded", False),
        ss=len((knowledge.get("scene") or {}).get("sources") or []),
        fs=len((knowledge.get("flow") or {}).get("sources") or []),
    )
    return f"{KNOWLEDGE_PREAMBLE}\n{knowledge_context}"


async def build_system_prompt(
    room_id: str = "",
    avatar_id: str = "",
    scene_id: str = "",
    api_url: str | None = None,
) -> str:
    """Build the full system prompt for the voice agent.

    Section order (S61 sandwich pattern):
      1. LANGUAGE directive            (top — strong steering)
      2. PERSONA + scene context       (varies by strategy)
      3. AUDIENCE                      (only when recipient_prompt is non-empty;
                                        injected between persona and knowledge)
      4. KNOWLEDGE                     (S56)
      5. SCENE DESCRIPTION / INSTRUCTION
      6. CANVAS ACTION TOOL GUIDANCE
      7. SCRIPTS                       (when present)
      8. LANGUAGE reminder             (bottom — sandwich)

    Strategy:
    1. If room_id is available, use the persona-prompt endpoint (includes
       persona + scene context); supplement with snapshot enrichment.
    2. Otherwise, build locally from avatar + scene.
    3. If everything fails, fall back to DEFAULT_PROMPT.

    The snapshot is fetched at most once and reused for the LANGUAGE
    directive, the AUDIENCE section, and per-strategy enrichment.
    """
    # ── Snapshot fetched once; powers LANGUAGE + AUDIENCE + body ──
    snapshot: dict | None = None
    if room_id:
        snapshot = await get_scene_snapshot(room_id, api_url)

    language = (snapshot or {}).get("language") or "en"
    audience_section = build_recipient_context((snapshot or {}).get("recipient_prompt"))

    body_parts: list[str] = []

    # ── Strategy 1: Use persona-prompt endpoint (Session 43) ──
    if room_id:
        persona_prompt = await get_persona_prompt(room_id, api_url)
        if persona_prompt:
            logger.info(f"Loaded persona prompt from backend for room {room_id}")
            body_parts.append(persona_prompt)

            # AUDIENCE — between persona and knowledge (S61)
            if audience_section:
                body_parts.append(audience_section.lstrip("\n"))

            if snapshot:
                # Knowledge section (S56) — after persona/audience, before tools
                knowledge_block = _build_knowledge_block(snapshot)
                if knowledge_block:
                    body_parts.append(knowledge_block)

                # Add canvas tools section (for Session 47)
                tools = build_canvas_tools_section(snapshot)
                if tools:
                    body_parts.append(tools)

                # Add scripts section (for Session 49)
                scripts_section = build_scripts_section(snapshot)
                if scripts_section:
                    body_parts.append(scripts_section)

            return _wrap_language_sandwich(body_parts, language, audience_section)

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
            body_parts.append("\n".join(parts))

            if avatar.get("knowledge"):
                body_parts.append(f"## Your Knowledge\n{avatar['knowledge']}")

    # AUDIENCE — between persona and knowledge/scene (S61)
    if audience_section:
        body_parts.append(audience_section.lstrip("\n"))

    # Scene context (re-uses snapshot fetched at the top)
    if snapshot:
        # Knowledge section (S56) — between persona/audience and scene details
        knowledge_block = _build_knowledge_block(snapshot)
        if knowledge_block:
            body_parts.append(knowledge_block)

        body_parts.append(build_scene_description(snapshot))

        instruction = build_instruction_section(snapshot)
        if instruction:
            body_parts.append(instruction)

        tools = build_canvas_tools_section(snapshot)
        if tools:
            body_parts.append(tools)

        scripts_section = build_scripts_section(snapshot)
        if scripts_section:
            body_parts.append(scripts_section)
    elif scene_id:
        scene = await get_scene(scene_id)
        if scene:
            body_parts.append(f"## Current Scene: {scene.get('title', 'Untitled')}")
            if scene.get("instruction"):
                body_parts.append(f"## Scene Instruction\n{scene['instruction']}")

    # Guidelines (always included)
    body_parts.append("""## Guidelines
- Speak naturally and conversationally
- Keep responses concise — this is a voice conversation, not a text chat
- Reference elements visible on the canvas when relevant
- If the visitor asks about something on the canvas, describe it
- If you're in a multi-scene flow, you can navigate between scenes when appropriate
- Be warm and engaging — you're presenting this content to a real person""")

    if not body_parts:
        body_parts = [DEFAULT_PROMPT]

    return _wrap_language_sandwich(body_parts, language, audience_section)


def _wrap_language_sandwich(
    body_parts: list[str], language: str, audience_section: str
) -> str:
    """Wrap the body with the LANGUAGE directive (top) + reminder (bottom).

    Logs a structured summary of the assembled prompt's shape so we can
    debug later why an avatar did or didn't pick up the language /
    audience steering for a given session.
    """
    sections = [
        f"# LANGUAGE\n{build_language_directive(language)}",
        *body_parts,
        build_language_reminder(language),
    ]
    prompt = "\n\n".join(sections)
    logger.info(
        "System prompt assembled: language={} audience_present={} body_sections={} prompt_chars={}",
        language,
        bool(audience_section),
        len(body_parts),
        len(prompt),
    )
    return prompt
