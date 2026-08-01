"""Build system prompts for the Pipecat voice agent.

Session 45: Uses Session 43's persona-prompt endpoint for the base prompt,
then enriches with scene snapshot data for instruction + display mode awareness.
"""

import asyncio

from loguru import logger

from api_client import (
    get_avatar_config,
    get_persona_prompt,
    get_scene_snapshot,
)
from flow_knowledge_cache import FlowKnowledgeCache
from scene_context import (
    KNOWLEDGE_PREAMBLE,
    build_canvas_tools_section,
    build_flow_knowledge_section,
    build_instruction_section,
    build_language_directive,
    build_language_reminder,
    build_link_narration_directive,
    build_recipient_context,
    build_scene_description,
    build_scene_knowledge_section,
    build_scripts_section,
    compute_element_aliases,
)

# Fallback prompt when API is unavailable
DEFAULT_PROMPT = """## Your Identity
You are a friendly AI assistant for Human Virtual.

## Guidelines
- Speak naturally and conversationally
- Keep responses concise — this is a voice conversation, not a text chat
- Be warm and engaging — you're presenting content to a real person
- If asked something you don't know, say so honestly"""


def _build_knowledge_block(
    snapshot: dict,
    flow_cache: FlowKnowledgeCache | None = None,
) -> str:
    """Build the knowledge section for the system prompt.

    Returns preamble + formatted knowledge, or "" when the snapshot has no
    usable knowledge. Emits one log line with snapshot metadata when content
    is injected — helps debug why the avatar does/doesn't know something.

    S66 Block 5b — when ``flow_cache`` is supplied, the FLOW-scope render is
    memoised across scene-change refreshes within a session; the SCENE-scope
    block is always rebuilt (it varies per navigation).
    """
    knowledge = snapshot.get("knowledge")
    if not knowledge:
        return ""

    if flow_cache is not None:
        flow_section = flow_cache.get_or_build(knowledge)
    else:
        flow_section = build_flow_knowledge_section(knowledge)
    scene_section = build_scene_knowledge_section(knowledge)
    parts = [s for s in (flow_section, scene_section) if s]
    if not parts:
        return ""
    knowledge_context = "\n\n".join(parts)

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
    aliases_out: dict[str, str] | None = None,
    flow_cache: FlowKnowledgeCache | None = None,
    snapshot_scene_id: str | None = None,
    snapshot: dict | None = None,
) -> str:
    """Build the full system prompt for the voice agent.

    When ``aliases_out`` is provided (mutable dict), it is cleared and
    populated with the snapshot's element-alias map (S64c) — the same
    mapping the prompt's "Available canvas elements" listing uses. The
    voice agent stores this on ``CanvasToolContext.element_alias_map``
    so the tool handlers can translate aliases back to real UUIDs
    before dispatching canvas commands. Pass ``None`` to skip alias
    population.

    Section order (S61 sandwich pattern + S63 Block 7):
      1. LANGUAGE directive            (top — strong steering)
      2. PERSONA + scene context       (varies by strategy)
      3. AUDIENCE                      (only when recipient_prompt is non-empty;
                                        injected between persona and knowledge)
      4. KNOWLEDGE                     (S56)
      5. LINK NARRATION                (S63 — only when snapshot.link is set)
      6. SCENE DESCRIPTION / INSTRUCTION
      7. CANVAS ACTION TOOL GUIDANCE
      8. SCRIPTS                       (when present)
      9. LANGUAGE reminder             (bottom — sandwich)

    Strategy:
    1. If room_id is available, use the persona-prompt endpoint (includes
       persona + scene context); supplement with snapshot enrichment.
    2. Otherwise, build locally from avatar + scene.
    3. If everything fails, fall back to DEFAULT_PROMPT.

    The snapshot is fetched at most once and reused for the LANGUAGE
    directive, the AUDIENCE section, and per-strategy enrichment. Pass
    ``snapshot`` to skip the fetch entirely (P3 2026-07-13 — callers
    that already hold the post-nav snapshot thread it through instead
    of paying a duplicate backend round trip).
    """
    # ── Snapshot fetched once; powers LANGUAGE + AUDIENCE + body ──
    # S65 (Option B) — snapshot nested under {live_room, flow_state,
    # current_scene, knowledge, survey}. Pull live_room + current_scene
    # blocks once so the rest of the function reads from local handles.
    # S66 Block 5c — only ``snapshot_scene_id`` (the broadcast scene_id
    # from canvas.sceneChanged) drives the by-id snapshot fetch; the
    # legacy ``scene_id`` param is intentionally NOT forwarded here
    # because in production it carries the Pipecat runner-args body's
    # scene_id — a hint from whoever started the agent, which for flow
    # rooms can be stale (the room's cursor moves; the body doesn't).
    # ``scene_id`` stays Strategy-2-only (avatar+scene path below).
    # P3 (2026-07-13) — snapshot and persona-prompt are independent
    # backend reads; fetch them CONCURRENTLY instead of serially (the
    # persona fetch used to wait for the snapshot round trip to finish).
    # The persona fetch targets the SAME scene as the snapshot: the
    # broadcast scene_id when present, else the (pre-fetched) snapshot's
    # own scene_id. Cursor-relative persona reads race the shell's
    # background cursor advance (P2) — the old-scene-prompt bug.
    persona_prompt: str | None = None
    if room_id:
        if snapshot is None:
            snapshot, persona_prompt = await asyncio.gather(
                get_scene_snapshot(
                    room_id, api_url, scene_id=snapshot_scene_id or None
                ),
                get_persona_prompt(
                    room_id, api_url, scene_id=snapshot_scene_id or None
                ),
            )
        else:
            persona_scene_id = (
                snapshot_scene_id
                or ((snapshot.get("current_scene") or {}).get("scene_id"))
                or None
            )
            persona_prompt = await get_persona_prompt(
                room_id,
                api_url,
                scene_id=str(persona_scene_id) if persona_scene_id else None,
            )

    live_room_block = (snapshot or {}).get("live_room") or {}
    current_scene_block = (snapshot or {}).get("current_scene") or {}

    language = live_room_block.get("language") or "en"
    audience_section = build_recipient_context(live_room_block.get("recipient_prompt"))

    # S64c — element aliases for the canvas tools section. Computed from
    # the same snapshot used for everything else, so the prompt's listing
    # and the agent's translation map are guaranteed in sync.
    element_aliases: dict[str, str] = {}
    if snapshot:
        element_aliases = compute_element_aliases(
            current_scene_block.get("elements") or []
        )
    if aliases_out is not None:
        aliases_out.clear()
        aliases_out.update(element_aliases)

    body_parts: list[str] = []

    # ── Strategy 1: Use persona-prompt endpoint (Session 43) ──
    # persona_prompt was fetched above (concurrently with the snapshot).
    if room_id:
        if persona_prompt:
            logger.info(f"Loaded persona prompt from backend for room {room_id}")
            body_parts.append(persona_prompt)

            # AUDIENCE — between persona and knowledge (S61)
            if audience_section:
                body_parts.append(audience_section.lstrip("\n"))

            if snapshot:
                # Knowledge section (S56) — after persona/audience, before tools
                knowledge_block = _build_knowledge_block(
                    snapshot, flow_cache=flow_cache
                )
                if knowledge_block:
                    body_parts.append(knowledge_block)

                # LINK NARRATION (S63 Block 7) — after KNOWLEDGE.
                # S65 (Option B) — link nested under current_scene.
                link_narration = build_link_narration_directive(
                    current_scene_block.get("link")
                )
                if link_narration:
                    body_parts.append(link_narration)

                # Add canvas tools section (for Session 47).
                # S64c — aliases are computed once above and threaded here so
                # the LLM-facing listing surfaces alias names instead of UUIDs.
                # build_canvas_tools_section + build_scripts_section read
                # the nested snapshot shape directly.
                tools = build_canvas_tools_section(snapshot, aliases=element_aliases)
                if tools:
                    body_parts.append(tools)

                # Add scripts section (for Session 49)
                scripts_section = build_scripts_section(snapshot)
                if scripts_section:
                    body_parts.append(scripts_section)

            return _wrap_language_sandwich(body_parts, language, audience_section)

    # ── Strategy 2: Build locally from public live-room data ──
    logger.info("Building prompt locally (no room_id or persona-prompt unavailable)")

    # Avatar identity — via the PUBLIC avatar-config endpoint. The old
    # path fetched the authenticated /avatars/{id} with a static
    # HV_API_TOKEN that silently expired 2026-04-04 (every call 401'd);
    # persona/knowledge depth comes from Strategy 1's persona-prompt —
    # this last-resort fallback keeps identity only, from data the agent
    # can actually reach.
    if room_id:
        avatar_config = await get_avatar_config(room_id, api_url)
        if avatar_config and avatar_config.get("name"):
            body_parts.append(f"## Your Identity\nYou are {avatar_config['name']}.")
    elif avatar_id:
        logger.info(
            "No room_id — avatar identity unavailable via public endpoints; skipping identity section"
        )

    # AUDIENCE — between persona and knowledge/scene (S61)
    if audience_section:
        body_parts.append(audience_section.lstrip("\n"))

    # Scene context (re-uses snapshot fetched at the top)
    if snapshot:
        # Knowledge section (S56) — between persona/audience and scene details
        knowledge_block = _build_knowledge_block(snapshot)
        if knowledge_block:
            body_parts.append(knowledge_block)

        # LINK NARRATION (S63 Block 7) — after KNOWLEDGE, before scene details.
        # S65 (Option B) — link nested under current_scene.
        link_narration = build_link_narration_directive(current_scene_block.get("link"))
        if link_narration:
            body_parts.append(link_narration)

        body_parts.append(build_scene_description(snapshot, aliases=element_aliases))

        instruction = build_instruction_section(snapshot)
        if instruction:
            body_parts.append(instruction)

        tools = build_canvas_tools_section(snapshot, aliases=element_aliases)
        if tools:
            body_parts.append(tools)

        scripts_section = build_scripts_section(snapshot)
        if scripts_section:
            body_parts.append(scripts_section)
    elif scene_id:
        # Pre-2026-07-10 this fetched the authenticated /scenes/{id}
        # (dead — expired token). There is no public per-scene endpoint
        # without a room; the snapshot fetched at the top is the only
        # scene source, and it's None on this branch by construction.
        logger.info("No snapshot available — scene context skipped in local prompt")

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
