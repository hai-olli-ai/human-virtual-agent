"""Process scene snapshots into agent-readable context.

Takes a scene snapshot dict from the API and builds descriptive text
for the LLM system prompt.
"""
from loguru import logger
from typing import Any

KNOWLEDGE_PREAMBLE = (
    "You have access to the following knowledge base. When visitors ask "
    "questions, prefer answers grounded in this knowledge. If the visitor "
    "asks something not covered here, you can answer from general knowledge, "
    "but mention when you're outside the provided context."
)


# ──────────────────────────────────────────────────────────────────────
# Language directives (Session 61)
# Sandwich pattern: directive at the top of the system prompt + a short
# reminder at the bottom. LLMs weight the first and last sections most
# heavily, so this is materially more drift-resistant than top-only.
# ──────────────────────────────────────────────────────────────────────

LANGUAGE_NAMES: dict[str, str] = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "pt": "Portuguese",
    "ja": "Japanese",
    "ko": "Korean",
    "vi": "Vietnamese",
    "zh": "Chinese (Mandarin)",
}


def build_language_directive(language: str | None) -> str:
    """Top-of-prompt directive: 'always speak in {language}'.

    The backend's CHECK constraint and Pydantic Literal both prevent unknown
    values from reaching us, so the English fallback here is purely
    belt-and-suspenders.
    """
    name = LANGUAGE_NAMES.get(language or "en", "English")
    return (
        f"You are speaking in {name}. "
        f"Always respond in {name} regardless of the language the visitor uses. "
        f"If the visitor speaks a different language, gently continue in {name}."
    )


def build_language_reminder(language: str | None) -> str:
    """Bottom-of-prompt language reminder. Short and emphatic."""
    name = LANGUAGE_NAMES.get(language or "en", "English")
    return f"Remember: respond in {name}."


# ──────────────────────────────────────────────────────────────────────
# Recipient prompt / audience steering (Session 61)
# Steering, not knowledge — describes WHO the avatar is talking to, not
# WHAT it knows. Empty/whitespace-only prompts produce no section, which
# is the signal "general visitors".
# ──────────────────────────────────────────────────────────────────────

RECIPIENT_PREAMBLE = (
    "This live conversation is addressed to a specific audience. "
    "Tailor your tone, vocabulary, and emphasis to this audience "
    "throughout the conversation."
)


def build_recipient_context(recipient_prompt: str | None) -> str:
    """The AUDIENCE section, or "" when no recipient_prompt was provided.

    Returns a string starting with a leading newline so it slots cleanly
    when concatenated; call sites that join with "\\n\\n" should strip the
    leading newline (see persona.build_system_prompt).
    """
    if not recipient_prompt or not recipient_prompt.strip():
        return ""
    return f"\n# AUDIENCE\n{RECIPIENT_PREAMBLE}\n\n{recipient_prompt.strip()}"


# ──────────────────────────────────────────────────────────────────────
# Link narration directive (Session 63, Block 7)
# Tells the LLM HOW to use linked content already injected into the
# KNOWLEDGE section. Sits after KNOWLEDGE, before SCENE INSTRUCTION.
# ──────────────────────────────────────────────────────────────────────

NARRATION_MODE_DIRECTIVES: dict[str, str] = {
    "walk_through": (
        "Walk the visitor through the linked content step-by-step. "
        "Surface the key points in narrative order. Pause for questions "
        "after each major section."
    ),
    "summarize": (
        "Summarize the key points of the linked content concisely when "
        "it becomes relevant. Don't enumerate everything — pick the most "
        "salient 2-3 points and offer to go deeper if asked."
    ),
    "answer_questions": (
        "Reference the linked content reactively, only when the visitor "
        "asks about it. Don't volunteer details unprompted."
    ),
    "reference_as_needed": (
        "Treat the linked content as background. Mention it only when "
        "directly relevant to the visitor's question. Otherwise, don't "
        "reference it."
    ),
}


def build_link_narration_directive(link: dict | None) -> str:
    """Return a system-prompt section that tells the LLM how to use the
    linked content. Returns an empty string when no link is set or the
    narration_mode is unknown.

    Place AFTER the KNOWLEDGE section (S56) and BEFORE SCENE INSTRUCTION
    in the system-prompt sandwich. Defensive against stale snapshots that
    reference modes added in a future session — unknown modes drop out.
    """
    if not link:
        return ""
    mode = link.get("narration_mode", "walk_through")
    directive = NARRATION_MODE_DIRECTIVES.get(mode)
    if not directive:
        return ""

    source = link.get("source", "linked content")
    url = link.get("url", "")

    return (
        "# LINK NARRATION\n"
        f"The creator has attached a {source} link to this scene "
        f"({url}). Knowledge from this link has been included in the "
        f"KNOWLEDGE section above.\n"
        f"How to use it: {directive}"
    )


def _format_scope(scope_data: dict[str, Any] | None, scope_label: str) -> str:
    """Format one knowledge scope (scene or flow) into a markdown section.
    Returns empty string if scope is None or has no content.

    Priority order within a scope: FAQ → Documents → URLs.
    (FAQ first because it's curated and highest-signal.)
    """
    if not scope_data:
        return ""

    parts: list[str] = []

    faqs = scope_data.get("faqs") or []
    if faqs:
        faq_lines = ["## FAQ"]
        for faq in faqs:
            q = (faq.get("question") or "").strip()
            a = (faq.get("answer") or "").strip()
            if not q or not a:
                continue
            faq_lines.append(f"Q: {q}")
            faq_lines.append(f"A: {a}")
            faq_lines.append("")
        if len(faq_lines) > 1:
            parts.append("\n".join(faq_lines))

    sources = scope_data.get("sources") or []
    for src in sources:
        text = (src.get("extracted_text") or "").strip()
        if not text:
            continue
        name = src.get("file_name") or "document"
        parts.append(f"## Document: {name}\n{text}")

    urls = scope_data.get("urls") or []
    for url in urls:
        text = (url.get("markdown_content") or "").strip()
        if not text:
            continue
        header = (url.get("title") or "").strip() or url.get("url") or "web page"
        parts.append(f"## Web Page: {header}\n{text}")

    if not parts:
        return ""

    return f"\n# {scope_label} KNOWLEDGE\n\n" + "\n\n---\n\n".join(parts)


def build_knowledge_context(knowledge: dict[str, Any] | None) -> str:
    """Format the snapshot's knowledge dict into a system-prompt section.

    Args:
      knowledge: The `knowledge` object from scene-snapshot, or None.

    Shape:
      {
        "scene": { "sources": [...], "urls": [...], "faqs": [...] } | None,
        "flow":  { ... same shape ... } | None,
        "budget_exceeded": bool,
        "total_chars": int,
      }

    Returns:
      A markdown string with FLOW section first (broader context), SCENE
      section second (more specific). Empty string when no usable knowledge
      is present. Never raises — defensive against missing keys.
    """
    if not knowledge:
        return ""

    sections: list[str] = []

    # FLOW first — broader, applies across scenes
    flow_scope = knowledge.get("flow")
    if flow_scope:
        flow_str = _format_scope(flow_scope, "FLOW")
        if flow_str:
            sections.append(flow_str)

    # SCENE second — specific to this scene
    scene_scope = knowledge.get("scene")
    if scene_scope:
        scene_str = _format_scope(scene_scope, "SCENE")
        if scene_str:
            sections.append(scene_str)

    return "\n\n".join(sections)


VISION_MESSAGE = "This is the current scene canvas that the visitor is seeing. The canvas is 1280x720 pixels (origin top-left). Remember the layout, colors, positions, and content of all elements. When discussing the scene, reference what you see in this image. When using canvas action tools (highlight, arrow, annotation), estimate pixel coordinates from this image."

def build_vision_message(image_base64: str) -> dict:
    """Build an OpenAI-format user message with a canvas image for vision.

    This message is added to the LLM context so the model can "see" the canvas.
    """
    return {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{image_base64}",
                    "detail": "high",
                },
            },
            {
                "type": "text",
                "text": VISION_MESSAGE,
            },
        ],
    }

# ──────────────────────────────────────────────────────────────────────
# Element aliases (S64c)
# ──────────────────────────────────────────────────────────────────────
#
# Elements come from the backend with UUID7-shaped ids — long, opaque
# strings that often share long prefixes (because UUID7 is timestamp-
# ordered). LLMs fail to copy these reliably as `target.element_id` for
# canvas_highlight or `from`/`to` for canvas_action(verb=draw_arrow):
# the model picks the right element conceptually but mis-types the id
# mid-stream, which surfaces as "highlighted the wrong element".
#
# The alias layer fixes this without changing the wire protocol. We
# derive a short, distinctive name per element (e.g. text_1, avatar_1,
# emoji_2) and surface those in the prompt instead of the raw UUIDs.
# bot.py stores the alias→UUID map on canvas_ctx; the tool handlers
# translate aliases back to real UUIDs before sending the canvas.command
# to the frontend. The LLM only ever needs to copy the short alias.

def compute_element_aliases(elements: list[dict]) -> dict[str, str]:
    """Return ``{alias: element_id}`` for the given elements list.

    Aliases are formatted ``<type>_<ordinal>`` (1-based, scoped per type),
    so a scene with one text + one avatar + two emojis produces:
    ``{"text_1": "...", "avatar_1": "...", "emoji_1": "...", "emoji_2": "..."}``.

    Ordering is the elements list's natural order, which matches the
    visual stacking the creator intended. Deterministic — given the same
    list, the alias mapping is stable across calls. Elements without an
    ``id`` are skipped.
    """
    counts: dict[str, int] = {}
    aliases: dict[str, str] = {}
    for el in elements:
        eid = el.get("id")
        if not eid:
            continue
        et = (el.get("type") or "unknown").lower()
        counts[et] = counts.get(et, 0) + 1
        aliases[f"{et}_{counts[et]}"] = eid
    return aliases


def _summarize_element(el: dict) -> str:
    """Short human-readable description of an element for the alias list.

    The goal is to give the LLM enough to disambiguate when the visitor
    refers to "the title text" or "the sign-up button". Falls back to
    bare type when no descriptive content is available.
    """
    et = el.get("type") or "unknown"
    if et == "text" and el.get("text"):
        return f'text "{el["text"]}"'
    if et == "button" and el.get("title"):
        return f'button "{el["title"]}"'
    if et == "emoji" and el.get("emoji_character"):
        return f'emoji {el["emoji_character"]}'
    if el.get("label"):
        return f"{et} ({el['label']})"
    return et


def build_scene_description(snapshot: dict, aliases: dict[str, str] | None = None) -> str:
    """Build a human-readable scene description from a snapshot.

    The snapshot comes from GET /live-rooms/{room_id}/scene-snapshot.

    When ``aliases`` (alias → element_id) is provided, each element's
    ``[id: …]`` annotation in the prompt uses the short alias instead of
    the raw UUID. See ``compute_element_aliases`` for why.
    """
    if not snapshot:
        return "No scene is currently loaded."

    parts = []

    title = snapshot.get("title", "Untitled Scene")
    parts.append(f"## Current Scene: {title}")

    # Background
    bg_url = snapshot.get("background_url")
    bg_type = snapshot.get("background_type")
    if bg_url:
        parts.append(f"Background: {bg_type or 'image'}")

    # Avatar display mode
    display_mode = snapshot.get("avatar_display_mode", "normal")
    parts.append(f"Avatar display mode: {display_mode}")

    if display_mode == "invisible":
        parts.append("Note: You are in voice-only mode. The visitor cannot see you, only hear you. Focus entirely on verbal communication.")
    elif display_mode == "talking":
        parts.append("Note: You are rendered as a talking avatar with lip sync. The visitor can see your face moving as you speak.")
    elif display_mode == "3dgs":
        parts.append("Note: You are rendered as a 3D model. The visitor sees a 3D representation of you.")

    # Canvas elements
    elements = snapshot.get("elements", [])
    if elements:
        parts.append("\nElements on the canvas:")
        # Note (S59): Element type "button" is a visitor-clickable CTA on
        # the scene canvas. The agent has no canvas-click tool — it should
        # describe buttons by their `title` (e.g. "the 'Sign up now' button
        # in the lower right") and may use canvas_highlight to point at one,
        # but it must NOT attempt to "click" buttons on the visitor's behalf.
        # Buttons are part of the visual scene; clicks are exclusively the
        # visitor's affordance.
        # Reverse map for O(1) UUID → alias lookup when aliases are provided.
        uuid_to_alias = {uuid: alias for alias, uuid in (aliases or {}).items()}

        for el in elements:
            el_type = el.get("type", "unknown")
            desc = f"- {el_type}"
            # S64c — surface the element id (or its short alias when
            # available) so the LLM can pass it as
            # `target={element_id: "..."}` to canvas_highlight.
            eid = el.get("id")
            if eid:
                display_id = uuid_to_alias.get(eid, eid)
                desc += f" [id: {display_id}]"

            if el.get("text"):
                desc += f': "{el["text"]}"'
            if el.get("label"):
                desc += f' (label: {el["label"]})'
            if el.get("title"):
                desc += f' (title: {el["title"]})'
            # S64c (Option 2) — surface emoji_character so the LLM can tell
            # one emoji from another, and so emoji elements aren't a
            # content-less line in the prompt that the LLM might mistake
            # for "the title".
            if el.get("emoji_character"):
                desc += f' (emoji: {el["emoji_character"]})'
            if el.get("display_mode"):
                desc += f" [display: {el['display_mode']}]"

            # Position info for canvas actions (Session 47)
            pos = el.get("position", {})
            size = el.get("size", {})
            if pos and size:
                desc += f" at ({pos.get('x', 0)}, {pos.get('y', 0)}), size {size.get('width', 0)}x{size.get('height', 0)}"

            parts.append(desc)

    # Flow position
    total = snapshot.get("total_scenes", 1)
    if total > 1:
        index = snapshot.get("scene_index", 0)
        parts.append(f"\nThis is scene {index + 1} of {total} in a multi-scene flow.")
        parts.append("You can navigate between scenes when appropriate by using the navigate_scene tool.")

    return "\n".join(parts)


def build_instruction_section(snapshot: dict) -> str:
    """Build the scene instruction section from a snapshot."""
    instruction = snapshot.get("instruction")
    if not instruction:
        return ""

    return f"""## Scene Instruction
Follow these specific instructions for this scene:
{instruction}"""


def build_scripts_section(snapshot: dict) -> str:
    """Build the scripts section from a snapshot.

    Gives the LLM awareness of script content so it can reference it
    during conversation without repeating it verbatim.
    """
    scripts = snapshot.get("scripts", [])
    if not scripts:
        return ""

    sorted_scripts = sorted(scripts, key=lambda s: s.get("order", 0))
    lines = []
    for i, script in enumerate(sorted_scripts, 1):
        text = script.get("text", "").strip()
        if text:
            lines.append(f"{i}. {text}")

    if not lines:
        return ""

    return "Scene Scripts (you will present these to the visitor via TTS before conversation begins):\n" + "\n".join(lines)


def build_canvas_tools_section(
    snapshot: dict,
    aliases: dict[str, str] | None = None,
) -> str:
    """Build the canvas action tools description for the system prompt.

    Describes the 5 generic ``canvas_*`` tools available on the active
    Canvas Page. The dynamic CANVAS PAGE section in
    context.prompt_builder eventually subsumes this once bot.py is
    rewired to call build_system_prompt_split.

    When ``aliases`` (alias → element_id) is provided, the "Available
    canvas elements" listing surfaces short, distinctive aliases
    (e.g. ``text_1``, ``avatar_1``) instead of UUID7 ids. The LLM uses
    these aliases as ``target.element_id`` (highlight) or
    ``args.from`` / ``args.to`` (draw_arrow); the tool handlers in
    ``tools/canvas_protocol_tools.py`` translate aliases back to real
    UUIDs before sending the canvas.command. See
    ``compute_element_aliases`` for the rationale.
    """
    total = snapshot.get("total_scenes", 1)
    control_verbs = "next_scene, previous_scene, goto_scene, clear" if total > 1 else "clear"

    # Element listing: when aliases are provided, render alias + short
    # description so the LLM has a complete authoritative reference. Strategy
    # 1 in persona.build_system_prompt skips build_scene_description, so this
    # section is the only reliable place to surface element ids on the live
    # path.
    elements = snapshot.get("elements") or []
    if aliases:
        # Stable iteration order: follow the elements list (which matches the
        # creator's stacking) rather than the dict's insertion order — they
        # should be the same since compute_element_aliases iterates elements
        # in order, but using `elements` keeps it tied to the source of truth.
        uuid_to_alias = {uuid: alias for alias, uuid in aliases.items()}
        listed: list[str] = []
        for el in elements:
            eid = el.get("id")
            if not eid:
                continue
            alias = uuid_to_alias.get(eid)
            if not alias:
                continue
            listed.append(f"  - `{alias}` — {_summarize_element(el)}")
        if listed:
            ids_line = (
                "- Available canvas elements (pass these aliases as "
                "`element_id` for canvas_highlight, and as `from` / `to` "
                "for canvas_action with verb=draw_arrow):\n"
                + "\n".join(listed)
            )
        else:
            ids_line = (
                "- No canvas elements are available on this scene — "
                "`canvas_highlight` and `canvas_action(verb='draw_arrow')` cannot be called."
            )
    else:
        # Fallback path: aliases not wired (sync test builder, older callers).
        # Surface raw UUIDs so the existing behavior is preserved.
        element_ids = [el.get("id") for el in elements if el.get("id")]
        if element_ids:
            ids_line = "- Available element ids: " + ", ".join(f"`{eid}`" for eid in element_ids) + "."
        else:
            ids_line = "- No element ids are available on this scene — `canvas_highlight` cannot be called."

    return "\n".join([
        "## Canvas Actions",
        "",
        "**Use ONLY the 5 tools below for any canvas interaction.**",
        "",
        "1. `canvas_analyze(question, options={})` — answer a question about what is "
        "visible on the canvas using the active page's semantic state. Use when the "
        "visitor asks something you cannot determine from your existing context.",
        "",
        "2. `canvas_highlight(target, options={})` — draw a highlight on the canvas. "
        "`target` MUST be `{element_id: \"<id>\"}` using one of the ids listed in Notes "
        "below. Box-coordinate targets (`{box: [x, y, w, h]}`) are NOT supported on the "
        "v0.1 Composition page — do not use them.",
        "",
        f"3. `canvas_control(verb, args={{}})` — state-transition verbs. Supported: "
        f"{control_verbs}. Most take `args={{}}`.",
        "",
        "4. `canvas_action(verb, args)` — content-producing verbs. **Verb-specific "
        "fields go INSIDE `args` (a nested object), not at the top level next to `verb`.**",
        "   - `draw_arrow`: pass `args = {\"from\": \"<element_id>\", \"to\": \"<element_id>\"}`. "
        "Both ids MUST come from the Available element ids list in Notes below "
        "(UUID-shaped). Do NOT use overlay ids (e.g. `ovl_3`) or any other id-shaped "
        "strings returned from earlier tool results — those are not element ids.",
        "   - `add_annotation`: pass `args = {\"text\": \"<string>\", \"x\": <number>, "
        "\"y\": <number>}`. `x` and `y` are in 1280x720 design-space coordinates.",
        "",
        "5. `canvas_set_page(pageType, pageInit={})` — switch the active canvas page. "
        "`pageType` must be one of `composition`, `youtube`, `quiz`. v0.1 only allows "
        "`composition` — only call this if the visitor explicitly asks for a different mode.",
        "",
        "Notes:",
        "- The canvas is 1280x720 pixels (origin top-left).",
        ids_line,
        "- For arg-less verbs (next_scene, previous_scene, clear), call with verb only and args={}.",
        "- Use these tools naturally during conversation when they help the visitor.",
    ])


# ──────────────────────────────────────────────────────────────────────
# Sync prompt assembly (Session 61)
# ──────────────────────────────────────────────────────────────────────
#
# Snapshot-only assembly path. Used by callers that already have a
# snapshot in hand (and by unit tests). The runtime path in
# persona.build_system_prompt is async and still integrates the legacy
# persona-prompt endpoint (Strategy 1) — these two paths are
# intentionally parallel until that legacy endpoint is retired.
# When you add or remove a section here, mirror the change in
# persona.build_system_prompt.

def build_system_prompt(snapshot: dict | None) -> str:
    """Assemble the agent's system prompt from a scene snapshot.

    Section order (S61 sandwich pattern + S63 Block 7):
      1. LANGUAGE directive            (top — strong steering)
      2. PERSONA                       (when snapshot.persona is non-empty)
      3. AUDIENCE                      (when snapshot.recipient_prompt is non-empty)
      4. KNOWLEDGE                     (S56)
      5. LINK NARRATION                (S63 — when snapshot.link is set)
      6. SCENE INSTRUCTION             (instruction or scene_instruction)
      7. SCENE / DISPLAY / ELEMENTS    (build_scene_description)
      8. CANVAS ACTION TOOL GUIDANCE
      9. SCRIPTS                       (when present)
      10. LANGUAGE reminder            (bottom — sandwich)
    """
    snapshot = snapshot or {}
    language = snapshot.get("language") or "en"
    sections: list[str] = []

    # 1. LANGUAGE directive (top)
    sections.append(f"# LANGUAGE\n{build_language_directive(language)}")

    # 2. PERSONA
    persona = (snapshot.get("persona") or "").strip()
    if persona:
        sections.append(f"# PERSONA\n{persona}")

    # 3. AUDIENCE (only when recipient_prompt is non-empty)
    audience = build_recipient_context(snapshot.get("recipient_prompt"))
    if audience:
        sections.append(audience.lstrip("\n"))

    # 4. KNOWLEDGE (S56)
    knowledge_section = build_knowledge_context(snapshot.get("knowledge"))
    if knowledge_section:
        sections.append(knowledge_section.lstrip("\n"))

    # 5. LINK NARRATION (S63 Block 7) — between KNOWLEDGE and SCENE INSTRUCTION
    link_narration = build_link_narration_directive(snapshot.get("link"))
    if link_narration:
        sections.append(link_narration)

    # 6. SCENE INSTRUCTION — accept either "instruction" (current backend)
    #    or "scene_instruction" (forward-compat with snapshot rename).
    instruction_text = (
        snapshot.get("scene_instruction")
        or snapshot.get("instruction")
        or ""
    ).strip()
    if instruction_text:
        sections.append(f"# SCENE INSTRUCTION\n{instruction_text}")

    # 6. SCENE / DISPLAY / ELEMENTS — existing combined helper.
    scene_block = build_scene_description(snapshot)
    if scene_block:
        sections.append(scene_block)

    # 7. CANVAS ACTION TOOL GUIDANCE
    canvas_tools = build_canvas_tools_section(snapshot)
    if canvas_tools:
        sections.append(canvas_tools)

    # 8. SCRIPTS
    scripts_section = build_scripts_section(snapshot)
    if scripts_section:
        sections.append(scripts_section)

    # 9. LANGUAGE reminder (bottom)
    sections.append(build_language_reminder(language))

    prompt = "\n\n".join(sections)

    logger.info(
        "Sync system prompt assembled: language={} audience_present={} sections={} prompt_chars={}",
        language,
        bool(audience),
        len(sections),
        len(prompt),
    )
    return prompt
