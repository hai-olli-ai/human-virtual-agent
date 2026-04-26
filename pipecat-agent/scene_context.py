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

def build_scene_description(snapshot: dict) -> str:
    """Build a human-readable scene description from a snapshot.

    The snapshot comes from GET /live-rooms/{room_id}/scene-snapshot.
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
        # in the lower right") and may use the existing highlight_element
        # canvas action to point at one, but it must NOT attempt to "click"
        # buttons on the visitor's behalf. Buttons are part of the visual
        # scene; clicks are exclusively the visitor's affordance.
        for el in elements:
            el_type = el.get("type", "unknown")
            desc = f"- {el_type}"

            if el.get("text"):
                desc += f': "{el["text"]}"'
            if el.get("label"):
                desc += f' (label: {el["label"]})'
            if el.get("title"):
                desc += f' (title: {el["title"]})'
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


def build_canvas_tools_section(snapshot: dict) -> str:
    """Build the canvas action tools description for the system prompt."""
    parts = ["## Canvas Actions"]
    parts.append("You have tools to interact with the canvas visually:")
    parts.append("- highlight_element(x, y, width, height): Highlight a region to draw attention")
    parts.append("- draw_arrow(from_x, from_y, to_x, to_y): Draw an arrow between two points")
    parts.append("- place_annotation(text, x, y): Place a short text label at a position")
    parts.append("- clear_annotations: Remove all visual overlays")

    total = snapshot.get("total_scenes", 1)
    if total > 1:
        parts.append("- navigate_scene: Go to next/previous scene in the flow")

    parts.append("")
    parts.append("The canvas is 1280x720 pixels (origin at top-left).")
    parts.append("Use the canvas image to estimate pixel coordinates for these tools.")
    parts.append("Use these tools naturally during conversation when they help the visitor understand the content.")
    parts.append("When describing yourself, note that the visitor may see your profile photo on the canvas.")

    return "\n".join(parts)


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

    Section order (S61 sandwich pattern):
      1. LANGUAGE directive            (top — strong steering)
      2. PERSONA                       (when snapshot.persona is non-empty)
      3. AUDIENCE                      (when snapshot.recipient_prompt is non-empty)
      4. KNOWLEDGE                     (S56)
      5. SCENE INSTRUCTION             (instruction or scene_instruction)
      6. SCENE / DISPLAY / ELEMENTS    (build_scene_description)
      7. CANVAS ACTION TOOL GUIDANCE
      8. SCRIPTS                       (when present)
      9. LANGUAGE reminder             (bottom — sandwich)
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

    # 5. SCENE INSTRUCTION — accept either "instruction" (current backend)
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
