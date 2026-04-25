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
