"""Process scene snapshots into agent-readable context.

Takes a scene snapshot dict from the API and builds descriptive text
for the LLM system prompt.
"""
from loguru import logger


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


def build_canvas_tools_section(snapshot: dict) -> str:
    """Build the canvas action tools description.

    These tools will be wired in Session 47. For now, describe them
    so the LLM knows they exist but won't try to invoke them yet.
    """
    elements = snapshot.get("elements", [])
    if not elements:
        return ""

    return """## Canvas Actions (Coming Soon)
In future updates, you will be able to:
- Highlight specific elements on the canvas to draw attention
- Draw arrows between elements to show relationships
- Place text annotations on the canvas
- Navigate to different scenes in a multi-scene flow
For now, describe what you see on the canvas verbally."""
