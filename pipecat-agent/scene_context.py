"""Process scene snapshots into agent-readable context.

Takes a scene snapshot dict from the API and builds descriptive text
for the LLM system prompt.
"""
from loguru import logger

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
