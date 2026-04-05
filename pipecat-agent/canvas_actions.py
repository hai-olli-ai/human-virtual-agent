"""
Canvas action tools for the Pipecat voice agent.

Defines LLM function calling tools that let the agent interact with the
frontend canvas: highlight elements, draw arrows, place annotations,
navigate scenes, and clear overlays.

Tool calls are dispatched as data channel messages to the frontend.
"""
import json
from typing import Any

from loguru import logger

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.frames.frames import FunctionCallResultProperties
from pipecat.services.llm_service import FunctionCallParams


# ── Tool Definitions ──

highlight_element = FunctionSchema(
    name="highlight_element",
    description="Highlight an element or region on the scene canvas to draw the visitor's attention. Use this when pointing out specific content.",
    properties={
        "element_description": {
            "type": "string",
            "description": "Description of the element to highlight (e.g., 'the title text', 'the avatar', 'the pricing section')",
        },
        "color": {
            "type": "string",
            "enum": ["orange", "green", "blue", "red"],
            "description": "Highlight color. Default: orange",
        },
        "duration_seconds": {
            "type": "number",
            "description": "How long the highlight stays visible in seconds. Default: 3",
        },
    },
    required=["element_description"],
)

draw_arrow = FunctionSchema(
    name="draw_arrow",
    description="Draw an arrow on the canvas pointing from one element/area to another. Use this to show relationships or flow between elements.",
    properties={
        "from_description": {
            "type": "string",
            "description": "Description of the starting point (e.g., 'the input field')",
        },
        "to_description": {
            "type": "string",
            "description": "Description of the ending point (e.g., 'the output display')",
        },
        "color": {
            "type": "string",
            "enum": ["orange", "green", "blue", "red"],
            "description": "Arrow color. Default: orange",
        },
    },
    required=["from_description", "to_description"],
)

place_annotation = FunctionSchema(
    name="place_annotation",
    description="Place a short text label on the canvas near a specific element. Use this to add brief commentary or labels.",
    properties={
        "text": {
            "type": "string",
            "description": "The annotation text (keep under 40 characters)",
        },
        "near_description": {
            "type": "string",
            "description": "Description of the element to place the annotation near",
        },
    },
    required=["text", "near_description"],
)

navigate_scene_tool = FunctionSchema(
    name="navigate_scene",
    description="Go to the next or previous scene in a multi-scene flow. Only use this when the room has multiple scenes.",
    properties={
        "direction": {
            "type": "string",
            "enum": ["next", "previous"],
            "description": "Direction to navigate",
        },
    },
    required=["direction"],
)

clear_annotations_tool = FunctionSchema(
    name="clear_annotations",
    description="Remove all highlights, arrows, and text annotations from the canvas. Use this before showing new annotations or when the canvas is cluttered.",
    properties={},
    required=[],
)


def get_canvas_tools() -> ToolsSchema:
    """Get all canvas action tools as a ToolsSchema."""
    return ToolsSchema(
        standard_tools=[
            highlight_element,
            draw_arrow,
            place_annotation,
            navigate_scene_tool,
            clear_annotations_tool,
        ]
    )


# ── Color Mapping ──

COLOR_MAP = {
    "orange": "#C15F3C",
    "green": "#4A7C59",
    "blue": "#4A6FA5",
    "red": "#C1443C",
}


# ── Element Resolution ──

def resolve_element_region(description: str, elements: list[dict]) -> dict | None:
    """Find the element that best matches the description and return its bounding box.

    Uses simple keyword matching. The LLM provides descriptions like
    "the title text" or "the avatar" — we match against element type,
    text content, label, and title.

    Returns: { x, y, width, height } or None if no match found.
    """
    if not elements:
        return None

    desc_lower = description.lower()
    best_match = None
    best_score = 0

    for el in elements:
        score = 0
        el_type = (el.get("type") or "").lower()
        el_text = (el.get("text") or "").lower()
        el_label = (el.get("label") or "").lower()
        el_title = (el.get("title") or "").lower()

        # Type matching
        if el_type and el_type in desc_lower:
            score += 3
        if "avatar" in desc_lower and "avatar" in el_type:
            score += 5
        if "text" in desc_lower and el_type == "text":
            score += 2
        if "image" in desc_lower and el_type == "image":
            score += 2

        # Content matching
        for word in desc_lower.split():
            if len(word) > 2:  # Skip short words
                if word in el_text:
                    score += 4
                if word in el_label:
                    score += 3
                if word in el_title:
                    score += 3

        if score > best_score:
            best_score = score
            pos = el.get("position", {})
            size = el.get("size", {})
            best_match = {
                "x": pos.get("x", 0),
                "y": pos.get("y", 0),
                "width": size.get("width", 100),
                "height": size.get("height", 100),
            }

    if best_score >= 2:
        return best_match

    # Fallback: return the center of the canvas
    return {"x": 440, "y": 260, "width": 400, "height": 200}


# ── Tool Handler Factory ──

def create_canvas_action_handlers(
    transport,
    elements: list[dict],
    room_id: str = "",
    api_url: str | None = None,
):
    """Create function call handlers for all canvas action tools.

    Args:
        transport: The Pipecat transport (for sending data channel messages)
        elements: Scene elements from the snapshot (for coordinate resolution)
        room_id: Live room ID (for scene navigation)
        api_url: Backend API URL (for scene navigation)

    Returns:
        Dict of { function_name: handler_coroutine }
    """

    async def _send_canvas_action(action: dict):
        """Send a canvas action to the frontend via data channel."""
        payload = {"type": "canvas_action", "action": action}
        try:
            if hasattr(transport, "send_app_message"):
                await transport.send_app_message(payload)
            elif hasattr(transport, "send_message"):
                await transport.send_message(json.dumps(payload))
        except Exception as e:
            logger.warning(f"Failed to send canvas action: {e}")

    async def handle_highlight_element(params: FunctionCallParams):
        """Handle highlight_element tool call."""
        args = params.arguments
        description = args.get("element_description", "")
        color_name = args.get("color", "orange")
        duration = args.get("duration_seconds", 3)

        region = resolve_element_region(description, elements)
        color = COLOR_MAP.get(color_name, COLOR_MAP["orange"])

        await _send_canvas_action({
            "name": "highlight",
            "params": {
                "region": region,
                "color": color,
                "duration": int(duration * 1000),  # Convert to ms
            },
        })

        props = FunctionCallResultProperties(run_llm=False)
        await params.result_callback(
            {"status": "highlighted", "element": description},
            properties=props,
        )

    async def handle_draw_arrow(params: FunctionCallParams):
        """Handle draw_arrow tool call."""
        args = params.arguments
        from_desc = args.get("from_description", "")
        to_desc = args.get("to_description", "")
        color_name = args.get("color", "orange")

        from_region = resolve_element_region(from_desc, elements)
        to_region = resolve_element_region(to_desc, elements)
        color = COLOR_MAP.get(color_name, COLOR_MAP["orange"])

        # Arrow goes from center of source to center of target
        from_point = {
            "x": from_region["x"] + from_region["width"] // 2,
            "y": from_region["y"] + from_region["height"] // 2,
        } if from_region else {"x": 300, "y": 360}

        to_point = {
            "x": to_region["x"] + to_region["width"] // 2,
            "y": to_region["y"] + to_region["height"] // 2,
        } if to_region else {"x": 900, "y": 360}

        await _send_canvas_action({
            "name": "draw_arrow",
            "params": {
                "from": from_point,
                "to": to_point,
                "color": color,
                "duration": 5000,
            },
        })

        props = FunctionCallResultProperties(run_llm=False)
        await params.result_callback(
            {"status": "arrow_drawn", "from": from_desc, "to": to_desc},
            properties=props,
        )

    async def handle_place_annotation(params: FunctionCallParams):
        """Handle place_annotation tool call."""
        args = params.arguments
        text = args.get("text", "")[:40]  # Limit length
        near_desc = args.get("near_description", "")

        region = resolve_element_region(near_desc, elements)

        # Place annotation above the element
        position = {
            "x": region["x"] + region["width"] // 2,
            "y": max(region["y"] - 30, 10),
        } if region else {"x": 640, "y": 100}

        await _send_canvas_action({
            "name": "place_annotation",
            "params": {
                "text": text,
                "position": position,
                "duration": 6000,
            },
        })

        props = FunctionCallResultProperties(run_llm=False)
        await params.result_callback(
            {"status": "annotation_placed", "text": text},
            properties=props,
        )

    async def handle_navigate_scene(params: FunctionCallParams):
        """Handle navigate_scene tool call."""
        args = params.arguments
        direction = args.get("direction", "next")

        if room_id:
            from api_client import navigate_scene as api_navigate
            result = await api_navigate(room_id, direction, api_url=api_url)

            if result:
                # Send navigation action to frontend
                await _send_canvas_action({
                    "name": "navigate",
                    "params": {"direction": direction},
                })

                # Also re-fetch vision image for the new scene
                from api_client import get_scene_image_base64
                new_image = await get_scene_image_base64(room_id, api_url)
                if new_image:
                    from scene_context import build_vision_message
                    # Note: Updating vision context mid-conversation requires
                    # adding a new message to the context. This is handled
                    # by returning a result that tells the LLM about the new scene.
                    logger.info(f"Navigated to scene {result.get('current_scene_index', '?')}")

                props = FunctionCallResultProperties(run_llm=True)
                await params.result_callback(
                    {
                        "status": "navigated",
                        "direction": direction,
                        "current_scene_index": result.get("current_scene_index"),
                        "scene_title": result.get("scene_title"),
                        "total_scenes": result.get("total_scenes"),
                    },
                    properties=props,
                )
                return

        props = FunctionCallResultProperties(run_llm=True)
        await params.result_callback(
            {"status": "error", "message": "Navigation not available for this room"},
            properties=props,
        )

    async def handle_clear_annotations(params: FunctionCallParams):
        """Handle clear_annotations tool call."""
        await _send_canvas_action({
            "name": "clear_annotations",
            "params": {},
        })

        props = FunctionCallResultProperties(run_llm=False)
        await params.result_callback(
            {"status": "cleared"},
            properties=props,
        )

    return {
        "highlight_element": handle_highlight_element,
        "draw_arrow": handle_draw_arrow,
        "place_annotation": handle_place_annotation,
        "navigate_scene": handle_navigate_scene,
        "clear_annotations": handle_clear_annotations,
    }
