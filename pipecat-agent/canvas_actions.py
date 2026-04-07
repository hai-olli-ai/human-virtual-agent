"""
Canvas action tools for the Pipecat voice agent.

Defines LLM function calling tools that let the agent interact with the
frontend canvas: highlight elements, draw arrows, place annotations,
navigate scenes, and clear overlays.

The LLM uses vision to see the canvas and provides pixel coordinates
directly in tool calls. No server-side element resolution needed.

Tool calls are dispatched as data channel messages to the frontend.
"""
from loguru import logger

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.frames.frames import FunctionCallResultProperties, OutputTransportMessageFrame
from pipecat.services.llm_service import FunctionCallParams


# ── Tool Definitions ──

highlight_element = FunctionSchema(
    name="highlight_element",
    description="Highlight a rectangular region on the scene canvas to draw the visitor's attention. Use the canvas image to estimate pixel coordinates of the element you want to highlight.",
    properties={
        "x": {
            "type": "number",
            "description": "X coordinate of the top-left corner of the highlight region (pixels from left edge)",
        },
        "y": {
            "type": "number",
            "description": "Y coordinate of the top-left corner of the highlight region (pixels from top edge)",
        },
        "width": {
            "type": "number",
            "description": "Width of the highlight region in pixels",
        },
        "height": {
            "type": "number",
            "description": "Height of the highlight region in pixels",
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
    required=["x", "y", "width", "height"],
)

draw_arrow = FunctionSchema(
    name="draw_arrow",
    description="Draw an arrow on the canvas from one point to another. Use the canvas image to estimate pixel coordinates of the start and end points.",
    properties={
        "from_x": {
            "type": "number",
            "description": "X coordinate of the arrow start point",
        },
        "from_y": {
            "type": "number",
            "description": "Y coordinate of the arrow start point",
        },
        "to_x": {
            "type": "number",
            "description": "X coordinate of the arrow end point",
        },
        "to_y": {
            "type": "number",
            "description": "Y coordinate of the arrow end point",
        },
        "color": {
            "type": "string",
            "enum": ["orange", "green", "blue", "red"],
            "description": "Arrow color. Default: orange",
        },
    },
    required=["from_x", "from_y", "to_x", "to_y"],
)

place_annotation = FunctionSchema(
    name="place_annotation",
    description="Place a short text label at a specific position on the canvas. Use the canvas image to estimate where to place it.",
    properties={
        "text": {
            "type": "string",
            "description": "The annotation text (keep under 40 characters)",
        },
        "x": {
            "type": "number",
            "description": "X coordinate for the annotation position",
        },
        "y": {
            "type": "number",
            "description": "Y coordinate for the annotation position",
        },
    },
    required=["text", "x", "y"],
)

navigate_scene_tool = FunctionSchema(
    name="navigate_scene",
    description="Go to the next or previous scene in a multi-scene flow. Trigger this when the visitor says things like 'next', 'next slide', 'next one', 'go next', 'go back', 'previous', 'previous slide', 'previous one', or similar navigation requests.",
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


# ── Tool Handler Factory ──

def create_canvas_action_handlers(
    output_transport,
    context=None,
    llm=None,
    room_id: str = "",
    api_url: str | None = None,
):
    """Create function call handlers for all canvas action tools.

    Args:
        output_transport: The Pipecat output transport (from transport.output())
        context: The LLMContext (for updating vision after scene navigation)
        llm: The OpenAILLMService (for updating system_instruction after navigation)
        room_id: Live room ID (for scene navigation)
        api_url: Backend API URL (for scene navigation)

    Returns:
        Dict of { function_name: handler_coroutine }
    """

    async def _send_canvas_action(action: dict):
        """Send a canvas action to the frontend via data channel."""
        payload = {"type": "canvas_action", "action": action}
        try:
            frame = OutputTransportMessageFrame(message=payload)
            await output_transport.send_message(frame)
        except Exception as e:
            logger.warning(f"Failed to send canvas action: {e}")

    async def handle_highlight_element(params: FunctionCallParams):
        """Handle highlight_element tool call."""
        args = params.arguments
        region = {
            "x": args.get("x", 0),
            "y": args.get("y", 0),
            "width": args.get("width", 100),
            "height": args.get("height", 100),
        }
        color_name = args.get("color", "orange")
        duration = args.get("duration_seconds", 3)
        color = COLOR_MAP.get(color_name, COLOR_MAP["orange"])

        logger.info(f"highlight_element: region={region} color={color_name} duration={duration}s")

        await _send_canvas_action({
            "name": "highlight",
            "params": {
                "region": region,
                "color": color,
                "duration": int(duration * 1000),
            },
        })

        props = FunctionCallResultProperties(run_llm=False)
        await params.result_callback({"status": "highlighted"}, properties=props)

    async def handle_draw_arrow(params: FunctionCallParams):
        """Handle draw_arrow tool call."""
        args = params.arguments
        from_point = {"x": args.get("from_x", 0), "y": args.get("from_y", 0)}
        to_point = {"x": args.get("to_x", 0), "y": args.get("to_y", 0)}
        color_name = args.get("color", "orange")
        color = COLOR_MAP.get(color_name, COLOR_MAP["orange"])

        logger.info(f"draw_arrow: from={from_point} to={to_point} color={color_name}")

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
        await params.result_callback({"status": "arrow_drawn"}, properties=props)

    async def handle_place_annotation(params: FunctionCallParams):
        """Handle place_annotation tool call."""
        args = params.arguments
        text = args.get("text", "")[:40]
        position = {"x": args.get("x", 0), "y": args.get("y", 0)}

        logger.info(f"place_annotation: text='{text}' position={position}")

        await _send_canvas_action({
            "name": "place_annotation",
            "params": {
                "text": text,
                "position": position,
                "duration": 6000,
            },
        })

        props = FunctionCallResultProperties(run_llm=False)
        await params.result_callback({"status": "annotation_placed", "text": text}, properties=props)

    async def handle_navigate_scene(params: FunctionCallParams):
        """Handle navigate_scene tool call."""
        args = params.arguments
        direction = args.get("direction", "next")

        logger.info(f"navigate_scene: direction={direction}")

        # 1. Tell frontend to navigate (before backend, so frontend navigates
        #    relative to the current backend state — avoids double-navigation)
        await _send_canvas_action({
            "name": "navigate",
            "params": {"direction": direction},
        })

        # 2. Update backend state so image endpoint returns the new scene
        if room_id:
            from api_client import navigate_scene as api_navigate
            await api_navigate(room_id, direction, api_url=api_url)

        # 3. Rebuild system prompt with new scene's instruction
        if room_id and llm:
            from persona import build_system_prompt
            new_prompt = await build_system_prompt(room_id=room_id, api_url=api_url)
            llm._settings.system_instruction = new_prompt
            logger.info(f"Updated system prompt for new scene ({len(new_prompt)} chars)")

        # 4. Fetch new scene image and update LLM vision context
        if room_id and context:
            from api_client import get_scene_image_base64
            from scene_context import build_vision_message
            new_image = await get_scene_image_base64(room_id, api_url)
            if new_image:
                context.add_message(build_vision_message(new_image))
                logger.info("Updated vision context with new scene image")
            else:
                logger.warning("Could not fetch new scene image after navigation")

        props = FunctionCallResultProperties(run_llm=True)
        await params.result_callback(
            {"status": "navigated", "direction": direction},
            properties=props,
        )

    async def handle_clear_annotations(params: FunctionCallParams):
        """Handle clear_annotations tool call."""
        await _send_canvas_action({
            "name": "clear_annotations",
            "params": {},
        })

        props = FunctionCallResultProperties(run_llm=False)
        await params.result_callback({"status": "cleared"}, properties=props)

    return {
        "highlight_element": handle_highlight_element,
        "draw_arrow": handle_draw_arrow,
        "place_annotation": handle_place_annotation,
        "navigate_scene": handle_navigate_scene,
        "clear_annotations": handle_clear_annotations,
    }
