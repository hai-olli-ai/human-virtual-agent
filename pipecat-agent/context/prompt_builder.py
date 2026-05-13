"""
S64c prompt builder — Canvas Protocol generation.

Adds the CANVAS PAGE section (rendered from the active Page's manifest) and
replaces V2.13's hardcoded CANVAS ACTIONS section with a generic 5-tool
description. Exposes `build_system_prompt_split` returning the prompt as
(stable_prefix, dynamic_suffix) so the Anthropic LLM service can mark the
prefix with cache_control: ephemeral.

S64c sandwich order:
  1.  LANGUAGE (open)
  2.  PERSONA
  3.  AUDIENCE
  4.  KNOWLEDGE
  4b. LINK NARRATION       (carried forward from S63)
  5.  SCENE INSTRUCTION
  5b. CANVAS PAGE          ← NEW (per-Page manifest)
  --- split ---
  6.  DISPLAY MODE
  7.  CANVAS ELEMENTS
  8.  CANVAS ACTIONS       ← REPLACED (5 generic tools)
  8b. SCRIPTS              (carried forward from S49)
  9.  LANGUAGE (close)

The split point is between CANVAS PAGE (5b) and DISPLAY MODE (6). Within a
single scene the stable prefix is identical across turns — Anthropic can
hit the prompt cache, OpenAI/Gemini get implicit prefix-cache benefit.

This module sits alongside persona.py and scene_context.py during S64c.
Block 5 wires bot.py to call build_system_prompt_split() instead of
persona.build_system_prompt(). Block 7 deletes the legacy V2.13 helper
(scene_context.build_canvas_tools_section) after cutover.
"""

from __future__ import annotations

from typing import Optional

from scene_context import (
    build_knowledge_context,
    build_language_directive,
    build_language_reminder,
    build_link_narration_directive,
    build_recipient_context,
    build_scripts_section,
)


# ----------------------------------------------------------------------------
# CANVAS PAGE — NEW (S64c)
# ----------------------------------------------------------------------------

# Per-verb arg shapes that the LLM must pass when invoking known verbs. The
# same shapes are also documented in the canvas_control / canvas_action
# FunctionSchema descriptions (tools/canvas_protocol_tools.py) — the
# redundancy is intentional because early LLM calls routinely flatten
# verb-specific fields onto the top level alongside `verb`, and documenting
# the nesting on BOTH surfaces (prompt + schema) was what closed the class
# of flatten-bugs (CLAUDE.md "canvas_action verb-specific arg shape" note).
# Pages that introduce new verbs without entries here render as argless —
# the frontend returns INVALID_ARGS if the verb actually needs args, and
# the LLM self-corrects from the error result handed back as a tool reply.
VERB_ARG_SHAPES: dict[str, str] = {
    # control verbs
    "seek":       '{"seconds": <non-negative number>}',
    "set_speed":  '{"rate": <number; 1.0 normal, 0.5 half, 2.0 double>}',
    "goto_scene": '{"index": <zero-based integer scene index>}',
    # action verbs
    "draw_arrow":     '{"from": "<element_id>", "to": "<element_id>"}',
    "add_annotation": '{"text": "<string>", "x": <number>, "y": <number>}',
}


def _render_verb_list(label: str, verbs: list[str]) -> list[str]:
    """Render a verb listing with per-verb arg shapes inlined when known.

    Concise single-line when every verb in the list is argless; multi-line
    bulleted when at least one verb declares an arg shape, so the LLM has
    an unambiguous template to copy for each verb."""
    if not verbs:
        return []
    has_args = [v for v in verbs if v in VERB_ARG_SHAPES]
    if not has_args:
        return [f"- {label} verbs: {' | '.join(verbs)}."]
    argless = [v for v in verbs if v not in VERB_ARG_SHAPES]
    lines = [f"- {label} verbs:"]
    if argless:
        lines.append(f"  - {', '.join(argless)} — args: {{}}")
    for v in has_args:
        lines.append(f"  - {v} — args: {VERB_ARG_SHAPES[v]}")
    return lines


def render_canvas_page_section(manifest: Optional[dict]) -> str:
    """Render the CANVAS PAGE section from the active Page's manifest.

    The section is positioned at a stable offset between SCENE INSTRUCTION and
    DISPLAY MODE. Its content varies per scene only when canvas_page_type
    changes (S64d+). Within a single Page type, the CANVAS PAGE text is stable
    — content fits inside the prompt cache prefix on Anthropic.
    """
    if not manifest:
        return (
            "## CANVAS PAGE\n"
            "No page registered yet. Wait for canvas.register before issuing "
            "highlight/control/action/analyze tool calls. canvas_set_page is "
            "the only tool you can safely call without a registered page."
        )

    page_type = manifest.get("pageType", "unknown")
    version = manifest.get("version", "0.1")
    cap = manifest.get("capabilities") or {}

    lines = [f"## CANVAS PAGE", f"Active page: **{page_type}** (v{version})."]

    if cap.get("analyze", {}).get("supported"):
        lines.append("- analyze: supported (semantic state provider).")

    h = cap.get("highlight") or {}
    if h.get("supported"):
        targets = ", ".join(h.get("targets", []) or []) or "(unspecified)"
        lines.append(f"- highlight: targets={targets}.")

    lines.extend(_render_verb_list("control", (cap.get("control") or {}).get("verbs") or []))
    lines.extend(_render_verb_list("action", (cap.get("action") or {}).get("verbs") or []))

    lines.append("")
    lines.append("Use these verbs through canvas_control(verb=..., args={...}) and canvas_action(verb=..., args={...}).")
    lines.append("Verb-specific fields MUST be nested inside `args` — never at the top level alongside `verb`.")
    lines.append("If you call an unsupported verb, the dispatch returns UNSUPPORTED_VERB and you should pick a supported alternative.")

    return "\n".join(lines)


# ----------------------------------------------------------------------------
# CANVAS ACTIONS — REPLACED (S64c)
# ----------------------------------------------------------------------------

def render_canvas_actions_section() -> str:
    """Render the CANVAS ACTIONS section explaining how to use the 5 generic tools.

    This replaces the V2.13 section that listed 5 hardcoded tools individually.
    The text below is the same regardless of active page; per-page guidance
    lives in the CANVAS PAGE section above.
    """
    return (
        "## CANVAS ACTIONS\n"
        "You have exactly 5 tools to operate the canvas. Choose based on intent:\n"
        "\n"
        "1. **canvas_analyze(question, options={})** — ask a question about what is "
        "visible. Returns a text answer using the active Page's semantic state. "
        "Use when you cannot determine the answer from CANVAS ELEMENTS or your other context.\n"
        "\n"
        "2. **canvas_highlight(target, options={})** — draw a highlight on the canvas. "
        "target is either {element_id: \"el_xxx\"} (preferred when an id is known from "
        "CANVAS ELEMENTS) or {box: [x, y, w, h]} in 1280x720 coordinates for arbitrary regions.\n"
        "\n"
        "3. **canvas_control(verb, args={})** — invoke a state-transition verb. The CANVAS "
        "PAGE section above lists which verbs are supported. Most control verbs take {}.\n"
        "\n"
        "4. **canvas_action(verb, args={})** — invoke a content-producing verb. The CANVAS "
        "PAGE section above lists which verbs are supported. Action verbs typically take args "
        "(e.g. draw_arrow needs from + to element ids; add_annotation needs text + x + y).\n"
        "\n"
        "5. **canvas_set_page(pageType, pageInit={})** — switch the active Canvas Page. "
        "Use sparingly. pageType must be one of: composition, youtube, quiz. After a successful "
        "set_page, a new manifest arrives and your CANVAS PAGE section updates.\n"
        "\n"
        "Notes:\n"
        "- Highlights persist until canvas_control(verb=\"clear\") or scene change.\n"
        "- Reuse element ids from CANVAS ELEMENTS when possible — they are stable.\n"
        "- For arg-less verbs (next_scene, previous_scene, clear, pause, play, restart, "
        "next_question, previous_question), call control or action with verb only and args={}."
    )


# ----------------------------------------------------------------------------
# Small per-section renderers used only by the split builder.
# (DISPLAY MODE and CANVAS ELEMENTS are extracted out of
# scene_context.build_scene_description so the dynamic suffix can render
# them as discrete sections; the legacy V2.13 helper stays untouched.)
# ----------------------------------------------------------------------------

def _render_display_mode(display_mode: str | None) -> str:
    if not display_mode:
        return ""
    lines = ["## DISPLAY MODE", f"Avatar display mode: {display_mode}."]
    if display_mode == "invisible":
        lines.append(
            "You are in voice-only mode. The visitor cannot see you, only hear you. "
            "Focus entirely on verbal communication."
        )
    elif display_mode == "talking":
        lines.append(
            "You are rendered as a talking avatar with lip sync. The visitor can see "
            "your face moving as you speak."
        )
    elif display_mode == "3dgs":
        lines.append(
            "You are rendered as a 3D model. The visitor sees a 3D representation of you."
        )
    return "\n".join(lines)


def _render_canvas_elements(elements: list[dict] | None) -> str:
    if not elements:
        return ""
    lines = ["## CANVAS ELEMENTS", "Elements visible to the visitor:"]
    for el in elements:
        el_type = el.get("type", "unknown")
        desc = f"- {el_type}"
        if el.get("id"):
            desc += f" [id: {el['id']}]"
        if el.get("text"):
            desc += f': "{el["text"]}"'
        if el.get("label"):
            desc += f' (label: {el["label"]})'
        if el.get("title"):
            desc += f' (title: {el["title"]})'
        if el.get("display_mode"):
            desc += f" [display: {el['display_mode']}]"
        pos = el.get("position") or {}
        size = el.get("size") or {}
        if pos and size:
            desc += (
                f" at ({pos.get('x', 0)}, {pos.get('y', 0)}), "
                f"size {size.get('width', 0)}x{size.get('height', 0)}"
            )
        lines.append(desc)
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# Split builder — returns (stable_prefix, dynamic_suffix)
# ----------------------------------------------------------------------------

def build_system_prompt_split(
    *,
    snapshot: dict | None,
    canvas_manifest: Optional[dict] = None,
) -> tuple[str, str]:
    """Build the S64c system prompt as (stable_prefix, dynamic_suffix).

    The Anthropic LLM service can wrap the stable_prefix as a content block
    with cache_control={"type": "ephemeral"} and pass dynamic_suffix as a
    second uncached block. OpenAI and Gemini concatenate the two halves with
    "\\n\\n" — implicit prefix caching kicks in because the first half is
    stable across turns within a scene.

    Args:
      snapshot: scene-snapshot dict from GET /live-rooms/.../scene-snapshot,
        or None to render the minimal sandwich (LANGUAGE only).
      canvas_manifest: the active Page's manifest (registered via
        canvas.register), or None when no Page has registered yet.

    Returns:
      (stable_prefix, dynamic_suffix) — both strings, joined with "\\n\\n".
    """
    snapshot = snapshot or {}
    language = snapshot.get("language") or "en"

    # ── Stable prefix (sections 1 → 5b) ──
    stable_parts: list[str] = []

    stable_parts.append(f"# LANGUAGE\n{build_language_directive(language)}")

    persona = (snapshot.get("persona") or "").strip()
    if persona:
        stable_parts.append(f"# PERSONA\n{persona}")

    audience = build_recipient_context(snapshot.get("recipient_prompt"))
    if audience:
        stable_parts.append(audience.lstrip("\n"))

    knowledge_section = build_knowledge_context(snapshot.get("knowledge"))
    if knowledge_section:
        stable_parts.append(knowledge_section.lstrip("\n"))

    link_narration = build_link_narration_directive(snapshot.get("link"))
    if link_narration:
        stable_parts.append(link_narration)

    instruction_text = (
        snapshot.get("scene_instruction")
        or snapshot.get("instruction")
        or ""
    ).strip()
    if instruction_text:
        stable_parts.append(f"# SCENE INSTRUCTION\n{instruction_text}")

    stable_parts.append(render_canvas_page_section(canvas_manifest))

    stable = "\n\n".join(s for s in stable_parts if s)

    # ── Dynamic suffix (sections 6 → 9) ──
    dynamic_parts: list[str] = []

    display_block = _render_display_mode(
        snapshot.get("avatar_display_mode") or snapshot.get("display_mode")
    )
    if display_block:
        dynamic_parts.append(display_block)

    elements_block = _render_canvas_elements(snapshot.get("elements"))
    if elements_block:
        dynamic_parts.append(elements_block)

    dynamic_parts.append(render_canvas_actions_section())

    scripts_section = build_scripts_section(snapshot)
    if scripts_section:
        dynamic_parts.append(scripts_section)

    dynamic_parts.append(build_language_reminder(language))

    dynamic = "\n\n".join(s for s in dynamic_parts if s)

    return stable, dynamic
