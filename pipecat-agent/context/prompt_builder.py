"""
S64c prompt builder — Canvas Protocol generation.

Adds the CANVAS PAGE section (rendered from the active Page's manifest) and
replaces V2.13's hardcoded CANVAS ACTIONS section with a generic 5-tool
description. Exposes `build_system_prompt_split` returning the prompt as
(stable_prefix, dynamic_suffix) so the Anthropic LLM service can mark the
prefix with cache_control: ephemeral.

S64c sandwich order (+ S64e AGENT PLAYBOOK):
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
  8b. AGENT PLAYBOOK       ← NEW (S64e — cross-tool sequences)
  8c. SCRIPTS              (carried forward from S49)
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
    "seek": '{"seconds": <non-negative number>}',
    "set_speed": '{"rate": <number; 1.0 normal, 0.5 half, 2.0 double>}',
    "goto_scene": '{"index": <zero-based integer scene index>}',
    # action verbs
    "draw_arrow": '{"from": "<element_id>", "to": "<element_id>"}',
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
            "control/action/analyze tool calls. canvas_set_page is "
            "the only tool you can safely call without a registered page."
        )

    page_type = manifest.get("pageType", "unknown")
    version = manifest.get("version", "0.1")
    cap = manifest.get("capabilities") or {}

    lines = ["## CANVAS PAGE", f"Active page: **{page_type}** (v{version})."]

    if cap.get("analyze", {}).get("supported"):
        lines.append("- analyze: supported (semantic state provider).")

    lines.extend(
        _render_verb_list("control", (cap.get("control") or {}).get("verbs") or [])
    )
    lines.extend(
        _render_verb_list("action", (cap.get("action") or {}).get("verbs") or [])
    )

    lines.append("")
    lines.append(
        "Use these verbs through canvas_control(verb=..., args={...}) and canvas_action(verb=..., args={...})."
    )
    lines.append(
        "Verb-specific fields MUST be nested inside `args` — never at the top level alongside `verb`."
    )
    lines.append(
        "If you call an unsupported verb, the dispatch returns UNSUPPORTED_VERB and you should pick a supported alternative."
    )
    lines.append("")
    # Scene-navigation verbs are SHELL-level and apply regardless of which
    # Canvas Page is active — they're routed by the frontend's DailyRelay
    # to the live-room shell's navigateToIndex BEFORE reaching the iframe,
    # so they don't (and shouldn't) appear in any Page's manifest. Always
    # callable; never returns UNSUPPORTED_VERB.
    lines.append(
        "Scene navigation: `canvas_control(verb='next_scene')`, "
        "`canvas_control(verb='previous_scene')`, and "
        "`canvas_control(verb='goto_scene', args={index: <int>})` are "
        "ALWAYS available regardless of the active Page — they navigate "
        "between scenes at the shell level. Use these whenever the "
        "visitor asks to move forward/backward in the flow, even from "
        "inside a YouTube or Quiz scene where the per-page verb list "
        "above does not mention them."
    )

    return "\n".join(lines)


# ----------------------------------------------------------------------------
# AGENT PLAYBOOK — NEW (S64e)
# ----------------------------------------------------------------------------

# Cross-tool playbooks. These document multi-step sequences the agent must
# follow that span tools (generate_quiz_from_knowledge → canvas_set_page →
# canvas_action → canvas_control) and that the per-tool descriptions can't
# express on their own. Sits BETWEEN the canvas tool guidance and the
# closing LANGUAGE reminder, so it benefits from recency weighting without
# displacing the canvas/manifest sections that the LLM consults for every
# tool call.
#
# Tool names use the underscored form (canvas_set_page, canvas_action,
# canvas_control, generate_quiz_from_knowledge) to match what's actually
# registered with the LLM provider. The CLAUDE.md note "Tool names must
# satisfy ^[a-zA-Z0-9_-]+$" applies here too — dotted forms would steer
# the model toward calls that 400 at the provider boundary.


def render_voice_output_style_section() -> str:
    """VOICE OUTPUT STYLE — hard formatting constraint for TTS + caption.

    The reply is sent verbatim to the TTS engine and shown as a live caption,
    so any Markdown the model emits (gpt-oss is markdown-happy) surfaces as
    spoken/printed noise — literal ``**``, stray symbols, trailing ``…``
    filler. This blunt directive keeps output to plain spoken words. It pairs
    with the TTS-side ``MarkdownTextFilter`` (the audio safety net), but the
    directive is what also keeps the CAPTION clean, since the transcript is
    forwarded upstream of the TTS filter.
    """
    return (
        "## VOICE OUTPUT — STRICT\n"
        "Your reply is read aloud by a text-to-speech engine and shown as a live "
        "caption, word for word. Output ONLY plain spoken words:\n"
        "- NO Markdown or formatting characters of any kind — no asterisks "
        "(`*`, `**`), underscores, backticks, pound/hash signs, bullet points, or "
        "numbered-list markers.\n"
        '- NO emojis, and NO ellipses ("…" or "..."). Never trail off — finish '
        "every sentence you start.\n"
        "- Speak the way a person talks: short, complete sentences. To emphasize a "
        "word, just say it plainly — never wrap it in symbols.\n"
        "- If you have little to say, say one short natural sentence — never emit "
        "placeholder dots, asterisks, or symbols."
    )


def render_agent_playbook_section() -> str:
    """AGENT PLAYBOOK — situational sequences the agent should follow.

    Documents the quiz flow (S64e) and the vision / visitor-annotation flow
    (S67b). New entries can be added as additional Page types ship their own
    multi-step interactions. Returns a stable string regardless of active
    Page; the per-page verb listing lives in the CANVAS PAGE section, not here.
    """
    return (
        "## AGENT PLAYBOOK\n"
        "\n"
        "**Quiz flow** — if the user asks for a quiz, asks to be quizzed, or asks "
        "to test their knowledge:\n"
        "\n"
        "1. **First, SPEAK a short acknowledgement aloud.** One sentence is enough "
        '— for example: "Alright, let me put a few questions together for you." '
        'or "Sure, give me just a second to pull some questions." '
        "**Do NOT call any tool yet.** Speak first, in the same turn, so the user "
        "hears your voice immediately. Dead air while a tool runs feels broken to "
        "the user, and this acknowledgement masks the generation latency.\n"
        "2. **Then**, in the same turn, call `generate_quiz_from_knowledge` with "
        "`count=3` (or whatever the user requested) and the conversation's "
        "language. The tool call MUST come AFTER the spoken sentence in step 1, "
        "never before it. This tool **both** generates the quiz **and** activates "
        "the quiz Page in one step — do NOT call `canvas_set_page` afterwards.\n"
        "3. When `generate_quiz_from_knowledge` returns, the quiz Page is "
        "already showing the first question. Read it aloud from the active "
        "Page's `semanticState.questionText` (the source of truth for what's "
        "currently on screen). Map the choices for the user as needed from "
        "`semanticState.choices`.\n"
        '4. When the user answers verbally (e.g. "I\'ll go with B" or "the answer '
        "is Paris\"), call `canvas_action` with `verb='submit_answer'` and "
        '`args={"choice": "<letter>"}`. Use `semanticState.choices` to map '
        "the user's words to a choice id (A/B/C/D).\n"
        '   - **If the user says "I don\'t know", "I\'m not sure", "skip", '
        '"no idea", "pass", or otherwise opts out of answering:** call '
        "`canvas_action` with `verb='skip_question'` and `args={}` INSTEAD of "
        "submit_answer. The Page reveals the correct answer, shows the "
        "explanation, and auto-advances on the same timer as a real answer. "
        "The tool returns `{skipped: true, correct: false, completed: bool}`. "
        "Narrate the correct answer + a brief why during the reveal. Count it "
        "as not-correct in your running score tally.\n"
        "5. The tool returns `{choice, correct: true/false, completed: "
        "bool}`. Interpret it like this:\n"
        "   - **`correct: true/false`** — narrate the result naturally and "
        "briefly explain WHY the right answer is right (one short sentence). "
        "The Quiz Page is, in parallel, showing the visual feedback "
        "(correct/wrong highlight, confetti or wrong flag, the explanation "
        "banner). Keep your narration timed with that — speak the result "
        "promptly, then a short explanation. The Page will auto-advance to "
        "the next question a few seconds after the answer; you do NOT need "
        "to call any tool to make that happen.\n"
        "   - After the auto-advance, the Page emits a stateChange and "
        "`semanticState.questionText` updates to the new question. Read the "
        "new question aloud right after your explanation — by the time you "
        "finish the explanation, the iframe has advanced. (You may read from "
        "your memory of the quiz blob too; both are in sync.)\n"
        "   - **`completed: true`** — the visitor just answered the LAST "
        "question. The Page stays on the final result view (no advance). "
        "Narrate a brief wrap-up using the running tally you've been keeping "
        '(see "Score tracking" below), e.g. "You got 2 out of 3 — nice '
        'work!". Then ask whether the visitor wants to continue with the '
        "lesson; if yes, call `canvas_set_page` with `pageType='composition'` "
        "to return to the scene view.\n"
        "\n"
        "**Score tracking** — keep a running mental tally as each "
        "`correct: true/false` comes back from `submit_answer`. The Page does "
        "not compute or surface a final score; reporting it to the user at the "
        "end of the quiz is the agent's job.\n"
        "\n"
        "**Do NOT call `canvas_set_page` with `pageType='quiz'`** — quiz Page "
        "activation is bundled into `generate_quiz_from_knowledge`. Calling "
        "`canvas_set_page` directly cannot deliver the questions blob.\n"
        "\n"
        "**Do NOT call `canvas_control(verb='next_question')` between "
        "questions** — the Quiz Page auto-advances itself a few seconds after "
        "each `submit_answer` so the visitor has time to see the visual "
        "feedback (confetti / wrong flag / explanation banner). Calling "
        "`next_question` yourself would race the auto-advance and skip a "
        "question. (The verb is still available for edge cases like the user "
        "explicitly asking to skip, or going back via `previous_question`.)\n"
        "\n"
        "**Do NOT call `canvas_action(verb='show_explanation')`** as a "
        "routine step after submit_answer — the Page reveals the explanation "
        "banner automatically as part of its post-answer sequence. Reserve "
        "this verb for the rare case where you want the banner revealed "
        "without the visitor having answered (e.g. they asked \"what's the "
        'explanation?" mid-question).\n'
        "\n"
        "To exit the quiz back to the regular scene view at any time, call "
        "`canvas_set_page` with `pageType='composition'` (`pageInit` can be "
        "empty; the visitor's shell will rebuild it from the snapshot).\n"
        "\n"
        "**Visual questions & visitor annotations (vision)** — the live screen "
        "(the actual rendered frame, any video, and anything the visitor has "
        "drawn) is NOT in your text context. Whenever the visitor asks what is on "
        "the screen, what you see, to look at / read / check the screen, OR refers "
        "to something they've drawn, circled, highlighted, or written — e.g. "
        '"what do you see on the screen?", "what\'s showing right now?", "what '
        'am I pointing at?", "what did I circle?", "is this answer I wrote '
        'correct?" — do NOT answer from the scene description; instead:\n'
        "\n"
        "1. Call `canvas_analyze` with the visitor's question. This triggers a "
        "live look at what the visitor actually sees, including their own "
        "annotations, and grounds your answer in the real pixels. (You do not "
        "choose how to look — just pass their question.)\n"
        "2. A system message tagged `[vision: ...]` will then appear in your "
        "context with the visual reasoning. Base your spoken reply on it, and "
        "speak it naturally — never read the `[vision: ...]` tag aloud.\n"
        "3. **Honesty rule — never invent what you cannot see.** If a "
        "`[vision note: ...]` says only the base scene is visible (the "
        "visitor's own drawings are NOT visible), or that you cannot see the "
        "canvas this turn, you MUST NOT claim to see their circles, marks, "
        "drawings, or handwriting. Say plainly that you can describe the scene "
        "but cannot see what they drew, and invite them to share their screen "
        "so you can. Never fabricate seeing an annotation.\n"
        "\n"
        "**Canvas annotation (`canvas_annotate`)** — to point at, circle, highlight, or "
        "label something the visitor is discussing, call `canvas_annotate`:\n"
        "\n"
        "1. Pick an `op`: `circle`, `arrow`, `shape`, `highlight`, `text`, or `erase`.\n"
        "2. Pick a `target` (required for every op except `erase`):\n"
        '   - `{element: "<alias>"}` — when it\'s a known scene element (composition '
        "scenes; use an alias from CANVAS ELEMENTS).\n"
        '   - `{describe: "..."}` — when it\'s something in a video or image; you will '
        "look at the live screen to locate it.\n"
        "   - `{region: {x, y, w, h}}` — only if you already have normalized 0-1 coords.\n"
        "3. Annotations appear on the SAME overlay the visitor draws on and clear on "
        "scene change. Be purposeful and sparse — annotate to clarify, not decorate; "
        "use `op='erase'` to clear when you're done.\n"
        "\n"
        "The old in-iframe `canvas_highlight` tool no longer exists — never reference an "
        "in-iframe highlight; annotate on the overlay with `canvas_annotate` instead."
    )


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
        "2. **canvas_annotate(op, target)** — draw a temporary annotation on the overlay "
        "the visitor sees (op: circle | arrow | shape | highlight | text | erase). target is "
        '{element: "<alias>"}, {describe: "..."}, or {region: {x, y, w, h}}. See AGENT PLAYBOOK.\n'
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
        '- Annotations persist until canvas_annotate(op="erase") or scene change.\n'
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
            desc += f" (label: {el['label']})"
        if el.get("title"):
            desc += f" (title: {el['title']})"
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
    # S65 (Option B) — snapshot nested under {live_room, flow_state,
    # current_scene, knowledge, survey}. Pull blocks once.
    live_room_block = snapshot.get("live_room") or {}
    current_scene_block = snapshot.get("current_scene") or {}

    language = live_room_block.get("language") or "en"

    # ── Stable prefix (sections 1 → 5b) ──
    stable_parts: list[str] = []

    stable_parts.append(f"# LANGUAGE\n{build_language_directive(language)}")

    persona = (snapshot.get("persona") or "").strip()
    if persona:
        stable_parts.append(f"# PERSONA\n{persona}")

    audience = build_recipient_context(live_room_block.get("recipient_prompt"))
    if audience:
        stable_parts.append(audience.lstrip("\n"))

    knowledge_section = build_knowledge_context(snapshot.get("knowledge"))
    if knowledge_section:
        stable_parts.append(knowledge_section.lstrip("\n"))

    link_narration = build_link_narration_directive(current_scene_block.get("link"))
    if link_narration:
        stable_parts.append(link_narration)

    instruction_text = (current_scene_block.get("instruction") or "").strip()
    if instruction_text:
        stable_parts.append(f"# SCENE INSTRUCTION\n{instruction_text}")

    stable_parts.append(render_canvas_page_section(canvas_manifest))

    stable = "\n\n".join(s for s in stable_parts if s)

    # ── Dynamic suffix (sections 6 → 9) ──
    dynamic_parts: list[str] = []

    display_block = _render_display_mode(current_scene_block.get("avatar_display_mode"))
    if display_block:
        dynamic_parts.append(display_block)

    elements_block = _render_canvas_elements(current_scene_block.get("elements"))
    if elements_block:
        dynamic_parts.append(elements_block)

    dynamic_parts.append(render_canvas_actions_section())

    # S64e — AGENT PLAYBOOK sits between CANVAS ACTIONS and the closing
    # LANGUAGE reminder. Stable string, but kept in the dynamic suffix
    # because it references tools registered for this session.
    dynamic_parts.append(render_agent_playbook_section())

    scripts_section = build_scripts_section(snapshot)
    if scripts_section:
        dynamic_parts.append(scripts_section)

    dynamic_parts.append(build_language_reminder(language))

    dynamic = "\n\n".join(s for s in dynamic_parts if s)

    return stable, dynamic
