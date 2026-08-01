"""One-shot live probe for S67b canvas vision.

Run it from the agent to confirm the real Gemini path works end-to-end:

    python vision_probe.py            # uses the key in this dir's .env

It builds a describable image (a red circle, a blue rectangle, the text
"PROBE 7Q"), runs the REAL production path
(``services.vision_client.VisionClient.analyze_image`` — exactly what bot.py
calls), and prints the model's answer. If that degrades to the
``VISION_UNAVAILABLE`` sentinel, it falls back to a direct ``google-genai``
call that surfaces the raw error, then retries without ``thinking_config`` to
tell a model-ID problem apart from a param problem.

Exit code: 0 if the production path returned a real answer, 1 otherwise.
The API key is never printed — only its length.

``make_probe_image`` and ``EXPECTED_TOKENS`` are imported by
``tests/test_vision_live.py`` so the live test and this probe stay in sync.
"""

from __future__ import annotations

import asyncio
import io
import os
import sys

from dotenv import load_dotenv

# Tokens any real description of the probe image should contain (case-folded).
# Broad on purpose: the goal is "the model actually saw the image", not exact
# phrasing — Gemini said "blue square" for the rectangle in the A-* probe.
EXPECTED_TOKENS = (
    "circle",
    "red",
    "blue",
    "square",
    "rectangle",
    "probe",
    "7q",
    "shape",
    "text",
)


def make_probe_image() -> bytes:
    """Return JPEG bytes of a small, unambiguously describable scene."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (480, 280), "white")
    d = ImageDraw.Draw(img)
    d.ellipse([40, 60, 200, 220], fill=(220, 30, 30))  # red circle (left)
    d.rectangle([260, 60, 440, 220], fill=(30, 60, 220))  # blue rectangle (right)
    try:
        from PIL import ImageFont

        fnt = ImageFont.truetype(
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf", 40
        )
    except Exception:
        fnt = None  # default bitmap font — shapes/colors carry the signal anyway
    d.text((150, 12), "PROBE 7Q", fill=(0, 0, 0), font=fnt)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


async def _amain() -> int:
    load_dotenv(override=True)  # this dir's .env → GOOGLE_AI_API_KEY / VISION_MODEL

    from services.vision_client import VISION_UNAVAILABLE, VisionClient

    jpeg = make_probe_image()
    print(
        f"[env] key_len={len(os.getenv('GOOGLE_AI_API_KEY', ''))} "
        f"VISION_MODEL={os.getenv('VISION_MODEL', '(unset→default)')} image_bytes={len(jpeg)}"
    )

    # ── Layer 1: the real production code path ────────────────────────
    vc = VisionClient()
    print(f"[vc] enabled={vc.enabled} model={vc.model}")
    answer = await vc.analyze_image(jpeg, "describe")
    print(f"[vc] analyze_image('describe') -> {answer!r}")
    if answer != VISION_UNAVAILABLE:
        matched = [t for t in EXPECTED_TOKENS if t in answer.lower()]
        print(f"[RESULT] SUCCESS — live vision answer; matched tokens={matched}")
        return 0

    # ── Layer 2: direct SDK call to surface the raw failure ───────────
    print(
        "[direct] production path degraded; calling google-genai directly to surface the error…"
    )
    try:
        from google import genai
        from google.genai import types
    except Exception as e:
        print(f"[direct] google-genai not importable: {e!r}")
        return 1

    client = genai.Client(api_key=os.environ.get("GOOGLE_AI_API_KEY", ""))
    model = os.getenv("VISION_MODEL", "gemini-3.5-flash")
    parts = [
        types.Part.from_bytes(data=jpeg, mime_type="image/jpeg"),
        "Describe this image in one short sentence.",
    ]

    def _text(resp) -> str:
        try:
            return (resp.text or "").strip()
        except Exception as te:  # .text can raise when blocked / no candidates
            return f"<no .text: {te!r}>"

    try:
        r = await client.aio.models.generate_content(
            model=model,
            contents=parts,
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_level="low")
            ),
        )
        print(f"[direct +thinking] OK model={model!r} -> {_text(r)!r}")
    except Exception as e:
        print(f"[direct +thinking] ERROR {type(e).__name__}: {str(e)[:600]}")
        try:
            r2 = await client.aio.models.generate_content(model=model, contents=parts)
            print(
                f"[direct  plain  ] OK model={model!r} (thinking_config was the issue) -> {_text(r2)!r}"
            )
        except Exception as e2:
            print(f"[direct  plain  ] ERROR {type(e2).__name__}: {str(e2)[:600]}")
    return (
        1  # production path degraded — a failure signal even if the direct call worked
    )


def main() -> int:
    return asyncio.run(_amain())


if __name__ == "__main__":
    sys.exit(main())
