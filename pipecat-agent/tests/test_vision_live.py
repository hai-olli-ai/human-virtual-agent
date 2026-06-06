"""S67b — LIVE coverage of the real Gemini vision call.

Every test in test_canvas_vision.py mocks ``VisionClient`` (orchestration,
fallback, stub). This file is the one place that exercises the REAL call:
``VisionClient.analyze_image`` → ``gemini-3.5-flash`` over the network.

It is gated two ways so normal runs / CI never hit the paid API:

  * ``@pytest.mark.live`` (module-level) — excluded by the ``addopts``
    ``-m 'not live'`` default in pyproject.toml. Run it explicitly with:

        pytest -m live tests/test_vision_live.py -v

  * a runtime skip when ``GOOGLE_AI_API_KEY`` is absent — so ``pytest -m live``
    still passes cleanly on a machine without a key.

The image + expected tokens come from ``vision_probe`` so this test and the
standalone probe stay in lockstep.
"""

from __future__ import annotations

import asyncio
import os

import pytest
from dotenv import load_dotenv

from services.vision_client import VISION_UNAVAILABLE, VisionClient
from vision_probe import EXPECTED_TOKENS, make_probe_image

pytestmark = pytest.mark.live


def test_vision_client_live_describes_image():
    """The production VisionClient path returns a real, image-grounded answer."""
    load_dotenv(override=True)  # only when this live test actually runs
    if not os.getenv("GOOGLE_AI_API_KEY"):
        pytest.skip("GOOGLE_AI_API_KEY not set — live vision test skipped")

    vc = VisionClient()
    assert vc.enabled, "key present but VisionClient reports disabled"

    answer = asyncio.run(vc.analyze_image(make_probe_image(), "describe"))

    # The real call must NOT degrade to the fallback sentinel.
    assert answer != VISION_UNAVAILABLE, "real Gemini call degraded to VISION_UNAVAILABLE"
    assert answer.strip(), "vision returned empty text"

    # Proof it actually saw the pixels (not just that the API answered).
    low = answer.lower()
    assert any(tok in low for tok in EXPECTED_TOKENS), (
        f"vision answer mentioned no expected visual token {EXPECTED_TOKENS}; got: {answer!r}"
    )
