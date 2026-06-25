"""MAIN_LLM_SUPPORTS_VISION gate — text-only main-LLM compatibility (option 1).

The S46 vision path injects a scene image directly into the MAIN LLM's context
(``build_vision_message`` → an OpenAI ``image_url`` content block). Groq's
default ``gpt-oss-120b`` is text-only and returns ``400 — messages[].content
must be a string`` on image content, which would break the very first turn
whenever a scene image is present. ``bot.py`` therefore gates that injection
(session start + ``VISION_REFRESH_MODE=eager``) on ``config.MAIN_LLM_SUPPORTS_VISION``.

These tests pin the default-derivation + env-override semantics of the pure
resolver. They deliberately exercise ``_resolve_main_llm_vision`` directly
rather than reloading ``config`` — ``config.py`` runs ``load_dotenv(override=True)``
at import, so a reload would let the developer's local ``.env`` clobber any
monkeypatched provider and make the test environment-dependent.

The decoupled S67b Gemini path (``run_vision_query``) is unaffected by this flag
(it injects text reasoning, never a raw image) and is not exercised here.
"""

import pytest

from config import _resolve_main_llm_vision


def test_groq_defaults_text_only():
    # No override → derive from provider. Groq's default model is text-only.
    assert _resolve_main_llm_vision("groq", None) is False


@pytest.mark.parametrize("provider", ["openai", "anthropic", "gemini"])
def test_vision_providers_default_on(provider):
    assert _resolve_main_llm_vision(provider, None) is True


def test_override_opts_groq_back_in():
    # Operator pointing GROQ_MODEL at a multimodal Groq model flips it on.
    assert _resolve_main_llm_vision("groq", "true") is True


def test_override_can_force_off_for_vision_provider():
    assert _resolve_main_llm_vision("openai", "false") is False


@pytest.mark.parametrize("truthy", ["1", "true", "TRUE", "Yes", "on", " on "])
def test_override_truthy_values(truthy):
    assert _resolve_main_llm_vision("groq", truthy) is True


@pytest.mark.parametrize("falsey", ["0", "false", "no", "off", "", "maybe", "garbage"])
def test_override_non_truthy_is_false(falsey):
    # Anything not explicitly truthy (including empty / nonsense) reads as off.
    assert _resolve_main_llm_vision("openai", falsey) is False
