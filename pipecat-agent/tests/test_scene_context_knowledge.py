import pytest
from scene_context import build_knowledge_context, _format_scope


# Minimal fixtures

def _scope(faqs=None, sources=None, urls=None):
    return {
        "faqs":    faqs    or [],
        "sources": sources or [],
        "urls":    urls    or [],
    }


def test_none_returns_empty_string():
    assert build_knowledge_context(None) == ""
    assert build_knowledge_context({}) == ""


def test_only_scene_scope():
    knowledge = {
        "scene": _scope(faqs=[{"question": "Hi?", "answer": "Hello."}]),
        "flow": None,
    }
    out = build_knowledge_context(knowledge)
    assert "# SCENE KNOWLEDGE" in out
    assert "# FLOW KNOWLEDGE" not in out
    assert "Q: Hi?" in out
    assert "A: Hello." in out


def test_only_flow_scope():
    knowledge = {
        "scene": None,
        "flow": _scope(faqs=[{"question": "Where?", "answer": "SF."}]),
    }
    out = build_knowledge_context(knowledge)
    assert "# FLOW KNOWLEDGE" in out
    assert "# SCENE KNOWLEDGE" not in out


def test_flow_first_then_scene():
    knowledge = {
        "flow":  _scope(faqs=[{"question": "FQ", "answer": "FA"}]),
        "scene": _scope(faqs=[{"question": "SQ", "answer": "SA"}]),
    }
    out = build_knowledge_context(knowledge)
    flow_idx = out.index("# FLOW KNOWLEDGE")
    scene_idx = out.index("# SCENE KNOWLEDGE")
    assert flow_idx < scene_idx


def test_faq_before_documents_before_urls_within_scope():
    knowledge = {
        "scene": _scope(
            faqs=[{"question": "Q", "answer": "A"}],
            sources=[{"file_name": "doc.pdf", "extracted_text": "DOC TEXT"}],
            urls=[{"url": "https://x", "title": "Xpage", "markdown_content": "URL TEXT"}],
        ),
        "flow": None,
    }
    out = build_knowledge_context(knowledge)
    faq_idx = out.index("FAQ")
    doc_idx = out.index("DOC TEXT")
    url_idx = out.index("URL TEXT")
    assert faq_idx < doc_idx < url_idx


def test_empty_text_items_are_skipped():
    knowledge = {
        "scene": _scope(
            sources=[{"file_name": "empty.pdf", "extracted_text": ""}],
            urls=[{"url": "https://y", "title": None, "markdown_content": "   "}],
        ),
        "flow": None,
    }
    out = build_knowledge_context(knowledge)
    # All items skipped → no SCENE section at all
    assert out == ""


def test_missing_title_falls_back_to_url():
    knowledge = {
        "scene": _scope(
            urls=[{"url": "https://fallback.example", "title": None, "markdown_content": "content"}],
        ),
        "flow": None,
    }
    out = build_knowledge_context(knowledge)
    assert "https://fallback.example" in out


def test_defensive_against_missing_keys():
    # Real snapshots may omit keys; the function should not KeyError
    knowledge = {"scene": {"faqs": [{"question": "Q", "answer": "A"}]}, "flow": None}
    out = build_knowledge_context(knowledge)
    assert "Q: Q" in out
