"""
tests/test_chain.py
Basic tests — no AWS connection required (uses mocks).
Run: python -m pytest tests/
"""
from unittest.mock import patch, MagicMock
from query.chain import build_context, ask
from query.retriever import Chunk


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_chunk(text="Sample text", source="docs/manual.pdf", page=3, score=0.95):
    return Chunk(text=text, source=source, page=page, score=score)


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_build_context_format():
    chunks = [
        make_chunk("The system requires Python 3.10.", page=1),
        make_chunk("Installation takes about 5 minutes.", page=2),
    ]
    context = build_context(chunks)
    assert "manual.pdf" in context
    assert "p.1" in context
    assert "Python 3.10" in context
    assert "[1]" in context
    assert "[2]" in context


def test_build_context_empty():
    assert build_context([]) == ""


@patch("query.chain.retrieve")
@patch("query.chain.build_llm")
def test_ask_with_chunks(mock_llm, mock_retrieve):
    mock_retrieve.return_value = [make_chunk("Requires 8GB of RAM.")]

    mock_response         = MagicMock()
    mock_response.content = "The system requires 8GB of RAM.\n\n📄 Sources: [manual.pdf, page 3]"
    mock_chain            = MagicMock()
    mock_chain.invoke.return_value = mock_response
    mock_llm.return_value.__or__ = lambda self, other: mock_chain

    with patch("query.chain.ChatPromptTemplate") as mock_prompt:
        mock_prompt.from_template.return_value.__or__ = lambda self, other: mock_chain
        result = ask("How much RAM do I need?")

    assert "8GB" in result.answer
    assert len(result.sources) == 1
    assert result.question == "How much RAM do I need?"


@patch("query.chain.retrieve")
def test_ask_no_chunks(mock_retrieve):
    mock_retrieve.return_value = []
    result = ask("Anything?")
    assert "ingestion" in result.answer.lower() or "no indexed" in result.answer.lower()
    assert result.sources == []
