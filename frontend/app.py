"""
frontend/app.py
RAG chat interface built with Streamlit.
Run: python -m streamlit run frontend/app.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from query.chain import ask

# ── Config ────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="RAG · Bedrock",
    page_icon="📚",
    layout="centered",
)

st.title("📚 RAG with AWS Bedrock")
st.caption("Ask questions about your documents. Powered by Claude 3 + Titan Embeddings.")

# ── Conversation state ────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

# ── History ───────────────────────────────────────────────────────────────────

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📄 Sources used"):
                for s in msg["sources"]:
                    fname = s.source.split("/")[-1]
                    st.markdown(f"- **{fname}** · page {s.page} · score `{s.score:.3f}`")

# ── Input ─────────────────────────────────────────────────────────────────────

if prompt := st.chat_input("Ask your question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            result = ask(prompt)

        st.markdown(result.answer)

        if result.sources:
            with st.expander("📄 Sources used"):
                for s in result.sources:
                    fname = s.source.split("/")[-1]
                    st.markdown(f"- **{fname}** · page {s.page} · score `{s.score:.3f}`")

    st.session_state.messages.append({
        "role":    "assistant",
        "content": result.answer,
        "sources": result.sources,
    })
