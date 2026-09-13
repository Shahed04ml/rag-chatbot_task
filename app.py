"""Streamlit interface for the document RAG chatbot."""

import os
import tempfile
from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="Document RAG Chatbot",
    page_icon="📚",
    layout="centered",
)

st.title("📚 Document RAG Chatbot")
st.caption("Upload a PDF, index it, and ask questions answered only from the document.")

with st.sidebar:
    st.header("Settings")
    env_key = os.getenv("GROQ_API_KEY", "")
    api_key = st.text_input(
        "Groq API key",
        value=env_key,
        type="password",
        help="The key is used only for this session and is not written to a file.",
    )
    top_k = st.slider("Retrieved chunks", min_value=1, max_value=6, value=3)
    chunk_size = st.slider("Chunk size", min_value=300, max_value=1500, value=800, step=100)
    chunk_overlap = st.slider(
        "Chunk overlap", min_value=0, max_value=300, value=150, step=25
    )

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "document_name" not in st.session_state:
    st.session_state.document_name = None

if uploaded_file is not None:
    st.info(f"Selected: **{uploaded_file.name}**")

    if st.button("Process document", type="primary", use_container_width=True):
        if chunk_overlap >= chunk_size:
            st.error("Chunk overlap must be smaller than chunk size.")
        else:
            try:
                from rag import prepare_pdf

                with st.spinner("Reading, chunking, and creating embeddings..."):
                    suffix = Path(uploaded_file.name).suffix or ".pdf"
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                        temp_file.write(uploaded_file.getbuffer())
                        temp_path = temp_file.name

                    try:
                        vector_store, chunks = prepare_pdf(
                            temp_path,
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap,
                        )
                    finally:
                        Path(temp_path).unlink(missing_ok=True)

                st.session_state.vector_store = vector_store
                st.session_state.chunks = chunks
                st.session_state.document_name = uploaded_file.name
                st.success(
                    f"Document ready — {len(chunks)} chunks created. You can ask questions now."
                )
            except Exception as exc:
                st.session_state.vector_store = None
                st.error(f"Could not process the PDF: {exc}")

if st.session_state.vector_store is not None:
    st.divider()
    st.subheader("Ask the document")
    st.caption(f"Current document: {st.session_state.document_name}")

    question = st.text_input(
        "Question",
        placeholder="Example: When is the best time to visit Japan?",
    )

    if st.button("Ask", use_container_width=True):
        if not api_key.strip():
            st.error("Enter your Groq API key in the sidebar first.")
        elif not question.strip():
            st.warning("Write a question first.")
        else:
            try:
                from rag import answer_question

                with st.spinner("Retrieving relevant chunks and generating the answer..."):
                    answer, sources = answer_question(
                        st.session_state.vector_store,
                        question,
                        api_key,
                        top_k=top_k,
                    )

                st.markdown("### Answer")
                st.write(answer)

                st.markdown("### Sources")
                for index, doc in enumerate(sources, start=1):
                    page = doc.metadata.get("page", "?")
                    shown_page = page + 1 if isinstance(page, int) else page
                    with st.expander(f"Source {index} — page {shown_page}"):
                        st.write(doc.page_content)
            except Exception as exc:
                message = str(exc)
                if "401" in message or "authentication" in message.lower():
                    st.error("Groq rejected the API key. Check the key and try again.")
                elif "model" in message.lower() and "not" in message.lower():
                    st.error("The configured Groq model is unavailable. Check LLM_MODEL in rag.py.")
                else:
                    st.error(f"Could not generate an answer: {exc}")
else:
    st.caption("Process a PDF first; the question box will appear here afterward.")
