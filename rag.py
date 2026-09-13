"""Core RAG logic for the document chatbot."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Iterable
from uuid import uuid4

from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq

EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
LLM_MODEL = "openai/gpt-oss-20b"

SYSTEM_PROMPT = """You are a document question-answering assistant.
Answer ONLY from the supplied context.
- If the answer is supported by the context, answer clearly and concisely.
- If the answer is not in the context, say that the information is not available in the document.
- Do not invent facts.
- Keep the answer in the same language as the user's question when possible.

Context:
{context}
"""


def load_pdf(pdf_path: str | Path) -> list[Document]:
    """Read a PDF and return one LangChain Document per page."""
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError("Only PDF files are supported.")

    reader = PdfReader(str(path))
    documents: list[Document] = []

    for page_index, page in enumerate(reader.pages):
        text = (page.extract_text() or "").strip()
        if text:
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "page": page_index,
                        "source": path.name,
                    },
                )
            )

    if not documents:
        raise ValueError(
            "No readable text was found in the PDF. "
            "The file may be scanned and require OCR."
        )

    return documents


def split_documents(
    documents: Iterable[Document],
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> list[Document]:
    """Split pages into overlapping chunks for retrieval."""
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "، ", " ", ""],
    )
    return splitter.split_documents(list(documents))


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """Load the local sentence-transformer once per Python process."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def build_vector_store(chunks: list[Document]) -> Chroma:
    """Embed chunks and store them in an in-memory Chroma collection."""
    if not chunks:
        raise ValueError("No chunks were created from the PDF.")

    return Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        collection_name=f"document_rag_{uuid4().hex}",
    )


def _format_docs(docs: list[Document]) -> str:
    parts = []
    for doc in docs:
        # pypdf/LangChain page index is zero-based, so show humans page + 1.
        page = doc.metadata.get("page", "?")
        shown_page = page + 1 if isinstance(page, int) else page
        parts.append(f"[Page {shown_page}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def answer_question(
    vector_store: Chroma,
    question: str,
    api_key: str,
    top_k: int = 3,
) -> tuple[str, list[Document]]:
    """Retrieve relevant chunks and answer using Groq."""
    question = question.strip()
    if not question:
        raise ValueError("Question cannot be empty.")
    if not api_key.strip():
        raise ValueError("A Groq API key is required.")

    retriever = vector_store.as_retriever(search_kwargs={"k": top_k})
    retrieved_docs = retriever.invoke(question)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "Question: {question}"),
        ]
    )
    llm = ChatGroq(
        model=LLM_MODEL,
        temperature=0,
        api_key=api_key,
    )
    chain = prompt | llm | StrOutputParser()

    answer = chain.invoke(
        {
            "context": _format_docs(retrieved_docs),
            "question": question,
        }
    )
    return answer, retrieved_docs


def prepare_pdf(
    pdf_path: str | Path,
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> tuple[Chroma, list[Document]]:
    """Convenience function: load, split, and index a PDF."""
    pages = load_pdf(pdf_path)
    chunks = split_documents(pages, chunk_size, chunk_overlap)
    vector_store = build_vector_store(chunks)
    return vector_store, chunks
