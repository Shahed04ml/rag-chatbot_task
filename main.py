import os
import sys

from rag import answer_question, prepare_pdf


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python main.py path/to/file.pdf")
        sys.exit(1)

    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        print("Please set the GROQ_API_KEY environment variable first.")
        sys.exit(1)

    pdf_path = sys.argv[1]

    try:
        print("Reading PDF, splitting text, and building vector store...")
        vector_store, chunks = prepare_pdf(pdf_path)
        print(f"Ready. The file was split into {len(chunks)} chunks.")
    except Exception as exc:
        print(f"Could not prepare the document: {exc}")
        sys.exit(1)

    print("\nType a question, or 'exit' to quit.\n")
    while True:
        question = input("Your question: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue

        try:
            answer, sources = answer_question(vector_store, question, api_key)
            print(f"\nAnswer:\n{answer}\n")
            pages = []
            for doc in sources:
                page = doc.metadata.get("page", "?")
                pages.append(str(page + 1 if isinstance(page, int) else page))
            print("Sources (pages): " + ", ".join(dict.fromkeys(pages)))
            print("-" * 60)
        except Exception as exc:
            print(f"Error: {exc}\n")


if __name__ == "__main__":
    main()
