# Document RAG Chatbot

A small **Retrieval-Augmented Generation (RAG)** project built with LangChain components, Chroma, Hugging Face embeddings, Groq, and Streamlit.

The app lets a user upload a PDF, splits the text into chunks, converts those chunks into embeddings, stores them in a vector database, retrieves the most relevant chunks for each question, and asks an LLM to answer **only from the retrieved document context**.



## Features

- PDF upload through a Streamlit interface
- Recursive text chunking with overlap
- Multilingual Hugging Face embeddings
- Chroma vector database
- Similarity-based retrieval
- Groq LLM (`openai/gpt-oss-20b`)
- Hallucination control through a document-only system prompt
- Retrieved source chunks and page numbers shown with each answer
- Clear error handling for missing API keys, unreadable PDFs, and API errors
- Optional terminal version in `main.py`

## RAG pipeline

```text
PDF
  ↓
Extract text page by page
  ↓
Split into overlapping chunks
  ↓
Create embeddings
  ↓
Store chunks in Chroma
  ↓
User asks a question
  ↓
Retrieve the most relevant chunks
  ↓
Question + retrieved context → Groq LLM
  ↓
Answer + source pages
```

## Project structure

```text
document-rag-chatbot/
├── app.py                 # Streamlit interface
├── rag.py                 # PDF, chunking, embeddings, retrieval, LLM logic
├── main.py                # Optional command-line interface
├── requirements.txt
├── .gitignore
├── .env.example
├── sample_nlp_rag.pdf
└── sample_travel_guide.pdf
    
```

## Installation



Activate the virtual environment.


Install the dependencies:

```bash
pip install -r requirements.txt
```

The first time the project creates embeddings, the sentence-transformer model may need to download. Later runs can reuse the local model cache.

## Run the Streamlit app

```bash
streamlit run app.py
```

Then:

1. Enter your Groq API key in the sidebar.
2. Upload a PDF.
3. Click **Process document**.
4. Ask a question about the PDF.
5. Open the **Sources** expanders to see the chunks and page numbers used for retrieval.

The API key field is a password field and the app does not write the key to a file.





Example question:

```text
When is the best time to visit Japan?
```

## Main technologies

| Part | Tool |
|---|---|
| PDF extraction | `pypdf` |
| Chunking | `RecursiveCharacterTextSplitter` |
| Embeddings | `paraphrase-multilingual-MiniLM-L12-v2` |
| Vector database | Chroma |
| Retrieval | LangChain retriever |
| LLM | Groq `openai/gpt-oss-20b` |
| Interface | Streamlit |

## How hallucination control works

The system prompt tells the model to answer only from the retrieved chunks and to say that the information is unavailable when the document does not contain the answer. This reduces hallucinations, although no prompt can guarantee that hallucinations never happen.

## Concepts demonstrated

This project directly demonstrates:

- Embeddings
- Chunking and chunk overlap
- Vector databases
- Similarity retrieval
- Large Language Models (LLMs)
- Retrieval-Augmented Generation (RAG)
- Prompt-based hallucination control

Related concepts such as tokenization, TF-IDF, transformers, and encoder-vs-decoder architecture are learning topics around the project, even though they are not all implemented as separate features in the app.

## Example test cases

After processing `sample_travel_guide.pdf`, try questions such as:

```text
When is the best time to visit Japan?
```

Also test a question whose answer is not present in the document. The chatbot should say that the information is not available instead of inventing an answer.

## Notes

- Page numbers displayed in the interface are human-friendly and start from 1.
- The vector store is kept in memory for this project, so uploading/processing a new document rebuilds the index.

