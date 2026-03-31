````markdown
# Intelligent Document Query Engine

A FastAPI-based Retrieval-Augmented Generation (RAG) API for answering questions over PDF documents.

This project downloads a PDF from a URL, extracts and chunks its text, builds a FAISS vector index over semantic embeddings, reranks retrieved chunks with a cross-encoder, and generates grounded answers with page-level source references.

It is built as a clean, portfolio-ready document QA system.

---

## Overview

Given:
- a PDF URL
- a list of questions

the API will:
1. download the PDF
2. extract text page by page
3. split the text into chunks
4. embed the chunks using a sentence-transformer model
5. build an in-memory FAISS index
6. retrieve relevant chunks for each question
7. rerank retrieved chunks with a cross-encoder
8. generate an answer with an LLM
9. return the answer along with grounded source references

---

## Features

- PDF ingestion from URL
- text extraction with PyMuPDF
- page-aware chunking
- semantic retrieval with FAISS
- reranking with a cross-encoder
- grounded answer objects with:
  - page number
  - chunk ID
  - excerpt
- support for batch question answering
- per-question failure isolation
- Bearer token authentication
- in-memory document caching
- unit tests covering ingestion, retrieval, response structure, and cache behavior

---

## Tech Stack

- **Backend:** FastAPI
- **PDF Parsing:** PyMuPDF
- **Chunking:** LangChain RecursiveCharacterTextSplitter
- **Embeddings:** `all-MiniLM-L6-v2`
- **Vector Search:** FAISS
- **Reranker:** `cross-encoder/ms-marco-TinyBERT-L-2-v2`
- **LLM Inference:** Groq
- **Testing:** pytest

---

## How It Works

The pipeline is:

**PDF URL → download → text extraction → chunking → embeddings → FAISS retrieval → reranking → LLM answer generation → grounded response**

### Retrieval Pipeline
- The document is fetched from a remote URL.
- Text is extracted page by page.
- Each page is chunked using overlapping text windows.
- Chunks are embedded and stored in a FAISS index.
- For each question:
  - relevant chunks are retrieved by vector similarity
  - retrieved chunks are reranked
  - top chunks are passed to the LLM as context
- The final response includes both the answer and the supporting source references.

---

## API

### `POST /hackrx/run`

Processes a document and answers up to 10 questions.

### Request Body

```json
{
  "documents": "https://example.com/document.pdf",
  "questions": [
    "What is this document about?",
    "What files are included in the sample package?"
  ]
}
````

### Response Body

```json
{
  "answers": [
    {
      "question": "What is this document about?",
      "answer": "The document describes ...",
      "status": "ok",
      "sources": [
        {
          "page": 1,
          "chunk_id": 0,
          "excerpt": "Sample excerpt from the source text..."
        }
      ]
    }
  ]
}
```

### Answer Status Values

Each answer includes a `status` field:

* `ok` — the answer was generated successfully
* `no_context` — relevant information was not found in the document
* `error` — processing failed for that specific question

### Headers

The API also exposes a cache header:

* `X-Document-Cache: MISS` — document was processed fresh
* `X-Document-Cache: HIT` — cached document/index was reused

---

## Health Check

### `GET /health`

Returns a simple health response for the API.

Example:

```json
{
  "status": "healthy",
  "version": "2.1.0"
}
```

---

## Project Structure

```text
.
├── main.py
├── test_unit.py
├── test_api.py
├── requirements.txt
├── .env.example
└── README.md
```

### File Descriptions

* `main.py` — FastAPI app and complete RAG pipeline
* `test_unit.py` — unit tests for ingestion, retrieval, auth, cache behavior, and endpoint responses
* `test_api.py` — simple integration-style script for local API checks
* `requirements.txt` — Python dependencies
* `.env.example` — sample environment variable file

---

## Local Setup

### 1. Clone the repository

```bash
git clone <your-github-repo-url>
cd Intelligent-Document-Query-Engine-main
```

### 2. Create a virtual environment

#### Windows (PowerShell)

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

#### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. If needed, install `tf-keras`

Some environments may raise a TensorFlow / Keras compatibility error while loading `sentence_transformers`. If that happens, run:

```bash
pip install tf-keras
```

### 5. Set environment variables

#### Windows (PowerShell)

```powershell
$env:GROQ_API_KEY="your_groq_api_key"
$env:API_TOKEN="your_api_token"
```

#### macOS / Linux

```bash
export GROQ_API_KEY="your_groq_api_key"
export API_TOKEN="your_api_token"
```

### 6. Run the server

```bash
uvicorn main:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

---

## Environment Variables

### Required

* `GROQ_API_KEY` — API key used for Groq inference
* `API_TOKEN` — Bearer token required to access the API

### Optional

* `MAX_PDF_BYTES` — maximum allowed PDF size in bytes
  Default: `15728640` (15 MB)

---

## Example Usage

Create a file named `body.json`:

```json
{
  "documents": "https://www.adobe.com/support/products/enterprise/knowledgecenter/media/c4611_sample_explain.pdf",
  "questions": [
    "What features are demonstrated in this sample?",
    "What files are included in the sample package?",
    "How do you deploy the sample in your environment?"
  ]
}
```

### Windows (PowerShell)

```powershell
curl.exe -X POST "http://127.0.0.1:8000/hackrx/run" `
  -H "Authorization: Bearer your_api_token" `
  -H "Content-Type: application/json" `
  -d "@body.json"
```

### Example Response

```json
{
  "answers": [
    {
      "question": "What features are demonstrated in this sample?",
      "answer": "The sample demonstrates primary and secondary bookmarks in a PDF file.",
      "status": "ok",
      "sources": [
        {
          "page": 1,
          "chunk_id": 0,
          "excerpt": "Features Demonstrated: • Primary bookmarks in a PDF file. • Secondary bookmarks in a PDF file."
        }
      ]
    },
    {
      "question": "What files are included in the sample package?",
      "answer": "The sample package contains ap_bookmark.IFD, ap_bookmark.mdf, ap_bookmark.dat, ap_bookmark.bmk, ap_bookmark.pdf, and ap_bookmark_doc.pdf.",
      "status": "ok",
      "sources": [
        {
          "page": 3,
          "chunk_id": 5,
          "excerpt": "Sample Files This sample package contains: ap_bookmark.IFD ... ap_bookmark_doc.pdf ..."
        }
      ]
    }
  ]
}
```

---

## Testing

### Run unit tests

```bash
pytest test_unit.py -v
```

### Run the local integration script

Start the server first, then run:

```bash
python test_api.py
```
```
```
