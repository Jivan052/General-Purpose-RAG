# General Purpose RAG Lab

A lightweight local RAG playground for learning and experimenting with retrieval-augmented generation. Upload a PDF, TXT, or MD file, ask questions, and inspect the exact chunks that were retrieved and the timings for both retrieval and LLM generation.

This project uses:

- FastAPI for the backend API
- Qdrant in embedded local mode for vector storage
- OpenRouter for embeddings and LLM calls
- A single static HTML page for the frontend

No Docker is required.

## Project structure

```text
rag-lab/
├── README.md
├── .gitignore
├── backend/
│   ├── requirements.txt
│   ├── qdrant_data/
│   └── app/
│       ├── main.py
│       ├── config.py
│       ├── routers/
│       │   ├── ingest.py
│       │   └── query.py
│       └── services/
│           ├── extraction.py
│           ├── chunking.py
│           ├── embeddings.py
│           ├── llm.py
│           └── qdrant_store.py
└── frontend/
    └── index.html
```

## Stack

- Frontend: plain HTML/CSS/JS in `frontend/index.html`
- Backend: FastAPI in `backend/app`
- Vector DB: Qdrant embedded mode
- Extraction: PyMuPDF
- Embedding model: OpenRouter-compatible model such as `openai/text-embedding-3-small`
- LLM: OpenRouter-compatible model such as `openai/gpt-oss-20b:free` or another model you choose

## Quick start

### 1. Create a virtual environment and install dependencies

```bash
cd /path/to/rag-lab
python3 -m venv venv
source venv/bin/activate
pip install -r backend/requirements.txt
```

### 2. Start the backend

From the project root:

```bash
cd backend
../venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Check the app is running:

```bash
curl http://localhost:8000/health
```

Expected response:

```json
{"status":"ok"}
```

### 3. Open the frontend

Open the file directly in a browser:

```bash
open frontend/index.html
```

or open it with your browser manually from the project folder.

## Configure settings from the UI

The frontend includes settings for:

- LLM Base URL
- LLM API Key
- LLM Model
- Embedding Model
- Embedding Dimension
- Top K retrieval count

You can set these values directly in the page and click Save. The values are stored in browser `sessionStorage` while the tab remains open, so they are not lost on refresh during the current session.

The backend also exposes:

- `GET /health`
- `GET /config`
- `POST /config`

This runtime config is used by the app in memory and clears cached clients after updates.

## How the workflow works

1. Upload a supported file
2. The backend extracts text from the document
3. Text is chunked and embedded
4. Embeddings are stored in the local Qdrant collection
5. A user query is embedded and compared against stored vectors
6. The top matching chunks are returned to the LLM
7. The LLM generates a concise answer based on the retrieved context
8. The result is shown with retrieval and LLM timing metrics

## Example usage

1. Open the app in the browser.
2. Enter your OpenRouter details in the config panel.
3. Upload a PDF, TXT, or MD file.
4. Click Run RAG or ask a question.
5. Review:
   - the retrieval score cards
   - the retrieved chunks
   - the answer
   - the time spent on retrieval and LLM response

## OpenRouter setup

Create an API key at https://openrouter.ai and then add credit or use a free model.

Typical values:

```text
LLM Base URL: https://openrouter.ai/api/v1
LLM API Key: sk-or-v1-...
LLM Model: openai/gpt-oss-20b:free
Embedding Model: openai/text-embedding-3-small
Embedding Dim: 1536
```

The app is built to work with OpenRouter's OpenAI-compatible API endpoints.

## Runtime configuration notes

The app does not require a permanent `.env` edit for quick local use. The browser form saves values in session memory. If you want to set defaults at process startup, you can still use environment variables through the backend config loader.

The core settings are defined in `backend/app/config.py` and include:

- `qdrant_path`
- `qdrant_collection`
- `embedding_model`
- `embedding_dim`
- `chunk_size`
- `chunk_overlap`
- `llm_base_url`
- `llm_api_key`
- `llm_model`
- `top_k`

## Resetting the local data

To reset the vector database, remove the local Qdrant data directory:

```bash
rm -rf backend/qdrant_data
```

This clears the embedded collection and lets the app rebuild from newly uploaded files.

## Notes

- This lab is intentionally simple and transparent: you can inspect retrieval steps and chunk scores directly.
- The frontend shows a live step-by-step RAG pipeline so you can follow the process from query to answer.
- The answer is intentionally kept concise by the app prompt logic so it reads more like a usable summary than a long raw model dump.