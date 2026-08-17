# RAG Lab

A minimal, transparent RAG pipeline for learning: upload a PDF/TXT/MD file,
ask questions about it, and see exactly which chunks were retrieved and how
similar they were to your query.

**No Docker required.** Qdrant runs embedded (in-process, on-disk) via
`qdrant-client`'s local mode. Embeddings and the LLM both go through
OpenRouter, so there's no local ML model to download either — the whole
backend install is under ~50MB.

## Stack

- **Frontend:** plain HTML/CSS/JS (`frontend/index.html`) — no build step, no server
- **Backend:** FastAPI (`backend/`)
- **Vector DB:** Qdrant, embedded mode (just a folder on disk — no server/Docker)
- **Embeddings:** OpenRouter (`openai/text-embedding-3-small`)
- **LLM:** OpenRouter (`openai/gpt-4o-mini`, or any model you swap in)
- **PDF parsing:** PyMuPDF

## Setup 

### 1. Get an OpenRouter API key

1. Sign up at https://openrouter.ai
2. Go to https://openrouter.ai/keys and create a key (starts with `sk-or-v1-...`)
3. Add credit, or pick a free model (see "Using free models" below)

### 2. Set up a Python virtual environment

```bash
cd rag-lab/backend
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

This installs FastAPI, qdrant-client, PyMuPDF, and the OpenAI SDK — no
`sentence-transformers`, no `torch`, so it's a light install (a few
hundred MB at most, mostly from PyMuPDF).

### 3. Configure your key

```bash
cd ..                          # back to rag-lab/
cp .env.example .env
```

Edit `.env`:
```
LLM_API_KEY=sk-or-v1-your-real-key-here
LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_MODEL=openai/gpt-4o-mini
```

### 4. Run the backend

```bash
cd backend
source venv/bin/activate
uvicorn app.main:app --reload --port 8000
```

You should see `Application startup complete`. On the first `/ingest` or
`/query` call, `qdrant-client` will create a `qdrant_data/` folder next to
wherever you run uvicorn from — that's your entire vector database,
just files on disk.

### 5. Open the frontend

Double-click `frontend/index.html`, or:
```bash
open frontend/index.html        # macOS
xdg-open frontend/index.html    # Linux
start frontend/index.html       # Windows
```

It talks to `http://localhost:8000` directly via `fetch` — no server needed
for the frontend at all.

## Using it

1. Upload a PDF/TXT/MD under "Ingest File" — runs extract → chunk → embed → store.
2. Ask a question — runs embed → Qdrant similarity search → top-k chunks → LLM → answer.
3. "Retrieved context" shows exactly which chunks were used and their
   cosine similarity scores.

## Keeping it small

- **No local model download.** Embeddings run through OpenRouter's API
  instead of a local sentence-transformers model, so there's nothing
  multi-hundred-MB sitting on disk.
- **Embedded Qdrant, not a server.** `qdrant_data/` only grows with what
  you actually ingest — a handful of PDFs is a few MB, not a running
  service holding memory/disk permanently.
- **To reset everything:** just delete the `qdrant_data/` folder. There's
  no container, image, or volume to clean up.
- **Disk-conscious tip:** if you're testing and don't need data to survive
  a restart, you can switch `qdrant_path` in `.env` to `:memory:` — then
  Qdrant keeps vectors in RAM only and writes nothing to disk at all
  (data disappears when uvicorn stops).

## Using OpenRouter effectively

- **Model string format:** OpenRouter model IDs are `provider/model-name`,
  e.g. `openai/gpt-4o-mini`, `anthropic/claude-3.5-haiku`,
  `meta-llama/llama-3.1-8b-instruct`. Swap `LLM_MODEL` in `.env` freely.
- **Free models:** OpenRouter lists models with a `:free` suffix (e.g.
  `meta-llama/llama-3.1-8b-instruct:free`) that cost nothing but have
  lower rate limits — good for a lab/learning setup. Check current free
  options at https://openrouter.ai/models?max_price=0
- **Embeddings are billed separately from chat**, but `text-embedding-3-small`
  is very cheap (~$0.02 per million tokens) — a whole PDF costs a fraction
  of a cent to embed.
- **One key, two jobs:** the same `LLM_API_KEY` authenticates both the
  `/embeddings` and `/chat/completions` calls, since both go through
  OpenRouter's OpenAI-compatible endpoints — no separate embedding
  provider account needed.

## Tuning knobs (backend/app/config.py or .env)

| Setting | What it controls |
|---|---|
| `chunk_size` / `chunk_overlap` | How text is split before embedding |
| `embedding_model` | Which OpenRouter embedding model to use |
| `embedding_dim` | Must match the embedding model's output size |
| `top_k` | How many chunks are retrieved per query |
| `llm_model` | Which model answers the question |
| `qdrant_path` | Where the on-disk vector store lives (or `:memory:`) |

## Project layout

```
rag-lab/
├── .env.example
├── README.md
├── backend/
│   ├── requirements.txt
│   └── app/
│       ├── main.py            # FastAPI app + CORS
│       ├── config.py          # all tunable settings
│       ├── routers/
│       │   ├── ingest.py      # POST /ingest
│       │   └── query.py       # POST /query
│       └── services/
│           ├── extraction.py    # PDF/TXT/MD -> raw text (PyMuPDF)
│           ├── chunking.py      # raw text -> overlapping chunks
│           ├── embeddings.py    # text -> vectors (via OpenRouter)
│           ├── qdrant_store.py  # embedded collection setup, upsert, search
│           └── llm.py           # chunks + question -> answer (via OpenRouter)
└── frontend/
    └── index.html              # single-file UI, no build step
```

## Notes / things worth exploring next

- No chunk deduplication yet: re-ingesting the same file twice creates
  duplicate points. Worth adding a `source`-based delete-before-insert if
  you re-upload often.
- No streaming: the LLM call is a single blocking request — a good next
  step once the pipeline works end-to-end.
- If you outgrow embedded mode (large corpora, concurrent writers), that's
  the point to move to a real Qdrant server — swap `QdrantClient(path=...)`
  for `QdrantClient(host=..., port=...)` in `qdrant_store.py`. Nothing
  else in the pipeline changes.
