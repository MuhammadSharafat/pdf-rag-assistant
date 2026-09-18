# PDF RAG Assistant

A Retrieval-Augmented Generation (RAG) app that lets you upload PDFs and ask questions about their content, grounded strictly in what's in the document. Built with FastAPI + Inngest for the backend pipeline, Qdrant as the vector store, and Streamlit for the UI — fully deployed and running on free-tier infrastructure end to end.

**🔗 Live demo:** [pdf-rag-assistant-jvukgqede7zkzf4grwrycs.streamlit.app](https://pdf-rag-assistant-jvukgqede7zkzf4grwrycs.streamlit.app/)

## How it works

1. **Upload** — You upload a PDF through the Streamlit UI.
2. **Ingest** — An Inngest function loads the PDF, splits it into overlapping text chunks, embeds each chunk, and stores the vectors in Qdrant.
3. **Ask** — When you ask a question, it's embedded the same way, and Qdrant returns the `top_k` most similar chunks.
4. **Answer** — Those chunks are passed as context to an LLM (via Groq), which generates an answer grounded in the retrieved text — and says so when the context doesn't contain the answer, rather than guessing.

```
PDF ─▶ chunk ─▶ embed ─▶ Qdrant
                                    question ─▶ embed ─▶ Qdrant search ─▶ top_k chunks ─▶ LLM ─▶ answer
```

Ingestion and querying both run as durable, retryable background jobs via Inngest, so a slow embedding call or a flaky API doesn't take down the whole request.

## Architecture

| Layer | Service | Why |
|---|---|---|
| Frontend | Streamlit Community Cloud | Free hosting, simple file-upload UI |
| Backend | FastAPI + Inngest, hosted on Render | Inngest handles retries/orchestration; Render's free tier runs it |
| Vector store | Qdrant Cloud | Free 1GB cluster, purpose-built for similarity search |
| Embeddings | `fastembed` (ONNX runtime, `all-MiniLM-L6-v2`, 384-dim) | Runs locally with no API cost, and — critically — no PyTorch dependency, so it fits in 512MB of RAM |
| LLM | Groq API (`openai/gpt-oss-20b`) | Free tier, OpenAI-compatible SDK, fast inference |
| PDF parsing / chunking | LlamaIndex (`PDFReader` + `SentenceSplitter`) | Reliable chunking with configurable overlap |
| Orchestration | Inngest Cloud | Free tier; durable functions with automatic retries |

The frontend and backend run on **separate servers**, so PDF bytes are base64-encoded and sent directly inside the Inngest event payload rather than relying on a shared local filesystem.

## Project structure

```
.
├── main.py              # FastAPI app + Inngest functions (ingest, query)
├── data_loader.py         # PDF loading, chunking, and embedding
├── vector_db.py            # Qdrant wrapper (create/upsert/search)
├── streamlit_app.py        # UI — upload PDFs, ask questions
├── requirements.txt          # Shared by both Render (backend) and Streamlit Cloud (frontend)
└── .env                       # API keys for local dev (not committed)
```

## Running it locally

### Prerequisites

- Python 3.12+
- Docker (for a local Qdrant instance)
- A free [Groq API key](https://console.groq.com/keys)

### Setup

```bash
pip install -r requirements.txt

docker run -d --name qdrantRagDb -p 6333:6333 -v "$(pwd)/qdrant_storage:/qdrant/storage" qdrant/qdrant
```

Create a `.env` file:

```
GROQ_API_KEY=your_groq_key_here
```

Run two processes in separate terminals:

```bash
# Terminal 1 — backend + Inngest Dev Server sync
python -m uvicorn main:app --reload

# Terminal 2 — frontend
streamlit run streamlit_app.py
```

Locally, the app auto-detects it's in dev mode (no `INNGEST_SIGNING_KEY` set) and talks to the Inngest Dev Server at `localhost:8288` and Qdrant at `localhost:6333`.

## Deploying it

The same codebase runs locally or in production — it switches behavior based on which environment variables are present:

| Variable | Local dev | Production |
|---|---|---|
| `QDRANT_URL` / `QDRANT_API_KEY` | unset → defaults to `localhost:6333`, no auth | Qdrant Cloud cluster URL + API key |
| `INNGEST_SIGNING_KEY` / `INNGEST_EVENT_KEY` | unset → uses local Dev Server | Inngest Cloud keys → production mode |
| `GROQ_API_KEY` | required either way | required either way |

**To deploy:**
1. Create a free cluster on [Qdrant Cloud](https://cloud.qdrant.io) → get URL + API key.
2. Create a free app on [Inngest Cloud](https://app.inngest.com) → get Signing Key + Event Key.
3. Deploy `main.py` to [Render](https://render.com) (or any host) as a web service, with all five keys above set as environment variables. Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`.
4. In the Inngest Cloud dashboard, sync your backend using `https://<your-backend>/api/inngest`.
5. Deploy `streamlit_app.py` to [Streamlit Community Cloud](https://share.streamlit.io), with the same keys set in **Secrets** (TOML format).

## Configuration

- **`chunk_size` / `chunk_overlap`** (`data_loader.py`) — controls how the PDF is split before embedding. Smaller chunks give more precise retrieval but less context per chunk.
- **`top_k`** (adjustable in the Streamlit UI) — how many chunks are retrieved per question. Too low and multi-part questions come back incomplete; too high adds noise and latency. In testing, questions whose answer spans two sections of a document (e.g. comparing two topics covered in different chunks) failed outright at `top_k=1` and succeeded fully at `top_k=5` — a clean demonstration of why this matters.
- **`EMBED_MODEL`** (`data_loader.py`) — swapping this requires re-ingesting all documents, since embeddings from different models aren't compatible, and the Qdrant collection's vector dimension must match.
- **`GROQ_MODEL`** (`main.py`) — available models depend on your Groq account; check `https://api.groq.com/openai/v1/models` with your key to see what you have access to before assuming a model name from the docs actually works on your account.

## Problems hit along the way (and fixes)

Building and deploying this surfaced a handful of real issues worth documenting, since they're the kind that cost the most debugging time:

- **`inngest.Throttle(count=...)` → `FunctionConfigInvalidError`.** The correct field name is `limit`, not `count`.
- **Silent failures with no error detail.** The initial polling code (`streamlit_app.py`) only checked run *status*, not the run's actual output/error payload. Fixed by fetching the per-run detail endpoint and surfacing the real exception on failure instead of a bare "Function run Failed".
- **`qdrant-client`'s `.search()` was removed** in a newer library version in favor of `.query_points()` — a breaking API change that only showed up at runtime, not install time.
- **Groq model name from the docs (`llama-3.3-70b-versatile`) returned 404.** Not every model listed in Groq's docs is available on every account. Fixed by querying `/v1/models` directly with the actual API key to see what's genuinely accessible.
- **`sentence-transformers` (PyTorch) caused Render's free tier to OOM** at ~512MB RAM. Switched to `fastembed` (ONNX runtime, no PyTorch) — same model, same 384-dim output, a fraction of the memory.
- **PDF ingestion broke in production** because the original code passed a local file *path* between the frontend and backend — which only works when both run on the same machine. Fixed by base64-encoding the PDF bytes and sending them directly in the event payload.
- **Streamlit Cloud secrets not taking effect** until the app was manually rebooted after saving — saving Secrets alone doesn't restart a running app.

## Known limitations

- Render's free tier spins down after inactivity, so the first request after idling takes 30-60+ seconds (cold start).
- Local embeddings are lightweight by design; retrieval quality on very technical or ambiguous documents may benefit from a larger, hosted embedding model.
- This is a demo/portfolio-scale deployment (single free-tier instance, no auth on the app itself) — a production version would need rate limiting, user auth, and a paid tier to avoid cold starts.
