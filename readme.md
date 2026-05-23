# AI Research Paper Understanding Agent

An API that analyzes research papers and returns a structured report (summary, contributions, method, datasets, limitations, future work, equations, implementation ideas, and research gaps).

The backend supports:
- arXiv URL analysis
- direct PDF upload analysis

## Project Structure

- `backend/` FastAPI app and analysis pipeline
- `backend/api/main.py` API endpoints
- `backend/analysis/` section-wise analysis agents
- `backend/rag/` chunking, embeddings, vector store
- `backend/tools/` loaders and PDF utilities

## Features

- RAG pipeline over paper text using FAISS + sentence-transformers
- Agent-style section generation (9 report sections)
- PDF upload endpoint (`multipart/form-data`)
- Groq OpenAI-compatible LLM integration
- Basic retry/backoff handling for Groq rate limits

## Tech Stack

- Python, FastAPI, Uvicorn
- PyMuPDF
- sentence-transformers
- FAISS
- Groq Chat Completions API (OpenAI-compatible)

## Setup

From repo root (`d:\RAG_agent`):

```powershell
.\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
```

Create/update `.env` in repo root:

```env
GROQ_API_KEY=your_groq_key
GROQ_MODEL=openai/gpt-oss-120b
GROQ_BASE_URL=https://api.groq.com/openai/v1

# Optional tuning for rate limits
GROQ_MAX_TOKENS=400
GROQ_MAX_PROMPT_CHARS=12000
GROQ_RETRIES=2
GROQ_RETRY_FALLBACK_SECONDS=8
```

## Run API

```powershell
cd backend
..\.venv\Scripts\python.exe -m uvicorn api.main:app --reload --port 8000
```

Open Swagger UI:

- `http://127.0.0.1:8000/docs`

## API Endpoints

### 1) Analyze by arXiv URL

`POST /analyze-paper`

Request body:

```json
{
  "paper_url": "https://arxiv.org/abs/1706.03762"
}
```

### 2) Analyze by PDF upload

`POST /analyze-upload`

- Content type: `multipart/form-data`
- Field name: `file`
- Supported: `.pdf`

cURL example:

```powershell
curl -X POST "http://127.0.0.1:8000/analyze-upload" `
  -H "accept: application/json" `
  -H "Content-Type: multipart/form-data" `
  -F "file=@D:/path/to/paper.pdf"
```

## Response Format

Both endpoints return:

```json
{
  "summary": "...",
  "contributions": "...",
  "method": "...",
  "dataset": "...",
  "limitations": "...",
  "future_work": "...",
  "equations": "...",
  "implementation": "...",
  "research_gaps": "..."
}
```

## Troubleshooting

- `Groq rate limit exceeded`:
  - wait and retry
  - reduce `GROQ_MAX_TOKENS` (e.g., `250`)
  - use a smaller/faster model or higher Groq tier
- `Analysis features require a Groq API key`:
  - set `GROQ_API_KEY` in `.env`
- Upload errors for form-data:
  - ensure `python-multipart` is installed (already in `backend/requirements.txt`)

## Notes

- Keep keys only in `.env`; do not commit secrets.
- If keys were exposed, rotate them immediately.
