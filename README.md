---
title: Intelligent Document Query Engine
emoji: 📄
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
---

# Intelligent Document Query Engine

A single-container FastAPI + React application for grounded PDF question answering. In production, FastAPI serves both the backend API and the built frontend from the same origin, which makes this repository suitable for a Hugging Face Docker Space.

## What It Does

- Query a remote PDF URL with `POST /hackrx/run`
- Upload a PDF directly with `POST /hackrx/upload-run`
- Return grounded answers with source excerpts and claim verification details
- Reuse cached document indexes for repeat queries
- Expose backend status from `GET /health`
- Serve the built React frontend from `/`

## Hugging Face Docker Space Deployment

This repository is configured for a single-container Hugging Face Docker Space:

1. Create a new Space and choose **Docker**.
2. Push this repository to the Space.
3. Configure these Space secrets:
   - `GROQ_API_KEY`
   - `API_TOKEN`
4. Let the Space build the multi-stage Dockerfile.
5. The container listens on port `7860`, and the frontend UI is served by FastAPI at the root URL.

### Required Space Secrets

- `GROQ_API_KEY`: Groq API key used for answer generation and claim verification
- `API_TOKEN`: Bearer token required by `/hackrx/run` and `/hackrx/upload-run`

## Local Development

### Backend

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:GROQ_API_KEY="your_groq_api_key"
$env:API_TOKEN="your_api_token"
$env:PORT="7860"
py start.py
```

### Frontend

```powershell
cd frontend
copy .env.example .env
npm install
npm run dev
```

Set `VITE_API_BASE_URL=http://127.0.0.1:8000` or another backend URL in `frontend/.env` when you want the Vite dev server to talk to a separate backend. If `VITE_API_BASE_URL` is not set, the built frontend uses same-origin requests.

## Production Container Build

```powershell
docker build -t intelligent-document-query-engine .
docker run --rm -p 7860:7860 --env GROQ_API_KEY=your_groq_api_key --env API_TOKEN=your_api_token intelligent-document-query-engine
```

## API Endpoints

### `POST /hackrx/run`

```json
{
  "documents": "https://example.com/document.pdf",
  "questions": [
    "What is this document about?",
    "What are the key exclusions?"
  ]
}
```

### `POST /hackrx/upload-run`

Send multipart form data with:
- `file`: the PDF upload
- `questions_json`: a JSON array of question strings

### `GET /health`

Returns health information including cache size and model load status.

## Deployment Notes

- FastAPI serves API routes first, then falls back to the built SPA for frontend refreshes.
- Static frontend assets are served from `frontend/dist` when that directory exists.
- The Hugging Face deployment path relies on the root `Dockerfile` and this README front matter.
- Existing Railway-oriented files remain in the repository for historical or local use, but they are not required for Hugging Face deployment.