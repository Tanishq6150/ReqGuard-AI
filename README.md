# ReqGuard AI

NLP-based software requirement analysis and intelligent test-case generation.

ReqGuard AI analyzes natural-language software requirements, flags ambiguous or missing details, suggests a clearer requirement, and generates structured test cases. Local NLP handles deterministic analysis. An LLM is used for improvement and test-case generation.

AI-generated test cases should be reviewed by a qualified software tester before being used in production.

## Features

- Requirement input and validation
- spaCy / rule-based component extraction
- Ambiguity and completeness scoring
- LLM-backed requirement improvement
- Structured test cases (functional, positive, negative, boundary, edge, security)
- JSON schema validation and duplicate filtering
- Streamlit UI with JSON/CSV export
- FastAPI backend and Docker deployment

## Project structure

```text
app/                 FastAPI application, NLP, LLM, and prompts
frontend/app.py      Streamlit UI
config/config.yaml   Runtime configuration
tests/               pytest suite
Dockerfile           Backend container
docker-compose.yml   API + UI services
```

## Prerequisites

- Python 3.11+
- An LLM API key (Google Gemini or OpenAI)

## Local setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
cp .env.example .env
```

Edit `.env` and set `LLM_API_KEY`. Choose a provider with `LLM_PROVIDER=gemini` or `LLM_PROVIDER=openai`, and set `LLM_MODEL` accordingly.

### Start the API

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Start the UI

```bash
streamlit run frontend/app.py
```

Open the Streamlit URL, enter a requirement, then run **Analyze Requirement**, optionally **Improve Requirement**, then **Generate Test Cases**.

## API

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/api/v1/health` | Health check |
| POST | `/api/v1/analyze` | NLP (+ optional LLM) analysis |
| POST | `/api/v1/improve` | Rewrite the requirement |
| POST | `/api/v1/generate` | Generate structured test cases |

Example:

```bash
curl -X POST http://localhost:8000/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d "{\"requirement\": \"The system should allow users to upload files quickly.\"}"
```

## Configuration

`config/config.yaml` controls model names, timeouts, retries, generation limits, and the NLP similarity threshold. Secrets stay in environment variables and are never committed.

## Tests

```bash
pytest
```

LLM calls are mocked in API tests. spaCy and sentence-transformer models are optional for unit tests; rule-based NLP and TF-IDF similarity are used as fallbacks.

## Docker

```bash
cp .env.example .env
docker compose up --build
```

- API: `http://localhost:8000/api/v1/health`
- UI: `http://localhost:8501`

## Notes

- The LLM may infer details; the UI labels those as assumptions.
- Generated tests are recommendations, not guaranteed coverage.
- Do not commit `.env` or API keys.
