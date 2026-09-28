# LabLens AI

AI-powered HealthTech platform that turns an uploaded lab report (CBC / Lipid Profile) into a safe, evidence-grounded, plain-language explanation — without diagnosing the patient.

Built for **IT 3041 — Information Retrieval and Web Analytics**, SLIIT.

## System overview

LabLens AI is a multi-agent (Agentic AI) system with four specialized agents coordinated by a central Coordinator:

| Agent | Responsibility |
|---|---|
| Document Agent | OCR + NLP extraction of test results from PDF/PNG/JPG reports |
| Medical Retrieval Agent | Semantic/keyword retrieval from a curated medical knowledge base (RAG) |
| Explanation Agent | LLM-based generation of plain-language explanations |
| Safety Agent | Rule-based validation of every explanation before it reaches the user |

Full architecture, agent communication flow, and design rationale: see `/docs` and the project report.

## Tech stack

- **Frontend:** React + TypeScript + Tailwind CSS
- **Backend:** FastAPI (Python)
- **Database:** MongoDB
- **Vector DB:** ChromaDB + Sentence Transformers
- **OCR/NLP:** Tesseract, spaCy, PyMuPDF, Regex
- **LLM:** GPT (via API)
- **Auth:** JWT + bcrypt

## Setup

### Backend
```bash
cd backend
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
cp .env.example .env          # fill in API keys, DB connection string
uvicorn main:app --reload
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```

### Environment variables
See `.env.example` for required keys (LLM API key, MongoDB URI, JWT secret).

## Usage

1. Register / log in.
2. Upload a CBC or Lipid Profile report (PDF/PNG/JPG).
3. View extracted results, plain-language explanations, and their sources.
4. Ask follow-up questions about any test.

## Project structure

```
backend/
  main.py
  explanation_agent/    # Explanation Agent (Member 3)
  tests/
frontend/
  src/
```

Document, Retrieval, Safety, and the Coordinator are owned by the other members and are not in this tree yet.

## Explanation Agent

`POST /explanation` turns one lab result into four educational fields. It does not assign a personal condition, recommend medication, or change the recorded value.

The Coordinator should calculate status with `calculate_status` and pass it in. If `status` is omitted, the agent uses that same function. A missing or unreadable reference range stays `unknown`.

```python
from explanation_agent import calculate_status, build_draft_response, ExplanationRequest
```

`build_draft_response` joins `what_it_measures`, `explanation`, `possible_meaning`, and `recommended_discussion`, then appends the standard disclaimer once. Send that string to the Safety Agent. Do not append the disclaimer again.

Optional `retrieved_sources` carries Retrieval Agent citations (`title`, `excerpt`, optional `url` and `source_id`). The prompt may use only those sources. With no usable source, the agent says there is not enough reliable information.

Regeneration: send `rejection_feedback` and, when you have it, `previous_draft`. The response sets `regenerated` to true.

| Environment variable | Purpose |
|---|---|
| `EXPLANATION_PROVIDER` | `openai` (default) or `template` for a local source-only preview |
| `EXPLANATION_MODEL` | OpenAI model name, default `gpt-4o-mini` |
| `OPENAI_API_KEY` | Required for `openai`. If it is missing, the agent returns `generation_mode: unavailable` and does not invent an explanation |
| `EXPLANATION_RATE_LIMIT` | Requests per minute per client. `0` disables the limit |

```bash
cd backend
python -m pytest
```

## Contributors

| Name | Role |
|---|---|
| Member 1 | Document Agent, Repository & Project Management |
| Member 2 | Medical Retrieval Agent, Backend Skeleton |
| Member 3 | Explanation Agent, Frontend Skeleton |
| Member 4 | Safety Agent, Auth & Audit Logging, Shared Tooling |

## Responsible AI

LabLens AI is an **educational tool, not a diagnostic system**. It never diagnoses conditions or recommends medication. All explanations are grounded in a curated medical knowledge base and validated by a dedicated Safety Agent before being shown to the user. See the final report for the full Responsible AI plan.

## License

MIT



## Run the Project

cd backend
.\venv\Scripts\python.exe -m uvicorn main:app --reload --host 127.0.0.1 --port 8000

cd frontend
npm run dev