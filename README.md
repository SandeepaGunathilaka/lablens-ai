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
venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
cp .env.example .env          # fill in API keys, DB connection string
Generate a JWT secret: python -c "import secrets; print(secrets.token_hex(32))"
Paste the output into .env as JWT_SECRET_KEY=<secret>. The server refuses to start without it.
uvicorn main:app --reload
```

The Document Agent also requires the Tesseract system executable. On Windows, install
it with:

```powershell
winget install --id UB-Mannheim.TesseractOCR --exact
```

Restart the terminal after installation so `tesseract` is available on `PATH`.

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

### Document Agent API

`POST /agents/document/extract` accepts one multipart field named `file`. Supported
uploads are CBC or Lipid Profile reports in PDF, PNG, JPG, or JPEG format, up to
10 MB. It returns raw extracted fields only; it does not label results as normal or
abnormal and does not diagnose.

```json
{
  "extraction_method": "pdf_text",
  "report_type": "cbc",
  "results": [
    {
      "test": "Hemoglobin",
      "value": 11.2,
      "unit": "g/dL",
      "reference_range": "12-16",
      "confidence": 0.99,
      "needs_verification": false
    }
  ]
}
```

OCR-derived fields are deliberately marked `needs_verification: true`, even when
Tesseract reports a high character score, because OCR can misread medical digits.

## Project structure

```
backend/
  agents/
    document_agent.py
    safety_agent.py
  security/
  tests/
    fixtures/document_agent/
    test_document_agent.py
    test_safety_agent.py
  main.py
  requirements.txt
frontend/
  src/
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
