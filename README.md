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

### Embedding model setup

The embedding component uses `sentence-transformers/all-MiniLM-L6-v2` on CPU by
default. Configure it with the `EMBEDDING_*` settings in `backend/.env.example`.
No model is loaded at import or construction time. The first real encoding
downloads the public model unless cached; `EMBEDDING_LOCAL_FILES_ONLY=true`
requires local cached files. An optional model revision can be set for repeatable
experiments. Model files use the library's external cache, not the repository.

Unit tests inject fake models and do not download weights. The real-model test
is skipped unless explicitly enabled. From `backend/`, run it with:

```powershell
$env:RUN_EMBEDDING_INTEGRATION = '1'
.\venv\Scripts\python.exe -m pytest tests/test_embedding_integration.py -m embedding_integration -s
Remove-Item Env:RUN_EMBEDDING_INTEGRATION
```

This optional test may download the model unless local-files-only mode is set.
It checks vector dimensions, normalization, ordering, and input token lengths.

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

## Curated knowledge vector index (Member 2)

ChromaDB 1.5.9 stores only the seven static curated knowledge documents, their
indexing text, source attribution, explicitly supplied embeddings, and build
provenance. Never put patient data, uploaded reports, identifiers, explanations,
or query history into this collection. Query vectors are transient.

`CHROMA_PERSIST_DIRECTORY=data/chroma` resolves against `backend/`, independently
of the terminal directory. Absolute paths are also supported.
`CHROMA_COLLECTION_NAME=lablens_medical_kb_v1` selects the collection. The generated
`backend/data/chroma/` directory is ignored by Git; the curated JSON remains tracked.

From `backend/`, after installing requirements:

```powershell
.\venv\Scripts\python.exe -m agents.build_vector_index
.\venv\Scripts\python.exe -m agents.build_vector_index --rebuild
```

The first command builds a missing index, leaves a valid index unchanged without
loading the model, and rejects a stale index with instructions to rebuild. Building
uses the configured EmbeddingService and may download its model unless cached or
local-files-only mode is enabled. The default model produces 384-dimensional
normalized vectors. Chroma has no automatic embedding function and uses the 1.5.9
`configuration={"hnsw": {"space": "cosine"}}` API.

Preparation completes before replacing the configured collection; unrelated
collections remain intact. Replacement is not transactional. A failed insertion
leaves an incomplete index that requires an explicit rebuild. Freshness checks
compare the raw-file KB fingerprint, model/revision, dimension, normalization,
embedding-text version, actual cosine configuration, seven IDs, count, and stored
text/source metadata. Queries never build automatically. An empty compatible
collection returns no candidates; strict build validation still rejects it.

Normal tests use fake vectors in temporary directories and do not load MiniLM.
The optional integration test loads the real model, uses only pytest temporary
persistence, and verifies document-vector self-retrieval, not semantic quality:

```powershell
$env:RUN_VECTOR_STORE_INTEGRATION="1"
.\venv\Scripts\python.exe -m pytest tests/test_vector_store_integration.py -v
Remove-Item Env:RUN_VECTOR_STORE_INTEGRATION
```

This component returns nearest vector candidates only. It does not implement a
SemanticRetriever, relevance thresholds, or patient explanations.

## Ranked semantic search (Member 2)

`agents.semantic_retriever.SemanticRetriever` composes the existing embedding
service, vector store, and knowledge-base loader. `search(query, n_results=3)`
returns ranked `SemanticSearchResult` objects with authoritative documents and
source attribution. Query text and vectors remain transient; search performs no
index builds, rebuilds, writes, or query logging.

The encoder's model/revision must match the store configuration. Query vectors
must have the configured dimension and unit norm. The vector store checks index
freshness, including normalization and embedding-text version; the semantic layer
also checks each recovered document fingerprint. Errors propagate without an
automatic rebuild. Returned documents are isolated copies of curated KB records.

The search() method remains ranked search only. The retrieve() method applies the frozen policy documented below; hybrid retrieval is not implemented.
`similarity = 1 - distance` is cosine similarity, not probability or confidence,
and is not clamped. Formal semantic evaluation comes later in Step 12.3.

Normal tests use injected fakes without loading MiniLM. The optional smoke test
uses real MiniLM with temporary Chroma storage, never the development index:

```powershell
# From backend/
$env:RUN_SEMANTIC_RETRIEVER_INTEGRATION="1"
.\venv\Scripts\python.exe -m pytest tests/test_semantic_retriever_integration.py -v
Remove-Item Env:RUN_SEMANTIC_RETRIEVER_INTEGRATION
```

The smoke test checks execution, ranked result structure, and source attribution;
it does not establish semantic retrieval quality.

## Semantic retrieval baseline evaluation (Member 2)

From `backend/`, run:

```powershell
.\venv\Scripts\python.exe -m agents.evaluate_semantic_retrieval
```

The fixed 144-query semantic fixture contains 84 supported cases and 60 negatives,
with 96 calibration and 48 held-out cases. Labels and wording are fixed before
predictions. This evaluates ranked search with all seven candidates, not an
acceptance policy: this runner applies no threshold and negative cases still receive nearest
neighbors. Full MRR, Top-1, Hit/Recall@3, per-category/test/split metrics, score
distributions, and exact-keyword comparison on the same supported cases are
written to `backend/evaluation_results/semantic_baseline.json`.

The real evaluation uses the cached model and existing development index. It does
not download a model or rebuild an index. Missing/stale index errors are recorded
and produce a failing exit code; they are never successful negative rejections.
Normal evaluation tests use synthetic rankings without MiniLM or network access.
Threshold calibration uses calibration cases only; held-out
results must not be used to tune thresholds or revise the benchmark.

## Semantic retrieval acceptance-policy calibration (Member 2)

From `backend/`, run:

```powershell
.\venv\Scripts\python.exe -m agents.calibrate_semantic_retrieval
```

Calibrates a conjunctive acceptance policy (`top1_similarity >= similarity_threshold AND similarity_margin >= margin_threshold`) using the 96 calibration cases only. The selection criteria enforce `accepted_precision >= 0.95` and `negative_false_accept_rate <= 0.05` while maximizing supported correct coverage. Held-out cases (48 queries) remain completely firewalled from threshold selection. Results and candidate search metadata are saved to `backend/evaluation_results/semantic_calibration.json`.


## Frozen semantic acceptance (Member 2 Step 12.5)

`SemanticRetriever.search()` remains pure ranked search.
`SemanticRetriever.retrieve(query)` calls `search(query, n_results=2)` and returns
an immutable `SemanticRetrievalDecision`. The frozen `SemanticAcceptancePolicy`
accepts only when both inclusive comparisons hold:

- Top-1 similarity >= **0.0**.
- Top-1 similarity minus Top-2 similarity >= **0.22541916370391846**.

These values were selected only from Step 12.4A calibration and retained unchanged
in Step 12.4B held-out evaluation. No calibration code or report loading runs in
runtime retrieval. Zero or one candidate causes abstention without an invented
margin. Accepted decisions retain the authoritative Top-1 document object;
abstentions have `document=None` and retain available candidates for inspection.
Search errors propagate instead of being converted into abstentions.

Held-out accepted precision was **100% (3/3 accepted)**, supported correct coverage
was **10.71% (3/28 supported)**, and negative false-accept rate was **0% (0/20)**.
The policy is intentionally conservative; these figures do not mean 100% semantic
retrieval accuracy. Similarity is not a probability or confidence score.

Retrieval does not write query text, vectors, decisions, patient identifiers, or
history. It never rebuilds Chroma. Keyword-plus-semantic hybrid retrieval is not
implemented.
