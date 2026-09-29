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

The search() method remains ranked search only. The retrieve() method applies the frozen policy documented below; keyword-first composition is documented in Step 13 below.
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
history. It never rebuilds Chroma. Keyword-first hybrid retrieval composes this frozen policy as documented below.

## Keyword-first hybrid retrieval (Member 2 Step 13)

`HybridRetriever(keyword_retriever=None, semantic_retriever=None).retrieve(query)`
tries the existing exact canonical/approved-alias matcher first. An exact match
returns immediately with `method="keyword"`; semantic retrieval is not called or
reranked against it. This preserves deterministic approved vocabulary behavior.
The default semantic component is constructed only after a keyword miss.

A miss calls `SemanticRetriever.retrieve()` with its unchanged frozen policy.
Acceptance returns `method="semantic"`; insufficient evidence returns
`found=False`, `document=None`, and `method="none"`. Semantic decisions are retained
for inspection. Scores are never combined, and similarity is not confidence.

`HybridRetrievalResult` is a frozen dataclass containing `document`, `found`,
`method`, and `semantic_decision`. Non-string or blank queries raise `ValueError`.
Other query text is forwarded unchanged; existing components own normalization.
Infrastructure and integrity exceptions propagate instead of becoming no-match.
The component stores no query history and performs no Chroma writes or rebuilds.
No runtime calibration or RetrievalRequest/RetrievalResponse orchestration is added.

## End-to-end hybrid evaluation (Member 2 Step 14)

From `backend/`, run `python -m agents.evaluate_hybrid_retrieval`.
The primary benchmark contains all **68 frozen keyword cases** and only the
**48 semantic held-out cases**: **116 total, 82 supported and 34 negative**.
The 96 semantic calibration cases are excluded; they are not independent test
data. No new labels or fixture are created. The runner calls the public
`HybridRetriever.retrieve()` for every case; observers record routing without
reimplementing it. No calibration or threshold changes occur.

The report is `backend/evaluation_results/hybrid_evaluation.json`, including
accuracy, accepted precision, supported coverage, negative false accepts, method
utilization, routing, subset/category/test breakdowns, and query-level decisions.
Accepted precision is not overall retrieval accuracy. Keyword negative labels
retain their original exact-match meaning; they are not universal statements of
semantic irrelevance. Results are reported against those unchanged labels.

The two blank keyword cases are still executed. The hybrid API rejects them;
these are recorded as errors, retained in denominators, and never counted as
correct abstentions. A report containing any errors produces exit code 1.
Infrastructure/integrity failures receive the same explicit error treatment.

The CLI verifies frozen fixture/KB hashes before execution and checks them again
afterward. It reads the existing development index into a temporary byte copy,
then queries that copy in a child process using cached MiniLM. This isolates any
Chroma internal housekeeping from the development files; it does not build or
rebuild an index. The temporary copy is removed after the child exits, and source
index hashes must remain unchanged. Normal tests use mocks without MiniLM.

**Primary 116-case results (keyword SHA-256 `ee70d2ff`, semantic SHA-256 `e481c484`, KB SHA-256 `f704ff3d`):**

| Metric | Value |
|---|---|
| Overall decision accuracy | 75.00% (87/116) |
| Accepted precision | 96.61% (57/59 accepted) |
| Supported correct coverage | 69.51% (57/82 supported) |
| Supported abstention rate | 30.49% (25/82) |
| Negative false-accept rate | 5.88% (2/34) |
| Negative abstention rate | 88.24% (30/34) |
| Correct keyword accepts | 54 |
| Correct semantic accepts | 3 |
| Wrong keyword accepts | 0 |
| Wrong semantic accepts | 2 |
| Correct abstentions | 30 |
| Missed supported | 25 |
| Errors (blank queries) | 2 |

**Routing:** 54 keyword hits, 60 keyword misses => 60 semantic fallbacks => 5 semantic accepts, 55 semantic abstentions. No keyword hit triggered a semantic call.

**Keyword subset (68):** accuracy 94.12% (64/68), precision 96.43% (54/56 accepted), coverage 100% (54/54 supported). Negative subset (14 cases): 10 correct abstentions, 2 negative false accepts (`unsupported_4` 'Hemoglob' and `unsupported_12` 'Triglycer'), and 2 blank-query validation errors (`unsupported_9` '' and `unsupported_10` '   '). Negative false-accept rate is 14.29% (2/14) and negative abstention rate is 71.43% (10/14).

**Semantic held-out subset (48):** 3 correct semantic accepts, 25 missed supported, 0 wrong accepts, 20 correct abstentions, 0 errors. Precision 100% (3/3), supported coverage 10.71% (3/28), negative false-accept rate 0.00% (0/20), and negative abstention rate 100% (20/20).

**Audit reconciliation of false accepts and errors:**
The 2 negative false accepts (and 2 wrong semantic accepts) in primary totals are genuine semantic fallback acceptances of keyword fixture negative queries: `unsupported_4` ('Hemoglob') and `unsupported_12` ('Triglycer'). Under the strict keyword benchmark contract, truncated prefixes are labelled negative (`expected_document_id=None`). When the keyword matcher missed them, semantic fallback accepted them as hemoglobin and triglycerides respectively due to high similarity and sufficient margin.
The 2 errors are blank/whitespace queries (`unsupported_9` and `unsupported_10`) that failed input validation with `ValueError` before retrieval routing; they are recorded as `outcome="error"`, have `found=None`, remain in overall denominators, and are strictly separate from false accepts.
On the independent 48-case semantic held-out subset, zero false accepts occurred.
