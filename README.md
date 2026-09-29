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
  explanation_agent/    # Explanation Agent (Member 3)
  security/
  tests/
    fixtures/document_agent/
    test_document_agent.py
    test_safety_agent.py
    test_explanation_service.py
    test_status.py
  main.py
  requirements.txt
frontend/
  src/
```

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
No runtime calibration is added. Request/response orchestration is described below.

## End-to-end hybrid evaluation (Member 2 Steps 14 & 14.2)

From `backend/`, run `python -m agents.evaluate_hybrid_retrieval`.

The evaluation protocol clearly distinguishes four complementary benchmarks and diagnostics:

1. **Keyword contract benchmark:** Evaluates deterministic exact canonical/alias matching on `keyword_queries.json` (68 cases). Its 14 negative cases test keyword-matcher rejection (including truncated prefixes and blank strings), not universal medical unsupportedness.
2. **Semantic held-out benchmark:** Evaluates independent semantic ranking and frozen-policy acceptance on `semantic_queries.json` held-out cases (48 queries: 28 supported, 20 negative). Zero calibration queries are included.
3. **Primary hybrid benchmark (102 operational cases):** The primary independent evaluation combines:
   - **54 keyword-supported cases** (approved canonical names and aliases)
   - **48 semantic held-out cases** (28 supported paraphrases/descriptions + 20 independent negatives: 12 unsupported medical, 8 unrelated)
   - **Total:** 82 supported, 20 negative = 102 queries.
   - The 14 keyword-contract negatives are excluded from primary safety and precision metrics because their labels define exact-match behavior rather than universal semantic irrelevance.
4. **Keyword-contract diagnostic (68 cases):** Evaluates all 68 keyword cases under `HybridRetriever` to monitor how the hybrid pipeline handles keyword-contract rejections (exact keyword hits, semantic fallbacks, semantic accepts of truncated stems, and validation errors).

The report is `backend/evaluation_results/hybrid_evaluation.json`, including accuracy, accepted precision, supported coverage, negative false accepts, method utilization, routing, subset/category/test breakdowns, query-level decisions, and the diagnostic analysis.

The CLI verifies frozen fixture/KB hashes before execution and checks them again afterward. It reads the existing development index into a temporary byte copy, then queries that copy in a child process using cached MiniLM. This isolates any Chroma internal housekeeping from the development files; it does not build or rebuild an index. The temporary copy is removed after the child exits, and source index hashes must remain unchanged. Normal tests use mocks without MiniLM.

**Primary 102-case results (keyword SHA-256 `ee70d2ff`, semantic SHA-256 `e481c484`, KB SHA-256 `f704ff3d`):**

| Metric | Value |
|---|---|
| Overall decision accuracy | 75.49% (77/102) |
| Accepted precision | 100.00% (57/57 accepted) |
| Supported correct coverage | 69.51% (57/82 supported) |
| Supported abstention rate | 30.49% (25/82) |
| Negative false-accept count | 0 |
| Negative false-accept rate | 0.00% (0/20) |
| Negative abstention rate | 100.00% (20/20) |
| Correct keyword accepts | 54 |
| Correct semantic accepts | 3 |
| Wrong accepts | 0 |
| Correct abstentions | 20 |
| Missed supported | 25 |
| Errors | 0 |

**Primary routing (102 queries):** 54 keyword hits, 48 keyword misses => 48 semantic fallbacks => 3 semantic accepts, 45 semantic abstentions. No keyword hit triggered semantic retrieval.

**Semantic held-out subset (48 queries):** 3 correct semantic accepts, 25 missed supported, 0 wrong accepts, 20 correct abstentions, 0 errors. Precision 100% (3/3), supported coverage 10.71% (3/28), negative false-accept rate 0.00% (0/20), and negative abstention rate 100% (20/20).

**Keyword-contract diagnostic (68 cases):**
- 54 supported cases: 54 exact keyword hits (100% coverage).
- 14 contract-negative cases:
  - 14 correctly rejected by keyword matcher.
  - 2 pre-routing input validation errors: `unsupported_9` (empty) and `unsupported_10` (whitespace), raising `ValueError`.
  - 12 semantic fallbacks attempted:
    - 2 semantic accepts: `unsupported_4` ('Hemoglob' -> hemoglobin, similarity 0.5431, margin 0.3907) and `unsupported_12` ('Triglycer' -> triglycerides, similarity 0.4863, margin 0.2877). These truncated stems clear semantic threshold margins but were rejected by exact keyword rules.
    - 10 semantic abstentions.

## Medical Retrieval Agent (Member 2 Step 15)

`MedicalRetrievalAgent(hybrid_retriever=None).retrieve(request: RetrievalRequest)`
returns the existing `RetrievalResponse` contract:

```text
MedicalRetrievalAgent -> HybridRetriever -> keyword first
                                        -> semantic fallback
                                        -> abstention
```

The Retrieval Agent returns curated evidence and provenance. It does not diagnose
or produce final patient-facing explanations. It calls the hybrid retriever once
per requested test, preserving request order, duplicate entries, and the exact
`task_id`, `report_id`, and `user_id`. A single query uses a one-element
`test_names` list. No request or response schema changes were needed.

Each success has one match. Its `information` contains labeled, verbatim curated
canonical test name, document title, definition, what it measures, general
information, source publisher, and source accessed date. The existing structured
source carries the original source title and URL. Publisher and accessed date
are passage text because the source contract has no dedicated fields for them.
The result's `test_name` preserves the requested name/query, including aliases.
Aliases are not copied from the knowledge document. The public contract has no
retrieval-method field, so method and internal semantic scores are not exported;
similarity is never presented as confidence.

Abstention is explicitly `found=False, matches=[]`, meaning no reliable curated
information was retrieved. Rejected semantic candidates never become evidence.
Validation, model, missing/stale-index and KB/integrity failures propagate to the
caller instead of becoming not-found. An infrastructure failure aborts the call;
ordinary abstentions leave the other successful items intact.

The agent performs no persistence, keeps no request history, and adds no LLM,
patient-value interpretation, REST routes, or Coordinator integration. Hybrid
retrieval, the frozen thresholds, the KB, and Chroma remain unchanged. Offline
agent tests inject a mock hybrid retriever and need no MiniLM or Chroma:

```powershell
cd backend
.\venv\Scripts\python.exe -m pytest tests/test_retrieval_agent.py
```

## Medical Retrieval API (Member 2 Step 16)

`POST /api/retrieval` uses the existing `RetrievalRequest` body and returns
`RetrievalResponse` directly, without an envelope:

```text
Medical Retrieval API -> MedicalRetrievalAgent -> HybridRetriever
```

The FastAPI `Depends(get_retrieval_agent)` provider lazily constructs and caches
one agent for reuse per process. Importing the router does not load MiniLM or
Chroma; semantic dependencies remain deferred until a keyword miss. The route
calls `agent.retrieve(request)` once and adds no retrieval logic or persistence.

Minimal request:

```json
{"task_id":"task-1","report_id":"report-1","user_id":"user-1","test_names":["unsupported test"]}
```

If retrieval completes with insufficient evidence, the HTTP 200 response is:

```json
{"task_id":"task-1","report_id":"report-1","user_id":"user-1","results":[{"test_name":"unsupported test","found":false,"matches":[]}]}
```

Successful results contain the agent's curated passages and source title/URL.
Identifiers, order, and duplicates are preserved. Mixed success and abstention
also return HTTP 200. Invalid request bodies use FastAPI's standard HTTP 422.
Infrastructure failures propagate to the existing non-debug server error handler,
which returns HTTP 500 with the generic text `Internal Server Error`, without
exception details or filesystem paths. No new exception handler or sensitive
retrieval logging is added. A missing index is a system failure, not abstention.

The generated `/docs` and `/openapi.json` expose the existing request/response
schemas. This endpoint returns evidence only, with no diagnosis, patient
interpretation, LLM calls, or Coordinator integration.
