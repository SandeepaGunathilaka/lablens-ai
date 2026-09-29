"""Ranked semantic search and frozen-policy acceptance for the medical knowledge base.

search() — returns all ranked candidates; no filtering or thresholds applied.
retrieve() — applies the frozen calibrated acceptance policy and returns a decision.

The acceptance policy (SemanticAcceptancePolicy) was selected exclusively from
the 96-query calibration split (Step 12.4A).  It was independently evaluated on
the 48-query held-out split (Step 12.4B) without any threshold adjustment.
Thresholds are hard-coded constants; no calibration code runs at runtime.
"""

from dataclasses import dataclass
import math

from agents.embedding_service import EmbeddingService
from agents.knowledge_base import KnowledgeBaseLoader, KnowledgeDocument
from agents.knowledge_base_fingerprint import document_sha256
from agents.vector_store import ChromaVectorStore, VectorIndexStaleError, validate_vector


class SemanticRetrievalIntegrityError(RuntimeError):
    """A vector candidate cannot be reconciled with authoritative curated content."""


@dataclass(frozen=True)
class SemanticSearchResult:
    document_id: str
    document: KnowledgeDocument
    distance: float
    rank: int

    @property
    def test_name(self) -> str:
        return self.document.test_name

    @property
    def similarity(self) -> float:
        """Cosine similarity, not probability or confidence; deliberately unclamped."""
        return 1.0 - self.distance


@dataclass(frozen=True)
class SemanticAcceptancePolicy:
    """Immutable conjunctive acceptance policy frozen after Step 12.4A calibration.

    A result is accepted only when BOTH conditions hold simultaneously:
        top1.similarity >= similarity_threshold
        AND
        (top1.similarity - top2.similarity) >= margin_threshold

    These exact values were selected from the 96-query calibration split and
    independently evaluated on the 48-query held-out split (Step 12.4B).
    They must NOT be derived dynamically or re-calibrated at runtime.

    Calibration constraints satisfied:
        accepted_precision          >= 0.95   (held-out: 1.0000)
        negative_false_accept_rate  <= 0.05   (held-out: 0.0000)

    The policy is intentionally conservative: held-out supported correct coverage
    was 10.71% and abstention rate was 93.75%.  This reflects insufficient margin
    separation in the current KB, not a retrieval failure.
    """

    # Selected from calibration split (Step 12.4A) — DO NOT CHANGE
    similarity_threshold: float = 0.0
    margin_threshold: float = 0.22541916370391846

    def accepts(self, top1_similarity: float, similarity_margin: float) -> bool:
        """Apply exactly the two inclusive, calibration-selected comparisons."""
        return (top1_similarity >= self.similarity_threshold
                and similarity_margin >= self.margin_threshold)


@dataclass(frozen=True)
class SemanticRetrievalDecision:
    """Outcome of a single retrieve() call under the frozen acceptance policy.

    accepted=True  — the top-ranked candidate cleared both policy thresholds.
                     document is the authoritative KnowledgeDocument for that candidate.
    accepted=False — evidence was insufficient; the retrieval system abstained.
                     document is None.  top_candidate / runner_up are preserved
                     so callers can inspect why abstention occurred.

    Abstention is NOT an error.  It means retrieval evidence was insufficient,
    not that the retrieval system failed.  Infrastructure failures (stale index,
    missing index, embedding errors, integrity errors) raise exceptions instead.

    similarity is cosine similarity (1 - distance), not a probability or confidence.
    similarity_margin is top1.similarity - top2.similarity when two candidates exist.
    """

    accepted: bool
    document: KnowledgeDocument | None
    top_candidate: SemanticSearchResult | None
    runner_up: SemanticSearchResult | None
    similarity_margin: float | None


# Module-level singleton — constructed once; frozen so callers cannot mutate it.
FROZEN_POLICY = SemanticAcceptancePolicy()


class SemanticRetriever:
    """Join existing encoding, vector search, and authoritative KB recovery.

    The encoder always requests normalized embeddings. The store validates vector
    dimension and index provenance (including normalization and text version).
    We additionally compare encoder model/revision and verify unit query norm.
    No index lifecycle operations or writes are performed here.
    """

    def __init__(self, embedding_service: EmbeddingService | None = None,
                 vector_store: ChromaVectorStore | None = None,
                 loader: KnowledgeBaseLoader | None = None):
        self._vector_store = vector_store if vector_store is not None else ChromaVectorStore()
        self._embedding_service = embedding_service if embedding_service is not None else EmbeddingService(
            model_name=self._vector_store.model_name, revision=self._vector_store.model_revision or "",
        )
        self._loader = loader if loader is not None else KnowledgeBaseLoader(self._vector_store.kb_directory)

    def search(self, query: str, n_results: int = 3) -> list[SemanticSearchResult]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a nonblank string")
        if isinstance(n_results, bool) or not isinstance(n_results, int) or n_results <= 0:
            raise ValueError("n_results must be an integer greater than zero")
        service, store = self._embedding_service, self._vector_store
        if service.model_name != store.model_name or service.revision != store.model_revision:
            raise VectorIndexStaleError("Query encoder model/revision differs from vector-store configuration")
        vector = service.encode_text(query.strip())
        checked = validate_vector(vector, store.dimension)
        if not math.isclose(math.hypot(*checked), 1.0, rel_tol=1e-5, abs_tol=1e-5):
            raise ValueError("Query embedding must already be normalized")
        candidates = store.query(vector, n_results=n_results)
        if not candidates:
            return []
        documents = {document.id: document for document in self._loader.load_all()}
        results = []
        for rank, candidate in enumerate(candidates, start=1):
            document = documents.get(candidate.document_id)
            if document is None:
                raise SemanticRetrievalIntegrityError(f"Unknown curated document ID: {candidate.document_id!r}")
            if candidate.metadata.get("document_sha256") != document_sha256(document):
                raise SemanticRetrievalIntegrityError(
                    f"Stale content or missing fingerprint for document {candidate.document_id!r}"
                )
            results.append(SemanticSearchResult(
                document_id=document.id, document=document.model_copy(deep=True),
                distance=candidate.distance, rank=rank,
            ))
        return results

    def retrieve(self, query: str) -> SemanticRetrievalDecision:
        """Apply the frozen acceptance policy to a single query.

        Internally calls search(query, n_results=2).  Infrastructure errors
        (stale index, missing index, embedding errors, integrity errors) are
        propagated as-is; they are never converted to abstentions.

        Abstention occurs only when retrieval evidence is insufficient:
          - zero candidates returned
          - only one candidate (margin cannot be computed)
          - top-1 similarity below similarity_threshold
          - rank-1/rank-2 similarity margin below margin_threshold

        No query text, vectors, or results are persisted anywhere.
        This method is read-only with respect to the vector index.
        """
        results = self.search(query, n_results=2)

        # Conservative policy: margin cannot be computed without two candidates.
        if len(results) < 2:
            top = results[0] if results else None
            return SemanticRetrievalDecision(
                accepted=False,
                document=None,
                top_candidate=top,
                runner_up=None,
                similarity_margin=None,
            )

        top1, top2 = results[0], results[1]
        margin = top1.similarity - top2.similarity

        if FROZEN_POLICY.accepts(top1.similarity, margin):
            return SemanticRetrievalDecision(
                accepted=True,
                document=top1.document,
                top_candidate=top1,
                runner_up=top2,
                similarity_margin=margin,
            )

        return SemanticRetrievalDecision(
            accepted=False,
            document=None,
            top_candidate=top1,
            runner_up=top2,
            similarity_margin=margin,
        )
