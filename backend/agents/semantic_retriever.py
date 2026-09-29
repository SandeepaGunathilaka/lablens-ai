"""Ranked semantic candidates only; no acceptance policy or query persistence."""

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
