"""Explicit vector persistence for the static curated KB; no retrieval policy."""

from dataclasses import dataclass
import json
import math
from numbers import Real
import os
from pathlib import Path
from typing import Callable

from dotenv import load_dotenv

from agents.embedding_service import DEFAULT_MODEL_NAME, EMBEDDING_TEXT_VERSION, build_embedding_text
from agents.knowledge_base import KNOWLEDGE_BASE_DIR, KnowledgeBaseLoader, KnowledgeDocument
from agents.knowledge_base_fingerprint import (
    BACKEND_DIR, DOCUMENT_IDS, document_sha256, knowledge_base_sha256, resolve_backend_path,
)


DEFAULT_COLLECTION_NAME = "lablens_medical_kb_v1"
DEFAULT_DIMENSION = 384


class VectorIndexNotBuiltError(RuntimeError):
    """An explicit build is required before this index can be read."""


class VectorIndexStaleError(RuntimeError):
    """The persisted index is incomplete or incompatible; rebuild explicitly."""


class VectorIndexBuildError(RuntimeError):
    """Preparing or writing the replacement index failed."""


@dataclass(frozen=True)
class VectorSearchResult:
    document_id: str
    distance: float
    metadata: dict
    document_text: str


def validate_vector(vector, dimension: int) -> list[float]:
    if not isinstance(vector, (list, tuple)) or not vector:
        raise ValueError("Vector must be a nonempty numeric list")
    if len(vector) != dimension:
        raise ValueError(f"Vector dimension must be {dimension}")
    result = []
    for value in vector:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError("Vector values must be numeric real numbers")
        try:
            number = float(value)
        except (ValueError, OverflowError) as exc:
            raise ValueError("Vector values must be finite") from exc
        if not math.isfinite(number):
            raise ValueError("Vector values must be finite")
        result.append(number)
    norm = math.hypot(*result)
    if not math.isfinite(norm) or norm == 0:
        raise ValueError("Cosine vectors must have a finite nonzero norm")
    return result


def _create_client(*, path: str):
    import chromadb
    return chromadb.PersistentClient(path=path)


class ChromaVectorStore:
    """Lazy store. Every query checks current KB provenance, without writing.

    Only rebuild() may create persistence. No query text or patient identifiers
    are accepted or persisted. Callers supply embeddings explicitly.
    """

    def __init__(self, directory: Path | str | None = None, *,
                 collection_name: str | None = None, model_name: str | None = None,
                 model_revision: str | None = None, dimension: int = DEFAULT_DIMENSION,
                 kb_directory: Path | str = KNOWLEDGE_BASE_DIR,
                 client_factory: Callable | None = None):
        load_dotenv(BACKEND_DIR / ".env")
        self.directory = resolve_backend_path(directory if directory is not None else
                                              os.getenv("CHROMA_PERSIST_DIRECTORY", "data/chroma"))
        self.collection_name = (collection_name if collection_name is not None else
                                os.getenv("CHROMA_COLLECTION_NAME", DEFAULT_COLLECTION_NAME))
        self.model_name = model_name if model_name is not None else os.getenv("EMBEDDING_MODEL_NAME", DEFAULT_MODEL_NAME)
        self.model_revision = (model_revision if model_revision is not None else
                               os.getenv("EMBEDDING_MODEL_REVISION", "")).strip() or None
        if not self.collection_name.strip() or not self.model_name.strip():
            raise ValueError("Collection and model names must not be blank")
        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
            raise ValueError("Embedding dimension must be a positive integer")
        self.dimension = dimension
        self.kb_directory = resolve_backend_path(kb_directory)
        self._client_factory = client_factory or _create_client
        self._client = None

    def _get_client(self, *, create: bool = False):
        if not create and not (self.directory / "chroma.sqlite3").is_file():
            raise VectorIndexNotBuiltError("Vector index is missing; run python -m agents.build_vector_index")
        if self._client is None:
            self._client = self._client_factory(path=str(self.directory))
        return self._client

    def _open_collection(self):
        from chromadb.errors import NotFoundError
        client = self._get_client()
        try:
            return client.get_collection(name=self.collection_name, embedding_function=None)
        except NotFoundError as exc:
            raise VectorIndexNotBuiltError(f"Collection {self.collection_name!r} is missing; build explicitly") from exc

    def _provenance(self, kb_hash: str) -> dict:
        metadata = {
            "kb_sha256": kb_hash, "embedding_model": self.model_name,
            "embedding_dimension": self.dimension, "normalized_embeddings": True,
            "embedding_text_version": EMBEDDING_TEXT_VERSION,
        }
        if self.model_revision:
            metadata["model_revision"] = self.model_revision
        return metadata

    def _record_metadata(self, document: KnowledgeDocument, kb_hash: str) -> dict:
        return {
            **self._provenance(kb_hash), "test_name": document.test_name,
            "report_type": document.report_type, "title": document.title,
            "aliases_json": json.dumps(document.aliases, ensure_ascii=False, separators=(",", ":")),
            "source_publisher": document.source.publisher, "source_title": document.source.title,
            "source_url": document.source.url, "source_accessed_date": document.source.accessed_date,
            "document_sha256": document_sha256(document),
        }

    def _verify(self, collection, *, status: str = "complete", allow_empty: bool = False) -> int:
        try:
            kb_hash = knowledge_base_sha256(self.kb_directory)
            expected = {**self._provenance(kb_hash), "build_status": status,
                        "distance_metric": "cosine", "expected_record_count": len(DOCUMENT_IDS)}
            metadata = collection.metadata or {}
            for key, value in expected.items():
                if type(metadata.get(key)) is not type(value) or metadata.get(key) != value:
                    raise ValueError(f"collection {key} does not match expected {value!r}")
            if metadata.get("model_revision") != self.model_revision:
                raise ValueError("model_revision does not match")
            if collection.configuration.get("hnsw", {}).get("space") != "cosine":
                raise ValueError("actual collection distance configuration must be cosine")
            count = collection.count()
            # Explicit query contract: a complete, compatible but empty collection
            # has no candidates. validate_index() still rejects it as incomplete.
            if count == 0 and allow_empty:
                return 0
            if count != len(DOCUMENT_IDS):
                raise ValueError("record count must be seven")
            records = collection.get(include=["documents", "metadatas"])
            ids = records["ids"]
            if len(ids) != len(DOCUMENT_IDS) or set(ids) != set(DOCUMENT_IDS):
                raise ValueError("record IDs must match the seven approved documents")
            documents = {doc.id: doc for doc in KnowledgeBaseLoader(self.kb_directory).load_all()}
            if set(documents) != set(DOCUMENT_IDS):
                raise ValueError("curated document IDs differ from the approved set")
            if len(records["documents"]) != count or len(records["metadatas"]) != count:
                raise ValueError("stored record fields are incomplete")
            for identifier, text, record_metadata in zip(ids, records["documents"], records["metadatas"]):
                document = documents[identifier]
                if text != build_embedding_text(document) or record_metadata != self._record_metadata(document, kb_hash):
                    raise ValueError(f"record {identifier!r} text or metadata is stale")
            return count
        except Exception as exc:
            raise VectorIndexStaleError(f"Index is stale or incomplete: {exc}. Use --rebuild explicitly.") from exc

    def validate_index(self) -> int:
        """Strict completeness and compatibility check without model loading."""
        return self._verify(self._open_collection())

    def rebuild(self, documents: list[KnowledgeDocument], embeddings: list[list[float]]) -> int:
        """Prepare completely, then replace only this collection (not transactional)."""
        try:
            ids = [doc.id for doc in documents]
            if len(ids) != len(set(ids)):
                raise ValueError("Duplicate document IDs")
            if len(ids) != len(DOCUMENT_IDS) or set(ids) != set(DOCUMENT_IDS):
                raise ValueError("Build requires exactly the seven approved document IDs")
            if len(embeddings) != len(documents):
                raise ValueError("Embedding count must match document count")
            texts = [build_embedding_text(doc) for doc in documents]
            vectors = [validate_vector(vector, self.dimension) for vector in embeddings]
            if any(not math.isclose(math.hypot(*vector), 1.0, rel_tol=1e-5, abs_tol=1e-5) for vector in vectors):
                raise ValueError("Document embeddings must already be normalized")
            kb_hash = knowledge_base_sha256(self.kb_directory)
            approved = {doc.id: document_sha256(doc) for doc in KnowledgeBaseLoader(self.kb_directory).load_all()}
            if {doc.id: document_sha256(doc) for doc in documents} != approved:
                raise ValueError("Documents must match the current curated knowledge base")
            metadatas = [self._record_metadata(doc, kb_hash) for doc in documents]
            metadata = {**self._provenance(kb_hash), "build_status": "incomplete",
                        "distance_metric": "cosine", "expected_record_count": len(DOCUMENT_IDS)}
            if knowledge_base_sha256(self.kb_directory) != kb_hash:
                raise ValueError("Knowledge base changed during preparation")
        except Exception as exc:
            raise VectorIndexBuildError(f"Index preparation failed; prior collection untouched: {exc}") from exc

        from chromadb.errors import NotFoundError
        try:
            client = self._get_client(create=True)
            try:
                client.get_collection(name=self.collection_name, embedding_function=None)
            except NotFoundError:
                pass
            else:
                client.delete_collection(name=self.collection_name)
            collection = client.create_collection(
                name=self.collection_name, embedding_function=None,
                configuration={"hnsw": {"space": "cosine"}}, metadata=metadata,
            )
            collection.add(ids=ids, embeddings=vectors, documents=texts, metadatas=metadatas)
            self._verify(collection, status="incomplete")
            collection.modify(metadata={**metadata, "build_status": "complete"})
        except Exception as exc:
            raise VectorIndexBuildError(f"Index write failed; replacement must not be used: {exc}") from exc
        return len(ids)

    def query(self, embedding: list[float], n_results: int) -> list[VectorSearchResult]:
        vector = validate_vector(embedding, self.dimension)
        if isinstance(n_results, bool) or not isinstance(n_results, int) or n_results <= 0:
            raise ValueError("n_results must be an integer greater than zero")
        collection = self._open_collection()
        count = self._verify(collection, allow_empty=True)
        if count == 0:
            return []
        result = collection.query(query_embeddings=[vector], n_results=min(n_results, count),
                                  include=["distances", "metadatas", "documents"])
        fields = [result.get(key) for key in ("ids", "distances", "metadatas", "documents")]
        if any(not isinstance(field, list) or len(field) != 1 or not isinstance(field[0], list) for field in fields):
            raise VectorIndexStaleError("Expected exactly one nested Chroma query result")
        ids, distances, metadatas, documents = [field[0] for field in fields]
        if len({len(field[0]) for field in fields}) != 1:
            raise VectorIndexStaleError("Chroma returned inconsistent result lengths")
        return [VectorSearchResult(identifier, float(distance), metadata, text)
                for identifier, distance, metadata, text in zip(ids, distances, metadatas, documents)]
