"""Opt-in persistence/self-retrieval smoke test, not semantic-quality evaluation."""

import math
import os

import pytest

from agents.embedding_service import DEFAULT_MODEL_NAME, EmbeddingService, build_embedding_text
from agents.knowledge_base import KnowledgeBaseLoader
from agents.vector_store import ChromaVectorStore


@pytest.mark.vector_store_integration
@pytest.mark.skipif(os.getenv("RUN_VECTOR_STORE_INTEGRATION") != "1",
                    reason="Set RUN_VECTOR_STORE_INTEGRATION=1 to load real MiniLM")
def test_real_document_embedding_persistence_and_self_retrieval(tmp_path):
    documents = KnowledgeBaseLoader().load_all()
    service = EmbeddingService(model_name=DEFAULT_MODEL_NAME, device="cpu")
    vectors = service.encode_documents(documents)
    assert len(vectors) == 7
    for vector in vectors:
        assert len(vector) == 384
        assert all(math.isfinite(value) for value in vector)
        assert math.hypot(*vector) == pytest.approx(1.0, abs=1e-5)
    kwargs = {"model_name": service.model_name, "model_revision": service.revision or "", "dimension": 384}
    store = ChromaVectorStore(tmp_path / "chroma", **kwargs)
    store.rebuild(documents, vectors)
    reopened = ChromaVectorStore(tmp_path / "chroma", **kwargs)
    assert reopened.validate_index() == 7
    for index in (0, 3, 6):
        result = reopened.query(vectors[index], 1)[0]
        assert result.document_id == documents[index].id
        assert result.distance == pytest.approx(0, abs=1e-5)
        assert result.document_text == build_embedding_text(documents[index])
        assert result.metadata["source_url"] == documents[index].source.url
        assert result.metadata["embedding_dimension"] == 384
        assert result.metadata["normalized_embeddings"] is True
