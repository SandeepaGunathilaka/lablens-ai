"""Opt-in natural-language execution smoke test, not semantic evaluation."""

import math
import os

import pytest

from agents.embedding_service import DEFAULT_MODEL_NAME, EmbeddingService
from agents.knowledge_base import KnowledgeBaseLoader
from agents.semantic_retriever import SemanticRetriever, SemanticSearchResult
from agents.vector_store import ChromaVectorStore


@pytest.mark.semantic_retriever_integration
@pytest.mark.skipif(os.getenv("RUN_SEMANTIC_RETRIEVER_INTEGRATION") != "1",
                    reason="Set RUN_SEMANTIC_RETRIEVER_INTEGRATION=1 to load real MiniLM")
def test_real_ranked_semantic_search(tmp_path):
    loader = KnowledgeBaseLoader()
    documents = loader.load_all()
    encoder = EmbeddingService(model_name=DEFAULT_MODEL_NAME, device="cpu")
    store = ChromaVectorStore(tmp_path / "chroma", collection_name="semantic_smoke",
                              model_name=encoder.model_name, model_revision=encoder.revision or "")
    store.rebuild(documents, encoder.encode_documents(documents))
    retriever = SemanticRetriever(encoder, store, loader)
    authoritative = {document.id: document for document in documents}
    for query in ("oxygen carrying protein in red blood cells", "cells that help the immune system",
                  "blood components involved in clotting"):
        results = retriever.search(query)
        assert len(results) == 3
        assert [result.rank for result in results] == [1, 2, 3]
        assert [result.distance for result in results] == sorted(result.distance for result in results)
        for result in results:
            assert isinstance(result, SemanticSearchResult)
            assert math.isfinite(result.distance)
            assert result.similarity == 1 - result.distance
            assert result.document == authoritative[result.document_id]
            assert all(result.document.source.model_dump().values())
