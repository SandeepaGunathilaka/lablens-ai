"""Opt-in real-model verification; normal test runs never load model weights."""

import math
import os

import pytest

from agents.embedding_service import DEFAULT_MODEL_NAME, EmbeddingService, build_embedding_text
from agents.knowledge_base import KnowledgeBaseLoader


@pytest.mark.embedding_integration
@pytest.mark.skipif(os.getenv("RUN_EMBEDDING_INTEGRATION") != "1", reason="Real model requires explicit opt-in")
def test_real_model_seven_documents():
    from sentence_transformers import SentenceTransformer

    loaded = []

    def factory(name, **options):
        model = SentenceTransformer(name, **options)
        loaded.append(model)
        return model

    service = EmbeddingService(model_factory=factory, model_name=DEFAULT_MODEL_NAME, device="cpu")
    documents = KnowledgeBaseLoader().load_all()
    vectors = service.encode_documents(documents)
    assert len(vectors) == len(documents) == 7
    model = loaded[0]
    for document, vector in zip(documents, vectors):
        assert len(vector) == 384
        assert all(math.isfinite(value) for value in vector)
        assert math.sqrt(sum(value * value for value in vector)) == pytest.approx(1.0, abs=1e-5)
        text = build_embedding_text(document)
        assert vector == pytest.approx(service.encode_text(text), abs=1e-5)
        length = len(model.tokenizer(text, truncation=False)["input_ids"])
        print(f"{document.test_name}: {length} tokens; limit={model.max_seq_length}; exceeds={length > model.max_seq_length}")
        assert length <= model.max_seq_length, f"{document.test_name} would be truncated"
