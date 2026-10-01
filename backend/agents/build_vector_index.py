"""Explicitly build the curated KB index: python -m agents.build_vector_index."""

import argparse
import sys
from typing import Callable

from agents.embedding_service import EmbeddingService
from agents.knowledge_base import KnowledgeBaseLoader
from agents.knowledge_base_fingerprint import knowledge_base_sha256
from agents.vector_store import ChromaVectorStore, VectorIndexNotBuiltError


def main(argv: list[str] | None = None, *, store: ChromaVectorStore | None = None,
         embedding_factory: Callable = EmbeddingService) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild", action="store_true", help="Explicitly replace only the configured collection")
    args = parser.parse_args(argv)
    try:
        store = store if store is not None else ChromaVectorStore()
        existing = False
        if not args.rebuild:
            try:
                store.validate_index()
                existing = True
            except VectorIndexNotBuiltError:
                pass
        if not existing:
            # No persistent writes until ALL document encoding/preparation succeeds.
            before_hash = knowledge_base_sha256(store.kb_directory)
            documents = KnowledgeBaseLoader(store.kb_directory).load_all()
            service = embedding_factory(model_name=store.model_name, revision=store.model_revision)
            vectors = service.encode_documents(documents)
            if knowledge_base_sha256(store.kb_directory) != before_hash:
                raise ValueError("Knowledge base changed during encoding; retry the build")
            store.rebuild(documents, vectors)
        count = store.validate_index()
        print("Index already exists; no overwrite." if existing else "Index built successfully.")
        print(f"Collection: {store.collection_name}\nPersistence: {store.directory}\n"
              f"Documents: {count}\nEmbedding model: {store.model_name}\nDimension: {store.dimension}\n"
              f"KB SHA-256: {knowledge_base_sha256(store.kb_directory)}\nStatus: complete")
        return 0
    except Exception as exc:
        print(f"Vector index build failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
