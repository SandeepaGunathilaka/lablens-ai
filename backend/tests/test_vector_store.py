"""Offline vector-store tests, including real Chroma with synthetic vectors."""

import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from unittest.mock import Mock

from chromadb.errors import NotFoundError
import pytest

from agents import embedding_service, vector_store as module
from agents.embedding_service import build_embedding_text
from agents.knowledge_base import KNOWLEDGE_BASE_DIR, KnowledgeBaseLoader
from agents.knowledge_base_fingerprint import (
    BACKEND_DIR, DOCUMENT_IDS, document_sha256, knowledge_base_sha256,
)
from agents.vector_store import (
    ChromaVectorStore, VectorIndexBuildError, VectorIndexNotBuiltError,
    VectorIndexStaleError, VectorSearchResult,
)


class FakeCollection:
    def __init__(self, metadata, configuration):
        self.metadata = copy.deepcopy(metadata)
        self.configuration = copy.deepcopy(configuration)
        self.records = {"ids": [], "documents": [], "metadatas": [], "embeddings": []}
        self.query = Mock(side_effect=self._query)

    def add(self, **records):
        self.records = copy.deepcopy(records)

    def get(self, **kwargs):
        return copy.deepcopy(self.records)

    def count(self):
        return len(self.records["ids"])

    def modify(self, *, metadata):
        self.metadata = copy.deepcopy(metadata)

    def _query(self, *, n_results, **kwargs):
        return {"ids": [self.records["ids"][:n_results]], "distances": [[0.0] * n_results],
                "documents": [self.records["documents"][:n_results]],
                "metadatas": [self.records["metadatas"][:n_results]]}


class FakeClient:
    def __init__(self):
        self.collections = {}
        self.deleted = []
        self.created = []
        self.opened = []
        self.fail_add = False
        self.fail_verification = False

    def factory(self, *, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "chroma.sqlite3").touch()
        return self

    def get_collection(self, *, name, embedding_function):
        assert embedding_function is None
        self.opened.append(name)
        if name not in self.collections:
            raise NotFoundError(name)
        return self.collections[name]

    def create_collection(self, **kwargs):
        assert kwargs["embedding_function"] is None
        self.created.append(copy.deepcopy(kwargs))
        collection = FakeCollection(kwargs["metadata"], kwargs["configuration"])
        self.collections[kwargs["name"]] = collection
        if self.fail_add:
            collection.add = Mock(side_effect=RuntimeError("synthetic insertion failure"))
        if self.fail_verification:
            collection.count = lambda: 6
        return collection

    def delete_collection(self, *, name):
        self.deleted.append(name)
        del self.collections[name]


def fake_vectors(documents, dimension=7):
    return [[float(index == coordinate) for coordinate in range(dimension)] for index, _ in enumerate(documents)]


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch):
    monkeypatch.setattr(module, "load_dotenv", lambda *args: None)
    for key in ("CHROMA_PERSIST_DIRECTORY", "CHROMA_COLLECTION_NAME", "EMBEDDING_MODEL_NAME", "EMBEDDING_MODEL_REVISION"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(embedding_service, "_create_model", Mock(side_effect=AssertionError("No real models in unit tests")))


@pytest.fixture
def documents():
    return KnowledgeBaseLoader().load_all()


@pytest.fixture
def setup_store(tmp_path, documents):
    client = FakeClient()
    store = ChromaVectorStore(tmp_path / "index", dimension=7, client_factory=client.factory)
    return store, client, documents, fake_vectors(documents)


@pytest.fixture
def built(setup_store):
    store, client, documents, vectors = setup_store
    store.rebuild(documents, vectors)
    return store, client, documents, vectors


def test_default_configuration_is_lazy(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    factory = Mock(side_effect=AssertionError("must remain lazy"))
    store = ChromaVectorStore(client_factory=factory)
    assert store.directory == BACKEND_DIR / "data" / "chroma"
    assert store.collection_name == "lablens_medical_kb_v1"
    assert store.dimension == 384
    factory.assert_not_called()


def test_environment_paths(monkeypatch, tmp_path):
    monkeypatch.setenv("CHROMA_PERSIST_DIRECTORY", "custom/vectors")
    monkeypatch.setenv("CHROMA_COLLECTION_NAME", "custom_collection")
    monkeypatch.setenv("EMBEDDING_MODEL_REVISION", "abc123")
    monkeypatch.chdir(tmp_path)
    store = ChromaVectorStore()
    assert store.directory == BACKEND_DIR / "custom" / "vectors"
    assert store.collection_name == "custom_collection"
    assert store.model_revision == "abc123"
    assert ChromaVectorStore(tmp_path).directory == tmp_path


@pytest.mark.parametrize("dimension", [0, -1, True, 1.5])
def test_invalid_dimension(dimension):
    with pytest.raises(ValueError, match="dimension"):
        ChromaVectorStore(dimension=dimension)


def test_build_metadata_and_explicit_api(built):
    store, client, documents, vectors = built
    collection = client.collections[store.collection_name]
    creation = client.created[0]
    assert creation["configuration"] == {"hnsw": {"space": "cosine"}}
    assert creation["embedding_function"] is None
    assert creation["metadata"]["build_status"] == "incomplete"
    assert collection.metadata["build_status"] == "complete"
    assert collection.metadata["expected_record_count"] == 7
    assert collection.records["embeddings"] == vectors
    assert set(collection.records["ids"]) == set(DOCUMENT_IDS)
    for doc, text, metadata in zip(documents, collection.records["documents"], collection.records["metadatas"]):
        assert text == build_embedding_text(doc)
        assert metadata["aliases_json"] == json.dumps(doc.aliases, ensure_ascii=False, separators=(",", ":"))
        assert metadata["source_url"] == doc.source.url
        assert metadata["source_publisher"] == doc.source.publisher
        assert metadata["source_title"] == doc.source.title
        assert metadata["source_accessed_date"] == doc.source.accessed_date
        assert metadata["document_sha256"] == document_sha256(doc)
        assert metadata["kb_sha256"] == knowledge_base_sha256()
        assert metadata["normalized_embeddings"] is True
        assert metadata["embedding_text_version"] == "1"
        assert "model_revision" not in metadata
        assert all(isinstance(value, (str, int, float, bool)) for value in metadata.values())
        assert not {"task_id", "report_id", "user_id", "patient_name", "query_history"} & metadata.keys()
    assert store.validate_index() == 7


def test_revision_included(setup_store):
    store, client, docs, vectors = setup_store
    store.model_revision = "pinned-revision"
    store.rebuild(docs, vectors)
    assert client.collections[store.collection_name].metadata["model_revision"] == "pinned-revision"
    assert all(row["model_revision"] == "pinned-revision" for row in client.collections[store.collection_name].records["metadatas"])


def test_document_hash_includes_source(documents):
    doc = documents[0]
    changed = doc.model_copy(deep=True)
    changed.source.url += "?changed"
    assert build_embedding_text(changed) == build_embedding_text(doc)
    assert document_sha256(changed) != document_sha256(doc)
    assert document_sha256(doc.model_copy(deep=True)) == document_sha256(doc)


def test_fingerprint_preserves_original_algorithm_and_evaluation(monkeypatch, tmp_path):
    from agents import evaluate_keyword_retrieval as evaluation
    legacy = hashlib.sha256()
    for path in sorted(KNOWLEDGE_BASE_DIR.glob("*.json"), key=lambda path: path.name):
        for raw in (path.name.encode("utf-8"), path.read_bytes()):
            legacy.update(len(raw).to_bytes(8, "big"))
            legacy.update(raw)
    monkeypatch.chdir(tmp_path)
    assert knowledge_base_sha256() == legacy.hexdigest()
    assert knowledge_base_sha256("data/knowledge_base") == legacy.hexdigest()
    assert evaluation.knowledge_base_sha256() == legacy.hexdigest()
    assert evaluation.build_report()["evaluation"]["knowledge_base_sha256"] == legacy.hexdigest()


INVALID_VECTORS = [[], [1], [0.0] * 7, [float("nan")] + [0.0] * 6,
                   [float("inf")] + [0.0] * 6, ["1"] + [0.0] * 6,
                   [True] + [0.0] * 6, [[1]] + [0.0] * 6]


@pytest.mark.parametrize("vector", INVALID_VECTORS)
def test_invalid_write_preserves_existing(built, vector):
    store, client, docs, vectors = built
    old = client.collections[store.collection_name]
    vectors[0] = vector
    with pytest.raises(VectorIndexBuildError, match="preparation"):
        store.rebuild(docs, vectors)
    assert client.collections[store.collection_name] is old
    assert client.deleted == []


@pytest.mark.parametrize("vector", INVALID_VECTORS)
def test_invalid_query_rejected_before_client(setup_store, vector):
    store, client, _, _ = setup_store
    with pytest.raises(ValueError):
        store.query(vector, 1)
    assert client.opened == []
    assert not store.directory.exists()


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "2", None])
def test_invalid_n_results(setup_store, value):
    store, _, _, vectors = setup_store
    with pytest.raises(ValueError, match="n_results"):
        store.query(vectors[0], value)


@pytest.mark.parametrize("failure", ["duplicate", "missing", "wrong_id", "vector_count", "unnormalized", "unapproved_content"])
def test_preparation_failure_leaves_previous_untouched(built, failure):
    store, client, docs, vectors = built
    previous = client.collections[store.collection_name]
    if failure == "duplicate":
        docs[1] = docs[0]
    elif failure == "missing":
        docs.pop()
    elif failure == "wrong_id":
        docs[0].id = "unknown"
    elif failure == "vector_count":
        vectors.pop()
    elif failure == "unnormalized":
        vectors[0] = [2.0] + [0.0] * 6
    else:
        docs[0].definition = "unapproved replacement"
    with pytest.raises(VectorIndexBuildError):
        store.rebuild(docs, vectors)
    assert client.collections[store.collection_name] is previous
    assert not client.deleted


def test_missing_index_query_does_not_create(tmp_path):
    factory = Mock(side_effect=AssertionError("No client for missing index"))
    store = ChromaVectorStore(tmp_path / "absent", dimension=7, client_factory=factory)
    with pytest.raises(VectorIndexNotBuiltError):
        store.query([1.0] + [0.0] * 6, 1)
    factory.assert_not_called()
    assert not store.directory.exists()
    store.directory.mkdir()
    with pytest.raises(VectorIndexNotBuiltError):
        store.query([1.0] + [0.0] * 6, 1)
    assert list(store.directory.iterdir()) == []


def test_missing_collection_does_not_create(setup_store):
    store, client, _, vectors = setup_store
    client.factory(path=str(store.directory))
    with pytest.raises(VectorIndexNotBuiltError):
        store.query(vectors[0], 1)
    assert not client.created


def test_query_flattening_clamping_and_explicit_embeddings(built):
    store, client, docs, vectors = built
    result = store.query(vectors[0], 20)
    assert len(result) == 7
    assert result[0] == VectorSearchResult(docs[0].id, 0.0,
        store._record_metadata(docs[0], knowledge_base_sha256()), build_embedding_text(docs[0]))
    client.collections[store.collection_name].query.assert_called_once_with(
        query_embeddings=[vectors[0]], n_results=7, include=["distances", "metadatas", "documents"])


@pytest.mark.parametrize("field,value", [
    ("kb_sha256", "changed"), ("embedding_model", "different"), ("model_revision", "different"),
    ("embedding_dimension", 384), ("normalized_embeddings", False), ("normalized_embeddings", 1),
    ("embedding_text_version", "2"), ("distance_metric", "l2"), ("expected_record_count", 6),
    ("build_status", "incomplete"),
])
def test_stale_collection_metadata(built, field, value):
    store, client, _, vectors = built
    collection = client.collections[store.collection_name]
    collection.metadata[field] = value
    with pytest.raises(VectorIndexStaleError, match="rebuild"):
        store.query(vectors[0], 1)
    collection.query.assert_not_called()


@pytest.mark.parametrize("failure", ["metric", "count", "ids", "text", "source"])
def test_stale_records_or_actual_configuration(built, failure):
    store, client, _, vectors = built
    collection = client.collections[store.collection_name]
    if failure == "metric":
        collection.configuration["hnsw"]["space"] = "l2"
    elif failure == "count":
        collection.records["ids"].pop()
    elif failure == "ids":
        collection.records["ids"][0] = "unexpected"
    elif failure == "text":
        collection.records["documents"][0] = "modified"
    else:
        collection.records["metadatas"][0]["source_url"] = "modified"
    with pytest.raises(VectorIndexStaleError):
        store.query(vectors[0], 1)


def test_empty_compatible_collection(built):
    store, client, _, vectors = built
    collection = client.collections[store.collection_name]
    collection.records = {key: [] for key in collection.records}
    assert store.query(vectors[0], 1) == []
    with pytest.raises(VectorIndexStaleError, match="count"):
        store.validate_index()
    collection.query.assert_not_called()


@pytest.mark.parametrize("response", [
    {"ids": [["a"], ["b"]], "distances": [[0.0]], "metadatas": [[{}]], "documents": [["text"]]},
    {"ids": [["a"]], "distances": [[]], "metadatas": [[{}]], "documents": [["text"]]},
    {"ids": None},
])
def test_malformed_query_shape(built, response):
    store, client, _, vectors = built
    client.collections[store.collection_name].query = Mock(return_value=response)
    with pytest.raises(VectorIndexStaleError):
        store.query(vectors[0], 1)


def test_rebuild_deletes_only_configured_collection(built):
    store, client, docs, vectors = built
    unrelated = object()
    client.collections["unrelated"] = unrelated
    store.rebuild(docs, vectors)
    assert client.deleted == [store.collection_name]
    assert client.collections["unrelated"] is unrelated
    assert store.validate_index() == 7


@pytest.mark.parametrize("failure", ["fail_add", "fail_verification"])
def test_failed_write_never_marks_complete(built, failure):
    store, client, docs, vectors = built
    setattr(client, failure, True)
    with pytest.raises(VectorIndexBuildError, match="write failed"):
        store.rebuild(docs, vectors)
    assert client.collections[store.collection_name].metadata["build_status"] == "incomplete"


def _local_chroma_scenario(path):
    """Runs in a child process so Windows releases SQLite/HNSW handles on exit."""
    import chromadb
    path = Path(path)
    kb = path / "kb"
    shutil.copytree(KNOWLEDGE_BASE_DIR, kb)
    docs = KnowledgeBaseLoader(kb).load_all()
    vectors = fake_vectors(docs)
    directory = path / "persistence"
    store = ChromaVectorStore(directory, dimension=7, kb_directory=kb, model_revision="", model_name="fake-model")
    assert store.rebuild(docs, vectors) == 7
    client = chromadb.PersistentClient(path=str(directory))
    client.create_collection(name="unrelated_collection", embedding_function=None)
    reopened = ChromaVectorStore(directory, dimension=7, kb_directory=kb, model_revision="", model_name="fake-model")
    assert reopened.validate_index() == 7
    results = reopened.query(vectors[2], 99)
    assert len(results) == 7
    assert results[0].document_id == docs[2].id
    assert results[0].distance == pytest.approx(0.0, abs=1e-6)
    assert all(result.distance == pytest.approx(1.0, abs=1e-6) for result in results[1:])
    assert results[0].document_text == build_embedding_text(docs[2])
    assert results[0].metadata["source_url"] == docs[2].source.url
    collection = client.get_collection(name=store.collection_name, embedding_function=None)
    assert set(collection.get()["ids"]) == set(DOCUMENT_IDS)
    assert collection.configuration["hnsw"]["space"] == "cosine"
    assert collection.configuration["embedding_function"] is None
    wrong = ChromaVectorStore(directory, dimension=7, kb_directory=kb, model_name="wrong", model_revision="")
    with pytest.raises(VectorIndexStaleError):
        wrong.query(vectors[0], 1)
    with (kb / "hdl.json").open("ab") as stream:
        stream.write(b"\n")
    with pytest.raises(VectorIndexStaleError):
        reopened.query(vectors[0], 1)
    reopened.rebuild(docs, vectors)
    assert reopened.validate_index() == 7
    assert client.get_collection(name="unrelated_collection", embedding_function=None).count() == 0
    # Leave persistence for a second, fresh process to reopen after all handles close.



def _reopen_local_chroma_snapshot(path):
    import chromadb
    path = Path(path)
    store = ChromaVectorStore(path / "persistence", dimension=7, kb_directory=path / "kb",
                              model_revision="", model_name="fake-model")
    docs = KnowledgeBaseLoader(path / "kb").load_all()
    assert store.validate_index() == 7
    assert store.query(fake_vectors(docs)[0], 1)[0].document_id == docs[0].id
    client = chromadb.PersistentClient(path=str(store.directory))
    assert client.get_collection(name="unrelated_collection", embedding_function=None).count() == 0
    client.delete_collection(name=store.collection_name)
    client.delete_collection(name="unrelated_collection")


def test_real_local_chroma_persistence_with_fake_vectors(tmp_path):
    # Two processes prove disk persistence independently of Chroma's client cache.
    for function in ("_local_chroma_scenario", "_reopen_local_chroma_snapshot"):
        code = ("import sys; sys.path.insert(0, 'tests'); from test_vector_store import "
                + function + "; " + function + "(sys.argv[1])")
        result = subprocess.run([sys.executable, "-B", "-c", code, str(tmp_path)], cwd=BACKEND_DIR,
                                capture_output=True, text=True, timeout=90)
        assert result.returncode == 0, result.stdout + result.stderr
