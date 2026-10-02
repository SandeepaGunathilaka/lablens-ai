"""CLI orchestration uses only injected encoders and temporary fake persistence."""

from unittest.mock import Mock

import pytest

from agents.build_vector_index import main
from test_vector_store import (
    built, documents, fake_vectors, isolated_environment, setup_store,
)


def encoder_factory(documents):
    return Mock(return_value=Mock(encode_documents=Mock(return_value=fake_vectors(documents))))


def test_initial_build_and_summary(setup_store, capsys):
    store, client, docs, _ = setup_store
    factory = encoder_factory(docs)
    assert main([], store=store, embedding_factory=factory) == 0
    assert store.validate_index() == 7
    factory.return_value.encode_documents.assert_called_once()
    text = capsys.readouterr().out
    for expected in ("built successfully", store.collection_name, str(store.directory), "Documents: 7",
                     store.model_name, "Dimension: 7", "KB SHA-256:", "Status: complete"):
        assert expected in text
    assert "[1.0" not in text
    assert client.deleted == []


def test_existing_index_does_not_load_model_or_overwrite(built, capsys):
    store, client, _, _ = built
    factory = Mock(side_effect=AssertionError("Existing index must not load encoder"))
    assert main([], store=store, embedding_factory=factory) == 0
    assert "already exists" in capsys.readouterr().out
    factory.assert_not_called()
    assert client.deleted == []
    assert len(client.created) == 1


def test_stale_index_requires_explicit_rebuild(built, capsys):
    store, client, _, _ = built
    client.collections[store.collection_name].metadata["kb_sha256"] = "stale"
    factory = Mock(side_effect=AssertionError("Stale index must not load encoder"))
    assert main([], store=store, embedding_factory=factory) == 1
    assert "--rebuild" in capsys.readouterr().err
    factory.assert_not_called()
    assert client.deleted == []


def test_explicit_rebuild_replaces_only_selected_collection(built):
    store, client, docs, _ = built
    unrelated = object()
    client.collections["unrelated"] = unrelated
    client.collections[store.collection_name].metadata["build_status"] = "incomplete"
    assert main(["--rebuild"], store=store, embedding_factory=encoder_factory(docs)) == 0
    assert client.deleted == [store.collection_name]
    assert client.collections["unrelated"] is unrelated
    assert store.validate_index() == 7


@pytest.mark.parametrize("failure", ["encoding", "invalid_vector"])
def test_preparation_failure_preserves_previous_index(built, failure, capsys):
    store, client, docs, _ = built
    previous = client.collections[store.collection_name]
    factory = encoder_factory(docs)
    if failure == "encoding":
        factory.return_value.encode_documents.side_effect = RuntimeError("synthetic encoder failure")
    else:
        factory.return_value.encode_documents.return_value[0] = []
    assert main(["--rebuild"], store=store, embedding_factory=factory) == 1
    assert "failed" in capsys.readouterr().err
    assert client.collections[store.collection_name] is previous
    assert client.deleted == []
    assert store.validate_index() == 7


def test_failed_insertion_exit_code(built, capsys):
    store, client, docs, _ = built
    client.fail_add = True
    assert main(["--rebuild"], store=store, embedding_factory=encoder_factory(docs)) == 1
    assert "write failed" in capsys.readouterr().err
    assert client.collections[store.collection_name].metadata["build_status"] == "incomplete"


def test_invalid_cli_argument():
    with pytest.raises(SystemExit) as exc:
        main(["--unknown"])
    assert exc.value.code == 2
