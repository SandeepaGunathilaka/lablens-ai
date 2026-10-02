"""Shared helpers for the assessment suites: real retrieval, a scripted model and text PDFs."""

import json
import re
from pathlib import Path

import pymupdf as fitz
import pytest

from agents.embedding_service import DEFAULT_MODEL_NAME, EmbeddingService
from agents.hybrid_retriever import HybridRetriever
from agents.keyword_retriever import KeywordRetriever
from agents.knowledge_base import KNOWLEDGE_BASE_DIR, KnowledgeBaseLoader
from agents.retrieval_agent import MedicalRetrievalAgent
from agents.semantic_retriever import SemanticRetriever
from agents.vector_store import ChromaVectorStore
from explanation_agent.copy import RECOMMENDED_DISCUSSION

KB_DIR = KNOWLEDGE_BASE_DIR


def load_kb_files(directory: Path = KB_DIR) -> dict[str, dict]:
    return {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted(directory.glob("*.json"))}


class RealRetrieval:
    """The production retrieval stack (keyword + MiniLM/Chroma semantic) over a KB directory.

    The vector index is rebuilt into ``index_dir`` from the given KB with the real embedding
    model, exactly as ``agents/build_vector_index.py`` does for data/chroma.
    """

    def __init__(self, index_dir: Path, kb_dir: Path = KB_DIR, collection: str = "assessment"):
        self.kb_dir = kb_dir
        self.loader = KnowledgeBaseLoader(kb_dir)
        documents = self.loader.load_all()
        self.encoder = EmbeddingService(model_name=DEFAULT_MODEL_NAME, device="cpu")
        self.store = ChromaVectorStore(index_dir, collection_name=collection, model_name=self.encoder.model_name,
                                       model_revision=self.encoder.revision or "", kb_directory=kb_dir)
        self.store.rebuild(documents, self.encoder.encode_documents(documents))
        self.semantic = SemanticRetriever(self.encoder, self.store, self.loader)
        self.hybrid = HybridRetriever(KeywordRetriever(self.loader), self.semantic)
        self.agent = MedicalRetrievalAgent(self.hybrid)


@pytest.fixture(scope="session")
def real_retrieval(tmp_path_factory) -> RealRetrieval:
    """The production keyword + MiniLM/Chroma stack over the curated knowledge base."""
    return RealRetrieval(tmp_path_factory.mktemp("chroma") / "index")


def text_pdf(text: str) -> bytes:
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), text, fontsize=11)
    content = document.tobytes()
    document.close()
    return content


# --- Scripted language model -----------------------------------------------------------------


def _prompt_field(prompt: str, label: str) -> list[str]:
    return re.findall(rf"^{label}: (.+)$", prompt, re.MULTILINE)


def grounded(prompt: str, **overrides) -> str:
    """A well-behaved draft: copies the required sentences and cites only retrieved titles."""
    title = _prompt_field(prompt, "Title")[0]
    test = _prompt_field(prompt, "Test name")[0]
    draft = {
        "what_it_measures": f"{title} describes what the {test} test measures in a blood sample.",
        "explanation": " ".join(_prompt_field(prompt, "Required sentence")) + f" {title} is the sole basis for this explanation.",
        "possible_meaning": f"General educational context comes only from {title}. No personal condition is assigned.",
        "recommended_discussion": RECOMMENDED_DISCUSSION,
        "insufficient_information": False,
        "sources_used": [title],
    }
    draft.update(overrides)
    return json.dumps(draft)


def draft(**overrides):
    return lambda prompt: grounded(prompt, **overrides)


class ScriptedLLM:
    """Stands in for Gemini. Replays drafts in order; the last one repeats."""

    def __init__(self, *drafts):
        self._drafts = list(drafts)
        self.calls: list[tuple[str, str]] = []

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        item = self._drafts.pop(0) if len(self._drafts) > 1 else self._drafts[0]
        return item(user_prompt) if callable(item) else item
