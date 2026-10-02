"""Exact approved vocabulary first, then curated lay phrases, then frozen-policy semantic fallback."""

from dataclasses import dataclass
from typing import Literal

from agents.knowledge_base import KnowledgeDocument
from agents.keyword_retriever import KeywordRetriever
from agents.lay_terms import match_lay_term
from agents.semantic_retriever import SemanticRetriever, SemanticRetrievalDecision


@dataclass(frozen=True)
class HybridRetrievalResult:
    document: KnowledgeDocument | None
    found: bool
    method: Literal["keyword", "lay_term", "semantic", "none"]
    semantic_decision: SemanticRetrievalDecision | None


class HybridRetriever:
    """Read-only retrieval; components own normalization and acceptance policy.

    Default semantic construction is deferred until a keyword miss, so approved
    vocabulary needs neither an embedding model nor a usable vector index.
    Infrastructure/integrity errors propagate unchanged.
    """

    def __init__(self, keyword_retriever: KeywordRetriever | None = None,
                 semantic_retriever: SemanticRetriever | None = None):
        self._keyword_retriever = keyword_retriever if keyword_retriever is not None else KeywordRetriever()
        self._semantic_retriever = semantic_retriever

    def retrieve(self, query: str) -> HybridRetrievalResult:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a nonblank string")
        document = self._keyword_retriever.retrieve(query)
        if document is not None:
            return HybridRetrievalResult(document, True, "keyword", None)
        lay_name = match_lay_term(query)
        if lay_name is not None:
            document = self._keyword_retriever.retrieve(lay_name)
            if document is not None:
                return HybridRetrievalResult(document, True, "lay_term", None)
        if self._semantic_retriever is None:
            self._semantic_retriever = SemanticRetriever()
        decision = self._semantic_retriever.retrieve(query)
        if decision.accepted:
            return HybridRetrievalResult(decision.document, True, "semantic", decision)
        return HybridRetrievalResult(None, False, "none", decision)
