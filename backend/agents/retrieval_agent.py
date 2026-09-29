"""Read-only orchestration of curated evidence through the hybrid retriever."""

from agents.hybrid_retriever import HybridRetriever
from agents.retrieval_models import (
    RetrievalMatch,
    RetrievalRequest,
    RetrievalResponse,
    RetrievalResult,
    RetrievalSource,
)


class MedicalRetrievalAgent:
    """Return evidence, never patient interpretation or generated explanations.

    Request validation belongs to RetrievalRequest. Retrieval errors propagate to
    the caller; only a completed hybrid abstention becomes a not-found result.
    """

    def __init__(self, hybrid_retriever: HybridRetriever | None = None):
        self._hybrid_retriever = (
            hybrid_retriever if hybrid_retriever is not None else HybridRetriever()
        )

    def retrieve(self, request: RetrievalRequest) -> RetrievalResponse:
        """Retrieve once per input position, preserving order, duplicates and IDs."""
        results = []
        for query in request.test_names:
            retrieval = self._hybrid_retriever.retrieve(query)
            if not retrieval.found:
                results.append(RetrievalResult(test_name=query, found=False))
                continue

            document = retrieval.document
            if document is None:
                raise ValueError("Successful hybrid retrieval requires a KnowledgeDocument")

            # The existing passage contract has no structured medical-content or
            # publisher/date fields. Keep curated text verbatim in labeled sections.
            information = "\n\n".join((
                f"Test: {document.test_name}",
                f"Title: {document.title}",
                f"Definition: {document.definition}",
                f"What it measures: {document.what_it_measures}",
                f"General information: {document.general_information}",
                f"Source publisher: {document.source.publisher}",
                f"Source accessed date: {document.source.accessed_date}",
            ))
            results.append(RetrievalResult(
                test_name=query,
                found=True,
                matches=[RetrievalMatch(
                    information=information,
                    sources=[RetrievalSource(
                        title=document.source.title,
                        url=document.source.url,
                    )],
                )],
            ))

        return RetrievalResponse(
            task_id=request.task_id,
            report_id=request.report_id,
            user_id=request.user_id,
            results=results,
        )
