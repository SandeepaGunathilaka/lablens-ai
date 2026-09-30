"""JSON contracts for retrieving curated educational evidence, not explanations.

The caller carries the same task/report/user IDs into the response and returns
one result per requested test, in request order. Unknown names are valid inputs:
the retrieval implementation must return ``found=False`` with no matches.
These models do not authenticate users or verify that content is KB-approved.
"""

from typing import Annotated

from pydantic import BaseModel, Field, HttpUrl, StringConstraints, model_validator


NonBlankText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class RetrievalRequest(BaseModel):
    """Test names only; patient laboratory values are not needed for retrieval."""

    task_id: str
    report_id: str
    user_id: str
    test_names: list[NonBlankText] = Field(min_length=1)


class RetrievalSource(BaseModel):
    """Attribution for a curated piece of educational information."""

    title: NonBlankText
    url: HttpUrl


class RetrievalMatch(BaseModel):
    """An educational passage with at least one supporting source."""

    information: NonBlankText
    sources: list[RetrievalSource] = Field(min_length=1)


class RetrievalResult(BaseModel):
    """Evidence for one requested test; false means no reliable information."""

    test_name: NonBlankText
    found: bool
    matches: list[RetrievalMatch] = Field(default_factory=list)

    @model_validator(mode="after")
    def matches_agree_with_found(self) -> "RetrievalResult":
        if self.found != bool(self.matches):
            raise ValueError("found must be true exactly when matches are present")
        return self


class RetrievalResponse(BaseModel):
    """Per-test evidence with the originating request's correlation IDs."""

    task_id: str
    report_id: str
    user_id: str
    results: list[RetrievalResult] = Field(min_length=1)
