"""Select the retrieved evidence for one finding and turn it into cited passages.

Evidence without a titled source is dropped: an explanation must be attributable.
"""

from agents.safety_agent import RetrievedSource
from explanation_agent.models import Passage

_EXCERPT_LIMIT = 4000


def passages_for(test_name: str, retrieved_sources: list[RetrievedSource]) -> list[Passage]:
    wanted = test_name.casefold()
    passages: list[Passage] = []
    seen: set[str] = set()
    for entry in retrieved_sources:
        if entry.test_name.strip().casefold() != wanted:
            continue
        excerpt = " ".join(" ".join(_strings(entry.information)).split())
        if not excerpt:
            continue
        if len(excerpt) > _EXCERPT_LIMIT:
            excerpt = excerpt[:_EXCERPT_LIMIT].rstrip() + "..."
        for source in entry.sources:
            if not isinstance(source, dict):
                continue
            title = str(source.get("title") or "").strip()
            if not title or title.casefold() in seen:
                continue
            seen.add(title.casefold())
            url = str(source.get("url") or "").strip() or None
            passages.append(Passage(title=title, url=url, excerpt=excerpt))
    return passages


def _strings(value: object) -> list[str]:
    """String values inside nested dicts and lists. Keys are labels, not evidence."""
    if isinstance(value, dict):
        return [s for item in value.values() for s in _strings(item)]
    if isinstance(value, (list, tuple)):
        return [s for item in value for s in _strings(item)]
    if isinstance(value, str):
        return [value]
    return []
