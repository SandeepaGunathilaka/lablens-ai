"""Deterministic fingerprints of static curated knowledge, without network I/O."""

import hashlib
import json
from pathlib import Path

from agents.knowledge_base import KNOWLEDGE_BASE_DIR, KnowledgeDocument


BACKEND_DIR = Path(__file__).resolve().parents[1]
DOCUMENT_IDS = ("hemoglobin", "wbc", "platelets", "hdl", "ldl", "triglycerides", "total_cholesterol")


def resolve_backend_path(directory: Path | str) -> Path:
    path = Path(directory)
    return (path if path.is_absolute() else BACKEND_DIR / path).resolve()


def knowledge_base_sha256(directory: Path | str = KNOWLEDGE_BASE_DIR) -> str:
    directory = resolve_backend_path(directory)
    paths = sorted(directory.glob("*.json"), key=lambda path: path.name)
    if {path.name for path in paths} != {f"{identifier}.json" for identifier in DOCUMENT_IDS}:
        raise ValueError("KB hashing requires exactly the seven approved JSON filenames")
    digest = hashlib.sha256()
    for path in paths:
        for data in (path.name.encode("utf-8"), path.read_bytes()):
            digest.update(len(data).to_bytes(8, "big"))
            digest.update(data)
    return digest.hexdigest()


def document_sha256(document: KnowledgeDocument) -> str:
    payload = json.dumps(document.model_dump(mode="json"), sort_keys=True,
                         ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
