import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pymongo.errors import PyMongoError


load_dotenv(Path(__file__).resolve().parent / ".env")

from agents.document_agent import router as document_agent_router  # noqa: E402
from agents.safety_agent import router as safety_agent_router  # noqa: E402
from api.analysis import router as analysis_router  # noqa: E402
from api.audit import router as audit_router  # noqa: E402
from api.chats import router as chats_router  # noqa: E402
from api.reports import ensure_report_indexes, router as reports_router  # noqa: E402
from api.retrieval import router as retrieval_router  # noqa: E402
from database import (  # noqa: E402
    get_audit_logs_collection,
    get_chats_collection,
    get_report_files_collection,
    get_reports_collection,
    get_users_collection,
)
from explanation_agent.router import router as explanation_router  # noqa: E402
from logging_service import ensure_audit_log_indexes  # noqa: E402
from security.auth import ensure_user_indexes, router as auth_router  # noqa: E402

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        ensure_user_indexes(get_users_collection())
        ensure_audit_log_indexes(get_audit_logs_collection())
        ensure_report_indexes(get_reports_collection(), get_report_files_collection(), get_chats_collection())
    except PyMongoError as exc:
        logger.warning("Could not create MongoDB indexes (is MongoDB running?): %s", exc)
    yield


app = FastAPI(title="LabLens AI API", version="0.1.0", lifespan=lifespan)

DEFAULT_CORS_ORIGINS = "http://localhost:5173,http://127.0.0.1:5173,http://localhost:8080,http://127.0.0.1:8080"
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", DEFAULT_CORS_ORIGINS).split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(document_agent_router)
app.include_router(safety_agent_router)
app.include_router(analysis_router)
app.include_router(audit_router)
app.include_router(retrieval_router)
app.include_router(explanation_router)
app.include_router(reports_router)
app.include_router(chats_router)


@app.get("/")
def read_root():
    return {"message": "LabLens AI backend is running"}


@app.get("/health")
def health_check():
    return {"status": "ok"}
