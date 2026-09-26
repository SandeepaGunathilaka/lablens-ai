import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pymongo.errors import PyMongoError

from agents.safety_agent import router as safety_agent_router
from agents.document_agent import router as document_agent_router
from database import get_users_collection
from security.auth import ensure_user_indexes, router as auth_router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        ensure_user_indexes(get_users_collection())
    except PyMongoError as exc:
        logger.warning("Could not create MongoDB indexes (is MongoDB running?): %s", exc)
    yield


app = FastAPI(title="LabLens AI API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(safety_agent_router)
app.include_router(document_agent_router)


@app.get("/")
def read_root():
    return {"message": "LabLens AI backend is running"}


@app.get("/health")
def health_check():
    return {"status": "ok"}
