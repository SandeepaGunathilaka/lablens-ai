import os

from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.collection import Collection

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/lablens")
DATABASE_NAME = os.getenv("DATABASE_NAME", "lablens")

# If MongoDB is unreachable, give up after this long instead of pymongo's default 30s,
# so requests (and startup) fail fast rather than hanging.
SERVER_SELECTION_TIMEOUT_MS = 5000

# MongoClient is lazy: it doesn't connect until the first real query,
# so importing this module is safe even when MongoDB isn't running (e.g. in tests).
client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=SERVER_SELECTION_TIMEOUT_MS)
db = client[DATABASE_NAME]


def get_users_collection() -> Collection:
    """FastAPI dependency that returns the users collection.

    Tests override this to swap in an in-memory fake database.
    """
    return db["users"]


def get_audit_logs_collection() -> Collection:
    """FastAPI dependency that returns the audit_logs collection.

    Tests override this to swap in an in-memory fake database.
    """
    return db["audit_logs"]


def get_reports_collection() -> Collection:
    """FastAPI dependency that returns the saved reports collection."""
    return db["reports"]


def get_report_files_collection() -> Collection:
    """FastAPI dependency that returns the uploaded original files, one document per report."""
    return db["report_files"]


def get_chats_collection() -> Collection:
    """FastAPI dependency that returns the chats collection (messages are embedded)."""
    return db["chats"]
