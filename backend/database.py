import os

from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.collection import Collection

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/lablens")
DATABASE_NAME = os.getenv("DATABASE_NAME", "lablens")

# MongoClient is lazy: it doesn't connect until the first real query,
# so importing this module is safe even when MongoDB isn't running (e.g. in tests).
client = MongoClient(MONGODB_URI)
db = client[DATABASE_NAME]


def get_users_collection() -> Collection:
    """FastAPI dependency that returns the users collection.

    Tests override this to swap in an in-memory fake database.
    """
    return db["users"]
