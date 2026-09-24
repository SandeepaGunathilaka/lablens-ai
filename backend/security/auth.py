import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr, Field, field_validator
from pymongo.collection import Collection
from pymongo.errors import DuplicateKeyError

from database import get_users_collection
from security.dependencies import get_current_user
from security.tokens import JWT_EXPIRE_MINUTES, create_access_token

router = APIRouter(prefix="/auth", tags=["auth"])

# bcrypt is deliberately slow and salts every hash, so identical passwords
# produce different hashes and brute-forcing a leaked hash is expensive.
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# bcrypt only looks at the first 72 bytes of a password; longer ones are rejected
# so two passwords sharing the same first 72 bytes can't both "match".
BCRYPT_MAX_BYTES = 72


class RegisterRequest(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    email: EmailStr
    password: str = Field(min_length=8)

    @field_validator("name")
    @classmethod
    def name_not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Name cannot be blank")
        return value

    @field_validator("email")
    @classmethod
    def normalize_email(cls, value: str) -> str:
        # Store emails lowercase so "Bob@X.com" and "bob@x.com" count as duplicates.
        return value.lower()

    @field_validator("password")
    @classmethod
    def password_strength(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Password cannot be only whitespace")
        if len(value.encode("utf-8")) > BCRYPT_MAX_BYTES:
            raise ValueError(f"Password cannot be longer than {BCRYPT_MAX_BYTES} bytes")
        return value


class LoginRequest(BaseModel):
    email: EmailStr
    password: str

    @field_validator("email")
    @classmethod
    def normalize_email(cls, value: str) -> str:
        return value.lower()


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int  # seconds until the token expires


class UserPublic(BaseModel):
    """The only user fields ever sent back to a client."""

    user_id: str
    name: str
    email: EmailStr


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    return pwd_context.verify(password, password_hash)


def ensure_user_indexes(users: Collection) -> None:
    """Unique index on email: the database itself refuses duplicate emails."""
    users.create_index("email", unique=True)


@router.post("/register", response_model=UserPublic, status_code=status.HTTP_201_CREATED)
def register(payload: RegisterRequest, users: Collection = Depends(get_users_collection)):
    # Fast, friendly check for the common case.
    if users.find_one({"email": payload.email}):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Email is already registered")

    user = {
        "user_id": str(uuid.uuid4()),
        "name": payload.name,
        "email": payload.email,
        "password_hash": hash_password(payload.password),
        "created_at": datetime.now(timezone.utc),
    }

    try:
        users.insert_one(user)
    except DuplicateKeyError:
        # Two requests with the same email raced past find_one; the unique index caught it.
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Email is already registered")

    # response_model=UserPublic strips everything except user_id, name and email.
    return user


@router.post("/login", response_model=TokenResponse)
def login(payload: LoginRequest, users: Collection = Depends(get_users_collection)):
    user = users.find_one({"email": payload.email})

    if user is None:
        # Spend the same bcrypt time as a real check, so response timing doesn't
        # reveal whether the email is registered.
        pwd_context.dummy_verify()
        password_ok = False
    else:
        password_ok = verify_password(payload.password, user["password_hash"])

    if not password_ok:
        # One message for both cases, so attackers can't probe which emails exist.
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return TokenResponse(
        access_token=create_access_token(user["user_id"]),
        expires_in=JWT_EXPIRE_MINUTES * 60,
    )


# Throwaway route to prove the JWT flow end-to-end; remove once real protected routes exist.
@router.get("/me")
def me(user_id: str = Depends(get_current_user)):
    return {"user_id": user_id}
