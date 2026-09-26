import os
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv
from jose import JWTError, jwt

load_dotenv()

JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
JWT_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "60"))
JWT_ALGORITHM = "HS256"

# Anyone who knows the secret can forge tokens, so never fall back to a default.
if not JWT_SECRET_KEY:
    raise RuntimeError("JWT_SECRET_KEY is not set. Add it to backend/.env (see .env.example).")


def create_access_token(user_id: str, expires_delta: timedelta | None = None) -> str:
    """Sign a JWT whose subject ("sub") is the user's id."""
    now = datetime.now(timezone.utc)
    expire = now + (expires_delta or timedelta(minutes=JWT_EXPIRE_MINUTES))
    claims = {"sub": user_id, "iat": now, "exp": expire}
    return jwt.encode(claims, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


def decode_access_token(token: str) -> str | None:
    """Return the user_id inside a valid token, or None if it's invalid or expired."""
    try:
        # Checks the signature and the "exp" claim. Pinning algorithms stops an
        # attacker from picking a weaker one (e.g. "none") in the token header.
        claims = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
    except JWTError:
        return None

    user_id = claims.get("sub")
    if not isinstance(user_id, str) or not user_id:
        return None
    return user_id
