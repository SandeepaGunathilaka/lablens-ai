from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from security.tokens import decode_access_token

# Reads "Authorization: Bearer <token>" and adds an "Authorize" button to /docs.
# auto_error=False lets us return our own 401 instead of FastAPI's default error.
bearer_scheme = HTTPBearer(auto_error=False)


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> str:
    """FastAPI dependency: returns the logged-in user's id, or raises 401."""
    user_id = decode_access_token(credentials.credentials) if credentials else None
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user_id
