import base64
import json
from datetime import timedelta

import pytest

from security.tokens import JWT_EXPIRE_MINUTES, create_access_token

USER = {"name": "Grace Hopper", "email": "grace@example.com", "password": "compile-me-1952"}


@pytest.fixture
def registered_user(client):
    response = client.post("/auth/register", json=USER)
    assert response.status_code == 201
    return response.json()


def login(client, email=USER["email"], password=USER["password"]):
    return client.post("/auth/login", json={"email": email, "password": password})


def auth_header(token):
    return {"Authorization": f"Bearer {token}"}


def test_login_success_returns_token(client, registered_user):
    # Mixed-case email should still find the (lowercased) stored user.
    response = login(client, email="Grace@Example.com")

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"access_token", "token_type", "expires_in"}
    assert body["token_type"] == "bearer"
    assert body["expires_in"] == JWT_EXPIRE_MINUTES * 60
    assert body["access_token"]


def test_login_wrong_password_returns_401(client, registered_user):
    response = login(client, password="not-the-password")

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid email or password"


def test_login_unknown_email_returns_same_401(client, registered_user):
    unknown = login(client, email="nobody@example.com")
    wrong_password = login(client, password="not-the-password")

    assert unknown.status_code == 401
    # Identical responses, so an attacker can't tell which part was wrong.
    assert unknown.json() == wrong_password.json()


def test_me_without_token_returns_401(client):
    response = client.get("/auth/me")

    assert response.status_code == 401


def test_me_with_valid_token_returns_user_id(client, registered_user):
    token = login(client).json()["access_token"]

    response = client.get("/auth/me", headers=auth_header(token))

    assert response.status_code == 200
    assert response.json() == {"user_id": registered_user["user_id"]}


def test_me_with_garbage_or_tampered_token_returns_401(client, registered_user):
    token = login(client).json()["access_token"]
    header, _payload, signature = token.split(".")
    # Pretend to be another user by swapping in a new payload but keeping the original signature.
    forged_payload = base64.urlsafe_b64encode(
        json.dumps({"sub": "someone-else", "exp": 9999999999}).encode()
    ).rstrip(b"=").decode()
    tampered = f"{header}.{forged_payload}.{signature}"

    for bad_token in ["not-a-jwt", tampered]:
        response = client.get("/auth/me", headers=auth_header(bad_token))
        assert response.status_code == 401


def test_me_with_expired_token_returns_401(client, registered_user):
    expired = create_access_token(registered_user["user_id"], expires_delta=timedelta(minutes=-1))

    response = client.get("/auth/me", headers=auth_header(expired))

    assert response.status_code == 401
