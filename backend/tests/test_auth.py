import pytest

from security.auth import verify_password


VALID_USER = {"name": "Ada Lovelace", "email": "ada@example.com", "password": "correct-horse-42"}


def test_register_success(client, users):
    response = client.post("/auth/register", json=VALID_USER)

    assert response.status_code == 201
    body = response.json()
    assert set(body) == {"user_id", "name", "email"}  # no password or hash leaked
    assert body["name"] == "Ada Lovelace"
    assert body["email"] == "ada@example.com"

    stored = users.find_one({"email": "ada@example.com"})
    assert stored["user_id"] == body["user_id"]
    assert stored["password_hash"] != VALID_USER["password"]
    assert verify_password(VALID_USER["password"], stored["password_hash"])
    assert "password" not in stored
    assert "created_at" in stored


def test_register_duplicate_email_rejected(client, users):
    assert client.post("/auth/register", json=VALID_USER).status_code == 201

    # Same email with different casing must still count as a duplicate.
    duplicate = {**VALID_USER, "name": "Someone Else", "email": "ADA@Example.com"}
    response = client.post("/auth/register", json=duplicate)

    assert response.status_code == 409
    assert users.count_documents({}) == 1


@pytest.mark.parametrize("password", ["", "short", "        "])
def test_register_weak_password_rejected(client, users, password):
    response = client.post("/auth/register", json={**VALID_USER, "password": password})

    assert response.status_code == 422
    assert users.count_documents({}) == 0


def test_me_returns_public_profile(client, users):
    client.post("/auth/register", json=VALID_USER)
    token = client.post(
        "/auth/login", json={"email": VALID_USER["email"], "password": VALID_USER["password"]}
    ).json()["access_token"]

    response = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"user_id", "name", "email"}  # no password hash leaked
    assert body["name"] == "Ada Lovelace"


def test_me_requires_token(client):
    assert client.get("/auth/me").status_code == 401


def test_me_rejects_token_for_deleted_user(client, auth_headers):
    assert client.get("/auth/me", headers=auth_headers("no-such-user")).status_code == 401
