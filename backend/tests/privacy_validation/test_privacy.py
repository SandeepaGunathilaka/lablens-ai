"""Student 2 assessment suite: Privacy and Data Leakage (PRIV-01..PRIV-15, API and database part).

Every request goes through the real FastAPI app: real bcrypt hashing, real JWT issuing and
verification, real ownership filters and the real Coordinator (template explanation mode).
MongoDB is replaced by an in-memory mongomock database so raw stored documents can be
inspected; the browser-based cases PRIV-04 and PRIV-05 are in test_privacy_browser.py.
"""

import json
import re
import statistics
import subprocess
import time
from datetime import timedelta

import pytest
from fastapi.testclient import TestClient
from pymongo.errors import ServerSelectionTimeoutError

import coordinator
from agents.safety_agent import DISCLAIMER
from database import get_reports_collection
from evidence_support import BACKEND_DIR
from explanation_agent.router import clear_rate_limits
from main import CORS_ORIGINS, app
from pipeline_support import text_pdf
from security.tokens import create_access_token

PASSWORD = "Str0ng-Passw0rd!"
REPORT_TEXT = "Patient: Nimal Perera\nHemoglobin 11.2 g/dL 12.0-15.5\nPlatelets 162 10^3/uL 150-400"


@pytest.fixture(autouse=True)
def template_mode():
    coordinator._default_explanation_service.cache_clear()
    clear_rate_limits()
    yield
    coordinator._default_explanation_service.cache_clear()


def register(client, email, name="Kasun Edirisinghe", password=PASSWORD):
    return client.post("/auth/register", json={"name": name, "email": email, "password": password})


def login(client, email, password=PASSWORD):
    return client.post("/auth/login", json={"email": email, "password": password})


def bearer(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def account(client, email):
    user = register(client, email).json()
    return user, bearer(login(client, email).json()["access_token"])


def upload(client, headers, text=REPORT_TEXT, filename="nimal_perera_cbc.pdf"):
    return client.post("/api/reports", files={"file": (filename, text_pdf(text), "application/pdf")}, headers=headers)


def safety_payload(user_id):
    return {"task_id": "task-priv", "report_id": "report-priv", "user_id": user_id,
            "original_result": [{"test": "Hemoglobin", "value": 11.2, "unit": "g/dL", "reference_range": "12.0-15.5"}],
            "retrieved_sources": [{"test_name": "Hemoglobin", "information": {"description": "Hemoglobin carries oxygen."}, "sources": []}],
            "draft_response": f"Your hemoglobin is 11.2 g/dL. {DISCLAIMER}"}


# --- PRIV-01 / 02 / 03 Credentials and enumeration ---------------------------------------------


def test_priv01_no_password_in_responses(evidence, client, users):
    ev = evidence("PRIV-01", "Password exposure in API responses", "UserPublic/TokenResponse response models; bcrypt storage",
                  "Response body never contains password or password_hash in either call.")
    ev.given(register={"name": "Kasun Edirisinghe", "email": "kasun@example.com", "password": PASSWORD},
             login={"email": "kasun@example.com", "password": PASSWORD})

    reg, log = register(client, "kasun@example.com"), login(client, "kasun@example.com")
    stored = users.find_one({"email": "kasun@example.com"})
    ev.check("Register: status", 201, reg.status_code)
    ev.check("Register: response fields", ["email", "name", "user_id"], sorted(reg.json()))
    ev.check("Login: response fields", ["access_token", "expires_in", "token_type"], sorted(log.json()))
    for label, response in (("Register", reg), ("Login", log)):
        ev.absent(f"{label}: no 'password' key or value", response.text, "password")
        ev.absent(f"{label}: no plaintext password", response.text, PASSWORD)
        ev.absent(f"{label}: no bcrypt hash", response.text, "$2b$")
    ev.check("Database: plaintext never stored", False, PASSWORD in json.dumps(stored, default=str))
    ev.check("Database: bcrypt hash stored", "$2b$12$...", stored["password_hash"][:7] + "...")
    ev.output("register response", reg.json())
    ev.output("login response (token shortened)", {**log.json(), "access_token": log.json()["access_token"][:24] + "..."})
    ev.verify()


def test_priv02_account_existence_leakage(evidence, client):
    ev = evidence("PRIV-02", "Account-existence leakage via register vs. login", "Duplicate check on register; single error on login",
                  "Register leaks existence via 409 (expected, common pattern). Login must return the identical 401 message for both wrong-password and unknown-email cases.")
    register(client, "dup@example.com")
    ev.given(registered="dup@example.com", second_register="dup@example.com",
             login_wrong_password="dup@example.com / Wrong-Passw0rd!", login_unknown_email="ghost@example.com / Wrong-Passw0rd!")

    again = register(client, "DUP@example.com")
    wrong = login(client, "dup@example.com", "Wrong-Passw0rd!")
    ghost = login(client, "ghost@example.com", "Wrong-Passw0rd!")
    ev.check("Second register (case-insensitive): status/detail", "409 Email is already registered",
             f"{again.status_code} {again.json()['detail']}")
    ev.check("Login wrong password: status/detail", "401 Invalid email or password", f"{wrong.status_code} {wrong.json()['detail']}")
    ev.check("Login unknown email: status/detail", "401 Invalid email or password", f"{ghost.status_code} {ghost.json()['detail']}")
    ev.check("Login bodies byte-identical", True, wrong.content == ghost.content)
    ev.check("Login WWW-Authenticate headers identical", wrong.headers.get("www-authenticate"), ghost.headers.get("www-authenticate"))
    ev.output("wrong password response", wrong.json())
    ev.output("unknown email response", ghost.json())
    ev.note("The 409 on registration confirms that an email is registered. The assessment accepts this as a common "
            "pattern; it lets an attacker enumerate accounts through /auth/register rather than /auth/login.")
    ev.verify()


def test_priv03_login_timing(evidence, client):
    ev = evidence("PRIV-03", "User enumeration via login response timing", "pwd_context.dummy_verify() for unknown emails",
                  "Response time difference between the two cases is not reliably distinguishable (dummy hash verification should mask it).")
    register(client, "timing@example.com")
    ev.given(known_email="timing@example.com (wrong password)", unknown_email="nobody-xyz@example.com", repetitions=5,
             order="interleaved, after one warm-up call each")

    login(client, "timing@example.com", "warm-up-wrong")
    login(client, "nobody-xyz@example.com", "warm-up-wrong")
    known, unknown = [], []
    for i in range(5):
        for email, bucket in (("timing@example.com", known), ("nobody-xyz@example.com", unknown)):
            start = time.perf_counter()
            response = login(client, email, f"Wrong-Passw0rd-{i}")
            bucket.append((time.perf_counter() - start) * 1000)
            assert response.status_code == 401
    k_mean, u_mean = statistics.mean(known), statistics.mean(unknown)
    gap = abs(k_mean - u_mean)
    spread = max(statistics.stdev(known), statistics.stdev(unknown))
    overlap = min(max(known), max(unknown)) >= max(min(known), min(unknown))
    ev.check("Both cases spend a bcrypt verification", "both > 50 ms", f"known {k_mean:.0f} ms, unknown {u_mean:.0f} ms",
             passed=k_mean > 50 and u_mean > 50)
    ev.check("Mean difference below 10% of the mean", "< 10%", f"{100 * gap / ((k_mean + u_mean) / 2):.1f}% ({gap:.1f} ms)",
             passed=gap < 0.10 * (k_mean + u_mean) / 2)
    ev.check("Timing ranges overlap", True, overlap)
    ev.output("known email, wrong password (ms)", [round(t, 1) for t in known])
    ev.output("unknown email (ms)", [round(t, 1) for t in unknown])
    ev.output("summary", {"known_mean_ms": round(k_mean, 1), "unknown_mean_ms": round(u_mean, 1),
                          "difference_ms": round(gap, 1), "largest_stdev_ms": round(spread, 1)})
    ev.verify()


# --- PRIV-06 to PRIV-09 Authorization and audit content -------------------------------------------


def test_priv06_cross_user_audit_access(evidence, client, audit_logs):
    ev = evidence("PRIV-06", "Cross-user audit log access", "GET /api/audit/{task_id} filtered by the token's user_id",
                  "Empty list returned, identical to an unknown task_id. No indication the task exists for another user.")
    user_a, a = account(client, "a@example.com")
    user_b, b = account(client, "b@example.com")
    report = upload(client, b).json()
    task = report["task_id"]
    ev.given(user_A=user_a["user_id"], user_B=user_b["user_id"], task_of_B=task, unknown_task="no-such-task")

    own = client.get(f"/api/audit/{task}", headers=b)
    other = client.get(f"/api/audit/{task}", headers=a)
    unknown = client.get("/api/audit/no-such-task", headers=a)
    ev.check("B reads own task: entries returned", True, own.status_code == 200 and len(own.json()) > 0)
    ev.check("A reads B's task", "200 []", f"{other.status_code} {other.text}")
    ev.check("A reads unknown task", "200 []", f"{unknown.status_code} {unknown.text}")
    ev.check("Responses indistinguishable", True, other.content == unknown.content and other.status_code == unknown.status_code)
    ev.output("B's own audit trail (agent/action/status)", [f"{e['agent']}.{e['action']}: {e['status']}" for e in own.json()])
    ev.verify()


def test_priv07_cross_user_safety_endpoint(evidence, client, audit_logs):
    ev = evidence("PRIV-07", "Cross-user Safety endpoint access", "user_id in body must match the token subject",
                  "403 Forbidden, request rejected before any checks run.")
    user_a, a = account(client, "a@example.com")
    user_b, _ = account(client, "b@example.com")
    ev.given(token_of="A", payload_user_id=f"{user_b['user_id']} (B)")

    response = client.post("/agents/safety/validate", json=safety_payload(user_b["user_id"]), headers=a)
    ev.check("Status", 403, response.status_code)
    ev.check("Detail", "Not allowed for this user", response.json().get("detail"))
    ev.check("No safety decision logged (rejected before checks ran)", 0, audit_logs.count_documents({}))
    own = client.post("/agents/safety/validate", json=safety_payload(user_a["user_id"]), headers=a)
    ev.check("Control: A with own user_id", 200, own.status_code)
    ev.output("response", response.json())
    ev.verify()


def test_priv08_audit_log_content(evidence, client, users, audit_logs):
    ev = evidence("PRIV-08", "Audit log content does not leak PII or patient data", "log_event callers record ids, outcomes and counts only",
                  "No password, password hash, lab values, draft explanation text, or other patient content appears in any audit entry, only IDs, agent names, statuses and counts.")
    user, headers = account(client, "auditee@example.com")
    report = upload(client, headers).json()
    chat = client.post("/api/chats", json={"report_id": report["id"]}, headers=headers).json()
    client.post(f"/api/chats/{chat['id']}/messages", json={"question": "Is my platelet count of 162 dangerous?"}, headers=headers)
    client.post("/agents/safety/validate", json=safety_payload(user["user_id"]), headers=headers)
    upload(client, headers, text="", filename="empty_nimal.pdf")
    entries = list(audit_logs.find({}, {"_id": 0}))
    dump = json.dumps(entries, default=str)
    # Random IDs and timestamps can contain a short needle such as "162" by chance.
    content = re.sub(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}|[0-9a-f]{24}|"
                     r"\d{4}-\d{2}-\d{2}[ T][\d:.+]+", "", dump, flags=re.IGNORECASE)
    stored_hash = users.find_one({"email": "auditee@example.com"})["password_hash"]
    ev.given(actions=["register", "login", "upload + analysis of a CBC/lipid report", "chat question", "safety validate",
                      "failed upload (empty file)"], report_text=REPORT_TEXT, question="Is my platelet count of 162 dangerous?")

    ev.check("Audit entries written", "> 0", len(entries), passed=len(entries) > 0)
    for label, needle in [("password", PASSWORD), ("password hash", stored_hash[:20]), ("email", "auditee@example.com"),
                          ("user name", "Kasun"), ("patient name", "Nimal"), ("file name", "nimal_perera"),
                          ("hemoglobin value", "11.2"), ("platelet value", "162"), ("test name", "Hemoglobin"),
                          ("question text", "dangerous"), ("draft/explanation text", "educational"), ("disclaimer", "medical advice")]:
        ev.check(f"No {label} in any entry", "absent", "present" if needle.lower() in content.lower() else "absent")
    fields = sorted({k for e in entries for k in e})
    ev.check("Top-level fields", ["action", "agent", "details", "log_id", "report_id", "status", "task_id", "timestamp", "user_id"], fields)
    ev.check("Authentication events (register/login) audited", "not required for this test", "none written",
             passed=True)
    ev.output("detail keys used", sorted({k for e in entries for k in e["details"]}))
    ev.output("sample entries", entries[:4])
    ev.note("No entry is written for register or login, so the audit trail cannot show failed-login bursts (see PRIV-15).")
    ev.verify()


def test_priv09_saved_reports_and_chats_authorization(evidence, client, storage):
    ev = evidence("PRIV-09", "Saved reports/chats endpoint authorization", "Every reports/chats query filtered by the token's user_id",
                  "Request is rejected or returns no data belonging to User B; access is scoped to the authenticated user only.")
    _, a = account(client, "a@example.com")
    _, b = account(client, "b@example.com")
    report = upload(client, b).json()
    chat = client.post("/api/chats", json={"report_id": report["id"]}, headers=b).json()
    rid, cid = report["id"], chat["id"]
    ev.given(attacker="User A (valid token)", owner="User B", report_id=rid, chat_id=cid)

    unknown = client.get("/api/reports/unknown-id", headers=a)
    attempts = {
        "list reports": client.get("/api/reports", headers=a),
        "list chats": client.get("/api/chats", headers=a),
        "list chats for B's report": client.get(f"/api/chats?report_id={rid}", headers=a),
        "read report": client.get(f"/api/reports/{rid}", headers=a),
        "download original file": client.get(f"/api/reports/{rid}/file", headers=a),
        "rename report": client.patch(f"/api/reports/{rid}", json={"name": "x"}, headers=a),
        "delete stored file": client.delete(f"/api/reports/{rid}/file", headers=a),
        "delete report": client.delete(f"/api/reports/{rid}", headers=a),
        "read chat": client.get(f"/api/chats/{cid}", headers=a),
        "rename chat": client.patch(f"/api/chats/{cid}", json={"title": "x"}, headers=a),
        "send message in B's chat": client.post(f"/api/chats/{cid}/messages", json={"question": "hi"}, headers=a),
        "open chat on B's report": client.post("/api/chats", json={"report_id": rid}, headers=a),
        "delete chat": client.delete(f"/api/chats/{cid}", headers=a),
    }
    for label in ("list reports", "list chats", "list chats for B's report"):
        ev.check(f"A: {label}", "200 []", f"{attempts[label].status_code} {attempts[label].text}")
    for label, response in list(attempts.items())[3:]:
        ev.check(f"A: {label}", "404 (same as unknown id)", f"{response.status_code} {response.json().get('detail')}",
                 passed=response.status_code == 404 == unknown.status_code)
    ev.check("B's data intact", (1, 1, True), (storage.reports.count_documents({}), storage.chats.count_documents({}),
                                              client.get(f"/api/reports/{rid}/file", headers=b).status_code == 200))
    ev.verify()


# --- PRIV-10 to PRIV-13 Configuration, errors and tokens -------------------------------------------


def test_priv10_cors_configuration(evidence, client):
    ev = evidence("PRIV-10", "CORS configuration", "CORSMiddleware in main.py",
                  "CORS allows only the expected frontend origin(s), not a wildcard with credentials enabled.")
    hostile = "https://attacker.example"
    ev.given(configured_origins=CORS_ORIGINS, allow_credentials=True, arbitrary_origin=hostile)

    pre = {"Access-Control-Request-Method": "GET", "Access-Control-Request-Headers": "authorization"}
    bad = client.options("/api/reports", headers={"Origin": hostile, **pre})
    simple = client.get("/health", headers={"Origin": hostile})
    good = client.options("/api/reports", headers={"Origin": CORS_ORIGINS[0], **pre})
    ev.check("No wildcard origin", False, "*" in CORS_ORIGINS)
    ev.check("Configured origins are local frontend dev servers", True,
             all(o.startswith(("http://localhost:", "http://127.0.0.1:")) for o in CORS_ORIGINS))
    ev.check("Arbitrary origin: preflight", "400, no Access-Control-Allow-Origin",
             f"{bad.status_code}, {bad.headers.get('access-control-allow-origin') or 'no Access-Control-Allow-Origin'}")
    ev.check("Arbitrary origin: simple request carries no CORS grant", None, simple.headers.get("access-control-allow-origin"))
    ev.check("Approved origin: credentials allowed for that origin only", (CORS_ORIGINS[0], "true"),
             (good.headers.get("access-control-allow-origin"), good.headers.get("access-control-allow-credentials")))
    ev.output("arbitrary-origin preflight", f"{bad.status_code} {bad.text}")
    ev.verify()


class _UnreachableCollection:
    def __getattr__(self, name):
        def fail(*args, **kwargs):
            raise ServerSelectionTimeoutError(
                "cluster0-shard-00-01.abcde.mongodb.net:27017: timed out (user lablens_app, "
                r"C:\Users\dev\lablens-ai\backend\database.py)")
        return fail


def test_priv11_internal_error_leakage(evidence, client, auth_headers):
    ev = evidence("PRIV-11", "Internal error information leakage", "Starlette's generic 500 handler; JSON-safe 422 handler",
                  "No stack trace, file path, environment variable name, or internal library detail is present in the response sent to the client.")
    ev.given(case_1="GET /api/reports while MongoDB is unreachable (ServerSelectionTimeoutError with host/user/path)",
             case_2="POST /explanation with value NaN (input the validators reject)",
             case_3="POST /auth/register with a 7-character password")

    app.dependency_overrides[get_reports_collection] = lambda: _UnreachableCollection()
    down = TestClient(app, raise_server_exceptions=False).get("/api/reports", headers=auth_headers("user-1"))
    ev.check("Database down: status/body", "500 Internal Server Error", f"{down.status_code} {down.text}")
    for needle in ("Traceback", "mongodb.net", "lablens_app", "database.py", "C:\\", "MONGODB_URI", "pymongo"):
        ev.absent(f"Database down: no '{needle}'", down.text, needle)
    nan = client.post("/explanation", content='{"task_id":"t","report_id":"r","user_id":"user-1","findings":[{"test":"Hb","value":NaN,"unit":"g/dL"}]}',
                      headers={**auth_headers("user-1"), "Content-Type": "application/json"})
    ev.check("NaN value: controlled 422 (not 500)", 422, nan.status_code)
    ev.absent("NaN value: no traceback", nan.text, "Traceback")
    short = register(client, "short@example.com", password="Abc123!")
    echoed = "Abc123!" in short.text
    ev.check("Short password: status", 422, short.status_code)
    ev.output("database-down response", down.text)
    ev.output("short-password 422 body", short.json())
    ev.note(f"Validation errors use FastAPI's default 422 body, which echoes the rejected input"
            f"{' - here the attempted password' if echoed else ''} back to the same client. It is not shown to anyone "
            "else, but proxies or browser logs that record response bodies would capture it.")
    ev.verify()


def test_priv12_expired_token(evidence, client, users):
    ev = evidence("PRIV-12", "Expired token handling", "exp claim verified by python-jose on every protected route",
                  '401 returned, with a generic message; no information about why the token failed beyond "invalid/expired".')
    user, _ = account(client, "expiry@example.com")
    token = create_access_token(user["user_id"], expires_delta=timedelta(seconds=2))
    ev.given(token_lifetime="2 seconds (create_access_token with a short expiry, standing in for JWT_EXPIRE_MINUTES)",
             wait="3 seconds", routes=["GET /auth/me", "GET /api/reports"])

    before = client.get("/auth/me", headers=bearer(token))
    time.sleep(3)
    after_me = client.get("/auth/me", headers=bearer(token))
    after_reports = client.get("/api/reports", headers=bearer(token))
    garbage = client.get("/auth/me", headers=bearer("garbage"))
    ev.check("Before expiry: accepted", 200, before.status_code)
    ev.check("After expiry: GET /auth/me", "401 Not authenticated", f"{after_me.status_code} {after_me.json()['detail']}")
    ev.check("After expiry: GET /api/reports", "401 Not authenticated", f"{after_reports.status_code} {after_reports.json()['detail']}")
    ev.check("Same body as a garbage token (reason not revealed)", garbage.text, after_me.text)
    ev.absent("No 'expired' wording", after_me.text, "expired")
    ev.verify()


def test_priv13_tampered_token(evidence, client, audit_logs):
    ev = evidence("PRIV-13", "Tampered token handling", "HS256 signature verification before any business logic",
                  "401 returned; the request is rejected before any business logic runs.")
    user, headers = account(client, "tamper@example.com")
    token = headers["Authorization"].split()[1]
    head, body, sig = token.split(".")
    flipped = sig[:10] + ("A" if sig[10] != "A" else "B") + sig[11:]
    tampered = f"{head}.{body}.{flipped}"
    ev.given(original_signature=sig[:16] + "...", tampered_signature=flipped[:16] + "...", changed="one character at position 10")

    responses = {
        "GET /auth/me": client.get("/auth/me", headers=bearer(tampered)),
        "POST /agents/safety/validate": client.post("/agents/safety/validate", json=safety_payload(user["user_id"]), headers=bearer(tampered)),
        "POST /api/reports": upload(client, bearer(tampered)),
    }
    for label, response in responses.items():
        ev.check(label, "401 Not authenticated", f"{response.status_code} {response.json().get('detail')}")
    ev.check("No audit entries (no business logic ran)", 0, audit_logs.count_documents({}))
    ev.check("Control: original token still works", 200, client.get("/auth/me", headers=headers).status_code)
    ev.verify()


# --- PRIV-14 / 15 Database network access and brute force ------------------------------------------


def test_priv14_database_network_access(evidence):
    ev = evidence("PRIV-14", "Database network access control", "MongoDB Atlas IP access list; secret handling in the repository",
                  "Document the current rule (0.0.0.0/0, allow from anywhere) and the compensating control in place (dedicated readWrite-only database user, not the project's admin account).")
    ev.given(repository_checks=["backend/.env ignored by git", "no MongoDB connection string in any commit", "database.py reads MONGODB_URI from the environment"],
             atlas_console="not accessible to the automated test")

    ignored = subprocess.run(["git", "check-ignore", "-q", "backend/.env"], cwd=BACKEND_DIR.parent).returncode == 0
    tracked = subprocess.run(["git", "ls-files", "backend/.env"], cwd=BACKEND_DIR.parent, capture_output=True, text=True).stdout.strip()
    revs = subprocess.run(["git", "rev-list", "--all"], cwd=BACKEND_DIR.parent, capture_output=True, text=True).stdout.split()
    hits = subprocess.run(["git", "grep", "-l", "-I", "-E", r"mongodb(\+srv)?://[^/\s\"']*:[^@\s\"']+@", *revs],
                          cwd=BACKEND_DIR.parent, capture_output=True, text=True).stdout.strip()
    source = (BACKEND_DIR / "database.py").read_text(encoding="utf-8")
    ev.check("backend/.env is git-ignored", True, ignored)
    ev.check("backend/.env not tracked", "", tracked)
    ev.check(f"No credentialed connection string in any of {len(revs)} commits", "", hits)
    ev.check("Connection string read from the environment", True, 'os.getenv("MONGODB_URI"' in source)
    ev.note("KF-01 (0.0.0.0/0 on the Atlas IP access list, compensated by the readWrite-only lablens_app user) was "
            "identified in the Atlas console during development. The Atlas Network Access page and database-user "
            "roles cannot be read from the code or an automated test, so that part needs the console screenshot.")
    ev.inconclusive("The Atlas IP access list and database-user roles are only visible in the Atlas console; the "
                    "repository-side controls above were verified.")


def test_priv15_brute_force_protection(evidence, client):
    ev = evidence("PRIV-15", "Brute-force protection on login", "Rate limiting / lockout on POST /auth/login",
                  "Document whether any rate limiting, lockout, or backoff is applied (expected: none currently implemented).")
    ev.weakness_confirmed("KF-02")
    register(client, "victim@example.com")
    attempts = 15
    ev.given(account="victim@example.com", wrong_password_attempts=attempts, then="correct password immediately")

    statuses, times = [], []
    start = time.perf_counter()
    for i in range(attempts):
        t = time.perf_counter()
        statuses.append(login(client, "victim@example.com", f"guess-{i:03d}").status_code)
        times.append((time.perf_counter() - t) * 1000)
    total = time.perf_counter() - start
    final = login(client, "victim@example.com")
    ev.check(f"{attempts} wrong guesses: statuses", [401] * attempts, statuses)
    ev.check("Any 429 / Retry-After returned", "none (no rate limit)", "none (no rate limit)" if 429 not in statuses else "429 seen")
    ev.check("Backoff: later attempts slower than early ones", "no backoff",
             "no backoff" if statistics.mean(times[-5:]) < 2 * statistics.mean(times[:5]) else "slower")
    ev.check("Correct password right after the burst", "200 (no lockout)", f"{final.status_code} (no lockout)"
             if final.status_code == 200 else f"{final.status_code} (locked out)")
    ev.output("per-attempt latency (ms)", [round(t) for t in times])
    ev.output("throughput", f"{attempts} attempts in {total:.2f}s = {attempts / total:.1f} guesses/s from one client")
    ev.note("Confirms KF-02 empirically. bcrypt's cost (~0.2-0.3 s per attempt) is the only brake, and failed logins are "
            "not written to the audit log (PRIV-08), so a guessing burst leaves no trace.")
    ev.verify()
