"""PRIV-04 and PRIV-05: token storage and logout, observed in the real frontend.

The real React frontend (Vite dev server) is driven in Microsoft Edge through Playwright
against the real FastAPI backend served by uvicorn, with MongoDB replaced by mongomock so
no shared database is touched. Screenshots show the page with an evidence overlay that
reproduces what DevTools > Application > Storage shows for the origin.
"""

import base64
import json
import os
import re
import subprocess
import threading
import time
from pathlib import Path

import httpx
import mongomock
import pytest
import uvicorn

import coordinator
from api.reports import ensure_report_indexes
from database import (
    get_audit_logs_collection,
    get_chats_collection,
    get_report_files_collection,
    get_reports_collection,
    get_users_collection,
)
from evidence_support import BACKEND_DIR
from logging_service import ensure_audit_log_indexes
from main import CORS_ORIGINS, app
from security.auth import ensure_user_indexes

API_PORT = 8765
SHOTS = Path(__file__).parent / "evidence" / "browser"
EMAIL, PASSWORD = "kasun.browser@example.com", "Str0ng-Passw0rd!"

OVERLAY = """(text) => {
  document.getElementById('__evidence')?.remove();
  const d = document.createElement('div');
  d.id = '__evidence';
  d.style.cssText = 'position:fixed;right:16px;bottom:16px;width:600px;background:#1e1e1e;color:#d4d4d4;' +
    'font:12px Consolas,monospace;padding:12px 14px;border-radius:8px;z-index:2147483647;' +
    'box-shadow:0 6px 24px rgba(0,0,0,.45);white-space:pre-wrap;word-break:break-all;border:2px solid #f0b400';
  d.textContent = text;
  document.body.appendChild(d);
}"""

STORAGE = "() => ({local: {...localStorage}, session: {...sessionStorage}, cookie: document.cookie})"


def _frontend_origin() -> tuple[str, int]:
    for origin in CORS_ORIGINS:
        match = re.fullmatch(r"http://(localhost|127\.0\.0\.1):(\d+)", origin)
        if match:
            return origin, int(match.group(2))
    pytest.skip("No local frontend origin in CORS_ORIGINS")


def _wait_for(url: str, timeout: float) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if httpx.get(url, timeout=3).status_code < 500:
                return
        except httpx.HTTPError:
            pass
        time.sleep(1)
    raise TimeoutError(f"{url} did not come up within {timeout:.0f}s")


def _provider(collection):
    return lambda: collection


@pytest.fixture(scope="module")
def live_stack():
    db = mongomock.MongoClient().db
    ensure_user_indexes(db.users)
    ensure_audit_log_indexes(db.audit_logs)
    ensure_report_indexes(db.reports, db.report_files, db.chats)
    overrides = {get_users_collection: db.users, get_audit_logs_collection: db.audit_logs,
                 get_reports_collection: db.reports, get_report_files_collection: db.report_files,
                 get_chats_collection: db.chats}
    for dependency, collection in overrides.items():
        app.dependency_overrides[dependency] = _provider(collection)
    coordinator._default_explanation_service.cache_clear()

    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=API_PORT, lifespan="off", log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    _wait_for(f"http://127.0.0.1:{API_PORT}/health", 30)

    origin, port = _frontend_origin()
    host = origin.split("//")[1].split(":")[0]
    SHOTS.mkdir(parents=True, exist_ok=True)
    log = open(SHOTS / "vite.log", "w", encoding="utf-8")
    vite = subprocess.Popen(
        ["npx.cmd", "vite", "dev", "--host", host, "--port", str(port), "--strictPort"],
        cwd=BACKEND_DIR.parent / "frontend", stdout=log, stderr=subprocess.STDOUT,
        env={**os.environ, "VITE_API_BASE_URL": f"http://127.0.0.1:{API_PORT}", "BROWSER": "none"},
    )
    try:
        _wait_for(origin, 180)
        yield {"origin": origin, "api": f"http://127.0.0.1:{API_PORT}", "db": db}
    finally:
        subprocess.run(["taskkill", "/PID", str(vite.pid), "/T", "/F"], capture_output=True)
        log.close()
        server.should_exit = True
        thread.join(timeout=10)
        for dependency in overrides:
            app.dependency_overrides.pop(dependency, None)


@pytest.fixture(scope="module")
def browser_run(live_stack):
    """Register and log in through the UI, inspect storage, log out, then replay the token."""
    from playwright.sync_api import sync_playwright

    origin, api = live_stack["origin"], live_stack["api"]
    run: dict = {"origin": origin}
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge")
        context = browser.new_context(viewport={"width": 1366, "height": 860})
        page = context.new_page()
        page.goto(f"{origin}/register", wait_until="networkidle", timeout=120_000)
        page.fill("#name", "Kasun Edirisinghe")
        page.fill("#email", EMAIL)
        page.fill("#pw", PASSWORD)
        page.click("button[type=submit]")
        page.wait_for_url(f"{origin}/app", timeout=60_000)
        page.get_by_role("button", name="Log out").last.wait_for(state="visible", timeout=60_000)
        page.wait_for_load_state("networkidle")
        page.wait_for_timeout(1500)

        run["after_login"] = page.evaluate(STORAGE)
        run["cookies_after_login"] = context.cookies()
        run["injected_script_read"] = page.evaluate(
            "() => { const s = document.createElement('script');"
            " s.textContent = \"window.__stolen = localStorage.getItem('lablens_token')\";"
            " document.head.appendChild(s); return window.__stolen || null; }")
        token = run["after_login"]["local"].get("lablens_token")
        run["token"] = token
        cookie_names = [c["name"] for c in run["cookies_after_login"]] or ["(none)"]
        page.evaluate(OVERLAY, (
            "EVIDENCE OVERLAY (injected by the test) - storage for " + origin + "\n\n"
            "Local Storage:\n  lablens_token = " + (token[:46] + "..." if token else "(absent)") + "\n"
            "Session Storage: " + (json.dumps(run["after_login"]["session"]) if run["after_login"]["session"] else "(empty)") + "\n"
            "Cookies: " + ", ".join(cookie_names) + "\n\n"
            "Script injected into the page read the token: " + ("YES" if run["injected_script_read"] == token else "no")))
        page.screenshot(path=str(SHOTS / "priv04_token_in_localstorage.png"))

        page.get_by_role("button", name="Log out").last.click()
        page.wait_for_function("() => !localStorage.getItem('lablens_token') && !location.pathname.startsWith('/app')",
                               timeout=30_000)
        page.wait_for_load_state("networkidle")
        run["after_logout"] = page.evaluate(STORAGE)
        run["url_after_logout"] = page.url

        replay = {path: httpx.get(f"{api}{path}", headers={"Authorization": f"Bearer {token}"}, timeout=10)
                  for path in ("/auth/me", "/api/reports")}
        run["replay"] = {path: {"status": r.status_code, "body": r.json()} for path, r in replay.items()}
        page.evaluate(OVERLAY, (
            "EVIDENCE OVERLAY (injected by the test) - after clicking 'Log out'\n\n"
            "Local Storage: lablens_token = " + str(run["after_logout"]["local"].get("lablens_token")) + "\n\n"
            "Replaying the captured token against the API:\n"
            + "\n".join(f"  GET {path} -> HTTP {v['status']}" + (f"  {json.dumps(v['body'])}" if path == '/auth/me' else
                        f"  ({len(v['body'])} reports)") for path, v in run["replay"].items())))
        page.screenshot(path=str(SHOTS / "priv05_token_replayed_after_logout.png"))
        browser.close()
    run["user"] = live_stack["db"].users.find_one({"email": EMAIL}, {"_id": 0, "user_id": 1})
    return run


def _claims(token: str) -> dict:
    payload = token.split(".")[1]
    return json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))


def test_priv04_jwt_storage_location(evidence, browser_run):
    ev = evidence("PRIV-04", "JWT storage location and XSS exposure", "Token handling in frontend/src/lib/api.ts",
                  "Document the actual storage mechanism (localStorage/sessionStorage/cookie) and assess its exposure to a hypothetical XSS vector.")
    ev.weakness_confirmed("PF-01")
    token = browser_run["token"]
    ev.given(frontend=f"real React frontend at {browser_run['origin']} (Vite dev server), Microsoft Edge",
             steps=["register Kasun Edirisinghe / " + EMAIL, "automatic login", "inspect storage", "inject a script into the page"])

    claims = _claims(token) if token else {}
    ev.check("Token stored in localStorage under 'lablens_token'", True, bool(token))
    ev.check("Stored value is the backend-issued JWT for this user", browser_run["user"]["user_id"], claims.get("sub"))
    ev.check("sessionStorage", {}, browser_run["after_login"]["session"])
    ev.check("Token in any cookie (HttpOnly protection)", "none", ", ".join(c["name"] for c in browser_run["cookies_after_login"]
                                                                     if token and token in c["value"]) or "none")
    ev.check("A script injected into the page can read the token", "readable (XSS exposure)",
             "readable (XSS exposure)" if browser_run["injected_script_read"] == token else "not readable")
    ev.output("JWT claims (decoded, not verified)", {k: claims.get(k) for k in ("sub", "iat", "exp")})
    ev.output("storage after login", {**browser_run["after_login"],
                                      "local": {k: v[:30] + "..." for k, v in browser_run["after_login"]["local"].items()}})
    ev.image(SHOTS / "priv04_token_in_localstorage.png", "Real frontend after login, with storage overlay")
    ev.note("localStorage is readable by any JavaScript running on the origin, so a single XSS flaw (or a compromised "
            "npm dependency) would let an attacker copy the bearer token and use it from anywhere until it expires "
            "(60 minutes). An HttpOnly, Secure, SameSite cookie would keep it out of reach of page scripts.")
    ev.verify()


def test_priv05_token_valid_after_logout(evidence, browser_run):
    ev = evidence("PRIV-05", "Token validity after logout", "Client-side logout (setToken(null)); stateless JWT verification",
                  "Document whether the token is still accepted (expected: yes, since JWTs are stateless) and treat this as a documented limitation, not a surprise.")
    ev.weakness_confirmed("PF-02")
    token = browser_run["token"]
    ev.given(steps=["log in through the UI", "capture the token from localStorage", "click 'Log out'",
                    "replay the captured token: GET /auth/me and GET /api/reports"])

    replay = browser_run["replay"]
    claims = _claims(token)
    ev.check("Token removed from the browser on logout", None, browser_run["after_logout"]["local"].get("lablens_token"))
    ev.check("UI left the authenticated area", "a public page (outside /app)", browser_run["url_after_logout"],
             passed="/app" not in browser_run["url_after_logout"])
    ev.check("Replayed token: GET /auth/me", "200 (still accepted)", f"{replay['/auth/me']['status']} "
             f"({'still accepted' if replay['/auth/me']['status'] == 200 else 'rejected'})")
    ev.check("Replayed token: identifies the logged-out user", EMAIL, replay["/auth/me"]["body"].get("email"))
    ev.check("Replayed token: GET /api/reports", 200, replay["/api/reports"]["status"])
    ev.check("Remaining lifetime of the replayed token", "about 60 minutes", f"{(claims['exp'] - time.time()) / 60:.0f} minutes",
             passed=(claims["exp"] - time.time()) > 50 * 60)
    ev.output("replay responses", replay)
    ev.image(SHOTS / "priv05_token_replayed_after_logout.png", "After logout: storage cleared, captured token still accepted by the API")
    ev.note("Logout only deletes the browser's copy. The backend has no logout endpoint, token denylist or token "
            "version, so a copied token keeps working until 'exp'.")
    ev.verify()
