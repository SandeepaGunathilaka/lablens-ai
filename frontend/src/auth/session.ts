// The JWT lives in sessionStorage: it survives a page reload but is dropped when
// the tab closes, and it is never sent automatically the way a cookie would be.
// The backend also expires it (JWT_EXPIRE_MINUTES), and we log out at that time.
const STORAGE_KEY = 'lablens.session'

export type StoredSession = { token: string; expiresAt: number }

export function readSession(): StoredSession | null {
  try {
    const raw = sessionStorage.getItem(STORAGE_KEY)
    if (!raw) return null
    const session = JSON.parse(raw) as StoredSession
    if (typeof session.token !== 'string' || typeof session.expiresAt !== 'number') return null
    if (session.expiresAt <= Date.now()) {
      clearSession()
      return null
    }
    return session
  } catch {
    return null
  }
}

export function writeSession(token: string, expiresInSeconds: number): StoredSession {
  const session = { token, expiresAt: Date.now() + expiresInSeconds * 1000 }
  try {
    sessionStorage.setItem(STORAGE_KEY, JSON.stringify(session))
  } catch {
    // Storage can be blocked (private mode); the session then lasts until reload.
  }
  return session
}

export function clearSession() {
  try {
    sessionStorage.removeItem(STORAGE_KEY)
  } catch {
    // Nothing stored, nothing to clear.
  }
}
