import { useCallback, useEffect, useMemo, useState, type ReactNode } from 'react'

import { apiFetch } from '@/lib/api'
import { AuthContext, type AuthStatus, type User } from './auth-context'
import { clearSession, readSession, writeSession, type StoredSession } from './session'

type TokenResponse = { access_token: string; token_type: string; expires_in: number }

export function AuthProvider({ children }: { children: ReactNode }) {
  const [session, setSession] = useState<StoredSession | null>(readSession)
  const [user, setUser] = useState<User | null>(null)
  const [status, setStatus] = useState<AuthStatus>(session ? 'loading' : 'anonymous')

  const logout = useCallback(() => {
    clearSession()
    setSession(null)
    setUser(null)
    setStatus('anonymous')
  }, [])

  // Whenever we hold a token, confirm it with the backend and load the profile.
  useEffect(() => {
    if (!session) return
    let cancelled = false
    apiFetch<User>('/auth/me', { token: session.token })
      .then((profile) => {
        if (cancelled) return
        setUser(profile)
        setStatus('authenticated')
      })
      .catch(() => {
        // Expired, revoked, or backend unreachable: fall back to logged out.
        if (!cancelled) logout()
      })
    return () => {
      cancelled = true
    }
  }, [session, logout])

  // Log out the moment the token expires rather than on the next failed request.
  useEffect(() => {
    if (!session) return
    const timer = window.setTimeout(logout, Math.max(0, session.expiresAt - Date.now()))
    return () => window.clearTimeout(timer)
  }, [session, logout])

  const login = useCallback(async (email: string, password: string) => {
    const result = await apiFetch<TokenResponse>('/auth/login', {
      method: 'POST',
      json: { email, password },
    })
    setStatus('loading')
    setSession(writeSession(result.access_token, result.expires_in))
  }, [])

  const register = useCallback(
    async (name: string, email: string, password: string) => {
      await apiFetch<User>('/auth/register', { method: 'POST', json: { name, email, password } })
      await login(email, password)
    },
    [login],
  )

  const value = useMemo(
    () => ({ status, user, token: session?.token ?? null, login, register, logout }),
    [status, user, session, login, register, logout],
  )

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}
