import { Navigate, Outlet, useLocation } from 'react-router'

import { useAuth } from './auth-context'

function FullPageSpinner() {
  return (
    <div className="grid min-h-dvh place-items-center" role="status" aria-live="polite">
      <div className="flex items-center gap-3 text-sm text-muted-foreground">
        <span className="size-4 animate-spin rounded-full border-2 border-primary/25 border-t-primary" />
        Checking your session…
      </div>
    </div>
  )
}

/** Pages under this route need a logged-in user; others are sent to /login. */
export function RequireAuth() {
  const { status } = useAuth()
  const location = useLocation()

  if (status === 'loading') return <FullPageSpinner />
  if (status === 'anonymous') {
    return <Navigate to="/login" replace state={{ from: location.pathname }} />
  }
  return <Outlet />
}

/** Login and register make no sense once logged in; go to the app instead. */
export function GuestOnly() {
  const { status } = useAuth()
  const location = useLocation()
  const from = (location.state as { from?: string } | null)?.from

  if (status === 'loading') return <FullPageSpinner />
  if (status === 'authenticated') return <Navigate to={from ?? '/app'} replace />
  return <Outlet />
}
