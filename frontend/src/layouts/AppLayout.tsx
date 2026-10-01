import { FileText, LayoutGrid, LogOut, Menu, MessageSquareText, Upload, X } from 'lucide-react'
import { useEffect, useState } from 'react'
import { NavLink, Outlet, useLocation } from 'react-router'

import { useAuth } from '@/auth/auth-context'
import { DISCLAIMER } from '@/components/brand/Disclaimer'
import { Logo } from '@/components/brand/Logo'
import { Avatar, AvatarFallback } from '@/components/ui/avatar'
import { Button } from '@/components/ui/button'
import { apiFetch } from '@/lib/api'
import { cn } from '@/lib/utils'

const APP_NAV = [
  { to: '/app', label: 'Overview', icon: LayoutGrid, end: true },
  { to: '/app/upload', label: 'Upload report', icon: Upload },
  { to: '/app/reports', label: 'My reports', icon: FileText },
  { to: '/app/explanations', label: 'Explanations', icon: MessageSquareText },
]

function initials(name: string) {
  return name
    .split(/\s+/)
    .filter(Boolean)
    .slice(0, 2)
    .map((part) => part[0]?.toUpperCase())
    .join('')
}

function BackendStatus() {
  const [state, setState] = useState<'checking' | 'up' | 'down'>('checking')

  useEffect(() => {
    let cancelled = false
    apiFetch<{ status: string }>('/health')
      .then(() => !cancelled && setState('up'))
      .catch(() => !cancelled && setState('down'))
    return () => {
      cancelled = true
    }
  }, [])

  const label = { checking: 'Checking backend…', up: 'Backend connected', down: 'Backend unreachable' }[state]
  return (
    <p
      role="status"
      className="inline-flex items-center gap-2 rounded-full border bg-card px-3 py-1 font-label text-xs font-medium text-muted-foreground"
    >
      <span
        aria-hidden="true"
        className={cn(
          'size-2 rounded-full',
          state === 'up' && 'bg-ok',
          state === 'down' && 'bg-flag',
          state === 'checking' && 'animate-pulse bg-muted-foreground/50',
        )}
      />
      {label}
    </p>
  )
}

function SidebarContent({ onNavigate }: { onNavigate?: () => void }) {
  const { user, logout } = useAuth()

  return (
    <div className="flex h-full flex-col">
      <div className="px-5 pt-6 pb-8">
        <Logo to="/app" />
      </div>

      <nav aria-label="App" className="flex-1 px-3">
        <ul className="space-y-1">
          {APP_NAV.map(({ to, label, icon: Icon, end }) => (
            <li key={to}>
              <NavLink
                to={to}
                end={end}
                onClick={onNavigate}
                className={({ isActive }) =>
                  cn(
                    'relative flex items-center gap-3 rounded-md px-3 py-2.5 font-label text-[13.5px] font-medium transition-colors',
                    isActive
                      ? 'bg-accent text-accent-foreground before:absolute before:inset-y-2 before:left-0 before:w-0.5 before:rounded-full before:bg-primary'
                      : 'text-muted-foreground hover:bg-secondary hover:text-foreground',
                  )
                }
              >
                <Icon className="size-4" aria-hidden="true" />
                {label}
              </NavLink>
            </li>
          ))}
        </ul>
      </nav>

      <div className="space-y-4 p-4">
        <p className="rounded-lg border bg-card p-3 text-xs leading-relaxed text-muted-foreground">
          <span className="eyebrow mb-1 block text-primary">Educational, not diagnostic</span>
          Discuss your results with a clinician.
        </p>
        {user && (
          <div className="flex items-center gap-3 border-t pt-4">
            <Avatar className="size-9">
              <AvatarFallback className="bg-accent font-label text-xs font-semibold text-primary">
                {initials(user.name)}
              </AvatarFallback>
            </Avatar>
            <div className="min-w-0 flex-1">
              <p className="truncate text-sm font-medium">{user.name}</p>
              <p className="truncate text-xs text-muted-foreground">{user.email}</p>
            </div>
            <Button variant="ghost" size="icon-sm" onClick={logout} aria-label="Log out" title="Log out">
              <LogOut />
            </Button>
          </div>
        )}
      </div>
    </div>
  )
}

export default function AppLayout() {
  const [menuOpen, setMenuOpen] = useState(false)
  const { pathname } = useLocation()
  const current = [...APP_NAV].reverse().find((item) => (item.end ? pathname === item.to : pathname.startsWith(item.to)))

  // Close the mobile menu with Escape.
  useEffect(() => {
    if (!menuOpen) return
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && setMenuOpen(false)
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [menuOpen])

  return (
    <div className="min-h-dvh lg:grid lg:grid-cols-[248px_minmax(0,1fr)]">
      <div className="hidden border-r bg-paper/70 lg:block">
        <aside className="sticky top-0 h-dvh">
          <SidebarContent />
        </aside>
      </div>

      {menuOpen && (
        <div className="fixed inset-0 z-40 lg:hidden" role="dialog" aria-modal="true" aria-label="Navigation">
          <button
            type="button"
            aria-label="Close menu"
            className="absolute inset-0 bg-foreground/30"
            onClick={() => setMenuOpen(false)}
          />
          <div className="animate-rise absolute inset-y-0 left-0 w-72 max-w-[85vw] border-r bg-background">
            <SidebarContent onNavigate={() => setMenuOpen(false)} />
          </div>
        </div>
      )}

      <div className="flex min-h-dvh flex-col">
        <header className="sticky top-0 z-30 flex h-16 items-center gap-3 border-b bg-background/90 px-4 backdrop-blur-sm sm:px-8">
          <Button
            variant="ghost"
            size="icon"
            className="lg:hidden"
            onClick={() => setMenuOpen((v) => !v)}
            aria-label={menuOpen ? 'Close menu' : 'Open menu'}
            aria-expanded={menuOpen}
          >
            {menuOpen ? <X /> : <Menu />}
          </Button>
          <p className="flex min-w-0 items-baseline gap-3">
            <span className="eyebrow hidden text-muted-foreground sm:inline">Dashboard</span>
            <span className="hidden text-border sm:inline" aria-hidden="true">/</span>
            <span className="truncate font-serif text-xl">{current?.label ?? 'LabLens'}</span>
          </p>
          <div className="ml-auto">
            <BackendStatus />
          </div>
        </header>

        <main id="main" className="flex-1 px-4 py-8 sm:px-8 lg:py-10">
          <div className="mx-auto max-w-5xl">
            <Outlet />
          </div>
        </main>

        <footer className="border-t px-4 py-5 text-xs leading-relaxed text-muted-foreground sm:px-8">
          <p className="mx-auto max-w-5xl">{DISCLAIMER}</p>
        </footer>
      </div>
    </div>
  )
}
