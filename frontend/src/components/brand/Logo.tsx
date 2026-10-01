import { Link } from 'react-router'

import { cn } from '@/lib/utils'

export function LogoMark({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 32 32" aria-hidden="true" className={cn('size-8 shrink-0', className)}>
      <rect width="32" height="32" rx="7" className="fill-primary" />
      <circle cx="14" cy="14" r="6.5" fill="none" className="stroke-primary-foreground" strokeWidth="2.4" />
      <path d="M18.8 18.8 24 24" className="stroke-primary-foreground" strokeWidth="2.6" strokeLinecap="round" />
      <path d="M11 14h6" className="stroke-primary-foreground" strokeWidth="1.8" strokeLinecap="round" />
    </svg>
  )
}

export function Logo({ to = '/', className }: { to?: string; className?: string }) {
  return (
    <Link
      to={to}
      className={cn('inline-flex items-center gap-2.5 rounded-md focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none', className)}
    >
      <LogoMark />
      <span className="font-serif text-[1.4rem] leading-none tracking-tight text-foreground">
        LabLens <em className="text-primary">AI</em>
      </span>
    </Link>
  )
}
