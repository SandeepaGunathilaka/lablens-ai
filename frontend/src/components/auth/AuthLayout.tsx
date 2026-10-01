import { ArrowLeft } from 'lucide-react'
import type { ReactNode } from 'react'
import { Link } from 'react-router'

import { Logo } from '@/components/brand/Logo'

/** Split screen: editorial reference panel on the left, the form on the right. */
export function AuthLayout({
  aside,
  switchPrompt,
  children,
}: {
  aside: ReactNode
  switchPrompt: ReactNode
  children: ReactNode
}) {
  return (
    <div className="grid min-h-dvh lg:grid-cols-[minmax(0,0.9fr)_minmax(0,1.1fr)]">
      <aside className="hidden flex-col border-r bg-paper px-10 py-10 lg:flex xl:px-14">
        <div className="flex items-center justify-between">
          <Logo />
          <Link
            to="/"
            className="inline-flex items-center gap-1.5 font-label text-[13px] text-muted-foreground hover:text-foreground"
          >
            <ArrowLeft className="size-3.5" aria-hidden="true" /> Back to overview
          </Link>
        </div>
        <div className="my-auto py-12">{aside}</div>
      </aside>

      <main className="flex flex-col px-4 py-6 sm:px-10 sm:py-10">
        <div className="flex items-center justify-between gap-4 lg:justify-end">
          <Logo className="lg:hidden" />
          <p className="text-sm text-muted-foreground">{switchPrompt}</p>
        </div>
        <div className="animate-rise mx-auto my-auto w-full max-w-[400px] py-12">{children}</div>
      </main>
    </div>
  )
}
