import { Link } from 'react-router'

import { Logo } from '@/components/brand/Logo'
import { Button } from '@/components/ui/button'

export default function NotFound() {
  return (
    <main className="flex min-h-dvh flex-col items-center justify-center px-4 text-center">
      <Logo />
      <p className="eyebrow mt-12 text-primary">Error 404</p>
      <h1 className="mt-3 text-5xl font-normal tracking-tight">This page isn’t on the report.</h1>
      <p className="mt-4 max-w-md text-muted-foreground">The link may be old, or the address mistyped.</p>
      <Button asChild className="mt-8">
        <Link to="/">Go to the home page</Link>
      </Button>
    </main>
  )
}
