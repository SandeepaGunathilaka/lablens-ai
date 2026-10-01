import { Construction } from 'lucide-react'
import { Link } from 'react-router'

import { Button } from '@/components/ui/button'

/** Placeholder for app sections that other team members are building. */
export default function ComingSoon({ title, description }: { title: string; description: string }) {
  return (
    <section className="animate-rise flex flex-col items-center rounded-2xl border border-dashed border-input bg-card px-6 py-16 text-center">
      <span className="grid size-12 place-items-center rounded-full bg-secondary text-muted-foreground">
        <Construction className="size-5" aria-hidden="true" />
      </span>
      <h1 className="mt-5 text-3xl font-normal">{title}</h1>
      <p className="mt-2 max-w-md text-muted-foreground">{description}</p>
      <Button asChild variant="outline" className="mt-7">
        <Link to="/app">Back to overview</Link>
      </Button>
    </section>
  )
}
