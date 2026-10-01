import { ArrowRight, FileUp, Lock } from 'lucide-react'
import { Link } from 'react-router'

import { useAuth } from '@/auth/auth-context'
import { Button } from '@/components/ui/button'

const PIPELINE = [
  ['01', 'Document Agent', 'reads the values'],
  ['02', 'Retrieval Agent', 'finds curated sources'],
  ['03', 'Explanation Agent', 'writes plain language'],
  ['04', 'Safety Agent', 'blocks diagnoses'],
]

function greeting(hour: number) {
  if (hour < 12) return 'Good morning'
  if (hour < 18) return 'Good afternoon'
  return 'Good evening'
}

export default function Overview() {
  const { user } = useAuth()
  const firstName = user?.name.split(/\s+/)[0]

  return (
    <div className="animate-rise">
      <h1 className="text-4xl font-normal tracking-tight sm:text-5xl">
        {greeting(new Date().getHours())}
        {firstName ? `, ${firstName}` : ''}.
      </h1>
      <p className="mt-3 max-w-xl text-lg text-muted-foreground">
        Upload a CBC or Lipid Profile report to get a plain-language explanation.
      </p>

      <section
        aria-labelledby="empty-title"
        className="mt-10 flex flex-col items-center rounded-2xl border border-dashed border-input bg-card px-6 py-14 text-center"
      >
        <span className="grid size-14 place-items-center rounded-full bg-accent text-primary">
          <FileUp className="size-6" aria-hidden="true" />
        </span>
        <h2 id="empty-title" className="mt-5 text-2xl font-normal">
          No reports yet
        </h2>
        <p className="mt-2 max-w-sm text-muted-foreground">
          Add a lab report and LabLens will explain each value, with its source.
        </p>
        <Button asChild size="lg" className="mt-7 h-11 px-5">
          <Link to="/app/upload">
            Upload your first report <ArrowRight />
          </Link>
        </Button>
        <p className="mt-4 font-label text-xs text-muted-foreground">PDF, PNG or JPG</p>
      </section>

      <div className="mt-8 grid gap-5 md:grid-cols-3">
        <article className="rounded-xl border bg-card p-6">
          <p className="eyebrow text-primary">Pipeline</p>
          <h2 className="mt-2 text-xl font-normal">How LabLens reads your report</h2>
          <ol className="mt-4 space-y-2.5">
            {PIPELINE.map(([n, name, what]) => (
              <li key={n} className="flex gap-3 text-sm">
                <span className="font-mono text-xs leading-5 text-primary">{n}</span>
                <span>
                  <span className="font-medium">{name}</span>{' '}
                  <span className="text-muted-foreground">{what}</span>
                </span>
              </li>
            ))}
          </ol>
        </article>

        <article className="rounded-xl border bg-card p-6">
          <p className="eyebrow text-primary">Scope</p>
          <h2 className="mt-2 text-xl font-normal">Supported tests</h2>
          <p className="mt-4 text-sm font-medium">Complete Blood Count</p>
          <p className="mt-1 text-sm text-muted-foreground">Hemoglobin, RBC, WBC, Platelets, Hematocrit, MCV</p>
          <p className="mt-4 text-sm font-medium">Lipid Profile</p>
          <p className="mt-1 text-sm text-muted-foreground">Total Cholesterol, LDL, HDL, Triglycerides</p>
        </article>

        <article className="rounded-xl border bg-card p-6">
          <p className="eyebrow text-primary">Privacy</p>
          <h2 className="mt-2 text-xl font-normal">Your session</h2>
          <ul className="mt-4 space-y-3 text-sm text-muted-foreground">
            <li className="flex gap-2.5">
              <Lock className="mt-0.5 size-4 shrink-0 text-primary" aria-hidden="true" />
              Your password is stored only as a bcrypt hash.
            </li>
            <li className="flex gap-2.5">
              <Lock className="mt-0.5 size-4 shrink-0 text-primary" aria-hidden="true" />
              Your session expires automatically, and closing this tab logs you out.
            </li>
          </ul>
        </article>
      </div>
    </div>
  )
}
