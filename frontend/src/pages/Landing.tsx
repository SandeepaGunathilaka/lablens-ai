import { ArrowRight, Check, KeyRound, Lock, ShieldCheck, UserRound } from 'lucide-react'
import { Link } from 'react-router'

import { useAuth } from '@/auth/auth-context'
import { DISCLAIMER } from '@/components/brand/Disclaimer'
import { Logo } from '@/components/brand/Logo'
import { ReportSheet } from '@/components/landing/ReportSheet'
import { CBC_SAMPLE } from '@/components/landing/samples'
import { Button } from '@/components/ui/button'

const NAV = [
  { href: '#how-it-works', label: 'How it works' },
  { href: '#safety', label: 'Safety' },
  { href: '#tests', label: 'Supported tests' },
]

const AGENTS = [
  {
    n: '01',
    name: 'Document Agent',
    role: 'Extraction',
    body: 'Reads your PDF or photo with OCR, then picks out each test name, value, unit and reference range.',
  },
  {
    n: '02',
    name: 'Retrieval Agent',
    role: 'Grounding',
    body: 'Searches a curated medical knowledge base, combining keyword and semantic search, for passages about each test.',
  },
  {
    n: '03',
    name: 'Explanation Agent',
    role: 'Plain language',
    body: 'Writes a short explanation using only the passages it was given, with the source attached.',
  },
  {
    n: '04',
    name: 'Safety Agent',
    role: 'Guardrail',
    body: 'Rule-based checks block diagnoses, treatment advice and unsupported claims before anything reaches you.',
    guard: true,
  },
]

const TESTS = [
  {
    name: 'Complete Blood Count',
    tag: 'CBC',
    body: 'The cells in your blood: how much oxygen they can carry, and the cells involved in immunity and clotting.',
    items: ['Hemoglobin', 'RBC', 'WBC', 'Platelets', 'Hematocrit', 'MCV'],
  },
  {
    name: 'Lipid Profile',
    tag: 'Lipids',
    body: 'The fats carried in your blood, reported as cholesterol fractions and triglycerides.',
    items: ['Total Cholesterol', 'LDL', 'HDL', 'Triglycerides'],
  },
]

function Header() {
  const { status } = useAuth()
  const signedIn = status === 'authenticated'

  return (
    <header className="sticky top-0 z-30 border-b bg-background/90 backdrop-blur-sm">
      <div className="mx-auto flex h-16 max-w-6xl items-center justify-between gap-6 px-4 sm:px-6">
        <Logo />
        <nav aria-label="Main" className="hidden items-center gap-8 md:flex">
          {NAV.map((item) => (
            <a
              key={item.href}
              href={item.href}
              className="font-label text-[13px] font-medium text-muted-foreground transition-colors hover:text-foreground"
            >
              {item.label}
            </a>
          ))}
        </nav>
        <div className="flex items-center gap-2">
          {signedIn ? (
            <Button asChild>
              <Link to="/app">
                Open dashboard <ArrowRight />
              </Link>
            </Button>
          ) : (
            <>
              <Button asChild variant="ghost" className="hidden sm:inline-flex">
                <Link to="/login">Log in</Link>
              </Button>
              <Button asChild>
                <Link to="/register">Create account</Link>
              </Button>
            </>
          )}
        </div>
      </div>
    </header>
  )
}

function Hero() {
  return (
    <section className="mx-auto grid max-w-6xl items-center gap-14 px-4 pt-14 pb-20 sm:px-6 lg:grid-cols-[1.05fr_1fr] lg:gap-16 lg:pt-20 lg:pb-28">
      <div className="animate-rise">
        <p className="eyebrow inline-flex items-center gap-2 rounded-full border bg-card px-3 py-1 text-primary">
          <span className="size-1.5 rounded-full bg-primary" aria-hidden="true" />
          For CBC &amp; Lipid Profile reports
        </p>
        <h1 className="mt-6 text-[2.75rem] leading-[1.02] font-normal tracking-[-0.02em] sm:text-6xl lg:text-[4.25rem]">
          Understand your lab report, <em className="text-primary">line by line.</em>
        </h1>
        <p className="mt-6 max-w-xl text-lg leading-relaxed text-muted-foreground">
          Upload a report and LabLens explains each value in plain language, grounded in curated medical
          sources and checked by a safety agent before you see it.
        </p>
        <div className="mt-9 flex flex-wrap gap-3">
          <Button asChild size="lg" className="h-11 px-5">
            <Link to="/register">
              Create a free account <ArrowRight />
            </Link>
          </Button>
          <Button asChild size="lg" variant="outline" className="h-11 px-5">
            <a href="#how-it-works">See how it works</a>
          </Button>
        </div>
        <p className="mt-8 flex items-center gap-2 border-t pt-5 text-sm text-muted-foreground">
          <ShieldCheck className="size-4 text-primary" aria-hidden="true" />
          Educational, not diagnostic. Always discuss results with a clinician.
        </p>
      </div>

      <div className="animate-rise relative [animation-delay:120ms]">
        <ReportSheet panel="Complete Blood Count" rows={CBC_SAMPLE} highlight="Hemoglobin">
          <div className="relative mt-5 rounded-lg border border-dashed border-primary/35 bg-accent/60 p-4">
            <div className="flex items-center justify-between gap-3">
              <p className="eyebrow text-primary">Plain-language explanation</p>
              <p className="flex items-center gap-1 font-label text-[11px] font-medium text-ok">
                <Check className="size-3.5" aria-hidden="true" /> Safety checked
              </p>
            </div>
            <p className="mt-2 text-[15px] leading-relaxed">
              <strong className="font-semibold">Hemoglobin</strong> is the protein in red blood cells that carries
              oxygen. This result is below the reference range printed on your report. Only a clinician can say
              what that means for you.
            </p>
            <p className="mt-3 border-t border-primary/15 pt-2.5 font-mono text-[11px] text-muted-foreground">
              Source: LabLens curated knowledge base · Hemoglobin
            </p>
          </div>
        </ReportSheet>
      </div>
    </section>
  )
}

function HowItWorks() {
  return (
    <section id="how-it-works" className="scroll-mt-20 border-t">
      <div className="mx-auto max-w-6xl px-4 py-20 sm:px-6 lg:py-24">
        <div className="max-w-2xl">
          <p className="eyebrow text-primary">How it works</p>
          <h2 className="mt-3 text-4xl font-normal tracking-tight sm:text-5xl">Four agents. One careful answer.</h2>
          <p className="mt-4 text-lg leading-relaxed text-muted-foreground">
            Each agent has one job, and the explanation only moves forward when the previous step has done its
            part.
          </p>
        </div>

        <ol className="relative mt-14 grid gap-5 md:grid-cols-2 lg:grid-cols-4">
          <span aria-hidden="true" className="absolute top-[38px] right-10 left-10 hidden h-px bg-border lg:block" />
          {AGENTS.map((agent) => (
            <li key={agent.n} className="relative flex flex-col rounded-xl border bg-card p-6">
              <div className="flex items-center justify-between">
                <span
                  className={`grid size-8 place-items-center rounded-full border bg-card font-mono text-xs ${agent.guard ? 'border-flag/40 text-flag' : 'text-primary'}`}
                >
                  {agent.n}
                </span>
                <span className="eyebrow text-muted-foreground">{agent.role}</span>
              </div>
              <h3 className="mt-6 text-2xl font-normal">{agent.name}</h3>
              <p className="mt-2 text-[15px] leading-relaxed text-muted-foreground">{agent.body}</p>
            </li>
          ))}
        </ol>
      </div>
    </section>
  )
}

function Safety() {
  const boundaries = [
    ['No diagnoses or treatment advice', 'The safety agent rejects any output that names a condition or suggests a treatment.'],
    ['Every explanation cites its source', 'Explanations are written only from retrieved passages, and the source is shown.'],
    ['Flags are facts, not verdicts', 'A value outside the reference range is marked as such, never labelled as a disease.'],
  ]
  const privacy = [
    [KeyRound, 'Passwords hashed with bcrypt', 'Salted and deliberately slow to crack. The plain password is never stored.'],
    [Lock, 'Signed sessions that expire', 'You get a signed JWT that stops working after a set time, and you are logged out.'],
    [UserRound, 'Your own private dashboard', 'The app is behind sign-in, and every request carries your personal session token.'],
  ] as const

  return (
    <section id="safety" className="scroll-mt-20 border-t bg-paper/60">
      <div className="mx-auto grid max-w-6xl gap-6 px-4 py-20 sm:px-6 lg:grid-cols-2 lg:py-24">
        <div className="rounded-xl border bg-card p-7 sm:p-9">
          <p className="eyebrow text-primary">Clinical boundaries</p>
          <h2 className="mt-3 text-3xl font-normal">Built to inform, not to diagnose.</h2>
          <ul className="mt-7 space-y-5">
            {boundaries.map(([title, body]) => (
              <li key={title} className="flex gap-3">
                <span className="mt-0.5 grid size-5 shrink-0 place-items-center rounded-full bg-accent text-primary">
                  <Check className="size-3" aria-hidden="true" />
                </span>
                <div>
                  <p className="font-medium">{title}</p>
                  <p className="mt-0.5 text-sm leading-relaxed text-muted-foreground">{body}</p>
                </div>
              </li>
            ))}
          </ul>
        </div>
        <div className="rounded-xl border bg-card p-7 sm:p-9">
          <p className="eyebrow text-primary">Privacy</p>
          <h2 className="mt-3 text-3xl font-normal">Your account stays yours.</h2>
          <ul className="mt-7 space-y-5">
            {privacy.map(([Icon, title, body]) => (
              <li key={title} className="flex gap-3">
                <Icon className="mt-0.5 size-5 shrink-0 text-primary" aria-hidden="true" />
                <div>
                  <p className="font-medium">{title}</p>
                  <p className="mt-0.5 text-sm leading-relaxed text-muted-foreground">{body}</p>
                </div>
              </li>
            ))}
          </ul>
        </div>
      </div>
    </section>
  )
}

function SupportedTests() {
  return (
    <section id="tests" className="scroll-mt-20 border-t">
      <div className="mx-auto max-w-6xl px-4 py-20 sm:px-6 lg:py-24">
        <p className="eyebrow text-primary">Supported tests</p>
        <h2 className="mt-3 text-4xl font-normal tracking-tight">Two panels, read carefully.</h2>
        <p className="mt-4 max-w-2xl text-lg text-muted-foreground">
          LabLens deliberately covers a small scope so each explanation can be checked properly.
        </p>
        <div className="mt-12 grid gap-5 md:grid-cols-2">
          {TESTS.map((panel) => (
            <article key={panel.name} className="rounded-xl border bg-card p-7">
              <div className="flex items-start justify-between gap-4">
                <h3 className="text-2xl font-normal">{panel.name}</h3>
                <span className="eyebrow rounded border px-2 py-1 text-muted-foreground">{panel.tag}</span>
              </div>
              <p className="mt-3 text-[15px] leading-relaxed text-muted-foreground">{panel.body}</p>
              <ul className="mt-6 flex flex-wrap gap-2" aria-label={`${panel.name} tests`}>
                {panel.items.map((item) => (
                  <li key={item} className="rounded-md border bg-paper/60 px-2.5 py-1 font-label text-xs font-medium">
                    {item}
                  </li>
                ))}
              </ul>
            </article>
          ))}
        </div>
      </div>
    </section>
  )
}

function ClosingCta() {
  return (
    <section className="border-t">
      <div className="mx-auto max-w-6xl px-4 py-20 sm:px-6">
        <div className="flex flex-col items-start justify-between gap-8 rounded-2xl bg-primary px-8 py-12 text-primary-foreground sm:px-12 md:flex-row md:items-center">
          <div>
            <h2 className="text-3xl font-normal sm:text-4xl">Read your results with more confidence.</h2>
            <p className="mt-3 max-w-lg text-primary-foreground/75">
              Free to use, and it takes a minute to set up.
            </p>
          </div>
          <Button asChild size="lg" variant="secondary" className="h-11 shrink-0 bg-primary-foreground px-5 text-primary hover:bg-primary-foreground/90">
            <Link to="/register">
              Create a free account <ArrowRight />
            </Link>
          </Button>
        </div>
      </div>
    </section>
  )
}

function Footer() {
  return (
    <footer className="border-t bg-paper/60">
      <div className="mx-auto max-w-6xl px-4 py-12 sm:px-6">
        <div className="flex flex-col justify-between gap-8 md:flex-row">
          <div className="max-w-sm">
            <Logo />
            <p className="mt-4 text-sm leading-relaxed text-muted-foreground">
              A student project for IT3041 Information Retrieval &amp; Web Analytics, SLIIT.
            </p>
          </div>
          <nav aria-label="Footer" className="flex gap-8 text-sm">
            <ul className="space-y-2">
              {NAV.map((item) => (
                <li key={item.href}>
                  <a href={item.href} className="text-muted-foreground hover:text-foreground">
                    {item.label}
                  </a>
                </li>
              ))}
            </ul>
            <ul className="space-y-2">
              <li><Link to="/login" className="text-muted-foreground hover:text-foreground">Log in</Link></li>
              <li><Link to="/register" className="text-muted-foreground hover:text-foreground">Create account</Link></li>
            </ul>
          </nav>
        </div>
        <p className="mt-10 border-t pt-6 text-xs leading-relaxed text-muted-foreground">
          {DISCLAIMER} © {new Date().getFullYear()} LabLens AI.
        </p>
      </div>
    </footer>
  )
}

export default function Landing() {
  return (
    <div className="min-h-dvh">
      <a href="#main" className="sr-only focus:not-sr-only focus:fixed focus:top-3 focus:left-3 focus:z-50 focus:rounded-md focus:bg-card focus:px-3 focus:py-2">
        Skip to content
      </a>
      <Header />
      <main id="main">
        <Hero />
        <HowItWorks />
        <Safety />
        <SupportedTests />
        <ClosingCta />
      </main>
      <Footer />
    </div>
  )
}
