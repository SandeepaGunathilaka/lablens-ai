import { ArrowRight, Check, Lock, X } from 'lucide-react'
import { useState, type FormEvent, type ReactNode } from 'react'
import { Link } from 'react-router'

import { useAuth } from '@/auth/auth-context'
import { AuthLayout } from '@/components/auth/AuthLayout'
import { FormAlert } from '@/components/auth/FormAlert'
import { PasswordInput } from '@/components/auth/PasswordInput'
import { passwordChecks } from '@/components/auth/password-rules'
import { DisclaimerNote } from '@/components/brand/Disclaimer'
import { Button } from '@/components/ui/button'
import { Checkbox } from '@/components/ui/checkbox'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { ApiError } from '@/lib/api'
import { cn } from '@/lib/utils'

const STEPS = [
  ['Upload a CBC or Lipid Profile report', 'PDF, PNG or JPG, straight from your lab.'],
  ['We extract each value and look up curated sources', 'OCR reads the report; retrieval finds passages for each test.'],
  ['Read a plain-language explanation', 'Only explanations that pass the safety check are shown to you.'],
]

function RegisterAside() {
  return (
    <div className="max-w-md">
      <h2 className="text-[2.6rem] leading-[1.1] font-normal tracking-tight">What happens after you sign up</h2>
      <ol className="mt-10 divide-y rounded-xl border bg-card">
        {STEPS.map(([title, body], i) => (
          <li key={title} className="flex gap-4 p-5">
            <span className="grid size-7 shrink-0 place-items-center rounded-full border font-mono text-xs text-primary">
              {i + 1}
            </span>
            <div>
              <p className="font-medium">{title}</p>
              <p className="mt-1 text-sm leading-relaxed text-muted-foreground">{body}</p>
            </div>
          </li>
        ))}
      </ol>
      <DisclaimerNote className="mt-8" />
    </div>
  )
}

type FieldErrors = Partial<Record<'name' | 'email' | 'password' | 'confirm' | 'consent', string>>

export default function Register() {
  const { register } = useAuth()

  const [name, setName] = useState('')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [confirm, setConfirm] = useState('')
  const [consent, setConsent] = useState(false)
  const [fieldErrors, setFieldErrors] = useState<FieldErrors>({})
  const [error, setError] = useState<ReactNode>('')
  const [submitting, setSubmitting] = useState(false)

  const checks = passwordChecks(password)
  const metCount = checks.filter((c) => c.met).length
  const mismatch = confirm.length > 0 && confirm !== password

  function validate(): FieldErrors {
    const errors: FieldErrors = {}
    if (!name.trim()) errors.name = 'Enter your name.'
    else if (name.trim().length > 100) errors.name = 'Name must be 100 characters or fewer.'
    if (!/^\S+@\S+\.\S+$/.test(email.trim())) errors.email = 'Enter a valid email address.'
    if (metCount < checks.length) errors.password = 'Password does not meet the requirements below.'
    if (confirm !== password) errors.confirm = "Passwords don't match."
    if (!consent) errors.consent = 'Please confirm you understand LabLens is educational.'
    return errors
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    setError('')
    const errors = validate()
    setFieldErrors(errors)
    if (Object.keys(errors).length > 0) {
      const first = Object.keys(errors)[0]
      document.getElementById(first === 'consent' ? 'consent' : first)?.focus()
      return
    }

    setSubmitting(true)
    try {
      // Registers, then logs in; GuestOnly redirects to the dashboard.
      await register(name.trim(), email.trim(), password)
    } catch (err) {
      if (err instanceof ApiError && err.status === 409) {
        setError(
          <>
            An account with this email already exists.{' '}
            <Link to="/login" className="font-medium underline">
              Log in instead
            </Link>
            .
          </>,
        )
      } else {
        setError(err instanceof Error ? err.message : 'Something went wrong. Please try again.')
      }
      setSubmitting(false)
    }
  }

  return (
    <AuthLayout
      aside={<RegisterAside />}
      switchPrompt={
        <>
          Already have an account?{' '}
          <Link to="/login" className="font-medium text-primary hover:underline">
            Log in
          </Link>
        </>
      }
    >
      <p className="eyebrow text-primary">Get started</p>
      <h1 className="mt-2 text-4xl font-normal tracking-tight">Create your account</h1>
      <p className="mt-2 text-muted-foreground">Free, private, and only takes a minute.</p>

      <form onSubmit={handleSubmit} noValidate className="mt-8 space-y-5">
        {error && <FormAlert>{error}</FormAlert>}

        <Field id="name" label="Full name" error={fieldErrors.name}>
          <Input
            id="name"
            autoComplete="name"
            maxLength={100}
            value={name}
            onChange={(e) => setName(e.target.value)}
            aria-invalid={!!fieldErrors.name}
            aria-describedby={fieldErrors.name ? 'name-error' : undefined}
            className="h-11 bg-card"
          />
        </Field>

        <Field id="email" label="Email" error={fieldErrors.email}>
          <Input
            id="email"
            type="email"
            autoComplete="email"
            placeholder="you@example.com"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            aria-invalid={!!fieldErrors.email}
            aria-describedby={fieldErrors.email ? 'email-error' : undefined}
            className="h-11 bg-card"
          />
        </Field>

        <Field id="password" label="Password" error={fieldErrors.password}>
          <PasswordInput
            id="password"
            autoComplete="new-password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            aria-invalid={!!fieldErrors.password}
            aria-describedby="password-rules"
            className="h-11 bg-card"
          />
          <div className="mt-2.5 grid grid-cols-3 gap-1.5" aria-hidden="true">
            {checks.map((c, i) => (
              <span
                key={c.label}
                className={cn('h-1 rounded-full bg-border transition-colors', i < metCount && 'bg-primary')}
              />
            ))}
          </div>
          <ul id="password-rules" className="mt-3 space-y-1.5 rounded-lg border bg-paper/60 px-3.5 py-3">
            {checks.map((c) => (
              <li
                key={c.label}
                className={cn('flex items-center gap-2 text-[13px]', c.met ? 'text-foreground' : 'text-muted-foreground')}
              >
                {c.met ? (
                  <Check className="size-3.5 text-primary" aria-hidden="true" />
                ) : (
                  <X className="size-3.5" aria-hidden="true" />
                )}
                {c.label}
                <span className="sr-only">{c.met ? '(met)' : '(not met)'}</span>
              </li>
            ))}
          </ul>
        </Field>

        <Field id="confirm" label="Confirm password" error={fieldErrors.confirm ?? (mismatch ? "Passwords don't match." : undefined)}>
          <PasswordInput
            id="confirm"
            autoComplete="new-password"
            value={confirm}
            onChange={(e) => setConfirm(e.target.value)}
            aria-invalid={mismatch || !!fieldErrors.confirm}
            aria-describedby={mismatch || fieldErrors.confirm ? 'confirm-error' : undefined}
            className="h-11 bg-card"
          />
        </Field>

        <div>
          <div className="flex items-start gap-3">
            <Checkbox
              id="consent"
              checked={consent}
              onCheckedChange={(v) => setConsent(v === true)}
              aria-invalid={!!fieldErrors.consent}
              aria-describedby={fieldErrors.consent ? 'consent-error' : undefined}
              className="mt-0.5"
            />
            <Label htmlFor="consent" className="text-sm leading-relaxed font-normal text-muted-foreground">
              I understand LabLens provides educational information, not medical advice.
            </Label>
          </div>
          {fieldErrors.consent && (
            <p id="consent-error" className="mt-1.5 text-[13px] text-destructive">
              {fieldErrors.consent}
            </p>
          )}
        </div>

        <Button type="submit" size="lg" className="h-11 w-full" disabled={submitting}>
          {submitting ? 'Creating account…' : (
            <>
              Create account <ArrowRight />
            </>
          )}
        </Button>
      </form>

      <p className="mt-10 flex items-start gap-2.5 border-t pt-5 text-xs leading-relaxed text-muted-foreground">
        <Lock className="mt-0.5 size-3.5 shrink-0 text-primary" aria-hidden="true" />
        Your password is hashed with bcrypt and never stored in plain text.
      </p>
    </AuthLayout>
  )
}

function Field({
  id,
  label,
  error,
  children,
}: {
  id: string
  label: string
  error?: string
  children: ReactNode
}) {
  return (
    <div className="space-y-2">
      <Label htmlFor={id} className="eyebrow text-foreground">
        {label}
      </Label>
      {children}
      {error && (
        <p id={`${id}-error`} className="text-[13px] text-destructive">
          {error}
        </p>
      )}
    </div>
  )
}
