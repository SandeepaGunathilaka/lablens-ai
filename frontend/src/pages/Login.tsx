import { ArrowRight, Lock, ShieldCheck } from 'lucide-react'
import { useState, type FormEvent } from 'react'
import { Link } from 'react-router'

import { useAuth } from '@/auth/auth-context'
import { AuthLayout } from '@/components/auth/AuthLayout'
import { FormAlert } from '@/components/auth/FormAlert'
import { PasswordInput } from '@/components/auth/PasswordInput'
import { ReportSheet } from '@/components/landing/ReportSheet'
import { LIPID_SAMPLE } from '@/components/landing/samples'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { ApiError } from '@/lib/api'

function LoginAside() {
  return (
    <div className="max-w-md">
      <p className="font-serif text-[2.6rem] leading-[1.1] tracking-tight italic">
        “Every value, explained with its source.”
      </p>
      <ReportSheet panel="Lipid Profile" rows={LIPID_SAMPLE} className="mt-10" />
      <p className="mt-10 flex items-center gap-2 text-sm text-muted-foreground">
        <ShieldCheck className="size-4 text-primary" aria-hidden="true" />
        Educational, not diagnostic. Always discuss lab values with a clinician.
      </p>
    </div>
  )
}

export default function Login() {
  const { login } = useAuth()

  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [submitting, setSubmitting] = useState(false)

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    setError('')
    setSubmitting(true)
    try {
      // On success the GuestOnly route redirects to the dashboard.
      await login(email.trim(), password)
    } catch (err) {
      setError(
        err instanceof ApiError && err.status === 401
          ? 'Invalid email or password.'
          : err instanceof Error
            ? err.message
            : 'Something went wrong. Please try again.',
      )
      setSubmitting(false)
    }
  }

  return (
    <AuthLayout
      aside={<LoginAside />}
      switchPrompt={
        <>
          New to LabLens?{' '}
          <Link to="/register" className="font-medium text-primary hover:underline">
            Create an account
          </Link>
        </>
      }
    >
      <p className="eyebrow text-primary">Welcome back</p>
      <h1 className="mt-2 text-4xl font-normal tracking-tight">Log in to LabLens</h1>
      <p className="mt-2 text-muted-foreground">Pick up where you left off with your saved reports.</p>

      <form onSubmit={handleSubmit} className="mt-8 space-y-5">
        {error && <FormAlert>{error}</FormAlert>}

        <div className="space-y-2">
          <Label htmlFor="email" className="eyebrow text-foreground">Email</Label>
          <Input
            id="email"
            type="email"
            autoComplete="email"
            placeholder="you@example.com"
            required
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            className="h-11 bg-card"
          />
        </div>

        <div className="space-y-2">
          <Label htmlFor="password" className="eyebrow text-foreground">Password</Label>
          <PasswordInput
            id="password"
            autoComplete="current-password"
            required
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            className="h-11 bg-card"
          />
        </div>

        <Button type="submit" size="lg" className="h-11 w-full" disabled={submitting}>
          {submitting ? 'Logging in…' : (
            <>
              Log in <ArrowRight />
            </>
          )}
        </Button>
      </form>

      <p className="mt-10 flex items-start gap-2.5 border-t pt-5 text-xs leading-relaxed text-muted-foreground">
        <Lock className="mt-0.5 size-3.5 shrink-0 text-primary" aria-hidden="true" />
        Passwords are hashed with bcrypt. Sessions are signed, expire automatically, and end when you close this tab.
      </p>
    </AuthLayout>
  )
}
