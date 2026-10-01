import { ShieldCheck } from 'lucide-react'
import type { ReactNode } from 'react'

import { cn } from '@/lib/utils'

export const DISCLAIMER =
  'LabLens AI provides educational information only and is not a substitute for professional medical advice, diagnosis or treatment.'

/** The bordered "educational, not diagnostic" note used across the app. */
export function DisclaimerNote({ className, children }: { className?: string; children?: ReactNode }) {
  return (
    <aside className={cn('rounded-lg border bg-paper/70 p-4', className)}>
      <p className="eyebrow flex items-center gap-1.5 text-primary">
        <ShieldCheck className="size-3.5" aria-hidden="true" />
        Educational, not diagnostic
      </p>
      <p className="mt-1.5 text-sm leading-relaxed text-muted-foreground">
        {children ?? 'LabLens never diagnoses conditions or recommends treatment. Discuss your results with a clinician.'}
      </p>
    </aside>
  )
}
