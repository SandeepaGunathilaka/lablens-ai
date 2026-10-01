import type { ReactNode } from 'react'

import { cn } from '@/lib/utils'

export type ReportRow = {
  test: string
  result: string
  unit: string
  reference: string
  flag?: 'high' | 'low'
}

/**
 * A typeset sample lab sheet. Purely illustrative: the values are made up and
 * shown so visitors can see what LabLens reads.
 */
export function ReportSheet({
  panel,
  rows,
  highlight,
  className,
  children,
}: {
  panel: string
  rows: ReportRow[]
  highlight?: string
  className?: string
  children?: ReactNode
}) {
  return (
    <figure className={cn('rounded-xl border bg-card p-5 shadow-[0_18px_40px_-24px_rgb(27_31_30/0.25)] sm:p-6', className)}>
      <figcaption className="flex items-start justify-between gap-4 border-b pb-4">
        <div>
          <p className="eyebrow text-muted-foreground">Sample report</p>
          <p className="mt-1 font-serif text-xl leading-tight">{panel}</p>
        </div>
        <p className="text-right font-mono text-[11px] leading-5 text-muted-foreground">
          Specimen LL-0427
          <br />
          Not real patient data
        </p>
      </figcaption>

      <table className="mt-2 w-full text-left text-sm">
        <thead>
          <tr className="font-label text-[10.5px] tracking-[0.1em] text-muted-foreground uppercase">
            <th scope="col" className="py-2.5 font-medium">Test</th>
            <th scope="col" className="py-2.5 text-right font-medium">Result</th>
            <th scope="col" className="hidden py-2.5 pl-4 font-medium sm:table-cell">Unit</th>
            <th scope="col" className="py-2.5 pl-4 font-medium">Reference</th>
            <th scope="col" className="py-2.5 text-right font-medium">
              <span className="sr-only">Status</span>
            </th>
          </tr>
        </thead>
        <tbody className="tabular">
          {rows.map((row) => (
            <tr
              key={row.test}
              className={cn('border-t', row.test === highlight && 'bg-flag-soft/45')}
            >
              <th scope="row" className="py-3 pr-2 font-normal">
                <span className="flex items-center gap-2">
                  {row.flag && <span className="size-1.5 rounded-full bg-flag" aria-hidden="true" />}
                  {row.test}
                </span>
              </th>
              <td className={cn('py-3 text-right font-label text-[15px] font-semibold', row.flag && 'text-flag')}>
                {row.result}
              </td>
              <td className="hidden py-3 pl-4 font-mono text-xs text-muted-foreground sm:table-cell">{row.unit}</td>
              <td className="py-3 pl-4 font-mono text-xs text-muted-foreground">{row.reference}</td>
              <td className="py-3 pl-2 text-right">
                {row.flag ? (
                  <span className="rounded-full border border-flag/30 bg-flag-soft px-2 py-0.5 font-label text-[10px] font-semibold tracking-wider text-flag uppercase">
                    {row.flag}
                  </span>
                ) : (
                  <span className="font-label text-[11px] text-muted-foreground">In range</span>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      {children}
    </figure>
  )
}
