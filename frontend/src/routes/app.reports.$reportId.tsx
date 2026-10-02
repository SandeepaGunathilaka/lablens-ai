import { createFileRoute, Link, useNavigate } from "@tanstack/react-router";
import { useEffect, useState } from "react";
import { z } from "zod";
import { ArrowLeft, CircleAlert, Download, ExternalLink, FileImage, Info, Workflow } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { Sheet, SheetContent, SheetHeader, SheetTitle, SheetTrigger } from "@/components/ui/sheet";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Disclaimer, EmptyState, Loading, StatusBadge } from "@/components/lab/shared";
import { AgentTimeline } from "@/components/lab/AgentTimeline";
import { ReportChat } from "@/components/lab/ReportChat";
import { api, type ReportDetail as Report } from "@/lib/api";
import { DISCLAIMER, formatDate, reportTypeLabel, toStatus } from "@/lib/labels";
import { useReport } from "@/lib/store";

export const Route = createFileRoute("/app/reports/$reportId")({
  validateSearch: z.object({ chat: z.string().optional() }),
  head: () => ({
    meta: [
      { title: "Report — LabLens AI" },
      { name: "description", content: "Plain-language explanation of your lab results." },
      { property: "og:title", content: "Report — LabLens AI" },
      { property: "og:description", content: "Plain-language explanation of your lab results." },
    ],
  }),
  component: ReportDetail,
});

/** final_response is one paragraph block per test, in result order, followed by the disclaimer. */
function explanationSections(report: Report): string[][] | null {
  if (!report.final_response) return null;
  const blocks = report.final_response
    .split(/\n\s*\n/)
    .map((b) => b.trim())
    .filter((b) => b && b.replace(/\s+/g, " ") !== DISCLAIMER);
  if (blocks.length !== report.results.length) return null;
  return blocks.map((b) => b.split("\n").map((l) => l.trim()).filter(Boolean));
}

function OriginalFile({ report }: { report: Report }) {
  const [open, setOpen] = useState(false);
  const [url, setUrl] = useState<string>();
  const [error, setError] = useState("");

  useEffect(() => {
    if (!open || !report.has_file) return;
    let objectUrl: string | undefined;
    api.getReportFile(report.id)
      .then((blob) => { objectUrl = URL.createObjectURL(blob); setUrl(objectUrl); })
      .catch((e) => setError(e instanceof Error ? e.message : "Couldn't load the file."));
    return () => { if (objectUrl) URL.revokeObjectURL(objectUrl); setUrl(undefined); setError(""); };
  }, [open, report.id, report.has_file]);

  const isPdf = report.content_type === "application/pdf";
  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild><Button variant="outline" size="sm"><FileImage /> View original</Button></DialogTrigger>
      <DialogContent className="max-w-3xl">
        <DialogHeader><DialogTitle>{report.original_filename || "Original report"}</DialogTitle></DialogHeader>
        {!report.has_file ? (
          <p className="text-sm text-muted-foreground">The original file was deleted. Your results are still saved.</p>
        ) : error ? (
          <p className="text-sm text-muted-foreground">{error}</p>
        ) : !url ? <Loading /> : (
          <div className="space-y-3">
            {isPdf
              ? <iframe src={url} title="Original report" className="h-[70vh] w-full rounded-xl border" />
              : <img src={url} alt="Original report" className="max-h-[70vh] w-full rounded-xl border object-contain" />}
            <Button variant="outline" size="sm" asChild>
              <a href={url} download={report.original_filename}><Download /> Download</a>
            </Button>
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
}

function ReportDetail() {
  const { reportId } = Route.useParams();
  const { chat } = Route.useSearch();
  const navigate = useNavigate({ from: Route.fullPath });
  const { data: report, isLoading, error } = useReport(reportId);

  if (isLoading) return <Loading text="Loading report…" />;
  if (!report) {
    return <EmptyState icon={<CircleAlert />} title="Report not found" text={error?.message ?? "It may have been deleted."}
      action={<Button asChild><Link to="/app/reports">Back to My Reports</Link></Button>} />;
  }

  const sections = explanationSections(report);
  const links = new Map<string, { title: string; url: string; tests: string[] }>();
  for (const s of report.sources) {
    for (const l of s.links) {
      if (!l.url) continue;
      const existing = links.get(l.url);
      if (existing) existing.tests.push(s.test_name);
      else links.set(l.url, { title: l.title, url: l.url, tests: [s.test_name] });
    }
  }
  const missing = report.results.filter((r) => !report.sources.some((s) => s.test_name.toLowerCase() === r.test.toLowerCase()));

  return (
    <div className="mx-auto max-w-7xl">
      <Link to="/app/reports" className="mb-4 inline-flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground"><ArrowLeft className="h-4 w-4" /> My Reports</Link>
      <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
        <div className="space-y-6">
          <div className="rounded-2xl border bg-card p-5">
            <div className="flex flex-wrap items-start justify-between gap-3">
              <div>
                <span className="rounded-full bg-accent px-2.5 py-0.5 text-xs font-semibold text-accent-foreground">{reportTypeLabel(report.report_type)}</span>
                <h1 className="mt-2 text-2xl font-semibold tracking-tight">{report.name}</h1>
                <p className="text-sm text-muted-foreground">Uploaded {formatDate(report.created_at)}</p>
              </div>
              <div className="flex gap-2">
                <OriginalFile report={report} />
                <Sheet>
                  <SheetTrigger asChild><Button variant="outline" size="sm"><Workflow /> How this was generated</Button></SheetTrigger>
                  <SheetContent className="overflow-y-auto">
                    <SheetHeader><SheetTitle>How this was generated</SheetTitle></SheetHeader>
                    <AgentTimeline taskId={report.task_id} />
                  </SheetContent>
                </Sheet>
              </div>
            </div>

            {report.status === "fallback" && report.message && (
              <p className="mt-4 flex items-start gap-2 rounded-xl bg-warning-soft px-4 py-3 text-sm text-warning-foreground">
                <Info className="mt-0.5 h-4 w-4 shrink-0" /> {report.message}
              </p>
            )}

            <div className="mt-5 overflow-x-auto">
              <Table>
                <TableHeader><TableRow><TableHead>Test</TableHead><TableHead>Value</TableHead><TableHead>Unit</TableHead><TableHead>Reference range</TableHead><TableHead>Status</TableHead></TableRow></TableHeader>
                <TableBody>
                  {report.results.map((r, i) => (
                    <TableRow key={`${r.test}-${i}`}>
                      <TableCell className="font-medium">
                        {r.test}
                        {r.needs_verification && <span className="ml-2 inline-flex rounded-full border border-dashed border-muted-foreground/40 px-2 py-0.5 text-[10px] font-medium text-muted-foreground" title={r.warning ?? "Please check this value against your original report."}>Please verify</span>}
                      </TableCell>
                      <TableCell>{r.value}</TableCell>
                      <TableCell className="text-muted-foreground">{r.unit ?? "—"}</TableCell>
                      <TableCell className="text-muted-foreground">{r.reference_range ?? "—"}</TableCell>
                      <TableCell><StatusBadge status={toStatus(r.status)} /></TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          </div>

          {report.final_response && (
            <div className="rounded-2xl border bg-card p-5">
              <h2 className="font-semibold">Explanations</h2>
              {sections ? (
                <Accordion type="multiple" className="mt-2">
                  {report.results.map((r, i) => (
                    <AccordionItem key={`${r.test}-${i}`} value={String(i)}>
                      <AccordionTrigger><span className="flex items-center gap-2">{r.test} <StatusBadge status={toStatus(r.status)} /></span></AccordionTrigger>
                      <AccordionContent className="space-y-2 text-sm">
                        {(sections[i] ?? []).map((line, j) => <p key={j} className={j === 0 ? "font-medium" : undefined}>{line}</p>)}
                      </AccordionContent>
                    </AccordionItem>
                  ))}
                </Accordion>
              ) : (
                <p className="mt-2 whitespace-pre-line text-sm">{report.final_response}</p>
              )}
            </div>
          )}

          <div className="rounded-2xl border bg-card p-5">
            <h2 className="font-semibold">Sources</h2>
            {links.size === 0 && <p className="mt-3 text-sm text-muted-foreground">No trusted sources were found for this report.</p>}
            <ul className="mt-3 space-y-2">
              {[...links.values()].map((s) => (
                <li key={s.url}>
                  <a href={s.url} target="_blank" rel="noreferrer" className="flex items-center justify-between gap-3 rounded-xl p-3 hover:bg-muted">
                    <div><p className="text-sm font-medium">{s.title}</p><p className="text-xs text-muted-foreground">{new URL(s.url).hostname.replace(/^www\./, "")} · {s.tests.join(", ")}</p></div>
                    <ExternalLink className="h-4 w-4 text-primary" />
                  </a>
                </li>
              ))}
            </ul>
            {missing.length > 0 && (
              <p className="mt-3 text-xs text-muted-foreground">
                No approved information was found for: {missing.map((r) => r.test).join(", ")}.
              </p>
            )}
          </div>
          <Disclaimer />
        </div>

        <div className="lg:sticky lg:top-6 lg:self-start">
          <ReportChat report={report} activeId={chat} onSelect={(id) => navigate({ search: { chat: id } })} />
        </div>
      </div>
    </div>
  );
}
