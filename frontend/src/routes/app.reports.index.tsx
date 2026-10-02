import { createFileRoute, Link } from "@tanstack/react-router";
import { useState } from "react";
import { FileText, ImageOff, MoreVertical, Pencil, Search, Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "@/components/ui/dropdown-menu";
import { ConfirmDelete, EmptyState, Loading, RenameDialog, StatusBadge } from "@/components/lab/shared";
import { formatDate, reportTypeLabel, toStatus } from "@/lib/labels";
import { useActions, useReports } from "@/lib/store";
import { cn } from "@/lib/utils";

export const Route = createFileRoute("/app/reports/")({
  head: () => ({
    meta: [
      { title: "My Reports — LabLens AI" },
      { name: "description", content: "All of your uploaded lab reports." },
      { property: "og:title", content: "My Reports — LabLens AI" },
      { property: "og:description", content: "All of your uploaded lab reports." },
    ],
  }),
  component: ReportsPage,
});

const FILTERS = ["All", "CBC", "Lipid Profile", "Outside range"] as const;

function ReportsPage() {
  const { data: reports = [], isLoading, error } = useReports();
  const { renameReport, deleteReport, deleteImage } = useActions();
  const [q, setQ] = useState("");
  const [f, setF] = useState<(typeof FILTERS)[number]>("All");

  const list = reports.filter((r) => {
    if (!r.name.toLowerCase().includes(q.toLowerCase())) return false;
    if (f === "CBC" || f === "Lipid Profile") return reportTypeLabel(r.report_type) === f;
    if (f === "Outside range") return r.result_statuses.some((s) => s === "low" || s === "high");
    return true;
  });

  return (
    <div className="mx-auto max-w-5xl">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-3xl font-semibold tracking-tight">My Reports</h1>
          <p className="mt-1 text-muted-foreground">{reports.length} report{reports.length !== 1 && "s"}</p>
        </div>
        <Button asChild><Link to="/app/upload">Upload report</Link></Button>
      </div>

      <div className="mt-6 flex flex-col gap-3 sm:flex-row sm:items-center">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
          <Input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Search reports" className="bg-card pl-9" />
        </div>
        <div className="flex flex-wrap gap-2">
          {FILTERS.map((x) => (
            <button key={x} onClick={() => setF(x)}
              className={cn("rounded-full border px-3 py-1.5 text-sm font-medium transition-colors", f === x ? "border-primary bg-primary text-primary-foreground" : "bg-card hover:bg-muted")}>
              {x}
            </button>
          ))}
        </div>
      </div>

      <div className="mt-6">
        {isLoading ? <Loading /> : error ? (
          <EmptyState icon={<FileText />} title="Couldn't load your reports" text={error.message} />
        ) : list.length === 0 ? (
          <EmptyState icon={<FileText />} title={reports.length ? "No matching reports" : "No reports yet"}
            text={reports.length ? "Try a different search or filter." : "Upload a lab report to see it here."}
            action={!reports.length && <Button asChild><Link to="/app/upload">Upload report</Link></Button>} />
        ) : (
          <div className="grid gap-5 sm:grid-cols-2 lg:grid-cols-3">
            {list.map((r) => (
              <div key={r.id} className="flex flex-col rounded-2xl border bg-card p-5">
                <div className="flex items-start justify-between">
                  <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-accent text-primary"><FileText className="h-5 w-5" /></div>
                  <DropdownMenu>
                    <DropdownMenuTrigger asChild><Button variant="ghost" size="icon" aria-label="Report actions"><MoreVertical /></Button></DropdownMenuTrigger>
                    <DropdownMenuContent align="end">
                      <RenameDialog title="Rename report" value={r.name} onSave={(v) => renameReport(r.id, v)}
                        trigger={<DropdownMenuItem onSelect={(e) => e.preventDefault()}><Pencil /> Rename</DropdownMenuItem>} />
                      <ConfirmDelete title="Delete original file?" description="The uploaded file will be removed. Your results and chats stay."
                        onConfirm={() => deleteImage(r.id)}
                        trigger={<DropdownMenuItem disabled={!r.has_file} onSelect={(e) => e.preventDefault()}><ImageOff /> Delete original file</DropdownMenuItem>} />
                      <ConfirmDelete title="Delete this report?" description="This removes the report, its uploaded file and all of its chats. This can't be undone."
                        onConfirm={() => deleteReport(r.id)}
                        trigger={<DropdownMenuItem className="text-destructive" onSelect={(e) => e.preventDefault()}><Trash2 /> Delete report</DropdownMenuItem>} />
                    </DropdownMenuContent>
                  </DropdownMenu>
                </div>
                <h3 className="mt-4 font-semibold">{r.name}</h3>
                <p className="text-sm text-muted-foreground">{reportTypeLabel(r.report_type)} · {formatDate(r.created_at)}</p>
                <div className="mt-3 flex flex-wrap gap-1.5">
                  {r.result_statuses.map((s, i) => <StatusBadge key={i} status={toStatus(s)} />)}
                </div>
                <p className="mt-3 text-xs text-muted-foreground">
                  {r.test_count} test{r.test_count !== 1 && "s"} · {r.chat_count} chat{r.chat_count !== 1 && "s"}
                  {!r.has_file && " · Original file deleted"}
                </p>
                <Button variant="outline" className="mt-5" asChild><Link to="/app/reports/$reportId" params={{ reportId: r.id }}>Open</Link></Button>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
