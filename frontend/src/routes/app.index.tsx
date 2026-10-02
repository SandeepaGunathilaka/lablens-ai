import { createFileRoute, Link } from "@tanstack/react-router";
import { ArrowRight, FileText, MessagesSquare, UploadCloud } from "lucide-react";
import { Button } from "@/components/ui/button";
import { EmptyState, Loading, StatusBadge } from "@/components/lab/shared";
import { formatDate, reportTypeLabel } from "@/lib/labels";
import { useAuth, useChats, useReports } from "@/lib/store";

export const Route = createFileRoute("/app/")({
  head: () => ({
    meta: [
      { title: "Dashboard — LabLens AI" },
      { name: "description", content: "Your recent lab reports and chats." },
      { property: "og:title", content: "Dashboard — LabLens AI" },
      { property: "og:description", content: "Your recent lab reports and chats." },
    ],
  }),
  component: Dashboard,
});

function Dashboard() {
  const { user } = useAuth();
  const { data: reports = [], isLoading: loadingReports } = useReports();
  const { data: chats = [], isLoading: loadingChats } = useChats();
  return (
    <div className="mx-auto max-w-5xl space-y-8">
      <div>
        <h1 className="text-3xl font-semibold tracking-tight">Hello{user ? `, ${user.name.split(" ")[0]}` : ""}</h1>
        <p className="mt-1 text-muted-foreground">Here's what's happening with your lab reports.</p>
      </div>

      <Link to="/app/upload" className="group flex flex-col items-start gap-4 rounded-2xl border-2 border-dashed border-primary/30 bg-accent/40 p-6 transition-colors hover:bg-accent sm:flex-row sm:items-center">
        <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-primary text-primary-foreground"><UploadCloud className="h-6 w-6" /></div>
        <div className="flex-1">
          <h2 className="font-semibold">Upload a new report</h2>
          <p className="text-sm text-muted-foreground">PDF, PNG or JPG up to 10 MB. CBC and Lipid Profile supported.</p>
        </div>
        <ArrowRight className="h-5 w-5 text-primary transition-transform group-hover:translate-x-1" />
      </Link>

      <div className="grid gap-6 lg:grid-cols-2">
        <section className="rounded-2xl border bg-card p-5">
          <div className="mb-4 flex items-center justify-between">
            <h2 className="font-semibold">Recent reports</h2>
            <Button variant="ghost" size="sm" asChild><Link to="/app/reports">View all</Link></Button>
          </div>
          {loadingReports ? <Loading /> : reports.length === 0 ? (
            <EmptyState icon={<FileText />} title="No reports yet" text="Upload your first lab report to get started." />
          ) : (
            <ul className="space-y-2">
              {reports.slice(0, 3).map((r) => {
                const flagged = r.result_statuses.filter((s) => s === "low" || s === "high").length;
                return (
                  <li key={r.id}>
                    <Link to="/app/reports/$reportId" params={{ reportId: r.id }} className="flex items-center gap-3 rounded-xl p-3 hover:bg-muted">
                      <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-accent text-primary"><FileText className="h-5 w-5" /></div>
                      <div className="min-w-0 flex-1">
                        <p className="truncate font-medium">{r.name}</p>
                        <p className="text-xs text-muted-foreground">{reportTypeLabel(r.report_type)} · {formatDate(r.created_at)}</p>
                      </div>
                      {flagged > 0
                        ? <span className="text-xs text-muted-foreground">{flagged} outside range</span>
                        : <StatusBadge status="Normal" />}
                    </Link>
                  </li>
                );
              })}
            </ul>
          )}
        </section>

        <section className="rounded-2xl border bg-card p-5">
          <div className="mb-4 flex items-center justify-between">
            <h2 className="font-semibold">Recent chats</h2>
            <Button variant="ghost" size="sm" asChild><Link to="/app/chats">View all</Link></Button>
          </div>
          {loadingChats ? <Loading /> : chats.length === 0 ? (
            <EmptyState icon={<MessagesSquare />} title="No chats yet" text="Open a report and ask a question." />
          ) : (
            <ul className="space-y-2">
              {chats.slice(0, 4).map((c) => {
                const r = reports.find((x) => x.id === c.report_id);
                return (
                  <li key={c.id}>
                    <Link to="/app/reports/$reportId" params={{ reportId: c.report_id }} search={{ chat: c.id }} className="flex items-center gap-3 rounded-xl p-3 hover:bg-muted">
                      <MessagesSquare className="h-4 w-4 text-primary" />
                      <div className="min-w-0 flex-1">
                        <p className="truncate font-medium">{c.title}</p>
                        <p className="truncate text-xs text-muted-foreground">{r?.name}</p>
                      </div>
                    </Link>
                  </li>
                );
              })}
            </ul>
          )}
        </section>
      </div>
    </div>
  );
}
