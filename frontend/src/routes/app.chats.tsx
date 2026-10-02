import { createFileRoute, Link } from "@tanstack/react-router";
import { FileText, MessagesSquare, Pencil, Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { ConfirmDelete, EmptyState, Loading, RenameDialog } from "@/components/lab/shared";
import { formatDate, reportTypeLabel } from "@/lib/labels";
import { useActions, useChats, useReports } from "@/lib/store";

export const Route = createFileRoute("/app/chats")({
  head: () => ({
    meta: [
      { title: "Chats — LabLens AI" },
      { name: "description", content: "All your conversations about your lab reports." },
      { property: "og:title", content: "Chats — LabLens AI" },
      { property: "og:description", content: "All your conversations about your lab reports." },
    ],
  }),
  component: ChatsPage,
});

function ChatsPage() {
  const { data: reports = [], isLoading: loadingReports } = useReports();
  const { data: chats = [], isLoading: loadingChats } = useChats();
  const { renameChat, deleteChat } = useActions();
  const groups = reports.map((r) => ({ r, items: chats.filter((c) => c.report_id === r.id) })).filter((g) => g.items.length);

  return (
    <div className="mx-auto max-w-4xl">
      <h1 className="text-3xl font-semibold tracking-tight">Chats</h1>
      <p className="mt-1 text-muted-foreground">Your conversations, grouped by report.</p>
      <div className="mt-8 space-y-8">
        {loadingReports || loadingChats ? <Loading /> : groups.length === 0 ? (
          <EmptyState icon={<MessagesSquare />} title="No chats yet" text="Open a report and ask a question to start a chat."
            action={<Button asChild><Link to="/app/reports">Go to My Reports</Link></Button>} />
        ) : groups.map(({ r, items }) => (
          <section key={r.id}>
            <div className="mb-3 flex items-center gap-2 text-sm font-semibold text-muted-foreground">
              <FileText className="h-4 w-4" /> {r.name} <span className="font-normal">· {reportTypeLabel(r.report_type)}</span>
            </div>
            <ul className="divide-y rounded-2xl border bg-card">
              {items.map((c) => (
                <li key={c.id} className="flex flex-wrap items-center gap-3 p-4">
                  <MessagesSquare className="h-4 w-4 text-primary" />
                  <div className="min-w-0 flex-1">
                    <p className="truncate font-medium">{c.title}</p>
                    <p className="text-xs text-muted-foreground">{c.message_count} messages · {formatDate(c.updated_at)}</p>
                  </div>
                  <Button size="sm" asChild><Link to="/app/reports/$reportId" params={{ reportId: r.id }} search={{ chat: c.id }}>Continue</Link></Button>
                  <RenameDialog title="Rename chat" value={c.title} onSave={(v) => renameChat(c.id, v)}
                    trigger={<Button size="icon" variant="ghost" aria-label="Rename chat"><Pencil /></Button>} />
                  <ConfirmDelete title="Delete this chat?" description="This conversation will be removed permanently."
                    onConfirm={() => deleteChat(c.id)}
                    trigger={<Button size="icon" variant="ghost" aria-label="Delete chat"><Trash2 /></Button>} />
                </li>
              ))}
            </ul>
          </section>
        ))}
      </div>
    </div>
  );
}
