import { useEffect, useRef, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { ExternalLink, Info, Loader2, MessageSquarePlus, Pencil, SendHorizontal, Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { cn } from "@/lib/utils";
import { api, type AnswerFinding, type ChatDetail, type ChatMessage, type ReportDetail } from "@/lib/api";
import { formatValue, suggestedQuestions, toStatus } from "@/lib/labels";
import { useActions, useChat, useChats } from "@/lib/store";
import { ConfirmDelete, Disclaimer, Loading, RenameDialog, SafetyBadge, StatusBadge } from "./shared";

const MAX_QUESTION = 1000;

/** Tests named in the question; empty lets the backend answer about every test. */
function mentionedTests(report: ReportDetail, question: string): string[] {
  const q = question.toLowerCase();
  return report.results.filter((r) => q.includes(r.test.toLowerCase())).map((r) => r.test);
}

function hostname(url: string | null) {
  try {
    return url ? new URL(url).hostname.replace(/^www\./, "") : "";
  } catch {
    return "";
  }
}

function FindingCard({ f }: { f: AnswerFinding }) {
  const sections = [
    ["What it measures", f.what_it_measures],
    ["Explanation", f.explanation],
    ["What it may mean", f.possible_meaning],
    ["Questions for your doctor", f.recommended_discussion],
  ].filter(([, text]) => text);
  return (
    <div className={cn("rounded-xl border bg-background p-4", f.insufficient_information && "border-dashed bg-muted/40")}>
      <div className="flex flex-wrap items-center justify-between gap-2">
        <p className="font-semibold">{f.test}</p>
        <div className="flex items-center gap-2 text-sm text-muted-foreground">{formatValue(f.value, f.unit)} <StatusBadge status={toStatus(f.status)} /></div>
      </div>
      {f.insufficient_information && <p className="mt-2 text-xs font-medium text-muted-foreground">Not enough reliable information</p>}
      {f.generation_mode === "unavailable" && <p className="mt-2 text-xs font-medium text-muted-foreground">Explanations are temporarily unavailable.</p>}
      <dl className="mt-3 space-y-3 text-sm">
        {sections.map(([label, text]) => (
          <div key={label}><dt className="text-xs font-semibold uppercase tracking-wide text-primary">{label}</dt><dd className="mt-0.5">{text}</dd></div>
        ))}
      </dl>
      {f.sources.length > 0 && (
        <div className="mt-3 flex flex-wrap gap-1.5">
          {f.sources.map((s) => s.url && (
            <a key={s.url} href={s.url} target="_blank" rel="noreferrer" title={s.title}
              className="inline-flex items-center gap-1 rounded-full border bg-card px-2.5 py-0.5 text-xs hover:border-primary hover:text-primary">
              {hostname(s.url) || s.title} <ExternalLink className="h-3 w-3" />
            </a>
          ))}
        </div>
      )}
    </div>
  );
}

function Message({ m }: { m: ChatMessage }) {
  if (m.role === "user") {
    return (
      <div className="flex justify-end">
        <p className="max-w-[85%] whitespace-pre-line rounded-2xl rounded-br-md bg-primary px-4 py-2.5 text-sm text-primary-foreground">{m.text}</p>
      </div>
    );
  }
  const a = m.answer;
  if (!a) return null;
  if (a.status === "fallback") {
    return <p className="flex items-start gap-2 rounded-xl bg-warning-soft px-4 py-3 text-sm text-warning-foreground"><Info className="mt-0.5 h-4 w-4 shrink-0" />{a.message}</p>;
  }
  return (
    <div className="space-y-3">
      {a.findings.map((f, i) => <FindingCard key={`${f.test}-${i}`} f={f} />)}
      <SafetyBadge />
    </div>
  );
}

export function ReportChat({ report, activeId, onSelect }: { report: ReportDetail; activeId: string | undefined; onSelect: (id: string | undefined) => void }) {
  const qc = useQueryClient();
  const { newChat, renameChat, deleteChat, refresh } = useActions();
  const { data: mine = [] } = useChats(report.id);
  const currentId = activeId ?? mine[0]?.id;
  const { data: chat, isLoading } = useChat(currentId);
  const [text, setText] = useState("");
  const [selected, setSelected] = useState<string[]>([]);
  const [pending, setPending] = useState<string | null>(null);
  const [error, setError] = useState("");
  const ta = useRef<HTMLTextAreaElement>(null);
  const end = useRef<HTMLDivElement>(null);

  useEffect(() => { ta.current?.focus(); }, [currentId, pending]);
  useEffect(() => { end.current?.scrollIntoView({ behavior: "smooth", block: "nearest" }); }, [chat?.messages.length, pending]);

  const startChat = async () => {
    const id = await newChat(report.id);
    if (id) onSelect(id);
    return id;
  };

  const send = async (q: string) => {
    const question = q.trim();
    if (!question || pending) return;
    setError("");
    const id = currentId ?? (await startChat());
    if (!id) return;
    if (id !== activeId) onSelect(id);
    setPending(question);
    setText("");
    try {
      const tests = selected.length ? selected : mentionedTests(report, question);
      const res = await api.sendMessage(id, question, tests);
      qc.setQueryData<ChatDetail>(["chat", id], (old) => old && {
        ...old, ...res.chat, messages: [...old.messages, res.user_message, res.assistant_message],
      });
      await refresh(["chats"], ["reports"]);
    } catch (e) {
      setText(question);
      setError(e instanceof Error ? e.message : "Something went wrong.");
    } finally {
      setPending(null);
    }
  };

  const toggle = (test: string) => setSelected((s) => (s.includes(test) ? s.filter((x) => x !== test) : [...s, test]));
  const messages = chat?.messages ?? [];

  return (
    <div className="flex h-full min-h-[640px] overflow-hidden rounded-2xl border bg-card">
      <aside className="hidden w-48 shrink-0 flex-col border-r bg-muted/40 xl:flex">
        <div className="p-3">
          <Button size="sm" className="w-full" onClick={startChat}><MessageSquarePlus /> New chat</Button>
        </div>
        <ul className="flex-1 space-y-0.5 overflow-y-auto px-2 pb-2">
          {mine.map((c) => (
            <li key={c.id} className={cn("group flex items-center rounded-lg", c.id === currentId ? "bg-accent text-accent-foreground" : "hover:bg-muted")}>
              <button className="min-w-0 flex-1 truncate px-2.5 py-2 text-left text-sm" onClick={() => onSelect(c.id)}>{c.title}</button>
              <div className="flex opacity-0 transition-opacity group-hover:opacity-100">
                <RenameDialog title="Rename chat" value={c.title} onSave={(v) => renameChat(c.id, v)}
                  trigger={<button className="p-1 text-muted-foreground hover:text-foreground" aria-label="Rename"><Pencil className="h-3.5 w-3.5" /></button>} />
                <ConfirmDelete title="Delete this chat?" description="This conversation will be removed permanently."
                  onConfirm={async () => { await deleteChat(c.id); if (c.id === currentId) onSelect(undefined); }}
                  trigger={<button className="p-1 pr-2 text-muted-foreground hover:text-destructive" aria-label="Delete"><Trash2 className="h-3.5 w-3.5" /></button>} />
              </div>
            </li>
          ))}
          {!mine.length && <li className="px-2.5 py-2 text-xs text-muted-foreground">No chats yet</li>}
        </ul>
      </aside>

      <div className="flex min-w-0 flex-1 flex-col">
        <div className="flex items-center justify-between gap-2 border-b px-4 py-3">
          <p className="truncate font-semibold">{chat?.title ?? "Ask about this report"}</p>
          <Button size="sm" variant="outline" className="xl:hidden" onClick={startChat}><MessageSquarePlus /> New</Button>
        </div>

        <div className="flex-1 space-y-4 overflow-y-auto p-4" style={{ maxHeight: 620 }}>
          {currentId && isLoading ? <Loading /> : messages.length === 0 && !pending ? (
            <div className="py-8 text-center">
              <p className="font-display text-lg font-semibold">What would you like to understand?</p>
              <p className="mt-1 text-sm text-muted-foreground">Pick a question below or type your own.</p>
            </div>
          ) : messages.map((m) => <Message key={m.id} m={m} />)}
          {pending && (
            <>
              <div className="flex justify-end">
                <p className="max-w-[85%] whitespace-pre-line rounded-2xl rounded-br-md bg-primary/80 px-4 py-2.5 text-sm text-primary-foreground">{pending}</p>
              </div>
              <div className="flex items-center gap-2 text-sm text-muted-foreground"><Loader2 className="h-4 w-4 animate-spin" /> Thinking and running a safety check…</div>
            </>
          )}
          <div ref={end} />
        </div>

        <div className="space-y-3 border-t p-4">
          <div className="flex flex-wrap items-center gap-1.5">
            <span className="text-xs text-muted-foreground">Ask about:</span>
            <button onClick={() => setSelected([])}
              className={cn("rounded-full border px-2.5 py-0.5 text-xs", !selected.length ? "border-primary bg-primary text-primary-foreground" : "hover:bg-muted")}>Auto</button>
            {report.results.map((r) => (
              <button key={r.test} onClick={() => toggle(r.test)}
                className={cn("rounded-full border px-2.5 py-0.5 text-xs", selected.includes(r.test) ? "border-primary bg-primary text-primary-foreground" : "hover:bg-muted")}>{r.test}</button>
            ))}
          </div>
          <div className="flex flex-wrap gap-2">
            {suggestedQuestions(report.report_type).map((q) => (
              <button key={q} onClick={() => send(q)} disabled={!!pending}
                className="rounded-full border border-primary/30 bg-accent/50 px-3 py-1 text-xs font-medium text-accent-foreground hover:bg-accent disabled:opacity-50">{q}</button>
            ))}
          </div>
          {error && <p role="alert" className="rounded-xl bg-warning-soft px-3 py-2 text-xs text-warning-foreground">{error}</p>}
          <form onSubmit={(e) => { e.preventDefault(); send(text); }} className="flex items-end gap-2">
            <div className="flex-1">
              <Textarea ref={ta} value={text} onChange={(e) => setText(e.target.value.slice(0, MAX_QUESTION))} rows={1} placeholder="Ask about a result…"
                onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(text); } }} className="min-h-10 resize-none" />
              {text.length > MAX_QUESTION * 0.8 && <p className="mt-1 text-right text-[10px] text-muted-foreground">{text.length}/{MAX_QUESTION}</p>}
            </div>
            <Button type="submit" size="icon" disabled={!text.trim() || !!pending} aria-label="Send"><SendHorizontal /></Button>
          </form>
          <Disclaimer className="text-xs" />
        </div>
      </div>
    </div>
  );
}
