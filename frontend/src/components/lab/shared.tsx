import { useState, type ReactNode } from "react";
import { Info, Loader2, ShieldCheck } from "lucide-react";
import { cn } from "@/lib/utils";
import { DISCLAIMER, type Status } from "@/lib/labels";
import {
  AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent, AlertDialogDescription,
  AlertDialogFooter, AlertDialogHeader, AlertDialogTitle, AlertDialogTrigger,
} from "@/components/ui/alert-dialog";
import { Dialog, DialogContent, DialogFooter, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";

const statusStyles: Record<Status, string> = {
  Low: "bg-warning-soft text-warning-foreground border-warning/30",
  High: "bg-warning-soft text-warning-foreground border-warning/30",
  Normal: "bg-success-soft text-success-foreground border-success/30",
  Unknown: "bg-muted text-muted-foreground border-border",
};

export function StatusBadge({ status }: { status: Status }) {
  return (
    <span className={cn("inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-semibold", statusStyles[status])}>
      {status}
    </span>
  );
}

export function SafetyBadge() {
  return (
    <span className="inline-flex items-center gap-1 rounded-full bg-accent px-2.5 py-0.5 text-xs font-medium text-accent-foreground">
      <ShieldCheck className="h-3.5 w-3.5" /> Safety checked
    </span>
  );
}

export function Disclaimer({ className }: { className?: string }) {
  return (
    <div className={cn("flex items-start gap-2 rounded-xl border border-primary/20 bg-accent/60 px-4 py-3 text-sm text-accent-foreground", className)}>
      <Info className="mt-0.5 h-4 w-4 shrink-0" />
      <p>{DISCLAIMER}</p>
    </div>
  );
}

export function ConfirmDelete({ trigger, title, description, onConfirm }: { trigger: ReactNode; title: string; description: string; onConfirm: () => void }) {
  return (
    <AlertDialog>
      <AlertDialogTrigger asChild>{trigger}</AlertDialogTrigger>
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle>{title}</AlertDialogTitle>
          <AlertDialogDescription>{description}</AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter>
          <AlertDialogCancel>Cancel</AlertDialogCancel>
          <AlertDialogAction onClick={onConfirm} className="bg-destructive text-destructive-foreground hover:bg-destructive/90">
            Delete
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
}

export function RenameDialog({ trigger, title, value, onSave }: { trigger: ReactNode; title: string; value: string; onSave: (v: string) => void }) {
  const [open, setOpen] = useState(false);
  const [v, setV] = useState(value);
  return (
    <Dialog open={open} onOpenChange={(o) => { setOpen(o); if (o) setV(value); }}>
      <DialogTrigger asChild>{trigger}</DialogTrigger>
      <DialogContent>
        <DialogHeader><DialogTitle>{title}</DialogTitle></DialogHeader>
        <form onSubmit={(e) => { e.preventDefault(); if (v.trim()) { onSave(v.trim()); setOpen(false); } }} className="space-y-4">
          <Input value={v} onChange={(e) => setV(e.target.value)} autoFocus />
          <DialogFooter>
            <Button type="button" variant="outline" onClick={() => setOpen(false)}>Cancel</Button>
            <Button type="submit">Save</Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}

export function EmptyState({ icon, title, text, action }: { icon: ReactNode; title: string; text: string; action?: ReactNode }) {
  return (
    <div className="flex flex-col items-center justify-center rounded-2xl border border-dashed bg-card px-6 py-14 text-center">
      <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-full bg-accent text-primary">{icon}</div>
      <h3 className="font-display text-lg font-semibold">{title}</h3>
      <p className="mt-1 max-w-sm text-sm text-muted-foreground">{text}</p>
      {action && <div className="mt-5">{action}</div>}
    </div>
  );
}

export function Loading({ text = "Loading…" }: { text?: string }) {
  return (
    <div className="flex items-center justify-center gap-2 py-14 text-sm text-muted-foreground">
      <Loader2 className="h-4 w-4 animate-spin" /> {text}
    </div>
  );
}

export function Logo() {
  return (
    <span className="flex items-center gap-2 font-display text-lg font-bold tracking-tight">
      <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary text-primary-foreground">
        <svg viewBox="0 0 24 24" className="h-4.5 w-4.5" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round"><circle cx="11" cy="11" r="6" /><path d="M20 20l-4.5-4.5M8.5 11h5M11 8.5v5" /></svg>
      </span>
      LabLens <span className="text-primary">AI</span>
    </span>
  );
}
