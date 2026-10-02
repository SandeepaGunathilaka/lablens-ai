import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { useEffect, useRef, useState } from "react";
import { Check, FileUp, Loader2, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { api } from "@/lib/api";
import { useActions } from "@/lib/store";

export const Route = createFileRoute("/app/upload")({
  head: () => ({
    meta: [
      { title: "Upload a report — LabLens AI" },
      { name: "description", content: "Upload a CBC or lipid report as PDF, PNG or JPG." },
      { property: "og:title", content: "Upload a report — LabLens AI" },
      { property: "og:description", content: "Upload a CBC or lipid report as PDF, PNG or JPG." },
    ],
  }),
  component: UploadPage,
});

const STEPS = ["Reading", "Checking ranges", "Finding sources", "Explaining", "Safety check"];
const OK = ["application/pdf", "image/png", "image/jpeg"];

function UploadPage() {
  const navigate = useNavigate();
  const { refresh } = useActions();
  const input = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [error, setError] = useState("");
  const [drag, setDrag] = useState(false);
  const [step, setStep] = useState(-1);
  const [done, setDone] = useState(false);

  const pick = (f?: File) => {
    if (!f) return;
    if (!OK.includes(f.type)) return setError("Please choose a PDF, PNG or JPG file.");
    if (f.size > 10 * 1024 * 1024) return setError("That file is larger than 10 MB.");
    setError(""); setFile(f);
  };

  const analyze = async () => {
    if (!file) return;
    setStep(0); setDone(false); setError("");
    try {
      const report = await api.uploadReport(file);
      setDone(true);
      setStep(STEPS.length);
      await refresh(["reports"]);
      setTimeout(() => navigate({ to: "/app/reports/$reportId", params: { reportId: report.id } }), 600);
    } catch (e) {
      setStep(-1);
      setError(e instanceof Error ? e.message : "We couldn't analyze this report.");
    }
  };

  // The stepper is paced by time while the single analyze request runs; it holds on the last step until the server answers.
  useEffect(() => {
    if (step < 0 || done || step >= STEPS.length - 1) return;
    const t = setTimeout(() => setStep((s) => s + 1), 2500);
    return () => clearTimeout(t);
  }, [step, done]);

  return (
    <div className="mx-auto max-w-2xl">
      <h1 className="text-3xl font-semibold tracking-tight">Upload a report</h1>
      <p className="mt-1 text-muted-foreground">We support CBC and Lipid Profile reports.</p>

      {step < 0 ? (
        <div className="mt-8 space-y-4">
          <div
            onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
            onDragLeave={() => setDrag(false)}
            onDrop={(e) => { e.preventDefault(); setDrag(false); pick(e.dataTransfer.files[0]); }}
            onClick={() => input.current?.click()}
            className={cn("flex cursor-pointer flex-col items-center rounded-3xl border-2 border-dashed bg-card px-6 py-16 text-center transition-colors",
              drag ? "border-primary bg-accent" : "border-border hover:border-primary/50")}
          >
            <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-accent text-primary"><FileUp className="h-7 w-7" /></div>
            <p className="mt-4 font-semibold">Drag & drop your report here</p>
            <p className="text-sm text-muted-foreground">or click to browse · PDF, PNG, JPG · max 10 MB</p>
            <input ref={input} type="file" accept=".pdf,.png,.jpg,.jpeg" className="hidden" onChange={(e) => pick(e.target.files?.[0])} />
          </div>
          {error && <p className="rounded-xl bg-warning-soft px-4 py-3 text-sm text-warning-foreground">{error}</p>}
          {file && (
            <div className="flex items-center gap-3 rounded-2xl border bg-card p-4">
              <FileUp className="h-5 w-5 text-primary" />
              <div className="min-w-0 flex-1"><p className="truncate font-medium">{file.name}</p><p className="text-xs text-muted-foreground">{(file.size / 1024 / 1024).toFixed(2)} MB</p></div>
              <Button variant="ghost" size="icon" onClick={() => setFile(null)} aria-label="Remove file"><X /></Button>
            </div>
          )}
          <Button size="lg" className="w-full" disabled={!file} onClick={analyze}>Analyze report</Button>
        </div>
      ) : (
        <div className="mt-8 rounded-3xl border bg-card p-6 md:p-8">
          <p className="font-medium">Analyzing <span className="text-primary">{file?.name}</span></p>
          <ol className="mt-6 space-y-1">
            {STEPS.map((s, i) => {
              const done = i < step, active = i === step;
              return (
                <li key={s} className="flex gap-4">
                  <div className="flex flex-col items-center">
                    <div className={cn("flex h-9 w-9 items-center justify-center rounded-full border-2 text-sm font-semibold transition-colors",
                      done ? "border-primary bg-primary text-primary-foreground" : active ? "border-primary text-primary" : "border-border text-muted-foreground")}>
                      {done ? <Check className="h-4 w-4" /> : active ? <Loader2 className="h-4 w-4 animate-spin" /> : i + 1}
                    </div>
                    {i < STEPS.length - 1 && <div className={cn("h-6 w-0.5", done ? "bg-primary" : "bg-border")} />}
                  </div>
                  <p className={cn("pt-1.5 font-medium", !done && !active && "text-muted-foreground")}>{s}</p>
                </li>
              );
            })}
          </ol>
          {step >= STEPS.length && <p className="mt-4 text-sm font-medium text-success-foreground">All done — opening your report…</p>}
        </div>
      )}
    </div>
  );
}
