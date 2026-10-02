import { useQuery } from "@tanstack/react-query";
import { CheckCircle2, ShieldAlert } from "lucide-react";
import { api, type AuditEntry } from "@/lib/api";
import { cn } from "@/lib/utils";
import { Loading } from "./shared";

const STAGES: Record<string, string> = {
  document_agent: "Reading",
  retrieval_agent: "References",
  explanation_agent: "Explaining",
  safety_agent: "Safety review",
  coordinator: "Checking",
};

const REJECTIONS: Record<string, string> = {
  patient_values_verified: "A number in the draft didn't match your report, so it was rewritten with your exact values.",
  diagnosis_detected: "The draft sounded like a diagnosis, so it was rewritten without diagnostic wording.",
  medication_detected: "The draft mentioned medication or treatment, so that was removed.",
  unsupported_claim_detected: "The draft included a claim not found in the trusted sources, so it was removed.",
  disclaimer_present: "The draft was missing the educational disclaimer.",
};

interface Step {
  stage: string;
  title: string;
  detail: string;
  flagged: boolean;
}

interface Details {
  attempt?: number;
  error?: string;
  result_count?: number;
  extraction_method?: string;
  computed?: number;
  unknown?: number;
  tests_with_sources?: number;
  tests_requested?: number;
  reason?: string;
  outcome?: string;
}

function describe(e: AuditEntry): Pick<Step, "title" | "detail"> {
  const d = e.details as Details;
  const attempt = typeof d.attempt === "number" && d.attempt > 1 ? ` (attempt ${d.attempt} of 3)` : "";
  if (e.status === "error") {
    return { title: `This step couldn't be completed${attempt}`, detail: e.agent === "document_agent" && d.error ? String(d.error) : "A safe message was shown instead." };
  }
  switch (`${e.agent}/${e.action}`) {
    case "document_agent/extract":
      return { title: "Read your report", detail: `Found ${d.result_count} test result(s) ${d.extraction_method === "ocr" ? "in your photo or scan" : "in your PDF"}.` };
    case "coordinator/compute_status":
      return { title: "Compared each value with its range", detail: `Labels come straight from the reference ranges printed on your report${Number(d.unknown) ? `; ${d.unknown} value(s) had no usable range` : ""}.` };
    case "retrieval_agent/retrieve":
      return { title: "Looked up trusted medical references", detail: `Found reference information for ${d.tests_with_sources} of ${d.tests_requested} test(s).` };
    case "explanation_agent/explain":
      return { title: `Wrote plain-language explanations${attempt}`, detail: "Based only on the trusted references found for your tests." };
    case "safety_agent/validate":
      return e.status === "rejected"
        ? { title: `Revised the wording${attempt}`, detail: REJECTIONS[String(d.reason)] ?? "Part of the draft didn't meet our safety rules, so it was rewritten." }
        : { title: "Passed our safety review", detail: "No diagnosis, no medication advice, values match your report." };
    case "coordinator/analyze_report":
    case "coordinator/answer_question":
      return d.outcome === "approved"
        ? { title: "Ready for you", detail: "Only reviewed text is shown." }
        : { title: "Showed a safe message instead", detail: "We couldn't produce an explanation that met every safety rule, so none was shown." };
    default:
      return { title: "Processing step", detail: "" };
  }
}

export function toSteps(entries: AuditEntry[]): Step[] {
  return entries.map((e) => ({
    stage: STAGES[e.agent] ?? "Processing",
    ...describe(e),
    flagged: e.status === "rejected" || e.status === "error",
  }));
}

export function AgentTimeline({ taskId }: { taskId: string }) {
  const { data, isLoading, error } = useQuery({ queryKey: ["audit", taskId], queryFn: () => api.getAudit(taskId) });
  if (isLoading) return <Loading />;
  if (error) return <p className="px-4 text-sm text-muted-foreground">{error.message}</p>;
  const steps = toSteps(data ?? []);
  if (!steps.length) return <p className="px-4 text-sm text-muted-foreground">No steps were recorded for this analysis.</p>;

  return (
    <ol className="mt-4 px-4">
      {steps.map((s, i) => (
        <li key={i} className="relative flex gap-3 pb-6">
          {i < steps.length - 1 && <span className="absolute left-[11px] top-7 h-full w-0.5 bg-border" />}
          <span className={cn("z-10 flex h-6 w-6 shrink-0 items-center justify-center rounded-full", s.flagged ? "bg-warning-soft text-warning-foreground" : "bg-accent text-primary")}>
            {s.flagged ? <ShieldAlert className="h-3.5 w-3.5" /> : <CheckCircle2 className="h-3.5 w-3.5" />}
          </span>
          <div className={cn("flex-1", s.flagged && "-mt-1 rounded-xl bg-warning-soft/60 p-3")}>
            <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">{s.stage}</p>
            <p className="font-medium">{s.title}</p>
            {s.detail && <p className="text-sm text-muted-foreground">{s.detail}</p>}
          </div>
        </li>
      ))}
    </ol>
  );
}
