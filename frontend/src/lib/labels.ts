import type { ResultStatus } from "./api";

export type Status = "Low" | "Normal" | "High" | "Unknown";

/** Must match DISCLAIMER in backend/agents/safety_agent.py: the report page strips it from final_response by exact text. */
export const DISCLAIMER =
  "This summary is for general education only and is not medical advice. Please talk to your doctor about your results.";

export function toStatus(status: ResultStatus | undefined): Status {
  if (status === "low") return "Low";
  if (status === "high") return "High";
  if (status === "normal") return "Normal";
  return "Unknown";
}

export function reportTypeLabel(reportType: string | null | undefined): string {
  if (reportType === "cbc") return "CBC";
  if (reportType === "lipid") return "Lipid Profile";
  return "Lab report";
}

export function formatDate(iso: string): string {
  return new Date(iso).toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" });
}

export function formatValue(value: number, unit: string | null | undefined): string {
  const number = Number.isInteger(value) ? String(value) : String(Number(value.toFixed(4)));
  return unit ? `${number} ${unit}` : number;
}

export function suggestedQuestions(reportType: string | null | undefined): string[] {
  if (reportType === "lipid") {
    return ["What does my LDL result mean?", "What is HDL cholesterol?", "Which of my results are outside the range?"];
  }
  if (reportType === "cbc") {
    return ["What does my hemoglobin result mean?", "What do white blood cells do?", "Which of my results are outside the range?"];
  }
  return ["Explain my results in simple terms.", "Which of my results are outside the range?"];
}
