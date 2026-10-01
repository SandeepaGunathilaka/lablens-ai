import { createFileRoute, Link } from "@tanstack/react-router";
import { ArrowRight, BookOpen, FileUp, FolderHeart, Lock, MessageCircleQuestion, ShieldCheck, Sparkles, Stethoscope } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Disclaimer, Logo } from "@/components/lab/shared";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "LabLens AI — Your lab report, in plain language" },
      { name: "description", content: "Upload a CBC or lipid report and get calm, plain-language explanations with trusted sources. Educational only." },
      { property: "og:title", content: "LabLens AI — Your lab report, in plain language" },
      { property: "og:description", content: "Calm, source-backed explanations of your blood test results." },
    ],
  }),
  component: Landing,
});

const steps = [
  { icon: FileUp, name: "Upload your report", text: "Add a PDF or a clear photo of your Complete Blood Count or Lipid Profile." },
  { icon: Sparkles, name: "See your results clearly", text: "Each test, your value and the reference range from your report, with a simple Low, Normal or High label." },
  { icon: BookOpen, name: "Understand what they mean", text: "Plain-language explanations based on trusted medical references, with links so you can read more." },
  { icon: MessageCircleQuestion, name: "Ask follow-up questions", text: "Chat about any result and get questions worth bringing to your next appointment." },
];

const features = [
  { icon: ShieldCheck, name: "Careful by design", text: "Every answer is checked before you see it. LabLens never diagnoses a condition or recommends medication." },
  { icon: FolderHeart, name: "Keep everything in one place", text: "Your reports and conversations are saved, so you can come back, continue a chat, or delete them any time." },
  { icon: Lock, name: "Private to you", text: "Only you can see your reports. Remove the original file or a whole report whenever you like." },
];

const tests = [
  { group: "Complete Blood Count (CBC)", items: ["Hemoglobin", "White Blood Cells (WBC)", "Platelets"] },
  { group: "Lipid Profile", items: ["Total Cholesterol", "LDL Cholesterol", "HDL Cholesterol", "Triglycerides"] },
];

function Landing() {
  return (
    <div className="min-h-screen">
      <header className="mx-auto flex max-w-6xl items-center justify-between px-5 py-5">
        <Logo />
        <div className="flex gap-2">
          <Button variant="ghost" asChild><Link to="/login">Log in</Link></Button>
          <Button asChild><Link to="/register">Sign up</Link></Button>
        </div>
      </header>

      <section className="hero-glow">
        <div className="mx-auto grid max-w-6xl items-center gap-12 px-5 pb-20 pt-12 md:grid-cols-2 md:pt-20">
          <div>
            <span className="inline-flex rounded-full bg-accent px-3 py-1 text-xs font-semibold text-accent-foreground">CBC & Lipid Profile</span>
            <h1 className="mt-5 text-4xl font-semibold leading-tight tracking-tight md:text-5xl">
              Your lab report, explained <span className="text-primary">calmly</span> and clearly.
            </h1>
            <p className="mt-5 max-w-md text-lg text-muted-foreground">
              Upload your results and get plain-language explanations, trusted sources and good questions to bring to your doctor.
            </p>
            <div className="mt-8 flex flex-wrap gap-3">
              <Button size="lg" asChild><Link to="/register">Get started <ArrowRight /></Link></Button>
              <Button size="lg" variant="outline" asChild><Link to="/login">Log in</Link></Button>
            </div>
          </div>
          <div className="rounded-3xl border bg-card p-5 shadow-xl shadow-primary/5">
            <p className="text-sm font-medium text-muted-foreground">Example · Complete Blood Count</p>
            {[
              ["Hemoglobin", "11.2 g/dL", "Low", "bg-warning-soft text-warning-foreground"],
              ["White Blood Cells", "7.0 ×10³/µL", "Normal", "bg-success-soft text-success-foreground"],
              ["Platelets", "450 ×10³/µL", "Normal", "bg-success-soft text-success-foreground"],
            ].map(([t, v, s, c]) => (
              <div key={t} className="mt-3 flex items-center justify-between rounded-xl bg-muted/60 px-4 py-3">
                <div><p className="font-medium">{t}</p><p className="text-sm text-muted-foreground">{v}</p></div>
                <span className={`rounded-full px-2.5 py-0.5 text-xs font-semibold ${c}`}>{s}</span>
              </div>
            ))}
            <div className="mt-4 rounded-xl border border-primary/20 bg-accent/50 p-4 text-sm text-accent-foreground">
              Hemoglobin is a protein in red blood cells that carries oxygen. This value is below the range printed on the report — a good thing to ask a doctor about.
            </div>
          </div>
        </div>
      </section>

      <section className="mx-auto max-w-6xl px-5 py-20">
        <h2 className="text-3xl font-semibold tracking-tight">How it works</h2>
        <p className="mt-2 text-muted-foreground">From upload to understanding in a few minutes.</p>
        <div className="mt-10 grid gap-5 sm:grid-cols-2 lg:grid-cols-4">
          {steps.map((s, i) => (
            <div key={s.name} className="rounded-2xl border bg-card p-6">
              <div className="flex items-center justify-between">
                <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-accent text-primary"><s.icon className="h-5 w-5" /></div>
                <span className="font-display text-3xl font-semibold text-border">0{i + 1}</span>
              </div>
              <h3 className="mt-5 font-semibold">{s.name}</h3>
              <p className="mt-2 text-sm text-muted-foreground">{s.text}</p>
            </div>
          ))}
        </div>
      </section>

      <section className="bg-muted/40">
        <div className="mx-auto grid max-w-6xl gap-5 px-5 py-20 md:grid-cols-3">
          {features.map((f) => (
            <div key={f.name}>
              <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-accent text-primary"><f.icon className="h-5 w-5" /></div>
              <h3 className="mt-4 font-semibold">{f.name}</h3>
              <p className="mt-2 text-sm text-muted-foreground">{f.text}</p>
            </div>
          ))}
        </div>
      </section>

      <section className="mx-auto max-w-6xl px-5 py-20">
        <h2 className="text-3xl font-semibold tracking-tight">Tests we can explain</h2>
        <p className="mt-2 text-muted-foreground">Upload one report at a time. More tests are on the way.</p>
        <div className="mt-8 grid gap-5 md:grid-cols-2">
          {tests.map((g) => (
            <div key={g.group} className="rounded-2xl border bg-card p-6">
              <h3 className="font-semibold">{g.group}</h3>
              <ul className="mt-4 flex flex-wrap gap-2">
                {g.items.map((t) => <li key={t} className="rounded-full bg-accent px-3 py-1 text-sm text-accent-foreground">{t}</li>)}
              </ul>
            </div>
          ))}
        </div>

        <div className="mt-12 flex flex-col items-start gap-4 rounded-2xl border bg-card p-6 sm:flex-row sm:items-center">
          <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-xl bg-accent text-primary"><Stethoscope className="h-5 w-5" /></div>
          <p className="flex-1 text-sm text-muted-foreground">
            LabLens helps you understand your numbers and prepare for your appointment. It is not a substitute for your doctor, who can interpret your results alongside your health history.
          </p>
          <Button asChild><Link to="/register">Get started <ArrowRight /></Link></Button>
        </div>
        <Disclaimer className="mt-6" />
      </section>
    </div>
  );
}
