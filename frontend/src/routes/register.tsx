import { createFileRoute } from "@tanstack/react-router";
import { AuthForm } from "@/components/lab/AuthForm";

export const Route = createFileRoute("/register")({
  head: () => ({
    meta: [
      { title: "Sign up — LabLens AI" },
      { name: "description", content: "Create a LabLens AI account to understand your lab results." },
      { property: "og:title", content: "Sign up — LabLens AI" },
      { property: "og:description", content: "Create a LabLens AI account to understand your lab results." },
    ],
  }),
  component: () => <AuthForm mode="register" />,
});
