import { createFileRoute } from "@tanstack/react-router";
import { AuthForm } from "@/components/lab/AuthForm";

export const Route = createFileRoute("/login")({
  head: () => ({
    meta: [
      { title: "Log in — LabLens AI" },
      { name: "description", content: "Log in to view your explained lab reports." },
      { property: "og:title", content: "Log in — LabLens AI" },
      { property: "og:description", content: "Log in to view your explained lab reports." },
    ],
  }),
  component: () => <AuthForm mode="login" />,
});
