import { useEffect, useState } from "react";
import { Link, useNavigate } from "@tanstack/react-router";
import { Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { api } from "@/lib/api";
import { useAuth } from "@/lib/store";
import { Logo } from "./shared";

export function AuthForm({ mode }: { mode: "login" | "register" }) {
  const navigate = useNavigate();
  const { ready, loggedIn } = useAuth();
  const isReg = mode === "register";
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (ready && loggedIn) navigate({ to: "/app" });
  }, [ready, loggedIn, navigate]);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (isReg && new TextEncoder().encode(password).length > 72) return setError("Password must be at most 72 bytes.");
    setBusy(true);
    setError("");
    try {
      if (isReg) await api.register(name, email, password);
      await api.login(email, password);
      navigate({ to: "/app" });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong.");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="hero-glow flex min-h-screen items-center justify-center px-4">
      <div className="w-full max-w-md rounded-3xl border bg-card p-8 shadow-xl shadow-primary/5">
        <Link to="/"><Logo /></Link>
        <h1 className="mt-6 text-2xl font-semibold">{isReg ? "Create your account" : "Welcome back"}</h1>
        <p className="mt-1 text-sm text-muted-foreground">{isReg ? "Start understanding your lab reports." : "Log in to see your reports."}</p>
        <form className="mt-6 space-y-4" onSubmit={submit}>
          {isReg && (
            <div className="space-y-1.5"><Label htmlFor="name">Name</Label><Input id="name" value={name} onChange={(e) => setName(e.target.value)} placeholder="Alex Perera" maxLength={100} required /></div>
          )}
          <div className="space-y-1.5"><Label htmlFor="email">Email</Label><Input id="email" type="email" value={email} onChange={(e) => setEmail(e.target.value)} placeholder="you@example.com" autoComplete="email" required /></div>
          <div className="space-y-1.5">
            <Label htmlFor="pw">Password</Label>
            <Input id="pw" type="password" value={password} onChange={(e) => setPassword(e.target.value)} placeholder="••••••••"
              minLength={isReg ? 8 : undefined} autoComplete={isReg ? "new-password" : "current-password"} required />
            {isReg && <p className="text-xs text-muted-foreground">At least 8 characters.</p>}
          </div>
          {error && <p role="alert" className="rounded-xl bg-warning-soft px-4 py-3 text-sm text-warning-foreground">{error}</p>}
          <Button type="submit" className="w-full" size="lg" disabled={busy}>
            {busy && <Loader2 className="animate-spin" />} {isReg ? "Sign up" : "Log in"}
          </Button>
        </form>
        <p className="mt-6 text-center text-sm text-muted-foreground">
          {isReg ? "Already have an account? " : "New to LabLens? "}
          <Link to={isReg ? "/login" : "/register"} className="font-medium text-primary hover:underline">{isReg ? "Log in" : "Sign up"}</Link>
        </p>
      </div>
    </div>
  );
}
