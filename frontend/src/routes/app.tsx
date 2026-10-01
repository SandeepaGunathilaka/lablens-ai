import { createFileRoute, Link, Outlet, useNavigate } from "@tanstack/react-router";
import { useEffect } from "react";
import { FileText, LayoutDashboard, LogOut, MessagesSquare, Upload } from "lucide-react";
import { Loading, Logo } from "@/components/lab/shared";
import { logout } from "@/lib/api";
import { useAuth } from "@/lib/store";

export const Route = createFileRoute("/app")({
  head: () => ({ meta: [{ name: "robots", content: "noindex" }] }),
  component: AppLayout,
});

const nav = [
  { to: "/app", label: "Dashboard", icon: LayoutDashboard, exact: true },
  { to: "/app/upload", label: "Upload", icon: Upload },
  { to: "/app/reports", label: "My Reports", icon: FileText },
  { to: "/app/chats", label: "Chats", icon: MessagesSquare },
] as const;

function AppLayout() {
  const { ready, loggedIn, user } = useAuth();
  const navigate = useNavigate();

  useEffect(() => {
    if (ready && !loggedIn) navigate({ to: "/login" });
  }, [ready, loggedIn, navigate]);

  if (!ready || !loggedIn) return <Loading />;

  return (
    <div className="flex min-h-screen flex-col md:flex-row">
      <aside className="border-b bg-sidebar md:sticky md:top-0 md:flex md:h-screen md:w-60 md:flex-col md:border-b-0 md:border-r">
        <div className="flex items-center justify-between px-5 py-4 md:py-6">
          <Link to="/"><Logo /></Link>
          <button onClick={() => { logout(); navigate({ to: "/" }); }} aria-label="Log out"
            className="rounded-lg p-2 text-muted-foreground hover:bg-muted md:hidden">
            <LogOut className="h-4 w-4" />
          </button>
        </div>
        <nav className="flex gap-1 overflow-x-auto px-3 pb-3 md:flex-1 md:flex-col md:pb-0">
          {nav.map((n) => (
            <Link
              key={n.to}
              to={n.to}
              activeOptions={{ exact: "exact" in n }}
              className="flex shrink-0 items-center gap-3 rounded-xl px-3 py-2.5 text-sm font-medium text-sidebar-foreground transition-colors hover:bg-muted"
              activeProps={{ className: "!bg-sidebar-accent !text-sidebar-accent-foreground" }}
            >
              <n.icon className="h-4 w-4" /> {n.label}
            </Link>
          ))}
        </nav>
        <div className="hidden p-3 md:block">
          {user && <p className="truncate px-3 pb-1 text-xs text-muted-foreground" title={user.email}>{user.email}</p>}
          <button onClick={() => { logout(); navigate({ to: "/" }); }}
            className="flex w-full items-center gap-3 rounded-xl px-3 py-2.5 text-sm text-muted-foreground hover:bg-muted">
            <LogOut className="h-4 w-4" /> Log out
          </button>
        </div>
      </aside>
      <main className="min-w-0 flex-1 px-4 py-6 md:px-8 md:py-8">
        <Outlet />
      </main>
    </div>
  );
}
