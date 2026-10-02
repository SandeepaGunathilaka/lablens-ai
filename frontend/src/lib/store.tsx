/** Session state and server data hooks. Server data lives in the TanStack Query cache. */

import { createContext, useContext, useEffect, useState, type ReactNode } from "react";
import { useQuery, useQueryClient, type QueryKey } from "@tanstack/react-query";
import { toast } from "sonner";
import { api, getToken, onAuthChange, type User } from "./api";

interface AuthState {
  /** False until the stored token has been checked (always false during SSR). */
  ready: boolean;
  loggedIn: boolean;
  user: User | null;
}

const AuthContext = createContext<AuthState>({ ready: false, loggedIn: false, user: null });

export function StoreProvider({ children }: { children: ReactNode }) {
  const queryClient = useQueryClient();
  const [auth, setAuth] = useState<AuthState>({ ready: false, loggedIn: false, user: null });

  useEffect(() => {
    let cancelled = false;
    const sync = async () => {
      if (!getToken()) {
        queryClient.clear();
        if (!cancelled) setAuth({ ready: true, loggedIn: false, user: null });
        return;
      }
      try {
        const user = await api.me();
        if (!cancelled) setAuth({ ready: true, loggedIn: true, user });
      } catch {
        // An invalid or expired token is cleared by the API client, which calls sync again.
        if (!cancelled && !getToken()) setAuth({ ready: true, loggedIn: false, user: null });
      }
    };
    void sync();
    const unsubscribe = onAuthChange(() => void sync());
    return () => {
      cancelled = true;
      unsubscribe();
    };
  }, [queryClient]);

  return <AuthContext.Provider value={auth}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthState {
  return useContext(AuthContext);
}

export function useReports() {
  const { loggedIn } = useAuth();
  return useQuery({ queryKey: ["reports"], queryFn: api.listReports, enabled: loggedIn });
}

export function useReport(id: string) {
  const { loggedIn } = useAuth();
  return useQuery({ queryKey: ["report", id], queryFn: () => api.getReport(id), enabled: loggedIn && !!id, retry: false });
}

/** All chats, or only one report's chats. Both live under the ["chats"] key prefix. */
export function useChats(reportId?: string) {
  const { loggedIn } = useAuth();
  return useQuery({
    queryKey: reportId ? ["chats", reportId] : ["chats"],
    queryFn: () => api.listChats(reportId),
    enabled: loggedIn,
  });
}

export function useChat(id: string | undefined) {
  const { loggedIn } = useAuth();
  return useQuery({ queryKey: ["chat", id], queryFn: () => api.getChat(id!), enabled: loggedIn && !!id });
}

function failed(error: unknown) {
  toast.error(error instanceof Error ? error.message : "Something went wrong.");
}

export function useActions() {
  const queryClient = useQueryClient();

  /** Refetch every query whose key starts with one of the given prefixes. */
  const refresh = async (...keys: QueryKey[]) => {
    await Promise.all(keys.map((queryKey) => queryClient.invalidateQueries({ queryKey })));
  };

  const run = async <T,>(action: () => Promise<T>, ...keys: QueryKey[]): Promise<T | undefined> => {
    try {
      const result = await action();
      await refresh(...keys);
      return result;
    } catch (error) {
      failed(error);
      return undefined;
    }
  };

  return {
    refresh,
    renameReport: (id: string, name: string) => run(() => api.renameReport(id, name), ["reports"], ["report", id]),
    deleteReport: (id: string) => run(() => api.deleteReport(id), ["reports"], ["chats"]),
    deleteImage: (id: string) => run(() => api.deleteReportFile(id), ["reports"], ["report", id]),
    newChat: async (reportId: string) => (await run(() => api.createChat(reportId), ["chats"], ["reports"]))?.id,
    renameChat: (id: string, title: string) => run(() => api.renameChat(id, title), ["chats"], ["chat", id]),
    deleteChat: (id: string) => run(() => api.deleteChat(id), ["chats"], ["reports"]),
  };
}
