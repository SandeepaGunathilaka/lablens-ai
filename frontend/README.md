# LabLens AI frontend

React + TypeScript + TanStack Start + Tailwind + shadcn/ui. The UI was first scaffolded with Lovable.

```sh
npm install
npm run dev   # http://localhost:8080
```

Configuration comes from `../backend/.env` (see `vite.config.ts`). Only `VITE_*` keys are exposed to the browser:

| Variable | Default | Purpose |
|---|---|---|
| `VITE_API_BASE_URL` | `http://127.0.0.1:8000` | Backend URL |

All backend calls live in `src/lib/api.ts`; data hooks are in `src/lib/store.tsx`.
