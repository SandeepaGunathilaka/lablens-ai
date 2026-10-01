import { createBrowserRouter, RouterProvider } from 'react-router'

import { AuthProvider } from '@/auth/AuthProvider'
import { GuestOnly, RequireAuth } from '@/auth/guards'
import AppLayout from '@/layouts/AppLayout'
import ComingSoon from '@/pages/app/ComingSoon'
import Overview from '@/pages/app/Overview'
import Landing from '@/pages/Landing'
import Login from '@/pages/Login'
import NotFound from '@/pages/NotFound'
import Register from '@/pages/Register'

// Public: /, /login, /register. Everything under /app needs a valid session.
const router = createBrowserRouter([
  { path: '/', element: <Landing /> },
  {
    element: <GuestOnly />,
    children: [
      { path: '/login', element: <Login /> },
      { path: '/register', element: <Register /> },
    ],
  },
  {
    path: '/app',
    element: <RequireAuth />,
    children: [
      {
        element: <AppLayout />,
        children: [
          { index: true, element: <Overview /> },
          {
            path: 'upload',
            element: (
              <ComingSoon
                title="Upload a report"
                description="Report upload connects to the Document Agent and is coming in the next phase."
              />
            ),
          },
          {
            path: 'reports',
            element: <ComingSoon title="My reports" description="Your uploaded reports will be listed here." />,
          },
          {
            path: 'explanations',
            element: (
              <ComingSoon
                title="Explanations"
                description="Safety-checked, plain-language explanations of your results will appear here."
              />
            ),
          },
        ],
      },
    ],
  },
  { path: '*', element: <NotFound /> },
])

export default function App() {
  return (
    <AuthProvider>
      <RouterProvider router={router} />
    </AuthProvider>
  )
}
