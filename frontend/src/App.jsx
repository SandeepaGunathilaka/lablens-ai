import { useEffect, useState } from 'react'
import './App.css'

function App() {
  const [backendStatus, setBackendStatus] = useState({
    loading: true,
    message: '',
    error: '',
  })

  useEffect(() => {
    const fetchBackendHealth = async () => {
      try {
        const response = await fetch('/api/health')

        if (!response.ok) {
          throw new Error('API request failed')
        }

        const data = await response.json()
        setBackendStatus({
          loading: false,
          message: data.status || 'ok',
          error: '',
        })
      } catch (error) {
        setBackendStatus({
          loading: false,
          message: '',
          error: 'Unable to reach the backend API at http://127.0.0.1:8000',
        })
      }
    }

    fetchBackendHealth()
  }, [])

  return (
    <main className="app-shell">
      <section className="status-card">
        <p className="eyebrow">LabLens AI</p>
        <h1>Backend connection status</h1>

        {backendStatus.loading ? (
          <p className="status-text">Checking backend health...</p>
        ) : backendStatus.error ? (
          <p className="status-error">{backendStatus.error}</p>
        ) : (
          <p className="status-success">
            Connected successfully. Backend status: <strong>{backendStatus.message}</strong>
          </p>
        )}
      </section>
    </main>
  )
}

export default App
