import { useEffect, useState } from 'react'
import './App.css'
import ExplanationResults from './ExplanationResults.jsx'

function StatusPanel() {
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
      } catch {
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
    <section className="card status-card">
      <p className="brand-mark">Connection</p>
      <h2>Backend connection status</h2>
      {backendStatus.loading ? (
        <p>Checking backend health...</p>
      ) : backendStatus.error ? (
        <p className="error-text">{backendStatus.error}</p>
      ) : (
        <p className="status-success">
          Connected successfully. Backend status: <strong>{backendStatus.message}</strong>
        </p>
      )}
    </section>
  )
}

function App() {
  const [view, setView] = useState('results')

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          <p className="brand-mark">LabLens AI</p>
          <h1>Explanation results</h1>
        </div>
        <nav className="nav" aria-label="Primary">
          <button
            type="button"
            aria-current={view === 'results' ? 'page' : undefined}
            onClick={() => setView('results')}
          >
            Results
          </button>
          <button
            type="button"
            aria-current={view === 'status' ? 'page' : undefined}
            onClick={() => setView('status')}
          >
            Connection
          </button>
        </nav>
      </header>
      <main className="page">
        {view === 'results' ? <ExplanationResults /> : <StatusPanel />}
      </main>
    </div>
  )
}

export default App
