import { useState } from 'react'

// The dev proxy strips the first /api; the retrieval router itself is mounted at /api/retrieval.
const RETRIEVAL_URL = '/api/api/retrieval'
const EXPLANATION_URL = '/api/explanation'
const LOGIN_URL = '/api/auth/login'

const EMPTY_FORM = {
  test: 'Hemoglobin',
  value: '10.2',
  unit: 'g/dL',
  referenceRange: '12.0-15.5',
  status: 'low',
  userQuestion: '',
}

const PRESETS = [
  { label: 'Hemoglobin', test: 'Hemoglobin', value: '10.2', unit: 'g/dL', referenceRange: '12.0-15.5', status: 'low' },
  { label: 'WBC', test: 'WBC', value: '7.0', unit: 'x10^9/L', referenceRange: '4.0-11.0', status: 'normal' },
  { label: 'Platelets', test: 'Platelets', value: '450', unit: 'x10^9/L', referenceRange: '150-450', status: 'normal' },
  { label: 'HDL', test: 'HDL', value: '35', unit: 'mg/dL', referenceRange: '>=40', status: 'low' },
  { label: 'LDL', test: 'LDL', value: '160', unit: 'mg/dL', referenceRange: '<100', status: 'high' },
  { label: 'Triglycerides', test: 'Triglycerides', value: '140', unit: 'mg/dL', referenceRange: '<150', status: 'normal' },
  { label: 'Total Cholesterol', test: 'Total Cholesterol', value: '180', unit: 'mg/dL', referenceRange: '<200', status: 'normal' },
  { label: 'Missing range', test: 'Hemoglobin', value: '13.4', unit: 'g/dL', referenceRange: '', status: '' },
  { label: 'Not in knowledge base', test: 'Ferritin', value: '40', unit: 'ng/mL', referenceRange: '20-250', status: 'normal' },
]

function formatApiError(data, fallback) {
  if (!data) {
    return fallback
  }
  if (typeof data.detail === 'string') {
    return data.detail
  }
  if (Array.isArray(data.detail)) {
    return data.detail
      .map((item) => {
        const field = Array.isArray(item.loc)
          ? item.loc.filter((part) => part !== 'body').join('.')
          : 'request'
        return `${field}: ${item.msg}`
      })
      .join(' ')
  }
  return fallback
}

async function postJson(url, body, fallbackError, token) {
  const headers = { 'Content-Type': 'application/json' }
  if (token) {
    headers.Authorization = `Bearer ${token}`
  }
  const response = await fetch(url, { method: 'POST', headers, body: JSON.stringify(body) })
  const data = await response.json().catch(() => null)
  if (!response.ok) {
    throw new Error(formatApiError(data, fallbackError))
  }
  return data
}

function userIdFromToken(token) {
  const payload = token.split('.')[1] || ''
  const json = atob(payload.replace(/-/g, '+').replace(/_/g, '/'))
  return JSON.parse(json).sub
}

// Same conversion the Coordinator applies before calling Explanation and Safety.
function toRetrievedSources(retrieval) {
  return retrieval.results
    .filter((result) => result.found)
    .map((result) => ({
      test_name: result.test_name,
      information: { passages: result.matches.map((match) => match.information) },
      sources: result.matches.flatMap((match) => match.sources),
    }))
}

function bannerCopy(finding) {
  if (finding.generation_mode === 'insufficient') {
    return 'There is not enough reliable source information, so no meaning was inferred.'
  }
  if (finding.generation_mode === 'unavailable') {
    return 'The explanation model is unavailable. No medical explanation was generated.'
  }
  if (finding.generation_mode === 'safe_fallback') {
    return 'The draft could not be verified against the retrieved sources, so it was replaced with a safe response.'
  }
  if (finding.generation_mode === 'template') {
    return 'This development preview uses only the retrieved source text.'
  }
  return 'This explanation uses only the sources returned by the Retrieval Agent.'
}

function SignIn({ onSignedIn }) {
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  async function submit(event) {
    event.preventDefault()
    setLoading(true)
    setError('')
    try {
      const data = await postJson(LOGIN_URL, { email, password }, 'Sign-in failed.')
      onSignedIn({ token: data.access_token, userId: userIdFromToken(data.access_token) })
    } catch (requestError) {
      setError(requestError.message || 'Unable to reach the sign-in service.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <form className="card" onSubmit={submit}>
      <h2>Sign in</h2>
      <p className="card-note">Explanations are only available to signed-in users.</p>
      <div className="form-grid">
        <label className="field field-wide">
          <span>Email</span>
          <input type="email" value={email} onChange={(event) => setEmail(event.target.value)} required />
        </label>
        <label className="field field-wide">
          <span>Password</span>
          <input
            type="password"
            value={password}
            onChange={(event) => setPassword(event.target.value)}
            required
          />
        </label>
      </div>
      {error ? <p className="error-text">{error}</p> : null}
      <div className="actions">
        <button className="primary" type="submit" disabled={loading}>
          {loading ? 'Signing in...' : 'Sign in'}
        </button>
      </div>
    </form>
  )
}

function ExplanationResults() {
  const [session, setSession] = useState(null)
  const [form, setForm] = useState(EMPTY_FORM)
  const [finding, setFinding] = useState(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  function updateField(key, value) {
    setForm((current) => ({ ...current, [key]: value }))
    setFinding(null)
    setError('')
  }

  function applyPreset(preset) {
    setForm((current) => ({
      ...current,
      test: preset.test,
      value: preset.value,
      unit: preset.unit,
      referenceRange: preset.referenceRange,
      status: preset.status,
    }))
    setFinding(null)
    setError('')
  }

  async function submitExplanation(event) {
    event.preventDefault()
    const value = Number(form.value)
    if (!Number.isFinite(value)) {
      setError('Value must be a number.')
      return
    }

    setLoading(true)
    setError('')
    setFinding(null)

    const ids = {
      task_id: crypto.randomUUID(),
      report_id: crypto.randomUUID(),
      user_id: session.userId,
    }
    const test = form.test.trim()

    try {
      const retrieval = await postJson(
        RETRIEVAL_URL,
        { ...ids, test_names: [test] },
        'The retrieval request failed.',
      )
      const explanation = await postJson(
        EXPLANATION_URL,
        {
          ...ids,
          findings: [
            {
              test,
              value,
              unit: form.unit.trim() || null,
              reference_range: form.referenceRange.trim() || null,
              status: form.status || null,
            },
          ],
          retrieved_sources: toRetrievedSources(retrieval),
          user_question: form.userQuestion.trim() || null,
        },
        'The explanation request failed.',
        session.token,
      )
      setFinding(explanation.findings[0])
    } catch (requestError) {
      setError(requestError.message || 'Unable to reach the explanation service.')
    } finally {
      setLoading(false)
    }
  }

  if (!session) {
    return <SignIn onSignedIn={setSession} />
  }

  const caution = finding && finding.generation_mode !== 'llm' && finding.generation_mode !== 'template'

  return (
    <>
      <p className="intro">
        Sources come from the Retrieval Agent's approved knowledge base. Status is supplied with the
        result; the language model never chooses it. Possible meaning stays educational: it does not
        assign a personal condition or recommend medication.
      </p>
      <div className="layout">
        <form className="card" onSubmit={submitExplanation}>
          <h2>Lab result</h2>
          <p className="card-note">
            A test that is not in the knowledge base gets a "not enough reliable information"
            response instead of an invented explanation.
          </p>
          <div className="presets" aria-label="Sample results">
            {PRESETS.map((preset) => (
              <button
                key={preset.label}
                type="button"
                className="preset"
                onClick={() => applyPreset(preset)}
              >
                {preset.label}
              </button>
            ))}
          </div>
          <div className="form-grid">
            <label className="field">
              <span>Test name</span>
              <input
                value={form.test}
                onChange={(event) => updateField('test', event.target.value)}
                required
              />
            </label>
            <label className="field">
              <span>Status</span>
              <select value={form.status} onChange={(event) => updateField('status', event.target.value)}>
                <option value="low">low</option>
                <option value="normal">normal</option>
                <option value="high">high</option>
                <option value="">could not be determined</option>
              </select>
            </label>
            <label className="field">
              <span>Value</span>
              <input
                inputMode="decimal"
                value={form.value}
                onChange={(event) => updateField('value', event.target.value)}
                required
              />
            </label>
            <label className="field">
              <span>Unit</span>
              <input value={form.unit} onChange={(event) => updateField('unit', event.target.value)} />
            </label>
            <label className="field field-wide">
              <span>Reference range</span>
              <input
                value={form.referenceRange}
                onChange={(event) => updateField('referenceRange', event.target.value)}
                placeholder="12.0-15.5, <200, >=40, or leave blank"
              />
            </label>
            <label className="field field-wide">
              <span>Follow-up question</span>
              <input
                value={form.userQuestion}
                onChange={(event) => updateField('userQuestion', event.target.value)}
              />
            </label>
          </div>
          <div className="actions">
            <button className="primary" type="submit" disabled={loading}>
              {loading ? 'Writing explanation...' : 'Explain result'}
            </button>
            <button className="secondary" type="button" onClick={() => setSession(null)}>
              Sign out
            </button>
          </div>
        </form>

        <section className="card" aria-live="polite">
          <h2>Explanation</h2>
          {error ? <p className="error-text">{error}</p> : null}
          {!finding && !error ? (
            <p className="empty-result">
              Submit a result to see what it measures, the explanation, a possible meaning, and what
              to discuss with a clinician.
            </p>
          ) : null}
          {finding ? (
            <>
              <div className={caution ? 'banner banner-caution' : 'banner'}>{bannerCopy(finding)}</div>
              <div className="result-head">
                <div>
                  <h3>
                    {finding.test_name} {finding.result}
                  </h3>
                </div>
                <span className={`status-pill status-${finding.status || 'unknown'}`}>
                  {finding.status || 'unknown'}
                </span>
              </div>
              <div className="sections">
                <article className="section">
                  <h3>What it measures</h3>
                  <p>{finding.what_it_measures}</p>
                </article>
                <article className="section">
                  <h3>Explanation</h3>
                  <p>{finding.explanation}</p>
                </article>
                <article className="section">
                  <h3>Possible meaning</h3>
                  <p>{finding.possible_meaning}</p>
                </article>
                <article className="section">
                  <h3>Recommended discussion</h3>
                  <p>{finding.recommended_discussion}</p>
                </article>
              </div>
              {finding.sources_used.length > 0 ? (
                <>
                  <h3>Sources used</h3>
                  <ul className="source-list">
                    {finding.sources_used.map((source) => (
                      <li key={source}>{source}</li>
                    ))}
                  </ul>
                </>
              ) : null}
            </>
          ) : null}
        </section>
      </div>
    </>
  )
}

export default ExplanationResults
