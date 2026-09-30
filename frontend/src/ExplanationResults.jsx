import { useState } from 'react'

const EMPTY_FORM = {
  testName: 'Hemoglobin',
  value: '10.2',
  unit: 'g/dL',
  referenceRange: '12.0-15.5',
  status: 'auto',
  sourceTitle: '',
  sourceExcerpt: '',
  sourceUrl: '',
  userQuestion: '',
  rejectionFeedback: '',
}

const PRESETS = [
  { label: 'Hemoglobin', testName: 'Hemoglobin', value: '10.2', unit: 'g/dL', referenceRange: '12.0-15.5' },
  { label: 'WBC', testName: 'WBC', value: '7.0', unit: 'x10^9/L', referenceRange: '4.0-11.0' },
  { label: 'Platelets', testName: 'Platelets', value: '450', unit: 'x10^9/L', referenceRange: '150-450' },
  { label: 'HDL', testName: 'HDL', value: '35', unit: 'mg/dL', referenceRange: '>=40' },
  { label: 'LDL', testName: 'LDL', value: '160', unit: 'mg/dL', referenceRange: '<100' },
  { label: 'Triglycerides', testName: 'Triglycerides', value: '140', unit: 'mg/dL', referenceRange: '<150' },
  { label: 'Total Cholesterol', testName: 'Total Cholesterol', value: '180', unit: 'mg/dL', referenceRange: '<200' },
  { label: 'Missing range', testName: 'Hemoglobin', value: '13.4', unit: 'g/dL', referenceRange: '' },
  { label: 'Unreadable range', testName: 'Hemoglobin', value: '13.4', unit: 'g/dL', referenceRange: 'see note' },
]

const SAMPLE_SOURCE = {
  sourceTitle: 'Sample hemoglobin note',
  sourceExcerpt:
    'Hemoglobin is a protein in red blood cells that carries oxygen. A laboratory report compares the measured amount with a reference range supplied by the lab.',
  sourceUrl: '',
}

function formatApiError(data) {
  if (!data) {
    return 'The explanation request failed.'
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
  return 'The explanation request failed.'
}

function bannerCopy(result) {
  if (result.generation_mode === 'insufficient') {
    return 'There is not enough reliable source information, so no meaning was inferred.'
  }
  if (result.generation_mode === 'unavailable') {
    return 'The explanation model is unavailable. No medical explanation was generated.'
  }
  if (result.generation_mode === 'safe_fallback') {
    return 'The draft could not be verified against the retrieved sources, so it was replaced with a safe response.'
  }
  if (result.generation_mode === 'template') {
    return 'This development preview uses only the retrieved source text you supplied.'
  }
  return 'This explanation uses only the retrieved sources supplied with the result.'
}

function ExplanationResults() {
  const [form, setForm] = useState(EMPTY_FORM)
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  function updateField(key, value) {
    setForm((current) => ({ ...current, [key]: value }))
    setResult(null)
    setError('')
  }

  function applyPreset(preset) {
    setForm((current) => ({
      ...current,
      testName: preset.testName,
      value: preset.value,
      unit: preset.unit,
      referenceRange: preset.referenceRange,
      status: 'auto',
    }))
    setResult(null)
    setError('')
  }

  async function submitExplanation(event) {
    event.preventDefault()
    setLoading(true)
    setError('')
    setResult(null)

    const payload = {
      task_id: crypto.randomUUID(),
      test_name: form.testName.trim(),
      value: form.value.trim(),
      unit: form.unit.trim(),
    }
    if (form.referenceRange.trim()) {
      payload.reference_range = form.referenceRange.trim()
    }
    if (form.status !== 'auto') {
      payload.status = form.status
    }
    if (form.sourceTitle.trim() && form.sourceExcerpt.trim()) {
      payload.retrieved_sources = [
        {
          title: form.sourceTitle.trim(),
          excerpt: form.sourceExcerpt.trim(),
          url: form.sourceUrl.trim() || null,
        },
      ]
    }
    const hasTitle = Boolean(form.sourceTitle.trim())
    const hasExcerpt = Boolean(form.sourceExcerpt.trim())
    if (hasTitle !== hasExcerpt) {
      setLoading(false)
      setError('A retrieved source needs both a title and an excerpt.')
      return
    }
    if (form.userQuestion.trim()) {
      payload.user_question = form.userQuestion.trim()
    }
    if (form.rejectionFeedback.trim()) {
      payload.rejection_feedback = [form.rejectionFeedback.trim()]
    }

    try {
      const response = await fetch('/api/explanation', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })
      const data = await response.json()
      if (!response.ok) {
        throw new Error(formatApiError(data))
      }
      setResult(data)
    } catch (requestError) {
      setError(requestError.message || 'Unable to reach the explanation service.')
    } finally {
      setLoading(false)
    }
  }

  const caution = result && result.generation_mode !== 'llm' && result.generation_mode !== 'template'

  return (
    <>
      <p className="intro">
        Status is calculated from the value and reference range before any explanation is written.
        The language model does not choose it. Possible meaning stays educational: it does not
        assign a personal condition or recommend medication.
      </p>
      <div className="layout">
        <form className="card" onSubmit={submitExplanation}>
          <h2>Lab result</h2>
          <p className="card-note">
            Use a retrieved source when you have one. With no source, the agent says the information
            is not enough instead of filling in a medical claim.
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
                value={form.testName}
                onChange={(event) => updateField('testName', event.target.value)}
                required
              />
            </label>
            <label className="field">
              <span>Status input</span>
              <select
                value={form.status}
                onChange={(event) => updateField('status', event.target.value)}
              >
                <option value="auto">Calculate from the range</option>
                <option value="low">Caller supplied: low</option>
                <option value="normal">Caller supplied: normal</option>
                <option value="high">Caller supplied: high</option>
                <option value="unknown">Caller supplied: unknown</option>
              </select>
            </label>
            <label className="field">
              <span>Value</span>
              <input
                value={form.value}
                onChange={(event) => updateField('value', event.target.value)}
                required
              />
            </label>
            <label className="field">
              <span>Unit</span>
              <input
                value={form.unit}
                onChange={(event) => updateField('unit', event.target.value)}
                required
              />
            </label>
            <label className="field field-wide">
              <span>Reference range</span>
              <input
                value={form.referenceRange}
                onChange={(event) => updateField('referenceRange', event.target.value)}
                placeholder="12.0-15.5, <200, >=40, or leave blank"
              />
            </label>
            <label className="field">
              <span>Source title</span>
              <input
                value={form.sourceTitle}
                onChange={(event) => updateField('sourceTitle', event.target.value)}
              />
            </label>
            <label className="field">
              <span>Source URL</span>
              <input
                value={form.sourceUrl}
                onChange={(event) => updateField('sourceUrl', event.target.value)}
              />
            </label>
            <label className="field field-wide">
              <span>Retrieved source excerpt</span>
              <textarea
                rows="4"
                value={form.sourceExcerpt}
                onChange={(event) => updateField('sourceExcerpt', event.target.value)}
              />
            </label>
            <label className="field field-wide">
              <span>Follow-up question</span>
              <input
                value={form.userQuestion}
                onChange={(event) => updateField('userQuestion', event.target.value)}
              />
            </label>
            <label className="field field-wide">
              <span>Safety rejection feedback</span>
              <textarea
                rows="2"
                value={form.rejectionFeedback}
                onChange={(event) => updateField('rejectionFeedback', event.target.value)}
                placeholder="Optional. Sends this request as a regeneration."
              />
            </label>
          </div>
          <div className="actions">
            <button className="primary" type="submit" disabled={loading}>
              {loading ? 'Writing explanation...' : 'Explain result'}
            </button>
            <button
              className="secondary"
              type="button"
              onClick={() => {
                setForm((current) => ({ ...current, ...SAMPLE_SOURCE }))
                setResult(null)
                setError('')
              }}
            >
              Use sample source
            </button>
          </div>
        </form>

        <section className="card" aria-live="polite">
          <h2>Explanation</h2>
          {error ? <p className="error-text">{error}</p> : null}
          {!result && !error ? (
            <p className="empty-result">
              Submit a result to see what it measures, the explanation, a possible meaning, and what
              to discuss with a clinician.
            </p>
          ) : null}
          {result ? (
            <>
              <div className={caution ? 'banner banner-caution' : 'banner'}>{bannerCopy(result)}</div>
              <div className="result-head">
                <div>
                  <h3>
                    {result.test_name} {result.value} {result.unit}
                  </h3>
                  <p className="status-line">{result.status_detail}</p>
                </div>
                <span className={`status-pill status-${result.status}`}>{result.status}</span>
              </div>
              <div className="sections">
                <article className="section">
                  <h3>What it measures</h3>
                  <p>{result.what_it_measures}</p>
                </article>
                <article className="section">
                  <h3>Explanation</h3>
                  <p>{result.explanation}</p>
                </article>
                <article className="section">
                  <h3>Possible meaning</h3>
                  <p>{result.possible_meaning}</p>
                </article>
                <article className="section">
                  <h3>Recommended discussion</h3>
                  <p>{result.recommended_discussion}</p>
                </article>
              </div>
              {result.sources_used.length > 0 ? (
                <>
                  <h3>Sources used</h3>
                  <ul className="source-list">
                    {result.sources_used.map((source) => (
                      <li key={source}>{source}</li>
                    ))}
                  </ul>
                </>
              ) : null}
              {result.regenerated ? (
                <p className="card-note">This response was treated as a regeneration.</p>
              ) : null}
            </>
          ) : null}
        </section>
      </div>
    </>
  )
}

export default ExplanationResults
