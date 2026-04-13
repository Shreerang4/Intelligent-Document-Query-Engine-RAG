import { useEffect, useRef, useState } from 'react';
import './App.css';

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || '').trim().replace(/\/$/, '');
const API_BASE_LABEL = API_BASE_URL || window.location.origin;

function apiUrl(path) {
  return `${API_BASE_URL}${path}`;
}

const MODE_OPTIONS = [
  { value: 'url', label: 'PDF URL' },
  { value: 'upload', label: 'Upload PDF' },
];

const ANSWER_STATUS_LABELS = {
  ok: 'OK',
  no_context: 'No Context',
  error: 'Error',
};

const CLAIM_VERDICT_LABELS = {
  supported: 'Supported',
  weakly_supported: 'Weakly Supported',
  unsupported: 'Unsupported',
};

function parseQuestions(text) {
  return text
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);
}

async function readResponseBody(response) {
  const contentType = response.headers.get('content-type') || '';

  if (contentType.includes('application/json')) {
    return response.json();
  }

  const text = await response.text();
  if (!text) {
    return null;
  }

  try {
    return JSON.parse(text);
  } catch {
    return text;
  }
}

function getErrorMessage(payload, fallbackMessage) {
  if (!payload) {
    return fallbackMessage;
  }

  if (typeof payload === 'string') {
    return payload;
  }

  if (typeof payload.detail === 'string') {
    return payload.detail;
  }

  if (Array.isArray(payload.detail)) {
    return payload.detail
      .map((item) => {
        if (typeof item === 'string') {
          return item;
        }
        if (item && typeof item.msg === 'string') {
          return item.msg;
        }
        return null;
      })
      .filter(Boolean)
      .join(' ');
  }

  if (typeof payload.message === 'string') {
    return payload.message;
  }

  return fallbackMessage;
}

function formatFileSize(bytes) {
  if (!Number.isFinite(bytes) || bytes <= 0) {
    return '';
  }

  if (bytes < 1024 * 1024) {
    return `${(bytes / 1024).toFixed(1)} KB`;
  }

  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

function App() {
  const fileInputRef = useRef(null);
  const [mode, setMode] = useState('url');
  const [documentUrl, setDocumentUrl] = useState('');
  const [selectedFile, setSelectedFile] = useState(null);
  const [questionsText, setQuestionsText] = useState('');
  const [token, setToken] = useState('');
  const [showToken, setShowToken] = useState(false);
  const [answers, setAnswers] = useState([]);
  const [formError, setFormError] = useState('');
  const [requestError, setRequestError] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [runMeta, setRunMeta] = useState({ cacheStatus: '', cacheEntries: '' });
  const [health, setHealth] = useState({
    loading: true,
    error: '',
    data: null,
  });

  const questions = parseQuestions(questionsText);

  async function refreshHealth() {
    setHealth((current) => ({
      ...current,
      loading: true,
      error: '',
    }));

    try {
      const response = await fetch(apiUrl('/health'));
      const payload = await readResponseBody(response);

      if (!response.ok) {
        throw new Error(getErrorMessage(payload, `Health check failed with status ${response.status}.`));
      }

      setHealth({
        loading: false,
        error: '',
        data: payload,
      });
    } catch (error) {
      setHealth({
        loading: false,
        error: error instanceof Error ? error.message : 'Unable to reach the backend.',
        data: null,
      });
    }
  }

  useEffect(() => {
    void refreshHealth();
  }, []);

  function handleModeChange(nextMode) {
    setMode(nextMode);
    setFormError('');
    setRequestError('');
  }

  function handleFileChange(event) {
    const nextFile = event.target.files?.[0] || null;
    setFormError('');

    if (!nextFile) {
      setSelectedFile(null);
      return;
    }

    const isPdf = nextFile.type === 'application/pdf' || nextFile.name.toLowerCase().endsWith('.pdf');
    if (!isPdf) {
      setSelectedFile(null);
      setFormError('Please choose a PDF file.');
      event.target.value = '';
      return;
    }

    setSelectedFile(nextFile);
  }

  function handleReset() {
    setMode('url');
    setDocumentUrl('');
    setSelectedFile(null);
    setQuestionsText('');
    setToken('');
    setShowToken(false);
    setAnswers([]);
    setFormError('');
    setRequestError('');
    setRunMeta({ cacheStatus: '', cacheEntries: '' });

    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  }

  function validateForm() {
    if (!token.trim()) {
      return 'API token is required.';
    }

    if (mode === 'url' && !documentUrl.trim()) {
      return 'PDF URL is required.';
    }

    if (mode === 'upload' && !selectedFile) {
      return 'Please choose a PDF file to upload.';
    }

    if (questions.length === 0) {
      return 'Enter at least one question.';
    }

    return '';
  }

  async function handleSubmit(event) {
    event.preventDefault();

    const validationMessage = validateForm();
    if (validationMessage) {
      setFormError(validationMessage);
      return;
    }

    setFormError('');
    setRequestError('');
    setAnswers([]);
    setRunMeta({ cacheStatus: '', cacheEntries: '' });
    setIsSubmitting(true);

    try {
      const authHeader = { Authorization: `Bearer ${token.trim()}` };
      let response;

      if (mode === 'url') {
        response = await fetch(apiUrl('/hackrx/run'), {
          method: 'POST',
          headers: {
            ...authHeader,
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            documents: documentUrl.trim(),
            questions,
          }),
        });
      } else {
        const formData = new FormData();
        formData.append('file', selectedFile);
        formData.append('questions_json', JSON.stringify(questions));

        response = await fetch(apiUrl('/hackrx/upload-run'), {
          method: 'POST',
          headers: authHeader,
          body: formData,
        });
      }

      const payload = await readResponseBody(response);

      if (!response.ok) {
        throw new Error(getErrorMessage(payload, `Request failed with status ${response.status}.`));
      }

      setAnswers(Array.isArray(payload?.answers) ? payload.answers : []);
      setRunMeta({
        cacheStatus: response.headers.get('X-Document-Cache') || '',
        cacheEntries: response.headers.get('X-Cache-Entries') || '',
      });
      await refreshHealth();
    } catch (error) {
      setRequestError(error instanceof Error ? error.message : 'Request failed.');
    } finally {
      setIsSubmitting(false);
    }
  }

  const healthStatusClass = health.loading ? 'checking' : health.data?.status === 'healthy' ? 'healthy' : 'unhealthy';
  const healthStatusLabel = health.loading ? 'Checking' : health.data?.status === 'healthy' ? 'Healthy' : 'Unhealthy';

  return (
    <div className="app-shell">
      <header className="hero">
        <p className="eyebrow">Intelligent Document Query Engine</p>
        <h1>Ask questions over PDF documents</h1>
        <p className="hero-subtitle">
          Ask questions over PDF documents using retrieval-augmented generation.
        </p>
      </header>

      <div className="top-grid">
        <form className="panel" onSubmit={handleSubmit}>
          <div className="panel-header">
            <div>
              <p className="panel-kicker">Query setup</p>
              <h2>Document input</h2>
            </div>
            <span className="subtle-copy">{questions.length} question{questions.length === 1 ? '' : 's'}</span>
          </div>

          <div className="mode-switch" role="tablist" aria-label="PDF input mode">
            {MODE_OPTIONS.map((option) => (
              <button
                key={option.value}
                type="button"
                className={`mode-button ${mode === option.value ? 'is-active' : ''}`}
                onClick={() => handleModeChange(option.value)}
              >
                {option.label}
              </button>
            ))}
          </div>

          {mode === 'url' ? (
            <label className="field">
              <span className="field-label">PDF URL</span>
              <input
                type="url"
                className="field-input"
                placeholder="https://example.com/document.pdf"
                value={documentUrl}
                onChange={(event) => setDocumentUrl(event.target.value)}
              />
            </label>
          ) : (
            <label className="field">
              <span className="field-label">Upload PDF</span>
              <input
                ref={fileInputRef}
                type="file"
                className="field-input field-input-file"
                accept=".pdf,application/pdf"
                onChange={handleFileChange}
              />
              <span className="field-help">
                {selectedFile
                  ? `${selectedFile.name} ${formatFileSize(selectedFile.size)}`.trim()
                  : 'Choose a PDF file from your machine.'}
              </span>
            </label>
          )}

          <label className="field">
            <span className="field-label">Questions</span>
            <textarea
              className="field-input field-textarea"
              placeholder={'What is this document about?\nWhat are the key dates?\nWhich sections mention exclusions?'}
              value={questionsText}
              onChange={(event) => setQuestionsText(event.target.value)}
            />
            <span className="field-help">Enter one question per line</span>
          </label>

          <label className="field">
            <span className="field-label">Bearer token</span>
            <div className="token-row">
              <input
                type={showToken ? 'text' : 'password'}
                className="field-input"
                placeholder="Paste the API token used by the backend"
                value={token}
                onChange={(event) => setToken(event.target.value)}
              />
              <button
                type="button"
                className="ghost-button"
                onClick={() => setShowToken((current) => !current)}
              >
                {showToken ? 'Hide' : 'Show'}
              </button>
            </div>
          </label>

          {formError ? <div className="notice notice-error">{formError}</div> : null}

          <div className="actions">
            <button type="submit" className="primary-button" disabled={isSubmitting}>
              {isSubmitting ? 'Running query...' : 'Run Query'}
            </button>
            <button type="button" className="secondary-button" onClick={handleReset} disabled={isSubmitting}>
              Clear
            </button>
          </div>
        </form>

        <aside className="panel status-panel">
          <div className="panel-header">
            <div>
              <p className="panel-kicker">Service status</p>
              <h2>Backend health</h2>
            </div>
            <button type="button" className="ghost-button" onClick={() => void refreshHealth()} disabled={health.loading}>
              Refresh
            </button>
          </div>

          <div className="health-card">
            <div className="health-summary">
              <span className={`status-pill status-${healthStatusClass}`}>{healthStatusLabel}</span>
              <span className="version-copy">
                {health.data?.version ? `v${health.data.version}` : 'No version available'}
              </span>
            </div>

            <dl className="health-grid">
              <div>
                <dt>API base</dt>
                <dd>{API_BASE_LABEL}</dd>
              </div>
              <div>
                <dt>Cache entries</dt>
                <dd>{health.data?.cache_entries ?? 'n/a'}</dd>
              </div>
              <div>
                <dt>Embeddings</dt>
                <dd>{health.data ? (health.data.embedding_model_loaded ? 'Loaded' : 'Idle') : 'Unknown'}</dd>
              </div>
              <div>
                <dt>Reranker</dt>
                <dd>{health.data ? (health.data.reranker_loaded ? 'Loaded' : 'Idle') : 'Unknown'}</dd>
              </div>
              <div>
                <dt>Groq client</dt>
                <dd>{health.data ? (health.data.groq_client_loaded ? 'Loaded' : 'Idle') : 'Unknown'}</dd>
              </div>
            </dl>

            {health.error ? <p className="status-message">{health.error}</p> : null}
          </div>

          <div className="support-card">
            <p className="support-title">Supported modes</p>
            <p className="support-copy">Run the existing URL workflow or upload a local PDF without changing the backend contract.</p>
          </div>
        </aside>
      </div>

      <section className="panel results-panel">
        <div className="panel-header">
          <div>
            <p className="panel-kicker">Results</p>
            <h2>Answers</h2>
          </div>
          <div className="results-meta">
            {runMeta.cacheStatus ? (
              <span>
                Cache {runMeta.cacheStatus}
                {runMeta.cacheEntries ? ` | ${runMeta.cacheEntries} entr${runMeta.cacheEntries === '1' ? 'y' : 'ies'}` : ''}
              </span>
            ) : (
              <span>Ready for a demo run</span>
            )}
          </div>
        </div>

        {requestError ? <div className="notice notice-error">{requestError}</div> : null}

        {isSubmitting ? (
          <div className="loading-box" aria-live="polite">
            <span className="spinner" aria-hidden="true" />
            Running document query...
          </div>
        ) : null}

        {!isSubmitting && !requestError && answers.length === 0 ? (
          <div className="empty-state">
            Submit a PDF URL or upload a PDF to see grounded answers with source excerpts.
          </div>
        ) : null}

        <div className="answers-grid">
          {answers.map((answer, index) => (
            <article className="answer-card" key={`${answer.question}-${index}`}>
              <div className="answer-header">
                <div>
                  <p className="section-label">Question</p>
                  <h3>{answer.question}</h3>
                </div>
                <span className={`status-pill status-${answer.status || 'error'}`}>
                  {ANSWER_STATUS_LABELS[answer.status] || answer.status || 'Unknown'}
                </span>
              </div>

              <div className="answer-copy">
                <p className="section-label">Answer</p>
                <p>{answer.answer}</p>
              </div>

              {answer.claim_verifications?.length ? (
                <div className="claim-section">
                  <div className="source-header">
                    <p className="section-label">Claim Verification</p>
                    <span className="subtle-copy">{answer.claim_verifications.length}</span>
                  </div>

                  <div className="claim-list">
                    {answer.claim_verifications.map((verification, verificationIndex) => (
                      <div className="claim-card" key={`${verification.claim}-${verificationIndex}`}>
                        <div className="claim-card-header">
                          <p className="claim-text">{verification.claim}</p>
                          <span className={`status-pill status-${verification.verdict || 'unsupported'}`}>
                            {CLAIM_VERDICT_LABELS[verification.verdict] || verification.verdict || 'Unknown'}
                          </span>
                        </div>

                        <p className="claim-rationale">{verification.rationale}</p>

                        <div className="claim-sources">
                          <div className="claim-sources-header">
                            <p className="section-label">Supporting Sources</p>
                            <span className="subtle-copy">{verification.sources?.length || 0}</span>
                          </div>

                          {verification.sources?.length ? (
                            <div className="source-list claim-source-list">
                              {verification.sources.map((source, sourceIndex) => (
                                <div className="source-card claim-source-card" key={`${source.page}-${source.chunk_id}-${sourceIndex}`}>
                                  <div className="source-meta">
                                    <span>Page {source.page}</span>
                                    <span>Chunk {source.chunk_id}</span>
                                  </div>
                                  <p>{source.excerpt}</p>
                                </div>
                              ))}
                            </div>
                          ) : (
                            <p className="empty-sources">No supporting sources were selected for this claim.</p>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              ) : null}

              <div className="source-section">
                <div className="source-header">
                  <p className="section-label">Sources</p>
                  <span className="subtle-copy">{answer.sources?.length || 0}</span>
                </div>

                {answer.sources?.length ? (
                  <div className="source-list">
                    {answer.sources.map((source, sourceIndex) => (
                      <div className="source-card" key={`${source.page}-${source.chunk_id}-${sourceIndex}`}>
                        <div className="source-meta">
                          <span>Page {source.page}</span>
                          <span>Chunk {source.chunk_id}</span>
                        </div>
                        <p>{source.excerpt}</p>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="empty-sources">No source excerpts were returned for this answer.</p>
                )}
              </div>
            </article>
          ))}
        </div>
      </section>
    </div>
  );
}

export default App;
