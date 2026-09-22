import { useEffect, useRef, useState } from 'react';
import { API_BASE_URL, getErrorMessage, rawApiFetch, readResponseBody } from './api/client.js';
import { DuplicateDocumentError, getDocumentStatus, queryDocument, uploadDocument } from './api/documentWorkflow.js';
import AuthScreen from './auth/AuthScreen.jsx';
import { useAuth } from './auth/AuthContext.jsx';
import './App.css';

const API_BASE_LABEL = API_BASE_URL || window.location.origin;

const MODE_OPTIONS = [
  { value: 'upload', label: 'Upload PDF' },
  { value: 'url', label: 'PDF URL' },
];

const VIEW_OPTIONS = [
  { value: 'query', label: 'Query' },
  { value: 'history', label: 'History' },
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

function formatFileSize(bytes) {
  if (!Number.isFinite(bytes) || bytes <= 0) {
    return '';
  }

  if (bytes < 1024 * 1024) {
    return `${(bytes / 1024).toFixed(1)} KB`;
  }

  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

function formatDateTime(value) {
  if (!value) {
    return 'Unknown date';
  }

  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return 'Unknown date';
  }

  return date.toLocaleString();
}

function formatDocumentTitle(document) {
  return document.filename || document.source_url || `Document ${document.id.slice(0, 8)}`;
}

function AuthenticatedApp() {
  const { user, logout, authenticatedFetch } = useAuth();
  const fileInputRef = useRef(null);
  const pendingUploadRequestRef = useRef(null);
  const uploadGenerationRef = useRef(0);
  const [activeView, setActiveView] = useState('query');
  const [mode, setMode] = useState('upload');
  const [documentUrl, setDocumentUrl] = useState('');
  const [selectedFile, setSelectedFile] = useState(null);
  const [uploadedDocument, setUploadedDocument] = useState(null);
  const [duplicateDocument, setDuplicateDocument] = useState(null);
  const [pollingError, setPollingError] = useState('');
  const [questionsText, setQuestionsText] = useState('');
  const [answers, setAnswers] = useState([]);
  const [formError, setFormError] = useState('');
  const [requestError, setRequestError] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isLoggingOut, setIsLoggingOut] = useState(false);
  const [runMeta, setRunMeta] = useState({ cacheStatus: '', cacheEntries: '' });
  const [historyDocuments, setHistoryDocuments] = useState([]);
  const [historyQueries, setHistoryQueries] = useState([]);
  const [selectedHistoryDocument, setSelectedHistoryDocument] = useState(null);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [historyError, setHistoryError] = useState('');
  const [health, setHealth] = useState({
    loading: true,
    error: '',
    data: null,
  });

  const questions = parseQuestions(questionsText);
  const uploadReady = uploadedDocument?.status === 'ready';
  const uploadProcessing = uploadedDocument?.status === 'queued' || uploadedDocument?.status === 'processing';

  async function refreshHealth() {
    setHealth((current) => ({
      ...current,
      loading: true,
      error: '',
    }));

    try {
      const response = await rawApiFetch('/health');
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

  useEffect(() => {
    if (mode !== 'upload' || !uploadedDocument || !uploadProcessing || pollingError) {
      return undefined;
    }

    let cancelled = false;
    let timer;
    const controller = new AbortController();

    async function pollStatus() {
      try {
        const document = await getDocumentStatus(
          authenticatedFetch, uploadedDocument.document_id, controller.signal,
        );
        if (cancelled) return;
        if (!['queued', 'processing', 'ready', 'failed'].includes(document.status)) {
          throw new Error('Document status is unavailable.');
        }
        setUploadedDocument(document);
        if (document.status === 'queued' || document.status === 'processing') {
          timer = window.setTimeout(pollStatus, 2000);
        }
      } catch (error) {
        if (!cancelled) {
          setPollingError(error instanceof Error ? error.message : 'Document status is unavailable.');
        }
      }
    }

    timer = window.setTimeout(pollStatus, 1500);
    return () => {
      cancelled = true;
      controller.abort();
      window.clearTimeout(timer);
    };
  }, [mode, uploadedDocument?.document_id, uploadedDocument?.status, uploadProcessing, pollingError, authenticatedFetch]);

  async function loadHistoryDocuments() {
    setHistoryLoading(true);
    setHistoryError('');
    setSelectedHistoryDocument(null);
    setHistoryQueries([]);

    try {
      const response = await authenticatedFetch('/history/documents');
      const payload = await readResponseBody(response);

      if (!response.ok) {
        throw new Error(getErrorMessage(payload, `History request failed with status ${response.status}.`));
      }

      setHistoryDocuments(Array.isArray(payload?.documents) ? payload.documents : []);
    } catch (error) {
      setHistoryError(error instanceof Error ? error.message : 'Unable to load history.');
    } finally {
      setHistoryLoading(false);
    }
  }

  async function loadHistoryQueries(document) {
    setHistoryLoading(true);
    setHistoryError('');
    setSelectedHistoryDocument(document);
    setHistoryQueries([]);

    try {
      const response = await authenticatedFetch(
        `/history/documents/${encodeURIComponent(document.id)}/queries`,
      );
      const payload = await readResponseBody(response);

      if (!response.ok) {
        throw new Error(getErrorMessage(payload, `History request failed with status ${response.status}.`));
      }

      setHistoryQueries(Array.isArray(payload?.queries) ? payload.queries : []);
    } catch (error) {
      setHistoryError(error instanceof Error ? error.message : 'Unable to load query history.');
    } finally {
      setHistoryLoading(false);
    }
  }

  function handleViewChange(nextView) {
    setActiveView(nextView);
    setHistoryError('');

    if (nextView === 'history' && historyDocuments.length === 0) {
      void loadHistoryDocuments();
    }
  }

  function handleModeChange(nextMode) {
    uploadGenerationRef.current += 1;
    setMode(nextMode);
    setFormError('');
    setRequestError('');
  }

  function handleFileChange(event) {
    const nextFile = event.target.files?.[0] || null;
    uploadGenerationRef.current += 1;
    setFormError('');
    setRequestError('');
    setPollingError('');
    setUploadedDocument(null);
    setDuplicateDocument(null);
    setAnswers([]);
    pendingUploadRequestRef.current = null;

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
    uploadGenerationRef.current += 1;
    setMode('upload');
    setDocumentUrl('');
    setSelectedFile(null);
    setUploadedDocument(null);
    setDuplicateDocument(null);
    setPollingError('');
    setQuestionsText('');
    setAnswers([]);
    setFormError('');
    setRequestError('');
    setRunMeta({ cacheStatus: '', cacheEntries: '' });
    setHistoryDocuments([]);
    setHistoryQueries([]);
    setSelectedHistoryDocument(null);
    setHistoryError('');
    pendingUploadRequestRef.current = null;

    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  }

  function validateForm() {
    if (mode === 'url' && !documentUrl.trim()) {
      return 'PDF URL is required.';
    }

    if (mode === 'upload') {
      if (!uploadReady) {
        if (duplicateDocument) return 'Choose Open existing or Upload again.';
        if (uploadProcessing) return 'Document is still processing.';
        if (!selectedFile) return 'Please choose a PDF file to upload.';
        return '';
      }
      if (questions.length !== 1) return 'Enter one question for this document.';
    } else if (questions.length === 0) {
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
    if (mode === 'url' || !uploadReady) setAnswers([]);
    setRunMeta({ cacheStatus: '', cacheEntries: '' });
    setIsSubmitting(true);

    try {
      let response;

      if (mode === 'upload' && !uploadReady) {
        await submitUpload(false);
        return;
      }

      if (mode === 'url') {
        response = await authenticatedFetch('/hackrx/run', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            documents: documentUrl.trim(),
            questions,
          }),
        });
      } else {
        const generation = uploadGenerationRef.current;
        const result = await queryDocument(authenticatedFetch, uploadedDocument.document_id, questions[0]);
        if (generation !== uploadGenerationRef.current) return;
        response = result.response;
        const nextAnswers = Array.isArray(result.payload?.answers) ? result.payload.answers : [];
        setAnswers((current) => [...current, ...nextAnswers]);
        setRunMeta({
          cacheStatus: response.headers.get('X-Document-Cache') || '',
          cacheEntries: response.headers.get('X-Cache-Entries') || '',
        });
        setQuestionsText('');
        await refreshHealth();
        return;
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
      handleUploadError(error);
    } finally {
      setIsSubmitting(false);
    }
  }

  async function submitUpload(allowDuplicate) {
    const pendingRequest = pendingUploadRequestRef.current;
    const requestId = pendingRequest?.file === selectedFile
      ? pendingRequest.requestId
      : crypto.randomUUID();
    pendingUploadRequestRef.current = { file: selectedFile, requestId };
    const generation = uploadGenerationRef.current;
    const document = await uploadDocument(authenticatedFetch, selectedFile, requestId, allowDuplicate);
    if (generation !== uploadGenerationRef.current) return;
    setUploadedDocument(document);
    setDuplicateDocument(null);
    setPollingError('');
    pendingUploadRequestRef.current = null;
  }

  function handleUploadError(error) {
    if (error instanceof DuplicateDocumentError) {
      setDuplicateDocument(error.document);
      return;
    }
    setRequestError(error instanceof Error ? error.message : 'Request failed.');
  }

  function handleOpenExisting() {
    setUploadedDocument(duplicateDocument);
    setDuplicateDocument(null);
    setPollingError('');
    setRequestError('');
    pendingUploadRequestRef.current = null;
  }

  async function handleUploadAgain() {
    setIsSubmitting(true);
    setRequestError('');
    try {
      await submitUpload(true);
    } catch (error) {
      handleUploadError(error);
    } finally {
      setIsSubmitting(false);
    }
  }

  async function handleLogout() {
    setIsLoggingOut(true);
    try {
      await logout();
    } catch {
      // The auth session records a safe notice and still clears all in-memory
      // credentials. The login screen presents that server-revocation caveat.
    } finally {
      setIsLoggingOut(false);
    }
  }

  const healthStatusClass = health.loading ? 'checking' : health.data?.status === 'healthy' ? 'healthy' : 'unhealthy';
  const healthStatusLabel = health.loading ? 'Checking' : health.data?.status === 'healthy' ? 'Healthy' : 'Unhealthy';

  return (
    <div className="app-shell">
      <header className="hero">
        <div>
          <p className="eyebrow">Intelligent Document Query Engine</p>
          <h1>Ask questions over PDF documents</h1>
          <p className="hero-subtitle">
            Ask questions over PDF documents using retrieval-augmented generation.
          </p>
        </div>
        <div className="account-menu">
          <div>
            <span className="account-label">Signed in as</span>
            <strong>{user.display_name || user.email}</strong>
            {user.display_name ? <span>{user.email}</span> : null}
          </div>
          <button
            type="button"
            className="ghost-button"
            onClick={() => void handleLogout()}
            disabled={isLoggingOut}
          >
            {isLoggingOut ? 'Signing out...' : 'Sign out'}
          </button>
        </div>
      </header>

      <div className="view-switch" role="tablist" aria-label="Application view">
        {VIEW_OPTIONS.map((option) => (
          <button
            key={option.value}
            type="button"
            className={`view-button ${activeView === option.value ? 'is-active' : ''}`}
            onClick={() => handleViewChange(option.value)}
          >
            {option.label}
          </button>
        ))}
      </div>

      {activeView === 'query' ? (
        <>
      <div className="top-grid">
        <form className="panel" onSubmit={handleSubmit}>
          <div className="panel-header">
            <div>
              <p className="panel-kicker">Query setup</p>
              <h2>Document input</h2>
            </div>
            <span className="subtle-copy">
              {mode === 'upload' ? 'Upload, then ask' : `${questions.length} question${questions.length === 1 ? '' : 's'}`}
            </span>
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

          {mode === 'upload' && uploadedDocument ? (
            <div className="document-progress" role="status" aria-live="polite">
              <span className={`status-pill status-${uploadedDocument.status}`}>
                {uploadedDocument.status === 'ready' ? 'Ready' : uploadedDocument.status === 'failed' ? 'Failed' : 'Processing'}
              </span>
              <span>
                {uploadReady
                  ? 'Document ready. Ask a question below.'
                  : uploadedDocument.status === 'failed'
                    ? 'Document processing failed. Upload the PDF again to retry.'
                    : 'Processing your PDF. The question box will open when it is ready.'}
              </span>
              {pollingError ? (
                <div className="notice notice-error">
                  {pollingError}
                  <button type="button" className="ghost-button" onClick={() => setPollingError('')}>
                    Retry status
                  </button>
                </div>
              ) : null}
            </div>
          ) : null}

          {mode === 'upload' && duplicateDocument ? (
            <div className="notice duplicate-confirmation" role="alertdialog" aria-labelledby="duplicate-title">
              <strong id="duplicate-title">You've already uploaded this document.</strong>
              <p>{duplicateDocument.filename || 'Existing PDF'} · {duplicateDocument.status}</p>
              <div className="actions">
                <button type="button" className="secondary-button" onClick={handleOpenExisting} disabled={isSubmitting}>
                  Open existing
                </button>
                <button type="button" className="primary-button" onClick={() => void handleUploadAgain()} disabled={isSubmitting}>
                  {isSubmitting ? 'Uploading...' : 'Upload again'}
                </button>
              </div>
            </div>
          ) : null}

          <label className="field">
            <span className="field-label">{mode === 'upload' ? 'Question' : 'Questions'}</span>
            <textarea
              className="field-input field-textarea"
              placeholder={mode === 'upload'
                ? 'What is this document about?'
                : 'What is this document about?\nWhat are the key dates?\nWhich sections mention exclusions?'}
              value={questionsText}
              onChange={(event) => setQuestionsText(event.target.value)}
              disabled={mode === 'upload' && !uploadReady}
            />
            <span className="field-help">
              {mode === 'upload'
                ? uploadReady ? 'Ask one question at a time.' : 'Available after document processing finishes.'
                : 'Enter one question per line'}
            </span>
          </label>

          {formError ? <div className="notice notice-error">{formError}</div> : null}

          <div className="actions">
            <button type="submit" className="primary-button" disabled={isSubmitting || (mode === 'upload' && (uploadProcessing || Boolean(duplicateDocument)))}>
              {isSubmitting
                ? mode === 'upload' && !uploadReady ? 'Uploading...' : 'Running query...'
                : mode === 'upload' && !uploadReady ? 'Upload PDF' : 'Run Query'}
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
            <p className="support-copy">Query a PDF URL directly, or upload a PDF and ask questions after processing finishes.</p>
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
            {mode === 'upload' && !uploadReady ? 'Uploading PDF...' : 'Running document query...'}
          </div>
        ) : null}

        {!isSubmitting && !requestError && answers.length === 0 ? (
          <div className="empty-state">
            {mode === 'upload' && !uploadReady
              ? 'Upload a PDF and wait for processing before asking a question.'
              : 'Ask a question to see a grounded answer with source excerpts.'}
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
        </>
      ) : (
        <section className="panel history-panel">
          <div className="panel-header">
            <div>
              <p className="panel-kicker">History</p>
              <h2>{selectedHistoryDocument ? 'Document queries' : 'Documents'}</h2>
            </div>
            <div className="history-actions">
              {selectedHistoryDocument ? (
                <button
                  type="button"
                  className="ghost-button"
                  onClick={() => {
                    setSelectedHistoryDocument(null);
                    setHistoryQueries([]);
                    setHistoryError('');
                  }}
                  disabled={historyLoading}
                >
                  Back
                </button>
              ) : null}
              <button
                type="button"
                className="ghost-button"
                onClick={() => void loadHistoryDocuments()}
                disabled={historyLoading}
              >
                Refresh
              </button>
            </div>
          </div>

          {selectedHistoryDocument ? (
            <div className="history-context">
              <p className="section-label">Selected document</p>
              <h3>{formatDocumentTitle(selectedHistoryDocument)}</h3>
              <p>
                {selectedHistoryDocument.source_type} | {selectedHistoryDocument.chunk_count} chunks |{' '}
                {selectedHistoryDocument.query_count} queries
              </p>
            </div>
          ) : null}

          {historyError ? <div className="notice notice-error">{historyError}</div> : null}

          {historyLoading ? (
            <div className="loading-box" aria-live="polite">
              <span className="spinner" aria-hidden="true" />
              Loading history...
            </div>
          ) : null}

          {!historyLoading && !historyError && !selectedHistoryDocument && historyDocuments.length === 0 ? (
            <div className="empty-state">No persisted documents found for the current user.</div>
          ) : null}

          {!historyLoading && !historyError && selectedHistoryDocument && historyQueries.length === 0 ? (
            <div className="empty-state">No persisted queries found for this document.</div>
          ) : null}

          {!selectedHistoryDocument ? (
            <div className="history-list">
              {historyDocuments.map((document) => (
                <button
                  type="button"
                  className="history-row"
                  key={document.id}
                  onClick={() => void loadHistoryQueries(document)}
                >
                  <span>
                    <strong>{formatDocumentTitle(document)}</strong>
                    <span className="history-meta">
                      {document.source_type} | {document.status} | {formatDateTime(document.created_at)}
                    </span>
                  </span>
                  <span className="history-counts">
                    {document.chunk_count} chunks
                    <br />
                    {document.query_count} queries
                  </span>
                </button>
              ))}
            </div>
          ) : (
            <div className="history-list">
              {historyQueries.map((query) => (
                <article className="history-query" key={query.id}>
                  <div className="history-query-header">
                    <div>
                      <p className="section-label">Question</p>
                      <h3>{query.question}</h3>
                    </div>
                    <span className={`status-pill status-${query.status || 'error'}`}>
                      {query.is_abstained ? 'Abstained' : ANSWER_STATUS_LABELS[query.status] || query.status || 'Unknown'}
                    </span>
                  </div>
                  <p>{query.answer}</p>
                  <div className="history-meta">
                    {formatDateTime(query.created_at)}
                    {Number.isFinite(query.latency_ms) ? ` | ${Math.round(query.latency_ms)} ms` : ''}
                  </div>
                </article>
              ))}
            </div>
          )}
        </section>
      )}
    </div>
  );
}

function App() {
  const { status } = useAuth();

  if (status === 'loading') {
    return (
      <main className="auth-shell">
        <section className="auth-card auth-loading" aria-live="polite">
          <span className="spinner" aria-hidden="true" />
          <div>
            <p className="eyebrow">Intelligent Document Query Engine</p>
            <h1>Restoring your session</h1>
          </div>
        </section>
      </main>
    );
  }

  if (status === 'unauthenticated') {
    return <AuthScreen />;
  }

  return <AuthenticatedApp />;
}

export default App;
