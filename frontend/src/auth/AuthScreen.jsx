import { useState } from 'react';
import { useAuth } from './AuthContext.jsx';


export default function AuthScreen() {
  const { login, register, notice } = useAuth();
  const [mode, setMode] = useState('login');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [displayName, setDisplayName] = useState('');
  const [error, setError] = useState('');
  const [submitting, setSubmitting] = useState(false);

  function switchMode(nextMode) {
    setMode(nextMode);
    setPassword('');
    setDisplayName('');
    setError('');
  }

  async function handleSubmit(event) {
    event.preventDefault();
    setError('');
    setSubmitting(true);
    try {
      if (mode === 'register') {
        await register({ email, password, displayName: displayName.trim() });
      } else {
        await login({ email, password });
      }
    } catch (requestError) {
      setError(
        requestError instanceof Error
          ? requestError.message
          : 'Authentication failed. Please try again.',
      );
    } finally {
      setSubmitting(false);
    }
  }

  const registering = mode === 'register';

  return (
    <main className="auth-shell">
      <section className="auth-card" aria-labelledby="auth-title">
        <div className="auth-intro">
          <p className="eyebrow">Intelligent Document Query Engine</p>
          <h1 id="auth-title">{registering ? 'Create your account' : 'Welcome back'}</h1>
          <p>
            {registering
              ? 'Use a strong passphrase to start a private session.'
              : 'Sign in to continue to your document workspace.'}
          </p>
        </div>

        <div className="auth-mode-switch" role="tablist" aria-label="Authentication mode">
          <button
            type="button"
            className={mode === 'login' ? 'is-active' : ''}
            onClick={() => switchMode('login')}
          >
            Sign in
          </button>
          <button
            type="button"
            className={mode === 'register' ? 'is-active' : ''}
            onClick={() => switchMode('register')}
          >
            Register
          </button>
        </div>

        <form onSubmit={handleSubmit}>
          {registering ? (
            <label className="field">
              <span className="field-label">Display name <span className="optional-label">Optional</span></span>
              <input
                className="field-input"
                type="text"
                autoComplete="name"
                maxLength={255}
                value={displayName}
                onChange={(event) => setDisplayName(event.target.value)}
              />
            </label>
          ) : null}

          <label className="field">
            <span className="field-label">Email</span>
            <input
              className="field-input"
              type="email"
              autoComplete="email"
              required
              value={email}
              onChange={(event) => setEmail(event.target.value)}
            />
          </label>

          <label className="field">
            <span className="field-label">Password</span>
            <input
              className="field-input"
              type="password"
              autoComplete={registering ? 'new-password' : 'current-password'}
              required
              minLength={registering ? 15 : 1}
              maxLength={128}
              value={password}
              onChange={(event) => setPassword(event.target.value)}
            />
            {registering ? <span className="field-help">Use 15 to 128 characters.</span> : null}
          </label>

          {notice ? <div className="notice notice-warning">{notice}</div> : null}
          {error ? <div className="notice notice-error" role="alert">{error}</div> : null}

          <button type="submit" className="primary-button auth-submit" disabled={submitting}>
            {submitting ? 'Please wait...' : registering ? 'Create account' : 'Sign in'}
          </button>
        </form>
      </section>
    </main>
  );
}
