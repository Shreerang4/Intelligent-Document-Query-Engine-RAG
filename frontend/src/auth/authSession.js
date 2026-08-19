import { API_BASE_URL, apiUrl, getErrorMessage, readResponseBody } from '../api/client.js';


export class AuthRequestError extends Error {
  constructor(message, { status = 0 } = {}) {
    super(message);
    this.name = 'AuthRequestError';
    this.status = status;
  }
}

const INITIAL_SNAPSHOT = Object.freeze({
  status: 'loading',
  accessToken: null,
  user: null,
  notice: '',
});

export const AUTH_REFRESH_LOCK_NAME = 'idqe-auth-refresh';

export function createAuthSession({
  fetchImpl = (...args) => globalThis.fetch(...args),
  baseUrl = API_BASE_URL,
  lockManager = globalThis.navigator?.locks,
} = {}) {
  let snapshot = INITIAL_SNAPSHOT;
  let refreshPromise = null;
  let bootstrapPromise = null;
  const listeners = new Set();

  function publish(nextSnapshot) {
    snapshot = Object.freeze(nextSnapshot);
    listeners.forEach((listener) => listener());
  }

  function setAuthenticated(payload) {
    if (!payload || typeof payload.access_token !== 'string' || !payload.access_token || !payload.user) {
      throw new AuthRequestError('The authentication response was incomplete.');
    }
    publish({
      status: 'authenticated',
      accessToken: payload.access_token,
      user: payload.user,
      notice: '',
    });
    return payload;
  }

  function setUnauthenticated(notice = '') {
    publish({
      status: 'unauthenticated',
      accessToken: null,
      user: null,
      notice,
    });
  }

  async function rawAuthRequest(path, { method = 'POST', body } = {}) {
    const headers = new Headers();
    const options = {
      method,
      credentials: 'include',
      headers,
    };
    if (body !== undefined) {
      headers.set('Content-Type', 'application/json');
      options.body = JSON.stringify(body);
    }

    let response;
    try {
      response = await fetchImpl(apiUrl(path, baseUrl), options);
    } catch (error) {
      const requestError = new AuthRequestError('Unable to reach the authentication service.');
      requestError.cause = error;
      throw requestError;
    }
    const payload = await readResponseBody(response);
    if (!response.ok) {
      throw new AuthRequestError(
        getErrorMessage(payload, `Authentication request failed with status ${response.status}.`),
        { status: response.status },
      );
    }
    return payload;
  }

  function runRefreshWithCrossTabLock() {
    const sendRefreshRequest = () => rawAuthRequest('/auth/refresh');
    if (!lockManager || typeof lockManager.request !== 'function') {
      return sendRefreshRequest();
    }

    return lockManager.request(
      AUTH_REFRESH_LOCK_NAME,
      () => sendRefreshRequest(),
    );
  }

  function refresh() {
    if (refreshPromise) {
      return refreshPromise;
    }

    refreshPromise = (async () => {
      try {
        const payload = await runRefreshWithCrossTabLock();
        return setAuthenticated(payload);
      } catch (error) {
        const notice = error instanceof AuthRequestError && error.status !== 401
          ? 'Your session could not be restored. Please sign in again.'
          : '';
        setUnauthenticated(notice);
        throw error;
      }
    })().finally(() => {
      refreshPromise = null;
    });

    return refreshPromise;
  }

  function bootstrap() {
    if (!bootstrapPromise) {
      bootstrapPromise = refresh().catch(() => null);
    }
    return bootstrapPromise;
  }

  async function login({ email, password }) {
    const payload = await rawAuthRequest('/auth/login', {
      body: { email, password },
    });
    return setAuthenticated(payload);
  }

  async function register({ email, password, displayName }) {
    const body = { email, password };
    if (displayName) {
      body.display_name = displayName;
    }
    const payload = await rawAuthRequest('/auth/register', { body });
    return setAuthenticated(payload);
  }

  async function logout() {
    let serverRevoked = false;
    try {
      await rawAuthRequest('/auth/logout');
      serverRevoked = true;
      setUnauthenticated();
      return { serverRevoked };
    } catch (error) {
      setUnauthenticated(
        'Signed out locally, but the server could not confirm refresh-session revocation.',
      );
      throw error;
    }
  }

  async function authenticatedFetch(path, options = {}) {
    const requestOnce = (accessToken) => {
      const headers = new Headers(options.headers || {});
      if (accessToken) {
        headers.set('Authorization', `Bearer ${accessToken}`);
      }
      return fetchImpl(apiUrl(path, baseUrl), {
        ...options,
        credentials: options.credentials || 'same-origin',
        headers,
      });
    };

    const firstResponse = await requestOnce(snapshot.accessToken);
    if (firstResponse.status !== 401) {
      return firstResponse;
    }

    await refresh();
    return requestOnce(snapshot.accessToken);
  }

  async function getCurrentUser() {
    const response = await authenticatedFetch('/auth/me');
    const payload = await readResponseBody(response);
    if (!response.ok) {
      throw new AuthRequestError(
        getErrorMessage(payload, `Current-user request failed with status ${response.status}.`),
        { status: response.status },
      );
    }
    publish({ ...snapshot, user: payload });
    return payload;
  }

  return Object.freeze({
    subscribe(listener) {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
    getSnapshot() {
      return snapshot;
    },
    bootstrap,
    login,
    register,
    refresh,
    logout,
    authenticatedFetch,
    getCurrentUser,
  });
}

export const browserAuthSession = createAuthSession();
