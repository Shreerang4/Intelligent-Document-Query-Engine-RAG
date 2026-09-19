import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';

import {
  AUTH_REFRESH_LOCK_NAME,
  AuthRequestError,
  createAuthSession,
} from '../src/auth/authSession.js';


const OLD_AUTH = {
  access_token: 'old-access-jwt',
  token_type: 'bearer',
  expires_in: 600,
  user: { id: 'user-1', email: 'person@example.com', display_name: null },
};

const NEW_AUTH = {
  ...OLD_AUTH,
  access_token: 'new-access-jwt',
  user: { ...OLD_AUTH.user, display_name: 'Person' },
};

function jsonResponse(status, payload) {
  return {
    status,
    ok: status >= 200 && status < 300,
    headers: new Headers({ 'content-type': 'application/json' }),
    async json() {
      return payload;
    },
    async text() {
      return JSON.stringify(payload);
    },
  };
}

function emptyResponse(status = 204) {
  return {
    status,
    ok: status >= 200 && status < 300,
    headers: new Headers(),
    async json() {
      return null;
    },
    async text() {
      return '';
    },
  };
}

function createSerialLockManager() {
  let queueTail = Promise.resolve();
  return {
    requests: [],
    request(name, callback) {
      this.requests.push(name);
      const result = queueTail.then(() => callback({ name }));
      queueTail = result.catch(() => undefined);
      return result;
    },
  };
}

test('bootstrap refresh restores authenticated state and includes browser credentials', async () => {
  const calls = [];
  const session = createAuthSession({
    baseUrl: 'https://app.example',
    fetchImpl: async (url, options) => {
      calls.push({ url, options });
      return jsonResponse(200, OLD_AUTH);
    },
  });

  assert.equal(session.getSnapshot().status, 'loading');
  await session.bootstrap();

  assert.deepEqual(session.getSnapshot(), {
    status: 'authenticated',
    accessToken: 'old-access-jwt',
    user: OLD_AUTH.user,
    notice: '',
  });
  assert.equal(calls.length, 1);
  assert.equal(calls[0].url, 'https://app.example/auth/refresh');
  assert.equal(calls[0].options.method, 'POST');
  assert.equal(calls[0].options.credentials, 'include');
});

test('bootstrap refresh 401 produces unauthenticated state without another refresh', async () => {
  let refreshCalls = 0;
  const session = createAuthSession({
    fetchImpl: async () => {
      refreshCalls += 1;
      return jsonResponse(401, { detail: 'Invalid refresh credential.' });
    },
  });

  await session.bootstrap();

  assert.equal(refreshCalls, 1);
  assert.deepEqual(session.getSnapshot(), {
    status: 'unauthenticated',
    accessToken: null,
    user: null,
    notice: '',
  });
});

test('login and registration keep returned access JWT only in session memory', async () => {
  const bodies = [];
  const session = createAuthSession({
    fetchImpl: async (url, options) => {
      bodies.push({ url, body: JSON.parse(options.body) });
      return jsonResponse(200, url.endsWith('/login') ? OLD_AUTH : NEW_AUTH);
    },
  });

  await session.login({ email: 'person@example.com', password: 'private login password' });
  assert.equal(session.getSnapshot().accessToken, OLD_AUTH.access_token);

  await session.register({
    email: 'new@example.com',
    password: 'private register password',
    displayName: 'New Person',
  });
  assert.equal(session.getSnapshot().accessToken, NEW_AUTH.access_token);
  assert.deepEqual(bodies[1].body, {
    email: 'new@example.com',
    password: 'private register password',
    display_name: 'New Person',
  });
});

test('authentication errors expose only the backend safe message', async () => {
  const password = 'never render this password';
  const session = createAuthSession({
    fetchImpl: async () => jsonResponse(401, { detail: 'Invalid email or password.' }),
  });

  await assert.rejects(
    session.login({ email: 'person@example.com', password }),
    (error) => {
      assert.ok(error instanceof AuthRequestError);
      assert.equal(error.message, 'Invalid email or password.');
      assert.equal(error.message.includes(password), false);
      return true;
    },
  );
});

test('simultaneous 401 responses share one refresh and all retry with the new token', async () => {
  let releaseRefresh;
  const refreshGate = new Promise((resolve) => {
    releaseRefresh = resolve;
  });
  let markRefreshStarted;
  const refreshStarted = new Promise((resolve) => {
    markRefreshStarted = resolve;
  });
  let refreshCalls = 0;
  const lockManager = createSerialLockManager();
  const resourceAuthorization = [];
  const session = createAuthSession({
    lockManager,
    fetchImpl: async (url, options) => {
      if (url.endsWith('/auth/login')) {
        return jsonResponse(200, OLD_AUTH);
      }
      if (url.endsWith('/auth/refresh')) {
        refreshCalls += 1;
        markRefreshStarted();
        await refreshGate;
        return jsonResponse(200, NEW_AUTH);
      }
      resourceAuthorization.push(options.headers.get('Authorization'));
      return options.headers.get('Authorization') === 'Bearer old-access-jwt'
        ? jsonResponse(401, { detail: 'Expired.' })
        : jsonResponse(200, { ok: true });
    },
  });
  await session.login({ email: 'person@example.com', password: 'password' });

  const requests = [
    session.authenticatedFetch('/resource/a'),
    session.authenticatedFetch('/resource/b'),
    session.authenticatedFetch('/resource/c'),
  ];
  await refreshStarted;
  assert.equal(refreshCalls, 1);
  releaseRefresh();
  const responses = await Promise.all(requests);

  assert.deepEqual(responses.map((response) => response.status), [200, 200, 200]);
  assert.equal(refreshCalls, 1);
  assert.deepEqual(lockManager.requests, [AUTH_REFRESH_LOCK_NAME]);
  assert.deepEqual(resourceAuthorization, [
    'Bearer old-access-jwt',
    'Bearer old-access-jwt',
    'Bearer old-access-jwt',
    'Bearer new-access-jwt',
    'Bearer new-access-jwt',
    'Bearer new-access-jwt',
  ]);
});

test('RAG requests use the in-memory JWT and reuse the existing refresh retry path', async () => {
  const calls = [];
  let refreshCalls = 0;
  const session = createAuthSession({
    fetchImpl: async (url, options) => {
      calls.push({ url, authorization: options.headers.get('Authorization') });
      if (url.endsWith('/auth/login')) {
        return jsonResponse(200, OLD_AUTH);
      }
      if (url.endsWith('/auth/refresh')) {
        refreshCalls += 1;
        return jsonResponse(200, NEW_AUTH);
      }
      if (url.endsWith('/hackrx/run')) {
        return options.headers.get('Authorization') === 'Bearer old-access-jwt'
          ? jsonResponse(401, { detail: 'Invalid or missing access token.' })
          : jsonResponse(200, { answers: [] });
      }
      throw new Error(`Unexpected request: ${url}`);
    },
  });
  await session.login({ email: 'person@example.com', password: 'password' });

  const response = await session.authenticatedFetch('/hackrx/run', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ documents: 'https://example.com/a.pdf', questions: ['Question?'] }),
  });

  assert.equal(response.status, 200);
  assert.equal(refreshCalls, 1);
  assert.deepEqual(
    calls.filter((call) => call.url.endsWith('/hackrx/run')).map((call) => call.authorization),
    ['Bearer old-access-jwt', 'Bearer new-access-jwt'],
  );
  assert.equal(session.getSnapshot().accessToken, 'new-access-jwt');
});

test('refresh HTTP dispatch executes inside the Web Lock callback', async () => {
  let insideLock = false;
  let refreshCalls = 0;
  const lockManager = {
    request(name, callback) {
      assert.equal(name, AUTH_REFRESH_LOCK_NAME);
      insideLock = true;
      return Promise.resolve(callback({ name })).finally(() => {
        insideLock = false;
      });
    },
  };
  const session = createAuthSession({
    lockManager,
    fetchImpl: async (url) => {
      if (url.endsWith('/auth/refresh')) {
        refreshCalls += 1;
        assert.equal(insideLock, true);
      }
      return jsonResponse(200, OLD_AUTH);
    },
  });

  await session.refresh();

  assert.equal(refreshCalls, 1);
  assert.equal(insideLock, false);
});

test('two tab sessions serialize refresh dispatch and keep separate access JWTs', async () => {
  const lockManager = createSerialLockManager();
  let releaseFirstRefresh;
  const firstRefreshGate = new Promise((resolve) => {
    releaseFirstRefresh = resolve;
  });
  let markFirstRefreshStarted;
  const firstRefreshStarted = new Promise((resolve) => {
    markFirstRefreshStarted = resolve;
  });
  const dispatchedRequests = [];
  const tabAAuth = { ...OLD_AUTH, access_token: 'tab-a-access-jwt' };
  const tabBAuth = { ...NEW_AUTH, access_token: 'tab-b-access-jwt' };

  const fetchImpl = async (url) => {
    assert.equal(url.endsWith('/auth/refresh'), true);
    const requestNumber = dispatchedRequests.length + 1;
    dispatchedRequests.push(requestNumber);
    if (requestNumber === 1) {
      markFirstRefreshStarted();
      await firstRefreshGate;
      return jsonResponse(200, tabAAuth);
    }
    return jsonResponse(200, tabBAuth);
  };
  const tabA = createAuthSession({ fetchImpl, lockManager });
  const tabB = createAuthSession({ fetchImpl, lockManager });

  const tabARefresh = tabA.refresh();
  const tabBRefresh = tabB.refresh();
  await firstRefreshStarted;

  assert.deepEqual(dispatchedRequests, [1]);
  assert.deepEqual(lockManager.requests, [
    AUTH_REFRESH_LOCK_NAME,
    AUTH_REFRESH_LOCK_NAME,
  ]);

  releaseFirstRefresh();
  await Promise.all([tabARefresh, tabBRefresh]);

  assert.deepEqual(dispatchedRequests, [1, 2]);
  assert.equal(tabA.getSnapshot().accessToken, 'tab-a-access-jwt');
  assert.equal(tabB.getSnapshot().accessToken, 'tab-b-access-jwt');
  assert.notEqual(tabA.getSnapshot().accessToken, tabB.getSnapshot().accessToken);
});

test('refresh falls back cleanly when Web Locks is unavailable', async () => {
  let refreshCalls = 0;
  const session = createAuthSession({
    lockManager: null,
    fetchImpl: async (url) => {
      assert.equal(url.endsWith('/auth/refresh'), true);
      refreshCalls += 1;
      return jsonResponse(200, OLD_AUTH);
    },
  });

  await Promise.all([session.refresh(), session.refresh(), session.refresh()]);

  assert.equal(refreshCalls, 1);
  assert.equal(session.getSnapshot().accessToken, OLD_AUTH.access_token);
});

test('an original request retries only once and never loops on a second 401', async () => {
  let resourceCalls = 0;
  let refreshCalls = 0;
  const session = createAuthSession({
    fetchImpl: async (url) => {
      if (url.endsWith('/auth/login')) {
        return jsonResponse(200, OLD_AUTH);
      }
      if (url.endsWith('/auth/refresh')) {
        refreshCalls += 1;
        return jsonResponse(200, NEW_AUTH);
      }
      resourceCalls += 1;
      return jsonResponse(401, { detail: 'Still unauthorized.' });
    },
  });
  await session.login({ email: 'person@example.com', password: 'password' });

  const response = await session.authenticatedFetch('/always-unauthorized');

  assert.equal(response.status, 401);
  assert.equal(resourceCalls, 2);
  assert.equal(refreshCalls, 1);
});

test('failed refresh clears auth state and does not enter a refresh loop', async () => {
  let refreshCalls = 0;
  const session = createAuthSession({
    fetchImpl: async (url) => {
      if (url.endsWith('/auth/login')) {
        return jsonResponse(200, OLD_AUTH);
      }
      if (url.endsWith('/auth/refresh')) {
        refreshCalls += 1;
        return jsonResponse(401, { detail: 'Invalid refresh credential.' });
      }
      return jsonResponse(401, { detail: 'Expired access token.' });
    },
  });
  await session.login({ email: 'person@example.com', password: 'password' });

  await assert.rejects(session.authenticatedFetch('/resource'), AuthRequestError);

  assert.equal(refreshCalls, 1);
  assert.equal(session.getSnapshot().status, 'unauthenticated');
  assert.equal(session.getSnapshot().accessToken, null);
  assert.equal(session.getSnapshot().user, null);
});

test('logout sends the cookie-enabled request and clears memory after success', async () => {
  const calls = [];
  const session = createAuthSession({
    fetchImpl: async (url, options) => {
      calls.push({ url, options });
      if (url.endsWith('/auth/login')) {
        return jsonResponse(200, OLD_AUTH);
      }
      return emptyResponse();
    },
  });
  await session.login({ email: 'person@example.com', password: 'password' });

  const result = await session.logout();

  assert.deepEqual(result, { serverRevoked: true });
  assert.equal(calls.at(-1).url.endsWith('/auth/logout'), true);
  assert.equal(calls.at(-1).options.credentials, 'include');
  assert.equal(session.getSnapshot().status, 'unauthenticated');
  assert.equal(session.getSnapshot().accessToken, null);
});

test('logout failure clears local memory but does not claim server revocation', async () => {
  const session = createAuthSession({
    fetchImpl: async (url) => (
      url.endsWith('/auth/login')
        ? jsonResponse(200, OLD_AUTH)
        : jsonResponse(503, { detail: 'Authentication service is unavailable.' })
    ),
  });
  await session.login({ email: 'person@example.com', password: 'password' });

  await assert.rejects(session.logout(), AuthRequestError);

  assert.equal(session.getSnapshot().status, 'unauthenticated');
  assert.equal(session.getSnapshot().accessToken, null);
  assert.match(session.getSnapshot().notice, /could not confirm/i);
});

test('auth implementation contains no persistent token storage or refresh-cookie reads', async () => {
  const sources = await Promise.all([
    readFile(new URL('../src/auth/authSession.js', import.meta.url), 'utf8'),
    readFile(new URL('../src/auth/AuthContext.jsx', import.meta.url), 'utf8'),
    readFile(new URL('../src/auth/AuthScreen.jsx', import.meta.url), 'utf8'),
    readFile(new URL('../src/App.jsx', import.meta.url), 'utf8'),
    readFile(new URL('../src/api/client.js', import.meta.url), 'utf8'),
  ]);
  const combined = sources.join('\n');

  assert.doesNotMatch(combined, /localStorage|sessionStorage|indexedDB/i);
  assert.doesNotMatch(combined, /document\.cookie/i);
  assert.doesNotMatch(combined, /console\.(?:log|debug|info|warn|error)/);
  assert.doesNotMatch(combined, /dangerouslySetInnerHTML/);
  assert.doesNotMatch(sources.slice(0, 3).join('\n'), /BroadcastChannel|setTimeout|setInterval/);
});

test('React RAG and document callers use authenticatedFetch with no manual token UI', async () => {
  const [appSource, clientSource, documentSource] = await Promise.all([
    readFile(new URL('../src/App.jsx', import.meta.url), 'utf8'),
    readFile(new URL('../src/api/client.js', import.meta.url), 'utf8'),
    readFile(new URL('../src/api/documentWorkflow.js', import.meta.url), 'utf8'),
  ]);
  const combined = `${appSource}\n${clientSource}\n${documentSource}`;

  for (const path of [
    '/hackrx/run',
    '/history/documents',
  ]) {
    assert.match(appSource, new RegExp(`authenticatedFetch\\(['\"]${path.replace('/', '\\/')}`));
  }
  assert.match(documentSource, /authenticatedFetch\('\/documents\/upload'/);
  assert.match(documentSource, /authenticatedFetch\(\s*`\/documents\/\$\{encodeURIComponent\(documentId\)\}`/);
  assert.match(documentSource, /\/queries`/);
  assert.doesNotMatch(appSource, /\/hackrx\/upload-run/);
  assert.doesNotMatch(combined, /legacyRagApiFetch/);
  assert.doesNotMatch(combined, /API_TOKEN|API token|shared token/i);
  assert.doesNotMatch(appSource, /setToken|showToken|legacy-token-field|token-row/);
  assert.match(appSource, /status === 'unauthenticated'/);
  assert.match(appSource, /return <AuthScreen \/>/);
  assert.match(appSource, /return <AuthenticatedApp \/>/);
});
