const viteEnvironment = import.meta.env || {};

export const API_BASE_URL = (viteEnvironment.VITE_API_BASE_URL || '').trim().replace(/\/$/, '');

export function apiUrl(path, baseUrl = API_BASE_URL) {
  return `${baseUrl}${path}`;
}

export async function readResponseBody(response) {
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

export function getErrorMessage(payload, fallbackMessage) {
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

export function rawApiFetch(path, options = {}) {
  return fetch(apiUrl(path), options);
}
