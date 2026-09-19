import { getErrorMessage, readResponseBody } from './client.js';

async function readDocumentResponse(response, fallback) {
  const payload = await readResponseBody(response);
  if (!response.ok) {
    throw new Error(getErrorMessage(payload, fallback));
  }
  return payload;
}

export async function uploadDocument(authenticatedFetch, file, uploadRequestId) {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('upload_request_id', uploadRequestId);
  const response = await authenticatedFetch('/documents/upload', {
    method: 'POST',
    body: formData,
  });
  return readDocumentResponse(response, 'Document upload failed.');
}

export async function getDocumentStatus(authenticatedFetch, documentId, signal) {
  const response = await authenticatedFetch(
    `/documents/${encodeURIComponent(documentId)}`,
    { signal },
  );
  return readDocumentResponse(response, 'Document status is unavailable.');
}

export async function queryDocument(authenticatedFetch, documentId, question) {
  const response = await authenticatedFetch(
    `/documents/${encodeURIComponent(documentId)}/queries`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question }),
    },
  );
  const payload = await readDocumentResponse(response, 'Document query failed.');
  return { payload, response };
}
