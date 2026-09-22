import { getErrorMessage, readResponseBody } from './client.js';

export class DuplicateDocumentError extends Error {
  constructor(document) {
    super("You've already uploaded this document.");
    this.name = 'DuplicateDocumentError';
    this.document = document;
  }
}

async function readDocumentResponse(response, fallback) {
  const payload = await readResponseBody(response);
  if (!response.ok) {
    throw new Error(getErrorMessage(payload, fallback));
  }
  return payload;
}

export async function uploadDocument(authenticatedFetch, file, uploadRequestId, allowDuplicate = false) {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('upload_request_id', uploadRequestId);
  if (allowDuplicate) formData.append('allow_duplicate', 'true');
  const response = await authenticatedFetch('/documents/upload', {
    method: 'POST',
    body: formData,
  });
  const payload = await readResponseBody(response);
  if (response.status === 409 && payload?.code === 'duplicate_document' && payload.document) {
    throw new DuplicateDocumentError(payload.document);
  }
  if (!response.ok) {
    throw new Error(getErrorMessage(payload, 'Document upload failed.'));
  }
  return payload;
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
