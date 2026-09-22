import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';

import { DuplicateDocumentError, getDocumentStatus, queryDocument, uploadDocument } from '../src/api/documentWorkflow.js';


function jsonResponse(payload, status = 200) {
  return {
    ok: status >= 200 && status < 300,
    status,
    headers: new Headers({ 'content-type': 'application/json' }),
    json: async () => payload,
  };
}

test('upload sends a PDF and idempotency key to the async endpoint', async () => {
  const calls = [];
  const authenticatedFetch = async (path, options) => {
    calls.push({ path, options });
    return jsonResponse({ document_id: 'doc-1', status: 'queued' }, 202);
  };

  const document = await uploadDocument(
    authenticatedFetch,
    new Blob(['%PDF-1.7'], { type: 'application/pdf' }),
    'request-1',
  );

  assert.equal(document.status, 'queued');
  assert.equal(calls[0].path, '/documents/upload');
  assert.equal(calls[0].options.method, 'POST');
  assert.equal(calls[0].options.body.get('upload_request_id'), 'request-1');
  assert.equal(calls[0].options.body.has('allow_duplicate'), false);
  assert.equal(calls[0].options.body.get('file').type, 'application/pdf');
});

test('status polling and document queries use the selected document ID', async () => {
  const calls = [];
  const authenticatedFetch = async (path, options) => {
    calls.push({ path, options });
    return jsonResponse(path.endsWith('/queries')
      ? { answers: [{ answer: 'Grounded answer', sources: [{ page: 1, chunk_id: 2, excerpt: 'Evidence' }] }] }
      : { document_id: 'doc/1', status: 'ready' });
  };
  const signal = new AbortController().signal;

  const document = await getDocumentStatus(authenticatedFetch, 'doc/1', signal);
  const result = await queryDocument(authenticatedFetch, document.document_id, 'What happened?');

  assert.equal(document.status, 'ready');
  assert.equal(calls[0].path, '/documents/doc%2F1');
  assert.equal(calls[0].options.signal, signal);
  assert.equal(calls[1].path, '/documents/doc%2F1/queries');
  assert.deepEqual(JSON.parse(calls[1].options.body), { question: 'What happened?' });
  assert.equal(result.payload.answers[0].sources[0].excerpt, 'Evidence');
});

test('document API failures use the backend safe message', async () => {
  const authenticatedFetch = async () => jsonResponse({ detail: 'Document processing failed.' }, 409);
  await assert.rejects(
    queryDocument(authenticatedFetch, 'doc-1', 'Why?'),
    /Document processing failed\./,
  );
});

test('duplicate response provides an existing document for the confirmation choice', async () => {
  const existing = { document_id: 'existing-1', filename: 'report.pdf', status: 'ready' };
  const authenticatedFetch = async () => jsonResponse({
    code: 'duplicate_document', document: existing,
  }, 409);

  await assert.rejects(
    uploadDocument(authenticatedFetch, new Blob(['%PDF-1.7']), 'request-1'),
    (error) => error instanceof DuplicateDocumentError && error.document === existing,
  );
});

test('Upload again sends allow_duplicate=true with the same request ID', async () => {
  const calls = [];
  const authenticatedFetch = async (path, options) => {
    calls.push({ path, options });
    return jsonResponse({ document_id: 'new-1', status: 'queued' }, 202);
  };
  const document = await uploadDocument(
    authenticatedFetch, new Blob(['%PDF-1.7']), 'request-1', true,
  );

  assert.equal(document.document_id, 'new-1');
  assert.equal(calls[0].options.body.get('upload_request_id'), 'request-1');
  assert.equal(calls[0].options.body.get('allow_duplicate'), 'true');
});

test('generic upload failures remain ordinary errors', async () => {
  const authenticatedFetch = async () => jsonResponse({ detail: 'Upload unavailable.' }, 503);
  await assert.rejects(
    uploadDocument(authenticatedFetch, new Blob(['%PDF-1.7']), 'request-1'),
    (error) => !(error instanceof DuplicateDocumentError) && error.message === 'Upload unavailable.',
  );
});

test('upload view gates questions until ready and cancels status polling', async () => {
  const source = await readFile(new URL('../src/App.jsx', import.meta.url), 'utf8');
  assert.match(source, /disabled=\{mode === 'upload' && !uploadReady\}/);
  assert.match(source, /window\.setTimeout\(pollStatus, 2000\)/);
  assert.match(source, /window\.clearTimeout\(timer\)/);
  assert.match(source, /controller\.abort\(\)/);
});

test('duplicate choice opens the returned document or retries an intentional upload', async () => {
  const source = await readFile(new URL('../src/App.jsx', import.meta.url), 'utf8');
  assert.match(source, /You've already uploaded this document\./);
  assert.match(source, /function handleOpenExisting\(\) \{\s*setUploadedDocument\(duplicateDocument\)/);
  assert.match(source, /function handleUploadAgain\(\)[\s\S]*?submitUpload\(true\)/);
  assert.match(source, /uploadedDocument\?\.status === 'ready'/);
  assert.match(source, /uploadedDocument\?\.status === 'queued' \|\| uploadedDocument\?\.status === 'processing'/);
});
