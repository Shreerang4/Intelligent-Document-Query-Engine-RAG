import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';

import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';


const currentDirectory = path.dirname(fileURLToPath(import.meta.url));
const sourceRoot = path.resolve(currentDirectory, '../src');


test('React renders attacker-like user content as escaped text', () => {
  const payload = '<img src=x onerror="globalThis.__idqeXssExecuted=true"><script>alert(1)</script>';
  const markup = renderToStaticMarkup(React.createElement('p', null, payload));

  assert.match(markup, /&lt;img/);
  assert.match(markup, /&lt;script&gt;/);
  assert.doesNotMatch(markup, /<img/);
  assert.doesNotMatch(markup, /<script>/);
});


test('frontend source contains no executable HTML or dynamic script sinks', () => {
  const sourceFiles = [
    'App.jsx',
    'auth/AuthScreen.jsx',
    'auth/AuthContext.jsx',
    'auth/authSession.js',
    'api/client.js',
    'main.jsx',
  ];
  const combinedSource = sourceFiles
    .map((relativePath) => fs.readFileSync(path.join(sourceRoot, relativePath), 'utf8'))
    .join('\n');

  for (const forbiddenPattern of [
    /dangerouslySetInnerHTML/,
    /\.innerHTML\s*=/,
    /\beval\s*\(/,
    /new\s+Function\b/,
    /createElement\s*\(\s*['"]script['"]/,
  ]) {
    assert.doesNotMatch(combinedSource, forbiddenPattern);
  }
});
