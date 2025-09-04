// API layer (DIP): pure functions for server endpoints
export async function fetchJson(url, options = {}) {
  const res = await fetch(url, options);
  const ct = res.headers.get('content-type') || '';
  if (!ct.includes('application/json')) {
    const text = await res.text();
    throw new Error(`Unexpected response (${res.status}): ${text}`);
  }
  const data = await res.json();
  if (!res.ok) {
    const msg = data && (data.message || data.error);
    throw new Error(msg || `Request failed: ${res.status}`);
  }
  return data;
}

export async function getJson(url) {
  const res = await fetch(url);
  return res.json();
}

export async function postJson(url, body) {
  return fetchJson(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body || {})
  });
}

export async function uploadVideos({ reference, current }) {
  const form = new FormData();
  if (reference) form.append('reference', reference);
  if (current) form.append('current', current);
  const res = await fetch('/upload_videos', { method: 'POST', body: form });
  return res.json();
}

export function setVideos({ reference_file, current_file, device }) {
  return postJson('/set_videos', { reference_file, current_file, device });
}

export async function analyze() {
  return postJson('/analyze');
}

export async function listVideos() {
  return getJson('/list_videos');
}

export async function systemUsage() {
  return getJson('/system_usage');
}

export async function chatbotStatus() {
  return getJson('/chatbot_status');
}

export async function initChatbot() {
  try { return await postJson('/init_chatbot'); } catch (e) { return { status: 'error', message: e.message }; }
}

export async function getMessages() {
  return getJson('/messages');
}

export async function sendMessage(message) {
  return postJson('/messages', { message });
}

export async function sendMessageStream(message, onChunk) {
  const res = await fetch('/messages', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Accept': 'text/event-stream'
    },
    body: JSON.stringify({ message })
  });

  const reader = res.body && res.body.getReader ? res.body.getReader() : null;
  if (!reader) return;
  const decoder = new TextDecoder('utf-8');
  let buffer = '';
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const parts = buffer.split('\n\n');
    buffer = parts.pop() || '';
    for (const part of parts) {
      if (part.startsWith('data:')) {
        const token = part.slice(5).trim();
        if (token === '[DONE]') return;
        onChunk(token);
      }
    }
  }
}

