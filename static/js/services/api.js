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

export function setVideos({ reference_file, current_file, device, backend }) {
  return postJson('/set_videos', { reference_file, current_file, device, backend });
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
  return new Promise((resolve, reject) => {
    const protocol = location.protocol === 'https:' ? 'wss' : 'ws';
    const ws = new WebSocket(`${protocol}://${location.host}/ws/messages`);
    ws.onopen = () => {
      ws.send(JSON.stringify({ message }));
    };
    ws.onmessage = (event) => {
      const token = event.data;
      if (token === '[DONE]') {
        ws.close();
        resolve();
      } else {
        onChunk(token);
      }
    };
    ws.onerror = (err) => {
      reject(err);
    };
  });
}

