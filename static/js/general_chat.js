const box = document.getElementById('chat-messages');
const input = document.getElementById('chat-input');
const btn = document.getElementById('send-btn');
const backendSel = document.getElementById('backend-select');
const deviceSel = document.getElementById('device-select');
const latencyEl = document.getElementById('latency');

function append(role, text) {
  const p = document.createElement('p');
  const prefix = role === 'user' ? 'あなた: ' : 'AI: ';
  p.textContent = prefix + text;
  p.classList.add('chat-msg', role === 'user' ? 'user-msg' : 'assistant-msg');
  box.appendChild(p);
  box.scrollTop = box.scrollHeight;
}

function appendTyping(role, text, speed = 30) {
  const p = document.createElement('p');
  const prefix = role === 'user' ? 'あなた: ' : 'AI: ';
  p.textContent = prefix;
  p.classList.add('chat-msg', role === 'user' ? 'user-msg' : 'assistant-msg');
  box.appendChild(p);
  box.scrollTop = box.scrollHeight;
  let i = 0;
  function typeNext() {
    if (i < text.length) {
      p.textContent += text.charAt(i);
      i += 1;
      box.scrollTop = box.scrollHeight;
      setTimeout(typeNext, speed);
    }
  }
  typeNext();
}

async function loadMessages() {
  try {
    const res = await fetch('/chat_messages');
    const msgs = await res.json();
    msgs.forEach(m => append(m.role, m.content));
  } catch (err) {
    console.error('Failed to load messages', err);
  }
}

async function send() {
  const text = input.value.trim();
  if (!text) return;
  append('user', text);
  input.value = '';
  try {
    const t0 = performance.now();
    const res = await fetch('/chat_messages', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text })
    });
    const data = await res.json();
    appendTyping('assistant', data.reply || '');
    const elapsed = typeof data.elapsed_ms === 'number' ? data.elapsed_ms : Math.round(performance.now() - t0);
    if (latencyEl) latencyEl.textContent = `最終応答時間: ${elapsed} ms`;
  } catch (err) {
    append('assistant', 'エラーが発生しました');
  }
}

btn.addEventListener('click', send);
input.addEventListener('keydown', e => {
  if (e.key === 'Enter') send();
});

async function loadChatSettings() {
  try {
    const res = await fetch('/chat_settings');
    const cfg = await res.json();
    if (backendSel && cfg.backend) {
      const val = (cfg.backend || '').toLowerCase();
      if ([...backendSel.options].some(o => o.value === val)) backendSel.value = val;
    }
    if (deviceSel && cfg.openvino_device) {
      const dev = cfg.openvino_device.toUpperCase();
      if ([...deviceSel.options].some(o => o.value === dev)) deviceSel.value = dev;
      else deviceSel.value = 'CPU';
    }
  } catch (e) {
    console.warn('Failed to load chat settings', e);
  }
}

async function applyChatSettings() {
  const payload = {
    backend: backendSel ? backendSel.value : undefined,
    device: deviceSel ? deviceSel.value : undefined,
  };
  try {
    await fetch('/chat_settings', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    // Clear local chat view after backend/device change to avoid confusion
    box.innerHTML = '';
    append('assistant', '設定を適用しました。新しいバックエンド/デバイスで応答します。');
  } catch (e) {
    console.warn('Failed to apply chat settings', e);
  }
}

if (backendSel) backendSel.addEventListener('change', applyChatSettings);
if (deviceSel) deviceSel.addEventListener('change', applyChatSettings);

loadChatSettings().then(loadMessages);
