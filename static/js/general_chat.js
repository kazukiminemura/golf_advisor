const box = document.getElementById('chat-messages');
const input = document.getElementById('chat-input');
const btn = document.getElementById('send-btn');

function append(role, text) {
  const p = document.createElement('p');
  const prefix = role === 'user' ? 'あなた: ' : 'AI: ';
  p.textContent = prefix + text;
  p.className = 'chat-msg';
  box.appendChild(p);
  box.scrollTop = box.scrollHeight;
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
    const res = await fetch('/chat_messages', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text })
    });
    const data = await res.json();
    append('assistant', data.reply || '');
  } catch (err) {
    append('assistant', 'エラーが発生しました');
  }
}

btn.addEventListener('click', send);
input.addEventListener('keydown', e => {
  if (e.key === 'Enter') send();
});

loadMessages();
