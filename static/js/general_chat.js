const box = document.getElementById('chat-messages');
const input = document.getElementById('chat-input');
const btn = document.getElementById('send-btn');

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
    const res = await fetch('/chat_messages', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text })
    });
    const data = await res.json();
    appendTyping('assistant', data.reply || '');
  } catch (err) {
    append('assistant', 'エラーが発生しました');
  }
}

btn.addEventListener('click', send);
input.addEventListener('keydown', e => {
  if (e.key === 'Enter') send();
});

loadMessages();
