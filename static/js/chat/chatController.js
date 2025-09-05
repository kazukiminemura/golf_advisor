import { chatbotStatus, getMessages, sendMessageStream, initChatbot } from '../services/api.js';
import { setStatusPreparing, setStatusGenerating, setStatusReady, setStatusError } from './statusView.js';

export class ChatController {
  constructor({ enabled, hasResults }) {
    this.enabled = enabled;
    this.hasResults = hasResults;
    this.ready = false;
    this.box = document.getElementById('chat-messages');
    this.input = document.getElementById('chat-input');
    this.sendBtn = document.getElementById('send-btn');
    this.statusDiv = document.getElementById('chat-status');
  }

  cleanLLMText(text) {
    if (!text) return '';
    let t = text;
    // Remove visible speaker prefixes
    t = t.replace(/^\s*(あなた|コーチ)[:：]\s*/gm, '');
    // Remove Markdown heading markers like ### Heading
    t = t.replace(/(^|\n)\s*#{1,6}\s+/g, (m, p1) => (p1 ? '\n' : ''));
    return t;
  }

  ensureGeneratingPlaceholder() {
    if (!this.box) return;
    if (!document.getElementById('init-loading')) {
      const p = document.createElement('p');
      p.id = 'init-loading';
      p.textContent = 'メッセージ生成中...';
      p.style.opacity = '0.7';
      p.className = 'chat-msg assistant-msg';
      this.box.appendChild(p);
      this.box.scrollTop = this.box.scrollHeight;
    }
  }

  removeGeneratingPlaceholder() {
    const p = document.getElementById('init-loading');
    if (p && p.parentElement) p.parentElement.removeChild(p);
  }

  setInteractable(enabled) {
    if (this.input) this.input.disabled = !enabled;
    if (this.sendBtn) this.sendBtn.disabled = !enabled;
  }

  typeText(el, text, speed = 30) {
    let i = 0;
    const step = () => {
      if (i < text.length) {
        el.textContent += text.charAt(i);
        i += 1;
        if (this.box) this.box.scrollTop = this.box.scrollHeight;
        setTimeout(step, speed);
      }
    };
    step();
  }

  async updateStatus() {
    try {
      const status = await chatbotStatus();
      if (status.initialized) {
        const hasRealMsgs = this.box ? this.box.querySelectorAll('.chat-msg').length > 0 : false;
        if (hasRealMsgs) {
          setStatusReady(this.statusDiv, 'チャットボットの準備ができました');
          this.setInteractable(true);
          this.ready = true;
        } else {
          setStatusGenerating(this.statusDiv, 'メッセージ生成中...');
          this.setInteractable(false);
          this.ready = false;
          this.ensureGeneratingPlaceholder();
        }
      } else {
        setStatusPreparing(this.statusDiv);
        this.setInteractable(false);
        this.ready = false;
      }
    } catch (e) {
      setStatusError(this.statusDiv);
      this.ready = false;
    }
  }

  async loadMessages() {
    try {
      const msgs = await getMessages();
      if (!this.box) return;
      this.box.innerHTML = '';
      msgs.forEach(m => {
        const p = document.createElement('p');
        p.textContent = this.cleanLLMText(m.content || '');
        p.className = 'chat-msg ' + (m.role === 'user' ? 'user-msg' : 'assistant-msg');
        this.box.appendChild(p);
      });
      this.box.scrollTop = this.box.scrollHeight;
      if (msgs.length > 0) this.removeGeneratingPlaceholder();
    } catch (e) {
      console.error('Failed to load messages:', e);
    }
  }

  bindSend() {
    if (!this.sendBtn || !this.input) return;
    this.sendBtn.onclick = async () => {
      if (!this.ready) return;
      const text = this.input.value.trim();
      if (!text) return;
      const userP = document.createElement('p');
      userP.textContent = this.cleanLLMText(text);
      userP.className = 'chat-msg user-msg';
      this.box.appendChild(userP);
      this.input.value = '';

      const coachP = document.createElement('p');
      coachP.textContent = '';
      coachP.className = 'chat-msg assistant-msg';
      this.box.appendChild(coachP);
      this.box.scrollTop = this.box.scrollHeight;

      try {
        let full = '';
        await sendMessageStream(text, (chunk) => {
          full += chunk;
        });
        // Remove any accidental prefixes inside streamed content
        this.typeText(coachP, this.cleanLLMText(full));
      } catch (e) {
        coachP.textContent = '返答の取得に失敗しました';
        coachP.style.color = 'red';
      }
    };

    this.input.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey && this.ready) {
        e.preventDefault();
        this.sendBtn.click();
      }
    });
  }

  async init() {
    if (!this.enabled || !this.input) return; // chat panel might be hidden

    await this.updateStatus();
    if (this.hasResults) {
      setStatusGenerating(this.statusDiv, 'メッセージ生成中...');
      this.ensureGeneratingPlaceholder();
      await initChatbot();
    }
    await this.updateStatus();
    await this.loadMessages();
    this.bindSend();

    // background refresh
    setInterval(() => this.updateStatus(), 5000);
  }
}

