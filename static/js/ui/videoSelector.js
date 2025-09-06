import { listVideos, uploadVideos, setVideos, analyze } from '../services/api.js';

export class VideoSelectorController {
  constructor({ currentDevice, hasChat, backend }) {
    this.currentDevice = currentDevice;
    this.currentBackend = backend;
    this.hasChat = hasChat;
  }

  async loadList() {
    const files = await listVideos();
    const refSel = document.getElementById('ref-file');
    const curSel = document.getElementById('cur-file');
    if (!refSel || !curSel) return;
    refSel.innerHTML = '';
    curSel.innerHTML = '';
    files.forEach(f => {
      const o1 = document.createElement('option'); o1.value = f; o1.textContent = f; refSel.appendChild(o1);
      const o2 = document.createElement('option'); o2.value = f; o2.textContent = f; curSel.appendChild(o2);
    });
  }

  setDefaults({ refName, curName }) {
    const refSel = document.getElementById('ref-file');
    const curSel = document.getElementById('cur-file');
    const devSel = document.getElementById('device-select');
    const backendSel = document.getElementById('backend-select');
    if (refSel) refSel.value = refName || '';
    if (curSel) curSel.value = curName || '';
    if (devSel) devSel.value = this.currentDevice || 'CPU';
    if (backendSel) backendSel.value = this.currentBackend || 'auto';
  }

  bindProcess() {
    const btn = document.getElementById('process-btn');
    if (!btn) return;
    btn.onclick = async () => {
      const refUp = document.getElementById('ref-upload').files[0];
      const curUp = document.getElementById('cur-upload').files[0];
      const refSel = document.getElementById('ref-file').value;
      const curSel = document.getElementById('cur-file').value;
      const device = document.getElementById('device-select').value;
      const backend = document.getElementById('backend-select').value;

      let refFile = refSel, curFile = curSel;
      if (refUp || curUp) {
        const uploaded = await uploadVideos({ reference: refUp, current: curUp });
        if (uploaded.reference_file) refFile = uploaded.reference_file;
        if (uploaded.current_file) curFile = uploaded.current_file;
      }

      await setVideos({ reference_file: refFile, current_file: curFile, device, backend });

      const status = document.getElementById('status');
      if (status) status.textContent = '動画解析中です。完了までお待ちください';

      const chatStatus = document.getElementById('chat-status');
      if (this.hasChat && chatStatus) {
        chatStatus.textContent = '動画解析中です。完了までお待ちください...';
        chatStatus.className = 'chat-status preparing';
        const input = document.getElementById('chat-input');
        const send = document.getElementById('send-btn');
        if (input) input.disabled = true;
        if (send) send.disabled = true;
      }

      let dots = 0;
      const timer = setInterval(() => {
        dots = (dots + 1) % 4;
        if (status) status.textContent = '動画解析中です。完了までお待ちください' + '.'.repeat(dots);
      }, 500);

      try {
        const res = await analyze();
        clearInterval(timer);
        if (res && res.error) {
          if (status) status.textContent = '解析に失敗しました: ' + res.error;
          return;
        }
        if (status) status.textContent = '動画解析が完了しました';
        if (this.hasChat && chatStatus) {
          chatStatus.textContent = 'メッセージ生成中...';
          chatStatus.className = 'chat-status generating';
        }
        setTimeout(() => location.reload(), 1000);
      } catch (e) {
        clearInterval(timer);
        if (status) status.textContent = '解析に失敗しました';
      }
    };
  }
}

