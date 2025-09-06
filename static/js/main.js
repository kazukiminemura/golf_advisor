import { setupCanvas } from './video/canvasPlayer.js';
import { bindSynchronizedControls } from './video/playerControls.js';
import { UsageController } from './usage/usageController.js';
import { ChatController } from './chat/chatController.js';
import { VideoSelectorController } from './ui/videoSelector.js';

// Read config injected by template
const cfg = window.APP_CONFIG || {};

// Fix UI labels per request
try {
  const devLabel = document.querySelector('label[for="device-select"]');
  if (devLabel) devLabel.textContent = '姿勢推定デバイス';
  const beLabel = document.querySelector('label[for="backend-select"]');
  if (beLabel) beLabel.textContent = 'チャットボットバックエンド';
} catch (_) { /* no-op */ }

// Load current chatbot settings (backend/device) into controls
async function loadChatSettingsIntoControls() {
  const backendSel = document.getElementById('backend-select');
  const chatDeviceSel = document.getElementById('chat-device-select');
  if (!backendSel && !chatDeviceSel) return;
  try {
    const res = await fetch('/chat_settings');
    const cfg = await res.json();
    if (backendSel && cfg.backend) {
      const val = (cfg.backend || '').toLowerCase();
      if ([...backendSel.options].some(o => o.value === val)) backendSel.value = val;
    }
    if (chatDeviceSel && cfg.openvino_device) {
      const dev = (cfg.openvino_device || 'CPU').toUpperCase();
      if ([...chatDeviceSel.options].some(o => o.value === dev)) chatDeviceSel.value = dev;
      else chatDeviceSel.value = 'CPU';
    }
  } catch (e) {
    // ignore
  }
}

async function applyChatSettingsFromControls() {
  const backendSel = document.getElementById('backend-select');
  const chatDeviceSel = document.getElementById('chat-device-select');
  const payload = {
    backend: backendSel ? backendSel.value : undefined,
    device: chatDeviceSel ? chatDeviceSel.value : undefined,
  };
  try {
    await fetch('/chat_settings', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    // Clear chat messages to avoid context confusion after backend/device change
    const box = document.getElementById('chat-messages');
    if (box) {
      box.innerHTML = '';
      const p = document.createElement('p');
      p.textContent = '設定を適用しました。新しいバックエンド/デバイスで応答します。';
      p.className = 'chat-msg assistant-msg';
      box.appendChild(p);
      box.scrollTop = box.scrollHeight;
    }
  } catch (e) {
    // ignore
  }
}

// Initialize video selector and analysis flow
const videoSelector = new VideoSelectorController({ currentDevice: cfg.device, hasChat: cfg.chatbotEnabled, backend: cfg.llmBackend });
await videoSelector.loadList();
videoSelector.setDefaults({ refName: cfg.refVideoName, curName: cfg.curVideoName });
videoSelector.bindProcess();

// Extract test UI removed per request

// Initialize videos and overlays if available
if (cfg.hasResults) {
  setupCanvas('ref-video', 'ref-canvas', cfg.refKpUrl);
  setupCanvas('cur-video', 'cur-canvas', cfg.curKpUrl);
  bindSynchronizedControls();
}

// Initialize chat
if (cfg.chatbotEnabled && document.getElementById('chat-input')) {
  const chat = new ChatController({ enabled: cfg.chatbotEnabled, hasResults: cfg.hasResults });
  await chat.init();
}

// Hook chat settings controls
await loadChatSettingsIntoControls();
const backendSelCtl = document.getElementById('backend-select');
const chatDeviceSelCtl = document.getElementById('chat-device-select');
if (backendSelCtl) backendSelCtl.addEventListener('change', applyChatSettingsFromControls);
if (chatDeviceSelCtl) chatDeviceSelCtl.addEventListener('change', applyChatSettingsFromControls);

// Start system usage polling
const usage = new UsageController();
usage.start(1000);

