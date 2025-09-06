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
  if (devLabel) devLabel.textContent = '動画分析デバイス';
  const beLabel = document.querySelector('label[for="backend-select"]');
  if (beLabel) beLabel.textContent = 'チャットボットバックエンド';
} catch (_) { /* no-op */ }

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

// Start system usage polling
const usage = new UsageController();
usage.start(1000);

