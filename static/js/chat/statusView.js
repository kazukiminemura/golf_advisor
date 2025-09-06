// Presentation for chatbot status
export const StatusClass = Object.freeze({
  PREPARING: 'chat-status preparing',
  GENERATING: 'chat-status generating',
  READY: 'chat-status ready',
  ERROR: 'chat-status error',
});

export function setStatusPreparing(el, text) {
  el.textContent = text || 'チャットボットを準備中です。まず動画を解析してください。';
  el.className = StatusClass.PREPARING;
}

export function setStatusGenerating(el, text) {
  el.textContent = text || 'メッセージ生成中...';
  el.className = StatusClass.GENERATING;
}

export function setStatusReady(el, text) {
  el.textContent = text || 'チャットボットの準備ができました';
  el.className = StatusClass.READY;
}

export function setStatusError(el, text) {
  el.textContent = text || 'チャットボット状態の取得に失敗しました';
  el.className = StatusClass.ERROR;
}

