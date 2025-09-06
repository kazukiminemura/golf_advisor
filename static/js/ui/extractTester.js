import { listVideos, extractPose } from '../services/api.js';

export class ExtractTester {
  constructor({ defaultDevice }) {
    this.defaultDevice = defaultDevice || 'CPU';
  }

  async init() {
    try {
      const sel = document.getElementById('extract-file');
      const runBtn = document.getElementById('extract-btn');
      const result = document.getElementById('extract-result');
      if (!sel || !runBtn || !result) return;

      // Populate with available videos
      const files = await listVideos();
      sel.innerHTML = '';
      files.forEach(f => {
        const o = document.createElement('option');
        o.value = f; o.textContent = f; sel.appendChild(o);
      });

      // Default to current selection if exists
      const curSel = document.getElementById('cur-file');
      if (curSel && curSel.value) sel.value = curSel.value;

      runBtn.onclick = async () => {
        const poseSel = document.getElementById('pose-select');
        const devSel = document.getElementById('device-select');
        const video_file = sel.value;
        const pose_model = poseSel ? poseSel.value : 'openvino';
        const device = devSel ? devSel.value : this.defaultDevice;
        result.textContent = '抽出中...';
        try {
          const res = await extractPose({ video_file, pose_model, device });
          result.textContent = JSON.stringify(res, null, 2);
        } catch (e) {
          result.textContent = '抽出に失敗しました: ' + (e && e.message ? e.message : String(e));
        }
      };
    } catch (_) { /* no-op */ }
  }
}

