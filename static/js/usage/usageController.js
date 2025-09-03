import { systemUsage } from '../services/api.js';

function usageColor(v) {
  if (v < 25) return 'green';
  if (v < 50) return 'blue';
  if (v < 75) return 'yellow';
  return 'red';
}

export class UsageController {
  async update() {
    try {
      const data = await systemUsage();
      const cpu = data.cpu || 0;
      const gpu = data.gpu || 0;
      const npu = data.npu || 0;
      const mem = data.memory_percent || 0;

      const set = (id, value) => { const el = document.getElementById(id); if (el) el.value = value; };
      const txt = (id, text) => { const el = document.getElementById(id); if (el) el.textContent = text; };
      const color = (id, v) => { const el = document.getElementById(id); if (el) el.style.setProperty('--bar-color', usageColor(v)); };
      const colorText = (id, v) => { const el = document.getElementById(id); if (el) el.style.color = usageColor(v); };

      set('cpu-bar', cpu);
      set('gpu-bar', gpu);
      set('npu-bar', npu);
      set('mem-bar', mem);

      txt('cpu-text', `${cpu.toFixed(1)}%`);
      txt('gpu-text', `${gpu.toFixed(1)}%`);
      txt('npu-text', `${npu.toFixed(1)}%`);
      txt('mem-text', `${mem.toFixed(1)}%`);

      color('cpu-bar', cpu);
      color('gpu-bar', gpu);
      color('npu-bar', npu);
      color('mem-bar', mem);

      colorText('cpu-text', cpu);
      colorText('gpu-text', gpu);
      colorText('npu-text', npu);
      colorText('mem-text', mem);

      const md = document.getElementById('mem-detail');
      if (md && data.memory_used_gb != null && data.memory_total_gb != null) {
        md.textContent = `${data.memory_used_gb.toFixed(2)} / ${data.memory_total_gb.toFixed(2)} GB`;
      }
    } catch (_) {}
  }

  start(intervalMs = 1000) {
    this.update();
    this.timer = setInterval(() => this.update(), intervalMs);
  }

  stop() {
    if (this.timer) clearInterval(this.timer);
  }
}

