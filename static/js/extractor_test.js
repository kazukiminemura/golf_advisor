import { listVideos } from './services/api.js';
import { ExtractTester } from './ui/extractTester.js';

// Populate initial video list for the standalone page and bind tester
async function main() {
  // Populate dropdown
  const sel = document.getElementById('extract-file');
  if (sel) {
    try {
      const files = await listVideos();
      sel.innerHTML = '';
      files.forEach(f => { const o = document.createElement('option'); o.value = f; o.textContent = f; sel.appendChild(o); });
    } catch (_) { /* ignore */ }
  }

  const tester = new ExtractTester({ defaultDevice: document.getElementById('device-select')?.value || 'CPU' });
  await tester.init();
}

main();

