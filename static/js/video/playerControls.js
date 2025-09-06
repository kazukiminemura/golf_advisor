export function bindSynchronizedControls() {
  const playBtn = document.getElementById('play-both-btn');
  if (playBtn) {
    playBtn.addEventListener('click', () => {
      const ref = document.getElementById('ref-video');
      const cur = document.getElementById('cur-video');
      if (ref && cur) {
        ref.currentTime = 0;
        cur.currentTime = 0;
        ref.play();
        cur.play();
      }
    });
  }
  const stopBtn = document.getElementById('stop-both-btn');
  if (stopBtn) {
    stopBtn.addEventListener('click', () => {
      const ref = document.getElementById('ref-video');
      const cur = document.getElementById('cur-video');
      if (ref && cur) {
        ref.pause();
        cur.pause();
        ref.currentTime = 0;
        cur.currentTime = 0;
      }
    });
  }
}

