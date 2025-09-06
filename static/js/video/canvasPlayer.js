import { drawSkeleton } from './skeleton.js';

export async function setupCanvas(videoId, canvasId, jsonUrl) {
  try {
    const res = await fetch(jsonUrl);
    const data = await res.json();
    const fps = data.fps || 30;
    const kps = data.keypoints || [];
    const video = document.getElementById(videoId);
    const canvas = document.getElementById(canvasId);
    const ctx = canvas.getContext('2d');

    video.addEventListener('loadedmetadata', () => {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
    });

    let reqId;
    function render() {
      if (video.paused || video.ended) return;
      const idx = Math.min(Math.floor(video.currentTime * fps), kps.length - 1);
      if (idx >= 0) {
        drawSkeleton(ctx, kps[idx], canvas.width, canvas.height);
      }
      reqId = requestAnimationFrame(render);
    }

    video.addEventListener('play', () => { reqId = requestAnimationFrame(render); });
    video.addEventListener('pause', () => cancelAnimationFrame(reqId));
    video.addEventListener('seeked', () => {
      const idx = Math.min(Math.floor(video.currentTime * fps), kps.length - 1);
      if (idx >= 0) drawSkeleton(ctx, kps[idx], canvas.width, canvas.height);
    });
  } catch (e) {
    console.error('Failed to load keypoints:', e);
  }
}

