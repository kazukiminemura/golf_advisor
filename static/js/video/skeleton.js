export const POSE_PAIRS = [
  [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7],
  [1, 8], [8, 9], [9, 10], [8, 12], [12, 13], [13, 14],
  [0, 1], [0, 15], [15, 17], [0, 16], [16, 18]
];

export function drawSkeleton(ctx, keypoints, w, h) {
  ctx.clearRect(0, 0, w, h);
  keypoints.forEach(([x, y, c]) => {
    if (c > 0.3) {
      ctx.beginPath();
      ctx.arc(x * w, y * h, 4, 0, 2 * Math.PI);
      ctx.fillStyle = 'rgb(0,255,0)';
      ctx.fill();
    }
  });
  POSE_PAIRS.forEach(([a, b]) => {
    if (a < keypoints.length && b < keypoints.length) {
      const [x1, y1, c1] = keypoints[a];
      const [x2, y2, c2] = keypoints[b];
      if (c1 > 0.3 && c2 > 0.3) {
        ctx.beginPath();
        ctx.moveTo(x1 * w, y1 * h);
        ctx.lineTo(x2 * w, y2 * h);
        ctx.strokeStyle = 'rgb(255,0,0)';
        ctx.lineWidth = 2;
        ctx.stroke();
      }
    }
  });
}

