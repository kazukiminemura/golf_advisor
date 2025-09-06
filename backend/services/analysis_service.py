from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional, Tuple

from backend.config import Settings
from backend.analysis import compare_swings, draw_skeleton
from backend.inference import extract_keypoints


logger = logging.getLogger("uvicorn.error")


class AnalysisService:
    """Encapsulates video/keypoint analysis and artifacts lifecycle.

    - Single Responsibility: analysis only (no HTTP, no chatbot state)
    - Dependency Inversion: depends on abstracted functions via injection when
      desired (defaults to project helpers).
    """

    def __init__(
        self,
        model_xml: str = Settings.MODEL_XML,
        device: str = Settings.DEVICE,
        data_dir: Path = Settings.DATA_DIR,
        static_dir: Path = Settings.STATIC_DIR,
    ) -> None:
        self.model_xml = model_xml
        self.device = device
        self.data_dir = data_dir
        self.static_dir = static_dir

        # Default video selections
        self.ref_video = data_dir / "reference.mp4"
        self.cur_video = data_dir / "current.mp4"

        # Output artifact paths
        self.out_ref = static_dir / "reference_annotated.mp4"
        self.out_cur = static_dir / "current_annotated.mp4"
        self.ref_kp_json = static_dir / "reference_keypoints.json"
        self.cur_kp_json = static_dir / "current_keypoints.json"

        # Runtime state
        self.score: Optional[float] = None
        self.ref_keypoints = None
        self.cur_keypoints = None
        self.analysis_running = False

    # ----------------------------- public API -----------------------------
    def set_videos(self, ref_file: Optional[str], cur_file: Optional[str], device: Optional[str]) -> None:
        if ref_file:
            self.ref_video = self.data_dir / ref_file
        if cur_file:
            self.cur_video = self.data_dir / cur_file
        if device and device.upper() in {"CPU", "GPU", "NPU"}:
            self.device = device.upper()
            logger.info("Analysis device set to %s", self.device)
        # Clear previous results/state and artifacts when switching videos
        self._clear_state(remove_artifacts=True)

    def prepare(self) -> None:
        """Run extraction, scoring, and artifact generation synchronously."""
        if (
            self.score is not None
            and self.out_ref.exists()
            and self.out_cur.exists()
            and self.ref_kp_json.exists()
            and self.cur_kp_json.exists()
        ):
            logger.info("Videos already processed, skipping")
            return

        self.analysis_running = True
        try:
            import cv2

            # Guard files
            if not self.ref_video.exists():
                raise FileNotFoundError(f"Reference video not found: {self.ref_video}")
            if not self.cur_video.exists():
                raise FileNotFoundError(f"Current video not found: {self.cur_video}")

            # Extract keypoints
            logger.info("Extracting keypoints from reference video on %s...", self.device)
            self.ref_keypoints = extract_keypoints(self.ref_video, self.model_xml, self.device)
            logger.info("Extracting keypoints from current video on %s...", self.device)
            self.cur_keypoints = extract_keypoints(self.cur_video, self.model_xml, self.device)

            # FPS for JSON
            ref_fps = self._get_fps(self.ref_video)
            cur_fps = self._get_fps(self.cur_video)

            # Compute score
            score, _ = compare_swings(self.ref_keypoints, self.cur_keypoints)
            self.score = score

            # Create outputs
            logger.info("Creating annotated videos and keypoint JSONs...")
            self._annotate_video(self.ref_video, self.ref_keypoints, self.out_ref)
            self._annotate_video(self.cur_video, self.cur_keypoints, self.out_cur)
            self._save_keypoints_json(self.ref_keypoints, ref_fps, self.ref_kp_json)
            self._save_keypoints_json(self.cur_keypoints, cur_fps, self.cur_kp_json)
        except Exception as e:
            logger.exception("Critical error during video analysis: %s", e)
            self._clear_state(remove_artifacts=True)
            raise
        finally:
            self.analysis_running = False

    def results_ready(self) -> bool:
        if self.score is not None and self.ref_keypoints is not None and self.cur_keypoints is not None:
            return True
        return self.score is not None and self.ref_kp_json.exists() and self.cur_kp_json.exists()

    # --------------------------- helper methods ---------------------------
    @staticmethod
    def _annotate_video(src: Path, keypoints, dst: Path) -> None:
        import cv2

        cap = cv2.VideoCapture(str(src))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(dst), fourcc, fps, (width, height))

        for kp in keypoints:
            ret, frame = cap.read()
            if not ret:
                break
            scaled = [(x * width, y * height, c) for x, y, c in kp]
            draw_skeleton(frame, scaled)
            writer.write(frame)

        cap.release()
        writer.release()

    @staticmethod
    def _save_keypoints_json(keypoints, fps: float, dst: Path) -> None:
        serializable = [[list(map(float, kp)) for kp in frame] for frame in keypoints]
        with dst.open("w") as f:
            json.dump({"fps": fps, "keypoints": serializable}, f)

    @staticmethod
    def _get_fps(video: Path) -> float:
        import cv2

        cap = cv2.VideoCapture(str(video))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()
        return float(fps)

    def _clear_state(self, remove_artifacts: bool = False) -> None:
        self.score = None
        self.ref_keypoints = None
        self.cur_keypoints = None
        if remove_artifacts:
            for p in (self.out_ref, self.out_cur, self.ref_kp_json, self.cur_kp_json):
                try:
                    if p.exists():
                        p.unlink()
                except Exception:
                    pass
