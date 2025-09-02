"""
Revised golf_swing_compare.py

This module implements stricter golf swing comparison utilities.  It keeps the
public API of the previous version (``compare_swings``, ``analyze_differences``
``GolfSwingAnalyzer`` and ``EnhancedSwingChatBot``) but introduces golf specific
penalties so that poor swings receive noticeably lower scores.

Keypoint format per frame: ``List[Tuple(x, y, conf)]`` with coordinates in the
range ``[0,1]``.  Confidence values below ``MISSING_CONF`` are considered
unreliable and incur penalties.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

# Import helpers from the OpenPose wrapper so that external modules can reuse
# the original extraction pipeline.  ``extract_keypoints`` is re-exported for
# backwards compatibility with ``app.py``.
from openpose_extractor import extract_keypoints  # type: ignore

# ============================================================================
# Tunable parameters (stricter than previous version)
# ============================================================================
STRICTNESS_FACTOR = 4.0  # higher -> lower scores for the same deviation
MISSING_CONF = 0.20      # below this, a keypoint is considered unreliable
MISSING_PENALTY = 0.20   # additive penalty for each critical missing kp (per frame)
ANGLE_SCALE = 0.75       # scale for angle penalties (radians)

# Per-keypoint importance during position matching (golf-relevant joints heavier)
KEYPOINT_WEIGHTS: Dict[int, float] = {
    # 0:nose, 1:neck, 2:r_sho, 3:r_elb, 4:r_wri, 5:l_sho, 6:l_elb, 7:l_wri,
    # 8:mid_hip, 9:r_hip, 10:r_knee, 11:r_ankle, 12:l_hip, 13:l_knee, 14:l_ankle
    1: 1.2,   # neck
    2: 1.2, 3: 1.4, 4: 1.6,   # trail arm chain
    5: 1.2, 6: 1.4, 7: 1.6,   # lead arm chain
    8: 1.5, 9: 1.2, 12: 1.2,  # pelvis/COG anchors
    10: 0.8, 11: 0.6, 13: 0.8, 14: 0.6,  # legs (important but slightly lighter)
}
DEFAULT_KP_WEIGHT = 0.8

# ============================================================================
# Utilities
# ============================================================================
Point = Tuple[float, float, float]
Frame = Sequence[Point]


def _kp_weight(idx: int) -> float:
    return KEYPOINT_WEIGHTS.get(idx, DEFAULT_KP_WEIGHT)


def _to_np_xy(frame: Frame) -> np.ndarray:
    """Return an ``(N,2)`` array of XY coordinates (confidence dropped)."""
    return np.array([[p[0], p[1]] for p in frame], dtype=np.float32)


def _conf(frame: Frame) -> np.ndarray:
    return np.array([p[2] if len(p) > 2 else 1.0 for p in frame], dtype=np.float32)


def _bbox_scale_and_center(frame: Frame) -> Tuple[np.ndarray, float, np.ndarray]:
    """Return (pts_xy, scale, center) using shoulder-hip box for normalization.

    The bounding box is defined by ``(l_sho, r_sho, l_hip, r_hip)``.  When any of
    these anchors is missing we fall back to the whole body bounding box.
    ``scale`` is the larger of width/height so that poses are scaled uniformly.
    """
    pts = _to_np_xy(frame)
    conf = _conf(frame)

    # anchor indices
    L_SHO, R_SHO, L_HIP, R_HIP = 5, 2, 12, 9
    anchors = [L_SHO, R_SHO, L_HIP, R_HIP]

    if all(conf[i] > MISSING_CONF for i in anchors if i < len(frame)):
        sel = pts[anchors]
    else:
        sel = pts  # fallback

    min_xy = sel.min(axis=0)
    max_xy = sel.max(axis=0)
    center = (min_xy + max_xy) / 2.0
    wh = max(max_xy[0] - min_xy[0], max_xy[1] - min_xy[1])
    scale = max(wh, 1e-6)
    return pts, scale, center


def _normalize(frame: Frame) -> Tuple[np.ndarray, np.ndarray]:
    """Return normalized coordinates and confidences."""
    pts, scale, center = _bbox_scale_and_center(frame)
    norm = (pts - center) / scale
    return norm, _conf(frame)


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle ABC in radians at point B."""
    v1 = a - b
    v2 = c - b
    n1 = np.linalg.norm(v1) + 1e-9
    n2 = np.linalg.norm(v2) + 1e-9
    cos = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(math.acos(cos))


def _safe_idx(frame: Frame, idx: int) -> bool:
    return idx < len(frame)


# ============================================================================
# Core scoring (frame level)
# ============================================================================

def frame_difference(ref: Frame, test: Frame) -> float:
    """Compute per-frame difference with positional + angle + missing penalties.

    The returned value is an additive distance where larger values represent a
    worse swing.  It is later converted to a similarity score using an
    exponential transformation.
    """
    ref_xy, ref_cf = _normalize(ref)
    tst_xy, tst_cf = _normalize(test)

    N = min(len(ref), len(test))

    # 1) Weighted positional difference
    pos = 0.0
    wsum = 0.0
    for i in range(N):
        w = _kp_weight(i)
        ref_weight = 0.5 if ref_cf[i] <= MISSING_CONF else 1.0
        pos += w * ref_weight * float(np.linalg.norm(ref_xy[i] - tst_xy[i]))
        wsum += w * ref_weight
    pos = pos / max(wsum, 1e-9)

    # 2) Angle penalties on key joints (arms + legs + spine tilt)
    ang_pen = 0.0
    ang_cnt = 0

    triplets = [
        (2, 3, 4),   # right arm
        (5, 6, 7),   # left arm
        (9, 10, 11),  # right leg
        (12, 13, 14), # left leg
    ]

    # Spine tilt vs reference
    if all(_safe_idx(ref, k) and _safe_idx(test, k) for k in [8, 1]):
        ref_vec = ref_xy[1] - ref_xy[8]
        tst_vec = tst_xy[1] - tst_xy[8]
        ref_tilt = math.atan2(ref_vec[1], ref_vec[0])
        tst_tilt = math.atan2(tst_vec[1], tst_vec[0])
        dtilt = abs((tst_tilt - ref_tilt + math.pi) % (2 * math.pi) - math.pi)
        ang_pen += dtilt
        ang_cnt += 1

    for a_idx, b_idx, c_idx in triplets:
        if all(_safe_idx(ref, k) and _safe_idx(test, k) for k in (a_idx, b_idx, c_idx)):
            ref_ang = _angle(ref_xy[a_idx], ref_xy[b_idx], ref_xy[c_idx])
            tst_ang = _angle(tst_xy[a_idx], tst_xy[b_idx], tst_xy[c_idx])
            ang_pen += abs(ref_ang - tst_ang)
            ang_cnt += 1

    ang = (ang_pen / max(ang_cnt, 1)) * ANGLE_SCALE

    # 3) Missing/implausible penalties (based on test confidences)
    missing_cnt = 0
    for i in range(N):
        if tst_cf[i] <= MISSING_CONF and _kp_weight(i) >= 1.2:
            missing_cnt += 1
    miss = missing_cnt * MISSING_PENALTY

    # 4) Lateral sway penalty: pelvis (mean of hips) should be close to ref
    if all(_safe_idx(ref, k) and _safe_idx(test, k) for k in (9, 12)):
        ref_pelvis = 0.5 * (ref_xy[9] + ref_xy[12])
        tst_pelvis = 0.5 * (tst_xy[9] + tst_xy[12])
        sway = float(abs(tst_pelvis[0] - ref_pelvis[0]))
    else:
        sway = 0.0

    return pos + ang + miss + 0.5 * sway


# ============================================================================
# Phase detection & phase-aware scoring
# ============================================================================

PHASES = (
    "address", "backswing", "top", "downswing", "impact", "follow_through"
)


@dataclass
class PhaseBoundaries:
    indices: Dict[str, Tuple[int, int]]  # phase -> (start, end) frame indices inclusive


def _hand_height(frame: Frame) -> float:
    # Use higher-confidence wrist; if both present, take mean
    cf = _conf(frame)
    y = []
    for idx in (4, 7):
        if idx < len(frame) and cf[idx] > MISSING_CONF:
            y.append(frame[idx][1])
    return float(np.mean(y)) if y else 0.0


def _pelvis_x(frame: Frame) -> float:
    if len(frame) > 12:
        return float(0.5 * (frame[9][0] + frame[12][0]))
    return 0.0


def detect_phases(kp_seq: Sequence[Frame]) -> PhaseBoundaries:
    """Heuristic phase detection from kinematics on the *reference* sequence."""
    n = len(kp_seq)
    if n < 6:
        step = max(1, n // len(PHASES))
        idx = {p: (i * step, min((i + 1) * step - 1, n - 1)) for i, p in enumerate(PHASES)}
        return PhaseBoundaries(idx)

    hands_y = np.array([_hand_height(f) for f in kp_seq])
    pelvis_x = np.array([_pelvis_x(f) for f in kp_seq])

    top_i = int(np.argmax(hands_y))

    def hand_x(frame: Frame) -> float:
        xs, c = [], _conf(frame)
        for idx in (4, 7):
            if idx < len(frame) and c[idx] > MISSING_CONF:
                xs.append(frame[idx][0])
        return float(np.mean(xs)) if xs else 0.0

    hx = np.array([hand_x(f) for f in kp_seq])
    dist = np.abs(hx - pelvis_x)
    search_start = max(int(0.55 * n), top_i)
    impact_i = search_start + int(np.argmin(dist[search_start:]))

    address_end = max(1, int(0.15 * n))
    downswing_start = min(top_i + 1, n - 2)

    idx = {
        "address": (0, address_end),
        "backswing": (address_end + 1, max(top_i - 1, address_end + 1)),
        "top": (max(top_i - 1, 0), min(top_i + 1, n - 1)),
        "downswing": (downswing_start, max(impact_i - 1, downswing_start)),
        "impact": (max(impact_i - 1, 0), min(impact_i + 1, n - 1)),
        "follow_through": (min(impact_i + 2, n - 1), n - 1),
    }

    last = -1
    for name in PHASES:
        s, e = idx[name]
        if s <= last or e < s:
            step = max(1, n // len(PHASES))
            idx = {p: (i * step, min((i + 1) * step - 1, n - 1)) for i, p in enumerate(PHASES)}
            break
        last = e
    return PhaseBoundaries(idx)


def _phase_score(ref_kp: Sequence[Frame], tst_kp: Sequence[Frame], start: int, end: int, strictness: float = 5.0) -> float:
    """Score a sub-sequence with higher strictness and phase-specific checks."""
    length = min(len(ref_kp), len(tst_kp), end - start + 1)
    if length <= 0:
        return 1.0

    dsum = 0.0
    lowconf_pen = 0.0
    for i in range(length):
        r = ref_kp[start + i]
        t = tst_kp[start + i]
        dsum += frame_difference(r, t)
        cf = _conf(t)
        imp = sum(1 for idx in KEYPOINT_WEIGHTS if idx < len(t) and cf[idx] <= MISSING_CONF)
        lowconf_pen += imp / max(len(KEYPOINT_WEIGHTS), 1)

    avg = dsum / length
    reliability = lowconf_pen / length
    return float(math.exp(-(strictness * avg + reliability)))


# ============================================================================
# Top level scoring utilities
# ============================================================================

def compare_swings(ref_kp: Sequence[Frame], test_kp: Sequence[Frame]) -> float:
    """Return similarity score between two swings in ``[0,1]``.

    The score combines frame wise distances with phase aware penalties.  Higher
    values indicate better similarity while poor swings receive low scores.
    """
    length = min(len(ref_kp), len(test_kp))
    if length == 0:
        return 0.0

    diff_sum = sum(frame_difference(ref_kp[i], test_kp[i]) for i in range(length))
    base = diff_sum / length
    base_score = math.exp(-STRICTNESS_FACTOR * base)

    phases = detect_phases(ref_kp)
    phase_scores = [
        _phase_score(ref_kp, test_kp, s, e, strictness=STRICTNESS_FACTOR)
        for s, e in phases.indices.values()
    ]
    phase_factor = float(np.mean(phase_scores)) if phase_scores else 1.0

    score = base_score * phase_factor
    return float(max(min(score, 1.0), 0.0))


def analyze_differences(ref_kp: Sequence[Frame], test_kp: Sequence[Frame]) -> Dict[str, float]:
    """Compute average per-keypoint positional differences."""
    length = min(len(ref_kp), len(test_kp))
    if length == 0:
        return {}

    num_kp = min(len(ref_kp[0]), len(test_kp[0]))
    diff_sum = np.zeros(num_kp, dtype=np.float32)
    counts = np.zeros(num_kp, dtype=np.float32)
    for i in range(length):
        ref_xy, ref_cf = _normalize(ref_kp[i])
        tst_xy, tst_cf = _normalize(test_kp[i])
        for j in range(num_kp):
            if ref_cf[j] > MISSING_CONF and tst_cf[j] > MISSING_CONF:
                diff_sum[j] += np.linalg.norm(ref_xy[j] - tst_xy[j])
                counts[j] += 1.0
    diff_avg = np.divide(diff_sum, counts, out=np.zeros_like(diff_sum), where=counts > 0)
    names = {
        0: "nose", 1: "neck", 2: "right shoulder", 3: "right elbow", 4: "right wrist",
        5: "left shoulder", 6: "left elbow", 7: "left wrist", 8: "mid hip",
        9: "right hip", 10: "right knee", 11: "right ankle",
        12: "left hip", 13: "left knee", 14: "left ankle",
    }
    return {names.get(i, str(i)): float(diff_avg[i]) for i in range(num_kp)}


# ============================================================================
# Analysis wrapper and chatbot
# ============================================================================

class GolfSwingAnalyzer:
    """Golf swing analysis wrapper keeping prior public API."""

    def __init__(self, ref_kp: Sequence[Frame], test_kp: Sequence[Frame]):
        self.ref_kp = ref_kp
        self.test_kp = test_kp
        self.analysis_results = self._perform_analysis()

    def _perform_analysis(self) -> Dict[str, object]:
        length = min(len(self.ref_kp), len(self.test_kp))
        score = compare_swings(self.ref_kp, self.test_kp)

        spine_ref: List[float] = []
        spine_tst: List[float] = []
        for i in range(length):
            ref_xy, _ = _normalize(self.ref_kp[i])
            tst_xy, _ = _normalize(self.test_kp[i])
            if ref_xy.shape[0] > 8 and tst_xy.shape[0] > 8:
                vref = ref_xy[1] - ref_xy[8]
                vtst = tst_xy[1] - tst_xy[8]
                spine_ref.append(math.degrees(math.atan2(vref[1], vref[0])))
                spine_tst.append(math.degrees(math.atan2(vtst[1], vtst[0])))
        spine_ref = np.array(spine_ref) if spine_ref else np.array([0.0])
        spine_tst = np.array(spine_tst) if spine_tst else np.array([0.0])

        result = {
            "overall_score": score,
            "keypoint_differences": analyze_differences(self.ref_kp, self.test_kp),
            "posture_analysis": {
                "spine_angle_difference": float(np.mean(np.abs(spine_ref - spine_tst))),
                "spine_consistency": float(np.std(spine_tst)),
            },
        }
        return result


class EnhancedSwingChatBot:
    """Minimal conversational stub that reports swing quality."""

    def __init__(self, ref_kp, test_kp, score: float | None = None):
        self.analyzer = GolfSwingAnalyzer(ref_kp, test_kp)
        self.score = score if score is not None else self.analyzer.analysis_results["overall_score"]
        self.analysis = self.analyzer.analysis_results

    def initial_message(self) -> str:
        s = self.score
        band = "優秀" if s > 0.90 else ("良好" if s > 0.80 else "要改善")
        return (
            f"🏌️ 解析完了  総合スコア: {s:.3f}  評価: {band}\n"
            f"姿勢(平均脊柱角差): {self.analysis['posture_analysis']['spine_angle_difference']:.1f}°\n"
        )

    def ask(self, message: str) -> str:  # pragma: no cover - simple stub
        return "詳報を準備しました。気になるフェーズを指定してください。"


# ============================================================================
# Basic skeleton drawing utilities (kept for compatibility with app.py)
# ============================================================================

POSE_PAIRS = [
    (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7),
    (1, 8), (8, 9), (9, 10), (8, 12), (12, 13), (13, 14),
    (0, 1), (0, 15), (15, 17), (0, 16), (16, 18),
]


def draw_skeleton(frame, keypoints):
    """Draw keypoints and skeleton on ``frame``.

    ``keypoints`` can be in absolute pixel coordinates or normalized ``[0,1]``
    coordinates.  Only points with confidence greater than ``0.3`` are rendered.
    """
    import cv2

    height, width = frame.shape[:2]

    def _scale(pt):
        x, y, conf = pt
        if x <= 1.0 and y <= 1.0:
            return x * width, y * height, conf
        return x, y, conf

    scaled_kp = [_scale(p) for p in keypoints]

    for x, y, conf in scaled_kp:
        if conf > 0.3:
            cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 0), -1)
    for a, b in POSE_PAIRS:
        if a < len(scaled_kp) and b < len(scaled_kp):
            x1, y1, c1 = scaled_kp[a]
            x2, y2, c2 = scaled_kp[b]
            if c1 > 0.3 and c2 > 0.3:
                cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)


__all__ = [
    "compare_swings",
    "analyze_differences",
    "GolfSwingAnalyzer",
    "EnhancedSwingChatBot",
    "draw_skeleton",
    "extract_keypoints",
]

