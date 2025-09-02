import argparse
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
import math
import torch
import numpy as np

# Scoring parameters (stricter)
STRICTNESS_FACTOR = 4.0
MISSING_CONF = 0.20
MISSING_PENALTY = 0.20
ANGLE_SCALE = 0.75

# Per-keypoint importance (golf-relevant joints heavier)
KEYPOINT_WEIGHTS: Dict[int, float] = {
    # 0:nose, 1:neck, 2:r_sho, 3:r_elb, 4:r_wri, 5:l_sho, 6:l_elb, 7:l_wri,
    # 8:mid_hip, 9:r_hip, 10:r_knee, 11:r_ankle, 12:l_hip, 13:l_knee, 14:l_ankle
    1: 1.2,
    2: 1.2, 3: 1.4, 4: 1.6,
    5: 1.2, 6: 1.4, 7: 1.6,
    8: 1.5, 9: 1.2, 12: 1.2,
    10: 0.8, 11: 0.6, 13: 0.8, 14: 0.6,
}
DEFAULT_KP_WEIGHT = 0.8

Point = Tuple[float, float, float]
Frame = Sequence[Point]


def _kp_weight(idx: int) -> float:
    return KEYPOINT_WEIGHTS.get(idx, DEFAULT_KP_WEIGHT)


def _to_np_xy(frame: Frame) -> np.ndarray:
    return np.array([[p[0], p[1]] for p in frame], dtype=np.float32)


def _conf(frame: Frame) -> np.ndarray:
    return np.array([p[2] if len(p) > 2 else 1.0 for p in frame], dtype=np.float32)


def _bbox_scale_and_center(frame: Frame) -> Tuple[np.ndarray, float, np.ndarray]:
    pts = _to_np_xy(frame)
    conf = _conf(frame)
    L_SHO, R_SHO, L_HIP, R_HIP = 5, 2, 12, 9
    anchors = [L_SHO, R_SHO, L_HIP, R_HIP]
    if all(conf[i] > MISSING_CONF for i in anchors):
        sel = pts[anchors]
    else:
        sel = pts
    min_xy = sel.min(axis=0)
    max_xy = sel.max(axis=0)
    center = (min_xy + max_xy) / 2.0
    wh = max(max_xy[0] - min_xy[0], max_xy[1] - min_xy[1])
    scale = max(wh, 1e-6)
    return pts, scale, center


def _normalize(frame: Frame) -> Tuple[np.ndarray, np.ndarray]:
    pts, scale, center = _bbox_scale_and_center(frame)
    norm = (pts - center) / scale
    return norm, _conf(frame)


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    v1 = a - b
    v2 = c - b
    n1 = np.linalg.norm(v1) + 1e-9
    n2 = np.linalg.norm(v2) + 1e-9
    cos = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(math.acos(cos))


def _safe_idx(frame: Frame, idx: int) -> bool:
    return idx < len(frame)


# OpenPose keypoint extraction
from openpose_extractor import extract_keypoints


def frame_difference(ref: Frame, test: Frame) -> float:
    """Per-frame difference with position, angle, and missing penalties."""
    ref_xy, ref_conf = _normalize(ref)
    test_xy, test_conf = _normalize(test)
    n = min(len(ref_xy), len(test_xy))
    pos_sum = 0.0
    weight_sum = 0.0
    missing = 0.0
    for i in range(n):
        w = _kp_weight(i)
        if ref_conf[i] <= MISSING_CONF or test_conf[i] <= MISSING_CONF:
            missing += MISSING_PENALTY * w
            continue
        d = np.linalg.norm(ref_xy[i] - test_xy[i])
        pos_sum += w * d
        weight_sum += w
    pos = pos_sum / max(weight_sum, 1e-9)

    # Angle penalties (elbows and knees)
    angle_pairs = [
        (2, 3, 4), (5, 6, 7),
        (9, 10, 11), (12, 13, 14),
    ]
    ang = 0.0
    for a, b, c in angle_pairs:
        if all(_safe_idx(ref, idx) and _safe_idx(test, idx) for idx in (a, b, c)):
            if all(ref_conf[idx] > MISSING_CONF and test_conf[idx] > MISSING_CONF for idx in (a, b, c)):
                ang_ref = _angle(ref_xy[a], ref_xy[b], ref_xy[c])
                ang_test = _angle(test_xy[a], test_xy[b], test_xy[c])
                ang += abs(ang_ref - ang_test)
    ang *= ANGLE_SCALE

    # Spine tilt difference
    if all(_safe_idx(ref, idx) and _safe_idx(test, idx) for idx in (8, 1)):
        if ref_conf[8] > MISSING_CONF and ref_conf[1] > MISSING_CONF and test_conf[8] > MISSING_CONF and test_conf[1] > MISSING_CONF:
            ref_tilt = _angle(ref_xy[8], ref_xy[1], ref_xy[8] + np.array([0, 1]))
            test_tilt = _angle(test_xy[8], test_xy[1], test_xy[8] + np.array([0, 1]))
            ang += abs(ref_tilt - test_tilt)

    # Pelvis sway penalty (horizontal displacement of mid hip)
    pelvis = 0.0
    if _safe_idx(ref, 8) and _safe_idx(test, 8):
        if ref_conf[8] > MISSING_CONF and test_conf[8] > MISSING_CONF:
            pelvis = abs(ref_xy[8][0] - test_xy[8][0])

    return pos + ang + missing + pelvis


def compare_swings(ref_kp, test_kp):
    length = min(len(ref_kp), len(test_kp))
    if length == 0:
        return 0.0
    total = 0.0
    for i in range(length):
        total += frame_difference(ref_kp[i], test_kp[i])
    avg = total / length
    return float(math.exp(-STRICTNESS_FACTOR * avg))


def analyze_differences(ref_kp, test_kp):
    """Compute average per-keypoint differences between two swings."""
    length = min(len(ref_kp), len(test_kp))
    if length == 0:
        return {}
    num_kp = min(len(ref_kp[0]), len(test_kp[0]))
    diff_sum = np.zeros(num_kp, dtype=np.float32)
    counts = np.zeros(num_kp, dtype=np.float32)
    for i in range(length):
        ref_xy, ref_conf = _normalize(ref_kp[i])
        test_xy, test_conf = _normalize(test_kp[i])
        for j in range(num_kp):
            if ref_conf[j] > MISSING_CONF and test_conf[j] > MISSING_CONF:
                diff_sum[j] += np.linalg.norm(ref_xy[j] - test_xy[j])
                counts[j] += 1.0
    diff_avg = np.divide(diff_sum, counts, out=np.zeros_like(diff_sum), where=counts > 0)
    names = {
        0: "nose", 1: "neck", 2: "right shoulder", 3: "right elbow", 4: "right wrist",
        5: "left shoulder", 6: "left elbow", 7: "left wrist", 8: "mid hip",
        9: "right hip", 10: "right knee", 11: "right ankle",
        12: "left hip", 13: "left knee", 14: "left ankle"
    }
    return {names.get(i, str(i)): float(diff_avg[i]) for i in range(num_kp)}


class GolfSwingAnalyzer:
    """Advanced golf swing analysis using pose estimation data."""
    
    def __init__(self, ref_kp, test_kp):
        self.ref_kp = ref_kp
        self.test_kp = test_kp
        self.keypoint_names = {
            0: "nose", 1: "neck", 2: "right_shoulder", 3: "right_elbow", 4: "right_wrist",
            5: "left_shoulder", 6: "left_elbow", 7: "left_wrist", 8: "mid_hip",
            9: "right_hip", 10: "right_knee", 11: "right_ankle",
            12: "left_hip", 13: "left_knee", 14: "left_ankle"
        }
        self.analysis_results = self._perform_detailed_analysis()
    
    def _perform_detailed_analysis(self):
        """Perform comprehensive swing analysis."""
        results = {
            "overall_score": compare_swings(self.ref_kp, self.test_kp),
            "keypoint_differences": analyze_differences(self.ref_kp, self.test_kp),
            "swing_phases": self._analyze_swing_phases(),
            "posture_analysis": self._analyze_posture(),
            "tempo_analysis": self._analyze_tempo(),
            "balance_analysis": self._analyze_balance()
        }
        return results
    
    def _analyze_swing_phases(self):
        """Analyze different phases of the golf swing."""
        length = min(len(self.ref_kp), len(self.test_kp))
        if length == 0:
            return {}
        
        # Divide swing into phases (address, backswing, downswing, follow-through)
        phases = {
            "address": (0, int(length * 0.15)),
            "backswing": (int(length * 0.15), int(length * 0.45)),
            "downswing": (int(length * 0.45), int(length * 0.65)),
            "follow_through": (int(length * 0.65), length)
        }
        
        phase_scores = {}
        for phase_name, (start, end) in phases.items():
            phase_diff = 0.0
            frame_count = 0
            for i in range(start, min(end, length)):
                ref = np.array([p[:2] for p in self.ref_kp[i]])
                test = np.array([p[:2] for p in self.test_kp[i]])
                phase_diff += np.linalg.norm(ref - test) / ref.size
                frame_count += 1
            avg_diff = phase_diff / max(frame_count, 1)
            phase_scores[phase_name] = float(
                np.exp(-avg_diff * STRICTNESS_FACTOR)
            )

        return phase_scores
    
    def _analyze_posture(self):
        """Analyze posture-related aspects."""
        length = min(len(self.ref_kp), len(self.test_kp))
        if length == 0:
            return {}
        
        spine_angles_ref = []
        spine_angles_test = []
        
        for i in range(length):
            # Calculate spine angle using neck and mid hip
            ref_neck = self.ref_kp[i][1][:2]
            ref_hip = self.ref_kp[i][8][:2]
            test_neck = self.test_kp[i][1][:2]
            test_hip = self.test_kp[i][8][:2]
            
            ref_angle = np.degrees(np.arctan2(ref_neck[1] - ref_hip[1], ref_neck[0] - ref_hip[0]))
            test_angle = np.degrees(np.arctan2(test_neck[1] - test_hip[1], test_neck[0] - test_hip[0]))
            
            spine_angles_ref.append(ref_angle)
            spine_angles_test.append(test_angle)
        
        return {
            "spine_angle_difference": np.mean(np.abs(np.array(spine_angles_ref) - np.array(spine_angles_test))),
            "spine_consistency": np.std(spine_angles_test)
        }
    
    def _analyze_tempo(self):
        """Analyze swing tempo and timing."""
        # Simple tempo analysis based on major position changes
        ref_tempo = self._calculate_tempo(self.ref_kp)
        test_tempo = self._calculate_tempo(self.test_kp)
        
        return {
            "tempo_difference": abs(ref_tempo - test_tempo),
            "ref_tempo": ref_tempo,
            "test_tempo": test_tempo
        }
    
    def _calculate_tempo(self, keypoints):
        """Calculate swing tempo based on hand movement."""
        if len(keypoints) < 2:
            return 0
        
        hand_speeds = []
        for i in range(1, len(keypoints)):
            prev_hand = keypoints[i-1][4][:2]  # right wrist
            curr_hand = keypoints[i][4][:2]
            speed = np.linalg.norm(np.array(curr_hand) - np.array(prev_hand))
            hand_speeds.append(speed)
        
        return np.mean(hand_speeds) if hand_speeds else 0
    
    def _analyze_balance(self):
        """Analyze balance and weight distribution."""
        length = min(len(self.ref_kp), len(self.test_kp))
        if length == 0:
            return {}
        
        balance_scores = []
        for i in range(length):
            # Calculate center of gravity using hip positions
            ref_left_hip = self.ref_kp[i][12][:2]
            ref_right_hip = self.ref_kp[i][9][:2]
            test_left_hip = self.test_kp[i][12][:2]
            test_right_hip = self.test_kp[i][9][:2]
            
            ref_center = [(ref_left_hip[0] + ref_right_hip[0])/2, (ref_left_hip[1] + ref_right_hip[1])/2]
            test_center = [(test_left_hip[0] + test_right_hip[0])/2, (test_left_hip[1] + test_right_hip[1])/2]
            
            balance_diff = np.linalg.norm(np.array(ref_center) - np.array(test_center))
            balance_scores.append(balance_diff)
        
        return {
            "balance_consistency": np.std(balance_scores),
            "average_balance_difference": np.mean(balance_scores)
        }


class EnhancedSwingChatBot:
    """Enhanced conversational AI for detailed golf swing coaching."""

    def __init__(self, ref_kp, test_kp, score):
        self.analyzer = GolfSwingAnalyzer(ref_kp, test_kp)
        self.score = score
        self.analysis = self.analyzer.analysis_results
        
        # Initialize conversation state
        self.conversation_history = []
        self.current_topic = None
        
    def initial_message(self):
        """Generate initial analysis summary."""
        message = f"""
🏌️ ゴルフスイング解析が完了しました！

📊 総合評価スコア: {self.score:.3f}
{'優秀' if self.score > 0.9 else '良好' if self.score > 0.8 else '要改善'}

📈 フェーズ別スコア:
• アドレス: {self.analysis['swing_phases']['address']:.3f}
• バックスイング: {self.analysis['swing_phases']['backswing']:.3f}
• ダウンスイング: {self.analysis['swing_phases']['downswing']:.3f}
• フォロースルー: {self.analysis['swing_phases']['follow_through']:.3f}

どの部分について詳しく知りたいですか？
1. 姿勢とバランス
2. スイングテンポ
3. 各部位の動き
4. 具体的な改善アドバイス
        """.strip()
        
        self.conversation_history.append(("bot", message))
        return message

    def ask(self, message: str) -> str:
        """Process user question and provide detailed coaching response."""
        self.conversation_history.append(("user", message))
        
        message_lower = message.lower()
        response = ""
        
        # Topic detection and response generation
        if any(word in message_lower for word in ['姿勢', 'バランス', '重心']):
            response = self._discuss_posture_balance()
        elif any(word in message_lower for word in ['テンポ', 'リズム', 'タイミング', '速度']):
            response = self._discuss_tempo()
        elif any(word in message_lower for word in ['手', '腕', '肩', '肘', '手首']):
            response = self._discuss_arm_movement()
        elif any(word in message_lower for word in ['腰', 'ヒップ', '回転']):
            response = self._discuss_hip_movement()
        elif any(word in message_lower for word in ['改善', 'アドバイス', '練習', 'コツ']):
            response = self._provide_improvement_advice()
        elif any(word in message_lower for word in ['アドレス']):
            response = self._discuss_address()
        elif any(word in message_lower for word in ['バックスイング']):
            response = self._discuss_backswing()
        elif any(word in message_lower for word in ['ダウンスイング', 'インパクト']):
            response = self._discuss_downswing()
        elif any(word in message_lower for word in ['フォロースルー', 'フィニッシュ']):
            response = self._discuss_followthrough()
        else:
            response = self._general_response()
        
        self.conversation_history.append(("bot", response))
        return response
    
    def _discuss_posture_balance(self):
        """Discuss posture and balance analysis."""
        posture = self.analysis['posture_analysis']
        balance = self.analysis['balance_analysis']
        
        spine_diff = posture['spine_angle_difference']
        balance_consistency = balance['balance_consistency']
        
        if spine_diff < 5:
            posture_eval = "優秀な姿勢を保っています"
        elif spine_diff < 15:
            posture_eval = "概ね良好ですが、わずかに改善の余地があります"
        else:
            posture_eval = "姿勢に大きな違いがあります"
        
        return f"""
🏃 姿勢・バランス解析:

📐 脊椎角度の差: {spine_diff:.1f}度
{posture_eval}

⚖️ バランス評価:
• 一貫性: {balance_consistency:.3f} ({'安定' if balance_consistency < 0.05 else '要改善'})
• 平均差: {balance['average_balance_difference']:.3f}

💡 アドバイス:
{'姿勢が安定しています。この調子を維持してください。' if spine_diff < 10 else 
 '背筋をより真っ直ぐ保ち、重心を意識した練習をお勧めします。'}
        """
    
    def _discuss_tempo(self):
        """Discuss swing tempo analysis."""
        tempo = self.analysis['tempo_analysis']
        tempo_diff = tempo['tempo_difference']
        
        if tempo_diff < 0.01:
            tempo_eval = "理想的なテンポです"
        elif tempo_diff < 0.05:
            tempo_eval = "良好なテンポですが、わずかに調整が必要"
        else:
            tempo_eval = "テンポに大きな違いがあります"
        
        return f"""
🎵 スイングテンポ解析:

📊 テンポ差: {tempo_diff:.4f}
{tempo_eval}

⏱️ 詳細:
• 基準テンポ: {tempo['ref_tempo']:.4f}
• あなたのテンポ: {tempo['test_tempo']:.4f}

💡 改善ポイント:
{'現在のテンポを維持してください。' if tempo_diff < 0.02 else
 'メトロノームを使った練習で、一定のリズムを身につけましょう。'}
        """
    
    def _discuss_arm_movement(self):
        """Discuss arm and upper body movement."""
        kp_diff = self.analysis['keypoint_differences']
        
        arm_points = {
            '右肩': kp_diff.get('right shoulder', 0),
            '右肘': kp_diff.get('right elbow', 0),
            '右手首': kp_diff.get('right wrist', 0),
            '左肩': kp_diff.get('left shoulder', 0),
            '左肘': kp_diff.get('left elbow', 0),
            '左手首': kp_diff.get('left wrist', 0)
        }
        
        worst_point = max(arm_points.items(), key=lambda x: x[1])
        
        return f"""
💪 腕・上半身の動き解析:

📍 各部位の差異:
• 右肩: {arm_points['右肩']:.3f}
• 右肘: {arm_points['右肘']:.3f}
• 右手首: {arm_points['右手首']:.3f}
• 左肩: {arm_points['左肩']:.3f}
• 左肘: {arm_points['左肘']:.3f}
• 左手首: {arm_points['左手首']:.3f}

⚠️ 注目ポイント: {worst_point[0]}の動きに最も大きな差があります（{worst_point[1]:.3f}）

💡 練習アドバイス:
{worst_point[0]}の動きを意識して、ゆっくりとしたスイング練習から始めてみてください。
        """
    
    def _discuss_hip_movement(self):
        """Discuss hip and lower body movement."""
        kp_diff = self.analysis['keypoint_differences']
        
        hip_points = {
            '腰中央': kp_diff.get('mid hip', 0),
            '右腰': kp_diff.get('right hip', 0),
            '左腰': kp_diff.get('left hip', 0)
        }
        
        avg_hip_diff = np.mean(list(hip_points.values()))
        
        return f"""
🏋️ 腰・下半身の動き解析:

📍 腰部の動き:
• 腰中央: {hip_points['腰中央']:.3f}
• 右腰: {hip_points['右腰']:.3f}
• 左腰: {hip_points['左腰']:.3f}
• 平均差: {avg_hip_diff:.3f}

📊 評価: {'優秀' if avg_hip_diff < 0.1 else '良好' if avg_hip_diff < 0.2 else '要改善'}

💡 改善のポイント:
腰の回転は飛距離と精度の鍵です。
{'現在の腰の動きは理想的です。' if avg_hip_diff < 0.15 else
 '腰の回転をもっと意識して、下半身主導のスイングを心がけてください。'}
        """
    
    def _provide_improvement_advice(self):
        """Provide specific improvement recommendations."""
        phases = self.analysis['swing_phases']
        worst_phase = min(phases.items(), key=lambda x: x[1])
        
        advice_map = {
            'address': '構えでは、足幅と重心配分を意識してください。',
            'backswing': 'バックスイングでは、肩の回転と手首のコックを意識しましょう。',
            'downswing': 'ダウンスイングでは、下半身主導で腰の回転を先行させてください。',
            'follow_through': 'フォロースルーでは、しっかりと振り切ることを意識してください。'
        }
        
        return f"""
🎯 具体的な改善アドバイス:

📊 最も改善が必要なフェーズ: {worst_phase[0]}（スコア: {worst_phase[1]:.3f}）

💪 重点練習メニュー:
1. {advice_map.get(worst_phase[0], '基本動作の確認')}
2. ハーフスイングでの反復練習
3. ミラーを使った姿勢チェック

📝 短期目標:
• 総合スコア {min(1, self.score + 0.05):.3f} を目指しましょう
• {worst_phase[0]}フェーズの改善に集中

🏌️ 次回の練習で特に意識するポイントをお伝えしますか？
        """
    
    def _discuss_address(self):
        """Discuss address position."""
        address_score = self.analysis['swing_phases']['address']
        return f"""
🏌️ アドレス解析:

📊 アドレススコア: {address_score:.3f}
評価: {'優秀' if address_score > 0.90 else '良好' if address_score > 0.86 else '要改善'}

💡 アドレスのポイント:
• 足幅は肩幅程度
• 重心は土踏まずに
• 背筋を真っ直ぐ保つ
• ボールとの距離を一定に

{f'アドレスが安定しています。' if address_score > 0.88 else
 'アドレスでの基本姿勢を見直してみましょう。'}
        """
    
    def _discuss_backswing(self):
        """Discuss backswing analysis."""
        backswing_score = self.analysis['swing_phases']['backswing']
        return f"""
🔄 バックスイング解析:

📊 バックスイングスコア: {backswing_score:.3f}
評価: {'優秀' if backswing_score > 0.86 else '良好' if backswing_score > 0.78 else '要改善'}

🎯 バックスイングのキーポイント:
• 肩の十分な回転（90度以上）
• 左腕の伸び
• 手首のコック
• 重心の右足移動

{f'バックスイングが安定しています。' if backswing_score > 0.82 else
 'バックスイングでの体の回転をもっと意識してみましょう。'}
        """
    
    def _discuss_downswing(self):
        """Discuss downswing analysis."""
        downswing_score = self.analysis['swing_phases']['downswing']
        return f"""
⚡ ダウンスイング解析:

📊 ダウンスイングスコア: {downswing_score:.3f}
評価: {'優秀' if downswing_score > 0.86 else '良好' if downswing_score > 0.78 else '要改善'}

🎯 ダウンスイングの重要ポイント:
• 下半身主導の始動
• 腰の回転が先行
• ハンドファーストでのインパクト
• 重心の左足移動

{f'ダウンスイングが理想的です。' if downswing_score > 0.82 else
 '下半身主導のダウンスイングを意識してみてください。'}
        """
    
    def _discuss_followthrough(self):
        """Discuss follow-through analysis."""
        followthrough_score = self.analysis['swing_phases']['follow_through']
        return f"""
🎊 フォロースルー解析:

📊 フォロースルースコア: {followthrough_score:.3f}
評価: {'優秀' if followthrough_score > 0.86 else '良好' if followthrough_score > 0.78 else '要改善'}

🎯 フォロースルーのポイント:
• 最後まで振り切る
• 体重の完全な左足移動
• バランスの良いフィニッシュ
• 目標方向への体の向き

{f'フォロースルーが素晴らしいです。' if followthrough_score > 0.82 else
 'もっと大きく振り切ることを意識してみましょう。'}
        """
    
    def _general_response(self):
        """Provide general guidance."""
        return f"""
🏌️ どのようなことについて知りたいですか？

以下のトピックについてお答えできます：
• 姿勢とバランス分析
• スイングテンポとリズム  
• 各部位（腕、肩、腰など）の動き
• スイングフェーズ別の分析
• 具体的な改善アドバイス

現在のスコア: {self.score:.3f}
お気軽に質問してください！
        """


# Rest of the original code remains the same...
POSE_PAIRS = [
    (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7),
    (1, 8), (8, 9), (9, 10), (8, 12), (12, 13), (13, 14),
    (0, 1), (0, 15), (15, 17), (0, 16), (16, 18),
]


def draw_skeleton(frame, keypoints):
    """Draw detected keypoints and skeleton on a frame.

    This function now accepts keypoints either in pixel coordinates or
    normalized coordinates (0-1). When normalized coordinates are
    provided, they are scaled to the frame size internally. This makes
    the function robust to different keypoint formats and fixes issues
    where joints were not rendered due to mismatched scales.
    """
    import cv2

    height, width = frame.shape[:2]

    def _scale(pt):
        x, y, conf = pt
        if x <= 1.0 and y <= 1.0:
            # Assume normalized coordinates
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
                cv2.line(
                    frame,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    (255, 0, 0),
                    2,
                )


def play_openpose_side_by_side(video_path: Path, keypoints) -> None:
    """Display original and OpenPose-annotated videos simultaneously.

    Args:
        video_path (Path): Source video to read frames from.
        keypoints (list): Per-frame keypoints returned by ``extract_keypoints``.
    """
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_idx = 0

    while cap.isOpened() and frame_idx < len(keypoints):
        ret, frame = cap.read()
        if not ret:
            break

        annotated = frame.copy()
        draw_skeleton(annotated, keypoints[frame_idx])
        combined = cv2.hconcat([frame, annotated])
        cv2.imshow("OpenPose Result", combined)

        if cv2.waitKey(int(1000 / fps)) & 0xFF == ord("q"):
            break

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()


def show_comparison_with_chat(
    ref_path: Path, test_path: Path, ref_kp, test_kp, score, start_paused: bool = False
):
    """Display swings alongside an enhanced chat panel."""
    import cv2
    import tkinter as tk
    from tkinter import scrolledtext
    from PIL import Image, ImageTk

    bot = EnhancedSwingChatBot(ref_kp, test_kp, score)

    root = tk.Tk()
    root.title("ゴルフスイング解析システム")
    root.geometry("1200x800")

    # Create main frames
    video_frame = tk.Frame(root)
    video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    
    chat_frame = tk.Frame(root, width=400)
    chat_frame.pack(side=tk.RIGHT, fill=tk.BOTH)
    chat_frame.pack_propagate(False)

    # Video display
    video_label = tk.Label(video_frame)
    video_label.pack(pady=10)

    # Chat interface
    tk.Label(chat_frame, text="🏌️ AIゴルフコーチ", font=("Arial", 16, "bold")).pack(pady=5)
    
    chat_display = scrolledtext.ScrolledText(chat_frame, height=25, wrap=tk.WORD, font=("Arial", 10))
    chat_display.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
    
    entry = tk.Entry(chat_frame, font=("Arial", 12))
    entry.pack(fill=tk.X, padx=10, pady=5)
    
    def send_message(event=None):
        user_input = entry.get().strip()
        if not user_input:
            return
        entry.delete(0, tk.END)
        
        chat_display.insert(tk.END, f"あなた: {user_input}\n", "user")
        chat_display.insert(tk.END, "\n")
        
        response = bot.ask(user_input)
        chat_display.insert(tk.END, f"🤖 AIコーチ: {response}\n", "bot")
        chat_display.insert(tk.END, "\n" + "="*50 + "\n")
        chat_display.see(tk.END)

    entry.bind("<Return>", send_message)
    
    send_button = tk.Button(chat_frame, text="送信", command=send_message, font=("Arial", 12))
    send_button.pack(pady=5)
    
    # Configure text tags
    chat_display.tag_configure("user", foreground="blue")
    chat_display.tag_configure("bot", foreground="green")
    
    # Display initial message
    initial_msg = bot.initial_message()
    chat_display.insert(tk.END, f"🤖 AIコーチ: {initial_msg}\n", "bot")
    chat_display.insert(tk.END, "\n" + "="*50 + "\n")

    # Video playback logic
    cap_ref = cv2.VideoCapture(str(ref_path))
    cap_test = cv2.VideoCapture(str(test_path))
    frame_idx = 0
    frame_count = min(len(ref_kp), len(test_kp))
    paused = start_paused

    def update_frame():
        nonlocal frame_idx
        if frame_count == 0:
            return
            
        cap_ref.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        cap_test.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret_ref, frame_ref = cap_ref.read()
        ret_test, frame_test = cap_test.read()
        
        if ret_ref and ret_test:
            # Scale frames to reasonable size
            frame_ref = cv2.resize(frame_ref, (320, 240))
            frame_test = cv2.resize(frame_test, (320, 240))
            
            # Convert keypoints to scaled coordinates
            ref_scaled = [(kp[0] * 320, kp[1] * 240, kp[2]) for kp in ref_kp[frame_idx]]
            test_scaled = [(kp[0] * 320, kp[1] * 240, kp[2]) for kp in test_kp[frame_idx]]
            
            draw_skeleton(frame_ref, ref_scaled)
            draw_skeleton(frame_test, test_scaled)
            combined = cv2.hconcat([frame_ref, frame_test])
            
            # Add labels and score
            cv2.putText(combined, "Reference", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(combined, "Your Swing", (330, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(combined, f"Score: {score:.4f}", (10, combined.shape[0] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(combined, f"Frame: {frame_idx+1}/{frame_count}", (450, combined.shape[0] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            img = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
            imgtk = ImageTk.PhotoImage(Image.fromarray(img))
            video_label.imgtk = imgtk
            video_label.configure(image=imgtk)
        
        if not paused:
            frame_idx = (frame_idx + 1) % frame_count
        root.after(30, update_frame)

    def toggle_pause(event=None):
        nonlocal paused
        paused = not paused

    def step_forward(event=None):
        nonlocal frame_idx
        if paused:
            frame_idx = (frame_idx + 1) % frame_count

    def step_backward(event=None):
        nonlocal frame_idx
        if paused:
            frame_idx = (frame_idx - 1) % frame_count

    # Video control buttons
    control_frame = tk.Frame(video_frame)
    control_frame.pack(pady=10)
    
    tk.Button(control_frame, text="⏯️ 再生/停止", command=toggle_pause, font=("Arial", 10)).pack(side=tk.LEFT, padx=5)
    tk.Button(control_frame, text="⏪ 前フレーム", command=step_backward, font=("Arial", 10)).pack(side=tk.LEFT, padx=5)
    tk.Button(control_frame, text="⏩ 次フレーム", command=step_forward, font=("Arial", 10)).pack(side=tk.LEFT, padx=5)

    # Keyboard bindings
    root.bind("<space>", toggle_pause)
    root.bind("<Left>", step_backward)
    root.bind("<Right>", step_forward)
    root.focus_set()  # Allow keyboard input

    update_frame()
    root.mainloop()
    cap_ref.release()
    cap_test.release()


def show_comparison(ref_path: Path, test_path: Path, ref_kp, test_kp, score, start_paused: bool = False):
    """Display the reference and test swings side by side with skeletons."""
    import cv2
    import numpy as np

    cap_ref = cv2.VideoCapture(str(ref_path))
    cap_test = cv2.VideoCapture(str(test_path))
    frame_idx = 0
    frame_count = min(len(ref_kp), len(test_kp))
    paused = start_paused
    combined = None
    
    while True:
        frame_idx %= frame_count
        cap_ref.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        cap_test.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret_ref, frame_ref = cap_ref.read()
        ret_test, frame_test = cap_test.read()
        
        if not ret_ref or not ret_test:
            break
            
        draw_skeleton(frame_ref, ref_kp[frame_idx])
        draw_skeleton(frame_test, test_kp[frame_idx])
        combined = cv2.hconcat([frame_ref, frame_test])
        
        # Add labels
        cv2.putText(combined, "Reference", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(combined, "Your Swing", (frame_ref.shape[1] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(combined, f"Score: {score:.4f}", (10, combined.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        cv2.imshow("Golf Swing Comparison", combined)

        key = cv2.waitKey(0 if paused else 30) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):
            paused = not paused
        elif key == 83 and paused:  # Right arrow
            frame_idx = (frame_idx + 1) % frame_count
            continue
        elif key == 81 and paused:  # Left arrow
            frame_idx = (frame_idx - 1) % frame_count
            continue
        elif not paused:
            frame_idx = (frame_idx + 1) % frame_count

    cap_ref.release()
    cap_test.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Enhanced Golf Swing Analysis with AI Coaching")
    parser.add_argument("--reference", required=True, help="Reference swing video path")
    parser.add_argument("--test", required=True, help="Test swing video path")
    parser.add_argument(
        "--model",
        default="intel/human-pose-estimation-0001/INT8/human-pose-estimation-0001.xml",
        help="Path to OpenVINO pose model (.xml)",
    )
    parser.add_argument("--device", default="CPU", help="Device name for inference")
    parser.add_argument(
        "--step",
        action="store_true",
        help="Start playback paused for frame-by-frame stepping",
    )
    parser.add_argument(
        "--chat",
        action="store_true",
        help="Show enhanced AI coaching chat panel alongside comparison",
    )
    parser.add_argument(
        "--analysis-only",
        action="store_true",
        help="Show detailed analysis without video display",
    )
    parser.add_argument(
        "--show-openpose",
        action="store_true",
        help="Display each input video with OpenPose results side by side",
    )
    
    args = parser.parse_args()

    print("🏌️ ゴルフスイング解析を開始します...")
    print("📹 動画からキーポイントを抽出中...")
    
    ref_kp = extract_keypoints(Path(args.reference), args.model, args.device)
    test_kp = extract_keypoints(Path(args.test), args.model, args.device)
    
    print("📊 スイング解析を実行中...")
    score = compare_swings(ref_kp, test_kp)
    
    # Create analyzer for detailed analysis
    analyzer = GolfSwingAnalyzer(ref_kp, test_kp)
    analysis = analyzer.analysis_results
    
    print(f"\n🎯 解析結果:")
    print(f"総合スコア: {score:.4f}")
    print(f"評価: {'優秀' if score > 0.9 else '良好' if score > 0.8 else '要改善'}")
    
    if args.analysis_only:
        # Display detailed analysis in console
        print(f"\n📈 フェーズ別スコア:")
        for phase, phase_score in analysis['swing_phases'].items():
            print(f"  • {phase}: {phase_score:.4f}")
        
        print(f"\n🏃 姿勢解析:")
        posture = analysis['posture_analysis']
        print(f"  • 脊椎角度差: {posture['spine_angle_difference']:.2f}度")
        print(f"  • 脊椎一貫性: {posture['spine_consistency']:.4f}")
        
        print(f"\n🎵 テンポ解析:")
        tempo = analysis['tempo_analysis']
        print(f"  • テンポ差: {tempo['tempo_difference']:.4f}")
        print(f"  • 基準テンポ: {tempo['ref_tempo']:.4f}")
        print(f"  • あなたのテンポ: {tempo['test_tempo']:.4f}")
        
        print(f"\n⚖️ バランス解析:")
        balance = analysis['balance_analysis']
        print(f"  • バランス一貫性: {balance['balance_consistency']:.4f}")
        print(f"  • 平均バランス差: {balance['average_balance_difference']:.4f}")
        
        print(f"\n💪 部位別差異 (上位5位):")
        kp_diff = analysis['keypoint_differences']
        sorted_diff = sorted(kp_diff.items(), key=lambda x: x[1], reverse=True)[:5]
        for i, (part, diff) in enumerate(sorted_diff, 1):
            print(f"  {i}. {part}: {diff:.4f}")

    if args.show_openpose:
        play_openpose_side_by_side(Path(args.reference), ref_kp)
        play_openpose_side_by_side(Path(args.test), test_kp)

    elif args.chat:
        print("🤖 AIコーチングシステムを起動します...")
        show_comparison_with_chat(
            Path(args.reference),
            Path(args.test),
            ref_kp,
            test_kp,
            score,
            start_paused=args.step,
        )
    else:
        print("📺 比較表示を開始します...")
        print("操作方法: スペース=再生/停止, 左右矢印=フレーム送り(停止時), q=終了")
        show_comparison(
            Path(args.reference),
            Path(args.test),
            ref_kp,
            test_kp,
            score,
            start_paused=args.step,
        )

    print("✅ 解析完了")


if __name__ == "__main__":
    main()
