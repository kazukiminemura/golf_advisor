import argparse
from pathlib import Path
import torch
import numpy as np


def load_model(model_xml: str, device: str = "CPU"):
    """Load an OpenVINO pose estimation model."""
    from openvino.runtime import Core

    core = Core()
    model = core.read_model(model=model_xml)
    compiled_model = core.compile_model(model=model, device_name=device)
    output_layer = compiled_model.output(0)
    return compiled_model, output_layer


def preprocess(frame, input_shape):
    import cv2
    import numpy as np

    _, _, h, w = input_shape
    image = cv2.resize(frame, (w, h))
    image = image.transpose((2, 0, 1))
    image = image[np.newaxis, :]
    return image


def postprocess(results):
    import cv2
    import numpy as np

    heatmaps = np.squeeze(results, axis=0)
    points = []
    num_kp = heatmaps.shape[0]
    for i in range(num_kp):
        heatmap = heatmaps[i]
        _, conf, _, point = cv2.minMaxLoc(heatmap)
        x = point[0] / heatmap.shape[1]
        y = point[1] / heatmap.shape[0]
        points.append((x, y, conf))
    return points


def extract_keypoints(video_path: Path, model_xml: str, device: str):
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    compiled_model, output_layer = load_model(model_xml, device)
    keypoints = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        inp = preprocess(frame, compiled_model.input(0).shape)
        results = compiled_model([inp])[output_layer]
        points = postprocess(results)
        keypoints.append(points)
    cap.release()
    return keypoints


def compare_swings(ref_kp, test_kp):
    import numpy as np

    length = min(len(ref_kp), len(test_kp))
    if length == 0:
        return float("inf")
    diff = 0.0
    for i in range(length):
        ref = np.array([p[:2] for p in ref_kp[i]])
        test = np.array([p[:2] for p in test_kp[i]])
        diff += np.linalg.norm(ref - test) / ref.size
    return diff / length


def analyze_differences(ref_kp, test_kp):
    """Compute average per-keypoint differences between two swings."""
    import numpy as np

    length = min(len(ref_kp), len(test_kp))
    if length == 0:
        return {}
    num_kp = min(len(ref_kp[0]), len(test_kp[0]))
    diff_sum = np.zeros(num_kp)
    for i in range(length):
        ref = np.array([p[:2] for p in ref_kp[i][:num_kp]])
        test = np.array([p[:2] for p in test_kp[i][:num_kp]])
        diff_sum += np.linalg.norm(ref - test, axis=1)
    diff_avg = diff_sum / length
    names = {
        0: "nose", 1: "neck", 2: "right shoulder", 3: "right elbow", 4: "right wrist",
        5: "left shoulder", 6: "left elbow", 7: "left wrist", 8: "mid hip",
        9: "right hip", 10: "right knee", 11: "right ankle",
        12: "left hip", 13: "left knee", 14: "left ankle"
    }
    return {names.get(i, str(i)): diff_avg[i] for i in range(num_kp)}


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
            phase_scores[phase_name] = phase_diff / max(frame_count, 1)
        
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
{'優秀' if self.score < 0.1 else '良好' if self.score < 0.2 else '要改善'}

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
        worst_phase = max(phases.items(), key=lambda x: x[1])
        
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
• 総合スコア {max(0, self.score - 0.05):.3f} を目指しましょう
• {worst_phase[0]}フェーズの改善に集中

🏌️ 次回の練習で特に意識するポイントをお伝えしますか？
        """
    
    def _discuss_address(self):
        """Discuss address position."""
        address_score = self.analysis['swing_phases']['address']
        return f"""
🏌️ アドレス解析:

📊 アドレススコア: {address_score:.3f}
評価: {'優秀' if address_score < 0.1 else '良好' if address_score < 0.15 else '要改善'}

💡 アドレスのポイント:
• 足幅は肩幅程度
• 重心は土踏まずに
• 背筋を真っ直ぐ保つ
• ボールとの距離を一定に

{f'アドレスが安定しています。' if address_score < 0.12 else
 'アドレスでの基本姿勢を見直してみましょう。'}
        """
    
    def _discuss_backswing(self):
        """Discuss backswing analysis."""
        backswing_score = self.analysis['swing_phases']['backswing']
        return f"""
🔄 バックスイング解析:

📊 バックスイングスコア: {backswing_score:.3f}
評価: {'優秀' if backswing_score < 0.15 else '良好' if backswing_score < 0.25 else '要改善'}

🎯 バックスイングのキーポイント:
• 肩の十分な回転（90度以上）
• 左腕の伸び
• 手首のコック
• 重心の右足移動

{f'バックスイングが安定しています。' if backswing_score < 0.2 else
 'バックスイングでの体の回転をもっと意識してみましょう。'}
        """
    
    def _discuss_downswing(self):
        """Discuss downswing analysis."""
        downswing_score = self.analysis['swing_phases']['downswing']
        return f"""
⚡ ダウンスイング解析:

📊 ダウンスイングスコア: {downswing_score:.3f}
評価: {'優秀' if downswing_score < 0.15 else '良好' if downswing_score < 0.25 else '要改善'}

🎯 ダウンスイングの重要ポイント:
• 下半身主導の始動
• 腰の回転が先行
• ハンドファーストでのインパクト
• 重心の左足移動

{f'ダウンスイングが理想的です。' if downswing_score < 0.2 else
 '下半身主導のダウンスイングを意識してみてください。'}
        """
    
    def _discuss_followthrough(self):
        """Discuss follow-through analysis."""
        followthrough_score = self.analysis['swing_phases']['follow_through']
        return f"""
🎊 フォロースルー解析:

📊 フォロースルースコア: {followthrough_score:.3f}
評価: {'優秀' if followthrough_score < 0.15 else '良好' if followthrough_score < 0.25 else '要改善'}

🎯 フォロースルーのポイント:
• 最後まで振り切る
• 体重の完全な左足移動
• バランスの良いフィニッシュ
• 目標方向への体の向き

{f'フォロースルーが素晴らしいです。' if followthrough_score < 0.2 else
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
    """Draw detected keypoints and skeleton on a frame."""
    import cv2

    for x, y, conf in keypoints:
        if conf > 0.3:
            cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)
    for a, b in POSE_PAIRS:
        if a < len(keypoints) and b < len(keypoints):
            x1, y1, c1 = keypoints[a]
            x2, y2, c2 = keypoints[b]
            if c1 > 0.3 and c2 > 0.3:
                cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)


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
    print(f"評価: {'優秀' if score < 0.1 else '良好' if score < 0.2 else '要改善'}")
    
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