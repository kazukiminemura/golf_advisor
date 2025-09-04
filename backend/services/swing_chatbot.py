from __future__ import annotations

from typing import Dict

from backend.analysis.swing_compare import GolfSwingAnalyzer
from backend.config import Settings
from backend.simple_chatbot import SimpleChatBot
from backend.simple_chatbot.config import ChatbotConfig


class EnhancedSwingChatBot:
    """Chatbot for explaining swing quality, keeping responsibilities separate.

    - Depends on analysis via `GolfSwingAnalyzer` (DIP).
    - Uses a lightweight `SimpleChatBot` for phrasing (SRP: generation only).
    """

    PHASE_JP = {
        "address": "アドレス",
        "backswing": "バックスイング",
        "downswing": "ダウンスイング",
        "follow_through": "フォロースルー",
    }

    def __init__(self, ref_kp, test_kp, score: float | None = None):
        self.analyzer = GolfSwingAnalyzer(ref_kp, test_kp)
        self.score = score if score is not None else self.analyzer.analysis_results["overall_score"]
        self.analysis: Dict[str, object] = self.analyzer.analysis_results
        cfg = ChatbotConfig()
        self._simple_bot = SimpleChatBot(
            model_name=cfg.model_name,
            gguf_filename=cfg.gguf_filename,
            backend=cfg.backend,
        )
        phases = self.analysis.get("swing_phases", {})  # type: ignore[assignment]
        worst = sorted(phases.items(), key=lambda x: x[1])[:2] if isinstance(phases, dict) else []

        def jp_name(k: str) -> str:
            return self.PHASE_JP.get(k, k)

        worst_txt = ", ".join(f"{jp_name(k)}({v:.1f})" for k, v in worst) if worst else "なし"
        sys_prompt = (
            "あなたはプロのゴルフコーチです。厳しめだが建設的に、短く具体的に助言します。\n"
            "- 日本語で回答する\n"
            "- 箇条書き中心、各行は簡潔に\n"
            "- 次の練習アクションを1-3個提示\n"
            f"【文脈】総合スコア: {self.score:.1f}。弱点: {worst_txt}。\n"
            "質問に合わせて、必要なら数値や体の位置を具体的に示す"
        )
        try:
            self._simple_bot.set_system_prompt(sys_prompt)
        except Exception:
            pass

    def initial_message(self) -> str:
        s = self.score
        band = "優秀" if s > 90.0 else ("良好" if s > 80.0 else "要改善")
        phases = self.analysis.get("swing_phases", {}) if isinstance(self.analysis, dict) else {}
        base = (
            f"🏌️ 解析完了  総合スコア: {s:.1f}/100  評価: {band}\n"
            f"姿勢(平均脊柱角差): {self.analysis['posture_analysis']['spine_angle_difference']:.1f}°\n"
            "📈 フェーズ別スコア:\n"
            f" • アドレス: {phases.get('address', 0.0):.3f}\n"
            f" • バックスイング: {phases.get('backswing', 0.0):.3f}\n"
            f" • ダウンスイング: {phases.get('downswing', 0.0):.3f}\n"
            f" • フォロースルー: {phases.get('follow_through', 0.0):.3f}"
        )
        advice = self._generate_advice()
        full = f"{base}\n\n📋 アドバイス:\n{advice}"
        try:
            return self._simple_bot.ask(
                "以下のスコアとアドバイスをこの順番で分かりやすく伝えてください。スコアを先に、続いてアドバイスを述べてください:\n"
                + full
            )
        except Exception:
            return full

    def _generate_advice(self) -> str:
        phases = self.analysis.get("swing_phases", {}) if isinstance(self.analysis, dict) else {}
        sorted_phases = sorted(phases.items(), key=lambda x: x[1]) if isinstance(phases, dict) else []
        s = self.score
        if s >= 90:
            base = "総合的に素晴らしいスイングです。この調子で練習を続けましょう。"
        elif s >= 75:
            base = "概ね良いスイングですが、さらに伸ばせる余地があります。"
        else:
            base = "基礎フォームの見直しが必要です。以下のポイントを重点的に練習しましょう。"

        advice_lines: list[str] = []
        for name, score in sorted_phases:
            jp = self.PHASE_JP.get(name, name)
            if score >= 90:
                advice_lines.append(f"{jp}は非常に良いです (スコア {score:.1f})")
            elif score >= 80:
                advice_lines.append(f"{jp}は概ね良好です。細かな改善を目指しましょう (スコア {score:.1f})")
            else:
                advice_lines.append(f"{jp}を重点的に練習しましょう (スコア {score:.1f})")

        return base + "\n" + "\n".join(f" • {line}" for line in advice_lines)

    def ask(self, message: str) -> str:
        try:
            return self._simple_bot.ask(message)
        except Exception:
            return self._generate_advice()

    def ask_stream(self, message: str):
        """Yield a reply incrementally for streaming clients."""
        try:
            for chunk in self._simple_bot.ask_stream(message):
                yield chunk
        except Exception:
            advice = self._generate_advice()
            for ch in advice:
                yield ch


__all__ = ["EnhancedSwingChatBot"]
