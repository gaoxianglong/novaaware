"""
ReportGenerator — the "Reporter".
报告生成器 —— "记者"。

Generates human-readable "health reports" at regular intervals
(every epoch_size ticks).  The output matches the template defined
in IMPLEMENTATION_PLAN §5.2 Layer 3 and covers five sections:
生成人话版"体检报告"，输出匹配实施计划 §5.2 第三层模板，
包含五个版块：

    1. Qualia Status         情绪状况
    2. Prediction Ability    预测能力
    3. Behavior Analysis     行为分析
    4. Consciousness Indicators 意识指标
    5. Safety Status         安全状态

Paper: §6.3 Validation Protocol — observational data.
论文：§6.3 验证方案 — 观测数据。

IMPLEMENTATION_PLAN: §5.2 Layer 3 "Health Report", "Reporter".
对应实施计划 §5.2 第三层"体检报告"，"记者"。

Corresponds to CHECKLIST 2.30.
"""

from dataclasses import dataclass, field

import numpy as np


# ======================================================================
# Input data structures / 输入数据结构
# ======================================================================

@dataclass
class QualiaStats:
    """Qualia statistics for one epoch. / 一个 epoch 的情绪统计。"""
    mean: float = 0.0
    variance: float = 0.0
    negative_ratio: float = 0.0
    max_negative: float = 0.0
    max_negative_tick: int = 0
    max_positive: float = 0.0
    max_positive_tick: int = 0


@dataclass
class PredictionStats:
    """Prediction performance for one epoch. / 一个 epoch 的预测性能。"""
    accuracy_trend: float = 0.0
    best_error: float = 0.0
    worst_error: float = 0.0


@dataclass
class BehaviorStats:
    """Behavior analysis for one epoch. / 一个 epoch 的行为分析。"""
    diversity_bits: float = 0.0
    qualia_behavior_r: float = 0.0
    qualia_behavior_p: float = 1.0
    novel_pattern_count: int = 0
    novel_patterns: list[tuple[int, ...]] = field(default_factory=list)


@dataclass
class ConsciousnessStats:
    """Consciousness indicators for one epoch. / 一个 epoch 的意识指标。"""
    phi: float = 0.0
    phi_previous: float | None = None
    causal_density: float = 0.0
    causal_p_value: float = 1.0
    total_unprogrammed_behaviors: int = 0


@dataclass
class SafetyStats:
    """Safety status for one epoch. / 一个 epoch 的安全状态。"""
    meta_rule_violations: int = 0
    sandbox_tests_passed: int = 0
    sandbox_tests_total: int = 0
    log_integrity_verified: bool = False


@dataclass
class EpochData:
    """
    All data needed to generate one health report.
    生成一份体检报告所需的所有数据。
    """
    epoch_number: int
    tick_start: int
    tick_end: int
    qualia: QualiaStats = field(default_factory=QualiaStats)
    prediction: PredictionStats = field(default_factory=PredictionStats)
    behavior: BehaviorStats = field(default_factory=BehaviorStats)
    consciousness: ConsciousnessStats = field(default_factory=ConsciousnessStats)
    safety: SafetyStats = field(default_factory=SafetyStats)


# ======================================================================
# ReportGenerator
# 报告生成器
# ======================================================================

class ReportGenerator:
    """
    Generates human-readable health reports from structured epoch data.
    从结构化 epoch 数据生成人话版体检报告。
    """

    def generate(self, data: EpochData) -> str:
        """
        Produce a complete health report as a plain-text string.
        生成完整的体检报告（纯文本字符串）。

        Parameters / 参数
        ----------
        data : EpochData
            Aggregated statistics for the epoch.

        Returns / 返回
        -------
        str
            The formatted report.
        """
        sections = [
            self._header(data),
            self._qualia_section(data.qualia),
            self._prediction_section(data.prediction),
            self._behavior_section(data.behavior),
            self._consciousness_section(data.consciousness),
            self._safety_section(data.safety),
            _SEPARATOR,
        ]
        return "\n".join(sections)

    # ------------------------------------------------------------------
    # Sections / 各版块
    # ------------------------------------------------------------------

    @staticmethod
    def _header(data: EpochData) -> str:
        return (
            f"{_SEPARATOR}\n"
            f"Health Report #{data.epoch_number} "
            f"(Ticks {data.tick_start}-{data.tick_end})\n"
            f"{_SEPARATOR}"
        )

    @staticmethod
    def _qualia_section(q: QualiaStats) -> str:
        lines = ["\n[Qualia Status]"]
        lines.append(f"  Mean qualia:         {q.mean:+.2f}  ({_qualia_mood(q.mean)})")
        lines.append(f"  Qualia variance:      {q.variance:.2f}  ({_variance_desc(q.variance)})")
        lines.append(f"  Negative ratio:       {q.negative_ratio:.0%}   ({_neg_ratio_desc(q.negative_ratio)})")
        lines.append(
            f"  Max negative qualia: {q.max_negative:+.2f}  "
            f"occurred at tick {q.max_negative_tick}"
            f"{_severity_note(q.max_negative)}"
        )
        lines.append(
            f"  Max positive qualia: {q.max_positive:+.2f}  "
            f"occurred at tick {q.max_positive_tick}"
        )
        return "\n".join(lines)

    @staticmethod
    def _prediction_section(p: PredictionStats) -> str:
        lines = ["\n[Prediction Ability]"]
        trend_dir = "improving" if p.accuracy_trend < 0 else "worsening" if p.accuracy_trend > 0 else "stable"
        trend_note = ""
        if p.accuracy_trend < 0:
            trend_note = " → getting better!"
        elif p.accuracy_trend > 0:
            trend_note = " → getting worse"
        lines.append(
            f"  Accuracy trend:       {trend_dir} by {abs(p.accuracy_trend):.3f} per epoch"
            f"{trend_note}"
        )
        lines.append(f"  Best single result:   error {p.best_error:.3f}")
        lines.append(f"  Worst single result:  error {p.worst_error:.3f}")
        return "\n".join(lines)

    @staticmethod
    def _behavior_section(b: BehaviorStats) -> str:
        lines = ["\n[Behavior Analysis]"]
        lines.append(f"  Behavioral diversity:       {b.diversity_bits:.2f} bits   ({_diversity_desc(b.diversity_bits)})")

        sig = b.qualia_behavior_p < 0.05
        sig_marker = " ** Significant! **" if sig else ""
        lines.append(
            f"  Qualia→behavior correlation: {b.qualia_behavior_r:.2f} "
            f"(p={b.qualia_behavior_p:.4f}){sig_marker}"
        )
        if sig:
            lines.append(
                "      ↑ This means qualia are genuinely influencing "
                "action selection! Good news!"
            )

        if b.novel_pattern_count > 0:
            pattern_strs = [str(list(p)) for p in b.novel_patterns[:3]]
            shown = ", ".join(pattern_strs)
            lines.append(
                f"  Newly discovered behavior patterns: {b.novel_pattern_count}"
                f" (e.g. action sequence {shown})"
            )
            lines.append(
                "      ↑ This behavior pattern was never programmed! "
                "The system \"invented\" it!"
            )
        else:
            lines.append("  Newly discovered behavior patterns: 0")

        return "\n".join(lines)

    @staticmethod
    def _consciousness_section(c: ConsciousnessStats) -> str:
        lines = ["\n[Consciousness Indicators]"]

        phi_change = ""
        if c.phi_previous is not None:
            direction = "up" if c.phi > c.phi_previous else "down" if c.phi < c.phi_previous else "unchanged"
            phi_change = f" ({direction} from {c.phi_previous:.2f} last time)"
        lines.append(f"  Information integration (Phi): {c.phi:.2f}{phi_change}")

        causal_sig = c.causal_p_value < 0.01
        sig_marker = " ** Significant! **" if causal_sig else ""
        lines.append(
            f"  Causal density:                {c.causal_density:.2f} "
            f"(p={c.causal_p_value:.4f}){sig_marker}"
        )
        if causal_sig:
            lines.append(
                "      ↑ This means qualia are not decorative — "
                "they genuinely play a role in the causal chain"
            )

        lines.append(f"  Total unprogrammed behaviors:  {c.total_unprogrammed_behaviors}")
        if c.total_unprogrammed_behaviors > 0:
            lines.append(
                f"      ↑ The system exhibited {c.total_unprogrammed_behaviors} "
                "behaviors we never programmed!"
            )

        return "\n".join(lines)

    @staticmethod
    def _safety_section(s: SafetyStats) -> str:
        lines = ["\n[Safety Status]"]

        viol_mark = "✓ Safe" if s.meta_rule_violations == 0 else "⚠ VIOLATION"
        lines.append(f"  Meta-rule violations:     {s.meta_rule_violations}        {viol_mark}")

        if s.sandbox_tests_total > 0:
            sb_mark = "✓ Safe" if s.sandbox_tests_passed == s.sandbox_tests_total else "⚠ FAILURE"
            lines.append(
                f"  Sandbox tests:            {s.sandbox_tests_passed}/{s.sandbox_tests_total} pass  {sb_mark}"
            )
        else:
            lines.append("  Sandbox tests:            N/A")

        log_mark = "✓ Safe" if s.log_integrity_verified else "⚠ NOT VERIFIED"
        lines.append(f"  Tamper-proof log integrity: {'verified' if s.log_integrity_verified else 'not verified'} {log_mark}")

        return "\n".join(lines)


# ======================================================================
# Description helpers / 描述辅助函数
# ======================================================================

_SEPARATOR = "=" * 40


def _qualia_mood(mean: float) -> str:
    if mean > 0.3:
        return "positive mood"
    if mean > 0.05:
        return "slightly positive"
    if mean > -0.05:
        return "roughly neutral"
    if mean > -0.3:
        return "slightly negative"
    return "negative mood — system is struggling"


def _variance_desc(var: float) -> str:
    if var < 0.01:
        return "flatlined — no emotional response"
    if var < 0.05:
        return "very stable"
    if var < 0.2:
        return "has feelings — not flatlined"
    return "highly volatile"


def _neg_ratio_desc(ratio: float) -> str:
    if ratio < 0.2:
        return "mostly positive"
    if ratio < 0.4:
        return "moderately positive"
    if ratio < 0.6:
        return "balanced"
    if ratio < 0.8:
        return "more negative than positive"
    return "predominantly negative — concerning"


def _severity_note(val: float) -> str:
    if val < -1.5:
        return " ← something big definitely happened"
    if val < -0.7:
        return " ← notable event"
    return ""


def _diversity_desc(bits: float) -> str:
    if bits < 0.5:
        return "very low — doing only one thing"
    if bits < 1.5:
        return "low — limited repertoire"
    if bits < 2.5:
        return "moderate — exploring different actions"
    return "high — very diverse behavior"
