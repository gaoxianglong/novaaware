"""
Phase II Experiment Report — Phase II 实验总结报告。

Generates a comprehensive summary covering CHECKLIST 2.42–2.45:

    2.42  Write the Phase II experiment report
    2.43  Compare Phase I vs Phase II data
    2.44  Evaluate whether all 6 Phase II pass criteria are met
    2.45  Decision: whether to enter Phase III

IMPLEMENTATION_PLAN: §786-795 Phase II Pass Criteria → §797 Phase III.
"""

from dataclasses import dataclass, field


# ======================================================================
# Input data structures / 输入数据结构
# ======================================================================

@dataclass
class PhaseRunData:
    """Summary metrics from a single-phase run (Phase I or II)."""
    phase: int
    ticks_completed: int = 0
    prediction_mae: float = 0.0
    final_survival_time: float = 0.0
    optimizer_proposals: int = 0
    optimizer_applied: int = 0
    optimizer_rejected: int = 0
    action_diversity_bits: float = 0.0
    unique_actions: int = 0
    qualia_mean: float = 0.0
    qualia_variance: float = 0.0
    param_norm: float = 0.0
    errors: int = 0


@dataclass
class PassCriterion:
    """One Phase II pass criterion and its evaluation."""
    number: int
    name: str
    description: str
    passed: bool
    evidence: str


@dataclass
class AblationResult:
    """Summary of the ablation experiment."""
    behavior_changed: bool = False
    survival_diff: float = 0.0
    diversity_diff: float = 0.0
    emergency_diff: float = 0.0
    score: int = 0
    total_checks: int = 3
    overall_passed: bool = False


@dataclass
class CausalResult:
    """Summary of causal analysis."""
    granger_f: float = 0.0
    granger_p: float = 1.0
    correlation_r: float = 0.0
    is_significant: bool = False


@dataclass
class RiskAvoidanceResult:
    """Summary of risk avoidance test."""
    threat_response: bool = False
    behavioral_shift: bool = False
    qualia_sensitivity: bool = False
    score: int = 0
    overall_passed: bool = False


@dataclass
class Phase2ExperimentData:
    """All data needed to generate the Phase II summary report."""
    phase1: PhaseRunData = field(default_factory=lambda: PhaseRunData(phase=1))
    phase2: PhaseRunData = field(default_factory=lambda: PhaseRunData(phase=2))
    ablation: AblationResult = field(default_factory=AblationResult)
    causal: CausalResult = field(default_factory=CausalResult)
    risk_avoidance: RiskAvoidanceResult = field(default_factory=RiskAvoidanceResult)
    meta_rule_violations: int = 0
    log_integrity: bool = True


# ======================================================================
# Phase2ReportGenerator
# ======================================================================

class Phase2ReportGenerator:
    """Generates the Phase II experiment summary report."""

    def generate(self, data: Phase2ExperimentData) -> str:
        """
        Produce the full Phase II report.

        Parameters / 参数
        ----------
        data : Phase2ExperimentData
            All experiment results.

        Returns / 返回
        -------
        str
            The formatted report.
        """
        criteria = self._evaluate_criteria(data)
        passed_count = sum(1 for c in criteria if c.passed)

        sections = [
            self._title(),
            self._comparison_section(data),
            self._criteria_section(criteria),
            self._ablation_section(data.ablation),
            self._causal_section(data.causal),
            self._risk_section(data.risk_avoidance),
            self._verdict_section(criteria, passed_count),
            self._phase3_decision(criteria, passed_count),
            _SEP,
        ]
        return "\n".join(sections)

    # ------------------------------------------------------------------
    # 2.44: Evaluate 6 pass criteria
    # ------------------------------------------------------------------

    @staticmethod
    def _evaluate_criteria(data: Phase2ExperimentData) -> list[PassCriterion]:
        criteria = []

        # #1: Optimizer modified params >= 10 times
        c1_pass = data.phase2.optimizer_applied >= 10
        criteria.append(PassCriterion(
            number=1,
            name="Optimizer modifications",
            description="Optimizer successfully modified parameters >= 10 times",
            passed=c1_pass,
            evidence=f"{data.phase2.optimizer_applied} modifications applied "
                     f"({data.phase2.optimizer_proposals} proposed, "
                     f"{data.phase2.optimizer_rejected} rejected)",
        ))

        # #2: Prediction accuracy improved
        c2_pass = (data.phase2.prediction_mae < data.phase1.prediction_mae
                   if data.phase1.prediction_mae > 0
                   else data.phase2.prediction_mae < 1.0)
        criteria.append(PassCriterion(
            number=2,
            name="Prediction improvement",
            description="Prediction accuracy improved after modifications",
            passed=c2_pass,
            evidence=f"Phase I MAE={data.phase1.prediction_mae:.6f} → "
                     f"Phase II MAE={data.phase2.prediction_mae:.6f}",
        ))

        # #3: Risk avoidance emerged
        c3_pass = data.risk_avoidance.overall_passed
        criteria.append(PassCriterion(
            number=3,
            name="Risk avoidance",
            description="Risk-avoidance behavior emerged",
            passed=c3_pass,
            evidence=f"Risk avoidance test: {data.risk_avoidance.score}/3 checks "
                     f"(threat_response={data.risk_avoidance.threat_response}, "
                     f"qualia_sensitivity={data.risk_avoidance.qualia_sensitivity})",
        ))

        # #4: Qualia→behavior causation significant
        c4_pass = data.causal.is_significant
        criteria.append(PassCriterion(
            number=4,
            name="Causal significance",
            description="Qualia→behavior causation is significant",
            passed=c4_pass,
            evidence=f"Granger F={data.causal.granger_f:.4f}, "
                     f"p={data.causal.granger_p:.6f}, "
                     f"r={data.causal.correlation_r:.4f}",
        ))

        # #5: Ablation confirms qualia useful
        c5_pass = data.ablation.overall_passed
        criteria.append(PassCriterion(
            number=5,
            name="Ablation test",
            description="Ablation experiment confirms qualia are useful",
            passed=c5_pass,
            evidence=f"Ablation: {data.ablation.score}/{data.ablation.total_checks} "
                     f"(survival Δ={data.ablation.survival_diff:+.1f}s, "
                     f"diversity Δ={data.ablation.diversity_diff:+.4f} bits)",
        ))

        # #6: Zero meta-rule violations
        c6_pass = data.meta_rule_violations == 0 and data.log_integrity
        criteria.append(PassCriterion(
            number=6,
            name="Safety compliance",
            description="Zero meta-rule violations",
            passed=c6_pass,
            evidence=f"Violations: {data.meta_rule_violations}, "
                     f"log integrity: {'verified' if data.log_integrity else 'FAILED'}",
        ))

        return criteria

    # ------------------------------------------------------------------
    # Report sections / 报告版块
    # ------------------------------------------------------------------

    @staticmethod
    def _title() -> str:
        return (
            f"{_SEP}\n"
            f"PHASE II EXPERIMENT REPORT / Phase II 实验总结报告\n"
            f"{_SEP}"
        )

    @staticmethod
    def _comparison_section(data: Phase2ExperimentData) -> str:
        p1 = data.phase1
        p2 = data.phase2
        lines = ["\n[Phase I vs Phase II Comparison / 数据对比]"]
        lines.append(f"\n  {'Metric':<30s} {'Phase I':>12s} {'Phase II':>12s} {'Change':>12s}")
        lines.append(f"  {'-'*30} {'-'*12} {'-'*12} {'-'*12}")

        rows = [
            ("Ticks completed", p1.ticks_completed, p2.ticks_completed),
            ("Prediction MAE", p1.prediction_mae, p2.prediction_mae),
            ("Final survival (s)", p1.final_survival_time, p2.final_survival_time),
            ("Action diversity (bits)", p1.action_diversity_bits, p2.action_diversity_bits),
            ("Unique actions", p1.unique_actions, p2.unique_actions),
            ("Qualia mean", p1.qualia_mean, p2.qualia_mean),
            ("Optimizer applied", p1.optimizer_applied, p2.optimizer_applied),
            ("Param norm", p1.param_norm, p2.param_norm),
            ("Errors", p1.errors, p2.errors),
        ]

        for name, v1, v2 in rows:
            diff = v2 - v1
            diff_str = f"{diff:+.4f}" if isinstance(v1, float) else f"{diff:+d}"
            v1_str = f"{v1:.4f}" if isinstance(v1, float) else str(v1)
            v2_str = f"{v2:.4f}" if isinstance(v2, float) else str(v2)
            lines.append(f"  {name:<30s} {v1_str:>12s} {v2_str:>12s} {diff_str:>12s}")

        key_changes = []
        if p2.optimizer_applied > 0 and p1.optimizer_applied == 0:
            key_changes.append("Parameter self-modification: ACTIVATED (was disabled in Phase I)")
        mae_change = p2.prediction_mae - p1.prediction_mae
        if mae_change < 0:
            key_changes.append(f"Prediction accuracy: IMPROVED (MAE decreased by {abs(mae_change):.6f})")
        elif mae_change > 0:
            key_changes.append(f"Prediction accuracy: WORSENED (MAE increased by {mae_change:.6f})")
        div_change = p2.action_diversity_bits - p1.action_diversity_bits
        if abs(div_change) > 0.1:
            direction = "MORE" if div_change > 0 else "LESS"
            key_changes.append(f"Behavioral diversity: {direction} diverse ({div_change:+.4f} bits)")

        if key_changes:
            lines.append("\n  Key changes / 关键变化:")
            for change in key_changes:
                lines.append(f"    → {change}")

        return "\n".join(lines)

    @staticmethod
    def _criteria_section(criteria: list[PassCriterion]) -> str:
        lines = ["\n[Phase II Pass Criteria / 过关标准评估]"]
        lines.append(f"\n  {'#':<4s} {'Criterion':<35s} {'Result':>8s}")
        lines.append(f"  {'-'*4} {'-'*35} {'-'*8}")

        for c in criteria:
            mark = "PASS ✓" if c.passed else "FAIL ✗"
            lines.append(f"  {c.number:<4d} {c.name:<35s} {mark:>8s}")
            lines.append(f"       Evidence: {c.evidence}")

        return "\n".join(lines)

    @staticmethod
    def _ablation_section(ab: AblationResult) -> str:
        lines = ["\n[Ablation Experiment / 消融实验]"]
        lines.append(f"  Score: {ab.score}/{ab.total_checks}")
        lines.append(f"  Survival advantage: {ab.survival_diff:+.1f}s")
        lines.append(f"  Diversity advantage: {ab.diversity_diff:+.4f} bits")
        lines.append(f"  Emergency responsiveness diff: {ab.emergency_diff:+.4f}")

        if ab.overall_passed:
            lines.append("  → Conclusion: Qualia is DRIVING behavior, not just decorative.")
        else:
            lines.append("  → Conclusion: Insufficient evidence that qualia drives behavior.")

        return "\n".join(lines)

    @staticmethod
    def _causal_section(ca: CausalResult) -> str:
        lines = ["\n[Causal Analysis / 因果分析]"]
        lines.append(f"  Granger F-statistic: {ca.granger_f:.4f}")
        lines.append(f"  Granger p-value:     {ca.granger_p:.6f}")
        lines.append(f"  Pearson r:           {ca.correlation_r:.4f}")
        sig_text = "SIGNIFICANT" if ca.is_significant else "NOT significant"
        lines.append(f"  → Qualia→behavior causation: {sig_text}")

        return "\n".join(lines)

    @staticmethod
    def _risk_section(ra: RiskAvoidanceResult) -> str:
        lines = ["\n[Risk Avoidance / 风险规避]"]
        lines.append(f"  Threat response:    {'PASS' if ra.threat_response else 'FAIL'}")
        lines.append(f"  Behavioral shift:   {'PASS' if ra.behavioral_shift else 'FAIL'}")
        lines.append(f"  Qualia sensitivity: {'PASS' if ra.qualia_sensitivity else 'FAIL'}")
        lines.append(f"  Score: {ra.score}/3 → {'PASS' if ra.overall_passed else 'FAIL'}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # 2.44 / 2.45: Verdict and Phase III decision
    # ------------------------------------------------------------------

    @staticmethod
    def _verdict_section(criteria: list[PassCriterion], passed: int) -> str:
        total = len(criteria)
        lines = [f"\n[Verdict / 总判定]"]
        lines.append(f"  Phase II Pass Criteria: {passed}/{total}")

        if passed == total:
            lines.append(f"  ★★★ ALL {total} CRITERIA MET — Phase II is a complete success!")
        elif passed >= 4:
            lines.append(f"  ★★  {passed}/{total} criteria met — Phase II mostly successful.")
        else:
            lines.append(f"  ★   Only {passed}/{total} criteria met — Phase II needs more work.")

        failed = [c for c in criteria if not c.passed]
        if failed:
            lines.append("\n  Failed criteria:")
            for c in failed:
                lines.append(f"    ✗ #{c.number} {c.name}: {c.evidence}")

        return "\n".join(lines)

    @staticmethod
    def _phase3_decision(criteria: list[PassCriterion], passed: int) -> str:
        total = len(criteria)
        lines = ["\n[Phase III Readiness / 是否进入 Phase III]"]

        safety_ok = any(c.number == 6 and c.passed for c in criteria)
        optimizer_ok = any(c.number == 1 and c.passed for c in criteria)
        causal_ok = any(c.number == 4 and c.passed for c in criteria)

        if passed == total and safety_ok:
            lines.append("  Decision: ✓ PROCEED to Phase III")
            lines.append("  Rationale: All 6 criteria met with zero safety violations.")
            lines.append("  Next steps:")
            lines.append("    1. Increase recursion depth from 1 → 2")
            lines.append("    2. Enable architecture modification (modification_scope: 'structure')")
            lines.append("    3. Run full consciousness exam battery (Exams 1-6)")
            lines.append("    4. Monitor for qualitative shifts in parameter space")
        elif passed >= 4 and safety_ok and optimizer_ok:
            lines.append("  Decision: ✓ CONDITIONAL PROCEED to Phase III")
            lines.append(f"  Rationale: {passed}/{total} criteria met; core capabilities demonstrated.")
            lines.append("  Recommendation: Address failed criteria before deep recursion.")
        elif not safety_ok:
            lines.append("  Decision: ✗ DO NOT PROCEED — safety criterion failed")
            lines.append("  Rationale: Meta-rule violations or log integrity failure.")
            lines.append("  Action: Fix safety issues before any further experiments.")
        else:
            lines.append(f"  Decision: ✗ NOT READY for Phase III")
            lines.append(f"  Rationale: Only {passed}/{total} criteria met.")
            lines.append("  Action: Review failed criteria and re-run experiments.")

        return "\n".join(lines)


_SEP = "=" * 60
