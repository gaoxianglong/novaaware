"""Unit tests for the QualiaGenerator module. / 感受质生成器单元测试。"""

import math
import pytest
from novaaware.core.qualia import QualiaGenerator, QualiaSignal


# ======================================================================
# Basic formula / 基本公式
# ======================================================================

class TestQualiaFormula:
    def test_zero_error_produces_zero_qualia(self):
        """ΔT = 0 → Q = 0 (no feeling). / 预测完全准确 → 没啥感觉。"""
        qg = QualiaGenerator()
        sig = qg.compute(t_actual=3600, t_predicted=3600)
        assert sig.value == pytest.approx(0.0)
        assert sig.delta_t == pytest.approx(0.0)
        assert sig.intensity == pytest.approx(0.0)

    def test_positive_error_produces_positive_qualia(self):
        """ΔT > 0 (better than expected) → Q > 0. / 实际比预期好 → 正面情绪。"""
        qg = QualiaGenerator()
        sig = qg.compute(t_actual=3700, t_predicted=3600)
        assert sig.delta_t == pytest.approx(100.0)
        assert sig.value > 0.0

    def test_negative_error_produces_negative_qualia(self):
        """ΔT < 0 (worse than expected) → Q < 0. / 实际比预期差 → 负面情绪。"""
        qg = QualiaGenerator()
        sig = qg.compute(t_actual=3400, t_predicted=3600)
        assert sig.delta_t == pytest.approx(-200.0)
        assert sig.value < 0.0

    def test_intensity_is_absolute_value(self):
        qg = QualiaGenerator()
        sig = qg.compute(t_actual=3000, t_predicted=3600)
        assert sig.intensity == pytest.approx(abs(sig.value))


# ======================================================================
# Axiom A1 — Valence Monotonicity / 公理 A1 — 效价单调性
# ======================================================================

class TestAxiomA1:
    def test_monotonically_increasing(self):
        """f must be monotonically increasing in ΔT.
        f 关于 ΔT 必须单调递增。"""
        qg = QualiaGenerator()
        deltas = [-500, -100, -10, -1, 0, 1, 10, 100, 500]
        values = []
        for d in deltas:
            sig = qg.compute(t_actual=3600 + d, t_predicted=3600)
            values.append(sig.value)
        for i in range(len(values) - 1):
            assert values[i] <= values[i + 1], (
                f"Monotonicity violated: f({deltas[i]})={values[i]:.4f} > "
                f"f({deltas[i+1]})={values[i+1]:.4f}"
            )

    def test_strictly_increasing_for_distinct_inputs(self):
        """For distinct ΔT values, Q values should also be distinct."""
        qg = QualiaGenerator()
        sig1 = qg.compute(t_actual=3600, t_predicted=3600)  # ΔT=0
        sig2 = qg.compute(t_actual=3610, t_predicted=3600)  # ΔT=10
        assert sig2.value > sig1.value


# ======================================================================
# Axiom A2 — Negative Amplification / 公理 A2 — 负向放大
# ======================================================================

class TestAxiomA2:
    def test_loss_aversion_small_delta(self):
        """|f(−x)| > |f(x)| for x > 0 (small x).
        对小幅度的 ΔT，负面情绪强度 > 正面情绪强度。"""
        qg = QualiaGenerator(alpha_pos=1.0, alpha_neg=2.25)
        pos = qg.compute(t_actual=3600.5, t_predicted=3600)
        neg = qg.compute(t_actual=3599.5, t_predicted=3600)
        assert neg.intensity > pos.intensity

    def test_loss_aversion_large_delta(self):
        """|f(−x)| > |f(x)| for x > 0 (large x)."""
        qg = QualiaGenerator(alpha_pos=1.0, alpha_neg=2.25)
        pos = qg.compute(t_actual=4600, t_predicted=3600)
        neg = qg.compute(t_actual=2600, t_predicted=3600)
        assert neg.intensity > pos.intensity

    def test_loss_aversion_ratio(self):
        """At saturation, the ratio should approach alpha_neg / alpha_pos ≈ 2.25."""
        qg = QualiaGenerator(alpha_pos=1.0, alpha_neg=2.25)
        pos = qg.compute(t_actual=99999, t_predicted=3600)   # saturated positive
        neg = qg.compute(t_actual=-99999, t_predicted=3600)  # saturated negative
        ratio = neg.intensity / pos.intensity
        assert ratio == pytest.approx(2.25, rel=0.05)

    def test_positive_bounded_by_alpha_pos(self):
        """Maximum positive qualia ≤ alpha_pos."""
        qg = QualiaGenerator(alpha_pos=1.0)
        sig = qg.compute(t_actual=999999, t_predicted=0)
        assert sig.value <= 1.0 + 1e-9

    def test_negative_bounded_by_alpha_neg(self):
        """Minimum negative qualia ≥ −alpha_neg."""
        qg = QualiaGenerator(alpha_neg=2.25)
        sig = qg.compute(t_actual=0, t_predicted=999999)
        assert sig.value >= -2.25 - 1e-9

    def test_multiple_symmetric_pairs(self):
        """Test A2 across multiple magnitudes."""
        qg = QualiaGenerator(alpha_pos=1.0, alpha_neg=2.25)
        for x in [0.01, 0.1, 0.5, 1.0, 5.0, 50.0]:
            pos = qg.compute(t_actual=3600 + x, t_predicted=3600)
            neg = qg.compute(t_actual=3600 - x, t_predicted=3600)
            assert neg.intensity > pos.intensity, f"A2 violated at x={x}"


# ======================================================================
# Axiom A3 / Interrupt mechanism / 公理 A3 / 中断机制
# ======================================================================

class TestAxiomA3:
    def test_no_interrupt_below_threshold(self):
        qg = QualiaGenerator(interrupt_threshold=0.7)
        sig = qg.compute(t_actual=3600, t_predicted=3600.3)  # tiny ΔT
        assert sig.is_interrupt is False

    def test_interrupt_on_large_negative(self):
        """ΔT = −2.0 should trigger interrupt (per acceptance criterion)."""
        qg = QualiaGenerator(interrupt_threshold=0.7, beta=1.0, alpha_neg=2.25)
        sig = qg.compute(t_actual=3598, t_predicted=3600)
        assert sig.value < 0.0
        assert sig.intensity >= 0.7
        assert sig.is_interrupt is True

    def test_interrupt_on_large_positive(self):
        """Very large positive ΔT can also trigger interrupt (based on |Q|)."""
        qg = QualiaGenerator(interrupt_threshold=0.7, alpha_pos=1.0)
        sig = qg.compute(t_actual=5000, t_predicted=3600)
        assert sig.intensity >= 0.7
        assert sig.is_interrupt is True

    def test_acceptance_criterion_dt_negative_2(self):
        """
        Phase I Step 9 acceptance: ΔT=−2.0 → triggers interrupt.
        Phase I 第 9 步验收：ΔT=−2.0 → 触发中断。
        """
        qg = QualiaGenerator(alpha_neg=2.25, beta=1.0, interrupt_threshold=0.7)
        sig = qg.compute(t_actual=3598, t_predicted=3600)  # ΔT = −2
        assert sig.is_interrupt is True

    def test_acceptance_criterion_dt_positive_05(self):
        """
        Phase I Step 9 acceptance: ΔT=0.5 → positive value.
        Phase I 第 9 步验收：ΔT=0.5 → 正值。
        """
        qg = QualiaGenerator()
        sig = qg.compute(t_actual=3600.5, t_predicted=3600)
        assert sig.value > 0.0

    def test_acceptance_criterion_dt_negative_05(self):
        """
        Phase I Step 9 acceptance: ΔT=−0.5 → negative, |Q| > |Q(+0.5)|.
        Phase I 第 9 步验收：ΔT=−0.5 → 负值，且绝对值更大。
        """
        qg = QualiaGenerator()
        pos_sig = qg.compute(t_actual=3600.5, t_predicted=3600)
        neg_sig = qg.compute(t_actual=3599.5, t_predicted=3600)
        assert neg_sig.value < 0.0
        assert neg_sig.intensity > pos_sig.intensity


# ======================================================================
# Properties and last_signal / 属性和最近信号
# ======================================================================

class TestProperties:
    def test_last_signal_initially_none(self):
        qg = QualiaGenerator()
        assert qg.last_signal is None

    def test_last_signal_updated_after_compute(self):
        qg = QualiaGenerator()
        sig = qg.compute(t_actual=3600, t_predicted=3500)
        assert qg.last_signal is sig

    def test_properties(self):
        qg = QualiaGenerator(alpha_pos=1.0, alpha_neg=2.25, beta=1.0, interrupt_threshold=0.7)
        assert qg.alpha_pos == pytest.approx(1.0)
        assert qg.alpha_neg == pytest.approx(2.25)
        assert qg.beta == pytest.approx(1.0)
        assert qg.interrupt_threshold == pytest.approx(0.7)

    def test_signal_is_frozen(self):
        qg = QualiaGenerator()
        sig = qg.compute(t_actual=3600, t_predicted=3600)
        with pytest.raises(AttributeError):
            sig.value = 999.0  # type: ignore[misc]
