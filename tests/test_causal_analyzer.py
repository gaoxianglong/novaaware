"""
Unit tests for CausalAnalyzer — the "Causal Detective".
因果分析器单元测试 —— "因果侦探"。

Tests cover:
  - Granger causality with known causal vs non-causal data
  - Controlled Granger causality (partialling out environment)
  - Edge cases: insufficient data, constant series, length mismatch
  - F-distribution p-value accuracy (compared against known values)
  - BIC-based lag selection

Corresponds to CHECKLIST 2.25–2.26.
"""

import math

import numpy as np
import pytest

from novaaware.observation.causal_analyzer import (
    GrangerResult,
    granger_causality,
    controlled_granger_causality,
    _f_survival,
    _betainc,
)


# ======================================================================
# 1. F-distribution / Beta function internals
# ======================================================================

class TestFDistribution:

    def test_f_survival_zero_returns_one(self):
        assert _f_survival(0.0, 5, 50) == 1.0

    def test_f_survival_negative_returns_one(self):
        assert _f_survival(-1.0, 5, 50) == 1.0

    def test_f_survival_large_f_near_zero(self):
        """Very large F → p ≈ 0."""
        p = _f_survival(100.0, 5, 100)
        assert p < 0.001

    def test_f_survival_f_equals_one(self):
        """F = 1 with equal df should give p ≈ 0.5 (approximately)."""
        p = _f_survival(1.0, 10, 10)
        assert 0.3 < p < 0.7

    def test_f_survival_known_value_d1_1_d2_1(self):
        """F(1,1) at f=1 → p = 0.5 (by symmetry of Cauchy)."""
        p = _f_survival(1.0, 1, 1)
        assert abs(p - 0.5) < 0.05

    def test_f_survival_known_chi2_limit(self):
        """F(d1, ∞) approaches chi² / d1. For large d2, compare."""
        p = _f_survival(3.0, 5, 10000)
        assert 0.0 < p < 0.02

    def test_f_survival_monotone(self):
        """P(F ≥ f) should decrease as f increases."""
        p_values = [_f_survival(f, 3, 50) for f in [0.5, 1.0, 2.0, 5.0, 10.0]]
        for i in range(len(p_values) - 1):
            assert p_values[i] >= p_values[i + 1]

    def test_betainc_boundary_zero(self):
        assert _betainc(2.0, 3.0, 0.0) == 0.0

    def test_betainc_boundary_one(self):
        assert _betainc(2.0, 3.0, 1.0) == 1.0

    def test_betainc_symmetry(self):
        """I_x(a, b) = 1 - I_{1-x}(b, a)."""
        a, b, x = 3.0, 5.0, 0.4
        lhs = _betainc(a, b, x)
        rhs = 1.0 - _betainc(b, a, 1.0 - x)
        assert abs(lhs - rhs) < 1e-10

    def test_betainc_half_when_a_equals_b(self):
        """I_{0.5}(a, a) = 0.5 by symmetry."""
        val = _betainc(5.0, 5.0, 0.5)
        assert abs(val - 0.5) < 1e-8


# ======================================================================
# 2. Granger Causality (2.25): Q → A
# ======================================================================

class TestGrangerCausality:

    def test_insufficient_data(self):
        """Too few data points → p = 1.0, not significant."""
        result = granger_causality([0.1] * 10, [1] * 10, max_lag=5)
        assert result.p_value == 1.0
        assert result.is_significant is False

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            granger_causality([1.0, 2.0], [1.0])

    def test_independent_series_not_significant(self):
        """Two independent random series → no Granger causality."""
        np.random.seed(42)
        n = 500
        cause = list(np.random.randn(n))
        effect = list(np.random.randn(n))
        result = granger_causality(cause, effect, max_lag=3)
        assert result.p_value > 0.01
        assert result.is_significant is False

    def test_causal_series_significant(self):
        """
        Construct A(t) = 0.5 * Q(t-1) + noise.
        Q should Granger-cause A.
        """
        np.random.seed(42)
        n = 500
        q = np.random.randn(n)
        a = np.zeros(n)
        for t in range(1, n):
            a[t] = 0.5 * q[t - 1] + np.random.randn() * 0.3

        result = granger_causality(list(q), list(a), max_lag=3)
        assert result.is_significant is True
        assert result.p_value < 0.01
        assert result.f_statistic > 0

    def test_reverse_causality_not_significant(self):
        """
        If A(t) = 0.5 * Q(t-1) + noise, then A should NOT
        Granger-cause Q (reverse direction).
        """
        np.random.seed(42)
        n = 500
        q = np.random.randn(n)
        a = np.zeros(n)
        for t in range(1, n):
            a[t] = 0.5 * q[t - 1] + np.random.randn() * 0.3

        result = granger_causality(list(a), list(q), max_lag=3)
        assert result.p_value > 0.01

    def test_strong_causality_very_low_p(self):
        """Strong causal signal → very low p-value."""
        np.random.seed(42)
        n = 1000
        q = np.random.randn(n)
        a = np.zeros(n)
        for t in range(1, n):
            a[t] = 0.8 * q[t - 1] + np.random.randn() * 0.1

        result = granger_causality(list(q), list(a), max_lag=3)
        assert result.p_value < 0.001

    def test_constant_cause_not_significant(self):
        """Constant cause cannot predict anything."""
        np.random.seed(42)
        n = 200
        cause = [0.5] * n
        effect = list(np.random.randn(n))
        result = granger_causality(cause, effect, max_lag=3)
        assert result.is_significant is False

    def test_result_fields(self):
        """GrangerResult should have all expected fields."""
        np.random.seed(42)
        n = 200
        cause = list(np.random.randn(n))
        effect = list(np.random.randn(n))
        result = granger_causality(cause, effect, max_lag=3)
        assert isinstance(result.f_statistic, float)
        assert isinstance(result.p_value, float)
        assert isinstance(result.is_significant, bool)
        assert isinstance(result.lag_order, int)
        assert result.lag_order >= 1
        assert result.num_observations > 0

    def test_lag_selection(self):
        """Lag should be between 1 and max_lag."""
        np.random.seed(42)
        n = 300
        cause = list(np.random.randn(n))
        effect = list(np.random.randn(n))
        result = granger_causality(cause, effect, max_lag=5)
        assert 1 <= result.lag_order <= 5

    def test_multi_lag_causality(self):
        """
        A(t) depends on Q(t-3) → should detect with max_lag ≥ 3.
        """
        np.random.seed(42)
        n = 500
        q = np.random.randn(n)
        a = np.zeros(n)
        for t in range(3, n):
            a[t] = 0.6 * q[t - 3] + np.random.randn() * 0.3

        result = granger_causality(list(q), list(a), max_lag=5)
        assert result.is_significant is True

    def test_rss_unrestricted_leq_restricted(self):
        """Unrestricted model should fit at least as well as restricted."""
        np.random.seed(42)
        n = 300
        q = np.random.randn(n)
        a = np.zeros(n)
        for t in range(1, n):
            a[t] = 0.5 * q[t - 1] + np.random.randn() * 0.5

        result = granger_causality(list(q), list(a), max_lag=3)
        assert result.rss_unrestricted <= result.rss_restricted + 1e-8


# ======================================================================
# 3. Controlled Granger Causality (2.26): Q → A | E
# ======================================================================

class TestControlledGrangerCausality:

    def test_insufficient_data(self):
        result = controlled_granger_causality(
            [0.1] * 10, [1] * 10, [[0.5] * 10], max_lag=5,
        )
        assert result.p_value == 1.0

    def test_length_mismatch_cause_effect(self):
        with pytest.raises(ValueError, match="same length"):
            controlled_granger_causality([1.0], [1.0, 2.0], [[1.0]])

    def test_length_mismatch_control(self):
        with pytest.raises(ValueError, match="control"):
            controlled_granger_causality(
                [1.0] * 100, [1.0] * 100, [[0.5] * 50],
            )

    def test_spurious_correlation_removed(self):
        """
        E drives both Q and A. After controlling for E, Q → A should
        NOT be significant.
        """
        np.random.seed(42)
        n = 500
        e = np.random.randn(n)
        q = 0.8 * e + np.random.randn(n) * 0.2
        a = np.zeros(n)
        for t in range(1, n):
            a[t] = 0.8 * e[t - 1] + np.random.randn() * 0.2

        result = controlled_granger_causality(
            list(q), list(a), [list(e)], max_lag=3,
        )
        assert result.p_value > 0.01 or not result.is_significant

    def test_genuine_causality_survives_control(self):
        """
        Q independently drives A (even after controlling for E).
        Q → A | E should still be significant.
        """
        np.random.seed(42)
        n = 500
        e = np.random.randn(n)
        q = np.random.randn(n)
        a = np.zeros(n)
        for t in range(1, n):
            a[t] = 0.5 * q[t - 1] + 0.3 * e[t - 1] + np.random.randn() * 0.3

        result = controlled_granger_causality(
            list(q), list(a), [list(e)], max_lag=3,
        )
        assert result.is_significant is True
        assert result.p_value < 0.01

    def test_multiple_control_variables(self):
        """Support multiple control variables simultaneously."""
        np.random.seed(42)
        n = 500
        e1 = np.random.randn(n)
        e2 = np.random.randn(n)
        q = np.random.randn(n)
        a = np.zeros(n)
        for t in range(1, n):
            a[t] = 0.5 * q[t - 1] + 0.2 * e1[t - 1] + 0.2 * e2[t - 1] + np.random.randn() * 0.3

        result = controlled_granger_causality(
            list(q), list(a), [list(e1), list(e2)], max_lag=3,
        )
        assert result.is_significant is True

    def test_empty_control_equivalent_to_unconditional(self):
        """No control variables → should behave like unconditional Granger."""
        np.random.seed(42)
        n = 300
        q = np.random.randn(n)
        a = np.zeros(n)
        for t in range(1, n):
            a[t] = 0.5 * q[t - 1] + np.random.randn() * 0.3

        result_plain = granger_causality(list(q), list(a), max_lag=3)
        result_ctrl = controlled_granger_causality(
            list(q), list(a), [], max_lag=3,
        )

        assert result_plain.is_significant == result_ctrl.is_significant

    def test_result_fields(self):
        np.random.seed(42)
        n = 200
        result = controlled_granger_causality(
            list(np.random.randn(n)),
            list(np.random.randn(n)),
            [list(np.random.randn(n))],
            max_lag=3,
        )
        assert isinstance(result.f_statistic, float)
        assert isinstance(result.p_value, float)
        assert isinstance(result.is_significant, bool)
        assert result.lag_order >= 1


# ======================================================================
# 4. Cross-validation: known VAR process
# ======================================================================

class TestKnownVARProcess:

    def test_var_process_bidirectional(self):
        """
        VAR(1): Q(t) = 0.3 A(t-1) + ε₁
                A(t) = 0.5 Q(t-1) + ε₂

        Both directions should be significant.
        """
        np.random.seed(42)
        n = 500
        q = np.zeros(n)
        a = np.zeros(n)
        for t in range(1, n):
            q[t] = 0.3 * a[t - 1] + np.random.randn() * 0.3
            a[t] = 0.5 * q[t - 1] + np.random.randn() * 0.3

        qa = granger_causality(list(q), list(a), max_lag=3)
        aq = granger_causality(list(a), list(q), max_lag=3)

        assert qa.is_significant is True
        assert aq.is_significant is True

    def test_unidirectional_var(self):
        """
        VAR(1): Q(t) = ε₁ (exogenous)
                A(t) = 0.5 Q(t-1) + ε₂

        Q → A significant, A → Q not significant.
        """
        np.random.seed(42)
        n = 500
        q = np.random.randn(n)
        a = np.zeros(n)
        for t in range(1, n):
            a[t] = 0.5 * q[t - 1] + np.random.randn() * 0.3

        qa = granger_causality(list(q), list(a), max_lag=3)
        aq = granger_causality(list(a), list(q), max_lag=3)

        assert qa.is_significant is True
        assert aq.is_significant is False


# ======================================================================
# 5. Realistic NovaAware scenario
# ======================================================================

class TestRealisticScenario:

    def test_qualia_drives_action_in_simulation(self):
        """
        Simulate a NovaAware-like system: negative qualia → emergency
        action (high ID), positive qualia → idle (low ID).
        """
        np.random.seed(42)
        n = 500
        qualia: list[float] = []
        actions: list[float] = []

        for t in range(n):
            q = np.sin(2 * np.pi * t / 50) + np.random.randn() * 0.2
            qualia.append(q)
            if t == 0:
                actions.append(0.0)
            else:
                if qualia[t - 1] < -0.5:
                    a = 7.0 + np.random.randn() * 0.5
                elif qualia[t - 1] > 0.5:
                    a = 0.0 + np.random.randn() * 0.5
                else:
                    a = 3.0 + np.random.randn() * 0.5
                actions.append(a)

        result = granger_causality(qualia, actions, max_lag=3)
        assert result.is_significant is True
        assert result.p_value < 0.01

    def test_environment_confound_controlled(self):
        """
        Environment E drives both Q and A. After controlling for E,
        residual Q → A causality should be weak.
        """
        np.random.seed(42)
        n = 500
        env = np.sin(2 * np.pi * np.arange(n) / 100) + np.random.randn(n) * 0.1
        qualia = list(0.7 * env + np.random.randn(n) * 0.3)
        actions: list[float] = [0.0]
        for t in range(1, n):
            actions.append(0.6 * env[t - 1] + np.random.randn() * 0.3)

        plain = granger_causality(qualia, actions, max_lag=3)
        controlled = controlled_granger_causality(
            qualia, actions, [list(env)], max_lag=3,
        )

        if plain.is_significant:
            assert controlled.p_value >= plain.p_value or not controlled.is_significant
