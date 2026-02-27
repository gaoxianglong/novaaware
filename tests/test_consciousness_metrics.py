"""
Unit tests for ConsciousnessMetrics — the "consciousness dashboard".
意识指标模块单元测试 —— "意识仪表盘"。

Tests cover:
  - Phi (Φ) information integration: independent vs correlated dimensions
  - Behavioral diversity: single action, uniform, skewed distributions
  - Qualia-behavior correlation: no correlation, strong coupling, edge cases
  - Cross-metric consistency with complex scenarios

Corresponds to CHECKLIST 2.22–2.24.
"""

import math

import numpy as np
import pytest

from novaaware.observation.consciousness_metrics import (
    PhiResult,
    DiversityResult,
    CorrelationResult,
    compute_phi,
    compute_behavioral_diversity,
    compute_qualia_behavior_correlation,
)


# ======================================================================
# 1. Phi (Φ) — Information Integration / 信息整合度
# ======================================================================

class TestPhi:

    def test_empty_history_returns_zero(self):
        result = compute_phi([])
        assert result.value == 0.0
        assert result.num_samples == 0

    def test_insufficient_samples_returns_zero(self):
        states = [np.zeros(32) for _ in range(5)]
        result = compute_phi(states, min_samples=10)
        assert result.value == 0.0
        assert result.num_samples == 5

    def test_single_varying_dimension_returns_zero(self):
        """Only one active dimension → no integration possible."""
        np.random.seed(42)
        states = []
        for _ in range(100):
            s = np.zeros(32)
            s[0] = np.random.randn()
            states.append(s)
        result = compute_phi(states)
        assert result.value == 0.0

    def test_constant_state_returns_zero(self):
        """All dimensions constant → zero variance → Φ = 0."""
        states = [np.ones(16) * 0.5 for _ in range(50)]
        result = compute_phi(states)
        assert result.value == 0.0

    def test_independent_dimensions_low_phi(self):
        """Independent random dimensions → Φ near 0."""
        np.random.seed(42)
        states = [np.random.randn(32) for _ in range(200)]
        result = compute_phi(states)
        assert result.value < 2.0

    def test_correlated_dimensions_high_phi(self):
        """All dimensions driven by one common signal → high Φ."""
        np.random.seed(42)
        states = []
        for _ in range(200):
            base = np.random.randn()
            s = np.full(32, base) + np.random.randn(32) * 0.01
            states.append(s)
        result = compute_phi(states)
        assert result.value > 1.0

    def test_correlated_higher_than_independent(self):
        """Correlated states should have strictly higher Φ than independent."""
        np.random.seed(42)
        independent = [np.random.randn(32) for _ in range(200)]
        phi_indep = compute_phi(independent)

        np.random.seed(99)
        correlated = []
        for _ in range(200):
            base = np.random.randn()
            s = np.full(32, base) + np.random.randn(32) * 0.01
            correlated.append(s)
        phi_corr = compute_phi(correlated)

        assert phi_corr.value > phi_indep.value

    def test_constant_dimensions_filtered(self):
        """Constant dimensions should be ignored, not cause NaN."""
        np.random.seed(42)
        states = []
        for _ in range(100):
            s = np.zeros(32)
            s[0] = np.random.randn()
            s[1] = np.random.randn()
            states.append(s)
        result = compute_phi(states)
        assert math.isfinite(result.value)
        assert result.num_dimensions == 32

    def test_phi_always_non_negative(self):
        """Φ should always be ≥ 0, regardless of data."""
        for seed in range(10):
            np.random.seed(seed)
            states = [np.random.randn(8) for _ in range(50)]
            result = compute_phi(states)
            assert result.value >= 0.0

    def test_result_metadata(self):
        """PhiResult should carry correct metadata."""
        np.random.seed(42)
        states = [np.random.randn(16) for _ in range(60)]
        result = compute_phi(states)
        assert result.num_samples == 60
        assert result.num_dimensions == 16
        assert result.normalized >= 0.0

    def test_normalized_phi(self):
        """normalized = value / k_active should be smaller than value for k > 1."""
        np.random.seed(42)
        states = []
        for _ in range(200):
            base = np.random.randn()
            s = np.full(8, base) + np.random.randn(8) * 0.1
            states.append(s)
        result = compute_phi(states)
        assert result.value > 0
        assert result.normalized < result.value

    def test_two_correlated_dims(self):
        """Two perfectly correlated dimensions → Φ > 0."""
        np.random.seed(42)
        states = []
        for _ in range(100):
            v = np.random.randn()
            s = np.zeros(4)
            s[0] = v
            s[1] = v + np.random.randn() * 0.01
            states.append(s)
        result = compute_phi(states)
        assert result.value > 0.0


# ======================================================================
# 2. Behavioral Diversity — Shannon Entropy / 行为多样性 — 香农熵
# ======================================================================

class TestBehavioralDiversity:

    def test_empty_actions(self):
        result = compute_behavioral_diversity([])
        assert result.entropy == 0.0
        assert result.unique_actions == 0
        assert result.total_actions == 0

    def test_single_action_zero_entropy(self):
        """Repeating one action → H = 0."""
        result = compute_behavioral_diversity([3] * 100)
        assert result.entropy == 0.0
        assert result.unique_actions == 1
        assert result.normalized == 0.0

    def test_single_element(self):
        result = compute_behavioral_diversity([7])
        assert result.entropy == 0.0
        assert result.unique_actions == 1
        assert result.total_actions == 1

    def test_two_actions_equal_frequency(self):
        """50/50 split → H = 1.0 bit."""
        actions = [0] * 50 + [1] * 50
        result = compute_behavioral_diversity(actions)
        assert abs(result.entropy - 1.0) < 0.001
        assert result.unique_actions == 2
        assert abs(result.normalized - 1.0) < 0.001

    def test_uniform_distribution_max_entropy(self):
        """Uniform across N → H = log₂(N)."""
        actions = list(range(8)) * 100
        result = compute_behavioral_diversity(actions)
        expected = math.log2(8)
        assert abs(result.entropy - expected) < 0.001
        assert result.unique_actions == 8
        assert abs(result.normalized - 1.0) < 0.001

    def test_skewed_lower_than_uniform(self):
        """Skewed distribution should have lower H than uniform."""
        uniform = compute_behavioral_diversity(list(range(4)) * 25)
        skewed = compute_behavioral_diversity([0] * 90 + [1] * 5 + [2] * 3 + [3] * 2)
        assert skewed.entropy < uniform.entropy

    def test_entropy_increases_with_more_actions(self):
        few = compute_behavioral_diversity([0] * 50 + [1] * 50)
        many = compute_behavioral_diversity(list(range(10)) * 10)
        assert many.entropy > few.entropy

    def test_normalized_between_0_and_1(self):
        for n in [2, 5, 10]:
            actions = list(range(n)) * 50
            result = compute_behavioral_diversity(actions)
            assert 0.0 <= result.normalized <= 1.0 + 1e-9

    def test_three_actions_known_entropy(self):
        """p = [0.5, 0.25, 0.25] → H = 1.5 bits."""
        actions = [0] * 200 + [1] * 100 + [2] * 100
        result = compute_behavioral_diversity(actions)
        assert abs(result.entropy - 1.5) < 0.001

    def test_total_actions_correct(self):
        result = compute_behavioral_diversity([1, 2, 3, 4, 5])
        assert result.total_actions == 5
        assert result.unique_actions == 5


# ======================================================================
# 3. Qualia-Behavior Correlation / 情绪-行为相关性
# ======================================================================

class TestQualiaBehaviorCorrelation:

    def test_insufficient_data(self):
        result = compute_qualia_behavior_correlation([0.1, 0.2], [1, 2])
        assert result.pearson_r == 0.0
        assert result.is_significant is False

    def test_length_mismatch(self):
        result = compute_qualia_behavior_correlation([0.1, 0.2, 0.3], [1, 2])
        assert result.pearson_r == 0.0

    def test_perfect_positive_correlation(self):
        """Qualia linearly predicts action → r ≈ 1."""
        qualia = list(np.linspace(-1, 1, 100))
        actions = list(range(100))
        result = compute_qualia_behavior_correlation(qualia, actions)
        assert result.pearson_r > 0.9

    def test_perfect_negative_correlation(self):
        """Qualia inversely predicts action → r ≈ -1."""
        qualia = list(np.linspace(-1, 1, 100))
        actions = list(range(99, -1, -1))
        result = compute_qualia_behavior_correlation(qualia, actions)
        assert result.pearson_r < -0.9

    def test_no_correlation_random(self):
        """Random qualia vs random actions → |r| ≈ 0."""
        np.random.seed(42)
        qualia = list(np.random.randn(500))
        actions = list(np.random.randint(0, 8, 500))
        result = compute_qualia_behavior_correlation(qualia, actions)
        assert abs(result.pearson_r) < 0.15

    def test_constant_qualia_zero_r(self):
        qualia = [0.5] * 100
        actions = list(range(100))
        result = compute_qualia_behavior_correlation(qualia, actions)
        assert result.pearson_r == 0.0

    def test_constant_action_zero_r(self):
        qualia = list(np.linspace(-1, 1, 100))
        actions = [3] * 100
        result = compute_qualia_behavior_correlation(qualia, actions)
        assert result.pearson_r == 0.0

    def test_mutual_info_non_negative(self):
        np.random.seed(42)
        qualia = list(np.random.randn(100))
        actions = list(np.random.randint(0, 5, 100))
        result = compute_qualia_behavior_correlation(qualia, actions)
        assert result.mutual_info >= 0.0

    def test_mutual_info_positive_for_dependent(self):
        """When qualia determine actions, MI > 0."""
        qualia = []
        actions = []
        for i in range(200):
            q = i / 200.0 - 0.5
            a = 0 if q < -0.2 else 1 if q < 0.0 else 2 if q < 0.2 else 3
            qualia.append(q)
            actions.append(a)
        result = compute_qualia_behavior_correlation(qualia, actions)
        assert result.mutual_info > 0.05

    def test_qualia_driven_behavior(self):
        """Negative qualia → emergency action; positive → idle."""
        np.random.seed(42)
        qualia = []
        actions = []
        for _ in range(300):
            q = np.random.randn() * 0.5
            if q < -0.3:
                a = 7
            elif q > 0.3:
                a = 0
            else:
                a = np.random.randint(1, 6)
            qualia.append(q)
            actions.append(a)
        result = compute_qualia_behavior_correlation(qualia, actions)
        assert abs(result.pearson_r) > 0.05 or result.mutual_info > 0.01

    def test_significance_requires_large_n(self):
        """is_significant needs n >= 30."""
        qualia = list(np.linspace(-1, 1, 20))
        actions = list(range(20))
        result = compute_qualia_behavior_correlation(qualia, actions)
        assert result.is_significant is False

    def test_significance_with_large_n(self):
        """Strong correlation + n >= 30 → significant."""
        qualia = list(np.linspace(-1, 1, 50))
        actions = list(range(50))
        result = compute_qualia_behavior_correlation(qualia, actions)
        assert result.is_significant is True

    def test_effect_size_distinct_groups(self):
        """Cohen's d should be large when groups have very different qualia."""
        np.random.seed(42)
        qualia = list(np.random.normal(1.0, 0.3, 50)) + list(np.random.normal(-1.0, 0.3, 50))
        actions = [0] * 50 + [1] * 50
        result = compute_qualia_behavior_correlation(qualia, actions)
        assert abs(result.effect_size) > 1.0

    def test_effect_size_same_qualia_across_actions(self):
        """Cohen's d ≈ 0 when all action groups have similar qualia."""
        np.random.seed(42)
        qualia = list(np.random.randn(200))
        actions = [i % 4 for i in range(200)]
        result = compute_qualia_behavior_correlation(qualia, actions)
        assert abs(result.effect_size) < 1.0

    def test_result_fields_types(self):
        np.random.seed(42)
        qualia = list(np.random.randn(50))
        actions = list(np.random.randint(0, 5, 50))
        result = compute_qualia_behavior_correlation(qualia, actions)
        assert isinstance(result.pearson_r, float)
        assert isinstance(result.effect_size, float)
        assert isinstance(result.mutual_info, float)
        assert isinstance(result.is_significant, bool)


# ======================================================================
# 4. Cross-metric consistency / 跨指标一致性
# ======================================================================

class TestCrossMetricConsistency:

    def test_all_metrics_handle_minimal_data(self):
        """All three metrics should return valid zero-results for trivial input."""
        states = [np.zeros(8)]
        actions = [0]
        qualia = [0.0]

        phi = compute_phi(states)
        div = compute_behavioral_diversity(actions)
        corr = compute_qualia_behavior_correlation(qualia, actions)

        assert phi.value == 0.0
        assert div.entropy == 0.0
        assert corr.pearson_r == 0.0

    def test_complex_scenario(self):
        """A realistic scenario: periodic signal driving qualia → actions."""
        np.random.seed(42)
        n = 500
        states: list[np.ndarray] = []
        actions: list[int] = []
        qualia: list[float] = []

        for t in range(n):
            base = np.sin(2 * np.pi * t / 100)
            s = np.full(16, base) + np.random.randn(16) * 0.1
            states.append(s)

            q = base + np.random.randn() * 0.1
            qualia.append(q)

            if q < -0.5:
                a = 7
            elif q > 0.5:
                a = 0
            else:
                a = np.random.randint(1, 7)
            actions.append(a)

        phi = compute_phi(states)
        div = compute_behavioral_diversity(actions)
        corr = compute_qualia_behavior_correlation(qualia, actions)

        assert math.isfinite(phi.value)
        assert phi.value > 0
        assert math.isfinite(div.entropy)
        assert div.entropy > 0
        assert div.unique_actions >= 2
        assert math.isfinite(corr.pearson_r)
        assert math.isfinite(corr.mutual_info)
        assert corr.mutual_info > 0

    def test_increasing_correlation_over_phases(self):
        """
        Simulates Phase I (no coupling) vs Phase II (coupling).
        Phi and correlation should both be higher in Phase II.
        """
        np.random.seed(42)

        # Phase I: random, no coupling
        phase1_states = [np.random.randn(8) for _ in range(200)]
        phase1_qualia = list(np.random.randn(200))
        phase1_actions = list(np.random.randint(0, 8, 200))

        phi1 = compute_phi(phase1_states)
        corr1 = compute_qualia_behavior_correlation(phase1_qualia, phase1_actions)

        # Phase II: correlated states, qualia-driven actions
        np.random.seed(99)
        phase2_states = []
        phase2_qualia = []
        phase2_actions = []
        for t in range(200):
            base = np.sin(2 * np.pi * t / 50)
            s = np.full(8, base) + np.random.randn(8) * 0.05
            phase2_states.append(s)

            q = base + np.random.randn() * 0.05
            phase2_qualia.append(q)
            phase2_actions.append(0 if q > 0 else 7)

        phi2 = compute_phi(phase2_states)
        corr2 = compute_qualia_behavior_correlation(phase2_qualia, phase2_actions)

        assert phi2.value > phi1.value
        assert abs(corr2.pearson_r) > abs(corr1.pearson_r)
