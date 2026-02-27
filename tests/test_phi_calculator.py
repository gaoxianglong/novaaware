"""
Unit tests for PhiCalculator — the "Consciousness Thermometer".
Phi 计算器单元测试 —— "意识温度计"。

Tests cover:
  - Total correlation computation
  - Minimum Information Partition (MIP) — exhaustive and spectral
  - Stateful tracking (update / history)
  - Trend analysis and is_rising check (Scorecard #8)
  - Edge cases: insufficient data, constant dimensions, identity cov
  - Consistency between TC and MIP

Corresponds to CHECKLIST 2.29.
"""

import math

import numpy as np
import pytest

from novaaware.observation.phi_calculator import (
    PhiCalculator,
    PhiSnapshot,
    PhiTrend,
    _total_correlation,
    _partition_phi,
    _exhaustive_mip,
    _spectral_mip,
    _linear_regression,
)


def _make_states(n: int, k: int, seed: int = 42,
                 correlated: bool = False) -> list[np.ndarray]:
    """Generate synthetic state vectors."""
    rng = np.random.RandomState(seed)
    if correlated:
        base = rng.randn(n, 1)
        noise = rng.randn(n, k) * 0.3
        states = base + noise
    else:
        states = rng.randn(n, k)
    return [states[i] for i in range(n)]


def _make_block_states(n: int, k1: int, k2: int,
                       seed: int = 42) -> list[np.ndarray]:
    """
    Generate states with two independent blocks of correlated dimensions.
    This creates a natural bipartition — the MIP should split along the
    block boundary.
    """
    rng = np.random.RandomState(seed)
    base1 = rng.randn(n, 1)
    base2 = rng.randn(n, 1)
    block1 = base1 + rng.randn(n, k1) * 0.1
    block2 = base2 + rng.randn(n, k2) * 0.1
    states = np.hstack([block1, block2])
    return [states[i] for i in range(n)]


# ======================================================================
# 1. Total Correlation / 总相关量
# ======================================================================

class TestTotalCorrelation:

    def test_identity_covariance_zero(self):
        """Independent dimensions → TC = 0."""
        cov = np.eye(5)
        assert abs(_total_correlation(cov)) < 1e-6

    def test_perfectly_correlated_high_tc(self):
        """All dimensions identical → high TC."""
        states = _make_states(100, 4, correlated=True)
        data = np.array(states)
        cov = np.cov(data.T) + np.eye(4) * 1e-10
        tc = _total_correlation(cov)
        assert tc > 0.5

    def test_single_dim_returns_zero(self):
        cov = np.array([[1.0]])
        assert _total_correlation(cov) == 0.0

    def test_non_negative(self):
        """TC should always be >= 0."""
        for seed in range(10):
            states = _make_states(50, 6, seed=seed)
            data = np.array(states)
            cov = np.cov(data.T) + np.eye(6) * 1e-10
            assert _total_correlation(cov) >= -1e-10


# ======================================================================
# 2. Partition Phi / 分割 Phi
# ======================================================================

class TestPartitionPhi:

    def test_independent_blocks_low_partition_phi(self):
        """Two independent blocks → cutting between them loses little."""
        rng = np.random.RandomState(42)
        n = 200
        block1 = rng.randn(n, 3)
        block2 = rng.randn(n, 3)
        data = np.hstack([block1, block2])
        cov = np.cov(data.T) + np.eye(6) * 1e-10
        phi = _partition_phi(cov, [0, 1, 2], [3, 4, 5])
        assert phi < 0.5

    def test_correlated_blocks_high_partition_phi(self):
        """All dimensions correlated → any cut loses significant info."""
        states = _make_states(200, 6, correlated=True)
        data = np.array(states)
        cov = np.cov(data.T) + np.eye(6) * 1e-10
        phi = _partition_phi(cov, [0, 1, 2], [3, 4, 5])
        assert phi > 0.1

    def test_non_negative(self):
        rng = np.random.RandomState(42)
        data = rng.randn(100, 4)
        cov = np.cov(data.T) + np.eye(4) * 1e-10
        phi = _partition_phi(cov, [0, 1], [2, 3])
        assert phi >= 0.0

    def test_singleton_partition(self):
        """One side has 1 dim → TC(A) = 0, so Phi = TC(whole) - TC(B)."""
        states = _make_states(100, 4, correlated=True)
        data = np.array(states)
        cov = np.cov(data.T) + np.eye(4) * 1e-10
        phi = _partition_phi(cov, [0], [1, 2, 3])
        assert phi >= 0.0


# ======================================================================
# 3. Exhaustive MIP / 穷举 MIP
# ======================================================================

class TestExhaustiveMIP:

    def test_two_dim_single_partition(self):
        """With 2 dims there's only one bipartition: ({0}, {1})."""
        rng = np.random.RandomState(42)
        data = rng.randn(100, 2)
        cov = np.cov(data.T) + np.eye(2) * 1e-10
        phi, part = _exhaustive_mip(cov, [0, 1])
        assert phi >= 0.0
        assert set(part[0] + part[1]) == {0, 1}

    def test_block_structure_found(self):
        """MIP should discover the natural block boundary."""
        states = _make_block_states(300, 3, 3, seed=42)
        data = np.array(states)
        cov = np.cov(data.T) + np.eye(6) * 1e-10
        phi, part = _exhaustive_mip(cov, list(range(6)))

        set_a = set(part[0])
        set_b = set(part[1])
        block1 = {0, 1, 2}
        block2 = {3, 4, 5}
        correct = (set_a == block1 and set_b == block2) or \
                  (set_a == block2 and set_b == block1)
        assert correct, f"Expected blocks {{0,1,2}} / {{3,4,5}}, got {set_a} / {set_b}"

    def test_mip_phi_leq_total_correlation(self):
        """MIP-Phi should be <= TC (it's the minimum over partitions)."""
        states = _make_states(100, 5, correlated=True)
        data = np.array(states)
        cov = np.cov(data.T) + np.eye(5) * 1e-10
        tc = _total_correlation(cov)
        phi, _ = _exhaustive_mip(cov, list(range(5)))
        assert phi <= tc + 1e-6

    def test_independent_dims_low_mip_phi(self):
        """Independent dimensions → MIP-Phi ≈ 0."""
        cov = np.eye(4) + np.eye(4) * 1e-10
        phi, _ = _exhaustive_mip(cov, [0, 1, 2, 3])
        assert phi < 0.01


# ======================================================================
# 4. Spectral MIP (greedy heuristic) / 谱 MIP
# ======================================================================

class TestSpectralMIP:

    def test_returns_valid_partition(self):
        states = _make_states(100, 15, seed=42)
        data = np.array(states)
        cov = np.cov(data.T) + np.eye(15) * 1e-10
        phi, part = _spectral_mip(cov, list(range(15)))
        assert phi >= 0.0
        all_indices = set(part[0] + part[1])
        assert all_indices == set(range(15))

    def test_block_structure_approximated(self):
        """Spectral method should roughly find block boundaries."""
        states = _make_block_states(500, 7, 7, seed=42)
        data = np.array(states)
        cov = np.cov(data.T) + np.eye(14) * 1e-10
        phi, part = _spectral_mip(cov, list(range(14)))

        set_a = set(part[0])
        set_b = set(part[1])
        block1 = set(range(7))
        block2 = set(range(7, 14))
        overlap_a1 = len(set_a & block1) / max(len(set_a), 1)
        overlap_b2 = len(set_b & block2) / max(len(set_b), 1)
        overlap_a2 = len(set_a & block2) / max(len(set_a), 1)
        overlap_b1 = len(set_b & block1) / max(len(set_b), 1)
        quality = max(overlap_a1 + overlap_b2, overlap_a2 + overlap_b1) / 2
        assert quality > 0.6, f"Spectral partition quality too low: {quality}"


# ======================================================================
# 5. PhiCalculator — compute / PhiCalculator — 计算
# ======================================================================

class TestPhiCalculatorCompute:

    def test_insufficient_data_returns_zeros(self):
        calc = PhiCalculator(min_samples=10)
        states = _make_states(5, 4)
        snap = calc.compute(states, tick=0)
        assert snap.total_correlation == 0.0
        assert snap.mip_phi == 0.0
        assert snap.mip_partition is None

    def test_single_dimension_returns_zeros(self):
        calc = PhiCalculator(min_samples=5)
        states = [np.array([float(i)]) for i in range(20)]
        snap = calc.compute(states, tick=0)
        assert snap.total_correlation == 0.0
        assert snap.mip_phi == 0.0

    def test_constant_dimensions_filtered(self):
        """Dimensions with zero variance should be filtered out."""
        calc = PhiCalculator(min_samples=5)
        rng = np.random.RandomState(42)
        states = []
        for i in range(50):
            v = rng.randn(4)
            v[2] = 1.0  # constant
            states.append(v)
        snap = calc.compute(states, tick=0)
        assert snap.active_dimensions == 3
        assert snap.num_dimensions == 4

    def test_correlated_data_positive_phi(self):
        calc = PhiCalculator()
        states = _make_states(100, 5, correlated=True)
        snap = calc.compute(states, tick=100)
        assert snap.total_correlation > 0.1
        assert snap.mip_phi > 0.0
        assert snap.tick == 100

    def test_independent_data_low_phi(self):
        calc = PhiCalculator()
        states = _make_states(200, 5, correlated=False)
        snap = calc.compute(states, tick=0)
        assert snap.mip_phi < 0.5

    def test_snapshot_fields_complete(self):
        calc = PhiCalculator()
        states = _make_states(50, 4)
        snap = calc.compute(states, tick=42)
        assert isinstance(snap.tick, int)
        assert isinstance(snap.total_correlation, float)
        assert isinstance(snap.mip_phi, float)
        assert isinstance(snap.num_samples, int)
        assert isinstance(snap.num_dimensions, int)
        assert isinstance(snap.active_dimensions, int)

    def test_uses_spectral_for_many_dims(self):
        """With > max_partition_dims active dims, spectral method is used."""
        calc = PhiCalculator(max_partition_dims=5)
        states = _make_states(100, 8, seed=42, correlated=True)
        snap = calc.compute(states, tick=0)
        assert snap.mip_phi >= 0.0
        assert snap.active_dimensions == 8


# ======================================================================
# 6. Stateful tracking / 有状态跟踪
# ======================================================================

class TestStatefulTracking:

    def test_update_records_to_history(self):
        calc = PhiCalculator()
        states = _make_states(50, 4)
        calc.update(states, tick=10)
        calc.update(states, tick=20)
        assert len(calc.history) == 2
        assert calc.history[0].tick == 10
        assert calc.history[1].tick == 20

    def test_latest_property(self):
        calc = PhiCalculator()
        assert calc.latest is None
        states = _make_states(50, 4)
        calc.update(states, tick=5)
        assert calc.latest is not None
        assert calc.latest.tick == 5

    def test_clear_empties_history(self):
        calc = PhiCalculator()
        states = _make_states(50, 4)
        calc.update(states, tick=1)
        calc.update(states, tick=2)
        calc.clear()
        assert len(calc.history) == 0
        assert calc.latest is None


# ======================================================================
# 7. Trend analysis (Scorecard #8) / 趋势分析
# ======================================================================

class TestTrendAnalysis:

    def test_rising_phi_detected(self):
        """Phi increasing over time → is_rising = True."""
        calc = PhiCalculator(rising_threshold=0.001)
        for i in range(10):
            corr_strength = 0.1 + i * 0.08
            rng = np.random.RandomState(i)
            base = rng.randn(100, 1)
            noise = rng.randn(100, 4) * max(0.01, 1.0 - corr_strength)
            states_arr = base * corr_strength + noise
            states = [states_arr[j] for j in range(100)]
            calc.update(states, tick=i * 100)

        t = calc.trend(min_measurements=5)
        assert t.slope > 0
        assert t.is_rising
        assert t.last_phi > t.first_phi

    def test_constant_phi_not_rising(self):
        """Same data every time → slope ≈ 0 → not rising."""
        calc = PhiCalculator(rising_threshold=0.001)
        states = _make_states(50, 4, seed=42, correlated=True)
        for i in range(10):
            calc.update(states, tick=i * 100)
        t = calc.trend()
        assert abs(t.slope) < 0.001
        assert not t.is_rising

    def test_insufficient_measurements_not_rising(self):
        calc = PhiCalculator()
        states = _make_states(50, 4)
        calc.update(states, tick=0)
        assert not calc.is_rising(min_measurements=5)

    def test_trend_fields_complete(self):
        calc = PhiCalculator()
        states = _make_states(50, 4)
        for i in range(5):
            calc.update(states, tick=i)
        t = calc.trend()
        assert isinstance(t.slope, float)
        assert isinstance(t.r_squared, float)
        assert isinstance(t.is_rising, bool)
        assert isinstance(t.num_measurements, int)
        assert isinstance(t.first_phi, float)
        assert isinstance(t.last_phi, float)
        assert isinstance(t.mean_phi, float)
        assert 0.0 <= t.r_squared <= 1.0

    def test_is_rising_shortcut(self):
        calc = PhiCalculator(rising_threshold=0.001)
        for i in range(10):
            corr_strength = 0.1 + i * 0.08
            rng = np.random.RandomState(i)
            base = rng.randn(100, 1)
            noise = rng.randn(100, 4) * max(0.01, 1.0 - corr_strength)
            states_arr = base * corr_strength + noise
            states = [states_arr[j] for j in range(100)]
            calc.update(states, tick=i * 100)
        assert calc.is_rising(min_measurements=5)


# ======================================================================
# 8. Linear regression helper / 线性回归辅助
# ======================================================================

class TestLinearRegression:

    def test_perfect_line(self):
        x = np.arange(10, dtype=np.float64)
        y = 2.0 * x + 3.0
        slope, intercept = _linear_regression(x, y)
        assert abs(slope - 2.0) < 1e-8
        assert abs(intercept - 3.0) < 1e-8

    def test_constant_y(self):
        x = np.arange(10, dtype=np.float64)
        y = np.ones(10) * 5.0
        slope, intercept = _linear_regression(x, y)
        assert abs(slope) < 1e-8
        assert abs(intercept - 5.0) < 1e-8

    def test_single_point(self):
        x = np.array([1.0])
        y = np.array([3.0])
        slope, intercept = _linear_regression(x, y)
        assert abs(slope) < 1e-8


# ======================================================================
# 9. Consistency: TC vs MIP / 一致性：TC vs MIP
# ======================================================================

class TestConsistency:

    def test_mip_leq_tc(self):
        """MIP-Phi is always <= total correlation."""
        for seed in range(5):
            calc = PhiCalculator()
            states = _make_states(100, 5, seed=seed, correlated=True)
            snap = calc.compute(states, tick=0)
            assert snap.mip_phi <= snap.total_correlation + 1e-6

    def test_higher_correlation_higher_phi(self):
        """More correlated data → higher Phi."""
        calc = PhiCalculator()
        states_low = _make_states(200, 4, seed=42, correlated=False)
        states_high = _make_states(200, 4, seed=42, correlated=True)
        snap_low = calc.compute(states_low)
        snap_high = calc.compute(states_high)
        assert snap_high.mip_phi > snap_low.mip_phi
        assert snap_high.total_correlation > snap_low.total_correlation
