"""
PhiCalculator — the "Consciousness Thermometer".
Phi 计算器 —— "意识温度计"。

Implements a proper approximation of Integrated Information Theory (IIT)
Phi (Φ) with two complementary measures:
实现整合信息理论（IIT）Φ 值的正规近似，包含两个互补度量：

    1. Total Correlation (TC) — fast upper bound on Phi.
       总相关量 — Phi 的快速上界。
           TC = 0.5 × ( Σᵢ log σᵢ² − log det Σ )

    2. Minimum Information Partition (MIP) approximation — the true
       IIT notion: find the bipartition (A, B) of system dimensions
       that minimises the information lost when the partition is cut.
       最小信息分割（MIP）近似 — 真正的 IIT 概念：找到将系统维度
       二分为 (A, B) 后信息丢失最少的分割方式。

           Φ_MIP ≈ min over all bipartitions { TC(whole) − TC(A) − TC(B) }

       Since enumerating all 2^(k-1)−1 bipartitions is exponential,
       we use a greedy spectral heuristic for k > 12.
       由于枚举所有分割是指数级的，当 k > 12 时使用贪心谱方法。

Additionally provides *stateful* tracking of Phi over time to support
Scorecard #8: "Information integration (Phi) is steadily rising".
还提供 Phi 随时间的 *有状态* 跟踪，支持记分卡第 8 项检查。

Paper: §6.3 "information-theoretic validation (Φ measurement)"
论文：§6.3 "信息论验证（Φ 值测量）"

IMPLEMENTATION_PLAN: "Consciousness Thermometer", §5.4 Scorecard #8.
对应实施计划"意识温度计"，§5.4 记分卡第 8 项。

Corresponds to CHECKLIST 2.29.
"""

import math
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np


# ======================================================================
# Result data structures / 结果数据结构
# ======================================================================

@dataclass
class PhiSnapshot:
    """
    A single Phi measurement at a point in time.
    某一时刻的 Phi 测量值。
    """
    tick: int
    total_correlation: float
    mip_phi: float
    mip_partition: tuple[tuple[int, ...], tuple[int, ...]] | None
    num_samples: int
    num_dimensions: int
    active_dimensions: int


@dataclass
class PhiTrend:
    """
    Trend analysis over a series of Phi measurements.
    一系列 Phi 测量值的趋势分析。
    """
    slope: float
    r_squared: float
    is_rising: bool
    num_measurements: int
    first_phi: float
    last_phi: float
    mean_phi: float


# ======================================================================
# PhiCalculator — stateful Phi tracker
# Phi 计算器 — 有状态的 Phi 跟踪器
# ======================================================================

class PhiCalculator:
    """
    Stateful calculator that tracks information integration over time.
    有状态的计算器，跟踪信息整合度随时间的变化。

    Parameters / 参数
    ----------
    min_samples : int
        Minimum state vectors needed for a reliable Φ estimate.
        可靠 Φ 估计所需的最少状态向量数。
    max_partition_dims : int
        Maximum active dimensions for exhaustive MIP search.
        Beyond this, use spectral greedy approximation.
        详尽 MIP 搜索的最大活跃维度数，超过则用谱贪心近似。
    rising_threshold : float
        Minimum slope (per measurement) to consider Phi "rising".
        认为 Phi "上升" 的最小斜率（每次测量）。
    """

    def __init__(
        self,
        min_samples: int = 10,
        max_partition_dims: int = 12,
        rising_threshold: float = 0.001,
    ):
        self._min_samples = min_samples
        self._max_partition_dims = max_partition_dims
        self._rising_threshold = rising_threshold
        self._history: list[PhiSnapshot] = []

    # ------------------------------------------------------------------
    # Core computation / 核心计算
    # ------------------------------------------------------------------

    def compute(
        self,
        state_history: list[np.ndarray],
        tick: int = 0,
    ) -> PhiSnapshot:
        """
        Compute Phi (both TC and MIP approximation) from state history.
        从状态历史计算 Phi（总相关量和 MIP 近似）。

        Parameters / 参数
        ----------
        state_history : list[np.ndarray]
            Recent state vectors, each of shape (k,).
        tick : int
            Current tick number for recording.

        Returns / 返回
        -------
        PhiSnapshot
        """
        k = state_history[0].shape[0] if state_history else 0
        n = len(state_history)

        empty = PhiSnapshot(
            tick=tick, total_correlation=0.0, mip_phi=0.0,
            mip_partition=None, num_samples=n,
            num_dimensions=k, active_dimensions=0,
        )

        if n < self._min_samples or k < 2:
            return empty

        states = np.array(state_history)  # (n, k)

        variances = np.var(states, axis=0)
        active_mask = variances > 1e-12
        k_active = int(np.sum(active_mask))

        if k_active < 2:
            return empty

        active_states = states[:, active_mask]
        active_indices = list(np.where(active_mask)[0])

        cov = np.cov(active_states.T)
        if cov.ndim == 0:
            return empty

        cov += np.eye(k_active) * 1e-10

        tc = _total_correlation(cov)

        if k_active <= self._max_partition_dims:
            mip_phi, mip_part = _exhaustive_mip(cov, active_indices)
        else:
            mip_phi, mip_part = _spectral_mip(cov, active_indices)

        return PhiSnapshot(
            tick=tick,
            total_correlation=round(tc, 6),
            mip_phi=round(mip_phi, 6),
            mip_partition=mip_part,
            num_samples=n,
            num_dimensions=k,
            active_dimensions=k_active,
        )

    # ------------------------------------------------------------------
    # Stateful tracking / 有状态跟踪
    # ------------------------------------------------------------------

    def update(
        self,
        state_history: list[np.ndarray],
        tick: int,
    ) -> PhiSnapshot:
        """
        Compute Phi and record the measurement in internal history.
        计算 Phi 并将测量值记录到内部历史。
        """
        snapshot = self.compute(state_history, tick)
        self._history.append(snapshot)
        return snapshot

    def trend(self, min_measurements: int = 3) -> PhiTrend:
        """
        Analyse the trend of Phi (MIP) over recorded measurements.
        分析记录的 Phi (MIP) 测量值的趋势。

        Uses ordinary least-squares linear regression on the MIP-Phi
        series to compute slope and R².
        对 MIP-Phi 序列使用最小二乘线性回归计算斜率和 R²。

        Parameters / 参数
        ----------
        min_measurements : int
            Minimum measurements required for trend analysis.

        Returns / 返回
        -------
        PhiTrend
        """
        phis = [s.mip_phi for s in self._history]
        n = len(phis)

        if n < min_measurements:
            mean_phi = sum(phis) / n if n > 0 else 0.0
            return PhiTrend(
                slope=0.0, r_squared=0.0, is_rising=False,
                num_measurements=n,
                first_phi=phis[0] if phis else 0.0,
                last_phi=phis[-1] if phis else 0.0,
                mean_phi=mean_phi,
            )

        x = np.arange(n, dtype=np.float64)
        y = np.array(phis, dtype=np.float64)

        slope, intercept = _linear_regression(x, y)

        y_pred = slope * x + intercept
        ss_res = float(np.sum((y - y_pred) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
        r_squared = max(0.0, r_squared)

        return PhiTrend(
            slope=round(slope, 8),
            r_squared=round(r_squared, 6),
            is_rising=slope > self._rising_threshold,
            num_measurements=n,
            first_phi=phis[0],
            last_phi=phis[-1],
            mean_phi=round(float(np.mean(y)), 6),
        )

    def is_rising(self, min_measurements: int = 5) -> bool:
        """
        Scorecard #8: Is information integration steadily rising?
        记分卡第 8 项：信息整合度是否在稳步上升？
        """
        t = self.trend(min_measurements)
        return t.is_rising

    # ------------------------------------------------------------------
    # Properties / 属性
    # ------------------------------------------------------------------

    @property
    def history(self) -> list[PhiSnapshot]:
        """All recorded Phi measurements. / 所有记录的 Phi 测量值。"""
        return list(self._history)

    @property
    def latest(self) -> PhiSnapshot | None:
        """Most recent measurement. / 最近的测量值。"""
        return self._history[-1] if self._history else None

    def clear(self) -> None:
        """Clear measurement history. / 清除测量历史。"""
        self._history.clear()


# ======================================================================
# Core Phi algorithms / 核心 Phi 算法
# ======================================================================

def _total_correlation(cov: np.ndarray) -> float:
    """
    Total correlation (multi-information) under Gaussian assumption.
    高斯假设下的总相关量（多信息）。

        TC = 0.5 × ( Σᵢ log σᵢ² − log det Σ )

    Returns 0.0 if the covariance matrix is degenerate.
    """
    k = cov.shape[0]
    if k < 2:
        return 0.0

    diag_log_sum = float(np.sum(np.log(np.diag(cov))))
    sign, log_det = np.linalg.slogdet(cov)

    if sign <= 0:
        return 0.0

    tc = 0.5 * (diag_log_sum - float(log_det))
    return max(0.0, tc)


def _partition_phi(
    cov: np.ndarray,
    part_a: list[int],
    part_b: list[int],
) -> float:
    """
    Information lost when cutting the system into partitions A and B.
    将系统分割为 A 和 B 时丢失的信息。

        Φ_partition = TC(whole) − TC(A) − TC(B)

    This measures how much information integration is lost when the
    two partitions are disconnected.
    """
    tc_whole = _total_correlation(cov)

    if len(part_a) < 2:
        tc_a = 0.0
    else:
        tc_a = _total_correlation(cov[np.ix_(part_a, part_a)])

    if len(part_b) < 2:
        tc_b = 0.0
    else:
        tc_b = _total_correlation(cov[np.ix_(part_b, part_b)])

    return max(0.0, tc_whole - tc_a - tc_b)


def _exhaustive_mip(
    cov: np.ndarray,
    original_indices: list[int],
) -> tuple[float, tuple[tuple[int, ...], tuple[int, ...]]]:
    """
    Find the Minimum Information Partition by exhaustive search.
    通过穷举搜索找到最小信息分割。

    Enumerates all possible bipartitions of the k dimensions and
    returns the one with the *minimum* Φ_partition — this is the
    "weakest link" and represents true integrated information.
    """
    k = cov.shape[0]
    all_dims = list(range(k))

    min_phi = float("inf")
    min_part: tuple[tuple[int, ...], tuple[int, ...]] = ((), ())

    for r in range(1, k):
        for combo_a in combinations(all_dims, r):
            part_a = list(combo_a)
            part_b = [d for d in all_dims if d not in combo_a]
            if not part_b:
                continue

            phi = _partition_phi(cov, part_a, part_b)

            if phi < min_phi:
                min_phi = phi
                orig_a = tuple(original_indices[i] for i in part_a)
                orig_b = tuple(original_indices[i] for i in part_b)
                min_part = (orig_a, orig_b)

    if min_phi == float("inf"):
        min_phi = 0.0

    return min_phi, min_part


def _spectral_mip(
    cov: np.ndarray,
    original_indices: list[int],
) -> tuple[float, tuple[tuple[int, ...], tuple[int, ...]]]:
    """
    Approximate MIP using spectral bipartitioning (greedy heuristic).
    使用谱二分法（贪心启发式）近似 MIP。

    Strategy: compute the eigenvector corresponding to the second-
    smallest eigenvalue of the Laplacian of the absolute correlation
    matrix, then split dimensions by sign.  This tends to find weakly
    coupled subgroups.
    策略：计算绝对相关矩阵的拉普拉斯矩阵第二小特征值对应的特征向量，
    然后按符号分割维度。这倾向于找到弱耦合子组。
    """
    k = cov.shape[0]

    diag = np.diag(cov)
    std = np.sqrt(np.maximum(diag, 1e-12))
    corr = cov / np.outer(std, std)
    np.fill_diagonal(corr, 1.0)
    abs_corr = np.abs(corr)

    degree = np.sum(abs_corr, axis=1)
    laplacian = np.diag(degree) - abs_corr

    eigvals, eigvecs = np.linalg.eigh(laplacian)

    fiedler = eigvecs[:, 1]

    part_a_local = [i for i in range(k) if fiedler[i] >= 0]
    part_b_local = [i for i in range(k) if fiedler[i] < 0]

    if not part_a_local or not part_b_local:
        half = k // 2
        part_a_local = list(range(half))
        part_b_local = list(range(half, k))

    phi = _partition_phi(cov, part_a_local, part_b_local)

    orig_a = tuple(original_indices[i] for i in part_a_local)
    orig_b = tuple(original_indices[i] for i in part_b_local)

    return phi, (orig_a, orig_b)


def _linear_regression(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[float, float]:
    """Simple OLS: y = slope * x + intercept."""
    n = len(x)
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    ss_xy = float(np.sum((x - x_mean) * (y - y_mean)))
    ss_xx = float(np.sum((x - x_mean) ** 2))
    if ss_xx < 1e-12:
        return 0.0, float(y_mean)
    slope = ss_xy / ss_xx
    intercept = float(y_mean) - slope * float(x_mean)
    return float(slope), float(intercept)
