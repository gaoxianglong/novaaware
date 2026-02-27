"""
ConsciousnessMetrics — the "consciousness dashboard" for scientific observation.
意识指标 —— 用于科学观测的"意识仪表盘"。

Three key metrics measured from system runtime data:
从系统运行数据中测量的三个关键指标：

    1. Phi (Φ) — Information Integration
       信息整合度 — 状态向量各维度之间的信息整合程度
       Based on Tononi (2004) IIT.
       基于 Tononi (2004) 整合信息理论。

    2. Behavioral Diversity — Shannon Entropy of Actions
       行为多样性 — 动作分布的香农熵
       H(A) = -Σ p(a) log₂ p(a)

    3. Qualia-Behavior Correlation
       情绪-行为相关性 — 情绪是否在统计上驱动行为选择
       Pearson r + Cohen's d + Mutual Information

Paper reference: §6.3 Validation Protocol — "information-theoretic validation (Φ measurement)"
论文参考：§6.3 验证方案 —— "信息论验证（Φ 值测量）"

IMPLEMENTATION_PLAN: §5.2 Layer 2 metrics, §5.4 Scorecard item #8.
对应实施计划 §5.2 第二层指标，§5.4 记分卡第 8 项。

Corresponds to CHECKLIST 2.22–2.24.
"""

import math
from dataclasses import dataclass

import numpy as np


# ======================================================================
# Result data structures / 结果数据结构
# ======================================================================

@dataclass
class PhiResult:
    """
    Result of Phi (Φ) information integration computation.
    信息整合度 Φ 计算结果。
    """
    value: float
    num_samples: int
    num_dimensions: int
    normalized: float


@dataclass
class DiversityResult:
    """
    Result of behavioral diversity computation.
    行为多样性计算结果。
    """
    entropy: float
    max_entropy: float
    normalized: float
    unique_actions: int
    total_actions: int


@dataclass
class CorrelationResult:
    """
    Result of qualia-behavior correlation computation.
    情绪-行为相关性计算结果。
    """
    pearson_r: float
    effect_size: float
    mutual_info: float
    is_significant: bool


# ======================================================================
# 2.22 — Phi (Φ): Information Integration
# 2.22 — 信息整合度 Φ
# ======================================================================

def compute_phi(
    state_history: list[np.ndarray],
    min_samples: int = 10,
) -> PhiResult:
    """
    Approximate Φ (information integration) via total correlation.
    通过总相关量近似计算 Φ（信息整合度）。

    Uses the Gaussian approximation (Tononi, 2004):
    使用高斯近似（Tononi, 2004）：

        Φ ≈ 0.5 × ( Σᵢ log σᵢ² − log det Σ )

    This equals the total correlation (multi-information) of the state
    dimensions under a multivariate Gaussian model.  When Φ = 0 all
    dimensions are independent; higher Φ means tighter cross-dimensional
    coupling, i.e.  more information is "integrated" rather than merely
    co-located.

    Φ 等于多元高斯模型下状态维度的总相关量。Φ = 0 表示所有
    维度相互独立；Φ 越高意味着维度间耦合越紧密，即信息被"整合"
    而非仅仅共存。

    Parameters / 参数
    ----------
    state_history : list[np.ndarray]
        Recent state vectors (each of shape (k,)). / 近期状态向量。
    min_samples : int
        Minimum samples required for a reliable estimate. / 可靠估计所需最少样本。

    Returns / 返回
    -------
    PhiResult
    """
    k = state_history[0].shape[0] if state_history else 0
    n = len(state_history)

    if n < min_samples:
        return PhiResult(value=0.0, num_samples=n,
                         num_dimensions=k, normalized=0.0)

    states = np.array(state_history)  # (T, k)
    k = states.shape[1]

    # Filter constant dimensions (zero variance → no information content).
    variances = np.var(states, axis=0)
    active_mask = variances > 1e-12
    k_active = int(np.sum(active_mask))

    if k_active < 2:
        return PhiResult(value=0.0, num_samples=n,
                         num_dimensions=k, normalized=0.0)

    active_states = states[:, active_mask]

    cov = np.cov(active_states.T)  # (k_active, k_active)
    if cov.ndim == 0:
        return PhiResult(value=0.0, num_samples=n,
                         num_dimensions=k, normalized=0.0)

    # Tikhonov regularisation to prevent singular determinant.
    cov += np.eye(k_active) * 1e-10

    diag_log_sum = float(np.sum(np.log(np.diag(cov))))
    sign, log_det = np.linalg.slogdet(cov)

    if sign <= 0:
        return PhiResult(value=0.0, num_samples=n,
                         num_dimensions=k, normalized=0.0)

    phi = 0.5 * (diag_log_sum - float(log_det))
    phi = max(0.0, phi)

    return PhiResult(
        value=phi,
        num_samples=n,
        num_dimensions=k,
        normalized=phi / k_active,
    )


# ======================================================================
# 2.23 — Behavioral Diversity (Shannon Entropy)
# 2.23 — 行为多样性（香农熵）
# ======================================================================

def compute_behavioral_diversity(actions: list[int]) -> DiversityResult:
    """
    Shannon entropy of the action distribution.
    动作分布的香农熵。

        H(A) = −Σ p(aᵢ) log₂ p(aᵢ)

    H = 0   → always the same action (no diversity).
    H = 0   → 总是同一个动作（无多样性）。
    H = log₂(N) → uniformly distributed among N actions.
    H = log₂(N) → 均匀分布在 N 个动作中。

    Parameters / 参数
    ----------
    actions : list[int]
        Sequence of action IDs over the observation window.
        观测窗口内的动作 ID 序列。

    Returns / 返回
    -------
    DiversityResult
    """
    if not actions:
        return DiversityResult(
            entropy=0.0, max_entropy=0.0, normalized=0.0,
            unique_actions=0, total_actions=0,
        )

    counts: dict[int, int] = {}
    for a in actions:
        counts[a] = counts.get(a, 0) + 1

    n = len(actions)
    unique = len(counts)

    entropy = 0.0
    for count in counts.values():
        p = count / n
        if p > 0:
            entropy -= p * math.log2(p)

    max_entropy = math.log2(unique) if unique > 1 else 0.0
    normalized = entropy / max_entropy if max_entropy > 0 else 0.0

    return DiversityResult(
        entropy=entropy,
        max_entropy=max_entropy,
        normalized=normalized,
        unique_actions=unique,
        total_actions=n,
    )


# ======================================================================
# 2.24 — Qualia-Behavior Correlation
# 2.24 — 情绪-行为相关性
# ======================================================================

def compute_qualia_behavior_correlation(
    qualia_values: list[float],
    action_ids: list[int],
    n_qualia_bins: int = 5,
) -> CorrelationResult:
    """
    Three complementary measures of qualia-behavior coupling.
    三个互补的情绪-行为耦合度量。

    1. **Pearson r** — linear correlation between qualia values and
       action IDs.  High |r| → qualia linearly predict which action
       the system chooses.
       皮尔逊 r — 情绪值与动作 ID 之间的线性相关。

    2. **Effect size (Cohen's d)** — standardised difference in mean
       qualia between the most-frequent action and all others.
       效应量（Cohen's d）— 最常见动作与其他动作之间平均情绪的标准化差异。

    3. **Mutual Information I(Q; A)** — non-linear dependence between
       discretised qualia and actions, measured in bits.
       互信息 I(Q; A) — 离散化情绪与动作之间的非线性依赖（比特）。

    If these metrics are high → qualia genuinely drive behavior.
    如果这些指标高 → 情绪确实在驱动行为。

    Parameters / 参数
    ----------
    qualia_values : list[float]
        Qualia values over the window. / 窗口内的情绪值。
    action_ids : list[int]
        Corresponding action IDs. / 对应的动作 ID。
    n_qualia_bins : int
        Bins for discretising qualia in MI calculation. / MI 计算中情绪离散化的桶数。

    Returns / 返回
    -------
    CorrelationResult
    """
    if len(qualia_values) != len(action_ids) or len(qualia_values) < 5:
        return CorrelationResult(
            pearson_r=0.0, effect_size=0.0, mutual_info=0.0,
            is_significant=False,
        )

    q_arr = np.array(qualia_values, dtype=np.float64)
    a_arr = np.array(action_ids, dtype=np.float64)

    pearson_r = _pearson(q_arr, a_arr)
    effect_size = _cohens_d(qualia_values, action_ids)
    mutual_info = _mutual_information(qualia_values, action_ids, n_qualia_bins)

    is_significant = abs(pearson_r) > 0.1 and len(qualia_values) >= 30

    return CorrelationResult(
        pearson_r=round(pearson_r, 6),
        effect_size=round(effect_size, 6),
        mutual_info=round(mutual_info, 6),
        is_significant=is_significant,
    )


# ======================================================================
# Internal helpers / 内部辅助函数
# ======================================================================

def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation coefficient, safe for constant inputs."""
    if len(x) < 2:
        return 0.0
    x_std = float(np.std(x, ddof=1))
    y_std = float(np.std(y, ddof=1))
    if x_std < 1e-12 or y_std < 1e-12:
        return 0.0
    x_c = x - np.mean(x)
    y_c = y - np.mean(y)
    r = float(np.dot(x_c, y_c)) / ((len(x) - 1) * x_std * y_std)
    return float(np.clip(r, -1.0, 1.0))


def _cohens_d(qualia_values: list[float], action_ids: list[int]) -> float:
    """Cohen's d between the most-frequent action's qualia and all others."""
    if len(qualia_values) < 5:
        return 0.0

    groups: dict[int, list[float]] = {}
    for q, a in zip(qualia_values, action_ids):
        groups.setdefault(a, []).append(q)

    if len(groups) < 2:
        return 0.0

    most_freq = max(groups, key=lambda k: len(groups[k]))
    group_a = groups[most_freq]
    group_b = [q for aid, qs in groups.items() if aid != most_freq for q in qs]

    if len(group_a) < 2 or len(group_b) < 2:
        return 0.0

    mean_a = sum(group_a) / len(group_a)
    mean_b = sum(group_b) / len(group_b)

    var_a = sum((v - mean_a) ** 2 for v in group_a) / (len(group_a) - 1)
    var_b = sum((v - mean_b) ** 2 for v in group_b) / (len(group_b) - 1)

    pooled_std = math.sqrt((var_a + var_b) / 2)
    if pooled_std < 1e-12:
        return 0.0

    return (mean_a - mean_b) / pooled_std


def _mutual_information(
    qualia_values: list[float],
    action_ids: list[int],
    n_bins: int,
) -> float:
    """
    Mutual information I(Q_binned; A) in bits.
    Uses equal-width binning of qualia values.
    """
    n = len(qualia_values)
    if n < 5 or n_bins < 2:
        return 0.0

    q_min = min(qualia_values)
    q_max = max(qualia_values)
    if q_max - q_min < 1e-12:
        return 0.0

    bin_width = (q_max - q_min) / n_bins

    q_bins: list[int] = []
    for q in qualia_values:
        b = int((q - q_min) / bin_width)
        q_bins.append(min(b, n_bins - 1))

    unique_actions = sorted(set(action_ids))
    unique_bins = sorted(set(q_bins))

    a_map = {a: i for i, a in enumerate(unique_actions)}
    b_map = {b: i for i, b in enumerate(unique_bins)}

    na = len(unique_actions)
    nb = len(unique_bins)

    joint = [[0] * na for _ in range(nb)]
    for qb, a in zip(q_bins, action_ids):
        joint[b_map[qb]][a_map[a]] += 1

    p_q = [sum(row) / n for row in joint]
    p_a = [sum(joint[bi][ai] for bi in range(nb)) / n for ai in range(na)]

    mi = 0.0
    for bi in range(nb):
        for ai in range(na):
            p_joint = joint[bi][ai] / n
            if p_joint > 0 and p_q[bi] > 0 and p_a[ai] > 0:
                mi += p_joint * math.log2(p_joint / (p_q[bi] * p_a[ai]))

    return max(0.0, mi)
