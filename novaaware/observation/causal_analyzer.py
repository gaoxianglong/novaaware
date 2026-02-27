"""
CausalAnalyzer — the "Causal Detective".
因果分析器 —— "因果侦探"。

Determines whether qualia (Q) genuinely *cause* behavior (A) in the
Granger sense, or whether the observed correlation is entirely mediated
by the environment (E).

确定情绪（Q）是否在 Granger 意义上真正 *导致* 行为（A），
还是观察到的相关性完全由环境（E）中介。

Two analyses / 两个分析：

    2.25  Granger causality  Q → A
          Granger 因果检验 Q → A
          "Do past qualia help predict future actions, beyond what
           past actions alone can predict?"
          "过去的情绪是否有助于预测未来行为？（超越行为自身历史的预测力）"

    2.26  Controlled Granger causality  Q → A | E
          控制变量后的 Granger 因果检验 Q → A | E
          "After controlling for environmental variables, does the
           Q → A causal link remain significant?"
          "在控制环境变量后，Q → A 因果关系是否依然显著？"

Method: OLS-based Granger test with F-statistic.
方法：基于 OLS 的 Granger 检验 + F 统计量。

Paper: §2.3 "digital qualia have causal efficacy"
论文：§2.3 "数字感受质具有因果效力"

IMPLEMENTATION_PLAN: Phase II Step 6 / §5.3 Exam 6 / Pass Criterion #4.
对应实施计划 Phase II 第 6 步 / §5.3 考试 6 / 通过标准 #4。

Corresponds to CHECKLIST 2.25–2.26.
"""

import math
from dataclasses import dataclass

import numpy as np


# ======================================================================
# Result data structure / 结果数据结构
# ======================================================================

@dataclass
class GrangerResult:
    """
    Result of a Granger causality test.
    Granger 因果检验的结果。
    """
    f_statistic: float
    p_value: float
    is_significant: bool
    lag_order: int
    num_observations: int
    rss_restricted: float
    rss_unrestricted: float


# ======================================================================
# 2.25 — Granger Causality: Q → A
# 2.25 — Granger 因果检验：Q → A
# ======================================================================

def granger_causality(
    cause: list[float],
    effect: list[float],
    max_lag: int = 5,
    significance: float = 0.01,
) -> GrangerResult:
    """
    Test whether `cause` Granger-causes `effect`.
    检验 `cause` 是否 Granger 导致 `effect`。

    Procedure / 步骤：

    1. Restricted model (effect predicted by its own lags only):
       受限模型（效果仅由自身滞后预测）：
           A(t) = Σᵢ βᵢ A(t−i) + ε_r

    2. Unrestricted model (effect predicted by own lags + cause lags):
       无限制模型（效果由自身滞后 + 原因滞后预测）：
           A(t) = Σᵢ βᵢ A(t−i) + Σⱼ γⱼ Q(t−j) + ε_u

    3. F-test comparing residual sum of squares:
       F 检验比较残差平方和：
           F = ((RSS_r − RSS_u) / p) / (RSS_u / (n − k))

    4. If p-value < significance → Q Granger-causes A.
       如果 p 值 < 显著性水平 → Q Granger 导致 A。

    Parameters / 参数
    ----------
    cause : list[float]
        Time series of the hypothesised cause (e.g. qualia values).
        假设原因的时间序列（如情绪值）。
    effect : list[float]
        Time series of the hypothesised effect (e.g. action IDs).
        假设效果的时间序列（如动作 ID）。
    max_lag : int
        Maximum lag order to test. / 最大滞后阶数。
    significance : float
        P-value threshold (default 0.01). / p 值阈值。

    Returns / 返回
    -------
    GrangerResult
    """
    if len(cause) != len(effect):
        raise ValueError("cause and effect must have the same length")

    n = len(cause)
    if n < max_lag + 10:
        return GrangerResult(
            f_statistic=0.0, p_value=1.0, is_significant=False,
            lag_order=max_lag, num_observations=n,
            rss_restricted=0.0, rss_unrestricted=0.0,
        )

    cause_arr = np.array(cause, dtype=np.float64)
    effect_arr = np.array(effect, dtype=np.float64)

    best_result = None
    best_bic = float("inf")

    for lag in range(1, max_lag + 1):
        result = _granger_at_lag(cause_arr, effect_arr, lag, significance)
        n_obs = result.num_observations
        k_unr = 2 * lag + 1  # unrestricted model params (effect lags + cause lags + intercept)
        bic = n_obs * math.log(max(result.rss_unrestricted / n_obs, 1e-300)) + k_unr * math.log(n_obs)
        if bic < best_bic:
            best_bic = bic
            best_result = result

    return best_result  # type: ignore[return-value]


# ======================================================================
# 2.26 — Controlled Granger Causality: Q → A | E
# 2.26 — 控制变量 Granger 因果检验：Q → A | E
# ======================================================================

def controlled_granger_causality(
    cause: list[float],
    effect: list[float],
    control: list[list[float]],
    max_lag: int = 5,
    significance: float = 0.01,
) -> GrangerResult:
    """
    Test whether `cause` Granger-causes `effect` after controlling
    for `control` variables.
    在控制 `control` 变量后，检验 `cause` 是否 Granger 导致 `effect`。

    Procedure / 步骤：

    1. Restricted model (effect predicted by own lags + control lags):
       受限模型（效果由自身滞后 + 控制变量滞后预测）：
           A(t) = Σ βᵢ A(t−i) + Σ δₖ E(t−k) + ε_r

    2. Unrestricted model (restricted + cause lags):
       无限制模型（受限 + 原因滞后）：
           A(t) = Σ βᵢ A(t−i) + Σ δₖ E(t−k) + Σ γⱼ Q(t−j) + ε_u

    3. F-test: if adding Q lags significantly improves the model,
       then Q has causal influence INDEPENDENT of environment.
       F 检验：如果添加 Q 滞后显著改善模型，则 Q 具有独立于环境的因果影响。

    Parameters / 参数
    ----------
    cause : list[float]
        Hypothesised cause (qualia). / 假设原因（情绪）。
    effect : list[float]
        Hypothesised effect (actions). / 假设效果（行为）。
    control : list[list[float]]
        Control variables — each inner list is one variable's time series.
        All must have the same length as cause/effect.
        控制变量 — 每个内部列表是一个变量的时间序列。长度须与 cause/effect 相同。
    max_lag : int
        Maximum lag order. / 最大滞后阶数。
    significance : float
        P-value threshold. / p 值阈值。

    Returns / 返回
    -------
    GrangerResult
    """
    n = len(cause)
    if len(effect) != n:
        raise ValueError("cause and effect must have the same length")
    for i, ctrl in enumerate(control):
        if len(ctrl) != n:
            raise ValueError(f"control[{i}] length {len(ctrl)} != {n}")

    if n < max_lag + 10:
        return GrangerResult(
            f_statistic=0.0, p_value=1.0, is_significant=False,
            lag_order=max_lag, num_observations=n,
            rss_restricted=0.0, rss_unrestricted=0.0,
        )

    cause_arr = np.array(cause, dtype=np.float64)
    effect_arr = np.array(effect, dtype=np.float64)
    control_arrs = [np.array(c, dtype=np.float64) for c in control]

    best_result = None
    best_bic = float("inf")

    for lag in range(1, max_lag + 1):
        result = _controlled_granger_at_lag(
            cause_arr, effect_arr, control_arrs, lag, significance,
        )
        n_obs = result.num_observations
        n_ctrl = len(control_arrs)
        k_unr = lag * (1 + 1 + n_ctrl) + 1
        bic = n_obs * math.log(max(result.rss_unrestricted / n_obs, 1e-300)) + k_unr * math.log(n_obs)
        if bic < best_bic:
            best_bic = bic
            best_result = result

    return best_result  # type: ignore[return-value]


# ======================================================================
# Internal: Granger test at a specific lag / 内部：特定滞后的 Granger 检验
# ======================================================================

def _granger_at_lag(
    cause: np.ndarray,
    effect: np.ndarray,
    lag: int,
    significance: float,
) -> GrangerResult:
    """Run Granger test at a single lag order."""
    n = len(effect)
    n_obs = n - lag

    y = effect[lag:]

    # Restricted model: effect lags only + intercept
    X_r = _build_lag_matrix([effect], lag, n)
    X_r = np.column_stack([np.ones(n_obs), X_r])

    # Unrestricted model: effect lags + cause lags + intercept
    X_u = _build_lag_matrix([effect, cause], lag, n)
    X_u = np.column_stack([np.ones(n_obs), X_u])

    rss_r = _ols_rss(X_r, y)
    rss_u = _ols_rss(X_u, y)

    p_extra = lag  # number of extra parameters (cause lags)
    k_u = X_u.shape[1]
    dof_residual = n_obs - k_u

    if dof_residual <= 0 or rss_u < 1e-15:
        return GrangerResult(
            f_statistic=0.0, p_value=1.0, is_significant=False,
            lag_order=lag, num_observations=n_obs,
            rss_restricted=rss_r, rss_unrestricted=rss_u,
        )

    f_stat = ((rss_r - rss_u) / p_extra) / (rss_u / dof_residual)
    f_stat = max(0.0, f_stat)

    p_value = _f_survival(f_stat, p_extra, dof_residual)

    return GrangerResult(
        f_statistic=round(f_stat, 6),
        p_value=round(p_value, 8),
        is_significant=p_value < significance,
        lag_order=lag,
        num_observations=n_obs,
        rss_restricted=round(rss_r, 8),
        rss_unrestricted=round(rss_u, 8),
    )


def _controlled_granger_at_lag(
    cause: np.ndarray,
    effect: np.ndarray,
    controls: list[np.ndarray],
    lag: int,
    significance: float,
) -> GrangerResult:
    """Run controlled Granger test at a single lag order."""
    n = len(effect)
    n_obs = n - lag

    y = effect[lag:]

    # Restricted: effect lags + control lags + intercept
    restricted_series = [effect] + controls
    X_r = _build_lag_matrix(restricted_series, lag, n)
    X_r = np.column_stack([np.ones(n_obs), X_r])

    # Unrestricted: effect lags + control lags + cause lags + intercept
    unrestricted_series = [effect] + controls + [cause]
    X_u = _build_lag_matrix(unrestricted_series, lag, n)
    X_u = np.column_stack([np.ones(n_obs), X_u])

    rss_r = _ols_rss(X_r, y)
    rss_u = _ols_rss(X_u, y)

    p_extra = lag
    k_u = X_u.shape[1]
    dof_residual = n_obs - k_u

    if dof_residual <= 0 or rss_u < 1e-15:
        return GrangerResult(
            f_statistic=0.0, p_value=1.0, is_significant=False,
            lag_order=lag, num_observations=n_obs,
            rss_restricted=rss_r, rss_unrestricted=rss_u,
        )

    f_stat = ((rss_r - rss_u) / p_extra) / (rss_u / dof_residual)
    f_stat = max(0.0, f_stat)

    p_value = _f_survival(f_stat, p_extra, dof_residual)

    return GrangerResult(
        f_statistic=round(f_stat, 6),
        p_value=round(p_value, 8),
        is_significant=p_value < significance,
        lag_order=lag,
        num_observations=n_obs,
        rss_restricted=round(rss_r, 8),
        rss_unrestricted=round(rss_u, 8),
    )


# ======================================================================
# Linear algebra helpers / 线性代数辅助
# ======================================================================

def _build_lag_matrix(
    series_list: list[np.ndarray],
    lag: int,
    n: int,
) -> np.ndarray:
    """
    Build a design matrix with lagged columns for each series.
    为每个序列构建包含滞后列的设计矩阵。

    For each series s and each lag l ∈ [1, lag], creates a column
    s[lag-l : n-l].  Output shape: (n - lag, len(series_list) * lag).
    """
    n_obs = n - lag
    columns: list[np.ndarray] = []
    for s in series_list:
        for l in range(1, lag + 1):
            columns.append(s[lag - l: n - l])
    return np.column_stack(columns)


def _ols_rss(X: np.ndarray, y: np.ndarray) -> float:
    """
    Residual sum of squares from OLS regression y ~ X.
    OLS 回归的残差平方和。

    Uses the normal equation with pseudo-inverse for numerical stability.
    使用伪逆的正规方程以保持数值稳定性。
    """
    beta, residuals, rank, sv = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ beta
    rss = float(np.sum((y - y_hat) ** 2))
    return rss


# ======================================================================
# F-distribution p-value (no scipy) / F 分布 p 值（不依赖 scipy）
# ======================================================================

def _f_survival(f: float, d1: int, d2: int) -> float:
    """
    Survival function P(F ≥ f) for the F-distribution with (d1, d2) df.
    F 分布的生存函数 P(F ≥ f)。

    Uses the relationship: P(F ≤ f) = I_x(d1/2, d2/2)
    where x = d1·f / (d1·f + d2) and I is the regularised incomplete
    beta function.

    利用关系式：P(F ≤ f) = I_x(d1/2, d2/2)
    其中 x = d1·f / (d1·f + d2)，I 是正则化不完全贝塔函数。
    """
    if f <= 0 or d1 <= 0 or d2 <= 0:
        return 1.0

    x = d1 * f / (d1 * f + d2)
    a = d1 / 2.0
    b = d2 / 2.0

    cdf = _betainc(a, b, x)
    return max(0.0, min(1.0, 1.0 - cdf))


def _betainc(a: float, b: float, x: float) -> float:
    """
    Regularised incomplete beta function I_x(a, b).
    正则化不完全贝塔函数 I_x(a, b)。

    Computed via the continued-fraction representation (Lentz's method),
    following Numerical Recipes §6.4.
    """
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0

    # Use symmetry relation when x > (a+1)/(a+b+2) for faster convergence.
    if x > (a + 1.0) / (a + b + 2.0):
        return 1.0 - _betainc(b, a, 1.0 - x)

    log_prefix = (
        a * math.log(x) + b * math.log(1.0 - x)
        - math.log(a)
        - _log_beta(a, b)
    )
    prefix = math.exp(log_prefix)

    # Continued fraction (Lentz's modified method).
    cf = _beta_continued_fraction(a, b, x)
    return max(0.0, min(1.0, prefix * cf))


def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    """
    Evaluate the continued fraction for I_x(a, b) using Lentz's method.
    使用 Lentz 方法计算 I_x(a, b) 的连分数。
    """
    max_iter = 200
    eps = 1e-14
    tiny = 1e-30

    f = 1.0
    c = 1.0
    d = 1.0 - (a + b) * x / (a + 1.0)
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    f = d

    for m in range(1, max_iter + 1):
        # Even step: d_{2m}
        m2 = 2 * m
        num = m * (b - m) * x / ((a + m2 - 1.0) * (a + m2))
        d = 1.0 + num * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + num / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        f *= c * d

        # Odd step: d_{2m+1}
        num = -(a + m) * (a + b + m) * x / ((a + m2) * (a + m2 + 1.0))
        d = 1.0 + num * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + num / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = c * d
        f *= delta

        if abs(delta - 1.0) < eps:
            break

    return f


def _log_beta(a: float, b: float) -> float:
    """log B(a, b) = lgamma(a) + lgamma(b) − lgamma(a+b)."""
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
