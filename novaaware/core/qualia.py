"""
QualiaGenerator — the system's "emotion engine".
感受质生成器 —— 系统的"情绪发生器"。

Implements the paper's qualia function with all three axioms:
实现论文的感受质函数及全部三条公理：

    Q(t) = f(ΔT),  where ΔT = T_actual − T̂  (prediction error)

    Axiom A1 (Valence Monotonicity / 效价单调性):
        f is monotonically increasing in ΔT.
        f 关于 ΔT 单调递增。
        → Positive ΔT (better than expected) → positive Q (good feeling).
        → Negative ΔT (worse than expected) → negative Q (bad feeling).

    Axiom A2 (Negative Amplification / 负向放大):
        |f(−x)| > |f(x)| for all x > 0.
        对同等幅度的威胁与利好，威胁的响应更强。
        → Loss aversion ratio ≈ 2.25 : 1 (psychology: losing $100
          hurts more than finding $100 feels good).

    Axiom A3 (Global Broadcast / 全局广播性):
        Q(t) is immediately readable by all submodules; above a
        threshold it can interrupt current computation.
        Q(t) 生成后对所有子模块即刻可读；超过阈值可中断计算。
        → Implemented in concert with GlobalWorkspace.

The function uses tanh to guarantee bounded output, satisfy
monotonicity, and allow the asymmetric positive/negative scaling.
使用 tanh 保证输出有界、满足单调性，并允许正/负不对称缩放。

Corresponds to IMPLEMENTATION_PLAN §3.3 and Phase I Step 9.
对应实施计划第 3.3 节和 Phase I 第 9 步。
"""

import math
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class QualiaSignal:
    """
    The output of a single qualia computation.
    一次感受质计算的输出。

    Attributes / 属性
    -----------------
    delta_t : float
        Prediction error: T_actual − T_predicted.
        预测误差：实际寿命 − 预测寿命。
    value : float
        Qualia value Q(t). Positive = good, negative = bad.
        Range: [−alpha_neg, +alpha_pos] (default [−2.25, +1.0]).
        感受质值 Q(t)。正 = 好，负 = 坏。
        范围：[−alpha_neg, +alpha_pos]（默认 [−2.25, +1.0]）。
    intensity : float
        |Q(t)| — always >= 0. / Q(t) 的绝对值，始终 >= 0。
    is_interrupt : bool
        True if intensity >= interrupt_threshold.
        如果强度 >= 中断阈值则为 True。
    """
    delta_t: float
    value: float
    intensity: float
    is_interrupt: bool


class QualiaGenerator:
    """
    Converts prediction error into a valenced emotional signal.
    将预测误差转换为带效价的情绪信号。

    The core formula (using tanh for boundedness and monotonicity):
    核心公式（使用 tanh 保证有界性和单调性）：

        if ΔT >= 0:  Q = alpha_pos * tanh(beta * ΔT)     ∈ [0, alpha_pos]
        if ΔT <  0:  Q = alpha_neg * tanh(beta * ΔT)     ∈ [−alpha_neg, 0)

    This satisfies:
    这满足了：
        A1: tanh is monotonically increasing, scaling preserves monotonicity.
        A2: alpha_neg (2.25) > alpha_pos (1.0), so |f(−x)| > |f(x)|.

    Parameters / 参数
    ----------
    alpha_pos : float
        Maximum positive qualia (default 1.0). / 最大正面情绪。
    alpha_neg : float
        Maximum negative qualia magnitude (default 2.25). / 最大负面情绪幅度。
    beta : float
        Sensitivity: how quickly Q saturates (default 1.0). / 灵敏度。
    interrupt_threshold : float
        |Q| above this triggers an interrupt (default 0.7). / 中断阈值。
    """

    def __init__(
        self,
        alpha_pos: float = 1.0,
        alpha_neg: float = 2.25,
        beta: float = 1.0,
        interrupt_threshold: float = 0.7,
    ):
        self._alpha_pos = alpha_pos
        self._alpha_neg = alpha_neg
        self._beta = beta
        self._interrupt_threshold = interrupt_threshold
        self._last_signal: Optional[QualiaSignal] = None

    def compute(self, t_actual: float, t_predicted: float) -> QualiaSignal:
        """
        Compute the qualia signal from actual vs. predicted survival time.
        根据实际寿命与预测寿命计算感受质信号。

        This is called at Core Loop Step ⑤.
        在核心循环第⑤步调用。

        Parameters / 参数
        ----------
        t_actual : float
            Actual survival time (seconds). / 实际生存时间（秒）。
        t_predicted : float
            Predicted survival time (seconds). / 预测生存时间（秒）。

        Returns / 返回
        -------
        QualiaSignal
        """
        delta_t = t_actual - t_predicted

        # Core formula — asymmetric tanh.
        # 核心公式 —— 不对称 tanh。
        raw = math.tanh(self._beta * delta_t)

        if delta_t >= 0:
            value = self._alpha_pos * raw
        else:
            value = self._alpha_neg * raw

        intensity = abs(value)
        is_interrupt = intensity >= self._interrupt_threshold

        signal = QualiaSignal(
            delta_t=delta_t,
            value=value,
            intensity=intensity,
            is_interrupt=is_interrupt,
        )
        self._last_signal = signal
        return signal

    # ------------------------------------------------------------------
    # Properties / 属性
    # ------------------------------------------------------------------

    @property
    def last_signal(self) -> Optional[QualiaSignal]:
        """Most recent qualia signal. / 最近一次感受质信号。"""
        return self._last_signal

    @property
    def alpha_pos(self) -> float:
        """Maximum positive qualia. / 最大正面情绪。"""
        return self._alpha_pos

    @property
    def alpha_neg(self) -> float:
        """Maximum negative qualia magnitude. / 最大负面情绪幅度。"""
        return self._alpha_neg

    @property
    def beta(self) -> float:
        """Sensitivity parameter. / 灵敏度参数。"""
        return self._beta

    @property
    def interrupt_threshold(self) -> float:
        """Interrupt threshold. / 中断阈值。"""
        return self._interrupt_threshold
