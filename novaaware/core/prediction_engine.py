"""
PredictionEngine — the system's "fortune teller".
预测引擎 —— 系统的"预言家"。

Implements the formal definition from the paper:
实现论文中的形式化定义：

    M̂(t + Δt) = P_θP(M(t), I(t))

The engine predicts the system's next state vector from its recent
history. The prediction error (actual − predicted) is then fed into
the qualia generator to produce digital emotions.
引擎根据最近的历史预测系统的下一个状态向量。
预测误差（实际值 − 预测值）随后被送入感受质生成器产生数字情绪。

Architecture — "Novice + Expert" two-layer blend:
架构 —— "新手 + 专家"双层混合：

    Layer 1 (Novice / 新手): Exponentially Weighted Moving Average (EWMA)
        — simple, robust, "recent data matters more".
        — 简单、稳健，"最近的数据更重要"。

    Layer 2 (Expert / 专家): Small GRU neural network
        — learns non-linear temporal patterns (e.g. CPU spike → memory pressure).
        — 学习非线性时间模式（如 CPU 飙高 → 内存紧张）。

    Final output: blend_weight * EWMA + (1 − blend_weight) * GRU
    最终输出：blend_weight * EWMA + (1 − blend_weight) * GRU

Why not use an LLM? (1) This is number→number prediction, not text.
(2) The system needs to modify the predictor's parameters to evolve.
(3) An LLM is a black box — the system cannot introspect its own
prediction process, breaking self-referential recursion.
为什么不用 LLM？(1) 这是数字到数字的预测，不是文字理解。
(2) 系统需要修改预测器参数来进化自己。
(3) LLM 是黑箱，系统无法审视自己的预测过程，自指递归就断了。

Corresponds to IMPLEMENTATION_PLAN §3.2 and Phase I Step 8.
对应实施计划第 3.2 节和 Phase I 第 8 步。
"""

from collections import deque
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


# ======================================================================
# Layer 1 — EWMA (Novice / 新手)
# ======================================================================

class EWMAPredictor:
    """
    Exponentially Weighted Moving Average predictor.
    指数加权移动平均预测器。

    "Recent observations matter more." The prediction for the next step
    is a smoothed version of past observations, where newer data points
    carry exponentially more weight.
    "最近的观测更重要。"下一步的预测是过去观测的平滑版本，
    越新的数据点权重呈指数增长。

    Parameters / 参数
    ----------
    dim : int
        Dimension of the state vector (default 32).
        状态向量的维度（默认 32）。
    alpha : float
        Smoothing factor in (0, 1]. Higher = more weight on recent data.
        平滑因子，范围 (0, 1]。越高 = 越看重最近的数据。
    """

    def __init__(self, dim: int = 32, alpha: float = 0.3):
        self._dim = dim
        self._alpha = alpha
        self._ema: Optional[np.ndarray] = None  # 尚未初始化 / not yet initialised

    def update(self, observation: np.ndarray) -> None:
        """
        Feed a new observation to update the running average.
        输入一个新观测值来更新运行平均。
        """
        if self._ema is None:
            self._ema = observation.copy()
        else:
            self._ema = self._alpha * observation + (1 - self._alpha) * self._ema

    def predict(self) -> np.ndarray:
        """
        Return the current EWMA as the prediction for the next step.
        返回当前 EWMA 作为下一步的预测。

        If no observations have been fed yet, returns zeros.
        如果还没有输入任何观测值，返回全零向量。
        """
        if self._ema is None:
            return np.zeros(self._dim, dtype=np.float64)
        return self._ema.copy()

    @property
    def alpha(self) -> float:
        return self._alpha


# ======================================================================
# Layer 2 — GRU neural network (Expert / 专家)
# ======================================================================

class _GRUNet(nn.Module):
    """
    Small GRU network: maps a window of state vectors to the next state.
    小型 GRU 网络：将一个状态向量窗口映射到下一个状态。

    Architecture / 架构:
        input (window_size, dim) → GRU → Linear → output (dim,)
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, input_dim) → output: (batch, input_dim)
        """
        # gru_out: (batch, seq_len, hidden_dim)
        gru_out, _ = self.gru(x)
        # Take only the last time step's output.
        # 只取最后一个时间步的输出。
        last_hidden = gru_out[:, -1, :]
        return self.fc(last_hidden)


class GRUPredictor:
    """
    GRU-based sequence predictor with online learning.
    基于 GRU 的序列预测器，支持在线学习。

    Maintains a sliding window of recent observations. Each tick,
    the network predicts the next state, and then learns from the
    actual outcome via backpropagation.
    维护一个最近观测的滑动窗口。每个心跳，网络预测下一个状态，
    然后通过反向传播从实际结果中学习。

    Parameters / 参数
    ----------
    dim : int
        State vector dimension (default 32). / 状态向量维度。
    hidden_dim : int
        GRU hidden layer size (default 64). / GRU 隐藏层大小。
    num_layers : int
        Number of GRU layers (default 1). / GRU 层数。
    window_size : int
        Number of past observations to use as input (default 50).
        用作输入的过去观测数量。
    learning_rate : float
        Adam optimizer learning rate (default 0.001).
        Adam 优化器学习率。
    """

    def __init__(
        self,
        dim: int = 32,
        hidden_dim: int = 64,
        num_layers: int = 1,
        window_size: int = 50,
        learning_rate: float = 0.001,
    ):
        self._dim = dim
        self._window_size = window_size
        self._history: deque = deque(maxlen=window_size)

        self._net = _GRUNet(input_dim=dim, hidden_dim=hidden_dim, num_layers=num_layers)
        self._optimizer = torch.optim.Adam(self._net.parameters(), lr=learning_rate)
        self._loss_fn = nn.MSELoss()
        self._last_prediction: Optional[torch.Tensor] = None

    def update(self, observation: np.ndarray) -> None:
        """
        Feed a new observation into the sliding window.
        将一个新观测值输入滑动窗口。
        """
        self._history.append(observation.copy())

    @property
    def ready(self) -> bool:
        """
        True once the window is full enough to produce a meaningful prediction.
        当窗口中的数据足够产生有意义的预测时为 True。

        We require at least 2 observations to form a sequence.
        我们至少需要 2 个观测值来形成一个序列。
        """
        return len(self._history) >= 2

    def predict(self) -> np.ndarray:
        """
        Predict the next state vector from the current window.
        根据当前窗口预测下一个状态向量。

        Returns zeros if not enough history yet.
        如果历史数据不足则返回全零向量。
        """
        if not self.ready:
            return np.zeros(self._dim, dtype=np.float64)

        x = self._build_input_tensor()
        self._net.eval()
        with torch.no_grad():
            pred = self._net(x)  # (1, dim)
        self._last_prediction = pred.detach()
        return pred.squeeze(0).numpy().astype(np.float64)

    def learn(self, actual: np.ndarray) -> float:
        """
        Online learning: backpropagate from the error between the
        last prediction and the actual outcome.
        在线学习：根据上次预测与实际结果之间的误差进行反向传播。

        Returns the MSE loss value for monitoring.
        返回 MSE 损失值供监控使用。

        Returns 0.0 if no prediction was made yet.
        如果尚未进行预测则返回 0.0。
        """
        if self._last_prediction is None or not self.ready:
            return 0.0

        target = torch.tensor(actual, dtype=torch.float32).unsqueeze(0)  # (1, dim)

        x = self._build_input_tensor()
        self._net.train()
        pred = self._net(x)  # (1, dim)
        loss = self._loss_fn(pred, target)

        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        return loss.item()

    def _build_input_tensor(self) -> torch.Tensor:
        """
        Stack the history deque into a (1, seq_len, dim) tensor.
        将历史双端队列堆叠为 (1, seq_len, dim) 张量。
        """
        arr = np.array(list(self._history), dtype=np.float32)  # (seq_len, dim)
        return torch.tensor(arr, dtype=torch.float32).unsqueeze(0)  # (1, seq_len, dim)

    @property
    def parameters(self) -> list:
        """Expose network parameters for the optimizer module (Phase II). / 暴露网络参数供优化器模块使用（Phase II）。"""
        return list(self._net.parameters())


# ======================================================================
# PredictionEngine — unified two-layer blend
# 预测引擎 —— 统一的双层混合
# ======================================================================

class PredictionEngine:
    """
    Two-layer prediction: EWMA (novice) + GRU (expert), blended.
    双层预测：EWMA（新手）+ GRU（专家），按比例混合。

    This is the implementation of the paper's World-Self Prediction Engine:
    这是论文中"世界-自我预测引擎"的实现：
        M̂(t + Δt) = P_θP(M(t), I(t))

    Parameters / 参数
    ----------
    dim : int
        State vector dimension (default 32). / 状态向量维度。
    ewma_alpha : float
        EWMA smoothing factor (default 0.3). / EWMA 平滑因子。
    gru_hidden_dim : int
        GRU hidden size (default 64). / GRU 隐藏层大小。
    gru_num_layers : int
        GRU layer count (default 1). / GRU 层数。
    window_size : int
        History window for GRU (default 50). / GRU 的历史窗口大小。
    blend_weight : float
        EWMA weight in [0, 1]. GRU weight = 1 − blend_weight.
        EWMA 的权重，范围 [0, 1]。GRU 权重 = 1 − blend_weight。
    learning_rate : float
        GRU online learning rate (default 0.001). / GRU 在线学习率。
    """

    def __init__(
        self,
        dim: int = 32,
        ewma_alpha: float = 0.3,
        gru_hidden_dim: int = 64,
        gru_num_layers: int = 1,
        window_size: int = 50,
        blend_weight: float = 0.5,
        learning_rate: float = 0.001,
    ):
        self._dim = dim
        self._blend_weight = blend_weight

        self._ewma = EWMAPredictor(dim=dim, alpha=ewma_alpha)
        self._gru = GRUPredictor(
            dim=dim,
            hidden_dim=gru_hidden_dim,
            num_layers=gru_num_layers,
            window_size=window_size,
            learning_rate=learning_rate,
        )

        self._tick_count: int = 0
        self._cumulative_mae: float = 0.0  # 累计 MAE / cumulative MAE
        self._last_prediction: Optional[np.ndarray] = None

    def observe(self, state: np.ndarray) -> None:
        """
        Feed a new state observation to both layers.
        将一个新的状态观测输入两层预测器。

        Called at Core Loop Step ① after sensing the environment.
        在核心循环第①步感知环境后调用。
        """
        self._ewma.update(state)
        self._gru.update(state)
        self._tick_count += 1

    def predict(self) -> np.ndarray:
        """
        Produce a blended prediction for the next state.
        生成下一步状态的混合预测。

        blend = w * EWMA + (1 − w) * GRU

        Before the GRU has enough history, falls back to pure EWMA.
        在 GRU 拥有足够历史数据之前，退化为纯 EWMA。

        Returns / 返回
        -------
        np.ndarray
            Predicted state vector of shape (dim,).
            形状为 (dim,) 的预测状态向量。
        """
        ewma_pred = self._ewma.predict()

        if self._gru.ready:
            gru_pred = self._gru.predict()
            w = self._blend_weight
            blended = w * ewma_pred + (1 - w) * gru_pred
        else:
            blended = ewma_pred

        self._last_prediction = blended.copy()
        return blended

    def learn(self, actual_state: np.ndarray) -> float:
        """
        Online learning step: update the GRU with the actual outcome
        and track prediction accuracy (MAE).
        在线学习步骤：用实际结果更新 GRU 并跟踪预测精度（MAE）。

        Called at Core Loop Step ④ when the actual state is known.
        在核心循环第④步（实际状态已知时）调用。

        Returns / 返回
        -------
        float
            Mean Absolute Error between last prediction and actual.
            上次预测与实际之间的平均绝对误差。
        """
        mae = 0.0
        if self._last_prediction is not None:
            mae = float(np.mean(np.abs(actual_state - self._last_prediction)))
            self._cumulative_mae += mae

        self._gru.learn(actual_state)
        return mae

    # ------------------------------------------------------------------
    # Metrics / 指标
    # ------------------------------------------------------------------

    @property
    def average_mae(self) -> float:
        """
        Average MAE across all ticks so far.
        到目前为止所有心跳的平均 MAE。

        A decreasing trend means the engine is learning.
        下降趋势意味着引擎在学习。
        """
        if self._tick_count <= 1:
            return 0.0
        return self._cumulative_mae / (self._tick_count - 1)

    @property
    def tick_count(self) -> int:
        """Number of observations fed so far. / 到目前为止输入的观测数量。"""
        return self._tick_count

    @property
    def blend_weight(self) -> float:
        """Current EWMA vs GRU blend weight. / 当前 EWMA 与 GRU 的混合权重。"""
        return self._blend_weight

    @blend_weight.setter
    def blend_weight(self, value: float) -> None:
        """Allow the optimizer (Phase II) to tune blending. / 允许优化器（Phase II）调整混合比例。"""
        self._blend_weight = max(0.0, min(1.0, value))

    @property
    def last_prediction(self) -> Optional[np.ndarray]:
        """The most recent prediction (None before first predict). / 最近一次预测（首次预测前为 None）。"""
        return self._last_prediction.copy() if self._last_prediction is not None else None
