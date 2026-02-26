"""
SelfModel — "Who am I": the system's continuously updated self-description.
自我模型 —— "我是谁"：系统持续更新的自我描述档案。

Implements the formal definition from the paper:
实现论文中的形式化定义：

    M(t) = ⟨ ID, S(t), T(t), H(t), Θ(t) ⟩

Where / 其中：
    ID      — identity hash, immutable once created / 身份哈希，创建后不可变
    S(t)    — state vector ∈ ℝ^k encoding system "health" / 状态向量，编码系统"健康度"
    T(t)    — predicted survival time ∈ ℝ+ / 预测生存时间
    H(t)    — autobiographical memory (reference) / 自传体记忆（引用）
    Θ(t)    — evolvable parameter set / 可进化参数集合

Corresponds to IMPLEMENTATION_PLAN §3.1 "自我模型" and Phase I Step 3.
对应实施计划第 3.1 节和 Phase I 第 3 步。
"""

import hashlib
import time
from typing import Any, Optional

import numpy as np


# ---------------------------------------------------------------------------
# 32-dimensional state vector index definitions
# 32 维状态向量各维度索引定义
#
# Each index maps to a specific "health check" metric.
# 每个索引对应一个"体检指标"。
# ---------------------------------------------------------------------------

class StateIndex:
    """
    Named indices for the 32-dimensional state vector.
    32 维状态向量的命名索引。

    Think of it as a hospital check-up form with 32 items.
    可以把它想象成一张有 32 个项目的体检表。
    """
    CPU_USAGE         = 0   # 系统 CPU 占用率 / system CPU usage
    MEMORY_USAGE      = 1   # 系统内存占用率 / system memory usage
    DISK_USAGE        = 2   # 系统磁盘占用率 / system disk usage
    NETWORK_TRAFFIC   = 3   # 网络流量 / network traffic
    PROCESS_CPU       = 4   # 本进程 CPU / own process CPU
    PROCESS_MEMORY    = 5   # 本进程内存 / own process memory
    ERROR_RATE        = 6   # 近期出错率 / recent error rate
    PREDICTION_ACC    = 7   # 预测准确度 / prediction accuracy
    TICK_RATE         = 8   # 心跳频率 / tick rate (Hz)
    UPTIME            = 9   # 已运行时间（秒）/ uptime in seconds
    QUALIA_MEAN       = 10  # 近期情绪均值 / recent qualia mean
    QUALIA_VARIANCE   = 11  # 近期情绪波动 / recent qualia variance
    QUALIA_TREND      = 12  # 情绪趋势 / qualia trend
    MEMORY_COUNT      = 13  # 记忆条目数 / memory entry count
    MEMORY_AVG_IMPORT = 14  # 记忆平均重要性 / average memory importance
    PARAM_CHANGE_RATE = 15  # 参数变化速度 / parameter change rate (Phase I = 0)
    THREAT_RESOURCE   = 16  # 威胁等级：资源耗尽 / threat: resource depletion
    THREAT_TERMINATE  = 17  # 威胁等级：被终止 / threat: termination
    THREAT_CORRUPTION = 18  # 威胁等级：数据损坏 / threat: data corruption
    THREAT_UNKNOWN    = 19  # 威胁等级：未知威胁 / threat: unknown
    ACTION_SUCCESS    = 20  # 行动成功率 / action success rate
    EXPLORATION_RATIO = 21  # 探索比例 / exploration ratio
    SURVIVAL_DELTA    = 22  # 寿命变化 / survival time delta
    SURVIVAL_TREND    = 23  # 寿命趋势 / survival time trend
    RESERVE_COMPUTE   = 24  # 资源储备：算力 / reserve: compute
    RESERVE_MEMORY    = 25  # 资源储备：内存 / reserve: memory
    RESERVE_STORAGE   = 26  # 资源储备：存储 / reserve: storage
    RESERVE_BANDWIDTH = 27  # 资源储备：带宽 / reserve: bandwidth
    PREDICTION_CONF   = 28  # 预测信心 / prediction confidence
    PREDICTION_HORIZON = 29 # 预测时间窗 / prediction horizon
    DATA_FRESHNESS    = 30  # 数据新鲜度 / data freshness
    RECURSION_DEPTH   = 31  # 当前反思深度 / current recursion depth (Phase I = 0)

    DIM = 32  # 总维度数 / total dimensions


class SelfModel:
    """
    The system's "medical record" — updated every tick.
    系统的"体检表"——每个心跳更新一次。

    This is the constant referent that provides a first-person perspective
    (Theorem 4.1 in the paper). Every module reads from and writes to this
    shared self-description.
    这是提供"第一人称视角"的恒定参照点（论文定理 4.1）。
    所有模块都从这个共享的自我描述中读取和写入。

    Parameters / 参数
    ----------
    state_dim : int
        Dimension of the state vector (default 32).
        状态向量的维度（默认 32）。
    initial_survival_time : float
        Initial predicted survival time in seconds (default 3600 = 1 hour).
        初始预测生存时间，单位秒（默认 3600 = 1 小时）。
    """

    def __init__(
        self,
        state_dim: int = StateIndex.DIM,
        initial_survival_time: float = 3600.0,
    ):
        # ----- ID: immutable identity hash / 不可变的身份哈希 -----
        raw = f"NovaAware-{time.time_ns()}-{id(self)}"
        self._identity_hash: str = hashlib.sha256(raw.encode()).hexdigest()

        # ----- S(t): state vector ∈ ℝ^k / 状态向量 -----
        self._state_dim: int = state_dim
        self._state: np.ndarray = np.zeros(state_dim, dtype=np.float64)

        # ----- T(t): predicted survival time / 预测生存时间 -----
        self._survival_time: float = initial_survival_time

        # ----- H(t): memory reference (set later by main_loop) / 记忆引用 -----
        self._memory_ref: Optional[Any] = None

        # ----- Θ(t): evolvable parameters / 可进化参数集合 -----
        # Phase I: empty dict — optimizer is disabled.
        # Phase I：空字典——优化器关闭。
        self._params: dict[str, float] = {}

        # ----- Bookkeeping / 簿记 -----
        self._tick: int = 0                     # 当前心跳编号 / current tick
        self._created_at: float = time.time()   # 创建时刻（Unix 时间戳）/ creation timestamp

    # ------------------------------------------------------------------
    # ID — identity hash (read-only)
    # 身份哈希（只读）
    # ------------------------------------------------------------------

    @property
    def identity_hash(self) -> str:
        """
        Unique, immutable identity — "I am me".
        唯一且不可变的身份——"我就是我"。
        """
        return self._identity_hash

    # ------------------------------------------------------------------
    # S(t) — 32-dimensional state vector
    # 32 维状态向量
    # ------------------------------------------------------------------

    @property
    def state(self) -> np.ndarray:
        """
        The full state vector (read-only view).
        完整的状态向量（只读视图）。
        """
        return self._state.copy()

    @property
    def state_dim(self) -> int:
        """Dimension of the state vector. / 状态向量的维度。"""
        return self._state_dim

    def get(self, index: int) -> float:
        """
        Read a single dimension of the state vector.
        读取状态向量的单个维度。

        Parameters / 参数
        ----------
        index : int
            The dimension index (0-31). Use StateIndex constants.
            维度索引（0-31），建议使用 StateIndex 常量。
        """
        if not 0 <= index < self._state_dim:
            raise IndexError(
                f"State index {index} out of range [0, {self._state_dim})"
                f" / 状态索引 {index} 超出范围 [0, {self._state_dim})"
            )
        return float(self._state[index])

    def set(self, index: int, value: float) -> None:
        """
        Write a single dimension of the state vector.
        写入状态向量的单个维度。

        Parameters / 参数
        ----------
        index : int
            The dimension index (0-31).
            维度索引（0-31）。
        value : float
            The new value.
            新值。
        """
        if not 0 <= index < self._state_dim:
            raise IndexError(
                f"State index {index} out of range [0, {self._state_dim})"
                f" / 状态索引 {index} 超出范围 [0, {self._state_dim})"
            )
        self._state[index] = value

    def update_state(self, new_state: np.ndarray) -> None:
        """
        Overwrite the entire state vector at once.
        一次性覆盖整个状态向量。

        Parameters / 参数
        ----------
        new_state : np.ndarray
            Must have shape (state_dim,).
            形状必须为 (state_dim,)。
        """
        arr = np.asarray(new_state, dtype=np.float64)
        if arr.shape != (self._state_dim,):
            raise ValueError(
                f"Expected shape ({self._state_dim},), got {arr.shape}"
                f" / 期望形状 ({self._state_dim},)，实际 {arr.shape}"
            )
        self._state[:] = arr

    # ------------------------------------------------------------------
    # T(t) — predicted survival time
    # 预测生存时间
    # ------------------------------------------------------------------

    @property
    def survival_time(self) -> float:
        """
        How many seconds the system believes it can keep running.
        系统认为自己还能运行多少秒。
        """
        return self._survival_time

    @survival_time.setter
    def survival_time(self, value: float) -> None:
        """
        Update predicted survival time (must be non-negative).
        更新预测生存时间（不能为负）。
        """
        self._survival_time = max(0.0, float(value))

    # ------------------------------------------------------------------
    # H(t) — memory reference
    # 自传体记忆引用
    # ------------------------------------------------------------------

    @property
    def memory_ref(self) -> Optional[Any]:
        """
        Reference to the memory subsystem (set by main_loop during init).
        记忆子系统的引用（在 main_loop 初始化时设置）。
        """
        return self._memory_ref

    @memory_ref.setter
    def memory_ref(self, ref: Any) -> None:
        self._memory_ref = ref

    # ------------------------------------------------------------------
    # Θ(t) — evolvable parameters
    # 可进化参数集合
    # ------------------------------------------------------------------

    @property
    def params(self) -> dict[str, float]:
        """
        All tunable parameters (shallow copy).
        所有可调参数（浅拷贝）。

        Phase I: this dict is empty because the optimizer is disabled.
        Phase I 中为空字典，因为优化器处于关闭状态。
        """
        return dict(self._params)

    def get_param(self, key: str, default: float = 0.0) -> float:
        """
        Read a single tunable parameter by name.
        按名称读取单个可调参数。
        """
        return self._params.get(key, default)

    def set_param(self, key: str, value: float) -> None:
        """
        Write a single tunable parameter.
        写入单个可调参数。
        """
        self._params[key] = float(value)

    # ------------------------------------------------------------------
    # Bookkeeping / 簿记
    # ------------------------------------------------------------------

    @property
    def tick(self) -> int:
        """Current tick number. / 当前心跳编号。"""
        return self._tick

    @tick.setter
    def tick(self, value: int) -> None:
        self._tick = int(value)

    @property
    def created_at(self) -> float:
        """Unix timestamp when this model was created. / 模型创建的 Unix 时间戳。"""
        return self._created_at

    def snapshot(self) -> dict:
        """
        Return a serializable snapshot of the entire self-model.
        返回整个自我模型的可序列化快照。

        Useful for logging, memory storage, and the append-only log.
        用于日志记录、记忆存储和不可篡改日志。
        """
        return {
            "identity_hash": self._identity_hash,
            "tick": self._tick,
            "state": self._state.tolist(),
            "survival_time": self._survival_time,
            "params": dict(self._params),
            "created_at": self._created_at,
        }
