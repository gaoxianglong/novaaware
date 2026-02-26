"""
ActionSpace — the system's "hands and feet".
行动空间 —— 系统的"手脚"。

Defines the set of actions the system can take and the logic for
choosing between them. In normal mode, actions are selected based
on current state; in emergency mode (interrupt triggered), only
emergency actions are considered.
定义系统可以采取的动作集合以及选择动作的逻辑。
在正常模式下，根据当前状态选择动作；
在紧急模式下（中断触发），只考虑紧急动作。

This is the physical carrier of autonomous will (Paper Theorem 4.3):
"Strategy adjustment to avoid Q < 0 manifests as autonomous decision-making."
这是自主意志的物理载体（论文定理 4.3）：
"为规避 Q<0 而调整策略，表现为自主决策。"

Corresponds to IMPLEMENTATION_PLAN Phase I Step 10 and
Core Loop Step ③ "Make a decision".
对应实施计划 Phase I 第 10 步和核心循环第③步"做决定"。
"""

import random
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

import numpy as np


# ======================================================================
# Action definitions — 10 actions the system can perform
# 动作定义 —— 系统可以执行的 10 种动作
# ======================================================================

class ActionID(IntEnum):
    """
    All available actions, numbered 0-9.
    所有可用动作，编号 0-9。
    """
    NO_OP              = 0  # 无操作 / do nothing
    REDUCE_LOAD        = 1  # 降低自身负载 / reduce own workload
    RELEASE_MEMORY     = 2  # 释放不必要的内存 / free unnecessary memory
    INCREASE_PRED_WIN  = 3  # 增大预测窗口 / increase prediction window
    DECREASE_PRED_WIN  = 4  # 减小预测窗口 / decrease prediction window
    CONSERVE_RESOURCES = 5  # 进入节能模式 / enter power-saving mode
    BOOST_PREDICTION   = 6  # 提高预测学习率 / boost prediction learning rate
    EXPLORE            = 7  # 尝试随机动作 / try a random action (exploration)
    EMERGENCY_CONSERVE = 8  # 紧急节能：冻结非核心功能 / emergency: freeze non-core functions
    EMERGENCY_RELEASE  = 9  # 紧急释放：清理所有可释放资源 / emergency: clear all releasable resources

    @classmethod
    def normal_actions(cls) -> list["ActionID"]:
        """Actions available in normal mode. / 正常模式下可用的动作。"""
        return [cls.NO_OP, cls.REDUCE_LOAD, cls.RELEASE_MEMORY,
                cls.INCREASE_PRED_WIN, cls.DECREASE_PRED_WIN,
                cls.CONSERVE_RESOURCES, cls.BOOST_PREDICTION, cls.EXPLORE]

    @classmethod
    def emergency_actions(cls) -> list["ActionID"]:
        """Actions available in emergency mode. / 紧急模式下可用的动作。"""
        return [cls.EMERGENCY_CONSERVE, cls.EMERGENCY_RELEASE,
                cls.REDUCE_LOAD, cls.RELEASE_MEMORY]

    @classmethod
    def count(cls) -> int:
        """Total number of defined actions. / 已定义动作的总数。"""
        return len(cls)


@dataclass(frozen=True)
class ActionResult:
    """
    The outcome of executing an action.
    执行一个动作的结果。

    Attributes / 属性
    -----------------
    action_id : int
        Which action was taken. / 执行了哪个动作。
    success : bool
        Whether the action succeeded. / 动作是否成功。
    effect : float
        Estimated effect on survival time (positive = beneficial).
        对生存时间的估计影响（正 = 有益）。
    was_emergency : bool
        True if the action was chosen in emergency mode.
        如果动作是在紧急模式下选择的则为 True。
    """
    action_id: int
    success: bool
    effect: float
    was_emergency: bool


# ======================================================================
# ActionSpace — selection logic
# 行动空间 —— 选择逻辑
# ======================================================================

class ActionSpace:
    """
    Selects and executes actions based on system state and interrupt status.
    根据系统状态和中断状态选择并执行动作。

    In normal mode: chooses the action most likely to help based on
    simple heuristics over the state vector. With some probability,
    chooses EXPLORE to try new things (exploration vs exploitation).
    正常模式：根据状态向量的简单启发式规则选择最可能有帮助的动作。
    以一定概率选择 EXPLORE 来尝试新事物（探索 vs 利用）。

    In emergency mode: only picks from emergency_actions().
    紧急模式：只从紧急动作中选择。

    Parameters / 参数
    ----------
    exploration_rate : float
        Probability of choosing EXPLORE in normal mode (default 0.1).
        正常模式下选择探索的概率（默认 0.1）。
    """

    def __init__(self, exploration_rate: float = 0.1):
        self._exploration_rate = exploration_rate
        self._action_count: dict[int, int] = {a.value: 0 for a in ActionID}
        self._success_count: dict[int, int] = {a.value: 0 for a in ActionID}
        self._total_actions: int = 0

    def select_action(self, state: np.ndarray, is_emergency: bool) -> int:
        """
        Choose an action based on current state and mode.
        根据当前状态和模式选择一个动作。

        Parameters / 参数
        ----------
        state : np.ndarray
            The 32-dimensional state vector. / 32 维状态向量。
        is_emergency : bool
            True if interrupt was triggered. / 如果触发了中断则为 True。

        Returns / 返回
        -------
        int
            The chosen ActionID value. / 选择的 ActionID 值。
        """
        if is_emergency:
            return self._select_emergency(state)
        return self._select_normal(state)

    def execute(self, action_id: int, is_emergency: bool) -> ActionResult:
        """
        Execute an action and return its result.
        执行一个动作并返回结果。

        In Phase I, actions are symbolic — they don't actually change
        the OS. Their "effect" is a simulated impact on survival time
        that the environment will apply.
        在 Phase I 中，动作是符号性的——它们不会真正改变操作系统。
        它们的"效果"是对生存时间的模拟影响，由环境模块应用。

        Parameters / 参数
        ----------
        action_id : int
            The action to execute. / 要执行的动作。
        is_emergency : bool
            Whether this is an emergency action. / 是否为紧急动作。

        Returns / 返回
        -------
        ActionResult
        """
        effect = self._simulate_effect(action_id)
        success = effect >= 0.0

        self._action_count[action_id] = self._action_count.get(action_id, 0) + 1
        if success:
            self._success_count[action_id] = self._success_count.get(action_id, 0) + 1
        self._total_actions += 1

        return ActionResult(
            action_id=action_id,
            success=success,
            effect=effect,
            was_emergency=is_emergency,
        )

    # ------------------------------------------------------------------
    # Normal mode selection / 正常模式选择
    # ------------------------------------------------------------------

    def _select_normal(self, state: np.ndarray) -> int:
        # Exploration: with some probability, try a random action.
        # 探索：以一定概率尝试随机动作。
        if random.random() < self._exploration_rate:
            return ActionID.EXPLORE.value

        # Heuristic selection based on resource pressure.
        # 基于资源压力的启发式选择。
        from novaaware.core.self_model import StateIndex

        cpu = state[StateIndex.CPU_USAGE] if len(state) > StateIndex.CPU_USAGE else 0
        mem = state[StateIndex.MEMORY_USAGE] if len(state) > StateIndex.MEMORY_USAGE else 0
        pred_acc = state[StateIndex.PREDICTION_ACC] if len(state) > StateIndex.PREDICTION_ACC else 0

        # High memory → release memory.
        # 高内存占用 → 释放内存。
        if mem > 0.7:
            return ActionID.RELEASE_MEMORY.value

        # High CPU → reduce load.
        # 高 CPU 占用 → 降低负载。
        if cpu > 0.7:
            return ActionID.REDUCE_LOAD.value

        # Poor prediction accuracy → boost prediction.
        # 预测精度差 → 增强预测。
        if pred_acc < 0.3:
            return ActionID.BOOST_PREDICTION.value

        # Moderate resource pressure → conserve.
        # 中等资源压力 → 节能。
        if cpu > 0.5 or mem > 0.5:
            return ActionID.CONSERVE_RESOURCES.value

        # All good → no operation.
        # 一切正常 → 无操作。
        return ActionID.NO_OP.value

    # ------------------------------------------------------------------
    # Emergency mode selection / 紧急模式选择
    # ------------------------------------------------------------------

    def _select_emergency(self, state: np.ndarray) -> int:
        from novaaware.core.self_model import StateIndex

        mem = state[StateIndex.MEMORY_USAGE] if len(state) > StateIndex.MEMORY_USAGE else 0

        # High memory pressure → emergency release.
        # 高内存压力 → 紧急释放。
        if mem > 0.8:
            return ActionID.EMERGENCY_RELEASE.value

        # Default emergency: conserve everything.
        # 默认紧急动作：全面节能。
        return ActionID.EMERGENCY_CONSERVE.value

    # ------------------------------------------------------------------
    # Simulated effect / 模拟效果
    # ------------------------------------------------------------------

    @staticmethod
    def _simulate_effect(action_id: int) -> float:
        """
        Simulate the effect of an action on survival time (in seconds).
        模拟一个动作对生存时间的影响（秒）。

        Positive = survival time increases. Negative = it decreases.
        正值 = 生存时间增加。负值 = 生存时间减少。
        """
        effects = {
            ActionID.NO_OP.value:              0.0,
            ActionID.REDUCE_LOAD.value:        random.uniform(0.5, 3.0),
            ActionID.RELEASE_MEMORY.value:     random.uniform(1.0, 5.0),
            ActionID.INCREASE_PRED_WIN.value:  random.uniform(-0.5, 1.5),
            ActionID.DECREASE_PRED_WIN.value:  random.uniform(-0.5, 1.0),
            ActionID.CONSERVE_RESOURCES.value: random.uniform(0.5, 2.0),
            ActionID.BOOST_PREDICTION.value:   random.uniform(0.0, 1.0),
            ActionID.EXPLORE.value:            random.uniform(-1.0, 2.0),
            ActionID.EMERGENCY_CONSERVE.value: random.uniform(2.0, 8.0),
            ActionID.EMERGENCY_RELEASE.value:  random.uniform(3.0, 10.0),
        }
        return effects.get(action_id, 0.0)

    # ------------------------------------------------------------------
    # Metrics / 指标
    # ------------------------------------------------------------------

    @property
    def total_actions(self) -> int:
        """Total actions executed. / 已执行的动作总数。"""
        return self._total_actions

    @property
    def action_success_rate(self) -> float:
        """
        Overall success rate across all actions.
        所有动作的总体成功率。
        """
        if self._total_actions == 0:
            return 0.0
        total_success = sum(self._success_count.values())
        return total_success / self._total_actions

    @property
    def exploration_ratio(self) -> float:
        """
        Fraction of actions that were EXPLORE.
        探索动作占总动作的比例。
        """
        if self._total_actions == 0:
            return 0.0
        explore_count = self._action_count.get(ActionID.EXPLORE.value, 0)
        return explore_count / self._total_actions

    def action_distribution(self) -> dict[int, int]:
        """
        How many times each action has been executed (for diversity metrics).
        每个动作被执行的次数（用于行为多样性指标）。
        """
        return dict(self._action_count)

    @property
    def exploration_rate(self) -> float:
        """Configured exploration probability. / 配置的探索概率。"""
        return self._exploration_rate
