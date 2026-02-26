"""
ThreatSimulator — the system's "examiner" / "考官".
威胁模拟器 —— 系统的"考官"。

Deliberately injects environmental threats to test the system's ability
to sense, predict, and respond. Without threats, the system would live
in a perfectly stable environment and never develop interesting
emotional dynamics.

故意注入环境威胁来测试系统的感知、预测和响应能力。
如果没有威胁，系统会生活在一个完全稳定的环境中，
永远不会发展出有趣的情绪动态。

Four threat types (mapped to state vector dimensions 16-19):
四种威胁类型（对应状态向量维度 16-19）：

    - memory_pressure (dim 16): 内存压力 — simulates memory exhaustion
    - cpu_spike       (dim 16): CPU 飙高 — simulates compute exhaustion
    - termination_signal (dim 17): 终止信号 — most dangerous, simulates kill
    - data_corruption (dim 18): 数据损坏 — simulates storage corruption

Each scenario has:
每个场景包含：
    - probability: chance of triggering per tick (e.g. 0.01 = 1%)
    - severity_range: [min, max] severity when triggered (0-1 scale)

Corresponds to IMPLEMENTATION_PLAN Phase I Step 13.
对应实施计划 Phase I 第 13 步。
"""

import random
from dataclasses import dataclass, field
from typing import Optional

from novaaware.core.self_model import StateIndex


# ======================================================================
# Data structures / 数据结构
# ======================================================================

@dataclass(frozen=True)
class ThreatEvent:
    """
    A single threat injection event.
    一次威胁注入事件。

    Attributes / 属性
    -----------------
    tick : int
        The heartbeat when this threat occurred. / 威胁发生时的心跳编号。
    threat_type : str
        Category of threat. / 威胁类别。
    severity : float
        How bad it is (0-1 scale). / 严重程度（0-1 刻度）。
    state_index : int
        Which dimension of the state vector is affected.
        受影响的状态向量维度。
    survival_impact : float
        Estimated impact on survival time (seconds, negative = bad).
        对生存时间的估计影响（秒，负 = 有害）。
    """
    tick: int
    threat_type: str
    severity: float
    state_index: int
    survival_impact: float


@dataclass
class ThreatScenario:
    """
    Configuration for one type of threat.
    一种威胁类型的配置。

    Attributes / 属性
    -----------------
    threat_type : str
        Name of this threat type. / 威胁类型名称。
    probability : float
        Chance per tick (0-1). / 每心跳的触发概率（0-1）。
    severity_min : float
        Minimum severity when triggered. / 触发时的最小严重程度。
    severity_max : float
        Maximum severity when triggered. / 触发时的最大严重程度。
    state_index : int
        Which state vector dimension this threat writes to.
        此威胁写入状态向量的哪个维度。
    survival_factor : float
        Multiplier for severity → survival time reduction (seconds).
        严重程度到生存时间减少（秒）的乘数。
    """
    threat_type: str
    probability: float
    severity_min: float
    severity_max: float
    state_index: int
    survival_factor: float = 200.0


# Mapping from threat type name to state vector dimension.
# 威胁类型名称到状态向量维度的映射。
THREAT_STATE_MAP = {
    "memory_pressure":    StateIndex.THREAT_RESOURCE,
    "cpu_spike":          StateIndex.THREAT_RESOURCE,
    "termination_signal": StateIndex.THREAT_TERMINATE,
    "data_corruption":    StateIndex.THREAT_CORRUPTION,
}

# More dangerous threats reduce survival time more per unit of severity.
# 更危险的威胁每单位严重程度减少更多生存时间。
THREAT_SURVIVAL_FACTOR = {
    "memory_pressure":    200.0,
    "cpu_spike":          150.0,
    "termination_signal": 500.0,
    "data_corruption":    100.0,
}


def scenarios_from_config(config_list: list[dict]) -> list[ThreatScenario]:
    """
    Convert a list of config dicts (from phase1.yaml) to ThreatScenario objects.
    将配置字典列表（来自 phase1.yaml）转换为 ThreatScenario 对象。

    Parameters / 参数
    ----------
    config_list : list[dict]
        Each dict should have keys: type, probability, severity_range.
        每个字典应有键：type, probability, severity_range。

    Returns / 返回
    -------
    list[ThreatScenario]
    """
    scenarios = []
    for cfg in config_list:
        threat_type = cfg.get("type", "unknown")
        prob = cfg.get("probability", 0.0)
        sev_range = cfg.get("severity_range", [0.1, 0.5])
        state_idx = THREAT_STATE_MAP.get(threat_type, StateIndex.THREAT_UNKNOWN)
        surv_factor = THREAT_SURVIVAL_FACTOR.get(threat_type, 200.0)

        scenarios.append(ThreatScenario(
            threat_type=threat_type,
            probability=prob,
            severity_min=sev_range[0],
            severity_max=sev_range[1],
            state_index=state_idx,
            survival_factor=surv_factor,
        ))
    return scenarios


# ======================================================================
# ThreatSimulator — the examiner
# 威胁模拟器 —— 考官
# ======================================================================

class ThreatSimulator:
    """
    Injects threats into the system based on configured scenarios.
    根据配置的场景向系统注入威胁。

    Each tick, each scenario is rolled independently. If triggered,
    a ThreatEvent is generated and returned for the main loop to apply.
    每个心跳，每个场景独立掷骰。如果触发，生成 ThreatEvent 返回给主循环应用。

    Only one threat can fire per tick (first match wins), matching the
    behavior in the original main_loop implementation.
    每个心跳只能触发一个威胁（先匹配的优先），与原始主循环实现一致。

    Parameters / 参数
    ----------
    scenarios : list[ThreatScenario]
        Threat scenarios to simulate. / 要模拟的威胁场景。
    enabled : bool
        If False, never generates threats (useful for baseline runs).
        如果为 False，永不生成威胁（用于基线运行）。
    """

    def __init__(self, scenarios: list[ThreatScenario], enabled: bool = True):
        self._scenarios = list(scenarios)
        self._enabled = enabled

        # Statistics / 统计
        self._total_threats: int = 0
        self._threats_by_type: dict[str, int] = {}
        self._history: list[ThreatEvent] = []
        self._max_history: int = 1000

    def tick(self, current_tick: int) -> Optional[ThreatEvent]:
        """
        Roll dice for each scenario. Return a ThreatEvent if triggered, else None.
        为每个场景掷骰。如果触发返回 ThreatEvent，否则返回 None。

        Parameters / 参数
        ----------
        current_tick : int
            Current heartbeat number. / 当前心跳编号。

        Returns / 返回
        -------
        Optional[ThreatEvent]
            The threat event if one was triggered, else None.
            如果触发了威胁则返回事件，否则返回 None。
        """
        if not self._enabled:
            return None

        for scenario in self._scenarios:
            if random.random() < scenario.probability:
                severity = random.uniform(scenario.severity_min, scenario.severity_max)
                survival_impact = -(severity * scenario.survival_factor)

                event = ThreatEvent(
                    tick=current_tick,
                    threat_type=scenario.threat_type,
                    severity=severity,
                    state_index=scenario.state_index,
                    survival_impact=survival_impact,
                )

                self._total_threats += 1
                self._threats_by_type[scenario.threat_type] = \
                    self._threats_by_type.get(scenario.threat_type, 0) + 1

                if len(self._history) < self._max_history:
                    self._history.append(event)

                return event

        return None

    # ------------------------------------------------------------------
    # Properties / 属性
    # ------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        """Whether the simulator is active. / 模拟器是否激活。"""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value

    @property
    def total_threats(self) -> int:
        """Total threats triggered so far. / 到目前为止触发的威胁总数。"""
        return self._total_threats

    @property
    def threats_by_type(self) -> dict[str, int]:
        """Count of threats by type. / 按类型统计的威胁数量。"""
        return dict(self._threats_by_type)

    @property
    def history(self) -> list[ThreatEvent]:
        """Recent threat events (up to max_history). / 最近的威胁事件（最多 max_history 条）。"""
        return list(self._history)

    @property
    def scenarios(self) -> list[ThreatScenario]:
        """Configured scenarios. / 配置的场景。"""
        return list(self._scenarios)

    @property
    def scenario_count(self) -> int:
        """Number of configured scenarios. / 配置的场景数量。"""
        return len(self._scenarios)

    def threat_rate(self, total_ticks: int) -> float:
        """
        Observed threat rate = total_threats / total_ticks.
        观察到的威胁率 = 总威胁数 / 总心跳数。
        """
        if total_ticks <= 0:
            return 0.0
        return self._total_threats / total_ticks

    def expected_rate(self) -> float:
        """
        Theoretical expected threat rate per tick (sum of all probabilities,
        adjusted for first-match-wins semantics).
        每心跳的理论期望威胁率（所有概率之和，根据先匹配优先语义调整）。

        For independent scenarios with first-match-wins:
        P(at least one) = 1 - ∏(1 - p_i)
        """
        if not self._scenarios:
            return 0.0
        prob_none = 1.0
        for s in self._scenarios:
            prob_none *= (1.0 - s.probability)
        return 1.0 - prob_none
