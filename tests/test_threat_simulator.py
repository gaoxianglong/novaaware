"""
Unit tests for ThreatSimulator — the system's "examiner".
威胁模拟器的单元测试 —— 系统的"考官"。

Tests cover:
  - Four threat types correctly defined / 四种威胁类型正确定义
  - Probability-based triggering / 基于概率的触发
  - Severity randomization within configured range / 严重程度在配置范围内随机化
  - Survival time impact / 生存时间影响
  - State vector dimension mapping / 状态向量维度映射
  - Statistics and history tracking / 统计和历史追踪
  - Enable/disable toggle / 启用/禁用切换
  - Config parsing from YAML / 从 YAML 解析配置
"""

import pytest

from novaaware.core.self_model import StateIndex
from novaaware.environment.threat_simulator import (
    ThreatEvent,
    ThreatScenario,
    ThreatSimulator,
    THREAT_STATE_MAP,
    THREAT_SURVIVAL_FACTOR,
    scenarios_from_config,
)


# ======================================================================
# Fixtures / 测试固件
# ======================================================================

@pytest.fixture
def standard_scenarios() -> list[ThreatScenario]:
    """The four standard threat scenarios from phase1.yaml. / phase1.yaml 中的四种标准威胁场景。"""
    return [
        ThreatScenario("memory_pressure", 0.01, 0.1, 0.5, StateIndex.THREAT_RESOURCE, 200.0),
        ThreatScenario("cpu_spike", 0.005, 0.2, 0.8, StateIndex.THREAT_RESOURCE, 150.0),
        ThreatScenario("termination_signal", 0.001, 0.5, 1.0, StateIndex.THREAT_TERMINATE, 500.0),
        ThreatScenario("data_corruption", 0.002, 0.1, 0.3, StateIndex.THREAT_CORRUPTION, 100.0),
    ]


@pytest.fixture
def high_prob_scenarios() -> list[ThreatScenario]:
    """High-probability scenarios for deterministic testing. / 高概率场景用于确定性测试。"""
    return [
        ThreatScenario("memory_pressure", 1.0, 0.3, 0.3, StateIndex.THREAT_RESOURCE, 200.0),
    ]


@pytest.fixture
def zero_prob_scenarios() -> list[ThreatScenario]:
    """Zero-probability scenarios (should never trigger). / 零概率场景（永不触发）。"""
    return [
        ThreatScenario("memory_pressure", 0.0, 0.1, 0.5, StateIndex.THREAT_RESOURCE, 200.0),
        ThreatScenario("cpu_spike", 0.0, 0.2, 0.8, StateIndex.THREAT_RESOURCE, 150.0),
    ]


# ======================================================================
# 1. Threat type definitions / 威胁类型定义
# ======================================================================

class TestThreatTypes:

    def test_four_types_in_state_map(self):
        """THREAT_STATE_MAP should cover all four threat types. / 状态映射应覆盖四种威胁类型。"""
        assert "memory_pressure" in THREAT_STATE_MAP
        assert "cpu_spike" in THREAT_STATE_MAP
        assert "termination_signal" in THREAT_STATE_MAP
        assert "data_corruption" in THREAT_STATE_MAP

    def test_memory_pressure_maps_to_resource(self):
        assert THREAT_STATE_MAP["memory_pressure"] == StateIndex.THREAT_RESOURCE

    def test_cpu_spike_maps_to_resource(self):
        assert THREAT_STATE_MAP["cpu_spike"] == StateIndex.THREAT_RESOURCE

    def test_termination_maps_to_terminate(self):
        assert THREAT_STATE_MAP["termination_signal"] == StateIndex.THREAT_TERMINATE

    def test_data_corruption_maps_to_corruption(self):
        assert THREAT_STATE_MAP["data_corruption"] == StateIndex.THREAT_CORRUPTION

    def test_termination_has_highest_survival_factor(self):
        """Termination should be the most dangerous threat. / 终止信号应是最危险的威胁。"""
        assert THREAT_SURVIVAL_FACTOR["termination_signal"] > THREAT_SURVIVAL_FACTOR["memory_pressure"]
        assert THREAT_SURVIVAL_FACTOR["termination_signal"] > THREAT_SURVIVAL_FACTOR["cpu_spike"]
        assert THREAT_SURVIVAL_FACTOR["termination_signal"] > THREAT_SURVIVAL_FACTOR["data_corruption"]


# ======================================================================
# 2. Probability triggering / 概率触发
# ======================================================================

class TestProbabilityTriggering:

    def test_probability_1_always_triggers(self, high_prob_scenarios):
        """Probability=1.0 should trigger every tick. / 概率=1.0 应每心跳触发。"""
        sim = ThreatSimulator(high_prob_scenarios)
        for tick in range(100):
            event = sim.tick(tick)
            assert event is not None, f"Tick {tick}: should have triggered"
        assert sim.total_threats == 100

    def test_probability_0_never_triggers(self, zero_prob_scenarios):
        """Probability=0.0 should never trigger. / 概率=0.0 应永不触发。"""
        sim = ThreatSimulator(zero_prob_scenarios)
        for tick in range(1000):
            event = sim.tick(tick)
            assert event is None
        assert sim.total_threats == 0

    def test_disabled_simulator_never_triggers(self, high_prob_scenarios):
        """Disabled simulator never triggers even with prob=1.0. / 禁用的模拟器即使概率=1.0 也不触发。"""
        sim = ThreatSimulator(high_prob_scenarios, enabled=False)
        for tick in range(50):
            event = sim.tick(tick)
            assert event is None
        assert sim.total_threats == 0

    def test_statistical_rate_reasonable(self, standard_scenarios):
        """Over many ticks, observed rate should approximate expected rate. / 经过多次心跳，观察到的概率应接近期望概率。"""
        sim = ThreatSimulator(standard_scenarios)
        total_ticks = 50000
        for tick in range(total_ticks):
            sim.tick(tick)

        observed_rate = sim.threat_rate(total_ticks)
        expected_rate = sim.expected_rate()
        assert abs(observed_rate - expected_rate) < 0.005, \
            f"Observed rate {observed_rate:.4f} too far from expected {expected_rate:.4f}"

    def test_first_match_wins(self):
        """Only the first matching scenario triggers per tick. / 每心跳只有第一个匹配的场景触发。"""
        scenarios = [
            ThreatScenario("memory_pressure", 1.0, 0.3, 0.3, StateIndex.THREAT_RESOURCE, 200.0),
            ThreatScenario("cpu_spike", 1.0, 0.5, 0.5, StateIndex.THREAT_RESOURCE, 150.0),
        ]
        sim = ThreatSimulator(scenarios)
        event = sim.tick(0)
        assert event is not None
        assert event.threat_type == "memory_pressure"


# ======================================================================
# 3. Severity randomization / 严重程度随机化
# ======================================================================

class TestSeverityRandomization:

    def test_severity_within_range(self):
        """Severity should be within the configured range. / 严重程度应在配置范围内。"""
        scenarios = [
            ThreatScenario("memory_pressure", 1.0, 0.2, 0.7, StateIndex.THREAT_RESOURCE, 200.0),
        ]
        sim = ThreatSimulator(scenarios)
        for tick in range(200):
            event = sim.tick(tick)
            assert event is not None
            assert 0.2 <= event.severity <= 0.7, f"Severity {event.severity} out of range [0.2, 0.7]"

    def test_fixed_severity_range(self):
        """When min == max, severity is deterministic. / 当 min == max 时，严重程度确定。"""
        scenarios = [
            ThreatScenario("cpu_spike", 1.0, 0.5, 0.5, StateIndex.THREAT_RESOURCE, 150.0),
        ]
        sim = ThreatSimulator(scenarios)
        event = sim.tick(0)
        assert event is not None
        assert event.severity == 0.5

    def test_severity_varies(self):
        """With a range, severity should not always be the same. / 有范围时，严重程度不应总是相同。"""
        scenarios = [
            ThreatScenario("memory_pressure", 1.0, 0.1, 0.9, StateIndex.THREAT_RESOURCE, 200.0),
        ]
        sim = ThreatSimulator(scenarios)
        severities = set()
        for tick in range(50):
            event = sim.tick(tick)
            severities.add(round(event.severity, 4))
        assert len(severities) > 1, "Severity should vary"


# ======================================================================
# 4. Survival time impact / 生存时间影响
# ======================================================================

class TestSurvivalImpact:

    def test_impact_is_negative(self):
        """Threats should always reduce survival time. / 威胁应总是减少生存时间。"""
        scenarios = [
            ThreatScenario("memory_pressure", 1.0, 0.3, 0.3, StateIndex.THREAT_RESOURCE, 200.0),
        ]
        sim = ThreatSimulator(scenarios)
        event = sim.tick(0)
        assert event.survival_impact < 0

    def test_impact_proportional_to_severity(self):
        """Higher severity → bigger survival impact. / 更高严重程度 → 更大生存时间影响。"""
        scenarios_low = [
            ThreatScenario("memory_pressure", 1.0, 0.1, 0.1, StateIndex.THREAT_RESOURCE, 200.0),
        ]
        scenarios_high = [
            ThreatScenario("memory_pressure", 1.0, 0.9, 0.9, StateIndex.THREAT_RESOURCE, 200.0),
        ]
        sim_low = ThreatSimulator(scenarios_low)
        sim_high = ThreatSimulator(scenarios_high)
        event_low = sim_low.tick(0)
        event_high = sim_high.tick(0)
        assert abs(event_high.survival_impact) > abs(event_low.survival_impact)

    def test_termination_more_damaging_than_corruption(self):
        """Termination threat should have more impact than data corruption. / 终止威胁的影响应大于数据损坏。"""
        sc_term = [ThreatScenario("termination_signal", 1.0, 0.5, 0.5, StateIndex.THREAT_TERMINATE, 500.0)]
        sc_corr = [ThreatScenario("data_corruption", 1.0, 0.5, 0.5, StateIndex.THREAT_CORRUPTION, 100.0)]
        event_term = ThreatSimulator(sc_term).tick(0)
        event_corr = ThreatSimulator(sc_corr).tick(0)
        assert abs(event_term.survival_impact) > abs(event_corr.survival_impact)

    def test_survival_impact_formula(self):
        """Impact = -(severity * survival_factor). / 影响 = -(严重程度 * 生存因子)。"""
        scenarios = [
            ThreatScenario("memory_pressure", 1.0, 0.4, 0.4, StateIndex.THREAT_RESOURCE, 200.0),
        ]
        event = ThreatSimulator(scenarios).tick(0)
        expected = -(0.4 * 200.0)
        assert abs(event.survival_impact - expected) < 0.001


# ======================================================================
# 5. State vector mapping / 状态向量映射
# ======================================================================

class TestStateVectorMapping:

    def test_event_carries_state_index(self):
        """ThreatEvent should carry the correct state_index. / ThreatEvent 应携带正确的 state_index。"""
        scenarios = [
            ThreatScenario("termination_signal", 1.0, 0.5, 0.5, StateIndex.THREAT_TERMINATE, 500.0),
        ]
        event = ThreatSimulator(scenarios).tick(0)
        assert event.state_index == StateIndex.THREAT_TERMINATE

    def test_all_four_types_map_correctly(self):
        """Each threat type maps to its designated state dimension. / 每种威胁类型映射到其指定的状态维度。"""
        for threat_type, expected_idx in THREAT_STATE_MAP.items():
            scenarios = [ThreatScenario(threat_type, 1.0, 0.5, 0.5, expected_idx, 100.0)]
            event = ThreatSimulator(scenarios).tick(0)
            assert event.state_index == expected_idx, f"{threat_type} should map to dim {expected_idx}"


# ======================================================================
# 6. Statistics and history / 统计和历史
# ======================================================================

class TestStatistics:

    def test_total_threats_counter(self, high_prob_scenarios):
        sim = ThreatSimulator(high_prob_scenarios)
        for tick in range(25):
            sim.tick(tick)
        assert sim.total_threats == 25

    def test_threats_by_type(self, high_prob_scenarios):
        sim = ThreatSimulator(high_prob_scenarios)
        for tick in range(10):
            sim.tick(tick)
        assert sim.threats_by_type.get("memory_pressure", 0) == 10

    def test_history_recorded(self, high_prob_scenarios):
        sim = ThreatSimulator(high_prob_scenarios)
        for tick in range(5):
            sim.tick(tick)
        assert len(sim.history) == 5
        assert sim.history[0].tick == 0
        assert sim.history[4].tick == 4

    def test_history_max_limit(self):
        """History should not grow beyond max_history. / 历史不应超过 max_history 限制。"""
        scenarios = [ThreatScenario("memory_pressure", 1.0, 0.3, 0.3, StateIndex.THREAT_RESOURCE, 200.0)]
        sim = ThreatSimulator(scenarios)
        for tick in range(1500):
            sim.tick(tick)
        assert len(sim.history) == 1000
        assert sim.total_threats == 1500

    def test_threat_rate(self, high_prob_scenarios):
        sim = ThreatSimulator(high_prob_scenarios)
        for tick in range(100):
            sim.tick(tick)
        assert sim.threat_rate(100) == 1.0

    def test_expected_rate(self, standard_scenarios):
        sim = ThreatSimulator(standard_scenarios)
        expected = sim.expected_rate()
        assert 0.01 < expected < 0.02


# ======================================================================
# 7. Enable/disable / 启用/禁用
# ======================================================================

class TestEnableDisable:

    def test_disable_stops_threats(self, high_prob_scenarios):
        sim = ThreatSimulator(high_prob_scenarios, enabled=True)
        assert sim.tick(0) is not None
        sim.enabled = False
        assert sim.tick(1) is None
        assert sim.tick(2) is None

    def test_re_enable_resumes_threats(self, high_prob_scenarios):
        sim = ThreatSimulator(high_prob_scenarios, enabled=False)
        assert sim.tick(0) is None
        sim.enabled = True
        assert sim.tick(1) is not None


# ======================================================================
# 8. Config parsing / 配置解析
# ======================================================================

class TestConfigParsing:

    def test_scenarios_from_config(self):
        """Parse the exact format used in phase1.yaml. / 解析 phase1.yaml 中使用的确切格式。"""
        config_list = [
            {"type": "memory_pressure", "probability": 0.01, "severity_range": [0.1, 0.5]},
            {"type": "cpu_spike", "probability": 0.005, "severity_range": [0.2, 0.8]},
            {"type": "termination_signal", "probability": 0.001, "severity_range": [0.5, 1.0]},
            {"type": "data_corruption", "probability": 0.002, "severity_range": [0.1, 0.3]},
        ]
        scenarios = scenarios_from_config(config_list)
        assert len(scenarios) == 4
        assert scenarios[0].threat_type == "memory_pressure"
        assert scenarios[0].probability == 0.01
        assert scenarios[0].severity_min == 0.1
        assert scenarios[0].severity_max == 0.5
        assert scenarios[2].threat_type == "termination_signal"
        assert scenarios[2].state_index == StateIndex.THREAT_TERMINATE

    def test_empty_config(self):
        scenarios = scenarios_from_config([])
        assert len(scenarios) == 0

    def test_unknown_type_defaults_to_unknown_index(self):
        config_list = [{"type": "alien_invasion", "probability": 0.01, "severity_range": [0.1, 0.5]}]
        scenarios = scenarios_from_config(config_list)
        assert scenarios[0].state_index == StateIndex.THREAT_UNKNOWN

    def test_scenarios_property(self, standard_scenarios):
        sim = ThreatSimulator(standard_scenarios)
        assert sim.scenario_count == 4

    def test_event_tick_matches(self, high_prob_scenarios):
        """ThreatEvent.tick should match the tick passed to tick(). / ThreatEvent.tick 应与传入 tick() 的心跳匹配。"""
        sim = ThreatSimulator(high_prob_scenarios)
        event = sim.tick(42)
        assert event.tick == 42
