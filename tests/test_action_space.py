"""
Unit tests for ActionSpace — the system's "hands and feet".
行动空间的单元测试 —— 系统的"手脚"。

Tests cover:
  - 10 actions are defined
  - Normal mode selection heuristics
  - Emergency mode selection
  - Exploration rate
  - Action execution and metrics
  - Action distribution (behavioral diversity)
"""

import numpy as np
import pytest

from novaaware.environment.action_space import ActionID, ActionResult, ActionSpace
from novaaware.core.self_model import StateIndex


# ======================================================================
# 1. Action definition / 动作定义
# ======================================================================

class TestActionIDDefinitions:
    """Verify that all 10 actions are properly defined. / 验证 10 种动作的正确定义。"""

    def test_total_action_count(self):
        """There should be exactly 10 actions. / 应该恰好有 10 种动作。"""
        assert ActionID.count() == 10

    def test_action_values_are_sequential(self):
        """Action IDs should be 0-9 without gaps. / 动作 ID 应该是 0-9 无间隔。"""
        values = sorted(a.value for a in ActionID)
        assert values == list(range(10))

    def test_normal_actions_subset(self):
        """Normal actions should be a subset of all actions. / 正常动作应该是所有动作的子集。"""
        normal = ActionID.normal_actions()
        assert len(normal) == 8
        for a in normal:
            assert a in ActionID

    def test_emergency_actions_subset(self):
        """Emergency actions should include the two dedicated emergency actions. / 紧急动作应包含两个专用紧急动作。"""
        emergency = ActionID.emergency_actions()
        assert ActionID.EMERGENCY_CONSERVE in emergency
        assert ActionID.EMERGENCY_RELEASE in emergency
        assert len(emergency) == 4

    def test_emergency_actions_include_basic_survival(self):
        """Emergency mode also includes basic survival actions. / 紧急模式也包含基本生存动作。"""
        emergency = ActionID.emergency_actions()
        assert ActionID.REDUCE_LOAD in emergency
        assert ActionID.RELEASE_MEMORY in emergency


# ======================================================================
# 2. Normal mode selection / 正常模式选择
# ======================================================================

class TestNormalModeSelection:
    """Test heuristic-based selection in normal mode. / 测试正常模式下的启发式选择。"""

    @pytest.fixture
    def space(self):
        return ActionSpace(exploration_rate=0.0)

    def _make_state(self, **overrides) -> np.ndarray:
        """Build a 32-dim state vector with sensible defaults. / 构建具有默认值的 32 维状态向量。"""
        state = np.zeros(StateIndex.DIM)
        for key, val in overrides.items():
            idx = getattr(StateIndex, key)
            state[idx] = val
        return state

    def test_all_low_returns_noop(self, space):
        """When all resources are fine, select NO_OP. / 所有资源正常时选择无操作。"""
        state = self._make_state(CPU_USAGE=0.1, MEMORY_USAGE=0.1, PREDICTION_ACC=0.8)
        action = space.select_action(state, is_emergency=False)
        assert action == ActionID.NO_OP.value

    def test_high_memory_returns_release(self, space):
        """High memory pressure → RELEASE_MEMORY. / 高内存压力 → 释放内存。"""
        state = self._make_state(MEMORY_USAGE=0.85, CPU_USAGE=0.1)
        action = space.select_action(state, is_emergency=False)
        assert action == ActionID.RELEASE_MEMORY.value

    def test_high_cpu_returns_reduce_load(self, space):
        """High CPU → REDUCE_LOAD. / 高 CPU → 降低负载。"""
        state = self._make_state(CPU_USAGE=0.85, MEMORY_USAGE=0.1)
        action = space.select_action(state, is_emergency=False)
        assert action == ActionID.REDUCE_LOAD.value

    def test_low_pred_acc_returns_boost(self, space):
        """Low prediction accuracy → BOOST_PREDICTION. / 预测精度低 → 增强预测。"""
        state = self._make_state(PREDICTION_ACC=0.1, CPU_USAGE=0.1, MEMORY_USAGE=0.1)
        action = space.select_action(state, is_emergency=False)
        assert action == ActionID.BOOST_PREDICTION.value

    def test_moderate_pressure_returns_conserve(self, space):
        """Moderate resource usage → CONSERVE_RESOURCES. / 中等资源占用 → 节能。"""
        state = self._make_state(CPU_USAGE=0.55, MEMORY_USAGE=0.2, PREDICTION_ACC=0.8)
        action = space.select_action(state, is_emergency=False)
        assert action == ActionID.CONSERVE_RESOURCES.value

    def test_memory_priority_over_cpu(self, space):
        """Memory pressure is checked before CPU. / 内存压力优先于 CPU 检查。"""
        state = self._make_state(CPU_USAGE=0.9, MEMORY_USAGE=0.9)
        action = space.select_action(state, is_emergency=False)
        assert action == ActionID.RELEASE_MEMORY.value


# ======================================================================
# 3. Emergency mode selection / 紧急模式选择
# ======================================================================

class TestEmergencyModeSelection:
    """Test action selection in emergency mode. / 测试紧急模式下的动作选择。"""

    @pytest.fixture
    def space(self):
        return ActionSpace(exploration_rate=0.0)

    def _make_state(self, **overrides) -> np.ndarray:
        state = np.zeros(StateIndex.DIM)
        for key, val in overrides.items():
            idx = getattr(StateIndex, key)
            state[idx] = val
        return state

    def test_emergency_default_is_conserve(self, space):
        """Default emergency action is EMERGENCY_CONSERVE. / 默认紧急动作是紧急节能。"""
        state = self._make_state(CPU_USAGE=0.5, MEMORY_USAGE=0.3)
        action = space.select_action(state, is_emergency=True)
        assert action == ActionID.EMERGENCY_CONSERVE.value

    def test_emergency_high_mem_is_release(self, space):
        """High memory in emergency → EMERGENCY_RELEASE. / 紧急高内存 → 紧急释放。"""
        state = self._make_state(MEMORY_USAGE=0.9)
        action = space.select_action(state, is_emergency=True)
        assert action == ActionID.EMERGENCY_RELEASE.value

    def test_emergency_action_is_from_emergency_set(self, space):
        """Selected emergency action must be in the emergency set. / 紧急动作必须在紧急集合中。"""
        emergency_set = {a.value for a in ActionID.emergency_actions()}
        for _ in range(50):
            state = np.random.rand(StateIndex.DIM)
            action = space.select_action(state, is_emergency=True)
            assert action in emergency_set, f"Action {action} not in emergency set"


# ======================================================================
# 4. Exploration / 探索
# ======================================================================

class TestExploration:
    """Test exploration behavior. / 测试探索行为。"""

    def test_high_exploration_rate(self):
        """With rate=1.0, always explore. / 探索率=1.0 时总是探索。"""
        space = ActionSpace(exploration_rate=1.0)
        state = np.zeros(StateIndex.DIM)
        for _ in range(20):
            action = space.select_action(state, is_emergency=False)
            assert action == ActionID.EXPLORE.value

    def test_zero_exploration_never_explores(self):
        """With rate=0.0, never randomly explore. / 探索率=0.0 时永不随机探索。"""
        space = ActionSpace(exploration_rate=0.0)
        state = np.zeros(StateIndex.DIM)
        state[StateIndex.PREDICTION_ACC] = 0.8
        for _ in range(50):
            action = space.select_action(state, is_emergency=False)
            assert action != ActionID.EXPLORE.value

    def test_exploration_not_in_emergency(self):
        """Emergency mode never uses exploration. / 紧急模式从不使用探索。"""
        space = ActionSpace(exploration_rate=1.0)
        state = np.zeros(StateIndex.DIM)
        for _ in range(20):
            action = space.select_action(state, is_emergency=True)
            assert action != ActionID.EXPLORE.value


# ======================================================================
# 5. Action execution / 动作执行
# ======================================================================

class TestActionExecution:
    """Test action execution and result tracking. / 测试动作执行和结果追踪。"""

    @pytest.fixture
    def space(self):
        return ActionSpace()

    def test_execute_returns_action_result(self, space):
        """Execute should return an ActionResult. / 执行应返回 ActionResult。"""
        result = space.execute(ActionID.NO_OP.value, is_emergency=False)
        assert isinstance(result, ActionResult)
        assert result.action_id == ActionID.NO_OP.value
        assert not result.was_emergency

    def test_noop_has_zero_effect(self, space):
        """NO_OP should have exactly zero effect. / 无操作应该效果为零。"""
        result = space.execute(ActionID.NO_OP.value, is_emergency=False)
        assert result.effect == 0.0
        assert result.success is True

    def test_emergency_flag_is_tracked(self, space):
        """Emergency flag is propagated to result. / 紧急标志传播到结果中。"""
        result = space.execute(ActionID.EMERGENCY_CONSERVE.value, is_emergency=True)
        assert result.was_emergency is True

    def test_emergency_conserve_positive_effect(self, space):
        """Emergency conserve should always have positive effect. / 紧急节能应总是正效果。"""
        for _ in range(30):
            result = space.execute(ActionID.EMERGENCY_CONSERVE.value, is_emergency=True)
            assert result.effect > 0, "Emergency conserve should always help"
            assert result.success is True

    def test_emergency_release_positive_effect(self, space):
        """Emergency release should always have positive effect. / 紧急释放应总是正效果。"""
        for _ in range(30):
            result = space.execute(ActionID.EMERGENCY_RELEASE.value, is_emergency=True)
            assert result.effect > 0, "Emergency release should always help"
            assert result.success is True


# ======================================================================
# 6. Metrics / 指标
# ======================================================================

class TestMetrics:
    """Test action tracking metrics. / 测试动作追踪指标。"""

    @pytest.fixture
    def space(self):
        return ActionSpace()

    def test_total_actions_starts_at_zero(self, space):
        assert space.total_actions == 0

    def test_total_actions_increments(self, space):
        space.execute(ActionID.NO_OP.value, is_emergency=False)
        space.execute(ActionID.REDUCE_LOAD.value, is_emergency=False)
        assert space.total_actions == 2

    def test_success_rate_after_noop(self, space):
        """NO_OP always succeeds (effect = 0). / 无操作总是成功（效果=0）。"""
        space.execute(ActionID.NO_OP.value, is_emergency=False)
        assert space.action_success_rate == 1.0

    def test_exploration_ratio(self):
        """Track how often EXPLORE was executed. / 追踪探索动作的执行频率。"""
        space = ActionSpace()
        space.execute(ActionID.EXPLORE.value, is_emergency=False)
        space.execute(ActionID.NO_OP.value, is_emergency=False)
        assert space.exploration_ratio == 0.5

    def test_action_distribution_keys(self, space):
        """Distribution should cover all 10 actions. / 分布应覆盖所有 10 种动作。"""
        dist = space.action_distribution()
        assert len(dist) == 10
        for a in ActionID:
            assert a.value in dist

    def test_action_distribution_counts(self, space):
        """Counts should reflect execution. / 计数应反映执行情况。"""
        space.execute(ActionID.NO_OP.value, is_emergency=False)
        space.execute(ActionID.NO_OP.value, is_emergency=False)
        space.execute(ActionID.REDUCE_LOAD.value, is_emergency=False)
        dist = space.action_distribution()
        assert dist[ActionID.NO_OP.value] == 2
        assert dist[ActionID.REDUCE_LOAD.value] == 1

    def test_exploration_rate_property(self, space):
        assert space.exploration_rate == 0.1
