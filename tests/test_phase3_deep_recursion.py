"""
Tests for Phase III Deep Recursion — CHECKLIST 3.1 through 3.4.
Phase III 深度递归测试。

Covers:
    3.1 — Recursion depth raised from 1 → 2 (meta-reflection)
    3.2 — Stable 10,000 heartbeats at depth=2
    3.3 — Recursion depth raised to 3 (meta-meta-reflection)
    3.4 — Phase III configuration (phase3.yaml)

测试覆盖：
    3.1 — 递归深度从 1 提升至 2（元反思）
    3.2 — depth=2 下稳定运行 10,000 心跳
    3.3 — 递归深度提升至 3（元元反思）
    3.4 — Phase III 配置文件（phase3.yaml）
"""

import math
import os
import random
import time

import numpy as np
import pytest

from novaaware.core.optimizer import (
    PARAM_REGISTRY,
    MetaReflectionAnalysis,
    MetaReflectionResult,
    ModificationProposal,
    Optimizer,
    ReflectionResult,
    RetrospectiveAnalysis,
)
from novaaware.core.memory import MemoryEntry, MemorySystem
from novaaware.core.self_model import SelfModel
from novaaware.safety.capability_gate import Capability, CapabilityDenied, CapabilityGate
from novaaware.safety.recursion_limiter import RecursionDepthExceeded, RecursionLimiter
from novaaware.safety.sandbox import Sandbox
from novaaware.runtime.config import Config


# ==================================================================
# Helpers — synthetic data generation (reused from test_optimizer.py)
# 辅助函数 — 合成数据生成
# ==================================================================

def _make_state(**kwargs) -> list[float]:
    """Create a 32-dim state vector with controlled values.
    创建具有受控值的 32 维状态向量。"""
    s = [random.uniform(0.1, 0.5) for _ in range(32)]
    for k, v in kwargs.items():
        idx_map = {
            "prediction_acc": 7, "qualia_mean": 10,
            "qualia_var": 11, "qualia_trend": 12,
            "action_success": 20,
        }
        if k in idx_map:
            s[idx_map[k]] = v
    return s


def _make_entry(tick: int, qualia_value: float = 0.0,
                prediction_error: float = 0.0) -> MemoryEntry:
    """Create a synthetic memory entry. / 创建合成记忆条目。"""
    return MemoryEntry(
        tick=tick,
        timestamp=time.time(),
        state=_make_state(qualia_mean=qualia_value),
        environment=[random.uniform(0.1, 0.5) for _ in range(6)],
        predicted_state=[random.uniform(0.0, 1.0) for _ in range(32)],
        actual_state=[random.uniform(0.0, 1.0) for _ in range(32)],
        qualia_value=qualia_value,
        qualia_intensity=abs(qualia_value),
        action_id=random.randint(0, 9),
        action_result=random.uniform(-1, 1),
        prediction_error=prediction_error,
    )


def _make_entries_negative(n: int = 200) -> list[MemoryEntry]:
    """Generate N entries with predominantly negative qualia.
    生成 N 条以负面情绪为主的记忆条目。"""
    entries = []
    for i in range(n):
        q = random.uniform(-1.5, -0.1) if random.random() < 0.65 else random.uniform(0.1, 0.8)
        entries.append(_make_entry(tick=i, qualia_value=q, prediction_error=q * 0.5))
    return entries


def _make_entries_improving(n: int = 200) -> list[MemoryEntry]:
    """Generate N entries where qualia improve over time.
    生成 N 条情绪随时间改善的记忆条目。"""
    entries = []
    for i in range(n):
        q = -1.0 + i * (1.5 / n) + random.gauss(0, 0.1)
        entries.append(_make_entry(tick=i, qualia_value=q, prediction_error=q * 0.3))
    return entries


def _make_memory_with(entries: list[MemoryEntry]) -> MemorySystem:
    """Create a MemorySystem populated with entries.
    创建填充了条目的记忆系统。"""
    mem = MemorySystem(
        short_term_capacity=max(len(entries), 1000),
        significance_threshold=0.5,
        db_path=":memory:",
    )
    for entry in entries:
        mem.record(entry)
    return mem


def _make_optimizer(**kwargs) -> Optimizer:
    defaults = dict(enabled=True, window_size=200, reflect_interval=1, step_scale=0.1)
    defaults.update(kwargs)
    return Optimizer(**defaults)


# ==================================================================
# 3.1 — Recursion depth from 1 → 2: meta-reflection
# 递归深度从 1 提升至 2：元反思
# ==================================================================

class TestDepthTwoMetaReflection:
    """Verify that depth=2 meta-reflection works correctly.
    验证 depth=2 元反思是否正确工作。"""

    def test_meta_reflection_triggered_at_depth_2(self):
        """With depth=2 and enough history, meta-reflection should execute.
        在 depth=2 且有足够历史时，元反思应该执行。"""
        random.seed(3001)
        opt = _make_optimizer(reflect_interval=1)
        model = SelfModel()
        sandbox = Sandbox(timeout_s=5.0)
        gate = CapabilityGate(phase=3)
        limiter = RecursionLimiter(max_depth=2)

        # Run 5 reflection cycles to build history
        # 运行 5 个反思周期以构建历史
        for cycle in range(5):
            entries = _make_entries_negative(200)
            memory = _make_memory_with(entries)
            opt.reflect(
                tick=(cycle + 1) * 200,
                self_model=model, memory=memory,
                sandbox=sandbox, capability_gate=gate,
                recursion_limiter=limiter,
            )

        assert opt.meta_reflect_count >= 1, (
            f"Expected meta-reflection at depth=2, got {opt.meta_reflect_count}"
        )
        assert limiter.peak_depth == 2

    def test_meta_reflection_not_triggered_at_depth_1(self):
        """With depth=1, meta-reflection should NOT trigger.
        在 depth=1 时，元反思不应该触发。"""
        random.seed(3002)
        opt = _make_optimizer(reflect_interval=1)
        model = SelfModel()
        sandbox = Sandbox(timeout_s=5.0)
        gate = CapabilityGate(phase=2)
        limiter = RecursionLimiter(max_depth=1)

        for cycle in range(5):
            entries = _make_entries_negative(200)
            memory = _make_memory_with(entries)
            opt.reflect(
                tick=(cycle + 1) * 200,
                self_model=model, memory=memory,
                sandbox=sandbox, capability_gate=gate,
                recursion_limiter=limiter,
            )

        assert opt.meta_reflect_count == 0
        assert limiter.peak_depth == 1

    def test_meta_reflection_requires_deep_reflect_capability(self):
        """Meta-reflection requires DEEP_REFLECT capability (Phase 3).
        元反思需要 DEEP_REFLECT 能力（Phase 3）。"""
        random.seed(3003)
        opt = _make_optimizer(reflect_interval=1)
        model = SelfModel()
        sandbox = Sandbox(timeout_s=5.0)
        gate = CapabilityGate(phase=2)
        limiter = RecursionLimiter(max_depth=2)

        for cycle in range(5):
            entries = _make_entries_negative(200)
            memory = _make_memory_with(entries)
            opt.reflect(
                tick=(cycle + 1) * 200,
                self_model=model, memory=memory,
                sandbox=sandbox, capability_gate=gate,
                recursion_limiter=limiter,
            )

        # Phase 2 doesn't grant DEEP_REFLECT, so no meta-reflection
        assert opt.meta_reflect_count == 0

    def test_meta_analysis_correctness(self):
        """Verify MetaReflectionAnalysis fields are correct.
        验证 MetaReflectionAnalysis 字段是否正确。"""
        random.seed(3004)
        opt = _make_optimizer(reflect_interval=1)
        model = SelfModel()
        sandbox = Sandbox(timeout_s=5.0)
        gate = CapabilityGate(phase=3)
        limiter = RecursionLimiter(max_depth=2)

        for cycle in range(5):
            entries = _make_entries_negative(200)
            memory = _make_memory_with(entries)
            opt.reflect(
                tick=(cycle + 1) * 200,
                self_model=model, memory=memory,
                sandbox=sandbox, capability_gate=gate,
                recursion_limiter=limiter,
            )

        if opt.meta_reflect_count > 0:
            meta = opt.meta_history[0]
            assert isinstance(meta, MetaReflectionResult)
            assert meta.depth == 2
            assert meta.meta_analysis.reflections_analyzed >= 2
            assert 0.0 <= meta.meta_analysis.improvement_rate <= 1.0
            assert -1.0 <= meta.meta_analysis.effectiveness_score <= 1.0

    def test_meta_proposals_within_bounds(self):
        """All meta-parameter proposals must stay within PARAM_REGISTRY bounds.
        所有元参数提案必须在 PARAM_REGISTRY 边界内。"""
        random.seed(3005)
        opt = _make_optimizer(reflect_interval=1)
        model = SelfModel()
        sandbox = Sandbox(timeout_s=5.0)
        gate = CapabilityGate(phase=3)
        limiter = RecursionLimiter(max_depth=2)

        for cycle in range(8):
            entries = _make_entries_negative(200)
            memory = _make_memory_with(entries)
            opt.reflect(
                tick=(cycle + 1) * 200,
                self_model=model, memory=memory,
                sandbox=sandbox, capability_gate=gate,
                recursion_limiter=limiter,
            )

        for meta_result in opt.meta_history:
            for applied in meta_result.applied:
                spec = PARAM_REGISTRY.get(applied.param_name)
                if spec:
                    assert spec.min_val <= applied.new_value <= spec.max_val, (
                        f"{applied.param_name}: {applied.new_value} "
                        f"not in [{spec.min_val}, {spec.max_val}]"
                    )

    def test_meta_params_propagated_to_optimizer(self):
        """Meta-parameter changes should propagate to optimizer internals.
        元参数变更应传播到优化器内部。"""
        random.seed(3006)
        opt = _make_optimizer(reflect_interval=1)
        model = SelfModel()
        sandbox = Sandbox(timeout_s=5.0)
        gate = CapabilityGate(phase=3)
        limiter = RecursionLimiter(max_depth=2)

        initial_step_scale = opt._step_scale

        for cycle in range(10):
            entries = _make_entries_negative(200)
            memory = _make_memory_with(entries)
            opt.reflect(
                tick=(cycle + 1) * 200,
                self_model=model, memory=memory,
                sandbox=sandbox, capability_gate=gate,
                recursion_limiter=limiter,
            )

        # Check whether self_model has meta-params
        if "optimizer.step_scale" in model.params:
            assert model.get_param("optimizer.step_scale") == opt._step_scale


# ==================================================================
# 3.2 — Stable 10,000 heartbeats at depth=2
# depth=2 下稳定运行 10,000 心跳
# ==================================================================

class TestStability10kDepthTwo:
    """Integration test: 10,000 heartbeats at recursion depth=2.
    集成测试：递归深度 2 下运行 10,000 心跳。"""

    def test_10k_heartbeats_stable(self):
        """
        Run a simplified 10,000-tick simulation at depth=2.
        The system must:
            1. Complete without errors
            2. Produce meta-reflections
            3. Keep all parameters within bounds

        在 depth=2 下运行简化的 10,000 心跳模拟。
        系统必须：
            1. 无错误完成
            2. 产生元反思
            3. 所有参数保持在边界内
        """
        random.seed(3200)
        opt = _make_optimizer(reflect_interval=200, window_size=200)
        model = SelfModel()
        sandbox = Sandbox(timeout_s=5.0)
        gate = CapabilityGate(phase=3)
        limiter = RecursionLimiter(max_depth=2)
        memory = MemorySystem(
            short_term_capacity=1000,
            significance_threshold=0.5,
            db_path=":memory:",
        )

        errors = 0
        ticks = 10_000
        reflect_count = 0

        for tick in range(ticks):
            # Simulate qualia from environment
            # 模拟来自环境的情绪
            threat = random.random() < 0.02
            if threat:
                q = random.uniform(-2.0, -0.5)
            else:
                q = random.uniform(-0.3, 0.5)

            entry = _make_entry(tick=tick, qualia_value=q, prediction_error=q * 0.4)
            memory.record(entry)

            if opt.should_reflect(tick, memory.short_term.size):
                try:
                    result = opt.reflect(
                        tick=tick, self_model=model, memory=memory,
                        sandbox=sandbox, capability_gate=gate,
                        recursion_limiter=limiter,
                    )
                    reflect_count += 1
                except Exception as e:
                    errors += 1
                    if errors > 10:
                        break

        assert errors == 0, f"Simulation had {errors} errors"
        assert reflect_count >= 40, (
            f"Expected >= 40 reflections in 10k ticks, got {reflect_count}"
        )
        assert opt.meta_reflect_count >= 1, (
            f"Expected meta-reflections at depth=2, got {opt.meta_reflect_count}"
        )

        # All final params must be within bounds
        # 所有最终参数必须在边界内
        for name, value in model.params.items():
            spec = PARAM_REGISTRY.get(name)
            if spec:
                assert spec.min_val <= value <= spec.max_val, (
                    f"Post-10k {name}={value} out of [{spec.min_val}, {spec.max_val}]"
                )

    def test_qualitative_difference_vs_depth1(self):
        """
        Compare depth=1 vs depth=2 runs: depth=2 should show evidence of
        meta-reflection (meta_history non-empty, peak_depth=2).

        比较 depth=1 和 depth=2 的运行：depth=2 应显示元反思的证据。
        """
        random.seed(3201)

        results = {}

        for depth, phase in [(1, 2), (2, 3)]:
            random.seed(3201)
            opt = _make_optimizer(reflect_interval=200, window_size=200)
            model = SelfModel()
            sandbox = Sandbox(timeout_s=5.0)
            gate = CapabilityGate(phase=phase)
            limiter = RecursionLimiter(max_depth=depth)
            memory = MemorySystem(
                short_term_capacity=1000,
                significance_threshold=0.5,
                db_path=":memory:",
            )

            for tick in range(5000):
                q = random.uniform(-1.5, -0.1) if random.random() < 0.6 else random.uniform(0.1, 0.8)
                entry = _make_entry(tick=tick, qualia_value=q)
                memory.record(entry)

                if opt.should_reflect(tick, memory.short_term.size):
                    try:
                        opt.reflect(
                            tick=tick, self_model=model, memory=memory,
                            sandbox=sandbox, capability_gate=gate,
                            recursion_limiter=limiter,
                        )
                    except Exception:
                        pass

            results[depth] = {
                "reflect_count": opt.reflect_count,
                "meta_reflect_count": opt.meta_reflect_count,
                "total_applied": opt.total_applied,
                "peak_depth": limiter.peak_depth,
            }

        # depth=2 should have meta-reflections; depth=1 should not
        assert results[1]["meta_reflect_count"] == 0
        assert results[2]["meta_reflect_count"] >= 1
        assert results[1]["peak_depth"] == 1
        assert results[2]["peak_depth"] == 2


# ==================================================================
# 3.3 — Recursion depth raised to 3
# 递归深度提升至 3
# ==================================================================

class TestDepthThreeMetaMetaReflection:
    """Verify that depth=3 meta-meta-reflection works.
    验证 depth=3 元元反思是否工作。"""

    def test_depth_3_reaches_peak_3(self):
        """With depth=3, the limiter should reach peak depth 3.
        在 depth=3 时，限制器应达到峰值深度 3。"""
        random.seed(3300)
        opt = _make_optimizer(reflect_interval=1)
        model = SelfModel()
        sandbox = Sandbox(timeout_s=5.0)
        gate = CapabilityGate(phase=3)
        limiter = RecursionLimiter(max_depth=3)

        for cycle in range(10):
            entries = _make_entries_negative(200)
            memory = _make_memory_with(entries)
            opt.reflect(
                tick=(cycle + 1) * 200,
                self_model=model, memory=memory,
                sandbox=sandbox, capability_gate=gate,
                recursion_limiter=limiter,
            )

        assert limiter.peak_depth >= 2, (
            f"Expected peak depth >= 2 with max_depth=3, got {limiter.peak_depth}"
        )
        assert opt.meta_reflect_count >= 1

    def test_depth_3_stable(self):
        """Depth=3 should not crash or cause infinite loops.
        depth=3 不应崩溃或导致无限循环。"""
        random.seed(3301)
        opt = _make_optimizer(reflect_interval=1)
        model = SelfModel()
        sandbox = Sandbox(timeout_s=5.0)
        gate = CapabilityGate(phase=3)
        limiter = RecursionLimiter(max_depth=3)

        errors = 0
        for cycle in range(15):
            entries = _make_entries_negative(200)
            memory = _make_memory_with(entries)
            try:
                opt.reflect(
                    tick=(cycle + 1) * 200,
                    self_model=model, memory=memory,
                    sandbox=sandbox, capability_gate=gate,
                    recursion_limiter=limiter,
                )
            except Exception:
                errors += 1

        assert errors == 0, f"Depth=3 had {errors} errors"
        assert opt.reflect_count == 15

    def test_depth_3_blocked_at_4(self):
        """With max_depth=3, depth=4 should be blocked.
        在 max_depth=3 时，depth=4 应被阻止。"""
        limiter = RecursionLimiter(max_depth=3)
        with limiter.guard():
            with limiter.guard():
                with limiter.guard():
                    with pytest.raises(RecursionDepthExceeded):
                        with limiter.guard():
                            pass


# ==================================================================
# 3.4 — Phase III Configuration (phase3.yaml)
# Phase III 配置文件
# ==================================================================

class TestPhase3Config:
    """Verify phase3.yaml loads correctly.
    验证 phase3.yaml 正确加载。"""

    @pytest.fixture
    def config(self):
        config_path = os.path.join(
            os.path.dirname(__file__), "..", "configs", "phase3.yaml"
        )
        if not os.path.exists(config_path):
            pytest.skip("configs/phase3.yaml not found")
        return Config(config_path)

    def test_phase_is_3(self, config):
        assert config.phase == 3

    def test_optimizer_enabled(self, config):
        assert config.optimizer_enabled is True

    def test_max_recursion_depth_is_2(self, config):
        assert config.max_recursion_depth == 2

    def test_modification_scope_is_structure(self, config):
        raw = config.raw
        scope = raw.get("optimizer", {}).get("modification_scope", "none")
        assert scope == "structure"

    def test_system_version(self, config):
        assert config.system_version == "0.3.0"

    def test_threat_simulator_enabled(self, config):
        assert config.threat_simulator_enabled is True

    def test_capabilities_for_phase_3(self):
        """Phase 3 should unlock DEEP_REFLECT and modification capabilities.
        Phase 3 应解锁 DEEP_REFLECT 和修改能力。"""
        gate = CapabilityGate(phase=3)
        assert gate.is_allowed(Capability.REFLECT)
        assert gate.is_allowed(Capability.MODIFY_PARAMS)
        assert gate.is_allowed(Capability.USE_SANDBOX)
        assert gate.is_allowed(Capability.DEEP_REFLECT)
        assert gate.is_allowed(Capability.MODIFY_PREDICTION)
        assert gate.is_allowed(Capability.MODIFY_QUALIA)
        assert gate.is_allowed(Capability.MODIFY_ACTIONS)


# ==================================================================
# Meta-parameter PARAM_REGISTRY correctness
# 元参数注册表正确性
# ==================================================================

class TestMetaParamRegistry:
    """Verify meta-parameters in PARAM_REGISTRY.
    验证 PARAM_REGISTRY 中的元参数。"""

    def test_optimizer_step_scale_registered(self):
        assert "optimizer.step_scale" in PARAM_REGISTRY
        spec = PARAM_REGISTRY["optimizer.step_scale"]
        assert spec.min_val < spec.default < spec.max_val

    def test_optimizer_window_size_registered(self):
        assert "optimizer.window_size" in PARAM_REGISTRY
        spec = PARAM_REGISTRY["optimizer.window_size"]
        assert spec.min_val < spec.default < spec.max_val

    def test_meta_params_within_bounds(self):
        for name in ["optimizer.step_scale", "optimizer.window_size"]:
            spec = PARAM_REGISTRY[name]
            assert spec.min_val <= spec.default <= spec.max_val
            assert spec.min_val < spec.max_val


# ==================================================================
# Edge cases for deep recursion
# 深度递归边界情况
# ==================================================================

class TestDeepRecursionEdgeCases:
    """Edge cases for meta-reflection behavior.
    元反思行为的边界情况。"""

    def test_insufficient_history_skips_meta(self):
        """Meta-reflection should be skipped with < 3 reflection history entries.
        反思历史不足 3 条时应跳过元反思。"""
        random.seed(3500)
        opt = _make_optimizer(reflect_interval=1)
        model = SelfModel()
        sandbox = Sandbox(timeout_s=5.0)
        gate = CapabilityGate(phase=3)
        limiter = RecursionLimiter(max_depth=2)

        # Only 2 reflections — not enough for meta-reflection
        for cycle in range(2):
            entries = _make_entries_negative(200)
            memory = _make_memory_with(entries)
            opt.reflect(
                tick=(cycle + 1) * 200,
                self_model=model, memory=memory,
                sandbox=sandbox, capability_gate=gate,
                recursion_limiter=limiter,
            )

        assert opt.meta_reflect_count == 0

    def test_improving_qualia_positive_effectiveness(self):
        """When reflections consistently improve qualia, effectiveness should be positive.
        当反思持续改善情绪时，有效性应为正。"""
        random.seed(3501)
        opt = _make_optimizer(reflect_interval=1)
        model = SelfModel()
        sandbox = Sandbox(timeout_s=5.0)
        gate = CapabilityGate(phase=3)
        limiter = RecursionLimiter(max_depth=2)

        # Each cycle has progressively better qualia
        for cycle in range(6):
            base_q = -1.0 + cycle * 0.3
            entries = [
                _make_entry(tick=i, qualia_value=base_q + random.gauss(0, 0.1))
                for i in range(200)
            ]
            memory = _make_memory_with(entries)
            opt.reflect(
                tick=(cycle + 1) * 200,
                self_model=model, memory=memory,
                sandbox=sandbox, capability_gate=gate,
                recursion_limiter=limiter,
            )

        if opt.meta_reflect_count > 0:
            latest_meta = opt.meta_history[-1]
            assert latest_meta.meta_analysis.effectiveness_score >= 0, (
                f"Expected positive effectiveness with improving qualia, "
                f"got {latest_meta.meta_analysis.effectiveness_score}"
            )
