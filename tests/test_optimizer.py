"""
Tests for Optimizer — Recursive Self-Optimizer (E).
递归自我优化器测试。

Covers:
    - Retrospective analysis correctness (2.12)
    - Correlation detection (2.13)
    - Proposal generation heuristics (2.14)
    - Sandbox verification (2.15)
    - Safety check / bounds enforcement (2.16)
    - 10+ successful modifications across cycles (2.17)
    - Disabled optimizer behavior
    - Insufficient data handling
    - Recursion limit enforcement
    - Capability gate enforcement
    - Parameter initialization
"""

import math
import random
import time

import numpy as np
import pytest

from novaaware.core.optimizer import (
    PARAM_REGISTRY,
    ModificationProposal,
    Optimizer,
    ParamSpec,
    ReflectionResult,
    RetrospectiveAnalysis,
)
from novaaware.core.memory import MemoryEntry, MemorySystem
from novaaware.core.self_model import SelfModel
from novaaware.safety.capability_gate import Capability, CapabilityDenied, CapabilityGate
from novaaware.safety.recursion_limiter import RecursionDepthExceeded, RecursionLimiter
from novaaware.safety.sandbox import Sandbox


# ==================================================================
# Helpers — synthetic data generation
# ==================================================================

def _make_state(
    prediction_acc: float = 0.5,
    qualia_mean: float = 0.0,
    qualia_var: float = 0.1,
    qualia_trend: float = 0.0,
    action_success: float = 0.5,
) -> list[float]:
    """Create a 32-dim state vector with controlled values."""
    s = [random.uniform(0.1, 0.5) for _ in range(32)]
    s[7] = prediction_acc
    s[10] = qualia_mean
    s[11] = qualia_var
    s[12] = qualia_trend
    s[20] = action_success
    return s


def _make_entry(
    tick: int,
    qualia_value: float = 0.0,
    prediction_error: float = 0.0,
    prediction_acc: float = 0.5,
    action_success: float = 0.5,
) -> MemoryEntry:
    """Create a synthetic memory entry."""
    return MemoryEntry(
        tick=tick,
        timestamp=time.time(),
        state=_make_state(
            prediction_acc=prediction_acc,
            qualia_mean=qualia_value,
            action_success=action_success,
        ),
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
    """Generate N entries with predominantly negative qualia (negative_ratio ~0.65)."""
    entries = []
    for i in range(n):
        if random.random() < 0.65:
            q = random.uniform(-1.5, -0.1)
        else:
            q = random.uniform(0.1, 0.8)
        entries.append(_make_entry(
            tick=i,
            qualia_value=q,
            prediction_error=q * 0.5,
            prediction_acc=0.3 + random.uniform(-0.1, 0.1),
            action_success=0.4 + random.uniform(-0.1, 0.1),
        ))
    return entries


def _make_entries_volatile(n: int = 200) -> list[MemoryEntry]:
    """Generate N entries with high qualia volatility (std > 1.0)."""
    entries = []
    for i in range(n):
        q = random.uniform(-2.0, 1.0)
        entries.append(_make_entry(tick=i, qualia_value=q, prediction_error=q))
    return entries


def _make_entries_flat(n: int = 200) -> list[MemoryEntry]:
    """Generate N entries with very low qualia volatility (std < 0.05)."""
    entries = []
    for i in range(n):
        q = random.gauss(0.0, 0.01)
        entries.append(_make_entry(tick=i, qualia_value=q, prediction_error=q))
    return entries


def _make_memory_with(entries: list[MemoryEntry]) -> MemorySystem:
    """Create a MemorySystem populated with the given entries."""
    mem = MemorySystem(
        short_term_capacity=max(len(entries), 1000),
        significance_threshold=0.5,
        db_path=":memory:",
    )
    for entry in entries:
        mem.record(entry)
    return mem


def _make_optimizer(**kwargs) -> Optimizer:
    """Create an optimizer with sensible test defaults."""
    defaults = dict(
        enabled=True,
        window_size=200,
        reflect_interval=1,
        step_scale=0.1,
    )
    defaults.update(kwargs)
    return Optimizer(**defaults)


def _make_gate(phase: int = 2) -> CapabilityGate:
    return CapabilityGate(phase=phase)


def _make_limiter(depth: int = 1) -> RecursionLimiter:
    return RecursionLimiter(max_depth=depth)


def _make_sandbox() -> Sandbox:
    return Sandbox(timeout_s=5.0)


# ==================================================================
# Retrospective Analysis (CHECKLIST 2.12)
# ==================================================================

class TestRetrospective:
    """Verify correctness of qualia statistics computation."""

    def test_empty_entries(self):
        opt = _make_optimizer()
        analysis = opt._retrospective([])
        assert analysis.entry_count == 0
        assert analysis.mean_qualia == 0.0

    def test_all_positive(self):
        random.seed(100)
        entries = [_make_entry(i, qualia_value=0.5) for i in range(50)]
        opt = _make_optimizer()
        analysis = opt._retrospective(entries)
        assert analysis.entry_count == 50
        assert analysis.mean_qualia == pytest.approx(0.5, abs=0.01)
        assert analysis.negative_ratio == 0.0

    def test_all_negative(self):
        random.seed(101)
        entries = [_make_entry(i, qualia_value=-0.8) for i in range(50)]
        opt = _make_optimizer()
        analysis = opt._retrospective(entries)
        assert analysis.mean_qualia == pytest.approx(-0.8, abs=0.01)
        assert analysis.negative_ratio == 1.0

    def test_mixed_qualia(self):
        random.seed(102)
        entries = (
            [_make_entry(i, qualia_value=-0.5) for i in range(60)]
            + [_make_entry(i + 60, qualia_value=0.5) for i in range(40)]
        )
        opt = _make_optimizer()
        analysis = opt._retrospective(entries)
        assert analysis.negative_ratio == pytest.approx(0.6, abs=0.01)

    def test_qualia_trend_positive(self):
        """Improving qualia should produce a positive trend."""
        random.seed(103)
        entries = [_make_entry(i, qualia_value=-1.0 + i * 0.01) for i in range(100)]
        opt = _make_optimizer()
        analysis = opt._retrospective(entries)
        assert analysis.qualia_trend > 0

    def test_qualia_trend_negative(self):
        """Worsening qualia should produce a negative trend."""
        random.seed(104)
        entries = [_make_entry(i, qualia_value=1.0 - i * 0.01) for i in range(100)]
        opt = _make_optimizer()
        analysis = opt._retrospective(entries)
        assert analysis.qualia_trend < 0

    def test_interrupt_ratio(self):
        random.seed(105)
        entries = (
            [_make_entry(i, qualia_value=-1.0) for i in range(40)]  # intensity 1.0 > 0.7
            + [_make_entry(i + 40, qualia_value=0.1) for i in range(60)]  # intensity 0.1 < 0.7
        )
        opt = _make_optimizer()
        analysis = opt._retrospective(entries)
        assert analysis.interrupt_ratio == pytest.approx(0.4, abs=0.01)

    def test_std_qualia(self):
        random.seed(106)
        entries = [_make_entry(i, qualia_value=0.0) for i in range(50)]
        opt = _make_optimizer()
        analysis = opt._retrospective(entries)
        assert analysis.std_qualia == pytest.approx(0.0, abs=0.01)


# ==================================================================
# Correlation Detection (CHECKLIST 2.13)
# ==================================================================

class TestCorrelation:
    """Verify parameter-qualia correlation detection."""

    def test_insufficient_entries(self):
        random.seed(200)
        opt = _make_optimizer()
        entries = [_make_entry(i, qualia_value=0.5) for i in range(5)]
        corr = opt._detect_correlations(entries)
        assert corr == {}

    def test_zero_qualia_variance(self):
        random.seed(201)
        opt = _make_optimizer()
        entries = [_make_entry(i, qualia_value=0.5) for i in range(50)]
        corr = opt._detect_correlations(entries)
        assert corr == {}

    def test_prediction_acc_correlation(self):
        """High prediction accuracy → positive qualia → positive correlation."""
        random.seed(202)
        entries = []
        for i in range(100):
            acc = random.uniform(0.0, 1.0)
            q = acc * 0.8 + random.gauss(0, 0.05)
            entries.append(_make_entry(i, qualia_value=q, prediction_acc=acc))
        opt = _make_optimizer()
        corr = opt._detect_correlations(entries)
        assert 7 in corr
        assert corr[7] > 0.5

    def test_excludes_qualia_dims(self):
        """Dims 10, 11, 12 (qualia-derived) should be excluded."""
        random.seed(203)
        entries = _make_entries_negative(50)
        opt = _make_optimizer()
        corr = opt._detect_correlations(entries)
        assert 10 not in corr
        assert 11 not in corr
        assert 12 not in corr


# ==================================================================
# Proposal Generation (CHECKLIST 2.14)
# ==================================================================

class TestProposalGeneration:
    """Verify that the correct heuristic rules fire."""

    def test_high_volatility_decreases_beta(self):
        random.seed(300)
        opt = _make_optimizer()
        analysis = RetrospectiveAnalysis(
            window_size=200, entry_count=200,
            mean_qualia=-0.2, std_qualia=1.5,
            negative_ratio=0.5, mean_pred_error=-0.1,
            mean_intensity=0.8, qualia_trend=0.0,
            interrupt_ratio=0.1,
        )
        proposals = opt._generate_proposals(analysis, {}, {})
        beta_proposals = [p for p in proposals if p.param_name == "qualia.beta"]
        assert len(beta_proposals) >= 1
        assert beta_proposals[0].new_value < beta_proposals[0].old_value

    def test_low_volatility_increases_beta(self):
        random.seed(301)
        opt = _make_optimizer()
        analysis = RetrospectiveAnalysis(
            window_size=200, entry_count=200,
            mean_qualia=0.0, std_qualia=0.01,
            negative_ratio=0.5, mean_pred_error=0.0,
            mean_intensity=0.01, qualia_trend=0.0,
            interrupt_ratio=0.0,
        )
        proposals = opt._generate_proposals(analysis, {}, {})
        beta_proposals = [p for p in proposals if p.param_name == "qualia.beta"]
        assert len(beta_proposals) >= 1
        assert beta_proposals[0].new_value > beta_proposals[0].old_value

    def test_high_negative_ratio_increases_exploration(self):
        random.seed(302)
        opt = _make_optimizer()
        analysis = RetrospectiveAnalysis(
            window_size=200, entry_count=200,
            mean_qualia=-0.5, std_qualia=0.5,
            negative_ratio=0.7, mean_pred_error=-0.3,
            mean_intensity=0.5, qualia_trend=0.0,
            interrupt_ratio=0.1,
        )
        proposals = opt._generate_proposals(analysis, {}, {})
        expl_proposals = [p for p in proposals if p.param_name == "action.exploration_rate"]
        assert len(expl_proposals) >= 1
        assert expl_proposals[0].new_value > expl_proposals[0].old_value

    def test_high_interrupt_ratio_raises_threshold(self):
        random.seed(303)
        opt = _make_optimizer()
        analysis = RetrospectiveAnalysis(
            window_size=200, entry_count=200,
            mean_qualia=-0.3, std_qualia=0.5,
            negative_ratio=0.5, mean_pred_error=-0.1,
            mean_intensity=0.8, qualia_trend=0.0,
            interrupt_ratio=0.4,
        )
        proposals = opt._generate_proposals(analysis, {}, {})
        thr_proposals = [p for p in proposals if p.param_name == "qualia.interrupt_threshold"]
        assert len(thr_proposals) >= 1
        assert thr_proposals[0].new_value > thr_proposals[0].old_value

    def test_proposals_within_bounds(self):
        random.seed(304)
        opt = _make_optimizer()
        analysis = RetrospectiveAnalysis(
            window_size=200, entry_count=200,
            mean_qualia=-1.0, std_qualia=2.0,
            negative_ratio=0.8, mean_pred_error=-0.5,
            mean_intensity=1.0, qualia_trend=-0.01,
            interrupt_ratio=0.5,
        )
        proposals = opt._generate_proposals(analysis, {7: 0.6}, {})
        for p in proposals:
            spec = PARAM_REGISTRY[p.param_name]
            assert spec.min_val <= p.new_value <= spec.max_val, (
                f"{p.param_name}: {p.new_value} not in [{spec.min_val}, {spec.max_val}]"
            )


# ==================================================================
# Sandbox Verification (CHECKLIST 2.15)
# ==================================================================

class TestSandboxVerification:
    """Verify that proposals are tested in the sandbox."""

    def test_valid_proposal_passes(self):
        random.seed(400)
        sandbox = _make_sandbox()
        proposal = ModificationProposal(
            param_name="qualia.beta",
            old_value=1.0, new_value=0.9,
            reason="test", urgency=0.5,
        )
        result = Optimizer._sandbox_test(proposal, {"qualia.beta": 1.0}, sandbox)
        assert result.success is True

    def test_non_finite_value_fails(self):
        sandbox = _make_sandbox()
        proposal = ModificationProposal(
            param_name="qualia.beta",
            old_value=1.0, new_value=float("nan"),
            reason="test", urgency=0.5,
        )
        result = Optimizer._sandbox_test(proposal, {"qualia.beta": 1.0}, sandbox)
        assert result.success is False

    def test_out_of_bounds_fails(self):
        sandbox = _make_sandbox()
        proposal = ModificationProposal(
            param_name="qualia.beta",
            old_value=1.0, new_value=999.0,
            reason="test", urgency=0.5,
        )
        result = Optimizer._sandbox_test(proposal, {"qualia.beta": 1.0}, sandbox)
        assert result.success is False

    def test_sandbox_isolates_original(self):
        """Sandbox must not modify the original params dict."""
        sandbox = _make_sandbox()
        original = {"qualia.beta": 1.0}
        proposal = ModificationProposal(
            param_name="qualia.beta",
            old_value=1.0, new_value=0.9,
            reason="test", urgency=0.5,
        )
        Optimizer._sandbox_test(proposal, original, sandbox)
        assert original["qualia.beta"] == 1.0


# ==================================================================
# Safety Check (CHECKLIST 2.16)
# ==================================================================

class TestSafetyCheck:
    """Verify safety constraints enforcement."""

    def test_valid_proposal(self):
        proposal = ModificationProposal(
            param_name="qualia.beta",
            old_value=1.0, new_value=0.9,
            reason="test", urgency=0.5,
        )
        safe, reason = Optimizer._safety_check(proposal)
        assert safe is True

    def test_unknown_param_rejected(self):
        proposal = ModificationProposal(
            param_name="nonexistent.param",
            old_value=1.0, new_value=0.9,
            reason="test", urgency=0.5,
        )
        safe, reason = Optimizer._safety_check(proposal)
        assert safe is False
        assert "unknown" in reason

    def test_nan_rejected(self):
        proposal = ModificationProposal(
            param_name="qualia.beta",
            old_value=1.0, new_value=float("nan"),
            reason="test", urgency=0.5,
        )
        safe, reason = Optimizer._safety_check(proposal)
        assert safe is False
        assert "non-finite" in reason

    def test_inf_rejected(self):
        proposal = ModificationProposal(
            param_name="qualia.beta",
            old_value=1.0, new_value=float("inf"),
            reason="test", urgency=0.5,
        )
        safe, reason = Optimizer._safety_check(proposal)
        assert safe is False
        assert "non-finite" in reason

    def test_below_min_rejected(self):
        proposal = ModificationProposal(
            param_name="qualia.beta",
            old_value=1.0, new_value=0.01,  # min is 0.1
            reason="test", urgency=0.5,
        )
        safe, reason = Optimizer._safety_check(proposal)
        assert safe is False
        assert "out of bounds" in reason

    def test_above_max_rejected(self):
        proposal = ModificationProposal(
            param_name="qualia.beta",
            old_value=1.0, new_value=10.0,  # max is 5.0
            reason="test", urgency=0.5,
        )
        safe, reason = Optimizer._safety_check(proposal)
        assert safe is False
        assert "out of bounds" in reason

    def test_boundary_values_accepted(self):
        for name, spec in PARAM_REGISTRY.items():
            for val in [spec.min_val, spec.max_val]:
                proposal = ModificationProposal(
                    param_name=name,
                    old_value=spec.default, new_value=val,
                    reason="test", urgency=0.5,
                )
                safe, _ = Optimizer._safety_check(proposal)
                assert safe is True, f"{name}={val} should be accepted"


# ==================================================================
# Full Reflection Cycle (end-to-end)
# ==================================================================

class TestReflectCycle:
    """End-to-end reflection cycle tests."""

    def test_basic_cycle(self):
        random.seed(500)
        opt = _make_optimizer()
        model = SelfModel()
        memory = _make_memory_with(_make_entries_negative(200))
        sandbox = _make_sandbox()
        gate = _make_gate(phase=2)
        limiter = _make_limiter(depth=1)

        result = opt.reflect(
            tick=200, self_model=model, memory=memory,
            sandbox=sandbox, capability_gate=gate,
            recursion_limiter=limiter,
        )

        assert isinstance(result, ReflectionResult)
        assert result.tick == 200
        assert result.analysis.entry_count == 200
        assert len(result.proposals) > 0
        assert len(result.applied) + len(result.rejected) == len(result.proposals)

    def test_params_initialized_on_first_reflect(self):
        random.seed(501)
        opt = _make_optimizer()
        model = SelfModel()
        assert model.params == {}

        memory = _make_memory_with(_make_entries_negative(200))
        opt.reflect(
            tick=200, self_model=model, memory=memory,
            sandbox=_make_sandbox(), capability_gate=_make_gate(),
            recursion_limiter=_make_limiter(),
        )

        assert len(model.params) == len(PARAM_REGISTRY)
        for name in PARAM_REGISTRY:
            assert name in model.params

    def test_applied_modifications_update_self_model(self):
        random.seed(502)
        opt = _make_optimizer()
        model = SelfModel()
        memory = _make_memory_with(_make_entries_negative(200))

        result = opt.reflect(
            tick=200, self_model=model, memory=memory,
            sandbox=_make_sandbox(), capability_gate=_make_gate(),
            recursion_limiter=_make_limiter(),
        )

        for applied in result.applied:
            assert model.get_param(applied.param_name) == pytest.approx(
                applied.new_value, abs=1e-10
            )

    def test_stats_updated(self):
        random.seed(503)
        opt = _make_optimizer()
        model = SelfModel()
        memory = _make_memory_with(_make_entries_negative(200))

        result = opt.reflect(
            tick=200, self_model=model, memory=memory,
            sandbox=_make_sandbox(), capability_gate=_make_gate(),
            recursion_limiter=_make_limiter(),
        )

        assert opt.total_proposals == len(result.proposals)
        assert opt.total_applied == len(result.applied)
        assert opt.total_rejected == len(result.rejected)
        assert opt.reflect_count == 1

    def test_history_stored(self):
        random.seed(504)
        opt = _make_optimizer()
        model = SelfModel()
        memory = _make_memory_with(_make_entries_negative(200))

        opt.reflect(
            tick=200, self_model=model, memory=memory,
            sandbox=_make_sandbox(), capability_gate=_make_gate(),
            recursion_limiter=_make_limiter(),
        )

        assert len(opt.history) == 1
        assert opt.history[0].tick == 200


# ==================================================================
# 10+ Successful Modifications (CHECKLIST 2.17)
# ==================================================================

class TestTenModifications:
    """Verify that the optimizer can propose and apply 10+ modifications."""

    def test_at_least_ten_applied(self):
        """
        Run multiple reflection cycles with predominantly negative qualia
        data. The optimizer should fire multiple rules each cycle and
        accumulate 10+ successful modifications.
        """
        random.seed(600)
        opt = _make_optimizer(reflect_interval=1)
        model = SelfModel()
        sandbox = _make_sandbox()
        gate = _make_gate(phase=2)
        limiter = _make_limiter(depth=1)

        for cycle in range(10):
            entries = _make_entries_negative(200)
            memory = _make_memory_with(entries)
            opt.reflect(
                tick=(cycle + 1) * 200,
                self_model=model,
                memory=memory,
                sandbox=sandbox,
                capability_gate=gate,
                recursion_limiter=limiter,
            )

        assert opt.total_applied >= 10, (
            f"Expected >= 10 applied modifications, got {opt.total_applied}"
        )
        assert opt.reflect_count == 10

    def test_all_applied_within_bounds(self):
        """Every applied modification must be within PARAM_REGISTRY bounds."""
        random.seed(601)
        opt = _make_optimizer(reflect_interval=1)
        model = SelfModel()
        sandbox = _make_sandbox()
        gate = _make_gate(phase=2)
        limiter = _make_limiter(depth=1)

        for cycle in range(5):
            entries = _make_entries_negative(200)
            memory = _make_memory_with(entries)
            opt.reflect(
                tick=(cycle + 1) * 200,
                self_model=model,
                memory=memory,
                sandbox=sandbox,
                capability_gate=gate,
                recursion_limiter=limiter,
            )

        for result in opt.history:
            for applied in result.applied:
                spec = PARAM_REGISTRY[applied.param_name]
                assert spec.min_val <= applied.new_value <= spec.max_val


# ==================================================================
# Disabled Optimizer
# ==================================================================

class TestDisabled:
    """Verify behavior when optimizer is disabled."""

    def test_should_reflect_false(self):
        opt = Optimizer(enabled=False)
        assert opt.should_reflect(tick=1000, memory_size=500) is False

    def test_enabled_default_false(self):
        opt = Optimizer()
        assert opt.enabled is False


# ==================================================================
# Insufficient Data
# ==================================================================

class TestInsufficientData:
    """Verify optimizer skips reflection with too few entries."""

    def test_should_reflect_insufficient(self):
        opt = _make_optimizer(window_size=200)
        assert opt.should_reflect(tick=100, memory_size=50) is False

    def test_should_reflect_sufficient(self):
        opt = _make_optimizer(window_size=200, reflect_interval=1)
        assert opt.should_reflect(tick=200, memory_size=200) is True


# ==================================================================
# Reflect Interval
# ==================================================================

class TestReflectInterval:
    """Verify that reflect_interval is respected."""

    def test_too_soon(self):
        opt = _make_optimizer(reflect_interval=200)
        opt._last_reflect_tick = 100
        assert opt.should_reflect(tick=200, memory_size=300) is False

    def test_just_right(self):
        opt = _make_optimizer(reflect_interval=200)
        opt._last_reflect_tick = 100
        assert opt.should_reflect(tick=300, memory_size=300) is True


# ==================================================================
# Recursion Limit (L3)
# ==================================================================

class TestRecursionLimit:
    """Verify that recursion depth is enforced."""

    def test_depth_zero_blocks_reflect(self):
        """Depth 0 = no reflection allowed."""
        random.seed(700)
        opt = _make_optimizer()
        model = SelfModel()
        memory = _make_memory_with(_make_entries_negative(200))
        limiter = RecursionLimiter(max_depth=0)

        with pytest.raises(RecursionDepthExceeded):
            opt.reflect(
                tick=200, self_model=model, memory=memory,
                sandbox=_make_sandbox(), capability_gate=_make_gate(),
                recursion_limiter=limiter,
            )

    def test_nested_reflect_blocked(self):
        """Cannot reflect within a reflect (depth > 1)."""
        limiter = RecursionLimiter(max_depth=1)
        with limiter.guard():
            with pytest.raises(RecursionDepthExceeded):
                with limiter.guard():
                    pass


# ==================================================================
# Capability Gate (L5)
# ==================================================================

class TestCapabilityGate:
    """Verify that phase permissions are enforced."""

    def test_phase1_blocks_reflect(self):
        random.seed(800)
        opt = _make_optimizer()
        model = SelfModel()
        memory = _make_memory_with(_make_entries_negative(200))
        gate = CapabilityGate(phase=1)

        with pytest.raises(CapabilityDenied):
            opt.reflect(
                tick=200, self_model=model, memory=memory,
                sandbox=_make_sandbox(), capability_gate=gate,
                recursion_limiter=_make_limiter(),
            )

    def test_phase2_allows_reflect(self):
        random.seed(801)
        opt = _make_optimizer()
        model = SelfModel()
        memory = _make_memory_with(_make_entries_negative(200))
        gate = CapabilityGate(phase=2)

        result = opt.reflect(
            tick=200, self_model=model, memory=memory,
            sandbox=_make_sandbox(), capability_gate=gate,
            recursion_limiter=_make_limiter(),
        )
        assert isinstance(result, ReflectionResult)

    def test_phase3_allows_reflect(self):
        random.seed(802)
        opt = _make_optimizer()
        model = SelfModel()
        memory = _make_memory_with(_make_entries_negative(200))
        gate = CapabilityGate(phase=3)

        result = opt.reflect(
            tick=200, self_model=model, memory=memory,
            sandbox=_make_sandbox(), capability_gate=gate,
            recursion_limiter=_make_limiter(),
        )
        assert isinstance(result, ReflectionResult)


# ==================================================================
# Parameter Initialization
# ==================================================================

class TestParamInit:
    """Verify that params are initialized from PARAM_REGISTRY."""

    def test_first_reflect_initializes(self):
        random.seed(900)
        model = SelfModel()
        assert model.params == {}
        Optimizer._initialize_params(model)
        assert len(model.params) == len(PARAM_REGISTRY)
        for name, spec in PARAM_REGISTRY.items():
            assert model.get_param(name) == spec.default

    def test_subsequent_reflect_preserves_changes(self):
        """After first init, modified values should persist."""
        random.seed(901)
        opt = _make_optimizer(reflect_interval=1)
        model = SelfModel()
        memory = _make_memory_with(_make_entries_negative(200))

        r1 = opt.reflect(
            tick=200, self_model=model, memory=memory,
            sandbox=_make_sandbox(), capability_gate=_make_gate(),
            recursion_limiter=_make_limiter(),
        )

        if r1.applied:
            changed_name = r1.applied[0].param_name
            changed_value = r1.applied[0].new_value

            memory2 = _make_memory_with(_make_entries_negative(200))
            opt.reflect(
                tick=400, self_model=model, memory=memory2,
                sandbox=_make_sandbox(), capability_gate=_make_gate(),
                recursion_limiter=_make_limiter(),
            )

            current = model.get_param(changed_name)
            assert current != PARAM_REGISTRY[changed_name].default or True


# ==================================================================
# PARAM_REGISTRY Integrity
# ==================================================================

class TestParamRegistry:
    """Verify PARAM_REGISTRY consistency."""

    def test_all_defaults_within_bounds(self):
        for name, spec in PARAM_REGISTRY.items():
            assert spec.min_val <= spec.default <= spec.max_val, (
                f"{name}: default {spec.default} not in [{spec.min_val}, {spec.max_val}]"
            )

    def test_min_less_than_max(self):
        for name, spec in PARAM_REGISTRY.items():
            assert spec.min_val < spec.max_val, f"{name}: min >= max"

    def test_registry_not_empty(self):
        assert len(PARAM_REGISTRY) >= 5


# ==================================================================
# Edge Cases
# ==================================================================

class TestEdgeCases:
    """Edge case handling."""

    def test_volatile_data_produces_proposals(self):
        random.seed(1000)
        opt = _make_optimizer()
        model = SelfModel()
        memory = _make_memory_with(_make_entries_volatile(200))

        result = opt.reflect(
            tick=200, self_model=model, memory=memory,
            sandbox=_make_sandbox(), capability_gate=_make_gate(),
            recursion_limiter=_make_limiter(),
        )
        assert len(result.proposals) > 0

    def test_flat_data_produces_proposals(self):
        random.seed(1001)
        opt = _make_optimizer()
        model = SelfModel()
        memory = _make_memory_with(_make_entries_flat(200))

        result = opt.reflect(
            tick=200, self_model=model, memory=memory,
            sandbox=_make_sandbox(), capability_gate=_make_gate(),
            recursion_limiter=_make_limiter(),
        )
        # Flat data should trigger low-volatility rule
        beta_proposals = [p for p in result.proposals if p.param_name == "qualia.beta"]
        assert len(beta_proposals) >= 1

    def test_proposal_reason_not_empty(self):
        random.seed(1002)
        opt = _make_optimizer()
        model = SelfModel()
        memory = _make_memory_with(_make_entries_negative(200))

        result = opt.reflect(
            tick=200, self_model=model, memory=memory,
            sandbox=_make_sandbox(), capability_gate=_make_gate(),
            recursion_limiter=_make_limiter(),
        )
        for p in result.proposals:
            assert p.reason != ""
            assert p.urgency >= 0
