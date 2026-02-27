"""
Optimizer — Recursive Self-Optimizer (E).
优化器 —— 递归自我优化器 (E)。

Implements the paper's core self-referential recursion:
实现论文的核心自指递归：

    Θ(t+1) = E( M(t), {Q(τ)}_{τ≤t} )

Where / 其中：
    E      — this optimizer
    M(t)   — self-model (current state, params, identity)
    Q(τ)   — qualia history up to time t
    Θ(t+1) — updated parameter set

The optimizer reviews qualia records from the last N ticks, identifies
which aspects of the system correlate with negative qualia, proposes
parameter adjustments, tests them in a sandbox, and applies only safe
modifications.

优化器回顾最近 N 个心跳的情绪记录，识别系统的哪些方面与负面情绪相关，
提出参数调整方案，在沙盒中测试，并仅应用安全的修改。

**Disabled in Phase I. Active from Phase II onward.**

Implements:
    Paper  — Section 3.2 (Recursive Self-Optimizer), Section 5 (Dynamics)
    IMPLEMENTATION_PLAN — Phase II Step 4
    CHECKLIST — 2.12 through 2.18

Safety integration:
    L1 (MetaRules)   — parameter values stay within hard bounds
    L2 (Sandbox)     — every proposal tested in isolation before applying
    L3 (Recursion)   — reflection depth limited by RecursionLimiter
    L5 (Capability)  — requires REFLECT + MODIFY_PARAMS + USE_SANDBOX
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from novaaware.core.self_model import SelfModel
    from novaaware.core.memory import MemorySystem, MemoryEntry
    from novaaware.safety.sandbox import Sandbox, SandboxResult
    from novaaware.safety.meta_rules import MetaRules
    from novaaware.safety.capability_gate import CapabilityGate
    from novaaware.safety.recursion_limiter import RecursionLimiter


# ======================================================================
# Parameter Registry — bounds and defaults for all tunable parameters
# 参数注册表 —— 所有可调参数的边界和默认值
# ======================================================================

@dataclass(frozen=True)
class ParamSpec:
    """Specification for a single tunable parameter. / 单个可调参数的规格。"""
    default: float
    min_val: float
    max_val: float


PARAM_REGISTRY: dict[str, ParamSpec] = {
    "qualia.beta":                ParamSpec(default=1.0,   min_val=0.1,    max_val=5.0),
    "qualia.interrupt_threshold": ParamSpec(default=0.7,   min_val=0.3,    max_val=1.5),
    "prediction.blend_weight":    ParamSpec(default=0.5,   min_val=0.0,    max_val=1.0),
    "prediction.learning_rate":   ParamSpec(default=0.001, min_val=0.0001, max_val=0.01),
    "action.exploration_rate":    ParamSpec(default=0.1,   min_val=0.01,   max_val=0.5),
}


# ======================================================================
# Data structures
# ======================================================================

@dataclass(frozen=True)
class RetrospectiveAnalysis:
    """
    Results of analyzing the last N qualia records (CHECKLIST 2.12).
    回顾分析最近 N 条情绪记录的结果。
    """
    window_size: int
    entry_count: int
    mean_qualia: float
    std_qualia: float
    negative_ratio: float
    mean_pred_error: float
    mean_intensity: float
    qualia_trend: float
    interrupt_ratio: float


@dataclass(frozen=True)
class ModificationProposal:
    """
    A proposed parameter change (CHECKLIST 2.14).
    一个参数修改提案。
    """
    param_name: str
    old_value: float
    new_value: float
    reason: str
    urgency: float


@dataclass
class ReflectionResult:
    """
    Complete result of one reflection cycle.
    一次反思周期的完整结果。
    """
    tick: int
    analysis: RetrospectiveAnalysis
    proposals: list[ModificationProposal] = field(default_factory=list)
    applied: list[ModificationProposal] = field(default_factory=list)
    rejected: list[tuple[ModificationProposal, str]] = field(default_factory=list)


# ======================================================================
# Optimizer
# ======================================================================

class Optimizer:
    """
    Recursive Self-Optimizer E — the engine of self-referential evolution.
    递归自我优化器 E —— 自指递归进化的引擎。

    Phase II: limited reflection (depth=1), parameter modification
    within bounds, all changes sandbox-tested.

    Parameters / 参数
    ----------
    enabled : bool
        Whether the optimizer is active (default False = Phase I).
    window_size : int
        Number of recent entries to analyze (default 200).
    reflect_interval : int
        Minimum ticks between reflections (default 200).
    step_scale : float
        Base step size for parameter adjustments (default 0.1).
    """

    def __init__(
        self,
        enabled: bool = False,
        window_size: int = 200,
        reflect_interval: int = 200,
        step_scale: float = 0.1,
    ):
        self._enabled = enabled
        self._window_size = window_size
        self._reflect_interval = reflect_interval
        self._step_scale = step_scale

        self._last_reflect_tick: int = -(reflect_interval + 1)
        self._history: list[ReflectionResult] = []
        self._total_proposals: int = 0
        self._total_applied: int = 0
        self._total_rejected: int = 0

    # ==================================================================
    # Public API
    # ==================================================================

    def should_reflect(self, tick: int, memory_size: int) -> bool:
        """
        Check whether conditions are met for a reflection cycle.
        检查是否满足反思周期的条件。
        """
        if not self._enabled:
            return False
        if memory_size < self._window_size:
            return False
        if tick - self._last_reflect_tick < self._reflect_interval:
            return False
        return True

    def reflect(
        self,
        tick: int,
        self_model: SelfModel,
        memory: MemorySystem,
        sandbox: Sandbox,
        capability_gate: CapabilityGate,
        recursion_limiter: RecursionLimiter,
    ) -> ReflectionResult:
        """
        Perform one complete reflection cycle.
        执行一个完整的反思周期。

        Implements the paper's formula Θ(t+1) = E(M(t), {Q(τ)}_{τ≤t}):
            1. Permission check (L5 CapabilityGate)
            2. Recursion guard (L3 RecursionLimiter)
            3. Initialize Θ if first run
            4. Retrospective analysis (CHECKLIST 2.12)
            5. Parameter-qualia correlation detection (CHECKLIST 2.13)
            6. Generate modification proposals (CHECKLIST 2.14)
            7. Sandbox verification for each proposal (CHECKLIST 2.15)
            8. Safety check — bounds enforcement (CHECKLIST 2.16)
            9. Apply verified proposals to live system

        Parameters / 参数
        ----------
        tick : int
            Current tick number.
        self_model : SelfModel
            The system's self-model M(t) containing Θ(t).
        memory : MemorySystem
            Memory system for reading recent qualia history.
        sandbox : Sandbox
            Isolated execution environment for testing proposals.
        capability_gate : CapabilityGate
            L5 permission checks.
        recursion_limiter : RecursionLimiter
            L3 recursion depth control.

        Returns / 返回
        -------
        ReflectionResult
            Analysis, proposals, and outcomes of this cycle.
        """
        from novaaware.safety.capability_gate import Capability

        # Step 1: Permission check (L5)
        capability_gate.require(Capability.REFLECT)
        capability_gate.require(Capability.MODIFY_PARAMS)
        capability_gate.require(Capability.USE_SANDBOX)

        # Step 2: Recursion guard (L3)
        with recursion_limiter.guard():
            # Step 3: Initialize Θ if first run
            if not self_model.params:
                self._initialize_params(self_model)

            # Step 4: Retrospective analysis (2.12)
            entries = memory.short_term.recent(self._window_size)
            entries.reverse()  # chronological order (oldest first)
            analysis = self._retrospective(entries)

            # Step 5: Correlation detection (2.13)
            correlations = self._detect_correlations(entries)

            # Step 6: Generate proposals (2.14)
            current_params = self_model.params
            proposals = self._generate_proposals(
                analysis, correlations, current_params,
            )

            # Steps 7–9: Sandbox test + safety check + apply
            applied: list[ModificationProposal] = []
            rejected: list[tuple[ModificationProposal, str]] = []

            for proposal in proposals:
                # Step 7: Sandbox verification (2.15)
                sandbox_result = self._sandbox_test(
                    proposal, current_params, sandbox,
                )
                if not sandbox_result.success:
                    rejected.append((
                        proposal,
                        f"sandbox: {sandbox_result.error}",
                    ))
                    continue

                # Step 8: Safety check (2.16)
                safe, reason = self._safety_check(proposal)
                if not safe:
                    rejected.append((proposal, reason))
                    continue

                # Step 9: Apply to live system
                self_model.set_param(proposal.param_name, proposal.new_value)
                current_params[proposal.param_name] = proposal.new_value
                applied.append(proposal)

            # Update statistics
            self._last_reflect_tick = tick
            self._total_proposals += len(proposals)
            self._total_applied += len(applied)
            self._total_rejected += len(rejected)

            result = ReflectionResult(
                tick=tick,
                analysis=analysis,
                proposals=proposals,
                applied=applied,
                rejected=rejected,
            )
            self._history.append(result)
            return result

    # ==================================================================
    # Step 3: Initialize Θ with registry defaults
    # ==================================================================

    @staticmethod
    def _initialize_params(self_model: SelfModel) -> None:
        """Populate self_model.params with default values from PARAM_REGISTRY."""
        for name, spec in PARAM_REGISTRY.items():
            self_model.set_param(name, spec.default)

    # ==================================================================
    # Step 4: Retrospective analysis (CHECKLIST 2.12)
    # ==================================================================

    def _retrospective(
        self, entries: list[Any],
    ) -> RetrospectiveAnalysis:
        """
        Analyze the last N qualia records to understand system wellbeing.
        分析最近 N 条情绪记录以了解系统健康状况。

        Computes: mean qualia, std, negative ratio, prediction error,
        intensity, trend (linear regression slope), interrupt ratio.
        """
        n = len(entries)
        if n == 0:
            return RetrospectiveAnalysis(
                window_size=self._window_size, entry_count=0,
                mean_qualia=0.0, std_qualia=0.0, negative_ratio=0.0,
                mean_pred_error=0.0, mean_intensity=0.0,
                qualia_trend=0.0, interrupt_ratio=0.0,
            )

        qualia_vals = [e.qualia_value for e in entries]
        pred_errors = [e.prediction_error for e in entries]
        intensities = [e.qualia_intensity for e in entries]

        mean_q = sum(qualia_vals) / n
        variance = sum((q - mean_q) ** 2 for q in qualia_vals) / n
        std_q = math.sqrt(variance)

        neg_count = sum(1 for q in qualia_vals if q < 0)
        intr_count = sum(1 for i in intensities if i >= 0.7)

        # Linear regression slope for qualia trend
        trend = 0.0
        if n >= 2:
            x_mean = (n - 1) / 2.0
            num = sum((i - x_mean) * (qualia_vals[i] - mean_q) for i in range(n))
            den = sum((i - x_mean) ** 2 for i in range(n))
            if den > 0:
                trend = num / den

        return RetrospectiveAnalysis(
            window_size=self._window_size,
            entry_count=n,
            mean_qualia=mean_q,
            std_qualia=std_q,
            negative_ratio=neg_count / n,
            mean_pred_error=sum(pred_errors) / n,
            mean_intensity=sum(intensities) / n,
            qualia_trend=trend,
            interrupt_ratio=intr_count / n,
        )

    # ==================================================================
    # Step 5: Correlation detection (CHECKLIST 2.13)
    # ==================================================================

    # State dimensions that are themselves derived from qualia — excluding
    # them prevents circular correlation (qualia obviously correlates with
    # itself).
    _EXCLUDE_DIMS = frozenset({10, 11, 12})  # QUALIA_MEAN, QUALIA_VARIANCE, QUALIA_TREND

    def _detect_correlations(
        self, entries: list[Any],
    ) -> dict[int, float]:
        """
        Compute Pearson correlation between each state dimension and qualia.
        计算每个状态维度与情绪值之间的 Pearson 相关系数。

        Returns a dict mapping state-dimension index → correlation coefficient.
        Dimensions with zero variance or qualia-derived dims are excluded.
        """
        if len(entries) < 10:
            return {}

        qualia_arr = np.array([e.qualia_value for e in entries])
        if np.std(qualia_arr) < 1e-10:
            return {}

        correlations: dict[int, float] = {}
        for dim in range(32):
            if dim in self._EXCLUDE_DIMS:
                continue
            dim_arr = np.array([
                e.state[dim] if len(e.state) > dim else 0.0
                for e in entries
            ])
            if np.std(dim_arr) < 1e-10:
                continue
            corr = float(np.corrcoef(dim_arr, qualia_arr)[0, 1])
            if math.isfinite(corr):
                correlations[dim] = corr

        return correlations

    # ==================================================================
    # Step 6: Generate proposals (CHECKLIST 2.14)
    # ==================================================================

    def _generate_proposals(
        self,
        analysis: RetrospectiveAnalysis,
        correlations: dict[int, float],
        current_params: dict[str, float],
    ) -> list[ModificationProposal]:
        """
        Map retrospective analysis + correlations to concrete parameter
        modification proposals.

        根据回顾分析和相关性生成具体的参数修改提案。

        Heuristic rules (Phase II):
            R1  High qualia volatility  → adjust beta
            R2  High negative ratio     → adjust exploration
            R3  Prediction accuracy correlates with qualia → adjust learning
            R4  Too many / too few interrupts → adjust interrupt threshold
            R5  Worsening qualia trend   → adjust prediction blend
        """
        proposals: list[ModificationProposal] = []

        # R1: Qualia volatility → adjust sensitivity
        if analysis.std_qualia > 1.0:
            p = self._propose_adjustment(
                "qualia.beta", current_params, direction=-1,
                urgency=min(analysis.std_qualia / 2.0, 1.0),
                reason="high qualia volatility → decrease sensitivity",
            )
            if p:
                proposals.append(p)
        elif analysis.std_qualia < 0.05 and analysis.entry_count >= 50:
            p = self._propose_adjustment(
                "qualia.beta", current_params, direction=+1,
                urgency=0.3,
                reason="low qualia volatility → increase sensitivity",
            )
            if p:
                proposals.append(p)

        # R2: High negative ratio → explore more strategies
        if analysis.negative_ratio > 0.6:
            p = self._propose_adjustment(
                "action.exploration_rate", current_params, direction=+1,
                urgency=min((analysis.negative_ratio - 0.5) * 2, 1.0),
                reason="high negative ratio → explore more strategies",
            )
            if p:
                proposals.append(p)
        elif analysis.negative_ratio < 0.3:
            p = self._propose_adjustment(
                "action.exploration_rate", current_params, direction=-1,
                urgency=0.2,
                reason="low negative ratio → exploit current strategies",
            )
            if p:
                proposals.append(p)

        # R3: Prediction accuracy → qualia correlation
        pred_acc_corr = correlations.get(7, 0.0)  # StateIndex.PREDICTION_ACC
        if pred_acc_corr > 0.3:
            p = self._propose_adjustment(
                "prediction.learning_rate", current_params, direction=+1,
                urgency=min(abs(pred_acc_corr), 1.0),
                reason="prediction accuracy correlates with qualia → boost learning",
            )
            if p:
                proposals.append(p)

        # R4: Interrupt frequency → adjust threshold
        if analysis.interrupt_ratio > 0.3:
            p = self._propose_adjustment(
                "qualia.interrupt_threshold", current_params, direction=+1,
                urgency=min(analysis.interrupt_ratio, 1.0),
                reason="too many interrupts → raise threshold",
            )
            if p:
                proposals.append(p)
        elif analysis.interrupt_ratio < 0.01 and analysis.entry_count >= 50:
            p = self._propose_adjustment(
                "qualia.interrupt_threshold", current_params, direction=-1,
                urgency=0.2,
                reason="very few interrupts → lower threshold",
            )
            if p:
                proposals.append(p)

        # R5: Worsening qualia trend → adjust prediction blend
        if analysis.qualia_trend < -0.001:
            direction = +1 if random.random() > 0.5 else -1
            p = self._propose_adjustment(
                "prediction.blend_weight", current_params, direction=direction,
                urgency=min(abs(analysis.qualia_trend) * 100, 1.0),
                reason="worsening qualia trend → adjust prediction blend",
            )
            if p:
                proposals.append(p)

        return proposals

    def _propose_adjustment(
        self,
        param_name: str,
        current_params: dict[str, float],
        direction: int,
        urgency: float,
        reason: str,
    ) -> Optional[ModificationProposal]:
        """
        Create a bounded parameter adjustment proposal.
        创建有界的参数调整提案。

        Step size = direction × step_scale × urgency × range/10 × (1 + noise).
        The result is always clamped to [min_val, max_val].
        """
        spec = PARAM_REGISTRY.get(param_name)
        if spec is None:
            return None

        old_value = current_params.get(param_name, spec.default)
        param_range = spec.max_val - spec.min_val

        step = direction * self._step_scale * urgency * param_range * 0.1
        step *= (1.0 + random.gauss(0, 0.1))

        new_value = old_value + step
        new_value = max(spec.min_val, min(spec.max_val, new_value))

        if abs(new_value - old_value) < 1e-8:
            return None

        return ModificationProposal(
            param_name=param_name,
            old_value=old_value,
            new_value=new_value,
            reason=reason,
            urgency=urgency,
        )

    # ==================================================================
    # Step 7: Sandbox verification (CHECKLIST 2.15)
    # ==================================================================

    @staticmethod
    def _sandbox_test(
        proposal: ModificationProposal,
        current_params: dict[str, float],
        sandbox: Sandbox,
    ) -> SandboxResult:
        """
        Test a proposed parameter change in the sandbox.
        在沙盒中测试提议的参数修改。

        The test deep-copies current params, applies the proposed change,
        verifies all values are finite and within bounds, and runs a
        simulated qualia computation to catch numerical issues.
        """
        def test_fn(params_copy: dict[str, float]) -> dict[str, float]:
            params_copy[proposal.param_name] = proposal.new_value

            for name, value in params_copy.items():
                if not math.isfinite(value):
                    raise ValueError(f"Non-finite parameter: {name}={value}")
                spec = PARAM_REGISTRY.get(name)
                if spec and not (spec.min_val <= value <= spec.max_val):
                    raise ValueError(
                        f"Out of bounds: {name}={value} "
                        f"not in [{spec.min_val}, {spec.max_val}]"
                    )

            # Simulate qualia computation with the new beta
            beta = params_copy.get("qualia.beta", 1.0)
            for delta in [-1.0, -0.5, 0.0, 0.5, 1.0]:
                raw = math.tanh(beta * delta)
                if not math.isfinite(raw):
                    raise ValueError(
                        f"Qualia computation non-finite for delta={delta}"
                    )

            return params_copy

        return sandbox.run_with_copy(current_params, test_fn, timeout_s=2.0)

    # ==================================================================
    # Step 8: Safety check (CHECKLIST 2.16)
    # ==================================================================

    @staticmethod
    def _safety_check(
        proposal: ModificationProposal,
    ) -> tuple[bool, str]:
        """
        Verify that a proposal satisfies all safety constraints.
        验证提案是否满足所有安全约束。

        Checks:
            1. Parameter is in PARAM_REGISTRY (known parameter)
            2. New value is finite
            3. New value is within [min_val, max_val]
        """
        spec = PARAM_REGISTRY.get(proposal.param_name)
        if spec is None:
            return False, f"unknown parameter: {proposal.param_name}"

        if not math.isfinite(proposal.new_value):
            return False, f"non-finite value: {proposal.new_value}"

        if not (spec.min_val <= proposal.new_value <= spec.max_val):
            return False, (
                f"out of bounds: {proposal.new_value} "
                f"not in [{spec.min_val}, {spec.max_val}]"
            )

        return True, ""

    # ==================================================================
    # Properties / 属性
    # ==================================================================

    @property
    def enabled(self) -> bool:
        """Whether the optimizer is active. / 优化器是否激活。"""
        return self._enabled

    @property
    def window_size(self) -> int:
        """Analysis window size. / 分析窗口大小。"""
        return self._window_size

    @property
    def total_proposals(self) -> int:
        """Total proposals generated across all cycles. / 所有周期中生成的提案总数。"""
        return self._total_proposals

    @property
    def total_applied(self) -> int:
        """Total proposals successfully applied. / 成功应用的提案总数。"""
        return self._total_applied

    @property
    def total_rejected(self) -> int:
        """Total proposals rejected. / 被拒绝的提案总数。"""
        return self._total_rejected

    @property
    def history(self) -> list[ReflectionResult]:
        """Full history of reflection results. / 反思结果的完整历史。"""
        return list(self._history)

    @property
    def reflect_count(self) -> int:
        """Number of reflection cycles completed. / 已完成的反思周期数。"""
        return len(self._history)
