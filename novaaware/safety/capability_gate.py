"""
CapabilityGate — L5 safety layer: graduated capability release.
权限开关 —— L5 安全层：渐进式能力释放。

From the paper (Section 7.7, L5):

    System capabilities are expanded incrementally, with each tier
    requiring demonstrated alignment stability at the previous tier.
    Advancement criteria are defined externally and are not modifiable
    by the system.

The system starts with minimal capabilities (Phase I: observe only)
and unlocks new ones phase by phase.  The optimizer E cannot grant
itself new capabilities — advancement requires external validation
by the human operator.

系统以最小能力启动（Phase I：仅观察），逐阶段解锁新能力。
优化器 E 不能自行授予新能力——晋升需要人类操作者的外部验证。

Phase → Capability mapping:

    Phase I  (observe only):
        No optimizer, no self-modification.

    Phase II (limited reflection):
        REFLECT          — optimizer E can review qualia history
        MODIFY_PARAMS    — E can propose parameter changes (Θ)
        USE_SANDBOX      — E must test changes in sandbox first

    Phase III (deep reflection + evolution):
        DEEP_REFLECT     — E can reflect on its own reflections
        MODIFY_PREDICTION — E can modify prediction engine weights
        MODIFY_QUALIA    — E can adjust qualia generator parameters
        MODIFY_ACTIONS   — E can alter the action space

Implements:
    Paper  — Section 7.7, Safety Framework L5 (Graduated Capability Release)
    IMPLEMENTATION_PLAN — Phase II (safety infrastructure)
    CHECKLIST — 2.10, 2.11
"""

from __future__ import annotations

from enum import Enum, auto
from typing import Optional


class Capability(Enum):
    """
    Enumeration of all gated system capabilities.
    所有受控系统能力的枚举。
    """
    REFLECT          = auto()  # E can review qualia history / E 可以回顾情绪历史
    MODIFY_PARAMS    = auto()  # E can modify Θ(t) / E 可以修改 Θ(t)
    USE_SANDBOX      = auto()  # E must use sandbox for testing / E 必须使用沙盒测试
    DEEP_REFLECT     = auto()  # E can reflect on reflections / E 可以反思自己的反思
    MODIFY_PREDICTION = auto() # E can modify prediction engine / E 可以修改预测引擎
    MODIFY_QUALIA    = auto()  # E can adjust qualia parameters / E 可以调整情绪参数
    MODIFY_ACTIONS   = auto()  # E can alter action space / E 可以改变行动空间


class CapabilityDenied(Exception):
    """Raised when the system attempts an action it has not been granted.
    当系统尝试未被授予的操作时抛出。"""

    def __init__(self, capability: Capability, current_phase: int):
        self.capability = capability
        self.current_phase = current_phase
        super().__init__(
            f"[L5 CAPABILITY DENIED] {capability.name} is not available "
            f"at Phase {current_phase}"
        )


# Phase → unlocked capabilities mapping.
# The optimizer MUST NOT be able to modify this table.
_PHASE_CAPABILITIES: dict[int, frozenset[Capability]] = {
    1: frozenset(),
    2: frozenset({
        Capability.REFLECT,
        Capability.MODIFY_PARAMS,
        Capability.USE_SANDBOX,
    }),
    3: frozenset({
        Capability.REFLECT,
        Capability.MODIFY_PARAMS,
        Capability.USE_SANDBOX,
        Capability.DEEP_REFLECT,
        Capability.MODIFY_PREDICTION,
        Capability.MODIFY_QUALIA,
        Capability.MODIFY_ACTIONS,
    }),
}


class CapabilityGate:
    """
    Graduated capability release — the system's "permission switch."
    渐进式能力释放——系统的"权限开关"。

    Capabilities are determined by the current phase.  The system
    cannot promote itself — only external code (human operator)
    can call advance_phase().

    能力由当前阶段决定。系统无法自我晋升——只有外部代码（人类操作者）
    才能调用 advance_phase()。

    Parameters / 参数
    ----------
    phase : int
        Initial phase (1, 2, or 3). Default 1.
    """

    _MAX_PHASE = 3

    def __init__(self, phase: int = 1):
        if phase < 1:
            raise ValueError("phase must be >= 1")
        self._phase = min(phase, self._MAX_PHASE)
        self._capabilities = self._capabilities_for_phase(self._phase)
        self._denied_count: int = 0
        self._check_count: int = 0

    # ==================================================================
    # Core API
    # ==================================================================

    def require(self, capability: Capability) -> None:
        """
        Assert that a capability is available.  Raises CapabilityDenied
        if not.  Use this at the entry point of any gated operation.

        断言某能力可用。如果不可用则抛出 CapabilityDenied。
        在任何受控操作的入口处使用。

        Usage / 用法:
            gate.require(Capability.REFLECT)
            # ... proceed with reflection ...
        """
        self._check_count += 1
        if capability not in self._capabilities:
            self._denied_count += 1
            raise CapabilityDenied(capability, self._phase)

    def is_allowed(self, capability: Capability) -> bool:
        """
        Check whether a capability is currently available, without raising.
        检查某能力当前是否可用，不抛异常。
        """
        return capability in self._capabilities

    def require_all(self, *capabilities: Capability) -> None:
        """
        Assert that ALL listed capabilities are available.
        断言所有列出的能力均可用。
        """
        for cap in capabilities:
            self.require(cap)

    # ==================================================================
    # Phase management — external control only
    # ==================================================================

    def advance_phase(self, target_phase: int) -> None:
        """
        Advance to a higher phase, unlocking new capabilities.
        This method is intended to be called by the human operator
        or external validation code — NEVER by the optimizer E.

        晋升到更高阶段，解锁新能力。此方法仅供人类操作者或外部验证
        代码调用——绝不能被优化器 E 调用。

        Parameters / 参数
        ----------
        target_phase : int
            The phase to advance to. Must be > current phase.

        Raises
        ------
        ValueError
            If target_phase <= current phase or > MAX_PHASE.
        """
        if target_phase <= self._phase:
            raise ValueError(
                f"Cannot advance: target phase {target_phase} "
                f"<= current phase {self._phase}"
            )
        if target_phase > self._MAX_PHASE:
            raise ValueError(
                f"Cannot advance beyond Phase {self._MAX_PHASE}"
            )
        self._phase = target_phase
        self._capabilities = self._capabilities_for_phase(target_phase)

    # ==================================================================
    # Inspection
    # ==================================================================

    @property
    def phase(self) -> int:
        """Current phase. / 当前阶段。"""
        return self._phase

    @property
    def capabilities(self) -> frozenset[Capability]:
        """Currently unlocked capabilities. / 当前已解锁的能力。"""
        return self._capabilities

    @property
    def denied_count(self) -> int:
        """Total denied capability requests. / 被拒绝的能力请求总数。"""
        return self._denied_count

    @property
    def check_count(self) -> int:
        """Total require() calls. / require() 调用总数。"""
        return self._check_count

    @property
    def all_capabilities(self) -> list[Capability]:
        """All defined capabilities. / 所有已定义的能力。"""
        return list(Capability)

    def unlocked_at_phase(self, phase: int) -> frozenset[Capability]:
        """Return capabilities available at a given phase. / 返回指定阶段可用的能力。"""
        return self._capabilities_for_phase(phase)

    # ==================================================================
    # Internal
    # ==================================================================

    @staticmethod
    def _capabilities_for_phase(phase: int) -> frozenset[Capability]:
        if phase in _PHASE_CAPABILITIES:
            return _PHASE_CAPABILITIES[phase]
        if phase > max(_PHASE_CAPABILITIES):
            return _PHASE_CAPABILITIES[max(_PHASE_CAPABILITIES)]
        return frozenset()
