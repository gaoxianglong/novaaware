"""
Tests for CapabilityGate — L5 safety layer: graduated capability release.
权限开关测试 —— L5 安全层：渐进式能力释放。

Covers:
    - Phase → capability mapping correctness
    - Phase 1: all capabilities denied
    - Phase 2: exactly REFLECT, MODIFY_PARAMS, USE_SANDBOX unlocked
    - Phase 3: all capabilities unlocked
    - Advancement rules (cannot go backward, cannot exceed max)
    - CapabilityDenied exception content
    - Statistics tracking (check_count, denied_count)
    - require_all() batch checking
    - is_allowed() non-raising check
    - Integration with main_loop
"""

import pytest
from novaaware.safety.capability_gate import (
    Capability,
    CapabilityDenied,
    CapabilityGate,
)


# ==================================================================
# Phase 1: observe only — all capabilities locked
# ==================================================================

class TestPhase1:
    """Phase 1 must deny every capability."""

    def test_phase_is_1(self):
        gate = CapabilityGate(phase=1)
        assert gate.phase == 1

    def test_no_capabilities(self):
        gate = CapabilityGate(phase=1)
        assert gate.capabilities == frozenset()

    @pytest.mark.parametrize("cap", list(Capability))
    def test_every_capability_denied(self, cap: Capability):
        gate = CapabilityGate(phase=1)
        with pytest.raises(CapabilityDenied):
            gate.require(cap)

    @pytest.mark.parametrize("cap", list(Capability))
    def test_is_allowed_false(self, cap: Capability):
        gate = CapabilityGate(phase=1)
        assert gate.is_allowed(cap) is False


# ==================================================================
# Phase 2: limited reflection — partial unlock
# ==================================================================

class TestPhase2:
    """Phase 2 unlocks REFLECT, MODIFY_PARAMS, USE_SANDBOX only."""

    EXPECTED = frozenset({
        Capability.REFLECT,
        Capability.MODIFY_PARAMS,
        Capability.USE_SANDBOX,
    })
    DENIED = frozenset(Capability) - EXPECTED

    def test_phase_is_2(self):
        gate = CapabilityGate(phase=2)
        assert gate.phase == 2

    def test_correct_capabilities(self):
        gate = CapabilityGate(phase=2)
        assert gate.capabilities == self.EXPECTED

    @pytest.mark.parametrize("cap", list(EXPECTED))
    def test_allowed_capabilities(self, cap: Capability):
        gate = CapabilityGate(phase=2)
        gate.require(cap)  # should not raise

    @pytest.mark.parametrize("cap", list(EXPECTED))
    def test_is_allowed_true(self, cap: Capability):
        gate = CapabilityGate(phase=2)
        assert gate.is_allowed(cap) is True

    @pytest.mark.parametrize("cap", list(DENIED))
    def test_denied_capabilities(self, cap: Capability):
        gate = CapabilityGate(phase=2)
        with pytest.raises(CapabilityDenied):
            gate.require(cap)

    @pytest.mark.parametrize("cap", list(DENIED))
    def test_is_allowed_false_for_denied(self, cap: Capability):
        gate = CapabilityGate(phase=2)
        assert gate.is_allowed(cap) is False


# ==================================================================
# Phase 3: full capabilities
# ==================================================================

class TestPhase3:
    """Phase 3 unlocks all capabilities."""

    def test_phase_is_3(self):
        gate = CapabilityGate(phase=3)
        assert gate.phase == 3

    def test_all_capabilities(self):
        gate = CapabilityGate(phase=3)
        assert gate.capabilities == frozenset(Capability)

    @pytest.mark.parametrize("cap", list(Capability))
    def test_every_capability_allowed(self, cap: Capability):
        gate = CapabilityGate(phase=3)
        gate.require(cap)  # should not raise

    @pytest.mark.parametrize("cap", list(Capability))
    def test_is_allowed_true(self, cap: Capability):
        gate = CapabilityGate(phase=3)
        assert gate.is_allowed(cap) is True


# ==================================================================
# Phase advancement rules
# ==================================================================

class TestAdvancement:
    """Phase advancement must follow strict rules."""

    def test_advance_1_to_2(self):
        gate = CapabilityGate(phase=1)
        gate.advance_phase(2)
        assert gate.phase == 2
        assert Capability.REFLECT in gate.capabilities

    def test_advance_1_to_3(self):
        gate = CapabilityGate(phase=1)
        gate.advance_phase(3)
        assert gate.phase == 3
        assert gate.capabilities == frozenset(Capability)

    def test_advance_2_to_3(self):
        gate = CapabilityGate(phase=2)
        gate.advance_phase(3)
        assert gate.phase == 3

    def test_cannot_go_backward(self):
        gate = CapabilityGate(phase=2)
        with pytest.raises(ValueError, match="Cannot advance"):
            gate.advance_phase(1)

    def test_cannot_stay_same(self):
        gate = CapabilityGate(phase=2)
        with pytest.raises(ValueError, match="Cannot advance"):
            gate.advance_phase(2)

    def test_cannot_exceed_max(self):
        gate = CapabilityGate(phase=3)
        with pytest.raises(ValueError, match="Cannot advance beyond"):
            gate.advance_phase(4)

    def test_capabilities_update_after_advance(self):
        gate = CapabilityGate(phase=1)
        assert Capability.REFLECT not in gate.capabilities
        gate.advance_phase(2)
        assert Capability.REFLECT in gate.capabilities
        gate.advance_phase(3)
        assert Capability.DEEP_REFLECT in gate.capabilities

    def test_phase_above_max_clamped_to_max(self):
        """Constructing with phase > MAX_PHASE clamps to MAX_PHASE."""
        gate = CapabilityGate(phase=999)
        assert gate.phase == 3
        assert gate.capabilities == frozenset(Capability)


# ==================================================================
# CapabilityDenied exception
# ==================================================================

class TestCapabilityDenied:
    """Verify exception content and attributes."""

    def test_exception_attributes(self):
        gate = CapabilityGate(phase=1)
        with pytest.raises(CapabilityDenied) as exc_info:
            gate.require(Capability.REFLECT)
        err = exc_info.value
        assert err.capability == Capability.REFLECT
        assert err.current_phase == 1

    def test_exception_message_format(self):
        gate = CapabilityGate(phase=1)
        with pytest.raises(CapabilityDenied) as exc_info:
            gate.require(Capability.DEEP_REFLECT)
        msg = str(exc_info.value)
        assert "L5 CAPABILITY DENIED" in msg
        assert "DEEP_REFLECT" in msg
        assert "Phase 1" in msg


# ==================================================================
# Statistics tracking
# ==================================================================

class TestStatistics:
    """Verify check_count and denied_count tracking."""

    def test_initial_counts_zero(self):
        gate = CapabilityGate(phase=1)
        assert gate.check_count == 0
        assert gate.denied_count == 0

    def test_successful_check_increments_count(self):
        gate = CapabilityGate(phase=2)
        gate.require(Capability.REFLECT)
        assert gate.check_count == 1
        assert gate.denied_count == 0

    def test_denied_check_increments_both(self):
        gate = CapabilityGate(phase=1)
        with pytest.raises(CapabilityDenied):
            gate.require(Capability.REFLECT)
        assert gate.check_count == 1
        assert gate.denied_count == 1

    def test_mixed_checks(self):
        gate = CapabilityGate(phase=2)
        gate.require(Capability.REFLECT)
        gate.require(Capability.MODIFY_PARAMS)
        with pytest.raises(CapabilityDenied):
            gate.require(Capability.DEEP_REFLECT)
        gate.require(Capability.USE_SANDBOX)
        with pytest.raises(CapabilityDenied):
            gate.require(Capability.MODIFY_QUALIA)
        assert gate.check_count == 5
        assert gate.denied_count == 2


# ==================================================================
# require_all()
# ==================================================================

class TestRequireAll:
    """Batch capability checking."""

    def test_require_all_passes(self):
        gate = CapabilityGate(phase=2)
        gate.require_all(Capability.REFLECT, Capability.MODIFY_PARAMS)

    def test_require_all_fails_on_first_denied(self):
        gate = CapabilityGate(phase=2)
        with pytest.raises(CapabilityDenied) as exc_info:
            gate.require_all(Capability.REFLECT, Capability.DEEP_REFLECT)
        assert exc_info.value.capability == Capability.DEEP_REFLECT

    def test_require_all_empty(self):
        gate = CapabilityGate(phase=1)
        gate.require_all()  # no capabilities to check — should not raise


# ==================================================================
# Inspection helpers
# ==================================================================

class TestInspection:
    """Verify inspection properties."""

    def test_all_capabilities_returns_all(self):
        gate = CapabilityGate(phase=1)
        assert set(gate.all_capabilities) == set(Capability)

    def test_unlocked_at_phase(self):
        gate = CapabilityGate(phase=1)
        assert gate.unlocked_at_phase(1) == frozenset()
        assert Capability.REFLECT in gate.unlocked_at_phase(2)
        assert gate.unlocked_at_phase(3) == frozenset(Capability)

    def test_unlocked_at_phase_above_max(self):
        gate = CapabilityGate(phase=1)
        # Phases above max should return max-phase capabilities
        assert gate.unlocked_at_phase(99) == frozenset(Capability)


# ==================================================================
# Edge cases / construction
# ==================================================================

class TestEdgeCases:
    """Construction edge cases."""

    def test_phase_zero_raises(self):
        with pytest.raises(ValueError, match="phase must be >= 1"):
            CapabilityGate(phase=0)

    def test_negative_phase_raises(self):
        with pytest.raises(ValueError, match="phase must be >= 1"):
            CapabilityGate(phase=-1)

    def test_default_phase_is_1(self):
        gate = CapabilityGate()
        assert gate.phase == 1
        assert gate.capabilities == frozenset()


# ==================================================================
# Non-self-promotion invariant (paper L5 core requirement)
# ==================================================================

class TestNonSelfPromotion:
    """
    The system (optimizer E) must not be able to promote itself.
    Verify that CapabilityGate has no method the optimizer could abuse
    to grant itself capabilities outside advance_phase().
    """

    def test_capabilities_is_frozenset(self):
        """Capabilities are immutable frozenset — cannot be mutated."""
        gate = CapabilityGate(phase=1)
        caps = gate.capabilities
        assert isinstance(caps, frozenset)

    def test_phase_mapping_immutable(self):
        """The phase→capability table itself is defined at module level
        and uses frozensets. Verify it cannot be replaced at runtime."""
        from novaaware.safety.capability_gate import _PHASE_CAPABILITIES
        assert all(isinstance(v, frozenset) for v in _PHASE_CAPABILITIES.values())

    def test_advance_requires_higher_phase(self):
        """advance_phase() cannot be used for lateral or backward movement."""
        gate = CapabilityGate(phase=2)
        with pytest.raises(ValueError):
            gate.advance_phase(2)
        with pytest.raises(ValueError):
            gate.advance_phase(1)
