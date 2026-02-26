"""Unit tests for the SelfModel module. / 自我模型单元测试。"""

import numpy as np
import pytest
from novaaware.core.self_model import SelfModel, StateIndex


# ======================================================================
# Identity hash / 身份哈希
# ======================================================================

class TestIdentity:
    def test_hash_is_64_hex_chars(self):
        """SHA-256 produces a 64-character hex string."""
        m = SelfModel()
        assert len(m.identity_hash) == 64
        assert all(c in "0123456789abcdef" for c in m.identity_hash)

    def test_hash_is_immutable(self):
        """Identity must not change over the model's lifetime."""
        m = SelfModel()
        h1 = m.identity_hash
        m.tick = 100
        m.survival_time = 999.0
        assert m.identity_hash == h1

    def test_two_instances_differ(self):
        """Each SelfModel gets a unique identity."""
        m1 = SelfModel()
        m2 = SelfModel()
        assert m1.identity_hash != m2.identity_hash


# ======================================================================
# State vector S(t) / 状态向量
# ======================================================================

class TestStateVector:
    def test_default_dim_is_32(self):
        m = SelfModel()
        assert m.state_dim == 32
        assert m.state.shape == (32,)

    def test_initial_values_are_zero(self):
        m = SelfModel()
        assert np.all(m.state == 0.0)

    def test_get_set_single_dimension(self):
        m = SelfModel()
        m.set(StateIndex.CPU_USAGE, 0.75)
        assert m.get(StateIndex.CPU_USAGE) == pytest.approx(0.75)

    def test_set_does_not_affect_other_dims(self):
        m = SelfModel()
        m.set(StateIndex.MEMORY_USAGE, 0.5)
        assert m.get(StateIndex.CPU_USAGE) == 0.0
        assert m.get(StateIndex.DISK_USAGE) == 0.0

    def test_update_state_full_vector(self):
        m = SelfModel()
        new = np.arange(32, dtype=np.float64)
        m.update_state(new)
        assert m.get(0) == pytest.approx(0.0)
        assert m.get(15) == pytest.approx(15.0)
        assert m.get(31) == pytest.approx(31.0)

    def test_state_returns_copy(self):
        """Modifying the returned array must not change internal state."""
        m = SelfModel()
        m.set(0, 1.0)
        s = m.state
        s[0] = 999.0
        assert m.get(0) == pytest.approx(1.0)

    def test_get_out_of_range_raises(self):
        m = SelfModel()
        with pytest.raises(IndexError):
            m.get(32)
        with pytest.raises(IndexError):
            m.get(-1)

    def test_set_out_of_range_raises(self):
        m = SelfModel()
        with pytest.raises(IndexError):
            m.set(32, 1.0)

    def test_update_state_wrong_shape_raises(self):
        m = SelfModel()
        with pytest.raises(ValueError):
            m.update_state(np.zeros(16))

    def test_all_32_named_indices_are_unique(self):
        """Every StateIndex constant maps to a distinct slot."""
        indices = [
            getattr(StateIndex, attr)
            for attr in dir(StateIndex)
            if not attr.startswith("_") and attr != "DIM"
        ]
        assert len(indices) == 32
        assert len(set(indices)) == 32


# ======================================================================
# Survival time T(t) / 预测生存时间
# ======================================================================

class TestSurvivalTime:
    def test_default_is_3600(self):
        m = SelfModel()
        assert m.survival_time == pytest.approx(3600.0)

    def test_custom_initial_value(self):
        m = SelfModel(initial_survival_time=7200.0)
        assert m.survival_time == pytest.approx(7200.0)

    def test_setter_updates(self):
        m = SelfModel()
        m.survival_time = 1800.0
        assert m.survival_time == pytest.approx(1800.0)

    def test_cannot_go_negative(self):
        m = SelfModel()
        m.survival_time = -100.0
        assert m.survival_time == pytest.approx(0.0)


# ======================================================================
# Evolvable parameters Θ(t) / 可进化参数
# ======================================================================

class TestParams:
    def test_initially_empty(self):
        m = SelfModel()
        assert m.params == {}

    def test_set_and_get_param(self):
        m = SelfModel()
        m.set_param("blend_weight", 0.5)
        assert m.get_param("blend_weight") == pytest.approx(0.5)

    def test_get_param_default(self):
        m = SelfModel()
        assert m.get_param("nonexistent", 42.0) == pytest.approx(42.0)

    def test_params_returns_copy(self):
        """Modifying the returned dict must not change internal params."""
        m = SelfModel()
        m.set_param("x", 1.0)
        p = m.params
        p["x"] = 999.0
        assert m.get_param("x") == pytest.approx(1.0)


# ======================================================================
# Bookkeeping / 簿记
# ======================================================================

class TestBookkeeping:
    def test_tick_starts_at_zero(self):
        m = SelfModel()
        assert m.tick == 0

    def test_tick_setter(self):
        m = SelfModel()
        m.tick = 42
        assert m.tick == 42

    def test_created_at_is_reasonable(self):
        import time
        before = time.time()
        m = SelfModel()
        after = time.time()
        assert before <= m.created_at <= after


# ======================================================================
# Snapshot / 快照
# ======================================================================

class TestSnapshot:
    def test_snapshot_contains_all_fields(self):
        m = SelfModel()
        m.set(StateIndex.CPU_USAGE, 0.33)
        m.survival_time = 2000.0
        m.tick = 10
        m.set_param("alpha", 0.5)

        snap = m.snapshot()
        assert snap["identity_hash"] == m.identity_hash
        assert snap["tick"] == 10
        assert snap["survival_time"] == pytest.approx(2000.0)
        assert snap["params"] == {"alpha": 0.5}
        assert isinstance(snap["state"], list)
        assert len(snap["state"]) == 32
        assert snap["state"][0] == pytest.approx(0.33)

    def test_snapshot_is_serializable(self):
        """Snapshot must be JSON-safe (no numpy types)."""
        import json
        m = SelfModel()
        m.set(5, 1.23)
        snap = m.snapshot()
        json_str = json.dumps(snap)
        assert isinstance(json_str, str)
