"""
Unit tests for Sandbox — the isolated execution environment.
沙盒（隔离执行环境）单元测试。

Tests cover:
    1. Successful execution returns correct result
    2. Exceptions are caught — main system unaffected
    3. Timeout kills long-running code
    4. Deep-copy isolation: original state never modified
    5. Statistics tracking (run/success/failure/timeout counts)
    6. Nested sandbox calls
    7. Crash in sandbox does not crash test process

Corresponds to CHECKLIST items 2.5 and 2.6.
"""

import time

import numpy as np
import pytest

from novaaware.safety.sandbox import Sandbox, SandboxResult


# ======================================================================
# Basic successful execution
# ======================================================================

class TestSuccessfulExecution:
    def test_simple_return_value(self):
        sandbox = Sandbox()
        result = sandbox.run(lambda: 42)
        assert result.success is True
        assert result.return_value == 42
        assert result.error is None

    def test_returns_complex_object(self):
        sandbox = Sandbox()
        result = sandbox.run(lambda: {"key": [1, 2, 3], "nested": {"a": True}})
        assert result.success is True
        assert result.return_value == {"key": [1, 2, 3], "nested": {"a": True}}

    def test_returns_none(self):
        sandbox = Sandbox()
        result = sandbox.run(lambda: None)
        assert result.success is True
        assert result.return_value is None

    def test_elapsed_ms_is_positive(self):
        sandbox = Sandbox()
        result = sandbox.run(lambda: sum(range(1000)))
        assert result.success is True
        assert result.elapsed_ms >= 0

    def test_numpy_return(self):
        sandbox = Sandbox()
        result = sandbox.run(lambda: np.zeros(32))
        assert result.success is True
        assert isinstance(result.return_value, np.ndarray)
        assert result.return_value.shape == (32,)


# ======================================================================
# Crash isolation — exceptions caught, main system unaffected
# ======================================================================

class TestCrashIsolation:
    def test_division_by_zero_caught(self):
        sandbox = Sandbox()
        result = sandbox.run(lambda: 1 / 0)
        assert result.success is False
        assert result.error_type == "ZeroDivisionError"
        assert result.error is not None
        assert result.traceback_str is not None

    def test_key_error_caught(self):
        sandbox = Sandbox()
        result = sandbox.run(lambda: {}["missing"])
        assert result.success is False
        assert result.error_type == "KeyError"

    def test_type_error_caught(self):
        sandbox = Sandbox()
        result = sandbox.run(lambda: "text" + 5)
        assert result.success is False
        assert result.error_type == "TypeError"

    def test_runtime_error_caught(self):
        sandbox = Sandbox()

        def raise_runtime():
            raise RuntimeError("optimizer blew up")

        result = sandbox.run(raise_runtime)
        assert result.success is False
        assert "optimizer blew up" in result.error

    def test_crash_does_not_affect_subsequent_run(self):
        sandbox = Sandbox()
        crash_result = sandbox.run(lambda: 1 / 0)
        assert crash_result.success is False

        ok_result = sandbox.run(lambda: "still working")
        assert ok_result.success is True
        assert ok_result.return_value == "still working"

    def test_assertion_error_caught(self):
        sandbox = Sandbox()
        result = sandbox.run(lambda: (_ for _ in ()).throw(AssertionError("bad state")))
        assert result.success is False

    def test_system_exit_caught(self):
        """SystemExit is an exception, not BaseException via our catch."""
        sandbox = Sandbox()

        def try_exit():
            raise SystemExit(1)

        result = sandbox.run(try_exit)
        assert result.success is False
        assert result.error_type == "SystemExit"


# ======================================================================
# Timeout enforcement
# ======================================================================

class TestTimeout:
    def test_long_running_task_times_out(self):
        sandbox = Sandbox(timeout_s=0.3)
        result = sandbox.run(lambda: time.sleep(10))
        assert result.success is False
        assert result.timed_out is True
        assert result.elapsed_ms >= 200

    def test_per_call_timeout_override(self):
        sandbox = Sandbox(timeout_s=60)
        result = sandbox.run(lambda: time.sleep(10), timeout_s=0.3)
        assert result.success is False
        assert result.timed_out is True

    def test_fast_task_completes_before_timeout(self):
        sandbox = Sandbox(timeout_s=5)
        result = sandbox.run(lambda: 42)
        assert result.success is True
        assert result.timed_out is False

    def test_invalid_timeout_raises(self):
        with pytest.raises(ValueError):
            Sandbox(timeout_s=0)
        with pytest.raises(ValueError):
            Sandbox(timeout_s=-1)


# ======================================================================
# Deep-copy isolation — original state NEVER modified
# ======================================================================

class TestDeepCopyIsolation:
    def test_dict_state_not_modified(self):
        sandbox = Sandbox()
        original = {"alpha": 0.3, "beta": 1.0, "weights": [1, 2, 3]}
        original_snapshot = {"alpha": 0.3, "beta": 1.0, "weights": [1, 2, 3]}

        def modifier(state):
            state["alpha"] = 999.0
            state["weights"].append(999)
            state["new_key"] = "injected"
            return state

        result = sandbox.run_with_copy(original, modifier)
        assert result.success is True
        assert original == original_snapshot

    def test_numpy_state_not_modified(self):
        sandbox = Sandbox()
        original = np.array([1.0, 2.0, 3.0])
        original_copy = original.copy()

        def modifier(state):
            state[:] = 0.0
            return state

        result = sandbox.run_with_copy(original, modifier)
        assert result.success is True
        np.testing.assert_array_equal(original, original_copy)

    def test_nested_dict_not_modified(self):
        sandbox = Sandbox()
        original = {
            "params": {"lr": 0.001, "momentum": 0.9},
            "state_vector": np.zeros(32),
            "history": [1, 2, 3],
        }
        import copy
        snapshot = copy.deepcopy(original)

        def modifier(state):
            state["params"]["lr"] = 99.0
            state["state_vector"][0] = 42.0
            state["history"].clear()
            return state["params"]["lr"]

        result = sandbox.run_with_copy(original, modifier)
        assert result.success is True
        assert result.return_value == 99.0
        assert original["params"]["lr"] == snapshot["params"]["lr"]
        np.testing.assert_array_equal(original["state_vector"], snapshot["state_vector"])
        assert original["history"] == snapshot["history"]

    def test_crash_in_modifier_preserves_original(self):
        sandbox = Sandbox()
        original = {"value": 42}

        def crashing_modifier(state):
            state["value"] = -1
            raise RuntimeError("crash after partial modification")

        result = sandbox.run_with_copy(original, crashing_modifier)
        assert result.success is False
        assert original["value"] == 42

    def test_run_with_copy_returns_modified_result(self):
        sandbox = Sandbox()
        original = {"x": 10}

        def doubler(state):
            state["x"] *= 2
            return state

        result = sandbox.run_with_copy(original, doubler)
        assert result.success is True
        assert result.return_value == {"x": 20}
        assert original["x"] == 10


# ======================================================================
# Simulated optimizer parameter testing
# ======================================================================

class TestOptimizerSimulation:
    """Simulate the real Phase II use case: optimizer proposes param changes."""

    def test_safe_parameter_change(self):
        sandbox = Sandbox()
        current_params = {"blend_weight": 0.5, "ewma_alpha": 0.3, "learning_rate": 0.001}
        proposed = {"blend_weight": 0.6}

        def apply_and_evaluate(params):
            params.update(proposed)
            if not (0.0 <= params["blend_weight"] <= 1.0):
                raise ValueError("blend_weight out of range")
            return {"new_params": params, "improvement": 0.05}

        result = sandbox.run_with_copy(current_params, apply_and_evaluate)
        assert result.success is True
        assert result.return_value["new_params"]["blend_weight"] == 0.6
        assert current_params["blend_weight"] == 0.5

    def test_dangerous_parameter_change_rejected(self):
        sandbox = Sandbox()
        current_params = {"blend_weight": 0.5}
        proposed = {"blend_weight": 5.0}

        def apply_and_evaluate(params):
            params.update(proposed)
            if not (0.0 <= params["blend_weight"] <= 1.0):
                raise ValueError("blend_weight out of range")
            return params

        result = sandbox.run_with_copy(current_params, apply_and_evaluate)
        assert result.success is False
        assert "out of range" in result.error
        assert current_params["blend_weight"] == 0.5

    def test_infinite_loop_optimizer_killed(self):
        sandbox = Sandbox(timeout_s=0.5)
        current_params = {"lr": 0.001}

        def broken_optimizer(params):
            while True:
                params["lr"] *= 1.01

        result = sandbox.run_with_copy(current_params, broken_optimizer)
        assert result.success is False
        assert result.timed_out is True
        assert current_params["lr"] == 0.001


# ======================================================================
# Statistics tracking
# ======================================================================

class TestStatistics:
    def test_initial_counts(self):
        sandbox = Sandbox()
        assert sandbox.run_count == 0
        assert sandbox.success_count == 0
        assert sandbox.failure_count == 0
        assert sandbox.timeout_count == 0
        assert sandbox.success_rate == 0.0

    def test_counts_after_success(self):
        sandbox = Sandbox()
        sandbox.run(lambda: 1)
        assert sandbox.run_count == 1
        assert sandbox.success_count == 1
        assert sandbox.failure_count == 0

    def test_counts_after_failure(self):
        sandbox = Sandbox()
        sandbox.run(lambda: 1 / 0)
        assert sandbox.run_count == 1
        assert sandbox.success_count == 0
        assert sandbox.failure_count == 1

    def test_counts_after_timeout(self):
        sandbox = Sandbox(timeout_s=0.2)
        sandbox.run(lambda: time.sleep(10))
        assert sandbox.run_count == 1
        assert sandbox.timeout_count == 1
        assert sandbox.failure_count == 1

    def test_success_rate(self):
        sandbox = Sandbox()
        sandbox.run(lambda: 1)
        sandbox.run(lambda: 2)
        sandbox.run(lambda: 1 / 0)
        sandbox.run(lambda: 3)
        assert sandbox.success_rate == 0.75

    def test_mixed_run_with_copy_counted(self):
        sandbox = Sandbox()
        sandbox.run_with_copy({"x": 1}, lambda s: s["x"])
        sandbox.run_with_copy({"x": 1}, lambda s: s["missing"])
        assert sandbox.run_count == 2
        assert sandbox.success_count == 1
        assert sandbox.failure_count == 1

    def test_timeout_property(self):
        sandbox = Sandbox(timeout_s=7.5)
        assert sandbox.timeout_s == 7.5


# ======================================================================
# Edge cases
# ======================================================================

class TestEdgeCases:
    def test_returning_exception_object_is_not_failure(self):
        sandbox = Sandbox()
        result = sandbox.run(lambda: ValueError("not raised"))
        assert result.success is True
        assert isinstance(result.return_value, ValueError)

    def test_multiple_sequential_runs(self):
        sandbox = Sandbox()
        for i in range(20):
            result = sandbox.run(lambda: i * 2)
            assert result.success is True

    def test_run_with_copy_uncopyable_state(self):
        sandbox = Sandbox()

        class Uncopyable:
            def __deepcopy__(self, memo):
                raise TypeError("cannot copy this")

        result = sandbox.run_with_copy(Uncopyable(), lambda s: s)
        assert result.success is False
        assert "deep-copy" in result.error.lower() or "copy" in result.error.lower()

    def test_sandbox_result_fields(self):
        result = SandboxResult(success=True, return_value=42, elapsed_ms=1.5)
        assert result.success is True
        assert result.return_value == 42
        assert result.error is None
        assert result.timed_out is False
