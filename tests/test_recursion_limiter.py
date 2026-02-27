"""
Unit tests for RecursionLimiter — the L3 safety layer.
递归限制器（L3 安全层）单元测试。

Tests cover:
    1. Depth 0 blocks all reflection (Phase I behavior)
    2. Depth 1 allows one level, blocks depth 2 (Phase II behavior)
    3. Depth properly decrements after exiting guard
    4. Thread safety — each thread has independent depth tracking
    5. Hard ceiling cannot be exceeded even by config
    6. Statistics tracking
    7. Context manager exception safety

Corresponds to CHECKLIST items 2.8 and 2.9.
"""

import threading
import time

import pytest

from novaaware.safety.recursion_limiter import (
    RecursionDepthExceeded,
    RecursionLimiter,
)


# ======================================================================
# Phase I behavior: depth=0 blocks all reflection
# ======================================================================

class TestDepthZero:
    """Phase I: optimizer disabled, no reflection allowed."""

    def test_depth_0_blocks_any_entry(self):
        limiter = RecursionLimiter(max_depth=0)
        with pytest.raises(RecursionDepthExceeded) as exc_info:
            with limiter.guard():
                pass
        assert exc_info.value.attempted == 1
        assert exc_info.value.max_allowed == 0

    def test_check_returns_false_at_depth_0(self):
        limiter = RecursionLimiter(max_depth=0)
        assert limiter.check(required_depth=1) is False

    def test_current_depth_is_0(self):
        limiter = RecursionLimiter(max_depth=0)
        assert limiter.current_depth == 0

    def test_remaining_depth_is_0(self):
        limiter = RecursionLimiter(max_depth=0)
        assert limiter.remaining_depth == 0


# ======================================================================
# Phase II behavior: depth=1 allows one level, blocks depth 2
# ======================================================================

class TestDepthOne:
    """Phase II: optimizer can reflect once, cannot reflect on reflection."""

    def test_depth_1_allows_first_level(self):
        limiter = RecursionLimiter(max_depth=1)
        with limiter.guard() as depth:
            assert depth == 1

    def test_depth_1_blocks_second_level(self):
        limiter = RecursionLimiter(max_depth=1)
        with limiter.guard():
            with pytest.raises(RecursionDepthExceeded) as exc_info:
                with limiter.guard():
                    pass
            assert exc_info.value.attempted == 2
            assert exc_info.value.max_allowed == 1

    def test_depth_resets_after_exit(self):
        limiter = RecursionLimiter(max_depth=1)
        with limiter.guard():
            assert limiter.current_depth == 1
        assert limiter.current_depth == 0

    def test_can_re_enter_after_exit(self):
        limiter = RecursionLimiter(max_depth=1)
        with limiter.guard():
            pass
        with limiter.guard():
            pass
        assert limiter.current_depth == 0

    def test_check_inside_guard(self):
        limiter = RecursionLimiter(max_depth=1)
        assert limiter.check(1) is True
        with limiter.guard():
            assert limiter.check(1) is False

    def test_remaining_depth_inside_guard(self):
        limiter = RecursionLimiter(max_depth=1)
        assert limiter.remaining_depth == 1
        with limiter.guard():
            assert limiter.remaining_depth == 0


# ======================================================================
# Deeper recursion (Phase III scenario)
# ======================================================================

class TestDeeperRecursion:
    def test_depth_3_allows_three_levels(self):
        limiter = RecursionLimiter(max_depth=3)
        with limiter.guard() as d1:
            assert d1 == 1
            with limiter.guard() as d2:
                assert d2 == 2
                with limiter.guard() as d3:
                    assert d3 == 3

    def test_depth_3_blocks_fourth_level(self):
        limiter = RecursionLimiter(max_depth=3)
        with limiter.guard():
            with limiter.guard():
                with limiter.guard():
                    with pytest.raises(RecursionDepthExceeded):
                        with limiter.guard():
                            pass

    def test_depth_fully_unwinds(self):
        limiter = RecursionLimiter(max_depth=3)
        with limiter.guard():
            with limiter.guard():
                with limiter.guard():
                    assert limiter.current_depth == 3
                assert limiter.current_depth == 2
            assert limiter.current_depth == 1
        assert limiter.current_depth == 0


# ======================================================================
# Hard ceiling — config cannot override absolute maximum
# ======================================================================

class TestHardCeiling:
    def test_capped_at_absolute_max(self):
        limiter = RecursionLimiter(max_depth=100)
        assert limiter.max_depth == 5

    def test_negative_depth_raises(self):
        with pytest.raises(ValueError):
            RecursionLimiter(max_depth=-1)

    def test_zero_is_valid(self):
        limiter = RecursionLimiter(max_depth=0)
        assert limiter.max_depth == 0


# ======================================================================
# Thread safety
# ======================================================================

class TestThreadSafety:
    def test_independent_depth_per_thread(self):
        limiter = RecursionLimiter(max_depth=2)
        results = {}

        def thread_work(thread_id):
            with limiter.guard():
                results[f"{thread_id}_outer"] = limiter.current_depth
                time.sleep(0.05)
                with limiter.guard():
                    results[f"{thread_id}_inner"] = limiter.current_depth
            results[f"{thread_id}_after"] = limiter.current_depth

        threads = [threading.Thread(target=thread_work, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        for i in range(4):
            assert results[f"{i}_outer"] == 1
            assert results[f"{i}_inner"] == 2
            assert results[f"{i}_after"] == 0

    def test_blocked_in_one_thread_doesnt_affect_another(self):
        limiter = RecursionLimiter(max_depth=1)
        blocked = []
        allowed = []

        def thread_a():
            with limiter.guard():
                try:
                    with limiter.guard():
                        blocked.append(False)
                except RecursionDepthExceeded:
                    blocked.append(True)

        def thread_b():
            time.sleep(0.02)
            with limiter.guard():
                allowed.append(True)

        ta = threading.Thread(target=thread_a)
        tb = threading.Thread(target=thread_b)
        ta.start()
        tb.start()
        ta.join()
        tb.join()

        assert blocked == [True]
        assert allowed == [True]


# ======================================================================
# Statistics
# ======================================================================

class TestStatistics:
    def test_initial_stats(self):
        limiter = RecursionLimiter(max_depth=1)
        assert limiter.total_entries == 0
        assert limiter.blocked_count == 0
        assert limiter.peak_depth == 0

    def test_entries_counted(self):
        limiter = RecursionLimiter(max_depth=2)
        with limiter.guard():
            with limiter.guard():
                pass
        assert limiter.total_entries == 2

    def test_blocked_counted(self):
        limiter = RecursionLimiter(max_depth=1)
        with limiter.guard():
            for _ in range(3):
                try:
                    with limiter.guard():
                        pass
                except RecursionDepthExceeded:
                    pass
        assert limiter.blocked_count == 3

    def test_peak_depth_tracked(self):
        limiter = RecursionLimiter(max_depth=3)
        with limiter.guard():
            with limiter.guard():
                pass
        assert limiter.peak_depth == 2

        with limiter.guard():
            with limiter.guard():
                with limiter.guard():
                    pass
        assert limiter.peak_depth == 3


# ======================================================================
# Exception safety — depth unwinds even on exception
# ======================================================================

class TestExceptionSafety:
    def test_depth_unwinds_on_exception(self):
        limiter = RecursionLimiter(max_depth=2)
        try:
            with limiter.guard():
                with limiter.guard():
                    raise RuntimeError("optimizer crashed")
        except RuntimeError:
            pass
        assert limiter.current_depth == 0

    def test_can_re_enter_after_exception(self):
        limiter = RecursionLimiter(max_depth=1)
        try:
            with limiter.guard():
                raise ValueError("bad param")
        except ValueError:
            pass

        with limiter.guard():
            assert limiter.current_depth == 1

    def test_depth_resets_after_recursion_exceeded(self):
        limiter = RecursionLimiter(max_depth=1)
        with limiter.guard():
            try:
                with limiter.guard():
                    pass
            except RecursionDepthExceeded:
                pass
            assert limiter.current_depth == 1
        assert limiter.current_depth == 0


# ======================================================================
# RecursionDepthExceeded exception
# ======================================================================

class TestExceptionContent:
    def test_exception_message(self):
        limiter = RecursionLimiter(max_depth=1)
        with limiter.guard():
            try:
                with limiter.guard():
                    pass
            except RecursionDepthExceeded as e:
                assert e.attempted == 2
                assert e.max_allowed == 1
                assert "RECURSION LIMIT" in str(e)
                assert "metacognitive escalation" in str(e)

    def test_exception_is_catchable(self):
        limiter = RecursionLimiter(max_depth=0)
        caught = False
        try:
            with limiter.guard():
                pass
        except RecursionDepthExceeded:
            caught = True
        assert caught is True
