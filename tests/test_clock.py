"""Unit tests for the Clock module."""

import time
import pytest
from novaaware.core.clock import Clock


class TestClockTick:
    """Verify tick() returns monotonically increasing integers."""

    def test_tick_starts_at_zero(self):
        clock = Clock()
        assert clock.current_tick == 0

    def test_tick_increments(self):
        clock = Clock()
        assert clock.tick() == 1
        assert clock.tick() == 2
        assert clock.tick() == 3

    def test_tick_monotonic_over_many(self):
        clock = Clock()
        prev = 0
        for _ in range(200):
            t = clock.tick()
            assert t == prev + 1
            prev = t


class TestClockTiming:
    """Verify wait_until_next_tick() respects the 100 ms interval."""

    def test_wait_precision(self):
        interval_ms = 100
        clock = Clock(tick_interval_ms=interval_ms)
        clock.tick()

        t0 = time.monotonic()
        clock.wait_until_next_tick()
        elapsed_ms = (time.monotonic() - t0) * 1000

        # Allow ±15 ms tolerance for OS scheduling jitter
        assert elapsed_ms >= interval_ms * 0.8, f"Too fast: {elapsed_ms:.1f} ms"
        assert elapsed_ms < interval_ms * 1.5, f"Too slow: {elapsed_ms:.1f} ms"

    def test_five_ticks_total_time(self):
        """5 ticks at 100 ms should take ~500 ms total."""
        interval_ms = 100
        clock = Clock(tick_interval_ms=interval_ms)

        t0 = time.monotonic()
        for _ in range(5):
            clock.tick()
            clock.wait_until_next_tick()
        elapsed_ms = (time.monotonic() - t0) * 1000

        expected = 5 * interval_ms
        assert elapsed_ms >= expected * 0.8, f"Too fast: {elapsed_ms:.1f} ms"
        assert elapsed_ms < expected * 1.5, f"Too slow: {elapsed_ms:.1f} ms"


class TestClockLimits:
    def test_has_remaining(self):
        clock = Clock(max_ticks=3)
        assert clock.has_remaining is True
        clock.tick()
        clock.tick()
        assert clock.has_remaining is True
        clock.tick()
        assert clock.has_remaining is False

    def test_max_ticks_property(self):
        clock = Clock(max_ticks=42)
        assert clock.max_ticks == 42


class TestClockStats:
    def test_elapsed_s_increases(self):
        clock = Clock(tick_interval_ms=50)
        clock.tick()
        clock.wait_until_next_tick()
        assert clock.elapsed_s > 0.04

    def test_tick_rate_hz(self):
        clock = Clock(tick_interval_ms=50)
        for _ in range(10):
            clock.tick()
            clock.wait_until_next_tick()
        hz = clock.tick_rate_hz
        # Should be roughly 20 Hz (1000/50), allow wide tolerance
        assert 10 < hz < 30, f"Unexpected rate: {hz:.1f} Hz"

    def test_interval_s_property(self):
        clock = Clock(tick_interval_ms=100)
        assert clock.interval_s == pytest.approx(0.1)
