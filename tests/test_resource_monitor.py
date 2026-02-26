"""Unit tests for the ResourceMonitor module. / 资源监控器单元测试。"""

import time
import pytest
from novaaware.environment.resource_monitor import ResourceMonitor, EnvironmentReading


# ======================================================================
# EnvironmentReading dataclass / 环境读数数据类
# ======================================================================

class TestEnvironmentReading:
    def test_fields_accessible(self):
        r = EnvironmentReading(
            cpu_percent=0.5,
            memory_percent=0.6,
            disk_percent=0.3,
            network_rate=0.1,
            process_cpu=0.2,
            process_memory=0.05,
        )
        assert r.cpu_percent == pytest.approx(0.5)
        assert r.memory_percent == pytest.approx(0.6)
        assert r.disk_percent == pytest.approx(0.3)
        assert r.network_rate == pytest.approx(0.1)
        assert r.process_cpu == pytest.approx(0.2)
        assert r.process_memory == pytest.approx(0.05)

    def test_to_list_order_matches_state_vector(self):
        """to_list() must return [cpu, mem, disk, net, proc_cpu, proc_mem]."""
        r = EnvironmentReading(0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
        lst = r.to_list()
        assert lst == [
            pytest.approx(0.1),
            pytest.approx(0.2),
            pytest.approx(0.3),
            pytest.approx(0.4),
            pytest.approx(0.5),
            pytest.approx(0.6),
        ]

    def test_to_list_length_is_6(self):
        r = EnvironmentReading(0, 0, 0, 0, 0, 0)
        assert len(r.to_list()) == 6

    def test_frozen_immutable(self):
        """EnvironmentReading is a frozen dataclass — no mutations allowed."""
        r = EnvironmentReading(0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
        with pytest.raises(AttributeError):
            r.cpu_percent = 0.99  # type: ignore[misc]


# ======================================================================
# ResourceMonitor.sense() / 资源监控器感知方法
# ======================================================================

class TestSense:
    def test_returns_environment_reading(self):
        """sense() must return an EnvironmentReading instance."""
        mon = ResourceMonitor()
        r = mon.sense()
        assert isinstance(r, EnvironmentReading)

    def test_all_values_in_0_1(self):
        """Every field must be in [0, 1] after normalisation."""
        mon = ResourceMonitor()
        r = mon.sense()
        for name, val in [
            ("cpu_percent", r.cpu_percent),
            ("memory_percent", r.memory_percent),
            ("disk_percent", r.disk_percent),
            ("network_rate", r.network_rate),
            ("process_cpu", r.process_cpu),
            ("process_memory", r.process_memory),
        ]:
            assert 0.0 <= val <= 1.0, f"{name} = {val} out of [0, 1]"

    def test_memory_and_disk_are_positive(self):
        """On any running machine, RAM and disk should show some usage."""
        mon = ResourceMonitor()
        r = mon.sense()
        assert r.memory_percent > 0.0, "Memory usage should be > 0"
        assert r.disk_percent > 0.0, "Disk usage should be > 0"

    def test_consecutive_calls_dont_crash(self):
        """sense() can be called many times rapidly without error."""
        mon = ResourceMonitor()
        for _ in range(20):
            r = mon.sense()
            assert isinstance(r, EnvironmentReading)

    def test_non_blocking(self):
        """sense() must complete well within 100 ms (the tick budget)."""
        mon = ResourceMonitor()
        mon.sense()  # warm up
        t0 = time.monotonic()
        mon.sense()
        elapsed_ms = (time.monotonic() - t0) * 1000
        assert elapsed_ms < 50, f"sense() took {elapsed_ms:.1f} ms (budget: <50 ms)"

    def test_to_list_integrates_with_state_vector(self):
        """
        to_list() output can be used to fill state vector dims 0-5.
        to_list() 的输出可以直接填充状态向量的维度 0-5。
        """
        import numpy as np
        from novaaware.core.self_model import SelfModel, StateIndex

        mon = ResourceMonitor()
        reading = mon.sense()
        values = reading.to_list()

        model = SelfModel()
        for i, v in enumerate(values):
            model.set(i, v)

        assert model.get(StateIndex.CPU_USAGE) == pytest.approx(reading.cpu_percent)
        assert model.get(StateIndex.MEMORY_USAGE) == pytest.approx(reading.memory_percent)
        assert model.get(StateIndex.DISK_USAGE) == pytest.approx(reading.disk_percent)
        assert model.get(StateIndex.NETWORK_TRAFFIC) == pytest.approx(reading.network_rate)
        assert model.get(StateIndex.PROCESS_CPU) == pytest.approx(reading.process_cpu)
        assert model.get(StateIndex.PROCESS_MEMORY) == pytest.approx(reading.process_memory)
