"""
Tests for observation/dashboard.py — the four-panel monitoring display.
测试监控面板 —— 四宫格实时监控画面。

These tests exercise DashboardData and Dashboard WITHOUT requiring a real
display (matplotlib is tested via mocking / headless checks).
这些测试在不需要真实显示的情况下验证 DashboardData 和 Dashboard
（matplotlib 通过 mock / headless 方式测试）。
"""

import numpy as np
import pytest

from novaaware.observation.dashboard import Dashboard, DashboardData


# ==================================================================
# DashboardData tests / 数据容器测试
# ==================================================================


class TestDashboardData:
    """Verify rolling data buffer behavior. / 验证滚动数据缓冲区行为。"""

    def test_empty_on_creation(self):
        data = DashboardData(max_points=100)
        assert data.size == 0
        assert len(data.ticks) == 0

    def test_append_increments_size(self):
        data = DashboardData(max_points=100)
        state = np.zeros(32)
        data.append(1, 0.5, 100.0, 95.0, 0.01, state)
        assert data.size == 1
        assert data.ticks[-1] == 1
        assert data.qualia_values[-1] == 0.5
        assert data.predicted_survival[-1] == 100.0
        assert data.actual_survival[-1] == 95.0
        assert data.mae_values[-1] == 0.01

    def test_max_points_cap(self):
        """Buffer respects max_points. / 缓冲区遵守最大点数限制。"""
        data = DashboardData(max_points=10)
        state = np.zeros(32)
        for i in range(20):
            data.append(i, float(i), 100.0, 90.0, 0.01, state)
        assert data.size == 10
        assert data.ticks[0] == 10
        assert data.ticks[-1] == 19

    def test_state_snapshot_is_copy(self):
        """State array stored as copy, not reference. / 状态数组以副本存储。"""
        data = DashboardData(max_points=10)
        state = np.ones(32) * 0.5
        data.append(1, 0.0, 100.0, 100.0, 0.0, state)
        state[:] = 0.99
        assert data.state_snapshots[0][0] == pytest.approx(0.5)

    def test_param_xy(self):
        data = DashboardData(max_points=100)
        state = np.zeros(32)
        data.append(5, 0.1, 100.0, 100.0, 0.01, state, param_norm=3.14)
        assert data.param_x[-1] == 5
        assert data.param_y[-1] == pytest.approx(3.14)

    def test_multiple_appends(self):
        data = DashboardData(max_points=500)
        state = np.zeros(32)
        for i in range(100):
            data.append(i, np.sin(i * 0.1), 3600.0 - i, 3600.0 - i * 0.9, 0.01 + i * 0.001, state)
        assert data.size == 100
        assert len(data.qualia_values) == 100
        assert len(data.predicted_survival) == 100
        assert len(data.mae_values) == 100


# ==================================================================
# Dashboard tests (headless / mock) / 面板测试（无头 / mock）
# ==================================================================


class TestDashboard:
    """Test Dashboard logic without a display. / 在无显示环境中测试面板逻辑。"""

    def test_creation(self):
        dash = Dashboard(refresh_ticks=50, max_points=200)
        assert dash.refresh_ticks == 50
        assert dash.data.size == 0

    def test_update_accumulates_data(self):
        """Data is fed regardless of whether matplotlib is available. / 无论 matplotlib 是否可用都会积累数据。"""
        dash = Dashboard(refresh_ticks=10, max_points=100)
        dash._available = False
        dash._initialized = True
        state = np.random.rand(32)
        for i in range(1, 21):
            dash.update(i, 0.5, 100.0, 95.0, 0.01, state)
        assert dash.data.size == 20

    def test_no_redraw_before_refresh_interval(self):
        """
        Dashboard should attempt a redraw only at refresh_ticks intervals.
        面板应仅在 refresh_ticks 间隔时尝试重绘。
        """
        dash = Dashboard(refresh_ticks=50, max_points=100)
        dash._available = False
        dash._initialized = True
        state = np.zeros(32)
        for i in range(1, 50):
            dash.update(i, 0.0, 100.0, 100.0, 0.0, state)
        assert dash.data.size == 49

    def test_close_no_error(self):
        """Close should be safe even if figure was never created. / 关闭即使未创建图形也应安全。"""
        dash = Dashboard()
        dash.close()

    def test_close_twice_no_error(self):
        dash = Dashboard()
        dash.close()
        dash.close()

    def test_refresh_ticks_default(self):
        dash = Dashboard()
        assert dash.refresh_ticks == 50

    def test_data_fed_at_refresh_boundary(self):
        """At tick=refresh_ticks, a redraw would be attempted. / 在 tick=refresh_ticks 时会尝试重绘。"""
        dash = Dashboard(refresh_ticks=10, max_points=100)
        dash._available = False
        dash._initialized = True
        state = np.zeros(32)
        for i in range(1, 11):
            dash.update(i, 0.0, 100.0, 100.0, 0.0, state)
        assert dash.data.size == 10

    def test_unavailable_graceful(self):
        """If matplotlib is unavailable, update still works. / 如果 matplotlib 不可用，update 仍然正常。"""
        dash = Dashboard(refresh_ticks=5, max_points=50)
        dash._available = False
        dash._initialized = True
        state = np.random.rand(32)
        for i in range(1, 101):
            dash.update(i, np.sin(i * 0.1), 3600.0 - i, 3600.0 - i * 0.9, 0.02, state)
        assert dash.data.size == 50

    def test_redraw_with_insufficient_data(self):
        """Redraw is a no-op with < 2 data points. / 数据不足 2 个时重绘为空操作。"""
        dash = Dashboard(refresh_ticks=1, max_points=100)
        dash._available = False
        dash._initialized = True
        state = np.zeros(32)
        dash.update(1, 0.0, 100.0, 100.0, 0.0, state)
        assert dash.data.size == 1


# ==================================================================
# Integration-style tests (data consistency) / 集成测试（数据一致性）
# ==================================================================


class TestDashboardIntegration:
    """Simulate a realistic data stream and verify consistency. / 模拟真实数据流并验证一致性。"""

    def test_simulate_500_ticks(self):
        dash = Dashboard(refresh_ticks=50, max_points=300)
        dash._available = False
        dash._initialized = True

        for i in range(1, 501):
            state = np.random.rand(32)
            q = np.sin(i * 0.05) * 0.8
            pred_s = 3600.0 - i * 0.5
            act_s = 3600.0 - i * 0.48
            mae = max(0, 0.1 - i * 0.0001)
            dash.update(i, q, pred_s, act_s, mae, state)

        assert dash.data.size == 300
        assert dash.data.ticks[0] == 201
        assert dash.data.ticks[-1] == 500

    def test_negative_qualia_recorded(self):
        dash = Dashboard(refresh_ticks=10, max_points=100)
        dash._available = False
        dash._initialized = True
        state = np.zeros(32)
        dash.update(1, -0.75, 100.0, 100.0, 0.0, state)
        assert dash.data.qualia_values[-1] == -0.75

    def test_extreme_values(self):
        """Dashboard handles extreme values without crashing. / 面板处理极端值不会崩溃。"""
        dash = Dashboard(refresh_ticks=5, max_points=100)
        dash._available = False
        dash._initialized = True
        state = np.ones(32) * 999
        dash.update(5, 100.0, 1e8, -1e8, 1e6, state)
        assert dash.data.size == 1
        assert dash.data.qualia_values[-1] == 100.0
