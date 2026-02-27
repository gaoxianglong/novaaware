"""Unit tests for the PredictionEngine module. / 预测引擎单元测试。"""

import numpy as np
import pytest
import torch
from novaaware.core.prediction_engine import (
    EWMAPredictor,
    GRUPredictor,
    PredictionEngine,
)


DIM = 32


# ======================================================================
# Layer 1 — EWMA / 第一层 — 指数加权移动平均
# ======================================================================

class TestEWMA:
    def test_predict_before_any_observation_returns_zeros(self):
        ewma = EWMAPredictor(dim=DIM, alpha=0.3)
        pred = ewma.predict()
        assert pred.shape == (DIM,)
        assert np.all(pred == 0.0)

    def test_first_observation_becomes_prediction(self):
        ewma = EWMAPredictor(dim=DIM, alpha=0.3)
        obs = np.ones(DIM)
        ewma.update(obs)
        pred = ewma.predict()
        np.testing.assert_array_almost_equal(pred, obs)

    def test_ewma_formula_correct(self):
        """After two updates, EWMA = α * x2 + (1-α) * x1."""
        ewma = EWMAPredictor(dim=1, alpha=0.4)
        ewma.update(np.array([10.0]))
        ewma.update(np.array([20.0]))
        expected = 0.4 * 20.0 + 0.6 * 10.0  # 14.0
        assert ewma.predict()[0] == pytest.approx(expected)

    def test_higher_alpha_tracks_faster(self):
        """Higher alpha → prediction closer to the latest observation."""
        fast = EWMAPredictor(dim=1, alpha=0.9)
        slow = EWMAPredictor(dim=1, alpha=0.1)
        for v in [1.0, 2.0, 3.0, 10.0]:
            obs = np.array([v])
            fast.update(obs)
            slow.update(obs)
        # fast should be much closer to 10.0
        assert abs(fast.predict()[0] - 10.0) < abs(slow.predict()[0] - 10.0)

    def test_predict_returns_copy(self):
        ewma = EWMAPredictor(dim=DIM, alpha=0.3)
        ewma.update(np.ones(DIM))
        p = ewma.predict()
        p[0] = 999.0
        assert ewma.predict()[0] == pytest.approx(1.0)


# ======================================================================
# Layer 2 — GRU / 第二层 — GRU 神经网络
# ======================================================================

class TestGRU:
    def test_not_ready_with_zero_observations(self):
        gru = GRUPredictor(dim=DIM, window_size=5)
        assert gru.ready is False

    def test_not_ready_with_one_observation(self):
        gru = GRUPredictor(dim=DIM, window_size=5)
        gru.update(np.zeros(DIM))
        assert gru.ready is False

    def test_ready_with_two_observations(self):
        gru = GRUPredictor(dim=DIM, window_size=5)
        gru.update(np.zeros(DIM))
        gru.update(np.ones(DIM))
        assert gru.ready is True

    def test_predict_returns_correct_shape(self):
        gru = GRUPredictor(dim=DIM, window_size=5)
        for _ in range(3):
            gru.update(np.random.randn(DIM))
        pred = gru.predict()
        assert pred.shape == (DIM,)

    def test_predict_before_ready_returns_zeros(self):
        gru = GRUPredictor(dim=DIM, window_size=5)
        pred = gru.predict()
        assert np.all(pred == 0.0)

    def test_learn_returns_loss(self):
        gru = GRUPredictor(dim=DIM, window_size=5, learning_rate=0.01)
        for _ in range(5):
            gru.update(np.random.randn(DIM))
        gru.predict()
        loss = gru.learn(np.random.randn(DIM))
        assert isinstance(loss, float)
        assert loss >= 0.0

    def test_learn_before_predict_returns_zero(self):
        gru = GRUPredictor(dim=DIM, window_size=5)
        loss = gru.learn(np.zeros(DIM))
        assert loss == 0.0

    def test_parameters_exposed(self):
        gru = GRUPredictor(dim=DIM)
        params = gru.parameters
        assert len(params) > 0
        assert all(hasattr(p, 'data') for p in params)


# ======================================================================
# PredictionEngine (unified blend) / 预测引擎（统一混合）
# ======================================================================

class TestPredictionEngine:
    def test_predict_output_shape(self):
        engine = PredictionEngine(dim=DIM)
        engine.observe(np.zeros(DIM))
        pred = engine.predict()
        assert pred.shape == (DIM,)

    def test_predict_before_observe_returns_zeros(self):
        engine = PredictionEngine(dim=DIM)
        pred = engine.predict()
        np.testing.assert_array_equal(pred, np.zeros(DIM))

    def test_observe_increments_tick_count(self):
        engine = PredictionEngine(dim=DIM)
        assert engine.tick_count == 0
        engine.observe(np.zeros(DIM))
        engine.observe(np.ones(DIM))
        assert engine.tick_count == 2

    def test_blend_weight_property(self):
        engine = PredictionEngine(dim=DIM, blend_weight=0.7)
        assert engine.blend_weight == pytest.approx(0.7)

    def test_blend_weight_setter_clamps(self):
        engine = PredictionEngine(dim=DIM)
        engine.blend_weight = 1.5
        assert engine.blend_weight == pytest.approx(1.0)
        engine.blend_weight = -0.3
        assert engine.blend_weight == pytest.approx(0.0)

    def test_early_ticks_use_pure_ewma(self):
        """Before GRU is ready, prediction equals EWMA output."""
        engine = PredictionEngine(dim=DIM, window_size=50, blend_weight=0.5)
        obs = np.full(DIM, 5.0)
        engine.observe(obs)
        pred = engine.predict()
        np.testing.assert_array_almost_equal(pred, obs)

    def test_last_prediction_property(self):
        engine = PredictionEngine(dim=DIM)
        assert engine.last_prediction is None
        engine.observe(np.ones(DIM))
        engine.predict()
        assert engine.last_prediction is not None
        assert engine.last_prediction.shape == (DIM,)

    def test_learn_returns_mae(self):
        engine = PredictionEngine(dim=DIM)
        engine.observe(np.ones(DIM))
        engine.predict()
        mae = engine.learn(np.ones(DIM) * 2.0)
        assert isinstance(mae, float)
        assert mae > 0.0

    def test_50_step_input_produces_32_dim_output(self):
        """Core acceptance: 50 steps history → 32-dim prediction."""
        engine = PredictionEngine(dim=DIM, window_size=50)
        for i in range(50):
            obs = np.random.randn(DIM) * 0.1 + 0.5
            engine.observe(obs)
        pred = engine.predict()
        assert pred.shape == (DIM,)
        assert not np.all(pred == 0.0)


# ======================================================================
# MAE decreasing trend over 1000 steps / 1000 步后 MAE 下降趋势
# ======================================================================

class TestMAEDecreasingTrend:
    def test_mae_decreases_on_learnable_pattern(self):
        """
        Core acceptance criterion: after 1000 ticks on a repeating
        pattern, MAE should show a decreasing trend.
        核心验收标准：在重复模式上跑 1000 个心跳后，MAE 应呈下降趋势。

        We generate a simple periodic signal and verify that
        the average MAE in the last 200 ticks is lower than
        the average MAE in the first 200 ticks.
        我们生成一个简单的周期信号，验证最后 200 个心跳的平均 MAE
        低于前 200 个心跳的平均 MAE。
        """
        np.random.seed(42)
        torch.manual_seed(42)
        engine = PredictionEngine(
            dim=DIM,
            ewma_alpha=0.3,
            gru_hidden_dim=32,
            gru_num_layers=1,
            window_size=20,
            blend_weight=0.5,
            learning_rate=0.005,
        )

        maes: list = []
        total_ticks = 1000
        period = 10

        for t in range(total_ticks):
            # Repeating sinusoidal pattern with slight noise.
            # 带轻微噪声的重复正弦模式。
            phase = 2 * np.pi * (t % period) / period
            obs = np.full(DIM, 0.5 + 0.3 * np.sin(phase), dtype=np.float64)
            obs += np.random.randn(DIM) * 0.01

            engine.observe(obs)
            pred = engine.predict()
            mae = engine.learn(obs)
            maes.append(mae)

        # Compare early vs late MAE.
        # 对比早期和晚期的 MAE。
        early_mae = np.mean(maes[:200])
        late_mae = np.mean(maes[800:1000])

        assert late_mae < early_mae, (
            f"MAE should decrease: early={early_mae:.4f}, late={late_mae:.4f} / "
            f"MAE 应下降：早期={early_mae:.4f}，晚期={late_mae:.4f}"
        )
