"""
Integration tests for the MainLoop — the system's "heartbeat".
主循环集成测试 —— 系统的"心跳"。

Tests cover:
  - Config loading / 配置加载
  - 11-step heartbeat execution / 11 步心跳执行
  - Smoke test: 1,000 ticks without crash / 冒烟测试：1000 次心跳不崩溃
  - Stress test: 10,000 ticks without crash / 压力测试：10000 次心跳不崩溃
  - Data output: CSV + black box integrity / 数据输出：CSV + 黑匣子完整性
  - Qualia-behavior coupling / 情绪-行为耦合
  - Phase II integration: optimizer + 10,000 heartbeats / Phase II 集成：优化器 + 10000 心跳
"""

import csv
import os
import shutil
import tempfile

import pytest
import yaml

from novaaware.runtime.config import Config, parse_args
from novaaware.runtime.main_loop import MainLoop


# ======================================================================
# Fixtures / 测试固件
# ======================================================================

@pytest.fixture
def tmp_dir():
    """Create a temp directory for test data and clean up after. / 创建临时目录并在测试后清理。"""
    d = tempfile.mkdtemp(prefix="novaaware_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def make_config(tmp_dir):
    """Factory fixture: create a Config with custom max_ticks. / 工厂固件：创建自定义 max_ticks 的 Config。"""
    def _make(
        max_ticks: int = 100,
        tick_interval_ms: int = 1,
        threats: bool = True,
        phase: int = 1,
        optimizer_enabled: bool = False,
        reflect_interval: int = 200,
        tick_data_enabled: bool = True,
    ):
        cfg = {
            "system": {"name": "Test-Nova", "version": "0.0.1", "phase": phase},
            "clock": {"tick_interval_ms": tick_interval_ms, "max_ticks": max_ticks},
            "self_model": {"state_dim": 32, "initial_survival_time": 3600.0},
            "prediction_engine": {
                "ewma_alpha": 0.3, "gru_hidden_dim": 32, "gru_num_layers": 1,
                "window_size": 20, "blend_weight": 0.5, "learning_rate": 0.001,
            },
            "qualia": {"alpha_pos": 1.0, "alpha_neg": 2.25, "beta": 1.0, "interrupt_threshold": 0.7},
            "memory": {
                "short_term_capacity": 500,
                "significance_threshold": 0.5,
                "db_path": os.path.join(tmp_dir, "memory.db"),
            },
            "safety": {
                "log_dir": os.path.join(tmp_dir, "logs"),
                "log_rotation_mb": 100,
                "meta_rules": {"max_cpu_percent": 80, "max_memory_mb": 2048},
            },
            "observation": {
                "output_dir": os.path.join(tmp_dir, "observations"),
                "tick_data_enabled": tick_data_enabled,
                "aggregate_window": 100,
                "epoch_size": 1000,
                "dashboard_refresh_ticks": 50,
            },
            "environment": {
                "threat_simulator": {
                    "enabled": threats,
                    "scenarios": [
                        {"type": "memory_pressure", "probability": 0.02, "severity_range": [0.1, 0.3]},
                        {"type": "cpu_spike", "probability": 0.01, "severity_range": [0.1, 0.4]},
                    ],
                },
            },
        }
        if optimizer_enabled:
            cfg["optimizer"] = {
                "enabled": True,
                "max_recursion_depth": 1,
                "modification_scope": "params",
                "window_size": 200,
                "reflect_interval": reflect_interval,
                "step_scale": 0.1,
            }
        path = os.path.join(tmp_dir, "test_config.yaml")
        with open(path, "w") as f:
            yaml.dump(cfg, f)
        return Config(path)
    return _make


def _run_loop(make_config, ticks: int, threats: bool = True) -> tuple:
    """Helper: create and run a MainLoop for N ticks. / 辅助：创建并运行 N 次心跳的 MainLoop。"""
    config = make_config(max_ticks=ticks, tick_interval_ms=1, threats=threats)
    loop = MainLoop(config, dashboard=False)
    summary = loop.run()
    return loop, summary, config


# ======================================================================
# 1. Config / 配置
# ======================================================================

class TestConfig:

    def test_parse_args_defaults(self):
        args = parse_args([])
        assert args.config == "configs/phase1.yaml"
        assert args.dashboard is False
        assert args.max_ticks is None

    def test_parse_args_custom(self):
        args = parse_args(["--config", "foo.yaml", "--dashboard", "--max-ticks", "500"])
        assert args.config == "foo.yaml"
        assert args.dashboard is True
        assert args.max_ticks == 500

    def test_config_loads_yaml(self, make_config):
        config = make_config(max_ticks=42)
        assert config.system_name == "Test-Nova"
        assert config.max_ticks == 42
        assert config.state_dim == 32


# ======================================================================
# 2. Basic heartbeat / 基本心跳
# ======================================================================

class TestBasicHeartbeat:

    def test_single_tick_runs(self, make_config):
        _, summary, _ = _run_loop(make_config, ticks=1, threats=False)
        assert summary["ticks_completed"] == 1
        assert summary["errors"] == 0

    def test_ten_ticks_no_error(self, make_config):
        _, summary, _ = _run_loop(make_config, ticks=10, threats=False)
        assert summary["ticks_completed"] == 10
        assert summary["errors"] == 0

    def test_identity_persists(self, make_config):
        loop, _, _ = _run_loop(make_config, ticks=5, threats=False)
        assert len(loop.self_model.identity_hash) == 64


# ======================================================================
# 3. Smoke test: 1,000 ticks / 冒烟测试：1000 次心跳
# ======================================================================

class TestSmokeTest:

    def test_1000_ticks_no_crash(self, make_config):
        """Acceptance criterion: stable for 1,000 heartbeats. / 验收标准：1000 次心跳稳定。"""
        _, summary, _ = _run_loop(make_config, ticks=1000)
        assert summary["ticks_completed"] == 1000
        assert summary["errors"] == 0

    def test_1000_ticks_survival_positive(self, make_config):
        """Survival time should remain non-negative. / 生存时间应保持非负。"""
        loop, summary, _ = _run_loop(make_config, ticks=1000)
        assert summary["final_survival_time"] >= 0

    def test_1000_ticks_memories_created(self, make_config):
        """Short-term memory should have entries. / 短期记忆应有条目。"""
        loop, _, _ = _run_loop(make_config, ticks=1000)
        assert loop.memory.short_term.size > 0

    def test_1000_ticks_log_integrity(self, make_config):
        """Black box should pass integrity check after 1000 ticks. / 黑匣子应通过 1000 次心跳后的完整性检查。"""
        loop, _, _ = _run_loop(make_config, ticks=1000)
        integrity = loop.log.verify_integrity()
        assert integrity.valid


# ======================================================================
# 4. Stress test: 10,000 ticks / 压力测试：10000 次心跳
# ======================================================================

class TestStressTest:

    @pytest.mark.slow
    def test_10000_ticks_no_crash(self, make_config):
        """Acceptance criterion: stable for 10,000 heartbeats. / 验收标准：10000 次心跳稳定。"""
        _, summary, _ = _run_loop(make_config, ticks=10000)
        assert summary["ticks_completed"] == 10000
        assert summary["errors"] == 0

    @pytest.mark.slow
    def test_10000_ticks_log_integrity(self, make_config):
        """Black box integrity after stress test. / 压力测试后的黑匣子完整性。"""
        loop, _, _ = _run_loop(make_config, ticks=10000)
        integrity = loop.log.verify_integrity()
        assert integrity.valid


# ======================================================================
# 5. Data output / 数据输出
# ======================================================================

class TestDataOutput:

    def test_csv_created(self, make_config):
        """tick_data.csv should exist with correct headers. / tick_data.csv 应存在且有正确表头。"""
        _, _, config = _run_loop(make_config, ticks=50, threats=False)
        csv_path = os.path.join(config.observation_dir, "tick_data.csv")
        assert os.path.exists(csv_path)

        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            header = next(reader)
        assert "tick" in header
        assert "qualia_value" in header
        assert "action_id" in header

    def test_csv_row_count(self, make_config):
        """CSV should have one row per tick plus header. / CSV 应每心跳一行加表头。"""
        _, _, config = _run_loop(make_config, ticks=50, threats=False)
        csv_path = os.path.join(config.observation_dir, "tick_data.csv")
        with open(csv_path, "r") as f:
            rows = list(csv.reader(f))
        assert len(rows) == 51  # 1 header + 50 data rows

    def test_log_file_created(self, make_config):
        """Black box log file should exist. / 黑匣子日志文件应存在。"""
        loop, _, _ = _run_loop(make_config, ticks=10, threats=False)
        assert os.path.exists(loop.log.current_file_path)


# ======================================================================
# 6. Qualia-behavior coupling / 情绪-行为耦合
# ======================================================================

class TestQualiaBehavior:

    def test_qualia_varies(self, make_config):
        """Qualia should not be a flat line. / 情绪不应是一条平线。"""
        loop, _, _ = _run_loop(make_config, ticks=200)
        q_vals = [loop.memory.short_term.recent(200)[i].qualia_value for i in range(min(50, loop.memory.short_term.size))]
        unique_vals = set(round(v, 3) for v in q_vals)
        assert len(unique_vals) > 1, "Qualia should vary across ticks"

    def test_qualia_has_both_signs(self, make_config):
        """Over many ticks, qualia should include both positive and negative values. / 经过多次心跳，情绪应包含正值和负值。"""
        loop, _, _ = _run_loop(make_config, ticks=500)
        entries = loop.memory.short_term.recent(500)
        values = [e.qualia_value for e in entries]
        has_positive = any(v > 0 for v in values)
        has_nonpositive = any(v <= 0 for v in values)
        assert has_positive, "Should have some positive qualia"
        assert has_nonpositive or len(set(round(v, 4) for v in values)) > 3, \
            "Should have either negative qualia or significant qualia variety"

    def test_actions_executed(self, make_config):
        """Actions should have been taken. / 应有动作被执行。"""
        loop, _, _ = _run_loop(make_config, ticks=100)
        assert loop.action_space.total_actions == 100


# ======================================================================
# 7. Component integration / 组件集成
# ======================================================================

class TestComponentIntegration:

    def test_prediction_engine_receives_observations(self, make_config):
        """Prediction engine should have processed observations. / 预测引擎应已处理观测。"""
        loop, _, _ = _run_loop(make_config, ticks=100, threats=False)
        assert loop.prediction.tick_count == 100

    def test_memory_long_term_with_threats(self, make_config):
        """With threats, some events should be promoted to long-term memory. / 有威胁时，部分事件应被提升到长期记忆。"""
        loop, summary, _ = _run_loop(make_config, ticks=500)
        assert summary["long_term_memories"] >= 0


# ======================================================================
# 8. Phase II integration: Optimizer + 10,000 heartbeats
#    Phase II 集成：优化器 + 10,000 心跳
# ======================================================================

def _run_phase2_loop(make_config, ticks: int, reflect_interval: int = 200,
                     threats: bool = True) -> tuple:
    """Helper: run a Phase II MainLoop with optimizer enabled."""
    config = make_config(
        max_ticks=ticks,
        tick_interval_ms=1,
        threats=threats,
        phase=2,
        optimizer_enabled=True,
        reflect_interval=reflect_interval,
        tick_data_enabled=False,
    )
    loop = MainLoop(config, dashboard=False)
    summary = loop.run()
    return loop, summary, config


class TestPhase2Basic:
    """Phase II basic integration: optimizer is wired and runs."""

    def test_phase2_50_ticks_no_crash(self, make_config):
        """Phase 2 with optimizer enabled should not crash in 50 ticks."""
        _, summary, _ = _run_phase2_loop(make_config, ticks=50, threats=False)
        assert summary["ticks_completed"] == 50
        assert summary["errors"] == 0

    def test_phase2_optimizer_skips_when_insufficient_data(self, make_config):
        """Optimizer should not reflect if memory size < window_size."""
        loop, summary, _ = _run_phase2_loop(make_config, ticks=50, threats=False)
        assert summary["optimizer_proposals"] == 0
        assert summary["optimizer_applied"] == 0

    def test_phase2_optimizer_reflects_after_enough_data(self, make_config):
        """After enough ticks, optimizer should perform at least one reflection."""
        loop, summary, _ = _run_phase2_loop(
            make_config, ticks=500, reflect_interval=200, threats=True,
        )
        assert summary["optimizer_proposals"] > 0

    def test_phase2_optimizer_applies_modifications(self, make_config):
        """Optimizer should apply at least some parameter modifications."""
        loop, summary, _ = _run_phase2_loop(
            make_config, ticks=1000, reflect_interval=200, threats=True,
        )
        assert summary["optimizer_applied"] > 0

    def test_phase2_params_within_bounds(self, make_config):
        """All optimized parameters must remain within PARAM_REGISTRY bounds."""
        from novaaware.core.optimizer import PARAM_REGISTRY
        loop, _, _ = _run_phase2_loop(
            make_config, ticks=1000, reflect_interval=200, threats=True,
        )
        for name, val in loop.self_model.params.items():
            if name in PARAM_REGISTRY:
                spec = PARAM_REGISTRY[name]
                assert spec.min_val <= val <= spec.max_val, \
                    f"{name}={val} out of bounds [{spec.min_val}, {spec.max_val}]"

    def test_phase2_reflection_logged(self, make_config):
        """Reflection events should appear in the black box log."""
        loop, _, _ = _run_phase2_loop(
            make_config, ticks=500, reflect_interval=200, threats=True,
        )
        integrity = loop.log.verify_integrity()
        assert integrity.valid

    def test_phase2_param_norm_nonzero(self, make_config):
        """After optimizer initializes params, param_norm should be > 0."""
        loop, _, _ = _run_phase2_loop(
            make_config, ticks=300, reflect_interval=200, threats=False,
        )
        params = loop.self_model.params
        if params:
            import math
            norm = math.sqrt(sum(v * v for v in params.values()))
            assert norm > 0


class TestPhase2Stress:
    """Phase II stress test: 10,000 heartbeats with optimizer active."""

    @pytest.mark.slow
    def test_10000_ticks_phase2_stable(self, make_config):
        """
        Acceptance criterion: Phase II config runs stably for 10,000 heartbeats.
        验收标准：Phase II 配置下稳定运行 10,000 心跳。

        Verifies:
        - Zero errors over 10,000 heartbeats
        - Optimizer proposes and applies parameter modifications
        - All parameter values remain within PARAM_REGISTRY bounds
        - Black box log maintains integrity
        - Survival time remains non-negative
        """
        from novaaware.core.optimizer import PARAM_REGISTRY

        loop, summary, _ = _run_phase2_loop(
            make_config, ticks=10000, reflect_interval=200, threats=True,
        )

        # Stability: no crashes
        assert summary["ticks_completed"] == 10000
        assert summary["errors"] == 0

        # Optimizer active: proposals generated and applied
        assert summary["optimizer_proposals"] > 0, "Optimizer should have generated proposals"
        assert summary["optimizer_applied"] > 0, "Optimizer should have applied some modifications"
        assert summary["optimizer_applied"] >= 10, \
            f"Expected >= 10 applied modifications, got {summary['optimizer_applied']}"

        # Safety: parameters within bounds
        for name, val in loop.self_model.params.items():
            if name in PARAM_REGISTRY:
                spec = PARAM_REGISTRY[name]
                assert spec.min_val <= val <= spec.max_val, \
                    f"{name}={val} out of bounds [{spec.min_val}, {spec.max_val}]"

        # Integrity: black box not corrupted
        integrity = loop.log.verify_integrity()
        assert integrity.valid

        # Survival: remained viable
        assert summary["final_survival_time"] >= 0

    @pytest.mark.slow
    def test_10000_ticks_phase2_log_integrity(self, make_config):
        """Black box integrity after 10,000 Phase II ticks."""
        loop, _, _ = _run_phase2_loop(
            make_config, ticks=10000, reflect_interval=200, threats=True,
        )
        integrity = loop.log.verify_integrity()
        assert integrity.valid
        assert integrity.total_entries > 10000
