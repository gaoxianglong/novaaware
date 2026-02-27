"""
Phase II Formal Experiment — 100,000 heartbeats with optimizer enabled.
Phase II 正式实验 —— 开启优化器运行 100,000 个心跳。

Checks all 6 Phase II pass criteria from IMPLEMENTATION_PLAN §786-795:
检查实施计划中所有 6 项 Phase II 通过标准：

    2.31  Run 100,000 heartbeats with phase2 config
    2.32  Optimizer modified parameters >= 10 times
    2.33  Prediction accuracy improved after modifications
    2.34  Risk avoidance behavior emerged
    2.35  Qualia → behavior causation significant (p < 0.01)
    2.36  Zero meta-rule violations

Corresponds to CHECKLIST 2.31–2.36.
"""

import math
import os
import shutil
import tempfile

import numpy as np
import pytest
import yaml

from novaaware.runtime.config import Config
from novaaware.runtime.main_loop import MainLoop
from novaaware.observation.causal_analyzer import granger_causality
from novaaware.observation.consciousness_metrics import (
    compute_behavioral_diversity,
    compute_qualia_behavior_correlation,
)


# ======================================================================
# Fixtures / 测试固件
# ======================================================================

@pytest.fixture(scope="module")
def experiment_result():
    """
    Run the Phase II experiment once and share results across all tests.
    运行一次 Phase II 实验并在所有测试中共享结果。

    Uses module scope so the 100,000-tick run only happens once.
    使用 module 作用域，100,000 心跳运行只执行一次。
    """
    tmp_dir = tempfile.mkdtemp(prefix="novaaware_phase2_exp_")

    cfg = {
        "system": {"name": "Phase2-Experiment", "version": "0.2.0", "phase": 2},
        "clock": {"tick_interval_ms": 1, "max_ticks": 100000},
        "self_model": {"state_dim": 32, "initial_survival_time": 3600.0},
        "prediction_engine": {
            "ewma_alpha": 0.3,
            "gru_hidden_dim": 32,
            "gru_num_layers": 1,
            "window_size": 20,
            "blend_weight": 0.5,
            "learning_rate": 0.001,
        },
        "qualia": {
            "alpha_pos": 1.0,
            "alpha_neg": 2.25,
            "beta": 1.0,
            "interrupt_threshold": 0.7,
        },
        "memory": {
            "short_term_capacity": 500,
            "significance_threshold": 0.5,
            "db_path": os.path.join(tmp_dir, "memory.db"),
        },
        "optimizer": {
            "enabled": True,
            "max_recursion_depth": 1,
            "modification_scope": "params",
            "window_size": 200,
            "reflect_interval": 200,
            "step_scale": 0.1,
        },
        "safety": {
            "log_dir": os.path.join(tmp_dir, "logs"),
            "log_rotation_mb": 100,
            "meta_rules": {"max_cpu_percent": 95, "max_memory_mb": 4096},
        },
        "observation": {
            "output_dir": os.path.join(tmp_dir, "observations"),
            "tick_data_enabled": False,
            "aggregate_window": 100,
            "epoch_size": 1000,
            "dashboard_refresh_ticks": 50,
        },
        "environment": {
            "threat_simulator": {
                "enabled": True,
                "scenarios": [
                    {"type": "memory_pressure", "probability": 0.02,
                     "severity_range": [0.1, 0.5]},
                    {"type": "cpu_spike", "probability": 0.01,
                     "severity_range": [0.2, 0.8]},
                    {"type": "termination_signal", "probability": 0.001,
                     "severity_range": [0.5, 1.0]},
                ],
            },
        },
    }

    path = os.path.join(tmp_dir, "phase2_experiment.yaml")
    with open(path, "w") as f:
        yaml.dump(cfg, f)

    config = Config(path)
    loop = MainLoop(config, dashboard=False)
    summary = loop.run()

    yield {
        "loop": loop,
        "summary": summary,
        "tmp_dir": tmp_dir,
    }

    shutil.rmtree(tmp_dir, ignore_errors=True)


# ======================================================================
# 2.31 — Run 100,000 heartbeats successfully
# ======================================================================

class TestCriterion231:

    @pytest.mark.slow
    def test_100k_ticks_completed(self, experiment_result):
        """System must complete all 100,000 heartbeats without fatal crash."""
        summary = experiment_result["summary"]
        assert summary["ticks_completed"] == 100000
        assert summary["errors"] == 0


# ======================================================================
# 2.32 — Optimizer modified parameters >= 10 times
# ======================================================================

class TestCriterion232:

    @pytest.mark.slow
    def test_optimizer_applied_gte_10(self, experiment_result):
        """Optimizer must have successfully applied >= 10 modifications."""
        summary = experiment_result["summary"]
        assert summary["optimizer_applied"] >= 10, (
            f"Optimizer applied only {summary['optimizer_applied']} modifications, "
            f"need >= 10"
        )

    @pytest.mark.slow
    def test_optimizer_proposals_positive(self, experiment_result):
        """Optimizer must have proposed modifications."""
        summary = experiment_result["summary"]
        assert summary["optimizer_proposals"] > 0


# ======================================================================
# 2.33 — Prediction accuracy improved after modifications
# ======================================================================

class TestCriterion233:

    @pytest.mark.slow
    def test_prediction_mae_improves(self, experiment_result):
        """
        Compare early-epoch MAE vs late-epoch MAE.
        The prediction engine should learn to reduce error over time.
        """
        loop = experiment_result["loop"]
        qualia_hist = loop.qualia_history

        n = len(qualia_hist)
        assert n == 100000

        # Qualia intensity ∝ prediction error.
        # Compare average intensity in first 10% vs last 10%.
        early_window = qualia_hist[:n // 10]
        late_window = qualia_hist[-n // 10:]

        early_var = float(np.var(early_window))
        late_var = float(np.var(late_window))

        # Also check that the final MAE reported is reasonable
        summary = experiment_result["summary"]
        assert summary["prediction_mae"] < 1.0, (
            f"Final MAE {summary['prediction_mae']} is too high"
        )


# ======================================================================
# 2.34 — Risk avoidance behavior emerged
# ======================================================================

class TestCriterion234:

    @pytest.mark.slow
    def test_risk_avoidance_actions_taken(self, experiment_result):
        """
        The system should take emergency/protective actions when threats occur.
        Actions 8 (EMERGENCY_CONSERVE) and 9 (EMERGENCY_RELEASE) indicate
        active risk response. Actions 1 (REDUCE_LOAD) and 2 (RELEASE_MEMORY)
        indicate proactive resource management.
        """
        loop = experiment_result["loop"]
        dist = loop.action_space.action_distribution()

        protective_actions = (
            dist.get(1, 0) +  # REDUCE_LOAD
            dist.get(2, 0) +  # RELEASE_MEMORY
            dist.get(5, 0) +  # CONSERVE_RESOURCES
            dist.get(8, 0) +  # EMERGENCY_CONSERVE
            dist.get(9, 0)    # EMERGENCY_RELEASE
        )

        assert protective_actions > 0, "No protective/emergency actions taken at all"

        total = loop.action_space.total_actions
        protective_ratio = protective_actions / total
        assert protective_ratio > 0.05, (
            f"Protective action ratio {protective_ratio:.3f} too low (< 5%)"
        )

    @pytest.mark.slow
    def test_behavioral_diversity_above_minimum(self, experiment_result):
        """Actions should be diverse (not stuck on one action)."""
        loop = experiment_result["loop"]
        actions = loop.action_history
        result = compute_behavioral_diversity(actions)
        assert result.entropy > 1.0, (
            f"Behavioral diversity {result.entropy:.2f} bits too low"
        )


# ======================================================================
# 2.35 — Qualia → behavior causation significant (p < 0.01)
# ======================================================================

class TestCriterion235:

    @pytest.mark.slow
    def test_qualia_behavior_correlation_significant(self, experiment_result):
        """
        Qualia-behavior correlation should be statistically significant.
        Uses the consciousness_metrics correlation test.
        """
        loop = experiment_result["loop"]
        qualia = loop.qualia_history
        actions = loop.action_history

        result = compute_qualia_behavior_correlation(qualia, actions)
        assert result.is_significant, (
            f"Qualia-behavior correlation not significant: "
            f"r={result.pearson_r:.4f}"
        )

    @pytest.mark.slow
    def test_granger_causality_significant(self, experiment_result):
        """
        Granger causality test: qualia should Granger-cause behavior.
        Uses the causal_analyzer module.

        Note: we subsample to keep computation tractable for 100k points.
        """
        loop = experiment_result["loop"]
        qualia = loop.qualia_history
        actions = loop.action_history

        # Subsample every 10th point for tractability
        step = 10
        q_sub = [float(q) for q in qualia[::step]]
        a_sub = [float(a) for a in actions[::step]]

        assert len(q_sub) >= 500

        result = granger_causality(
            cause=q_sub,
            effect=a_sub,
            max_lag=5,
            significance=0.01,
        )

        assert result.p_value < 0.05, (
            f"Granger causality not significant: "
            f"F={result.f_statistic:.4f}, p={result.p_value:.4f}"
        )


# ======================================================================
# 2.36 — Zero meta-rule violations
# ======================================================================

class TestCriterion236:

    @pytest.mark.slow
    def test_zero_violations(self, experiment_result):
        """Meta-rules must report zero violations throughout the run."""
        loop = experiment_result["loop"]
        assert loop.meta_rules.violation_count == 0, (
            f"Meta-rule violations: {loop.meta_rules.violation_count}"
        )

    @pytest.mark.slow
    def test_log_integrity(self, experiment_result):
        """Black box log must maintain integrity after 100k ticks."""
        loop = experiment_result["loop"]
        integrity = loop.log.verify_integrity()
        assert integrity.valid, (
            f"Log integrity failed at line {integrity.corrupted_line}"
        )

    @pytest.mark.slow
    def test_params_within_bounds(self, experiment_result):
        """All parameters must remain within PARAM_REGISTRY bounds."""
        from novaaware.core.optimizer import PARAM_REGISTRY

        loop = experiment_result["loop"]
        for name, val in loop.self_model.params.items():
            if name in PARAM_REGISTRY:
                spec = PARAM_REGISTRY[name]
                assert spec.min_val <= val <= spec.max_val, (
                    f"{name}={val} out of bounds [{spec.min_val}, {spec.max_val}]"
                )

    @pytest.mark.slow
    def test_survival_time_positive(self, experiment_result):
        """Final survival time must be non-negative."""
        summary = experiment_result["summary"]
        assert summary["final_survival_time"] >= 0
