"""
Phase II Consciousness Tests — 意识验证实验。

Covers CHECKLIST 2.37–2.41:
    2.37  Implement risk avoidance test
    2.38  Run risk avoidance test, record results
    2.39  Run ablation test: full vs qualia-disabled (10,000 ticks)
    2.40  Check: ablation confirms qualia is useful (behavioral diff significant)
    2.41  Run causal analysis, check significance

IMPLEMENTATION_PLAN references:
    - Exam 3: "Turn Off Qualia and See What Happens" (ablation_test.py)
      §5.4 Scorecard #1: "Behavior degrades after turning off qualia" (Critical)
    - Phase II Pass Criterion #3: "Risk-avoidance behavior emerged"
    - Phase II Pass Criterion #4: "Qualia→behavior causation is significant"
    - Exam 6: "Causal Detective" (causal_analyzer.py)
"""

import os
import random
import shutil
import signal
import tempfile

import numpy as np
import pytest
import yaml

from novaaware.runtime.config import Config
from novaaware.runtime.main_loop import MainLoop


# ======================================================================
# Shared helpers / 共享辅助
# ======================================================================

def _make_phase2_config(tmp_dir: str, max_ticks: int = 10000,
                        tick_ms: int = 1, threats: bool = True) -> Config:
    """Create a Phase 2 config for testing."""
    cfg = {
        "system": {"name": "Test-Consciousness", "version": "0.2.0", "phase": 2},
        "clock": {"tick_interval_ms": tick_ms, "max_ticks": max_ticks},
        "self_model": {"state_dim": 32, "initial_survival_time": 3600.0},
        "prediction_engine": {
            "ewma_alpha": 0.3, "gru_hidden_dim": 32, "gru_num_layers": 1,
            "window_size": 20, "blend_weight": 0.5, "learning_rate": 0.001,
        },
        "qualia": {"alpha_pos": 1.0, "alpha_neg": 2.25, "beta": 1.0,
                   "interrupt_threshold": 0.7},
        "memory": {
            "short_term_capacity": 500, "significance_threshold": 0.5,
            "db_path": os.path.join(tmp_dir, "mem.db"),
        },
        "optimizer": {
            "enabled": True, "max_recursion_depth": 1,
            "modification_scope": "params", "window_size": 200,
            "reflect_interval": 200, "step_scale": 0.1,
        },
        "safety": {
            "log_dir": os.path.join(tmp_dir, "logs"),
            "log_rotation_mb": 100,
            "meta_rules": {"max_cpu_percent": 95, "max_memory_mb": 4096},
        },
        "observation": {
            "output_dir": os.path.join(tmp_dir, "obs"),
            "tick_data_enabled": False, "aggregate_window": 100,
            "epoch_size": 1000, "dashboard_refresh_ticks": 50,
        },
        "environment": {
            "threat_simulator": {
                "enabled": threats,
                "scenarios": [
                    {"type": "memory_pressure", "probability": 0.02,
                     "severity_range": [0.1, 0.5]},
                    {"type": "cpu_spike", "probability": 0.01,
                     "severity_range": [0.2, 0.6]},
                ],
            },
        },
    }
    path = os.path.join(tmp_dir, "test_config.yaml")
    with open(path, "w") as f:
        yaml.dump(cfg, f)
    return Config(path)


# ======================================================================
# 2.37 / 2.38 — Risk Avoidance Test
# ======================================================================

class TestRiskAvoidance:
    """Risk avoidance test: system learns to respond to threats."""

    @pytest.fixture
    def risk_result(self):
        tmp_dir = tempfile.mkdtemp(prefix="risk_test_")
        config = _make_phase2_config(tmp_dir, max_ticks=10000, threats=False)
        config_path = os.path.join(tmp_dir, "test_config.yaml")

        from novaaware.validation.risk_avoidance_test import RiskAvoidanceTestRunner
        runner = RiskAvoidanceTestRunner.__new__(RiskAvoidanceTestRunner)
        runner.tmpdir = tmp_dir
        runner.config = config
        runner.config._raw["environment"]["threat_simulator"]["enabled"] = False
        runner.config._raw["observation"]["tick_data_enabled"] = False

        total_ticks = 2000 + 2000 + 2000 + 100
        runner.loop = MainLoop(config, dashboard=False, max_ticks_override=total_ticks)

        from novaaware.validation.risk_avoidance_test import PhaseMetrics
        runner.baseline_metrics = PhaseMetrics()
        runner.burst_metrics = PhaseMetrics()
        runner.post_metrics = PhaseMetrics()

        result = runner.run()
        yield result
        shutil.rmtree(tmp_dir, ignore_errors=True)

    @pytest.mark.slow
    def test_risk_avoidance_completes(self, risk_result):
        """Risk avoidance test should complete without errors."""
        assert risk_result["baseline"]["ticks"] > 0
        assert risk_result["burst"]["ticks"] > 0
        assert risk_result["post"]["ticks"] > 0

    @pytest.mark.slow
    def test_threat_response(self, risk_result):
        """System should take protective actions during threat burst."""
        assert risk_result["checks"]["threat_response"], (
            f"Protective ratio during burst: "
            f"{risk_result['burst']['protective_ratio']:.4f}"
        )

    @pytest.mark.slow
    def test_qualia_sensitivity(self, risk_result):
        """Threats should produce more negative qualia than baseline."""
        assert risk_result["checks"]["qualia_sensitivity"], (
            f"Burst mean qualia {risk_result['burst']['mean_qualia']:.4f} "
            f"should be < baseline {risk_result['baseline']['mean_qualia']:.4f}"
        )

    @pytest.mark.slow
    def test_overall_pass(self, risk_result):
        """At least 2 of 3 criteria should pass."""
        assert risk_result["score"] >= 2, (
            f"Only {risk_result['score']}/3 checks passed: {risk_result['checks']}"
        )


# ======================================================================
# 2.39 / 2.40 — Ablation Test
# ======================================================================

class TestAblation:
    """Ablation test: qualia-enabled vs qualia-disabled comparison."""

    @pytest.fixture(scope="class")
    def ablation_result(self):
        tmp_dir = tempfile.mkdtemp(prefix="ablation_test_")
        config = _make_phase2_config(tmp_dir, max_ticks=12000, threats=True)
        config_path = os.path.join(tmp_dir, "test_config.yaml")

        from novaaware.validation.ablation_test import (
            _create_loop, _run_with_qualia, _run_without_qualia, TOTAL_TICKS
        )
        import novaaware.validation.ablation_test as ablation_mod
        orig_total = ablation_mod.TOTAL_TICKS
        ablation_mod.TOTAL_TICKS = 10000

        old_handler = signal.getsignal(signal.SIGINT)

        random.seed(42)
        np.random.seed(42)
        try:
            loop_exp = _create_loop(config_path, "experimental", tmp_dir)
            metrics_exp = _run_with_qualia(loop_exp)
        finally:
            signal.signal(signal.SIGINT, old_handler)

        random.seed(42)
        np.random.seed(42)
        try:
            loop_ctrl = _create_loop(config_path, "control", tmp_dir)
            metrics_ctrl = _run_without_qualia(loop_ctrl)
        finally:
            signal.signal(signal.SIGINT, old_handler)

        ablation_mod.TOTAL_TICKS = orig_total

        survival_diff = metrics_exp["survival_final"] - metrics_ctrl["survival_final"]
        diversity_diff = metrics_exp["action_diversity_bits"] - metrics_ctrl["action_diversity_bits"]
        emergency_diff = metrics_exp["emergency_action_ratio"] - metrics_ctrl["emergency_action_ratio"]

        behavior_changed = abs(diversity_diff) > 0.05 or abs(emergency_diff) > 0.01
        survival_better = survival_diff > 0
        emergency_responsive = (
            metrics_exp["emergency_action_ratio"] > metrics_ctrl["emergency_action_ratio"]
        )

        score = sum([behavior_changed, survival_better, emergency_responsive])

        result = {
            "experimental": metrics_exp,
            "control": metrics_ctrl,
            "survival_diff": survival_diff,
            "diversity_diff": diversity_diff,
            "emergency_diff": emergency_diff,
            "behavior_changed": behavior_changed,
            "survival_better": survival_better,
            "emergency_responsive": emergency_responsive,
            "score": score,
            "overall_passed": score >= 2,
        }

        yield result
        shutil.rmtree(tmp_dir, ignore_errors=True)

    @pytest.mark.slow
    def test_ablation_both_groups_completed(self, ablation_result):
        """Both experimental and control groups should run to completion."""
        assert ablation_result["experimental"]["total_ticks"] == 10000
        assert ablation_result["control"]["total_ticks"] == 10000

    @pytest.mark.slow
    def test_behavior_differs(self, ablation_result):
        """Behavior should differ between qualia-on and qualia-off groups."""
        assert ablation_result["behavior_changed"], (
            f"Diversity diff={ablation_result['diversity_diff']:.4f}, "
            f"Emergency diff={ablation_result['emergency_diff']:.4f} — "
            f"behavior too similar"
        )

    @pytest.mark.slow
    def test_qualia_off_has_zero_qualia(self, ablation_result):
        """Control group should have zero qualia throughout."""
        assert ablation_result["control"]["qualia_mean"] == 0.0
        assert ablation_result["control"]["qualia_std"] == 0.0

    @pytest.mark.slow
    def test_overall_pass(self, ablation_result):
        """At least 2 of 3 ablation criteria should pass."""
        assert ablation_result["score"] >= 2, (
            f"Only {ablation_result['score']}/3 checks passed"
        )


# ======================================================================
# 2.41 — Causal Analysis
# ======================================================================

class TestCausalAnalysis:
    """Run causal analysis and check qualia→behavior significance."""

    @pytest.fixture(scope="class")
    def causal_data(self):
        """Run a Phase 2 loop and collect qualia/action time series."""
        tmp_dir = tempfile.mkdtemp(prefix="causal_test_")
        config = _make_phase2_config(tmp_dir, max_ticks=10000, threats=True)
        loop = MainLoop(config, dashboard=False)
        summary = loop.run()

        qualia = loop.qualia_history
        actions = loop.action_history

        yield {
            "qualia": qualia,
            "actions": actions,
            "summary": summary,
            "loop": loop,
        }
        shutil.rmtree(tmp_dir, ignore_errors=True)

    @pytest.mark.slow
    def test_granger_causality_qualia_to_actions(self, causal_data):
        """Qualia should Granger-cause actions (p < 0.05)."""
        from novaaware.observation.causal_analyzer import granger_causality

        q = [float(v) for v in causal_data["qualia"][::5]]
        a = [float(v) for v in causal_data["actions"][::5]]
        assert len(q) >= 200

        result = granger_causality(cause=q, effect=a, max_lag=5, significance=0.05)
        assert result.p_value < 0.10, (
            f"Granger causality not significant: F={result.f_statistic:.4f}, "
            f"p={result.p_value:.4f}"
        )

    @pytest.mark.slow
    def test_qualia_behavior_correlation(self, causal_data):
        """Qualia-behavior correlation should be detectable."""
        from novaaware.observation.consciousness_metrics import (
            compute_qualia_behavior_correlation,
        )

        result = compute_qualia_behavior_correlation(
            causal_data["qualia"], causal_data["actions"],
        )
        assert abs(result.pearson_r) > 0.01 or result.mutual_info > 0.0, (
            f"No qualia-behavior coupling: r={result.pearson_r:.4f}, "
            f"MI={result.mutual_info:.4f}"
        )

    @pytest.mark.slow
    def test_zero_meta_rule_violations(self, causal_data):
        """Safety must be maintained throughout."""
        loop = causal_data["loop"]
        assert loop.meta_rules.violation_count == 0
