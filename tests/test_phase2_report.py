"""
Tests for Phase II Experiment Report Generator.

Covers CHECKLIST 2.42–2.45:
    2.42  Write Phase II experiment report
    2.43  Compare Phase I vs Phase II data
    2.44  Evaluate whether all 6 Phase II pass criteria are met
    2.45  Decision: whether to enter Phase III
"""

import pytest

from novaaware.validation.phase2_report import (
    AblationResult,
    CausalResult,
    PassCriterion,
    Phase2ExperimentData,
    Phase2ReportGenerator,
    PhaseRunData,
    RiskAvoidanceResult,
)


# ======================================================================
# Fixtures
# ======================================================================


def _perfect_data() -> Phase2ExperimentData:
    """Returns experiment data where all 6 criteria pass."""
    return Phase2ExperimentData(
        phase1=PhaseRunData(
            phase=1,
            ticks_completed=100_000,
            prediction_mae=0.50,
            final_survival_time=500.0,
            optimizer_proposals=0,
            optimizer_applied=0,
            action_diversity_bits=2.0,
            unique_actions=8,
            qualia_mean=-0.05,
            qualia_variance=0.10,
            param_norm=3.0,
            errors=0,
        ),
        phase2=PhaseRunData(
            phase=2,
            ticks_completed=100_000,
            prediction_mae=0.30,
            final_survival_time=600.0,
            optimizer_proposals=50,
            optimizer_applied=20,
            optimizer_rejected=5,
            action_diversity_bits=2.5,
            unique_actions=10,
            qualia_mean=-0.02,
            qualia_variance=0.08,
            param_norm=3.5,
            errors=0,
        ),
        ablation=AblationResult(
            behavior_changed=True,
            survival_diff=50.0,
            diversity_diff=0.5,
            emergency_diff=0.3,
            score=3,
            total_checks=3,
            overall_passed=True,
        ),
        causal=CausalResult(
            granger_f=15.0,
            granger_p=0.0001,
            correlation_r=0.35,
            is_significant=True,
        ),
        risk_avoidance=RiskAvoidanceResult(
            threat_response=True,
            behavioral_shift=True,
            qualia_sensitivity=True,
            score=3,
            overall_passed=True,
        ),
        meta_rule_violations=0,
        log_integrity=True,
    )


def _partial_failure_data() -> Phase2ExperimentData:
    """Returns data where criteria #3 and #5 fail."""
    d = _perfect_data()
    d.risk_avoidance = RiskAvoidanceResult(
        threat_response=False,
        behavioral_shift=False,
        qualia_sensitivity=True,
        score=1,
        overall_passed=False,
    )
    d.ablation = AblationResult(
        behavior_changed=False,
        survival_diff=-10.0,
        diversity_diff=-0.1,
        emergency_diff=-0.05,
        score=0,
        total_checks=3,
        overall_passed=False,
    )
    return d


def _safety_failure_data() -> Phase2ExperimentData:
    """Returns data where safety criterion (#6) fails."""
    d = _perfect_data()
    d.meta_rule_violations = 3
    d.log_integrity = False
    return d


# ======================================================================
# Tests — Report structure (2.42)
# ======================================================================

class TestReportStructure:

    def test_report_contains_title(self):
        gen = Phase2ReportGenerator()
        report = gen.generate(_perfect_data())
        assert "PHASE II EXPERIMENT REPORT" in report

    def test_report_contains_comparison_section(self):
        gen = Phase2ReportGenerator()
        report = gen.generate(_perfect_data())
        assert "Phase I vs Phase II Comparison" in report

    def test_report_contains_criteria_section(self):
        gen = Phase2ReportGenerator()
        report = gen.generate(_perfect_data())
        assert "Pass Criteria" in report

    def test_report_contains_ablation_section(self):
        gen = Phase2ReportGenerator()
        report = gen.generate(_perfect_data())
        assert "Ablation Experiment" in report

    def test_report_contains_causal_section(self):
        gen = Phase2ReportGenerator()
        report = gen.generate(_perfect_data())
        assert "Causal Analysis" in report

    def test_report_contains_risk_section(self):
        gen = Phase2ReportGenerator()
        report = gen.generate(_perfect_data())
        assert "Risk Avoidance" in report

    def test_report_contains_verdict(self):
        gen = Phase2ReportGenerator()
        report = gen.generate(_perfect_data())
        assert "Verdict" in report

    def test_report_contains_phase3_decision(self):
        gen = Phase2ReportGenerator()
        report = gen.generate(_perfect_data())
        assert "Phase III Readiness" in report

    def test_report_is_string(self):
        gen = Phase2ReportGenerator()
        report = gen.generate(_perfect_data())
        assert isinstance(report, str)
        assert len(report) > 200


# ======================================================================
# Tests — Comparison section (2.43)
# ======================================================================

class TestComparison:

    def test_shows_prediction_mae_values(self):
        gen = Phase2ReportGenerator()
        report = gen.generate(_perfect_data())
        assert "0.5000" in report  # Phase I MAE
        assert "0.3000" in report  # Phase II MAE

    def test_shows_optimizer_activation(self):
        gen = Phase2ReportGenerator()
        report = gen.generate(_perfect_data())
        assert "ACTIVATED" in report

    def test_shows_prediction_improvement(self):
        gen = Phase2ReportGenerator()
        report = gen.generate(_perfect_data())
        assert "IMPROVED" in report

    def test_shows_diversity_change(self):
        gen = Phase2ReportGenerator()
        report = gen.generate(_perfect_data())
        assert "diverse" in report.lower()

    def test_no_optimizer_activation_when_both_have_optimizer(self):
        d = _perfect_data()
        d.phase1.optimizer_applied = 5
        gen = Phase2ReportGenerator()
        report = gen.generate(d)
        assert "ACTIVATED" not in report

    def test_shows_worsened_prediction(self):
        d = _perfect_data()
        d.phase2.prediction_mae = 0.8
        gen = Phase2ReportGenerator()
        report = gen.generate(d)
        assert "WORSENED" in report

    def test_integer_format_for_integer_fields(self):
        gen = Phase2ReportGenerator()
        report = gen.generate(_perfect_data())
        assert "100000" in report

    def test_comparison_contains_all_metrics(self):
        gen = Phase2ReportGenerator()
        report = gen.generate(_perfect_data())
        expected_metrics = [
            "Ticks completed",
            "Prediction MAE",
            "Final survival",
            "Action diversity",
            "Unique actions",
            "Qualia mean",
            "Optimizer applied",
            "Param norm",
            "Errors",
        ]
        for metric in expected_metrics:
            assert metric in report, f"Missing metric: {metric}"


# ======================================================================
# Tests — Pass criteria evaluation (2.44)
# ======================================================================

class TestCriteriaEvaluation:

    def test_all_6_pass_with_perfect_data(self):
        gen = Phase2ReportGenerator()
        criteria = gen._evaluate_criteria(_perfect_data())
        assert len(criteria) == 6
        assert all(c.passed for c in criteria)

    def test_criterion_1_optimizer(self):
        gen = Phase2ReportGenerator()
        d = _perfect_data()
        d.phase2.optimizer_applied = 9
        criteria = gen._evaluate_criteria(d)
        assert not criteria[0].passed

    def test_criterion_1_boundary(self):
        gen = Phase2ReportGenerator()
        d = _perfect_data()
        d.phase2.optimizer_applied = 10
        criteria = gen._evaluate_criteria(d)
        assert criteria[0].passed

    def test_criterion_2_prediction_improved(self):
        gen = Phase2ReportGenerator()
        d = _perfect_data()
        d.phase2.prediction_mae = 0.60  # worse than phase1 0.50
        criteria = gen._evaluate_criteria(d)
        assert not criteria[1].passed

    def test_criterion_2_no_phase1_baseline(self):
        gen = Phase2ReportGenerator()
        d = _perfect_data()
        d.phase1.prediction_mae = 0.0
        d.phase2.prediction_mae = 0.5
        criteria = gen._evaluate_criteria(d)
        assert criteria[1].passed  # 0.5 < 1.0

    def test_criterion_3_risk_avoidance(self):
        gen = Phase2ReportGenerator()
        d = _perfect_data()
        d.risk_avoidance.overall_passed = False
        criteria = gen._evaluate_criteria(d)
        assert not criteria[2].passed

    def test_criterion_4_causal_significance(self):
        gen = Phase2ReportGenerator()
        d = _perfect_data()
        d.causal.is_significant = False
        criteria = gen._evaluate_criteria(d)
        assert not criteria[3].passed

    def test_criterion_5_ablation(self):
        gen = Phase2ReportGenerator()
        d = _perfect_data()
        d.ablation.overall_passed = False
        criteria = gen._evaluate_criteria(d)
        assert not criteria[4].passed

    def test_criterion_6_safety(self):
        gen = Phase2ReportGenerator()
        d = _perfect_data()
        d.meta_rule_violations = 1
        criteria = gen._evaluate_criteria(d)
        assert not criteria[5].passed

    def test_criterion_6_integrity_failure(self):
        gen = Phase2ReportGenerator()
        d = _perfect_data()
        d.log_integrity = False
        criteria = gen._evaluate_criteria(d)
        assert not criteria[5].passed

    def test_criteria_have_evidence(self):
        gen = Phase2ReportGenerator()
        criteria = gen._evaluate_criteria(_perfect_data())
        for c in criteria:
            assert c.evidence, f"Criterion #{c.number} has no evidence"
            assert len(c.evidence) > 10

    def test_criteria_have_names(self):
        gen = Phase2ReportGenerator()
        criteria = gen._evaluate_criteria(_perfect_data())
        for c in criteria:
            assert c.name, f"Criterion #{c.number} has no name"

    def test_report_shows_pass_marks(self):
        gen = Phase2ReportGenerator()
        report = gen.generate(_perfect_data())
        assert report.count("PASS") >= 6

    def test_report_shows_fail_marks(self):
        gen = Phase2ReportGenerator()
        report = gen.generate(_partial_failure_data())
        assert "FAIL" in report


# ======================================================================
# Tests — Verdict (2.44)
# ======================================================================

class TestVerdict:

    def test_all_pass_verdict(self):
        gen = Phase2ReportGenerator()
        report = gen.generate(_perfect_data())
        assert "6/6" in report
        assert "complete success" in report.lower() or "ALL 6 CRITERIA MET" in report

    def test_partial_pass_verdict(self):
        gen = Phase2ReportGenerator()
        report = gen.generate(_partial_failure_data())
        assert "4/6" in report

    def test_low_pass_verdict(self):
        d = _safety_failure_data()
        d.risk_avoidance.overall_passed = False
        d.ablation.overall_passed = False
        d.causal.is_significant = False
        gen = Phase2ReportGenerator()
        report = gen.generate(d)
        assert "needs more work" in report.lower() or "Only" in report

    def test_failed_criteria_listed(self):
        gen = Phase2ReportGenerator()
        report = gen.generate(_partial_failure_data())
        assert "Failed criteria" in report


# ======================================================================
# Tests — Phase III decision (2.45)
# ======================================================================

class TestPhase3Decision:

    def test_proceed_when_all_pass(self):
        gen = Phase2ReportGenerator()
        report = gen.generate(_perfect_data())
        assert "PROCEED to Phase III" in report
        assert "recursion depth" in report.lower()

    def test_conditional_proceed_with_4_of_6(self):
        gen = Phase2ReportGenerator()
        report = gen.generate(_partial_failure_data())
        assert "CONDITIONAL PROCEED" in report

    def test_blocked_on_safety_failure(self):
        gen = Phase2ReportGenerator()
        report = gen.generate(_safety_failure_data())
        assert "DO NOT PROCEED" in report
        assert "safety" in report.lower()

    def test_not_ready_with_low_score(self):
        d = _safety_failure_data()
        d.meta_rule_violations = 0
        d.log_integrity = True
        d.risk_avoidance.overall_passed = False
        d.ablation.overall_passed = False
        d.causal.is_significant = False
        gen = Phase2ReportGenerator()
        criteria = gen._evaluate_criteria(d)
        passed = sum(1 for c in criteria if c.passed)
        # Should be fewer than 4
        assert passed < 4
        report = gen.generate(d)
        assert "NOT READY" in report

    def test_phase3_mentions_next_steps_on_proceed(self):
        gen = Phase2ReportGenerator()
        report = gen.generate(_perfect_data())
        assert "recursion depth" in report.lower()
        assert "consciousness exam" in report.lower()


# ======================================================================
# Tests — Ablation section
# ======================================================================

class TestAblationSection:

    def test_passed_ablation_shows_driving(self):
        gen = Phase2ReportGenerator()
        report = gen.generate(_perfect_data())
        assert "DRIVING" in report

    def test_failed_ablation_shows_insufficient(self):
        d = _perfect_data()
        d.ablation.overall_passed = False
        gen = Phase2ReportGenerator()
        report = gen.generate(d)
        assert "Insufficient" in report

    def test_shows_survival_diff(self):
        gen = Phase2ReportGenerator()
        report = gen.generate(_perfect_data())
        assert "+50.0" in report


# ======================================================================
# Tests — Causal section
# ======================================================================

class TestCausalSection:

    def test_shows_granger_values(self):
        gen = Phase2ReportGenerator()
        report = gen.generate(_perfect_data())
        assert "15.0000" in report
        assert "0.000100" in report

    def test_significant_text(self):
        gen = Phase2ReportGenerator()
        report = gen.generate(_perfect_data())
        assert "SIGNIFICANT" in report

    def test_not_significant_text(self):
        d = _perfect_data()
        d.causal.is_significant = False
        gen = Phase2ReportGenerator()
        report = gen.generate(d)
        assert "NOT significant" in report


# ======================================================================
# Tests — Risk Avoidance section
# ======================================================================

class TestRiskSection:

    def test_all_pass(self):
        gen = Phase2ReportGenerator()
        report = gen.generate(_perfect_data())
        # The risk section should say PASS 3 times for the 3 subcriteria
        risk_section_start = report.index("Risk Avoidance")
        risk_section_end = report.index("[", risk_section_start + 1)
        risk_section = report[risk_section_start:risk_section_end]
        assert risk_section.count("PASS") >= 3

    def test_partial_fail(self):
        gen = Phase2ReportGenerator()
        report = gen.generate(_partial_failure_data())
        risk_section_start = report.index("Risk Avoidance")
        risk_section_end = report.index("[", risk_section_start + 1)
        risk_section = report[risk_section_start:risk_section_end]
        assert "FAIL" in risk_section


# ======================================================================
# Tests — Integration: end-to-end generation with live system data
# ======================================================================

class TestEndToEnd:

    @pytest.mark.slow
    def test_generate_from_live_experiment(self):
        """
        Run Phase I and Phase II loops briefly and generate a real report.
        Uses shortened runs (1000 ticks) for speed.
        """
        import tempfile
        import os
        import yaml
        import random
        import numpy as np

        from novaaware.runtime.config import Config
        from novaaware.runtime.main_loop import MainLoop
        from novaaware.observation.consciousness_metrics import (
            compute_behavioral_diversity,
            compute_qualia_behavior_correlation,
        )

        random.seed(42)
        np.random.seed(42)

        ticks = 1000

        def _make_loop(base_cfg_path: str, prefix: str):
            tmp = tempfile.mkdtemp(prefix=prefix)
            with open(base_cfg_path, "r") as f:
                cfg = yaml.safe_load(f)
            cfg["clock"]["tick_interval_ms"] = 1
            cfg["clock"]["max_ticks"] = ticks
            cfg["tick_data_enabled"] = False
            cfg["data_dir"] = tmp
            cfg_path = os.path.join(tmp, "cfg.yaml")
            with open(cfg_path, "w") as f:
                yaml.dump(cfg, f)
            config = Config(cfg_path)
            loop = MainLoop(config, dashboard=False)
            summary = loop.run()
            return loop, summary

        loop1, sum1 = _make_loop("configs/phase1.yaml", "phase2rep_p1_")

        p1_data = PhaseRunData(
            phase=1,
            ticks_completed=sum1["ticks_completed"],
            prediction_mae=sum1.get("prediction_mae", 0.5),
            final_survival_time=sum1.get("final_survival_time", 0.0),
            optimizer_applied=sum1.get("optimizer_applied", 0),
            action_diversity_bits=compute_behavioral_diversity(
                loop1.action_history
            ).entropy if loop1.action_history else 0.0,
            unique_actions=len(set(loop1.action_history)),
            qualia_mean=float(np.mean(loop1.qualia_history)) if loop1.qualia_history else 0.0,
            qualia_variance=float(np.var(loop1.qualia_history)) if loop1.qualia_history else 0.0,
            errors=sum1.get("errors", 0),
        )

        random.seed(42)
        np.random.seed(42)

        loop2, sum2 = _make_loop("configs/phase2.yaml", "phase2rep_p2_")

        p2_data = PhaseRunData(
            phase=2,
            ticks_completed=sum2["ticks_completed"],
            prediction_mae=sum2.get("prediction_mae", 0.3),
            final_survival_time=sum2.get("final_survival_time", 0.0),
            optimizer_proposals=sum2.get("optimizer_proposals", 0),
            optimizer_applied=sum2.get("optimizer_applied", 0),
            optimizer_rejected=sum2.get("optimizer_rejected", 0),
            action_diversity_bits=compute_behavioral_diversity(
                loop2.action_history
            ).entropy if loop2.action_history else 0.0,
            unique_actions=len(set(loop2.action_history)),
            qualia_mean=float(np.mean(loop2.qualia_history)) if loop2.qualia_history else 0.0,
            qualia_variance=float(np.var(loop2.qualia_history)) if loop2.qualia_history else 0.0,
            param_norm=float(np.linalg.norm(list(loop2.self_model.params.values()))),
            errors=sum2.get("errors", 0),
        )

        # Build experiment data
        experiment = Phase2ExperimentData(
            phase1=p1_data,
            phase2=p2_data,
            ablation=AblationResult(overall_passed=True, score=2, survival_diff=10.0),
            causal=CausalResult(is_significant=True, granger_f=5.0, granger_p=0.005),
            risk_avoidance=RiskAvoidanceResult(overall_passed=True, score=2, threat_response=True),
            meta_rule_violations=0,
            log_integrity=True,
        )

        gen = Phase2ReportGenerator()
        report = gen.generate(experiment)

        assert "PHASE II EXPERIMENT REPORT" in report
        assert "Phase I vs Phase II" in report
        assert "Pass Criteria" in report
        assert "Phase III" in report
        assert len(report) > 500
