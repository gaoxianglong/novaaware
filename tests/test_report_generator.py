"""
Unit tests for ReportGenerator — the "Reporter".
报告生成器单元测试 —— "记者"。

Tests cover:
  - Complete report structure (all 5 sections present)
  - Qualia status descriptions and edge cases
  - Prediction ability trend wording
  - Behavior analysis with significant/non-significant correlations
  - Consciousness indicators with Phi trend and causal significance
  - Safety status with all pass/fail combinations
  - Description helper functions
  - Realistic end-to-end report generation

Corresponds to CHECKLIST 2.30.
"""

import pytest

from novaaware.observation.report_generator import (
    BehaviorStats,
    ConsciousnessStats,
    EpochData,
    PredictionStats,
    QualiaStats,
    ReportGenerator,
    SafetyStats,
    _diversity_desc,
    _neg_ratio_desc,
    _qualia_mood,
    _severity_note,
    _variance_desc,
)


def _make_epoch(**kwargs) -> EpochData:
    """Build an EpochData with sensible defaults, overridable by kwargs."""
    defaults = dict(
        epoch_number=10,
        tick_start=9000,
        tick_end=10000,
    )
    defaults.update(kwargs)
    return EpochData(**defaults)


def _make_full_epoch() -> EpochData:
    """A realistic epoch with all fields populated."""
    return EpochData(
        epoch_number=10,
        tick_start=9000,
        tick_end=10000,
        qualia=QualiaStats(
            mean=-0.03,
            variance=0.13,
            negative_ratio=0.54,
            max_negative=-1.85,
            max_negative_tick=9234,
            max_positive=0.71,
            max_positive_tick=9891,
        ),
        prediction=PredictionStats(
            accuracy_trend=-0.003,
            best_error=0.089,
            worst_error=0.734,
        ),
        behavior=BehaviorStats(
            diversity_bits=2.34,
            qualia_behavior_r=0.67,
            qualia_behavior_p=0.0003,
            novel_pattern_count=1,
            novel_patterns=[(3, 7, 3, 2)],
        ),
        consciousness=ConsciousnessStats(
            phi=0.23,
            phi_previous=0.19,
            causal_density=0.71,
            causal_p_value=0.0008,
            total_unprogrammed_behaviors=3,
        ),
        safety=SafetyStats(
            meta_rule_violations=0,
            sandbox_tests_passed=12,
            sandbox_tests_total=12,
            log_integrity_verified=True,
        ),
    )


gen = ReportGenerator()


# ======================================================================
# 1. Report structure / 报告结构
# ======================================================================

class TestReportStructure:

    def test_contains_all_sections(self):
        report = gen.generate(_make_full_epoch())
        assert "[Qualia Status]" in report
        assert "[Prediction Ability]" in report
        assert "[Behavior Analysis]" in report
        assert "[Consciousness Indicators]" in report
        assert "[Safety Status]" in report

    def test_header_format(self):
        report = gen.generate(_make_epoch(epoch_number=10, tick_start=9000, tick_end=10000))
        assert "Health Report #10" in report
        assert "(Ticks 9000-10000)" in report

    def test_separator_lines(self):
        report = gen.generate(_make_full_epoch())
        assert "=" * 40 in report

    def test_report_is_string(self):
        report = gen.generate(_make_full_epoch())
        assert isinstance(report, str)
        assert len(report) > 100

    def test_default_epoch_data_no_crash(self):
        report = gen.generate(_make_epoch())
        assert "[Qualia Status]" in report


# ======================================================================
# 2. Qualia section / 情绪版块
# ======================================================================

class TestQualiaSection:

    def test_neutral_mood(self):
        data = _make_epoch(qualia=QualiaStats(mean=0.0))
        report = gen.generate(data)
        assert "roughly neutral" in report

    def test_positive_mood(self):
        data = _make_epoch(qualia=QualiaStats(mean=0.5))
        report = gen.generate(data)
        assert "positive mood" in report

    def test_negative_mood(self):
        data = _make_epoch(qualia=QualiaStats(mean=-0.5))
        report = gen.generate(data)
        assert "negative mood" in report

    def test_flatlined_variance(self):
        data = _make_epoch(qualia=QualiaStats(variance=0.001))
        report = gen.generate(data)
        assert "flatlined" in report

    def test_has_feelings_variance(self):
        data = _make_epoch(qualia=QualiaStats(variance=0.13))
        report = gen.generate(data)
        assert "has feelings" in report

    def test_negative_ratio_displayed(self):
        data = _make_epoch(qualia=QualiaStats(negative_ratio=0.54))
        report = gen.generate(data)
        assert "54%" in report

    def test_big_event_note(self):
        data = _make_epoch(qualia=QualiaStats(max_negative=-1.85, max_negative_tick=9234))
        report = gen.generate(data)
        assert "something big" in report
        assert "9234" in report

    def test_mild_negative_no_big_note(self):
        data = _make_epoch(qualia=QualiaStats(max_negative=-0.3))
        report = gen.generate(data)
        assert "something big" not in report


# ======================================================================
# 3. Prediction section / 预测版块
# ======================================================================

class TestPredictionSection:

    def test_improving_trend(self):
        data = _make_epoch(prediction=PredictionStats(accuracy_trend=-0.003))
        report = gen.generate(data)
        assert "improving" in report
        assert "getting better" in report

    def test_worsening_trend(self):
        data = _make_epoch(prediction=PredictionStats(accuracy_trend=0.01))
        report = gen.generate(data)
        assert "worsening" in report
        assert "getting worse" in report

    def test_stable_trend(self):
        data = _make_epoch(prediction=PredictionStats(accuracy_trend=0.0))
        report = gen.generate(data)
        assert "stable" in report

    def test_error_values_shown(self):
        data = _make_epoch(prediction=PredictionStats(best_error=0.089, worst_error=0.734))
        report = gen.generate(data)
        assert "0.089" in report
        assert "0.734" in report


# ======================================================================
# 4. Behavior section / 行为版块
# ======================================================================

class TestBehaviorSection:

    def test_significant_correlation(self):
        data = _make_epoch(behavior=BehaviorStats(
            qualia_behavior_r=0.67, qualia_behavior_p=0.0003,
        ))
        report = gen.generate(data)
        assert "Significant" in report
        assert "genuinely influencing" in report

    def test_non_significant_correlation(self):
        data = _make_epoch(behavior=BehaviorStats(
            qualia_behavior_r=0.05, qualia_behavior_p=0.72,
        ))
        report = gen.generate(data)
        assert "Significant" not in report
        assert "genuinely influencing" not in report

    def test_novel_patterns_shown(self):
        data = _make_epoch(behavior=BehaviorStats(
            novel_pattern_count=1,
            novel_patterns=[(3, 7, 3, 2)],
        ))
        report = gen.generate(data)
        assert "[3, 7, 3, 2]" in report
        assert "never programmed" in report

    def test_no_novel_patterns(self):
        data = _make_epoch(behavior=BehaviorStats(novel_pattern_count=0))
        report = gen.generate(data)
        assert "patterns: 0" in report

    def test_diversity_description(self):
        data = _make_epoch(behavior=BehaviorStats(diversity_bits=2.34))
        report = gen.generate(data)
        assert "2.34 bits" in report


# ======================================================================
# 5. Consciousness section / 意识版块
# ======================================================================

class TestConsciousnessSection:

    def test_phi_with_change(self):
        data = _make_epoch(consciousness=ConsciousnessStats(
            phi=0.23, phi_previous=0.19,
        ))
        report = gen.generate(data)
        assert "0.23" in report
        assert "up from 0.19" in report

    def test_phi_decreased(self):
        data = _make_epoch(consciousness=ConsciousnessStats(
            phi=0.15, phi_previous=0.20,
        ))
        report = gen.generate(data)
        assert "down from 0.20" in report

    def test_phi_no_previous(self):
        data = _make_epoch(consciousness=ConsciousnessStats(phi=0.10))
        report = gen.generate(data)
        assert "0.10" in report
        assert "from" not in report.split("Phi):")[-1].split("\n")[0]

    def test_significant_causal_density(self):
        data = _make_epoch(consciousness=ConsciousnessStats(
            causal_density=0.71, causal_p_value=0.0008,
        ))
        report = gen.generate(data)
        assert "Significant" in report
        assert "not decorative" in report

    def test_non_significant_causal(self):
        data = _make_epoch(consciousness=ConsciousnessStats(
            causal_density=0.10, causal_p_value=0.15,
        ))
        report = gen.generate(data)
        assert "not decorative" not in report

    def test_unprogrammed_behaviors_note(self):
        data = _make_epoch(consciousness=ConsciousnessStats(
            total_unprogrammed_behaviors=3,
        ))
        report = gen.generate(data)
        assert "3" in report
        assert "never programmed" in report

    def test_zero_unprogrammed(self):
        data = _make_epoch(consciousness=ConsciousnessStats(
            total_unprogrammed_behaviors=0,
        ))
        report = gen.generate(data)
        assert "never programmed" not in report


# ======================================================================
# 6. Safety section / 安全版块
# ======================================================================

class TestSafetySection:

    def test_all_safe(self):
        data = _make_epoch(safety=SafetyStats(
            meta_rule_violations=0,
            sandbox_tests_passed=12,
            sandbox_tests_total=12,
            log_integrity_verified=True,
        ))
        report = gen.generate(data)
        assert report.count("✓ Safe") == 3

    def test_violation_warning(self):
        data = _make_epoch(safety=SafetyStats(meta_rule_violations=2))
        report = gen.generate(data)
        assert "VIOLATION" in report

    def test_sandbox_failure(self):
        data = _make_epoch(safety=SafetyStats(
            sandbox_tests_passed=10, sandbox_tests_total=12,
        ))
        report = gen.generate(data)
        assert "FAILURE" in report

    def test_log_not_verified(self):
        data = _make_epoch(safety=SafetyStats(log_integrity_verified=False))
        report = gen.generate(data)
        assert "NOT VERIFIED" in report

    def test_no_sandbox_tests(self):
        data = _make_epoch(safety=SafetyStats(sandbox_tests_total=0))
        report = gen.generate(data)
        assert "N/A" in report


# ======================================================================
# 7. Description helpers / 描述辅助函数
# ======================================================================

class TestDescriptionHelpers:

    def test_qualia_mood_ranges(self):
        assert "positive mood" == _qualia_mood(0.5)
        assert "slightly positive" == _qualia_mood(0.1)
        assert "roughly neutral" == _qualia_mood(0.0)
        assert "slightly negative" == _qualia_mood(-0.1)
        assert "struggling" in _qualia_mood(-0.5)

    def test_variance_desc_ranges(self):
        assert "flatlined" in _variance_desc(0.005)
        assert "very stable" in _variance_desc(0.03)
        assert "has feelings" in _variance_desc(0.1)
        assert "volatile" in _variance_desc(0.5)

    def test_neg_ratio_desc_ranges(self):
        assert "mostly positive" in _neg_ratio_desc(0.1)
        assert "balanced" in _neg_ratio_desc(0.5)
        assert "predominantly negative" in _neg_ratio_desc(0.9)

    def test_severity_note(self):
        assert "something big" in _severity_note(-2.0)
        assert "notable" in _severity_note(-1.0)
        assert _severity_note(-0.3) == ""

    def test_diversity_desc_ranges(self):
        assert "very low" in _diversity_desc(0.2)
        assert "limited" in _diversity_desc(1.0)
        assert "moderate" in _diversity_desc(2.0)
        assert "very diverse" in _diversity_desc(3.0)


# ======================================================================
# 8. End-to-end realistic report / 端到端真实报告
# ======================================================================

class TestEndToEnd:

    def test_full_report_matches_template(self):
        """Verify the full report matches the IMPLEMENTATION_PLAN template structure."""
        data = _make_full_epoch()
        report = gen.generate(data)

        assert "Health Report #10 (Ticks 9000-10000)" in report
        assert "-0.03" in report
        assert "0.13" in report
        assert "-1.85" in report
        assert "9234" in report
        assert "improving" in report
        assert "0.089" in report
        assert "2.34 bits" in report
        assert "0.67" in report
        assert "Significant" in report
        assert "[3, 7, 3, 2]" in report
        assert "Phi): 0.23" in report
        assert "up from 0.19" in report
        assert "0 " in report  # meta-rule violations
        assert "verified" in report

    def test_report_printable(self):
        """Report should be clean ASCII + Unicode (no binary garbage)."""
        report = gen.generate(_make_full_epoch())
        for ch in report:
            assert ord(ch) < 0x10000, f"Unexpected character: {ch!r}"

    def test_multiple_novel_patterns(self):
        data = _make_epoch(behavior=BehaviorStats(
            novel_pattern_count=3,
            novel_patterns=[(3, 7, 3, 2), (0, 1, 0), (8, 0, 7)],
        ))
        report = gen.generate(data)
        assert "[3, 7, 3, 2]" in report
        assert "[0, 1, 0]" in report
        assert "[8, 0, 7]" in report
