"""
Unit tests for EmergenceDetector — the "Surprise Detector".
涌现检测器单元测试 —— "惊喜探测器"。

Tests cover:
  - N-gram counting and recording (2.28)
  - Novel pattern detection via surprise metric (2.27)
  - Behavioral complexity and temporal correlation
  - Edge cases: empty, single, constant sequences
  - Realistic NovaAware simulation scenarios

Corresponds to CHECKLIST 2.27–2.28.
"""

import math

import numpy as np
import pytest

from novaaware.observation.emergence_detector import (
    EmergenceDetector,
    EmergenceReport,
    NovelPattern,
)


# ======================================================================
# 1. Basic observation and recording (2.28)
# ======================================================================

class TestObservation:

    def test_empty_detector(self):
        det = EmergenceDetector()
        assert det.total_observations == 0
        assert det.action_history == []
        report = det.detect()
        assert report.total_patterns_observed == 0
        assert report.novel_patterns == []

    def test_single_observation(self):
        det = EmergenceDetector()
        det.observe(0, 3)
        assert det.total_observations == 1
        assert det.action_history == [3]

    def test_records_action_history(self):
        det = EmergenceDetector()
        for i, a in enumerate([0, 1, 2, 3, 4]):
            det.observe(i, a)
        assert det.action_history == [0, 1, 2, 3, 4]

    def test_ngram_counts_bigram(self):
        det = EmergenceDetector(ngram_sizes=[2])
        for i, a in enumerate([0, 1, 0, 1, 0]):
            det.observe(i, a)
        counts = det.all_ngram_counts[2]
        assert counts[(0, 1)] == 2
        assert counts[(1, 0)] == 2

    def test_ngram_counts_trigram(self):
        det = EmergenceDetector(ngram_sizes=[3])
        for i, a in enumerate([0, 1, 2, 0, 1, 2]):
            det.observe(i, a)
        counts = det.all_ngram_counts[3]
        assert counts[(0, 1, 2)] == 2

    def test_first_seen_tick_recorded(self):
        det = EmergenceDetector(ngram_sizes=[2], min_pattern_count=1,
                                surprise_threshold=-100)
        det.observe(10, 3)
        det.observe(11, 7)
        det.observe(12, 3)
        det.observe(13, 7)
        report = det.detect()
        pattern_37 = [p for p in report.novel_patterns if p.pattern == (3, 7)]
        assert len(pattern_37) == 1
        assert pattern_37[0].first_seen_tick == 11

    def test_multiple_ngram_sizes(self):
        det = EmergenceDetector(ngram_sizes=[2, 3])
        for i in range(10):
            det.observe(i, i % 3)
        counts_2 = det.all_ngram_counts[2]
        counts_3 = det.all_ngram_counts[3]
        assert len(counts_2) > 0
        assert len(counts_3) > 0


# ======================================================================
# 2. Novel pattern detection (2.27)
# ======================================================================

class TestNovelPatternDetection:

    def test_uniform_random_no_novel_patterns(self):
        """Uniform random actions → no significant temporal structure."""
        np.random.seed(42)
        det = EmergenceDetector(ngram_sizes=[2, 3], min_pattern_count=3,
                                surprise_threshold=2.0)
        for i in range(500):
            det.observe(i, int(np.random.randint(0, 10)))
        report = det.detect()
        # With 10 actions, any specific bigram has ~1% chance.
        # Over 500 observations, most bigrams appear ~5 times.
        # Under independence, surprise should be low.
        assert len(report.novel_patterns) < 20

    def test_repeating_pattern_detected(self):
        """A repeating cycle should produce high-surprise n-grams."""
        det = EmergenceDetector(ngram_sizes=[2, 3], min_pattern_count=3,
                                surprise_threshold=1.0)
        cycle = [3, 7, 3, 2]
        for i in range(200):
            det.observe(i, cycle[i % len(cycle)])
        report = det.detect()
        assert len(report.novel_patterns) > 0

        patterns = [p.pattern for p in report.novel_patterns]
        assert (3, 7) in patterns or (7, 3) in patterns

    def test_constant_action_no_novel_patterns(self):
        """Same action every tick → bigrams are (a, a), which is expected."""
        det = EmergenceDetector(ngram_sizes=[2], min_pattern_count=3,
                                surprise_threshold=1.0)
        for i in range(100):
            det.observe(i, 5)
        report = det.detect()
        # Only one unique bigram (5, 5). Under independence with p(5)=1,
        # expected = observed → surprise = 0.
        for p in report.novel_patterns:
            assert p.surprise < 1.0

    def test_surprise_calculation_correct(self):
        """Verify surprise = log₂(observed / expected)."""
        det = EmergenceDetector(ngram_sizes=[2], min_pattern_count=1,
                                surprise_threshold=-100)
        # 50% action 0, 50% action 1, but always alternating 0,1,0,1
        for i in range(100):
            det.observe(i, i % 2)

        report = det.detect()
        bigram_01 = [p for p in report.novel_patterns if p.pattern == (0, 1)]
        assert len(bigram_01) == 1

        p = bigram_01[0]
        # p(0) = p(1) = 0.5, so expected under independence:
        # E[(0,1)] = 99 * 0.5 * 0.5 = 24.75
        # Observed: 50
        # Surprise = log₂(50 / 24.75) ≈ 1.015
        assert 0.9 < p.surprise < 1.1

    def test_high_surprise_for_structured_data(self):
        """Highly structured sequence → high surprise."""
        det = EmergenceDetector(ngram_sizes=[3], min_pattern_count=3,
                                surprise_threshold=0.5)
        # Always [0, 0, 1] repeated, but unigram dist is 2/3 zero, 1/3 one
        pattern = [0, 0, 1]
        for i in range(300):
            det.observe(i, pattern[i % 3])
        report = det.detect()
        # The trigram (0, 0, 1) should be very surprising
        novel = [p for p in report.novel_patterns if p.pattern == (0, 0, 1)]
        assert len(novel) == 1
        assert novel[0].surprise > 1.0

    def test_novel_patterns_sorted_by_surprise(self):
        det = EmergenceDetector(ngram_sizes=[2], min_pattern_count=1,
                                surprise_threshold=-100)
        cycle = [0, 1, 2, 3]
        for i in range(200):
            det.observe(i, cycle[i % 4])
        report = det.detect()
        if len(report.novel_patterns) >= 2:
            surprises = [p.surprise for p in report.novel_patterns]
            for i in range(len(surprises) - 1):
                assert surprises[i] >= surprises[i + 1]

    def test_min_pattern_count_filter(self):
        """Patterns below min_pattern_count should not appear."""
        det = EmergenceDetector(ngram_sizes=[2], min_pattern_count=5,
                                surprise_threshold=-100)
        # Only 3 repetitions of (0, 1) → below threshold of 5
        for i in range(6):
            det.observe(i, i % 2)
        report = det.detect()
        for p in report.novel_patterns:
            assert p.count >= 5


# ======================================================================
# 3. Behavioral complexity / 行为复杂性
# ======================================================================

class TestBehavioralComplexity:

    def test_constant_action_low_complexity(self):
        det = EmergenceDetector(ngram_sizes=[2])
        for i in range(100):
            det.observe(i, 0)
        report = det.detect()
        assert report.behavioral_complexity < 0.01

    def test_diverse_actions_higher_complexity(self):
        det1 = EmergenceDetector(ngram_sizes=[2])
        det2 = EmergenceDetector(ngram_sizes=[2])
        for i in range(200):
            det1.observe(i, 0)
            det2.observe(i, i % 5)
        r1 = det1.detect()
        r2 = det2.detect()
        assert r2.behavioral_complexity > r1.behavioral_complexity

    def test_complexity_between_0_and_1(self):
        det = EmergenceDetector(ngram_sizes=[2, 3])
        np.random.seed(42)
        for i in range(300):
            det.observe(i, int(np.random.randint(0, 5)))
        report = det.detect()
        assert 0.0 <= report.behavioral_complexity <= 1.0


# ======================================================================
# 4. Temporal correlation / 时间相关性
# ======================================================================

class TestTemporalCorrelation:

    def test_random_actions_low_autocorrelation(self):
        """Independent random actions → autocorrelation ≈ 0."""
        np.random.seed(42)
        det = EmergenceDetector()
        for i in range(500):
            det.observe(i, int(np.random.randint(0, 10)))
        report = det.detect()
        assert abs(report.temporal_correlation) < 0.15

    def test_alternating_pattern_negative_autocorrelation(self):
        """Alternating 0, 1, 0, 1 → negative autocorrelation."""
        det = EmergenceDetector()
        for i in range(200):
            det.observe(i, i % 2)
        report = det.detect()
        assert report.temporal_correlation < -0.5

    def test_constant_action_zero_autocorrelation(self):
        det = EmergenceDetector()
        for i in range(100):
            det.observe(i, 5)
        report = det.detect()
        assert report.temporal_correlation == 0.0

    def test_strongly_correlated_sequence(self):
        """Slowly changing action → positive autocorrelation."""
        det = EmergenceDetector()
        for i in range(200):
            det.observe(i, i // 20)
        report = det.detect()
        assert report.temporal_correlation > 0.5

    def test_insufficient_data_returns_zero(self):
        det = EmergenceDetector()
        for i in range(5):
            det.observe(i, i)
        report = det.detect()
        assert report.temporal_correlation == 0.0


# ======================================================================
# 5. get_patterns_by_size / 按大小获取模式
# ======================================================================

class TestPatternsBySize:

    def test_filter_by_bigram(self):
        det = EmergenceDetector(ngram_sizes=[2, 3], min_pattern_count=1,
                                surprise_threshold=-100)
        cycle = [0, 1, 2]
        for i in range(90):
            det.observe(i, cycle[i % 3])
        bigrams = det.get_patterns_by_size(2)
        trigrams = det.get_patterns_by_size(3)
        for p in bigrams:
            assert len(p.pattern) == 2
        for p in trigrams:
            assert len(p.pattern) == 3

    def test_no_patterns_for_unsupported_size(self):
        det = EmergenceDetector(ngram_sizes=[2])
        for i in range(50):
            det.observe(i, i % 3)
        assert det.get_patterns_by_size(5) == []


# ======================================================================
# 6. Realistic NovaAware simulation / 真实 NovaAware 模拟
# ======================================================================

class TestRealisticScenario:

    def test_qualia_driven_action_cycle_detected(self):
        """
        Simulate: negative qualia → emergency action (8),
        then recovery → no-op (0), then explore (7).
        This creates a repeating cycle [8, 0, 7] that is NOT
        programmed (action_space is memoryless).
        """
        det = EmergenceDetector(ngram_sizes=[2, 3], min_pattern_count=3,
                                surprise_threshold=0.5)
        cycle = [8, 0, 7]
        for i in range(300):
            det.observe(i, cycle[i % 3])
        report = det.detect()
        assert len(report.novel_patterns) > 0

        trigrams = [p for p in report.novel_patterns if len(p.pattern) == 3]
        trigram_patterns = [p.pattern for p in trigrams]
        assert (8, 0, 7) in trigram_patterns or (0, 7, 8) in trigram_patterns

    def test_mixed_random_and_structured(self):
        """
        First 200 ticks random, then structured [0, 1, 0, 1] cycle.
        The structured part should create detectable temporal patterns.
        """
        np.random.seed(42)
        det = EmergenceDetector(ngram_sizes=[2], min_pattern_count=3,
                                surprise_threshold=1.0)
        for i in range(200):
            det.observe(i, int(np.random.randint(0, 10)))
        for i in range(200, 600):
            det.observe(i, i % 2)
        report = det.detect()
        bigram_01 = [p for p in report.novel_patterns if p.pattern == (0, 1)]
        assert len(bigram_01) > 0

    def test_novel_pattern_count_property(self):
        det = EmergenceDetector(ngram_sizes=[2], min_pattern_count=3,
                                surprise_threshold=0.5)
        cycle = [0, 1, 2]
        for i in range(300):
            det.observe(i, cycle[i % 3])
        assert det.novel_pattern_count > 0

    def test_report_fields_complete(self):
        np.random.seed(42)
        det = EmergenceDetector()
        for i in range(100):
            det.observe(i, int(np.random.randint(0, 5)))
        report = det.detect()
        assert isinstance(report.total_patterns_observed, int)
        assert isinstance(report.novel_patterns, list)
        assert isinstance(report.behavioral_complexity, float)
        assert isinstance(report.temporal_correlation, float)
