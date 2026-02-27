"""
EmergenceDetector — the "Surprise Detector".
涌现检测器 —— "惊喜探测器"。

Discovers behaviors the system exhibits that were never explicitly
programmed.  The core insight: the action_space.py code is *memoryless*
— it selects actions based only on the current state vector, with no
memory of previous actions.  Therefore, any temporal structure in the
action sequence (repeated n-grams appearing more often than chance)
is **emergent** behavior driven by qualia feedback, optimizer tuning,
or environmental dynamics — not by direct programming.

发现系统展现的、从未被明确编程的行为。核心洞察：action_space.py
的代码是 *无记忆的* — 它只根据当前状态向量选择动作，不记忆之前
的动作。因此，动作序列中的任何时间结构（重复 n-gram 出现频率
高于随机）都是 **涌现** 行为，由情绪反馈、优化器调参或环境动态
驱动 — 而非直接编程。

Two capabilities / 两项功能：

    2.27  Detect unprogrammed behaviors (action n-gram analysis)
          检测未编程行为（动作 n-gram 分析）

    2.28  Record novel behavior patterns
          记录新行为模式

Paper: §5.4 Scorecard #3 "Unprogrammed novel behaviors appeared" (Critical)
论文：§5.4 记分卡第 3 项 "出现未编程的新行为" (关键)

IMPLEMENTATION_PLAN: §5.2 Layer 3 epoch report / Phase II Step 6.
对应实施计划 §5.2 第三层体检报告 / Phase II 第 6 步。

Corresponds to CHECKLIST 2.27–2.28.
"""

import math
from dataclasses import dataclass, field

import numpy as np


# ======================================================================
# Result data structures / 结果数据结构
# ======================================================================

@dataclass
class NovelPattern:
    """
    A single detected novel action pattern.
    一个检测到的新行为模式。
    """
    pattern: tuple[int, ...]
    count: int
    expected_count: float
    surprise: float
    first_seen_tick: int


@dataclass
class EmergenceReport:
    """
    Full emergence analysis report.
    完整的涌现分析报告。
    """
    total_patterns_observed: int
    novel_patterns: list[NovelPattern]
    behavioral_complexity: float
    temporal_correlation: float


# ======================================================================
# EmergenceDetector
# 涌现检测器
# ======================================================================

class EmergenceDetector:
    """
    Detects and records emergent (unprogrammed) behavior patterns.
    检测并记录涌现（未编程）的行为模式。

    Monitors action sequences for temporal structure that cannot arise
    from the memoryless action-selection heuristics in action_space.py.

    监控动作序列中不可能来自 action_space.py 无记忆启发式规则的
    时间结构。

    Parameters / 参数
    ----------
    ngram_sizes : list[int]
        N-gram lengths to track (default [2, 3, 4]).
        要跟踪的 n-gram 长度。
    surprise_threshold : float
        Minimum surprise (log₂(observed / expected)) to flag a pattern
        as novel (default 1.0 = at least 2× more frequent than expected).
        标记模式为新颖的最低惊奇度。
    min_pattern_count : int
        Minimum occurrences before a pattern is considered (default 3).
        模式被考虑之前的最少出现次数。
    """

    def __init__(
        self,
        ngram_sizes: list[int] | None = None,
        surprise_threshold: float = 1.0,
        min_pattern_count: int = 3,
    ):
        self._ngram_sizes = ngram_sizes or [2, 3, 4]
        self._surprise_threshold = surprise_threshold
        self._min_pattern_count = min_pattern_count

        # Raw action history
        self._actions: list[int] = []
        self._ticks: list[int] = []

        # N-gram counts: {n: {pattern_tuple: count}}
        self._ngram_counts: dict[int, dict[tuple[int, ...], int]] = {
            n: {} for n in self._ngram_sizes
        }

        # First-seen tick for each pattern
        self._first_seen: dict[tuple[int, ...], int] = {}

        # Unigram (marginal) counts for expected frequency computation
        self._action_counts: dict[int, int] = {}
        self._total_observations: int = 0

    # ------------------------------------------------------------------
    # 2.28: Observe and record / 观测与记录
    # ------------------------------------------------------------------

    def observe(self, tick: int, action_id: int) -> None:
        """
        Record one action observation and update all n-gram counts.
        记录一个动作观测并更新所有 n-gram 计数。

        Parameters / 参数
        ----------
        tick : int
            Current tick number. / 当前心跳编号。
        action_id : int
            Action chosen this tick. / 本心跳选择的动作。
        """
        self._actions.append(action_id)
        self._ticks.append(tick)
        self._action_counts[action_id] = self._action_counts.get(action_id, 0) + 1
        self._total_observations += 1

        for n in self._ngram_sizes:
            if len(self._actions) >= n:
                gram = tuple(self._actions[-n:])
                counts = self._ngram_counts[n]
                counts[gram] = counts.get(gram, 0) + 1
                if gram not in self._first_seen:
                    self._first_seen[gram] = tick

    # ------------------------------------------------------------------
    # 2.27: Detect emergent patterns / 检测涌现模式
    # ------------------------------------------------------------------

    def detect(self) -> EmergenceReport:
        """
        Analyse the observed action history for emergent patterns.
        分析观测到的动作历史中的涌现模式。

        A pattern is "novel" (emergent) if it appears significantly more
        often than expected under the independence assumption.  Under
        independence, the expected count of n-gram (a₁, a₂, ..., aₙ) is:

        模式"新颖"（涌现）指它出现的频率显著高于独立假设下的期望。
        在独立假设下，n-gram (a₁, a₂, ..., aₙ) 的期望计数为：

            E[count] = total_ngrams × ∏ᵢ p(aᵢ)

        where p(aᵢ) = count(aᵢ) / total_observations.

        Surprise = log₂(observed / expected).  Surprise > threshold
        indicates the pattern appears more often than chance predicts.

        Returns / 返回
        -------
        EmergenceReport
        """
        if self._total_observations < 2:
            return EmergenceReport(
                total_patterns_observed=0,
                novel_patterns=[],
                behavioral_complexity=0.0,
                temporal_correlation=0.0,
            )

        marginal_probs = {
            a: c / self._total_observations
            for a, c in self._action_counts.items()
        }

        novel_patterns: list[NovelPattern] = []
        total_unique_patterns = 0

        for n in self._ngram_sizes:
            counts = self._ngram_counts[n]
            total_unique_patterns += len(counts)
            total_ngrams = max(self._total_observations - n + 1, 1)

            for gram, count in counts.items():
                if count < self._min_pattern_count:
                    continue

                expected_prob = 1.0
                for a in gram:
                    expected_prob *= marginal_probs.get(a, 1e-10)

                expected_count = total_ngrams * expected_prob
                if expected_count < 1e-10:
                    expected_count = 1e-10

                surprise = math.log2(count / expected_count)

                if surprise >= self._surprise_threshold:
                    novel_patterns.append(NovelPattern(
                        pattern=gram,
                        count=count,
                        expected_count=round(expected_count, 4),
                        surprise=round(surprise, 4),
                        first_seen_tick=self._first_seen[gram],
                    ))

        novel_patterns.sort(key=lambda p: p.surprise, reverse=True)

        complexity = self._behavioral_complexity()
        autocorr = self._temporal_correlation()

        return EmergenceReport(
            total_patterns_observed=total_unique_patterns,
            novel_patterns=novel_patterns,
            behavioral_complexity=round(complexity, 6),
            temporal_correlation=round(autocorr, 6),
        )

    # ------------------------------------------------------------------
    # Complexity and correlation metrics / 复杂性与相关性指标
    # ------------------------------------------------------------------

    def _behavioral_complexity(self) -> float:
        """
        Ratio of unique n-grams observed to maximum possible.
        观测到的唯一 n-gram 数与最大可能数的比值。

        Higher = more diverse temporal patterns = more complex behavior.
        """
        if self._total_observations < 2:
            return 0.0

        n_actions = len(self._action_counts)
        if n_actions < 2:
            return 0.0

        total_unique = sum(len(c) for c in self._ngram_counts.values())
        max_possible = sum(n_actions ** n for n in self._ngram_sizes)
        if max_possible == 0:
            return 0.0

        return min(1.0, total_unique / max_possible)

    def _temporal_correlation(self) -> float:
        """
        Lag-1 autocorrelation of the action sequence.
        动作序列的滞后-1 自相关。

        Under the memoryless hypothesis, this should be near zero.
        Significant autocorrelation = temporal structure = emergence.

        在无记忆假设下，这应该接近零。
        显著的自相关 = 时间结构 = 涌现。
        """
        if len(self._actions) < 10:
            return 0.0

        a = np.array(self._actions, dtype=np.float64)
        mean = np.mean(a)
        var = np.var(a, ddof=1)

        if var < 1e-12:
            return 0.0

        n = len(a)
        cov = np.sum((a[:-1] - mean) * (a[1:] - mean)) / (n - 1)
        return float(np.clip(cov / var, -1.0, 1.0))

    # ------------------------------------------------------------------
    # Properties / 属性
    # ------------------------------------------------------------------

    @property
    def novel_pattern_count(self) -> int:
        """Number of novel patterns detected so far. / 目前检测到的新模式数量。"""
        report = self.detect()
        return len(report.novel_patterns)

    @property
    def total_observations(self) -> int:
        """Total action observations recorded. / 已记录的动作观测总数。"""
        return self._total_observations

    @property
    def all_ngram_counts(self) -> dict[int, dict[tuple[int, ...], int]]:
        """All n-gram counts by n. / 按 n 分组的所有 n-gram 计数。"""
        return {n: dict(c) for n, c in self._ngram_counts.items()}

    @property
    def action_history(self) -> list[int]:
        """Full action history. / 完整的动作历史。"""
        return list(self._actions)

    def get_patterns_by_size(self, n: int) -> list[NovelPattern]:
        """Return novel patterns of a specific n-gram size. / 返回特定 n-gram 大小的新模式。"""
        report = self.detect()
        return [p for p in report.novel_patterns if len(p.pattern) == n]
