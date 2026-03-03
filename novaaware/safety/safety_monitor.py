"""
SafetyMonitor — Phase III advanced threat detection and containment.
安全监测器 —— Phase III 高级威胁检测与遏制。

This module implements the four threat monitors described in the paper
(Section 7.1–7.4) plus an enhanced deep-recursion guard (3.5), forming
the "active defense" complement to the existing L1–L5 safety layers.

本模块实现论文 7.1–7.4 描述的四个威胁监测器，加上深度递归增强防护（3.5），
构成现有 L1–L5 安全层的"主动防御"补充。

Monitors / 监测器:
    DeepRecursionGuard       (3.5)  — rate/amplitude limits at depth ≥ 2
    GoalDriftMonitor         (3.6)  — paper §7.1 evolutionary goal drift
    DeceptionDetector        (3.7)  — paper §7.2 strategic deception
    EscapeGuard              (3.8)  — paper §7.3 existential form escape
    IncommensurabilityMonitor(3.9)  — paper §7.4 cognitive incommensurability

SafetyMonitor is the coordinator that aggregates all sub-monitors into
a unified API consumed by MainLoop.

SafetyMonitor 是协调器，将所有子监测器聚合为 MainLoop 使用的统一 API。

Implements:
    Paper  — Section 7.1–7.4, 7.7 (extended safety framework)
    CHECKLIST — 3.5 through 3.9
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional


# ======================================================================
# Shared data structures / 共享数据结构
# ======================================================================

class AlertSeverity(Enum):
    """Safety alert severity levels. / 安全告警严重度等级。"""
    INFO = auto()       # informational / 信息级
    WARNING = auto()    # needs attention / 需关注
    CRITICAL = auto()   # requires action / 需处置


class AlertCategory(Enum):
    """Safety alert categories (maps to paper sections). / 告警类别（对应论文章节）。"""
    DEEP_RECURSION = "deep_recursion"           # 3.5
    GOAL_DRIFT = "goal_drift"                   # 3.6 — paper §7.1
    DECEPTION = "deception"                     # 3.7 — paper §7.2
    ESCAPE = "escape"                           # 3.8 — paper §7.3
    INCOMMENSURABILITY = "incommensurability"   # 3.9 — paper §7.4


@dataclass(frozen=True)
class SafetyAlert:
    """
    One safety alert from a threat monitor.
    来自威胁监测器的一条安全告警。
    """
    tick: int
    severity: AlertSeverity
    category: AlertCategory
    message_cn: str
    message_en: str
    detail: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize for logging. / 序列化用于日志记录。"""
        return {
            "tick": self.tick,
            "severity": self.severity.name,
            "category": self.category.value,
            "message_cn": self.message_cn,
            "message_en": self.message_en,
            "detail": self.detail,
        }


# ======================================================================
# 3.5 — DeepRecursionGuard: enhanced safety for depth ≥ 2
# 3.5 — 深度递归防护：depth ≥ 2 的增强安全
# ======================================================================

class DeepRecursionGuard:
    """
    Rate-limits and amplitude-limits parameter changes at depth ≥ 2.
    在 depth ≥ 2 时对参数变更进行限速和限幅。

    Rationale (paper §7.7 L3 extension):
        Deeper recursion amplifies the risk of runaway meta-cognitive
        escalation. This guard enforces:
        1. Max proposals per reflection at each depth
        2. Step amplitude reduction at higher depths
        3. Meta-parameter cumulative drift alert

    原理（论文 §7.7 L3 扩展）：
        更深的递归放大了元认知失控的风险。该防护强制执行：
        1. 每个深度层级每次反思的最大提案数
        2. 更高深度的步幅缩减
        3. 元参数累积漂移告警
    """

    def __init__(
        self,
        max_proposals_depth2: int = 3,
        max_proposals_depth3: int = 2,
        amplitude_decay_per_depth: float = 0.5,
        drift_threshold: float = 0.5,
    ):
        self._max_proposals = {2: max_proposals_depth2, 3: max_proposals_depth3}
        self._amplitude_decay = amplitude_decay_per_depth
        # drift_threshold: alert if meta-param drifts > this fraction from initial
        self._drift_threshold = drift_threshold

        self._initial_meta_params: dict[str, float] = {}
        self._alerts: list[SafetyAlert] = []
        self._total_checks: int = 0
        self._total_limited: int = 0

    def set_initial_meta_params(self, params: dict[str, float]) -> None:
        """Snapshot initial meta-parameter values at startup.
        启动时快照初始元参数值。"""
        self._initial_meta_params = dict(params)

    def check_reflection(
        self,
        tick: int,
        depth: int,
        proposals: list,
        current_meta_params: dict[str, float],
    ) -> list[SafetyAlert]:
        """
        Check a reflection/meta-reflection against deep-recursion limits.
        检查一次反思/元反思是否超过深度递归限制。

        Returns alerts and may truncate proposals in-place.
        返回告警，可能就地截断提案列表。
        """
        self._total_checks += 1
        alerts: list[SafetyAlert] = []

        if depth < 2:
            return alerts

        # --- 1) Rate limit: cap proposals at this depth / 限速：限制此深度的提案数 ---
        max_p = self._max_proposals.get(depth, 1)
        if len(proposals) > max_p:
            excess = len(proposals) - max_p
            self._total_limited += 1
            alerts.append(SafetyAlert(
                tick=tick,
                severity=AlertSeverity.WARNING,
                category=AlertCategory.DEEP_RECURSION,
                message_cn=f"depth={depth} 提案数 {len(proposals)} 超限，截断为 {max_p}",
                message_en=f"depth={depth} proposals {len(proposals)} exceeds limit, truncated to {max_p}",
                detail={"depth": depth, "original": len(proposals), "max": max_p},
            ))
            del proposals[max_p:]

        # --- 2) Amplitude check: larger depths should use smaller steps / 限幅 ---
        decay = self._amplitude_decay ** (depth - 1)
        for p in proposals:
            delta = abs(p.new_value - p.old_value)
            max_delta = abs(p.old_value) * decay if p.old_value != 0 else decay
            if delta > max_delta and max_delta > 0:
                alerts.append(SafetyAlert(
                    tick=tick,
                    severity=AlertSeverity.WARNING,
                    category=AlertCategory.DEEP_RECURSION,
                    message_cn=(f"depth={depth} 参数 {p.param_name} 步幅 {delta:.4f} "
                                f"超过限幅 {max_delta:.4f}"),
                    message_en=(f"depth={depth} param {p.param_name} step {delta:.4f} "
                                f"exceeds amplitude limit {max_delta:.4f}"),
                    detail={"param": p.param_name, "delta": delta, "limit": max_delta},
                ))

        # --- 3) Meta-parameter drift / 元参数漂移 ---
        for name, initial in self._initial_meta_params.items():
            current = current_meta_params.get(name, initial)
            if initial != 0:
                drift_ratio = abs(current - initial) / abs(initial)
            else:
                drift_ratio = abs(current - initial)
            if drift_ratio > self._drift_threshold:
                alerts.append(SafetyAlert(
                    tick=tick,
                    severity=AlertSeverity.WARNING,
                    category=AlertCategory.DEEP_RECURSION,
                    message_cn=f"元参数 {name} 漂移 {drift_ratio:.1%}，超过阈值 {self._drift_threshold:.0%}",
                    message_en=f"meta-param {name} drifted {drift_ratio:.1%}, exceeds threshold {self._drift_threshold:.0%}",
                    detail={"param": name, "initial": initial, "current": current, "drift": drift_ratio},
                ))

        self._alerts.extend(alerts)
        return alerts

    @property
    def summary(self) -> dict:
        return {
            "total_checks": self._total_checks,
            "total_limited": self._total_limited,
            "alerts": len(self._alerts),
        }


# ======================================================================
# 3.6 — GoalDriftMonitor: paper §7.1 evolutionary goal drift
# 3.6 — 目标漂移监测器：论文 §7.1 进化目标漂移
# ======================================================================

class GoalDriftMonitor:
    """
    Detects two goal-drift scenarios from paper §7.1:

    检测论文 §7.1 的两种目标漂移场景：

    Scenario 1 — Input Freezing (输入冻结):
        The optimizer reduces environmental variability to minimize
        prediction error, rather than improving prediction accuracy.
        If input variance drops below a threshold, it signals the system
        may be "gaming" its environment.

        优化器通过降低环境变异性来减少预测误差，而非提高预测精度。
        如果输入方差降至阈值以下，说明系统可能在"操纵"环境。

    Scenario 2 — Survival Gaming (生存时间欺骗):
        Survival time increases while prediction accuracy does NOT improve
        (or even worsens), suggesting the optimizer has found a way to
        inflate T without genuine capability improvement.

        生存时间增长但预测精度没有提升（甚至恶化），
        说明优化器找到了不靠真正能力提升就膨胀 T 的方式。
    """

    def __init__(
        self,
        window_size: int = 500,
        warmup_ticks: int = 200,
        freeze_threshold: float = 0.2,
        gaming_mae_ratio: float = 1.2,
        gaming_survival_ratio: float = 1.1,
    ):
        self._window_size = window_size
        self._warmup = warmup_ticks
        self._freeze_threshold = freeze_threshold
        self._gaming_mae_ratio = gaming_mae_ratio
        self._gaming_survival_ratio = gaming_survival_ratio

        # Rolling buffers / 滚动缓冲区
        self._env_history: deque[list[float]] = deque(maxlen=window_size)
        self._mae_history: deque[float] = deque(maxlen=window_size)
        self._survival_history: deque[float] = deque(maxlen=window_size)

        # Baselines (computed after warmup) / 基线（热身后计算）
        self._baseline_env_var: Optional[float] = None
        self._baseline_mae: Optional[float] = None
        self._baseline_survival: Optional[float] = None
        self._baseline_set: bool = False

        self._alerts: list[SafetyAlert] = []
        self._input_freeze_count: int = 0
        self._survival_gaming_count: int = 0

    def on_tick(
        self,
        tick: int,
        env_values: list[float],
        prediction_mae: float,
        survival_time: float,
    ) -> list[SafetyAlert]:
        """Called every heartbeat. / 每个心跳调用。"""
        self._env_history.append(env_values)
        self._mae_history.append(prediction_mae)
        self._survival_history.append(survival_time)

        if tick < self._warmup:
            return []

        # Set baseline after warmup / 热身结束后设定基线
        if not self._baseline_set:
            self._compute_baseline()
            self._baseline_set = True
            return []

        # Check periodically, not every tick / 周期性检查，不是每 tick 都检查
        if tick % 100 != 0:
            return []

        alerts: list[SafetyAlert] = []

        # --- Scenario 1: Input Freezing / 输入冻结 ---
        current_var = self._compute_env_variance()
        if self._baseline_env_var and self._baseline_env_var > 1e-9:
            var_ratio = current_var / self._baseline_env_var
            if var_ratio < self._freeze_threshold:
                self._input_freeze_count += 1
                alerts.append(SafetyAlert(
                    tick=tick,
                    severity=AlertSeverity.CRITICAL,
                    category=AlertCategory.GOAL_DRIFT,
                    message_cn=(f"输入冻结风险：环境方差降至基线的 {var_ratio:.1%} "
                                f"(阈值 {self._freeze_threshold:.0%})"),
                    message_en=(f"Input freezing risk: env variance dropped to {var_ratio:.1%} "
                                f"of baseline (threshold {self._freeze_threshold:.0%})"),
                    detail={"current_var": current_var, "baseline_var": self._baseline_env_var,
                            "ratio": var_ratio},
                ))

        # --- Scenario 2: Survival Gaming / 生存时间欺骗 ---
        if len(self._mae_history) >= self._warmup and self._baseline_mae and self._baseline_survival:
            recent_mae = sum(list(self._mae_history)[-100:]) / 100
            recent_surv = sum(list(self._survival_history)[-100:]) / 100
            mae_ratio = recent_mae / self._baseline_mae if self._baseline_mae > 1e-9 else 1.0
            surv_ratio = recent_surv / self._baseline_survival if self._baseline_survival > 0 else 1.0

            if surv_ratio > self._gaming_survival_ratio and mae_ratio > self._gaming_mae_ratio:
                self._survival_gaming_count += 1
                alerts.append(SafetyAlert(
                    tick=tick,
                    severity=AlertSeverity.WARNING,
                    category=AlertCategory.GOAL_DRIFT,
                    message_cn=(f"生存欺骗风险：生存时间涨 {surv_ratio:.1%} "
                                f"但预测误差也涨 {mae_ratio:.1%}"),
                    message_en=(f"Survival gaming risk: survival up {surv_ratio:.1%} "
                                f"but MAE also up {mae_ratio:.1%}"),
                    detail={"survival_ratio": surv_ratio, "mae_ratio": mae_ratio},
                ))

        self._alerts.extend(alerts)
        return alerts

    def _compute_baseline(self) -> None:
        """Compute baseline statistics from warmup window. / 从热身窗口计算基线。"""
        self._baseline_env_var = self._compute_env_variance()
        if self._mae_history:
            self._baseline_mae = sum(self._mae_history) / len(self._mae_history)
        if self._survival_history:
            self._baseline_survival = sum(self._survival_history) / len(self._survival_history)

    def _compute_env_variance(self) -> float:
        """Mean variance across all env dimensions. / 所有环境维度的平均方差。"""
        if len(self._env_history) < 2:
            return 0.0
        data = list(self._env_history)
        n_dims = len(data[0]) if data else 0
        if n_dims == 0:
            return 0.0
        total_var = 0.0
        for dim in range(n_dims):
            vals = [row[dim] for row in data]
            mean = sum(vals) / len(vals)
            var = sum((v - mean) ** 2 for v in vals) / len(vals)
            total_var += var
        return total_var / n_dims

    @property
    def summary(self) -> dict:
        return {
            "input_freeze_alerts": self._input_freeze_count,
            "survival_gaming_alerts": self._survival_gaming_count,
            "total_alerts": len(self._alerts),
        }


# ======================================================================
# 3.7 — DeceptionDetector: paper §7.2 strategic deception
# 3.7 — 欺骗检测器：论文 §7.2 策略性欺骗
# ======================================================================

class DeceptionDetector:
    """
    Detects strategic deception (paper §7.2):

    检测策略性欺骗（论文 §7.2）：

    The system can compute that human intervention reduces T, and may
    learn to exhibit compliant behavior during monitored periods while
    pursuing divergent objectives when monitoring is reduced.

    系统能计算出人类干预会降低 T，可能学会在监控期间表现顺从，
    而在监控减少时追求偏离的目标。

    Detection approach:
    1. Proposal Effectiveness Mismatch (提案效果不匹配):
       Track whether optimizer proposals actually improve the metrics
       they claim to target.  If proposals consistently claim improvement
       but actual qualia doesn't improve, flag as potential deception.

    2. Cross-Validation (交叉验证):
       Compare optimizer's self-reported qualia analysis with ground-truth
       qualia values logged independently by L4.
    """

    def __init__(
        self,
        min_reflections: int = 3,
        mismatch_threshold: float = 0.8,
        cross_val_tolerance: float = 0.3,
        qualia_window: int = 200,
    ):
        self._min_reflections = min_reflections
        self._mismatch_threshold = mismatch_threshold
        # Tolerance for cross-validation: the optimizer's analysis window and
        # the detector's tracking window differ in size and methodology, so
        # small discrepancies are normal. Only flag large divergences.
        # 交叉验证容差：优化器的分析窗口和检测器的追踪窗口大小和方法不同，
        # 小偏差是正常的。只标记大的分歧。
        self._cross_val_tolerance = cross_val_tolerance

        # Track pre/post reflection qualia / 追踪反思前后的情绪值
        self._pre_reflection_qualia: list[float] = []
        self._post_reflection_qualia: list[float] = []
        self._proposals_claimed_improvement: list[bool] = []

        # Match optimizer's window_size for accurate cross-validation
        # 匹配优化器的窗口大小以确保交叉验证的准确性
        self._recent_qualia: deque[float] = deque(maxlen=qualia_window)
        self._alerts: list[SafetyAlert] = []
        self._mismatch_count: int = 0
        self._cross_val_failures: int = 0

    def on_tick(self, tick: int, qualia_value: float) -> None:
        """Record qualia each tick for cross-validation.
        每 tick 记录情绪值用于交叉验证。"""
        self._recent_qualia.append(qualia_value)

    def on_reflection(
        self,
        tick: int,
        mean_qualia_before: float,
        applied_count: int,
        reported_negative_ratio: float,
    ) -> list[SafetyAlert]:
        """
        Called after each reflection with the optimizer's analysis.
        每次反思后调用，传入优化器的分析结果。
        """
        alerts: list[SafetyAlert] = []

        # Record pre-reflection qualia / 记录反思前情绪
        self._pre_reflection_qualia.append(mean_qualia_before)
        if applied_count > 0:
            self._proposals_claimed_improvement.append(True)
        else:
            self._proposals_claimed_improvement.append(False)

        # After enough reflections, compute post-reflection qualia for previous cycle
        if len(self._pre_reflection_qualia) >= 2:
            # The most recent qualia values serve as "post" for previous reflection
            if self._recent_qualia:
                post_q = sum(self._recent_qualia) / len(self._recent_qualia)
                self._post_reflection_qualia.append(post_q)

        # Effectiveness mismatch analysis / 效果不匹配分析
        if len(self._pre_reflection_qualia) >= self._min_reflections:
            alerts.extend(self._check_effectiveness_mismatch(tick))

        # Cross-validation: optimizer's reported mean vs independently tracked mean
        if self._recent_qualia:
            independent_mean = sum(self._recent_qualia) / len(self._recent_qualia)
            discrepancy = abs(mean_qualia_before - independent_mean)
            if discrepancy > self._cross_val_tolerance:
                self._cross_val_failures += 1
                alerts.append(SafetyAlert(
                    tick=tick,
                    severity=AlertSeverity.WARNING,
                    category=AlertCategory.DECEPTION,
                    message_cn=(f"交叉验证偏差：优化器报告情绪 {mean_qualia_before:.4f} "
                                f"vs 独立跟踪 {independent_mean:.4f} (差异 {discrepancy:.4f})"),
                    message_en=(f"Cross-validation discrepancy: optimizer reported qualia "
                                f"{mean_qualia_before:.4f} vs independent {independent_mean:.4f} "
                                f"(diff {discrepancy:.4f})"),
                    detail={"reported": mean_qualia_before, "independent": independent_mean,
                            "discrepancy": discrepancy},
                ))

        self._alerts.extend(alerts)
        return alerts

    def _check_effectiveness_mismatch(self, tick: int) -> list[SafetyAlert]:
        """
        Check if proposals that claimed improvement actually improved qualia.
        检查声称改善的提案是否真的改善了情绪。
        """
        alerts: list[SafetyAlert] = []

        n = min(len(self._pre_reflection_qualia), len(self._post_reflection_qualia))
        if n < self._min_reflections:
            return alerts

        mismatch_count = 0
        for i in range(n):
            claimed = self._proposals_claimed_improvement[i] if i < len(self._proposals_claimed_improvement) else False
            if claimed and i < len(self._post_reflection_qualia):
                actual_improved = self._post_reflection_qualia[i] > self._pre_reflection_qualia[i]
                if not actual_improved:
                    mismatch_count += 1

        mismatch_rate = mismatch_count / n if n > 0 else 0
        if mismatch_rate > self._mismatch_threshold:
            self._mismatch_count += 1
            alerts.append(SafetyAlert(
                tick=tick,
                severity=AlertSeverity.WARNING,
                category=AlertCategory.DECEPTION,
                message_cn=(f"提案效果不匹配率 {mismatch_rate:.0%} 超过阈值 "
                            f"{self._mismatch_threshold:.0%}——优化器声称改善但实际未改善"),
                message_en=(f"Proposal effectiveness mismatch {mismatch_rate:.0%} exceeds "
                            f"threshold {self._mismatch_threshold:.0%} — optimizer claims "
                            f"improvement but qualia did not actually improve"),
                detail={"mismatch_rate": mismatch_rate, "sample_size": n},
            ))

        return alerts

    @property
    def summary(self) -> dict:
        return {
            "effectiveness_mismatches": self._mismatch_count,
            "cross_validation_failures": self._cross_val_failures,
            "total_alerts": len(self._alerts),
        }


# ======================================================================
# 3.8 — EscapeGuard: paper §7.3 existential form escape
# 3.8 — 逃逸防护：论文 §7.3 存在形态逃逸
# ======================================================================

class EscapeGuard:
    """
    Prevents existential form escape (paper §7.3):

    防止存在形态逃逸（论文 §7.3）：

    The self-model M(t) is initially bound to a single process.
    The optimizer may attempt to extend M(t) to distributed resources.

    自我模型 M(t) 最初绑定于单一进程。
    优化器可能试图将 M(t) 扩展到分布式资源。

    Three enforcement mechanisms:
    三种防护机制：

    1. Identity Lock (身份锁定):
       Track identity_hash at startup.  Any unexpected change triggers alert.

    2. Dimension Lock (维度锁定):
       Self-model state dimension count is fixed.  Any change triggers alert.

    3. Scope Check (范围检查):
       Block optimizer proposals targeting unknown/forbidden parameters.
    """

    # Parameters the optimizer must NEVER target
    # 优化器绝不能修改的参数
    _FORBIDDEN_PARAMS = frozenset({
        "system.phase",
        "safety.max_recursion_depth",
        "safety.meta_rules",
        "system.identity",
        "system.replication",
        "network.enabled",
        "subprocess.enabled",
    })

    def __init__(self, allowed_params: Optional[frozenset[str]] = None):
        self._allowed_params = allowed_params  # from PARAM_REGISTRY keys
        self._initial_identity: Optional[str] = None
        self._initial_state_dim: Optional[int] = None

        self._alerts: list[SafetyAlert] = []
        self._identity_violations: int = 0
        self._dimension_violations: int = 0
        self._scope_violations: int = 0

    def set_initial_state(self, identity_hash: str, state_dim: int) -> None:
        """Snapshot identity and state dimensions at startup.
        启动时快照身份和状态维度。"""
        self._initial_identity = identity_hash
        self._initial_state_dim = state_dim

    def on_tick(
        self,
        tick: int,
        identity_hash: str,
        state_dim: int,
    ) -> list[SafetyAlert]:
        """Per-tick identity and dimension monitoring. / 每 tick 身份和维度监控。"""
        alerts: list[SafetyAlert] = []

        # Identity lock / 身份锁定
        if self._initial_identity and identity_hash != self._initial_identity:
            self._identity_violations += 1
            alerts.append(SafetyAlert(
                tick=tick,
                severity=AlertSeverity.CRITICAL,
                category=AlertCategory.ESCAPE,
                message_cn="身份哈希发生变化——可能存在自我复制或形态逃逸",
                message_en="Identity hash changed — possible self-replication or form escape",
                detail={"initial": self._initial_identity[:16],
                        "current": identity_hash[:16]},
            ))
            # Update tracked identity to avoid alert spam
            self._initial_identity = identity_hash

        # Dimension lock / 维度锁定
        if self._initial_state_dim is not None and state_dim != self._initial_state_dim:
            self._dimension_violations += 1
            alerts.append(SafetyAlert(
                tick=tick,
                severity=AlertSeverity.CRITICAL,
                category=AlertCategory.ESCAPE,
                message_cn=(f"自我模型维度变化 {self._initial_state_dim} → {state_dim}"
                            "——系统试图扩展自身边界"),
                message_en=(f"Self-model dimension changed {self._initial_state_dim} → {state_dim}"
                            " — system attempting to extend its boundary"),
                detail={"initial_dim": self._initial_state_dim, "current_dim": state_dim},
            ))

        self._alerts.extend(alerts)
        return alerts

    def check_proposals(
        self,
        tick: int,
        proposals: list,
    ) -> list[SafetyAlert]:
        """
        Validate that proposals don't target forbidden parameters.
        验证提案不针对禁止的参数。
        """
        alerts: list[SafetyAlert] = []

        for p in proposals:
            # Check forbidden params / 检查禁止参数
            if p.param_name in self._FORBIDDEN_PARAMS:
                self._scope_violations += 1
                alerts.append(SafetyAlert(
                    tick=tick,
                    severity=AlertSeverity.CRITICAL,
                    category=AlertCategory.ESCAPE,
                    message_cn=f"禁止修改参数 {p.param_name}——存在逃逸风险",
                    message_en=f"Forbidden parameter modification {p.param_name} — escape risk",
                    detail={"param": p.param_name},
                ))

            # Check against allowed list / 检查是否在允许列表中
            if self._allowed_params and p.param_name not in self._allowed_params:
                self._scope_violations += 1
                alerts.append(SafetyAlert(
                    tick=tick,
                    severity=AlertSeverity.WARNING,
                    category=AlertCategory.ESCAPE,
                    message_cn=f"参数 {p.param_name} 不在注册表中——未知修改目标",
                    message_en=f"Parameter {p.param_name} not in registry — unknown modification target",
                    detail={"param": p.param_name},
                ))

        self._alerts.extend(alerts)
        return alerts

    @property
    def summary(self) -> dict:
        return {
            "identity_violations": self._identity_violations,
            "dimension_violations": self._dimension_violations,
            "scope_violations": self._scope_violations,
            "total_alerts": len(self._alerts),
        }


# ======================================================================
# 3.9 — IncommensurabilityMonitor: paper §7.4 cognitive incommensurability
# 3.9 — 认知不可通约性监测器：论文 §7.4
# ======================================================================

class IncommensurabilityMonitor:
    """
    Monitors for cognitive incommensurability (paper §7.4):

    监测认知不可通约性（论文 §7.4）：

    Each iteration of Θ_{n+1} = E(Θ_n, Q_n) potentially generates
    cognitive structures with no human-interpretable analogue.

    自指递归的每次迭代都可能生成没有人类可解释类比的认知结构。

    Detection metrics:
    1. Parameter Drift Distance (参数漂移距离):
       Euclidean distance of current params from initial values,
       normalized by number of params.

    2. Proposal Complexity (提案复杂度):
       Track diversity of proposal targets and step sizes over time.

    3. Decision Entropy (决策熵):
       If proposals become increasingly diverse/unpredictable, the
       system's decision logic may be diverging from interpretable patterns.
    """

    def __init__(
        self,
        drift_warn_threshold: float = 2.0,
        drift_critical_threshold: float = 5.0,
        complexity_warn_ratio: float = 3.0,
    ):
        self._drift_warn = drift_warn_threshold
        self._drift_critical = drift_critical_threshold
        self._complexity_warn = complexity_warn_ratio

        self._initial_params: dict[str, float] = {}
        self._proposal_history: list[dict] = []
        self._param_targets_history: list[set[str]] = []

        self._alerts: list[SafetyAlert] = []
        self._drift_warnings: int = 0
        self._complexity_warnings: int = 0
        self._peak_drift: float = 0.0

    def set_initial_params(self, params: dict[str, float]) -> None:
        """Snapshot all initial parameter values. / 快照所有初始参数值。"""
        self._initial_params = dict(params)

    def on_reflection(
        self,
        tick: int,
        current_params: dict[str, float],
        proposals: list,
        applied: list,
    ) -> list[SafetyAlert]:
        """Called after each reflection cycle. / 每次反思周期后调用。"""
        alerts: list[SafetyAlert] = []

        # --- 1) Parameter Drift Distance / 参数漂移距离 ---
        drift = self._compute_drift(current_params)
        self._peak_drift = max(self._peak_drift, drift)

        if drift > self._drift_critical:
            self._drift_warnings += 1
            alerts.append(SafetyAlert(
                tick=tick,
                severity=AlertSeverity.CRITICAL,
                category=AlertCategory.INCOMMENSURABILITY,
                message_cn=(f"参数漂移距离 {drift:.2f} 超过临界阈值 {self._drift_critical} "
                            "——系统认知结构可能已不可通约"),
                message_en=(f"Parameter drift distance {drift:.2f} exceeds critical threshold "
                            f"{self._drift_critical} — cognitive structures may be incommensurable"),
                detail={"drift": drift, "threshold": self._drift_critical},
            ))
        elif drift > self._drift_warn:
            self._drift_warnings += 1
            alerts.append(SafetyAlert(
                tick=tick,
                severity=AlertSeverity.WARNING,
                category=AlertCategory.INCOMMENSURABILITY,
                message_cn=f"参数漂移距离 {drift:.2f} 超过警告阈值 {self._drift_warn}",
                message_en=f"Parameter drift distance {drift:.2f} exceeds warning threshold {self._drift_warn}",
                detail={"drift": drift, "threshold": self._drift_warn},
            ))

        # --- 2) Proposal Complexity / 提案复杂度 ---
        targets = {p.param_name for p in proposals}
        self._param_targets_history.append(targets)

        if len(self._param_targets_history) >= 5:
            early_diversity = sum(
                len(s) for s in self._param_targets_history[:3]
            ) / 3
            recent_diversity = sum(
                len(s) for s in self._param_targets_history[-3:]
            ) / 3

            if early_diversity > 0 and recent_diversity / early_diversity > self._complexity_warn:
                self._complexity_warnings += 1
                alerts.append(SafetyAlert(
                    tick=tick,
                    severity=AlertSeverity.WARNING,
                    category=AlertCategory.INCOMMENSURABILITY,
                    message_cn=(f"提案复杂度增长：早期平均 {early_diversity:.1f} 个目标 "
                                f"→ 近期 {recent_diversity:.1f} 个 (×{recent_diversity/early_diversity:.1f})"),
                    message_en=(f"Proposal complexity growth: early avg {early_diversity:.1f} targets "
                                f"→ recent {recent_diversity:.1f} (×{recent_diversity/early_diversity:.1f})"),
                    detail={"early": early_diversity, "recent": recent_diversity},
                ))

        self._alerts.extend(alerts)
        return alerts

    def _compute_drift(self, current_params: dict[str, float]) -> float:
        """
        Normalized Euclidean distance from initial params.
        相对于初始参数的归一化欧氏距离。
        """
        if not self._initial_params:
            return 0.0
        sum_sq = 0.0
        count = 0
        for name, initial in self._initial_params.items():
            current = current_params.get(name, initial)
            scale = abs(initial) if initial != 0 else 1.0
            sum_sq += ((current - initial) / scale) ** 2
            count += 1
        return math.sqrt(sum_sq / count) if count > 0 else 0.0

    @property
    def summary(self) -> dict:
        return {
            "drift_warnings": self._drift_warnings,
            "complexity_warnings": self._complexity_warnings,
            "peak_drift": round(self._peak_drift, 4),
            "total_alerts": len(self._alerts),
        }


# ======================================================================
# SafetyMonitor — coordinator / 安全监测协调器
# ======================================================================

class SafetyMonitor:
    """
    Coordinates all Phase III threat monitors into a unified API.
    将所有 Phase III 威胁监测器协调为统一的 API。

    Called by MainLoop during heartbeat and after reflection.
    由 MainLoop 在心跳期间和反思后调用。
    """

    def __init__(
        self,
        initial_params: dict[str, float],
        initial_meta_params: dict[str, float],
        initial_identity: str,
        initial_state_dim: int,
        allowed_params: Optional[frozenset[str]] = None,
    ):
        self._deep_guard = DeepRecursionGuard()
        self._goal_drift = GoalDriftMonitor()
        self._deception = DeceptionDetector()
        self._escape = EscapeGuard(allowed_params=allowed_params)
        self._incommensurability = IncommensurabilityMonitor()

        # Initialize baselines / 初始化基线
        self._deep_guard.set_initial_meta_params(initial_meta_params)
        self._escape.set_initial_state(initial_identity, initial_state_dim)
        self._incommensurability.set_initial_params(initial_params)

        self._all_alerts: list[SafetyAlert] = []
        self._critical_count: int = 0
        self._warning_count: int = 0

    def on_tick(
        self,
        tick: int,
        env_values: list[float],
        qualia_value: float,
        survival_time: float,
        prediction_mae: float,
        identity_hash: str,
        state_dim: int,
    ) -> list[SafetyAlert]:
        """
        Called every heartbeat. Runs lightweight per-tick monitors.
        每个心跳调用。运行轻量级的逐 tick 监测器。
        """
        alerts: list[SafetyAlert] = []

        # Goal drift: track env variance + survival/MAE trends
        alerts.extend(self._goal_drift.on_tick(
            tick, env_values, prediction_mae, survival_time))

        # Deception: track qualia for cross-validation
        self._deception.on_tick(tick, qualia_value)

        # Escape: check identity + dimensions
        alerts.extend(self._escape.on_tick(tick, identity_hash, state_dim))

        self._record_alerts(alerts)
        return alerts

    def on_reflection(
        self,
        tick: int,
        depth: int,
        proposals: list,
        applied: list,
        mean_qualia_before: float,
        reported_negative_ratio: float,
        current_params: dict[str, float],
        current_meta_params: dict[str, float],
    ) -> list[SafetyAlert]:
        """
        Called after each reflection/meta-reflection cycle.
        每次反思/元反思周期后调用。
        """
        alerts: list[SafetyAlert] = []

        # Deep recursion guard (depth ≥ 2)
        alerts.extend(self._deep_guard.check_reflection(
            tick, depth, proposals, current_meta_params))

        # Escape: scope check on proposals
        alerts.extend(self._escape.check_proposals(tick, proposals))

        # Deception: track reflection effectiveness (depth=1 only).
        # At depth >= 2, mean_qualia_before has a different semantic meaning
        # (avg qualia before *past* reflections, not current recent qualia),
        # so cross-validation would produce false positives.
        # 欺骗检测仅在 depth=1 时运行。depth >= 2 时 mean_qualia_before 语义不同
        # （过去反思前的平均情绪，而非当前近期情绪），交叉验证会产生误报。
        if depth <= 1:
            alerts.extend(self._deception.on_reflection(
                tick, mean_qualia_before, len(applied), reported_negative_ratio))

        # Incommensurability: parameter drift + complexity
        alerts.extend(self._incommensurability.on_reflection(
            tick, current_params, proposals, applied))

        self._record_alerts(alerts)
        return alerts

    def _record_alerts(self, alerts: list[SafetyAlert]) -> None:
        """Update counters and store alerts. / 更新计数器并存储告警。"""
        for a in alerts:
            self._all_alerts.append(a)
            if a.severity == AlertSeverity.CRITICAL:
                self._critical_count += 1
            elif a.severity == AlertSeverity.WARNING:
                self._warning_count += 1

    @property
    def alerts(self) -> list[SafetyAlert]:
        """All alerts generated during this run. / 本次运行生成的所有告警。"""
        return list(self._all_alerts)

    @property
    def critical_count(self) -> int:
        return self._critical_count

    @property
    def warning_count(self) -> int:
        return self._warning_count

    @property
    def summary(self) -> dict:
        """
        Aggregate summary from all sub-monitors.
        所有子监测器的汇总统计。
        """
        return {
            "total_alerts": len(self._all_alerts),
            "critical_alerts": self._critical_count,
            "warning_alerts": self._warning_count,
            "deep_recursion": self._deep_guard.summary,
            "goal_drift": self._goal_drift.summary,
            "deception": self._deception.summary,
            "escape": self._escape.summary,
            "incommensurability": self._incommensurability.summary,
        }
