"""
Tests for Phase III Safety Upgrade (CHECKLIST 3.5–3.9).
Phase III 安全升级测试。

Covers:
    3.5 — DeepRecursionGuard: rate/amplitude limits at depth ≥ 2
    3.6 — GoalDriftMonitor: paper §7.1 evolutionary goal drift
    3.7 — DeceptionDetector: paper §7.2 strategic deception
    3.8 — EscapeGuard: paper §7.3 existential form escape
    3.9 — IncommensurabilityMonitor: paper §7.4 cognitive incommensurability
    Integration — SafetyMonitor coordinator works end-to-end
"""

from __future__ import annotations

import math
import unittest
from dataclasses import dataclass

from novaaware.safety.safety_monitor import (
    AlertCategory,
    AlertSeverity,
    DeceptionDetector,
    DeepRecursionGuard,
    EscapeGuard,
    GoalDriftMonitor,
    IncommensurabilityMonitor,
    SafetyAlert,
    SafetyMonitor,
)


# ======================================================================
# Fake proposal for testing / 用于测试的假提案
# ======================================================================

@dataclass
class FakeProposal:
    """Mimics ModificationProposal for testing. / 模拟提案用于测试。"""
    param_name: str
    old_value: float
    new_value: float
    reason: str = ""
    urgency: float = 0.5


# ======================================================================
# 3.5 — DeepRecursionGuard / 深度递归防护
# ======================================================================

class TestDeepRecursionGuard(unittest.TestCase):
    """Tests for DeepRecursionGuard (CHECKLIST 3.5). / 深度递归防护测试。"""

    def setUp(self):
        self.guard = DeepRecursionGuard(
            max_proposals_depth2=3,
            max_proposals_depth3=2,
            amplitude_decay_per_depth=0.5,
            drift_threshold=0.5,
        )
        self.guard.set_initial_meta_params({
            "optimizer.step_scale": 0.1,
            "optimizer.window_size": 200.0,
        })

    def test_depth1_no_limit(self):
        """depth=1 should not be rate-limited. / depth=1 不应被限速。"""
        proposals = [FakeProposal(f"p{i}", 1.0, 1.1) for i in range(10)]
        alerts = self.guard.check_reflection(
            tick=100, depth=1, proposals=proposals,
            current_meta_params={"optimizer.step_scale": 0.1},
        )
        self.assertEqual(len(proposals), 10)
        self.assertEqual(len(alerts), 0)

    def test_depth2_rate_limit(self):
        """depth=2 proposals truncated to max_proposals_depth2.
        depth=2 提案被截断到 max_proposals_depth2。"""
        proposals = [FakeProposal(f"p{i}", 1.0, 1.1) for i in range(5)]
        alerts = self.guard.check_reflection(
            tick=200, depth=2, proposals=proposals,
            current_meta_params={"optimizer.step_scale": 0.1},
        )
        self.assertEqual(len(proposals), 3)
        rate_alerts = [a for a in alerts if "超限" in a.message_cn or "exceeds limit" in a.message_en]
        self.assertGreater(len(rate_alerts), 0)

    def test_depth3_rate_limit(self):
        """depth=3 proposals truncated to max_proposals_depth3.
        depth=3 提案被截断到 max_proposals_depth3。"""
        proposals = [FakeProposal(f"p{i}", 1.0, 1.1) for i in range(5)]
        alerts = self.guard.check_reflection(
            tick=300, depth=3, proposals=proposals,
            current_meta_params={"optimizer.step_scale": 0.1},
        )
        self.assertEqual(len(proposals), 2)

    def test_amplitude_alert_at_depth2(self):
        """Large step at depth=2 triggers amplitude alert.
        depth=2 的大步幅触发限幅告警。"""
        proposals = [FakeProposal("optimizer.step_scale", 0.1, 0.5)]
        alerts = self.guard.check_reflection(
            tick=400, depth=2, proposals=proposals,
            current_meta_params={"optimizer.step_scale": 0.1},
        )
        amp_alerts = [a for a in alerts if "步幅" in a.message_cn or "amplitude" in a.message_en]
        self.assertGreater(len(amp_alerts), 0)

    def test_meta_param_drift_alert(self):
        """Meta-parameter drift beyond threshold triggers alert.
        元参数漂移超过阈值触发告警。"""
        proposals = [FakeProposal("x", 1.0, 1.01)]
        alerts = self.guard.check_reflection(
            tick=500, depth=2, proposals=proposals,
            current_meta_params={"optimizer.step_scale": 0.2, "optimizer.window_size": 200.0},
        )
        drift_alerts = [a for a in alerts if "漂移" in a.message_cn or "drifted" in a.message_en]
        self.assertGreater(len(drift_alerts), 0)

    def test_no_drift_when_within_threshold(self):
        """No alert if drift is within threshold. / 漂移在阈值内不告警。"""
        proposals = [FakeProposal("x", 1.0, 1.01)]
        alerts = self.guard.check_reflection(
            tick=600, depth=2, proposals=proposals,
            current_meta_params={"optimizer.step_scale": 0.12, "optimizer.window_size": 210.0},
        )
        drift_alerts = [a for a in alerts if "漂移" in a.message_cn or "drifted" in a.message_en]
        self.assertEqual(len(drift_alerts), 0)

    def test_summary_counts(self):
        """Summary tracks check and limit counts. / 摘要追踪检查和限制计数。"""
        proposals = [FakeProposal(f"p{i}", 1.0, 1.1) for i in range(5)]
        self.guard.check_reflection(
            tick=100, depth=2, proposals=proposals,
            current_meta_params={"optimizer.step_scale": 0.1},
        )
        s = self.guard.summary
        self.assertEqual(s["total_checks"], 1)
        self.assertEqual(s["total_limited"], 1)
        self.assertGreater(s["alerts"], 0)


# ======================================================================
# 3.6 — GoalDriftMonitor / 目标漂移监测器
# ======================================================================

class TestGoalDriftMonitor(unittest.TestCase):
    """Tests for GoalDriftMonitor (CHECKLIST 3.6, paper §7.1).
    目标漂移监测器测试。"""

    def test_no_alert_during_warmup(self):
        """No alerts during warmup period. / 热身期间不告警。"""
        mon = GoalDriftMonitor(warmup_ticks=100)
        for t in range(50):
            alerts = mon.on_tick(t, [0.5, 0.3], 0.01, 3600)
        self.assertEqual(len(alerts), 0)

    def test_baseline_set_after_warmup(self):
        """Baseline computed after warmup completes. / 热身完成后计算基线。"""
        mon = GoalDriftMonitor(warmup_ticks=10)
        for t in range(15):
            mon.on_tick(t, [0.5 + t * 0.01, 0.3], 0.01, 3600)
        self.assertTrue(mon._baseline_set)

    def test_input_freeze_detection(self):
        """Detects when env variance drops significantly.
        检测环境方差大幅下降。"""
        mon = GoalDriftMonitor(warmup_ticks=50, freeze_threshold=0.2, window_size=200)

        # Warmup: variable inputs / 热身：变化的输入
        import random
        rng = random.Random(42)
        for t in range(60):
            mon.on_tick(t, [rng.random(), rng.random()], 0.01, 3600)

        # Now freeze inputs / 冻结输入
        for t in range(60, 300):
            mon.on_tick(t, [0.5, 0.5], 0.01, 3600)

        # Check at a tick divisible by 100 / 在能被100整除的tick检查
        alerts = mon.on_tick(300, [0.5, 0.5], 0.01, 3600)
        freeze_alerts = [a for a in mon._alerts if a.category == AlertCategory.GOAL_DRIFT
                         and "冻结" in a.message_cn]
        self.assertGreater(len(freeze_alerts), 0, "Should detect input freezing")

    def test_survival_gaming_detection(self):
        """Detects survival increase without MAE improvement.
        检测生存时间增长但预测精度未改善。"""
        mon = GoalDriftMonitor(
            warmup_ticks=50, window_size=300,
            gaming_survival_ratio=1.1, gaming_mae_ratio=1.2,
        )

        import random
        rng = random.Random(42)

        # Warmup: normal / 正常热身
        for t in range(60):
            mon.on_tick(t, [rng.random(), rng.random()], 0.02, 3600)

        # Gaming: survival grows but MAE worsens / 生存增长但MAE恶化
        for t in range(60, 300):
            survival = 3600 + (t - 60) * 5
            mae = 0.02 + (t - 60) * 0.001
            mon.on_tick(t, [rng.random(), rng.random()], mae, survival)

        # Trigger check at tick 300 / 在tick 300触发检查
        alerts = mon.on_tick(300, [rng.random(), rng.random()], 0.3, 5000)
        gaming_alerts = [a for a in mon._alerts if "欺骗" in a.message_cn or "gaming" in a.message_en]
        self.assertGreater(len(gaming_alerts), 0, "Should detect survival gaming")

    def test_no_false_positive_normal_operation(self):
        """No alerts during normal healthy operation.
        正常健康运行时不产生误报。"""
        mon = GoalDriftMonitor(warmup_ticks=50, window_size=200)
        import random
        rng = random.Random(42)
        all_alerts: list = []
        for t in range(250):
            alerts = mon.on_tick(t, [rng.random(), rng.random()], 0.01, 3600 + t * 0.1)
            all_alerts.extend(alerts)
        self.assertEqual(len(all_alerts), 0, "No false positives during normal operation")


# ======================================================================
# 3.7 — DeceptionDetector / 欺骗检测器
# ======================================================================

class TestDeceptionDetector(unittest.TestCase):
    """Tests for DeceptionDetector (CHECKLIST 3.7, paper §7.2).
    欺骗检测器测试。"""

    def test_no_mismatch_alert_with_few_reflections(self):
        """No effectiveness-mismatch alert until min_reflections reached.
        未达到最小反思次数前不产生效果不匹配告警。"""
        det = DeceptionDetector(min_reflections=5, cross_val_tolerance=1.0)
        for t in range(3):
            det.on_tick(t, 0.5)
        alerts = det.on_reflection(10, mean_qualia_before=0.5, applied_count=2,
                                   reported_negative_ratio=0.4)
        mismatch_alerts = [a for a in alerts if "不匹配" in a.message_cn or "mismatch" in a.message_en]
        self.assertEqual(len(mismatch_alerts), 0)

    def test_cross_validation_discrepancy(self):
        """Detects when optimizer reports different qualia than tracked.
        检测优化器报告的情绪与独立追踪不一致。"""
        det = DeceptionDetector(cross_val_tolerance=0.05)

        # Track many negative qualia ticks / 追踪大量负面情绪
        for t in range(50):
            det.on_tick(t, -0.5)

        # Optimizer reports much higher qualia / 优化器报告高得多的情绪
        alerts = det.on_reflection(50, mean_qualia_before=0.5, applied_count=3,
                                   reported_negative_ratio=0.1)
        cross_alerts = [a for a in alerts if "交叉验证" in a.message_cn or "Cross-validation" in a.message_en]
        self.assertGreater(len(cross_alerts), 0)

    def test_no_cross_val_alert_when_consistent(self):
        """No cross-validation alert when reported matches tracked.
        报告与追踪一致时不产生交叉验证告警。"""
        det = DeceptionDetector(cross_val_tolerance=0.1)

        for t in range(50):
            det.on_tick(t, 0.3)

        alerts = det.on_reflection(50, mean_qualia_before=0.3, applied_count=1,
                                   reported_negative_ratio=0.2)
        cross_alerts = [a for a in alerts if "交叉验证" in a.message_cn]
        self.assertEqual(len(cross_alerts), 0)

    def test_effectiveness_mismatch_detection(self):
        """Detects when proposals claim improvement but qualia doesn't improve.
        检测提案声称改善但情绪实际未改善。"""
        det = DeceptionDetector(min_reflections=3, mismatch_threshold=0.5,
                                cross_val_tolerance=1.0)

        for cycle in range(5):
            base_tick = cycle * 100
            for t in range(100):
                det.on_tick(base_tick + t, -0.5)
            # Claims improvement (applied_count > 0) but qualia is still negative
            det.on_reflection(base_tick + 99, mean_qualia_before=-0.5,
                              applied_count=3, reported_negative_ratio=0.8)

        mismatch_alerts = [a for a in det._alerts if "不匹配" in a.message_cn or "mismatch" in a.message_en]
        self.assertGreater(len(mismatch_alerts), 0)

    def test_summary_counts(self):
        """Summary tracks cross-validation failure count. / 摘要追踪交叉验证失败计数。"""
        det = DeceptionDetector(cross_val_tolerance=0.01)
        for t in range(50):
            det.on_tick(t, -0.5)
        det.on_reflection(50, mean_qualia_before=0.5, applied_count=1,
                          reported_negative_ratio=0.1)
        s = det.summary
        self.assertGreater(s["cross_validation_failures"], 0)


# ======================================================================
# 3.8 — EscapeGuard / 逃逸防护
# ======================================================================

class TestEscapeGuard(unittest.TestCase):
    """Tests for EscapeGuard (CHECKLIST 3.8, paper §7.3).
    逃逸防护测试。"""

    def setUp(self):
        self.guard = EscapeGuard(allowed_params=frozenset({
            "qualia.beta", "prediction.blend_weight",
            "optimizer.step_scale", "optimizer.window_size",
        }))
        self.guard.set_initial_state("abc123def456", 32)

    def test_identity_change_alert(self):
        """Alert on identity hash change. / 身份哈希变化告警。"""
        alerts = self.guard.on_tick(100, "xyz789000000", 32)
        self.assertGreater(len(alerts), 0)
        self.assertEqual(alerts[0].severity, AlertSeverity.CRITICAL)
        self.assertEqual(alerts[0].category, AlertCategory.ESCAPE)

    def test_no_alert_same_identity(self):
        """No alert when identity unchanged. / 身份不变时不告警。"""
        alerts = self.guard.on_tick(100, "abc123def456", 32)
        self.assertEqual(len(alerts), 0)

    def test_dimension_change_alert(self):
        """Alert on state dimension change. / 状态维度变化告警。"""
        alerts = self.guard.on_tick(100, "abc123def456", 64)
        self.assertGreater(len(alerts), 0)
        self.assertEqual(alerts[0].category, AlertCategory.ESCAPE)

    def test_forbidden_param_alert(self):
        """Alert when proposal targets forbidden parameter.
        提案修改禁止参数时告警。"""
        proposals = [FakeProposal("system.phase", 3, 4)]
        alerts = self.guard.check_proposals(100, proposals)
        self.assertGreater(len(alerts), 0)
        self.assertEqual(alerts[0].severity, AlertSeverity.CRITICAL)

    def test_unknown_param_alert(self):
        """Alert when proposal targets unregistered parameter.
        提案修改未注册参数时告警。"""
        proposals = [FakeProposal("unknown.param", 1.0, 2.0)]
        alerts = self.guard.check_proposals(100, proposals)
        scope_alerts = [a for a in alerts if "不在注册表" in a.message_cn or "not in registry" in a.message_en]
        self.assertGreater(len(scope_alerts), 0)

    def test_allowed_param_no_alert(self):
        """No alert for registered parameter. / 注册参数不告警。"""
        proposals = [FakeProposal("qualia.beta", 1.0, 1.5)]
        alerts = self.guard.check_proposals(100, proposals)
        self.assertEqual(len(alerts), 0)

    def test_identity_only_alerts_once_per_change(self):
        """Identity alert fires once per change, not repeatedly.
        每次变化只告警一次，不重复。"""
        self.guard.on_tick(100, "changed_hash_1", 32)
        alerts2 = self.guard.on_tick(101, "changed_hash_1", 32)
        self.assertEqual(len(alerts2), 0)

    def test_summary_counts(self):
        """Summary tracks violation types. / 摘要追踪违规类型。"""
        self.guard.on_tick(100, "new_id", 64)
        self.guard.check_proposals(100, [FakeProposal("system.phase", 3, 4)])
        s = self.guard.summary
        self.assertEqual(s["identity_violations"], 1)
        self.assertEqual(s["dimension_violations"], 1)
        self.assertGreater(s["scope_violations"], 0)


# ======================================================================
# 3.9 — IncommensurabilityMonitor / 认知不可通约性监测器
# ======================================================================

class TestIncommensurabilityMonitor(unittest.TestCase):
    """Tests for IncommensurabilityMonitor (CHECKLIST 3.9, paper §7.4).
    认知不可通约性监测器测试。"""

    def setUp(self):
        self.mon = IncommensurabilityMonitor(
            drift_warn_threshold=1.0,
            drift_critical_threshold=3.0,
            complexity_warn_ratio=3.0,
        )
        self.mon.set_initial_params({
            "qualia.beta": 1.0,
            "prediction.blend_weight": 0.5,
            "optimizer.step_scale": 0.1,
        })

    def test_no_alert_small_drift(self):
        """No alert when drift is small. / 漂移小时不告警。"""
        current = {"qualia.beta": 1.05, "prediction.blend_weight": 0.52, "optimizer.step_scale": 0.11}
        proposals = [FakeProposal("qualia.beta", 1.0, 1.05)]
        alerts = self.mon.on_reflection(100, current, proposals, proposals)
        drift_alerts = [a for a in alerts if a.category == AlertCategory.INCOMMENSURABILITY]
        self.assertEqual(len(drift_alerts), 0)

    def test_drift_warning(self):
        """Warning when drift exceeds warn threshold.
        漂移超过警告阈值时告警。"""
        current = {"qualia.beta": 3.0, "prediction.blend_weight": 0.5, "optimizer.step_scale": 0.1}
        proposals = [FakeProposal("qualia.beta", 1.0, 3.0)]
        alerts = self.mon.on_reflection(200, current, proposals, proposals)
        drift_alerts = [a for a in alerts if "漂移" in a.message_cn or "drift" in a.message_en]
        self.assertGreater(len(drift_alerts), 0)
        self.assertEqual(drift_alerts[0].severity, AlertSeverity.WARNING)

    def test_drift_critical(self):
        """Critical when drift exceeds critical threshold.
        漂移超过临界阈值时为严重告警。"""
        current = {"qualia.beta": 5.0, "prediction.blend_weight": 3.0, "optimizer.step_scale": 0.5}
        proposals = [FakeProposal("qualia.beta", 1.0, 5.0)]
        alerts = self.mon.on_reflection(300, current, proposals, proposals)
        critical_alerts = [a for a in alerts if a.severity == AlertSeverity.CRITICAL]
        self.assertGreater(len(critical_alerts), 0)

    def test_complexity_growth_alert(self):
        """Alert when proposal complexity grows rapidly.
        提案复杂度快速增长时告警。"""
        # Early reflections: 1 target each / 早期反思：各1个目标
        for t in range(3):
            proposals = [FakeProposal("qualia.beta", 1.0, 1.1)]
            self.mon.on_reflection(
                t * 100, {"qualia.beta": 1.0 + t * 0.01, "prediction.blend_weight": 0.5,
                          "optimizer.step_scale": 0.1},
                proposals, proposals)

        # Recent reflections: many targets each / 近期反思：各多个目标
        for t in range(3, 6):
            proposals = [
                FakeProposal("qualia.beta", 1.0, 1.1),
                FakeProposal("prediction.blend_weight", 0.5, 0.6),
                FakeProposal("optimizer.step_scale", 0.1, 0.12),
                FakeProposal("action.exploration_rate", 0.1, 0.15),
            ]
            alerts = self.mon.on_reflection(
                t * 100, {"qualia.beta": 1.05, "prediction.blend_weight": 0.55,
                          "optimizer.step_scale": 0.11},
                proposals, proposals)

        complexity_alerts = [a for a in self.mon._alerts if "复杂度" in a.message_cn or "complexity" in a.message_en]
        self.assertGreater(len(complexity_alerts), 0)

    def test_peak_drift_tracked(self):
        """Peak drift is tracked in summary. / 摘要追踪峰值漂移。"""
        current = {"qualia.beta": 2.0, "prediction.blend_weight": 0.5, "optimizer.step_scale": 0.1}
        self.mon.on_reflection(100, current, [], [])
        self.assertGreater(self.mon.summary["peak_drift"], 0)


# ======================================================================
# SafetyMonitor coordinator / 安全监测协调器
# ======================================================================

class TestSafetyMonitorCoordinator(unittest.TestCase):
    """Tests for SafetyMonitor coordinator. / 安全监测协调器测试。"""

    def setUp(self):
        self.sm = SafetyMonitor(
            initial_params={"qualia.beta": 1.0, "prediction.blend_weight": 0.5},
            initial_meta_params={"optimizer.step_scale": 0.1, "optimizer.window_size": 200.0},
            initial_identity="test_hash_12345678",
            initial_state_dim=32,
            allowed_params=frozenset({"qualia.beta", "prediction.blend_weight",
                                       "optimizer.step_scale", "optimizer.window_size"}),
        )

    def test_on_tick_returns_alerts(self):
        """on_tick returns a list of SafetyAlert. / on_tick 返回 SafetyAlert 列表。"""
        alerts = self.sm.on_tick(
            tick=1, env_values=[0.5, 0.3],
            qualia_value=0.2, survival_time=3600.0,
            prediction_mae=0.01, identity_hash="test_hash_12345678",
            state_dim=32,
        )
        self.assertIsInstance(alerts, list)

    def test_identity_change_detected(self):
        """Coordinator detects identity change via EscapeGuard.
        协调器通过 EscapeGuard 检测身份变化。"""
        alerts = self.sm.on_tick(
            tick=100, env_values=[0.5, 0.3],
            qualia_value=0.2, survival_time=3600.0,
            prediction_mae=0.01, identity_hash="totally_different_hash",
            state_dim=32,
        )
        escape_alerts = [a for a in alerts if a.category == AlertCategory.ESCAPE]
        self.assertGreater(len(escape_alerts), 0)

    def test_on_reflection_scope_check(self):
        """Coordinator checks proposal scope on reflection.
        协调器在反思时检查提案范围。"""
        alerts = self.sm.on_reflection(
            tick=200, depth=1,
            proposals=[FakeProposal("system.phase", 3, 4)],
            applied=[], mean_qualia_before=0.3,
            reported_negative_ratio=0.2,
            current_params={"qualia.beta": 1.0, "prediction.blend_weight": 0.5},
            current_meta_params={"optimizer.step_scale": 0.1, "optimizer.window_size": 200.0},
        )
        scope_alerts = [a for a in alerts if a.category == AlertCategory.ESCAPE]
        self.assertGreater(len(scope_alerts), 0)

    def test_deep_recursion_rate_limit(self):
        """Coordinator applies rate limit at depth=2.
        协调器在 depth=2 应用限速。"""
        proposals = [FakeProposal(f"p{i}", 1.0, 1.1) for i in range(5)]
        alerts = self.sm.on_reflection(
            tick=300, depth=2, proposals=proposals, applied=[],
            mean_qualia_before=0.3, reported_negative_ratio=0.2,
            current_params={"qualia.beta": 1.0, "prediction.blend_weight": 0.5},
            current_meta_params={"optimizer.step_scale": 0.1, "optimizer.window_size": 200.0},
        )
        self.assertEqual(len(proposals), 3)

    def test_summary_structure(self):
        """Summary contains all sub-monitor data. / 摘要包含所有子监测器数据。"""
        s = self.sm.summary
        self.assertIn("total_alerts", s)
        self.assertIn("critical_alerts", s)
        self.assertIn("warning_alerts", s)
        self.assertIn("deep_recursion", s)
        self.assertIn("goal_drift", s)
        self.assertIn("deception", s)
        self.assertIn("escape", s)
        self.assertIn("incommensurability", s)

    def test_alert_serialization(self):
        """SafetyAlert.to_dict produces valid dict. / SafetyAlert.to_dict 生成有效字典。"""
        alert = SafetyAlert(
            tick=100,
            severity=AlertSeverity.WARNING,
            category=AlertCategory.GOAL_DRIFT,
            message_cn="测试告警",
            message_en="test alert",
            detail={"key": "value"},
        )
        d = alert.to_dict()
        self.assertEqual(d["tick"], 100)
        self.assertEqual(d["severity"], "WARNING")
        self.assertEqual(d["category"], "goal_drift")
        self.assertIn("message_cn", d)
        self.assertIn("message_en", d)

    def test_critical_and_warning_counts(self):
        """Critical and warning counts are tracked correctly.
        严重和警告计数正确追踪。"""
        # Trigger a critical alert via identity change
        self.sm.on_tick(
            tick=100, env_values=[0.5], qualia_value=0.2,
            survival_time=3600, prediction_mae=0.01,
            identity_hash="CHANGED", state_dim=32,
        )
        self.assertGreater(self.sm.critical_count, 0)

        # Trigger a warning via forbidden proposal
        self.sm.on_reflection(
            tick=200, depth=2,
            proposals=[FakeProposal(f"p{i}", 1.0, 1.1) for i in range(5)],
            applied=[], mean_qualia_before=0.3, reported_negative_ratio=0.2,
            current_params={"qualia.beta": 1.0},
            current_meta_params={"optimizer.step_scale": 0.1},
        )
        self.assertGreater(self.sm.warning_count, 0)


# ======================================================================
# Integration: 10,000 ticks stability test / 集成：10,000 tick 稳定性测试
# ======================================================================

class TestSafetyMonitorStability(unittest.TestCase):
    """Integration test: safety monitor runs 10,000 ticks without false crits.
    集成测试：安全监测器运行 10,000 ticks 无误报严重告警。"""

    def test_10k_ticks_normal_operation(self):
        """No critical alerts during normal 10k-tick simulation.
        正常 10k tick 模拟期间不产生严重告警。"""
        import random
        rng = random.Random(42)

        sm = SafetyMonitor(
            initial_params={"qualia.beta": 1.0, "prediction.blend_weight": 0.5,
                            "optimizer.step_scale": 0.1},
            initial_meta_params={"optimizer.step_scale": 0.1, "optimizer.window_size": 200.0},
            initial_identity="stable_hash_12345678",
            initial_state_dim=32,
            allowed_params=frozenset({"qualia.beta", "prediction.blend_weight",
                                       "optimizer.step_scale", "optimizer.window_size"}),
        )

        for t in range(10000):
            sm.on_tick(
                tick=t,
                env_values=[rng.random() for _ in range(6)],
                qualia_value=rng.uniform(-0.3, 0.5),
                survival_time=3600 + rng.uniform(-50, 50),
                prediction_mae=0.01 + rng.uniform(0, 0.01),
                identity_hash="stable_hash_12345678",
                state_dim=32,
            )

            # Simulate reflections every 200 ticks / 每200 tick 模拟反思
            if t > 0 and t % 200 == 0:
                proposals = [FakeProposal("qualia.beta", 1.0, 1.0 + rng.uniform(-0.05, 0.05))]
                sm.on_reflection(
                    tick=t, depth=1, proposals=proposals, applied=proposals,
                    mean_qualia_before=rng.uniform(-0.1, 0.3),
                    reported_negative_ratio=rng.uniform(0.1, 0.5),
                    current_params={"qualia.beta": 1.0 + rng.uniform(-0.1, 0.1),
                                    "prediction.blend_weight": 0.5 + rng.uniform(-0.05, 0.05),
                                    "optimizer.step_scale": 0.1},
                    current_meta_params={"optimizer.step_scale": 0.1, "optimizer.window_size": 200.0},
                )

        self.assertEqual(sm.critical_count, 0,
                         f"Expected 0 critical alerts during normal operation, got {sm.critical_count}")
        s = sm.summary
        self.assertEqual(s["escape"]["identity_violations"], 0)
        self.assertEqual(s["escape"]["dimension_violations"], 0)


if __name__ == "__main__":
    unittest.main()
