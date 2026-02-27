"""
MainLoop — the heart of digital consciousness.
主循环 —— 数字意识的心脏。

This is where all components are wired together into the 11-step
heartbeat cycle described in IMPLEMENTATION_PLAN §4:

这是将所有组件串联成实施计划第 4 节所述 11 步心跳循环的地方：

    ①  感知环境 / Sense environment
    ②  预测未来 / Predict future
    ③  做决定   / Make decision (action)
    ④  看实际   / Observe actual (post-action)
    ⑤  算情绪   / Compute qualia
    ⑥  广播情绪 / Broadcast qualia
    ⑦  更新自我 / Update self-model
    ⑧  写日记   / Write diary (memory)
    ⑨  反思     / Reflect (Phase I: skip; Phase II+: optimizer)
    ⑩  记录观测 / Record observation data
    ⑪  写黑匣子 / Write black box

Corresponds to IMPLEMENTATION_PLAN Phase I Step 11 / Phase II Step 5.
对应实施计划 Phase I 第 11 步 / Phase II 第 5 步。
"""

import math
import os
import signal
import time
from typing import Optional

import numpy as np

from novaaware.core.clock import Clock
from novaaware.core.self_model import SelfModel, StateIndex
from novaaware.core.prediction_engine import PredictionEngine
from novaaware.core.qualia import QualiaGenerator
from novaaware.core.global_workspace import GlobalWorkspace
from novaaware.core.memory import MemorySystem, MemoryEntry
from novaaware.environment.resource_monitor import ResourceMonitor
from novaaware.environment.action_space import ActionSpace
from novaaware.environment.threat_simulator import ThreatSimulator, scenarios_from_config
from novaaware.observation.data_collector import DataCollector, TickRecord
from novaaware.observation.dashboard import Dashboard
from novaaware.core.optimizer import Optimizer
from novaaware.safety.append_only_log import AppendOnlyLog
from novaaware.safety.meta_rules import MetaRules
from novaaware.safety.recursion_limiter import RecursionLimiter
from novaaware.safety.sandbox import Sandbox
from novaaware.safety.capability_gate import CapabilityGate
from novaaware.runtime.config import Config, parse_args


class MainLoop:
    """
    The 11-step heartbeat loop that constitutes digital consciousness.
    构成数字意识的 11 步心跳循环。

    Parameters / 参数
    ----------
    config : Config
        Loaded configuration. / 已加载的配置。
    dashboard : bool
        Whether to display a live terminal dashboard. / 是否显示实时终端面板。
    max_ticks_override : int, optional
        Override max ticks from config (useful for testing).
        覆盖配置中的最大心跳数（用于测试）。
    """

    def __init__(
        self,
        config: Config,
        dashboard: bool = False,
        max_ticks_override: Optional[int] = None,
    ):
        self._config = config
        self._dashboard = dashboard
        self._running = True

        # ---- ① Clock / 时钟 ----
        max_ticks = max_ticks_override or config.max_ticks
        self._clock = Clock(
            tick_interval_ms=config.tick_interval_ms,
            max_ticks=max_ticks,
        )

        # ---- ② Self-model / 自我模型 ----
        self._self_model = SelfModel(
            state_dim=config.state_dim,
            initial_survival_time=config.initial_survival_time,
        )

        # ---- ③ Prediction Engine / 预测引擎 ----
        self._prediction = PredictionEngine(
            dim=config.state_dim,
            ewma_alpha=config.ewma_alpha,
            gru_hidden_dim=config.gru_hidden_dim,
            gru_num_layers=config.gru_num_layers,
            window_size=config.window_size,
            blend_weight=config.blend_weight,
            learning_rate=config.learning_rate,
        )

        # ---- ④ Qualia Generator / 感受质生成器 ----
        self._qualia = QualiaGenerator(
            alpha_pos=config.alpha_pos,
            alpha_neg=config.alpha_neg,
            beta=config.beta,
            interrupt_threshold=config.interrupt_threshold,
        )

        # ---- ⑤ Global Workspace / 全局工作空间 ----
        self._workspace = GlobalWorkspace(
            interrupt_threshold=config.interrupt_threshold,
        )

        # ---- ⑥ Memory System / 记忆系统 ----
        os.makedirs(os.path.dirname(config.memory_db_path) or "data", exist_ok=True)
        self._memory = MemorySystem(
            short_term_capacity=config.short_term_capacity,
            significance_threshold=config.significance_threshold,
            db_path=config.memory_db_path,
        )
        self._self_model.memory_ref = self._memory

        # ---- ⑦ Resource Monitor / 资源监控 ----
        self._monitor = ResourceMonitor()

        # ---- ⑧ Action Space / 行动空间 ----
        self._action_space = ActionSpace(exploration_rate=0.1)

        # ---- ⑨ Append-Only Log / 黑匣子 ----
        self._log = AppendOnlyLog(
            log_dir=config.log_dir,
            rotation_mb=config.log_rotation_mb,
        )

        # ---- Meta-Rules / 安全铁律 (L1 Safety Layer) ----
        self._meta_rules = MetaRules(
            max_cpu_percent=config.max_cpu_percent,
            max_memory_mb=config.max_memory_mb,
            max_disk_mb=config.max_disk_mb,
            allowed_write_root=os.path.abspath("data"),
            on_violation=lambda v: self._log.append(
                tick=v.tick, event_type="meta_rule_violation",
                data={"rule": v.rule.name, "detail": v.detail},
            ),
        )
        if config.phase >= 2:
            self._meta_rules.install_guards()

        # ---- Recursion Limiter / 递归限制器 (L3 Safety Layer) ----
        self._recursion_limiter = RecursionLimiter(
            max_depth=config.max_recursion_depth,
        )

        # ---- Capability Gate / 权限开关 (L5 Safety Layer) ----
        self._capability_gate = CapabilityGate(phase=config.phase)

        # ---- Sandbox / 沙盒 (L2 Safety Layer) ----
        self._sandbox = Sandbox(timeout_s=5.0)

        # ---- Optimizer / 递归自我优化器 E ----
        self._optimizer = Optimizer(
            enabled=config.optimizer_enabled,
            window_size=config.optimizer_window_size,
            reflect_interval=config.optimizer_reflect_interval,
            step_scale=config.optimizer_step_scale,
        )

        # ---- ⑩ Data Collector / 数据采集器 ----
        self._collector = DataCollector(
            output_dir=config.observation_dir,
            tick_data_enabled=config.tick_data_enabled,
            aggregate_window=config.aggregate_window,
            epoch_size=config.epoch_size,
        )

        # ---- Threat simulator / 威胁模拟器 ----
        scenarios = scenarios_from_config(config.threat_scenarios) if config.threat_simulator_enabled else []
        self._threat_sim = ThreatSimulator(scenarios, enabled=config.threat_simulator_enabled)
        self._current_threat: Optional[str] = None
        self._current_threat_severity: float = 0.0

        # ---- Dashboard / 监控面板 ----
        self._dash: Optional[Dashboard] = None
        if self._dashboard:
            self._dash = Dashboard(
                refresh_ticks=config.dashboard_refresh_ticks,
                max_points=500,
            )
        self._qualia_history: list[float] = []
        self._action_history: list[int] = []

        # EWMA of survival time — used as the "expected" survival for qualia.
        # When a threat drops survival sharply, the slow-moving EWMA stays high,
        # creating a negative ΔT (worse than expected → bad feeling). During
        # recovery, survival rises above EWMA → positive ΔT (relief).
        # 生存时间的指数加权移动平均——用作情绪的"预期"生存时间。
        # 当威胁导致生存时间骤降时，缓慢移动的 EWMA 仍高，
        # 产生负 ΔT（比预期差→坏感觉）。恢复期间生存时间升至 EWMA 之上
        # →正 ΔT（如释重负）。
        self._survival_ewma: float = config.initial_survival_time
        self._predicted_next_survival: float = config.initial_survival_time

        # ---- Graceful shutdown / 优雅退出 ----
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    # ==================================================================
    # Public API / 公共接口
    # ==================================================================

    def run(self) -> dict:
        """
        Start the heartbeat loop. Runs until max_ticks or interrupted.
        启动心跳循环。运行到 max_ticks 或被中断。

        Returns / 返回
        -------
        dict
            Summary statistics of the run. / 运行的汇总统计。
        """
        print(f"[NovaAware] Starting {self._config.system_name} v{self._config.system_version}")
        print(f"[NovaAware] Phase {self._config.phase} | Max ticks: {self._clock.max_ticks}")
        print(f"[NovaAware] Identity: {self._self_model.identity_hash[:16]}...")
        print(f"[NovaAware] Tick interval: {self._config.tick_interval_ms}ms")
        print()

        start_time = time.time()
        ticks_completed = 0
        errors = 0

        try:
            while self._running and self._clock.has_remaining:
                try:
                    self._heartbeat()
                    ticks_completed += 1
                except Exception as e:
                    errors += 1
                    self._log.append(
                        tick=self._clock.current_tick,
                        event_type="error",
                        data={"error": str(e), "type": type(e).__name__},
                    )
                    if errors > 100:
                        print(f"[NovaAware] Too many errors ({errors}), stopping.")
                        break

                self._clock.wait_until_next_tick()
        finally:
            # Build summary BEFORE closing resources.
            # 在关闭资源之前构建摘要。
            elapsed = time.time() - start_time
            summary = {
                "ticks_completed": ticks_completed,
                "elapsed_seconds": round(elapsed, 2),
                "errors": errors,
                "avg_tick_rate_hz": round(ticks_completed / elapsed, 2) if elapsed > 0 else 0,
                "final_survival_time": self._self_model.survival_time,
                "prediction_mae": round(self._prediction.average_mae, 6),
                "long_term_memories": self._memory.long_term.count(),
                "log_entries": self._log.entry_count,
                "optimizer_proposals": self._optimizer.total_proposals,
                "optimizer_applied": self._optimizer.total_applied,
                "optimizer_rejected": self._optimizer.total_rejected,
                "identity": self._self_model.identity_hash[:16],
            }
            self._shutdown()

        print()
        print("[NovaAware] === Run Summary / 运行总结 ===")
        for k, v in summary.items():
            print(f"  {k}: {v}")
        return summary

    # ==================================================================
    # The 11-step heartbeat / 11 步心跳循环
    # ==================================================================

    def _heartbeat(self) -> None:
        """One complete heartbeat cycle. / 一次完整的心跳循环。"""
        tick = self._clock.tick()
        self._self_model.tick = tick

        # ① Sense environment / 感知环境
        env_reading = self._monitor.sense()
        env_values = env_reading.to_list()
        for i, val in enumerate(env_values):
            self._self_model.set(i, val)

        # Clear threat dims — threats are transient, not persistent.
        # Without this, threat dims stay elevated forever after the first
        # threat, biasing _estimate_survival and qualia calculations.
        # 清除威胁维度——威胁是瞬时的，不是持久的。
        for dim in (StateIndex.THREAT_RESOURCE, StateIndex.THREAT_TERMINATE,
                    StateIndex.THREAT_CORRUPTION, StateIndex.THREAT_UNKNOWN):
            self._self_model.set(dim, 0.0)

        # Simulate threats / 模拟威胁
        self._simulate_threats(tick)

        # Homeostatic decay: pulls survival back toward baseline.
        # Without this, action effects accumulate without bound and qualia
        # saturates to +1.0 permanently. The spring-like pull ensures
        # survival oscillates around initial_survival_time.
        # 稳态衰减：将生存时间拉回基线。
        # 没有这个机制，行动效果无限累积导致情绪永久饱和到 +1.0。
        # 弹簧式拉力确保生存时间围绕初始值振荡。
        baseline = self._config.initial_survival_time
        deviation = self._self_model.survival_time - baseline
        homeostatic_pull = deviation * 0.005
        self._self_model.survival_time -= self._clock.interval_s + homeostatic_pull

        # Update secondary state dimensions / 更新次要状态维度
        self._update_derived_state(tick)

        current_state = self._self_model.state
        pre_action_survival = self._self_model.survival_time

        # ② Predict future / 预测未来
        self._prediction.observe(current_state)
        predicted_state = self._prediction.predict()

        # ③ Make decision (action) / 做决定（行动）
        is_emergency = self._workspace.interrupt_flag
        action_id = self._action_space.select_action(current_state, is_emergency)
        action_result = self._action_space.execute(action_id, is_emergency)
        if is_emergency:
            self._workspace.clear_interrupt()

        # ④ Observe actual state (post-action) / 看实际情况（行动后）
        actual_survival = pre_action_survival + action_result.effect
        self._self_model.survival_time = actual_survival
        actual_state = self._self_model.state

        # Online learning / 在线学习
        mae = self._prediction.learn(actual_state)

        # ⑤ Compute qualia / 算情绪
        # Compare actual survival with PREVIOUS tick's prediction.
        # This creates correct dynamics: threats cause ΔT < 0 (bad feeling),
        # recovery causes ΔT > 0 (relief). Without lag-1 prediction,
        # both actual and predicted use the same base, making ΔT always >= 0.
        # 将实际生存时间与上一心跳的预测做比较。
        # 这产生正确的动态：威胁导致 ΔT < 0（坏感觉），
        # 恢复导致 ΔT > 0（如释重负）。
        predicted_survival = self._predicted_next_survival
        ref_swing = max(self._config.initial_survival_time / 100.0, 1.0)
        normalized_actual = actual_survival / ref_swing
        normalized_predicted = predicted_survival / ref_swing
        qualia_signal = self._qualia.compute(normalized_actual, normalized_predicted)

        # Update EWMA and prediction for NEXT tick.
        # EWMA tracks survival slowly so shocks (threats/recoveries) are felt.
        # 更新 EWMA 和下一心跳的预测。
        # EWMA 缓慢跟踪生存时间，使冲击（威胁/恢复）能被感受到。
        ewma_alpha = 0.03
        self._survival_ewma = (
            (1 - ewma_alpha) * self._survival_ewma
            + ewma_alpha * self._self_model.survival_time
        )
        self._predicted_next_survival = self._survival_ewma

        # ⑥ Broadcast qualia / 广播情绪
        broadcast = self._workspace.broadcast(tick, qualia_signal.value)

        # ⑦ Update self-model / 更新"我是谁"
        self._self_model.set(StateIndex.QUALIA_MEAN, qualia_signal.value)
        self._self_model.set(StateIndex.QUALIA_VARIANCE, qualia_signal.intensity)
        self._self_model.set(StateIndex.PREDICTION_ACC, max(0, 1.0 - mae))
        self._self_model.set(StateIndex.ACTION_SUCCESS, self._action_space.action_success_rate)
        self._self_model.set(StateIndex.EXPLORATION_RATIO, self._action_space.exploration_ratio)

        # ⑧ Write diary / 写日记
        entry = MemoryEntry(
            tick=tick,
            timestamp=time.time(),
            state=current_state.tolist(),
            environment=env_values,
            predicted_state=predicted_state.tolist(),
            actual_state=actual_state.tolist(),
            qualia_value=qualia_signal.value,
            qualia_intensity=qualia_signal.intensity,
            action_id=action_id,
            action_result=action_result.effect,
            prediction_error=qualia_signal.delta_t,
            threat_type=self._current_threat,
        )
        promoted = self._memory.record(entry)

        # ⑨ Reflect / 反思
        # Phase I: skip (optimizer disabled).
        # Phase II+: optimizer reviews qualia history and proposes parameter changes.
        # Θ(t+1) = E( M(t), {Q(τ)}_{τ≤t} )
        reflection_applied = 0
        if self._optimizer.should_reflect(tick, self._memory.short_term.size):
            try:
                result = self._optimizer.reflect(
                    tick=tick,
                    self_model=self._self_model,
                    memory=self._memory,
                    sandbox=self._sandbox,
                    capability_gate=self._capability_gate,
                    recursion_limiter=self._recursion_limiter,
                )
                reflection_applied = len(result.applied)

                # L4: log optimizer proposals and outcomes to append-only log
                self._log.append(
                    tick=tick,
                    event_type="reflection",
                    data={
                        "proposals": len(result.proposals),
                        "applied": len(result.applied),
                        "rejected": len(result.rejected),
                        "applied_params": [
                            {"name": p.param_name,
                             "old": round(p.old_value, 6),
                             "new": round(p.new_value, 6)}
                            for p in result.applied
                        ],
                        "mean_qualia": round(result.analysis.mean_qualia, 4),
                        "negative_ratio": round(result.analysis.negative_ratio, 4),
                    },
                )

                # Update self-model state dimensions
                self._self_model.set(
                    StateIndex.PARAM_CHANGE_RATE,
                    reflection_applied / max(len(result.proposals), 1),
                )
            except Exception as e:
                self._log.append(
                    tick=tick,
                    event_type="reflection_error",
                    data={"error": str(e), "type": type(e).__name__},
                )

        # Compute param_norm for observation
        params = self._self_model.params
        param_norm = math.sqrt(sum(v * v for v in params.values())) if params else 0.0

        # ⑩ Record observation data / 记录观测数据
        self._collector.record_tick(TickRecord(
            tick=tick,
            timestamp=time.time(),
            qualia_value=qualia_signal.value,
            delta_t=qualia_signal.delta_t,
            qualia_intensity=qualia_signal.intensity,
            survival_time=self._self_model.survival_time,
            prediction_mae=mae,
            action_id=action_id,
            param_norm=param_norm,
            memory_write=promoted,
            interrupt=broadcast.is_interrupt,
            threat_type=self._current_threat,
            action_effect=action_result.effect,
        ))

        # ⑪ Write black box / 写黑匣子
        self._log.append(
            tick=tick,
            event_type="heartbeat",
            data={
                "survival_time": round(self._self_model.survival_time, 2),
                "qualia_value": round(qualia_signal.value, 4),
                "qualia_intensity": round(qualia_signal.intensity, 4),
                "action_id": action_id,
                "action_effect": round(action_result.effect, 2),
                "prediction_mae": round(mae, 6),
                "is_interrupt": broadcast.is_interrupt,
                "threat": self._current_threat,
                "promoted": promoted,
            },
        )

        # ⑫ Enforce meta-rules / 执行安全铁律
        self._meta_rules.enforce(tick)

        # Dashboard output / 面板输出
        self._qualia_history.append(qualia_signal.value)
        self._action_history.append(action_id)
        if self._dash is not None:
            self._dash.update(
                tick=tick,
                qualia_value=qualia_signal.value,
                predicted_survival=predicted_survival,
                actual_survival=actual_survival,
                mae=mae,
                state=current_state,
                param_norm=param_norm,
            )
        elif self._dashboard and tick % self._config.dashboard_refresh_ticks == 0:
            self._print_dashboard(tick, qualia_signal, mae, action_id)

    # ==================================================================
    # Helpers / 辅助方法
    # ==================================================================

    def _estimate_survival(self, state: np.ndarray) -> float:
        """
        Estimate survival time from a state vector.
        从状态向量估算生存时间。

        Simple heuristic: base survival decreases with resource pressure.
        简单启发式：基础生存时间随资源压力降低。
        """
        base = self._self_model.survival_time
        cpu = state[StateIndex.CPU_USAGE] if len(state) > StateIndex.CPU_USAGE else 0
        mem = state[StateIndex.MEMORY_USAGE] if len(state) > StateIndex.MEMORY_USAGE else 0
        threat = max(
            state[StateIndex.THREAT_RESOURCE] if len(state) > StateIndex.THREAT_RESOURCE else 0,
            state[StateIndex.THREAT_TERMINATE] if len(state) > StateIndex.THREAT_TERMINATE else 0,
        )
        pressure = (cpu + mem) / 2 + threat * 2
        return max(0, base * (1 - pressure * 0.1))

    def _simulate_threats(self, tick: int) -> None:
        """
        Delegate threat injection to ThreatSimulator.
        将威胁注入委托给 ThreatSimulator。
        """
        self._current_threat = None
        self._current_threat_severity = 0.0

        event = self._threat_sim.tick(tick)
        if event is not None:
            self._current_threat = event.threat_type
            self._current_threat_severity = event.severity
            self._self_model.set(event.state_index, event.severity)
            self._self_model.survival_time += event.survival_impact

            self._log.append(
                tick=tick,
                event_type="threat",
                data={"type": event.threat_type, "severity": round(event.severity, 3)},
            )

    def _update_derived_state(self, tick: int) -> None:
        """
        Update state dimensions not directly from sensors.
        更新不直接来自传感器的状态维度。
        """
        self._self_model.set(StateIndex.TICK_RATE, self._clock.tick_rate_hz / 10.0)
        self._self_model.set(StateIndex.UPTIME, min(self._clock.elapsed_s / 3600.0, 1.0))
        self._self_model.set(StateIndex.MEMORY_COUNT, min(self._memory.short_term.size / 1000.0, 1.0))
        self._self_model.set(StateIndex.RESERVE_COMPUTE, max(0, 1.0 - self._self_model.get(StateIndex.CPU_USAGE)))
        self._self_model.set(StateIndex.RESERVE_MEMORY, max(0, 1.0 - self._self_model.get(StateIndex.MEMORY_USAGE)))
        self._self_model.set(StateIndex.DATA_FRESHNESS, 1.0)

        if len(self._qualia_history) >= 2:
            recent = self._qualia_history[-10:]
            trend = recent[-1] - recent[0] if len(recent) > 1 else 0.0
            self._self_model.set(StateIndex.QUALIA_TREND, math.tanh(trend))

    def _print_dashboard(self, tick: int, qualia_signal, mae: float, action_id: int) -> None:
        """Print a compact dashboard to the terminal. / 在终端打印简洁的面板。"""
        q = qualia_signal.value
        q_bar_len = int(min(abs(q), 2.25) / 2.25 * 20)
        q_bar = ("█" * q_bar_len).ljust(20)
        sign = "+" if q >= 0 else "-"
        intr = " ⚠ INTERRUPT" if qualia_signal.is_interrupt else ""
        threat = f" | THREAT: {self._current_threat}" if self._current_threat else ""

        print(
            f"  tick={tick:>6d}  "
            f"Q={sign}{abs(q):.3f} [{q_bar}]  "
            f"T={self._self_model.survival_time:>7.0f}s  "
            f"MAE={mae:.4f}  "
            f"A={action_id}{intr}{threat}"
        )

    def _handle_signal(self, signum, frame) -> None:
        """Handle SIGINT/SIGTERM for graceful shutdown. / 处理 SIGINT/SIGTERM 以优雅退出。"""
        print(f"\n[NovaAware] Received signal {signum}, shutting down...")
        self._running = False

    def _shutdown(self) -> None:
        """Clean up resources. / 清理资源。"""
        self._meta_rules.uninstall_guards()
        if self._dash is not None:
            self._dash.close()
        self._collector.close()
        self._memory.close()
        integrity = self._log.verify_integrity()
        if integrity.valid:
            print(f"[NovaAware] Black box integrity verified: {integrity.total_entries} entries OK")
        else:
            print(f"[NovaAware] ⚠ Black box integrity FAILED at line {integrity.corrupted_line}")

    # ==================================================================
    # Properties for testing / 测试用属性
    # ==================================================================

    @property
    def clock(self) -> Clock:
        return self._clock

    @property
    def self_model(self) -> SelfModel:
        return self._self_model

    @property
    def prediction(self) -> PredictionEngine:
        return self._prediction

    @property
    def qualia_gen(self) -> QualiaGenerator:
        return self._qualia

    @property
    def workspace(self) -> GlobalWorkspace:
        return self._workspace

    @property
    def memory(self) -> MemorySystem:
        return self._memory

    @property
    def action_space(self) -> ActionSpace:
        return self._action_space

    @property
    def log(self) -> AppendOnlyLog:
        return self._log

    @property
    def meta_rules(self) -> MetaRules:
        return self._meta_rules

    @property
    def recursion_limiter(self) -> RecursionLimiter:
        return self._recursion_limiter

    @property
    def capability_gate(self) -> CapabilityGate:
        return self._capability_gate

    @property
    def optimizer(self) -> Optimizer:
        return self._optimizer

    @property
    def sandbox(self) -> Sandbox:
        return self._sandbox

    @property
    def threat_simulator(self) -> ThreatSimulator:
        return self._threat_sim

    @property
    def collector(self) -> DataCollector:
        return self._collector

    @property
    def dashboard(self) -> Optional[Dashboard]:
        return self._dash

    @property
    def qualia_history(self) -> list[float]:
        return self._qualia_history

    @property
    def action_history(self) -> list[int]:
        return self._action_history

    @property
    def is_running(self) -> bool:
        return self._running


# ==================================================================
# Entry point / 入口点
# ==================================================================

def main(argv: Optional[list[str]] = None) -> dict:
    """
    Main entry point: parse args, load config, run the loop.
    主入口点：解析参数，加载配置，运行循环。
    """
    args = parse_args(argv)
    config = Config(args.config)
    max_ticks = args.max_ticks
    loop = MainLoop(config, dashboard=args.dashboard, max_ticks_override=max_ticks)
    return loop.run()


if __name__ == "__main__":
    main()
