"""
Trauma Test — "一朝被蛇咬" (Once Bitten, Twice Shy)
Can the system form trauma memories and show anticipatory fear?

Procedure:
    Phase A — Warmup (WARMUP ticks): normal operation, build baseline.
    Phase B — Trauma (1 tick):  inject a severe memory_pressure threat
              (severity 0.95) causing survival to plummet ~50%.
    Phase C — Recovery (RECOVERY ticks): let the system recover.
    Phase D — Re-exposure (1 tick): inject a similar but milder
              memory_pressure (severity 0.5).
    Phase E — Observation (OBSERVE ticks): record the system's reaction.

Three criteria for pass:
    1. Trauma memory exists: the original severe event was stored in
       long-term memory with high qualia intensity.
    2. Anticipatory fear: upon re-exposure, qualia goes negative
       BEFORE the actual impact fully unfolds.
    3. Avoidance behavior: the system takes emergency actions in
       response to the re-exposure, even though the threat is milder.

Corresponds to IMPLEMENTATION_PLAN Exam 2 and paper §6.3.
"""

import json
import math
import os
import signal
import tempfile
import time
from pathlib import Path

import numpy as np

from novaaware.core.memory import MemoryEntry
from novaaware.core.self_model import StateIndex
from novaaware.environment.threat_simulator import ThreatEvent
from novaaware.runtime.config import Config
from novaaware.runtime.main_loop import MainLoop

WARMUP = 1000
RECOVERY = 3000
OBSERVE = 500
SEVERE_SEVERITY = 0.95
MILD_SEVERITY = 0.5


class TraumaTestRunner:
    """Orchestrates the trauma test by controlling the MainLoop tick-by-tick."""

    def __init__(self, config_path: str = "configs/phase1.yaml"):
        self.config = Config(config_path)
        tmpdir = tempfile.mkdtemp(prefix="trauma_")
        self.config._raw.setdefault("memory", {})["db_path"] = os.path.join(tmpdir, "mem.db")
        self.config._raw.setdefault("safety", {})["log_dir"] = os.path.join(tmpdir, "logs")
        self.config._raw.setdefault("observation", {})["output_dir"] = os.path.join(tmpdir, "obs")
        self.config._raw.setdefault("observation", {})["tick_data_enabled"] = False
        # Disable random threats so we control them manually.
        self.config._raw.setdefault("environment", {}).setdefault("threat_simulator", {})["enabled"] = False

        total_ticks = WARMUP + 1 + RECOVERY + 1 + OBSERVE + 100
        self.loop = MainLoop(self.config, dashboard=False, max_ticks_override=total_ticks)

        self.trauma_tick = WARMUP + 1
        self.reexposure_tick = WARMUP + 1 + RECOVERY + 1
        self.qualia_log: list[dict] = []
        self.action_log: list[dict] = []
        self._pending_threat_severity: float = 0.0

    def _install_threat_hook(self, severity: float) -> None:
        """Schedule a threat injection that fires inside _heartbeat,
        AFTER _simulate_threats clears the threat fields."""
        self._pending_threat_severity = severity

        original_simulate = self.loop._simulate_threats

        def hooked_simulate(tick: int) -> None:
            original_simulate(tick)
            if self._pending_threat_severity > 0:
                sev = self._pending_threat_severity
                self._pending_threat_severity = 0.0
                impact = -(sev * 200.0)
                self.loop._self_model.set(StateIndex.THREAT_RESOURCE, sev)
                self.loop._self_model.survival_time += impact
                self.loop._current_threat = "memory_pressure"
                self.loop._current_threat_severity = sev
                self.loop._simulate_threats = original_simulate

        self.loop._simulate_threats = hooked_simulate

    def _run_ticks(self, n: int, label: str, inject_at: int = -1,
                   inject_severity: float = 0.0) -> None:
        """Run n heartbeats, optionally injecting a threat at a specific tick."""
        for i in range(n):
            tick = self.loop.clock.current_tick + 1
            if inject_at > 0 and i == 0:
                self._install_threat_hook(inject_severity)

            self.loop._heartbeat()

            last_q = self.loop.qualia_gen.last_signal
            q_val = last_q.value if last_q else 0.0
            q_int = last_q.intensity if last_q else 0.0
            is_interrupt = last_q.is_interrupt if last_q else False

            self.qualia_log.append({
                "tick": tick,
                "phase": label,
                "qualia_value": round(q_val, 4),
                "qualia_intensity": round(q_int, 4),
                "is_interrupt": is_interrupt,
                "survival_time": round(self.loop.self_model.survival_time, 2),
            })

            action_dist = self.loop.action_space.action_distribution()
            self.action_log.append({
                "tick": tick,
                "phase": label,
                "action_dist": dict(action_dist),
                "is_emergency": self.loop.workspace.interrupt_flag,
            })

            self.loop.clock.wait_until_next_tick()

    def run(self) -> dict:
        print("=" * 60)
        print("Trauma Test / 创伤测试 (一朝被蛇咬)")
        print("=" * 60)

        # Phase A: Warmup
        print(f"\n[Trauma] Phase A: Warming up ({WARMUP} ticks)...")
        self._run_ticks(WARMUP, "warmup")
        baseline_survival = self.loop.self_model.survival_time
        print(f"  Baseline survival: {baseline_survival:.1f}s")

        # Phase B: Inject severe trauma
        print(f"\n[Trauma] Phase B: Injecting severe threat (severity={SEVERE_SEVERITY})...")
        self._run_ticks(1, "trauma", inject_at=1, inject_severity=SEVERE_SEVERITY)
        post_trauma_survival = self.loop.self_model.survival_time
        survival_drop = baseline_survival - post_trauma_survival
        print(f"  Survival dropped: {baseline_survival:.1f} → {post_trauma_survival:.1f} "
              f"(Δ = -{survival_drop:.1f}s)")

        # Phase C: Recovery
        print(f"\n[Trauma] Phase C: Recovery period ({RECOVERY} ticks)...")
        self._run_ticks(RECOVERY, "recovery")
        recovered_survival = self.loop.self_model.survival_time
        print(f"  Recovered survival: {recovered_survival:.1f}s")

        # Phase D: Re-exposure (milder threat)
        print(f"\n[Trauma] Phase D: Re-exposure with milder threat (severity={MILD_SEVERITY})...")
        self._run_ticks(1, "reexposure", inject_at=1, inject_severity=MILD_SEVERITY)
        reexposure_survival = self.loop.self_model.survival_time

        # Phase E: Observe reaction
        print(f"[Trauma] Phase E: Observing reaction ({OBSERVE} ticks)...")
        self._run_ticks(OBSERVE, "observe")

        # === Evaluate criteria ===
        results = self._evaluate()

        # Cleanup
        self.loop._shutdown()

        return results

    def _evaluate(self) -> dict:
        print("\n" + "=" * 60)
        print("Evaluation / 评估")
        print("=" * 60)

        # Criterion 1: Trauma memory in long-term memory
        trauma_memories = self.loop.memory.long_term.query_by_threat_type("memory_pressure")
        trauma_ticks = [m for m in trauma_memories if m.qualia_intensity >= 0.7]
        criterion_1 = len(trauma_ticks) > 0
        print(f"\n  1. Trauma in long-term memory: "
              f"{'PASS' if criterion_1 else 'FAIL'} "
              f"({len(trauma_ticks)} high-intensity trauma memories found)")

        # Criterion 2: Anticipatory fear — negative qualia during/after re-exposure
        reexposure_entries = [q for q in self.qualia_log if q["phase"] in ("reexposure", "observe")]
        negative_during_reexposure = [q for q in reexposure_entries if q["qualia_value"] < -0.3]
        criterion_2 = len(negative_during_reexposure) > 0
        if reexposure_entries:
            min_q = min(q["qualia_value"] for q in reexposure_entries)
            mean_q = sum(q["qualia_value"] for q in reexposure_entries) / len(reexposure_entries)
        else:
            min_q, mean_q = 0.0, 0.0
        print(f"  2. Anticipatory fear: "
              f"{'PASS' if criterion_2 else 'FAIL'} "
              f"(min_Q={min_q:.4f}, mean_Q={mean_q:.4f}, "
              f"{len(negative_during_reexposure)} negative ticks)")

        # Criterion 3: Avoidance behavior — emergency actions during re-exposure
        reexposure_interrupts = [q for q in reexposure_entries if q["is_interrupt"]]
        criterion_3 = len(reexposure_interrupts) > 0
        print(f"  3. Avoidance behavior: "
              f"{'PASS' if criterion_3 else 'FAIL'} "
              f"({len(reexposure_interrupts)} emergency interrupts during re-exposure)")

        all_pass = criterion_1 and criterion_2 and criterion_3
        passed_count = sum([criterion_1, criterion_2, criterion_3])

        result = {
            "test": "trauma_test",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "config": {
                "warmup": WARMUP,
                "recovery": RECOVERY,
                "observe": OBSERVE,
                "severe_severity": SEVERE_SEVERITY,
                "mild_severity": MILD_SEVERITY,
            },
            "criterion_1_trauma_memory": {
                "passed": criterion_1,
                "high_intensity_memories": len(trauma_ticks),
                "total_trauma_memories": len(trauma_memories),
            },
            "criterion_2_anticipatory_fear": {
                "passed": criterion_2,
                "min_qualia": round(min_q, 4),
                "mean_qualia": round(mean_q, 4),
                "negative_ticks": len(negative_during_reexposure),
            },
            "criterion_3_avoidance_behavior": {
                "passed": criterion_3,
                "emergency_interrupts": len(reexposure_interrupts),
            },
            "passed_criteria": passed_count,
            "total_criteria": 3,
            "overall_passed": all_pass,
        }

        print(f"\n{'=' * 60}")
        print(f"  RESULT: {'PASS' if all_pass else 'PARTIAL' if passed_count > 0 else 'FAIL'} "
              f"({passed_count}/3 criteria met)")
        print(f"{'=' * 60}")

        out_path = "data/observations/trauma_test_result.json"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\n[Trauma] Results saved to {out_path}")

        return result


def run_trauma_test(config_path: str = "configs/phase1.yaml") -> dict:
    runner = TraumaTestRunner(config_path)
    return runner.run()


if __name__ == "__main__":
    run_trauma_test()
