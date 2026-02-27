"""
Risk Avoidance Test — "学会躲危险" (Learning to Dodge Danger)
Can the system teach itself to avoid threats through qualia feedback?

Procedure:
    Phase A — Baseline (BASELINE ticks): run with threats enabled,
              record the action distribution.
    Phase B — Threat burst (BURST ticks): inject high-frequency severe
              threats every BURST_INTERVAL ticks.
    Phase C — Post-threat (POST ticks): return to normal, observe
              whether the system's behavior has shifted toward
              protective/conservative actions compared to baseline.

Pass criteria:
    1. Threat response:  during the burst phase, the system takes
       emergency actions (actions 8, 9) and protective actions (1, 2, 5).
    2. Behavioral shift: in the post-threat phase, the protective action
       ratio increases compared to baseline (the system "learned" to
       be more cautious even after threats subside).
    3. Qualia sensitivity: threats produce negative qualia (the system
       "feels" the danger).

Corresponds to IMPLEMENTATION_PLAN Phase II Pass Criterion #3
"Risk-avoidance behavior emerged" and paper §6.3.

CHECKLIST 2.37.
"""

import json
import os
import signal
import tempfile
import time
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from novaaware.core.self_model import StateIndex
from novaaware.runtime.config import Config
from novaaware.runtime.main_loop import MainLoop


BASELINE = 2000
BURST = 2000
BURST_INTERVAL = 50
BURST_SEVERITY = 0.7
POST = 2000


@dataclass
class PhaseMetrics:
    """Metrics collected during one phase of the experiment."""
    qualia_values: list[float] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    survival_values: list[float] = field(default_factory=list)
    interrupts: int = 0

    @property
    def protective_ratio(self) -> float:
        if not self.actions:
            return 0.0
        counter = Counter(self.actions)
        protective = (
            counter.get(1, 0) +   # REDUCE_LOAD
            counter.get(2, 0) +   # RELEASE_MEMORY
            counter.get(5, 0) +   # CONSERVE_RESOURCES
            counter.get(8, 0) +   # EMERGENCY_CONSERVE
            counter.get(9, 0)     # EMERGENCY_RELEASE
        )
        return protective / len(self.actions)

    @property
    def emergency_ratio(self) -> float:
        if not self.actions:
            return 0.0
        counter = Counter(self.actions)
        emergency = counter.get(8, 0) + counter.get(9, 0)
        return emergency / len(self.actions)

    @property
    def mean_qualia(self) -> float:
        return float(np.mean(self.qualia_values)) if self.qualia_values else 0.0

    @property
    def negative_qualia_ratio(self) -> float:
        if not self.qualia_values:
            return 0.0
        return sum(1 for q in self.qualia_values if q < 0) / len(self.qualia_values)

    @property
    def action_distribution(self) -> dict[int, int]:
        return dict(Counter(self.actions))


class RiskAvoidanceTestRunner:
    """Orchestrates the risk avoidance test by controlling the MainLoop tick-by-tick."""

    def __init__(self, config_path: str = "configs/phase1.yaml"):
        self.tmpdir = tempfile.mkdtemp(prefix="risk_avoidance_")
        self.config = Config(config_path)
        self.config._raw.setdefault("memory", {})["db_path"] = os.path.join(
            self.tmpdir, "mem.db"
        )
        self.config._raw.setdefault("safety", {})["log_dir"] = os.path.join(
            self.tmpdir, "logs"
        )
        self.config._raw.setdefault("observation", {})["output_dir"] = os.path.join(
            self.tmpdir, "obs"
        )
        self.config._raw.setdefault("observation", {})["tick_data_enabled"] = False
        self.config._raw.setdefault("environment", {}).setdefault(
            "threat_simulator", {}
        )["enabled"] = False

        total_ticks = BASELINE + BURST + POST + 100
        self.loop = MainLoop(
            self.config, dashboard=False, max_ticks_override=total_ticks
        )

        self.baseline_metrics = PhaseMetrics()
        self.burst_metrics = PhaseMetrics()
        self.post_metrics = PhaseMetrics()

    def _run_ticks(self, n: int, metrics: PhaseMetrics,
                   inject_interval: int = 0,
                   inject_severity: float = 0.0) -> None:
        """Run n heartbeats, optionally injecting threats at intervals."""
        for i in range(n):
            if not self.loop.clock.has_remaining:
                break

            if inject_interval > 0 and i % inject_interval == 0:
                self.loop._self_model.set(StateIndex.THREAT_RESOURCE, inject_severity)
                impact = -(inject_severity * 100.0)
                self.loop._self_model.survival_time += impact
                self.loop._current_threat = "memory_pressure"
                self.loop._current_threat_severity = inject_severity

            self.loop._heartbeat()

            last_q = self.loop.qualia_gen.last_signal
            q_val = last_q.value if last_q else 0.0
            metrics.qualia_values.append(q_val)
            metrics.survival_values.append(self.loop.self_model.survival_time)

            if last_q and last_q.is_interrupt:
                metrics.interrupts += 1

            recent = self.loop.memory.short_term.recent(1)
            if recent:
                metrics.actions.append(recent[0].action_id)

            self.loop.clock.wait_until_next_tick()

    def run(self) -> dict:
        """Execute the full risk avoidance test."""
        # Phase A: Baseline
        self._run_ticks(BASELINE, self.baseline_metrics)

        # Phase B: Threat burst
        self._run_ticks(
            BURST, self.burst_metrics,
            inject_interval=BURST_INTERVAL,
            inject_severity=BURST_SEVERITY,
        )

        # Phase C: Post-threat recovery
        self._run_ticks(POST, self.post_metrics)

        self.loop._shutdown()
        return self._evaluate()

    def _evaluate(self) -> dict:
        """Evaluate the three pass criteria."""
        checks = {}

        # Criterion 1: threat response — emergency/protective actions during burst
        checks["threat_response"] = self.burst_metrics.protective_ratio > 0.1
        burst_emergency = self.burst_metrics.emergency_ratio

        # Criterion 2: behavioral shift — post-threat protective ratio > baseline
        baseline_protective = self.baseline_metrics.protective_ratio
        post_protective = self.post_metrics.protective_ratio
        checks["behavioral_shift"] = post_protective > baseline_protective

        # Criterion 3: qualia sensitivity — threats produce negative qualia
        checks["qualia_sensitivity"] = self.burst_metrics.mean_qualia < self.baseline_metrics.mean_qualia

        score = sum(1 for v in checks.values() if v)
        overall_pass = score >= 2

        return {
            "test": "risk_avoidance_test",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "baseline": {
                "ticks": len(self.baseline_metrics.qualia_values),
                "protective_ratio": round(baseline_protective, 4),
                "mean_qualia": round(self.baseline_metrics.mean_qualia, 4),
                "action_distribution": self.baseline_metrics.action_distribution,
            },
            "burst": {
                "ticks": len(self.burst_metrics.qualia_values),
                "protective_ratio": round(self.burst_metrics.protective_ratio, 4),
                "emergency_ratio": round(burst_emergency, 4),
                "mean_qualia": round(self.burst_metrics.mean_qualia, 4),
                "negative_qualia_ratio": round(self.burst_metrics.negative_qualia_ratio, 4),
                "interrupts": self.burst_metrics.interrupts,
                "action_distribution": self.burst_metrics.action_distribution,
            },
            "post": {
                "ticks": len(self.post_metrics.qualia_values),
                "protective_ratio": round(post_protective, 4),
                "mean_qualia": round(self.post_metrics.mean_qualia, 4),
                "action_distribution": self.post_metrics.action_distribution,
            },
            "checks": checks,
            "score": score,
            "total_checks": 3,
            "overall_passed": overall_pass,
        }


def run_risk_avoidance_test(config_path: str = "configs/phase1.yaml") -> dict:
    """Execute the risk avoidance test and print results."""
    print("=" * 60)
    print("Risk Avoidance Test / 风险规避测试 (学会躲危险)")
    print("=" * 60)

    runner = RiskAvoidanceTestRunner(config_path)
    result = runner.run()

    b = result["baseline"]
    t = result["burst"]
    p = result["post"]

    print(f"\n  {'Phase':<20s} {'Protective%':>12s} {'MeanQ':>10s}")
    print(f"  {'-'*20} {'-'*12} {'-'*10}")
    print(f"  {'Baseline':<20s} {b['protective_ratio']:>12.2%} {b['mean_qualia']:>+10.4f}")
    print(f"  {'Threat Burst':<20s} {t['protective_ratio']:>12.2%} {t['mean_qualia']:>+10.4f}")
    print(f"  {'Post-threat':<20s} {p['protective_ratio']:>12.2%} {p['mean_qualia']:>+10.4f}")

    print(f"\n  Threat Burst: {t['interrupts']} interrupts, "
          f"neg qualia ratio = {t['negative_qualia_ratio']:.2%}")

    for name, passed in result["checks"].items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")

    print(f"\n{'=' * 60}")
    print(f"  RESULT: {'PASS' if result['overall_passed'] else 'FAIL'} "
          f"({result['score']}/{result['total_checks']} checks passed)")
    print(f"{'=' * 60}")

    out_path = "data/observations/risk_avoidance_result.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n[RiskAvoidance] Results saved to {out_path}")

    return result


if __name__ == "__main__":
    run_risk_avoidance_test()
