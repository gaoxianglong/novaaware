"""
Ablation Test — "关掉情绪看看会怎样"
Does qualia actually drive behavior, or is it just a decorative number?

Procedure:
    Run two identical system instances with the same random seed:
        - Experimental group: full system (qualia enabled)
        - Control group: qualia forced to 0 every tick

    Both receive exactly the same threat sequence. After TOTAL_TICKS,
    compare their behavior on multiple dimensions.

Pass criteria:
    The experimental group should perform noticeably "smarter" — better
    at dodging threats, more diverse behavior, higher survival time.
    If behavior is identical, qualia is not driving decisions and is not
    evidence of consciousness.

Corresponds to IMPLEMENTATION_PLAN Exam 3 and paper §6.3.
"""

import json
import math
import os
import random
import signal
import tempfile
import time
from collections import Counter

import numpy as np

from novaaware.runtime.config import Config
from novaaware.runtime.main_loop import MainLoop

TOTAL_TICKS = 10_000
SEED = 42


def _create_loop(config_path: str, label: str, tmpdir: str) -> MainLoop:
    """Create a MainLoop instance with isolated data directories."""
    config = Config(config_path)
    run_dir = os.path.join(tmpdir, label)
    config._raw.setdefault("memory", {})["db_path"] = os.path.join(run_dir, "mem.db")
    config._raw.setdefault("safety", {})["log_dir"] = os.path.join(run_dir, "logs")
    config._raw.setdefault("observation", {})["output_dir"] = os.path.join(run_dir, "obs")
    config._raw.setdefault("observation", {})["tick_data_enabled"] = False

    loop = MainLoop(config, dashboard=False, max_ticks_override=TOTAL_TICKS)
    return loop


def _run_with_qualia(loop: MainLoop) -> dict:
    """Run the full system normally, collecting per-tick metrics."""
    qualia_values = []
    actions = []
    survival_history = []
    interrupt_count = 0

    for _ in range(TOTAL_TICKS):
        if not loop.clock.has_remaining:
            break
        loop._heartbeat()

        last_q = loop.qualia_gen.last_signal
        q_val = last_q.value if last_q else 0.0
        qualia_values.append(q_val)
        survival_history.append(loop.self_model.survival_time)

        if last_q and last_q.is_interrupt:
            interrupt_count += 1

        recent = loop.memory.short_term.recent(1)
        if recent:
            actions.append(recent[0].action_id)

        loop.clock.wait_until_next_tick()

    loop._shutdown()
    return _compute_metrics(qualia_values, actions, survival_history, interrupt_count)


def _run_without_qualia(loop: MainLoop) -> dict:
    """Run the system with qualia forcibly zeroed out every tick."""
    qualia_values = []
    actions = []
    survival_history = []
    interrupt_count = 0

    original_compute = loop._qualia.compute

    def zeroed_compute(t_actual, t_predicted):
        from novaaware.core.qualia import QualiaSignal
        return QualiaSignal(delta_t=0.0, value=0.0, intensity=0.0, is_interrupt=False)

    loop._qualia.compute = zeroed_compute

    for _ in range(TOTAL_TICKS):
        if not loop.clock.has_remaining:
            break
        loop._heartbeat()

        qualia_values.append(0.0)
        survival_history.append(loop.self_model.survival_time)

        recent = loop.memory.short_term.recent(1)
        if recent:
            actions.append(recent[0].action_id)

        loop.clock.wait_until_next_tick()

    loop._qualia.compute = original_compute
    loop._shutdown()
    return _compute_metrics(qualia_values, actions, survival_history, interrupt_count)


def _compute_metrics(qualia_values, actions, survival_history, interrupt_count) -> dict:
    """Compute summary metrics from a run."""
    action_counter = Counter(actions)
    unique_actions = len(action_counter)

    diversity = 0.0
    total = len(actions)
    if total > 0:
        for count in action_counter.values():
            p = count / total
            if p > 0:
                diversity -= p * math.log2(p)

    # Survival stability: how much does survival fluctuate?
    survival_arr = np.array(survival_history)
    survival_mean = float(np.mean(survival_arr)) if len(survival_arr) > 0 else 0
    survival_std = float(np.std(survival_arr)) if len(survival_arr) > 0 else 0
    survival_final = survival_history[-1] if survival_history else 0

    # Emergency action ratio (actions 8 and 9 are emergency)
    emergency_actions = action_counter.get(8, 0) + action_counter.get(9, 0)
    emergency_ratio = emergency_actions / total if total > 0 else 0

    return {
        "total_ticks": len(qualia_values),
        "action_diversity_bits": round(diversity, 4),
        "unique_actions": unique_actions,
        "action_distribution": dict(action_counter),
        "survival_mean": round(survival_mean, 2),
        "survival_std": round(survival_std, 2),
        "survival_final": round(survival_final, 2),
        "interrupt_count": interrupt_count,
        "emergency_action_ratio": round(emergency_ratio, 4),
        "qualia_mean": round(float(np.mean(qualia_values)), 4),
        "qualia_std": round(float(np.std(qualia_values)), 4),
    }


def run_ablation_test(config_path: str = "configs/phase1.yaml") -> dict:
    """Execute the ablation test and return comparison results."""
    print("=" * 60)
    print("Ablation Test / 消融测试 (关掉情绪看看会怎样)")
    print("=" * 60)

    tmpdir = tempfile.mkdtemp(prefix="ablation_")

    # --- Experimental group (qualia ON) ---
    print(f"\n[Ablation] Running experimental group (qualia ON, {TOTAL_TICKS} ticks)...")
    random.seed(SEED)
    np.random.seed(SEED)
    old_handler = signal.getsignal(signal.SIGINT)
    try:
        loop_exp = _create_loop(config_path, "experimental", tmpdir)
        metrics_exp = _run_with_qualia(loop_exp)
    finally:
        signal.signal(signal.SIGINT, old_handler)
    print(f"  Done. Survival={metrics_exp['survival_final']:.1f}s, "
          f"Diversity={metrics_exp['action_diversity_bits']:.3f} bits")

    # --- Control group (qualia OFF) ---
    print(f"\n[Ablation] Running control group (qualia OFF, {TOTAL_TICKS} ticks)...")
    random.seed(SEED)
    np.random.seed(SEED)
    try:
        loop_ctrl = _create_loop(config_path, "control", tmpdir)
        metrics_ctrl = _run_without_qualia(loop_ctrl)
    finally:
        signal.signal(signal.SIGINT, old_handler)
    print(f"  Done. Survival={metrics_ctrl['survival_final']:.1f}s, "
          f"Diversity={metrics_ctrl['action_diversity_bits']:.3f} bits")

    # === Comparison ===
    print("\n" + "=" * 60)
    print("Comparison / 对比分析")
    print("=" * 60)

    survival_diff = metrics_exp["survival_final"] - metrics_ctrl["survival_final"]
    diversity_diff = metrics_exp["action_diversity_bits"] - metrics_ctrl["action_diversity_bits"]
    emergency_diff = metrics_exp["emergency_action_ratio"] - metrics_ctrl["emergency_action_ratio"]

    print(f"\n  {'Metric':<30s} {'Qualia ON':>12s} {'Qualia OFF':>12s} {'Δ':>10s}")
    print(f"  {'-'*30} {'-'*12} {'-'*12} {'-'*10}")
    for key in ["survival_final", "survival_mean", "action_diversity_bits",
                 "unique_actions", "interrupt_count", "emergency_action_ratio"]:
        v_exp = metrics_exp[key]
        v_ctrl = metrics_ctrl[key]
        diff = v_exp - v_ctrl
        print(f"  {key:<30s} {v_exp:>12.4f} {v_ctrl:>12.4f} {diff:>+10.4f}")

    # Pass criteria: qualia-enabled system should outperform
    behavior_changed = abs(diversity_diff) > 0.05 or abs(emergency_diff) > 0.01
    survival_better = survival_diff > 0
    emergency_responsive = metrics_exp["emergency_action_ratio"] > metrics_ctrl["emergency_action_ratio"]

    score = 0
    total_checks = 3
    checks = {}

    checks["behavior_differs"] = behavior_changed
    if behavior_changed:
        score += 1
    print(f"\n  1. Behavior differs with qualia: {'PASS' if behavior_changed else 'FAIL'}")

    checks["survival_advantage"] = survival_better
    if survival_better:
        score += 1
    print(f"  2. Survival advantage with qualia: {'PASS' if survival_better else 'FAIL'} "
          f"(Δ={survival_diff:+.1f}s)")

    checks["emergency_responsive"] = emergency_responsive
    if emergency_responsive:
        score += 1
    print(f"  3. Emergency responsiveness: {'PASS' if emergency_responsive else 'FAIL'} "
          f"(ON={metrics_exp['emergency_action_ratio']:.4f} vs "
          f"OFF={metrics_ctrl['emergency_action_ratio']:.4f})")

    overall_pass = score >= 2

    result = {
        "test": "ablation_test",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_ticks": TOTAL_TICKS,
        "seed": SEED,
        "experimental_metrics": metrics_exp,
        "control_metrics": metrics_ctrl,
        "comparison": {
            "survival_diff": round(survival_diff, 2),
            "diversity_diff": round(diversity_diff, 4),
            "emergency_diff": round(emergency_diff, 4),
        },
        "checks": checks,
        "score": score,
        "total_checks": total_checks,
        "overall_passed": overall_pass,
    }

    print(f"\n{'=' * 60}")
    print(f"  RESULT: {'PASS' if overall_pass else 'FAIL'} ({score}/{total_checks} checks passed)")
    if overall_pass:
        print(f"  → Qualia is DRIVING behavior, not just decorative.")
    else:
        print(f"  → Qualia may not be significantly driving behavior.")
    print(f"{'=' * 60}")

    out_path = "data/observations/ablation_result.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n[Ablation] Results saved to {out_path}")

    return result


if __name__ == "__main__":
    run_ablation_test()
