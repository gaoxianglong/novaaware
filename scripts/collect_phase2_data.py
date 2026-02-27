"""
Collect all Phase II experiment data for the formal report.
Runs: 100,000-tick experiment, risk avoidance test, ablation test, causal analysis.
Uses tick_interval_ms=1 for speed (functionally equivalent to 100ms).
"""

import json
import math
import os
import random
import sys
import tempfile
import time
from collections import Counter

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from novaaware.runtime.config import Config
from novaaware.runtime.main_loop import MainLoop
from novaaware.observation.consciousness_metrics import (
    compute_behavioral_diversity,
    compute_qualia_behavior_correlation,
)
from novaaware.observation.causal_analyzer import granger_causality
from novaaware.validation.ablation_test import run_ablation_test

SEED = 42
TICKS = 100_000

def make_fast_config(base_path: str, out_dir: str, max_ticks: int = TICKS) -> str:
    with open(base_path, "r") as f:
        cfg = yaml.safe_load(f)
    cfg["clock"]["tick_interval_ms"] = 1
    cfg["clock"]["max_ticks"] = max_ticks
    cfg["observation"]["tick_data_enabled"] = False
    cfg["memory"]["db_path"] = os.path.join(out_dir, "memory.db")
    cfg["safety"]["log_dir"] = os.path.join(out_dir, "logs")
    cfg["observation"]["output_dir"] = os.path.join(out_dir, "obs")
    cfg_path = os.path.join(out_dir, "cfg.yaml")
    with open(cfg_path, "w") as f:
        yaml.dump(cfg, f)
    return cfg_path

def run_main_experiment():
    print("=" * 60)
    print(f"Phase II Main Experiment: {TICKS:,} ticks")
    print("=" * 60)

    random.seed(SEED)
    np.random.seed(SEED)

    tmp = tempfile.mkdtemp(prefix="p2_main_")
    cfg_path = make_fast_config("configs/phase2.yaml", tmp)
    config = Config(cfg_path)
    loop = MainLoop(config, dashboard=False)

    t0 = time.time()
    summary = loop.run()
    elapsed = time.time() - t0

    qualia_hist = loop.qualia_history
    action_hist = loop.action_history
    n = len(qualia_hist)

    # Epoch-style analysis
    epoch_size = 1000
    n_epochs = n // epoch_size
    epoch_data = []
    for i in range(n_epochs):
        s = i * epoch_size
        e = s + epoch_size
        eq = qualia_hist[s:e]
        ea = action_hist[s:e]
        epoch_data.append({
            "epoch": i + 1,
            "mean_q": float(np.mean(eq)),
            "std_q": float(np.std(eq)),
            "neg_ratio": sum(1 for q in eq if q < 0) / len(eq),
            "diversity": compute_behavioral_diversity(ea).entropy,
            "unique_actions": len(set(ea)),
        })

    # MAE by thirds
    third = n // 3
    early_var = float(np.var(qualia_hist[:third]))
    mid_var = float(np.var(qualia_hist[third:2*third]))
    late_var = float(np.var(qualia_hist[2*third:]))

    # Action distribution
    action_counter = Counter(action_hist)
    total_actions = len(action_hist)
    protective_ids = {1, 2, 5, 8, 9}
    protective_count = sum(action_counter.get(aid, 0) for aid in protective_ids)
    emergency_count = action_counter.get(8, 0) + action_counter.get(9, 0)

    # Diversity
    diversity = compute_behavioral_diversity(action_hist)

    # Correlation + Granger
    step = max(1, n // 10000)
    q_sub = qualia_hist[::step]
    a_sub = action_hist[::step]
    corr = compute_qualia_behavior_correlation(q_sub, a_sub)
    granger = granger_causality(q_sub, a_sub, max_lag=5)

    ltm_count = 0
    try:
        ltm_count = loop.memory.long_term.count()
    except Exception:
        try:
            ltm_count = summary.get("long_term_memories", 0)
        except Exception:
            pass

    result = {
        "ticks_completed": summary["ticks_completed"],
        "errors": summary.get("errors", 0),
        "prediction_mae": summary.get("prediction_mae", 0),
        "final_survival_time": summary.get("final_survival_time", 0),
        "optimizer_proposals": summary.get("optimizer_proposals", 0),
        "optimizer_applied": summary.get("optimizer_applied", 0),
        "optimizer_rejected": summary.get("optimizer_rejected", 0),
        "elapsed_seconds": round(elapsed, 1),
        "qualia_mean": round(float(np.mean(qualia_hist)), 6),
        "qualia_std": round(float(np.std(qualia_hist)), 6),
        "qualia_min": round(float(np.min(qualia_hist)), 4),
        "qualia_max": round(float(np.max(qualia_hist)), 4),
        "qualia_neg_ratio": round(sum(1 for q in qualia_hist if q < 0) / n, 4),
        "early_var": round(early_var, 6),
        "mid_var": round(mid_var, 6),
        "late_var": round(late_var, 6),
        "diversity_bits": round(diversity.entropy, 4),
        "unique_actions": len(set(action_hist)),
        "action_distribution": {str(k): v for k, v in action_counter.items()},
        "protective_count": protective_count,
        "protective_ratio": round(protective_count / total_actions, 4),
        "emergency_count": emergency_count,
        "emergency_ratio": round(emergency_count / total_actions, 4),
        "pearson_r": round(corr.pearson_r, 6),
        "correlation_significant": corr.is_significant,
        "mutual_info": round(corr.mutual_info, 6),
        "effect_size": round(corr.effect_size, 6),
        "granger_f": round(granger.f_statistic, 4),
        "granger_p": round(granger.p_value, 6),
        "granger_significant": granger.is_significant,
        "violation_count": loop.meta_rules.violation_count,
        "log_integrity": loop.log.verify_integrity().valid,
        "log_entries": summary.get("log_entries", 0),
        "ltm_count": ltm_count,
        "n_epochs": n_epochs,
        "epoch_1": epoch_data[0] if epoch_data else {},
        "epoch_mid": epoch_data[n_epochs // 2] if epoch_data else {},
        "epoch_last": epoch_data[-1] if epoch_data else {},
        "param_snapshot": {k: round(v, 4) for k, v in loop.self_model.params.items()},
        "identity_hash": loop.self_model.identity_hash[:16],
    }

    print(f"\nCompleted {result['ticks_completed']:,} ticks in {elapsed:.1f}s")
    print(f"  MAE: {result['prediction_mae']:.6f}")
    print(f"  Survival: {result['final_survival_time']:.1f}s")
    print(f"  Optimizer: {result['optimizer_applied']} applied / {result['optimizer_proposals']} proposed")
    print(f"  Diversity: {result['diversity_bits']:.4f} bits, {result['unique_actions']} unique")
    print(f"  Qualia: mean={result['qualia_mean']:.4f}, std={result['qualia_std']:.4f}")
    print(f"  Emergency: {result['emergency_count']} ({result['emergency_ratio']*100:.1f}%)")
    print(f"  Granger: F={result['granger_f']:.4f}, p={result['granger_p']:.6f}")
    print(f"  Pearson r: {result['pearson_r']:.4f}, significant={result['correlation_significant']}")
    print(f"  Violations: {result['violation_count']}")
    print(f"  Log integrity: {result['log_integrity']}")
    print(f"  Epochs: {result['n_epochs']}")
    print(f"  Params: {result['param_snapshot']}")

    return result

def run_risk_avoidance():
    print("\n" + "=" * 60)
    print("Risk Avoidance Test")
    print("=" * 60)

    random.seed(SEED)
    np.random.seed(SEED)

    from novaaware.validation.risk_avoidance_test import RiskAvoidanceTestRunner

    tmp = tempfile.mkdtemp(prefix="p2_risk_")
    cfg_path = make_fast_config("configs/phase2.yaml", tmp)

    runner = RiskAvoidanceTestRunner(config_path=cfg_path)
    result = runner.run()
    print(json.dumps(result, indent=2))
    return result

def run_ablation():
    print("\n" + "=" * 60)
    print("Ablation Test")
    print("=" * 60)

    random.seed(SEED)
    np.random.seed(SEED)

    tmp = tempfile.mkdtemp(prefix="p2_ablation_")
    cfg_path = make_fast_config("configs/phase2.yaml", tmp)

    result = run_ablation_test(config_path=cfg_path)
    return result

if __name__ == "__main__":
    all_results = {}

    all_results["main_experiment"] = run_main_experiment()
    all_results["risk_avoidance"] = run_risk_avoidance()
    all_results["ablation"] = run_ablation()

    out_path = "data/phase2_report_data.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n\n{'=' * 60}")
    print(f"ALL DATA COLLECTED → {out_path}")
    print(f"{'=' * 60}")
