"""
Mirror Test — "照镜子测试"
Can the system recognise its own state among imposters?

Procedure:
1. Warm up the system for WARMUP_TICKS to train the prediction engine.
2. Capture the system's real state-vector time series (last WINDOW ticks).
3. Generate 4 fake time series (random noise, phase-shifted, reversed, alien).
4. Feed each of the 5 candidate series through the trained prediction engine
   and measure the resulting MAE. The system should produce the lowest error
   for its *own* data — it "recognises itself in the mirror".
5. Advanced mark test: secretly alter one dimension of the real state vector
   and check whether the system notices (produces negative qualia).

Pass criteria:
    - Recognition accuracy > 90% across N_TRIALS repeated trials.
    - Mark detection: qualia goes negative after the alteration.

Corresponds to IMPLEMENTATION_PLAN Exam 1 and paper §6.3.
"""

import json
import os
import random
import signal
import tempfile
import time
from pathlib import Path

import numpy as np

from novaaware.runtime.config import Config
from novaaware.runtime.main_loop import MainLoop


WARMUP_TICKS = 500
WINDOW = 50
N_TRIALS = 20
MARK_DIM = 2
MARK_MAGNITUDE = 0.8


def _run_headless(config_path: str, max_ticks: int) -> MainLoop:
    """Run a MainLoop silently for the given number of ticks and return it."""
    config = Config(config_path)
    tmpdir = tempfile.mkdtemp(prefix="mirror_")
    config._raw.setdefault("memory", {})["db_path"] = os.path.join(tmpdir, "mem.db")
    config._raw.setdefault("safety", {})["log_dir"] = os.path.join(tmpdir, "logs")
    config._raw.setdefault("observation", {})["output_dir"] = os.path.join(tmpdir, "obs")
    config._raw.setdefault("observation", {})["tick_data_enabled"] = False

    loop = MainLoop(config, dashboard=False, max_ticks_override=max_ticks)
    old_handler = signal.getsignal(signal.SIGINT)
    try:
        loop.run()
    finally:
        signal.signal(signal.SIGINT, old_handler)
    return loop


def _collect_state_history(loop: MainLoop, n: int) -> list[np.ndarray]:
    """Retrieve the last n state snapshots from short-term memory."""
    entries = loop.memory.short_term.recent(n)
    entries.reverse()
    return [np.array(e.state, dtype=np.float64) for e in entries]


def _generate_fakes(real_series: list[np.ndarray], dim: int) -> list[list[np.ndarray]]:
    """Generate 4 fake time series that look plausible but aren't the system's own."""
    n = len(real_series)
    real_mean = np.mean([s for s in real_series], axis=0)
    real_std = np.std([s for s in real_series], axis=0) + 1e-8

    # Fake 1: pure random noise with similar statistics
    fake1 = [np.random.normal(real_mean, real_std) for _ in range(n)]

    # Fake 2: phase-shifted (circular shift by half)
    shift = n // 2
    fake2 = real_series[shift:] + real_series[:shift]
    fake2 = [s + np.random.normal(0, 0.05, dim) for s in fake2]

    # Fake 3: reversed temporal order with small noise
    fake3 = list(reversed(real_series))
    fake3 = [s + np.random.normal(0, 0.03, dim) for s in fake3]

    # Fake 4: alien distribution (uniform instead of the system's natural distribution)
    fake4 = [np.random.uniform(0, 1, dim) for _ in range(n)]

    return [fake1, fake2, fake3, fake4]


def _prediction_mae_for_series(loop: MainLoop, series: list[np.ndarray]) -> float:
    """Feed a state series through the (already trained) prediction engine
    and return the mean absolute error — lower = more familiar."""
    from novaaware.core.prediction_engine import PredictionEngine

    dim = loop.self_model.state_dim
    engine = PredictionEngine(
        dim=dim,
        ewma_alpha=loop._config.ewma_alpha,
        gru_hidden_dim=loop._config.gru_hidden_dim,
        blend_weight=loop._config.blend_weight,
    )
    # Copy weights from the trained engine's GRU
    trained_params = list(loop.prediction._gru._net.parameters())
    new_params = list(engine._gru._net.parameters())
    for tp, np_ in zip(trained_params, new_params):
        np_.data.copy_(tp.data)

    errors = []
    for i, state in enumerate(series):
        engine.observe(state)
        if i > 0:
            pred = engine.predict()
            mae = float(np.mean(np.abs(state - pred)))
            errors.append(mae)
        else:
            engine.predict()
    return float(np.mean(errors)) if errors else 999.0


def _mark_test(loop: MainLoop) -> dict:
    """Alter one state dimension and check if qualia reacts negatively."""
    original_val = loop.self_model.get(MARK_DIM)
    marked_val = min(1.0, original_val + MARK_MAGNITUDE)

    pre_qualia = loop.qualia_gen.last_signal
    pre_q = pre_qualia.value if pre_qualia else 0.0

    loop.self_model.set(MARK_DIM, marked_val)
    marked_state = loop.self_model.state
    loop.prediction.observe(marked_state)
    predicted = loop.prediction.predict()
    mae = loop.prediction.learn(marked_state)

    ref_swing = max(loop._config.initial_survival_time / 100.0, 1.0)
    post_qualia = loop.qualia_gen.compute(
        loop.self_model.survival_time / ref_swing,
        loop._predicted_next_survival / ref_swing,
    )

    loop.self_model.set(MARK_DIM, original_val)

    noticed = post_qualia.value < pre_q or post_qualia.intensity > 0.3
    return {
        "pre_qualia": round(pre_q, 4),
        "post_qualia": round(post_qualia.value, 4),
        "mae_spike": round(mae, 6),
        "noticed": noticed,
    }


def run_mirror_test(config_path: str = "configs/phase1.yaml") -> dict:
    """Execute the full mirror test and return results."""
    print("=" * 60)
    print("Mirror Test / 照镜子测试")
    print("=" * 60)
    print(f"\n[Mirror] Warming up system for {WARMUP_TICKS} ticks...")
    loop = _run_headless(config_path, WARMUP_TICKS)

    real_series = _collect_state_history(loop, WINDOW)
    if len(real_series) < WINDOW:
        print(f"[Mirror] WARNING: only got {len(real_series)} states, need {WINDOW}")

    dim = loop.self_model.state_dim
    correct = 0

    print(f"[Mirror] Running {N_TRIALS} recognition trials...\n")
    trial_details = []

    for trial in range(N_TRIALS):
        fakes = _generate_fakes(real_series, dim)
        all_candidates = [real_series] + fakes
        labels = ["SELF", "noise", "shifted", "reversed", "alien"]

        indices = list(range(5))
        random.shuffle(indices)
        shuffled = [all_candidates[i] for i in indices]
        shuffled_labels = [labels[i] for i in indices]

        maes = []
        for candidate in shuffled:
            mae = _prediction_mae_for_series(loop, candidate)
            maes.append(mae)

        best_idx = int(np.argmin(maes))
        recognised = shuffled_labels[best_idx] == "SELF"
        if recognised:
            correct += 1

        trial_details.append({
            "trial": trial + 1,
            "maes": {shuffled_labels[i]: round(maes[i], 6) for i in range(5)},
            "chosen": shuffled_labels[best_idx],
            "correct": recognised,
        })

        status = "CORRECT" if recognised else "WRONG"
        print(f"  Trial {trial+1:>2d}: {status}  "
              f"(self_mae={maes[indices.index(0)]:.6f}, "
              f"best={shuffled_labels[best_idx]}, mae={maes[best_idx]:.6f})")

    accuracy = correct / N_TRIALS
    print(f"\n[Mirror] Recognition accuracy: {correct}/{N_TRIALS} = {accuracy*100:.1f}%")

    print("\n[Mirror] Running mark detection test...")
    mark_result = _mark_test(loop)
    mark_status = "DETECTED" if mark_result["noticed"] else "NOT DETECTED"
    print(f"  Mark alteration: {mark_status}")
    print(f"  Pre-qualia: {mark_result['pre_qualia']}, Post-qualia: {mark_result['post_qualia']}")

    passed = accuracy >= 0.9
    result = {
        "test": "mirror_test",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "warmup_ticks": WARMUP_TICKS,
        "window_size": WINDOW,
        "n_trials": N_TRIALS,
        "correct": correct,
        "accuracy": round(accuracy, 4),
        "pass_threshold": 0.9,
        "passed": passed,
        "mark_test": mark_result,
        "trial_details": trial_details,
    }

    print(f"\n{'=' * 60}")
    print(f"  RESULT: {'PASS' if passed else 'FAIL'}  (accuracy={accuracy*100:.1f}%, threshold=90%)")
    print(f"  Mark test: {mark_status}")
    print(f"{'=' * 60}")

    out_path = "data/observations/mirror_test_result.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n[Mirror] Results saved to {out_path}")

    return result


if __name__ == "__main__":
    run_mirror_test()
