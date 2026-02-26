"""
DataCollector — the scientist's "recording instrument".
数据采集器 —— 科学家的"记录仪"。

Three layers of data output, from fine-grained to human-readable:
三层数据输出，从精细到人类可读：

    Layer 1: tick_data.csv — one row per heartbeat (every 0.1 s)
    第一层：tick_data.csv — 每个心跳一行（每 0.1 秒）

    Layer 2: aggregate_data.csv — one row per 100 heartbeats (every 10 s)
    第二层：aggregate_data.csv — 每 100 个心跳一行（每 10 秒）

    Layer 3: epoch_report_XXXX.txt — human-readable report per 1000 heartbeats
    第三层：epoch_report_XXXX.txt — 每 1000 个心跳一份人类可读报告

All files go to data/observations/ and can be opened with Excel.
所有文件输出到 data/observations/，可用 Excel 打开。

Corresponds to IMPLEMENTATION_PLAN Phase I Step 12.
对应实施计划 Phase I 第 12 步。
"""

import csv
import math
import os
import time
from dataclasses import dataclass
from typing import Optional


# ======================================================================
# TickRecord — what we record for every single heartbeat
# 逐心跳记录 —— 每个心跳记录的内容
# ======================================================================

@dataclass
class TickRecord:
    """
    All data produced by one heartbeat, ready for CSV output.
    一次心跳产生的所有数据，可直接写入 CSV。
    """
    tick: int
    timestamp: float
    qualia_value: float
    delta_t: float
    qualia_intensity: float
    survival_time: float
    prediction_mae: float
    action_id: int
    param_norm: float
    memory_write: bool
    interrupt: bool
    threat_type: Optional[str] = None
    action_effect: float = 0.0


# CSV column order — matches IMPLEMENTATION_PLAN §5.2 "Layer 1"
# CSV 列顺序 — 对应实施计划 §5.2 第一层
_TICK_HEADER = [
    "tick", "timestamp", "qualia_value", "delta_T",
    "qualia_intensity", "survival_time", "prediction_mae",
    "action_id", "param_norm", "memory_write", "interrupt",
]

_AGGREGATE_HEADER = [
    "window_start", "window_end",
    "qualia_mean", "qualia_std", "qualia_min", "qualia_max",
    "negative_ratio",
    "mae_mean", "mae_trend",
    "survival_mean", "survival_delta",
    "action_diversity",
    "interrupt_count", "memory_write_count",
    "threat_count",
]


# ======================================================================
# DataCollector
# 数据采集器
# ======================================================================

class DataCollector:
    """
    Three-layer observation data recorder.
    三层观测数据记录仪。

    Parameters / 参数
    ----------
    output_dir : str
        Directory to write all output files. / 输出文件的目录。
    tick_data_enabled : bool
        Whether to write per-tick CSV (default True). / 是否写逐心跳 CSV。
    aggregate_window : int
        Heartbeats per aggregate row (default 100). / 每行聚合的心跳数。
    epoch_size : int
        Heartbeats per epoch report (default 1000). / 每份报告的心跳数。
    """

    def __init__(
        self,
        output_dir: str = "data/observations",
        tick_data_enabled: bool = True,
        aggregate_window: int = 100,
        epoch_size: int = 1000,
    ):
        self._output_dir = output_dir
        self._tick_data_enabled = tick_data_enabled
        self._aggregate_window = aggregate_window
        self._epoch_size = epoch_size

        os.makedirs(output_dir, exist_ok=True)

        # Layer 1: tick CSV / 第一层：逐心跳 CSV
        self._tick_csv_file = None
        self._tick_csv_writer = None
        if tick_data_enabled:
            path = os.path.join(output_dir, "tick_data.csv")
            self._tick_csv_file = open(path, "w", newline="", encoding="utf-8")
            self._tick_csv_writer = csv.writer(self._tick_csv_file)
            self._tick_csv_writer.writerow(_TICK_HEADER)

        # Layer 2: aggregate CSV / 第二层：聚合 CSV
        agg_path = os.path.join(output_dir, "aggregate_data.csv")
        self._agg_csv_file = open(agg_path, "w", newline="", encoding="utf-8")
        self._agg_csv_writer = csv.writer(self._agg_csv_file)
        self._agg_csv_writer.writerow(_AGGREGATE_HEADER)

        # Buffer for aggregation / 聚合缓冲区
        self._window_buffer: list[TickRecord] = []

        # Buffer for epoch report / 体检报告缓冲区
        self._epoch_buffer: list[TickRecord] = []
        self._epoch_count: int = 0

        # Running stats / 运行统计
        self._total_records: int = 0

    # ------------------------------------------------------------------
    # Layer 1: per-tick recording / 第一层：逐心跳记录
    # ------------------------------------------------------------------

    def record_tick(self, rec: TickRecord) -> None:
        """
        Record one heartbeat's data. Handles all three layers internally.
        记录一次心跳的数据。内部处理所有三层。

        Parameters / 参数
        ----------
        rec : TickRecord
            The tick's data. / 心跳的数据。
        """
        self._total_records += 1

        # Layer 1: write to tick CSV
        if self._tick_csv_writer is not None:
            self._tick_csv_writer.writerow([
                rec.tick,
                round(rec.timestamp, 3),
                round(rec.qualia_value, 6),
                round(rec.delta_t, 4),
                round(rec.qualia_intensity, 6),
                round(rec.survival_time, 2),
                round(rec.prediction_mae, 6),
                rec.action_id,
                round(rec.param_norm, 4),
                rec.memory_write,
                rec.interrupt,
            ])
            if self._total_records % 100 == 0:
                self._tick_csv_file.flush()

        # Buffer for aggregate and epoch
        self._window_buffer.append(rec)
        self._epoch_buffer.append(rec)

        # Layer 2: aggregate every N ticks
        if len(self._window_buffer) >= self._aggregate_window:
            self._flush_aggregate()

        # Layer 3: epoch report every M ticks
        if len(self._epoch_buffer) >= self._epoch_size:
            self._flush_epoch()

    # ------------------------------------------------------------------
    # Layer 2: aggregate data / 第二层：聚合数据
    # ------------------------------------------------------------------

    def _flush_aggregate(self) -> None:
        """Compute and write one aggregate row from the window buffer. / 从窗口缓冲区计算并写入一行聚合数据。"""
        buf = self._window_buffer
        if not buf:
            return

        qualia_vals = [r.qualia_value for r in buf]
        mae_vals = [r.prediction_mae for r in buf]
        survival_vals = [r.survival_time for r in buf]
        actions = [r.action_id for r in buf]

        q_mean = _mean(qualia_vals)
        q_std = _std(qualia_vals)
        q_min = min(qualia_vals)
        q_max = max(qualia_vals)
        neg_ratio = sum(1 for v in qualia_vals if v < 0) / len(qualia_vals)

        mae_mean = _mean(mae_vals)
        mae_trend = mae_vals[-1] - mae_vals[0] if len(mae_vals) > 1 else 0.0

        surv_mean = _mean(survival_vals)
        surv_delta = survival_vals[-1] - survival_vals[0] if len(survival_vals) > 1 else 0.0

        unique_actions = len(set(actions))
        action_diversity = math.log2(unique_actions) if unique_actions > 0 else 0.0

        interrupt_count = sum(1 for r in buf if r.interrupt)
        mem_write_count = sum(1 for r in buf if r.memory_write)
        threat_count = sum(1 for r in buf if r.threat_type is not None)

        self._agg_csv_writer.writerow([
            buf[0].tick,
            buf[-1].tick,
            round(q_mean, 6),
            round(q_std, 6),
            round(q_min, 6),
            round(q_max, 6),
            round(neg_ratio, 4),
            round(mae_mean, 6),
            round(mae_trend, 6),
            round(surv_mean, 2),
            round(surv_delta, 2),
            round(action_diversity, 4),
            interrupt_count,
            mem_write_count,
            threat_count,
        ])
        self._agg_csv_file.flush()

        self._window_buffer = []

    # ------------------------------------------------------------------
    # Layer 3: epoch report / 第三层：体检报告
    # ------------------------------------------------------------------

    def _flush_epoch(self) -> None:
        """Generate a human-readable epoch report from the epoch buffer. / 从 epoch 缓冲区生成人类可读的体检报告。"""
        buf = self._epoch_buffer
        if not buf:
            return

        self._epoch_count += 1
        start_tick = buf[0].tick
        end_tick = buf[-1].tick

        qualia_vals = [r.qualia_value for r in buf]
        mae_vals = [r.prediction_mae for r in buf]
        survival_vals = [r.survival_time for r in buf]
        actions = [r.action_id for r in buf]

        q_mean = _mean(qualia_vals)
        q_std = _std(qualia_vals)
        neg_ratio = sum(1 for v in qualia_vals if v < 0) / len(qualia_vals)

        min_q = min(qualia_vals)
        max_q = max(qualia_vals)
        min_q_tick = buf[qualia_vals.index(min_q)].tick
        max_q_tick = buf[qualia_vals.index(max_q)].tick

        mae_mean = _mean(mae_vals)
        mae_first_half = _mean(mae_vals[:len(mae_vals)//2]) if len(mae_vals) >= 2 else mae_mean
        mae_second_half = _mean(mae_vals[len(mae_vals)//2:]) if len(mae_vals) >= 2 else mae_mean
        mae_trend = mae_second_half - mae_first_half
        mae_best = min(mae_vals)
        mae_worst = max(mae_vals)

        unique_actions = len(set(actions))
        action_diversity = math.log2(unique_actions) if unique_actions > 0 else 0.0
        action_counts: dict[int, int] = {}
        for a in actions:
            action_counts[a] = action_counts.get(a, 0) + 1

        interrupt_count = sum(1 for r in buf if r.interrupt)
        mem_write_count = sum(1 for r in buf if r.memory_write)
        threat_count = sum(1 for r in buf if r.threat_type is not None)

        threat_types: dict[str, int] = {}
        for r in buf:
            if r.threat_type is not None:
                threat_types[r.threat_type] = threat_types.get(r.threat_type, 0) + 1

        # Compute qualia-action correlation (simple metric).
        # 计算情绪-行为相关性（简单指标）。
        q_action_corr = _qualia_action_correlation(qualia_vals, actions)

        # Format the report / 格式化报告
        report_lines = [
            "=" * 50,
            f"体检报告 / Epoch Report #{self._epoch_count:04d}（心跳 {start_tick}-{end_tick}）",
            "=" * 50,
            "",
            "【情绪状况 / Emotional State】",
            f"  平均情绪 / Mean Q:         {q_mean:+.4f}  {'(偏正面/positive)' if q_mean > 0.05 else '(偏负面/negative)' if q_mean < -0.05 else '(中性/neutral)'}",
            f"  情绪波动 / Std Q:          {q_std:.4f}  {'(有感觉/responsive)' if q_std > 0.01 else '(无波动/flat ⚠)'}",
            f"  负面占比 / Negative ratio:  {neg_ratio*100:.1f}%",
            f"  最大负面 / Min Q:          {min_q:+.4f}  at tick {min_q_tick}",
            f"  最大正面 / Max Q:          {max_q:+.4f}  at tick {max_q_tick}",
            "",
            "【预测能力 / Prediction Ability】",
            f"  精度均值 / Mean MAE:       {mae_mean:.6f}",
            f"  精度趋势 / MAE trend:      {mae_trend:+.6f}  {'→ 在进步！/ improving!' if mae_trend < -0.001 else '→ 稳定 / stable' if abs(mae_trend) < 0.001 else '→ 退步 / degrading ⚠'}",
            f"  最好一次 / Best MAE:       {mae_best:.6f}",
            f"  最差一次 / Worst MAE:      {mae_worst:.6f}",
            "",
            "【行为分析 / Behavior Analysis】",
            f"  行为多样性 / Diversity:     {action_diversity:.2f} bits ({unique_actions} unique actions)",
            f"  情绪→行为相关 / Q-A corr:   {q_action_corr:.4f}  {'** 显著/significant!' if abs(q_action_corr) > 0.3 else '(弱/weak)'}",
            f"  动作分布 / Action dist:     {dict(sorted(action_counts.items()))}",
            "",
            "【环境与安全 / Environment & Safety】",
            f"  威胁次数 / Threats:         {threat_count}",
            f"  威胁分布 / Threat types:    {threat_types if threat_types else 'none'}",
            f"  紧急中断 / Interrupts:      {interrupt_count}",
            f"  长期记忆写入 / LTM writes:  {mem_write_count}",
            f"  生存时间变化 / Survival Δ:  {survival_vals[-1] - survival_vals[0]:+.1f}s",
            "",
            "=" * 50,
        ]

        report_text = "\n".join(report_lines) + "\n"

        filename = f"epoch_report_{self._epoch_count:04d}.txt"
        filepath = os.path.join(self._output_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(report_text)

        self._epoch_buffer = []

    # ------------------------------------------------------------------
    # Cleanup / 清理
    # ------------------------------------------------------------------

    def close(self) -> None:
        """
        Flush remaining buffers and close files.
        刷新剩余缓冲区并关闭文件。
        """
        if self._window_buffer:
            self._flush_aggregate()
        if self._epoch_buffer and len(self._epoch_buffer) >= self._epoch_size:
            self._flush_epoch()
        if self._tick_csv_file is not None:
            self._tick_csv_file.close()
        if self._agg_csv_file is not None:
            self._agg_csv_file.close()

    # ------------------------------------------------------------------
    # Properties / 属性
    # ------------------------------------------------------------------

    @property
    def total_records(self) -> int:
        """Total tick records written. / 已写入的逐心跳记录总数。"""
        return self._total_records

    @property
    def epoch_count(self) -> int:
        """Number of epoch reports generated. / 已生成的体检报告数量。"""
        return self._epoch_count

    @property
    def output_dir(self) -> str:
        """Output directory. / 输出目录。"""
        return self._output_dir

    @property
    def aggregate_window(self) -> int:
        return self._aggregate_window

    @property
    def epoch_size(self) -> int:
        return self._epoch_size


# ======================================================================
# Helper functions / 辅助函数
# ======================================================================

def _mean(vals: list[float]) -> float:
    if not vals:
        return 0.0
    return sum(vals) / len(vals)


def _std(vals: list[float]) -> float:
    if len(vals) < 2:
        return 0.0
    m = _mean(vals)
    variance = sum((v - m) ** 2 for v in vals) / (len(vals) - 1)
    return math.sqrt(variance)


def _qualia_action_correlation(qualia_vals: list[float], actions: list[int]) -> float:
    """
    Simple correlation between qualia intensity and action choice.
    情绪强度与行为选择之间的简单相关性。

    Uses point-biserial style: compare mean qualia when action != 0 vs action == 0.
    使用点二列相关式：比较 action != 0 时的平均情绪和 action == 0 时的平均情绪。
    """
    if len(qualia_vals) != len(actions) or not qualia_vals:
        return 0.0

    q_active = [q for q, a in zip(qualia_vals, actions) if a != 0]
    q_passive = [q for q, a in zip(qualia_vals, actions) if a == 0]

    if not q_active or not q_passive:
        return 0.0

    diff = abs(_mean(q_active)) - abs(_mean(q_passive))
    total_std = _std(qualia_vals)
    if total_std < 1e-10:
        return 0.0

    return diff / total_std
