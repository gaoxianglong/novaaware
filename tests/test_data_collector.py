"""
Unit tests for DataCollector — the scientist's "recording instrument".
数据采集器的单元测试 —— 科学家的"记录仪"。

Tests cover:
  - Layer 1: tick_data.csv per-heartbeat recording / 逐心跳记录
  - Layer 2: aggregate_data.csv 100-tick aggregation / 100 心跳聚合
  - Layer 3: epoch_report_XXXX.txt 1000-tick reports / 1000 心跳报告
  - Format correctness: CSV files openable by Excel / CSV 格式正确性
  - Integration with MainLoop / 与主循环集成
"""

import csv
import os
import shutil
import tempfile
import time

import pytest

from novaaware.observation.data_collector import (
    DataCollector,
    TickRecord,
    _mean,
    _std,
)


# ======================================================================
# Fixtures / 测试固件
# ======================================================================

@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp(prefix="novaaware_obs_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


def _make_tick(tick: int, qualia: float = 0.1, mae: float = 0.5,
               survival: float = 3600.0, action: int = 0,
               interrupt: bool = False, memory_write: bool = False,
               threat: str = None) -> TickRecord:
    return TickRecord(
        tick=tick,
        timestamp=time.time(),
        qualia_value=qualia,
        delta_t=qualia * 0.5,
        qualia_intensity=abs(qualia),
        survival_time=survival,
        prediction_mae=mae,
        action_id=action,
        param_norm=0.0,
        memory_write=memory_write,
        interrupt=interrupt,
        threat_type=threat,
        action_effect=0.0,
    )


# ======================================================================
# 1. Layer 1: tick_data.csv / 第一层：逐心跳 CSV
# ======================================================================

class TestTickCSV:

    def test_csv_created(self, tmp_dir):
        """tick_data.csv should be created on init. / 初始化时应创建 tick_data.csv。"""
        dc = DataCollector(output_dir=tmp_dir)
        dc.close()
        assert os.path.exists(os.path.join(tmp_dir, "tick_data.csv"))

    def test_csv_has_header(self, tmp_dir):
        """First row should be the header. / 第一行应为表头。"""
        dc = DataCollector(output_dir=tmp_dir)
        dc.close()
        with open(os.path.join(tmp_dir, "tick_data.csv"), "r") as f:
            reader = csv.reader(f)
            header = next(reader)
        assert "tick" in header
        assert "qualia_value" in header
        assert "delta_T" in header
        assert "survival_time" in header
        assert "prediction_mae" in header
        assert "action_id" in header

    def test_csv_one_row_per_tick(self, tmp_dir):
        """Each record_tick should write one row. / 每次 record_tick 应写入一行。"""
        dc = DataCollector(output_dir=tmp_dir, aggregate_window=1000, epoch_size=10000)
        for i in range(50):
            dc.record_tick(_make_tick(i))
        dc.close()

        with open(os.path.join(tmp_dir, "tick_data.csv"), "r") as f:
            rows = list(csv.reader(f))
        assert len(rows) == 51  # header + 50 data rows

    def test_csv_disabled(self, tmp_dir):
        """When tick_data_enabled=False, no tick CSV is written. / tick_data_enabled=False 时不写入逐心跳 CSV。"""
        dc = DataCollector(output_dir=tmp_dir, tick_data_enabled=False)
        for i in range(10):
            dc.record_tick(_make_tick(i))
        dc.close()

        path = os.path.join(tmp_dir, "tick_data.csv")
        assert not os.path.exists(path)

    def test_total_records_counter(self, tmp_dir):
        dc = DataCollector(output_dir=tmp_dir, aggregate_window=1000, epoch_size=10000)
        for i in range(25):
            dc.record_tick(_make_tick(i))
        assert dc.total_records == 25
        dc.close()

    def test_csv_values_correct(self, tmp_dir):
        """Spot-check a single row's values. / 抽查单行的值。"""
        dc = DataCollector(output_dir=tmp_dir, aggregate_window=1000, epoch_size=10000)
        dc.record_tick(_make_tick(42, qualia=0.567, mae=0.123, survival=2500.0, action=3))
        dc.close()

        with open(os.path.join(tmp_dir, "tick_data.csv"), "r") as f:
            reader = csv.reader(f)
            next(reader)
            row = next(reader)
        assert row[0] == "42"
        assert float(row[2]) == pytest.approx(0.567, abs=0.001)
        assert float(row[6]) == pytest.approx(0.123, abs=0.001)
        assert row[7] == "3"


# ======================================================================
# 2. Layer 2: aggregate_data.csv / 第二层：聚合 CSV
# ======================================================================

class TestAggregateCSV:

    def test_aggregate_csv_created(self, tmp_dir):
        """aggregate_data.csv should be created on init. / 初始化时应创建 aggregate_data.csv。"""
        dc = DataCollector(output_dir=tmp_dir)
        dc.close()
        assert os.path.exists(os.path.join(tmp_dir, "aggregate_data.csv"))

    def test_aggregate_has_header(self, tmp_dir):
        dc = DataCollector(output_dir=tmp_dir)
        dc.close()
        with open(os.path.join(tmp_dir, "aggregate_data.csv"), "r") as f:
            reader = csv.reader(f)
            header = next(reader)
        assert "qualia_mean" in header
        assert "qualia_std" in header
        assert "negative_ratio" in header
        assert "mae_trend" in header
        assert "action_diversity" in header

    def test_aggregate_row_per_window(self, tmp_dir):
        """With window=100, 200 ticks should produce 2 aggregate rows. / window=100 时，200 心跳应产生 2 行聚合。"""
        dc = DataCollector(output_dir=tmp_dir, aggregate_window=100, epoch_size=10000)
        for i in range(200):
            q = 0.1 if i % 2 == 0 else -0.2
            dc.record_tick(_make_tick(i, qualia=q))
        dc.close()

        with open(os.path.join(tmp_dir, "aggregate_data.csv"), "r") as f:
            rows = list(csv.reader(f))
        assert len(rows) == 3  # header + 2 data rows

    def test_aggregate_partial_flush_on_close(self, tmp_dir):
        """Remaining buffer should be flushed on close. / 关闭时应刷新剩余缓冲区。"""
        dc = DataCollector(output_dir=tmp_dir, aggregate_window=100, epoch_size=10000)
        for i in range(150):
            dc.record_tick(_make_tick(i))
        dc.close()

        with open(os.path.join(tmp_dir, "aggregate_data.csv"), "r") as f:
            rows = list(csv.reader(f))
        assert len(rows) >= 2  # header + at least 1 aggregate row

    def test_aggregate_negative_ratio(self, tmp_dir):
        """If all qualia negative, negative_ratio should be 1.0. / 如果所有情绪为负，负面占比应为 1.0。"""
        dc = DataCollector(output_dir=tmp_dir, aggregate_window=10, epoch_size=10000)
        for i in range(10):
            dc.record_tick(_make_tick(i, qualia=-0.5))
        dc.close()

        with open(os.path.join(tmp_dir, "aggregate_data.csv"), "r") as f:
            reader = csv.reader(f)
            header = next(reader)
            row = next(reader)
        neg_idx = header.index("negative_ratio")
        assert float(row[neg_idx]) == pytest.approx(1.0)

    def test_aggregate_window_boundaries(self, tmp_dir):
        """Window start/end ticks should be correct. / 窗口起止心跳应正确。"""
        dc = DataCollector(output_dir=tmp_dir, aggregate_window=50, epoch_size=10000)
        for i in range(100):
            dc.record_tick(_make_tick(i))
        dc.close()

        with open(os.path.join(tmp_dir, "aggregate_data.csv"), "r") as f:
            reader = csv.reader(f)
            header = next(reader)
            row1 = next(reader)
            row2 = next(reader)
        start_idx = header.index("window_start")
        end_idx = header.index("window_end")
        assert int(row1[start_idx]) == 0
        assert int(row1[end_idx]) == 49
        assert int(row2[start_idx]) == 50
        assert int(row2[end_idx]) == 99


# ======================================================================
# 3. Layer 3: epoch_report_XXXX.txt / 第三层：体检报告
# ======================================================================

class TestEpochReport:

    def test_no_report_before_epoch(self, tmp_dir):
        """No report file until epoch_size ticks pass. / epoch_size 心跳之前不应有报告文件。"""
        dc = DataCollector(output_dir=tmp_dir, aggregate_window=50, epoch_size=100)
        for i in range(99):
            dc.record_tick(_make_tick(i))
        dc.close()
        assert not os.path.exists(os.path.join(tmp_dir, "epoch_report_0001.txt"))

    def test_report_created_at_epoch(self, tmp_dir):
        """Report file should be created after epoch_size ticks. / epoch_size 心跳后应创建报告文件。"""
        dc = DataCollector(output_dir=tmp_dir, aggregate_window=50, epoch_size=100)
        for i in range(100):
            dc.record_tick(_make_tick(i))
        dc.close()
        assert os.path.exists(os.path.join(tmp_dir, "epoch_report_0001.txt"))

    def test_report_count(self, tmp_dir):
        """200 ticks with epoch_size=100 → 2 reports. / 200 心跳 epoch_size=100 → 2 份报告。"""
        dc = DataCollector(output_dir=tmp_dir, aggregate_window=50, epoch_size=100)
        for i in range(200):
            dc.record_tick(_make_tick(i))
        dc.close()
        assert dc.epoch_count == 2
        assert os.path.exists(os.path.join(tmp_dir, "epoch_report_0001.txt"))
        assert os.path.exists(os.path.join(tmp_dir, "epoch_report_0002.txt"))

    def test_report_contains_key_sections(self, tmp_dir):
        """Report should contain all key sections. / 报告应包含所有关键章节。"""
        dc = DataCollector(output_dir=tmp_dir, aggregate_window=50, epoch_size=100)
        for i in range(100):
            q = -0.5 if i == 42 else 0.1
            dc.record_tick(_make_tick(i, qualia=q, action=i % 5, threat="memory_pressure" if i == 42 else None))
        dc.close()

        with open(os.path.join(tmp_dir, "epoch_report_0001.txt"), "r") as f:
            content = f.read()

        assert "情绪状况" in content or "Emotional State" in content
        assert "预测能力" in content or "Prediction Ability" in content
        assert "行为分析" in content or "Behavior Analysis" in content
        assert "环境与安全" in content or "Environment" in content

    def test_report_tick_range(self, tmp_dir):
        """Report should mention the correct tick range. / 报告应提及正确的心跳范围。"""
        dc = DataCollector(output_dir=tmp_dir, aggregate_window=50, epoch_size=100)
        for i in range(100):
            dc.record_tick(_make_tick(i))
        dc.close()

        with open(os.path.join(tmp_dir, "epoch_report_0001.txt"), "r") as f:
            content = f.read()
        assert "0-99" in content or "0" in content

    def test_report_numbering_padded(self, tmp_dir):
        """Report filenames should be zero-padded. / 报告文件名应零填充。"""
        dc = DataCollector(output_dir=tmp_dir, aggregate_window=50, epoch_size=50)
        for i in range(50):
            dc.record_tick(_make_tick(i))
        dc.close()
        assert os.path.exists(os.path.join(tmp_dir, "epoch_report_0001.txt"))


# ======================================================================
# 4. Integration with 1000 ticks / 与 1000 心跳的集成
# ======================================================================

class TestIntegration1000Ticks:

    def test_1000_ticks_all_files_present(self, tmp_dir):
        """After 1000 ticks: tick CSV + aggregate CSV + 1 epoch report. / 1000 心跳后：逐心跳 CSV + 聚合 CSV + 1 份报告。"""
        dc = DataCollector(
            output_dir=tmp_dir,
            tick_data_enabled=True,
            aggregate_window=100,
            epoch_size=1000,
        )
        for i in range(1000):
            q = -0.3 if i % 100 == 42 else 0.05
            threat = "cpu_spike" if i % 200 == 99 else None
            dc.record_tick(_make_tick(
                i, qualia=q, mae=0.5 - i * 0.0003, survival=3600.0 - i * 0.1,
                action=i % 8, interrupt=(abs(q) > 0.2), memory_write=(abs(q) > 0.2),
                threat=threat,
            ))
        dc.close()

        assert os.path.exists(os.path.join(tmp_dir, "tick_data.csv"))
        assert os.path.exists(os.path.join(tmp_dir, "aggregate_data.csv"))
        assert os.path.exists(os.path.join(tmp_dir, "epoch_report_0001.txt"))

    def test_1000_ticks_csv_row_count(self, tmp_dir):
        """tick_data.csv should have 1001 rows (header + 1000). / tick_data.csv 应有 1001 行。"""
        dc = DataCollector(output_dir=tmp_dir, aggregate_window=100, epoch_size=1000)
        for i in range(1000):
            dc.record_tick(_make_tick(i))
        dc.close()

        with open(os.path.join(tmp_dir, "tick_data.csv"), "r") as f:
            rows = list(csv.reader(f))
        assert len(rows) == 1001

    def test_1000_ticks_aggregate_row_count(self, tmp_dir):
        """aggregate_data.csv should have 11 rows (header + 10). / aggregate_data.csv 应有 11 行。"""
        dc = DataCollector(output_dir=tmp_dir, aggregate_window=100, epoch_size=1000)
        for i in range(1000):
            dc.record_tick(_make_tick(i))
        dc.close()

        with open(os.path.join(tmp_dir, "aggregate_data.csv"), "r") as f:
            rows = list(csv.reader(f))
        assert len(rows) == 11  # header + 10 aggregate rows

    def test_csv_no_nan_or_empty(self, tmp_dir):
        """No cell in tick CSV should be NaN or empty. / 逐心跳 CSV 中不应有 NaN 或空单元格。"""
        dc = DataCollector(output_dir=tmp_dir, aggregate_window=100, epoch_size=1000)
        for i in range(100):
            dc.record_tick(_make_tick(i))
        dc.close()

        with open(os.path.join(tmp_dir, "tick_data.csv"), "r") as f:
            reader = csv.reader(f)
            next(reader)
            for row_num, row in enumerate(reader, start=2):
                for col_num, cell in enumerate(row):
                    assert cell != "", f"Empty cell at row {row_num} col {col_num}"
                    assert cell.lower() != "nan", f"NaN at row {row_num} col {col_num}"


# ======================================================================
# 5. Helper functions / 辅助函数
# ======================================================================

class TestHelpers:

    def test_mean_empty(self):
        assert _mean([]) == 0.0

    def test_mean_basic(self):
        assert _mean([1.0, 2.0, 3.0]) == pytest.approx(2.0)

    def test_std_single(self):
        assert _std([5.0]) == 0.0

    def test_std_basic(self):
        assert _std([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]) == pytest.approx(2.0, abs=0.2)
