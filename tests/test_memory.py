"""Unit tests for the Memory module. / 记忆系统单元测试。"""

import os
import tempfile
from typing import Optional
import pytest
from novaaware.core.memory import (
    MemoryEntry,
    ShortTermMemory,
    LongTermMemory,
    MemorySystem,
)


# ======================================================================
# Helpers / 辅助函数
# ======================================================================

def _make_entry(
    tick: int = 0,
    qualia_value: float = 0.0,
    qualia_intensity: float = 0.0,
    threat_type: Optional[str] = None,
    action_id: Optional[int] = None,
) -> MemoryEntry:
    """Create a MemoryEntry with sensible defaults for testing."""
    return MemoryEntry(
        tick=tick,
        timestamp=1000.0 + tick * 0.1,
        state=[float(tick)] * 32,
        environment=[0.5] * 6,
        predicted_state=[0.0] * 32,
        actual_state=[float(tick)] * 32,
        qualia_value=qualia_value,
        qualia_intensity=qualia_intensity,
        action_id=action_id,
        action_result=None,
        prediction_error=abs(qualia_value),
        threat_type=threat_type,
    )


# ======================================================================
# ShortTermMemory (ring buffer) / 短期记忆（环形缓冲区）
# ======================================================================

class TestShortTermMemory:
    def test_starts_empty(self):
        stm = ShortTermMemory(capacity=10)
        assert stm.size == 0
        assert stm.total_writes == 0

    def test_write_and_size(self):
        stm = ShortTermMemory(capacity=10)
        for i in range(5):
            stm.write(_make_entry(tick=i))
        assert stm.size == 5

    def test_capacity_limit(self):
        """After writing more than capacity, size stays at capacity."""
        stm = ShortTermMemory(capacity=3)
        for i in range(10):
            stm.write(_make_entry(tick=i))
        assert stm.size == 3
        assert stm.total_writes == 10

    def test_recent_returns_newest_first(self):
        stm = ShortTermMemory(capacity=100)
        for i in range(5):
            stm.write(_make_entry(tick=i))
        recent = stm.recent(3)
        assert len(recent) == 3
        assert recent[0].tick == 4  # newest
        assert recent[1].tick == 3
        assert recent[2].tick == 2

    def test_recent_more_than_available(self):
        stm = ShortTermMemory(capacity=100)
        stm.write(_make_entry(tick=0))
        stm.write(_make_entry(tick=1))
        recent = stm.recent(99)
        assert len(recent) == 2

    def test_recent_after_wrap(self):
        """Correct order even after the buffer has wrapped around."""
        stm = ShortTermMemory(capacity=3)
        for i in range(5):
            stm.write(_make_entry(tick=i))
        recent = stm.recent(3)
        assert [e.tick for e in recent] == [4, 3, 2]

    def test_get_all_chronological(self):
        stm = ShortTermMemory(capacity=100)
        for i in range(5):
            stm.write(_make_entry(tick=i))
        all_entries = stm.get_all()
        assert [e.tick for e in all_entries] == [0, 1, 2, 3, 4]

    def test_get_all_after_wrap(self):
        stm = ShortTermMemory(capacity=3)
        for i in range(5):
            stm.write(_make_entry(tick=i))
        all_entries = stm.get_all()
        assert [e.tick for e in all_entries] == [2, 3, 4]

    def test_empty_recent(self):
        stm = ShortTermMemory(capacity=10)
        assert stm.recent(5) == []


# ======================================================================
# LongTermMemory (SQLite) / 长期记忆（SQLite）
# ======================================================================

class TestLongTermMemory:
    @pytest.fixture(autouse=True)
    def _setup_db(self, tmp_path):
        """Use a temporary database for each test. / 每个测试使用临时数据库。"""
        self.db_path = str(tmp_path / "test_memory.db")
        self.ltm = LongTermMemory(db_path=self.db_path)
        yield
        self.ltm.close()

    def test_db_file_created(self):
        assert os.path.exists(self.db_path)

    def test_write_and_count(self):
        assert self.ltm.count() == 0
        self.ltm.write(_make_entry(tick=1, qualia_intensity=0.8))
        self.ltm.write(_make_entry(tick=2, qualia_intensity=1.2))
        assert self.ltm.count() == 2

    def test_write_returns_row_id(self):
        row_id = self.ltm.write(_make_entry(tick=1))
        assert isinstance(row_id, int)
        assert row_id >= 1

    def test_query_by_tick_range(self):
        for i in range(10):
            self.ltm.write(_make_entry(tick=i))
        results = self.ltm.query_by_tick_range(3, 7)
        assert [e.tick for e in results] == [3, 4, 5, 6]

    def test_query_by_intensity(self):
        self.ltm.write(_make_entry(tick=1, qualia_intensity=0.2))
        self.ltm.write(_make_entry(tick=2, qualia_intensity=0.8))
        self.ltm.write(_make_entry(tick=3, qualia_intensity=1.5))
        results = self.ltm.query_by_intensity(0.7)
        assert len(results) == 2
        assert results[0].tick == 2
        assert results[1].tick == 3

    def test_query_by_threat_type(self):
        self.ltm.write(_make_entry(tick=1, threat_type="cpu_spike"))
        self.ltm.write(_make_entry(tick=2, threat_type="memory_pressure"))
        self.ltm.write(_make_entry(tick=3, threat_type="cpu_spike"))
        self.ltm.write(_make_entry(tick=4, threat_type=None))
        results = self.ltm.query_by_threat_type("cpu_spike")
        assert len(results) == 2
        assert all(e.threat_type == "cpu_spike" for e in results)

    def test_recent(self):
        for i in range(5):
            self.ltm.write(_make_entry(tick=i))
        recent = self.ltm.recent(2)
        assert len(recent) == 2
        assert recent[0].tick == 4  # newest first
        assert recent[1].tick == 3

    def test_roundtrip_preserves_data(self):
        """Data survives write → read through JSON serialisation."""
        original = _make_entry(
            tick=42,
            qualia_value=-1.5,
            qualia_intensity=1.5,
            threat_type="termination_signal",
            action_id=3,
        )
        self.ltm.write(original)
        recovered = self.ltm.query_by_tick_range(42, 43)
        assert len(recovered) == 1
        r = recovered[0]
        assert r.tick == 42
        assert r.qualia_value == pytest.approx(-1.5)
        assert r.qualia_intensity == pytest.approx(1.5)
        assert r.threat_type == "termination_signal"
        assert r.action_id == 3
        assert len(r.state) == 32


# ======================================================================
# MemorySystem (unified facade + filtering) / 记忆系统（统一门面 + 筛选）
# ======================================================================

class TestMemorySystem:
    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        self.db_path = str(tmp_path / "sys_memory.db")
        self.ms = MemorySystem(
            short_term_capacity=5,
            significance_threshold=0.5,
            db_path=self.db_path,
        )
        yield
        self.ms.close()

    def test_always_writes_to_short_term(self):
        self.ms.record(_make_entry(tick=1, qualia_intensity=0.1))
        assert self.ms.short_term.size == 1

    def test_low_intensity_not_promoted(self):
        """Below threshold → short-term only, not long-term."""
        promoted = self.ms.record(_make_entry(tick=1, qualia_intensity=0.3))
        assert promoted is False
        assert self.ms.long_term.count() == 0

    def test_high_intensity_promoted(self):
        """At or above threshold → promoted to long-term."""
        promoted = self.ms.record(_make_entry(tick=1, qualia_intensity=0.5))
        assert promoted is True
        assert self.ms.long_term.count() == 1

    def test_above_threshold_promoted(self):
        promoted = self.ms.record(_make_entry(tick=1, qualia_intensity=1.8))
        assert promoted is True
        assert self.ms.long_term.count() == 1

    def test_mixed_batch(self):
        """Only significant events appear in long-term memory."""
        entries = [
            _make_entry(tick=1, qualia_intensity=0.1),  # not promoted
            _make_entry(tick=2, qualia_intensity=0.8),  # promoted
            _make_entry(tick=3, qualia_intensity=0.2),  # not promoted
            _make_entry(tick=4, qualia_intensity=1.5),  # promoted
            _make_entry(tick=5, qualia_intensity=0.4),  # not promoted
        ]
        for e in entries:
            self.ms.record(e)
        assert self.ms.short_term.size == 5
        assert self.ms.long_term.count() == 2

    def test_threshold_property(self):
        assert self.ms.significance_threshold == pytest.approx(0.5)

    def test_long_term_query_after_record(self):
        """Promoted entries can be queried from long-term memory."""
        self.ms.record(_make_entry(tick=10, qualia_intensity=0.9, threat_type="cpu_spike"))
        self.ms.record(_make_entry(tick=11, qualia_intensity=0.1))
        results = self.ms.long_term.query_by_threat_type("cpu_spike")
        assert len(results) == 1
        assert results[0].tick == 10
