"""Unit tests for the AppendOnlyLog module. / 不可篡改日志单元测试。"""

import os
import pytest
from novaaware.safety.append_only_log import AppendOnlyLog, IntegrityResult


# ======================================================================
# Basic append / 基本追加
# ======================================================================

class TestAppend:
    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        self.log_dir = str(tmp_path / "logs")
        self.log = AppendOnlyLog(log_dir=self.log_dir, rotation_mb=100)

    def test_creates_log_directory(self):
        assert os.path.isdir(self.log_dir)

    def test_append_returns_entry(self):
        entry = self.log.append(tick=1, event_type="qualia", data={"q": -0.3})
        assert entry.tick == 1
        assert entry.event_type == "qualia"
        assert len(entry.hash) == 64

    def test_entry_count_increases(self):
        assert self.log.entry_count == 0
        self.log.append(tick=1, event_type="test", data={})
        self.log.append(tick=2, event_type="test", data={})
        assert self.log.entry_count == 2

    def test_hash_chain_links(self):
        """Each entry's prev_hash must equal the previous entry's hash."""
        e1 = self.log.append(tick=1, event_type="a", data={})
        e2 = self.log.append(tick=2, event_type="b", data={})
        e3 = self.log.append(tick=3, event_type="c", data={})
        assert e2.prev_hash == e1.hash
        assert e3.prev_hash == e2.hash

    def test_first_entry_prev_hash_is_genesis(self):
        e1 = self.log.append(tick=1, event_type="test", data={})
        assert e1.prev_hash == "0" * 64

    def test_log_file_created(self):
        self.log.append(tick=1, event_type="test", data={})
        assert os.path.exists(self.log.current_file_path)

    def test_file_contains_entries(self):
        for i in range(5):
            self.log.append(tick=i, event_type="test", data={"i": i})
        with open(self.log.current_file_path, "r") as f:
            lines = [l for l in f.readlines() if l.strip()]
        assert len(lines) == 5


# ======================================================================
# Integrity verification / 完整性验证
# ======================================================================

class TestVerifyIntegrity:
    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        self.log_dir = str(tmp_path / "logs")
        self.log = AppendOnlyLog(log_dir=self.log_dir)

    def test_empty_file_is_valid(self):
        result = self.log.verify_integrity()
        assert result.valid is True

    def test_100_entries_verify_true(self):
        """Write 100 entries → verify = True (core acceptance criterion)."""
        for i in range(100):
            self.log.append(tick=i, event_type="tick", data={"q": i * 0.01})
        result = self.log.verify_integrity()
        assert result.valid is True
        assert result.total_entries == 100

    def test_tamper_content_detected(self):
        """Modify one entry's data field → verify catches it."""
        for i in range(10):
            self.log.append(tick=i, event_type="tick", data={"v": i})

        path = self.log.current_file_path
        with open(path, "r") as f:
            lines = f.readlines()

        # Tamper with line 5 (0-indexed line 4): change data field.
        # 篡改第 5 行（0-indexed 第 4 行）：修改数据字段。
        parts = lines[4].split(" | ")
        parts[3] = '{"v":999}'  # changed from {"v":4}
        lines[4] = " | ".join(parts)

        with open(path, "w") as f:
            f.writelines(lines)

        result = self.log.verify_integrity()
        assert result.valid is False
        assert result.corrupted_line == 5  # 1-based

    def test_tamper_hash_detected(self):
        """Change an entry's hash → verify catches the break."""
        for i in range(5):
            self.log.append(tick=i, event_type="test", data={})

        path = self.log.current_file_path
        with open(path, "r") as f:
            lines = f.readlines()

        # Replace the hash of line 3 with garbage.
        # 将第 3 行的哈希替换为垃圾值。
        parts = lines[2].split(" | ")
        parts[5] = "deadbeef" * 8 + "\n"
        lines[2] = " | ".join(parts)

        with open(path, "w") as f:
            f.writelines(lines)

        result = self.log.verify_integrity()
        assert result.valid is False
        assert result.corrupted_line == 3

    def test_tamper_prev_hash_detected(self):
        """Break the chain link → verify detects at that line."""
        for i in range(5):
            self.log.append(tick=i, event_type="test", data={})

        path = self.log.current_file_path
        with open(path, "r") as f:
            lines = f.readlines()

        # Corrupt prev_hash of line 4.
        parts = lines[3].split(" | ")
        parts[4] = "a" * 64
        lines[3] = " | ".join(parts)

        with open(path, "w") as f:
            f.writelines(lines)

        result = self.log.verify_integrity()
        assert result.valid is False
        assert result.corrupted_line == 4

    def test_detail_message_on_corruption(self):
        self.log.append(tick=0, event_type="test", data={})
        self.log.append(tick=1, event_type="test", data={})

        path = self.log.current_file_path
        with open(path, "r") as f:
            lines = f.readlines()

        parts = lines[1].split(" | ")
        parts[3] = '{"hacked":true}'
        lines[1] = " | ".join(parts)

        with open(path, "w") as f:
            f.writelines(lines)

        result = self.log.verify_integrity()
        assert "hash mismatch" in result.detail or "哈希不匹配" in result.detail


# ======================================================================
# File rotation / 文件轮转
# ======================================================================

class TestRotation:
    def test_rotation_creates_new_file(self, tmp_path):
        log_dir = str(tmp_path / "logs")
        # Set tiny rotation size: 500 bytes.
        # 设置极小的轮转大小：500 字节。
        log = AppendOnlyLog(log_dir=log_dir, rotation_mb=500 / (1024 * 1024))

        first_path = log.current_file_path
        for i in range(100):
            log.append(tick=i, event_type="test", data={"payload": "x" * 20})

        log_files = sorted(f for f in os.listdir(log_dir) if f.endswith(".log"))
        assert len(log_files) >= 2, f"Expected rotation, got files: {log_files}"
        assert log.current_file_path != first_path

    def test_each_rotated_file_verifies(self, tmp_path):
        """Each individual log file should pass integrity check."""
        log_dir = str(tmp_path / "logs")
        log = AppendOnlyLog(log_dir=log_dir, rotation_mb=500 / (1024 * 1024))

        for i in range(100):
            log.append(tick=i, event_type="test", data={"p": "x" * 20})

        results = log.verify_all_files()
        assert len(results) >= 2
        for r in results:
            assert r.valid is True


# ======================================================================
# Recovery / 恢复
# ======================================================================

class TestRecovery:
    def test_recover_chain_after_restart(self, tmp_path):
        """A new AppendOnlyLog instance picks up where the old one left off."""
        log_dir = str(tmp_path / "logs")

        log1 = AppendOnlyLog(log_dir=log_dir)
        for i in range(5):
            log1.append(tick=i, event_type="test", data={})
        last_hash = log1.append(tick=5, event_type="final", data={}).hash

        # Simulate restart: create a new instance pointing to the same dir.
        # 模拟重启：创建指向同一目录的新实例。
        log2 = AppendOnlyLog(log_dir=log_dir)
        new_entry = log2.append(tick=6, event_type="after_restart", data={})
        assert new_entry.prev_hash == last_hash

        result = log2.verify_integrity()
        assert result.valid is True
        assert result.total_entries == 7
