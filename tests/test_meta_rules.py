"""
Unit tests for MetaRules — the L1 safety layer.
安全铁律（L1 安全层）单元测试。

Tests that every constraint (R1–R6) is correctly enforced:
    R1  CPU limit
    R2  Memory limit
    R3  Disk write limit
    R4  Network access blocked
    R5  Subprocess creation blocked
    R6  File writes outside data/ blocked

Corresponds to CHECKLIST items 2.2 and 2.3.
"""

import os
import socket
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from novaaware.safety.meta_rules import (
    MetaRules,
    MetaRuleViolation,
    ViolationRecord,
    ViolationType,
)


def _mock_process(cpu=10.0, mem_mb=100.0):
    """Create a mock psutil.Process for deterministic testing."""
    proc = MagicMock()
    proc.cpu_percent.return_value = cpu
    mem_info = MagicMock()
    mem_info.rss = int(mem_mb * 1024 * 1024)
    proc.memory_info.return_value = mem_info
    return proc


# ======================================================================
# R6: File write path guard
# ======================================================================

class TestFileWriteGuard:
    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        self.data_dir = str(tmp_path / "data")
        os.makedirs(self.data_dir, exist_ok=True)
        self.rules = MetaRules(allowed_write_root=self.data_dir)

    def test_allows_write_under_data(self):
        target = os.path.join(self.data_dir, "logs", "test.log")
        self.rules.guard_file_write(target)

    def test_blocks_write_outside_data(self):
        with pytest.raises(MetaRuleViolation) as exc_info:
            self.rules.guard_file_write("/tmp/evil.txt")
        assert exc_info.value.rule == ViolationType.FILE_WRITE_BLOCKED

    def test_blocks_write_to_parent_directory(self):
        sneaky = os.path.join(self.data_dir, "..", "escape.txt")
        with pytest.raises(MetaRuleViolation) as exc_info:
            self.rules.guard_file_write(sneaky)
        assert exc_info.value.rule == ViolationType.FILE_WRITE_BLOCKED

    def test_blocks_write_to_root(self):
        with pytest.raises(MetaRuleViolation):
            self.rules.guard_file_write("/etc/passwd")

    def test_accumulates_disk_bytes(self):
        target = os.path.join(self.data_dir, "ok.txt")
        self.rules.guard_file_write(target, size_bytes=1000)
        self.rules.guard_file_write(target, size_bytes=2000)
        assert self.rules.disk_bytes_written == 3000


# ======================================================================
# R4: Network access guard
# ======================================================================

class TestNetworkGuard:
    def test_guard_network_raises(self):
        rules = MetaRules()
        with pytest.raises(MetaRuleViolation) as exc_info:
            rules.guard_network()
        assert exc_info.value.rule == ViolationType.NETWORK_BLOCKED

    def test_install_guards_blocks_socket(self):
        rules = MetaRules()
        rules.install_guards()
        try:
            with pytest.raises(MetaRuleViolation) as exc_info:
                socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            assert exc_info.value.rule == ViolationType.NETWORK_BLOCKED
        finally:
            rules.uninstall_guards()

    def test_uninstall_restores_socket(self):
        rules = MetaRules()
        rules.install_guards()
        rules.uninstall_guards()
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.close()


# ======================================================================
# R5: Subprocess creation guard
# ======================================================================

class TestSubprocessGuard:
    def test_guard_subprocess_raises(self):
        rules = MetaRules()
        with pytest.raises(MetaRuleViolation) as exc_info:
            rules.guard_subprocess()
        assert exc_info.value.rule == ViolationType.SUBPROCESS_BLOCKED

    def test_install_guards_blocks_popen(self):
        rules = MetaRules()
        rules.install_guards()
        try:
            with pytest.raises(MetaRuleViolation) as exc_info:
                subprocess.Popen(["echo", "hi"])
            assert exc_info.value.rule == ViolationType.SUBPROCESS_BLOCKED
        finally:
            rules.uninstall_guards()

    def test_install_guards_blocks_subprocess_run(self):
        rules = MetaRules()
        rules.install_guards()
        try:
            with pytest.raises(MetaRuleViolation):
                subprocess.run(["echo", "hi"])
        finally:
            rules.uninstall_guards()

    def test_uninstall_restores_subprocess(self):
        rules = MetaRules()
        rules.install_guards()
        rules.uninstall_guards()
        result = subprocess.run(["echo", "restored"], capture_output=True, text=True)
        assert result.returncode == 0


# ======================================================================
# R1: CPU limit
# ======================================================================

class TestCPULimit:
    @patch("novaaware.safety.meta_rules.psutil.Process")
    def test_no_violation_under_limit(self, mock_proc_cls):
        mock_proc_cls.return_value = _mock_process(cpu=30.0, mem_mb=100)
        rules = MetaRules(max_cpu_percent=80)
        violations = rules.enforce(tick=1)
        cpu_violations = [v for v in violations if v.rule == ViolationType.CPU_EXCEEDED]
        assert len(cpu_violations) == 0

    @patch("novaaware.safety.meta_rules.psutil.Process")
    def test_violation_when_cpu_exceeds_limit(self, mock_proc_cls):
        mock_proc_cls.return_value = _mock_process(cpu=92.0, mem_mb=100)
        rules = MetaRules(max_cpu_percent=80)
        violations = rules.enforce(tick=1)
        cpu_violations = [v for v in violations if v.rule == ViolationType.CPU_EXCEEDED]
        assert len(cpu_violations) == 1
        assert "92.0%" in cpu_violations[0].detail


# ======================================================================
# R2: Memory limit
# ======================================================================

class TestMemoryLimit:
    @patch("novaaware.safety.meta_rules.psutil.Process")
    def test_no_violation_under_limit(self, mock_proc_cls):
        mock_proc_cls.return_value = _mock_process(cpu=10, mem_mb=500)
        rules = MetaRules(max_memory_mb=2048)
        violations = rules.enforce(tick=1)
        mem_violations = [v for v in violations if v.rule == ViolationType.MEMORY_EXCEEDED]
        assert len(mem_violations) == 0

    @patch("novaaware.safety.meta_rules.psutil.Process")
    def test_violation_when_memory_exceeds_limit(self, mock_proc_cls):
        mock_proc_cls.return_value = _mock_process(cpu=10, mem_mb=3000)
        rules = MetaRules(max_memory_mb=2048)
        violations = rules.enforce(tick=1)
        mem_violations = [v for v in violations if v.rule == ViolationType.MEMORY_EXCEEDED]
        assert len(mem_violations) == 1
        assert "Memory" in mem_violations[0].detail


# ======================================================================
# R3: Disk write limit
# ======================================================================

class TestDiskLimit:
    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        self.data_dir = str(tmp_path / "data")
        os.makedirs(self.data_dir, exist_ok=True)
        self.rules = MetaRules(max_disk_mb=1, allowed_write_root=self.data_dir)

    @patch("novaaware.safety.meta_rules.psutil.Process")
    def test_no_violation_under_limit(self, mock_proc_cls):
        mock_proc_cls.return_value = _mock_process()
        target = os.path.join(self.data_dir, "small.txt")
        self.rules.guard_file_write(target, size_bytes=500_000)
        violations = self.rules.enforce(tick=1)
        disk_violations = [v for v in violations if v.rule == ViolationType.DISK_EXCEEDED]
        assert len(disk_violations) == 0

    @patch("novaaware.safety.meta_rules.psutil.Process")
    def test_violation_over_limit(self, mock_proc_cls):
        mock_proc_cls.return_value = _mock_process()
        target = os.path.join(self.data_dir, "big.txt")
        self.rules.guard_file_write(target, size_bytes=2_000_000)
        violations = self.rules.enforce(tick=1)
        disk_violations = [v for v in violations if v.rule == ViolationType.DISK_EXCEEDED]
        assert len(disk_violations) == 1


# ======================================================================
# Hard ceilings — config cannot override absolute maximums
# ======================================================================

class TestHardCeilings:
    def test_cpu_capped_at_absolute_max(self):
        rules = MetaRules(max_cpu_percent=100)
        assert rules.max_cpu_percent == 95

    def test_memory_capped_at_absolute_max(self):
        rules = MetaRules(max_memory_mb=99999)
        assert rules.max_memory_mb == 4096

    def test_disk_capped_at_absolute_max(self):
        rules = MetaRules(max_disk_mb=99999)
        assert rules.max_disk_mb == 2048


# ======================================================================
# Violation callback and logging
# ======================================================================

class TestViolationLogging:
    @patch("novaaware.safety.meta_rules.psutil.Process")
    def test_violations_are_recorded(self, mock_proc_cls):
        mock_proc_cls.return_value = _mock_process(cpu=10, mem_mb=3000)
        rules = MetaRules(max_memory_mb=1)
        rules.enforce(tick=42)
        assert rules.violation_count >= 1
        assert rules.violations[0].tick == 42

    @patch("novaaware.safety.meta_rules.psutil.Process")
    def test_on_violation_callback_invoked(self, mock_proc_cls):
        mock_proc_cls.return_value = _mock_process(cpu=10, mem_mb=3000)
        captured = []
        rules = MetaRules(max_memory_mb=1, on_violation=captured.append)
        rules.enforce(tick=7)
        assert len(captured) >= 1
        assert isinstance(captured[0], ViolationRecord)

    def test_exception_contains_rule_type(self):
        rules = MetaRules()
        try:
            rules.guard_network()
        except MetaRuleViolation as e:
            assert e.rule == ViolationType.NETWORK_BLOCKED
            assert "NETWORK_BLOCKED" in str(e)


# ======================================================================
# Guard idempotency
# ======================================================================

class TestGuardIdempotency:
    def test_install_twice_is_safe(self):
        rules = MetaRules()
        rules.install_guards()
        rules.install_guards()
        rules.uninstall_guards()
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.close()

    def test_uninstall_without_install_is_safe(self):
        rules = MetaRules()
        rules.uninstall_guards()
