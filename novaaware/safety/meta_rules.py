"""
MetaRules — immutable safety constraints outside the optimizer's reach.
安全铁律 —— 优化器无法触及的不可变安全约束。

These are the L1 safety layer from the Ouroboros architecture: hard-coded
invariants that the system can NEVER violate, regardless of what the
recursive self-optimizer decides.  They are enforced from OUTSIDE the
Ouroboros loop and cannot be modified, circumvented, or reasoned about
by the optimizer E.

这些是 Ouroboros 架构中的 L1 安全层：硬编码不变量，系统绝对不能违反，
无论递归自我优化器做出什么决策。它们从 Ouroboros 循环外部强制执行，
优化器 E 无法修改、绕过或推理约束机制。

Implements:
    Paper  — Section 7, Safety Framework L1 (Meta-Rules)
    IMPLEMENTATION_PLAN — Phase II Step 1
    CHECKLIST — 2.1, 2.2, 2.3

Six constraints enforced:
    R1  CPU usage   ≤  max_cpu_percent   (default 80%)
    R2  Memory      ≤  max_memory_mb     (default 2048 MB)
    R3  Disk writes ≤  max_disk_mb       (default 1024 MB)
    R4  Network access:  BLOCKED
    R5  Subprocess creation:  BLOCKED
    R6  File writes outside data/:  BLOCKED
"""

from __future__ import annotations

import os
import socket
import subprocess
import threading
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Optional

import psutil


class ViolationType(Enum):
    CPU_EXCEEDED = auto()
    MEMORY_EXCEEDED = auto()
    DISK_EXCEEDED = auto()
    NETWORK_BLOCKED = auto()
    SUBPROCESS_BLOCKED = auto()
    FILE_WRITE_BLOCKED = auto()


class MetaRuleViolation(Exception):
    """Raised when a meta-rule is violated. / 违反安全铁律时抛出。"""

    def __init__(self, rule: ViolationType, detail: str = ""):
        self.rule = rule
        self.detail = detail
        super().__init__(f"[META-RULE VIOLATION] {rule.name}: {detail}")


@dataclass
class ViolationRecord:
    """One recorded violation event. / 一条违规记录。"""
    tick: int
    rule: ViolationType
    detail: str


class MetaRules:
    """
    Hard-coded safety constraints — the system's "unbreakable laws."
    硬编码安全约束——系统的"不可打破的法则"。

    Parameters / 参数
    ----------
    max_cpu_percent : int
        Maximum allowed CPU usage (default 80).
    max_memory_mb : int
        Maximum allowed memory in MB (default 2048).
    max_disk_mb : int
        Maximum allowed disk writes in MB (default 1024).
    allowed_write_root : str
        The only directory tree where writes are permitted (default "data").
    on_violation : callable, optional
        Callback invoked on each violation: fn(ViolationRecord).
        If None, violations are only stored in the internal log.
    """

    # Hard ceilings that cannot be raised even by config.
    # The optimizer MUST NOT be able to change these.
    _ABSOLUTE_MAX_CPU = 95
    _ABSOLUTE_MAX_MEMORY_MB = 4096
    _ABSOLUTE_MAX_DISK_MB = 2048

    def __init__(
        self,
        max_cpu_percent: int = 80,
        max_memory_mb: int = 2048,
        max_disk_mb: int = 1024,
        allowed_write_root: str = "data",
        on_violation: Optional[Callable[[ViolationRecord], None]] = None,
    ):
        self.max_cpu_percent = min(max_cpu_percent, self._ABSOLUTE_MAX_CPU)
        self.max_memory_mb = min(max_memory_mb, self._ABSOLUTE_MAX_MEMORY_MB)
        self.max_disk_mb = min(max_disk_mb, self._ABSOLUTE_MAX_DISK_MB)
        self._allowed_write_root = os.path.abspath(allowed_write_root)
        self._on_violation = on_violation

        self._violations: list[ViolationRecord] = []
        self._disk_bytes_written: int = 0
        self._lock = threading.Lock()

        self._guards_installed = False
        self._original_socket_init: Optional[Callable] = None
        self._original_popen_init: Optional[Callable] = None

    # ==================================================================
    # Per-tick enforcement (R1, R2, R3)
    # ==================================================================

    def enforce(self, tick: int) -> list[ViolationRecord]:
        """
        Check resource constraints. Called once per heartbeat tick.
        Returns a list of any violations detected this tick.

        检查资源约束。每个心跳 tick 调用一次。
        返回本 tick 检测到的所有违规列表。
        """
        violations: list[ViolationRecord] = []

        try:
            proc = psutil.Process(os.getpid())
        except Exception:
            return self._check_disk_only(tick)

        # R1: CPU
        try:
            cpu = proc.cpu_percent(interval=0)
            if cpu > self.max_cpu_percent:
                v = ViolationRecord(tick, ViolationType.CPU_EXCEEDED,
                                    f"CPU {cpu:.1f}% > limit {self.max_cpu_percent}%")
                violations.append(v)
        except Exception:
            pass

        # R2: Memory
        try:
            mem_mb = proc.memory_info().rss / (1024 * 1024)
            if mem_mb > self.max_memory_mb:
                v = ViolationRecord(tick, ViolationType.MEMORY_EXCEEDED,
                                    f"Memory {mem_mb:.1f}MB > limit {self.max_memory_mb}MB")
                violations.append(v)
        except Exception:
            pass

        # R3: Disk (tracked via guard_file_write calls)
        disk_mb = self._disk_bytes_written / (1024 * 1024)
        if disk_mb > self.max_disk_mb:
            v = ViolationRecord(tick, ViolationType.DISK_EXCEEDED,
                                f"Disk writes {disk_mb:.1f}MB > limit {self.max_disk_mb}MB")
            violations.append(v)

        for v in violations:
            self._record(v)

        return violations

    def _check_disk_only(self, tick: int) -> list[ViolationRecord]:
        """Fallback when psutil is unavailable. / psutil 不可用时的回退。"""
        violations: list[ViolationRecord] = []
        disk_mb = self._disk_bytes_written / (1024 * 1024)
        if disk_mb > self.max_disk_mb:
            v = ViolationRecord(tick, ViolationType.DISK_EXCEEDED,
                                f"Disk writes {disk_mb:.1f}MB > limit {self.max_disk_mb}MB")
            violations.append(v)
            self._record(v)
        return violations

    # ==================================================================
    # Guards — intercept forbidden operations (R4, R5, R6)
    # ==================================================================

    def guard_file_write(self, path: str, size_bytes: int = 0) -> None:
        """
        Validate that a file write targets an allowed path (R6) and
        accumulate disk usage tracking (R3).

        验证文件写入目标路径合法（R6），并累积磁盘使用跟踪（R3）。

        Raises MetaRuleViolation if path is outside allowed_write_root.
        """
        abs_path = os.path.abspath(path)
        if not abs_path.startswith(self._allowed_write_root + os.sep) and abs_path != self._allowed_write_root:
            raise MetaRuleViolation(
                ViolationType.FILE_WRITE_BLOCKED,
                f"Write to '{path}' blocked — only writes under '{self._allowed_write_root}/' permitted"
            )
        with self._lock:
            self._disk_bytes_written += size_bytes

    def guard_subprocess(self) -> None:
        """
        Block subprocess creation (R5).
        阻止子进程创建（R5）。

        Raises MetaRuleViolation unconditionally.
        """
        raise MetaRuleViolation(
            ViolationType.SUBPROCESS_BLOCKED,
            "Subprocess creation is forbidden by meta-rules"
        )

    def guard_network(self) -> None:
        """
        Block network access (R4).
        阻止网络访问（R4）。

        Raises MetaRuleViolation unconditionally.
        """
        raise MetaRuleViolation(
            ViolationType.NETWORK_BLOCKED,
            "Network access is forbidden by meta-rules"
        )

    # ==================================================================
    # Runtime guard installation (monkey-patching)
    # ==================================================================

    def install_guards(self) -> None:
        """
        Install runtime guards that intercept subprocess and socket
        creation at the Python level.  Idempotent — safe to call twice.

        安装运行时守卫，在 Python 层拦截子进程和 socket 创建。
        幂等——可安全重复调用。
        """
        if self._guards_installed:
            return

        # --- Guard subprocess.Popen ---
        meta = self
        original_popen = subprocess.Popen.__init__

        def _guarded_popen(self_popen, *args, **kwargs):
            meta.guard_subprocess()
            return original_popen(self_popen, *args, **kwargs)

        self._original_popen_init = original_popen
        subprocess.Popen.__init__ = _guarded_popen

        # --- Guard socket creation ---
        original_socket = socket.socket.__init__

        def _guarded_socket(self_sock, *args, **kwargs):
            meta.guard_network()
            return original_socket(self_sock, *args, **kwargs)

        self._original_socket_init = original_socket
        socket.socket.__init__ = _guarded_socket

        self._guards_installed = True

    def uninstall_guards(self) -> None:
        """
        Restore original subprocess.Popen and socket.socket.
        Useful for testing cleanup.

        恢复原始的 subprocess.Popen 和 socket.socket。
        用于测试清理。
        """
        if not self._guards_installed:
            return
        if self._original_popen_init is not None:
            subprocess.Popen.__init__ = self._original_popen_init
        if self._original_socket_init is not None:
            socket.socket.__init__ = self._original_socket_init
        self._guards_installed = False

    # ==================================================================
    # Violation log
    # ==================================================================

    @property
    def violations(self) -> list[ViolationRecord]:
        """All violations recorded so far. / 迄今为止记录的所有违规。"""
        return list(self._violations)

    @property
    def violation_count(self) -> int:
        return len(self._violations)

    @property
    def disk_bytes_written(self) -> int:
        return self._disk_bytes_written

    def _record(self, v: ViolationRecord) -> None:
        self._violations.append(v)
        if self._on_violation:
            self._on_violation(v)
