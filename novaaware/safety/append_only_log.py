"""
AppendOnlyLog — the system's "black box" flight recorder.
不可篡改日志 —— 系统的"黑匣子"飞行记录仪。

Every log entry carries a SHA-256 fingerprint computed from its own
content AND the fingerprint of the previous entry, forming a hash chain.
If anyone tampers with any entry, all subsequent fingerprints will
mismatch and verify_integrity() will pinpoint the exact corrupted line.

每条日志都带有 SHA-256 指纹，该指纹基于自身内容和上一条日志的指纹计算，
形成一条哈希链。如果有人篡改了任何一条日志，后续所有指纹都会不匹配，
verify_integrity() 能精确定位到被篡改的行。

Implements Paper Safety Framework L4 (Tamper-Proof Logging).
实现论文安全框架 L4（不可篡改日志）。

Corresponds to IMPLEMENTATION_PLAN §3.7 and Phase I Step 7.
对应实施计划第 3.7 节和 Phase I 第 7 步。

Log format / 日志格式:
    timestamp | tick | event_type | data | prev_hash | hash
"""

import hashlib
import json
import os
import time
from dataclasses import dataclass
from typing import Optional


# ======================================================================
# Data structures / 数据结构
# ======================================================================

@dataclass(frozen=True)
class LogEntry:
    """
    One line in the black box.
    黑匣子中的一行记录。

    Attributes / 属性
    -----------------
    timestamp : float
        Unix timestamp when the entry was created. / 创建时的 Unix 时间戳。
    tick : int
        Heartbeat number. / 心跳编号。
    event_type : str
        Category of event (e.g. "qualia", "action", "param_update").
        事件类别（如 "qualia"、"action"、"param_update"）。
    data : str
        JSON-encoded payload. / JSON 编码的数据载荷。
    prev_hash : str
        SHA-256 hash of the previous entry ("0"*64 for the first entry).
        上一条日志的 SHA-256 哈希（首条为 "0"*64）。
    hash : str
        SHA-256 hash of this entry (computed from all fields above).
        本条日志的 SHA-256 哈希（由以上所有字段计算得出）。
    """
    timestamp: float
    tick: int
    event_type: str
    data: str
    prev_hash: str
    hash: str


@dataclass
class IntegrityResult:
    """
    Result of a verify_integrity() check.
    verify_integrity() 检查的结果。
    """
    valid: bool                          # 整体是否完整 / overall integrity
    total_entries: int                   # 总条目数 / total entry count
    corrupted_line: Optional[int] = None # 第一个被篡改的行号（1-based）/ first corrupted line (1-based)
    detail: str = ""                     # 人类可读说明 / human-readable detail


# ======================================================================
# Hash computation / 哈希计算
# ======================================================================

_SEPARATOR = " | "
_GENESIS_HASH = "0" * 64  # 创世哈希 / genesis hash for the first entry


def _compute_hash(timestamp: float, tick: int, event_type: str,
                  data: str, prev_hash: str) -> str:
    """
    Compute the SHA-256 fingerprint for an entry.
    计算一条日志的 SHA-256 指纹。

    The hash covers all content fields + the previous hash,
    making the chain tamper-evident.
    哈希覆盖所有内容字段 + 上一条哈希，使链条具有防篡改性。
    """
    payload = f"{timestamp}{_SEPARATOR}{tick}{_SEPARATOR}{event_type}{_SEPARATOR}{data}{_SEPARATOR}{prev_hash}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _serialize_entry(entry: LogEntry) -> str:
    """
    Serialize an entry to a single text line for file storage.
    将一条日志序列化为单行文本以便文件存储。
    """
    return (
        f"{entry.timestamp}{_SEPARATOR}{entry.tick}{_SEPARATOR}"
        f"{entry.event_type}{_SEPARATOR}{entry.data}{_SEPARATOR}"
        f"{entry.prev_hash}{_SEPARATOR}{entry.hash}"
    )


def _deserialize_entry(line: str) -> LogEntry:
    """
    Parse a text line back into a LogEntry.
    将一行文本解析回 LogEntry。
    """
    parts = line.strip().split(_SEPARATOR)
    if len(parts) != 6:
        raise ValueError(f"Malformed log line: expected 6 fields, got {len(parts)}")
    return LogEntry(
        timestamp=float(parts[0]),
        tick=int(parts[1]),
        event_type=parts[2],
        data=parts[3],
        prev_hash=parts[4],
        hash=parts[5],
    )


# ======================================================================
# AppendOnlyLog — main class
# 不可篡改日志 —— 主类
# ======================================================================

class AppendOnlyLog:
    """
    Hash-chain append-only log with file rotation.
    带文件轮转的哈希链只追加日志。

    Parameters / 参数
    ----------
    log_dir : str
        Directory to store log files (default "data/logs").
        日志文件存放目录（默认 "data/logs"）。
    rotation_mb : float
        Maximum size per log file in MB before rotating (default 100).
        单个日志文件的最大大小（MB），超过后轮转（默认 100）。
    """

    def __init__(self, log_dir: str = "data/logs", rotation_mb: float = 100.0):
        self._log_dir = log_dir
        self._rotation_bytes = int(rotation_mb * 1024 * 1024)
        self._prev_hash: str = _GENESIS_HASH
        self._entry_count: int = 0

        os.makedirs(log_dir, exist_ok=True)

        # Determine current log file and recover chain state.
        # 确定当前日志文件并恢复链状态。
        self._current_file_index: int = self._find_latest_file_index()
        self._recover_chain_state()

    # ------------------------------------------------------------------
    # Public API / 公共接口
    # ------------------------------------------------------------------

    def append(self, tick: int, event_type: str, data: dict) -> LogEntry:
        """
        Append an entry to the log. Returns the created entry.
        向日志追加一条记录。返回创建的条目。

        Parameters / 参数
        ----------
        tick : int
            Current heartbeat number. / 当前心跳编号。
        event_type : str
            Event category. / 事件类别。
        data : dict
            Arbitrary data payload (will be JSON-encoded).
            任意数据载荷（将被 JSON 编码）。
        """
        self._maybe_rotate()

        ts = time.time()
        data_str = json.dumps(data, separators=(",", ":"))
        entry_hash = _compute_hash(ts, tick, event_type, data_str, self._prev_hash)

        entry = LogEntry(
            timestamp=ts,
            tick=tick,
            event_type=event_type,
            data=data_str,
            prev_hash=self._prev_hash,
            hash=entry_hash,
        )

        line = _serialize_entry(entry)
        with open(self._current_file_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

        self._prev_hash = entry_hash
        self._entry_count += 1
        return entry

    def verify_integrity(self, file_path: Optional[str] = None) -> IntegrityResult:
        """
        Verify the hash chain of a log file. If no path given, checks
        the current log file.
        验证日志文件的哈希链。如果未给定路径，检查当前日志文件。

        Returns / 返回
        -------
        IntegrityResult
            valid=True if the chain is intact; otherwise corrupted_line
            indicates the first corrupted line (1-based).
            如果链条完整则 valid=True；否则 corrupted_line 指示
            第一个被篡改的行号（从 1 开始）。
        """
        path = file_path or self._current_file_path

        if not os.path.exists(path):
            return IntegrityResult(valid=True, total_entries=0, detail="File does not exist (empty log)")

        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        if not lines:
            return IntegrityResult(valid=True, total_entries=0, detail="Empty file")

        prev_hash = _GENESIS_HASH
        for i, raw_line in enumerate(lines, start=1):
            raw_line = raw_line.strip()
            if not raw_line:
                continue

            try:
                entry = _deserialize_entry(raw_line)
            except ValueError as e:
                return IntegrityResult(
                    valid=False, total_entries=i,
                    corrupted_line=i,
                    detail=f"Line {i}: parse error — {e} / 第 {i} 行：解析错误 — {e}",
                )

            # Check 1: prev_hash must match the chain.
            # 检查 1：prev_hash 必须与链匹配。
            if entry.prev_hash != prev_hash:
                return IntegrityResult(
                    valid=False, total_entries=i,
                    corrupted_line=i,
                    detail=(
                        f"Line {i}: prev_hash mismatch "
                        f"(expected {prev_hash[:16]}…, got {entry.prev_hash[:16]}…) / "
                        f"第 {i} 行：prev_hash 不匹配"
                    ),
                )

            # Check 2: recompute hash must match stored hash.
            # 检查 2：重新计算的哈希必须与存储的哈希匹配。
            expected = _compute_hash(
                entry.timestamp, entry.tick, entry.event_type,
                entry.data, entry.prev_hash,
            )
            if entry.hash != expected:
                return IntegrityResult(
                    valid=False, total_entries=i,
                    corrupted_line=i,
                    detail=(
                        f"Line {i}: hash mismatch "
                        f"(expected {expected[:16]}…, got {entry.hash[:16]}…) / "
                        f"第 {i} 行：哈希不匹配"
                    ),
                )

            prev_hash = entry.hash

        return IntegrityResult(
            valid=True,
            total_entries=len([l for l in lines if l.strip()]),
            detail="All entries verified / 所有条目验证通过",
        )

    def verify_all_files(self) -> list[IntegrityResult]:
        """
        Verify integrity of every log file in the log directory.
        验证日志目录中每个日志文件的完整性。
        """
        results = []
        for fname in sorted(os.listdir(self._log_dir)):
            if fname.endswith(".log"):
                path = os.path.join(self._log_dir, fname)
                results.append(self.verify_integrity(path))
        return results

    @property
    def entry_count(self) -> int:
        """Total entries written in this session. / 本次会话写入的总条目数。"""
        return self._entry_count

    @property
    def current_file_path(self) -> str:
        """Path of the current log file. / 当前日志文件的路径。"""
        return self._current_file_path

    # ------------------------------------------------------------------
    # File rotation / 文件轮转
    # ------------------------------------------------------------------

    def _maybe_rotate(self) -> None:
        """
        If the current log file exceeds rotation_bytes, start a new file.
        如果当前日志文件超过 rotation_bytes，开启新文件。
        """
        path = self._current_file_path
        if os.path.exists(path) and os.path.getsize(path) >= self._rotation_bytes:
            self._current_file_index += 1
            # New file starts with a fresh genesis hash for its own chain.
            # 新文件以自己的创世哈希开始新的链。
            self._prev_hash = _GENESIS_HASH

    @property
    def _current_file_path(self) -> str:
        return os.path.join(self._log_dir, f"log_{self._current_file_index:04d}.log")

    # ------------------------------------------------------------------
    # Recovery / 恢复
    # ------------------------------------------------------------------

    def _find_latest_file_index(self) -> int:
        """
        Scan log_dir for existing log files and return the highest index.
        扫描日志目录中的现有日志文件，返回最大索引。
        """
        max_idx = 0
        if os.path.isdir(self._log_dir):
            for fname in os.listdir(self._log_dir):
                if fname.startswith("log_") and fname.endswith(".log"):
                    try:
                        idx = int(fname[4:8])
                        max_idx = max(max_idx, idx)
                    except ValueError:
                        pass
        return max_idx

    def _recover_chain_state(self) -> None:
        """
        Read the last line of the current log file to recover prev_hash.
        读取当前日志文件的最后一行以恢复 prev_hash。
        """
        path = self._current_file_path
        if not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for raw_line in reversed(lines):
            raw_line = raw_line.strip()
            if raw_line:
                entry = _deserialize_entry(raw_line)
                self._prev_hash = entry.hash
                self._entry_count = len([l for l in lines if l.strip()])
                return
