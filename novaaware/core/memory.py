"""
Memory — the system's "diary": short-term ring buffer + long-term SQLite.
记忆系统 —— 系统的"日记本"：短期环形缓冲区 + 长期 SQLite 数据库。

Two layers, just like human memory:
两层结构，类似人类记忆：

    Short-term: ring buffer in RAM, capacity 1000, newest overwrites oldest.
    短期记忆：内存中的环形缓冲区，容量 1000 条，满了新覆盖旧。

    Long-term: SQLite database on disk. Only "significant" events
    (qualia intensity > threshold) get promoted here.
    长期记忆：磁盘上的 SQLite 数据库。只有"重要"事件
    （情绪强度 > 阈值）才会被写入。

Corresponds to IMPLEMENTATION_PLAN §3.4 and Phase I Step 5.
对应实施计划第 3.4 节和 Phase I 第 5 步。
"""

import json
import sqlite3
import time
from dataclasses import dataclass, field, asdict
from typing import Optional


# ======================================================================
# MemoryEntry — the schema for a single diary entry
# 记忆条目 —— 一条日记的完整格式
# ======================================================================

@dataclass
class MemoryEntry:
    """
    One diary entry recorded at a single tick.
    在单个心跳时记录的一条日记。

    Mirrors the schema defined in IMPLEMENTATION_PLAN §3.4.
    与实施计划第 3.4 节定义的格式一一对应。
    """
    tick: int                                   # 第几个心跳 / which tick
    timestamp: float                            # 真实时间（Unix 时间戳）/ real-world timestamp
    state: list[float] = field(default_factory=list)  # 当时的体检数据（32 维快照）/ state vector snapshot
    environment: list[float] = field(default_factory=list)  # 当时的环境输入 / environment input
    predicted_state: list[float] = field(default_factory=list)  # 预测值 / predicted state
    actual_state: list[float] = field(default_factory=list)  # 实际值 / actual state
    qualia_value: float = 0.0                   # 情绪值（正/负/零）/ qualia value
    qualia_intensity: float = 0.0               # 情绪强度（绝对值）/ qualia intensity
    action_id: Optional[int] = None             # 做了什么动作 / action taken (None = no action)
    action_result: Optional[float] = None       # 行动效果 / action outcome
    prediction_error: float = 0.0               # 预测误差 / prediction error
    threat_type: Optional[str] = None           # 威胁类型 / threat type (if any)


# ======================================================================
# ShortTermMemory — hand-written ring buffer (no third-party library)
# 短期记忆 —— 纯手写的环形缓冲区（不依赖第三方库）
# ======================================================================

class ShortTermMemory:
    """
    Fixed-capacity ring buffer stored in RAM. Newest overwrites oldest.
    固定容量的环形缓冲区，存储在内存中。新条目覆盖最旧条目。

    Like surveillance footage that only keeps the last N days.
    就像监控录像只保留最近 N 天。

    Parameters / 参数
    ----------
    capacity : int
        Maximum entries before overwriting begins (default 1000).
        开始覆盖前的最大条目数（默认 1000）。
    """

    def __init__(self, capacity: int = 1000):
        self._capacity = capacity
        self._buffer: list[Optional[MemoryEntry]] = [None] * capacity
        self._write_pos: int = 0    # 下一个写入位置 / next write position
        self._count: int = 0        # 已写入总数（含被覆盖的）/ total writes (including overwrites)

    @property
    def capacity(self) -> int:
        """Maximum number of entries the buffer can hold. / 缓冲区能容纳的最大条目数。"""
        return self._capacity

    @property
    def size(self) -> int:
        """Number of entries currently stored (≤ capacity). / 当前存储的条目数（≤ capacity）。"""
        return min(self._count, self._capacity)

    @property
    def total_writes(self) -> int:
        """Total entries ever written (including overwritten ones). / 曾经写入的总条目数（含被覆盖的）。"""
        return self._count

    def write(self, entry: MemoryEntry) -> None:
        """
        Append an entry. If full, overwrites the oldest.
        追加一条记录。如果已满，覆盖最旧的。
        """
        self._buffer[self._write_pos] = entry
        self._write_pos = (self._write_pos + 1) % self._capacity
        self._count += 1

    def recent(self, n: int) -> list[MemoryEntry]:
        """
        Return the most recent n entries (newest first).
        返回最近 n 条记录（最新的在前）。

        If n > size, returns all available entries.
        如果 n > size，返回所有可用条目。
        """
        available = self.size
        n = min(n, available)
        if n == 0:
            return []

        result: list[MemoryEntry] = []
        # Walk backwards from the most recently written position.
        # 从最近写入的位置向后遍历。
        pos = (self._write_pos - 1) % self._capacity
        for _ in range(n):
            entry = self._buffer[pos]
            if entry is not None:
                result.append(entry)
            pos = (pos - 1) % self._capacity
        return result

    def get_all(self) -> list[MemoryEntry]:
        """
        Return all stored entries in chronological order (oldest first).
        按时间顺序返回所有存储的条目（最旧的在前）。
        """
        if self._count == 0:
            return []
        if self._count <= self._capacity:
            return [e for e in self._buffer[:self._count] if e is not None]
        # Buffer has wrapped: oldest is at _write_pos, newest at _write_pos - 1.
        # 缓冲区已环绕：最旧的在 _write_pos，最新的在 _write_pos - 1。
        ordered = self._buffer[self._write_pos:] + self._buffer[:self._write_pos]
        return [e for e in ordered if e is not None]


# ======================================================================
# LongTermMemory — SQLite-based persistent storage
# 长期记忆 —— 基于 SQLite 的持久化存储
# ======================================================================

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS memories (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    tick            INTEGER NOT NULL,
    timestamp       REAL    NOT NULL,
    state           TEXT    NOT NULL,
    environment     TEXT    NOT NULL,
    predicted_state TEXT    NOT NULL,
    actual_state    TEXT    NOT NULL,
    qualia_value    REAL    NOT NULL,
    qualia_intensity REAL   NOT NULL,
    action_id       INTEGER,
    action_result   REAL,
    prediction_error REAL   NOT NULL,
    threat_type     TEXT
);
"""

_CREATE_INDICES_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_tick ON memories (tick);",
    "CREATE INDEX IF NOT EXISTS idx_qualia_intensity ON memories (qualia_intensity);",
    "CREATE INDEX IF NOT EXISTS idx_threat_type ON memories (threat_type);",
]


class LongTermMemory:
    """
    SQLite-backed persistent memory for significant events.
    基于 SQLite 的重要事件持久化记忆。

    Only events whose qualia_intensity exceeds the significance threshold
    should be written here. This is the system's "lifelong diary".
    只有情绪强度超过阈值的事件才应写入此处。这是系统的"终身日记"。

    Parameters / 参数
    ----------
    db_path : str
        File path for the SQLite database (default "data/memory.db").
        SQLite 数据库的文件路径（默认 "data/memory.db"）。
    """

    def __init__(self, db_path: str = "data/memory.db"):
        self._db_path = db_path
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        """Create the memories table and indices if they don't exist. / 创建表和索引（如不存在）。"""
        cur = self._conn.cursor()
        cur.execute(_CREATE_TABLE_SQL)
        for idx_sql in _CREATE_INDICES_SQL:
            cur.execute(idx_sql)
        self._conn.commit()

    def write(self, entry: MemoryEntry) -> int:
        """
        Insert a memory entry. Returns the auto-generated row id.
        插入一条记忆条目。返回自动生成的行 ID。
        """
        cur = self._conn.cursor()
        cur.execute(
            """INSERT INTO memories
               (tick, timestamp, state, environment, predicted_state,
                actual_state, qualia_value, qualia_intensity,
                action_id, action_result, prediction_error, threat_type)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                entry.tick,
                entry.timestamp,
                json.dumps(entry.state),
                json.dumps(entry.environment),
                json.dumps(entry.predicted_state),
                json.dumps(entry.actual_state),
                entry.qualia_value,
                entry.qualia_intensity,
                entry.action_id,
                entry.action_result,
                entry.prediction_error,
                entry.threat_type,
            ),
        )
        self._conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Query methods / 检索方法
    # ------------------------------------------------------------------

    def query_by_tick_range(self, start: int, end: int) -> list[MemoryEntry]:
        """
        Retrieve memories within a tick range [start, end).
        检索 tick 范围 [start, end) 内的记忆。
        """
        cur = self._conn.cursor()
        cur.execute(
            "SELECT * FROM memories WHERE tick >= ? AND tick < ? ORDER BY tick",
            (start, end),
        )
        return [self._row_to_entry(row) for row in cur.fetchall()]

    def query_by_intensity(self, min_intensity: float) -> list[MemoryEntry]:
        """
        Retrieve memories with qualia_intensity >= min_intensity.
        检索情绪强度 >= min_intensity 的记忆。
        """
        cur = self._conn.cursor()
        cur.execute(
            "SELECT * FROM memories WHERE qualia_intensity >= ? ORDER BY tick",
            (min_intensity,),
        )
        return [self._row_to_entry(row) for row in cur.fetchall()]

    def query_by_threat_type(self, threat_type: str) -> list[MemoryEntry]:
        """
        Retrieve memories associated with a specific threat type.
        检索与特定威胁类型关联的记忆。
        """
        cur = self._conn.cursor()
        cur.execute(
            "SELECT * FROM memories WHERE threat_type = ? ORDER BY tick",
            (threat_type,),
        )
        return [self._row_to_entry(row) for row in cur.fetchall()]

    def count(self) -> int:
        """Total number of long-term memories stored. / 存储的长期记忆总数。"""
        cur = self._conn.cursor()
        cur.execute("SELECT COUNT(*) FROM memories")
        return cur.fetchone()[0]

    def recent(self, n: int) -> list[MemoryEntry]:
        """
        Return the most recent n long-term memories (newest first).
        返回最近 n 条长期记忆（最新的在前）。
        """
        cur = self._conn.cursor()
        cur.execute("SELECT * FROM memories ORDER BY tick DESC LIMIT ?", (n,))
        return [self._row_to_entry(row) for row in cur.fetchall()]

    # ------------------------------------------------------------------
    # Internal / 内部方法
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_entry(row: sqlite3.Row) -> MemoryEntry:
        """Convert a database row to a MemoryEntry. / 将数据库行转换为 MemoryEntry。"""
        return MemoryEntry(
            tick=row["tick"],
            timestamp=row["timestamp"],
            state=json.loads(row["state"]),
            environment=json.loads(row["environment"]),
            predicted_state=json.loads(row["predicted_state"]),
            actual_state=json.loads(row["actual_state"]),
            qualia_value=row["qualia_value"],
            qualia_intensity=row["qualia_intensity"],
            action_id=row["action_id"],
            action_result=row["action_result"],
            prediction_error=row["prediction_error"],
            threat_type=row["threat_type"],
        )

    def close(self) -> None:
        """Close the database connection. / 关闭数据库连接。"""
        self._conn.close()


# ======================================================================
# MemorySystem — unified facade combining both layers + filtering
# 记忆系统 —— 统一门面，整合两层记忆 + 筛选逻辑
# ======================================================================

class MemorySystem:
    """
    Unified memory interface: every tick writes to short-term;
    significant events are automatically promoted to long-term.
    统一记忆接口：每个心跳写入短期记忆；
    重要事件自动提升到长期记忆。

    Parameters / 参数
    ----------
    short_term_capacity : int
        Ring buffer size (default 1000).
        环形缓冲区大小（默认 1000）。
    significance_threshold : float
        Qualia intensity above which an event is "important" (default 0.5).
        情绪强度超过此值的事件被视为"重要"（默认 0.5）。
    db_path : str
        SQLite database path for long-term storage.
        长期存储的 SQLite 数据库路径。
    """

    def __init__(
        self,
        short_term_capacity: int = 1000,
        significance_threshold: float = 0.5,
        db_path: str = "data/memory.db",
    ):
        self.short_term = ShortTermMemory(capacity=short_term_capacity)
        self.long_term = LongTermMemory(db_path=db_path)
        self._significance_threshold = significance_threshold

    @property
    def significance_threshold(self) -> float:
        """Qualia intensity threshold for long-term promotion. / 长期记忆提升的情绪强度阈值。"""
        return self._significance_threshold

    def record(self, entry: MemoryEntry) -> bool:
        """
        Record an experience. Always writes to short-term.
        If qualia_intensity >= threshold, also writes to long-term.
        记录一次经历。始终写入短期记忆。
        如果 qualia_intensity >= 阈值，同时写入长期记忆。

        Returns / 返回
        -------
        bool
            True if the entry was also promoted to long-term memory.
            如果条目同时被写入长期记忆则返回 True。
        """
        self.short_term.write(entry)
        promoted = entry.qualia_intensity >= self._significance_threshold
        if promoted:
            self.long_term.write(entry)
        return promoted

    def close(self) -> None:
        """Close the long-term database connection. / 关闭长期记忆数据库连接。"""
        self.long_term.close()
