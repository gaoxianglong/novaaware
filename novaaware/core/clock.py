"""
Clock — the heartbeat of the consciousness engine.
时钟 —— 意识引擎的"心跳"。

Drives the core loop at a fixed tick interval (default 10 Hz / 100 ms).
Every module in the system uses the Clock's tick counter as its global
time reference: the self-model records "how many ticks I have lived",
the prediction engine looks back over the last N ticks, and the main
loop calls wait_until_next_tick() to maintain a steady rhythm.

以固定间隔（默认 10 Hz / 100 ms）驱动核心循环。
系统中的所有模块都以时钟的 tick 计数作为全局时间参考：
自我模型记录"我已经活了多少个心跳"，预测引擎回溯最近 N 个心跳，
主循环调用 wait_until_next_tick() 来维持稳定的节奏。
"""

import time


class Clock:
    """
    Drives the core loop at a fixed tick interval.
    以固定间隔驱动核心循环的时钟。

    Parameters / 参数
    ----------
    tick_interval_ms : int
        Milliseconds between ticks (default 100 → 10 Hz).
        每个心跳之间的毫秒数（默认 100 → 10 Hz）。
    max_ticks : int
        Maximum number of ticks before the engine stops.
        引擎停止前的最大心跳数。
    """

    def __init__(self, tick_interval_ms: int = 100, max_ticks: int = 1_000_000):
        self._interval_s = tick_interval_ms / 1000.0   # 心跳间隔（秒） / tick interval in seconds
        self._max_ticks = max_ticks                     # 最大心跳数 / max heartbeat count
        self._current_tick: int = 0                     # 当前心跳编号 / current tick number
        self._start_time: float = time.monotonic()      # 时钟创建时刻 / clock creation timestamp
        self._last_tick_time: float = self._start_time  # 上次心跳时刻 / last tick timestamp

    # ------------------------------------------------------------------
    # Public API / 公共接口
    # ------------------------------------------------------------------

    def tick(self) -> int:
        """
        Advance the clock by one step and return the new tick number.
        推进时钟一步，返回新的心跳编号。

        This is called once at the top of each core-loop iteration.
        在每次核心循环迭代开始时调用一次。
        """
        self._current_tick += 1
        self._last_tick_time = time.monotonic()
        return self._current_tick

    def wait_until_next_tick(self) -> None:
        """
        Sleep until the next tick boundary, compensating for drift.
        休眠到下一个心跳边界，自动补偿时间漂移。

        If the core loop finishes its work in 60 ms, this method
        sleeps for the remaining 40 ms so the total cycle = 100 ms.
        如果核心循环在 60 ms 内完成工作，此方法会休眠剩余的 40 ms，
        使得总周期 = 100 ms。
        """
        target = self._last_tick_time + self._interval_s
        now = time.monotonic()
        remaining = target - now
        if remaining > 0:
            time.sleep(remaining)

    # ------------------------------------------------------------------
    # Properties / 属性
    # ------------------------------------------------------------------

    @property
    def current_tick(self) -> int:
        """Current tick number (starts at 0). / 当前心跳编号（从 0 开始）。"""
        return self._current_tick

    @property
    def max_ticks(self) -> int:
        """Maximum ticks allowed. / 允许的最大心跳数。"""
        return self._max_ticks

    @property
    def has_remaining(self) -> bool:
        """
        True if current_tick < max_ticks (engine should keep running).
        当 current_tick < max_ticks 时为 True（引擎应继续运行）。
        """
        return self._current_tick < self._max_ticks

    @property
    def interval_s(self) -> float:
        """Configured tick interval in seconds. / 配置的心跳间隔（秒）。"""
        return self._interval_s

    @property
    def elapsed_s(self) -> float:
        """
        Wall-clock seconds since the clock was created.
        自时钟创建以来经过的真实秒数。
        """
        return time.monotonic() - self._start_time

    @property
    def tick_rate_hz(self) -> float:
        """
        Actual average tick rate measured so far.
        到目前为止测量到的实际平均心跳频率（Hz）。
        """
        elapsed = self.elapsed_s
        if elapsed <= 0 or self._current_tick == 0:
            return 0.0
        return self._current_tick / elapsed
