"""
RecursionLimiter — L3 safety layer: hard cap on self-referential recursion depth.
递归限制器 —— L3 安全层：自指递归深度的硬性上限。

From the paper (Section 2.2, Definition 2.2):

    Self-referential recursion means the system can take its self-
    representation r_S as input to generate evaluation r_S^(2), then
    r_S^(3), ... forming an infinitely nestable self-mapping sequence
    — Hofstadter's "Strange Loop."

From the paper (Section 7.7, L3):

    Hard upper bounds on the depth of self-referential recursion,
    preventing unbounded metacognitive escalation in early deployment
    phases.

From the paper (Section 7.4):

    Each iteration Θ_{n+1} = E(Θ_n, Q_n) potentially generates
    cognitive structures with no human-interpretable analogue.
    After sufficient recursive depth, the system's decision-making
    process may become fundamentally incommensurable with human
    cognition.

Concrete semantics of depth levels:

    depth 0 — No reflection.  Phase I: optimizer E is disabled.
    depth 1 — One level: E reviews qualia history and modifies Θ.
              Phase II: E can reflect once but CANNOT reflect on
              the results of its own reflection.
    depth 2 — E can evaluate its own previous optimization step.
              Blocked in Phase II; may be unlocked in Phase III.
    depth n — n layers of recursive metacognition.

Implements:
    Paper  — Section 7.7, Safety Framework L3 (Recursion Depth Limits)
    IMPLEMENTATION_PLAN — Phase II Step 3
    CHECKLIST — 2.7, 2.8, 2.9
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Optional


class RecursionDepthExceeded(Exception):
    """Raised when self-referential recursion exceeds the allowed depth.
    当自指递归超过允许深度时抛出。"""

    def __init__(self, attempted: int, max_allowed: int):
        self.attempted = attempted
        self.max_allowed = max_allowed
        super().__init__(
            f"[L3 RECURSION LIMIT] Attempted depth {attempted} "
            f"exceeds maximum {max_allowed} — "
            f"metacognitive escalation blocked"
        )


class RecursionLimiter:
    """
    Hard cap on self-referential recursion depth — the system's
    "metacognitive ceiling."

    自指递归深度的硬性上限——系统的"元认知天花板"。

    The optimizer E operates at recursion depth 1: it reviews the
    system's qualia history and proposes parameter changes.  If E
    were allowed to recurse (reflect on its own reflection), it
    could potentially develop cognitive structures that are
    fundamentally incommensurable with human cognition (paper §7.4).
    This limiter prevents that.

    Thread-safe: each thread tracks its own recursion depth via
    threading.local(), so concurrent optimizer threads cannot
    interfere with each other's depth accounting.

    Parameters / 参数
    ----------
    max_depth : int
        Maximum allowed recursion depth (default 1 for Phase II).
        Phase I should use 0 (optimizer disabled).

    Usage / 用法
    -----
    ```python
    limiter = RecursionLimiter(max_depth=1)

    with limiter.guard():
        # depth is now 1 — optimizer E runs here
        optimizer.reflect_and_modify(params)

        with limiter.guard():
            # depth would be 2 — RecursionDepthExceeded raised!
            pass
    ```
    """

    # The optimizer MUST NOT be able to raise this ceiling.
    _ABSOLUTE_MAX_DEPTH = 5

    def __init__(self, max_depth: int = 1):
        if max_depth < 0:
            raise ValueError("max_depth must be non-negative")
        self._max_depth = min(max_depth, self._ABSOLUTE_MAX_DEPTH)
        self._local = threading.local()

        self._total_entries: int = 0
        self._blocked_count: int = 0
        self._peak_depth: int = 0
        self._lock = threading.Lock()

    # ==================================================================
    # Core API
    # ==================================================================

    @contextmanager
    def guard(self):
        """
        Context manager that increments recursion depth on entry and
        decrements on exit.  Raises RecursionDepthExceeded if the new
        depth would exceed max_depth.

        进入时递增递归深度，退出时递减。如果新深度超过 max_depth 则抛出异常。
        """
        current = self._current_depth
        new_depth = current + 1

        if new_depth > self._max_depth:
            with self._lock:
                self._blocked_count += 1
            raise RecursionDepthExceeded(new_depth, self._max_depth)

        self._set_depth(new_depth)
        with self._lock:
            self._total_entries += 1
            if new_depth > self._peak_depth:
                self._peak_depth = new_depth

        try:
            yield new_depth
        finally:
            self._set_depth(current)

    def check(self, required_depth: int = 1) -> bool:
        """
        Check whether entering *required_depth* more levels is allowed
        WITHOUT actually entering.  Returns True if allowed.

        检查是否可以再进入 required_depth 层，但不实际进入。
        """
        return (self._current_depth + required_depth) <= self._max_depth

    # ==================================================================
    # Depth inspection
    # ==================================================================

    @property
    def current_depth(self) -> int:
        """Current recursion depth for this thread. / 当前线程的递归深度。"""
        return self._current_depth

    @property
    def max_depth(self) -> int:
        """Configured maximum depth. / 配置的最大深度。"""
        return self._max_depth

    @property
    def remaining_depth(self) -> int:
        """How many more levels can be entered. / 还可以进入多少层。"""
        return max(0, self._max_depth - self._current_depth)

    # ==================================================================
    # Statistics
    # ==================================================================

    @property
    def total_entries(self) -> int:
        """Total successful guard entries. / 成功进入守卫的总次数。"""
        return self._total_entries

    @property
    def blocked_count(self) -> int:
        """Total blocked attempts. / 被阻止的总次数。"""
        return self._blocked_count

    @property
    def peak_depth(self) -> int:
        """Highest depth ever reached. / 曾达到的最大深度。"""
        return self._peak_depth

    # ==================================================================
    # Internal helpers
    # ==================================================================

    @property
    def _current_depth(self) -> int:
        return getattr(self._local, "depth", 0)

    def _set_depth(self, value: int) -> None:
        self._local.depth = value
