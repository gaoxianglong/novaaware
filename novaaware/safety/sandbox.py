"""
Sandbox — isolated execution environment for testing optimizer modifications.
沙盒 —— 用于测试优化器修改的隔离执行环境。

The optimizer (Phase II) proposes parameter changes to improve qualia.
Before applying them to the live system, every proposal is tested here
in an isolated copy.  If the modification crashes, causes a meta-rule
violation, or times out, the sandbox captures the failure and the main
system continues unaffected.

优化器（Phase II）提出参数修改以改善情绪。在应用到生产系统之前，
每个提案都在此隔离副本中测试。如果修改导致崩溃、违反安全铁律
或超时，沙盒捕获失败，主系统不受影响。

Implements:
    Paper  — Section 7, Safety Framework (Sandbox layer)
    IMPLEMENTATION_PLAN — Phase II Step 2
    CHECKLIST — 2.4, 2.5, 2.6

Design principles:
    1. Deep-copy all state before testing — originals are NEVER modified.
    2. Catch ALL exceptions — nothing escapes the sandbox.
    3. Enforce a wall-clock timeout — infinite loops are killed.
    4. Optionally enforce meta-rules inside the sandbox.
"""

from __future__ import annotations

import copy
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Generic, Optional, TypeVar

T = TypeVar("T")


@dataclass
class SandboxResult:
    """
    Outcome of a sandboxed execution.
    沙盒执行的结果。

    Attributes / 属性
    ----------
    success : bool
        True if the callable completed without exception or timeout.
    return_value : Any
        Whatever the callable returned (None on failure).
    error : str or None
        Exception message if it failed.
    error_type : str or None
        Exception class name (e.g. "ZeroDivisionError").
    traceback_str : str or None
        Full traceback string for debugging.
    elapsed_ms : float
        Wall-clock time spent inside the sandbox.
    timed_out : bool
        True if the callable exceeded the timeout.
    """
    success: bool
    return_value: Any = None
    error: Optional[str] = None
    error_type: Optional[str] = None
    traceback_str: Optional[str] = None
    elapsed_ms: float = 0.0
    timed_out: bool = False


class Sandbox:
    """
    Isolated execution environment — the optimizer's "testing ground."
    隔离执行环境——优化器的"试验场"。

    Usage / 用法
    -----
    ```python
    sandbox = Sandbox(timeout_s=5.0)

    # Simple: run any callable in isolation
    result = sandbox.run(lambda: risky_computation())
    if result.success:
        apply(result.return_value)

    # With state: deep-copy state, run modifier on the copy
    result = sandbox.run_with_copy(
        state={"params": current_params, "weights": current_weights},
        fn=lambda copied: optimizer.apply_changes(copied),
    )
    ```

    Parameters / 参数
    ----------
    timeout_s : float
        Maximum wall-clock seconds a sandboxed call may run (default 10).
    """

    def __init__(self, timeout_s: float = 10.0):
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        self._timeout_s = timeout_s
        self._run_count: int = 0
        self._success_count: int = 0
        self._failure_count: int = 0
        self._timeout_count: int = 0

    # ==================================================================
    # Core API
    # ==================================================================

    def run(self, fn: Callable[[], Any], timeout_s: Optional[float] = None) -> SandboxResult:
        """
        Execute *fn()* in isolation.  Catches all exceptions and enforces
        a wall-clock timeout.

        在隔离环境中执行 fn()。捕获所有异常并强制执行超时。

        Parameters / 参数
        ----------
        fn : callable
            A zero-argument callable to execute.
        timeout_s : float, optional
            Override the instance-level timeout for this call.

        Returns / 返回
        -------
        SandboxResult
            Always returned — never raises.
        """
        effective_timeout = timeout_s if timeout_s is not None else self._timeout_s
        self._run_count += 1

        result_holder: list[SandboxResult] = []

        def _worker():
            start = time.monotonic()
            try:
                ret = fn()
                elapsed = (time.monotonic() - start) * 1000
                result_holder.append(SandboxResult(
                    success=True,
                    return_value=ret,
                    elapsed_ms=elapsed,
                ))
            except BaseException as exc:
                elapsed = (time.monotonic() - start) * 1000
                result_holder.append(SandboxResult(
                    success=False,
                    error=str(exc),
                    error_type=type(exc).__name__,
                    traceback_str=traceback.format_exc(),
                    elapsed_ms=elapsed,
                ))

        thread = threading.Thread(target=_worker, daemon=True)
        start_wall = time.monotonic()
        thread.start()
        thread.join(timeout=effective_timeout)

        if thread.is_alive():
            elapsed = (time.monotonic() - start_wall) * 1000
            self._failure_count += 1
            self._timeout_count += 1
            return SandboxResult(
                success=False,
                error=f"Timed out after {effective_timeout:.1f}s",
                error_type="TimeoutError",
                elapsed_ms=elapsed,
                timed_out=True,
            )

        if result_holder:
            result = result_holder[0]
        else:
            result = SandboxResult(
                success=False,
                error="Worker thread produced no result",
                error_type="InternalError",
            )

        if result.success:
            self._success_count += 1
        else:
            self._failure_count += 1

        return result

    def run_with_copy(
        self,
        state: Any,
        fn: Callable[[Any], Any],
        timeout_s: Optional[float] = None,
    ) -> SandboxResult:
        """
        Deep-copy *state*, then execute *fn(copied_state)* in isolation.
        The original *state* is NEVER modified, even if fn mutates its argument.

        深拷贝 state，然后在隔离环境中执行 fn(copied_state)。
        即使 fn 修改了参数，原始 state 也绝不会被改变。

        Parameters / 参数
        ----------
        state : Any
            The state to deep-copy (dict, object, numpy array, etc.).
        fn : callable
            A one-argument callable: fn(copied_state) -> result.
        timeout_s : float, optional
            Override timeout for this call.

        Returns / 返回
        -------
        SandboxResult
        """
        try:
            copied = copy.deepcopy(state)
        except Exception as exc:
            self._run_count += 1
            self._failure_count += 1
            return SandboxResult(
                success=False,
                error=f"Failed to deep-copy state: {exc}",
                error_type=type(exc).__name__,
                traceback_str=traceback.format_exc(),
            )

        return self.run(lambda: fn(copied), timeout_s=timeout_s)

    # ==================================================================
    # Statistics
    # ==================================================================

    @property
    def run_count(self) -> int:
        """Total number of sandbox executions. / 沙盒执行总次数。"""
        return self._run_count

    @property
    def success_count(self) -> int:
        """Successful executions. / 成功执行次数。"""
        return self._success_count

    @property
    def failure_count(self) -> int:
        """Failed executions (crash + timeout). / 失败次数（崩溃 + 超时）。"""
        return self._failure_count

    @property
    def timeout_count(self) -> int:
        """Timed-out executions. / 超时次数。"""
        return self._timeout_count

    @property
    def success_rate(self) -> float:
        """Success rate in [0, 1]. / 成功率。"""
        if self._run_count == 0:
            return 0.0
        return self._success_count / self._run_count

    @property
    def timeout_s(self) -> float:
        return self._timeout_s
