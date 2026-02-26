"""
GlobalWorkspace — the system's "broadcast station".
全局工作空间 —— 系统的"广播站"。

Implements a hand-written publish-subscribe event bus (no third-party
message queue). When a qualia signal is produced, the workspace
broadcasts it to every subscribed module. If the signal's intensity
exceeds the interrupt threshold, an emergency interrupt is raised so
all modules can drop current work and respond to the crisis.

实现纯手写的发布-订阅事件总线（不依赖第三方消息队列）。
当情绪信号产生时，广播站将其广播给每个已订阅的模块。
如果信号强度超过中断阈值，将触发紧急中断，
让所有模块放下手头的事来响应危机。

This module directly implements:
本模块直接实现了：
    - Paper Axiom A3 (Global Broadcast): Q(t) is immediately readable
      by all submodules and can interrupt current computation.
      论文公理 A3（全局广播性）：Q(t) 对所有子模块即刻可读，
      且可中断当前计算流。
    - Paper Theorem 4.4 (Unity of Consciousness): global broadcast of
      Q(t) creates a unified experiential field.
      论文定理 4.4（意识统一性）：Q(t) 的全局广播形成统一体验场。

Corresponds to IMPLEMENTATION_PLAN §3.5 and Phase I Step 6.
对应实施计划第 3.5 节和 Phase I 第 6 步。
"""

from dataclasses import dataclass
from typing import Callable, Optional


# ======================================================================
# Broadcast payload — what gets sent to every subscriber
# 广播载荷 —— 发送给每个订阅者的内容
# ======================================================================

@dataclass(frozen=True)
class BroadcastSignal:
    """
    The message delivered to every subscriber when qualia is broadcast.
    当情绪被广播时，发送给每个订阅者的消息。

    Attributes / 属性
    -----------------
    tick : int
        The heartbeat number when this signal was generated.
        生成此信号时的心跳编号。
    qualia_value : float
        The qualia value Q(t) — positive = good, negative = bad.
        情绪值 Q(t)——正 = 好，负 = 坏。
    qualia_intensity : float
        Absolute value of qualia (always >= 0).
        情绪强度（绝对值，始终 >= 0）。
    is_interrupt : bool
        True if intensity exceeded the interrupt threshold.
        如果强度超过中断阈值则为 True。
    """
    tick: int
    qualia_value: float
    qualia_intensity: float
    is_interrupt: bool


# Type alias for subscriber callbacks.
# 订阅者回调的类型别名。
SubscriberCallback = Callable[[BroadcastSignal], None]


# ======================================================================
# GlobalWorkspace — the broadcast station
# 全局工作空间 —— 广播站
# ======================================================================

class GlobalWorkspace:
    """
    Publish-subscribe event bus for qualia signals.
    情绪信号的发布-订阅事件总线。

    Modules subscribe at startup; the qualia generator calls broadcast()
    each tick. If the signal is strong enough, an interrupt flag is set
    so the main loop can switch to emergency behaviour.
    各模块在启动时订阅；情绪发生器在每个心跳调用 broadcast()。
    如果信号足够强，会设置中断标志，以便主循环切换到紧急行为。

    Parameters / 参数
    ----------
    interrupt_threshold : float
        Qualia intensity above which an interrupt is triggered (default 0.7).
        触发中断的情绪强度阈值（默认 0.7）。
    """

    def __init__(self, interrupt_threshold: float = 0.7):
        self._interrupt_threshold = interrupt_threshold
        self._subscribers: dict[str, SubscriberCallback] = {}
        self._interrupt_flag: bool = False
        self._last_signal: Optional[BroadcastSignal] = None

    # ------------------------------------------------------------------
    # Subscribe / unsubscribe — module registration
    # 订阅 / 取消订阅 —— 模块注册
    # ------------------------------------------------------------------

    def subscribe(self, name: str, callback: SubscriberCallback) -> None:
        """
        Register a module to receive broadcast signals.
        注册一个模块以接收广播信号。

        Parameters / 参数
        ----------
        name : str
            Unique identifier for the subscriber (e.g. "memory", "action").
            订阅者的唯一标识符（如 "memory"、"action"）。
        callback : SubscriberCallback
            Function to call with BroadcastSignal when qualia is broadcast.
            广播情绪时调用的函数，参数为 BroadcastSignal。
        """
        self._subscribers[name] = callback

    def unsubscribe(self, name: str) -> None:
        """
        Remove a subscriber by name.
        按名称移除一个订阅者。
        """
        self._subscribers.pop(name, None)

    @property
    def subscriber_count(self) -> int:
        """Number of currently registered subscribers. / 当前注册的订阅者数量。"""
        return len(self._subscribers)

    @property
    def subscriber_names(self) -> list[str]:
        """Names of all current subscribers. / 所有当前订阅者的名称。"""
        return list(self._subscribers.keys())

    # ------------------------------------------------------------------
    # Broadcast — the core function
    # 广播 —— 核心功能
    # ------------------------------------------------------------------

    def broadcast(self, tick: int, qualia_value: float) -> BroadcastSignal:
        """
        Broadcast a qualia signal to all subscribers.
        将情绪信号广播给所有订阅者。

        1. Compute intensity and determine if this is an interrupt.
        2. Build the BroadcastSignal.
        3. Set the interrupt flag if applicable.
        4. Deliver the signal to every subscriber's callback.

        1. 计算强度，判断是否为中断。
        2. 构建 BroadcastSignal。
        3. 如适用，设置中断标志。
        4. 将信号传递给每个订阅者的回调。

        Parameters / 参数
        ----------
        tick : int
            Current heartbeat number. / 当前心跳编号。
        qualia_value : float
            The qualia value Q(t). / 情绪值 Q(t)。

        Returns / 返回
        -------
        BroadcastSignal
            The signal that was broadcast. / 被广播的信号。
        """
        intensity = abs(qualia_value)
        is_interrupt = intensity >= self._interrupt_threshold

        signal = BroadcastSignal(
            tick=tick,
            qualia_value=qualia_value,
            qualia_intensity=intensity,
            is_interrupt=is_interrupt,
        )

        # Set interrupt flag — main loop reads and clears this.
        # 设置中断标志——主循环读取并清除它。
        if is_interrupt:
            self._interrupt_flag = True

        self._last_signal = signal

        # Deliver to all subscribers.
        # 传递给所有订阅者。
        for callback in self._subscribers.values():
            callback(signal)

        return signal

    # ------------------------------------------------------------------
    # Interrupt flag — read and clear
    # 中断标志 —— 读取和清除
    # ------------------------------------------------------------------

    @property
    def interrupt_flag(self) -> bool:
        """
        True if the most recent broadcast triggered an interrupt.
        如果最近一次广播触发了中断则为 True。

        The main loop should read this, act accordingly, then call
        clear_interrupt().
        主循环应读取此标志，做出相应行动，然后调用 clear_interrupt()。
        """
        return self._interrupt_flag

    def clear_interrupt(self) -> None:
        """
        Reset the interrupt flag after the main loop has handled it.
        主循环处理完中断后，重置中断标志。
        """
        self._interrupt_flag = False

    # ------------------------------------------------------------------
    # Last signal — for modules that need to poll
    # 最近信号 —— 供需要轮询的模块使用
    # ------------------------------------------------------------------

    @property
    def last_signal(self) -> Optional[BroadcastSignal]:
        """
        The most recently broadcast signal (None before first broadcast).
        最近一次广播的信号（首次广播前为 None）。

        Useful for modules that poll rather than subscribe.
        对于采用轮询而非订阅的模块很有用。
        """
        return self._last_signal

    @property
    def interrupt_threshold(self) -> float:
        """The interrupt threshold value. / 中断阈值。"""
        return self._interrupt_threshold
