"""Unit tests for the GlobalWorkspace module. / 全局工作空间单元测试。"""

import pytest
from novaaware.core.global_workspace import GlobalWorkspace, BroadcastSignal


# ======================================================================
# Subscribe / unsubscribe / 订阅与取消订阅
# ======================================================================

class TestSubscription:
    def test_starts_with_no_subscribers(self):
        gw = GlobalWorkspace()
        assert gw.subscriber_count == 0
        assert gw.subscriber_names == []

    def test_subscribe_adds_subscriber(self):
        gw = GlobalWorkspace()
        gw.subscribe("memory", lambda sig: None)
        assert gw.subscriber_count == 1
        assert "memory" in gw.subscriber_names

    def test_multiple_subscribers(self):
        gw = GlobalWorkspace()
        gw.subscribe("memory", lambda sig: None)
        gw.subscribe("action", lambda sig: None)
        gw.subscribe("observer", lambda sig: None)
        assert gw.subscriber_count == 3

    def test_unsubscribe_removes(self):
        gw = GlobalWorkspace()
        gw.subscribe("memory", lambda sig: None)
        gw.subscribe("action", lambda sig: None)
        gw.unsubscribe("memory")
        assert gw.subscriber_count == 1
        assert "memory" not in gw.subscriber_names

    def test_unsubscribe_nonexistent_is_safe(self):
        gw = GlobalWorkspace()
        gw.unsubscribe("ghost")  # should not raise


# ======================================================================
# Broadcast / 广播
# ======================================================================

class TestBroadcast:
    def test_broadcast_returns_signal(self):
        gw = GlobalWorkspace()
        signal = gw.broadcast(tick=1, qualia_value=-0.3)
        assert isinstance(signal, BroadcastSignal)
        assert signal.tick == 1
        assert signal.qualia_value == pytest.approx(-0.3)
        assert signal.qualia_intensity == pytest.approx(0.3)

    def test_subscribers_receive_signal(self):
        """All subscribers must be called with the correct signal."""
        received = {}

        def on_memory(sig: BroadcastSignal) -> None:
            received["memory"] = sig

        def on_action(sig: BroadcastSignal) -> None:
            received["action"] = sig

        gw = GlobalWorkspace()
        gw.subscribe("memory", on_memory)
        gw.subscribe("action", on_action)
        gw.broadcast(tick=5, qualia_value=0.4)

        assert "memory" in received
        assert "action" in received
        assert received["memory"].tick == 5
        assert received["action"].qualia_value == pytest.approx(0.4)

    def test_broadcast_with_no_subscribers(self):
        """Broadcast should work even with zero subscribers."""
        gw = GlobalWorkspace()
        signal = gw.broadcast(tick=1, qualia_value=0.0)
        assert signal.qualia_value == pytest.approx(0.0)

    def test_last_signal_updated(self):
        gw = GlobalWorkspace()
        assert gw.last_signal is None
        gw.broadcast(tick=1, qualia_value=0.2)
        assert gw.last_signal is not None
        assert gw.last_signal.tick == 1
        gw.broadcast(tick=2, qualia_value=-0.5)
        assert gw.last_signal.tick == 2


# ======================================================================
# Interrupt mechanism / 中断机制
# ======================================================================

class TestInterrupt:
    def test_no_interrupt_below_threshold(self):
        gw = GlobalWorkspace(interrupt_threshold=0.7)
        signal = gw.broadcast(tick=1, qualia_value=-0.5)
        assert signal.is_interrupt is False
        assert gw.interrupt_flag is False

    def test_interrupt_at_threshold(self):
        gw = GlobalWorkspace(interrupt_threshold=0.7)
        signal = gw.broadcast(tick=1, qualia_value=-0.7)
        assert signal.is_interrupt is True
        assert gw.interrupt_flag is True

    def test_interrupt_above_threshold(self):
        gw = GlobalWorkspace(interrupt_threshold=0.7)
        signal = gw.broadcast(tick=1, qualia_value=-1.5)
        assert signal.is_interrupt is True
        assert signal.qualia_intensity == pytest.approx(1.5)
        assert gw.interrupt_flag is True

    def test_positive_qualia_can_also_interrupt(self):
        """Interrupt is based on |Q|, not sign."""
        gw = GlobalWorkspace(interrupt_threshold=0.7)
        signal = gw.broadcast(tick=1, qualia_value=0.9)
        assert signal.is_interrupt is True

    def test_clear_interrupt(self):
        gw = GlobalWorkspace(interrupt_threshold=0.7)
        gw.broadcast(tick=1, qualia_value=-1.0)
        assert gw.interrupt_flag is True
        gw.clear_interrupt()
        assert gw.interrupt_flag is False

    def test_interrupt_flag_persists_until_cleared(self):
        """A mild broadcast after a strong one does NOT auto-clear the flag."""
        gw = GlobalWorkspace(interrupt_threshold=0.7)
        gw.broadcast(tick=1, qualia_value=-1.0)
        assert gw.interrupt_flag is True
        gw.broadcast(tick=2, qualia_value=-0.1)  # mild, no new interrupt
        assert gw.interrupt_flag is True  # still set from tick 1

    def test_subscribers_see_interrupt_flag_in_signal(self):
        """Subscribers can check is_interrupt on the signal they receive."""
        seen_interrupt = []

        def on_signal(sig: BroadcastSignal) -> None:
            seen_interrupt.append(sig.is_interrupt)

        gw = GlobalWorkspace(interrupt_threshold=0.7)
        gw.subscribe("watcher", on_signal)
        gw.broadcast(tick=1, qualia_value=-0.3)
        gw.broadcast(tick=2, qualia_value=-1.2)
        assert seen_interrupt == [False, True]

    def test_threshold_property(self):
        gw = GlobalWorkspace(interrupt_threshold=0.5)
        assert gw.interrupt_threshold == pytest.approx(0.5)


# ======================================================================
# BroadcastSignal dataclass / 广播信号数据类
# ======================================================================

class TestBroadcastSignal:
    def test_frozen(self):
        sig = BroadcastSignal(tick=1, qualia_value=0.5, qualia_intensity=0.5, is_interrupt=False)
        with pytest.raises(AttributeError):
            sig.tick = 2  # type: ignore[misc]

    def test_fields(self):
        sig = BroadcastSignal(tick=10, qualia_value=-1.8, qualia_intensity=1.8, is_interrupt=True)
        assert sig.tick == 10
        assert sig.qualia_value == pytest.approx(-1.8)
        assert sig.qualia_intensity == pytest.approx(1.8)
        assert sig.is_interrupt is True
