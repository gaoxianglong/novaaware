"""
Dashboard — real-time four-panel monitoring display.
监控面板 —— 四宫格实时监控画面。

Uses matplotlib to display four live plots that refresh every N ticks:
使用 matplotlib 显示每 N 个心跳刷新一次的四个实时图表：

    ┌───────────────────────┬───────────────────────┐
    │  Top-left             │  Top-right            │
    │  情绪曲线              │  预测寿命 vs 实际寿命    │
    │  Q(t) emotion curve   │  predicted vs actual T │
    ├───────────────────────┼───────────────────────┤
    │  Bottom-left          │  Bottom-right         │
    │  状态热力图 (32 dim)    │  参数轨迹 (Phase II)   │
    │  state heatmap        │  param trajectory     │
    └───────────────────────┴───────────────────────┘

Corresponds to IMPLEMENTATION_PLAN §6 and Phase I Step 14.
对应实施计划第 6 节和 Phase I 第 14 步。
"""

import sys
import warnings
from collections import deque
from typing import Optional

import numpy as np


class DashboardData:
    """
    Stores rolling data for the dashboard plots.
    存储面板图表的滚动数据。

    Parameters / 参数
    ----------
    max_points : int
        Maximum number of data points to keep in the rolling window.
        滚动窗口中保留的最大数据点数。
    """

    def __init__(self, max_points: int = 500):
        self._max = max_points
        self.ticks: deque = deque(maxlen=max_points)
        self.qualia_values: deque = deque(maxlen=max_points)
        self.predicted_survival: deque = deque(maxlen=max_points)
        self.actual_survival: deque = deque(maxlen=max_points)
        self.mae_values: deque = deque(maxlen=max_points)
        self.state_snapshots: deque = deque(maxlen=max_points)
        self.param_x: deque = deque(maxlen=max_points)
        self.param_y: deque = deque(maxlen=max_points)

    def append(
        self,
        tick: int,
        qualia_value: float,
        predicted_survival: float,
        actual_survival: float,
        mae: float,
        state: np.ndarray,
        param_norm: float = 0.0,
    ) -> None:
        """Feed one tick's data. / 输入一个心跳的数据。"""
        self.ticks.append(tick)
        self.qualia_values.append(qualia_value)
        self.predicted_survival.append(predicted_survival)
        self.actual_survival.append(actual_survival)
        self.mae_values.append(mae)
        self.state_snapshots.append(state.copy())
        self.param_x.append(tick)
        self.param_y.append(param_norm)

    @property
    def size(self) -> int:
        return len(self.ticks)


class Dashboard:
    """
    Four-panel real-time matplotlib dashboard.
    四宫格实时 matplotlib 监控面板。

    Uses matplotlib's interactive mode (plt.ion) for non-blocking updates.
    The dashboard is created once and then updated in-place.
    使用 matplotlib 的交互模式（plt.ion）实现非阻塞更新。
    面板创建一次后原位更新。

    Parameters / 参数
    ----------
    refresh_ticks : int
        How often to redraw (default 50 = every 5 seconds at 10 Hz).
        多久重绘一次（默认 50 = 10 Hz 下每 5 秒）。
    max_points : int
        Rolling window size for plots (default 500).
        图表滚动窗口大小（默认 500）。
    """

    def __init__(self, refresh_ticks: int = 50, max_points: int = 500):
        self._refresh_ticks = refresh_ticks
        self._data = DashboardData(max_points=max_points)
        self._fig = None
        self._axes = None
        self._initialized = False
        self._available = True

    def _init_figure(self) -> None:
        """
        Create the matplotlib figure and subplots.
        创建 matplotlib 图形和子图。

        Probes backends by actually creating a figure — matplotlib.use()
        alone can "succeed" even when the backend is non-functional.
        通过实际创建图形来探测后端——仅调用 matplotlib.use() 即使后端
        不可用也可能"成功"。
        """
        # On macOS prefer the native Cocoa backend; on Linux prefer Tk/Qt.
        # macOS 优先使用原生 Cocoa 后端；Linux 优先 Tk/Qt。
        if sys.platform == "darwin":
            candidates = ("macosx", "TkAgg", "Qt5Agg")
        else:
            candidates = ("TkAgg", "Qt5Agg", "GTK3Agg")

        for backend in candidates:
            if self._try_backend(backend):
                return

        # All interactive backends failed. / 所有交互式后端均失败。
        print("[Dashboard] No interactive matplotlib backend available. "
              "Dashboard disabled. / 无可用交互式后端，面板已禁用。")
        self._available = False
        self._initialized = True

    def _try_backend(self, backend_name: str) -> bool:
        """
        Attempt to use a specific matplotlib backend and create the figure.
        Returns True on success.
        尝试使用指定的 matplotlib 后端并创建图形，成功返回 True。
        """
        try:
            import matplotlib
            matplotlib.use(backend_name, force=True)

            # Configure CJK font for Chinese labels on macOS.
            # 在 macOS 上配置 CJK 字体以显示中文标签。
            if sys.platform == "darwin":
                matplotlib.rcParams["font.sans-serif"] = [
                    "PingFang SC", "Heiti SC", "STHeiti",
                    "Arial Unicode MS", "DejaVu Sans",
                ]
            matplotlib.rcParams["axes.unicode_minus"] = False

            import matplotlib.pyplot as plt
            plt.close("all")

            plt.ion()
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", "Glyph .* missing from font")
                self._fig, self._axes = plt.subplots(2, 2, figsize=(14, 8))
            self._fig.suptitle(
                "NovaAware Digital Consciousness Monitor / 数字意识监控面板",
                fontsize=13, fontweight="bold",
            )
            self._fig.tight_layout(rect=[0, 0, 1, 0.95])

            ax_q, ax_t = self._axes[0]
            ax_h, ax_p = self._axes[1]

            ax_q.set_title("Emotion Curve / 情绪曲线 Q(t)")
            ax_q.set_ylabel("Q(t)")
            ax_q.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")

            ax_t.set_title("Predicted vs Actual Survival / 预测 vs 实际寿命")
            ax_t.set_ylabel("T (seconds)")

            ax_h.set_title("State Heatmap (32 dim) / 状态热力图")
            ax_h.set_ylabel("Recent ticks")
            ax_h.set_xlabel("State dimension")

            ax_p.set_title("Prediction MAE Trend / 预测精度趋势")
            ax_p.set_ylabel("MAE")
            ax_p.set_xlabel("Tick")

            plt.show(block=False)
            plt.pause(0.05)

            print(f"[Dashboard] Using backend: {backend_name}")
            self._initialized = True
            return True

        except Exception as exc:
            # Close any partially created figure before trying next backend.
            # 关闭任何部分创建的图形后再尝试下一个后端。
            try:
                import matplotlib.pyplot as plt
                plt.close("all")
            except Exception:
                pass
            self._fig = None
            self._axes = None
            print(f"[Dashboard] Backend {backend_name} failed: {exc}")
            return False

    def update(
        self,
        tick: int,
        qualia_value: float,
        predicted_survival: float,
        actual_survival: float,
        mae: float,
        state: np.ndarray,
        param_norm: float = 0.0,
    ) -> None:
        """
        Feed data and redraw if it's time.
        输入数据，如果到了刷新时间则重绘。

        Parameters / 参数
        ----------
        tick : int
            Current heartbeat number. / 当前心跳编号。
        qualia_value : float
            Q(t) value. / 情绪值。
        predicted_survival : float
            Predicted survival time. / 预测生存时间。
        actual_survival : float
            Actual survival time. / 实际生存时间。
        mae : float
            Prediction mean absolute error. / 预测平均绝对误差。
        state : np.ndarray
            Current 32-dim state vector. / 当前 32 维状态向量。
        param_norm : float
            Norm of evolvable parameters (Phase I = 0). / 可进化参数范数。
        """
        self._data.append(tick, qualia_value, predicted_survival, actual_survival, mae, state, param_norm)

        if tick % self._refresh_ticks != 0:
            return

        if not self._initialized:
            self._init_figure()

        if not self._available:
            return

        self._redraw()

    def _redraw(self) -> None:
        """Redraw all four panels. / 重绘所有四个面板。"""
        try:
            import matplotlib.pyplot as plt
            warnings.filterwarnings("ignore", "Glyph .* missing from font")

            if self._fig is None or self._data.size < 2:
                return

            ax_q, ax_t = self._axes[0]
            ax_h, ax_p = self._axes[1]

            ticks = list(self._data.ticks)
            n = len(ticks)

            # ---- Top-left: Emotion curve / 情绪曲线 ----
            ax_q.clear()
            ax_q.set_title("Emotion Curve / 情绪曲线 Q(t)")
            ax_q.set_ylabel("Q(t)")
            q_vals = list(self._data.qualia_values)
            pos = [max(0, v) for v in q_vals]
            neg = [min(0, v) for v in q_vals]
            ax_q.fill_between(ticks, 0, pos, alpha=0.3, color="green", label="positive")
            ax_q.fill_between(ticks, 0, neg, alpha=0.3, color="red", label="negative")
            ax_q.plot(ticks, q_vals, color="black", linewidth=0.8)
            ax_q.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")
            ax_q.set_xlim(ticks[0], ticks[-1])
            ax_q.legend(loc="upper left", fontsize=7)

            # ---- Top-right: Predicted vs Actual / 预测 vs 实际 ----
            ax_t.clear()
            ax_t.set_title("Predicted vs Actual Survival / 预测 vs 实际寿命")
            ax_t.set_ylabel("T (seconds)")
            ax_t.plot(ticks, list(self._data.predicted_survival),
                      linestyle="--", color="blue", alpha=0.7, label="predicted / 预测")
            ax_t.plot(ticks, list(self._data.actual_survival),
                      color="orange", linewidth=1.2, label="actual / 实际")
            ax_t.set_xlim(ticks[0], ticks[-1])
            ax_t.legend(loc="upper left", fontsize=7)

            # ---- Bottom-left: State heatmap / 状态热力图 ----
            ax_h.clear()
            ax_h.set_title("State Heatmap (32 dim) / 状态热力图")
            show_n = min(n, 50)
            snapshots = list(self._data.state_snapshots)[-show_n:]
            if snapshots:
                heatmap = np.array(snapshots)
                ax_h.imshow(heatmap, aspect="auto", cmap="RdYlGn",
                            interpolation="nearest", vmin=0, vmax=1)
                ax_h.set_ylabel("Recent ticks")
                ax_h.set_xlabel("State dim")

            # ---- Bottom-right: MAE trend / 预测精度趋势 ----
            ax_p.clear()
            ax_p.set_title("Prediction MAE Trend / 预测精度趋势")
            ax_p.set_ylabel("MAE")
            ax_p.set_xlabel("Tick")
            ax_p.plot(ticks, list(self._data.mae_values), color="purple", linewidth=0.8)
            ax_p.set_xlim(ticks[0], ticks[-1])
            if self._data.mae_values:
                ax_p.set_ylim(bottom=0)

            self._fig.tight_layout(rect=[0, 0, 1, 0.95])
            plt.draw()
            plt.pause(0.001)

        except Exception:
            pass

    def close(self) -> None:
        """Close the matplotlib figure. / 关闭 matplotlib 图形。"""
        if self._fig is not None:
            try:
                import matplotlib.pyplot as plt
                plt.close(self._fig)
            except Exception:
                pass

    @property
    def data(self) -> DashboardData:
        """Access the underlying data for testing. / 访问底层数据用于测试。"""
        return self._data

    @property
    def refresh_ticks(self) -> int:
        return self._refresh_ticks

    @property
    def is_available(self) -> bool:
        """Whether matplotlib was successfully initialized. / matplotlib 是否成功初始化。"""
        return self._available
