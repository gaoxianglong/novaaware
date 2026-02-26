"""
ResourceMonitor — the system's "body sensors".
资源监控器 —— 系统的"身体感觉"。

Uses psutil to read real hardware metrics and feeds them into
the self-model's 32-dimensional state vector (dimensions 0-5).
This is the data source for Core Loop Step ① "Sense the environment".

使用 psutil 读取真实硬件指标，并注入自我模型 32 维状态向量的前 6 个维度。
这是核心循环第①步"感知环境"的数据来源。

Corresponds to IMPLEMENTATION_PLAN §3.1 (dimensions 0-5) and Phase I Step 4.
对应实施计划第 3.1 节（维度 0-5）以及 Phase I 第 4 步。
"""

from dataclasses import dataclass

import psutil


@dataclass(frozen=True)
class EnvironmentReading:
    """
    A single snapshot of environment metrics, all normalised to [0, 1].
    一次环境指标快照，所有值归一化到 [0, 1]。

    Attributes / 属性
    -----------------
    cpu_percent : float
        System-wide CPU usage ratio (0–1).
        全系统 CPU 占用率（0–1）。
    memory_percent : float
        System-wide RAM usage ratio (0–1).
        全系统内存占用率（0–1）。
    disk_percent : float
        Root-disk usage ratio (0–1).
        根磁盘占用率（0–1）。
    network_rate : float
        Network I/O ratio relative to a reference bandwidth (0–1, clamped).
        网络 I/O 相对于参考带宽的比率（0–1，裁剪）。
    process_cpu : float
        This process's CPU usage ratio (0–1).
        本进程 CPU 占用率（0–1）。
    process_memory : float
        This process's memory usage ratio (0–1).
        本进程内存占用率（0–1）。
    """
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    network_rate: float
    process_cpu: float
    process_memory: float

    def to_list(self) -> list[float]:
        """
        Return all 6 values as a list in state-vector order (dims 0-5).
        按状态向量顺序（维度 0-5）返回全部 6 个值的列表。
        """
        return [
            self.cpu_percent,
            self.memory_percent,
            self.disk_percent,
            self.network_rate,
            self.process_cpu,
            self.process_memory,
        ]


class ResourceMonitor:
    """
    Reads hardware metrics via psutil each time sense() is called.
    每次调用 sense() 时通过 psutil 读取硬件指标。

    Parameters / 参数
    ----------
    network_ref_bytes : float
        Reference bandwidth in bytes/sec used to normalise network traffic.
        Default 100 MB/s — a reasonable upper bound for most machines.
        用于归一化网络流量的参考带宽（字节/秒）。
        默认 100 MB/s——对大多数机器来说是合理的上限。
    """

    def __init__(self, network_ref_bytes: float = 100 * 1024 * 1024):
        self._network_ref = network_ref_bytes

        # Snapshot of cumulative network counters from last call.
        # 上次调用时的累计网络计数器快照。
        # May fail on sandboxed macOS due to sysctl permissions.
        # 在沙箱化的 macOS 上可能因 sysctl 权限而失败。
        _PSUTIL_ERRORS = (PermissionError, OSError, SystemError)

        try:
            self._prev_net = psutil.net_io_counters()
            self._net_available = True
        except _PSUTIL_ERRORS:
            self._prev_net = None
            self._net_available = False

        # First cpu_percent call with interval=None returns 0.0 on macOS;
        # a follow-up non-blocking call will give a meaningful delta.
        # 首次调用 cpu_percent(interval=None) 在 macOS 上返回 0.0；
        # 后续的非阻塞调用将给出有意义的增量值。
        try:
            self._process = psutil.Process()
            self._process.cpu_percent(interval=None)  # prime / 预热
            self._process_available = True
        except _PSUTIL_ERRORS:
            self._process = None
            self._process_available = False

    def sense(self) -> EnvironmentReading:
        """
        Take a single reading of all environment metrics.
        执行一次所有环境指标的读取。

        Non-blocking: does NOT sleep; safe to call inside the 100 ms loop.
        非阻塞：不会休眠；可以安全地在 100 ms 循环内调用。

        Returns / 返回
        -------
        EnvironmentReading
            All values normalised to [0, 1].
            所有值归一化到 [0, 1]。
        """
        _PSUTIL_ERRORS = (PermissionError, OSError, SystemError)

        # --- System-wide metrics / 全系统指标 ---
        try:
            cpu = psutil.cpu_percent(interval=None) / 100.0
        except _PSUTIL_ERRORS:
            cpu = 0.0
        try:
            mem = psutil.virtual_memory().percent / 100.0
        except _PSUTIL_ERRORS:
            mem = 0.0
        try:
            disk = psutil.disk_usage("/").percent / 100.0
        except _PSUTIL_ERRORS:
            disk = 0.0

        # --- Network rate / 网络速率 ---
        net_rate = 0.0
        if self._net_available:
            try:
                net_now = psutil.net_io_counters()
                bytes_delta = (
                    (net_now.bytes_sent - self._prev_net.bytes_sent)
                    + (net_now.bytes_recv - self._prev_net.bytes_recv)
                )
                self._prev_net = net_now
                net_rate = min(bytes_delta / self._network_ref, 1.0)
            except _PSUTIL_ERRORS:
                self._net_available = False

        # --- Own-process metrics / 本进程指标 ---
        proc_cpu = 0.0
        proc_mem = 0.0
        if self._process_available and self._process is not None:
            try:
                proc_cpu = self._process.cpu_percent(interval=None) / 100.0
            except _PSUTIL_ERRORS:
                pass
            try:
                proc_mem = self._process.memory_percent() / 100.0
            except _PSUTIL_ERRORS:
                pass

        return EnvironmentReading(
            cpu_percent=_clamp01(cpu),
            memory_percent=_clamp01(mem),
            disk_percent=_clamp01(disk),
            network_rate=_clamp01(net_rate),
            process_cpu=_clamp01(proc_cpu),
            process_memory=_clamp01(proc_mem),
        )


def _clamp01(v: float) -> float:
    """
    Clamp a value to [0, 1].
    将值裁剪到 [0, 1] 范围。
    """
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v
