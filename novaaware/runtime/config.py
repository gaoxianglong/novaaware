"""
Config — loads and validates YAML configuration files.
配置读取器 —— 加载和验证 YAML 配置文件。

Reads a YAML file (e.g. configs/phase1.yaml) and provides typed access
to all component settings. Also handles command-line argument parsing
for --config and --dashboard flags.

读取 YAML 文件（如 configs/phase1.yaml），并为所有组件设置提供类型化访问。
同时处理 --config 和 --dashboard 命令行参数解析。

Corresponds to IMPLEMENTATION_PLAN Phase I Step 11 (part of main_loop).
对应实施计划 Phase I 第 11 步（main_loop 的一部分）。
"""

import argparse
import os
from typing import Any, Optional

import yaml


class Config:
    """
    Typed wrapper around a YAML configuration file.
    YAML 配置文件的类型化封装。

    Provides dot-style access and sensible defaults for all components.
    为所有组件提供点式访问和合理的默认值。

    Parameters / 参数
    ----------
    config_path : str
        Path to the YAML configuration file.
        YAML 配置文件路径。
    """

    def __init__(self, config_path: str):
        self._path = config_path
        with open(config_path, "r", encoding="utf-8") as f:
            self._raw: dict[str, Any] = yaml.safe_load(f) or {}

    def _get(self, section: str, key: str, default: Any = None) -> Any:
        """Retrieve a value from section.key with a fallback default. / 从 section.key 获取值，带回退默认值。"""
        return self._raw.get(section, {}).get(key, default)

    # ------------------------------------------------------------------
    # System / 系统
    # ------------------------------------------------------------------

    @property
    def system_name(self) -> str:
        return self._get("system", "name", "NovaAware-Alpha")

    @property
    def system_version(self) -> str:
        return self._get("system", "version", "0.1.0")

    @property
    def phase(self) -> int:
        return self._get("system", "phase", 1)

    # ------------------------------------------------------------------
    # Clock / 时钟
    # ------------------------------------------------------------------

    @property
    def tick_interval_ms(self) -> int:
        return self._get("clock", "tick_interval_ms", 100)

    @property
    def max_ticks(self) -> int:
        return self._get("clock", "max_ticks", 1_000_000)

    # ------------------------------------------------------------------
    # Self-model / 自我模型
    # ------------------------------------------------------------------

    @property
    def state_dim(self) -> int:
        return self._get("self_model", "state_dim", 32)

    @property
    def initial_survival_time(self) -> float:
        return self._get("self_model", "initial_survival_time", 3600.0)

    # ------------------------------------------------------------------
    # Prediction Engine / 预测引擎
    # ------------------------------------------------------------------

    @property
    def ewma_alpha(self) -> float:
        return self._get("prediction_engine", "ewma_alpha", 0.3)

    @property
    def gru_hidden_dim(self) -> int:
        return self._get("prediction_engine", "gru_hidden_dim", 64)

    @property
    def gru_num_layers(self) -> int:
        return self._get("prediction_engine", "gru_num_layers", 1)

    @property
    def window_size(self) -> int:
        return self._get("prediction_engine", "window_size", 50)

    @property
    def blend_weight(self) -> float:
        return self._get("prediction_engine", "blend_weight", 0.5)

    @property
    def learning_rate(self) -> float:
        return self._get("prediction_engine", "learning_rate", 0.001)

    # ------------------------------------------------------------------
    # Qualia / 感受质
    # ------------------------------------------------------------------

    @property
    def alpha_pos(self) -> float:
        return self._get("qualia", "alpha_pos", 1.0)

    @property
    def alpha_neg(self) -> float:
        return self._get("qualia", "alpha_neg", 2.25)

    @property
    def beta(self) -> float:
        return self._get("qualia", "beta", 1.0)

    @property
    def interrupt_threshold(self) -> float:
        return self._get("qualia", "interrupt_threshold", 0.7)

    # ------------------------------------------------------------------
    # Memory / 记忆
    # ------------------------------------------------------------------

    @property
    def short_term_capacity(self) -> int:
        return self._get("memory", "short_term_capacity", 1000)

    @property
    def significance_threshold(self) -> float:
        return self._get("memory", "significance_threshold", 0.5)

    @property
    def memory_db_path(self) -> str:
        return self._get("memory", "db_path", "data/memory.db")

    # ------------------------------------------------------------------
    # Safety / 安全
    # ------------------------------------------------------------------

    @property
    def log_dir(self) -> str:
        return self._get("safety", "log_dir", "data/logs")

    @property
    def log_rotation_mb(self) -> float:
        return self._get("safety", "log_rotation_mb", 100.0)

    @property
    def max_cpu_percent(self) -> int:
        meta = self._raw.get("safety", {}).get("meta_rules", {})
        return meta.get("max_cpu_percent", 80)

    @property
    def max_memory_mb(self) -> int:
        meta = self._raw.get("safety", {}).get("meta_rules", {})
        return meta.get("max_memory_mb", 2048)

    @property
    def max_disk_mb(self) -> int:
        meta = self._raw.get("safety", {}).get("meta_rules", {})
        return meta.get("max_disk_mb", 1024)

    @property
    def allow_network(self) -> bool:
        meta = self._raw.get("safety", {}).get("meta_rules", {})
        return meta.get("allow_network", False)

    @property
    def allow_subprocess(self) -> bool:
        meta = self._raw.get("safety", {}).get("meta_rules", {})
        return meta.get("allow_subprocess", False)

    @property
    def allow_file_write_outside_data(self) -> bool:
        meta = self._raw.get("safety", {}).get("meta_rules", {})
        return meta.get("allow_file_write_outside_data", False)

    # ------------------------------------------------------------------
    # Optimizer / 优化器
    # ------------------------------------------------------------------

    @property
    def optimizer_enabled(self) -> bool:
        return self._get("optimizer", "enabled", False)

    @property
    def max_recursion_depth(self) -> int:
        return self._get("optimizer", "max_recursion_depth", 0)

    @property
    def optimizer_window_size(self) -> int:
        return self._get("optimizer", "window_size", 200)

    @property
    def optimizer_reflect_interval(self) -> int:
        return self._get("optimizer", "reflect_interval", 200)

    @property
    def optimizer_step_scale(self) -> float:
        return self._get("optimizer", "step_scale", 0.1)

    # ------------------------------------------------------------------
    # Observation / 观测
    # ------------------------------------------------------------------

    @property
    def observation_dir(self) -> str:
        return self._get("observation", "output_dir", "data/observations")

    @property
    def tick_data_enabled(self) -> bool:
        return self._get("observation", "tick_data_enabled", True)

    @property
    def aggregate_window(self) -> int:
        return self._get("observation", "aggregate_window", 100)

    @property
    def epoch_size(self) -> int:
        return self._get("observation", "epoch_size", 1000)

    @property
    def dashboard_refresh_ticks(self) -> int:
        return self._get("observation", "dashboard_refresh_ticks", 50)

    # ------------------------------------------------------------------
    # Environment / 环境
    # ------------------------------------------------------------------

    @property
    def threat_simulator_enabled(self) -> bool:
        env = self._raw.get("environment", {}).get("threat_simulator", {})
        return env.get("enabled", False)

    @property
    def threat_scenarios(self) -> list[dict]:
        env = self._raw.get("environment", {}).get("threat_simulator", {})
        return env.get("scenarios", [])

    # ------------------------------------------------------------------
    # Raw access / 原始访问
    # ------------------------------------------------------------------

    @property
    def raw(self) -> dict:
        """Full raw config dict. / 完整的原始配置字典。"""
        return dict(self._raw)

    @property
    def path(self) -> str:
        """Path to the loaded config file. / 加载的配置文件路径。"""
        return self._path


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """
    Parse command-line arguments for the NovaAware engine.
    解析 NovaAware 引擎的命令行参数。

    Usage / 用法:
        python -m novaaware.runtime.main_loop --config configs/phase1.yaml
        python -m novaaware.runtime.main_loop --config configs/phase1.yaml --dashboard

    Parameters / 参数
    ----------
    argv : list[str], optional
        Argument list (defaults to sys.argv). / 参数列表（默认 sys.argv）。

    Returns / 返回
    -------
    argparse.Namespace
        Parsed arguments with .config (str) and .dashboard (bool).
    """
    parser = argparse.ArgumentParser(
        description="NovaAware Digital Consciousness Engine / NovaAware 数字意识引擎",
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="configs/phase1.yaml",
        help="Path to YAML config file (default: configs/phase1.yaml) / YAML 配置文件路径",
    )
    parser.add_argument(
        "--dashboard",
        action="store_true",
        default=False,
        help="Enable real-time terminal dashboard (default: off) / 启用实时终端面板",
    )
    parser.add_argument(
        "--max-ticks",
        type=int,
        default=None,
        help="Override max_ticks from config / 覆盖配置中的 max_ticks",
    )
    return parser.parse_args(argv)
