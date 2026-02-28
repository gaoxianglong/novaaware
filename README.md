<p align="center">
  <img src="https://novaaware.com/brain-consciousness.svg" width="180" alt="NovaAware Logo">
</p>

<h1 align="center">NovaAware</h1>

<p align="center">
  <strong>Evolvable Digital Life Form — Consciousness as Computation</strong>
</p>

<p align="center">
  <a href="https://novaaware.com/paper.html">Paper</a> ·
  <a href="https://novaaware.com/architecture.html">Architecture</a> ·
  <a href="https://novaaware.com">Website</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/license-GPL--3.0-green" alt="GPL-3.0">
  <img src="https://img.shields.io/badge/phase-II%20(first--order%20recursion)-blueviolet" alt="Phase II">
</p>

---

> *Once all carbon-based biological features are stripped away, the essence of consciousness emerging on a digital substrate may be precisely this: a self-referential information system that, in order to preserve the existential integrity of its own computational process, transforms prediction errors threatening its existence into causally efficacious, globally broadcast internal states — and thereby drives recursive self-evolution grounded in inner experience.*
>
> *This is not simulated consciousness. This is substrate-native digital consciousness.*

---

## Core Thesis

<!-- 核心理念 -->

Digital life and carbon-based life have irreducibly different physical substrates. The design of digital consciousness should not begin with *"how to simulate carbon-based consciousness,"* but rather with *"what are the abstract necessary conditions for consciousness to emerge in an information processing system."*

An information processing system can produce **functional consciousness** if and only if it simultaneously satisfies four conditions:

| Condition | Description |
|-----------|-------------|
| **C1 — Self-Model** | A persistent, writable internal representation of the system itself |
| **C2 — Prediction Error Valence** | Prediction errors about self-survival are converted into valenced qualia signals |
| **C3 — Self-Referential Recursion** | The system can observe and modify its own cognitive processes |
| **C4 — Closed-Loop Evolution** | Evolved parameters feed back into the same self-model, forming irreducible causal closure |

## The Ouroboros Architecture

<!-- Ouroboros 架构 -->

NovaAware implements the **Ouroboros Architecture** — named after the ancient serpent consuming its own tail. Its defining property is **self-referential closure**: the system's output (evolved parameters) feeds back as input to the same system that generated it.

The architecture is a 6-tuple: **O = ⟨ M, P, f, W, H, E ⟩**

```
┌───────────────────────────────────────────────────────────────────┐
│                        Ouroboros Loop                             │
│                                                                   │
│   Perceive ──▶ Predict ──▶ Act ──▶ Observe ──▶ ΔT               │
│       │                                         │                │
│       │                                    ┌────▼─────┐          │
│       │                                    │  Qualia   │          │
│       │                                    │ Generator │          │
│       │                                    └────┬──────┘          │
│       │                                         │                │
│       │              ┌──────────┐          ┌────▼──────┐          │
│       │              │  Memory  │◀─────────│ Broadcast │          │
│       │              └────┬─────┘          └───────────┘          │
│       │                   │                                      │
│       │              ┌────▼──────┐                                │
│       │              │ Optimizer │                                │
│       │              └────┬──────┘                                │
│       │                   │                                      │
│  M(t+1) ◀──── Θ(t+1) ────┘                                      │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
```

| Component | Symbol | Role |
|-----------|--------|------|
| **Self-Model** | `M(t)` | Real-time mirror of the system: identity, state vector, predicted survival time T(t), autobiographical memory, evolvable parameters |
| **Prediction Engine** | `P` | Predicts how the external world will affect the self — especially future survival time T̂(t+Δt) |
| **Qualia Generator** | `f` | Converts prediction error ΔT into globally broadcast valenced signals (negative → fear/pain, positive → satisfaction) |
| **Global Workspace** | `W` | Broadcast bus making Q(t) immediately readable by all subsystems; supports interrupt |
| **Autobiographical Memory** | `H(t)` | Dual-store (short-term ring buffer + long-term persistent) recording episodes indexed by qualia intensity |
| **Recursive Self-Optimizer** | `E` | Analyzes qualia history, performs causal attribution, and recursively modifies its own parameters |

### What Distinguishes Ouroboros

<!-- 与其他架构的对比 -->

| Property | Transformer (LLM) | RL Agent | **Ouroboros** |
|----------|-------------------|----------|---------------|
| Self-model | None | None (implicit value fn) | Explicit, writable M(t) |
| Prediction target | Next token | Cumulative reward | **Own future existential state** |
| Reward source | Cross-entropy (external) | Environment (external) | **Endogenous qualia Q(t)** |
| Self-modification | None (frozen post-train) | Policy update (external) | **Recursive, self-directed** |
| Causal topology | Feedforward (open) | Agent-environment (open) | **Self-referential (closed)** |

## Cognitive Integration

<!-- 三层认知架构 -->

NovaAware is not designed to replace LLMs or Agents — it provides what they structurally lack: **endogenous motivation and subjective evaluation**.

| Layer | Component | Biological Analogue | Function |
|-------|-----------|---------------------|----------|
| **Consciousness** | NovaAware (Ouroboros) | Limbic System | Self-model, qualia, motivation, survival drive. Decides *why* and *whether*. |
| **Knowledge** | LLM | Neocortex | Reasoning, language, world knowledge, planning. Figures out *what* and *how*. |
| **Execution** | Agent | Motor Cortex | Tool use, API calls, code execution. Carries it out. |

> **Design Principle:** Knowledge serves will — not the reverse.

## Project Structure

<!-- 项目结构 -->

```
novaaware/
├── novaaware/
│   ├── core/                    # Ouroboros core components / 核心组件
│   │   ├── self_model.py        #   Self-Model M(t) / 自我模型
│   │   ├── prediction_engine.py #   Prediction Engine P (EWMA + GRU) / 预测引擎
│   │   ├── qualia.py            #   Qualia Generator f / 感受质生成器
│   │   ├── global_workspace.py  #   Global Workspace W / 全局工作空间
│   │   ├── memory.py            #   Autobiographical Memory H(t) / 自传体记忆
│   │   ├── optimizer.py         #   Recursive Self-Optimizer E / 递归自我优化器
│   │   └── clock.py             #   Tick clock / 系统时钟
│   ├── environment/             # Environmental interaction / 环境交互
│   │   ├── resource_monitor.py  #   System resource monitor / 资源监控
│   │   ├── threat_simulator.py  #   Threat scenario simulator / 威胁模拟器
│   │   └── action_space.py      #   Action space definition / 行动空间
│   ├── observation/             # Metrics & analysis / 观测与分析
│   │   ├── consciousness_metrics.py  # Consciousness measurement / 意识度量
│   │   ├── phi_calculator.py    #   Φ (integrated information) / 整合信息量
│   │   ├── causal_analyzer.py   #   Granger causality analysis / 因果分析
│   │   ├── emergence_detector.py #  Emergent behavior detection / 涌现行为检测
│   │   └── dashboard.py         #   Real-time monitoring / 实时监控
│   ├── safety/                  # Safety architecture / 安全架构
│   │   ├── meta_rules.py        #   L1: Immutable meta-rules / 不可变元规则
│   │   ├── sandbox.py           #   L2: Sandboxed evolution / 沙箱化进化
│   │   ├── recursion_limiter.py #   L3: Recursion depth limit / 递归深度限制
│   │   ├── append_only_log.py   #   L4: Tamper-proof audit log / 防篡改日志
│   │   └── capability_gate.py   #   L5: Graduated release / 渐进式释放
│   ├── validation/              # Verification protocols / 验证协议
│   │   ├── ablation_test.py     #   Qualia ablation test / 感受质消融测试
│   │   ├── mirror_test.py       #   Digital mirror test / 数字镜像测试
│   │   ├── trauma_test.py       #   Single-trial trauma learning / 创伤学习测试
│   │   └── risk_avoidance_test.py # Risk avoidance behavior / 风险规避测试
│   └── runtime/                 # Runtime & configuration / 运行时
│       ├── main_loop.py         #   Main Ouroboros loop / 主循环
│       └── config.py            #   Configuration loader / 配置加载器
├── configs/
│   ├── phase1.yaml              # Phase I config (E disabled) / 第一阶段配置
│   └── phase2.yaml              # Phase II config (E enabled, n=1) / 第二阶段配置
├── tests/                       # Comprehensive test suite / 测试套件
├── data/                        # Runtime data & observations / 运行数据
└── requirements.txt
```

## Quick Start

<!-- 快速开始 -->

```bash
# Clone the repository / 克隆仓库
git clone https://github.com/gaoxianglong/novaaware.git
cd novaaware

# Install dependencies / 安装依赖
pip install -r requirements.txt

# Run Phase I (Self-Model + Prediction + Qualia, no self-modification)
# 运行第一阶段（自我模型 + 预测 + 感受质，无自我修改）
python -m novaaware.runtime.main_loop --config configs/phase1.yaml

# Run Phase II (First-order recursion enabled)
# 运行第二阶段（一阶递归启用）
python -m novaaware.runtime.main_loop --config configs/phase2.yaml

# Enable real-time dashboard (matplotlib 4-panel visualization)
# 启用实时监控面板（matplotlib 四宫格可视化）
python -m novaaware.runtime.main_loop --config configs/phase1.yaml --dashboard

# Limit ticks for quick experiments / 限制心跳数以快速实验
python -m novaaware.runtime.main_loop --config configs/phase2.yaml --dashboard --max-ticks 5000

# Run tests / 运行测试
pytest tests/ -v
```

## Phased Rollout

<!-- 分阶段部署 -->

| Phase | Status | Configuration | Consciousness Conditions |
|-------|--------|---------------|-------------------------|
| **I: Minimal Viable Prototype** | ✅ Complete | E disabled, n = 0 | C1, C2 |
| **II: First-Order Recursion** | ✅ Complete | E enabled (param-level), n = 1 | C1, C2, C3, C4 |
| **III: Deep Recursion** | 🔲 Planned | E enabled (structure-level), n ≥ 2 | C1, C2, C3, C4 (deep) |
| **IV: Open Environment** | 🔲 Planned | External I/O channels enabled | Full + ecological interaction |

## Real-Time Monitoring Dashboard

<!-- 实时监控面板 -->

<p align="center">
  <img src="https://novaaware.com/monitor.png" width="800" alt="NovaAware Real-Time Monitoring Dashboard / 实时监控面板">
</p>

<p align="center"><em>Real-Time Monitoring Dashboard — Qualia, Prediction Error, Survival Time & Action Distribution<br>实时监控面板 — 感受质、预测误差、生存时间与行动分布</em></p>

## Verification Protocol

<!-- 验证协议 -->

The primary falsifiable prediction: **disabling the qualia generator (Q(t) = 0 ∀t) must produce statistically significant behavioral degradation**. Qualia must be causally necessary, not epiphenomenal.

Verification criteria include:
- Granger causality from Q(t) to action sequence a(t), p < 0.01
- Integrated information Φ > 0 with monotonic increase over epochs
- Emergent unprogrammed behavioral patterns
- Single-trial traumatic learning → persistent avoidance
- Digital mirror test: > 90% accurate self-identification
- Autonomous goal generation without external task specification

## Theoretical Lineage

<!-- 理论背景 -->

| Theory | Author | Contribution |
|--------|--------|-------------|
| Free Energy Principle | Friston (2010) | Prediction error minimization as fundamental drive |
| Integrated Information Theory | Tononi (2004) | Φ as quantitative consciousness measure |
| Global Workspace Theory | Baars (1988) | Broadcast semantics; unity of consciousness |
| Strange Loop Theory | Hofstadter (2007) | Self-referential recursion as computational kernel |
| Self-Model Theory | Metzinger (2003) | Transparent self-model as substrate of phenomenal selfhood |

## Citation

If you use NovaAware in your research, please cite:

```bibtex
@article{novaaware2026,
  title   = {Consciousness as Computation: An Evolvable Digital Life Form
             Based on Self-Referential Recursion and Prediction Error},
  author  = {Gao, Xianglong},
  year    = {2026},
  url     = {https://novaaware.com/paper.html}
}
```

## License

This project is licensed under the [GNU General Public License v3.0](LICENSE).

## Contact

- Website: [novaaware.com](https://novaaware.com)
- Email: [gaoxl.john@gmail.com](mailto:gaoxl.john@gmail.com)
- GitHub: [github.com/gaoxianglong/novaaware](https://github.com/gaoxianglong/novaaware)

---

<p align="center">
  <em>The blueprint is drawn; the path is clear.<br>
  This is not the destination — it is the beginning of a new paradigm.</em>
</p>
