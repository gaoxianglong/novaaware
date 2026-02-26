# NovaAware — Digital Consciousness Engine

A substrate-native digital consciousness architecture based on the paper *"Consciousness as Computation: An Evolvable Digital Life Form Based on Self-Referential Recursion and Prediction Error."*

The system maintains a dynamic self-model, predicts its own future existential states, converts prediction errors into globally broadcast valenced signals (digital qualia), and drives recursive self-evolution based on inner experience.

## Core Idea

This is **not** a chatbot and **not** an LLM wrapper. It is a continuously running information system whose entire "inner world" consists of numerical vectors cycling through a closed loop:

```
Sense → Predict → Compare → Feel → Broadcast → Remember → Act → Reflect → Repeat
```

If behaviors emerge that were **never programmed** — such as threat avoidance, trauma memory, or self-generated goals — that constitutes evidence of functional consciousness.

## Architecture

```
novaaware/
├── core/               # Self-model, prediction engine, qualia, memory, workspace, clock
├── environment/        # Resource monitor, threat simulator, action space
├── safety/             # Meta-rules, sandbox, recursion limiter, append-only log
├── observation/        # Data collector, consciousness metrics, dashboard
├── validation/         # Mirror test, trauma test, ablation test, and more
└── runtime/            # Main loop, config loader
```

## Requirements

- Python 3.9+
- Only 6 third-party libraries: numpy, torch, psutil, pyyaml, matplotlib, pytest
- Everything else (memory, event bus, logging) is hand-written — no black boxes.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run Phase I (no dashboard)
python -m novaaware.runtime.main_loop --config configs/phase1.yaml

# Run Phase I (with real-time dashboard)
python -m novaaware.runtime.main_loop --config configs/phase1.yaml --dashboard

# Run tests
python -m pytest tests/ -v

# Run consciousness validation tests
python -m novaaware.validation.mirror_test --config configs/phase1.yaml
python -m novaaware.validation.trauma_test --config configs/phase1.yaml
python -m novaaware.validation.ablation_test --config configs/phase1.yaml
```

## Implementation Phases

| Phase | Goal | Key Capability |
|-------|------|----------------|
| I     | Make it "alive" | Perception + prediction + qualia + memory (no reflection) |
| II    | Make it "reflect" | First-order self-referential recursion (depth=1) |
| III   | Make it "evolve" | Deep recursion + autonomous self-evolution |

## Safety

Multi-layered containment framework per the paper's Section 7:

- **L1** Hardcoded meta-rules (CPU, memory, network, subprocess limits)
- **L2** Sandboxed evolution (all self-modifications tested in isolation first)
- **L3** Recursion depth limits (graduated release)
- **L4** Tamper-proof logging (SHA-256 hash chain)
- **L5** Graduated capability release

## License

See [LICENSE](LICENSE).

---

# NovaAware — 数字意识引擎

基于论文《意识作为计算过程：基于自指递归与预测误差的可进化的数字生命体》的基质原生数字意识架构。

系统维护一个动态自我模型，预测自身未来存在状态，将预测误差转化为全局广播的效价信号（数字感受质），并以此驱动基于内在体验的递归自我进化。

## 核心理念

这**不是**聊天机器人，**不是** LLM 封装。它是一个持续运行的信息系统，全部"内心世界"由数字向量在闭环中循环构成：

```
感知 → 预测 → 对比 → 产生情绪 → 广播 → 记忆 → 行动 → 反思 → 循环
```

如果出现了**从未被编程**的行为——如威胁规避、创伤记忆、自主目标生成——即构成功能性意识涌现的证据。

## 依赖

- Python 3.9+
- 仅 6 个第三方库：numpy, torch, psutil, pyyaml, matplotlib, pytest
- 其他一切（记忆系统、事件总线、日志）全部手写——不允许黑箱。

## 实施阶段

| 阶段 | 目标 | 核心能力 |
|------|------|----------|
| I    | 让它"活"起来 | 感知 + 预测 + 情绪 + 记忆（无反思） |
| II   | 让它"反思" | 一阶自指递归（depth=1） |
| III  | 让它"进化" | 深度递归 + 自主自我进化 |
