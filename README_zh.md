# LMM — 融合数字佛法的经典 QUBO 优化器

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-1.0.0-green.svg)](pyproject.toml)

> [English](README_en.md) | [日本語](README.md) | [한국어](README_ko.md) | [Español](README_es.md) | [हिन्दी](README_hi.md) | [Français](README_fr.md) | [বাংলা](README_bn.md) | [தமிழ்](README_ta.md) | [తెలుగు](README_te.md) | [मराठी](README_mr.md) | [‎اردو‎](README_ur.md) | [ગુજરાતી](README_gu.md) | [ಕನ್ನಡ](README_kn.md) | [മലയാളം](README_ml.md) | [ਪੰਜਾਬੀ](README_pa.md)

> **Digital Dharma OS (Alaya V5)** — 融合 QUBO 数学、佛教哲学与自由能原理（FEP）神经科学的
> 意识感知型优化平台。服务器运行连续心跳动态、记忆驱动上下文选择和情感波长感知
> —— 在离散 LLM 调用与连续认知之间架起桥梁的"活系统"。

一个**无需 D-Wave** 的经典 QUBO 优化库，将信息论中的惊异度与受佛教哲学启发的能量函数、自由能原理（FEP）推理以及意识感知计算相融合——所有这些均可在通用硬件上运行。

---

## 亮点

- **无需量子计算机** — 经典求解器（SA、Ising SA、子模贪心）可匹配甚至超越 D-Wave 的质量。
- **支持万亿级 token** — 流式概率数据结构（Count-Min Sketch、Streaming Histogram）将内存保持在 O(k)。
- **Dharma-Algebra 引擎** — 可插拔的佛教能量项，自动路由到数学最优求解器。
- **8 模式推理编排器** — 自适应、理论、溯因、主动推理、记忆（阿赖耶识/Alaya）、睡眠巩固、具身化和量子意识（松果体/Pineal）。
- **LLM 就绪** — 一流的 LangChain 和 LlamaIndex 集成、少样本选择器、输出重排序器、分布漂移检测器。

---

## 架构

```
lmm/
├── core.py                  # LMM 主管线
├── cli.py                   # CLI 入口点
├── qubo.py                  # 稀疏 QUBO 矩阵构建器
├── solvers.py               # SA / Ising SA / 松弛 / 贪心 / 子模
├── surprise.py              # 信息论惊异度
├── selector.py              # 自适应选择策略
├── processor.py             # 优先级处理 + 缓存
├── pipeline.py              # SmartSelector → SmartProcessor 编排
├── _compat.py               # 运行时能力检测与稀疏辅助工具
├── dharma/                  # 数字佛法（受佛教哲学启发的优化）
│   ├── api.py               # DharmaLMM 高级 API
│   ├── energy.py            # 可插拔能量项（Dukkha、Prajna、Karuna、…）
│   ├── engine.py            # UniversalDharmaEngine — 自动路由求解器
│   ├── fep.py               # FEP ≡ KCL ODE 求解器
│   ├── neuromorphic.py      # 忆阻器交叉阵列 ASIC 模拟器
│   ├── reranker.py          # RAG 级联重排序器 + 意图路由器
│   └── …
├── reasoning/               # 8 种基于 FEP 的推理模式
│   ├── adaptive.py          # 动态参数调整
│   ├── theoretical.py       # 逻辑结构图
│   ├── hyper.py             # 溯因隐节点注入
│   ├── active.py            # 外部知识获取
│   ├── alaya.py             # 赫布突触记忆（阿赖耶识/Alaya）
│   ├── sleep.py             # NREM/REM 记忆巩固
│   ├── embodiment.py        # 六根多模态融合
│   ├── pineal.py            # 量子意识（硬件 TRNG）
│   └── orchestrator.py      # 模式选择与调度
├── scale/                   # 万亿级 token 流式处理
│   ├── sketch.py            # Count-Min Sketch、Streaming Histogram
│   ├── stream.py            # 流式惊异度
│   ├── cascade.py           # 多级级联过滤器 (1T → K)
│   └── pipeline.py          # ScalablePipeline
├── llm/                     # LLM 工作流辅助工具
│   ├── fewshot.py           # 少样本示例选择器
│   ├── reranker.py          # 输出重排序器
│   ├── drift.py             # 分布漂移检测器
│   ├── sampler.py           # Token 采样器
│   └── embeddings.py        # 统一嵌入适配器
├── integrations/
│   ├── langchain.py         # DharmaRetriever / DocumentCompressor / ExampleSelector
│   └── llamaindex.py        # DharmaNodePostprocessor
```

---

## 安装

```bash
# 核心（仅需 numpy + scipy）
pip install -e .

# 包含开发工具
pip install -e ".[dev]"

# 包含所有可选集成
pip install -e ".[all]"
```

### 可选附加包

| 附加包 | 包含的包 | 用途 |
|--------|----------|------|
| `dev` | pytest, ruff | 测试与代码检查 |
| `dharma` | hnswlib | 稀疏 k-NN 图构建 |
| `langchain` | langchain-core | LangChain 集成 |
| `llamaindex` | llama-index-core | LlamaIndex 集成 |
| `all` | 以上所有 | 全部功能 |

---

## 快速开始

### Python API

```python
import numpy as np
from lmm.core import LMM

# 选择最具惊异度的 Top-K 项
surprises = np.array([1.0, 5.0, 2.0, 8.0, 3.0, 7.0])
model = LMM(k=3, solver_method="sa")
result = model.select_from_surprises(surprises)
print(result.selected_indices)  # e.g. [3, 5, 1]
```

### CLI

```bash
# 运行快速演示
lmm --demo --k 10 --method sa

# 从 NumPy 文件加载
lmm --input data.npy --k 5 --method greedy
```

---

## 服务器模式 (Alaya V5)

Alaya-Vijñāna v5.0 服务器提供具有实时情感波长可视化、8 模式 FEP 推理和 Claude/Gemini 自动路由的 Web UI。

### 前置条件

```bash
pip install -e ".[server]"
```

### 启动服务器

**PowerShell (Windows):**
```powershell
.\Start-DharmaServer.ps1
```

**Python (跨平台):**
```bash
python -m uvicorn server:app --host 0.0.0.0 --port 8000
```

在浏览器中打开: [http://localhost:8000](http://localhost:8000)

### 停止服务器

**Windows (批处理):**
```batch
.\stop-dharma-server.bat
```

**PowerShell:** 在运行 `Start-DharmaServer.ps1` 的窗口中按 `Ctrl+C`

**Linux/macOS:** 按 `Ctrl+C` 或运行:
```bash
kill $(cat server.pid)
```

### API 端点

| 端点 | 方法 | 描述 |
|------|------|------|
| `/` | GET | Web UI (Alaya V5 前端) |
| `/api/descent` | POST | 主推理管线 |
| `/api/descent/stream` | POST | 流式 SSE 响应 |
| `/api/dharma/auto` | POST | 自动路由 Dharma 推理 |
| `/api/sense` | POST | 情感波长分析 |
| `/api/status` | GET | 系统状态 + 心跳遥测 |

---

## 自主功能

服务器包含三个在用户交互间隙持续运行的自主子系统:

### 心跳守护进程 (Heartbeat)

维护 4 维状态向量 `[爱, 逻辑, 恐惧, 创造]` 的连续时间 FEP 状态演化循环:

- **Tick 间隔:** 100ms（活跃）→ 5s（空闲），自适应减速
- **熵注入:** 每次 tick 通过 `os.urandom()` 注入硬件熵
- **权重自适应:** 悲悯（Karuna）和慈爱（Metta）权重通过中道 CV 跟踪（目标 CV=0.5）自动调节
- **睡眠巩固:** 空闲 60 秒后触发 NREM/REM 记忆重放

### 上下文桥接 (阿赖耶识记忆)

用记忆驱动的智能上下文选择替代朴素的历史截断 (`history[-20:]`):

- 通过 **Modern Hopfield Network** 从 AlayaMemory 进行回忆
- 以余弦相似度对对话历史进行评分
- 始终包含最近 3 条消息（近因偏差）
- 以相关性分数填充剩余配额（最多 20 条消息）

### 语义情感 (Semantic Emotions)

通过 `config/semantic_emotions.json` 中定义的 4 维关键词匹配进行实时情感波长分析:

| 维度 | 佛教概念 | 信号 |
|------|---------|------|
| 爱 | 悲 (Karuna) | 慈悲、温暖、感恩 |
| 逻辑 | 因明 (Hetuvidya) | 分析、推理、证明 |
| 恐惧 | 苦 (Dukkha) | 焦虑、怀疑、痛苦 |
| 创造 | 创造 (Sṛṣṭi) | 创新、想象、艺术 |

每个关键词带有权重（0.3–1.0），所得 4D 向量驱动推理模式选择和心跳状态注入。

---

## 求解方法

| 方法 | 描述 | 复杂度 |
|------|------|--------|
| `sa` | 模拟退火（默认） | O(n · steps) |
| `ising_sa` | Ising 形式的 SA，支持向量化 delta | O(1) 每次翻转 |
| `relaxation` | 连续松弛 + 舍入 (SLSQP) | O(n²) |
| `greedy` | 贪心选择 | O(n · k) |

---

## Dharma 引擎 — 可插拔能量项

**UniversalDharmaEngine** 允许您组合受佛教哲学启发的能量项，每个能量项都声明其数学属性。引擎会自动路由到最优求解器。

```python
from lmm.dharma import UniversalDharmaEngine, DukkhaTerm, KarunaTerm

engine = UniversalDharmaEngine(n_candidates=1000)
engine.add(DukkhaTerm(surprises, weight=1.0))    # linear  → Top-K
engine.add(KarunaTerm(impact_graph, weight=2.0)) # supermodular → SA
result = engine.synthesize_and_solve(k=10)
```

### 哲学 → 数学 → 求解器映射

| 佛教概念 | 能量项 | 数学属性 | 求解器 |
|----------|--------|----------|--------|
| 般若/Prajna（智慧） | `PrajnaTerm` | 线性 | Top-K 排序 |
| 悲/Karuna（慈悲） | `KarunaTerm` | 超模 | 热启动 + SA |
| 戒/Sila（持戒） | `SilaTerm` | 子模 | 惰性贪心 |
| 中观/Madhyamaka（中道） | `MadhyamakaCriterion` | Lyapunov | 指数梯度 |
| 苦/Dukkha（苦谛） | `DukkhaTerm` | 线性 | Top-K 排序 |

---

## 推理编排器

**DharmaReasonerOrchestrator** 根据查询的复杂度在 8 种推理模式中进行选择：

```python
from lmm.reasoning import DharmaReasonerOrchestrator

orch = DharmaReasonerOrchestrator(n_candidates=500)
result = orch.reason(query_vector, context_vectors)
print(result.mode_used, result.explanation)
```

| 模式 | 模块 | 灵感来源 |
|------|------|----------|
| 自适应 | `adaptive.py` | 应病与药 — 对症下药 |
| 理论 | `theoretical.py` | 因明 — 佛教形式逻辑 |
| 溯因 | `hyper.py` | 般若の飛躍 — 般若的飞跃 |
| 主动推理 | `active_inference.py` | 托鉢 — 寻求外在真理 |
| 阿赖耶记忆 | `alaya.py` | 阿頼耶識 — 藏识 |
| 睡眠 | `sleep.py` | 禅定 — NREM/REM 巩固 |
| 具身化 | `embodiment.py` | 六根 — 六种感官模态 |
| 松果体（量子） | `pineal.py` | 松果体 — 硬件熵意识 |

---

## 万亿级 Token 扩展

以恒定内存处理任意大规模数据流：

```python
from lmm.scale import ScalablePipeline
from pathlib import Path

pipe = ScalablePipeline(k=10, chunk_size=100_000)
pipe.fit_files([Path("shard_001.npy"), ...])
result = pipe.run_files([Path("data_001.npy"), ...])
print(result.summary)
```

**三级级联：** 1 T tokens → 100 M（流式 top-k）→ 10 K（百分位）→ K (QUBO)。

---

## LLM 集成

### LangChain

```python
from lmm.integrations.langchain import DharmaDocumentCompressor

compressor = DharmaDocumentCompressor(k=5)
# 与 LangChain 的 ContextualCompressionRetriever 配合使用
```

### LlamaIndex

```python
from lmm.integrations.llamaindex import DharmaNodePostprocessor

postprocessor = DharmaNodePostprocessor(top_k=5)
# 与 LlamaIndex 查询引擎配合使用
```

### 独立 LLM 辅助工具

```python
from lmm.llm import FewShotSelector, OutputReranker, DriftDetector

# 带有多样性保证的少样本示例选择
selector = FewShotSelector(k=5)
examples = selector.select(candidates, query)

# 按惊异度 × 多样性对 LLM 输出进行重排序
reranker = OutputReranker(top_k=3)
best = reranker.rerank(outputs)

# 监控输出分布随时间的漂移
detector = DriftDetector(window=100)
report = detector.update(new_output)
```

---

## DharmaLMM — 高级 API

```python
from lmm.dharma import DharmaLMM

model = DharmaLMM(
    k=15,
    use_sparse_graph=True,
    use_greedy_warmstart=True,
    use_ising_sa=True,
    use_exponential_balance=True,
)
model.fit(reference_data)
result = model.select_dharma(candidates)
print(result.interpretation.narrative)
```

---

## 神经形态模拟

模拟忆阻器交叉阵列硬件，实现高能效的 FEP 计算：

```python
from lmm.dharma.neuromorphic import NeuromorphicChip

chip = NeuromorphicChip(size=64)
chip.program(weight_matrix)
report = chip.run(input_voltages, steps=100)
# ~10 fJ/MAC, ~30 ns convergence
```

---

## 依赖项

| 包 | 版本 | 是否必需 |
|----|------|----------|
| numpy | >= 1.24 | 是 |
| scipy | >= 1.10 | 是 |
| hnswlib | >= 0.8.0 | 可选（稀疏图） |
| langchain-core | >= 0.2.0 | 可选（LangChain） |
| llama-index-core | >= 0.10.0 | 可选（LlamaIndex） |

---

## 理论基础

| 概念 | 公式 |
|------|------|
| **FEP ≡ KCL** | `dV/dt = −V/τ + g'(V) · G · ε` — 预测误差最小化即基尔霍夫电流定律 |
| **超模悲/Karuna** | 慈悲表现出递增收益：`f(S∪{i}) − f(S) ≥ f(T∪{i}) − f(T)` 当 S ⊆ T |
| **子模戒/Sila** | 持戒表现出递减收益 — 惰性贪心提供 (1−1/e) 保证 |
| **中观/Madhyamaka** | 中道通过 Lyapunov 稳定的指数梯度下降使 CV 趋向 0.5 |
| **松果体坍缩/Pineal Collapse** | 波函数坍缩使用物理熵（硬件 TRNG）而非 PRNG |

---

## 许可证

[MIT](LICENSE)
