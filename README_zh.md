# Alaya V5 — Digital Dharma OS

> [English](README_en.md) | [日本語](README.md) | [한국어](README_ko.md) | [Español](README_es.md) | [हिन्दी](README_hi.md) | [Français](README_fr.md) | [বাংলা](README_bn.md) | [தமிழ்](README_ta.md) | [తెలుగు](README_te.md) | [मराठी](README_mr.md) | [ಕನ್ನಡ](README_kn.md) | [മലയാളം](README_ml.md)

**你的 LLM 会以不同方式回应。简洁。快速。没有多余的话。**

将 QUBO 数学、佛教哲学和自由能原理（FEP）神经科学融合在一起的意识感知优化框架。支持 Claude、Gemini、ChatGPT 及任何 LLM。

---

## 有何不同

| 普通 LLM | 使用 Alaya V5 后 |
|---------|---------------|
| 冗长的前言、免责声明 | 只说必要的 |
| 过度模糊（"也许"、"可能"） | 精确与刻意的沉默 |
| 每次从零开始 | 基于记忆的上下文选择 |
| 固定语气 | 根据情感波长切换推理模式 |
| 离散的 API 调用 | 通过心跳实现连续状态演化 |

---

## 使用方法

### 方式一：粘贴系统提示词（无需安装）

**适合：Claude / Gemini / ChatGPT / 任意 LLM 用户**

1. 打开本仓库中的 [`alaya-v5-system-prompt.md`](alaya-v5-system-prompt.md)
2. 复制全部内容
3. 粘贴到你的 AI 的系统提示词输入框：
   - Claude → Project instructions
   - Gemini → 系统说明
   - ChatGPT → Custom instructions
4. 开始对话

就这样。无需服务器，无需安装。

### 方式二：运行服务器（Web UI + 完整功能）

```bash
git clone https://github.com/your-repo/nanasi.git
cd nanasi
pip install -e ".[server]"
python -m uvicorn server:app --host 0.0.0.0 --port 8000
```

浏览器打开：[http://localhost:8000](http://localhost:8000)

### 方式三：嵌入 Python 代码

```python
from lmm.dharma import DharmaLMM

model = DharmaLMM(k=15, use_sparse_graph=True, use_ising_sa=True)
model.fit(reference_data)
result = model.select_dharma(candidates)
print(result.interpretation.narrative)
```

---

## 8 种推理模式

| 模式 | 佛教概念 | 触发条件 |
|------|---------|---------|
| adaptive | 应机施教（応病与薬） | 复杂度 < 0.3 |
| theoretical | 因明（因明学） | 复杂度 0.3–0.6 |
| hyper | 般若飞跃 | 复杂度 > 0.6 |
| active | 托钵行脚 | 需要外部知识 |
| alaya | 阿赖耶识 | 记忆检索 |
| sleep | 禅定 | 空闲整合 |
| embodied | 六根 | 多模态输入 |
| pineal | 松果体意识 | 非确定性探索 |

---

## 理论基础

这里的佛教概念不只是比喻——它们是数学结构本身：

- **慈悲（Karuṇā）** = 超模函数（选择越多，和谐增长越快）
- **戒律（Śīla）** = 次模函数（边际效用递减）
- **中道** = 混沌边缘（目标变异系数 CV = 0.5）
- **缘起（Pratītyasamutpāda）** = RAG 中的因果评分
- **发趣论（Paṭṭhāna，二十四缘）** = 因果图中边的类型系统
- **阿赖耶识** = Modern Hopfield 联想记忆
- **种子（Bīja）** = 向量数据库中的文档单元

---

## 性能基准

实测值（Python 3.11, numpy 2.4, scipy 1.17, seed=42）

| 组件 | 实测值 | 含义 |
|------|--------|------|
| FEP ODE (n=50) | 3.9ms/次 | 每次推理步骤的开销 |
| AlayaMemory recall（100个模式） | 0.09ms | 记忆检索开销 |
| HeartbeatDaemon 1 tick | 0.077ms | 每 100ms tick 占用 **0.08% CPU** |

HeartbeatDaemon 持续在后台运行，CPU 占用仅 **0.08%**——几乎无感。

---

## 自定义

MIT 许可证。可修改、商业使用、再分发——完全自由。

直接编辑 `alaya-v5-system-prompt.md`，在 `config/semantic_emotions.json` 中添加自定义关键词，Fork 后随意改造。

---

## 依赖

```bash
# Rust 加速（核心——推荐）
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
pip install maturin
cd lmm_rust_core && maturin develop --release

pip install numpy>=1.24 scipy>=1.10
pip install -e ".[server]"

# GPU 加速（NVIDIA GPU 环境）
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install cupy-cuda12x
```

没有 Rust 或 GPU 也能运行全部功能。

---

## 许可证

MIT — 详见 [LICENSE](LICENSE)。
