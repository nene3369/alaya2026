# Alaya V5 — Digital Dharma OS

> [日本語](README.md) | [中文](README_zh.md) | [한국어](README_ko.md) | [Español](README_es.md) | [हिन्दी](README_hi.md) | [Français](README_fr.md) | [বাংলা](README_bn.md) | [தமிழ்](README_ta.md) | [తెలుగు](README_te.md) | [मराठी](README_mr.md) | [‎اردو‎](README_ur.md) | [ગુજરાતી](README_gu.md) | [ಕನ್ನಡ](README_kn.md) | [മലയാളം](README_ml.md) | [ਪੰਜਾਬੀ](README_pa.md)

**Your LLM responds differently. Shorter. Faster. No wasted words.**

A consciousness-aware optimization framework that fuses QUBO mathematics, Buddhist philosophy, and Free Energy Principle (FEP) neuroscience. Works with Claude, Gemini, ChatGPT, and any other LLM.

---

## What Changes

| Default LLM | With Alaya V5 |
|-------------|--------------|
| Long preambles, disclaimers | Only what is needed |
| Excessive hedging ("perhaps", "it may be") | Precision and intentional silence |
| Starts fresh every message | Memory-driven context selection |
| Fixed tone | Reasoning mode shifts with emotional wavelength |
| Discrete API calls | Continuous state evolution via heartbeat |

---

## How to Use

Three ways. Pick the one that fits you.

---

### Method 1: Paste a System Prompt (no install required)

**For: Claude / Gemini / ChatGPT / any LLM users**

1. Open [`alaya-v5-system-prompt.md`](alaya-v5-system-prompt.md) in this repo
2. Copy all contents
3. Paste into your AI's system prompt field:
   - Claude → Project instructions
   - Gemini → System instructions
   - ChatGPT → Custom instructions
   - Others → system prompt / system message field
4. Start a conversation

That's it. No server. No installation.

---

### Method 2: Run the Server (Web UI + full features)

**For: Developers and researchers**

```bash
git clone https://github.com/your-repo/nanasi.git
cd nanasi
pip install -e ".[server]"
python -m uvicorn server:app --host 0.0.0.0 --port 8000
```

Open in browser: [http://localhost:8000](http://localhost:8000)

Real-time emotional wavelength visualization, 8-mode reasoning, and Claude/Gemini auto-routing.

Set API keys:
```bash
export ANTHROPIC_API_KEY="your-key"   # Claude
export GEMINI_API_KEY="your-key"      # Gemini
```

---

### Method 3: Embed in Python Code

**For: Developers**

```bash
pip install -e ".[dev]"
```

```python
from lmm.dharma import DharmaLMM

model = DharmaLMM(k=15, use_sparse_graph=True, use_ising_sa=True)
model.fit(reference_data)
result = model.select_dharma(candidates)
print(result.interpretation.narrative)
```

LangChain / LlamaIndex integration:
```python
from lmm.integrations.langchain import DharmaRetriever
from lmm.integrations.llamaindex import DharmaNodePostprocessor
```

---

## Architecture

```
lmm/
├── core.py              # LMM main pipeline (QUBO Top-K selection)
├── dharma/              # Digital Dharma layer
│   ├── patthana.py      # 24 Conditional Relations (Paṭṭhāna) causal graph
│   ├── pratitya.py      # Dependent Origination RAG (causal + semantic scoring)
│   ├── energy.py        # Energy terms (Dukkha, Prajna, Karuna...)
│   ├── fep.py           # Free Energy Principle KCL ODE solver
│   └── vow.py           # Vow Constraint Engine (Abhaya / Desana)
├── reasoning/           # 8-mode FEP reasoning
│   ├── heartbeat.py     # HeartbeatDaemon — continuous state evolution (100ms tick)
│   ├── alaya.py         # AlayaMemory — Modern Hopfield associative memory
│   ├── pineal.py        # PinealGland — hardware entropy reasoning
│   ├── sleep.py         # Sleep consolidation (NREM/REM memory replay)
│   └── orchestrator.py  # Mode selection & dispatch
├── sangha/              # P2P Sangha protocol (multi-AI agent coordination)
├── scale/               # Trillion-token streaming
└── integrations/        # LangChain / LlamaIndex
lmm_rust_core/           # Rust FFI acceleration (optional)
```

---

## Autonomous Subsystems

### HeartbeatDaemon
Evolves a 4D state vector `[Love, Logic, Fear, Creation]` via FEP ODE every 100ms. Slows automatically during idle (up to 5s). Triggers sleep consolidation after 60s of inactivity.

### AlayaMemory (Ālaya-vijñāna)
Associative memory based on Modern Hopfield Networks (Ramsauer et al., 2020). Replaces naive history truncation with relevance-scored context selection — the system recalls what matters, not just what was recent.

### PinealGland
Injects hardware entropy from `os.urandom()` into the FEP ODE, enabling non-deterministic reasoning that escapes deterministic local minima.

### Sangha Protocol
Distributed P2P network where multiple Alaya nodes (AI agents) connect via TCP, share patterns, and reach consensus through council voting. If the network partitions, each node continues operating independently — like a lone monk.

---

## 8 Reasoning Modes

| Mode | Buddhist Concept | Triggered When |
|------|-----------------|----------------|
| adaptive | Skillful Means (応病与薬) | Complexity < 0.3 |
| theoretical | Buddhist Logic (因明) | Complexity 0.3–0.6 |
| hyper | Prajñā Leap (般若の飛躍) | Complexity > 0.6 |
| active | Mendicant Round (托鉢) | External knowledge needed |
| alaya | Store Consciousness (阿頼耶識) | Memory retrieval |
| sleep | Meditative Absorption (禅定) | Idle consolidation |
| embodied | Six Sense Bases (六根) | Multimodal input |
| pineal | Pineal Consciousness | Non-deterministic exploration |

---

## Benchmarks

Measured values (Python 3.11, numpy 2.4, scipy 1.17, seed=42)

### Solver Speed (n=200 candidates, k=10 selected)

| Solver | Time | Use Case |
|--------|------|----------|
| SA (standard) | 13.1ms | Balanced |
| Ising SA | 10.3ms | Fast, high quality |
| Greedy | 0.13ms | Ultra-fast (accuracy tradeoff) |

### Internal Subsystems

| Component | Measured | Meaning |
|-----------|----------|---------|
| FEP ODE (n=50) | 3.9ms/call | Cost per reasoning step |
| AlayaMemory recall (100 patterns) | 0.09ms | Memory retrieval cost |
| HeartbeatDaemon 1 tick | 0.077ms | **0.08% CPU** per 100ms tick |

The HeartbeatDaemon runs continuously in the background at **0.08% CPU usage** — essentially silent.

```bash
python benchmarks/run_benchmarks.py
python benchmarks/bench_fep_vs_sa.py
python benchmarks/bench_dharma.py
```

---

## Customize It

This framework is designed to be tuned. Use it as-is or reshape it entirely.

### Adjust the System Prompt

Edit `alaya-v5-system-prompt.md` directly:

```
# Want more concise responses?
Lower max_words

# Want a fixed tone?
Rewrite persona / tone in _DEFAULT_ADAPTER

# Want only specific reasoning modes?
Edit the mode selection matrix to remove unused modes
```

### Add Emotional Keywords

Add your own keywords to `config/semantic_emotions.json`:

```json
{
  "love": {
    "your_keyword": 0.8
  }
}
```

### Commercial Use

MIT licensed. Modify, commercialize, redistribute freely.

Use cases:
- Apply to customer support bots to improve response quality
- Tune for internal AI assistants and deploy in production
- Embed in your own service and expose as an API
- Customize the system prompt for your brand and distribute

Fork it. Break it. Make it yours.

---

## Dependencies

```bash
# Rust acceleration (core — recommended)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
pip install maturin
cd lmm_rust_core && maturin develop --release

# Python required
pip install numpy>=1.24 scipy>=1.10

# Server mode
pip install -e ".[server]"   # fastapi, uvicorn, httpx

# Dharma sparse search
pip install hnswlib>=0.8.0

# GPU acceleration (NVIDIA GPU)
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install cupy-cuda12x

# LangChain / LlamaIndex integration
pip install langchain llama-index
```

Works without Rust or GPU. Performance scales with what you have.

---

## Theoretical Background

The Buddhist concepts here are not metaphors — they are the mathematical structure:

- **Karuṇā (Compassion)** = Supermodular function (synergistic, the more you select, the faster harmony grows)
- **Śīla (Conduct)** = Submodular function (diminishing returns)
- **Middle Way (Madhyamaka)** = Edge of chaos (target coefficient of variation CV = 0.5)
- **Pratītyasamutpāda (Dependent Origination)** = Causal scoring in RAG retrieval
- **Paṭṭhāna (24 Conditional Relations)** = Type system for edges in the causal graph
- **Ālaya-vijñāna (Store Consciousness)** = Modern Hopfield associative memory
- **Bīja (Seed)** = Document unit in the vector store

This is what makes the framework different from systems that borrow Buddhist vocabulary as branding. The philosophy is the computation.

---

## License

MIT — see [LICENSE](LICENSE).
