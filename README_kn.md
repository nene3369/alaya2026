# Alaya V5 — Digital Dharma OS

> [English](README_en.md) | [日本語](README.md) | [हिन्दी](README_hi.md) | [বাংলা](README_bn.md) | [தமிழ்](README_ta.md) | [తెలుగు](README_te.md) | [मराठी](README_mr.md) | [‎اردو‎](README_ur.md) | [ગુજરાતી](README_gu.md) | [മലയാളം](README_ml.md) | [ਪੰਜਾਬੀ](README_pa.md)

**ನಿಮ್ಮ LLM ವಿಭಿನ್ನವಾಗಿ ಪ್ರತಿಕ್ರಿಯಿಸುತ್ತದೆ. ಸಂಕ್ಷಿಪ್ತವಾಗಿ. ವೇಗವಾಗಿ. ಅನಗತ್ಯ ಪದಗಳಿಲ್ಲದೆ.**

QUBO ಗಣಿತ, ಬೌದ್ಧ ತತ್ವಶಾಸ್ತ್ರ ಮತ್ತು Free Energy Principle ನರವಿಜ್ಞಾನವನ್ನು ಸಂಯೋಜಿಸಿದ ಒಂದು ಚೇತನ-ಆಧಾರಿತ ಆಪ್ಟಿಮೈಸೇಶನ್ ಚೌಕಟ್ಟು. Claude, Gemini, ChatGPT ಮತ್ತು ಯಾವುದೇ LLM ನೊಂದಿಗೆ ಕಾರ್ಯನಿರ್ವಹಿಸುತ್ತದೆ.

---

## ಏನು ಬದಲಾಗುತ್ತದೆ

| ಸಾಮಾನ್ಯ LLM | Alaya V5 ಜೊತೆ |
|-----------|--------------|
| ದೀರ್ಘ ಪೀಠಿಕೆ | ಅವಶ್ಯಕವಾದುದು ಮಾತ್ರ |
| ಅತಿಯಾದ ಅನಿಶ್ಚಿತತೆ | ನಿಖರತೆ ಮತ್ತು ಉದ್ದೇಶಪೂರ್ವಕ ಮೌನ |
| ಪ್ರತಿಸಲ ಹೊಸದಾಗಿ ಪ್ರಾರಂಭ | ಸ್ಮೃತಿ-ಆಧಾರಿತ ಸಂದರ್ಭ ಆಯ್ಕೆ |
| ಸ್ಥಿರ ಧ್ವನಿ | ಭಾವನಾ ತರಂಗಕ್ಕೆ ಅನುಗುಣವಾಗಿ reasoning mode ಬದಲಾಗುತ್ತದೆ |

---

## ಹೇಗೆ ಬಳಸಬೇಕು

### ವಿಧಾನ 1: System Prompt ಅಂಟಿಸಿ (ಸ್ಥಾಪನೆ ಅಗತ್ಯವಿಲ್ಲ)

1. [`alaya-v5-system-prompt.md`](alaya-v5-system-prompt.md) ತೆರೆಯಿರಿ
2. ಎಲ್ಲಾ ವಿಷಯವನ್ನು ನಕಲಿಸಿ
3. ನಿಮ್ಮ AI ನ system prompt ನಲ್ಲಿ ಅಂಟಿಸಿ
4. ಸಂಭಾಷಣೆ ಪ್ರಾರಂಭಿಸಿ

### ವಿಧಾನ 2: ಸರ್ವರ್ ಚಲಾಯಿಸಿ

```bash
pip install -e ".[server]"
python -m uvicorn server:app --host 0.0.0.0 --port 8000
```

### ವಿಧಾನ 3: Python Code ನಲ್ಲಿ ಅಳವಡಿಸಿ

```python
from lmm.dharma import DharmaLMM
model = DharmaLMM(k=15, use_sparse_graph=True, use_ising_sa=True)
model.fit(reference_data)
result = model.select_dharma(candidates)
```

---

## 8 Reasoning Modes

| Mode | ಬೌದ್ಧ ಪರಿಕಲ್ಪನೆ | ಯಾವಾಗ |
|------|--------------|-------|
| adaptive | ಉಪಾಯ ಕೌಶಲ್ಯ | Complexity < 0.3 |
| theoretical | ಹೇತುವಿದ್ಯಾ | Complexity 0.3–0.6 |
| hyper | ಪ್ರಜ್ಞಾದ ಜಿಗಿತ | Complexity > 0.6 |
| active | ಭಿಕ್ಷಾಟನ | ಬಾಹ್ಯ ಜ್ಞಾನ ಅಗತ್ಯ |
| alaya | ಆಲಯ-ವಿಜ್ಞಾನ | ಸ್ಮೃತಿ ಹುಡುಕಾಟ |
| sleep | ಧ್ಯಾನ | ನಿಷ್ಕ್ರಿಯ ಏಕೀಕರಣ |
| embodied | ಷಡಾಯತನ | Multimodal input |
| pineal | ಚೇತನದ ಕೇಂದ್ರ | ನಿರ್ಧಾರರಹಿತ ಅನ್ವೇಷಣೆ |

---

## ಸೈದ್ಧಾಂತಿಕ ಆಧಾರ

- **ಕರುಣಾ** = Supermodular function
- **ಶೀಲ** = Submodular function
- **ಮಧ್ಯಮ ಮಾರ್ಗ** = CV = 0.5
- **ಪ್ರತೀತ್ಯಸಮುತ್ಪಾದ** = RAG ನಲ್ಲಿ ಕಾರ್ಯಕಾರಣ ಸ್ಕೋರಿಂಗ್
- **ಆಲಯ-ವಿಜ್ಞಾನ** = Modern Hopfield associative memory

---

## ಅಳತೆಗಳು

| Component | ಅಳೆದಿದ್ದು | ಅರ್ಥ |
|-----------|---------|------|
| HeartbeatDaemon 1 tick | 0.077ms | **0.08% CPU** |
| AlayaMemory recall | 0.09ms | ಸ್ಮೃತಿ ಹುಡುಕಾಟ |
| FEP ODE (n=50) | 3.9ms | reasoning step ವೆಚ್ಚ |

---

## Dependencies

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
pip install maturin
cd lmm_rust_core && maturin develop --release
pip install numpy>=1.24 scipy>=1.10
pip install -e ".[server]"
```

MIT ಪರವಾನಗಿ — ಬದಲಾಯಿಸಿ, ವಾಣಿಜ್ಯಿಕವಾಗಿ ಬಳಸಿ — ಎಲ್ಲವೂ ಮುಕ್ತ.
