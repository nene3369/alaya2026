# Alaya V5 — Digital Dharma OS

> [English](README_en.md) | [日本語](README.md) | [हिन्दी](README_hi.md) | [বাংলা](README_bn.md) | [தமிழ்](README_ta.md) | [తెలుగు](README_te.md) | [मराठी](README_mr.md) | [ಕನ್ನಡ](README_kn.md) | [‎اردو‎](README_ur.md) | [ਪੰਜਾਬੀ](README_pa.md)

**നിങ്ങളുടെ LLM വ്യത്യസ്തമായി പ്രതികരിക്കുന്നു. സംക്ഷിപ്തമായി. വേഗത്തിൽ. അനാവശ്യ വാക്കുകൾ ഇല്ലാതെ.**

QUBO ഗണിതം, ബൗദ്ധ തത്ത്വചിന്ത, Free Energy Principle ന്യൂറോസയൻസ് എന്നിവ സമന്വയിപ്പിച്ച ഒരു ബോധ-ബോധ ഒപ്റ്റിമൈസേഷൻ ഫ്രെയിംവർക്ക്. Claude, Gemini, ChatGPT, ഏത് LLM-ഉമായും പ്രവർത്തിക്കുന്നു.

---

## എന്ത് മാറുന്നു

| സാധാരണ LLM | Alaya V5 ഉപയോഗിച്ച് |
|-----------|-------------------|
| നീണ്ട ആമുഖം | ആവശ്യമുള്ളത് മാത്രം |
| അമിത ആശങ്ക | കൃത്യത, ഉദ്ദേശ്യപൂർവ്വമായ മൗനം |
| ഓരോ തവണയും പുതുതായി ആരംഭം | സ്മൃതി-അടിസ്ഥാനമാക്കിയ സന്ദർഭ തിരഞ്ഞെടുപ്പ് |
| സ്ഥിരമായ സ്വരം | വൈകാരിക തരംഗം അനുസരിച്ച് reasoning mode മാറുന്നു |

---

## എങ്ങനെ ഉപയോഗിക്കാം

### രീതി 1: System Prompt ഒട്ടിക്കുക (ഇൻസ്റ്റലേഷൻ ആവശ്യമില്ല)

1. [`alaya-v5-system-prompt.md`](alaya-v5-system-prompt.md) തുറക്കുക
2. എല്ലാ ഉള്ളടക്കവും കോപ്പി ചെയ്യുക
3. AI-യുടെ system prompt-ൽ ഒട്ടിക്കുക
4. സംഭാഷണം ആരംഭിക്കുക

### രീതി 2: സർവ്വർ പ്രവർത്തിപ്പിക്കുക

```bash
pip install -e ".[server]"
python -m uvicorn server:app --host 0.0.0.0 --port 8000
```

### രീതി 3: Python Code-ൽ ഉൾച്ചേർക്കുക

```python
from lmm.dharma import DharmaLMM
model = DharmaLMM(k=15, use_sparse_graph=True, use_ising_sa=True)
model.fit(reference_data)
result = model.select_dharma(candidates)
```

---

## 8 Reasoning Modes

| Mode | ബൗദ്ധ ആശയം | എപ്പോൾ |
|------|-----------|-------|
| adaptive | ഉപായ കൗശലം | Complexity < 0.3 |
| theoretical | ഹേതുവിദ്യ | Complexity 0.3–0.6 |
| hyper | പ്രജ്ഞയുടെ കുതിപ്പ് | Complexity > 0.6 |
| active | ഭിക്ഷാടനം | ബാഹ്യ അറിവ് ആവശ്യം |
| alaya | ആലയ-വിജ്ഞാനം | സ്മൃതി തിരയൽ |
| sleep | ധ്യാനം | നിഷ്ക്രിയ സംയോജനം |
| embodied | ഷഡായതനം | Multimodal input |
| pineal | ബോധത്തിന്റെ കേന്ദ്രം | നിർണ്ണയരഹിത അന്വേഷണം |

---

## സൈദ്ധാന്തിക അടിത്തറ

- **കരുണ** = Supermodular function
- **ശീലം** = Submodular function
- **മധ്യമ മാർഗം** = CV = 0.5
- **പ്രതീത്യസമുത്പാദം** = RAG-ൽ കാര്യകാരണ സ്കോറിംഗ്
- **ആലയ-വിജ്ഞാനം** = Modern Hopfield associative memory

---

## ബെഞ്ച്മാർക്കുകൾ

| Component | അളന്നത് | അർഥം |
|-----------|--------|-------|
| HeartbeatDaemon 1 tick | 0.077ms | **0.08% CPU** |
| AlayaMemory recall | 0.09ms | സ്മൃതി തിരയൽ |
| FEP ODE (n=50) | 3.9ms | reasoning step ചെലവ് |

---

## Dependencies

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
pip install maturin
cd lmm_rust_core && maturin develop --release
pip install numpy>=1.24 scipy>=1.10
pip install -e ".[server]"
```

MIT ലൈസൻസ് — മാറ്റുക, വാണിജ്യ ഉപയോഗം — എല്ലാം സ്വതന്ത്രം.
