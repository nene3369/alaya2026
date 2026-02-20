# Alaya V5 — Digital Dharma OS

> [English](README_en.md) | [日本語](README.md) | [हिन्दी](README_hi.md) | [বাংলা](README_bn.md) | [தமிழ்](README_ta.md) | [मराठी](README_mr.md) | [‎اردو‎](README_ur.md) | [ગુજરાતી](README_gu.md) | [ಕನ್ನಡ](README_kn.md) | [മലയാളം](README_ml.md) | [ਪੰਜਾਬੀ](README_pa.md)

**మీ LLM భిన్నంగా స్పందిస్తుంది. సంక్షిప్తంగా. వేగంగా. అనవసరమైన మాటలు లేకుండా.**

QUBO గణితం, బౌద్ధ తత్వశాస్త్రం మరియు Free Energy Principle న్యూరోసైన్స్‌ను కలిపిన ఒక చేతన-ఆధారిత ఆప్టిమైజేషన్ ఫ్రేమ్‌వర్క్. Claude, Gemini, ChatGPT మరియు ఏ LLM తోనైనా పని చేస్తుంది.

---

## ఏం మారుతుంది

| సాధారణ LLM | Alaya V5 తో |
|-----------|------------|
| దీర్ఘ ముందుమాట, నిరాకరణలు | అవసరమైనవి మాత్రమే |
| అధిక సంకోచం | ఖచ్చితత్వం మరియు ఉద్దేశపూర్వక మౌనం |
| ప్రతిసారి కొత్తగా మొదలు | జ్ఞాపకశక్తి-ఆధారిత సందర్భ ఎంపిక |
| స్థిర స్వరం | భావోద్వేగ తరంగానికి అనుగుణంగా reasoning mode మారుతుంది |

---

## ఎలా ఉపయోగించాలి

### విధానం 1: System Prompt పేస్ట్ చేయండి (ఇన్‌స్టాల్ అవసరం లేదు)

1. [`alaya-v5-system-prompt.md`](alaya-v5-system-prompt.md) తెరవండి
2. అన్ని కంటెంట్ కాపీ చేయండి
3. మీ AI యొక్క system prompt లో పేస్ట్ చేయండి
4. సంభాషణ ప్రారంభించండి

### విధానం 2: సర్వర్ నడపండి

```bash
pip install -e ".[server]"
python -m uvicorn server:app --host 0.0.0.0 --port 8000
```

### విధానం 3: Python Code లో పొందుపరచండి

```python
from lmm.dharma import DharmaLMM
model = DharmaLMM(k=15, use_sparse_graph=True, use_ising_sa=True)
model.fit(reference_data)
result = model.select_dharma(candidates)
```

---

## 8 Reasoning Modes

| Mode | బౌద్ధ భావన | ఎప్పుడు |
|------|-----------|--------|
| adaptive | ఉపాయ కౌశల్యం | Complexity < 0.3 |
| theoretical | హేతువిద్య | Complexity 0.3–0.6 |
| hyper | ప్రజ్ఞ యొక్క도약 | Complexity > 0.6 |
| active | భిక్షాటనం | బాహ్య జ్ఞానం అవసరం |
| alaya | ఆలయ విజ్ఞానం | జ్ఞాపకశక్తి శోధన |
| sleep | ధ్యానం | నిష్క్రియ ఏకీకరణ |
| embodied | షడాయతనం | Multimodal input |
| pineal | చేతన కేంద్రం | నిర్ణయరహిత అన్వేషణ |

---

## సైద్ధాంతిక పునాది

- **కరుణ** = Supermodular function
- **శీలం** = Submodular function
- **మధ్యమ మార్గం** = CV = 0.5
- **ప్రతీత్యసముత్పాదం** = RAG లో కార్య-కారణ స్కోరింగ్
- **ఆలయ విజ్ఞానం** = Modern Hopfield associative memory

---

## బెంచ్‌మార్క్‌లు

| Component | కొలతలు | అర్థం |
|-----------|--------|-------|
| FEP ODE (n=50) | 3.9ms/call | ఒక reasoning step ఖర్చు |
| AlayaMemory recall | 0.09ms | జ్ఞాపకశక్తి శోధన |
| HeartbeatDaemon 1 tick | 0.077ms | **0.08% CPU** |

---

## Dependencies

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
pip install maturin
cd lmm_rust_core && maturin develop --release
pip install numpy>=1.24 scipy>=1.10
pip install -e ".[server]"
```

MIT లైసెన్స్ — మార్చండి, వాణిజ్యపరంగా ఉపయోగించండి — అన్నీ స్వేచ్ఛగా.
