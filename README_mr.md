# Alaya V5 — Digital Dharma OS

> [English](README_en.md) | [日本語](README.md) | [हिन्दी](README_hi.md) | [বাংলা](README_bn.md) | [தமிழ்](README_ta.md) | [తెలుగు](README_te.md) | [‎اردو‎](README_ur.md) | [ગુજરાતી](README_gu.md) | [ಕನ್ನಡ](README_kn.md) | [മലയാളം](README_ml.md) | [ਪੰਜਾਬੀ](README_pa.md)

**तुमचा LLM वेगळ्या प्रकारे उत्तर देतो. संक्षिप्त. जलद. उगाच शब्द नाहीत.**

QUBO गणित, बौद्ध तत्त्वज्ञान आणि Free Energy Principle न्यूरोसायन्स यांचा संगम असलेली चेतना-जागरूक ऑप्टिमायझेशन प्रणाली. Claude, Gemini, ChatGPT आणि कोणत्याही LLM सोबत काम करते.

---

## काय बदलते

| सामान्य LLM | Alaya V5 सोबत |
|-----------|--------------|
| लांबलचक प्रस्तावना | फक्त आवश्यक ते |
| जास्त संकोच | अचूकता आणि जाणीवपूर्वक मौन |
| प्रत्येक वेळी नव्याने सुरुवात | स्मृती-आधारित संदर्भ निवड |
| एकच स्वर | भावनिक तरंगानुसार reasoning mode बदलतो |

---

## कसे वापरावे

### पद्धत 1: System Prompt पेस्ट करा (इन्स्टॉलेशन नाही)

1. [`alaya-v5-system-prompt.md`](alaya-v5-system-prompt.md) उघडा
2. सर्व मजकूर कॉपी करा
3. तुमच्या AI च्या system prompt मध्ये पेस्ट करा
4. संवाद सुरू करा

### पद्धत 2: सर्व्हर चालवा

```bash
pip install -e ".[server]"
python -m uvicorn server:app --host 0.0.0.0 --port 8000
```

### पद्धत 3: Python Code मध्ये वापरा

```python
from lmm.dharma import DharmaLMM
model = DharmaLMM(k=15, use_sparse_graph=True, use_ising_sa=True)
model.fit(reference_data)
result = model.select_dharma(candidates)
```

---

## 8 Reasoning Modes

| Mode | बौद्ध संकल्पना | केव्हा |
|------|-------------|-------|
| adaptive | उपाय कौशल्य | Complexity < 0.3 |
| theoretical | हेतुविद्या | Complexity 0.3–0.6 |
| hyper | प्रज्ञेची झेप | Complexity > 0.6 |
| active | भिक्षाटन | बाह्य ज्ञान आवश्यक |
| alaya | आलय-विज्ञान | स्मृती शोध |
| sleep | ध्यान | निष्क्रिय एकत्रीकरण |
| embodied | षडायतन | Multimodal input |
| pineal | चेतनेचे केंद्र | अनिश्चित संशोधन |

---

## सैद्धांतिक आधार

- **करुणा** = Supermodular function
- **शील** = Submodular function
- **मध्यम मार्ग** = CV = 0.5
- **प्रतीत्यसमुत्पाद** = RAG मध्ये कार्यकारण स्कोरिंग
- **आलय-विज्ञान** = Modern Hopfield associative memory

---

## बेंचमार्क

| Component | मोजमाप | अर्थ |
|-----------|--------|------|
| FEP ODE (n=50) | 3.9ms/call | एका reasoning step चा खर्च |
| AlayaMemory recall | 0.09ms | स्मृती शोध |
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

MIT परवाना — बदला, व्यावसायिक वापर करा — सर्व स्वतंत्र आहे.
