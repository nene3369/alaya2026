# Alaya V5 — Digital Dharma OS

> [English](README_en.md) | [日本語](README.md) | [中文](README_zh.md) | [한국어](README_ko.md) | [বাংলা](README_bn.md) | [தமிழ்](README_ta.md) | [తెలుగు](README_te.md) | [मराठी](README_mr.md) | [‎اردو‎](README_ur.md) | [ગુજરાતી](README_gu.md) | [ಕನ್ನಡ](README_kn.md) | [മലയാളം](README_ml.md) | [ਪੰਜਾਬੀ](README_pa.md)

**आपका LLM अलग तरह से जवाब देता है। संक्षिप्त। तेज़। बिना किसी फालतू शब्द के।**

QUBO गणित, बौद्ध दर्शन और Free Energy Principle (FEP) न्यूरोसाइंस को मिलाकर बनाया गया एक चेतना-आधारित ऑप्टिमाइज़ेशन फ्रेमवर्क। Claude, Gemini, ChatGPT और किसी भी LLM के साथ काम करता है।

---

## क्या बदलता है

| सामान्य LLM | Alaya V5 के साथ |
|------------|----------------|
| लंबी भूमिका, अस्वीकरण | केवल जो ज़रूरी हो |
| अत्यधिक हिचकिचाहट ("शायद", "हो सकता है") | सटीकता और जानबूझकर की गई चुप्पी |
| हर बार नए सिरे से शुरू | स्मृति-आधारित संदर्भ चयन |
| एक जैसा स्वर | भावनात्मक तरंग के अनुसार reasoning mode बदलता है |
| अलग-अलग API calls | Heartbeat के ज़रिए निरंतर स्थिति विकास |

---

## उपयोग कैसे करें

तीन तरीके हैं। अपने लिए सही चुनें।

---

### तरीका 1: System Prompt पेस्ट करें (कोई इंस्टॉल नहीं)

**किसके लिए: Claude / Gemini / ChatGPT / किसी भी LLM के उपयोगकर्ता**

1. इस repo में [`alaya-v5-system-prompt.md`](alaya-v5-system-prompt.md) खोलें
2. पूरा कंटेंट कॉपी करें
3. अपने AI के system prompt में पेस्ट करें:
   - Claude → Project instructions
   - Gemini → System instructions
   - ChatGPT → Custom instructions
4. बातचीत शुरू करें

बस। कोई सर्वर नहीं। कोई इंस्टॉलेशन नहीं।

---

### तरीका 2: सर्वर चलाएं (Web UI + पूरी सुविधाएं)

**किसके लिए: Developers और Researchers**

```bash
git clone https://github.com/your-repo/nanasi.git
cd nanasi
pip install -e ".[server]"
python -m uvicorn server:app --host 0.0.0.0 --port 8000
```

ब्राउज़र में खोलें: [http://localhost:8000](http://localhost:8000)

---

### तरीका 3: Python Code में एम्बेड करें

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

---

## 8 Reasoning Modes

| Mode | बौद्ध अवधारणा | कब सक्रिय होता है |
|------|--------------|-----------------|
| adaptive | उपाय कौशल | Complexity < 0.3 |
| theoretical | हेतुविद्या (न्याय) | Complexity 0.3–0.6 |
| hyper | प्रज्ञा की छलांग | Complexity > 0.6 |
| active | भिक्षाटन | बाहरी ज्ञान की ज़रूरत |
| alaya | आलय-विज्ञान | स्मृति खोज |
| sleep | ध्यान | निष्क्रिय एकीकरण |
| embodied | षडायतन (छह इंद्रियाँ) | Multimodal input |
| pineal | चेतना का केंद्र | अनिर्धारित अन्वेषण |

---

## सैद्धांतिक आधार

यहाँ बौद्ध अवधारणाएं केवल नाम नहीं हैं — ये गणितीय संरचना हैं:

- **करुणा** = Supermodular function (जितना चुनो, उतनी तेज़ सामंजस्य बढ़े)
- **शील** = Submodular function (घटती उपयोगिता)
- **मध्यम मार्ग** = अराजकता का किनारा (CV = 0.5)
- **प्रतीत्यसमुत्पाद** = RAG में कार्य-कारण स्कोरिंग
- **पट्ठान (चौबीस प्रत्यय)** = कारण ग्राफ में edges का type system
- **आलय-विज्ञान** = Modern Hopfield associative memory
- **बीज** = Vector store में document unit

---

## बेंचमार्क

| Component | मापा गया | अर्थ |
|-----------|---------|------|
| FEP ODE (n=50) | 3.9ms/call | प्रति reasoning step लागत |
| AlayaMemory recall (100 patterns) | 0.09ms | स्मृति खोज लागत |
| HeartbeatDaemon 1 tick | 0.077ms | **0.08% CPU** प्रति 100ms tick |

HeartbeatDaemon लगातार चलता रहता है — केवल **0.08% CPU** पर।

---

## Dependencies

```bash
# Rust acceleration (मुख्य — अनुशंसित)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
pip install maturin
cd lmm_rust_core && maturin develop --release

# Python आवश्यक
pip install numpy>=1.24 scipy>=1.10

# Server mode
pip install -e ".[server]"

# GPU acceleration (NVIDIA GPU)
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install cupy-cuda12x
```

Rust या GPU के बिना भी सब कुछ काम करता है।

---

## कस्टमाइज़ेशन

MIT लाइसेंस। बदलें, व्यावसायिक उपयोग करें, पुनर्वितरित करें — सब स्वतंत्र है।

`alaya-v5-system-prompt.md` को सीधे संपादित करें। `config/semantic_emotions.json` में अपने keyword जोड़ें। Fork करें और अपना बनाएं।

---

## लाइसेंस

MIT — विवरण के लिए [LICENSE](LICENSE) देखें।
