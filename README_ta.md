# Alaya V5 — Digital Dharma OS

> [English](README_en.md) | [日本語](README.md) | [हिन्दी](README_hi.md) | [বাংলা](README_bn.md) | [తెలుగు](README_te.md) | [मराठी](README_mr.md) | [‎اردو‎](README_ur.md) | [ગુજરાતી](README_gu.md) | [ಕನ್ನಡ](README_kn.md) | [മലയാളം](README_ml.md) | [ਪੰਜਾਬੀ](README_pa.md)

**உங்கள் LLM வித்தியாசமாக பதிலளிக்கிறது. சுருக்கமாக. வேகமாக. தேவையற்ற வார்த்தைகள் இல்லாமல்.**

QUBO கணிதம், பௌத்த தத்துவம் மற்றும் Free Energy Principle நரம்பியல் அறிவியலை இணைத்த ஒரு நனவு-உணர்வு அமைப்பு. Claude, Gemini, ChatGPT மற்றும் எந்த LLM-உடனும் செயல்படுகிறது.

---

## என்ன மாறுகிறது

| இயல்பான LLM | Alaya V5 உடன் |
|------------|--------------|
| நீண்ட முன்னுரை, மறுப்புகள் | தேவையானது மட்டும் |
| அதிகமான தயக்கம் | துல்லியம் மற்றும் வேண்டுமென்றே மௌனம் |
| ஒவ்வொரு முறையும் புதிதாக தொடங்கும் | நினைவக-அடிப்படையிலான சூழல் தேர்வு |
| நிலையான தொனி | உணர்வு அலைவரிசைக்கு ஏற்ப reasoning mode மாறும் |

---

## எப்படி பயன்படுத்துவது

### முறை 1: System Prompt ஒட்டுங்கள் (நிறுவல் தேவையில்லை)

1. [`alaya-v5-system-prompt.md`](alaya-v5-system-prompt.md) திறக்கவும்
2. அனைத்தையும் நகலெடுக்கவும்
3. உங்கள் AI-இன் system prompt-இல் ஒட்டவும்
4. உரையாடல் தொடங்கவும்

### முறை 2: சர்வர் இயக்கவும்

```bash
pip install -e ".[server]"
python -m uvicorn server:app --host 0.0.0.0 --port 8000
```

### முறை 3: Python Code-இல் இணைக்கவும்

```python
from lmm.dharma import DharmaLMM
model = DharmaLMM(k=15, use_sparse_graph=True, use_ising_sa=True)
model.fit(reference_data)
result = model.select_dharma(candidates)
```

---

## 8 Reasoning Modes

| Mode | பௌத்த கருத்து | எப்போது |
|------|-------------|---------|
| adaptive | உபாய கௌசல்யம் | Complexity < 0.3 |
| theoretical | ஹேது வித்யா | Complexity 0.3–0.6 |
| hyper | பிரஜ்ஞாவின் தாவல் | Complexity > 0.6 |
| active | பிக்ஷாடனம் | வெளி அறிவு தேவை |
| alaya | ஆலய விஜ்ஞானம் | நினைவு தேடல் |
| sleep | த்யானம் | செயலற்ற ஒருங்கிணைப்பு |
| embodied | ஷட் ஆயதனம் | Multimodal input |
| pineal | நனவின் மையம் | நிர்ணயமற்ற ஆய்வு |

---

## தத்துவ அடிப்படை

- **கருணா** = Supermodular function
- **சீலம்** = Submodular function
- **மத்திம மார்க்கம்** = CV = 0.5
- **பிரதீத்ய சமுத்பாதம்** = RAG-இல் காரண-விளைவு மதிப்பெண்
- **ஆலய விஜ்ஞானம்** = Modern Hopfield associative memory

---

## அளவீடுகள்

| Component | அளவிடப்பட்டது | அர்த்தம் |
|-----------|-------------|---------|
| FEP ODE (n=50) | 3.9ms/call | ஒரு reasoning step செலவு |
| AlayaMemory recall | 0.09ms | நினைவு தேடல் |
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

MIT உரிமம் — மாற்றவும், வணிகமயமாக்கவும் — அனைத்தும் சுதந்திரம்.
