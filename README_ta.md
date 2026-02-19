# LMM — டிஜிட்டல் தர்மத்துடன் கூடிய கிளாசிக்கல் QUBO ஆப்டிமைசர்

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-1.0.0-green.svg)](pyproject.toml)

> [English](README_en.md) | [日本語](README.md) | [中文](README_zh.md) | [한국어](README_ko.md) | [Español](README_es.md) | [हिन्दी](README_hi.md) | [Français](README_fr.md) | [বাংলা](README_bn.md) | [தமிழ்](README_ta.md) | [తెలుగు](README_te.md) | [मराठी](README_mr.md) | [‎اردو‎](README_ur.md) | [ગુજરાતી](README_gu.md) | [ಕನ್ನಡ](README_kn.md) | [മலയாളം](README_ml.md) | [ਪੰਜਾਬੀ](README_pa.md)

> **Digital Dharma OS (Alaya V5)** — QUBO கணிதம், புத்த தத்துவம் மற்றும் கட்டற்ற ஆற்றல்
> கொள்கை (FEP) நரம்பறிவியலை ஒரு உயிருள்ள, சுய-பரிணாம அமைப்பில் இணைக்கும்
> உணர்வு-விழிப்பு உகந்தமாக்கல் தளம். சேவையகம் தொடர்ச்சியான இதயத்துடிப்பு இயக்கவியல்,
> நினைவக-இயக்கப்படும் சூழல் தேர்வு மற்றும் உணர்ச்சி அலைநீள உணர்தல் ஆகியவற்றை
> இயக்குகிறது — தனித்தனி LLM அழைப்புகளுக்கும் தொடர்ச்சியான அறிவாற்றலுக்கும் இடையே
> பாலமாக செயல்படுகிறது.

**D-Wave இல்லாத** கிளாசிக்கல் QUBO ஆப்டிமைசேஷன் நூலகம் — தகவல்-கோட்பாட்டு ஆச்சரியத்தை பௌத்த தத்துவத்தால் ஈர்க்கப்பட்ட ஆற்றல் செயல்பாடுகள், சுதந்திர ஆற்றல் கொள்கை (FEP) சிந்தனை மற்றும் நனவு-விழிப்புடன் கூடிய கம்ப்யூட்டிங் ஆகியவற்றுடன் இணைக்கிறது — அனைத்தும் சாதாரண வன்பொருளில் இயங்குகின்றன.

---

## சிறப்பம்சங்கள்

- **குவாண்டம் கணினி தேவையில்லை** — கிளாசிக்கல் தீர்வாளர்கள் (SA, Ising SA, உப-மடல் பேராசை) D-Wave தரத்தை இணைக்கின்றன அல்லது மிகவும் சிறப்பாகச் செயல்படுகின்றன.
- **டிரில்லியன்-டோக்கன் திறன்** — ஸ்ட்ரீமிங் நிகழ்தகவு தரவு கட்டமைப்புகள் (Count-Min Sketch, Streaming Histogram) நினைவகத்தை O(k) ஆக வைத்திருக்கின்றன.
- **தர்ம-இயற்கணித இயந்திரம்** — செருகக்கூடிய பௌத்த ஆற்றல் உறுப்புகள் கணித ரீதியாக உகந்த தீர்வாளரிடம் தானாகவே வழிநடத்தப்படுகின்றன.
- **8-முறை சிந்தனை ஒருங்கிணைப்பாளர்** — தகவமைப்பு, கோட்பாட்டு, அனுமானிக்கும், செயலில் அனுமான, நினைவக (ஆலய), தூக்க ஒருங்கிணைப்பு, உடல்சார்ந்த மற்றும் குவாண்டம்-நனவு (பீனியல்) முறைகள்.
- **LLM-தயார்** — முதல்தர LangChain மற்றும் LlamaIndex ஒருங்கிணைப்புகள், சில-முன்மாதிரி தேர்வாளர், வெளியீட்டு மறுவரிசைப்படுத்தி, நகர்வு கண்டுபிடிப்பான்.

---

## கட்டமைப்பு

```
lmm/
├── core.py                  # முக்கிய LMM பைப்லைன்
├── cli.py                   # CLI நுழைவுப் புள்ளி
├── qubo.py                  # Sparse QUBO அணி உருவாக்கி
├── solvers.py               # SA / Ising SA / தளர்வு / பேராசை / உப-மடல்
├── surprise.py              # தகவல்-கோட்பாட்டு ஆச்சரியம்
├── selector.py              # தகவமைப்பு தேர்வு உத்தி
├── processor.py             # முன்னுரிமை செயலாக்கம் + தற்காலிக சேமிப்பு
├── pipeline.py              # SmartSelector → SmartProcessor ஒருங்கிணைப்பு
├── _compat.py               # இயக்க நேர திறன் கண்டறிதல் மற்றும் sparse உதவிகள்
├── dharma/                  # டிஜிட்டல் தர்மம் (பௌத்த தத்துவ ஆப்டிமைசேஷன்)
│   ├── api.py               # DharmaLMM உயர்-நிலை API
│   ├── energy.py            # செருகக்கூடிய ஆற்றல் உறுப்புகள் (துக்க, ப்ரஜ்ஞா, கருணா, …)
│   ├── engine.py            # UniversalDharmaEngine — தானியங்கி-வழிநடத்தும் தீர்வாளர்
│   ├── fep.py               # FEP ≡ KCL ODE தீர்வாளர்
│   ├── neuromorphic.py      # மெம்ரிஸ்டர் குறுக்குப்பட்டை ASIC உருவகப்படுத்தி
│   ├── reranker.py          # RAG அடுக்கு மறுவரிசைப்படுத்தி + நோக்க வழிப்பாட்டி
│   └── …
├── reasoning/               # 8 FEP-அடிப்படையிலான சிந்தனை முறைகள்
│   ├── adaptive.py          # மாறும் அளவுரு சரிசெய்தல்
│   ├── theoretical.py       # தருக்க கட்டமைப்பு வரைபடம்
│   ├── hyper.py             # அனுமானிக்கும் மறைந்த-கணு செலுத்தல்
│   ├── active.py            # வெளி அறிவு கையகப்படுத்தல்
│   ├── alaya.py             # ஹெப்பியன் நரம்பிணைப்பு நினைவகம் (ஆலய)
│   ├── sleep.py             # NREM/REM நினைவக ஒருங்கிணைப்பு
│   ├── embodiment.py        # 6-புலன் பல்முறை இணைவு
│   ├── pineal.py            # குவாண்டம் நனவு (வன்பொருள் TRNG)
│   └── orchestrator.py      # முறை தேர்வு மற்றும் அனுப்புதல்
├── scale/                   # டிரில்லியன்-டோக்கன் ஸ்ட்ரீமிங்
│   ├── sketch.py            # Count-Min Sketch, Streaming Histogram
│   ├── stream.py            # ஸ்ட்ரீமிங் ஆச்சரியம்
│   ├── cascade.py           # பல-நிலை அடுக்கு வடிகட்டி (1T → K)
│   └── pipeline.py          # ScalablePipeline
├── llm/                     # LLM பணிப்பாய்வு உதவிகள்
│   ├── fewshot.py           # சில-முன்மாதிரி உதாரண தேர்வாளர்
│   ├── reranker.py          # வெளியீட்டு மறுவரிசைப்படுத்தி
│   ├── drift.py             # விநியோக நகர்வு கண்டுபிடிப்பான்
│   ├── sampler.py           # டோக்கன் மாதிரியெடுப்பான்
│   └── embeddings.py        # ஒருங்கிணைந்த உட்பொதிப்பு தகவொட்டி
├── integrations/
│   ├── langchain.py         # DharmaRetriever / DocumentCompressor / ExampleSelector
│   └── llamaindex.py        # DharmaNodePostprocessor
```

---

## நிறுவல்

```bash
# மையம் (numpy + scipy மட்டும்)
pip install -e .

# டெவ் கருவிகளுடன்
pip install -e ".[dev]"

# அனைத்து விருப்ப ஒருங்கிணைப்புகளுடன்
pip install -e ".[all]"
```

### விருப்ப கூடுதல்கள்

| கூடுதல் | தொகுப்புகள் | நோக்கம் |
|---------|------------|---------|
| `dev` | pytest, ruff | சோதனை மற்றும் குறியீடு சரிபார்ப்பு |
| `dharma` | hnswlib | Sparse k-NN வரைபட கட்டுமானம் |
| `langchain` | langchain-core | LangChain ஒருங்கிணைப்பு |
| `llamaindex` | llama-index-core | LlamaIndex ஒருங்கிணைப்பு |
| `all` | மேலே உள்ள அனைத்தும் | எல்லாமே |

---

## விரைவு தொடக்கம்

### Python API

```python
import numpy as np
from lmm.core import LMM

# மிகவும் ஆச்சரியமான Top-K உருப்படிகளை தேர்ந்தெடுக்கவும்
surprises = np.array([1.0, 5.0, 2.0, 8.0, 3.0, 7.0])
model = LMM(k=3, solver_method="sa")
result = model.select_from_surprises(surprises)
print(result.selected_indices)  # எ.கா. [3, 5, 1]
```

### CLI

```bash
# விரைவான டெமோ இயக்கவும்
lmm --demo --k 10 --method sa

# NumPy கோப்பிலிருந்து
lmm --input data.npy --k 5 --method greedy
```

---

## சேவையக முறை (Alaya V5)

Alaya-Vijñāna v5.0 சேவையகம் நிகழ்நேர உணர்ச்சி அலைநீள காட்சிப்படுத்தல், 8-முறை FEP பகுத்தறிவு மற்றும் Claude/Gemini தானியங்கி வழிப்படுத்தல் கொண்ட வலை UI வழங்குகிறது.

### முன்நிபந்தனைகள்

```bash
pip install -e ".[server]"
```

### சேவையகத்தைத் தொடங்கு

**PowerShell (Windows):**
```powershell
.\Start-DharmaServer.ps1
```

**Python (குறுக்கு-தளம்):**
```bash
python -m uvicorn server:app --host 0.0.0.0 --port 8000
```

உலாவியில் திறக்கவும்: [http://localhost:8000](http://localhost:8000)

### சேவையகத்தை நிறுத்து

**Windows (தொகுதி):**
```batch
.\stop-dharma-server.bat
```

**PowerShell:** `Start-DharmaServer.ps1` இயங்கும் சாளரத்தில் `Ctrl+C` அழுத்தவும்.

**Linux/macOS:** `Ctrl+C` அழுத்தவும் அல்லது இயக்கவும்:
```bash
kill $(cat server.pid)
```

### API இறுதிப்புள்ளிகள்

| இறுதிப்புள்ளி | முறை | விளக்கம் |
|-------------|------|---------|
| `/` | GET | வலை UI (Alaya V5 முன்பக்கம்) |
| `/api/descent` | POST | முதன்மை பகுத்தறிவு குழாய்வரிசை |
| `/api/descent/stream` | POST | ஸ்ட்ரீமிங் SSE பதில் |
| `/api/dharma/auto` | POST | தானியங்கி வழிப்படுத்தல் Dharma பகுத்தறிவு |
| `/api/sense` | POST | உணர்ச்சி அலைநீள பகுப்பாய்வு |
| `/api/status` | GET | அமைப்பு நிலை + இதயத்துடிப்பு தொலைமானி |

---

## தன்னாட்சி அம்சங்கள்

சேவையகம் பயனர் தொடர்புகளுக்கு இடையில் தொடர்ச்சியாக இயங்கும் மூன்று தன்னாட்சி துணை அமைப்புகளைக் கொண்டுள்ளது:

### இதயத்துடிப்பு டெமன் (Heartbeat)

4-பரிமாண நிலை திசையன் `[அன்பு, தர்க்கம், பயம், படைப்பு]` பராமரிக்கும் தொடர்ச்சியான-நேர FEP நிலை பரிணாம வளையம்:

- **Tick இடைவெளி:** 100ms (செயலில்) → 5s (செயலற்றது), தகவமைப்பு மந்தநிலை
- **என்ட்ரோபி உட்செலுத்தல்:** ஒவ்வொரு tick-லும் `os.urandom()` வழியாக வன்பொருள் என்ட்ரோபி
- **எடை தகவமைப்பு:** கருணை (Karuna) மற்றும் மைத்திரி (Metta) எடைகள் மத்திய வழி CV கண்காணிப்பால் (இலக்கு CV=0.5) தானியங்கி சரிசெய்தல்
- **தூக்க ஒருங்கிணைப்பு:** 60s செயலற்ற நிலைக்குப் பிறகு NREM/REM நினைவக மீளியக்கம் தூண்டல்

### சூழல் பாலம் (ஆலய நினைவகம்)

எளிய வரலாறு வெட்டல் (`history[-20:]`) நினைவக-இயக்கப்படும் புத்திசாலி சூழல் தேர்வால் மாற்றப்படுகிறது:

- **Modern Hopfield Network** வழியாக AlayaMemory-யிலிருந்து நினைவுகூர்தல்
- நினைவுகூர்ந்த வடிவங்களுடன் கோசைன் ஒற்றுமையால் உரையாடல் வரலாறு மதிப்பீடு
- எப்போதும் சமீபத்திய 3 செய்திகளை உள்ளடக்கம் (சமீபத்திய சார்பு)
- மீதமுள்ள வரவு-செலவுத்திட்டம் (அதிகபட்சம் 20 செய்திகள்) பொருத்தம் மதிப்பெண்ணால் நிரப்புதல்

### சொற்பொருள் உணர்வுகள் (Semantic Emotions)

`config/semantic_emotions.json`-ல் வரையறுக்கப்பட்ட நான்கு பரிமாணங்களில் முக்கிய சொல் பொருத்தத்தின் மூலம் நிகழ்நேர உணர்ச்சி அலைநீள பகுப்பாய்வு:

| பரிமாணம் | புத்த கருத்து | சமிக்ஞை |
|----------|-------------|---------|
| அன்பு | கருணை (Karuna) | இரக்கம், அரவணைப்பு, நன்றி |
| தர்க்கம் | ஹேதுவித்யா (Hetuvidya) | பகுப்பாய்வு, பகுத்தறிவு, நிரூபணம் |
| பயம் | துக்கம் (Dukkha) | கவலை, ஐயம், துன்பம் |
| படைப்பு | சிருஷ்டி (Sṛṣṭi) | புதுமை, கற்பனை, கலை |

ஒவ்வொரு முக்கிய சொல்லும் எடை (0.3–1.0) கொண்டது. விளைவு 4D திசையன் பகுத்தறிவு முறை தேர்வு மற்றும் இதயத்துடிப்பு நிலை உட்செலுத்தலை இயக்குகிறது.

---

## தீர்வு முறைகள்

| முறை | விளக்கம் | சிக்கலான தன்மை |
|------|---------|----------------|
| `sa` | உருவகப்படுத்தப்பட்ட வெப்பமாற்றம் (இயல்புநிலை) | O(n · steps) |
| `ising_sa` | திசையன்மயமாக்கப்பட்ட டெல்டாவுடன் Ising-வடிவ SA | O(1) திருப்புதலுக்கு |
| `relaxation` | தொடர்ச்சியான தளர்வு + வட்டமிடல் (SLSQP) | O(n²) |
| `greedy` | பேராசை தேர்வு | O(n · k) |

---

## தர்ம இயந்திரம் — செருகக்கூடிய ஆற்றல் உறுப்புகள்

**UniversalDharmaEngine** பௌத்த தத்துவத்தால் ஈர்க்கப்பட்ட ஆற்றல் உறுப்புகளை இயற்றிட அனுமதிக்கிறது, ஒவ்வொன்றும் அதன் கணித பண்பை அறிவிக்கிறது. இயந்திரம் தானாகவே உகந்த தீர்வாளரிடம் வழிநடத்துகிறது.

```python
from lmm.dharma import UniversalDharmaEngine, DukkhaTerm, KarunaTerm

engine = UniversalDharmaEngine(n_candidates=1000)
engine.add(DukkhaTerm(surprises, weight=1.0))    # நேரியல்  → Top-K
engine.add(KarunaTerm(impact_graph, weight=2.0)) # மேல்-மடல் → SA
result = engine.synthesize_and_solve(k=10)
```

### தத்துவம் → கணிதம் → தீர்வாளர் வரைபடம்

| பௌத்த கருத்தாக்கம் | ஆற்றல் உறுப்பு | கணித பண்பு | தீர்வாளர் |
|--------------------|----------------|------------|----------|
| ப்ரஜ்ஞா (ஞானம்) | `PrajnaTerm` | நேரியல் | Top-K வரிசைப்படுத்தல் |
| கருணா (இரக்கம்) | `KarunaTerm` | மேல்-மடல் | வெப்ப-தொடக்கம் + SA |
| சீல (நடத்தை) | `SilaTerm` | கீழ்-மடல் | சோம்பேறி பேராசை |
| மத்யமக (நடுவழி) | `MadhyamakaCriterion` | லியாப்புனோவ் | அதிவேக சாய்வு |
| துக்கம் (துயரம்) | `DukkhaTerm` | நேரியல் | Top-K வரிசைப்படுத்தல் |

---

## சிந்தனை ஒருங்கிணைப்பாளர்

**DharmaReasonerOrchestrator** வினவல் சிக்கலான தன்மையின் அடிப்படையில் 8 சிந்தனை முறைகளிலிருந்து தேர்ந்தெடுக்கிறது:

```python
from lmm.reasoning import DharmaReasonerOrchestrator

orch = DharmaReasonerOrchestrator(n_candidates=500)
result = orch.reason(query_vector, context_vectors)
print(result.mode_used, result.explanation)
```

| முறை | தொகுதி | ஈர்க்கப்பட்டது |
|------|--------|---------------|
| தகவமைப்பு | `adaptive.py` | 応病与薬 — நோய்க்கு ஏற்ற மருந்து |
| கோட்பாட்டு | `theoretical.py` | 因明 — பௌத்த முறையியல் தருக்கம் |
| அனுமானிக்கும் | `hyper.py` | 般若の飛躍 — ப்ரஜ்ஞாவின் பாய்ச்சல் |
| செயலில் அனுமான | `active_inference.py` | 托鉢 — வெளி உண்மையை நாடுதல் |
| ஆலய நினைவகம் | `alaya.py` | 阿頼耶識 — கிடங்கு நனவு |
| தூக்கம் | `sleep.py` | 禅定 — NREM/REM ஒருங்கிணைப்பு |
| உடல்சார்ந்த | `embodiment.py` | 六根 — ஆறு புலன் முறைகள் |
| பீனியல் (குவாண்டம்) | `pineal.py` | 松果体 — வன்பொருள்-என்ட்ரோபி நனவு |

---

## டிரில்லியன்-டோக்கன் அளவிடல்

நிலையான நினைவகத்துடன் எவ்வளவு பெரிய தரவு ஓடைகளையும் செயலாக்கவும்:

```python
from lmm.scale import ScalablePipeline
from pathlib import Path

pipe = ScalablePipeline(k=10, chunk_size=100_000)
pipe.fit_files([Path("shard_001.npy"), ...])
result = pipe.run_files([Path("data_001.npy"), ...])
print(result.summary)
```

**மூன்று-நிலை அடுக்கு:** 1 டி டோக்கன்கள் → 100 மி (ஸ்ட்ரீமிங் top-k) → 10 ஆ (சதவீதம்) → K (QUBO).

---

## LLM ஒருங்கிணைப்பு

### LangChain

```python
from lmm.integrations.langchain import DharmaDocumentCompressor

compressor = DharmaDocumentCompressor(k=5)
# LangChain இன் ContextualCompressionRetriever உடன் பயன்படுத்தவும்
```

### LlamaIndex

```python
from lmm.integrations.llamaindex import DharmaNodePostprocessor

postprocessor = DharmaNodePostprocessor(top_k=5)
# LlamaIndex வினவல் இயந்திரத்துடன் பயன்படுத்தவும்
```

### தனி LLM உதவிகள்

```python
from lmm.llm import FewShotSelector, OutputReranker, DriftDetector

# பன்முகத்தன்மை உத்தரவாதத்துடன் சில-முன்மாதிரி உதாரண தேர்வு
selector = FewShotSelector(k=5)
examples = selector.select(candidates, query)

# ஆச்சரியம் × பன்முகத்தன்மையால் LLM வெளியீடுகளை மறுவரிசைப்படுத்தவும்
reranker = OutputReranker(top_k=3)
best = reranker.rerank(outputs)

# காலப்போக்கில் வெளியீட்டு விநியோக நகர்வை கண்காணிக்கவும்
detector = DriftDetector(window=100)
report = detector.update(new_output)
```

---

## DharmaLMM — உயர்-நிலை API

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

## நரம்பியல் உருவகப்படுத்தல்

ஆற்றல்-திறனான FEP கணிப்பிற்காக மெம்ரிஸ்டர்-குறுக்குப்பட்டை வன்பொருளை உருவகப்படுத்தவும்:

```python
from lmm.dharma.neuromorphic import NeuromorphicChip

chip = NeuromorphicChip(size=64)
chip.program(weight_matrix)
report = chip.run(input_voltages, steps=100)
# ~10 fJ/MAC, ~30 ns ஒருங்கிணைவு
```

---

## சார்புகள்

| தொகுப்பு | பதிப்பு | தேவையானதா |
|---------|--------|-----------|
| numpy | >= 1.24 | ஆம் |
| scipy | >= 1.10 | ஆம் |
| hnswlib | >= 0.8.0 | விருப்பம் (sparse வரைபடம்) |
| langchain-core | >= 0.2.0 | விருப்பம் (LangChain) |
| llama-index-core | >= 0.10.0 | விருப்பம் (LlamaIndex) |

---

## கோட்பாட்டு அடிப்படைகள்

| கருத்தாக்கம் | சூத்திரம் |
|------------|----------|
| **FEP ≡ KCL** | `dV/dt = −V/τ + g'(V) · G · ε` — கிர்க்ஹோஃப்பின் மின்னோட்டச் சட்டமாக கணிப்பு-பிழை குறைப்பு |
| **மேல்-மடல் கருணா** | இரக்கம் அதிகரிக்கும் வருமானத்தை காட்டுகிறது: `f(S∪{i}) − f(S) ≥ f(T∪{i}) − f(T)` S ⊆ T க்கு |
| **கீழ்-மடல் சீல** | நடத்தை குறைந்து வரும் வருமானத்தை காட்டுகிறது — சோம்பேறி பேராசை (1−1/e) உத்தரவாதம் தருகிறது |
| **மத்யமக** | நடுவழி CV = 0.5 ஐ லியாப்புனோவ்-நிலையான அதிவேக சாய்வு வழியாக இலக்காக கொள்கிறது |
| **பீனியல் சுருக்கம்** | அலை-செயல்பாடு சுருக்கம் PRNG க்கு பதிலாக இயற்பியல் என்ட்ரோபியை (வன்பொருள் TRNG) பயன்படுத்துகிறது |

---

## உரிமம்

[MIT](LICENSE)
