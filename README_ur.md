# LMM — ڈیجیٹل دھرم کے ساتھ کلاسیکل QUBO آپٹیمائزر

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-1.0.0-green.svg)](pyproject.toml)

> [English](README_en.md) | [日本語](README.md) | [中文](README_zh.md) | [한국어](README_ko.md) | [Español](README_es.md) | [हिन्दी](README_hi.md) | [Français](README_fr.md) | [বাংলা](README_bn.md) | [தமிழ்](README_ta.md) | [తెలుగు](README_te.md) | [मराठी](README_mr.md) | [اردو](README_ur.md) | [ગુજરાતી](README_gu.md) | [ಕನ್ನಡ](README_kn.md) | [മലയാളം](README_ml.md) | [ਪੰਜਾਬੀ](README_pa.md)

ایک **D-Wave سے آزاد** کلاسیکل QUBO آپٹیمائزیشن لائبریری جو معلوماتی نظریے کی حیرت کو بدھ فلسفے سے متاثر انرجی فنکشنز، فری انرجی پرنسپل (FEP) استدلال، اور شعور-باخبر کمپیوٹنگ کے ساتھ یکجا کرتی ہے — سب کچھ عام ہارڈویئر پر چلتا ہے۔

---

## اہم خصوصیات

- **کوئی کوانٹم کمپیوٹر ضروری نہیں** — کلاسیکل سالور (SA، Ising SA، سب ماڈیولر گریڈی) D-Wave کے معیار کے برابر یا اس سے بہتر نتائج دیتے ہیں۔
- **ٹریلین-ٹوکن قابل** — سٹریمنگ امکاناتی ڈیٹا ڈھانچے (Count-Min Sketch، Streaming Histogram) میموری کو O(k) رکھتے ہیں۔
- **دھرم-الجبرا انجن** — قابل توسیع بدھ انرجی اصطلاحات خود بخود ریاضیاتی طور پر بہترین سالور کی طرف رہنمائی کرتی ہیں۔
- **8-موڈ استدلال آرکیسٹریٹر** — موافق، نظری، ابداعی، فعال-اندازہ، یادداشت (آلایہ)، نیند استحکام، مجسم، اور کوانٹم-شعور (پینیل)۔
- **LLM کے لیے تیار** — LangChain اور LlamaIndex کے ساتھ اعلیٰ درجے کا انضمام، فیو-شاٹ سلیکٹر، آؤٹ پٹ ری رینکر، ڈرفٹ ڈیٹیکٹر۔

---

## فن تعمیر

```
lmm/
├── core.py                  # LMM کا مرکزی پائپ لائن
├── cli.py                   # CLI داخلہ نقطہ
├── qubo.py                  # سپارس QUBO میٹرکس بنانے والا
├── solvers.py               # SA / Ising SA / ریلیکسیشن / گریڈی / سب ماڈیولر
├── surprise.py              # معلوماتی نظریے کی حیرت
├── selector.py              # موافق انتخاب حکمت عملی
├── processor.py             # ترجیحی پروسیسنگ + کیش
├── pipeline.py              # SmartSelector → SmartProcessor آرکیسٹریشن
├── _compat.py               # رن ٹائم صلاحیت شناخت اور سپارس معاونین
├── dharma/                  # ڈیجیٹل دھرم (بدھ فلسفے سے متاثر آپٹیمائزیشن)
│   ├── api.py               # DharmaLMM اعلیٰ سطحی API
│   ├── energy.py            # قابل توسیع انرجی اصطلاحات (دکھا، پراجنا، کرونا، ...)
│   ├── engine.py            # UniversalDharmaEngine — خود-رہنمائی سالور
│   ├── fep.py               # FEP ≡ KCL ODE سالور
│   ├── neuromorphic.py      # میم ریزسٹر کراس بار ASIC سمولیٹر
│   ├── reranker.py          # RAG کاسکیڈ ری رینکر + ارادہ روٹر
│   └── …
├── reasoning/               # 8 FEP-بنیاد استدلال طریقے
│   ├── adaptive.py          # متحرک پیرامیٹر ٹیوننگ
│   ├── theoretical.py       # منطقی ڈھانچہ گراف
│   ├── hyper.py             # ابداعی لیٹنٹ-نوڈ انجیکشن
│   ├── active.py            # بیرونی علم کا حصول
│   ├── alaya.py             # ہیبین سیناپٹک میموری (آلایہ)
│   ├── sleep.py             # NREM/REM میموری استحکام
│   ├── embodiment.py        # 6-حواس ملٹی موڈل فیوژن
│   ├── pineal.py            # کوانٹم شعور (ہارڈویئر TRNG)
│   └── orchestrator.py      # موڈ انتخاب اور ڈسپیچ
├── scale/                   # ٹریلین-ٹوکن سٹریمنگ
│   ├── sketch.py            # Count-Min Sketch، Streaming Histogram
│   ├── stream.py            # سٹریمنگ حیرت
│   ├── cascade.py           # ملٹی-لیول کاسکیڈ فلٹر (1T → K)
│   └── pipeline.py          # ScalablePipeline
├── llm/                     # LLM ورک فلو معاونین
│   ├── fewshot.py           # فیو-شاٹ مثال سلیکٹر
│   ├── reranker.py          # آؤٹ پٹ ری رینکر
│   ├── drift.py             # تقسیم ڈرفٹ ڈیٹیکٹر
│   ├── sampler.py           # ٹوکن سیمپلر
│   └── embeddings.py        # یکجا ایمبیڈنگ اڈاپٹر
├── integrations/
│   ├── langchain.py         # DharmaRetriever / DocumentCompressor / ExampleSelector
│   └── llamaindex.py        # DharmaNodePostprocessor
```

---

## تنصیب

```bash
# بنیادی (صرف numpy + scipy)
pip install -e .

# ڈیو ٹولز کے ساتھ
pip install -e ".[dev]"

# تمام اختیاری انضمام کے ساتھ
pip install -e ".[all]"
```

### اختیاری اضافے

| اضافہ | پیکیجز | مقصد |
|-------|---------|-------|
| `dev` | pytest, ruff | جانچ اور لنٹنگ |
| `dharma` | hnswlib | سپارس k-NN گراف تعمیر |
| `langchain` | langchain-core | LangChain انضمام |
| `llamaindex` | llama-index-core | LlamaIndex انضمام |
| `all` | اوپر تمام | سب کچھ |

---

## فوری آغاز

### Python API

```python
import numpy as np
from lmm.core import LMM

# سب سے زیادہ حیران کن K آئٹمز منتخب کریں
surprises = np.array([1.0, 5.0, 2.0, 8.0, 3.0, 7.0])
model = LMM(k=3, solver_method="sa")
result = model.select_from_surprises(surprises)
print(result.selected_indices)  # مثلاً [3, 5, 1]
```

### CLI

```bash
# فوری ڈیمو چلائیں
lmm --demo --k 10 --method sa

# NumPy فائل سے
lmm --input data.npy --k 5 --method greedy
```

---

## سالور طریقے

| طریقہ | تفصیل | پیچیدگی |
|-------|--------|----------|
| `sa` | سمولیٹڈ اینیلنگ (پہلے سے طے شدہ) | O(n · steps) |
| `ising_sa` | ویکٹرائزڈ ڈیلٹا کے ساتھ Ising-فارم SA | O(1) فی فلپ |
| `relaxation` | مسلسل ریلیکسیشن + گول کرنا (SLSQP) | O(n²) |
| `greedy` | گریڈی انتخاب | O(n · k) |

---

## دھرم انجن — قابل توسیع انرجی اصطلاحات

**UniversalDharmaEngine** آپ کو بدھ فلسفے سے متاثر انرجی اصطلاحات کو یکجا کرنے دیتا ہے، جن میں سے ہر ایک اپنی ریاضیاتی خاصیت ظاہر کرتی ہے۔ انجن خود بخود بہترین سالور کی طرف رہنمائی کرتا ہے۔

```python
from lmm.dharma import UniversalDharmaEngine, DukkhaTerm, KarunaTerm

engine = UniversalDharmaEngine(n_candidates=1000)
engine.add(DukkhaTerm(surprises, weight=1.0))    # خطی → Top-K
engine.add(KarunaTerm(impact_graph, weight=2.0)) # سپر ماڈیولر → SA
result = engine.synthesize_and_solve(k=10)
```

### فلسفہ → ریاضی → سالور نقشہ

| بدھ تصور | انرجی اصطلاح | ریاضیاتی خاصیت | سالور |
|-----------|--------------|----------------|-------|
| پراجنا (حکمت) | `PrajnaTerm` | خطی | Top-K ترتیب |
| کرونا (شفقت) | `KarunaTerm` | سپر ماڈیولر | وارم-اسٹارٹ + SA |
| شیلا (اخلاق) | `SilaTerm` | سب ماڈیولر | لیزی گریڈی |
| مادھیاماکا (درمیانی راہ) | `MadhyamakaCriterion` | لیاپونوف | ایکسپونینشل گریڈینٹ |
| دکھا (تکلیف) | `DukkhaTerm` | خطی | Top-K ترتیب |

---

## استدلال آرکیسٹریٹر

**DharmaReasonerOrchestrator** سوال کی پیچیدگی کی بنیاد پر 8 استدلال طریقوں میں سے انتخاب کرتا ہے:

```python
from lmm.reasoning import DharmaReasonerOrchestrator

orch = DharmaReasonerOrchestrator(n_candidates=500)
result = orch.reason(query_vector, context_vectors)
print(result.mode_used, result.explanation)
```

| موڈ | ماڈیول | تحریک |
|-----|--------|--------|
| موافق | `adaptive.py` | 応病与薬 — بیماری کے مطابق دوا |
| نظری | `theoretical.py` | 因明 — بدھ رسمی منطق |
| ابداعی | `hyper.py` | 般若の飛躍 — پراجنا کی چھلانگ |
| فعال اندازہ | `active_inference.py` | 托鉢 — بیرونی حقیقت کی تلاش |
| آلایہ میموری | `alaya.py` | 阿頼耶識 — گودام شعور |
| نیند | `sleep.py` | 禅定 — NREM/REM استحکام |
| مجسم | `embodiment.py` | 六根 — چھ حواس کی طریقے |
| پینیل (کوانٹم) | `pineal.py` | 松果体 — ہارڈویئر-انٹروپی شعور |

---

## ٹریلین-ٹوکن اسکیلنگ

مستقل میموری کے ساتھ بے حد بڑے ڈیٹا سٹریمز کو پروسیس کریں:

```python
from lmm.scale import ScalablePipeline
from pathlib import Path

pipe = ScalablePipeline(k=10, chunk_size=100_000)
pipe.fit_files([Path("shard_001.npy"), ...])
result = pipe.run_files([Path("data_001.npy"), ...])
print(result.summary)
```

**تین مرحلہ کاسکیڈ:** 1 T ٹوکنز → 100 M (سٹریمنگ top-k) → 10 K (پرسینٹائل) → K (QUBO)۔

---

## LLM انضمام

### LangChain

```python
from lmm.integrations.langchain import DharmaDocumentCompressor

compressor = DharmaDocumentCompressor(k=5)
# LangChain کے ContextualCompressionRetriever کے ساتھ استعمال کریں
```

### LlamaIndex

```python
from lmm.integrations.llamaindex import DharmaNodePostprocessor

postprocessor = DharmaNodePostprocessor(top_k=5)
# LlamaIndex کوئری انجن کے ساتھ استعمال کریں
```

### اکیلے LLM معاونین

```python
from lmm.llm import FewShotSelector, OutputReranker, DriftDetector

# تنوع کی ضمانت کے ساتھ فیو-شاٹ مثال انتخاب
selector = FewShotSelector(k=5)
examples = selector.select(candidates, query)

# حیرت × تنوع کے ذریعے LLM آؤٹ پٹس کو دوبارہ ترتیب دیں
reranker = OutputReranker(top_k=3)
best = reranker.rerank(outputs)

# وقت کے ساتھ آؤٹ پٹ تقسیم کے ڈرفٹ کی نگرانی کریں
detector = DriftDetector(window=100)
report = detector.update(new_output)
```

---

## DharmaLMM API

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

## نیورومورفک سمولیشن

انرجی-موثر FEP حساب کے لیے میم ریزسٹر-کراس بار ہارڈویئر کی سمولیشن کریں:

```python
from lmm.dharma.neuromorphic import NeuromorphicChip

chip = NeuromorphicChip(size=64)
chip.program(weight_matrix)
report = chip.run(input_voltages, steps=100)
# ~10 fJ/MAC، ~30 ns کنورجنس
```

---

## انحصار

| پیکیج | ورژن | ضروری |
|-------|------|--------|
| numpy | >= 1.24 | ہاں |
| scipy | >= 1.10 | ہاں |
| hnswlib | >= 0.8.0 | اختیاری (سپارس گراف) |
| langchain-core | >= 0.2.0 | اختیاری (LangChain) |
| llama-index-core | >= 0.10.0 | اختیاری (LlamaIndex) |

---

## نظریاتی بنیادیں

| تصور | فارمولیشن |
|------|-----------|
| **FEP ≡ KCL** | `dV/dt = −V/τ + g'(V) · G · ε` — Kirchhoff کے کرنٹ قانون کے بطور پیشگوئی-غلطی کم سے کم کرنا |
| **سپر ماڈیولر کرونا** | شفقت بڑھتے منافع ظاہر کرتی ہے: `f(S∪{i}) − f(S) ≥ f(T∪{i}) − f(T)` برائے S ⊆ T |
| **سب ماڈیولر شیلا** | اخلاق گھٹتے منافع ظاہر کرتا ہے — لیزی گریڈی (1−1/e) ضمانت دیتا ہے |
| **مادھیاماکا** | درمیانی راہ Lyapunov-مستحکم ایکسپونینشل گریڈینٹ ڈیسینٹ کے ذریعے CV = 0.5 ہدف بناتی ہے |
| **پینیل کولیپس** | ویو فنکشن کولیپس PRNG کے بجائے طبیعی انٹروپی (ہارڈویئر TRNG) استعمال کرتی ہے |

---

## لائسنس

[MIT](LICENSE)
