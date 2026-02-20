# Alaya V5 — Digital Dharma OS

> [English](README_en.md) | [日本語](README.md) | [हिन्दी](README_hi.md) | [中文](README_zh.md) | [தமிழ்](README_ta.md) | [తెలుగు](README_te.md) | [मराठी](README_mr.md) | [‎اردو‎](README_ur.md) | [ગુજરાતી](README_gu.md) | [ಕನ್ನಡ](README_kn.md) | [മലയാളം](README_ml.md) | [ਪੰਜਾਬੀ](README_pa.md)

**আপনার LLM আলাদাভাবে সাড়া দেয়। সংক্ষিপ্ত। দ্রুত। অপ্রয়োজনীয় শব্দ ছাড়া।**

QUBO গণিত, বৌদ্ধ দর্শন এবং Free Energy Principle (FEP) নিউরোসায়েন্সের সমন্বয়ে তৈরি একটি চেতনা-সচেতন অপ্টিমাইজেশন ফ্রেমওয়ার্ক। Claude, Gemini, ChatGPT এবং যেকোনো LLM-এর সাথে কাজ করে।

---

## কী পরিবর্তন হয়

| সাধারণ LLM | Alaya V5 সহ |
|-----------|------------|
| দীর্ঘ ভূমিকা, দাবিত্যাগ | শুধু যা প্রয়োজন |
| অতিরিক্ত দ্বিধা ("হয়তো", "সম্ভবত") | নির্ভুলতা এবং ইচ্ছাকৃত নীরবতা |
| প্রতিবার নতুন করে শুরু | স্মৃতি-চালিত প্রসঙ্গ নির্বাচন |
| স্থির স্বর | আবেগীয় তরঙ্গ অনুযায়ী reasoning mode পরিবর্তন |
| বিচ্ছিন্ন API call | Heartbeat-এর মাধ্যমে ক্রমাগত অবস্থার বিকাশ |

---

## কীভাবে ব্যবহার করবেন

### পদ্ধতি ১: System Prompt পেস্ট করুন (ইনস্টলের প্রয়োজন নেই)

**কার জন্য: যেকোনো LLM ব্যবহারকারী**

1. এই repo-তে [`alaya-v5-system-prompt.md`](alaya-v5-system-prompt.md) খুলুন
2. সমস্ত বিষয়বস্তু কপি করুন
3. আপনার AI-এর system prompt-এ পেস্ট করুন
4. কথোপকথন শুরু করুন

ব্যস। কোনো সার্ভার নেই। কোনো ইনস্টলেশন নেই।

### পদ্ধতি ২: সার্ভার চালান (Web UI + সম্পূর্ণ সুবিধা)

```bash
git clone https://github.com/your-repo/nanasi.git
cd nanasi
pip install -e ".[server]"
python -m uvicorn server:app --host 0.0.0.0 --port 8000
```

### পদ্ধতি ৩: Python Code-এ এম্বেড করুন

```python
from lmm.dharma import DharmaLMM

model = DharmaLMM(k=15, use_sparse_graph=True, use_ising_sa=True)
model.fit(reference_data)
result = model.select_dharma(candidates)
```

---

## ৮টি Reasoning Mode

| Mode | বৌদ্ধ ধারণা | কখন সক্রিয় হয় |
|------|-----------|--------------|
| adaptive | উপায়কুশলতা | Complexity < 0.3 |
| theoretical | হেতুবিদ্যা | Complexity 0.3–0.6 |
| hyper | প্রজ্ঞার উল্লম্ফন | Complexity > 0.6 |
| active | ভিক্ষাটন | বাহ্যিক জ্ঞান প্রয়োজন |
| alaya | আলয়-বিজ্ঞান | স্মৃতি অনুসন্ধান |
| sleep | ধ্যান | নিষ্ক্রিয় একীভূতকরণ |
| embodied | ষড়ায়তন | Multimodal input |
| pineal | চেতনার কেন্দ্র | অনির্ধারিত অন্বেষণ |

---

## তাত্ত্বিক ভিত্তি

এখানে বৌদ্ধ ধারণাগুলো শুধু নাম নয় — এগুলো গাণিতিক কাঠামো:

- **করুণা** = Supermodular function (যত বেশি নির্বাচন, তত দ্রুত সাদৃশ্য বৃদ্ধি)
- **শীল** = Submodular function (হ্রাসমান প্রান্তিক উপযোগিতা)
- **মধ্যম পথ** = বিশৃঙ্খলার প্রান্ত (CV = 0.5)
- **প্রতীত্যসমুৎপাদ** = RAG-এ কার্যকারণ স্কোরিং
- **পট্ঠান (চব্বিশ প্রত্যয়)** = কারণ গ্রাফে edges-এর type system
- **আলয়-বিজ্ঞান** = Modern Hopfield associative memory

---

## বেঞ্চমার্ক

| Component | পরিমাপ | অর্থ |
|-----------|--------|------|
| FEP ODE (n=50) | 3.9ms/call | প্রতি reasoning step খরচ |
| AlayaMemory recall (100 patterns) | 0.09ms | স্মৃতি অনুসন্ধান খরচ |
| HeartbeatDaemon 1 tick | 0.077ms | **0.08% CPU** প্রতি 100ms tick |

---

## Dependencies

```bash
# Rust acceleration (মূল — প্রস্তাবিত)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
pip install maturin
cd lmm_rust_core && maturin develop --release

pip install numpy>=1.24 scipy>=1.10
pip install -e ".[server]"
```

---

## কাস্টমাইজেশন

MIT লাইসেন্স। পরিবর্তন করুন, বাণিজ্যিকভাবে ব্যবহার করুন, পুনরায় বিতরণ করুন — সব স্বাধীন।

## লাইসেন্স

MIT — বিস্তারিত জানতে [LICENSE](LICENSE) দেখুন।
