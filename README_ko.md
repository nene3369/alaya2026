# LMM — 디지털 법(法)을 활용한 고전적 QUBO 최적화기

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-1.0.0-green.svg)](pyproject.toml)

> [English](README_en.md) | [日本語](README.md) | [中文](README_zh.md) | [Español](README_es.md) | [हिन्दी](README_hi.md) | [Français](README_fr.md) | [বাংলা](README_bn.md) | [தமிழ்](README_ta.md) | [తెలుగు](README_te.md) | [मराठी](README_mr.md) | [اردو](README_ur.md) | [ગુજરાતી](README_gu.md) | [ಕನ್ನಡ](README_kn.md) | [മലയാളം](README_ml.md) | [ਪੰਜਾਬੀ](README_pa.md)

**D-Wave 불필요**한 고전적 QUBO 최적화 라이브러리로, 정보 이론적 놀라움(surprise)과
불교 철학에서 영감을 받은 에너지 함수, 자유 에너지 원리(FEP) 추론, 의식 인식
컴퓨팅을 융합합니다 — 모두 일반 하드웨어에서 실행됩니다.

---

## 주요 특징

- **양자 컴퓨터 불필요** — 고전적 솔버(SA, Ising SA, 하위모듈 탐욕법)가 D-Wave 품질과 동등하거나 이를 능가합니다.
- **1조 토큰 처리 가능** — 스트리밍 확률적 자료 구조(Count-Min Sketch, Streaming Histogram)로 메모리를 O(k)로 유지합니다.
- **법(Dharma)-대수 엔진** — 플러그인 방식의 불교 에너지 항이 수학적으로 최적인 솔버로 자동 라우팅됩니다.
- **8모드 추론 오케스트레이터** — 적응형, 이론적, 귀추적, 능동 추론, 기억(아뢰야식/Alaya), 수면 통합, 신체화, 양자 의식(송과체/Pineal).
- **LLM 연동 지원** — LangChain 및 LlamaIndex 1급 통합, 퓨샷 선택기, 출력 재순위기, 드리프트 감지기.

---

## 아키텍처

```
lmm/
├── core.py                  # 메인 LMM 파이프라인
├── cli.py                   # CLI 진입점
├── qubo.py                  # 희소 QUBO 행렬 빌더
├── solvers.py               # SA / Ising SA / 완화 / 탐욕법 / 하위모듈
├── surprise.py              # 정보 이론적 놀라움(surprise)
├── selector.py              # 적응적 선택 전략
├── processor.py             # 우선순위 처리 + 캐시
├── pipeline.py              # SmartSelector → SmartProcessor 오케스트레이션
├── _compat.py               # 런타임 기능 감지 및 희소 헬퍼
├── dharma/                  # 디지털 법(Digital Dharma) (불교 철학 최적화)
│   ├── api.py               # DharmaLMM 고수준 API
│   ├── energy.py            # 플러그인 에너지 항 (Dukkha, Prajna, Karuna, …)
│   ├── engine.py            # UniversalDharmaEngine — 자동 라우팅 솔버
│   ├── fep.py               # FEP ≡ KCL ODE 솔버
│   ├── neuromorphic.py      # 멤리스터 크로스바 ASIC 시뮬레이터
│   ├── reranker.py          # RAG 캐스케이드 재순위기 + 의도 라우터
│   └── …
├── reasoning/               # 8가지 FEP 기반 추론 모드
│   ├── adaptive.py          # 동적 매개변수 튜닝
│   ├── theoretical.py       # 논리 구조 그래프
│   ├── hyper.py             # 귀추적 잠재 노드 주입
│   ├── active.py            # 외부 지식 획득
│   ├── alaya.py             # 헤브 시냅스 기억 (아뢰야식/Alaya)
│   ├── sleep.py             # NREM/REM 기억 통합
│   ├── embodiment.py        # 육근(六根) 다중 모달 융합
│   ├── pineal.py            # 양자 의식 (하드웨어 TRNG)
│   └── orchestrator.py      # 모드 선택 및 디스패치
├── scale/                   # 1조 토큰 스트리밍
│   ├── sketch.py            # Count-Min Sketch, Streaming Histogram
│   ├── stream.py            # 스트리밍 놀라움(surprise)
│   ├── cascade.py           # 다단계 캐스케이드 필터 (1T → K)
│   └── pipeline.py          # ScalablePipeline
├── llm/                     # LLM 워크플로 헬퍼
│   ├── fewshot.py           # 퓨샷 예제 선택기
│   ├── reranker.py          # 출력 재순위기
│   ├── drift.py             # 분포 드리프트 감지기
│   ├── sampler.py           # 토큰 샘플러
│   └── embeddings.py        # 통합 임베딩 어댑터
├── integrations/
│   ├── langchain.py         # DharmaRetriever / DocumentCompressor / ExampleSelector
│   └── llamaindex.py        # DharmaNodePostprocessor
```

---

## 설치

```bash
# 코어 (numpy + scipy만)
pip install -e .

# 개발 도구 포함
pip install -e ".[dev]"

# 모든 선택적 통합 포함
pip install -e ".[all]"
```

### 선택적 추가 패키지

| 추가 항목 | 패키지 | 용도 |
|-------|----------|---------|
| `dev` | pytest, ruff | 테스트 및 린팅 |
| `dharma` | hnswlib | 희소 k-NN 그래프 구축 |
| `langchain` | langchain-core | LangChain 통합 |
| `llamaindex` | llama-index-core | LlamaIndex 통합 |
| `all` | 위의 모든 것 | 전체 |

---

## 빠른 시작

### Python API

```python
import numpy as np
from lmm.core import LMM

# 가장 놀라움이 큰 상위 K개 항목 선택
surprises = np.array([1.0, 5.0, 2.0, 8.0, 3.0, 7.0])
model = LMM(k=3, solver_method="sa")
result = model.select_from_surprises(surprises)
print(result.selected_indices)  # 예: [3, 5, 1]
```

### CLI

```bash
# 빠른 데모 실행
lmm --demo --k 10 --method sa

# NumPy 파일에서 실행
lmm --input data.npy --k 5 --method greedy
```

---

## 솔버 방법

| 방법 | 설명 | 복잡도 |
|--------|-------------|------------|
| `sa` | 시뮬레이티드 어닐링 (기본값) | O(n · steps) |
| `ising_sa` | 벡터화된 델타를 사용하는 Ising 형태 SA | O(1) per flip |
| `relaxation` | 연속 완화 + 반올림 (SLSQP) | O(n²) |
| `greedy` | 탐욕적 선택 | O(n · k) |

---

## Dharma 엔진 — 플러그인 에너지 항

**UniversalDharmaEngine**은 불교 철학에서 영감을 받은 에너지 항을 조합할 수 있으며,
각 항은 자신의 수학적 속성을 선언합니다. 엔진은 자동으로 최적의 솔버로
라우팅합니다.

```python
from lmm.dharma import UniversalDharmaEngine, DukkhaTerm, KarunaTerm

engine = UniversalDharmaEngine(n_candidates=1000)
engine.add(DukkhaTerm(surprises, weight=1.0))    # linear  → Top-K
engine.add(KarunaTerm(impact_graph, weight=2.0)) # supermodular → SA
result = engine.synthesize_and_solve(k=10)
```

### 철학 → 수학 → 솔버 매핑

| 불교 개념 | 에너지 항 | 수학적 속성 | 솔버 |
|-----------------|-------------|---------------|--------|
| 반야(般若, Prajna, 지혜) | `PrajnaTerm` | linear | Top-K sort |
| 자비(慈悲, Karuna, 연민) | `KarunaTerm` | supermodular | Warm-start + SA |
| 지계(持戒, Sila, 행위) | `SilaTerm` | submodular | Lazy greedy |
| 중도(中道, Madhyamaka) | `MadhyamakaCriterion` | Lyapunov | Exponential gradient |
| 고(苦, Dukkha, 괴로움) | `DukkhaTerm` | linear | Top-K sort |

---

## 추론 오케스트레이터

**DharmaReasonerOrchestrator**는 쿼리 복잡도에 따라 8가지 추론 모드 중에서
선택합니다:

```python
from lmm.reasoning import DharmaReasonerOrchestrator

orch = DharmaReasonerOrchestrator(n_candidates=500)
result = orch.reason(query_vector, context_vectors)
print(result.mode_used, result.explanation)
```

| 모드 | 모듈 | 영감의 원천 |
|------|--------|-------------|
| 적응형 | `adaptive.py` | 응병여약(應病與藥) — 병에 맞는 약 |
| 이론적 | `theoretical.py` | 인명(因明) — 불교 형식 논리학 |
| 귀추적 | `hyper.py` | 반야의 비약(般若の飛躍) — 반야의 도약 |
| 능동 추론 | `active_inference.py` | 탁발(托鉢) — 외부 진리 탐구 |
| 아뢰야 기억 | `alaya.py` | 아뢰야식(阿頼耶識) — 장식(藏識) |
| 수면 | `sleep.py` | 선정(禪定) — NREM/REM 통합 |
| 신체화 | `embodiment.py` | 육근(六根) — 여섯 가지 감각 양식 |
| 송과체 (양자) | `pineal.py` | 송과체(松果体) — 하드웨어 엔트로피 의식 |

---

## 1조 토큰 스케일링

일정한 메모리로 임의 크기의 데이터 스트림을 처리합니다:

```python
from lmm.scale import ScalablePipeline
from pathlib import Path

pipe = ScalablePipeline(k=10, chunk_size=100_000)
pipe.fit_files([Path("shard_001.npy"), ...])
result = pipe.run_files([Path("data_001.npy"), ...])
print(result.summary)
```

**3단계 캐스케이드:** 1T 토큰 → 100M (스트리밍 top-k) → 10K (백분위수) → K (QUBO).

---

## LLM 통합

### LangChain

```python
from lmm.integrations.langchain import DharmaDocumentCompressor

compressor = DharmaDocumentCompressor(k=5)
# LangChain의 ContextualCompressionRetriever와 함께 사용
```

### LlamaIndex

```python
from lmm.integrations.llamaindex import DharmaNodePostprocessor

postprocessor = DharmaNodePostprocessor(top_k=5)
# LlamaIndex 쿼리 엔진과 함께 사용
```

### 독립형 LLM 헬퍼

```python
from lmm.llm import FewShotSelector, OutputReranker, DriftDetector

# 다양성 보장이 있는 퓨샷 예제 선택
selector = FewShotSelector(k=5)
examples = selector.select(candidates, query)

# 놀라움 × 다양성으로 LLM 출력 재순위
reranker = OutputReranker(top_k=3)
best = reranker.rerank(outputs)

# 시간에 따른 출력 분포 드리프트 모니터링
detector = DriftDetector(window=100)
report = detector.update(new_output)
```

---

## DharmaLMM — 고수준 API

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

## 뉴로모픽 시뮬레이션

에너지 효율적인 FEP 계산을 위한 멤리스터 크로스바 하드웨어를 시뮬레이션합니다:

```python
from lmm.dharma.neuromorphic import NeuromorphicChip

chip = NeuromorphicChip(size=64)
chip.program(weight_matrix)
report = chip.run(input_voltages, steps=100)
# ~10 fJ/MAC, ~30 ns 수렴
```

---

## 의존성

| 패키지 | 버전 | 필수 여부 |
|---------|---------|----------|
| numpy | >= 1.24 | 예 |
| scipy | >= 1.10 | 예 |
| hnswlib | >= 0.8.0 | 선택 (희소 그래프) |
| langchain-core | >= 0.2.0 | 선택 (LangChain) |
| llama-index-core | >= 0.10.0 | 선택 (LlamaIndex) |

---

## 이론적 기초

| 개념 | 수식화 |
|---------|-------------|
| **FEP ≡ KCL** | `dV/dt = −V/τ + g'(V) · G · ε` — 예측 오차 최소화를 키르히호프 전류 법칙으로 표현 |
| **초모듈 자비(Karuna)** | 자비는 수확 체증을 보입니다: `f(S∪{i}) − f(S) ≥ f(T∪{i}) − f(T)` (S ⊆ T) |
| **하위모듈 지계(Sila)** | 행위는 수확 체감을 보입니다 — 지연 탐욕법이 (1−1/e) 보장을 제공합니다 |
| **중도(Madhyamaka)** | 중도는 리아푸노프 안정 지수 경사 하강법을 통해 CV = 0.5를 목표로 합니다 |
| **송과체 붕괴(Pineal Collapse)** | 파동 함수 붕괴는 PRNG 대신 물리적 엔트로피(하드웨어 TRNG)를 사용합니다 |

---

## 라이선스

[MIT](LICENSE)
