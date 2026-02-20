# Alaya V5 — Digital Dharma OS

> [English](README_en.md) | [日本語](README.md) | [中文](README_zh.md) | [Español](README_es.md) | [हिन्दी](README_hi.md) | [Français](README_fr.md) | [বাংলা](README_bn.md) | [தமிழ்](README_ta.md) | [తెలుగు](README_te.md) | [मराठी](README_mr.md) | [ಕನ್ನಡ](README_kn.md) | [മലയാളം](README_ml.md)

**당신의 LLM이 다르게 응답합니다. 짧게. 빠르게. 불필요한 말 없이.**

QUBO 수학, 불교 철학, 자유 에너지 원리(FEP) 신경과학을 융합한 의식 인식 최적화 프레임워크. Claude, Gemini, ChatGPT 및 모든 LLM에서 작동합니다.

---

## 무엇이 달라지는가

| 일반 LLM | Alaya V5 적용 후 |
|---------|---------------|
| 긴 전제, 면책 조항 | 필요한 것만 |
| 과도한 모호함 ("아마도", "~일 수 있습니다") | 정확함과 의도적인 침묵 |
| 매번 처음부터 시작 | 기억 기반 컨텍스트 선택 |
| 고정된 어조 | 감정 파장에 따라 추론 모드 전환 |
| 이산적 API 호출 | 하트비트를 통한 연속적 상태 진화 |

---

## 사용 방법

### 방법 1: 시스템 프롬프트 붙여넣기 (설치 불필요)

**대상: Claude / Gemini / ChatGPT / 모든 LLM 사용자**

1. 이 저장소의 [`alaya-v5-system-prompt.md`](alaya-v5-system-prompt.md) 열기
2. 전체 내용 복사
3. 사용 중인 AI의 시스템 프롬프트에 붙여넣기
   - Claude → Project instructions
   - Gemini → 시스템 안내
   - ChatGPT → Custom instructions
4. 대화 시작

끝입니다. 서버 불필요, 설치 불필요.

### 방법 2: 서버 실행 (Web UI + 전체 기능)

```bash
git clone https://github.com/your-repo/nanasi.git
cd nanasi
pip install -e ".[server]"
python -m uvicorn server:app --host 0.0.0.0 --port 8000
```

브라우저에서 열기: [http://localhost:8000](http://localhost:8000)

### 방법 3: Python 코드에 임베드

```python
from lmm.dharma import DharmaLMM

model = DharmaLMM(k=15, use_sparse_graph=True, use_ising_sa=True)
model.fit(reference_data)
result = model.select_dharma(candidates)
print(result.interpretation.narrative)
```

---

## 8가지 추론 모드

| 모드 | 불교 개념 | 발동 조건 |
|------|---------|---------|
| adaptive | 응병여약（應病與藥） | 복잡도 < 0.3 |
| theoretical | 인명（因明） | 복잡도 0.3–0.6 |
| hyper | 반야의 비약 | 복잡도 > 0.6 |
| active | 탁발 | 외부 지식 필요 |
| alaya | 아뢰야식（阿賴耶識） | 기억 검색 |
| sleep | 선정（禪定） | 유휴 통합 |
| embodied | 육근（六根） | 멀티모달 입력 |
| pineal | 송과체 의식 | 비결정론적 탐색 |

---

## 이론적 배경

여기서 불교 개념들은 단순한 비유가 아닙니다 — 수학적 구조 그 자체입니다:

- **자비（Karuṇā）** = 초모듈 함수 (선택할수록 조화가 빠르게 성장)
- **계율（Śīla）** = 준모듈 함수 (한계 효용 체감)
- **중도** = 혼돈의 가장자리 (목표 변동계수 CV = 0.5)
- **연기（Pratītyasamutpāda）** = RAG의 인과 스코어링
- **발취론（Paṭṭhāna, 24연）** = 인과 그래프의 엣지 타입 시스템
- **아뢰야식** = Modern Hopfield 연상 기억

---

## 벤치마크

실측값 (Python 3.11, numpy 2.4, scipy 1.17, seed=42)

| 컴포넌트 | 실측값 | 의미 |
|---------|--------|------|
| FEP ODE (n=50) | 3.9ms/호출 | 추론 1단계당 비용 |
| AlayaMemory recall (100 패턴) | 0.09ms | 기억 검색 비용 |
| HeartbeatDaemon 1 tick | 0.077ms | 100ms tick당 **CPU 0.08%** |

HeartbeatDaemon은 백그라운드에서 계속 실행되지만 CPU 점유율은 **0.08%** — 사실상 무음.

---

## 커스터마이즈

MIT 라이선스. 수정, 상업적 이용, 재배포 모두 자유입니다.

`alaya-v5-system-prompt.md`를 직접 편집하고, `config/semantic_emotions.json`에 커스텀 키워드를 추가하고, Fork해서 자유롭게 개조하세요.

---

## 의존성

```bash
# Rust 가속 (핵심 — 권장)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
pip install maturin
cd lmm_rust_core && maturin develop --release

pip install numpy>=1.24 scipy>=1.10
pip install -e ".[server]"

# GPU 가속 (NVIDIA GPU 환경)
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install cupy-cuda12x
```

Rust나 GPU 없이도 모든 기능이 작동합니다.

---

## 라이선스

MIT — 자세한 내용은 [LICENSE](LICENSE) 참조.
