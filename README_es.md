# LMM — Optimizador QUBO Clásico con Dharma Digital

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-1.0.0-green.svg)](pyproject.toml)

[English](README_en.md) | [日本語](README.md) | [中文](README_zh.md) | [한국어](README_ko.md) | [हिन्दी](README_hi.md) | [Français](README_fr.md) | [বাংলা](README_bn.md) | [தமிழ்](README_ta.md) | [తెలుగు](README_te.md) | [मराठी](README_mr.md) | [‎اردو‎](README_ur.md) | [ગુજરાતી](README_gu.md) | [ಕನ್ನಡ](README_kn.md) | [മലయാളം](README_ml.md) | [ਪੰਜਾਬੀ](README_pa.md)

Una biblioteca de optimización QUBO clásica **sin necesidad de D-Wave** que fusiona
la sorpresa teórico-informacional con funciones de energía inspiradas en la filosofía
budista, razonamiento basado en el Principio de Energía Libre (FEP) y computación
consciente — todo ejecutándose en hardware convencional.

---

## Aspectos Destacados

- **No requiere computadora cuántica** — los solucionadores clásicos (SA, Ising SA, greedy submodular) igualan o superan la calidad de D-Wave.
- **Capacidad de un billón de tokens** — estructuras de datos probabilísticas en streaming (Count-Min Sketch, Streaming Histogram) mantienen la memoria en O(k).
- **Motor Dharma-Algebra** — términos de energía budistas conectables que se enrutan automáticamente al solucionador matemáticamente óptimo.
- **Orquestador de razonamiento de 8 modos** — adaptativo, teórico, abductivo, inferencia activa, memoria (Alaya), consolidación del sueño, corporeizado y consciencia cuántica (Pineal).
- **Listo para LLM** — integraciones de primera clase con LangChain y LlamaIndex, selector few-shot, reordenador de salida, detector de deriva.

---

## Arquitectura

```
lmm/
├── core.py                  # Pipeline principal de LMM
├── cli.py                   # Punto de entrada CLI
├── qubo.py                  # Constructor de matrices QUBO dispersas
├── solvers.py               # SA / Ising SA / relajación / greedy / submodular
├── surprise.py              # Sorpresa teórico-informacional
├── selector.py              # Estrategia de selección adaptativa
├── processor.py             # Procesamiento con prioridad + caché
├── pipeline.py              # Orquestación SmartSelector → SmartProcessor
├── _compat.py               # Detección de capacidades en tiempo de ejecución y ayudantes dispersos
├── dharma/                  # Dharma Digital (optimización inspirada en filosofía budista)
│   ├── api.py               # API de alto nivel DharmaLMM
│   ├── energy.py            # Términos de energía conectables (Dukkha, Prajna, Karuna, …)
│   ├── engine.py            # UniversalDharmaEngine — solucionador con enrutamiento automático
│   ├── fep.py               # FEP ≡ KCL solucionador ODE
│   ├── neuromorphic.py      # Simulador ASIC de crossbar de memristores
│   ├── reranker.py          # Reordenador en cascada RAG + enrutador de intenciones
│   └── …
├── reasoning/               # 8 modos de razonamiento basados en FEP
│   ├── adaptive.py          # Ajuste dinámico de parámetros
│   ├── theoretical.py       # Grafo de estructura lógica
│   ├── hyper.py             # Inyección abductiva de nodos latentes
│   ├── active.py            # Adquisición de conocimiento externo
│   ├── alaya.py             # Memoria sináptica hebbiana (Alaya)
│   ├── sleep.py             # Consolidación de memoria NREM/REM
│   ├── embodiment.py        # Fusión multimodal de 6 sentidos
│   ├── pineal.py            # Consciencia cuántica (TRNG por hardware)
│   └── orchestrator.py      # Selección y despacho de modos
├── scale/                   # Streaming de un billón de tokens
│   ├── sketch.py            # Count-Min Sketch, Streaming Histogram
│   ├── stream.py            # Sorpresa en streaming
│   ├── cascade.py           # Filtro en cascada multinivel (1T → K)
│   └── pipeline.py          # ScalablePipeline
├── llm/                     # Ayudantes para flujos de trabajo LLM
│   ├── fewshot.py           # Selector de ejemplos few-shot
│   ├── reranker.py          # Reordenador de salida
│   ├── drift.py             # Detector de deriva de distribución
│   ├── sampler.py           # Muestreador de tokens
│   └── embeddings.py        # Adaptador de embeddings unificado
├── integrations/
│   ├── langchain.py         # DharmaRetriever / DocumentCompressor / ExampleSelector
│   └── llamaindex.py        # DharmaNodePostprocessor
```

---

## Instalación

```bash
# Núcleo (solo numpy + scipy)
pip install -e .

# Con herramientas de desarrollo
pip install -e ".[dev]"

# Con todas las integraciones opcionales
pip install -e ".[all]"
```

### Extras opcionales

| Extra | Paquetes | Propósito |
|-------|----------|-----------|
| `dev` | pytest, ruff | Pruebas y linting |
| `dharma` | hnswlib | Construcción de grafos k-NN dispersos |
| `langchain` | langchain-core | Integración con LangChain |
| `llamaindex` | llama-index-core | Integración con LlamaIndex |
| `all` | Todos los anteriores | Todo incluido |

---

## Inicio Rápido

### API de Python

```python
import numpy as np
from lmm.core import LMM

# Seleccionar los K elementos más sorprendentes
surprises = np.array([1.0, 5.0, 2.0, 8.0, 3.0, 7.0])
model = LMM(k=3, solver_method="sa")
result = model.select_from_surprises(surprises)
print(result.selected_indices)  # ej. [3, 5, 1]
```

### CLI

```bash
# Ejecutar una demostración rápida
lmm --demo --k 10 --method sa

# Desde un archivo NumPy
lmm --input data.npy --k 5 --method greedy
```

---

## Métodos de Resolución

| Método | Descripción | Complejidad |
|--------|-------------|-------------|
| `sa` | Recocido Simulado (por defecto) | O(n · steps) |
| `ising_sa` | SA en forma de Ising con delta vectorizado | O(1) por giro |
| `relaxation` | Relajación continua + redondeo (SLSQP) | O(n²) |
| `greedy` | Selección voraz | O(n · k) |

---

## Motor Dharma — Términos de Energía Conectables

El **UniversalDharmaEngine** permite componer términos de energía inspirados
en la filosofía budista, cada uno declarando su propiedad matemática. El motor
enruta automáticamente al solucionador óptimo.

```python
from lmm.dharma import UniversalDharmaEngine, DukkhaTerm, KarunaTerm

engine = UniversalDharmaEngine(n_candidates=1000)
engine.add(DukkhaTerm(surprises, weight=1.0))    # linear  → Top-K
engine.add(KarunaTerm(impact_graph, weight=2.0)) # supermodular → SA
result = engine.synthesize_and_solve(k=10)
```

### Correspondencia Filosofía → Matemáticas → Solucionador

| Concepto budista | Término de energía | Propiedad matemática | Solucionador |
|-----------------|-------------|---------------|--------|
| Prajna (Sabiduría) | `PrajnaTerm` | lineal | Ordenamiento Top-K |
| Karuna (Compasión) | `KarunaTerm` | supermodular | Arranque en caliente + SA |
| Sila (Conducta) | `SilaTerm` | submodular | Greedy perezoso |
| Madhyamaka (Camino Medio) | `MadhyamakaCriterion` | Lyapunov | Gradiente exponencial |
| Dukkha (Sufrimiento) | `DukkhaTerm` | lineal | Ordenamiento Top-K |

---

## Orquestador de Razonamiento

El **DharmaReasonerOrchestrator** selecciona entre 8 modos de razonamiento
según la complejidad de la consulta:

```python
from lmm.reasoning import DharmaReasonerOrchestrator

orch = DharmaReasonerOrchestrator(n_candidates=500)
result = orch.reason(query_vector, context_vectors)
print(result.mode_used, result.explanation)
```

| Modo | Módulo | Inspirado en |
|------|--------|-------------|
| Adaptativo | `adaptive.py` | 応病与薬 — medicina adecuada a la enfermedad |
| Teórico | `theoretical.py` | 因明 — lógica formal budista |
| Abductivo | `hyper.py` | 般若の飛躍 — el salto de Prajna |
| Inferencia Activa | `active_inference.py` | 托鉢 — búsqueda de la verdad externa |
| Memoria Alaya | `alaya.py` | 阿頼耶識 — consciencia almacén |
| Sueño | `sleep.py` | 禅定 — consolidación NREM/REM |
| Corporeizado | `embodiment.py` | 六根 — las seis modalidades sensoriales |
| Pineal (Cuántico) | `pineal.py` | 松果体 — consciencia por entropía de hardware |

---

## Escalamiento a un Billón de Tokens

Procese flujos de datos arbitrariamente grandes con memoria constante:

```python
from lmm.scale import ScalablePipeline
from pathlib import Path

pipe = ScalablePipeline(k=10, chunk_size=100_000)
pipe.fit_files([Path("shard_001.npy"), ...])
result = pipe.run_files([Path("data_001.npy"), ...])
print(result.summary)
```

**Cascada de tres etapas:** 1 T tokens → 100 M (streaming top-k) → 10 K (percentil) → K (QUBO).

---

## Integración con LLM

### LangChain

```python
from lmm.integrations.langchain import DharmaDocumentCompressor

compressor = DharmaDocumentCompressor(k=5)
# Usar con ContextualCompressionRetriever de LangChain
```

### LlamaIndex

```python
from lmm.integrations.llamaindex import DharmaNodePostprocessor

postprocessor = DharmaNodePostprocessor(top_k=5)
# Usar con el motor de consultas de LlamaIndex
```

### Ayudantes LLM independientes

```python
from lmm.llm import FewShotSelector, OutputReranker, DriftDetector

# Selección de ejemplos few-shot con garantía de diversidad
selector = FewShotSelector(k=5)
examples = selector.select(candidates, query)

# Reordenar salidas de LLM por sorpresa × diversidad
reranker = OutputReranker(top_k=3)
best = reranker.rerank(outputs)

# Monitorear la deriva de distribución de salida a lo largo del tiempo
detector = DriftDetector(window=100)
report = detector.update(new_output)
```

---

## DharmaLMM — API de Alto Nivel

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

## Simulación Neuromórfica

Simule hardware de crossbar de memristores para computación FEP energéticamente eficiente:

```python
from lmm.dharma.neuromorphic import NeuromorphicChip

chip = NeuromorphicChip(size=64)
chip.program(weight_matrix)
report = chip.run(input_voltages, steps=100)
# ~10 fJ/MAC, ~30 ns de convergencia
```

---

## Dependencias

| Paquete | Versión | Requerido |
|---------|---------|-----------|
| numpy | >= 1.24 | Sí |
| scipy | >= 1.10 | Sí |
| hnswlib | >= 0.8.0 | Opcional (grafo disperso) |
| langchain-core | >= 0.2.0 | Opcional (LangChain) |
| llama-index-core | >= 0.10.0 | Opcional (LlamaIndex) |

---

## Fundamentos Teóricos

| Concepto | Formulación |
|---------|-------------|
| **FEP ≡ KCL** | `dV/dt = −V/τ + g'(V) · G · ε` — minimización de error de predicción como Ley de Corrientes de Kirchhoff |
| **Karuna Supermodular** | La compasión exhibe rendimientos crecientes: `f(S∪{i}) − f(S) ≥ f(T∪{i}) − f(T)` para S ⊆ T |
| **Sila Submodular** | La conducta exhibe rendimientos decrecientes — el greedy perezoso garantiza (1−1/e) |
| **Madhyamaka** | El Camino Medio apunta a CV = 0.5 mediante descenso de gradiente exponencial estable de Lyapunov |
| **Colapso Pineal** | El colapso de la función de onda usa entropía física (TRNG por hardware) en lugar de PRNG |

---

## Licencia

[MIT](LICENSE)
