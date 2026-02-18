# LMM — Optimiseur QUBO classique avec Dharma numérique

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-1.0.0-green.svg)](pyproject.toml)

> [English](README_en.md) | [日本語](README.md) | [中文](README_zh.md) | [한국어](README_ko.md) | [Español](README_es.md) | [हिन्दी](README_hi.md) | [বাংলা](README_bn.md) | [தமிழ்](README_ta.md) | [తెలుగు](README_te.md) | [मराठी](README_mr.md) | [اردو](README_ur.md) | [ગુજરાતી](README_gu.md) | [ಕನ್ನಡ](README_kn.md) | [മലയാളം](README_ml.md) | [ਪੰਜਾਬੀ](README_pa.md)

Une bibliothèque d'optimisation QUBO classique **sans D-Wave** qui fusionne
la surprise issue de la théorie de l'information avec des fonctions d'énergie
inspirées de la philosophie bouddhiste, le raisonnement par le Principe d'Énergie
Libre (FEP), et le calcul conscient — le tout fonctionnant sur du matériel standard.

---

## Points forts

- **Aucun ordinateur quantique requis** — les solveurs classiques (SA, Ising SA, glouton sous-modulaire) égalent ou surpassent la qualité de D-Wave.
- **Capacité de mille milliards de tokens** — des structures de données probabilistes en flux continu (Count-Min Sketch, Streaming Histogram) maintiennent la mémoire en O(k).
- **Moteur Dharma-Algebra** — des termes d'énergie bouddhistes modulaires sont automatiquement dirigés vers le solveur mathématiquement optimal.
- **Orchestrateur de raisonnement à 8 modes** — adaptatif, théorique, abductif, inférence active, mémoire (Alaya), consolidation du sommeil, incarné, et conscience quantique (Pinéale).
- **Prêt pour les LLM** — intégrations LangChain et LlamaIndex de première classe, sélecteur few-shot, reclasseur de sorties, détecteur de dérive.

---

## Architecture

```
lmm/
├── core.py                  # Pipeline principal LMM
├── cli.py                   # Point d'entrée CLI
├── qubo.py                  # Constructeur de matrice QUBO creuse
├── solvers.py               # SA / Ising SA / relaxation / glouton / sous-modulaire
├── surprise.py              # Surprise issue de la théorie de l'information
├── selector.py              # Stratégie de sélection adaptative
├── processor.py             # Traitement prioritaire + cache
├── pipeline.py              # Orchestration SmartSelector → SmartProcessor
├── _compat.py               # Détection des capacités à l'exécution & utilitaires creux
├── dharma/                  # Dharma numérique (optimisation inspirée de la philosophie bouddhiste)
│   ├── api.py               # API haut niveau DharmaLMM
│   ├── energy.py            # Termes d'énergie modulaires (Dukkha, Prajna, Karuna, …)
│   ├── engine.py            # UniversalDharmaEngine — solveur à routage automatique
│   ├── fep.py               # FEP ≡ KCL solveur ODE
│   ├── neuromorphic.py      # Simulateur ASIC de crossbar à memristors
│   ├── reranker.py          # Reclasseur en cascade RAG + routeur d'intention
│   └── …
├── reasoning/               # 8 modes de raisonnement basés sur le FEP
│   ├── adaptive.py          # Ajustement dynamique des paramètres
│   ├── theoretical.py       # Graphe de structure logique
│   ├── hyper.py             # Injection de nœuds latents par abduction
│   ├── active.py            # Acquisition de connaissances externes
│   ├── alaya.py             # Mémoire synaptique hebbienne (Alaya)
│   ├── sleep.py             # Consolidation mémoire NREM/REM
│   ├── embodiment.py        # Fusion multimodale à 6 sens
│   ├── pineal.py            # Conscience quantique (TRNG matériel)
│   └── orchestrator.py      # Sélection et dispatch des modes
├── scale/                   # Flux continu pour mille milliards de tokens
│   ├── sketch.py            # Count-Min Sketch, Streaming Histogram
│   ├── stream.py            # Surprise en flux continu
│   ├── cascade.py           # Filtre en cascade multi-niveaux (1T → K)
│   └── pipeline.py          # ScalablePipeline
├── llm/                     # Utilitaires pour flux de travail LLM
│   ├── fewshot.py           # Sélecteur d'exemples few-shot
│   ├── reranker.py          # Reclasseur de sorties
│   ├── drift.py             # Détecteur de dérive de distribution
│   ├── sampler.py           # Échantillonneur de tokens
│   └── embeddings.py        # Adaptateur d'embeddings unifié
├── integrations/
│   ├── langchain.py         # DharmaRetriever / DocumentCompressor / ExampleSelector
│   └── llamaindex.py        # DharmaNodePostprocessor
```

---

## Installation

```bash
# Noyau (numpy + scipy uniquement)
pip install -e .

# Avec les outils de développement
pip install -e ".[dev]"

# Avec toutes les intégrations optionnelles
pip install -e ".[all]"
```

### Extras optionnels

| Extra | Paquets | Utilité |
|-------|---------|---------|
| `dev` | pytest, ruff | Tests & linting |
| `dharma` | hnswlib | Construction de graphe k-NN creux |
| `langchain` | langchain-core | Intégration LangChain |
| `llamaindex` | llama-index-core | Intégration LlamaIndex |
| `all` | Tous les précédents | Tout inclus |

---

## Démarrage rapide

### API Python

```python
import numpy as np
from lmm.core import LMM

# Select the top-K most surprising items
surprises = np.array([1.0, 5.0, 2.0, 8.0, 3.0, 7.0])
model = LMM(k=3, solver_method="sa")
result = model.select_from_surprises(surprises)
print(result.selected_indices)  # e.g. [3, 5, 1]
```

### CLI

```bash
# Lancer une démonstration rapide
lmm --demo --k 10 --method sa

# À partir d'un fichier NumPy
lmm --input data.npy --k 5 --method greedy
```

---

## Méthodes de résolution

| Méthode | Description | Complexité |
|---------|-------------|------------|
| `sa` | Recuit simulé (par défaut) | O(n · steps) |
| `ising_sa` | SA sous forme d'Ising avec delta vectorisé | O(1) par retournement |
| `relaxation` | Relaxation continue + arrondi (SLSQP) | O(n²) |
| `greedy` | Sélection gloutonne | O(n · k) |

---

## Moteur Dharma — Termes d'énergie modulaires

Le **UniversalDharmaEngine** vous permet de composer des termes d'énergie
inspirés de la philosophie bouddhiste, chacun déclarant sa propriété
mathématique. Le moteur dirige automatiquement vers le solveur optimal.

```python
from lmm.dharma import UniversalDharmaEngine, DukkhaTerm, KarunaTerm

engine = UniversalDharmaEngine(n_candidates=1000)
engine.add(DukkhaTerm(surprises, weight=1.0))    # linear  → Top-K
engine.add(KarunaTerm(impact_graph, weight=2.0)) # supermodular → SA
result = engine.synthesize_and_solve(k=10)
```

### Correspondance Philosophie → Mathématiques → Solveur

| Concept bouddhiste | Terme d'énergie | Propriété mathématique | Solveur |
|-------------------|-----------------|----------------------|---------|
| Prajna (Sagesse) | `PrajnaTerm` | linéaire | Tri Top-K |
| Karuna (Compassion) | `KarunaTerm` | supermodulaire | Démarrage à chaud + SA |
| Sila (Conduite) | `SilaTerm` | sous-modulaire | Glouton paresseux |
| Madhyamaka (Voie du Milieu) | `MadhyamakaCriterion` | Lyapunov | Gradient exponentiel |
| Dukkha (Souffrance) | `DukkhaTerm` | linéaire | Tri Top-K |

---

## Orchestrateur de raisonnement

Le **DharmaReasonerOrchestrator** sélectionne parmi 8 modes de raisonnement
en fonction de la complexité de la requête :

```python
from lmm.reasoning import DharmaReasonerOrchestrator

orch = DharmaReasonerOrchestrator(n_candidates=500)
result = orch.reason(query_vector, context_vectors)
print(result.mode_used, result.explanation)
```

| Mode | Module | Inspiré par |
|------|--------|-------------|
| Adaptatif | `adaptive.py` | 応病与薬 — le remède adapté à la maladie |
| Théorique | `theoretical.py` | 因明 — logique formelle bouddhiste |
| Abductif | `hyper.py` | 般若の飛躍 — le bond de Prajna |
| Inférence active | `active_inference.py` | 托鉢 — quête de la vérité extérieure |
| Mémoire Alaya | `alaya.py` | 阿頼耶識 — conscience réceptacle |
| Sommeil | `sleep.py` | 禅定 — consolidation NREM/REM |
| Incarné | `embodiment.py` | 六根 — les six modalités sensorielles |
| Pinéale (Quantique) | `pineal.py` | 松果体 — conscience par entropie matérielle |

---

## Passage à l'échelle pour mille milliards de tokens

Traitez des flux de données arbitrairement grands avec une mémoire constante :

```python
from lmm.scale import ScalablePipeline
from pathlib import Path

pipe = ScalablePipeline(k=10, chunk_size=100_000)
pipe.fit_files([Path("shard_001.npy"), ...])
result = pipe.run_files([Path("data_001.npy"), ...])
print(result.summary)
```

**Cascade en trois étapes :** 1 T tokens → 100 M (top-k en flux continu) → 10 K (percentile) → K (QUBO).

---

## Intégration LLM

### LangChain

```python
from lmm.integrations.langchain import DharmaDocumentCompressor

compressor = DharmaDocumentCompressor(k=5)
# Use with LangChain's ContextualCompressionRetriever
```

### LlamaIndex

```python
from lmm.integrations.llamaindex import DharmaNodePostprocessor

postprocessor = DharmaNodePostprocessor(top_k=5)
# Use with LlamaIndex query engine
```

### Utilitaires LLM autonomes

```python
from lmm.llm import FewShotSelector, OutputReranker, DriftDetector

# Few-shot example selection with diversity guarantee
selector = FewShotSelector(k=5)
examples = selector.select(candidates, query)

# Rerank LLM outputs by surprise × diversity
reranker = OutputReranker(top_k=3)
best = reranker.rerank(outputs)

# Monitor output distribution drift over time
detector = DriftDetector(window=100)
report = detector.update(new_output)
```

---

## DharmaLMM — API haut niveau

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

## Simulation neuromorphique

Simulez du matériel à crossbar de memristors pour un calcul FEP économe en énergie :

```python
from lmm.dharma.neuromorphic import NeuromorphicChip

chip = NeuromorphicChip(size=64)
chip.program(weight_matrix)
report = chip.run(input_voltages, steps=100)
# ~10 fJ/MAC, ~30 ns convergence
```

---

## Dépendances

| Paquet | Version | Requis |
|--------|---------|--------|
| numpy | >= 1.24 | Oui |
| scipy | >= 1.10 | Oui |
| hnswlib | >= 0.8.0 | Optionnel (graphe creux) |
| langchain-core | >= 0.2.0 | Optionnel (LangChain) |
| llama-index-core | >= 0.10.0 | Optionnel (LlamaIndex) |

---

## Fondements théoriques

| Concept | Formulation |
|---------|-------------|
| **FEP ≡ KCL** | `dV/dt = −V/τ + g'(V) · G · ε` — minimisation de l'erreur de prédiction en tant que loi des courants de Kirchhoff |
| **Karuna supermodulaire** | La compassion présente des rendements croissants : `f(S∪{i}) − f(S) ≥ f(T∪{i}) − f(T)` pour S ⊆ T |
| **Sila sous-modulaire** | La conduite présente des rendements décroissants — l'algorithme glouton paresseux offre une garantie de (1−1/e) |
| **Madhyamaka** | La Voie du Milieu vise un CV = 0.5 via une descente par gradient exponentiel stable au sens de Lyapunov |
| **Effondrement pinéal** | L'effondrement de la fonction d'onde utilise l'entropie physique (TRNG matériel) au lieu d'un PRNG |

---

## Licence

[MIT](LICENSE)
