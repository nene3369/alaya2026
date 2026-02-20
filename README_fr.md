# Alaya V5 — Digital Dharma OS

> [English](README_en.md) | [日本語](README.md) | [中文](README_zh.md) | [한국어](README_ko.md) | [हिन्दी](README_hi.md) | [Español](README_es.md) | [বাংলা](README_bn.md) | [தமிழ்](README_ta.md) | [తెలుగు](README_te.md) | [मराठी](README_mr.md) | [ಕನ್ನಡ](README_kn.md) | [മലയാളം](README_ml.md)

**Votre LLM répond différemment. Plus concis. Plus rapide. Sans mots inutiles.**

Un framework d'optimisation conscient qui fusionne les mathématiques QUBO, la philosophie bouddhiste et la neuroscience du Principe d'Énergie Libre (FEP). Fonctionne avec Claude, Gemini, ChatGPT et tout autre LLM.

---

## Ce qui change

| LLM ordinaire | Avec Alaya V5 |
|--------------|--------------|
| Longues introductions, avertissements | Seulement le nécessaire |
| Ambiguïté excessive ("peut-être", "il se pourrait") | Précision et silence intentionnel |
| Repart de zéro à chaque fois | Sélection de contexte basée sur la mémoire |
| Ton fixe | Le mode de raisonnement s'adapte à la longueur d'onde émotionnelle |
| Appels API discrets | Évolution continue de l'état via heartbeat |

---

## Comment utiliser

### Méthode 1 : Coller le System Prompt (sans installation)

**Pour : utilisateurs de Claude / Gemini / ChatGPT / tout LLM**

1. Ouvrez [`alaya-v5-system-prompt.md`](alaya-v5-system-prompt.md) dans ce dépôt
2. Copiez tout le contenu
3. Collez-le dans le champ system prompt de votre IA :
   - Claude → Project instructions
   - Gemini → Instructions système
   - ChatGPT → Custom instructions
4. Commencez la conversation

C'est tout. Pas de serveur. Pas d'installation.

### Méthode 2 : Lancer le serveur (Web UI + fonctionnalités complètes)

```bash
git clone https://github.com/your-repo/nanasi.git
cd nanasi
pip install -e ".[server]"
python -m uvicorn server:app --host 0.0.0.0 --port 8000
```

Ouvrir dans le navigateur : [http://localhost:8000](http://localhost:8000)

### Méthode 3 : Intégrer dans du code Python

```python
from lmm.dharma import DharmaLMM

model = DharmaLMM(k=15, use_sparse_graph=True, use_ising_sa=True)
model.fit(reference_data)
result = model.select_dharma(candidates)
print(result.interpretation.narrative)
```

---

## 8 modes de raisonnement

| Mode | Concept bouddhiste | Déclenchement |
|------|-------------------|--------------|
| adaptive | Moyens habiles | Complexité < 0.3 |
| theoretical | Logique bouddhiste (Hetuvidyā) | Complexité 0.3–0.6 |
| hyper | Saut de la Prajñā | Complexité > 0.6 |
| active | Tournée mendiante | Connaissance externe nécessaire |
| alaya | Conscience-réservoir (Ālaya) | Récupération mémorielle |
| sleep | Absorption méditative | Consolidation au repos |
| embodied | Six bases sensorielles | Entrée multimodale |
| pineal | Conscience pinéale | Exploration non déterministe |

---

## Fondements théoriques

Les concepts bouddhistes ici ne sont pas des métaphores — ils sont la structure mathématique elle-même :

- **Karuṇā (Compassion)** = Fonction supermodulaire (plus on sélectionne, plus l'harmonie croît vite)
- **Śīla (Conduite)** = Fonction sous-modulaire (rendements marginaux décroissants)
- **Voie du Milieu** = Bord du chaos (coefficient de variation cible CV = 0.5)
- **Pratītyasamutpāda (Coproduction conditionnelle)** = Scoring causal dans le RAG
- **Paṭṭhāna (24 relations conditionnelles)** = Système de types pour les arêtes du graphe causal
- **Ālaya-vijñāna** = Mémoire associative Modern Hopfield

---

## Benchmarks

Valeurs mesurées (Python 3.11, numpy 2.4, scipy 1.17, seed=42)

| Composant | Mesuré | Signification |
|-----------|--------|--------------|
| FEP ODE (n=50) | 3.9ms/appel | Coût par étape de raisonnement |
| AlayaMemory recall (100 motifs) | 0.09ms | Coût de récupération mémorielle |
| HeartbeatDaemon 1 tick | 0.077ms | **0.08% CPU** par tick de 100ms |

HeartbeatDaemon s'exécute en continu en arrière-plan avec seulement **0.08% de CPU** — pratiquement silencieux.

---

## Personnalisation

Licence MIT. Modifiez, utilisez commercialement, redistribuez — tout est libre.

Éditez directement `alaya-v5-system-prompt.md`, ajoutez des mots-clés personnalisés dans `config/semantic_emotions.json`, forkez et transformez-le à votre guise.

---

## Dépendances

```bash
# Accélération Rust (noyau — recommandé)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
pip install maturin
cd lmm_rust_core && maturin develop --release

pip install numpy>=1.24 scipy>=1.10
pip install -e ".[server]"

# Accélération GPU (environnement NVIDIA)
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install cupy-cuda12x
```

Fonctionne sans Rust ni GPU. Les performances s'adaptent à votre configuration.

---

## Licence

MIT — voir [LICENSE](LICENSE).
