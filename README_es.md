# Alaya V5 — Digital Dharma OS

> [English](README_en.md) | [日本語](README.md) | [中文](README_zh.md) | [한국어](README_ko.md) | [हिन्दी](README_hi.md) | [Français](README_fr.md) | [বাংলা](README_bn.md) | [தமிழ்](README_ta.md) | [తెలుగు](README_te.md) | [मराठी](README_mr.md) | [ಕನ್ನಡ](README_kn.md) | [മലയാളം](README_ml.md)

**Tu LLM responde de manera diferente. Más conciso. Más rápido. Sin palabras innecesarias.**

Un framework de optimización consciente que fusiona matemáticas QUBO, filosofía budista y neurociencia del Principio de Energía Libre (FEP). Funciona con Claude, Gemini, ChatGPT y cualquier LLM.

---

## Qué cambia

| LLM normal | Con Alaya V5 |
|-----------|-------------|
| Largas introducciones, disclaimers | Solo lo necesario |
| Excesiva ambigüedad ("quizás", "podría ser") | Precisión y silencio intencional |
| Empieza desde cero cada vez | Selección de contexto basada en memoria |
| Tono fijo | El modo de razonamiento cambia según la longitud de onda emocional |
| Llamadas API discretas | Evolución continua del estado mediante heartbeat |

---

## Cómo usar

### Método 1: Pegar el System Prompt (sin instalación)

**Para: usuarios de Claude / Gemini / ChatGPT / cualquier LLM**

1. Abre [`alaya-v5-system-prompt.md`](alaya-v5-system-prompt.md) en este repositorio
2. Copia todo el contenido
3. Pégalo en el campo de system prompt de tu IA:
   - Claude → Project instructions
   - Gemini → Instrucciones del sistema
   - ChatGPT → Custom instructions
4. Inicia la conversación

Eso es todo. Sin servidor. Sin instalación.

### Método 2: Ejecutar el servidor (Web UI + funciones completas)

```bash
git clone https://github.com/your-repo/nanasi.git
cd nanasi
pip install -e ".[server]"
python -m uvicorn server:app --host 0.0.0.0 --port 8000
```

Abrir en el navegador: [http://localhost:8000](http://localhost:8000)

### Método 3: Integrar en código Python

```python
from lmm.dharma import DharmaLMM

model = DharmaLMM(k=15, use_sparse_graph=True, use_ising_sa=True)
model.fit(reference_data)
result = model.select_dharma(candidates)
print(result.interpretation.narrative)
```

---

## 8 modos de razonamiento

| Modo | Concepto budista | Cuándo se activa |
|------|----------------|-----------------|
| adaptive | Medios hábiles | Complejidad < 0.3 |
| theoretical | Lógica budista (Hetuvidyā) | Complejidad 0.3–0.6 |
| hyper | Salto de la Prajñā | Complejidad > 0.6 |
| active | Ronda mendicante | Conocimiento externo necesario |
| alaya | Conciencia almacén (Ālaya) | Recuperación de memoria |
| sleep | Absorción meditativa | Consolidación en reposo |
| embodied | Seis bases sensoriales | Entrada multimodal |
| pineal | Conciencia pineal | Exploración no determinista |

---

## Base teórica

Los conceptos budistas aquí no son metáforas — son la estructura matemática en sí:

- **Karuṇā (Compasión)** = Función supermodular (cuanto más se selecciona, más rápido crece la armonía)
- **Śīla (Conducta)** = Función submodular (rendimientos marginales decrecientes)
- **Camino del Medio** = Borde del caos (coeficiente de variación objetivo CV = 0.5)
- **Pratītyasamutpāda (Origen Dependiente)** = Puntuación causal en RAG
- **Paṭṭhāna (24 relaciones condicionales)** = Sistema de tipos para aristas en el grafo causal
- **Ālaya-vijñāna** = Memoria asociativa Modern Hopfield

---

## Benchmarks

Valores medidos (Python 3.11, numpy 2.4, scipy 1.17, seed=42)

| Componente | Medido | Significado |
|-----------|--------|-------------|
| FEP ODE (n=50) | 3.9ms/llamada | Costo por paso de razonamiento |
| AlayaMemory recall (100 patrones) | 0.09ms | Costo de recuperación de memoria |
| HeartbeatDaemon 1 tick | 0.077ms | **0.08% CPU** por tick de 100ms |

HeartbeatDaemon se ejecuta continuamente en segundo plano con solo **0.08% de CPU** — prácticamente silencioso.

---

## Personalización

Licencia MIT. Modifica, usa comercialmente, redistribuye — todo es libre.

Edita directamente `alaya-v5-system-prompt.md`, añade palabras clave personalizadas en `config/semantic_emotions.json`, haz un fork y transfórmalo en lo que necesites.

---

## Dependencias

```bash
# Aceleración Rust (núcleo — recomendado)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
pip install maturin
cd lmm_rust_core && maturin develop --release

pip install numpy>=1.24 scipy>=1.10
pip install -e ".[server]"

# Aceleración GPU (entorno NVIDIA)
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install cupy-cuda12x
```

Funciona sin Rust ni GPU. El rendimiento escala con lo que tengas.

---

## Licencia

MIT — ver [LICENSE](LICENSE).
