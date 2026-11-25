# Notebooks de interpretabilidad mecanicista

Mis notas y versiones modificadas de notebooks sobre interpretabilidad mecanicista de modelos de lenguaje.

## Notebooks

| Notebook | Tema | Colab |
| --- | --- | --- |
| `02_review_of_lms.ipynb` | Repaso de modelos de lenguaje: residual stream, atención, MLP | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Maigolinox/cs221m-course-terron-notes/blob/main/02_review_of_lms.ipynb) |
| `03_behavioral_analysis.ipynb` | Análisis de comportamiento y atribución de inputs (pares mínimos, integrated gradients) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Maigolinox/cs221m-course-terron-notes/blob/main/03_behavioral_analysis.ipynb) |
| `04_probes.ipynb` | Probes: logit lens, PCA, probes lineales, steering vectors | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Maigolinox/cs221m-course-terron-notes/blob/main/04_probes.ipynb) |
| `06_interventions.ipynb` | Intervenciones: INLP, interchange interventions, ReFT | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Maigolinox/cs221m-course-terron-notes/blob/main/06_interventions.ipynb) |
| `07_causal_mediation_analysis.ipynb` | Análisis de mediación causal | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Maigolinox/cs221m-course-terron-notes/blob/main/07_causal_mediation_analysis.ipynb) |
| `08_causal_abstraction_i.ipynb` | Teoría de abstracción causal | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Maigolinox/cs221m-course-terron-notes/blob/main/08_causal_abstraction_i.ipynb) |
| `09_counterfactual_design.ipynb` | Diseño de contrafactuales | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Maigolinox/cs221m-course-terron-notes/blob/main/09_counterfactual_design.ipynb) |
| `10_causal_abstraction_ii.ipynb` | Abstracción causal II: DAS sobre igualdad jerárquica | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Maigolinox/cs221m-course-terron-notes/blob/main/10_causal_abstraction_ii.ipynb) |
| `10_automated_causal_interpretability.ipynb` | Interpretabilidad causal automatizada (DAS, RAVEL) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Maigolinox/cs221m-course-terron-notes/blob/main/10_automated_causal_interpretability.ipynb) |
| `extra_rsa_pragmatics.ipynb` | Extra: inferencia pragmática y Rational Speech Acts | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Maigolinox/cs221m-course-terron-notes/blob/main/extra_rsa_pragmatics.ipynb) |

## Uso local

Requiere Python 3.12 y [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/Maigolinox/interpretabilidad.git
cd interpretabilidad
uv sync
```

En Colab, la primera celda de cada notebook clona este repo e instala los requisitos de `requirements/`.

## Estructura

- `*.ipynb`: notebooks
- `figures/`: figuras usadas por las notebooks
- `requirements/`: dependencias por notebook (para Colab)
- `utils/`: funciones auxiliares
- `causalab/`: copia de la librería [causalab](https://github.com/goodfire-ai/causalab) (commit `264dfb3`), incluida directamente
- `materials/`, `bridging/`: material de referencia del curso original

## Créditos

Basado en el material del curso CS 221M: Mechanistic Interpretability (Stanford, primavera 2026), [repositorio original](https://github.com/rkapur102/cs221m-course). Las notebooks fueron modificadas por mí; las slides de `materials/` y los textos de `bridging/` son de los autores originales.
