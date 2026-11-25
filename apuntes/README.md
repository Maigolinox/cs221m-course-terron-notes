# Apuntes

Notas de estudio de cada notebook del curso y de la librería `causalab`. Cada archivo sigue la misma estructura: de qué trata, ideas clave, recorrido celda por celda con el código explicado línea por línea, ejercicios, una sección "Ojo" con trampas y dudas, y un resumen corto al final.

Lo que no se pudo comprobar sin ejecutar (no había torch/pyvene a mano) está marcado como **no verificado** dentro de cada archivo. Los números que aparecen vienen de las salidas guardadas en las notebooks o están marcados como calculados a mano.

## Orden sugerido

1. Curso, en orden: 02 → 03 → 04 → 06 → 07 → 08 → 09 → 10 (las dos) → extra.
2. Para entender lo que usan la 08, 09 y 10 por debajo: `causalab/00_vision_general.md` y después los módulos.
3. Demos de causalab, en orden (es un tutorial encadenado).
4. Las tareas de causalab al final; varias están desactualizadas respecto a la librería (ver abajo).

## Curso

| Notebook | Apuntes | Tema |
| --- | --- | --- |
| `02_review_of_lms.ipynb` | [curso/02_review_of_lms.md](curso/02_review_of_lms.md) | tokens y logits, residual stream, MLP vs atención, attention sinks |
| `03_behavioral_analysis.ipynb` | [curso/03_behavioral_analysis.md](curso/03_behavioral_analysis.md) | MMLU few-shot, perturbaciones, memorización |
| `04_probes.ipynb` | [curso/04_probes.md](curso/04_probes.md) | logit lens, PCA, probes lineales, difference-in-means steering |
| `06_interventions.ipynb` | [curso/06_interventions.md](curso/06_interventions.md) | INLP, MaxNN, zero-out vs interchange, ReFT |
| `07_causal_mediation_analysis.ipynb` | [curso/07_causal_mediation_analysis.md](curso/07_causal_mediation_analysis.md) | log odds, patching por capa/token y por cabeza, state tracking |
| `08_causal_abstraction_i.ipynb` | [curso/08_causal_abstraction_i.md](curso/08_causal_abstraction_i.md) | modelos causales, abstracción constructiva, igualdad jerárquica |
| `09_counterfactual_design.ipynb` | [curso/09_counterfactual_design.md](curso/09_counterfactual_design.md) | diseño de contrafactuales, MCQA pointer-value, entity binding, `utils/lecture_09_utils.py` |
| `10_causal_abstraction_ii.ipynb` | [curso/10_causal_abstraction_ii.md](curso/10_causal_abstraction_ii.md) | DAS, abstracción bajo traducción |
| `10_automated_causal_interpretability.ipynb` | [curso/10_automated_causal_interpretability.md](curso/10_automated_causal_interpretability.md) | DAS/DBM en RAVEL (notebook de solo lectura) |
| `extra_rsa_pragmatics.ipynb` | [curso/extra_rsa_pragmatics.md](curso/extra_rsa_pragmatics.md) | Rational Speech Acts, listener pragmático |

## causalab: la librería

| Apuntes | Qué cubre |
| --- | --- |
| [causalab/00_vision_general.md](causalab/00_vision_general.md) | qué es causal abstraction, arquitectura del paquete, flujo de un experimento |
| [causalab/01_causal.md](causalab/01_causal.md) | `causal/`: `CausalModel`, `Mechanism`, `CausalTrace`, contrafactuales |
| [causalab/02_neural.md](causalab/02_neural.md) | `neural/`: pipeline, model units, featurizers, token positions, `pyvene_core` |
| [causalab/03_experiments.md](causalab/03_experiments.md) | `experiments/`: métricas, entrenamiento DAS/DBM, jobs, visualizaciones |
| [causalab/04_tasks.md](causalab/04_tasks.md) | `tasks/`: MCQA, IOI, entity binding, suma |

## causalab: demos (tutorial)

| Notebook | Apuntes |
| --- | --- |
| `demos/causal_model_demo.ipynb` | [causalab/demos/causal_model_demo.md](causalab/demos/causal_model_demo.md) |
| `demos/onboarding_tutorial/01_define_MCQA_task.ipynb` | [causalab/demos/01_define_MCQA_task.md](causalab/demos/01_define_MCQA_task.md) |
| `.../02_trace_residual_stream.ipynb` | [causalab/demos/02_trace_residual_stream.md](causalab/demos/02_trace_residual_stream.md) |
| `.../03_localize_with_patching.ipynb` | [causalab/demos/03_localize_with_patching.md](causalab/demos/03_localize_with_patching.md) |
| `.../04_train_DAS_and_DBM.ipynb` | [causalab/demos/04_train_DAS_and_DBM.md](causalab/demos/04_train_DAS_and_DBM.md) |
| `.../05_PCA_DBM_DAS.ipynb` | [causalab/demos/05_PCA_DBM_DAS.md](causalab/demos/05_PCA_DBM_DAS.md) |
| `.../06_boundless_DAS.ipynb` | [causalab/demos/06_boundless_DAS.md](causalab/demos/06_boundless_DAS.md) |
| `.../07_cross_model_patch.ipynb` | [causalab/demos/07_cross_model_patch.md](causalab/demos/07_cross_model_patch.md) |

## causalab: tareas

| Notebook | Apuntes |
| --- | --- |
| `tasks/entity_binding/notebooks/01_task_and_causal_model.ipynb` | [causalab/tasks/entity_binding_01_task_and_causal_model.md](causalab/tasks/entity_binding_01_task_and_causal_model.md) |
| `.../02_token_positions_demo.ipynb` | [causalab/tasks/entity_binding_02_token_positions_demo.md](causalab/tasks/entity_binding_02_token_positions_demo.md) |
| `.../03_counterfactuals_and_positional_variables.ipynb` | [causalab/tasks/entity_binding_03_counterfactuals_and_positional_variables.md](causalab/tasks/entity_binding_03_counterfactuals_and_positional_variables.md) |
| `.../04_interchange_intervention_test.ipynb` | [causalab/tasks/entity_binding_04_interchange_intervention_test.md](causalab/tasks/entity_binding_04_interchange_intervention_test.md) |
| `tasks/general_addition/notebooks/01_addition_task_and_causal_models.ipynb` | [causalab/tasks/general_addition_01_addition_task_and_causal_models.md](causalab/tasks/general_addition_01_addition_task_and_causal_models.md) |
| `.../02_token_positions_for_digits.ipynb` | [causalab/tasks/general_addition_02_token_positions_for_digits.md](causalab/tasks/general_addition_02_token_positions_for_digits.md) |
| `.../03_counterfactuals_and_carry_variables.ipynb` | [causalab/tasks/general_addition_03_counterfactuals_and_carry_variables.md](causalab/tasks/general_addition_03_counterfactuals_and_carry_variables.md) |
| `tasks/IOI/ioi_demo.ipynb` | [causalab/tasks/IOI_demo.md](causalab/tasks/IOI_demo.md) |

## Cosas a tener presentes

- **Tareas de causalab actualizadas**: IOI, entity binding y general addition se portaron a la API actual de `causalab` y sus notebooks corren completas (salvo la 02 de general addition, que necesita tokenizers con acceso restringido de Gemma y Llama). Varios apuntes de esas tareas se escribieron antes del port: cada uno tiene una nota al principio con lo que cambió.
- **Demos 05 y 06**: se corrigió el guardado del dataset. Las demos 02 a 07 usan `meta-llama/Llama-3.2-1B-Instruct`, que necesita aceptar la licencia en Hugging Face y un token.
- **Notebooks del curso ejecutadas**: la 02, 04, 06 y 07 están ejecutadas en GPU y tienen las respuestas escritas. La 03 se corre en Colab, porque su modelo (7B) no entra en 8 GB de GPU.
- **Notebook 10 automatizada**: llama a scripts de una carpeta `causalab-internal` que no está en ningún repo, así que solo se puede leer.
- **pyvene** no está incluido en el repo; donde el código depende de su comportamiento interno, los apuntes lo marcan.
