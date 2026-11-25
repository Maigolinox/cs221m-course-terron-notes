# 06 — Boundless DAS: rotación y máscara por dimensión aprendidas juntas (`06_boundless_DAS.ipynb`)

> **Actualización (noviembre 2026):** se corrigió el guardado del dataset: ahora se escribe en un archivo `.json` en lugar de en una carpeta. El problema de "Ojo" sobre `save_counterfactual_examples` ya está resuelto.

Notebook: `causalab/demos/onboarding_tutorial/06_boundless_DAS.ipynb`.

Sexta notebook del onboarding. Junta las dos ideas de la 04 ([04_train_DAS_and_DBM.md](04_train_DAS_and_DBM.md)) y la 05 ([05_PCA_DBM_DAS.md](05_PCA_DBM_DAS.md)):
- de **DAS**: una rotación ortogonal **entrenable** del espacio de activaciones;
- de **DBM con `tie_masks=False`**: una máscara **por dimensión** (con temperatura y L1) que decide qué coordenadas rotadas patchear.

El resultado es **Boundless DAS**: en vez de fijar de antemano la dimensión `k` del subespacio (como en DAS), se aprende una rotación completa y la máscara decide cuántas y cuáles dimensiones forman el subespacio. Se aplica a la variable `answer` con contrafácticos `different_symbol`, sobre tres tipos de componentes: residual stream (un modelo por capa), attention heads (todas juntas) y MLPs (todos juntos). La pregunta: ¿qué componentes, y cuántas dimensiones dentro de cada uno, hacen falta para mover la respuesta?

Sigue en [07_cross_model_patch.md](07_cross_model_patch.md), que cambia de tema: patchear activaciones de un modelo en otro. Referencia: [../02_neural.md](../02_neural.md) y [../03_experiments.md](../03_experiments.md).

---

## Ideas clave / teoría

### De DAS a Boundless DAS

En DAS (04) el subespacio tiene una dimensión `k` elegida a mano. Si `k` es muy chico, no entra la variable; si es muy grande, se patchea información de más y aumenta el riesgo de overfitting. **Boundless DAS** (la idea viene de Wu et al., 2023, "Interpretability at Scale") aprende también **el tamaño** del subespacio.

Con una rotación ortogonal `R ∈ R^{d×n}` (acá `n = d`, una rotación completa) y una máscara `m ∈ R^n`:

  `f_b = x_b R`,  `f_s = x_s R`,  `g = σ(m / τ)`
  `f' = (1 − g) ⊙ f_b + g ⊙ f_s`
  `x' = f' Rᵀ + error_b`

- Con `n = d` y `R` ortogonal cuadrada, `error_b = x_b − x_b R Rᵀ = 0`: la reconstrucción es exacta.
- Las dimensiones rotadas con `g_j → 1` forman el subespacio que se intercambia. Su cantidad, `Σ_j 1[σ(m_j) > 0.5]`, es la **dimensión aprendida**.
- Si `g` fuera binaria con exactamente `k` unos, esto sería DAS con `W` = las `k` columnas de `R` seleccionadas. Boundless DAS lo relaja: `g` continua, con temperatura `τ` que baja de 1 a 0.001, y penalización `λ · Σ g / n` para preferir subespacios chicos.
- **Ambas cosas se entrenan**: la rotación `R` (vía la parametrización ortogonal, `trainable=True`) y la máscara `m`.

Diferencia con el paper: según recuerdo, en el paper original la máscara se parametriza con "fronteras" aprendidas (un intervalo de dimensiones contiguas). Esta implementación usa una sigmoide **independiente por dimensión** (`FeatureMaskIntervention` con `tie_masks=False`). Como la rotación es libre, el efecto es similar: la rotación puede reordenar las dimensiones.

### Relación con DBM

| | featurizer | máscara | qué selecciona |
|---|---|---|---|
| DBM `tie_masks=True` (04) | identidad | 1 escalar por unidad | unidades enteras |
| DBM sobre PCA (05) | PCA fija | 1 por componente | componentes PCA |
| **Boundless DAS (06)** | rotación **entrenable** | 1 por dimensión rotada | un subespacio aprendido dentro de cada unidad |

En código, Boundless DAS **es** `train_DBM_feature_masks` con un `SubspaceFeaturizer` entrenable ya puesto en cada unidad.

### Feature count heatmap

Como el resultado por unidad es un **número de dimensiones seleccionadas**, la visualización (`plot_feature_counts`) muestra, por ejemplo para attention heads, una grilla capa × head con cuántas de las 64 dimensiones de cada head se seleccionaron. Una head con 0 no participa; una con muchas participa con un subespacio grande.

---

## El código de la librería que usa esta notebook

### `train_boundless_DAS` (`causalab/causalab/experiments/jobs/boundless_DAS_feature_mask_grid.py`)

```python
config = merge_with_defaults(config)
interchange_target = copy.deepcopy(interchange_target)
n_features = config["DAS"]["n_features"]
...
for unit in all_units:
    if unit.featurizer.id == "null":
        if n_features > unit.shape[0]:
            logger.warning(...)
        unit.set_featurizer(
            SubspaceFeaturizer(shape=(unit.shape[0], n_features), trainable=True, id="boundless_DAS")
        )
...
result = train_DBM_feature_masks(..., config=config, title=title)
result["metadata"]["experiment_type"] = "boundless_DAS"
result["metadata"]["n_features"] = n_features
```

- Hace `deepcopy` del target para no mutar el del usuario. Así, a diferencia de `train_DAS`/DBM, los targets de la notebook quedan limpios después de entrenar.
- A cada unidad sin featurizer le pone una rotación ortogonal **aleatoria** `shape=(d_unidad, n_features)`, entrenable. Si `n_features > d`, avisa (proyectaría a un espacio más grande).
- Delega en `train_DBM_feature_masks` (`DBM_feature_masks_grid.py`), que fuerza `intervention_type="mask"` y `tie_masks=False` y llama `train_interventions` (`experiments/train.py`). Como los featurizers ya tienen id `"boundless_DAS"`, `_initialize_featurizers` no los toca.
- En el loop de `causalab/causalab/neural/pyvene_core/interchange.py`, el optimizador recibe `v.parameters()` de cada intervención. `FeatureMaskIntervention` guarda el featurizer como atributo (`self._featurizer = featurizer`), y como es un `nn.Module` queda registrado como submódulo: sus parámetros (la rotación) entran en el optimizador junto con `self.mask`. Por eso se entrenan juntos. (Esto lo deduzco de la semántica de `nn.Module.__setattr__`; no lo ejecuté.)
- Al final: `feature_indices = [j : σ(m_j) > 0.5]` por unidad. La evaluación en train/test usa la intervención `interchange` con la rotación aprendida y esos índices como `subspaces`.

### Formato de entrada: target único o dict por capa

`train_DBM_feature_masks` (y por lo tanto `train_boundless_DAS`) acepta:
- un `InterchangeTarget` → un entrenamiento, devuelve `train_score`, `test_score`, `feature_indices`;
- un `dict[int, InterchangeTarget]` (capa → target) → un entrenamiento **por capa**, devuelve `train_scores`/`test_scores` por capa y agrega a la metadata `best_layer`, `best_test_score`, `avg_test_score`.

### `mode="one_target_per_layer"` (`causalab/causalab/experiments/interchange_targets.py`)

`_group_units_into_targets` agrupa todas las unidades de una misma capa (acá, las 7 posiciones) en **un** `InterchangeTarget([group])` con clave `(layer,)`. Todas comparten contrafáctico y se entrenan juntas.

---

## Recorrido por la notebook

### Celda 1 — setup

```python
from causalab.experiments.interchange_targets import (
    build_residual_stream_targets, build_attention_head_targets, build_mlp_targets,
)
from causalab.experiments.jobs.boundless_DAS_feature_mask_grid import train_boundless_DAS
...
pipeline = LMPipeline(model_name, max_new_tokens=1, device=device,
                      dtype=torch.bfloat16 if device == "cuda" else torch.float32,
                      max_length=32)
pipeline.tokenizer.padding_side = "left"
```

- `padding_side = "left"` es redundante: `LMPipeline` ya lo pone por defecto.
- Salida guardada: cuda, 16 capas, 32 heads por capa, `hidden_size` 2048.

### Celda 3 — datasets

```python
size = 128
train_dataset = [different_symbol() for _ in range(size)]
filtered_train_dataset = filter_dataset(dataset=train_dataset, pipeline=pipeline,
                                        causal_model=causal_model, metric=checker, batch_size=128)
test_dataset = [different_symbol() for _ in range(size)]
filtered_test_dataset = filter_dataset(...)

train_dataset_path = "./temp_train_datasets/different_symbol/filtered_dataset"
test_dataset_path = "./temp_test_datasets/different_symbol/filtered_dataset"
os.makedirs(train_dataset_path, exist_ok=True)
os.makedirs(test_dataset_path, exist_ok=True)
save_counterfactual_examples(filtered_train_dataset, train_dataset_path)
save_counterfactual_examples(filtered_test_dataset, test_dataset_path)
```

- `different_symbol` porque la variable es `answer`: el contrafáctico tiene la misma pregunta con letras nuevas, así que el label es "la letra nueva en la misma posición".
- 128 pares por split (el doble que en la 04).
- **Ojo**: igual que en la 05, crea un **directorio** con el nombre del archivo y después intenta escribir JSON en él. Con el código actual (`save_counterfactual_examples` hace `open(path, "w")`) falla. La salida guardada ("Saving the dataset (0/1 shards)... 112 / 111 examples", "Saved train dataset: ...") es de una versión anterior que guardaba datasets HF en directorios; incluso los `print` de la salida no coinciden con los del código actual. Del output se deduce que quedaron 112 ejemplos de train y 111 de test.
- Además, la ruta coincide con el directorio de datasets que usa la 04 (`./temp_train_datasets/different_symbol/`), donde la 04 guarda `filtered_dataset.json`. No se pisan (los nombres difieren), pero conviene saber que conviven.

### Celda 4 — posiciones

Las mismas 7 posiciones de la 03/04 (`correct_symbol`, `correct_symbol_period`, `symbol0`, `symbol0_period`, `symbol1`, `symbol1_period`, `last_token`).

### Celda 6 — config

```python
config: PartialExperimentConfig = {
    "train_batch_size": 128,
    "evaluation_batch_size": 64,
    "training_epoch": 6,
    "init_lr": 0.001,
    "DAS": {"n_features": 2048},
    "masking": {
        "regularization_coefficient": 0.0001,
        "temperature_annealing_fraction": 0.5,
        "temperature_schedule": (1.0, 0.001),
    },
}
```

- `train_batch_size: 128` con ~112 ejemplos → **un batch por época**, 6 pasos de optimización en total. La temperatura baja de 1 a 0.001 en los primeros 3 pasos. Es muy poco entrenamiento para una rotación de 2048×2048; la máscara probablemente se mueve poco desde su valor inicial (`σ(0) = 0.5`). No lo pude verificar.
- `n_features: 2048 = hidden_size`: rotación **completa**, sin pérdida (`error = 0`).
- `regularization_coefficient: 1e-4`: L1 muy débil.
- El comentario dice que `train_boundless_DAS` setea `intervention_type="mask"` y `tie_masks=False`. En rigor lo hace `train_DBM_feature_masks`, que es a quien delega.

### Celda 8 — residual stream, un modelo por capa

```python
layers = list(range(pipeline.model.config.num_hidden_layers))       # 0..15, sin embeddings
residual_targets = build_residual_stream_targets(pipeline=pipeline, layers=layers,
                                                 token_positions=token_positions,
                                                 mode="one_target_per_layer")
residual_targets_by_layer = {key[0]: target for key, target in residual_targets.items()}
```

- `one_target_per_layer` → `{(layer,): target}` con 7 unidades cada uno. La comprensión de dict lo convierte a `{layer: target}`, el formato que `train_DBM_feature_masks` interpreta como "un entrenamiento por capa".
- 16 targets. A diferencia de la 03/04, **no** se incluye la capa -1 (embeddings).

```python
residual_result = train_boundless_DAS(
    causal_model=causal_model, interchange_target=residual_targets_by_layer,
    train_dataset_path=train_dataset_path, test_dataset_path=test_dataset_path,
    pipeline=pipeline, target_variable_group=("answer",),
    output_dir="./boundless_DAS_outputs/residual_stream", metric=checker, config=config,
)
print(f"  Best layer: {residual_result['metadata']['best_layer']}")
print(f"  Best test score: {residual_result['metadata']['best_test_score']:.3f}")
print(f"  Average test score: {residual_result['metadata']['avg_test_score']:.3f}")
```

- Por capa: 7 unidades, cada una con su propia rotación 2048×2048 entrenable y su máscara de 2048. Las 7 se intervienen **a la vez** con el mismo contrafáctico.
- Salida guardada: mejor capa 1, test 1.000; promedio sobre capas 0.977. Figura con 3 ejes (conteo de features + scores).
- Por qué casi todo da alto: en `different_symbol`, los únicos tokens que difieren entre base y contrafáctico son los símbolos, y las 7 posiciones cubren ambos símbolos, sus puntos y el último token. En capas tempranas la letra está en los símbolos; en capas tardías está en el último token. En cualquier capa, patchear las 7 posiciones juntas (con libertad de elegir dimensiones) alcanza para transferir la letra. Por eso este experimento dice poco sobre *dónde* está `answer` por capa; lo informativo es **cuántas dimensiones** usa cada posición (el heatmap de conteos).

### Celda 10 — attention heads, todas juntas

```python
heads = list(range(pipeline.model.config.num_attention_heads))
last_token_position = token_positions_dict["last_token"]
attention_targets = build_attention_head_targets(pipeline=pipeline, layers=layers, heads=heads,
                                                 token_position=last_token_position,
                                                 mode="one_target_all_units")
attention_target = attention_targets[("all",)]
attention_config = {**config, "DAS": {"n_features": 64}}
attention_result = train_boundless_DAS(..., interchange_target=attention_target,
                                       output_dir="./boundless_DAS_outputs/attention_heads",
                                       config=attention_config)
```

- 512 heads, **solo en el último token** (a diferencia de la DBM de la 04, que usaba todas las posiciones). Pregunta: ¿qué heads escriben la letra en el último token?
- `{**config, "DAS": {"n_features": 64}}`: copia el config y reemplaza la clave `DAS` entera. 64 = `head_dim` = 2048/32: rotación completa del espacio de cada head.
- Target único → un solo entrenamiento con 512 rotaciones de 64×64 y 512×64 máscaras.
- Salida guardada: train 0.991, test 0.991. Figura de 1800×640 (heatmap capa × head con número de dimensiones seleccionadas).

### Celda 12 — MLPs, todos juntos

```python
mlp_targets = build_mlp_targets(pipeline=pipeline, layers=layers, token_positions=token_positions,
                                mode="one_target_all_units")
mlp_target = mlp_targets[("all",)]
mlp_config = {**config, "DAS": {"n_features": 2048}}
mlp_result = train_boundless_DAS(..., interchange_target=mlp_target,
                                 output_dir="./boundless_DAS_outputs/mlp", config=mlp_config)
```

- 16 capas × 7 posiciones = 112 unidades `MLP(layer, pos, "mlp_output")`, cada una con una rotación 2048×2048. Son unos 470 M parámetros entrenables en total (112 × 2048²); pesado en memoria.
- `mlp_config` es igual a `config` (ya tenía `n_features: 2048`).
- Salida guardada: train 1.000, test 0.991.

### Celda 14 — comparación

```python
scores = {
    f"Residual Stream (layer {residual_result['metadata']['best_layer']})": residual_result["metadata"]["best_test_score"],
    "Attention Heads": attention_result["test_score"],
    "MLPs": mlp_result["test_score"],
}
best_component = max(scores, key=lambda k: scores[k])
```

- Toma el mejor test de cada tipo y elige el máximo. Salida guardada: residual stream (capa 1) con 1.000, heads 0.991, MLPs 0.991.
- La comparación es poco informativa: los tres están prácticamente saturados, y no son experimentos equivalentes (el residual stream es "la mejor de 16 capas", los otros son un único modelo; las heads miran solo el último token).

### Celda 15 — Takeaways

- Residual stream por capa: muestra cómo cambian las representaciones con la profundidad.
- Heads: el heatmap de conteos dice qué heads contribuyen y con cuántas dimensiones.
- MLPs: lo mismo por (capa, posición).
- DBM (`tie_masks=True`) selecciona unidades enteras; Boundless DAS (`tie_masks=False`) selecciona dimensiones dentro de cada unidad. Es más fino pero tiene muchos más parámetros.

---

## Ejercicios

No hay celdas de ejercicio. Para sacarle información real al experimento:
- Mirar los **heatmaps de conteo** (`boundless_DAS_outputs/*/heatmaps/answer_features.png`), no solo los scores: los scores están saturados.
- Subir `regularization_coefficient` (por ejemplo 0.01–1) y/o las épocas para forzar subespacios chicos, y ver en qué capa/posición sobrevive la información con pocas dimensiones.
- Repetir con `answer_position` y `same_symbol_different_position`; ahí debería verse la capa 10–11 en `last_token`, como en la 04.

---

## Ojo / errores comunes / dudas

- **Guardado de datasets en un directorio** (celda 3): falla con el código actual; las salidas guardadas son de una versión anterior.
- **Muy pocos pasos de entrenamiento**: batch 128 con ~112 ejemplos → 1 paso por época, 6 en total. Con L1 de 1e-4, no está claro que la máscara se aleje de su inicialización. Los scores saturados podrían deberse a que se patchea "casi todo" más que a un subespacio aprendido. Habría que mirar los conteos de features.
- **`deepcopy` del target**: `train_boundless_DAS` hace `copy.deepcopy(interchange_target)`. Las unidades guardan sus `TokenPosition`, y cada `TokenPosition` guarda una referencia al `pipeline` (`self.pipeline`). Si `deepcopy` sigue esa referencia, **copia también el pipeline con el modelo entero** en cada llamada. Lo deduzco del código (`token_position_builder.py`, `LM_units.py`); no lo verifiqué ejecutando, pero si te quedás sin memoria GPU es un sospechoso. El modelo copiado solo se usaría para tokenizar.
- **Posiciones duplicadas en el mismo target**: en el residual stream por capa, `correct_symbol` coincide con `symbol0` o `symbol1` (y `correct_symbol_period` con el punto correspondiente). Hay dos unidades sobre el mismo vector, cada una con su rotación y máscara. No verifiqué cómo pyvene compone dos intervenciones sobre la misma posición (presumiblemente en secuencia).
- **Boundless DAS no incluye la capa -1** en esta notebook (`layers = range(16)`).
- **Attention heads solo en `last_token`** acá, a diferencia de la DBM de la 04 (todas las posiciones). No compares directamente las heads elegidas.
- **Memoria de los MLPs**: 112 rotaciones de 2048×2048 más sus gradientes y estados de AdamW.
- **`best_test_score` del residual stream es un máximo sobre 16 capas** (sesgo de selección); los de heads/MLPs son un único modelo.

---

## Resumen de una línea por sección

- **Setup / datasets**: `different_symbol` (128 → ~112 tras filtrar) para la variable `answer`.
- **Config**: rotación completa (`n_features` = dimensión de la unidad), máscara por dimensión, L1 1e-4, 6 épocas.
- **Residual stream por capa**: 16 modelos (7 posiciones cada uno); mejor capa 1 con test 1.0, promedio 0.977. Saturado porque las 7 posiciones cubren todos los tokens que difieren.
- **Attention heads**: 512 heads en el último token, rotación 64×64 por head; test 0.991.
- **MLPs**: 112 (capa, posición) con rotación 2048×2048; test 0.991.
- **Comparación / takeaways**: Boundless DAS = DBM por dimensión sobre una rotación entrenable; lo informativo es cuántas dimensiones se seleccionan y dónde.
