# 05 — PCA como análisis no supervisado y como featurizer para DAS y DBM (`05_PCA_DBM_DAS.ipynb`)

> **Actualización (noviembre 2026):** se corrigió el guardado del dataset: ahora se escribe en un archivo `.json` en lugar de en una carpeta. El problema de "Ojo" sobre `save_counterfactual_examples` ya está resuelto.

Notebook: `causalab/demos/onboarding_tutorial/05_PCA_DBM_DAS.ipynb`.

Quinta notebook del onboarding. Hasta la 04 ([04_train_DAS_and_DBM.md](04_train_DAS_and_DBM.md)) todo fue **supervisado** por el modelo causal. Acá se agrega un método **no supervisado**: recolectar activaciones del residual stream (capa 10, último token, que en la 04 resultó ser el mejor lugar para `answer_position`) y hacer PCA/SVD para ver si los inputs se agrupan según las variables causales. Después se usa PCA de dos formas como **featurizer** (una base de coordenadas para intervenir):

1. **PCA → DAS**: inicializar el subespacio de DAS con las primeras componentes principales en vez de una rotación aleatoria.
2. **PCA → DBM**: dejar fija la base de PCA y aprender con DBM **qué componentes** hay que patchear.

La pregunta: ¿la estructura de varianza de las activaciones refleja las variables causales, y sirve como punto de partida para los métodos supervisados?

Sigue en [06_boundless_DAS.md](06_boundless_DAS.md): ahí la rotación **y** la máscara por dimensión se aprenden juntas (Boundless DAS), que es la versión entrenable de lo que acá se hace con PCA fija. Referencia: [../02_neural.md](../02_neural.md) (featurizers, `collect_features`, `compute_svd`) y [../03_experiments.md](../03_experiments.md).

---

## Ideas clave / teoría

### PCA / SVD sobre activaciones

Tenemos una matriz `X ∈ R^{N×d}` (N = 1024 inputs, d = 2048): la activación en una ubicación fija para cada input. SVD truncada: `X ≈ U Σ Vᵀ`, con `V ∈ R^{d×n}` de columnas ortonormales (las **componentes**). Proyectar `X V` da las coordenadas de cada input en esas direcciones.

- Si antes de la SVD se **centra** (y estandariza) `X`, la SVD es PCA: las componentes son las direcciones de máxima varianza.
- Si **no** se centra (`normalize=False`, que es lo que se usa en la primera parte de esta notebook), la primera componente tiende a apuntar hacia la **media** de los datos (la dirección con más "energía" `‖Xv‖²`, no con más varianza). Por eso, en la salida guardada, PC0 explica solo el 0.4% de la varianza y PC1 el 23.6%: `explained_variance_ratio_` de sklearn mide la **varianza** de cada proyección, y la proyección sobre la dirección de la media es casi constante. (Esta explicación es mía, a partir del código; la notebook no lo comenta.)

**Para qué sirve.** Si al colorear los puntos proyectados por `answer_position` aparecen dos grupos separados, la variable está codificada (al menos en parte) **linealmente** y en pocas direcciones de alta varianza. Es exploración, genera hipótesis; no prueba causalidad (una dirección puede separar los datos y no ser usada por la red).

### PCA como featurizer

Un featurizer es una base de coordenadas en la que se interviene (ver la 04). `SubspaceFeaturizer(rotation_subspace=V)` usa `W = V` (o `V[:, :k]`):
- features: `f = x W` (las coordenadas PCA del vector, sin centrar);
- reconstrucción: `x = f Wᵀ + error`, con `error = x − x W Wᵀ`.

Entonces:
- **DAS inicializado con PCA** (`trainable=True`, `k = 2`): la intervención es la de DAS, `x' = W Wᵀ x_s + (I − W Wᵀ) x_b`, pero `W` arranca en las 2 primeras componentes y el entrenamiento la ajusta. Es un *warm start*: si PCA ya capturó la dirección relevante, DAS arranca cerca.
- **DBM sobre componentes PCA** (`trainable=False`, `k = 512`): la base queda fija; se aprende una máscara por componente, `f'_j = (1 − g_j) f_{b,j} + g_j f_{s,j}`. Las componentes con `g_j = 1` son las que, patcheadas, hacen que la red se comporte como el modelo causal. Responde: "¿qué componentes principales son causalmente relevantes para `answer_position`?". Ojo que la parte de `x` fuera del span de las 512 componentes (`error`) **siempre** queda del base.

---

## El código de la librería que usa esta notebook

### `collect_features` (`causalab/causalab/neural/pyvene_core/collect.py`)

- `prepare_intervenable_model(pipeline, model_units, "collect")`: el tipo de intervención es `FeatureCollectIntervention` (`featurizers.py`), que **no modifica** la activación sino que la devuelve (featurizada; con el featurizer identidad, cruda).
- Por batch: toma solo `example["input"]` (el base; los contrafácticos se ignoran), calcula índices con cada unidad, tokeniza, y `_collect_activations_single_batch` hace `intervenable_model(inputs, unit_locations={"sources->base": (idx, idx)})[0][1]` (pyvene devuelve `((salida_base, activaciones_recolectadas), salida_cf)`).
- Reshapea cada activación a `(-1, hidden)` y las acumula. Devuelve `{unit_id: tensor (N, d)}`.

### `compute_svd` (`collect.py`)

- `n = min(min(N, d) − 1, n_components)`.
- Si `normalize`: resta la media y divide por el desvío (con `ε = 1e-6`), o sea PCA estandarizada.
- `TruncatedSVD(n_components=n, algorithm="randomized").fit(features)` (sklearn). **TruncatedSVD no centra los datos**: por eso, sin `normalize`, no es PCA estricta.
- Devuelve por unidad `{"components": (n, d), "explained_variance_ratio", "rotation": componentsᵀ (d, n) como tensor, "mean", "std", "n_components"}`. `rotation` es lo que se le pasa a `SubspaceFeaturizer`.

### `collect_and_compute_PCA` (`causalab/causalab/experiments/jobs/collect_PCA.py`)

1. Junta todas las unidades de todos los targets y llama **una vez** a `collect_features`.
2. `svd_results_by_unit = compute_svd(features_float32, n_components, normalize)`: una SVD **por unidad** (se usa para inicializar featurizers).
3. Por target combina las features de sus unidades:
   - `combine_mode="concatenate"` (default): `torch.cat(dim=1)` → `(N, U·d)`, un espacio conjunto;
   - `combine_mode="stack"`: `torch.cat(dim=0)` → `(U·N, d)`, todas las unidades como muestras de un mismo espacio, en orden **unit-major** (primero las N filas de la unidad 1, después las de la 2, etc.).
4. SVD sobre lo combinado → `svd_results_by_target` (se usa para graficar).
5. Guarda metadata, features (`.safetensors`), resúmenes de SVD (`.json`) y plots con `plot_pca_scatter`.
6. Devuelve `features_by_target`, `svd_results_by_target`, `svd_results_by_unit` (claves = `unit.id`), `metadata`, `output_paths`. Las claves de los dicts "by_target" son strings: `(10, "last_token")` → `"10__last_token"`, `("all",)` → `"all"`.
- Default `normalize=True` (en esta notebook se pasa `False` en la primera llamada y se deja el default en la de stack).

### `plot_pca_scatter` (`causalab/causalab/experiments/visualizations/pca_scatter.py`)

- Si el `svd_result` tiene `mean`/`std`, normaliza las features igual que en la SVD; proyecta `features @ rotation` y grafica las componentes de cada tupla (2D o 3D). Los ejes muestran el % de varianza explicada.
- `labels` controla el color y `shape_labels` el marcador (hay solo 10 formas).
- **Auto-replicación**: si `len(labels)` divide a `len(features)`, hace `list(labels) * n_repeats`. Esto es correcto para labels **por muestra** en modo stack (porque cada bloque de N filas es otra unidad sobre las mismas muestras), pero **no** para labels **por unidad** (ver "Ojo").

### `SubspaceFeaturizer` y `train_DBM_feature_masks`

- `SubspaceFeaturizer(rotation_subspace=R, trainable=..., id=...)` (`featurizers.py`): crea un `LowRankRotateLayer(d, k)` sin inicialización ortogonal, **copia** `R` en `weight`, lo envuelve con la parametrización ortogonal de PyTorch y le pone `requires_grad_(trainable)`. `n_features = k`. (Que la parametrización ortogonal preserve exactamente la `R` copiada depende de cómo `torch.nn.utils.parametrizations.orthogonal` inicializa la parametrización; como las columnas de `R` ya son ortonormales, debería quedar igual o casi igual, pero no lo verifiqué.)
- `unit.set_featurizer(f)` (`model_units.py`): reemplaza el featurizer y vuelve a chequear los `feature_indices` contra `n_features`.
- `train_DBM_feature_masks` (`causalab/causalab/experiments/jobs/DBM_feature_masks_grid.py`): acepta un target único o un dict `{layer: target}`. Exige que todas las unidades tengan `featurizer.n_features` definido (`_get_n_features_by_unit`), fuerza `intervention_type="mask"` y `featurizer_kwargs={"tie_masks": False}`, llama `train_interventions` (que **respeta** featurizers con id distinto de `"null"`), y dibuja un heatmap de "cuántas features se seleccionaron" con `plot_feature_counts`. Devuelve `train_score`, `test_score`, `feature_indices` (`{unit_id: [índices]}`), `n_features_by_unit`, `output_paths["heatmap"]`.

---

## Recorrido por la notebook

### Celda 0 — setup

Mismo pipeline (Llama-3.2-1B, `max_new_tokens=1`, `max_length=32`). Salida guardada: corrió en `cuda`, 16 capas.

### Celda 3 — dataset sin filtrar

```python
size = 1024
dataset = [different_symbol() for _ in range(size)]
dataset_path = "./temp_pca_dataset"
os.makedirs(dataset_path, exist_ok=True)
save_counterfactual_examples(dataset, dataset_path)
sample = dataset[0]
print(f"\nSample input: {sample['input'].to_dict()}")
```

- 1024 pares `different_symbol`. Para PCA solo importan los inputs base; los contrafácticos no se usan.
- **Sin filtrar**: no se hacen intervenciones, solo se miran activaciones, así que se quiere ver la distribución completa (incluidos los casos donde el modelo se equivoca).
- **Ojo**: `os.makedirs(dataset_path)` crea un **directorio** y después `save_counterfactual_examples` intenta `open(dataset_path, "w")`. Con el código actual de la librería (que escribe JSON a un archivo) eso falla con `IsADirectoryError`. La salida guardada muestra "Saving the dataset (0/1 shards)", que es el mensaje de `datasets.Dataset.save_to_disk` de HuggingFace: la notebook se corrió con una versión anterior de la librería que guardaba datasets HF en directorios. Lo mismo pasa en las celdas 28 de esta notebook y 3 de la 06. Para correrla con el código actual habría que usar rutas de archivo (`.../dataset.json`) y no crear el directorio.
- El `to_dict()` de la salida guardada no incluye `answer`, `answer_position` ni `raw_output`, otra señal de que es de una versión anterior.

### Celda 5 — labels desde las variables causales

```python
labels_answer_position = []
labels_answer = []
for example in dataset:
    trace = example["input"]
    labels_answer_position.append(trace["answer_position"])
    labels_answer.append(trace["answer"])
```

- Para cada input base se leen dos variables del trace: la posición (0/1) y la letra (A–Z). Sirven solo para colorear los gráficos, no entran en la PCA.
- Salida guardada: 517 / 507 por posición; letras entre 2.7% y 5.0% cada una.

### Celda 7 — qué ubicación analizar

```python
all_token_positions = create_token_positions(pipeline)
last_token_pos = all_token_positions["last_token"]
layers_to_analyze = [10]
targets = build_residual_stream_targets(pipeline=pipeline, layers=layers_to_analyze,
                                        token_positions=[last_token_pos],
                                        mode="one_target_per_unit")
```

- Un solo target `(10, "last_token")` con una unidad `ResidualStream(Layer-10, block_output, Token-last_token)`. Es la ubicación donde DAS encontró mejor `answer_position` en la 04.
- El comentario "layers 12-15 are interesting" no coincide con el código (solo capa 10).

### Celda 9 — recolectar y hacer la SVD

```python
result = collect_and_compute_PCA(
    interchange_targets=targets,
    data=dataset,
    pipeline=pipeline,
    labels=labels_answer_position,
    component_tuples=[(0, 1), (0, 2), (1, 2)],
    n_components=512,
    normalize=False,
    output_dir="./PCA_results_answer_position",
    verbose=True,
)
```

- `component_tuples`: qué pares de componentes graficar (3 gráficos 2D).
- `n_components=512`: se calculan 512 componentes (de un máximo de `min(1024, 2048) − 1 = 1023`). Más de las necesarias para graficar, pero se usan después para DBM.
- `normalize=False`: SVD sin centrar. El markdown de la celda 8 dice "with normalization", pero el código no normaliza.
- Por dentro: 32 batches de 32 → features `(1024, 2048)` → `compute_svd` → 3 scatter plots coloreados por `answer_position`.
- Salida guardada: tres figuras, guardadas en `./PCA_results_answer_position/plots`.

### Celda 12 — varianza explicada

```python
for key_str, svd_result in result["svd_results_by_target"].items():
    explained_var = svd_result["explained_variance_ratio"]
    cumulative_var = sum(explained_var)
    print(f"  Total variance explained by {len(explained_var)} components: {cumulative_var:.1%}")
    for i, var in enumerate(explained_var[:3]):
        print(f"    PC{i}: {var:.1%}")
```

- Salida guardada: 512 componentes explican 99.8%; PC0 0.4%, PC1 23.6%, PC2 8.3%. Que PC0 sea tan bajo se explica por la SVD sin centrar (ver teoría): PC0 apunta a la media.
- Que PC1 tenga 23.6% sugiere una dirección dominante. Si en el scatter PC1 separa por `answer_position`, es la candidata natural para DAS.

### Celdas 14–17 — 3D y doble codificación

```python
from causalab.experiments.visualizations.pca_scatter import plot_pca_scatter
key = list(result["features_by_target"].keys())[0]           # "10__last_token"
plot_pca_scatter(features=result["features_by_target"][key],
                 svd_result=result["svd_results_by_target"][key],
                 labels=labels_answer_position, component_tuples=[(0, 1, 2)],
                 title=f"{key} (3D)")
```

- Una tupla de 3 → gráfico 3D (interactivo en Jupyter).

```python
plot_pca_scatter(..., labels=labels_answer, shape_labels=labels_answer_position,
                 component_tuples=[(0, 1)], title="Answer Letter (color) x Position (shape)")
```

- Color = letra (26 valores), forma = posición (2 valores). La recomendación de la notebook: color para la variable con muchas categorías, forma para la que tiene pocas.
- Sirve para ver si los clusters los explica la posición, la letra, o ambas.

### Celdas 19–24 — stack mode

```python
stacked_targets = build_residual_stream_targets(pipeline=pipeline, layers=[0, 5, 10, 15],
                                                token_positions=[last_token_pos],
                                                mode="one_target_all_units")
```

- Un target `("all",)` con 4 unidades: el último token en las capas 0, 5, 10 y 15.

```python
result_stacked = collect_and_compute_PCA(
    interchange_targets=stacked_targets, data=dataset, pipeline=pipeline,
    labels=labels_answer_position, component_tuples=[(0, 1), (0, 2), (0, 1, 2)],
    combine_mode="stack", n_components=10, output_dir="./PCA_results_stacked", verbose=True,
)
print(f"Features shape: {list(result_stacked['features_by_target'].values())[0].shape}")
```

- `combine_mode="stack"`: las 4 unidades se apilan como muestras → `(4·1024, 2048) = (4096, 2048)` (coincide con la salida guardada). Todas las capas comparten un mismo espacio PCA.
- `normalize` no se pasa, así que vale `True` (PCA estandarizada). A diferencia de la llamada anterior.
- `n_components=10`.
- `labels_answer_position` (1024) se replica ×4 automáticamente, y en este caso **está bien**: el bloque de cada capa recorre las mismas 1024 muestras en el mismo orden.

```python
key_tuple = list(stacked_targets.keys())[0]
key_str = list(result_stacked["features_by_target"].keys())[0]
units = stacked_targets[key_tuple].flatten()
unit_labels = [f"Layer {u.layer}" for u in units]       # 4 labels
plot_pca_scatter(features=..., svd_result=..., labels=unit_labels,
                 component_tuples=[(0, 1)], title="Residual Stream by Layer")
```

- Intención: colorear cada punto por la capa de donde viene. Se espera que las capas formen grupos separados (las normas y direcciones del residual stream cambian mucho con la profundidad).
- **Ojo, posible bug**: `plot_pca_scatter` replica con `list(labels) * 1024` → `[L0, L5, L10, L15, L0, L5, ...]`, pero las features están en orden unit-major (`[L0]×1024, [L5]×1024, ...`). Así que el color de cada punto corresponde a `índice mod 4`, **no** a su capa. Para que sea correcto habría que pasar `[f"Layer {u.layer}" for u in units for _ in range(1024)]`. Lo mismo vale para las celdas 22 y 23 (en la 23 la forma, `answer_position`, sí queda bien). Lo deduzco leyendo `collect_PCA.py` y `pca_scatter.py`; no lo pude comprobar ejecutando.
- Celda 24: el mismo stack coloreado por `answer_position`, que sí se replica bien.

### Celda 25 — Takeaways de la parte no supervisada

PCA revela estructura, los labels causales la interpretan, la comparación entre capas muestra cómo cambia la información, y si una variable separa en PCA es buena candidata para DAS.

### Celdas 27–28 — datasets filtrados para DAS

```python
def checker(neural_output, causal_output):
    return causal_output in neural_output["string"] or neural_output["string"] in causal_output

train_size = 64
train_dataset = [same_symbol_different_position() for _ in range(train_size)]
filtered_train = filter_dataset(dataset=train_dataset, pipeline=pipeline,
                                causal_model=causal_model, metric=checker, batch_size=128)
test_dataset = [same_symbol_different_position() for _ in range(train_size)]
filtered_test = filter_dataset(...)
train_path = "./temp_das_train"
test_path = "./temp_das_test"
os.makedirs(train_path, exist_ok=True); os.makedirs(test_path, exist_ok=True)
save_counterfactual_examples(filtered_train, train_path)
save_counterfactual_examples(filtered_test, test_path)
```

- `same_symbol_different_position` porque la variable objetivo es `answer_position` (igual que la 04).
- Salida guardada: 57 train y 54 test (de 64).
- El mismo problema de `makedirs` + guardar en un directorio de la celda 3.

### Celda 30 — featurizers inicializados con PCA

```python
from causalab.neural.featurizers import SubspaceFeaturizer
svd_results_by_unit = result["svd_results_by_unit"]
for unit_id in svd_results_by_unit:
    rotation_shape = svd_results_by_unit[unit_id]["rotation"].shape
    print(f"  {unit_id}: rotation shape {rotation_shape}")
```

- `svd_results_by_unit` sale de la **primera** llamada (capa 10, sin normalizar). Clave = `"ResidualStream(Layer-10,block_output,Token-last_token)"`, `rotation` de shape `(2048, 512)` (salida guardada).

```python
n_features = 2
for key, target in targets.items():
    for unit in target.flatten():
        rotation = svd_results_by_unit[unit.id]["rotation"]
        featurizer = SubspaceFeaturizer(
            rotation_subspace=rotation[:, :n_features],
            trainable=True,
            id="PCA_init",
        )
        unit.set_featurizer(featurizer)
```

- Reusa los `targets` de la celda 7 (la misma unidad de la capa 10) y busca su rotación por `unit.id`.
- `rotation[:, :2]`: `W ∈ R^{2048×2}` = PC0 y PC1. Ojo que PC0 es la dirección "de la media" (SVD sin centrar) y PC1 la de mayor varianza.
- `trainable=True`: DAS la va a seguir ajustando.
- `id="PCA_init"`: cualquier id distinto de `"null"` hace que `_initialize_featurizers` **no** la pise.
- Importa que la SVD se haya hecho **sin normalizar**: `SubspaceFeaturizer` proyecta la activación cruda (`x W`), sin restar media ni dividir por el desvío. Si la base viniera de una PCA estandarizada, estaría expresada en otras coordenadas.

### Celda 32 — entrenar DAS desde PCA

```python
das_config = {
    "train_batch_size": 32, "evaluation_batch_size": 64,
    "training_epoch": 20, "init_lr": 0.001,
    "DAS": {"n_features": n_features},
}
das_result = train_DAS(causal_model=causal_model, interchange_targets=targets,
                       train_dataset_path=train_path, test_dataset_path=test_path,
                       pipeline=pipeline, target_variable_group=("answer_position",),
                       output_dir="./DAS_pca_init_results", metric=checker,
                       config=das_config, verbose=True)
for key, score in das_result["test_scores"].items():
    train_score = das_result["train_scores"][key]
    print(f"  Layer {key[0]:2d}: train={train_score:.3f}, test={score:.3f}")
```

- `DAS.n_features` acá **no tiene efecto**: solo se usa para crear featurizers en unidades con id `"null"`, y esta ya tiene uno de 2 dimensiones.
- 20 épocas, lr 1e-3 (más suave que la 04).
- Salida guardada: train 0.860, test 0.889 en la capa 10 con **solo 2 dimensiones**. En la 04 (32 dimensiones al azar, 3 épocas) el mejor test fue 0.789. Las configuraciones no son comparables directamente (otra cantidad de épocas, de dimensiones y otros datos), así que no se puede atribuir la mejora solo a la inicialización con PCA.
- Para una comparación justa habría que correr el mismo config con `SubspaceFeaturizer(shape=(2048, 2))` aleatorio (eso sugiere el resumen, punto "Comparable").

### Celda 36 — target para DBM

```python
dbm_targets = build_residual_stream_targets(
    pipeline=pipeline,
    layers=layers_to_analyze,  # [0, 5, 10, 15]
    token_positions=[last_token_pos],
    mode="one_target_all_units",
)
```

- **Ojo**: el comentario dice `[0, 5, 10, 15]`, pero `layers_to_analyze` vale `[10]` (celda 7). Por eso la salida guardada muestra **1 unidad** (capa 10), y el texto de la celda 35 ("reuse the same layers [0, 5, 10, 15]") no coincide con lo que corre.
- Son unidades **nuevas** (no las de `targets`, que ya tienen el featurizer de DAS entrenado).

### Celda 38 — featurizer PCA fijo

```python
for key, target in dbm_targets.items():
    for unit in target.flatten():
        rotation = svd_results_by_unit[unit.id]["rotation"]
        featurizer = SubspaceFeaturizer(rotation_subspace=rotation, trainable=False, id="PCA_for_DBM")
        unit.set_featurizer(featurizer)
```

- `W ∈ R^{2048×512}`: las 512 componentes. `trainable=False`: la base no se mueve, solo se aprenden máscaras.
- `n_features = 512`.

### Celda 40 — DBM sobre componentes

```python
dbm_config = {
    "train_batch_size": 32, "evaluation_batch_size": 64,
    "training_epoch": 20, "init_lr": 0.001,
    "masking": {"regularization_coefficient": 1e-4,
                "temperature_annealing_fraction": 0.5,
                "temperature_schedule": (1.0, 0.001)},
}
dbm_result = train_DBM_feature_masks(causal_model=causal_model, interchange_target=dbm_targets[("all",)],
                                     train_dataset_path=train_path, test_dataset_path=test_path,
                                     pipeline=pipeline, target_variable_group=("answer_position",),
                                     output_dir="./DBM_pca_init_results", metric=checker,
                                     config=dbm_config, verbose=True,
                                     title="PCA-Initialized DBM: Selected Components")
```

- `train_DBM_feature_masks` fuerza `tie_masks=False` → 512 máscaras (una por componente). El featurizer `PCA_for_DBM` se respeta.
- `regularization_coefficient = 1e-4`: muy chico, casi no empuja a la esparsidad.
- Durante el entrenamiento: `f_out = (1 − σ(m/τ)) ⊙ f_b + σ(m/τ) ⊙ f_s` en coordenadas PCA. Al final, `feature_indices = [j : σ(m_j) > 0.5]`, y la evaluación patchea solo esas componentes.
- Salida guardada: train 0.544, test 0.648. Bastante peor que PCA → DAS con 2 dimensiones entrenables.

### Celda 42 — qué componentes se eligieron

```python
heatmap_path = dbm_result["output_paths"]["heatmap"]
display(Image(filename=heatmap_path))
feature_indices = dbm_result["feature_indices"]
for unit_id, indices in feature_indices.items():
    explained_var = svd_results_by_unit[unit_id]["explained_variance_ratio"]
    if indices:
        selected_var = sum(explained_var[i] for i in indices)
        print(f"  Selected components: {indices}")
        print(f"  Variance explained by selected: {selected_var:.1%}")
```

- `feature_indices`: `{unit_id: [índices de componentes]}`.
- Salida guardada: se eligieron **261 de 512** componentes (0, 1, 2, 3, 4, 6, 7, 9, 10, ...), que juntas explican el 73.2% de la varianza. Con `λ = 1e-4` la máscara apenas se penaliza; como empieza en `σ(0) = 0.5` y el gradiente es débil, es plausible que muchas componentes hayan quedado de un lado u otro del umbral casi al azar (esto es una interpretación mía). El resultado no es la selección "esparsa e interpretable" que promete el resumen.

### Celdas 33 y 43 — resúmenes

- PCA → DAS: `collect_and_compute_PCA` → `SubspaceFeaturizer(rotation_subspace=...)` → `train_DAS` respeta los featurizers con id distinto de `"null"`.
- PCA + DBM: target con `one_target_all_units`, `SubspaceFeaturizer(..., trainable=False)`, `train_DBM_feature_masks`.
- El resumen dice "the number of components is small (e.g., 10 vs 2048)", pero en esta corrida se usaron 512.
- Anticipa la 06: `train_boundless_DAS` es "un wrapper que inicializa `SubspaceFeaturizer` con `trainable=True` antes de llamar a `train_DBM_feature_masks`".

---

## Ejercicios

No hay celdas de ejercicio. Experimentos que vale la pena hacer:
- Repetir la PCA con `normalize=True` y comparar la varianza de PC0 (debería dejar de ser ≈ 0). Ojo: entonces la rotación no sirve directamente como featurizer de activaciones crudas.
- PCA → DAS vs DAS aleatorio con el **mismo** config (2 dimensiones, 20 épocas, lr 1e-3).
- PCA → DBM con `regularization_coefficient` más grande (por ejemplo 0.1 o 1) para forzar una selección esparsa, y mirar si las componentes elegidas son las de alta varianza.
- Arreglar `unit_labels` en stack mode y ver si las capas se separan.

---

## Ojo / errores comunes / dudas

- **SVD sin centrar ≠ PCA.** Con `normalize=False`, PC0 ≈ dirección de la media; su "varianza explicada" es mínima aunque sea la componente de mayor energía.
- **El markdown dice "with normalization"** en la celda 8, pero la llamada usa `normalize=False`. La de stack mode usa el default `True`.
- **`save_counterfactual_examples` con un directorio**: con el código actual falla (las salidas guardadas son de una versión que guardaba datasets HF).
- **`layers_to_analyze` es `[10]`**, no `[0, 5, 10, 15]` como dicen el comentario y el markdown de la sección DBM.
- **Labels por unidad en stack mode** se replican mal (intercalados en vez de en bloques). Los colores "por capa" de las celdas 21–23 probablemente no corresponden a las capas.
- **`DAS.n_features` no hace nada** cuando la unidad ya tiene un featurizer.
- **El featurizer PCA no centra**: proyecta `x W`. Por eso hay que usar la rotación de una SVD sin normalizar.
- **DBM sobre 512 componentes deja fija la parte de `x` fuera de esas 512 direcciones** (`error` del base). Aunque se seleccionaran todas, no se patchearía el vector entero.
- **Las comparaciones de números entre la 04 y la 05 no son controladas** (distinta cantidad de épocas, de dimensiones, de datos).
- **Las rutas de los warnings** (`/mnt/polished-lake/home/atticus/github/causalab-internal-boundlessDAS/...`) muestran que la ejecución guardada es de una rama interna de la librería.

---

## Resumen de una línea por sección

- **Dataset sin filtrar**: 1024 inputs `different_symbol`; solo se usan los base.
- **Labels**: `answer_position` y `answer` leídos del trace, solo para colorear.
- **Unidad**: residual stream, capa 10, último token.
- **Collect + PCA**: `collect_features` + `TruncatedSVD` (sin centrar) con 512 componentes; PC1 explica 23.6%.
- **3D / doble label**: tuplas de 3 componentes y `shape_labels` para ver dos variables a la vez.
- **Stack mode**: 4 capas en un espacio común `(4096, 2048)`; ojo con la replicación de labels por unidad.
- **PCA → DAS**: `SubspaceFeaturizer(rotation[:, :2], trainable=True)`; test 0.889 en la capa 10.
- **PCA → DBM**: base PCA fija de 512 componentes y máscara por componente; test 0.648, selección poco esparsa.
