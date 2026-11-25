# 04 — Localización supervisada: DAS y DBM (`04_train_DAS_and_DBM.ipynb`)

Notebook: `causalab/demos/onboarding_tutorial/04_train_DAS_and_DBM.ipynb`.

Cuarta notebook del onboarding, y la central de la librería. En la 03 ([03_localize_with_patching.md](03_localize_with_patching.md)) se patcheaba el **vector entero** del residual stream y se *miraba* dónde coincidía con el modelo causal. Acá el modelo causal pasa a ser **supervisión**: se entrena *dónde* y *qué* intervenir para que la red intervenida responda lo mismo que el modelo causal intervenido. Dos métodos:

1. **DAS** (*Distributed Alignment Search*): en cada (capa, posición) se aprende un **subespacio lineal** del residual stream (32 direcciones ortonormales) que se comporta como `answer_position`.
2. **DBM** (*Desiderata-Based Masking*): se aprende una **máscara binaria** sobre componentes (512 attention heads, o 48 salidas de MLP) que selecciona cuáles hay que patchear para que la red se comporte como si se interviniera `answer`.

La pregunta: ¿se puede localizar `answer_position` con más precisión que en la 03, y qué heads y MLPs mueven la respuesta hasta el último token?

Sigue en [05_PCA_DBM_DAS.md](05_PCA_DBM_DAS.md), que introduce un método no supervisado (PCA) y lo combina con DAS y DBM. Referencia de la librería: [../02_neural.md](../02_neural.md) (featurizers, model units) y [../03_experiments.md](../03_experiments.md) (jobs de entrenamiento).

---

## Ideas clave / teoría

### Por qué no alcanza con patchear el vector entero

Un vector del residual stream (`d = 2048`) codifica muchas cosas a la vez: la letra, la posición, el objeto, el color, etc. Si patcheás el vector entero, copiás *todo* eso. Puede pasar que la variable que te interesa esté ahí pero que, al copiar también lo demás, la salida cambie de otra forma (o al revés: la variable está repartida en una dirección que el patch completo mezcla con otras cosas). La hipótesis de **representación lineal** dice que una variable vive en un **subespacio** de pocas dimensiones. Si se encuentra ese subespacio, se puede patchear **solo** esa parte.

### DAS: interchange intervention en un subespacio aprendido

Sea `W ∈ R^{d×k}` con columnas ortonormales (`WᵀW = I_k`). Para el vector base `x_b` y el del contrafáctico `x_s` en la misma ubicación, la intervención de DAS es:

  `x' = x_b + W Wᵀ (x_s − x_b) = W Wᵀ x_s + (I − W Wᵀ) x_b`

- `W Wᵀ` es el proyector ortogonal sobre el subespacio de dimensión `k`.
- `W Wᵀ x_s`: la componente del contrafáctico **dentro** del subespacio.
- `(I − W Wᵀ) x_b`: la componente del base **fuera** del subespacio, que se conserva intacta.

En coordenadas: `f = x W ∈ R^k` son las "features" (proyecciones). Se reemplazan las features del base por las del contrafáctico y se reconstruye.

**Entrenamiento.** Para cada par `(b, s)`, el label es la salida del modelo causal con `answer_position := answer_position(s)`. Se corre la red con la intervención de DAS y se minimiza la cross-entropy de los logits del primer token generado contra el token del label. Los pesos del LLM están congelados: lo único que se entrena es `W`. Como `W` está parametrizada como ortogonal, sigue siendo ortonormal durante todo el entrenamiento.

**Evaluación.** Con `W` fija, se calcula la IIA (igual que en la 03) en un **test set** generado igual pero independiente. DAS es aprendizaje supervisado con muchos parámetros (`2048 × 32` por ubicación), así que **puede sobreajustar**: encontrar un subespacio que "fuerce" las respuestas en train sin que sea la representación real. Por eso train/test.

### DBM: máscaras sobre componentes

Para un conjunto de unidades `u = 1..U` (por ejemplo, las 512 attention heads, en todas las posiciones), cada una con un parámetro escalar `m_u`, la intervención durante el entrenamiento es:

  `g_u = σ(m_u / τ)`,  `h'_u = (1 − g_u) · h_u^{(b)} + g_u · h_u^{(s)}`

- `σ` es la sigmoide. `g_u ∈ (0,1)` es una "compuerta" suave: 1 = patchear la unidad entera, 0 = dejar la del base.
- `τ` es la **temperatura**. Se reduce linealmente de 1.0 a 0.001 durante la primera mitad del entrenamiento, y después queda fija. Con `τ` chica, `σ(m/τ)` es casi un escalón: la compuerta se vuelve binaria.
- La pérdida es `CE(salida intervenida, label causal) + λ · (Σ_u g_u) / U`. El segundo término es una penalización L1 sobre las compuertas, normalizada por el número de elementos de máscara, que empuja a seleccionar **pocas** unidades. Acá `λ = 0.3`.
- En evaluación la compuerta es dura: `g_u = 1[σ(m_u) > 0.5]` (equivalente a `m_u > 0`).

Resultado: un subconjunto pequeño de componentes que, patcheados juntos con el contrafáctico, hacen que la red se comporte como el modelo causal con `answer` intervenida. Es un "circuito" mínimo para mover esa variable.

`tie_masks=True` significa **una máscara por unidad** (un escalar que selecciona la head entera). Con `tie_masks=False` hay una máscara por dimensión; eso es la base de Boundless DAS ([06_boundless_DAS.md](06_boundless_DAS.md)).

---

## El código de la librería que usa esta notebook

### Featurizers (`causalab/causalab/neural/featurizers.py`)

Un `Featurizer` es un par de módulos `featurizer: x → (f, error)` y `inverse_featurizer: (f, error) → x̂`, más `n_features`, `id` y `tie_masks`. A partir de ellos construye **clases** de intervención para pyvene (clases, no instancias, porque pyvene las instancia por su cuenta):

- `get_interchange_intervention()` → `FeatureInterchangeIntervention`. Su `forward(base, source, subspaces)`:
  ```python
  f_base, base_err = self._featurizer(base)
  f_src, _ = self._featurizer(source)
  if subspaces is None or _subspace_is_all_none(subspaces):
      f_out = f_src
  else:
      f_out = _do_intervention_by_swap(f_base, f_src, "interchange", ..., subspaces, ...)
  return self._inverse(f_out, base_err).to(base.dtype)
  ```
  Featuriza ambos y reemplaza las features del base por las del source: **todas** si `subspaces` es `None`, o solo los índices en `subspaces` (los `feature_indices` de la unidad). Reconstruye con el **error del base**.
- `get_mask_intervention()` → `FeatureMaskIntervention`. Tiene `self.mask = Parameter(zeros(1))` si `tie_masks` o `zeros(n_features)` si no, y una temperatura. `forward`: en modo training `gate = sigmoid(mask/temp)`, en eval `gate = (sigmoid(mask) > 0.5)`; `f_out = (1-gate)·f_base + gate·f_src`. `get_sparsity_loss()` = `‖sigmoid(mask/temp)‖₁`. Inicializar `mask = 0` da `gate = 0.5` al principio.
- `get_collect_intervention()` → `FeatureCollectIntervention`: devuelve las features (para recolectar activaciones; se usa en la 05 y la 07).

`SubspaceFeaturizer(shape=(d, k), trainable=True)` o `SubspaceFeaturizer(rotation_subspace=W)`:
- crea `pv.models.layers.LowRankRotateLayer(d, k)` (un `weight` de `d × k`), lo envuelve con `torch.nn.utils.parametrizations.orthogonal` (para que `weight` tenga siempre columnas ortonormales) y le pone `requires_grad_(trainable)`;
- `SubspaceFeaturizerModule.forward(x)`: `f = x @ W` (shape `(..., k)`), `error = x − f @ Wᵀ` (la parte de `x` fuera del subespacio);
- `SubspaceInverseFeaturizerModule.forward(f, error)` = `f @ Wᵀ + error`.

Con `f_out = f_src` y `error = error_base`: `x' = x_s W Wᵀ + x_b − x_b W Wᵀ`, que es exactamente la fórmula de DAS de arriba (con vectores fila).

### `train_interventions` (`causalab/causalab/experiments/train.py`)

Función genérica que usan DAS, DBM y Boundless DAS:

1. `config = merge_with_defaults(config)` (`causalab/causalab/experiments/configs/train_config.py`): mezcla el config parcial con `DEFAULT_CONFIG`, haciendo merge profundo de `DAS`, `masking` y `featurizer_kwargs`. Defaults relevantes: `intervention_type="mask"`, `training_epoch=3`, `init_lr=1e-2`, `DAS.n_features=32`, `masking.regularization_coefficient=0.1`, `temperature_schedule=(1.0, 0.001)`, `scheduler_type="constant"`, `shuffle=True`, `patience=None`.
2. Si `intervention_type == "mask"`, exige `featurizer_kwargs` en el config.
3. Si se le pasa un solo `InterchangeTarget`, lo envuelve como `{("single",): target}`.
4. Carga train/test con `load_counterfactual_examples`.
5. `_initialize_featurizers`: para cada unidad **cuyo featurizer tenga id `"null"`** (el identidad por defecto):
   - si es `mask`: `Featurizer(n_features=unit.shape[0], tie_masks=..., id=f"mask_{unit.id}")` (identidad, pero con `n_features`, que hace falta para construir la máscara);
   - si es `interchange`: `SubspaceFeaturizer(shape=(unit.shape[0], n_features), trainable=True, id=f"DAS_{unit.id}")`.

   Las unidades con un featurizer ya puesto (id distinto de `"null"`) **se respetan**. Esto se aprovecha en la 05 para inicializar con PCA.
6. Etiqueta el train con `causal_model.label_counterfactual_data(train, list(target_variable_group))` → agrega `"label"` = `raw_output` del modelo causal intervenido.
7. Para cada target: `train_interventions_pyvene(...)` (ver abajo), después evalúa en train y test con `run_interchange_interventions` + `causal_score_intervention_outputs` (IIA, igual que la 03). La evaluación usa la intervención `interchange` con los `feature_indices` aprendidos.
8. Guarda resultados, modelos (`target.save`) y metadata. Devuelve `results_by_key[key] = {"train_score", "test_score", "feature_indices", ...}`.

### El loop de entrenamiento (`train_interventions` en `causalab/causalab/neural/pyvene_core/interchange.py`)

- `prepare_intervenable_model(pipeline, target, intervention_type)`, `disable_model_gradients()` (congela el LLM), `eval()`.
- Optimizador: `AdamW` sobre `v.parameters()` de cada intervención (en DAS es la rotación; en DBM, la máscara, y también la rotación si el featurizer es entrenable), `weight_decay=0`, scheduler de `transformers.get_scheduler` (constante por default).
- Si es `mask`: arma el schedule de temperatura `linspace(start, end, annealing_steps+1)` seguido de `end` constante, con `annealing_steps = total_steps · temperature_annealing_fraction`.
- Por batch: `loss, metrics, _ = loss_and_metric_fn(...)`. Si es mask, suma `λ · Σ sparsity / Σ numel(mask)` y actualiza la temperatura. Después `backward`, `step`, `scheduler.step`, `set_zero_grad`.
- Al final, si es mask, **convierte las máscaras en `feature_indices`**:
  - `tie_masks=True`: si `σ(mask[0]) > 0.5`, `indices = None` (la unidad entera); si no, `indices = []` (ninguna feature, es decir, la unidad no se interviene);
  - `tie_masks=False`: `indices = [i : σ(mask_i) > 0.5]`.

  Y hace `model_unit.set_feature_indices(indices)`.

### La pérdida (`LM_loss_and_metric_fn` en `causalab/causalab/experiments/metric.py`)

```python
batched_base, batched_counterfactuals, inv_locations, feature_indices = \
    prepare_intervenable_inputs(pipeline, examples, interchange_target)
batched_inv_label = pipeline.load(label_traces, max_length=pipeline.max_new_tokens,
                                  padding_side="right", add_special_tokens=False)
for k in batched_base:
    batched_base[k] = torch.cat([batched_base[k], batched_inv_label[k]], dim=-1)
_, counterfactual_logits = intervenable_model(batched_base, batched_counterfactuals,
                                              unit_locations=inv_locations, subspaces=feature_indices)
labels = batched_inv_label["input_ids"]
logits = counterfactual_logits.logits[:, -labels.shape[-1] - 1 : -1]
loss = compute_cross_entropy_loss(logits, labels, pad_token_id)
```

- Tokeniza el label (`" K"`) a `max_new_tokens = 1` token y lo **concatena** al prompt base (teacher forcing). El base queda con 33 tokens.
- Forward de pyvene con intervenciones (las posiciones se calcularon sobre el prompt sin label, así que el token del label nunca se interviene).
- `logits[:, -2:-1]`: el logit en la última posición del prompt, que predice el label. CE ignorando el pad.
- También calcula accuracy con el `checker` sobre el argmax, para la barra de progreso.

### Jobs

- `train_DAS` (`causalab/causalab/experiments/jobs/DAS_grid.py`): detecta el tipo de componente, fuerza `config["intervention_type"] = "interchange"`, llama `train_interventions`, arma `train_scores`/`test_scores` (`{(layer, pos_id): IIA}`), agrega a la metadata `train_max_score`, `test_max_score`, mejores ubicaciones, y dibuja `heatmaps/train.png` y `heatmaps/test.png`.
- `train_DBM_binary_heatmaps` (`causalab/causalab/experiments/jobs/DBM_binary_grid.py`): detecta el tipo por el id de la primera unidad, sobreescribe `config["featurizer_kwargs"] = {"tie_masks": tie_masks}` (el argumento de la función manda), deja `intervention_type` en su default `"mask"`, llama `train_interventions` con el target único (clave `("single",)`), y con `get_selected_units(feature_indices)` (`experiments/visualizations/binary_mask.py`) extrae la lista de `(layer, head)` o `(layer, pos_id)` seleccionadas. Dibuja `heatmaps/<var>_mask.png`.

### Builders nuevos (`causalab/causalab/experiments/interchange_targets.py`)

- `build_attention_head_targets(pipeline, layers, heads, token_position, mode)`: una `AttentionHead(layer, head, token_indices=token_position, shape=(head_dim,))` por par. `AttentionHead` (`causalab/causalab/neural/LM_units.py`) usa `component_type = "head_attention_value_output"` y `unit = "h.pos"`, y su `index_component` devuelve `[[head]], [posiciones]` (pyvene indexa primero la head y después las posiciones). `head_dim` sale de `config.head_dim` o de `hidden_size // num_heads` (2048/32 = 64).
- `build_mlp_targets(pipeline, layers, token_positions, mode, location="mlp_output")`: `MLP(layer, pos, location)` con `component_type = location` y `shape = (hidden_size,)` (o `intermediate_size` si `location="mlp_activation"`).
- `mode="one_target_all_units"` → `{("all",): InterchangeTarget([todas_las_unidades])}`: **un solo grupo**, así que todas comparten el mismo contrafáctico y se entrenan juntas. Es lo que necesita DBM para elegir un subconjunto.

---

## Recorrido por la notebook

### Celda 0 — setup, datasets de train y test

```python
from causalab.neural.token_position_builder import get_all_tokens
from causalab.experiments.filter import filter_dataset
from causalab.causal.causal_utils import save_counterfactual_examples
...
size = 64
counterfactual_generators = {
    "different_symbol": different_symbol,
    "same_symbol_different_position": same_symbol_different_position,
    "random_counterfactual": random_counterfactual,
}
```

- Mismo pipeline y `checker` (sin `strip`) que la 03.
- Genera y filtra **dos veces** cada dataset: uno de train y uno de test, con el mismo generador pero muestras independientes.

```python
train_datasets_dir = "./temp_train_datasets"
test_datasets_dir = "./temp_test_datasets"
for name, dataset in filtered_datasets.items():
    dataset_path = os.path.join(train_datasets_dir, name, "filtered_dataset.json")
    os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
    save_counterfactual_examples(dataset, dataset_path)
...
token_positions = create_token_positions(pipeline)
```

- Guarda cada dataset en `temp_{train,test}_datasets/<nombre>/filtered_dataset.json` (archivo JSON; el directorio padre se crea antes).
- La última línea crea las posiciones, pero se vuelven a crear en la celda 5.

### Celda 3 — mostrar el heatmap de la 03

```python
if os.path.exists("vector_patching_results/heatmap_dataset_same_symbol_different_position_MCQA_vector_patching_variables_answer_position.png"):
    display(Image(filename=...))
```

- Quiere mostrar el heatmap de `answer_position` de la 03 para compararlo. Con la versión actual de la librería ese archivo está en `vector_patching_results/same_symbol_different_position/heatmaps/same_symbol_different_position_answer_position.png`, así que el `exists` da `False` y la celda no muestra nada (en la ejecución guardada, efectivamente no hay salida).

### Celda 5 — entrenar DAS para `answer_position`

```python
from causalab.experiments.jobs.DAS_grid import train_DAS
from causalab.experiments.interchange_targets import build_residual_stream_targets
from causalab.experiments.configs.train_config import PartialExperimentConfig

all_token_positions = create_token_positions(pipeline)
ordered_position_names = ["correct_symbol", "correct_symbol_period", "symbol0",
                          "symbol0_period", "symbol1", "symbol1_period", "last_token"]
token_positions = [all_token_positions[name] for name in ordered_position_names]

train_dataset_path = "./temp_train_datasets/same_symbol_different_position/filtered_dataset.json"
test_dataset_path = "./temp_test_datasets/same_symbol_different_position/filtered_dataset.json"
```

- Mismas 7 posiciones que la 03.
- Dataset `same_symbol_different_position`: es el que distingue `answer_position` de `answer` y de no intervenir (ver 03). Si entrenaras DAS para `answer_position` con `different_symbol`, el label siempre sería "no cambia nada" y DAS aprendería a no hacer nada.

```python
num_layers = pipeline.model.config.num_hidden_layers
layers = [-1] + list(range(num_layers))
targets = build_residual_stream_targets(pipeline=pipeline, layers=layers,
                                        token_positions=token_positions,
                                        mode="one_target_per_unit")
```

- 119 targets independientes, cada uno con una unidad `ResidualStream` (`shape=(2048,)`). **Se entrena un DAS por (capa, posición)**: 119 entrenamientos.

```python
das_config: PartialExperimentConfig = {
    "train_batch_size": 32,
    "evaluation_batch_size": 32,
    "training_epoch": 3,
    "init_lr": 1e-2,
    "DAS": {"n_features": 32},
}
```

- `PartialExperimentConfig` es un `TypedDict` con todos los campos opcionales (solo sirve para tipado).
- `n_features: 32`: `k = 32` dimensiones del subespacio. **Ojo**: el markdown anterior dice "16 orthogonal features", pero el código usa 32.
- 3 épocas con ~50 ejemplos y batch 32 son ~6 pasos de optimización por ubicación: muy poco entrenamiento, lr alto (1e-2).

```python
das_result = train_DAS(
    interchange_targets=targets, causal_model=causal_model,
    train_dataset_path=train_dataset_path, test_dataset_path=test_dataset_path,
    pipeline=pipeline, target_variable_group=("answer_position",),
    output_dir="./DAS_results", metric=checker, config=das_config, verbose=True,
)
train_results = das_result["train_scores"]
test_results = das_result["test_scores"]
print(f"✓ Best training score: {das_result['metadata']['train_max_score']:.3f}")
print(f"✓ Best test score: {das_result['metadata']['test_max_score']:.3f}")
```

- `target_variable_group=("answer_position",)`: los labels de entrenamiento son `run_interchange(b, {"answer_position": s})["raw_output"]`.
- Por dentro: `_initialize_featurizers` le pone a cada unidad un `SubspaceFeaturizer(shape=(2048, 32), trainable=True)` (rotación ortogonal aleatoria inicial), entrena, evalúa en train y test.
- Salida guardada: 119 targets en 4:48; mejor IIA en train = 0.810, en test = 0.789. Dos heatmaps (train y test).
- `train_results`/`test_results`: dict `{(layer, pos_id): IIA}`.

### Celdas 6–8 — interpretación

Según el texto de la notebook:
- El heatmap de test es más débil que el de train aunque los datos se generan igual: hay **overfitting**.
- El heatmap de test muestra **tres etapas** de recuperación de la posición en `last_token`: la mejor en la **capa 10**, una más débil en la **11** y otra aún más débil en las **12–13**. Con patching del vector completo (03) solo se veían las dos primeras.
- El de train es "puntiagudo" (spiky), más difícil de leer, otra vez por sobreajuste con pocos datos.

### Celda 10 — otro heatmap de la 03

Igual que la celda 3: intenta mostrar el heatmap de `answer` de la 03 con una ruta vieja, y no muestra nada.

### Celda 12 — targets de attention heads para DBM

```python
from causalab.experiments.jobs.DBM_binary_grid import train_DBM_binary_heatmaps
from causalab.experiments.interchange_targets import build_attention_head_targets

sample_input = filtered_datasets["different_symbol"][0]["input"]
all_tokens = get_all_tokens(sample_input, pipeline, padding=True)

num_layers = pipeline.model.config.num_hidden_layers       # 16
num_heads = pipeline.model.config.num_attention_heads      # 32

attention_targets = build_attention_head_targets(
    pipeline=pipeline, layers=list(range(num_layers)), heads=list(range(num_heads)),
    token_position=all_tokens, mode="one_target_all_units",
)
print(f"Built attention head target with {len(attention_targets[('all',)])} units")
```

- `get_all_tokens(input, pipeline, padding=True)` (`token_position_builder.py`) devuelve **un** `TokenPosition` con id `"all_tokens"`. Su indexer tokeniza el input que recibe y, con `padding=True`, devuelve `range(len(token_ids))`, o sea **todas las 32 posiciones, incluido el padding**. `sample_input` no se usa para nada más que la firma: el indexer se evalúa sobre cada input en el momento.
- Se patchea cada head **en todas las posiciones a la vez**. La pregunta es "¿qué heads, como componente entero, transportan la respuesta?".
- 16 × 32 = 512 unidades `AttentionHead`, todas en **un grupo** (`one_target_all_units`).
- **Ojo con el print**: `len(InterchangeTarget)` devuelve el número de **grupos** (1), no de unidades. Por eso la salida guardada dice "with 1 units". Las unidades son `len(attention_targets[('all',)].flatten())` = 512 (esa forma correcta se usa en la celda 17 para los MLPs).

### Celda 13 — entrenar DBM sobre heads para `answer`

```python
dbm_attention_config: PartialExperimentConfig = {
    "train_batch_size": 32,
    "evaluation_batch_size": 32,
    "training_epoch": 3,
    "init_lr": 1e-2,
    "masking": {
        "regularization_coefficient": 0.3,
        "temperature_annealing_fraction": 0.5,
        "temperature_schedule": (1.0, 0.001),
    },
    "featurizer_kwargs": {"tie_masks": True},
}
```

- `regularization_coefficient`: `λ` de la penalización L1 (normalizada por número de máscaras).
- `temperature_annealing_fraction = 0.5`: la temperatura baja de 1 a 0.001 en la primera mitad de los pasos y queda en 0.001.
- El comentario dice "Gumbel-Softmax", pero en el código no hay ruido de Gumbel: es una sigmoide con temperatura (ver `FeatureMaskIntervention.forward`).
- `tie_masks: True`: un escalar por head.

```python
dbm_result = train_DBM_binary_heatmaps(
    causal_model=causal_model,
    interchange_target=attention_targets[("all",)],
    train_dataset_path="./temp_train_datasets/different_symbol/filtered_dataset.json",
    test_dataset_path="./temp_test_datasets/different_symbol/filtered_dataset.json",
    pipeline=pipeline, target_variable_group=("answer",),
    output_dir="./DBM_attention_results", metric=checker,
    config=dbm_attention_config, tie_masks=True, verbose=True,
)
```

- Dataset `different_symbol` y variable `answer`: los labels son "la letra nueva del contrafáctico". Hay que encontrar qué heads, patcheadas juntas, hacen que la red diga la letra nueva.
- Por dentro: cada head recibe `Featurizer(n_features=64, tie_masks=True, id="mask_...")` (identidad + máscara escalar). Se entrena **un** modelo con 512 máscaras. Al final, las heads con `σ(m) > 0.5` quedan con `feature_indices = None` (patchear entera) y el resto con `[]`. La evaluación en train/test patchea **solo** las seleccionadas.
- Salida guardada: un target, 79 s, figura de máscara (capas × heads).

### Celda 14 — resumen

```python
print(f"Train Accuracy: {dbm_result['train_score']:.3f}")
print(f"Test Accuracy: {dbm_result['test_score']:.3f}")
print(f"Selected Units: {dbm_result['metadata']['num_selected_units']} out of {dbm_result['metadata']['num_units']}")
for layer, head in dbm_result["selected_units"]:
    print(f"  Layer {layer:2d}, Head {head:2d}")
```

- Salida guardada: train 0.966, test 0.982, **6 de 512** heads: (3, 11), (10, 3), (11, 14), (12, 13), (14, 11), (14, 14).
- El texto dice que las heads seleccionadas "generalizan perfectamente" al test (0.982), aunque sigue siendo un solo template.
- Lectura: hay heads en capas tempranas (3), medias (10–12) y tardías (14). Esto es coherente con la 02/03: la letra se mueve del token del símbolo al último token entre las capas ~10 y 12.

### Celdas 17–19 — DBM sobre MLPs

```python
from causalab.experiments.interchange_targets import build_mlp_targets
ordered_position_names = ["correct_symbol", "correct_symbol_period", "last_token"]
token_positions = [all_token_positions[name] for name in ordered_position_names]

mlp_targets = build_mlp_targets(pipeline=pipeline, layers=list(range(num_layers)),
                                token_positions=token_positions,
                                mode="one_target_all_units", location="mlp_output")
print(f"Built MLP target with {len(mlp_targets[('all',)].flatten())} units")   # 48
```

- A diferencia de las heads, acá se usan **posiciones concretas** (3) y cada (capa, posición) es una unidad: 16 × 3 = 48. `build_mlp_targets` recibe una **lista** de posiciones; `build_attention_head_targets` recibe **una** posición.
- `location="mlp_output"`: se patchea la salida del bloque MLP (lo que el MLP suma al residual stream).

```python
dbm_mlp_config = {... mismo que heads ...}
mlp_dbm_result = train_DBM_binary_heatmaps(
    causal_model=causal_model, interchange_target=mlp_targets[("all",)],
    train_dataset_path=".../different_symbol/filtered_dataset.json",
    test_dataset_path=".../different_symbol/filtered_dataset.json",
    pipeline=pipeline, target_variable_group=("answer",),
    output_dir="./DBM_mlp_results", metric=checker, config=dbm_mlp_config,
    tie_masks=True, verbose=True,
)
```

- Salida guardada: train 1.000, test 0.982, **1 de 48** unidades: MLP de la **capa 0** en `correct_symbol`.

### Celda 20 — heads vs MLPs

La interpretación de la notebook: las heads **mueven** la respuesta entre posiciones (desde el símbolo hasta el último token), y los MLPs **transforman** en cada posición. Que solo se elija el MLP 0 en el símbolo correcto sugiere que la transformación crítica es temprana: el primer MLP convierte el embedding del símbolo en una representación que las heads después pueden enrutar.

Cuidado con esta lectura: patchear la salida del MLP 0 en el símbolo correcto, con un contrafáctico que solo cambia la letra, básicamente reemplaza gran parte de la información de "qué letra es" en esa posición. Que con eso alcance para cambiar la respuesta muestra que la información downstream depende de esa salida, pero no prueba que el MLP 0 "haga" algo especial. La notebook lo presenta como sugerencia, no como prueba.

### Celda 15 — Takeaways

Usar el modelo causal como supervisión es poderoso, pero como ahora hay descenso por gradiente buscando evidencia de la hipótesis, es fácil sobreajustar. Siempre train/test.

---

## Ejercicios

No hay celdas de ejercicio. Experimentos sugeridos por los comentarios ("Modify them to experiment!"):
- Cambiar `n_features` de DAS (por ejemplo 1, 4, 16, 128) y ver cómo cambian train/test. Con `k` grande, DAS tiene más libertad y más riesgo de overfitting.
- Subir `size` (más datos) para ver si el heatmap de train deja de ser puntiagudo.
- Cambiar `regularization_coefficient` de DBM: más alto → menos heads seleccionadas; si es demasiado alto la accuracy cae.

---

## Ojo / errores comunes / dudas

- **"16 orthogonal features" en el markdown vs `n_features: 32` en el código.**
- **`len(target)` cuenta grupos, no unidades.** Usá `.flatten()`.
- **Las celdas 3 y 10 no muestran nada**: las rutas de imagen son de una versión vieja.
- **Los targets se mutan in place.** `train_DAS` y `train_DBM_binary_heatmaps` le ponen featurizers a las unidades del target que les pasás (y DBM también les pone `feature_indices`). Como `_initialize_featurizers` respeta los featurizers con id distinto de `"null"`, si reutilizás esos mismos targets en otro entrenamiento **no** se reinicializan. Hay que reconstruir los targets con el builder.
- **DBM con `tie_masks=True` selecciona componentes enteros** y los patchea en **todas** las posiciones (heads) o en las posiciones elegidas (MLPs).
- **El comentario "Gumbel-Softmax" es impreciso**: no hay muestreo de Gumbel, solo una sigmoide con temperatura.
- **`intervenable_model.eval()` en el loop de entrenamiento**: `FeatureMaskIntervention.forward` usa la compuerta suave solo si `self.training`. No pude verificar (pyvene no está instalado) si `IntervenableModel.eval()` propaga el modo eval a las intervenciones. Si lo hiciera, la compuerta sería dura durante el entrenamiento y la máscara no recibiría gradiente. Los resultados guardados (6 heads elegidas, accuracies altas) sugieren que el entrenamiento sí funciona.
- **Overfitting de DAS**: con ~50 ejemplos y 65 536 parámetros por ubicación, el heatmap de train es poco confiable. Interpretá el de test.
- **Las IIA de DAS en train/test son el promedio sobre un solo grupo de variables** (`answer_position`), calculado con la intervención `interchange` (no con la máscara).

---

## Resumen de una línea por sección

- **Setup**: tres datasets × (train, test) filtrados y guardados en JSON.
- **DAS**: por cada una de 119 ubicaciones se aprende `W ∈ R^{2048×32}` ortonormal con `x' = WWᵀx_s + (I−WWᵀ)x_b`, supervisado con los labels de `answer_position`; mejor test 0.789.
- **Held-out / tres etapas**: el test muestra `answer_position` en `last_token` en las capas 10 > 11 > 12–13; el train sobreajusta.
- **DBM en heads**: máscaras escalares con temperatura y L1 sobre 512 heads; quedan 6 heads (capas 3, 10, 11, 12, 14, 14) con test 0.982 para `answer`.
- **DBM en MLPs**: sobre 48 (capa, posición), queda solo el MLP 0 en `correct_symbol`.
- **Takeaway**: supervisar con el modelo causal es potente, pero obliga a separar train y test.
