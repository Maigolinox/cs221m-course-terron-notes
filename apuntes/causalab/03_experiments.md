# causalab/experiments: entrenar, medir y visualizar

Este submódulo junta el modelo causal (`causal/`) con la red (`neural/`): filtra datasets, arma grillas de lugares a intervenir, corre intervenciones, entrena DAS y DBM, compara las salidas del LM intervenido con los labels del modelo causal y dibuja los resultados. La pregunta que responde es: ¿en qué (capa, posición) o (capa, cabeza) la red se comporta como predice la variable X del algoritmo, y con qué precisión?

Archivos (en `causalab/causalab/experiments/`):

- `filter.py`: `filter_dataset`.
- `interchange_targets.py`: builders de grillas de `InterchangeTarget` y detección de tipo de componente.
- `metric.py`: scoring (`causal_score_intervention_outputs`, `InterchangeMetric`, `score_intervention_outputs`) y la loss de entrenamiento (`LM_loss_and_metric_fn`).
- `train.py`: `train_interventions` (DAS/DBM de punta a punta).
- `configs/train_config.py`: `DEFAULT_CONFIG` y `merge_with_defaults`.
- `io.py`: guardar resultados.
- `jobs/*.py`: experimentos empaquetados.
- `visualizations/*.py`: gráficos.
- `download_models.py`: script CLI para bajar modelos de HF a la cache.
- `__init__.py`: reexporta `filter_dataset` y las funciones de visualización.

El loop de entrenamiento propiamente dicho vive en `neural/pyvene_core/interchange.py:train_interventions`, pero lo explico acá porque solo tiene sentido junto con la loss y con `experiments/train.py`.

## Ideas clave / teoría

### Interchange intervention accuracy (IIA)

Para un target (conjunto de units) y un grupo de variables objetivo V:

```
IIA = (1/N) Σ_i  metric( LM_intervenido(base_i, cf_i),  label_i )
label_i = raw_output de run_interchange(base_i, {v: cf_i para v en V})
```

`metric` es una función booleana que pasa el usuario (por ejemplo, "el string generado contiene la letra esperada"). En el código, la IIA es el `avg_score`.

### DAS: entrenar la rotación

Con un `SubspaceFeaturizer` (W de d x k, ortogonal) en cada unit e intervención `"interchange"`, se minimiza la cross-entropy entre los logits del LM intervenido y los tokens del label:

`L_DAS(W) = CE( LM(base | h ← h_base + (h_src − h_base) W Wᵀ),  label )`

El LM está congelado; solo se entrena W (vía la parametrización ortogonal).

### DBM: entrenar una máscara

Con intervención `"mask"`, cada feature j (o cada unit, si `tie_masks=True`) tiene un logit `m_j` y una compuerta `g_j = sigmoid(m_j / T)`:

`f_out = (1 − g) ⊙ f_base + g ⊙ f_src`

La loss es

`L_DBM(m) = CE(LM(base | f_out), label) + λ · ( Σ_j g_j ) / (número total de compuertas)`

- El primer término empuja a abrir las compuertas que hacen falta para producir el label.
- El segundo (L1 normalizada) empuja a cerrar todo lo que no hace falta (λ = `regularization_coefficient`).
- La temperatura T se baja (annealing) de `temperature_schedule[0]` a `[1]` durante la primera fracción del entrenamiento, para que las compuertas terminen casi binarias. Al final se binariza: `g_j = 1` si `sigmoid(m_j) > 0.5`.

### Boundless DAS

Rotación entrenable (`SubspaceFeaturizer`) **más** máscara por feature, entrenadas juntas: la rotación busca una base y la máscara decide cuántas dimensiones de esa base hacen falta (sin fijar k de antemano, de ahí "boundless").

## `filter.py`: `filter_dataset`

```python
def filter_dataset(dataset, pipeline, causal_model, metric, batch_size=32, validate_counterfactuals=True):
    outputs = pipeline.compute_outputs(dataset, batch_size=batch_size)
    base_outputs_flat = outputs["base_outputs"]
    counterfactual_outputs_flat = outputs["counterfactual_outputs"] if validate_counterfactuals else []
    num_cf_per_example = len(dataset[0]["counterfactual_inputs"]) if len(dataset) > 0 else 0
    cf_idx = 0
    for example_idx in range(dataset_original):
        example = dataset[example_idx]
        base_output = base_outputs_flat[example_idx]
        base_expected = example["input"]["raw_output"]
        if not metric(base_output, base_expected):
            cf_idx += num_cf_per_example
            continue
        if validate_counterfactuals and num_cf_per_example > 0:
            cf_valid = True
            for i in range(num_cf_per_example):
                cf_output = counterfactual_outputs_flat[cf_idx + i]
                cf_expected = example["counterfactual_inputs"][i]["raw_output"]
                if not metric(cf_output, cf_expected):
                    cf_valid = False
                    break
            cf_idx += num_cf_per_example
            if not cf_valid:
                continue
        filtered_examples.append(example)
    return filtered_examples
```

- Corre el LM **sin intervenir** sobre todas las bases y contrafactuales (`compute_outputs`).
- Se queda solo con los ejemplos donde la salida del LM en la base coincide (según `metric(output_dict, expected)`) con el `raw_output` del modelo causal, y opcionalmente también en todos los contrafactuales.
- `cf_idx` recorre la lista aplanada de salidas contrafactuales; asume que **todos** los ejemplos tienen la misma cantidad de contrafactuales (la del primero).
- El argumento `causal_model` no se usa: el expected se lee directamente de la traza (`input_trace["raw_output"]`).
- `metric` recibe el dict de salida (`{"sequences", "scores", "string"}`) y el `raw_output`. Una métrica típica compara `output["string"]` contra el texto esperado.

## `interchange_targets.py`

### Builders

Todos devuelven `Dict[tuple, InterchangeTarget]` y aceptan `mode`:

- `"one_target_all_units"` → `{("all",): target}` con un único grupo que contiene todas las units (para DBM: la máscara decide cuáles usar).
- `"one_target_per_unit"` → `{key: InterchangeTarget([[unit]])}`, una por unit (para grillas de scores y DAS: un experimento por lugar).
- `"one_target_per_layer"` → `{(layer,): target}` con todas las units de la capa.

El agrupamiento lo hace `_group_units_into_targets(units_with_keys, layers, mode)`. En el modo por capa usa el primer elemento de la key como capa (`for (layer, *_), unit in units_with_keys`) y descarta capas vacías con un walrus (`if (group := layer_groups[layer])`).

```python
def build_residual_stream_targets(pipeline, layers, token_positions, mode="one_target_per_unit", target_output=True):
    hidden_size = pipeline.model.config.hidden_size
    for layer in layers:
        for pos in token_positions:
            actual_layer = layer
            actual_target_output = target_output
            if layer == -1:
                actual_layer = 0
                actual_target_output = False
            unit = ResidualStream(layer=actual_layer, token_indices=pos, featurizer=None,
                                  shape=(hidden_size,), feature_indices=None,
                                  target_output=actual_target_output)
            units_with_keys.append(((layer, pos.id), unit))
    return _group_units_into_targets(units_with_keys, layers, mode)
```

- Por defecto interviene `block_output` (residual stream después de la capa).
- **Capa -1** es una convención: se traduce a `block_input` de la capa 0, o sea, los **embeddings**. Así una grilla `[-1, 0, ..., L-1]` cubre "antes de todo" y "después de cada capa". La key conserva `-1`.
- `shape=(hidden_size,)` para que `train.py` sepa el tamaño al crear featurizers.
- Key `(layer, pos.id)`.

`build_mlp_targets(pipeline, layers, token_positions, mode, location="mlp_output")`: igual con `MLP`. La dimensión es `hidden_size`, salvo `mlp_activation` que usa `n_inner` o `intermediate_size` (o `4 * hidden_size` si no encuentra ninguno).

`build_attention_head_targets(pipeline, layers, heads, token_position, mode)`: una `AttentionHead` por (capa, cabeza) con **una sola** token position. `head_size` = `config.head_dim` si existe; si no `hidden_size // num_heads` (busca `n_head`, `num_attention_heads` o `num_heads`). Key `(layer, head)`.

### Utilidades de grilla

- `detect_component_type_from_targets(targets)`: mira el id de la primera unit del primer target y devuelve `"attention_head"`, `"residual_stream"` o `"mlp"` según el prefijo del id (`AttentionHead(...`, `ResidualStream(...`, `MLP(...`). Esto es lo que hace que los jobs "autodetecten" el tipo.
- `extract_grid_dimensions_from_targets(component_type, targets)`: de las keys saca `{"layers", "heads"}` (ordenadas) o `{"layers", "token_position_ids"}` (posiciones en orden de inserción, porque son strings).

## `configs/train_config.py`

`ExperimentConfig` es un `TypedDict` con todos los campos; `DEFAULT_CONFIG`:

| Clave | Default | Para qué |
|---|---|---|
| `intervention_type` | `"mask"` | `"mask"` = DBM, `"interchange"` = DAS |
| `train_batch_size` | 32 | batch de entrenamiento |
| `evaluation_batch_size` | 32 | batch de evaluación |
| `method_name`, `id` | `"default"`, `"default_experiment"` | etiquetas |
| `output_scores` | False | no lo lee ningún código del paquete |
| `top_k_logits` | 10 | no lo lee ningún código del paquete |
| `training_epoch` | 3 | épocas |
| `init_lr` | 1e-2 | learning rate de AdamW |
| `max_output_tokens` | 1 | no lo lee ningún código del paquete (el largo real sale de `pipeline.max_new_tokens`) |
| `log_dir` | `"logs"` | `train.py` lo pisa con `output_dir/logs`; no se escribe TensorBoard en el código actual |
| `patience` | None | early stopping (épocas sin mejora de la loss media) |
| `scheduler_type` | `"constant"` | scheduler de `transformers.get_scheduler` |
| `memory_cleanup_freq` | 50 | cada cuántos batches vaciar cache CUDA |
| `shuffle` | True | barajar por época |
| `DAS.n_features` | 32 | dimensión k del subespacio |
| `masking.regularization_coefficient` | 0.1 | λ de la L1 |
| `masking.temperature_annealing_fraction` | 0.5 | fracción de pasos con annealing |
| `masking.temperature_schedule` | (1.0, 0.001) | T inicial y final |
| `featurizer_kwargs.tie_masks` | False | máscara por unit (True) o por feature (False) |

`merge_with_defaults(config)`: deep copy de los defaults y merge **un nivel** de profundidad: para claves cuyo valor es dict (`DAS`, `masking`, `featurizer_kwargs`) hace `{**default, **user}`; el resto se pisa. Con `None` devuelve los defaults. Hay versiones `Partial...Config` (`total=False`) para tipar la entrada.

## `metric.py`

### `InterchangeMetric` y `make_causal_metric`

```python
@dataclass
class InterchangeMetric:
    fn: Callable[[Dict, Dict, Dict], float]   # (intervention_output, expected, original) -> score
    needs_causal_expected: bool = True
    needs_original_output: bool = False
    target_variables: Tuple[str, ...] | None = None
```

Una métrica general declara qué datos necesita: el label del modelo causal (`expected`), la salida original del LM sin intervenir (`original`), o ambos. Ejemplos del docstring: métrica causal (`output["string"] == expected["string"]`) y métrica de **faithfulness** (`output["string"] == original["string"]`, o sea "la intervención no cambió la salida").

`make_causal_metric(checker, target_variables)` envuelve un `checker(output_dict, expected) -> bool` en un `InterchangeMetric` que devuelve 1.0/0.0, con `needs_causal_expected=True` y las variables objetivo guardadas.

### `score_intervention_outputs(raw_results, dataset, metric, causal_model=None, original_outputs=None) -> {key: float}`

```python
if metric.needs_causal_expected:
    labeled_data = causal_model.label_counterfactual_data(copy.deepcopy(dataset), list(metric.target_variables))
    expected_outputs = [example["label"] for example in labeled_data]
else:
    expected_outputs = [{}] * len(dataset)
if original_outputs is None:
    original_outputs = [{}] * len(dataset)

for key, outputs in raw_results.items():
    string_outputs = outputs.get("string", [])
    flattened_outputs = []
    for item in string_outputs:
        if isinstance(item, list):
            flattened_outputs.extend(item)
        else:
            flattened_outputs.append(item)
    key_scores = []
    for idx, output_string in enumerate(flattened_outputs):
        if idx < len(expected_outputs):
            intervention_output = {"string": output_string}
            score = metric.fn(intervention_output, expected_outputs[idx], original_outputs[idx] ...)
            key_scores.append(float(score))
    scores[key] = sum(key_scores) / len(key_scores) if key_scores else 0.0
```

- Valida que estén los insumos que la métrica declara.
- Etiqueta el dataset **una vez** con el modelo causal (deep copy para no tocar el original).
- `raw_results` es `{key_del_target: salida de run_interchange_interventions}`; `outputs["string"]` es una lista por batch, cada una lista de strings (o un string suelto si el batch tenía un elemento), por eso el aplanado.
- Al checker solo le llega `{"string": ...}` (no sequences ni scores).
- Devuelve el promedio por key.

### `causal_score_intervention_outputs(raw_results, dataset, causal_model, target_variable_groups, metric) -> dict`

La función de scoring que usan `train.py` y los jobs:

- Para cada grupo de variables (p.ej. `("answer",)`, `("answer", "answer_position")`) arma `make_causal_metric(metric, grupo)` y llama `score_intervention_outputs`.
- Devuelve:
  - `results_by_key[key] = {"scores_by_variable": {str(grupo): score}, "avg_score": promedio sobre grupos, "raw_results": raw_results[key]}`,
  - `scores_by_variable[grupo]` = promedio sobre keys,
  - `avg_score` global.

Ojo: un grupo con varias variables, p.ej. `("answer", "answer_position")`, significa **intervenir ambas a la vez** con el mismo contrafactual (ver `label_counterfactual_data` en `01_causal.md`), no "promediar las dos".

### `LM_loss_and_metric_fn(pipeline, intervenable_model, examples, interchange_target, checker, source_pipeline=None, source_intervenable_model=None)`

La loss de entrenamiento de DAS y DBM. Es **teacher forcing**: en vez de generar, se concatena el label al prompt y se leen los logits en las posiciones del label.

```python
batched_base, batched_counterfactuals, inv_locations, feature_indices = (
    prepare_intervenable_inputs(pipeline, examples, interchange_target))
source_representations = None
if source_pipeline is not None:
    source_representations = collect_source_representations(...)
    batched_counterfactuals = None
```

Igual que en evaluación: posiciones, tokenización y (si aplica) activaciones del modelo fuente.

```python
batched_inv_label_strs = [ex["label"] for ex in examples]
if isinstance(batched_inv_label_strs[0], dict):
    batched_inv_label_strs = [item["string"] for item in batched_inv_label_strs]
batched_inv_label_traces = [CausalTrace(mechanisms={"raw_input": Mechanism(...)}, inputs={"raw_input": s})
                            for s in batched_inv_label_strs]
batched_inv_label = pipeline.load(batched_inv_label_traces, max_length=pipeline.max_new_tokens,
                                  padding_side="right", add_special_tokens=False, use_chat_template=False)
```

- Los labels son los `raw_output` contrafactuales (strings, o dicts con `"string"` como en IOI).
- Se tokenizan como si fueran prompts (de ahí las trazas mínimas), **padding a la derecha**, sin special tokens, a exactamente `max_new_tokens` tokens (truncando si el label es más largo).

```python
for k in batched_base:
    batched_base[k] = torch.cat([batched_base[k], batched_inv_label[k]], dim=-1)
```

Concatena `input_ids` y `attention_mask` del label a la derecha de la base. Como la base está paddeada a la izquierda, las posiciones de intervención calculadas antes siguen siendo válidas (agregar tokens a la derecha no corre los índices). Shape: `(batch, L_prompt + max_new_tokens)`.

```python
_, counterfactual_logits = intervenable_model(
    batched_base, batched_counterfactuals,
    unit_locations=inv_locations, subspaces=feature_indices,
    source_representations=source_representations)
labels = batched_inv_label["input_ids"]
logits = counterfactual_logits.logits[:, -labels.shape[-1] - 1 : -1]
pred_ids = torch.argmax(logits, dim=-1)
```

- Forward del modelo intervenible (con gradientes, porque este código no está bajo `no_grad`). pyvene devuelve `(salida_base_original, salida_intervenida)`; se usa la segunda.
- `labels`: `(batch, T)` con `T = max_new_tokens`.
- Los logits en la posición t predicen el token t+1. El primer token del label está en la posición `-T`, así que lo predice la posición `-T-1`; el último lo predice `-2`. Por eso el slice `[-T-1 : -1]`, que da `(batch, T, vocab)` alineado con `labels`.

```python
for i in range(pred_ids.shape[0]):
    pred_str = pipeline.dump(pred_ids[i : i + 1])
    score = checker({"string": pred_str}, examples[i]["label"])
    ...
accuracy = sum(scores) / len(scores) if scores else 1.0
eval_metrics = {"accuracy": accuracy, "token_accuracy": accuracy}
loss = compute_cross_entropy_loss(logits, labels, pipeline.tokenizer.pad_token_id)
```

- Accuracy de entrenamiento con el mismo checker que en evaluación, sobre las predicciones greedy por posición (no una generación real: cada token se predice dado el label correcto anterior). `token_accuracy` es el mismo número (no hay accuracy por token separada).
- `compute_cross_entropy_loss`: aplana a `(batch*T, vocab)` y `(batch*T,)` y hace `cross_entropy(..., ignore_index=pad_token_id)`. Como `pad_token = eos_token`, si el label legítimamente contiene EOS, ese token se ignora.
- Devuelve `(loss, eval_metrics, logging_info)`; `logging_info` trae predicciones y labels decodificados, ids y máscaras de base y contrafactuales, posiciones y feature indices (para debug; el loop no lo usa).

## El loop de entrenamiento: `neural/pyvene_core/interchange.py:train_interventions`

Firma: `train_interventions(pipeline, interchange_target, counterfactual_dataset, intervention_type, config, loss_and_metric_fn, source_pipeline=None) -> str`.

```python
intervenable_model = prepare_intervenable_model(pipeline, interchange_target, intervention_type=intervention_type)
intervenable_model.disable_model_gradients()
intervenable_model.eval()
```

- Modelo intervenible con intervenciones `"interchange"` (DAS) o `"mask"` (DBM).
- `disable_model_gradients()` (pyvene) congela el LM.
- `eval()` desactiva dropout del LM. (Ver la duda sobre el modo de la máscara en "Ojo" de `02_neural.md`.)

```python
train_batch_size = config["train_batch_size"]
num_batches = -(-len(counterfactual_dataset) // train_batch_size)   # ceil
num_epoch = config["training_epoch"]
regularization_coefficient = config.get("masking", {}).get("regularization_coefficient", 0.1)
patience = config.get("patience", None)
scheduler_type = config.get("scheduler_type", "constant")
```

Lectura de config. `-(-a // b)` es división techo.

```python
optimizer_params = []
for k, v in intervenable_model.interventions.items():
    optimizer_params += list(v.parameters())
optimizer = torch.optim.AdamW(optimizer_params, lr=config["init_lr"], weight_decay=0)
scheduler = transformers.get_scheduler(scheduler_type, optimizer=optimizer,
                                       num_training_steps=num_epoch * num_batches)
```

Solo los parámetros de las intervenciones: la rotación (registrada como submódulo vía el featurizer capturado) o la máscara. Si una intervención no tiene parámetros (featurizer identidad en interchange), la lista queda vacía y no hay nada que entrenar.

Schedule de temperatura (solo `"mask"`):

```python
total_steps = num_epoch * num_batches
annealing_steps = int(total_steps * temperature_annealing_fraction)
annealing_schedule = torch.linspace(temperature_start, temperature_end, annealing_steps + 1)
constant_schedule = torch.full((total_steps - annealing_steps,), temperature_end)
temperature_schedule = torch.cat([annealing_schedule, constant_schedule]).to(dtype).to(device)
for k, v in interventions.items():
    init_intervention = v[0] if isinstance(v, tuple) else v
    init_intervention.set_temperature(temperature_schedule[current_step])
```

Baja linealmente de T_inicial a T_final en la primera fracción de pasos y después queda constante. Se setea la temperatura inicial en todas las máscaras (pyvene a veces guarda las intervenciones como tuplas, de ahí el `v[0]`).

El loop:

```python
for epoch in train_iterator:
    indices = list(range(len(counterfactual_dataset)))
    if shuffle:
        random.shuffle(indices)
    for step, start in enumerate(epoch_iterator):
        examples = [counterfactual_dataset[i] for i in indices[start : start + train_batch_size]]
        loss, eval_metrics, _logging_info = loss_and_metric_fn(
            pipeline, intervenable_model, examples, interchange_target,
            source_pipeline, source_intervenable_model)
        if intervention_type == "mask":
            temp = temperature_schedule[current_step]
            total_sparsity = torch.tensor(0.0, device=loss.device)
            total_mask_elements = 0
            for k, v in interventions.items():
                mask_intervention = v[0] if isinstance(v, tuple) else v
                total_sparsity = total_sparsity + mask_intervention.get_sparsity_loss()
                total_mask_elements += mask_intervention.mask.numel()
                mask_intervention.set_temperature(temp)
            if total_mask_elements > 0:
                loss = loss + regularization_coefficient * (total_sparsity / total_mask_elements)
        ...
        loss.backward()
        optimizer.step()
        scheduler.step()
        current_step += 1
        intervenable_model.set_zero_grad()
        if step % memory_cleanup_freq == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
```

- Por época, baraja índices y arma batches.
- `loss_and_metric_fn` es la `LM_loss_and_metric_fn` (envuelta por `train.py` con el checker).
- Para máscaras: suma las L1 de todas las compuertas, normaliza por el número total de compuertas (así λ significa lo mismo con 1 o 1000 features) y la agrega a la loss. Ojo con el orden: la temperatura del paso `current_step` se setea **después** del forward de este paso, así que cada forward usa la temperatura del paso anterior (desfase de un paso, irrelevante en la práctica).
- Paso estándar: backward, step, scheduler, poner gradientes en cero (`set_zero_grad` de pyvene).
- Barra de progreso con loss y métricas medias.

Al final de cada época, si `patience` no es None: si la loss media de la época mejora, resetea el contador; si no, lo incrementa y corta al llegar a `patience`.

Conversión de máscaras a `feature_indices`:

```python
if intervention_type == "mask":
    model_units = interchange_target.flatten()
    for kv, model_unit in zip(intervenable_model.interventions.items(), model_units):
        k, v = kv
        intervention = v[0] if isinstance(v, tuple) else v
        if config["featurizer_kwargs"]["tie_masks"]:
            if torch.sigmoid(intervention.mask[0]) > 0.5:
                indices = None
            else:
                indices = []
        else:
            mask_binary = (torch.sigmoid(intervention.mask) > 0.5).float().cpu()
            indices = torch.nonzero(mask_binary).numpy().flatten().tolist()
        model_unit.set_feature_indices(indices)
```

- Empareja cada intervención con su unit por orden (asume que `interventions` está en el mismo orden que `flatten()`).
- `tie_masks=True`: unit seleccionada → `None` (todas las features), no seleccionada → `[]`.
- `tie_masks=False`: lista de features con compuerta abierta.
- Escribe los índices **en las units del target** (mutación in place). Así, en la evaluación posterior con intervención `"interchange"`, solo se intercambian las features elegidas. Ojo: para `tie_masks=False`, si ninguna feature quedó abierta el resultado es `[]` (no intercambia nada), y si quedaron todas es la lista completa (no `None`).

Limpieza y retorno de un string resumen (`"Trained intervention for ... Final metrics: loss: ... accuracy: ..."`).

Importante sobre cómo queda la unit después del entrenamiento:

- **DAS**: la rotación entrenada vive dentro del `SubspaceFeaturizer` de cada unit (los módulos son los mismos objetos que usaron las intervenciones). Al evaluar con `"interchange"` se crea un modelo intervenible nuevo, pero con el mismo featurizer, así que usa la rotación entrenada.
- **DBM**: la máscara vive en la intervención, que se borra; lo que persiste es `feature_indices`.

## `train.py`: `train_interventions` de alto nivel

```python
def train_interventions(causal_model, interchange_targets, train_dataset_path, test_dataset_path,
                        pipeline, target_variable_group, output_dir, metric, config=None,
                        save_results=True, source_pipeline=None) -> Dict[str, Any]:
```

- `interchange_targets`: dict `{key: target}` o un solo target (se envuelve como `{("single",): target}`).
- Los datasets se pasan como **rutas** a JSON (los que escribe `save_counterfactual_examples`), ya filtrados.
- `target_variable_group`: tupla de variables a intervenir juntas en el modelo causal para generar labels, p.ej. `("answer",)`.
- `metric(output_dict, expected) -> bool`.

Paso a paso:

1. `config = merge_with_defaults(config)`; valida `intervention_type` en `{"mask", "interchange"}` y que para `"mask"` exista `featurizer_kwargs` (siempre existe tras el merge, así que en la práctica no falla).
2. Carga el train con `load_counterfactual_examples` (deserializa a trazas), crea `output_dir/logs`.
3. `_initialize_featurizers(targets, config)`:

```python
for _key, target in interchange_targets.items():
    for unit in target.flatten():
        if unit.featurizer.id != "null":
            continue
        unit_shape = unit.shape
        if unit_shape is None:
            raise ValueError(...)
        if intervention_type == "mask":
            unit.set_featurizer(Featurizer(n_features=unit_shape[0], tie_masks=tie_masks, id=f"mask_{unit.id}"))
        else:
            unit.set_featurizer(SubspaceFeaturizer(shape=(unit_shape[0], n_features), trainable=True, id=f"DAS_{unit.id}"))
```

   - Respeta featurizers ya puestos (id distinto de `"null"`): así se puede preinicializar con PCA o con Boundless DAS.
   - DBM: featurizer **identidad** con `n_features = shape[0]` (la máscara opera sobre las coordenadas crudas) e id `mask_<unit>`.
   - DAS: `SubspaceFeaturizer` de `(d, n_features)` con inicialización ortogonal aleatoria, id `DAS_<unit>`.
   - Ojo: como el id deja de ser `"null"`, si después vuelvo a llamar `train_interventions` con los mismos targets, no se reinicializan.
4. Carga el test y etiqueta el train: `labeled_train_dataset = causal_model.label_counterfactual_data(deepcopy(train), list(target_variable_group))`.
5. Para cada `(key, target)`:
   - entrena con `train_interventions_pyvene(..., loss_and_metric_fn=lambda p, m, b, t, sp, sim: LM_loss_and_metric_fn(p, m, b, t, metric, source_pipeline=sp, source_intervenable_model=sim))` (la lambda inyecta el checker);
   - lee `target.get_feature_indices()`;
   - evalúa en train y test: `run_interchange_interventions(..., output_scores=False)` y `causal_score_intervention_outputs(...)` con `[target_variable_group]`;
   - guarda `train_score`, `test_score`, `feature_indices`, las evaluaciones y el target entrenado.
6. Promedios `avg_train`, `avg_test`; `num_selected` cuenta units con índices no vacíos (ojo: `None` cuenta como "no seleccionada" en este conteo porque `if indices and len(indices) > 0` es falso para `None`; con `tie_masks=True` las seleccionadas tienen `None`, así que este número puede no significar lo que su nombre sugiere. `DBM_binary_grid.py` recalcula `num_selected_units` correctamente con `get_selected_units`).
7. Metadata (tipo `"DBM"`/`"DAS"`, modelo, rutas, tamaños, scores, config de entrenamiento, `regularization_coefficient`/`tie_masks` o `n_features`).
8. Si `save_results`: `train_eval/` y `test_eval/` (scores.json, raw_results.json, raw_results.safetensors si hay sequences), `training/feature_indices.json`, `models/<key>/` (cada target con `InterchangeTarget.save`), `metadata.json`.

Devuelve `{"results_by_key", "avg_train_score", "avg_test_score", "metadata", "output_paths"}` (+ `num_selected_units` en DBM).

## `io.py`

- `_key_to_str(key)`: `(a, b)` → `"a__b"`, `(a,)` → `"a"`, otro → join con `__`.
- `save_experiment_metadata(metadata, output_dir, filename="metadata.json")`, `save_json_results(data, output_dir, filename)`, `save_tensor_results(tensors, output_dir, filename)` (safetensors).
- `save_intervention_results(results_by_key, output_dir, prefix="")`: `scores.json` = `{key_str: avg_score}`, `raw_results.json` = `{"string": [raw_results["string"] por key]}` y, si hay, concatena todas las `sequences` en un tensor y lo guarda en safetensors.
- `save_training_artifacts(results_by_key, output_dir)`: `training/feature_indices.json` y `models/<key_str>/` vía `trained_target.save`.
- `save_aggregate_metadata`: alias de `save_experiment_metadata`.

## `jobs/`

### `residual_stream_tracing.py`: `run_residual_stream_tracing`

```python
def run_residual_stream_tracing(pipeline, prompt, counterfactual_prompt, token_positions, output_dir,
                                layers=None, generate_visualization=True, save_results=True,
                                verbose=True, source_pipeline=None):
```

El experimento más simple: un solo par (prompt, prompt contrafactual), sin modelo causal.

- Arma un dataset de un ejemplo con dicts planos `{"raw_input": prompt}` (no trazas; funciona porque `pipeline.load` solo lee `item["raw_input"]`).
- Capas por defecto: `[-1] + range(num_layers)`.
- `build_residual_stream_targets(..., mode="one_target_per_unit")` y por cada `(layer, pos_id)` corre `run_interchange_interventions(batch_size=1, output_scores=False)`: full-vector patching del residual stream de ese token en esa capa.
- Guarda `results/intervention_results.json` (strings) y `.safetensors` (sequences), y un heatmap `residual_stream_heatmap.png` con `plot_residual_stream_intervention_heatmap(..., color_by_frequency=True)`: cada celda muestra el token que salió, coloreado por frecuencia.
- La lectura típica: en las capas bajas, patchear el token que difiere entre los prompts cambia la salida; en capas altas la información ya se movió al último token, y es patchear el último token lo que la cambia.

### `interchange_score_grid.py`

`run_interchange_score_heatmap(causal_model, interchange_targets, dataset_path, pipeline, target_variable_groups, batch_size, output_dir, metric, save_results=True, verbose=True, source_pipeline=None)` (marcado "currently unused but not dead code"):

1. Detecta tipo de componente y dimensiones de grilla.
2. Carga el dataset.
3. `run_interchange_interventions` por target (full vector con el featurizer que tengan las units, normalmente identidad).
4. `causal_score_intervention_outputs` con todos los grupos de variables.
5. Un heatmap por grupo en `heatmaps/{dataset}_{var}.png` con `plot_score_heatmap`.

Devuelve `{"scores": {grupo: {key: score}}, "component_type", "metadata", "output_paths"}`. Ojo: el docstring promete `results/scores.json` y `raw_results`, pero el código solo guarda heatmaps (y no escribe `metadata.json`).

`run_interchange_custom_score_heatmap(interchange_targets, dataset_path, pipeline, batch_size, output_dir, metric: InterchangeMetric, causal_model, ...)`: igual pero con un `InterchangeMetric` arbitrario; si la métrica necesita la salida original, corre `pipeline.compute_outputs` y pasa `base_outputs`. Útil para métricas de faithfulness. Ojo: el ejemplo del docstring no pasa `causal_model`, que es obligatorio en la firma (se usa para deserializar el dataset).

### `DAS_grid.py`: `train_DAS`

Wrapper de `train_interventions` para DAS sobre una grilla (normalmente `one_target_per_unit`):

- Config por defecto si no se pasa: `intervention_type="interchange"`, batch 32, eval batch 64, 4 épocas, lr 0.001, `n_features=32`. Si se pasa config, se mergea y se fuerza `intervention_type="interchange"`.
- Después de entrenar, extrae `train_scores` y `test_scores` por key, calcula promedio, máximo y mejor ubicación, y dibuja `heatmaps/train.png` y `heatmaps/test.png`. Reescribe `metadata.json` enriquecido.

### `DBM_binary_grid.py`: `train_DBM_binary_heatmaps`

DBM con `tie_masks=True` sobre **un solo** target (normalmente `one_target_all_units`): aprende qué units usar.

- Detecta el tipo de componente por el id de la primera unit.
- Config por defecto: batch 32, eval 64, **20 épocas**, lr 0.001, `regularization_coefficient=100`. Siempre fuerza `featurizer_kwargs={"tie_masks": tie_masks}`. Ojo: no fija `intervention_type`, se apoya en que el default es `"mask"`; si el config del usuario dice `"interchange"`, entrenaría DAS.
- `selected_units = get_selected_units(feature_indices)`: las units con `None` (seleccionadas), devueltas como tuplas (capa, cabeza) o (capa, posición).
- Dibuja `heatmaps/{var}_mask.png` con `plot_binary_mask` (grilla binaria de seleccionadas).

La lógica: todas las units del target comparten el contrafactual; la máscara aprende el subconjunto mínimo de units (por la L1) cuyo patching conjunto reproduce el label. Es la forma de encontrar "el circuito" de cabezas que cargan una variable.

### `DBM_feature_masks_grid.py`: `train_DBM_feature_masks`

DBM con `tie_masks=False`: qué **features** dentro de cada unit importan.

- Acepta un target o un `Dict[int, InterchangeTarget]` (modo por capa; se convierte a keys `(layer,)`).
- `_get_n_features_by_unit` exige que **todas** las units tengan `featurizer.n_features` definido. Ojo: una unit recién creada por los builders tiene `Featurizer()` con `n_features=None`, así que esta función lanza `ValueError` antes de llegar a `_initialize_featurizers`. La "Option 1" del docstring (featurizer identidad) no funciona tal cual: hay que setear antes un featurizer con `n_features` (p.ej. `Featurizer(n_features=d, id="identity_masks")`) o un `SubspaceFeaturizer` (PCA, que es el uso pensado).
- Config por defecto: 20 épocas, lr 0.001, `tie_masks=False`; fuerza `intervention_type="mask"`.
- Respeta featurizers pre-inicializados (p.ej. PCA congelado): la máscara elige componentes principales.
- Devuelve scores simples o por capa (con `best_layer`), `feature_indices`, `n_features_by_unit`, y dibuja `heatmaps/{var}_features.png` con `plot_feature_counts` (cuántas features quedaron seleccionadas por unit).
- No acepta `source_pipeline`.

### `boundless_DAS_feature_mask_grid.py`: `train_boundless_DAS`

```python
config = merge_with_defaults(config)
interchange_target = copy.deepcopy(interchange_target)
n_features = config["DAS"]["n_features"]
for unit in all_units:
    if unit.featurizer.id == "null":
        ...
        unit.set_featurizer(SubspaceFeaturizer(shape=(unit.shape[0], n_features), trainable=True, id="boundless_DAS"))
result = train_DBM_feature_masks(..., config=config, title=title)
result["metadata"]["experiment_type"] = "boundless_DAS"
```

- Deep-copia el target para no mutar el del llamador (ojo: entonces los featurizers y `feature_indices` entrenados **no** quedan en el objeto que pasaste; están en el resultado y en disco).
- Pone un `SubspaceFeaturizer` **entrenable** (k = `DAS.n_features`, default 32) en cada unit sin featurizer, con aviso si k > d.
- Delega en `train_DBM_feature_masks`. Como la intervención de máscara captura el featurizer, el optimizador ve tanto la máscara como la rotación y las entrena juntas.

### `collect_PCA.py`: `collect_and_compute_PCA`

```python
def collect_and_compute_PCA(interchange_targets, data, pipeline, labels, component_tuples,
                            shape_labels=None, combine_mode="concatenate", n_components=None,
                            normalize=True, output_dir=None, batch_size=32, save_results=True, verbose=True):
```

1. Valida argumentos; `n_components` por defecto = máximo índice pedido en `component_tuples` + 1.
2. Junta todas las units de todos los targets y hace **una sola** pasada de `collect_features` (eficiente).
3. `compute_svd` por unit (en float32) → `svd_results_by_unit`: es lo que se usa para inicializar featurizers PCA (`rotation` de `(d, n)`).
4. Por target, combina las features de sus units:
   - `"concatenate"`: `cat(dim=1)` → `(n_muestras, n_units * d)` (espacio conjunto),
   - `"stack"`: `cat(dim=0)` → `(n_muestras * n_units, d)` (las units como muestras distintas en un espacio compartido).
5. SVD sobre lo combinado por target (para graficar).
6. Si se guarda: `metadata.json`, `features/{key}.safetensors`, `svd/{key}.json` (varianza explicada y shape) y `plots/{key}/` con scatter 2D/3D (`plot_pca_scatter`), coloreados por `labels` y con forma por `shape_labels`.

El docstring muestra `data=[{"input": x} for x in inputs]`: solo hace falta la clave `input`.

## `visualizations/`

Todas son funciones de matplotlib que reciben dicts de scores o de `feature_indices` y opcionalmente guardan a archivo. Lo importante es qué consumen:

- **`score_heatmap.py`**
  - `plot_attention_head_heatmap(scores, layers, heads, title, save_path)`: matriz capa (y) x cabeza (x) con `NaN` donde falta.
  - `plot_residual_stream_heatmap(scores, layers, token_position_ids, ...)`: capa x token position.
  - `plot_score_heatmap(scores, interchange_targets, title, save_path)`: dispatcher que autodetecta el tipo con los targets. MLP usa la misma grilla que residual stream.
- **`string_heatmap.py`**: `plot_residual_stream_intervention_heatmap(intervention_results, prompt, layers, token_positions, pipeline, correct_answer=None, title=None, save_path=None, show_scores=True, color_by_frequency=False)`: celda = token generado tras intervenir (`result["string"][0]`), etiquetas del eje x = `"índice: token"` del prompt, capas invertidas para que -1 quede abajo; colorea por acierto contra `correct_answer` o por frecuencia (top 5 outputs con colores propios).
- **`binary_mask.py`**: `get_selected_heads / get_selected_residual_positions / get_selected_mlps` (units con `indices is None`), `get_selected_units` (dispatcher), `plot_attention_head_mask / plot_residual_stream_mask / plot_mlp_mask` y `plot_binary_mask(feature_indices, title, save_path)` que autodetecta el tipo a partir de los ids. Convención: `None` = seleccionada, `[]` = no.
- **`feature_masks.py`**: `count_selected_features(indices, n_features)` (`None` cuenta como todas), matrices de conteos por unit, `plot_*_feature_counts` y `plot_feature_counts(feature_indices, scores, n_features, title, save_path)` que soporta modo simple o por capa (con columna de accuracy por capa). `NFeatures` es `int | Dict[str, int]`.
- **`unit_id.py`**: parseo de ids (`extract_layer_from_unit_id`, `extract_token_position_from_unit_id`, `extract_layer_head_from_unit_id`), `detect_component_type(feature_indices)`, `is_per_layer_mode`, `extract_grid_dimensions`. Todo el sistema de visualización depende del formato de los ids de `LM_units.py`.
- **`pca_scatter.py`**: `plot_pca_scatter(features, svd_result, labels, component_tuples, shape_labels=None, title=None, save_dir=None, figsize=(8, 6))`: proyecta (normalizando con `mean`/`std` si están) y grafica 2D/3D; replica los labels si hay más puntos que labels y la cantidad es múltiplo (modo stack); máximo 10 formas de marcador.
- **`text_analysis.py`**: `print_residual_stream_patching_analysis(scores, layers, token_position_ids, title=None, save_path=None)`: análisis en texto de una grilla de scores: agrupa valores de score en clusters (scores a menos de 0.10 de la media del cluster) y reporta regiones (rangos de capas contiguos vía `_find_contiguous_ranges`) por posición. Devuelve el texto y lo imprime.
- **`utils.py`**: `create_heatmap`, `create_binary_mask_heatmap`, `create_feature_count_heatmap` y helpers de anotación; son el boilerplate de matplotlib que usan las demás.

## Ojo / errores comunes / dudas

- **Los datasets de `train.py` y los jobs se pasan por ruta**, no como lista: hay que guardarlos antes con `save_counterfactual_examples`.
- **`DBM_feature_masks` sin featurizer con `n_features` falla** (ver arriba).
- **`train_boundless_DAS` deep-copia los targets**: el resultado entrenado no queda en tus objetos.
- **`num_selected_units` de `train.py`** no cuenta bien las units seleccionadas con `tie_masks=True` (son `None`); el de `DBM_binary_grid.py` sí.
- **El label se trunca a `max_new_tokens` tokens** en la loss. Si el `raw_output` tokeniza en más tokens, se entrena solo el prefijo; y en evaluación la generación también tiene `max_new_tokens` tokens. Conviene que el checker sea tolerante (p.ej. "contiene" o comparar tras `strip()`).
- **Accuracy de entrenamiento ≠ IIA de evaluación**: la primera es teacher-forced sobre el dataset etiquetado y con la máscara en el modo que tenga en entrenamiento; la segunda es generación real con `feature_indices` binarizados.
- **Varias claves del config no se usan** (`output_scores`, `top_k_logits`, `max_output_tokens`, `method_name`).
- **Docstrings con imports viejos** (`causalab.experiments.scripts...`): el módulo real es `causalab.experiments.jobs`.

## Resumen de una línea por sección

- **IIA**: fracción de ejemplos donde el LM intervenido produce el label del modelo causal.
- **DAS / DBM / Boundless DAS**: rotación entrenada con CE; máscara sigmoide con temperatura + L1; ambas a la vez.
- **`filter_dataset`**: me quedo con ejemplos que el LM resuelve en base (y contrafactuales).
- **`interchange_targets`**: grillas de units por capa x posición o capa x cabeza; capa -1 = embeddings; tres modos de agrupamiento.
- **`train_config`**: defaults y merge de un nivel.
- **`metric`**: scoring por grupos de variables; loss teacher-forced con el label concatenado y logits desplazados un lugar.
- **Loop de entrenamiento**: AdamW sobre parámetros de intervenciones, annealing de temperatura, sparsity normalizada, early stopping, máscara → `feature_indices`.
- **`train.py`**: inicializa featurizers, etiqueta, entrena y evalúa train/test por target, guarda todo.
- **Jobs**: tracing, grid de scores, DAS, DBM binario, DBM por feature, Boundless DAS, PCA.
- **Visualizaciones**: heatmaps de scores, de tokens, de máscaras y de conteos, scatter PCA y análisis en texto; todas parsean los ids de las units.
