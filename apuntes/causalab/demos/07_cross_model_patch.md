# 07 — Cross-model activation patching (`07_cross_model_patch.ipynb`)

Notebook: `causalab/demos/onboarding_tutorial/07_cross_model_patch.ipynb`.

Séptima y última notebook del onboarding. Hasta ahora, en toda interchange intervention, la activación del contrafáctico salía **del mismo modelo** en el que se patcheaba. Acá se introduce el **cross-model patching**: recolectar activaciones de un *source model* con el input contrafáctico y patchearlas en un *target model* que corre el input base. La notebook no hace todavía un experimento entre modelos distintos: hace un **test de validación**. Usa como source el **mismo** pipeline que el target y verifica que los resultados sean idénticos a los del patching normal de la 03 ([03_localize_with_patching.md](03_localize_with_patching.md)). La pregunta: ¿la implementación del camino "recolectar en un modelo, inyectar en otro" es correcta?

Viene de la 06 ([06_boundless_DAS.md](06_boundless_DAS.md)), pero conceptualmente se apoya en la 02/03: es el mismo `run_interchange_score_heatmap` con un argumento nuevo, `source_pipeline`. Referencia: [../02_neural.md](../02_neural.md) (pyvene_core) y [../03_experiments.md](../03_experiments.md).

---

## Ideas clave / teoría

**Patching normal** (02–06): con un solo modelo `N`,

  `N_{L ← L^N(s)}(b)`: correr `N` en `b` con la activación en `L` reemplazada por la de `N` en `s`.

**Cross-model patching**: con un source `N_src` y un target `N_tgt`,

  `N_tgt{L ← L^{N_src}(s)}(b)`: la activación que se inyecta sale de **otro** modelo.

Para que tenga sentido, la ubicación `L` tiene que existir en ambos y tener la **misma dimensión** (por ejemplo, el residual stream de la capa 10 en dos checkpoints del mismo modelo). Usos que menciona la notebook:
- comparar checkpoints de un mismo entrenamiento (¿una representación de la época 1 "funciona" en el modelo final?);
- modelos con la misma arquitectura y distinto entrenamiento (por ejemplo, base vs fine-tuned);
- modelos distintos con la misma dimensión oculta;
- *transfer*: ver si representaciones de un modelo pueden controlar el comportamiento de otro.

**Test de validación.** Si `N_src = N_tgt`, el cross-model patching es matemáticamente igual al patching normal. Cualquier diferencia en los puntajes indicaría un bug (posiciones mal calculadas, orden de las activaciones mal alineado, etc.). Este es un *sanity check* clásico: antes de usar una herramienta nueva, comprobar que reproduce el caso conocido.

---

## El código de la librería que usa esta notebook

El parámetro `source_pipeline` está en casi todas las funciones de alto nivel (`run_interchange_score_heatmap`, `run_residual_stream_tracing`, `train_DAS`, `train_DBM_binary_heatmaps`, `train_interventions`) y se pasa hasta `causalab/causalab/neural/pyvene_core/interchange.py`.

### `run_interchange_interventions(..., source_pipeline=...)` (`interchange.py`)

```python
intervenable_model = prepare_intervenable_model(pipeline, interchange_target, intervention_type="interchange")
source_intervenable_model = None
if source_pipeline is not None:
    source_intervenable_model = prepare_intervenable_model(
        source_pipeline, interchange_target, intervention_type="collect"
    )
```

- Se arman **dos** `IntervenableModel` de pyvene: uno de intercambio sobre el target y uno de **recolección** (`FeatureCollectIntervention`) sobre el source, con las mismas unidades. Se crean una sola vez, fuera del loop de batches.

### `batched_interchange_intervention` (`interchange.py`)

```python
batched_base, batched_counterfactuals, inv_locations, feature_indices = \
    prepare_intervenable_inputs(pipeline, examples, interchange_target)

source_representations = None
if source_pipeline is not None:
    source_representations = collect_source_representations(
        source_pipeline, examples, interchange_target, source_intervenable_model
    )
    batched_counterfactuals = None

output = pipeline.intervenable_generate(intervenable_model, batched_base, batched_counterfactuals,
                                        inv_locations, feature_indices,
                                        source_representations=source_representations, ...)
```

1. Se preparan igual que siempre los inputs base, los índices de posición de base y contrafáctico, y los `feature_indices` (con el pipeline target).
2. Si hay source: se recolectan las activaciones del contrafáctico **en el source** y se ponen `batched_counterfactuals = None`. Así pyvene no corre el contrafáctico en el target.
3. `intervenable_generate` le pasa a pyvene `sources=None` y `source_representations=[...]`. pyvene usa esas activaciones ya calculadas como "source" de la intervención en vez de computarlas.

### `collect_source_representations` (`causalab/causalab/neural/pyvene_core/collect.py`)

```python
cf_inputs_raw = list(zip(*[ex["counterfactual_inputs"] for ex in examples]))
batched_cf_for_source = [source_pipeline.load(list(cf_group)) for cf_group in cf_inputs_raw]
source_cf_indices = [
    model_unit.index_component(list(cf_group), batch=True, is_original=False)
    for group, cf_group in zip(interchange_target, cf_inputs_raw)
    for model_unit in group
]
return collect_batch_representations(source_pipeline, batched_cf_for_source,
                                     interchange_target, source_cf_indices,
                                     intervenable_model=source_intervenable_model)
```

- `zip(*...)` transpone: de "por ejemplo, lista de contrafácticos" a "por grupo, batch de contrafácticos" (un grupo del `InterchangeTarget` ↔ un contrafáctico).
- Tokeniza los contrafácticos **con el tokenizer del source** (`source_pipeline.load`).
- Calcula los índices de posición con `model_unit.index_component`. Ese indexer es un `TokenPosition` que guarda **el pipeline con el que fue creado** (el target, porque `create_token_positions(pipeline)` se llamó con el target), así que las posiciones se calculan con el tokenizer del target.
- `collect_batch_representations`: por cada grupo corre `_collect_activations_single_batch(model, cf_input, group_indices)`, que hace un forward con la intervención de recolección y devuelve una lista de tensores (uno por unidad, en el orden de `sorted_keys` de pyvene). Junta todo en una lista plana.

### `prepare_intervenable_model` (`causalab/causalab/neural/pyvene_core/intervenable_model.py`)

Igual que siempre: un config de pyvene por unidad con `group_key = índice del grupo`. Con `intervention_type="collect"` usa `featurizer.get_collect_intervention()`.

---

## Recorrido por la notebook

### Celda 1 — setup

```python
pipeline = LMPipeline(model_name, max_new_tokens=1, device=device,
                      dtype=torch.bfloat16 if device == "cuda" else torch.float32,
                      max_length=32)

def metric(neural_output, causal_output):
    return causal_output in neural_output["string"] or neural_output["string"] in causal_output
```

- Un solo pipeline, que va a hacer de source y de target. Misma métrica que la 03.
- Salida guardada: corrió en cuda.

### Celda 3 — dataset filtrado

```python
size = 64
dataset = [different_symbol() for _ in range(size)]
filtered_dataset = filter_dataset(dataset=dataset, pipeline=pipeline, causal_model=causal_model,
                                  metric=metric, batch_size=32)
```

- 64 pares `different_symbol` → 58 tras filtrar (salida guardada). Ejemplo guardado: "X. green / H. yellow" vs "G. green / N. yellow".

### Celda 5 — guardar

```python
temp_dir = "./temp_cross_model_test"
os.makedirs(temp_dir, exist_ok=True)
dataset_path = os.path.join(temp_dir, "filtered_dataset")
save_counterfactual_examples(filtered_dataset, dataset_path)
```

- A diferencia de la 05/06, acá se crea solo el directorio padre, así que `filtered_dataset` es un **archivo** (JSON sin extensión) y funciona con el código actual.
- `run_interchange_score_heatmap` usa `Path(dataset_path).stem` = `"filtered_dataset"` como nombre del dataset en los nombres de las figuras.

### Celda 7 — targets

```python
all_token_positions = create_token_positions(pipeline)
ordered_position_names = ["correct_symbol", "correct_symbol_period", "symbol0", "symbol0_period",
                          "symbol1", "symbol1_period", "last_token"]
token_positions = [all_token_positions[name] for name in ordered_position_names]
num_layers = pipeline.model.config.num_hidden_layers
layers = [-1] + list(range(num_layers))
targets = build_residual_stream_targets(pipeline=pipeline, layers=layers,
                                        token_positions=token_positions, mode="one_target_per_unit")
```

- Exactamente la grilla de la 03: 17 capas × 7 posiciones = 119 targets (salida guardada).

### Celda 9 — patching normal

```python
regular_output_dir = os.path.join(temp_dir, "regular_patching")
regular_result = run_interchange_score_heatmap(
    causal_model=causal_model, interchange_targets=targets, dataset_path=dataset_path,
    pipeline=pipeline, batch_size=32,
    target_variable_groups=[("answer",), ("answer_position",)],
    output_dir=regular_output_dir, metric=metric, verbose=True,
    source_pipeline=None,
)
print(f"Average score: {regular_result['metadata']['avg_score']:.4f}")
```

- `source_pipeline=None`: el camino de siempre (pyvene corre el contrafáctico en el mismo modelo).
- `metadata["avg_score"]` es el promedio, sobre los dos grupos de variables, de la IIA promedio sobre los 119 targets.
- Salida guardada: 119 targets en 20 s; `avg_score` 0.4704.

### Celda 11 — cross-model con el mismo modelo

```python
cross_model_result = run_interchange_score_heatmap(
    ..., output_dir=cross_model_output_dir, ..., source_pipeline=pipeline,
)
```

- Igual, pero `source_pipeline=pipeline`. Ahora las activaciones del contrafáctico se recolectan aparte (con el `IntervenableModel` de recolección) y se pasan como `source_representations`.
- Salida guardada: 119 targets en 30 s (más lento por la pasada de recolección separada); `avg_score` 0.4704, igual.

### Celda 13 — comparar puntaje por puntaje

```python
all_match = True
for var_group in [("answer",), ("answer_position",)]:
    regular_scores = regular_result["scores"][var_group]
    cross_model_scores = cross_model_result["scores"][var_group]
    mismatches = []
    for key in regular_scores:
        if key not in cross_model_scores:
            mismatches.append((key, "missing", None))
        elif abs(regular_scores[key] - cross_model_scores[key]) > 1e-6:
            mismatches.append((key, regular_scores[key], cross_model_scores[key]))
    if mismatches:
        all_match = False
        ...
    else:
        scores_list = list(regular_scores.values())
        print(f"  All {len(regular_scores)} scores match!")
        print(f"  Mean score: {np.mean(scores_list):.4f}")
        ...
```

- `result["scores"]` es `{var_group: {(layer, pos_id): IIA}}`. Recorre las 119 claves y compara con tolerancia 1e-6. Guarda las que faltan o difieren y muestra las 5 primeras.
- Si no hay diferencias, imprime media, mínimo y máximo.
- Salida guardada: las 119 coinciden en ambos grupos. `answer`: media 0.1966 (mín 0, máx 1); `answer_position`: media 0.7441 (mín 0, máx 1). "SUCCESS". El `avg_score` de 0.4704 es justamente `(0.1966 + 0.7441) / 2`.
- Como la IIA es una proporción de aciertos discretos sobre 58 ejemplos, "coinciden" quiere decir que las generaciones fueron idénticas en todos los ejemplos. Con bfloat16, un orden de operaciones distinto podría en teoría cambiar algún argmax, pero acá no pasó.

### Celda 15 — heatmaps lado a lado

```python
for var_name in ["answer", "answer_position"]:
    regular_heatmap_path = os.path.join(regular_output_dir, "heatmaps", f"filtered_dataset_{var_name}.png")
    cross_model_heatmap_path = os.path.join(cross_model_output_dir, "heatmaps", f"filtered_dataset_{var_name}.png")
    if os.path.exists(regular_heatmap_path) and os.path.exists(cross_model_heatmap_path):
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        axes[0].imshow(Image.open(regular_heatmap_path)); ...
        axes[1].imshow(Image.open(cross_model_heatmap_path)); ...
        plt.show()
```

- Carga con PIL los PNG que generó `run_interchange_score_heatmap` (`<output_dir>/heatmaps/<dataset>_<var>.png`) y los muestra lado a lado. A diferencia de la 04, acá las rutas sí coinciden con la convención actual.

### Celda 17 — limpieza

Código para borrar `temp_dir` comentado; solo imprime dónde quedaron los archivos.

### Celda 18 — conclusión

- Patching normal y cross-model con el mismo modelo dan puntajes y heatmaps idénticos: la implementación queda validada.
- Observaciones de la notebook: el heatmap de `answer` muestra señal en `correct_symbol` en capas tempranas, y el de `answer_position`, "señal en `last_token` alrededor de la capa 15".
- Próximos pasos: análisis de checkpoints, comparación de modelos con arquitecturas compatibles, experimentos de transferencia.

---

## Ejercicios

No hay celdas de ejercicio. El paso natural es hacer cross-model "de verdad":
- Cargar un segundo `LMPipeline` (por ejemplo `meta-llama/Llama-3.2-1B`, la versión base, que tiene la misma arquitectura y el mismo tokenizer) y pasarlo como `source_pipeline`. Comparar los heatmaps con los del patching normal: donde la IIA se mantiene, las representaciones son "compatibles" entre los dos modelos en esa ubicación.
- También se puede pasar `source_pipeline` a `train_DAS`. Según el docstring de `train_interventions`, eso entrena features en el target que se alinean con las activaciones del source.

---

## Ojo / errores comunes / dudas

- **La interpretación de `answer_position` en la conclusión hay que tomarla con cuidado.** El dataset es `different_symbol`, que según la 01 y la 03 **confunde `answer_position` con no intervenir** (proporción 0). En ese heatmap, IIA = 1 significa "el patch no cambió la salida", no "acá está `answer_position`". La media alta (0.7441) refleja eso: la mayoría de las ubicaciones son irrelevantes. Lo que el heatmap de `answer_position` muestra en realidad es el "negativo" del de `answer`.
- **Las posiciones de token se calculan con el tokenizer del target.** En `collect_source_representations`, los inputs se tokenizan con `source_pipeline.load`, pero los índices salen de `TokenPosition`s creados con el pipeline target (`create_token_positions(pipeline)`). Si los dos modelos tienen tokenizers distintos (o distinto `max_length`/padding), las posiciones pueden no corresponder. Para modelos con distinto tokenizer harían falta posiciones construidas con el pipeline del source, y la librería no parece separar eso.
- **Las dimensiones tienen que coincidir** en cada unidad (por ejemplo, `hidden_size` para el residual stream). Las unidades se construyen con la config del target (`build_residual_stream_targets` lee `pipeline.model.config.hidden_size`).
- **Featurizers no triviales en cross-model.** La recolección usa `FeatureCollectIntervention`, que devuelve las activaciones **ya featurizadas**, y después `FeatureInterchangeIntervention` featuriza el "source" otra vez (`f_src, _ = self._featurizer(source)`). Con el featurizer identidad (esta notebook) da igual. Con un `SubspaceFeaturizer` (DAS) parecería que se featuriza dos veces, y las dimensiones no cerrarían (`k` vs `d`). No lo pude verificar porque depende de cómo pyvene pasa `source_representations` a la intervención y de lo que hace `_do_intervention_by_swap` en modo collect. Ojo antes de usar `source_pipeline` con DAS.
- **Con `source_pipeline = pipeline` hay dos `IntervenableModel` sobre el mismo `nn.Module`.** Funciona (lo demuestra el test), pero conviene saber que ambos comparten pesos.
- **Más lento**: la recolección separada agrega un forward por batch (20 s → 30 s en la ejecución guardada).

---

## Resumen de una línea por sección

- **Intro / validación**: si source = target, cross-model patching tiene que reproducir exactamente el patching normal.
- **Dataset**: `different_symbol` filtrado (58 pares), guardado como JSON.
- **Targets**: la grilla de la 03 (17 × 7 = 119, residual stream, vector completo).
- **Patching normal**: `run_interchange_score_heatmap(..., source_pipeline=None)` → `avg_score` 0.4704.
- **Cross-model**: `source_pipeline=pipeline` → recolecta con un `IntervenableModel` "collect" y pasa `source_representations` a pyvene → 0.4704.
- **Comparación**: las 119 IIA coinciden en `answer` y `answer_position`.
- **Conclusión**: implementación validada; lista para checkpoints, modelos compatibles y transferencia (con cuidado con tokenizers y featurizers).
