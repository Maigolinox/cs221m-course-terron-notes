# 03 — Localizar `answer` y `answer_position` con activation patching sobre datasets (`03_localize_with_patching.ipynb`)

Notebook: `causalab/demos/onboarding_tutorial/03_localize_with_patching.ipynb`.

Tercera notebook del onboarding. En la 02 ([02_trace_residual_stream.md](02_trace_residual_stream.md)) se hizo patching del residual stream para **un** par de prompts y se miró qué token salía. Acá se pasa a **datasets** de pares (los tres generadores de la 01: `different_symbol`, `same_symbol_different_position`, `random_counterfactual`), y en vez de mirar tokens se calcula un **puntaje**: la *interchange intervention accuracy* (IIA), que mide qué tan seguido la red intervenida responde lo mismo que el modelo causal intervenido sobre `answer` o sobre `answer_position`. La pregunta: ¿en qué (capa, posición) vive cada una de las dos variables del modelo causal, y cómo interpreto los heatmaps sabiendo qué confunde cada dataset?

Sigue en [04_train_DAS_and_DBM.md](04_train_DAS_and_DBM.md): en vez de patchear el vector completo, se entrena dónde y qué subespacio patchear. Referencia: [../03_experiments.md](../03_experiments.md).

---

## Ideas clave / teoría

**Interchange Intervention Accuracy (IIA).** Para un dataset de pares `{(b_i, s_i)}`, una ubicación de la red `L` (acá un vector del residual stream en una capa y posición) y una variable del modelo causal `X`:

  `IIA(L, X) = (1/N) · Σ_i  1[ N_{L ← L(s_i)}(b_i)  ≈  M_{X ← X(s_i)}(b_i) ]`

- `N_{L ← L(s_i)}(b_i)`: salida de la red con input `b_i`, con la activación en `L` reemplazada por la que tiene con input `s_i`.
- `M_{X ← X(s_i)}(b_i)`: `raw_output` del modelo causal con input `b_i` y `X` fijada a su valor con `s_i` (`run_interchange`).
- `≈`: la `metric` (inclusión de strings).

Si `IIA(L, X) = 1`, la ubicación `L` se comporta exactamente como la variable `X` bajo intervenciones en ese dataset: es evidencia de que la red **representa** `X` en `L`. La idea central de la abstracción causal es que el modelo causal es fiel a la red en la medida en que interchange interventions equivalentes tienen efectos equivalentes.

**El confounding vuelve en forma de "falsos positivos".** Si en un dataset la interchange sobre `X` da la misma salida que no intervenir (proporción 0 en `can_distinguish_with_dataset(..., [X], None)`), entonces **cualquier** ubicación causalmente irrelevante (donde el patch no cambia nada) obtiene IIA = 1 para `X`. El heatmap de `X` se ve "todo encendido" salvo justamente donde está la información que cambia la salida. Por eso cada heatmap hay que leerlo junto con la tabla de `can_distinguish_with_dataset`.

**Posiciones dinámicas.** Con pares aleatorios, el símbolo correcto puede estar en la posición 0 o 1, y la longitud en tokens del prompt varía (los objetos y colores pueden ocupar distinta cantidad de tokens). Por eso las posiciones se definen **por rol** ("el símbolo correcto", "el punto después del símbolo correcto") y se recalculan por input.

**Filtrado.** Solo se usan pares donde el LM responde bien el base **y** el contrafáctico. No es obligatorio para abstracción causal (se podría modelar también cuándo falla), pero simplifica: Llama-1B falla seguido en esta tarea.

---

## El código de la librería que usa esta notebook

### `filter_dataset` (`causalab/causalab/experiments/filter.py`)

- `pipeline.compute_outputs(dataset, batch_size)` (en `pipeline.py`) genera salidas para todos los inputs base y todos los contrafácticos (aplanados), en batches.
- Para cada ejemplo: si `metric(base_output, input["raw_output"])` falla, se descarta; si `validate_counterfactuals` (default `True`) y algún contrafáctico falla `metric(cf_output, cf["raw_output"])`, se descarta.
- Devuelve la sublista de ejemplos que pasaron.

### `create_token_positions` (`causalab/causalab/tasks/MCQA/token_positions.py`)

Define posiciones con un mini-lenguaje declarativo y las convierte en `TokenPosition` vía `build_token_position_factories` (`causalab/causalab/neural/token_position_builder.py`):

- `"last_token"`: `{"type": "index", "position": -1}` → último token (con padding a la izquierda, siempre el índice 31).
- `"symbol0"`, `"symbol1"`: `{"type": "variable", "name": "symbol0"}` → los tokens donde quedó el valor de esa variable del template.
- `"symbol0_period"`, `"symbol1_period"`: `{"type": "index", "position": +1, "relative_to": {"variable": "symbol0"}}` → el token siguiente al último token del símbolo (el `.`).
- `"correct_symbol"`: **dinámica**, un `lambda setting: {"type": "variable", "name": find_correct_symbol(setting)}`, donde `find_correct_symbol` devuelve `"symbol0"` o `"symbol1"` según cuál opción coincide con el color. `"correct_symbol_period"`: igual pero +1.

Cómo se encuentran los tokens de una variable: `Template.get_variable_positions` rellena el template llevando la cuenta de en qué rango de **caracteres** quedó cada variable, tokeniza con `pipeline.load(..., return_offsets_mapping=True)` y mapea caracteres → tokens con los offsets. El resultado se cachea por `(id(tokenizer), texto)`. Para una posición dinámica, `_build_dynamic_factory` evalúa la lambda **por input** y construye la posición concreta en el momento.

`TokenPosition.highlight_selected_token(input)` decodifica token por token y envuelve los seleccionados en `**...**` (lo que se ve en la salida de la celda 5).

### `run_interchange_score_heatmap` (`causalab/causalab/experiments/jobs/interchange_score_grid.py`)

1. Detecta el tipo de componente mirando el id de la primera unidad (`"ResidualStream"`, `"AttentionHead"` o `"MLP"`) y extrae las dimensiones de la grilla de las claves `(layer, pos_id)`.
2. `load_counterfactual_examples(dataset_path, causal_model)` (`causalab/causalab/causal/causal_utils.py`): lee el JSON y reconstruye cada trace con `causal_model.new_trace(dict)`.
3. Para cada target: `run_interchange_interventions(pipeline, dataset, target, batch_size, output_scores=False, source_pipeline=...)` (explicado en [02_trace_residual_stream.md](02_trace_residual_stream.md)).
4. `causal_score_intervention_outputs(raw_results, dataset, causal_model, target_variable_groups, metric)` (`causalab/causalab/experiments/metric.py`): para cada grupo de variables crea una métrica con `make_causal_metric` y llama `score_intervention_outputs`, que:
   - etiqueta el dataset con `causal_model.label_counterfactual_data(dataset, list(var_group))`: por cada ejemplo hace `run_interchange(input, {var: cf})` y guarda `label = raw_output` intervenido;
   - para cada target compara el string generado por la red en cada ejemplo con su label usando `metric` → 1.0/0.0, y promedia. **Ese promedio es la IIA.**
5. Dibuja un heatmap por grupo de variables con `plot_score_heatmap` en `output_dir/heatmaps/{dataset_name}_{var}.png`, donde `dataset_name = Path(dataset_path).stem`.
6. Devuelve `{"scores": {var_group: {(layer, pos_id): IIA}}, "component_type", "metadata", "output_paths"}`.

### `save_counterfactual_examples` (`causal_utils.py`)

Serializa cada ejemplo como `{"input": trace.to_dict(), "counterfactual_inputs": [...]}` y hace `json.dump` a la ruta (que tiene que ser un **archivo**). Los jobs de la librería leen de disco, por eso las notebooks guardan los datasets filtrados antes de llamarlos.

---

## Recorrido por la notebook

### Celda 0 — setup

```python
from causalab.tasks.MCQA.counterfactuals import (
    different_symbol, same_symbol_different_position, random_counterfactual,
)
from causalab.causal.counterfactual_dataset import CounterfactualExample
from causalab.neural.pipeline import LMPipeline
...
pipeline = LMPipeline(model_name, max_new_tokens=1, device=device,
                      dtype=torch.bfloat16 if device == "cuda" else torch.float32,
                      max_length=32)

def metric(neural_output: dict[str, Any], causal_output: str) -> bool:
    return (
        causal_output in neural_output["string"]
        or neural_output["string"] in causal_output
    )
```

- Mismo pipeline que la 02.
- `metric`: igual que el `checker` de la 02 pero **sin** `strip()`. Funciona porque tanto la red como el modelo causal producen `" L"` con espacio. Sigue aceptando un string vacío como correcto.

### Celda 2 — generar y filtrar los tres datasets

```python
from causalab.experiments.filter import filter_dataset
size = 64
counterfactual_generators = {
    "different_symbol": different_symbol,
    "same_symbol_different_position": same_symbol_different_position,
    "random_counterfactual": random_counterfactual,
}
counterfactual_datasets = {}
for name, sampler in counterfactual_generators.items():
    counterfactual_datasets[name] = [sampler() for _ in range(size)]
```

- 64 pares por generador; cada elemento es un `CounterfactualExample` (`{"input": trace, "counterfactual_inputs": [trace]}`).

```python
filtered_datasets = {}
for name, dataset in counterfactual_datasets.items():
    filtered_datasets[name] = filter_dataset(
        dataset=dataset, pipeline=pipeline, causal_model=causal_model,
        metric=metric, batch_size=128,
    )
different_symbol_pairs = filtered_datasets["different_symbol"]
same_symbol_diff_position_pairs = filtered_datasets["same_symbol_different_position"]
random_pairs = filtered_datasets["random_counterfactual"]
```

- `batch_size=128` > 64, así que cada dataset se procesa en un solo batch.
- Salida guardada: 174 ejemplos en total tras filtrar. Por los `count` que se imprimen más abajo, son 60 (`different_symbol`), 53 (`same_symbol_different_position`) y 61 (`random`).

### Celda 5 — posiciones de token

```python
from causalab.tasks.MCQA.token_positions import create_token_positions
all_token_positions = create_token_positions(pipeline)
ordered_position_names = [
    "correct_symbol", "correct_symbol_period",
    "symbol0", "symbol0_period",
    "symbol1", "symbol1_period",
    "last_token",
]
token_positions = [all_token_positions[name] for name in ordered_position_names]

for token_position in token_positions:
    example = next(iter(counterfactual_datasets.values()))[0]["input"]
    print(f"Selected token: {token_position.highlight_selected_token(example)}")
```

- `create_token_positions` devuelve un dict nombre → `TokenPosition`. El orden de la lista define el orden de las columnas del heatmap (con `last_token` al final).
- El ejemplo usado para resaltar es el primero del dataset `different_symbol` **sin filtrar**.
- Salida guardada: en "S. orange / L. pink", con color pink, `correct_symbol` resalta `L` (que es `symbol1`), `correct_symbol_period` el `.` siguiente, y `last_token` el `:` de `Answer:`. Notá que `correct_symbol` coincide con `symbol1` en este ejemplo, pero en otro coincidiría con `symbol0`.

### Celda 7 — la función de patching (definición)

```python
from causalab.experiments.jobs.interchange_score_grid import run_interchange_score_heatmap
from causalab.experiments.interchange_targets import build_residual_stream_targets
from causalab.causal.causal_utils import save_counterfactual_examples

results_dir = "vector_patching_results"
temp_datasets_dir = "./temp_filtered_datasets"
num_layers = pipeline.model.config.num_hidden_layers
layers = [-1] + list(range(num_layers))
results = {}
```

- `layers`: 17 filas (embeddings + capas 0..15).

```python
def patch_dataset(dataset_name: str) -> None:
    dataset = filtered_datasets[dataset_name]
    dataset_path = os.path.join(temp_datasets_dir, f"{dataset_name}.json")
    save_counterfactual_examples(dataset, dataset_path)
    dataset_output_dir = os.path.join(results_dir, dataset_name)

    targets = build_residual_stream_targets(
        pipeline=pipeline, layers=layers, token_positions=token_positions,
        mode="one_target_per_unit",
    )
    result = run_interchange_score_heatmap(
        causal_model=causal_model, interchange_targets=targets,
        dataset_path=dataset_path, pipeline=pipeline, batch_size=128,
        target_variable_groups=[("answer",), ("answer_position",)],
        output_dir=dataset_output_dir, metric=metric, verbose=True,
    )
    results[dataset_name] = {
        str(("answer",)): result["scores"][("answer",)],
        str(("answer_position",)): result["scores"][("answer_position",)],
    }
```

- Guarda el dataset filtrado en JSON (los jobs leen de disco).
- `targets`: 17 × 7 = 119 targets de una sola unidad, claves `(layer, pos_id)`. Cada unidad es `ResidualStream(layer, pos)` con featurizer identidad: se patchea **el vector entero** en esa posición.
- `target_variable_groups=[("answer",), ("answer_position",)]`: se corren las intervenciones **una vez** y se puntúan **dos veces**, una contra cada variable. Los dos heatmaps salen de las mismas salidas de la red; lo que cambia es el label del modelo causal.
- Las posiciones se evalúan por input y por separado para base y contrafáctico. Por ejemplo, con `correct_symbol`, si en el base el correcto es `symbol1` y en el contrafáctico `symbol0`, se copia el vector del `symbol0` del contrafáctico al `symbol1` del base.
- Ojo: la celda solo **define** la función e imprime "Interventions complete and scores computed!" sin haber corrido nada. Las corridas son en las celdas 13, 19 y 25.

### Celdas 9–13 — `different_symbol`: localizar `answer`

```python
print(different_symbol_pairs[0])
print(different_symbol_pairs[0]["input"]["raw_input"])
print(different_symbol_pairs[0]["counterfactual_inputs"][0]["raw_input"])
```

- Muestra un par: "S. orange / L. pink" vs "Q. orange / P. pink". Mismas opciones y misma posición, letras nuevas.

```python
causal_model.can_distinguish_with_dataset(different_symbol_pairs, ["answer"], None)            # 1.0
causal_model.can_distinguish_with_dataset(different_symbol_pairs, ["answer"], ["answer_position"])  # 1.0
causal_model.can_distinguish_with_dataset(different_symbol_pairs, ["answer_position"], None)   # 0.0
patch_dataset("different_symbol")
```

- Tabla (salida guardada, 60 ejemplos): `answer` se distingue de no intervenir y de `answer_position` siempre; `answer_position` **nunca** se distingue de no intervenir.
- Interpretación según la notebook:
  - **Heatmap de `answer`**: alto en la capa de embeddings sobre `correct_symbol` (ese token determina la salida). Un par de capas después, la señal pasa en parte al `.` siguiente. En `symbol0`/`symbol1` la señal es parcial porque el correcto está a veces en uno y a veces en el otro, con una asimetría (más débil sobre el primer símbolo). En capas tardías aparece en `last_token`.
  - **Heatmap de `answer_position`**: como este dataset confunde `answer_position` con no intervenir, toda ubicación donde el patch no cambia la salida obtiene IIA = 1. El heatmap queda casi todo encendido, y **apagado justo donde está la información de la respuesta**. No sirve para localizar `answer_position`; es un "negativo" del heatmap de `answer`.
- Salida guardada: 119 targets en 9:20 (≈ 4.7 s/target) y dos figuras.

### Celdas 15–19 — `same_symbol_different_position`: localizar `answer_position`

```python
print(same_symbol_diff_position_pairs[0]["input"]["raw_input"])
print(same_symbol_diff_position_pairs[0]["counterfactual_inputs"][0]["raw_input"])
```

- Par guardado: "M. green / D. pink" vs "D. pink / M. green". Misma respuesta (M), en la otra posición.

```python
causal_model.can_distinguish_with_dataset(same_symbol_diff_position_pairs, ["answer_position"], None)      # 1.0
causal_model.can_distinguish_with_dataset(same_symbol_diff_position_pairs, ["answer"], ["answer_position"])  # 1.0
causal_model.can_distinguish_with_dataset(same_symbol_diff_position_pairs, ["answer"], None)               # 0.0
patch_dataset("same_symbol_different_position")
```

- Por qué: `IntInv(b, s, answer_position)` = letra del base en la posición nueva = la **otra** letra (D) → cambia la salida. `IntInv(b, s, answer)` = letra del contrafáctico = M = la misma que el base → no cambia nada. Así que este dataset distingue `answer_position` y es ciego a `answer`.
- Interpretación según la notebook:
  - **Heatmap de `answer_position`**: alrededor de 50% sobre los tokens de los símbolos. La explicación de la notebook: patchear el vector de un símbolo deja la misma letra repetida dos veces en contexto, y la mitad de las veces la letra que "sobra" es la que el modelo causal espera. Fuera de eso, solo **una** ubicación tiene señal notable: `last_token` en la capa 11.
  - **Heatmap de `answer`**: confundido con no intervenir → todo encendido salvo la ubicación de `answer_position` (la capa 11 en el último token aparece apagada).

### Celdas 21–25 — `random_counterfactual`

```python
causal_model.can_distinguish_with_dataset(random_pairs, ["answer_position"], None)       # 0.508
causal_model.can_distinguish_with_dataset(random_pairs, ["answer"], ["answer_position"])  # 0.967
causal_model.can_distinguish_with_dataset(random_pairs, ["answer"], None)                 # 0.951
patch_dataset("random_counterfactual")
```

- Pares totalmente independientes (otra pregunta, otras opciones, otros símbolos).
- Interpretación según la notebook:
  - `answer_position`: parecido a `same_symbol_different_position`, el mejor lugar sigue siendo `last_token` en la capa 11. Pero la señal sobre los tokens de símbolo es más baja, porque ya no se trata de las mismas letras intercambiadas sino de letras nuevas.
  - `answer`: parecido a `different_symbol`, pero más difuso.
- Ojo: acá aproximadamente la mitad de los pares tiene la respuesta en la misma posición, y en esos pares `answer_position` es indistinguible de no intervenir. Eso infla el piso del heatmap de `answer_position`.

### Celda 26 — Takeaway

Patchear vectores enteros es grueso pero da una impresión rápida, cuantitativa y cualitativa. Ningún dataset solo alcanza: hay que armar varios datasets diagnósticos y cuantificar qué distinguen.

---

## Ejercicios

No hay celdas de ejercicio. Lo que hay que saber hacer con estos resultados:
- Para cada heatmap, antes de mirarlo, predecir con la tabla de `can_distinguish_with_dataset` si va a estar "encendido por defecto" (confundido con no intervenir) o "apagado por defecto".
- Ubicar la **capa 11 en `last_token`** como candidata para `answer_position`. En la 04 se refina con DAS y aparece una estructura de tres etapas (capas 10, 11, 12–13).

---

## Ojo / errores comunes / dudas

- **IIA alta no siempre es buena noticia**: si el dataset confunde `X` con no intervenir, IIA = 1 en todos los lugares irrelevantes. Siempre mirar `can_distinguish_with_dataset(..., [X], None)`.
- **Las dos variables se puntúan con las mismas corridas de la red**; lo único que cambia es el label del modelo causal.
- **`patch_dataset` se define en la celda 7 pero se ejecuta en 13, 19 y 25.** El mensaje "Interventions complete" de la celda 7 es engañoso.
- **Las rutas de las figuras**: `run_interchange_score_heatmap` las guarda en `vector_patching_results/<dataset>/heatmaps/<dataset>_<var>.png`. La 04 intenta mostrar archivos con otro nombre (`heatmap_dataset_..._MCQA_vector_patching_variables_...png`, de una versión vieja), y por eso esas celdas de la 04 no muestran nada.
- **Las salidas de las celdas 9 y 15 incluyen un `print(dict)`** que en el código actual de la celda 15 no está (restos de una versión anterior de la notebook). No afecta nada.
- **Costo**: cada dataset son 119 targets × 1 pasada por batch; en la ejecución guardada, ≈ 9 minutos por dataset.
- **Posiciones dinámicas y caché**: `Template.get_variable_positions` cachea por `(id(tokenizer), texto)`. Si cambiás el tokenizer o el template, se recalcula. No debería dar problemas, pero conviene saber que existe.

---

## Resumen de una línea por sección

- **Generar/filtrar**: tres datasets de 64 pares; se quedan solo los pares donde Llama acierta base y contrafáctico (60/53/61).
- **Token positions**: posiciones por rol (`correct_symbol`, `symbolX`, sus puntos, `last_token`) que se recalculan por input vía offsets de caracteres.
- **Patching con datasets**: 119 targets (17 capas × 7 posiciones), vector completo, puntuados contra `answer` y `answer_position` (IIA).
- **different_symbol**: localiza `answer` (embeddings del símbolo correcto → punto → último token); el heatmap de `answer_position` está confundido.
- **same_symbol_different_position**: localiza `answer_position` en `last_token`, capa 11; el heatmap de `answer` está confundido.
- **random**: resultados parecidos pero más difusos; la mitad de los pares no informa sobre `answer_position`.
- **Takeaway**: hacen falta varios datasets diagnósticos y leer cada heatmap con su tabla de confounding.
