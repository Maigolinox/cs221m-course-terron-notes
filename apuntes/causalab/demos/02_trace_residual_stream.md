# 02 — Rastrear información en el residual stream con activation patching (`02_trace_residual_stream.ipynb`)

Notebook: `causalab/demos/onboarding_tutorial/02_trace_residual_stream.ipynb`.

Segunda notebook del onboarding. Hasta ahora todas las interchange interventions fueron sobre el modelo causal de alto nivel (ver [01_define_MCQA_task.md](01_define_MCQA_task.md)). Acá por primera vez se carga un LLM (`meta-llama/Llama-3.2-1B-Instruct`) y se hace la misma operación **sobre la red**: tomar el vector del residual stream de una corrida con el prompt contrafáctico y pegarlo en la corrida con el prompt original, en **cada** capa y **cada** posición de token, para **un solo** par de prompts. El resultado es un mapa (capa × token) de qué token genera el modelo después de cada patch. La pregunta: ¿por dónde "viaja" la información de la respuesta desde el token del símbolo hasta el último token?

Sigue en [03_localize_with_patching.md](03_localize_with_patching.md), que hace lo mismo pero promediando sobre datasets enteros y puntuando contra el modelo causal. Referencia de la parte neural de la librería: [../02_neural.md](../02_neural.md) y [../03_experiments.md](../03_experiments.md).

---

## Ideas clave / teoría

**Residual stream.** En un transformer decoder-only, cada posición de token tiene un vector `h_ℓ ∈ R^d` (acá `d = 2048`) que se va actualizando capa por capa: `h_{ℓ+1} = h_ℓ + Attn_ℓ(h_ℓ) + MLP_ℓ(...)`. Ese vector acumulado es el *residual stream*. Todo lo que una capa "escribe" para que lo lean capas posteriores pasa por ahí. Llama-3.2-1B tiene 16 capas.

**Activation patching = interchange intervention sobre la red.** Con un prompt base `b` y uno contrafáctico `s`:
1. corrés la red sobre `s` y guardás `h_ℓ^{(s)}[t']` (capa ℓ, posición t');
2. corrés la red sobre `b`, pero en la capa ℓ y posición t reemplazás `h_ℓ^{(b)}[t]` por `h_ℓ^{(s)}[t']`;
3. mirás qué genera la red.

Es exactamente `IntInv(b, s, X)` de la notebook 01, donde X ahora es "el vector del residual stream en (ℓ, t)" en vez de una variable del modelo causal. Si el patch hace que la red responda lo que respondería con `s`, ese vector contiene (al menos parte de) la información que diferencia `b` de `s`.

**Qué esperar en los bordes.** En la capa de embeddings (ℓ = -1), la única posición que diferencia `b` de `s` es el token del símbolo que cambió, así que solo ahí el patch cambia la salida. En la última capa, solo el último token importa (es donde se hace el unembedding para predecir el próximo token). Lo interesante es el medio: en qué capa la información "salta" del token del símbolo al último token.

**Por qué un solo par.** Esto es exploratorio: un mapa cualitativo para generar hipótesis. La 03 lo cuantifica.

---

## El código de la librería que usa esta notebook

### `LMPipeline` (`causalab/causalab/neural/pipeline.py`)

Envoltorio de un modelo de HuggingFace + tokenizer con métodos para cargar inputs, generar y generar con intervenciones.

- `__init__(model_or_name, *, max_new_tokens=3, max_length=None, ..., padding_side="left", **kwargs)`: guarda opciones y llama `_setup_model`. `device` y `dtype` llegan por `kwargs`.
- `_setup_model`: `AutoTokenizer.from_pretrained` y `AutoModelForCausalLM.from_pretrained(..., torch_dtype=dtype).to(device)`. Después fuerza `config._attn_implementation = "eager"` (atención explícita, para que los hooks de pyvene sobre componentes de atención funcionen) y `config.use_cache = False`. Pone `pad_token = eos_token` y `padding_side = "left"`.
- `load(input: list[CausalTrace], ...)`: toma `item["raw_input"]` de cada elemento, opcionalmente aplica chat template, y tokeniza. Si `max_length` está seteado (acá 32), **paddea siempre a 32** (`padding="max_length"`) y trunca. Con padding a la izquierda, el último token real queda siempre en la posición 31. Devuelve tensores en el device del modelo (`input_ids`, `attention_mask`, shape `(batch, 32)`).
- `generate(input)`: `load` + `model.generate(..., max_new_tokens=self.max_new_tokens, do_sample=False, return_dict_in_generate=True, output_scores=True)`. Greedy. Devuelve `{"scores", "sequences", "string"}`, donde `sequences` son los últimos `max_new_tokens` tokens y `string` es el texto decodificado (`dump`). Si hay un solo input, `string` es un `str`; si hay varios, una lista.
- `intervenable_generate(intervenable_model, base, sources, map, feature_indices, source_representations=None)`: llama `intervenable_model.generate(base, sources=..., unit_locations=map, subspaces=feature_indices, intervene_on_prompt=True, ...)` de pyvene y devuelve el mismo formato.

### `get_list_of_each_token` (`causalab/causalab/neural/token_position_builder.py`)

- Si recibe un string, lo envuelve en un `CausalTrace` mínimo con una sola variable `raw_input`.
- Tokeniza con `pipeline.load` (o sea, con padding a 32) y, para cada índice `i` que **no** sea pad, crea un `TokenPosition` cuyo indexer es la constante `[i]` y cuyo `id` es `f"tok_{i}_{token}"`.
- `TokenPosition` hereda de `ComponentIndexer` (`causalab/causalab/neural/model_units.py`): un callable `input → list[int]` con un id. Es la forma en que la librería dice "en qué posiciones intervenir" para cada input.

### `build_residual_stream_targets` (`causalab/causalab/experiments/interchange_targets.py`)

- Para cada `layer` en `layers` y cada `pos` en `token_positions` crea un `ResidualStream(layer, token_indices=pos, shape=(hidden_size,), target_output=True)`.
- Caso especial `layer == -1`: usa `layer=0` con `target_output=False`, es decir, el `block_input` de la capa 0 = **los embeddings**.
- `ResidualStream` (`causalab/causalab/neural/LM_units.py`) es un `AtomicModelUnit` con `component_type = "block_output"` (salida de la capa ℓ) o `"block_input"`, `unit = "pos"`, featurizer identidad por defecto, e id tipo `ResidualStream(Layer-10,block_output,Token-last_token)`.
- `mode="one_target_per_unit"` → un dict `{(layer, pos.id): InterchangeTarget([[unit]])}`. Un `InterchangeTarget` es una lista de grupos de unidades; cada grupo comparte un contrafáctico.

### `run_interchange_interventions` (`causalab/causalab/neural/pyvene_core/interchange.py`)

1. `prepare_intervenable_model(pipeline, target, "interchange")` (en `intervenable_model.py`): por cada unidad arma un config de pyvene `{"component": "block_output", "unit": "pos", "layer": ℓ, "group_key": i, "intervention_type": featurizer.get_interchange_intervention()}` y construye `pv.IntervenableModel(configs, model=pipeline.model)`. pyvene registra hooks en ese componente.
2. Por batch: `prepare_intervenable_inputs` calcula los índices de posición del base (`is_original=True`) y del contrafáctico (`is_original=False`) con cada `TokenPosition`, los `feature_indices` (acá `None`), tokeniza con `pipeline.load`, y arma `{"sources->base": (cf_indices, base_indices)}`.
3. `pipeline.intervenable_generate(...)`: pyvene corre el contrafáctico, captura la activación en esas posiciones, corre el base, y en el hook reemplaza la activación del base por la del contrafáctico.
4. La intervención concreta es `FeatureInterchangeIntervention` (`causalab/causalab/neural/featurizers.py`). Con el featurizer identidad y `subspaces=None` hace `f_out = f_src`, o sea **reemplaza el vector entero**.
5. Devuelve `{"sequences": [...], "string": [...]}` por batch.

### `run_residual_stream_tracing` (`causalab/causalab/experiments/jobs/residual_stream_tracing.py`)

- Arma un dataset de un solo ejemplo **con dicts planos**: `[{"input": {"raw_input": prompt}, "counterfactual_inputs": [{"raw_input": counterfactual_prompt}]}]`. No hacen falta `CausalTrace` porque `load` solo lee `raw_input` y los indexers son constantes.
- `layers=None` → `[-1] + list(range(num_layers))` = 17 "capas" (embeddings + 16).
- Construye los targets con `build_residual_stream_targets(..., mode="one_target_per_unit")` y corre `run_interchange_interventions` (batch 1, sin scores) para cada uno. Con 22 tokens × 17 capas son 374 intervenciones (coincide con la barra de progreso guardada).
- Guarda `results/intervention_results.json` (strings), `.safetensors` (secuencias), `metadata.json`, y si `generate_visualization=True` dibuja `residual_stream_heatmap.png` con `plot_residual_stream_intervention_heatmap` (`experiments/visualizations/string_heatmap.py`): cada celda muestra el token generado, coloreado por frecuencia (`color_by_frequency=True`).
- Acepta `source_pipeline` para cross-model patching (se usa en la 07).

---

## Recorrido por la notebook

### Celda 0 — imports

```python
from causalab.tasks.MCQA.causal_models import positional_causal_model
from causalab.tasks.MCQA.counterfactuals import sample_answerable_question
causal_model = positional_causal_model
import random
```

Lo mismo de la 01: el modelo causal y el sampler de preguntas.

### Celda 2 — cargar el LLM

```python
from causalab.neural.pipeline import LMPipeline
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "meta-llama/Llama-3.2-1B-Instruct"
pipeline = LMPipeline(
    model_name,
    max_new_tokens=1,
    device=device,
    dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    max_length=32,
)
```

- `max_new_tokens=1`: solo nos interesa el primer token generado (la letra).
- `dtype`: bfloat16 en GPU (memoria), float32 en CPU (bfloat16 en CPU es lento y no siempre soportado).
- `max_length=32`: todos los prompts se paddean (a la izquierda) a 32 tokens. Esto hace que las posiciones sean comparables entre ejemplos de longitud parecida y que el último token esté siempre en el índice 31.
- El modelo es gated en HuggingFace: `_setup_model` toma el token de `HF_TOKEN` o `HUGGING_FACE_HUB_TOKEN` del entorno.

```python
def checker(neural_output: dict[str, Any], causal_output: str) -> bool:
    return (
        causal_output.strip() in neural_output["string"].strip()
        or neural_output["string"].strip() in causal_output.strip()
    )
```

- Compara la salida de la red (`{"string": " Y"}`) con la del modelo causal (`" Y"`), ignorando espacios y aceptando inclusión en cualquier dirección.
- Ojo: si la red genera un string vacío, `"" in causal_output` es `True` y el checker da verdadero. Con `max_new_tokens=1` y greedy esto es raro, pero es una debilidad del criterio.
- Salida guardada: `DEVICE: cpu` (esa ejecución fue en CPU).

### Celda 4 — samplear un par que el modelo resuelve bien

```python
while True:
    original = sample_answerable_question()
    full_setting = original

    cf_inputs = {var: original[var] for var in causal_model.inputs}
    answer_symbol_key = f"symbol{full_setting['answer_position']}"
    new_symbols = list({"A", "B", "C"}.difference({full_setting[answer_symbol_key]}))
    cf_inputs[answer_symbol_key] = random.choice(new_symbols)
    counterfactual_setting = causal_model.new_trace(cf_inputs)
```

- `original`: un `CausalTrace` con todo calculado.
- Contrafáctico: mismo prompt pero se cambia **solo el símbolo de la respuesta correcta** por una letra de {A,B,C} distinta. Así `b` y `s` difieren en exactamente un token (salvo que la letra nueva tokenice distinto), y la respuesta correcta de `s` es la letra nueva. Ojo: no se excluye el *otro* símbolo, así que en principio podría salir un prompt con la misma letra dos veces.
- Este experimento solo mira "dónde está la letra de la respuesta"; no tiene nada que ver con `answer_position` (la posición no cambia).

```python
    original_pred = pipeline.generate([full_setting])
    counterfactual_pred = pipeline.generate([counterfactual_setting])
    if checker(original_pred, full_setting["raw_output"]):
        if checker(counterfactual_pred, counterfactual_setting["raw_output"]):
            break
```

- `generate` recibe una lista de traces y devuelve `{"string": ...}`. Se repite hasta que el modelo acierte **ambos** prompts (Llama-1B no es perfecto en esta tarea). Si el modelo no acierta el base, patchear no tiene una interpretación limpia.
- Ejecución guardada: "The bag is blue... E. red / Y. blue" → " Y"; contrafáctico con "C. blue" → " C". La advertencia "generation flags are not valid... temperature, top_p" sale porque la `generation_config` de Llama-Instruct trae esos parámetros y acá se usa greedy (`do_sample=False`); es inofensiva.

### Celda 6 — el tracing

```python
from causalab.experiments.jobs.residual_stream_tracing import run_residual_stream_tracing
from causalab.neural.token_position_builder import get_list_of_each_token

token_positions = get_list_of_each_token(full_setting["raw_input"], pipeline)
print(f"Found {len(token_positions)} token positions")
```

- Un `TokenPosition` por token no-pad del prompt base. Salida guardada: 22 posiciones (incluye `<|begin_of_text|>`).
- Como las posiciones son índices fijos calculados sobre el prompt **base**, se aplican igual al contrafáctico. Esto está bien porque ambos prompts tienen la misma longitud en tokens (solo cambia una letra).

```python
result = run_residual_stream_tracing(
    pipeline=pipeline,
    prompt=full_setting["raw_input"],
    counterfactual_prompt=counterfactual_setting["raw_input"],
    token_positions=token_positions,
    output_dir="./tracing_results",
    layers=None,
    generate_visualization=True,
    verbose=True,
)
intervention_results = result["intervention_results"]
```

- `layers=None` → embeddings + 16 capas.
- `generate_visualization=True` genera el heatmap (el comentario "We'll visualize separately below" quedó desactualizado: la figura la produce esta misma llamada).
- `verbose=True` pone el logger `causalab` en DEBUG y muestra la barra de progreso.
- `intervention_results` es un dict `{(layer, "tok_i_X"): {"sequences": [...], "string": [...]}}`: una entrada por batch (acá un solo batch de tamaño 1, así que `"string"` es una lista con un string).
- Salida guardada: 374 intervenciones en 1:43 en CPU, y una figura de 2940×850.

**Cómo leer el heatmap**: filas = capas (-1 a 15), columnas = tokens. Cada celda es el token que genera el modelo después de reemplazar ese vector del residual stream por el del contrafáctico. Donde aparece la letra del contrafáctico (" C"), ese vector contiene la información de la respuesta. Donde aparece la del base (" Y"), el patch no tuvo efecto.

### Celda 7 — Takeaways

Según el texto de la notebook, para la mayoría de los pares la señal se queda en el token del símbolo correcto hasta la capa ~11, y ahí salta al último token. A veces también aparece en el token del punto que sigue al símbolo (capas 3–10). La recomendación es re-ejecutar con otros pares, porque el patrón varía.

---

## Ejercicios

No hay celdas de ejercicio. La invitación es re-correr las celdas 4 y 6 con pares nuevos y mirar:
- en qué capa la letra del contrafáctico deja de aparecer en la columna del símbolo y empieza a aparecer en la columna del último token (`Answer:` → el token `:`);
- si el token `.` después del símbolo correcto "se enciende" en capas intermedias (eso motiva la posición `correct_symbol_period` de la 03).

---

## Ojo / errores comunes / dudas

- **El padding a 32 es a la izquierda**, así que los índices de token (`tok_i_...`) incluyen el offset del padding. No compares índices entre prompts de distinta longitud.
- **Las posiciones se calculan sobre el base y se reutilizan para el contrafáctico.** Funciona porque los prompts difieren en una letra. Para pares con longitudes distintas hacen falta posiciones dinámicas (eso es lo que resuelve `create_token_positions` en la 03).
- **Patchear el vector completo es grueso**: cambia *toda* la información de esa posición, no solo la variable que te interesa. La 04 lo refina con DAS.
- **Un solo par = anécdota.** El patrón exacto depende del par; no saques conclusiones cuantitativas de acá.
- **El checker acepta strings vacíos** como correctos (ver arriba).
- **Capa -1** no es una capa del modelo: es el `block_input` de la capa 0 (embeddings).

---

## Resumen de una línea por sección

- **Cargar el LLM**: `LMPipeline` envuelve Llama-3.2-1B; greedy, 1 token nuevo, prompts paddeados a la izquierda a 32.
- **Samplear un par**: base y contrafáctico difieren solo en la letra de la respuesta; se exige que el modelo acierte ambos.
- **Tracing**: `run_residual_stream_tracing` patchea el vector entero del residual stream en cada (capa, token) y guarda el token generado.
- **Takeaways**: la información de la letra vive en el token del símbolo en capas tempranas y pasa al último token alrededor de la capa 11.
