# IOI demo: Indirect Object Identification con causalab

> **Actualización (noviembre 2026):** la notebook se reescribió con la API actual y corre completa en GPU:
> - filtrado con `filter_dataset`;
> - parcheo de cabezas con `build_attention_head_targets` + `run_interchange_interventions`;
> - diferencia de logits calculada con los logits completos (y el espacio inicial correcto en los dos nombres);
> - DBM con `train_DBM_binary_heatmaps`;
> - los contrafactuales conservan plantilla, lugar y objeto del base;
> - el coeficiente de regularización de DBM pasó de 0.1 a 10 para que la máscara seleccione pocas cabezas con el mismo IIA.
>
> Resultados de la ejecución: las cabezas con más efecto en la diferencia de logits son 8.6, 8.10, 7.9 y 7.3, las *S-inhibition heads* de Wang et al. (2023).

Notebook: `causalab/causalab/tasks/IOI/ioi_demo.ipynb`

**IOI** (Indirect Object Identification) es la tarea clásica de interpretabilidad de GPT-2 small: en *"As Christopher and Kevin left the restaurant, Kevin gave a ring to"* el modelo debería continuar con *" Christopher"*, el **objeto indirecto** (IO), es decir, el nombre que **no** se repite. Al nombre repetido se le llama sujeto (S). Esta notebook define un modelo causal de alto nivel de la tarea con dos variables intermedias, **`output_position`** (en qué posición estaba el IO: primero o segundo nombre) y **`output_token`** (qué nombre es el IO), genera varios datasets contrafactuales, filtra los ejemplos que GPT-2 resuelve bien, hace **activation patching por cabeza de atención** en el último token y entrena máscaras **DBM** (Desiderata-Based Masking) para encontrar qué cabezas cargan esas variables (las *name mover heads* y, según el título de la sección, *negative name mover heads*). La pregunta: ¿qué cabezas de atención de GPT-2, leídas en el último token, implementan las variables del modelo causal de IOI?

> La tarea está marcada **DEPRECATED** y la notebook dice usar "the new refactored API", pero varias clases que usa (`FilterExperiment`, `PatchAttentionHeads`, `compute_custom_scores`, `CounterfactualDataset.from_sampler`) **ya no existen** en el `causalab/` del repo (ver "Ojo"). Explico lo que se puede verificar en el código actual y lo que se infiere de las salidas guardadas. Código de la tarea: `causalab/causalab/tasks/IOI/` (`causal_models.py`, `counterfactuals.py`, `token_positions.py`, y los JSON de nombres, objetos, lugares y plantillas). Apuntes generales: `../01_causal.md` y siguientes.

---

## Ideas clave / teoría

### El modelo causal de IOI

Variables (`tasks/IOI/causal_models.py`, `create_ioi_causal_model`):

| Variable | Padres | Mecanismo |
|---|---|---|
| `template`, `name_A`, `name_B`, `name_C`, `place`, `object` | ninguno (entradas) | `random.choice` de los JSON |
| `raw_input` | las 6 entradas | `fill_template`: reemplaza `{name_A}`, `{name_B}`, `{name_C}`, `{place}`, `{object}` en la plantilla |
| `output_position` | A, B, C | 1 si C == A (la respuesta es B, segunda posición); 0 si C == B (la respuesta es A); `None` si C no es ninguno |
| `output_token` | A, B, C | el nombre que no es C (B si C == A, A si C == B); `None` si no hay match |
| `logits` | A, B, `output_token`, `output_position` | ver fórmula abajo |
| `raw_output` | A, B, `logits` | `{"string": " " + nombre con mayor logit, "logits": logits}` |

`name_A` y `name_B` son los dos nombres en el orden en que aparecen; `name_C` es el que aparece de nuevo antes de "gave a ... to". En un input bien formado C es igual a uno de los dos.

La fórmula de `get_logits` (líneas 95-117), con `bias = 2`, `token_coeff = -1`, `pos_coeff = -1` por defecto:

$$\text{logit}_A = \text{bias} - \text{token\_coeff}\cdot[\text{output\_token} = A] - \text{pos\_coeff}\cdot[\text{output\_position} = 0]$$

$$\text{logit}_B = \text{bias} - \text{token\_coeff}\cdot[\text{output\_token} = B] - \text{pos\_coeff}\cdot[\text{output\_position} = 1]$$

Con los valores por defecto: cada nombre parte de 2, gana +1 si es el `output_token` y +1 si está en la `output_position`. En un input normal las dos señales coinciden y el IO tiene 4 y el S tiene 2 (eso se ve en la salida guardada: `{'Christopher': 4.0, 'Kevin': 2.0}`). `raw_output` elige el de logit mayor; si empatan, elige A (`get_raw_output`, 119-139).

Por qué dos variables y no una: son dos hipótesis de cómo se "señala" la respuesta. Una señal de **token** ("el IO es Christopher", contenido) y una de **posición** ("el IO es el primer nombre", posición relativa). Con intervenciones se pueden separar: si intervengo solo una con un valor que contradice a la otra, los logits empatan (3 y 3); si intervengo la posición con un valor y el token con un nombre que no está en el prompt, gana la posición.

### Activation patching por cabeza de atención

Para cada cabeza (layer, head) de GPT-2 small (12 × 12 = 144), se corre el modelo en la base, pero la salida de esa cabeza **en el último token** se reemplaza por la que tiene en el contrafactual. Si la diferencia de logits (IO − S) cae mucho, la cabeza es importante para la tarea en esa posición. En causalab la unidad es `AttentionHead(layer, head, token_indices)` (`causalab/causalab/neural/LM_units.py`), con `component_type = "head_attention_value_output"`: la salida de la cabeza antes de la proyección de salida.

### DBM (Desiderata-Based Masking)

En vez de probar cabezas de una en una, se aprende una **máscara** sobre todas las cabezas a la vez. Con `tie_masks=True` cada cabeza tiene **un solo** parámetro escalar $m_h$ (`causalab/causalab/neural/featurizers.py`, `FeatureMaskIntervention`). En entrenamiento:

$$g_h = \sigma(m_h / T), \qquad \text{salida}_h = (1 - g_h)\,\text{base}_h + g_h\,\text{cf}_h$$

y en evaluación la compuerta se binariza: $g_h = [\sigma(m_h) > 0.5]$. La pérdida combina que la salida de la red intervenida coincida con la **etiqueta contrafactual** (lo que predice el modelo causal al intervenir las variables objetivo) más un término de esparsidad $\lambda \sum_h g_h$ (norma L1 de las compuertas; `regularization_coefficient` es $\lambda$). $T$ es una temperatura que se va bajando para que las compuertas se acerquen a 0/1. Resultado: un conjunto pequeño de cabezas que, al parcharlas juntas, reproducen el efecto de intervenir la variable. La métrica es la **IIA** (interchange intervention accuracy): fracción de pares donde la red intervenida da la etiqueta contrafactual.

---

## Recorrido por la notebook

### Imports

```python
from causalab.tasks.IOI.causal_models import create_ioi_causal_model
from causalab.tasks.IOI.counterfactuals import (
    sample_well_formed_input, swap_names, flip_name_C, flip_name_C_sample_new_IO,
    swap_names_and_flip_C, random_counterfactual, random_ABC)
from causalab.causal.causal_viz import print_structure
```

### Cargar el modelo causal

```python
causal_model = create_ioi_causal_model()
print_structure(causal_model)
```

- `create_ioi_causal_model(bias=2.0, token_coeff=-1.0, pos_coeff=-1.0)` arma variables, valores, padres y mecanismos como en la tabla. Los datos se cargan al importar el módulo: `NAMES` (99 nombres, `names.json`), `OBJECTS` (8: ring, kiss, bone, basketball, computer, necklace, drink, snack), `PLACES` (8), `TEMPLATES` (`templates.json`, 70 plantillas, la mayoría en dos versiones "a"/"an").
- Al final del módulo se crea `positional_causal_model = create_ioi_causal_model()`, una instancia por defecto que usan los generadores de contrafactuales.
- `print_structure(model)` (`causalab/causal/causal_viz.py` 644-683) dibuja el grafo con NetworkX: arcos padre → hijo, posiciones de `model.print_pos`. La salida guardada es solo la figura.

### Samplear y correr

```python
example = sample_well_formed_input()
print(example['name_A'], example['name_B'], example['name_C'], example['place'], example['object'])
print(example['raw_input'])
```

`sample_well_formed_input()` (`counterfactuals.py` 15-28):
1. `input_sample = model.sample_input()`: samplea todas las entradas al azar y corre el modelo (`CausalModel.sample_input` en `causal_model.py`).
2. Mientras `name_A == name_B`, re-samplea `name_B`.
3. `name_C = random.choice([name_A, name_B])`: fuerza un input bien formado.

Con la `CausalTrace` actual, asignar `input_sample["name_B"] = ...` llama a `intervene`, que recalcula `raw_input` y el resto. Salida guardada: Christopher / Kevin / Kevin / restaurant / ring → `"As Christopher and Kevin left the restaurant, Kevin gave a ring to"`.

```python
full_setting = causal_model.new_trace(example)
print(full_setting['output_position'], full_setting['output_token'], full_setting['logits'], full_setting['raw_output'])
```

C == B, así que `output_position = 0` (la respuesta está en la posición de A), `output_token = Christopher`, logits `{'Christopher': 4.0, 'Kevin': 2.0}`, `raw_output = {'string': ' Christopher', ...}`. Nota que `raw_output` es un **diccionario**, no un string; por eso más abajo el checker lee `causal_output["string"]`.

### Datasets contrafactuales

```python
size = 128
datasets = {
    "swap_names": CounterfactualDataset.from_sampler(size, swap_names, id="swap_names"),
    "flip_name_C": ..., "flip_name_C_sample_new_IO": ..., "swap_names_and_flip_C": ...,
    "random_counterfactual": ..., "random_ABC": ...,
}
```

Cada generador (`counterfactuals.py`) parte de una base bien formada (A, B, C) y crea el contrafactual con `model.new_trace({...nombres...})`. Qué hace cada uno y qué valores toman las variables intermedias en base (b) y contrafactual (c). Supongo C = B en la base, o sea `output_position(b) = 0`, `output_token(b) = A`:

| Generador | Contrafactual | `output_position(c)` | `output_token(c)` | Separa |
|---|---|---|---|---|
| `swap_names` (31-45) | (B, A, C): se invierte el orden de los nombres, mismo C | 1 | A | posición cambia, token igual |
| `flip_name_C` (48-67) | (A, B, A): C pasa a ser el otro nombre | 1 | B | ambas cambian |
| `flip_name_C_sample_new_IO` (70-93) | (A, B', A) con B' nuevo (si C == A en la base: (A', B, B)) | 1 | B' (un nombre que **no está** en la base) | posición cambia, token "inexistente" en la base |
| `swap_names_and_flip_C` (96-115) | (B, A, A): se invierten nombres y C pasa al otro | 0 | B | token cambia, posición igual |
| `random_counterfactual` (118-125) | otra base bien formada independiente | aleatoria | otro nombre | todo cambia |
| `random_ABC` (140-162) | tres nombres distintos al azar | `None` | `None` | contrafactual sin respuesta IOI definida |

Con esto se puede razonar cada experimento. Ejemplo de la salida guardada, `flip_name_C`: base `"Before Crystal and Erica left the house, Erica decided to give an ring to"` → ` Crystal` (`{'Crystal': 4.0, 'Erica': 2.0}`); contrafactual con C = Crystal → ` Erica`.

`CounterfactualDataset.from_sampler(n, f, id=...)` llamaba `n` veces al generador y guardaba los pares en un dataset (con `.dataset` como el dataset subyacente). En la API actual un dataset es solo una `list[CounterfactualExample]` (`causalab/causal/counterfactual_dataset.py`).

### Cargar GPT-2 small

```python
gpt_config = GPT2Config.from_pretrained(model_path)
pipeline = LMPipeline(model_path, max_new_tokens=1, device=device, max_length=128,
                      config=gpt_config, position_ids=True)

def checker(neural_output, causal_output):
    predicted_token = causal_output["string"].strip()
    return predicted_token in neural_output["string"] or neural_output["string"] in predicted_token
```

- `max_new_tokens=1`: solo se genera el siguiente token (el nombre).
- `max_length=128`: padding a la izquierda hasta 128 (`pipeline.py`: `padding_side="left"`, `pad_token = eos_token`).
- `position_ids=True`: en `LMPipeline.load`, se calculan `position_ids` con `model.prepare_inputs_for_generation(input_ids, attention_mask)`. Con padding a la izquierda esto es importante para GPT-2: sin `position_ids` explícitos, el primer token real tendría posición ~100 en vez de 0 y el modelo se comportaría distinto.
- `checker`: compara el string generado por la red con el del modelo causal, en ambas direcciones de "contiene". Es laxo: si el nombre se tokeniza en varias piezas y la red solo genera la primera (`" Chris"`), `" Chris" in "Christopher"` es falso por el espacio inicial y `"Christopher" in " Chris"` también; además, si la red generara un string vacío, `"" in predicted_token` daría `True`.

### Filtrar por desempeño de GPT-2

```python
filter_exp = FilterExperiment(pipeline, causal_model, checker)
datasets_for_filtering = {k: v for k, v in datasets.items() if k != "random_ABC"}
filtered_datasets = filter_exp.filter(datasets_for_filtering, verbose=True, batch_size=64)
filtered_datasets["random_ABC"] = datasets["random_ABC"]
```

- La idea (equivalente actual: `causalab/causalab/experiments/filter.py`, `filter_dataset`): correr la red en la base y en cada contrafactual y quedarse solo con los pares donde **ambas** salidas coinciden con lo que dice el modelo causal (según `checker`). No tiene sentido medir si una intervención "rompe" un comportamiento que la red no tenía.
- `random_ABC` no se filtra porque sus contrafactuales no tienen una respuesta correcta definida (el modelo causal daría el empate → A, que no es "correcto" en ningún sentido).
- Salida guardada: se conserva entre 60.9% (`swap_names`, 78/128) y 74.2% (`swap_names_and_flip_C`, 95/128); en total 431 de 640 (67.3%). Es decir, GPT-2 small con `max_new_tokens=1` resuelve bien ~2/3 de estos prompts (base y contrafactual a la vez).

### Patching por cabeza de atención (último token)

```python
config = {"id": "ioi_attention_heads_demo", "top_k_logits": 100, "batch_size": 64}
num_heads = pipeline.get_num_attention_heads()   # 12
num_layers = pipeline.get_num_layers()           # 12
last_token = TokenPosition(lambda x: get_last_token_index(x, pipeline), pipeline, id="last_token")
each_head = [[(layer, head)] for layer in range(0, num_layers) for head in range(num_heads)]
experiment = PatchAttentionHeads(pipeline, each_head, last_token, config=config)
raw_results = experiment.perform_interventions({"random_ABC": filtered_datasets["random_ABC"]},
                                               verbose=True, include_actual_outputs=True)
```

- `TokenPosition(indexer, pipeline, id)` (`token_position_builder.py`): `get_last_token_index` tokeniza el input y devuelve `[len(ids) - 1]`, que con padding izquierdo es el último token real ("to"/"for"). Es la posición donde el modelo decide el nombre.
- `each_head`: 144 listas, cada una con una sola cabeza → 144 experimentos separados (la barra de progreso guardada dice 144/144 en 2:10).
- `top_k_logits: 100`: se guardan solo los 100 logits más altos del siguiente token, en formato `{"top_k_logits", "top_k_indices", ...}` por posición (lo usa la métrica de abajo).
- `include_actual_outputs=True`: también guarda la salida sin intervenir, para comparar.
- Se usa `random_ABC`: el contrafactual tiene tres nombres distintos, así que la activación de la cabeza en el contrafactual **no contiene** la información "quién es el IO en la base". Parchearla es parecido a "borrar" la información IOI de esa cabeza (al estilo de los experimentos de path patching del paper de IOI, que usan un dataset ABC).

`PatchAttentionHeads` y `perform_interventions` no existen en el código actual; lo más cercano es `build_attention_head_targets(pipeline, layers, heads, token_position, mode="one_target_per_unit")` en `causalab/causalab/experiments/interchange_targets.py`.

### Métrica de diferencia de logits

```python
def ioi_logit_difference_metric(causal_model_input, raw_output, actual_output=None):
    base_input = causal_model_input['base_input']
    causal_output = causal_model.new_trace(base_input)
    correct_token_text = causal_output['raw_output']['string']        # " IO"
    for token_text in list(causal_output["raw_output"]["logits"].keys()):
        if token_text.strip() == correct_token_text.strip(): continue
        else: incorrect_token_text = token_text; break                 # "S" (sin espacio)
    correct_token_id = pipeline.tokenizer.encode(correct_token_text, add_special_tokens=False)[0]
    incorrect_token_id = pipeline.tokenizer.encode(incorrect_token_text, add_special_tokens=False)[0]
    def get_logit_for_token(output_dict, token_id):
        last_position_scores = output_dict["scores"][-1]
        top_k_indices = last_position_scores["top_k_indices"][0]
        top_k_logits = last_position_scores["top_k_logits"][0]
        for k_idx, idx in enumerate(top_k_indices):
            if idx.item() == token_id: return top_k_logits[k_idx].item()
        return -100.0
    diff_intervention = get_logit_for_token(raw_output, correct_token_id) - get_logit_for_token(raw_output, incorrect_token_id)
    diff_actual = get_logit_for_token(actual_output, correct_token_id) - get_logit_for_token(actual_output, incorrect_token_id)
    return diff_intervention - diff_actual
```

Línea por línea:
- Corre el modelo causal en la base para saber cuál es el IO (`correct_token_text`, con espacio inicial) y cuál el S (la otra clave de `logits`).
- `tokenizer.encode(...)[0]`: id del **primer** token de cada nombre.
- **Ojo:** `correct_token_text` tiene espacio (`" Christopher"`) pero `incorrect_token_text` es la clave del diccionario `logits`, **sin** espacio (`"Kevin"`). En GPT-2, `"Kevin"` y `" Kevin"` son tokens distintos; el que compite con `" Christopher"` después de "to" es `" Kevin"`. Así que el logit "incorrecto" se busca para el token sin espacio, que casi nunca estará en el top-100 y valdrá −100. No pude verificar el efecto en el heatmap guardado, pero la diferencia de logits resultante probablemente esté dominada por el logit del IO.
- `get_logit_for_token` busca el id en el top-k del **último paso** (`scores[-1]`; con `max_new_tokens=1` hay un solo paso) del primer elemento del batch; si no está devuelve −100.
- Devuelve $(\ell_{IO} - \ell_S)_{\text{intervenido}} - (\ell_{IO} - \ell_S)_{\text{sin intervenir}}$: negativo = parchear la cabeza reduce la preferencia por el IO (la cabeza era importante); positivo = la aumenta.

```python
results = compute_custom_scores(raw_results, custom_scoring_fn=ioi_logit_difference_metric,
                                metric_name="logit_difference", use_actual_outputs=True)
experiment.plot_heatmaps(results, ["logit_difference"], save_path=results_dir + "logit_difference_heatmaps/",
                         average_counterfactuals=False)
```

`compute_custom_scores` aplica la métrica a cada ejemplo y unidad; `plot_heatmaps` dibuja un mapa capa × cabeza. La salida guardada es solo la figura (no se ven valores numéricos en el texto). Lo esperable por la literatura de IOI es que destaquen cabezas de capas 9-10 (name movers), pero eso no lo puedo verificar con lo guardado.

### Name mover heads con DBM

```python
config = {"id": "ioi_dbm_name_mover_demo", "learning_rate": 0.001, "batch_size": 32,
          "evaluation_batch_size": 64, "training_epoch": 24,
          "masking": {"regularization_coefficient": 0.1}, "featurizer_kwargs": {"tie_masks": True}}
all_heads = [[(layer, head) for layer in range(0, num_layers) for head in range(num_heads)]]
experiment = PatchAttentionHeads(pipeline, all_heads, last_token, config=config)
labeled_data = causal_model.label_counterfactual_data(filtered_datasets["random_counterfactual"],
                                                      target_variables=["raw_output"])
experiment.train_interventions(labeled_dataset=labeled_data, method="DBM", verbose=True, checker=checker)
```

- `all_heads` es **una sola** lista con las 144 cabezas: una única unidad compuesta sobre la que se aprende la máscara.
- `tie_masks=True`: un escalar por cabeza (se prende o se apaga la cabeza entera, no dimensiones sueltas).
- `regularization_coefficient: 0.1`: peso de la penalización L1 sobre las compuertas (empuja a usar pocas cabezas).
- `label_counterfactual_data(examples, target_variables)` (`causal_model.py`): para cada par hace `run_interchange(base, {"raw_output": cf})` y guarda `label = setting["raw_output"]`. Intervenir `raw_output` es simplemente copiar la salida del contrafactual: la etiqueta es **la respuesta del contrafactual** (el IO de otro prompt, con otros nombres).
- Entonces DBM busca el conjunto de cabezas tal que, al copiar sus salidas en el último token desde el contrafactual, GPT-2 diga el nombre IO **del contrafactual** aunque ese nombre no aparezca en la base. Son las cabezas que "mueven" el nombre hacia la salida: **name mover heads**.
- Métricas finales guardadas del entrenamiento: `loss: 2.4788 accuracy: 0.9640`.

```python
raw_results = experiment.perform_interventions({"random_counterfactual": filtered_datasets["random_counterfactual"]},
                                               include_actual_outputs=True, verbose=True)
results = compute_interchange_scores(raw_results=raw_results, causal_model=causal_model,
    datasets={"random_counterfactual": filtered_datasets["random_counterfactual"]},
    target_variables_list=[["raw_output"]], checker=checker)
train_score = list(results["dataset"]["random_counterfactual"]["model_unit"].values())[0]["raw_output"]["average_score"]
experiment.plot_mask_heatmap(results, save_path=results_dir)
```

- `perform_interventions` aplica la máscara aprendida (binarizada) a los mismos datos.
- `compute_interchange_scores` (`causalab/causal/causal_utils.py`, sí existe en el código actual): toma las salidas crudas, etiqueta los datos con el modelo causal para cada grupo de variables objetivo y calcula `scores` y `average_score` con el `checker`, guardándolos en `results["dataset"][nombre]["model_unit"][unidad][variables]`.
- IIA guardada: **0.965** en los datos de entrenamiento (no hay test aparte, así que es optimista).
- `plot_mask_heatmap`: mapa capa × cabeza de qué cabezas quedaron seleccionadas (solo figura en la salida guardada).

### "Negative Name Mover Heads"

```python
example = filtered_datasets["flip_name_C_sample_new_IO"][0]
# Original:       Then, Sarah and Brian went to the restaurant. Brian gave a computer to
# Counterfactual: Then, Sarah and Melissa went to the restaurant. Sarah gave a computer to
causal_model.can_distinguish_with_dataset(filtered_datasets["flip_name_C_sample_new_IO"],
                                          ["output_position", "output_token"], ["raw_output"])
# 84 out of 84
```

Calculemos con el modelo causal qué pasa en este par:
- Base (A = Sarah, B = Brian, C = Brian): `output_position = 0`, `output_token = Sarah`, respuesta Sarah.
- Contrafactual (A = Sarah, B' = Melissa, C = Sarah): `output_position = 1`, `output_token = Melissa`.
- Intervenir **ambas** variables en la base con los valores del contrafactual: `output_token = Melissa` no es ni Sarah ni Brian, así que ninguna gana el +1 de token; `output_position = 1` le da +1 a B = Brian. Logits: Sarah 2, Brian 3 → respuesta **Brian**, el sujeto S de la base.
- Intervenir `raw_output`: copia la respuesta del contrafactual, **Melissa**.

Brian ≠ Melissa en todos los pares → 84/84 (100%). En el otro caso (C == A en la base) el resultado es simétrico: la etiqueta vuelve a ser el nombre repetido de la base. O sea, en este dataset la etiqueta al intervenir `[output_position, output_token]` es siempre "el sujeto de la base": el token del contrafactual no sirve (es un nombre ausente) y solo actúa la **señal de posición**, que ahora apunta al otro nombre.

```python
config = {"id": "ioi_dbm_negative_name_mover_demo", ..., "training_epoch": 12, ...}
experiment = PatchAttentionHeads(pipeline, all_heads, last_token, config=config)
labeled_data = causal_model.label_counterfactual_data(filtered_datasets["flip_name_C_sample_new_IO"],
                                                      target_variables=["output_position", "output_token"])
experiment.train_interventions(labeled_dataset=labeled_data, method="DBM", verbose=True, checker=checker)
...
results = compute_interchange_scores(..., target_variables_list=[["output_position", "output_token"]], checker=checker)
train_score = ...["output_position-output_token"]["average_score"]
```

- Igual que antes pero con 12 épocas, el dataset `flip_name_C_sample_new_IO` y la etiqueta de intervenir las dos variables intermedias (la clave de resultados es `"output_position-output_token"`).
- DBM busca cabezas que, al parchearlas desde el contrafactual, hagan que GPT-2 diga el **sujeto** de la base. Guardado: `loss: 1.5344 accuracy: 1.0000`, IIA en entrenamiento **1.000**.
- Por qué se llama "negative name mover": la notebook no lo explica. Mi lectura (no verificada) es que se buscan las cabezas que transportan la señal de posición que decide entre IO y S; en el análisis original de IOI las "negative name movers" son cabezas que escriben **en contra** del nombre que atienden. La conexión entre ese concepto y este experimento queda implícita.
- Nota: `perform_interventions` se llama aquí **sin** `include_actual_outputs`, a diferencia de antes; no afecta a `compute_interchange_scores`, que solo compara con las etiquetas.

---

## `token_positions.py` de IOI (no se usa en la notebook, pero es la otra pieza de la tarea)

La notebook solo usa el último token. `tasks/IOI/token_positions.py` define posiciones para los nombres:

- `get_last_token_of_name(input_sample, pipeline, name_var)` (líneas 16-125):
  1. Busca palabras con `re.finditer(r"\b\w+\b", prompt)` que sean A, B o C.
  2. Asigna la **primera** aparición de A a `name_A`, la primera de B a `name_B`, y la **última** aparición del valor de C a `name_C` (que, como C repite A o B, es la segunda mención).
  3. Tokeniza el prompt completo (con padding) y localiza dónde empieza el contenido (primer token no-pad).
  4. Tokeniza el prefijo `prompt[:end_pos]` (hasta el final del nombre) sin padding, y busca esa subsecuencia de tokens dentro del contenido; el índice del nombre es el último token de esa subsecuencia, convertido a coordenadas con padding.
- `create_name_A/B/C_token_position(pipeline)` y `create_last_token_position(pipeline)` envuelven esas funciones en `TokenPosition` con ids `"name_A"`, `"name_B"`, `"name_C"`, `"last_token"`; `create_token_positions(pipeline)` devuelve el diccionario con las cuatro.

Ojo: si un nombre fuera sufijo/prefijo de una palabra del texto, `\b\w+\b` lo trata como palabra entera, así que no confunde "Ann" con "Anna". Lo que sí podría fallar es la búsqueda de la subsecuencia de tokens si el tokenizer junta el último token del nombre con lo que sigue (por ejemplo, `"Kevin,"` tokenizado distinto en el prefijo que en el prompt completo); en GPT-2 los nombres con espacio inicial suelen ser un token aparte, así que en la práctica funciona.

---

## Ejercicios

No hay ejercicios ni celdas `# TERRON` / `FILL IN`. Para practicar con el modelo causal (sin GPU):
- Para cada dataset de la tabla, calcular a mano la etiqueta de `label_counterfactual_data(..., ["output_position"])`, `[... "output_token"]` y `["output_position", "output_token"]`. Por ejemplo, en `swap_names` intervenir solo `output_position` da empate 3-3 (token dice A, posición dice B), que `get_raw_output` resuelve a favor de A.
- Usar `causal_model.can_distinguish_with_dataset(dataset, ["output_position"], ["output_token"])` para ver qué datasets separan las dos señales.

---

## Ojo / errores comunes / dudas

- **API inexistente en el repo actual.** No encontré `causalab.experiments.filter_experiment.FilterExperiment`, `causalab.experiments.LM_experiments.attention_head_experiment.PatchAttentionHeads`, `causalab.experiments.experiment_utils.compute_custom_scores` ni `CounterfactualDataset.from_sampler` en `causalab/causalab/`. Sí existen `filter_dataset` (`experiments/filter.py`), `build_attention_head_targets` (`experiments/interchange_targets.py`), `train_interventions` (`experiments/train.py`) y `compute_interchange_scores` (`causal/causal_utils.py`). La notebook no corre con el código actual sin adaptarla.
- **Constructor de `CausalModel`.** `create_ioi_causal_model` usa `CausalModel(variables, values, parents, mechanisms, id="ioi")`, la firma vieja; la actual es `CausalModel(mechanisms, values, print_pos=None, id=...)` con objetos `Mechanism`.
- **Contrafactuales que solo pasan nombres.** `swap_names` & co. llaman `model.new_trace({"name_A": ..., "name_B": ..., "name_C": ...})` sin `template`, `place` ni `object`. En las salidas guardadas, base y contrafactual comparten plantilla, lugar y objeto, así que la versión vieja debía completarlos desde la base o de otra forma; con la `CausalTrace` actual `raw_input` no se podría calcular (faltan padres). No pude verificar de dónde salían esos valores.
- **Artículo "a/an".** `fill_template` tiene un comentario "Handle both 'a' and 'an' article cases" pero no hace nada: la plantilla se elige al azar independientemente del objeto. Por eso la salida guardada tiene "an ring". Esto mete ruido en los prompts (y posiblemente baja la precisión de GPT-2).
- **Espacio en el token incorrecto** dentro de `ioi_logit_difference_metric` (ver arriba).
- **`random_ABC` sin filtrar** y con contrafactuales sin respuesta definida: sirve para patching (como "borrado"), no para DBM con etiquetas.
- **IIA en datos de entrenamiento:** 0.965 y 1.000 son en los mismos ejemplos usados para entrenar las máscaras; no hay split de test.
- **Rutas relativas** (`results_dir = "tasks/IOI/results/demo/"`) dependen del directorio desde el que se corra el kernel.
- **`output_position`/`output_token` pueden ser `None`** (input mal formado, como en `random_ABC`); `get_logits` entonces da 2 a ambos y `raw_output` elige A por empate.

---

## Resumen de una línea por sección

- **Modelo causal:** entradas A, B, C, lugar, objeto, plantilla; intermedias `output_position` y `output_token`; logits = 2 + señal de token + señal de posición.
- **Sample:** `sample_well_formed_input` fuerza A ≠ B y C ∈ {A, B}; IO con logit 4, S con 2.
- **Datasets:** seis generadores que cambian posición, token, ambos, todo, o dejan el contrafactual sin respuesta (ABC).
- **GPT-2 + checker:** `max_new_tokens=1`, padding izquierdo con `position_ids`; checker por inclusión de strings.
- **Filtro:** se queda con ~67% de pares donde GPT-2 acierta base y contrafactual.
- **Patching por cabeza:** 144 cabezas, último token, contrafactual ABC; métrica = cambio en (logit IO − logit S).
- **DBM name movers:** máscara sobre las 144 cabezas para reproducir "decir el IO del contrafactual"; IIA 0.965 (train).
- **"Negative name movers":** con `flip_name_C_sample_new_IO` la etiqueta es siempre el sujeto de la base (solo actúa la posición); DBM llega a IIA 1.000 (train).
