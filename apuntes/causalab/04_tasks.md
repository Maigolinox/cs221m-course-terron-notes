# causalab/tasks: las tareas concretas

> **Actualización (noviembre 2026):** esta tarea ya corre con la versión de `causalab` incluida en el repo. Cambios principales:
> - el modelo causal se construye con objetos `Mechanism`, la API actual de `CausalModel`;
> - los generadores de contrafactuales devuelven `CausalTrace` y aceptan `model=` para armar los traces con el modelo que se analiza;
> - los datasets son listas simples, no `CounterfactualDataset.from_sampler`;
> - se restauró `statement_conjunction_function`.
>
> Lo que abajo se describe como "no corre" o "API vieja" ya está resuelto. La explicación de las ideas sigue valiendo.

Cada tarea es un "plugin" que define tres cosas: el **modelo causal** de alto nivel (qué algoritmo creemos que implementa el LM), los **generadores de contrafactuales** (qué pares base/contrafactual permiten distinguir hipótesis) y las **token positions** (en qué tokens del prompt buscar las variables). Este archivo recorre las cuatro tareas que hay en `causalab/causalab/tasks/`.

El `__init__.py` de `tasks/` lo dice explícitamente:

- **Activa**: `MCQA` (mantenida).
- **Deprecated**: `IOI`, `entity_binding`, `general_addition`. Cada uno de sus archivos arranca con "DEPRECATED: This task is outdated... See causalab/tasks/MCQA/ for an up-to-date example."

Y no es solo un aviso: las tres deprecated usan una API vieja de `CausalModel` que ya no existe (detalles en cada sección). MCQA es la referencia para entender cómo se arma una tarea con la API actual.

## Ideas clave / teoría

- **Modelo posicional vs. directo.** Varias tareas comparan dos hipótesis de algoritmo que dan la misma salida en datos normales pero difieren bajo intervención. Por ejemplo, en MCQA el modelo "posicional" dice: el LM primero calcula **en qué posición** está la respuesta (`answer_position`) y después lee **el símbolo** de esa posición (`answer`). Un contrafactual que cambie los símbolos pero no la posición (o viceversa) separa ambas variables.
- **Diseño de contrafactuales.** Un buen dataset contrafactual hace que intervenir la variable X cambie el label (si no, no se puede detectar X) y que intervenir X o Y den labels distintos (si no, no se pueden distinguir). Esto se mide con `CausalModel.can_distinguish_with_dataset` (ver `01_causal.md`).
- **Token positions.** Las variables de alto nivel no tienen "un lugar" en el texto, pero sus valores sí: el símbolo "A" del prompt, el último token antes de responder, etc. Las token positions dicen dónde mirar.

## MCQA (`tasks/MCQA/`)

Tarea de preguntas de opción múltiple sobre colores de objetos. Es la del tutorial (`demos/onboarding_tutorial/01_define_MCQA_task.ipynb` en adelante).

### `causal_models.py`

Constantes:

```python
OBJECTS = ["ball", "car", "house", "shirt", "flower", "pen", "cup", "hat", "bag", "shoe"]
COLORS = ["red", "blue", "green", "yellow", "purple", "orange", "pink", "brown", "black", "white"]
NUM_CHOICES = 2
ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
TEMPLATES = [
    "The {object} is {color}. What color is the {object}?"
    + "".join([f"\n{{symbol{str(i)}}}. {{choice{str(i)}}}" for i in range(NUM_CHOICES)])
    + "\nAnswer:"
]
```

Con `NUM_CHOICES = 2` el template queda:

```
The {object} is {color}. What color is the {object}?
{symbol0}. {choice0}
{symbol1}. {choice1}
Answer:
```

Ojo: `{object}` aparece dos veces y el doble llaves `{{symbol0}}` en el f-string produce el literal `{symbol0}`.

Valores (`values`): `choice{i}` y `color` ∈ COLORS, `symbol{i}` y `answer` ∈ letras, `answer_position` ∈ {0, 1}, `template` ∈ TEMPLATES, `object` ∈ OBJECTS, `raw_input`/`raw_output` = None.

Mecanismos:

```python
def _fill_template(t):
    template = t["template"]
    filled_template = template.replace("{object}", t["object"]).replace("{color}", t["color"])
    for i in range(NUM_CHOICES):
        filled_template = filled_template.replace(f"{{symbol{i}}}", t[f"symbol{i}"])
    for i in range(NUM_CHOICES):
        filled_template = filled_template.replace(f"{{choice{i}}}", t[f"choice{i}"])
    return filled_template

def _get_answer_position(t):
    color = t["color"]
    choices = [t[f"choice{i}"] for i in range(NUM_CHOICES)]
    for i in range(NUM_CHOICES):
        if choices[i] == color:
            return i
    raise ValueError(...)

def _get_answer(t):
    return t[f"symbol{t['answer_position']}"]
```

- `_fill_template`: sustitución de strings (con `replace`, que reemplaza todas las apariciones, así `{object}` se llena dos veces).
- `_get_answer_position`: la **primera** opción cuyo color coincide con el color del objeto. Si ninguna coincide, lanza `ValueError` (por eso no se puede usar `sample_input` a ciegas).
- `_get_answer`: el símbolo en esa posición.

```python
mechanisms = {
    "template": input_var(TEMPLATES),
    "object": input_var(OBJECTS),
    "color": input_var(COLORS),
    **{f"symbol{i}": input_var(list(ALPHABET)) for i in range(NUM_CHOICES)},
    **{f"choice{i}": input_var(COLORS) for i in range(NUM_CHOICES)},
    "raw_input": Mechanism(parents=["template", "object", "color", "symbol0", "symbol1", "choice0", "choice1"],
                           compute=_fill_template),
    "answer_position": Mechanism(parents=["color", "choice0", "choice1"], compute=_get_answer_position),
    "answer": Mechanism(parents=["answer_position", "symbol0", "symbol1"], compute=_get_answer),
    "raw_output": Mechanism(parents=["answer"], compute=lambda t: " " + t["answer"]),
}
positional_causal_model = CausalModel(mechanisms, values, id=f"{NUM_CHOICES}_answer_MCQA")
```

El DAG:

```
color, choice0, choice1 ──► answer_position ──┐
symbol0, symbol1 ─────────────────────────────┴─► answer ──► raw_output (" " + letra)
template, object, color, symbols, choices ──► raw_input
```

Las dos variables intermedias interesantes son `answer_position` (¿en qué renglón está la respuesta?) y `answer` (¿qué letra es?). `raw_output` lleva un espacio adelante porque así tokeniza la letra un LM tipo GPT-2 después de "Answer:".

### `counterfactuals.py`

```python
def sample_answerable_question() -> CausalTrace:
    template = random.choice(TEMPLATES)
    obj = random.choice(OBJECTS)
    color = random.choice(COLORS)
    other_colors = [c for c in COLORS if c != color]
    other_choices = random.sample(other_colors, NUM_CHOICES - 1)
    answer_position = random.randint(0, NUM_CHOICES - 1)
    choices = other_choices[:answer_position] + [color] + other_choices[answer_position:]
    symbols = random.sample(ALPHABET, NUM_CHOICES)
    input_dict = {"template": template, "object": obj, "color": color}
    for idx in range(NUM_CHOICES):
        input_dict[f"choice{idx}"] = choices[idx]
        input_dict[f"symbol{idx}"] = symbols[idx]
    return positional_causal_model.new_trace(input_dict)
```

Sampler propio que garantiza que el color correcto esté entre las opciones (y una sola vez), en una posición al azar, con símbolos distintos. Ojo: el color del objeto es aleatorio, no "realista" (puede salir "The ball is purple").

Los tres generadores devuelven `{"input": traza, "counterfactual_inputs": [traza]}`:

- **`same_symbol_different_position()`**: toma una pregunta, elige otra posición `new_pos`, e **intercambia** opción y símbolo entre `pos` y `new_pos`. Resultado: la respuesta correcta está en otro renglón, pero sigue teniendo la **misma letra** (porque el símbolo viajó con ella). En el contrafactual `answer` es igual a la base y `answer_position` es distinto. Con este dataset:
  - intervenir `answer` no cambia el label (misma letra) → no detecta `answer`;
  - intervenir `answer_position` sí cambia el label (la base lee la letra de otro renglón).
- **`different_symbol()`**: mismas opciones, **todas las letras nuevas** (tomadas del complemento de las actuales). `answer_position` igual, `answer` distinto. Intervenir `answer` cambia el label a una letra que ni aparece en el prompt base; intervenir `answer_position` no cambia nada.
- **`random_counterfactual()`**: dos preguntas independientes.

Todos construyen el contrafactual copiando solo las **variables de input** (`{var: input_sample[var] for var in positional_causal_model.inputs}`), modificándolas y corriendo `new_trace`, así las variables calculadas se recalculan coherentemente.

### `token_positions.py`: `create_token_positions(pipeline, template=None)`

```python
def find_correct_symbol(setting):
    for i in range(NUM_CHOICES):
        if setting[f"choice{i}"] == setting["color"]:
            return f"symbol{i}"
    raise ValueError(...)

token_position_specs = {
    "last_token": {"type": "index", "position": -1},
    "correct_symbol": lambda setting: {"type": "variable", "name": find_correct_symbol(setting)},
    "correct_symbol_period": lambda setting: {"type": "index", "position": +1,
                                              "relative_to": {"variable": find_correct_symbol(setting)}},
}
for i in range(NUM_CHOICES):
    token_position_specs[f"symbol{i}"] = {"type": "variable", "name": f"symbol{i}"}
    token_position_specs[f"symbol{i}_period"] = {"type": "index", "position": +1,
                                                 "relative_to": {"variable": f"symbol{i}"}}
factories = build_token_position_factories(token_position_specs, template)
return {name: factory(pipeline) for name, factory in factories.items()}
```

Usa el sistema declarativo de `neural/token_position_builder.py` (ver `02_neural.md`):

- `last_token`: el último token (el de "Answer:" / ":"), donde el LM tiene que decidir la respuesta.
- `symbol0`, `symbol1`: los tokens de cada letra.
- `symbol0_period`, `symbol1_period`: el token siguiente a cada letra (el ".").
- `correct_symbol`, `correct_symbol_period`: **dinámicos**: dependen de la traza (cuál opción es la correcta). Para eso el spec es un lambda que recibe la traza y devuelve un spec concreto.

Devuelve `{nombre: TokenPosition}` listo para usar en `build_residual_stream_targets` o `build_attention_head_targets`.

Ojo: el docstring de `make_causal_metric` en `experiments/metric.py` menciona `causalab.tasks.MCQA.checker`, pero ese módulo no existe: la métrica la define quien usa la tarea.

### Tests (`tasks/MCQA/tests/`)

- `test_causal_models.py`: estructura del modelo (variables, padres, inputs/outputs), que el sampler produce preguntas respondibles, cada mecanismo por separado (`_fill_template`, `answer_position`, `answer`), forward completo y repetido, el caso borde de colores duplicados (devuelve la **primera** posición coincidente) y las constantes.
- `test_counterfactuals.py`: que cada generador produce la estructura correcta y la relación esperada entre base y contrafactual (mismo símbolo/otra posición, todos los símbolos nuevos, aleatorio), validez de los contrafactuales y un test de **distinguibilidad** con `can_distinguish_with_dataset` (afirma que `different_symbol` distingue `answer` de `answer_position` y de no intervenir; el caso de `same_symbol_different_position` solo se imprime, sin assert).
- `test_token_positions.py`: con GPT-2 real (`LMPipeline("gpt2", max_new_tokens=1, max_length=32)`), que las posiciones de símbolos, símbolo correcto, último token y puntos caen en los tokens correctos, `highlight_selected_token`, y el error cuando no hay símbolo correcto.

## IOI (`tasks/IOI/`), DEPRECATED

**Indirect Object Identification**: "After the lunch, Mary and John went to the store. John gave a book to" → " Mary". El nombre que no se repite (el objeto indirecto) es la respuesta.

### Datos

`names.json`, `objects.json`, `places.json`, `templates.json` (cargados al importar). Los templates tienen `{name_A}`, `{name_B}`, `{place}`, `{name_C}`, `{object}`, por ejemplo `"After {name_A} and {name_B} went to the {place}, {name_C} gave a {object} to"` (con variantes "a"/"an").

### `causal_models.py`: `create_ioi_causal_model(bias=2.0, token_coeff=-1.0, pos_coeff=-1.0)`

Variables: `template, name_A, name_B, name_C, place, object` (inputs), `raw_input`, `output_position`, `output_token`, `logits`, `raw_output`.

- `fill_template`: reemplaza los placeholders.
- `get_output_position`: si `name_C == name_A` → 1 (la respuesta es B); si `name_C == name_B` → 0; si no, `None`.
- `get_output_token`: el nombre que no es `name_C` (o `None`).
- `get_logits` (closure con los coeficientes):

```python
token_signal_A = 1 if output_token == name_A else 0
pos_signal_A = 1 if output_position == 0 else 0
logit_A = bias - token_coeff * token_signal_A - pos_coeff * pos_signal_A
# análogo para B
return {name_A: logit_A, name_B: logit_B}
```

  Con los defaults (`token_coeff = pos_coeff = -1`) queda `logit = bias + señal_token + señal_posición`. Es un modelo causal con **dos mecanismos** que suman evidencia: uno basado en la identidad del token y otro en la posición. Separarlos permite contrafactuales que activan uno y no el otro (p.ej. intercambiar el orden de A y B).
- `get_raw_output`: devuelve un **dict** `{"string": " " + nombre_con_mayor_logit, "logits": logits}` (empate → A). Es el único caso donde el label es un dict; por eso `LM_loss_and_metric_fn` acepta labels dict con clave `"string"`.

Problema con la API actual: el modelo se construye con

```python
return CausalModel(variables, values, parents, mechanisms, id="ioi")
```

pero la firma actual es `CausalModel(mechanisms, values, print_pos=None, id="null")`. Con 4 posicionales más `id=`, Python lanza `TypeError` ("multiple values for argument 'id'"). Y como el archivo hace `positional_causal_model = create_ioi_causal_model()` al final, **importar `tasks.IOI.causal_models` falla**. Además los mecanismos son funciones `trace -> valor` sueltas, no objetos `Mechanism` con `parents`, que es lo que la API nueva espera.

### `counterfactuals.py`

- `sample_well_formed_input()`: `sample_input()`, fuerza `name_A != name_B` y pone `name_C` igual a uno de los dos (las asignaciones `input_sample["name_B"] = ...` son intervenciones sobre la traza; como son inputs, el efecto es recalcular).
- `swap_names()`: contrafactual con A y B intercambiados, mismo C. Cambia la posición de la respuesta pero no el token.
- `flip_name_C()`: C pasa a ser el otro nombre. Cambia token y posición de la respuesta.
- `flip_name_C_sample_new_IO()`: C pasa al otro nombre y además se muestrea un nuevo nombre para el objeto indirecto.
- `swap_names_and_flip_C()`: intercambia A/B y además invierte C.
- `random_counterfactual()`: dos inputs bien formados independientes.
- `random_ABC()`: base bien formada; contrafactual con tres nombres distintos (sin respuesta definida: `output_token = None`).

Ojo: los contrafactuales se construyen con `model.new_trace({"name_A": ..., "name_B": ..., "name_C": ...})` **sin** `template`, `place` ni `object`; con la semántica actual de `CausalTrace`, `raw_input` no se calcularía (le faltan padres). Otro síntoma de que la tarea quedó atrás.

### `token_positions.py`

`get_last_token_of_name(input_sample, pipeline, name_var)` encuentra el **último token** del nombre correspondiente:

1. Busca palabras del prompt que sean alguno de los tres nombres; asigna a `name_A` la primera aparición de A, a `name_B` la primera de B, y a `name_C` la **última** aparición del valor de C (que coincide con A o B, por eso hay que tomar la última).
2. Tokeniza el prompt completo (con padding) y el prefijo `prompt[:fin_del_nombre]` sin padding, busca la subsecuencia del prefijo dentro de los tokens del contenido, y convierte al sistema de coordenadas con padding (`content_start_idx + fin - 1`).

Es el enfoque "viejo" (buscar subsecuencias de tokens) previo al sistema de offsets de `token_position_builder.py`. Las fábricas `create_name_A/B/C_token_position`, `create_last_token_position` y `create_token_positions(pipeline)` devuelven `{"name_A", "name_B", "name_C", "last_token"}`.

Hay además un `ioi_demo.ipynb` en la carpeta.

## entity_binding (`tasks/entity_binding/`), DEPRECATED

Tarea de **binding** de entidades: "Pete loves jam, Ann loves pie, and Bob loves cake. What does Ann love?" → "pie". La pregunta es cómo el LM liga cada persona con su comida y recupera el par correcto.

### `config.py`

- `EntityBindingTaskConfig` (dataclass): `max_groups` (k grupos), `max_entities_per_group` (d entidades por grupo), `entity_roles` (`{0: "person", 1: "food"}`), `entity_pools` (valores posibles por rol), `statement_template` (`"{entity_e0} loves {entity_e1}"`), `question_templates` (dict `((índices_consultados,), índice_respuesta) -> template`), `delimiters`, `prompt_prefix`, `prompt_suffix`, `statement_question_separator`, `fixed_query_indices`.
- `EntityGroup(entities, group_index)`: un grupo G_i (p.ej. ("Pete", "jam")).
- `BindingMatrix(groups, max_groups, max_entities_per_group)`: la matriz de bindings; `get_entity(g, e)` devuelve `None` fuera de rango.
- `create_sample_love_config()`: 2 entidades por grupo (persona, comida), hasta 3 grupos, preguntas "What does {query_entity} love?" y "Who loves {query_entity}?".
- `create_sample_action_config()`: 3 entidades (persona, objeto, lugar), `"{entity_e0} put {entity_e1} in the {entity_e2}"`, 9 templates de pregunta (consultas por 1 o 2 entidades).

### `templates.py`: `TemplateProcessor`

- `fill_statement_template(binding_matrix)`: llena el template por grupo y los une con `statement_conjunction_function(filled_statements, delimiters)`.
- `select_question_template(query_indices, answer_index)`.
- `fill_question_template(template, query_group, query_indices, binding_matrix)`: arma el dict de sustitución con `query_entity` y con cada rol del grupo consultado (`person`, `object`, ...).
- `generate_full_prompt(...)`: `prefix + statement + separator + question + suffix` y la respuesta esperada.
- `create_sample_binding_matrix(config, num_groups)`: matriz aleatoria para tests.

Ojo: `templates.py` importa `statement_conjunction_function` desde `causalab.causal.causal_utils`, pero esa función **no existe** en `causal_utils.py`. Importar este módulo (y por lo tanto casi toda la tarea) falla con `ImportError`.

### `causal_models.py`

Tres variantes, las tres con la firma vieja `CausalModel(variables, values, parents, mechanisms, id=...)`:

1. **`create_direct_causal_model(config)`**: variables `entity_g{g}_e{e}` (una por celda de la matriz, con `None` para inactivas), `query_group`, `query_indices`, `answer_index`, `active_groups`, `entities_per_group`, `statement_template`, `question_template` (depende de `query_indices` y `answer_index`), `raw_input`, `raw_output`. El `raw_output` lee **directamente** `entity_g{query_group}_e{answer_index}`: hipótesis de "indexado directo" (el LM sabe qué grupo se consulta por su índice).
2. **`create_positional_causal_model(config)`**: agrega `query_entity` (la entidad consultada, por valor, como tupla) y `positional_query_group` (búsqueda: el grupo activo donde **todas** las entidades consultadas aparecen en sus posiciones; error si hay más de uno, `None` si ninguno). `raw_output` usa `positional_query_group`: hipótesis de "recuperación por contenido" (el LM busca dónde aparece "Ann" y lee la comida de ese grupo). En datos normales ambos modelos coinciden; difieren cuando se interviene `query_group` o se mueven entidades.
3. **`create_positional_entity_causal_model(config)`**: hace explícita la posición de cada entidad: `positional_entity_g{g}_e{e}` (el índice de grupo de esa entidad, o `None`), `positional_query_e{e}` (tupla de grupos donde aparece la entidad consultada en la posición e), `positional_answer` (intersección de los conjuntos de grupos de todas las entidades consultadas; error si es ambigua). `raw_output` lee la respuesta de `positional_answer`. Pensado para intervenciones del tipo "intercambiar las posiciones de dos entidades".

`sample_valid_entity_binding_input(config, ensure_positional_uniqueness=True)`: rejection sampling (hasta 100 intentos) que asegura grupos activos llenos, `query_group` dentro de los activos, respuesta distinta de lo consultado, que exista template de pregunta, entidades distintas dentro de cada grupo y (opcional) distintas por posición entre grupos (necesario para que la búsqueda posicional no sea ambigua). Si no lo logra, devuelve un input fijo de fallback.

Ojo en `create_positional_entity_causal_model`: `lambda t, g=g: g if t[f"entity_g{g}_e{e}"] is not None else None` captura `g` por default pero **no** `e`, así que todas las lambdas usan el último valor de `e` del loop.

### `counterfactuals.py`

Todos crean el modelo directo y usan `sample_valid_entity_binding_input`; devuelven **dicts** (no trazas) y borran `raw_input` del contrafactual para que se regenere:

- `swap_query_group(config)`: intercambia el grupo consultado con otro y actualiza `query_group` para que siga a la entidad consultada. La pregunta es la misma ("What does Ann love?"), cambia el orden de los hechos; la respuesta es la misma entidad pero en otra posición.
- `swap_query_group_preserve_answer(config)`: mismo intercambio (el código es prácticamente idéntico al anterior).
- `swap_non_query_groups(config)`: intercambia dos grupos no consultados; control, la respuesta no debería cambiar.
- `random_counterfactual(config)`.

### `token_positions.py` (posiciones "estructuradas")

En vez de buscar substrings, reconstruye la estructura del prompt:

- `PromptSegment` (texto, tipo de segmento, grupo/entidad, rango de caracteres, posiciones de token) y `ParsedPrompt` (segmentos + regiones de enunciado y pregunta) con `get_entity_segments` / `get_entity_tokens(group_idx, entity_idx, region)`.
- `PromptParser.parse_prompt(input_sample)`: rearma la `BindingMatrix` desde las variables, parsea el enunciado con los delimitadores y la pregunta, y ajusta offsets.
- `PromptTokenizer.tokenize_prompt(parsed)`: tokeniza el prompt completo y construye un mapa carácter → token decodificando prefijos de tokens de a uno.
- `get_entity_token_indices_structured(input_sample, pipeline, config, group_idx, entity_idx, region=None, token_idx=None)`: API de alto nivel; permite distinguir la aparición de una entidad en el enunciado vs. en la pregunta.

Dudas: `parse_prompt` calcula el inicio de la pregunta como `len(statement) + 1` (asume un separador de un carácter y no suma `prompt_prefix`), mientras que `experiment_config.get_task_config` usa prefijo y separador `"\n\n"`. No queda claro solo leyendo el código si las posiciones quedan bien alineadas en ese caso. Además `_parse_statement` vuelve a importar la inexistente `statement_conjunction_function`.

### `experiment_config.py`

Configuración para los scripts de `experiments/`:

- `get_task_config(name)`: `'love'` (2 grupos), `'action'` (3 grupos), `'positional_entity'` (love con `fixed_query_indices=(0,)`); a todos les agrega el prefijo "We will ask a question about the following sentences.\n\n", separador `"\n\n"` y sufijo `"\nAnswer:"`.
- `get_causal_model(config, 'positional' | 'positional_entity')`.
- `get_counterfactual_generator('swap_query_group', config)`.
- `get_token_positions(pipeline, config, 'last_token' | 'both_e1')`: `both_e1` devuelve los últimos tokens de la segunda entidad (la comida) de los grupos 0 y 1 **en orden invertido para el contrafactual** (usa el kwarg `is_original`); combinado con las variables `"A<-B"` esto implementa "patchear la comida del grupo 0 de la base con la del grupo 1 del contrafactual".
- `get_checker()`: acierta si el esperado está contenido en el string generado o el string (sin espacios) está contenido en el esperado.
- `get_default_target_variables(name)`: para love/action `[["positional_query_group"], ["query_entity"], ["raw_output"]]`; para positional_entity una lista con cuatro intervenciones cruzadas `"positional_entity_g0_e1<-positional_entity_g1_e1"`, etc. (la sintaxis `<-` de `run_interchange`).
- `get_pipeline_config(name)`: `max_new_tokens=5`, `max_length=256`.

`experiments/` tiene scripts (`generate_and_filter_dataset.py`, `run_interventions.py`, `visualize_results.py`, `run_all*.sh`) y hay 4 notebooks en `notebooks/`.

### Tests (`tasks/entity_binding/tests/`)

- `test_core_data_structures.py`: `EntityGroup`, `BindingMatrix` (incluyendo `None`), configs love/action y los wrappers de prompt.
- `test_template_processing.py`: llenado de enunciados y preguntas, selección de templates, prompts con y sin wrapper, la tarea de acción, conjunción de varios grupos.
- `test_causal_models.py`: modelos directo y posicional, que coinciden en inputs normales y **difieren tras un swap**, el mecanismo de búsqueda posicional, config de acción.
- `test_positional_entity_model.py`: las variables posicionales nuevas, equivalencia con el modelo posicional, intervenciones sobre `positional_entity`, `positional_query` y `positional_answer`, consultas de varias entidades.
- `test_counterfactual.py`: los cuatro generadores, varios ejemplos y la tarea de acción.

Dado el `ImportError` de `templates.py` y la firma vieja de `CausalModel`, es esperable que estos tests no pasen con el código actual (no los ejecuté).

## general_addition (`tasks/general_addition/`), DEPRECATED

Suma de K números de D dígitos: "The sum of 23 and 45 is" → " 68" (formateado por el template processor).

### `config.py`

`AdditionTaskConfig(max_numbers, max_digits, templates, prompt_prefix="", prompt_suffix="")` y fábricas: `create_two_number_two_digit_config()` ("The sum of {num0} and {num1} is"), `create_two_number_three_digit_config()`, `create_three_number_two_digit_config()` ("The sum of {num0}, {num1}, and {num2} is"), `create_general_config(max_numbers, max_digits)`.

### `templates.py`: `AdditionTemplateProcessor`

`digits_to_number`, `number_to_digits(number, num_digits)` (con ceros a la izquierda), `format_number`, `fill_template(template, numbers)` (con `str.format` sobre `{num0}`, `{num1}`, ...), `generate_prompt(digit_values, num_addends, num_digits, template)` (arma los números desde `digit_{k}_{d}`, aplica prefijo/sufijo), `compute_sum(numbers)` (suma enteros y devuelve D+1 dígitos), `format_output`.

### `causal_models.py`

1. **`create_basic_addition_model(config)`**: inputs `digit_{k}_{d}` (k = sumando, d = dígito contado **desde la izquierda**, 0 = más significativo), `num_addends`, `num_digits`, `template`; outputs `output_digit_{d}` para d = 0..D (cada uno recalcula la suma completa y toma su dígito); `raw_input` y `raw_output` (`" " + format_output(dígitos)`). Es un modelo "solo entrada-salida", sin estructura interna.
2. **`create_intermediate_addition_model(config)`**: solo 2 sumandos. Variables intermedias con índices **desde la derecha empezando en 1**:
   - `C_i`: carry de la posición i. `C_1 = 1 si x_1 + y_1 >= 10`; `C_i = 1 si x_i + y_i + C_{i-1} >= 10`.
   - `O_i`: dígito de salida. `O_1 = (x_1 + y_1) % 10`; `O_i = (x_i + y_i + C_{i-1}) % 10`; `O_{D+1} = C_D`.
   - La conversión de índice es `d = D - i` (posición i desde la derecha = dígito d desde la izquierda).
   - Los mecanismos se crean con closures y defaults (`d=d`, `make_carry_mechanism(pos, d=d)`) para capturar bien las variables del loop.
   - `raw_output` invierte `O_1..O_{D+1}` para escribir de izquierda a derecha.

   Este es el modelo interesante: permite preguntar si el LM calcula explícitamente los carries.

Los dos usan la firma vieja `CausalModel(variables, values, parents, mechanisms, id=model_id)`.

Samplers: `sample_valid_addition_input(config, num_addends, num_digits)` (primer dígito de cada número entre 1 y 9 para evitar ceros a la izquierda, dígitos sobrantes en 0) y `sample_sum_to_nine_input(config, digit_position)` (fuerza que en esa posición los dígitos sumen 9: el caso donde el carry de la posición anterior decide si hay carry en esta).

### `counterfactuals.py`

- `random_counterfactual(config, num_addends, num_digits)`: dos inputs independientes.
- `sum_to_nine_counterfactual(config, digit_position)`: base con suma 9 en esa posición, contrafactual aleatorio.

Ambos crean el modelo básico pero no lo usan, y devuelven **dicts de inputs** (sin `raw_input` ni variables calculadas); quien los use tiene que pasarlos por el modelo.

### `token_positions.py`

- `get_digit_token_position(input_sample, pipeline, addend_idx, digit_idx)`: arma el número del sumando como string (sin ceros a la izquierda) y usa `get_substring_token_ids(prompt, number_str, pipeline, add_special_tokens=False)`. Si el número es un solo token, lo devuelve; si son varios, devuelve **todos** los tokens del número. Ojo: `digit_idx` no se usa para elegir un token (hay un comentario reconociéndolo), y la variable `context` que construye tampoco se usa; además si el mismo número aparece dos veces, `get_substring_token_ids` toma la primera aparición.
- `get_delimiter_token_position(..., delimiter_type)` para "and", "is", "comma".
- `create_token_positions(pipeline, num_addends, num_digits)`: `digit_{k}_{d}` para cada dígito, `delimiter_and`, `delimiter_is`, `delimiter_comma` (si hay más de 2 sumandos) y `last_token`.

`experiments/` tiene scripts numerados (`01_generate_and_filter_dataset.py`, `02_run_interventions.py`, `03_compute_scores_and_visualize.py`), `tokenization_config.py` y shells; hay 3 notebooks.

### Tests (`tasks/general_addition/tests/`)

- `test_basic_functionality.py`: config, templates, modelos básico e intermedio, contrafactuales y el sampler de suma 9.
- `test_intermediate_model.py`: el modelo con carries: que sume bien, que `C_i` y `O_i` tengan los valores esperados y que las intervenciones sobre esas variables tengan el efecto esperado.
- `test_token_positions.py` y `test_token_positions_all_models.py`: posiciones de dígitos con tokenizers reales (varios modelos).
- `test_batch_size.py`: en realidad es un script para encontrar el batch size máximo que entra en la GPU (sube el batch hasta OOM); tiene un `sys.path` hardcodeado a una máquina del autor.
- `analyze_tokenization.py`: script de análisis (no es un test propiamente).

## Cómo armar una tarea nueva (siguiendo MCQA)

Con la API actual, el patrón es:

1. `causal_models.py`: constantes, `values`, funciones de mecanismo `trace -> valor`, dict `mechanisms` con `input_var(...)` para inputs y `Mechanism(parents=[...], compute=f)` para el resto, incluyendo `raw_input` (llenando un template) y `raw_output` (con el espacio inicial que espera el tokenizer). Construir `CausalModel(mechanisms, values, id=...)`.
2. `counterfactuals.py`: un sampler de inputs válidos que devuelva `model.new_trace(dict)`, y generadores sin argumentos que devuelvan `{"input": traza, "counterfactual_inputs": [traza]}` modificando solo **inputs** y recalculando con `new_trace`.
3. `token_positions.py`: specs declarativos sobre **el mismo template** que usa `raw_input`, con lambdas para posiciones que dependen de la traza, y `build_token_position_factories`.
4. Verificar con `can_distinguish_with_dataset` que los datasets separan las variables que querés comparar.

## Ojo / errores comunes / dudas

- **Solo MCQA funciona con la API actual.** IOI, entity_binding y general_addition llaman `CausalModel(variables, values, parents, mechanisms, id=...)`; entity_binding además importa una función inexistente. No los ejecuté (no hay torch en este entorno), pero la incompatibilidad se ve leyendo las firmas.
- **El template de token positions tiene que ser el mismo que llena `raw_input`.** En MCQA, `create_token_positions` usa `TEMPLATES[0]` por defecto, que es lo que usa `_fill_template`.
- **El espacio inicial del `raw_output`** (`" A"`, `" Mary"`) importa para que coincida con cómo el LM genera el token.
- **`answer_position` toma la primera coincidencia** si hay colores repetidos entre las opciones; el sampler de MCQA evita ese caso.
- **Labels dict en IOI** (`{"string", "logits"}`): la loss solo usa `"string"`.

## Resumen de una línea por sección

- **MCQA**: color → `answer_position` → `answer` (letra); contrafactuales que separan posición y símbolo; posiciones declarativas (último token, símbolos, puntos, símbolo correcto dinámico).
- **IOI**: dos señales (token y posición) suman logits para elegir el objeto indirecto; contrafactuales de swap/flip de nombres; posiciones por búsqueda de subsecuencias; API vieja, no importa.
- **entity_binding**: matriz de bindings; modelos directo, posicional y posicional-por-entidad; swaps de grupos; parser estructurado de prompts; API vieja e import roto.
- **general_addition**: modelo entrada-salida y modelo con carries `C_i`/`O_i`; sampler de suma 9; posiciones por substring del número; API vieja.
- **Tarea nueva**: seguir el patrón de MCQA con `Mechanism`, `new_trace` y specs declarativos sobre el mismo template.
