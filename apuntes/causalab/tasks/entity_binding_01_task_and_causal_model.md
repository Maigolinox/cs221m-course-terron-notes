# Entity binding 01: la tarea y su modelo causal

> **Actualización (noviembre 2026):** esta tarea ya corre con la versión de `causalab` incluida en el repo. Cambios principales:
> - el modelo causal se construye con objetos `Mechanism`, la API actual de `CausalModel`;
> - los generadores de contrafactuales devuelven `CausalTrace` y aceptan `model=` para armar los traces con el modelo que se analiza;
> - los datasets son listas simples, no `CounterfactualDataset.from_sampler`;
> - se restauró `statement_conjunction_function`.
>
> Lo que abajo se describe como "no corre" o "API vieja" ya está resuelto. La explicación de las ideas sigue valiendo.

Notebook: `causalab/causalab/tasks/entity_binding/notebooks/01_task_and_causal_model.ipynb`

Esta notebook construye, pieza por pieza, la tarea de **entity binding** ("ligado de entidades"): prompts del tipo *"Pete loves jam, and Ann loves pie. What does Pete love?"*, donde el modelo tiene que recuperar la entidad que está **ligada** (en el mismo grupo) a la entidad por la que se pregunta. La pregunta que responde es: ¿cómo representamos esta tarea de forma abstracta (configuración, matriz de ligados, plantillas) y cómo la convertimos en un **modelo causal** de causalab que genera el prompt (`raw_input`) y la respuesta correcta (`raw_output`)? Todavía no hay red neuronal: todo es simbólico.

> Ojo: la notebook y todo el código de `tasks/entity_binding/` están marcados como **DEPRECATED** ("See `causalab/tasks/MCQA/` for an up-to-date example"). Las salidas guardadas vienen de una versión anterior de la librería; más abajo (sección "Ojo") explico qué partes ya no cuadran con el código actual de `causalab/causal/`. Para la API general de modelos causales ver los apuntes generales (`../01_causal.md` y siguientes), pero este archivo está escrito para entenderse solo.

---

## Ideas clave / teoría

### La tarea

- Hay **grupos de entidades** (entity groups). En la tarea "love" cada grupo es un par (persona, comida): `(Pete, jam)`, `(Ann, pie)`. En la tarea "action" cada grupo es un trío (persona, objeto, lugar): `(Pete, jam, cup)`.
- Cada grupo se verbaliza con una **plantilla de enunciado** (`"{entity_e0} loves {entity_e1}"`) y los enunciados se unen con delimitadores (`", "`, `", and "`, `"."`).
- Luego viene una **pregunta** que menciona una o más entidades de un grupo (las *query entities*) y pide otra entidad del mismo grupo (la *answer entity*).

La notación que usa el código:

| Concepto | Nombre en el código | Ejemplo |
|---|---|---|
| Entidad del grupo `g`, posición `e` dentro del grupo | `entity_g{g}_e{e}` | `entity_g1_e0 = "Ann"` |
| Grupo por el que se pregunta | `query_group` | `1` |
| Posiciones (roles) que aparecen en la pregunta | `query_indices` (tupla) | `(0,)` = se menciona a la persona |
| Posición de la respuesta | `answer_index` | `1` = se pide la comida |
| Cuántos grupos se usan de verdad | `active_groups` | `2` |

La idea de fondo (lo que se va a estudiar en las notebooks 03 y 04) es: cuando un LM resuelve esto, ¿cómo encuentra el grupo correcto? ¿Usa algo como un "índice de grupo" (posición), o busca por contenido la entidad mencionada?

### Modelo causal en causalab (resumen mínimo)

Un modelo causal es un grafo dirigido acíclico de **variables**. Cada variable tiene:
- un conjunto de **valores** posibles,
- una lista de **padres**,
- un **mecanismo**: una función que calcula su valor a partir de los padres.

Las variables sin padres son las **entradas**. Correr el modelo ("forward") es fijar las entradas y calcular el resto en orden topológico. Causalab exige siempre dos variables especiales: `raw_input` (el texto que se le da al LM) y `raw_output` (la respuesta esperada). Intervenir una variable es fijar su valor a mano rompiendo el vínculo con sus padres; eso es la base de las **interchange interventions** que aparecen en las notebooks siguientes.

En esta notebook el modelo usado es el **directo** (`create_direct_causal_model`): no tiene variables intermedias; `raw_output` se calcula indexando directamente `entity_g{query_group}_e{answer_index}`.

---

## Recorrido por la notebook

### Step 1: estructuras de datos

#### Celda de imports

```python
%load_ext autoreload
%autoreload 2
import sys
sys.path.append("..")
from causalab.tasks.entity_binding.config import (
    EntityGroup, BindingMatrix, create_sample_love_config, create_sample_action_config,
)
```

- `%autoreload 2`: recarga automáticamente los módulos importados si cambias el `.py`, útil mientras se desarrolla la tarea.
- `sys.path.append("..")`: agrega la carpeta padre al path para poder importar `causalab`. Ojo, desde `tasks/entity_binding/notebooks/` el `..` es `tasks/entity_binding/`, no la raíz del repo; probablemente funcionaba porque `causalab` estaba instalado como paquete (`pip install -e`).
- Se importan las dos clases de datos y dos "fábricas" de configuraciones de ejemplo, todas de `causalab/causalab/tasks/entity_binding/config.py`.

#### `EntityBindingTaskConfig`: el plano de la tarea

```python
love_config = create_sample_love_config()
print(f"Max groups: {love_config.max_groups}")
...
print(f"Question templates: {love_config.question_templates}")
```

`EntityBindingTaskConfig` es un `@dataclass` (`config.py`, líneas 15-41) con estos campos:

- `max_groups`: máximo número de grupos (k). Para "love" es 3.
- `max_entities_per_group`: entidades por grupo (d). Para "love" es 2.
- `entity_roles`: `{0: "person", 1: "food"}`, nombre del rol de cada posición.
- `entity_pools`: `{0: [...personas...], 1: [...comidas...]}`, de dónde se samplea cada posición.
- `statement_template`: `"{entity_e0} loves {entity_e1}"`.
- `question_templates`: diccionario cuya **clave** es `(query_indices, answer_index)` y el valor es la plantilla. Para "love": `((0,), 1) -> "What does {query_entity} love?"` y `((1,), 0) -> "Who loves {query_entity}?"`.
- `delimiters`: `[", ", "FILL", ", and ", "."]` (ver abajo cómo se usan).
- `prompt_prefix`, `prompt_suffix`, `statement_question_separator` (por defecto `" "`), `fixed_query_indices` (si no es `None`, se fija siempre qué rol se pregunta). Estos se usan en la notebook 04.

La salida guardada imprime exactamente esos valores (Max groups 3, People `['Pete', 'Ann', 'Tim', 'Bob', 'Sue', 'Kate']`, etc.).

#### `EntityGroup` y `BindingMatrix`: una instancia concreta

```python
group1 = EntityGroup(["Pete", "jam"], group_index=0)
group2 = EntityGroup(["Ann", "pie"], group_index=1)
matrix = BindingMatrix([group1, group2],
                       max_groups=love_config.max_groups,
                       max_entities_per_group=love_config.max_entities_per_group)
print(f"Active groups: {matrix.active_groups}")
print(f"Group 0, Entity 0 (person): {matrix.get_entity(0, 0)}")
...
print(f"Group 2, Entity 0 (inactive): {matrix.get_entity(2, 0)}")
```

- `EntityGroup(entities, group_index)` (`config.py` 44-62) guarda la lista de entidades de un grupo; `get_entity(i)` devuelve la entidad `i` o `None` si el índice se sale.
- `BindingMatrix(groups, max_groups, max_entities_per_group)` (`config.py` 65-103) es la "matriz" G de la tarea: fila = grupo, columna = rol. `active_groups = len(groups)` (acá 2, aunque el máximo es 3).
- `get_entity(g, e)` devuelve `G[g][e]` o `None` si el grupo está inactivo (`g >= active_groups`). Por eso `get_entity(2, 0)` da `None` en la salida guardada.

La celda markdown "Flat Dictionary Representation" no tiene código asociado; la idea es que el modelo causal no usa `BindingMatrix` como variable, sino un diccionario plano con claves `entity_g{g}_e{e}`.

### Step 2: el sistema de plantillas

```python
from causalab.tasks.entity_binding.templates import TemplateProcessor
processor = TemplateProcessor(love_config)
```

`TemplateProcessor` (`templates.py`) solo guarda la config y ofrece métodos para convertir una `BindingMatrix` en texto.

#### Llenar el enunciado

```python
statement = processor.fill_statement_template(matrix)
# 'Pete loves jam, and Ann loves pie.'
```

Lo que hace `fill_statement_template` (`templates.py` 28-56), línea por línea:

1. Para cada grupo activo `e` arma `entity_dict = {"entity_e0": G[e][0], "entity_e1": G[e][1]}`.
2. Llena la plantilla con `str.format(**entity_dict)`: `"Pete loves jam"`, `"Ann loves pie"`.
3. Une la lista con `statement_conjunction_function(filled_statements, self.config.delimiters)`.

La función de conjunción no se ve en la notebook. Por la salida guardada, con 2 enunciados produce `"A, and B."` y con 3 produce `"A, B, and C."`, o sea: separador `", "` entre los del medio, `", and "` antes del último y `"."` al final; además pone en mayúscula la primera letra. Hay una reimplementación en `utils/lecture_09_utils.py` (líneas 31-53) que hace exactamente eso: `sep = delimiters[0]`, `last_sep = delimiters[-2]`, `end = delimiters[-1]` (el `"FILL"` del medio no se usa). Ojo: en la versión actual de `causalab/causal/causal_utils.py` esta función **ya no existe** (ver sección "Ojo").

#### Elegir y llenar la pregunta

```python
template1 = processor.select_question_template((0,), 1)
question1 = processor.fill_question_template(template1, 0, (0,), matrix)
# 'What does Pete love?'
template2 = processor.select_question_template((1,), 0)
question2 = processor.fill_question_template(template2, 0, (1,), matrix)
# 'Who loves jam?'
```

- `select_question_template(query_indices, answer_index)` (`templates.py` 58-77) arma la clave `(query_indices, answer_index)` y la busca en `config.question_templates`; si no existe lanza `KeyError`.
- `fill_question_template(template, query_group, query_indices, binding_matrix)` (79-120):
  - recorre `query_indices` y pone `substitution_dict["query_entity"] = G[query_group][idx]` (si hay varios índices, se queda con el último; el comentario dice "Simple case: single query entity");
  - además agrega **todas** las entidades del grupo con el nombre de su rol: `{"person": "Pete", "food": "jam"}`. Eso es lo que permite que las plantillas de la tarea "action" usen `{person}`, `{object}`, `{location}`;
  - finalmente `template.format(**substitution_dict)`. Como `format` ignora claves sobrantes, el mismo diccionario sirve para cualquier plantilla.

#### Prompt completo

```python
prompt1, answer1 = processor.generate_full_prompt(matrix, 0, (0,), 1)
# 'Pete loves jam, and Ann loves pie. What does Pete love?'  -> 'jam'
prompt2, answer2 = processor.generate_full_prompt(matrix, 0, (1,), 0)
# 'Pete loves jam, and Ann loves pie. Who loves jam?'        -> 'Pete'
```

`generate_full_prompt` (122-166) encadena lo anterior: enunciado + `statement_question_separator` + pregunta, envuelto en `prompt_prefix` y `prompt_suffix`; la respuesta es `G[query_group][answer_index]`. El punto que remarca la notebook: mismas entidades, distinta pregunta, distinta respuesta.

#### Tarea "action" (3 entidades por grupo)

```python
action_config = create_sample_action_config()
action_processor = TemplateProcessor(action_config)
action_group = EntityGroup(["Pete", "jam", "cup"], 0)
action_matrix = BindingMatrix([action_group], action_config.max_groups, action_config.max_entities_per_group)
action_statement = action_processor.fill_statement_template(action_matrix)
# 'Pete put jam in the cup.'
```

`create_sample_action_config` (`config.py` 131-170) define roles persona/objeto/lugar, pools de 8 elementos cada uno, plantilla `"{entity_e0} put {entity_e1} in the {entity_e2}"` y **9 plantillas de pregunta**: 6 que mencionan una entidad (`((0,), 1)`, `((0,), 2)`, ...) y 3 que mencionan dos (`((0, 1), 2)`: "Where did {person} put {object}?", etc.). Con un solo grupo, `statement_conjunction_function` devuelve solo el enunciado más el punto final.

### Step 3: integración en un modelo causal

```python
from causalab.tasks.entity_binding.causal_models import (
    create_direct_causal_model, sample_valid_entity_binding_input,
)
```

#### Estructura del modelo directo

```python
causal_model = create_direct_causal_model(love_config)
print(f"Causal Model: {causal_model.id}")          # entity_binding_direct_3g_2e
print(f"Total variables: {len(causal_model.variables)}")   # 15
entity_vars = [v for v in causal_model.variables if v.startswith("entity_")]
control_vars = [v for v in causal_model.variables
                if not v.startswith("entity_") and v not in ["raw_input", "raw_output"]]
```

Qué arma `create_direct_causal_model` (`causal_models.py` 22-251):

**Variables** (líneas 45-66): una `entity_g{g}_e{e}` por cada celda de la matriz (3×2 = 6), más `query_group`, `query_indices`, `answer_index`, `active_groups`, `entities_per_group`, `statement_template`, `question_template`, `raw_input`, `raw_output`. Total 6 + 9 = 15, que es lo que muestra la salida.

**Valores** (68-95): cada entidad puede ser cualquier elemento de su pool o `None` (grupo inactivo); `query_group ∈ {0,1,2}`; `query_indices ∈ {(0,), (1,)}`; `active_groups ∈ {1,2,3}`; `question_template` es la lista de plantillas; `raw_input`/`raw_output` tienen `None` porque se calculan.

**Padres** (97-141), que es lo importante del grafo:
- entidades y variables de control: sin padres (son entradas);
- `question_template` ← `query_indices`, `answer_index`;
- `raw_input` ← todas las entidades + `statement_template`, `question_template`, `query_group`, `query_indices`, `active_groups`, `entities_per_group`;
- `raw_output` ← todas las entidades + **`query_group`** + `answer_index`, `active_groups`, `entities_per_group`.

**Mecanismos** (143-245):
- entidades: `random.choice(pool)` (ojo al truco `lambda t, pool=pool:` para que cada lambda capture su propio pool y no el último del loop);
- `active_groups`: `random.randint(2, max_groups)` (aunque los valores declarados incluyen 1, el mecanismo nunca samplea 1);
- `question_template`: busca `(query_indices, answer_index)` en la config (convierte lista a tupla porque al serializar un Dataset las tuplas se vuelven listas); si no existe devuelve `"What is the answer?"`;
- `raw_input` (`generate_raw_input`, 185-225): reconstruye una `BindingMatrix` con los `active_groups` primeros grupos (pone `MISSING_g_e` si falta alguna entidad), y usa `TemplateProcessor` para enunciado + pregunta + prefijo/sufijo; si algo falla devuelve `"Invalid configuration"`;
- `raw_output` (`generate_raw_output`, 230-243): **recuperación directa por índice**: si `query_group < active_groups` devuelve `trace[f"entity_g{query_group}_e{answer_index}"]`, si no `"UNKNOWN"`.

El modelo se construye con `CausalModel(variables, values, parents, mechanisms, id=model_id)` y el id es `entity_binding_direct_{k}g_{d}e`. La frase de la notebook "The causal model has NO intermediate variables" se refiere justamente a que no hay nada entre las entradas y `raw_output`; la hipótesis "directa" es que la red indexa el grupo por su posición.

#### Samplear entradas válidas

```python
sample_input = sample_valid_entity_binding_input(love_config)
print(f"Active groups: {sample_input['active_groups']}")
...
for g in range(sample_input["active_groups"]):
    ...
```

`sample_valid_entity_binding_input(config, ensure_positional_uniqueness=True)` (`causal_models.py` 579-708) es más cuidadoso que los mecanismos aleatorios del modelo. Hasta 100 intentos:

1. `active_groups = randint(2, max_groups)`, `query_group = randint(0, active_groups-1)` (así el grupo preguntado siempre está activo).
2. `query_indices`: `config.fixed_query_indices` si está fijado; si no, una tupla de un solo índice al azar.
3. `answer_index` al azar; si cae dentro de `query_indices` (preguntar y responder lo mismo) se descarta el intento (`continue`).
4. Si no existe plantilla para `(query_indices, answer_index)`, también se descarta.
5. Samplea entidades con dos restricciones: **distintas dentro del grupo** (`used_entities_per_group`) y, si `ensure_positional_uniqueness`, **distintas en la misma posición entre grupos** (`used_entities_per_position`): no puede haber dos personas iguales ni dos comidas iguales. Esto es necesario para que el modelo posicional (notebook 03) no tenga ambigüedad.
6. Grupos inactivos quedan en `None`.

Si los 100 intentos fallan, devuelve un ejemplo fijo de respaldo. Importante: lo que devuelve es un `dict` con **solo las variables de entrada** (no incluye plantillas ni `raw_input`).

En la ejecución guardada salió: 3 grupos activos, `query_group = 2`, patrón `(0,) → 1`, grupos `(Pete, soup)`, `(Kate, pie)`, `(Bob, tea)`.

#### Correr el modelo

```python
output = causal_model.new_trace(sample_input)
print(f"Generated prompt: '{output['raw_input']}'")
print(f"Expected answer: '{output['raw_output']}'")
query_entity_key = f"entity_g{sample_input['query_group']}_e{sample_input['query_indices'][0]}"
answer_entity_key = f"entity_g{sample_input['query_group']}_e{sample_input['answer_index']}"
print(f"Output matches answer entity: {output['raw_output'] == sample_input[answer_entity_key]}")
```

- `causal_model.new_trace(inputs)` crea una **traza** (`CausalTrace`, en `causalab/causal/trace.py`): fija las entradas dadas y calcula recursivamente todos los descendientes cuyos padres ya están disponibles. El resultado se indexa como diccionario (`output['raw_input']`).
- Salida guardada: `'Pete loves soup, Kate loves pie, and Bob loves tea. What does Bob love?'` → `'tea'`, y el chequeo de consistencia da `True`.
- El comentario "pass sample_input as an intervention" es lenguaje de la API vieja; en la API actual `new_trace` simplemente fija las entradas (no es una intervención en el sentido de romper mecanismos).

#### Varios ejemplos

La celda de "Multiple Examples" repite 5 veces: samplear, `new_trace`, imprimir prompt/respuesta y verificar que `raw_output == entity_g{query_group}_e{answer_index}`. En la salida guardada salen ambos tipos de pregunta ("Who loves tea?" → `Sue`, "What does Ann love?" → `tea`) y todos con ✓.

### Sistema "action" con el modelo causal

La celda 30 es un "deep dive" de la tarea action sin modelo causal: para 3 escenarios fijos (`Pete/jam/cup`, `Ann/water/box`, `Bob/book/table`) imprime el enunciado y las 9 preguntas posibles con su respuesta, usando `fill_question_template(template, 0, query_indices, matrix)` y `matrix.get_entity(0, answer_idx)`. Sirve para ver que las plantillas de dos entidades (`"Where did Pete put jam?"`) funcionan gracias al diccionario por rol.

```python
action_config.max_groups = 4
action_causal_model = create_direct_causal_model(action_config)
# id: entity_binding_direct_4g_3e, 21 variables
for i in range(3):
    sample = sample_valid_entity_binding_input(action_config)
    output = action_causal_model.new_trace(sample)
    ...
```

- Se sube `max_groups` a 4 mutando la config; el modelo queda con 4×3 = 12 variables de entidad + 9 = 21 variables (coincide con la salida).
- Detalle: `sample_valid_entity_binding_input` solo samplea `query_indices` de **un elemento** (salvo que `fixed_query_indices` diga otra cosa), así que las plantillas de dos entidades nunca se usan desde el sampler. Y `values["query_indices"]` del modelo también solo lista tuplas de un elemento.
- El loop que arma `entities` recorre primero `e` y luego `g`, por eso en la salida aparecen agrupadas por rol (`person: Bob, person: Ann, object: watch, ...`), no por grupo. El comentario `# Focus on first group` no corresponde a lo que hace.
- Salida guardada: `'Bob put watch in the table, and Ann put water in the box. What did Bob put somewhere?'` → `'watch'`. Nota que la plantilla produce frases poco naturales ("put watch in the table") porque no hay artículos para el objeto.

La celda termina imprimiendo una comparación (2 plantillas para love, 9 para action) y la observación combinatoria: con N roles hay N·(N-1) preguntas de una entidad, etc.

### "What We've Built"

Markdown final: tres componentes (estructuras de datos, plantillas, modelo causal) y los próximos pasos: contrafactuales (notebook 03), token positions (notebook 02) e integración con experimentos de intervención (notebook 04).

---

## Ejercicios

La notebook no tiene ejercicios ni celdas `# TERRON` / `FILL IN`. Si quieres practicar, cosas que se pueden verificar con el mismo código:
- Cambiar `fixed_query_indices = (1,)` y ver que todas las preguntas pasan a ser "Who loves X?".
- Llamar `sample_valid_entity_binding_input(config, ensure_positional_uniqueness=False)` y buscar casos con dos personas iguales: ese es exactamente el caso ambiguo que rompería al modelo posicional.

---

## Ojo / errores comunes / dudas

- **Código deprecado y API vieja.** `create_direct_causal_model` termina con `CausalModel(variables, values, parents, mechanisms, id=...)`, pero el constructor actual en `causalab/causal/causal_model.py` es `CausalModel(mechanisms, values, print_pos=None, id="null")`, donde `mechanisms` es un dict de objetos `Mechanism(parents, compute)` y los padres se derivan de ahí. Con la librería actual esa llamada fallaría (4 argumentos posicionales más `id=` por keyword → "multiple values for argument 'id'"). Las salidas guardadas son de la versión anterior; no las pude reproducir.
- **`statement_conjunction_function` no existe** en el `causal_utils.py` actual; `templates.py` la importa en la línea 14. El repo del curso la "parchea" en `utils/lecture_09_utils.py`. Si importas `templates.py` sin ese parche, falla el import.
- **`query_group` es un índice, no un nombre.** En el modelo directo `raw_output` depende de `query_group` y no de qué nombre aparece en la pregunta. Por eso las dos hipótesis (directa vs posicional) dan la misma respuesta en entradas normales y solo se separan bajo intervenciones (notebook 03).
- **`active_groups` en los valores incluye 1 pero ni el mecanismo ni el sampler generan 1.** No afecta a esta notebook, pero importa si alguien hace `sample_input()` del modelo crudo.
- **Mecanismos aleatorios del modelo vs sampler.** Si corrieras el modelo sin pasar entradas, los mecanismos samplean cada variable independientemente (por ejemplo `query_group` puede caer en un grupo inactivo y dar `"UNKNOWN"`, o `answer_index == query_index`). Por eso la notebook siempre usa `sample_valid_entity_binding_input`.
- **`fill_question_template` con varias query entities** pisa `query_entity` en cada vuelta; las plantillas de dos entidades funcionan solo porque usan los nombres de rol (`{person}`, `{object}`), no `{query_entity}`.

---

## Resumen de una línea por sección

- **Step 1 (estructuras):** `EntityBindingTaskConfig` es el plano; `EntityGroup`/`BindingMatrix` son una instancia concreta; inactivos devuelven `None`.
- **Step 2 (plantillas):** `TemplateProcessor` llena enunciados por grupo, los une con delimitadores, elige la pregunta por `(query_indices, answer_index)` y devuelve `(prompt, respuesta)`.
- **Action:** mismo sistema con 3 roles y 9 plantillas de pregunta (incluye preguntas que mencionan dos entidades).
- **Step 3 (modelo causal directo):** 15 variables, sin intermedias; `raw_output = entity_g{query_group}_e{answer_index}`.
- **Sampler:** `sample_valid_entity_binding_input` garantiza grupo activo, pregunta ≠ respuesta, plantilla existente y entidades únicas por grupo y por posición.
- **Ojo:** el código usa una API de `CausalModel` que ya no existe; las salidas son de una versión anterior.
