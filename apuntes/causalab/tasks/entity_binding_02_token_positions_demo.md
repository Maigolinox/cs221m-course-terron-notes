# Entity binding 02: token positions conscientes de la plantilla

Notebook: `causalab/causalab/tasks/entity_binding/notebooks/02_token_positions_demo.ipynb`

Para hacer una interchange intervention en un transformer hay que decir **dónde** intervenir: en qué capa y en qué **posición de token**. Esta notebook muestra el sistema de token positions de la tarea entity binding, que resuelve un problema concreto: en *"Pete loves jam, and Ann loves pie. What does Pete love?"* el nombre "Pete" aparece **dos veces** (en el enunciado y en la pregunta), y buscar el substring "Pete" sin más siempre devolvería la primera aparición. La solución es **parsear el prompt usando las plantillas** que lo generaron, de modo que cada trozo de texto sabe qué es (entidad, texto de plantilla, delimitador, texto de pregunta), a qué grupo/rol pertenece y en qué región está (enunciado o pregunta). La pregunta que responde la notebook: ¿cómo obtengo los índices de token exactos de "la entidad g, e en la región R"?

> La tarea está marcada **DEPRECATED**. El código relevante es `causalab/causalab/tasks/entity_binding/token_positions.py`. Las salidas guardadas son de una versión anterior de la librería (ver "Ojo"). Contexto general de token positions en causalab: `causalab/causalab/neural/token_position_builder.py` (y los apuntes generales, p. ej. `../01_causal.md` y siguientes).

---

## Ideas clave / teoría

### Por qué importa la posición

En un transformer cada token tiene su vector en el **residual stream** (la "corriente" de activaciones que cada capa lee y a la que suma su salida). Una hipótesis del tipo "la red guarda el ligado Pete↔jam en la posición de jam" solo se puede testear interviniendo el vector de esa posición concreta. Si te equivocas de token (por ejemplo intervienes el "Pete" de la pregunta en vez del del enunciado) estás testeando otra hipótesis: el "Pete" del enunciado es donde se **escribe** la información del ligado, el de la pregunta es donde se **consulta**.

### Tokenización: caracteres ≠ tokens

El prompt es una cadena de caracteres, pero el modelo ve tokens (BPE en GPT-2). Una entidad puede ocupar uno o varios tokens, y un token puede incluir el espacio previo (`' Pete'`). Por eso el sistema trabaja en dos pasos:
1. **Parseo en caracteres:** reconstruye, a partir de las plantillas, qué rango de caracteres `[char_start, char_end)` ocupa cada segmento.
2. **Mapeo carácter → token:** decodifica el prompt token por token y asigna a cada carácter el índice del token que lo generó. Los tokens de un segmento son los tokens de todos sus caracteres.

Ejemplo de la salida guardada (GPT-2): `"Pete"` al inicio del texto se tokeniza como `'P'` + `'ete'` (tokens 0 y 1), mientras que `" Pete"` en la pregunta es un solo token (12). Esto pasa porque en el vocabulario de GPT-2 existe `" Pete"` (con espacio) pero no `"Pete"` sin espacio inicial.

---

## Recorrido por la notebook

### Setup

```python
from causalab.tasks.entity_binding.config import create_sample_love_config, create_sample_action_config
from causalab.tasks.entity_binding.causal_models import create_direct_causal_model, sample_valid_entity_binding_input
from causalab.tasks.entity_binding.token_positions import (
    PromptParser, PromptTokenizer, get_entity_token_indices_structured,
)
from causalab.neural.pipeline import LMPipeline
```

Lo nuevo son las tres piezas de `token_positions.py` y `LMPipeline`, el wrapper de causalab sobre un modelo de HuggingFace + su tokenizer (`causalab/causalab/neural/pipeline.py`). La salida muestra el aviso "nnsight is not detected", que es irrelevante acá (solo se usa el tokenizer).

### Part 1: el problema de ambigüedad

```python
pipeline = LMPipeline("gpt2")
config = create_sample_love_config()
```

- `LMPipeline("gpt2")` carga GPT-2 small y su tokenizer. En el código actual (`pipeline.py`, `_setup_model`) además pone `pad_token = eos_token` y `padding_side = "left"` por defecto.

```python
input_sample = {
    "entity_g0_e0": "Pete", "entity_g0_e1": "jam",
    "entity_g1_e0": "Ann",  "entity_g1_e1": "pie",
    "query_group": 0, "query_indices": (0,), "answer_index": 1,
    "active_groups": 2, "entities_per_group": 2,
    "raw_input": "Pete loves jam, and Ann loves pie. What does Pete love?",
}
```

Un ejemplo escrito a mano (no sale del modelo causal) donde el grupo preguntado es el 0 y la entidad preguntada es Pete, que aparece dos veces.

```python
tokens = pipeline.load(input_sample)["input_ids"][0].tolist()
for i, token_id in enumerate(tokens):
    token_str = pipeline.tokenizer.decode([token_id])
    ...
    print(f"Token {i:2d}: {repr(token_str):20s}{marker}")
```

- `pipeline.load(...)` tokeniza `raw_input` y devuelve el diccionario de HF (`input_ids`, `attention_mask`, ...) con tensores de shape `[batch, seq_len]`; `[0]` toma el único ejemplo y `.tolist()` lo pasa a lista de ints.
- Se decodifica cada id por separado para ver el texto de cada token. Los marcadores de "FIRST Pete" en 0-1 y "SECOND Pete" en 12 están **hardcodeados** en la celda (`if i in [0, 1]`, `elif i == 12`); todavía no los calcula ningún sistema.
- Salida guardada: 15 tokens, `'P' 'ete' ' loves' ' jam' ',' ' and' ' Ann' ' loves' ' pie' '.' ' What' ' does' ' Pete' ' love' '?'`.

### Part 2: parseo consciente de la plantilla

```python
parser = PromptParser(config)
parsed = parser.parse_prompt(input_sample)
print(f"  Total segments: {len(parsed.segments)}")      # 11
print(f"  Statement region: chars {parsed.statement_region}")   # (0, 34)
print(f"  Question region: chars {parsed.question_region}")     # (35, 55)
```

Las estructuras (`token_positions.py`):

- `PromptSegment` (dataclass, líneas 26-53): `segment_type` (`'template_text'`, `'delimiter'`, `'entity'`, `'question_text'`), `text`, `char_start`, `char_end` (exclusivo), y para entidades `group_idx`, `entity_idx`; para delimitadores `delimiter_idx`; y `token_positions` (se llena después).
- `ParsedPrompt` (56-149): `raw_text`, lista de `segments`, `statement_region` y `question_region` como rangos de caracteres.

Qué hace `PromptParser.parse_prompt` (159-217), paso a paso:

1. Toma `raw_input` del sample (si no está, error).
2. `_extract_binding_matrix` (219-237) reconstruye una `BindingMatrix` con las `entity_g{g}_e{e}` del sample (`None` si faltan).
3. `_parse_statement` (239-279): para cada grupo llena la plantilla del enunciado y luego une todo con `statement_conjunction_function` (la misma que usa `TemplateProcessor`), obteniendo el texto completo del enunciado. Después llama a `_parse_statement_with_delimiters`:
   - recorre los enunciados llenos en orden; para el primero busca la versión con mayúscula inicial (porque la conjunción capitaliza);
   - `full_statement.find(search_text, char_pos)` encuentra dónde empieza cada enunciado; lo que haya entre el final del anterior y este es un segmento **delimitador** (`", and "`);
   - dentro de cada enunciado, `_parse_filled_statement_at_position` (417-483) busca cada entidad con `filled_text.find(entity_value, current_pos)`, ordena por posición y crea segmentos `entity` y `template_text` (`" loves "`) con offsets absolutos (`char_offset = idx`);
   - lo que sobra al final (el `"."`) es el último delimitador.
4. Si el sample trae `query_group` y `query_indices`, `_parse_question` (485-523) vuelve a elegir y llenar la plantilla de pregunta con `TemplateProcessor`, arma el diccionario rol→entidad (más `query_entity`) y `_parse_question_template` (525-597) recorre los placeholders `{...}` de la plantilla: el texto antes de cada placeholder es `question_text`, el placeholder es una `entity` con `group_idx = query_group` y `entity_idx` deducido del rol (o de `query_indices[0]` si es `{query_entity}`).
5. Los offsets de la pregunta se desplazan a coordenadas del prompt completo sumando `len(statement_text) + 1` (**asume que el separador es un solo espacio**; ver "Ojo").
6. Regiones: enunciado `(0, len(statement))`, pregunta `(q_start, q_start + len(question))`.

Con el ejemplo: el enunciado `"Pete loves jam, and Ann loves pie."` tiene 34 caracteres → región `(0, 34)`; la pregunta empieza en 35 y tiene 20 caracteres → `(35, 55)`. Coincide con la salida.

```python
for i, seg in enumerate(parsed.segments):
    in_statement = parsed.statement_region and parsed.statement_region[0] <= seg.char_start < parsed.statement_region[1]
    in_question = parsed.question_region and parsed.question_region[0] <= seg.char_start < parsed.question_region[1]
    region = "STMT" if in_statement else ("QUES" if in_question else "???")
    print(f"{i:2d}. [{region}] {seg.segment_type:15s} | '{seg.text}'")
```

Clasifica cada segmento según dónde cae su `char_start`. Salida guardada: 11 segmentos, 8 en el enunciado (`Pete`, `' loves '`, `jam`, `', and '`, `Ann`, `' loves '`, `pie`, `'.'`) y 3 en la pregunta (`'What does '`, `Pete`, `' love?'`). Los dos `Pete` son segmentos distintos con el mismo `group_idx=0, entity_idx=0`.

### Tokenización de los segmentos

```python
tokenizer = PromptTokenizer(pipeline)
parsed = tokenizer.tokenize_prompt(parsed)
for seg in parsed.segments:
    if seg.segment_type == "entity":
        print(f"Entity g{seg.group_idx}_e{seg.entity_idx} ('{seg.text}'): tokens {seg.token_positions}")
```

`PromptTokenizer.tokenize_prompt` (`token_positions.py` 606-642):

1. Envuelve el texto en una `CausalTrace` mínima con una sola variable `raw_input` (así `pipeline.load` acepta el formato que espera, una lista de trazas).
2. `pipeline.load([trace], add_special_tokens=False)["input_ids"][0]`: ids sin BOS/EOS.
3. `_build_char_to_token_map` (644-668): para cada `token_idx`, decodifica `token_ids[:token_idx+1]` y marca los caracteres nuevos (desde `len(reconstructed)` hasta `len(decoded_so_far)`) como pertenecientes a ese token. Es un mapeo carácter→token construido por decodificación incremental.
4. Para cada segmento, junta los tokens de todos sus caracteres y los guarda ordenados en `segment.token_positions`.

Salida: `Pete` del enunciado → `[0, 1]`, `jam` → `[3]`, `Ann` → `[6]`, `pie` → `[8]`, `Pete` de la pregunta → `[12]`. Fíjate que `jam` (caracteres 11-13) cae en el token `' jam'` (que incluye el espacio en el carácter 10): como se toma cualquier token que toque el rango, el resultado es el token correcto.

### Part 3: consultas por región

```python
pete_statement = get_entity_token_indices_structured(input_sample, pipeline, config, 0, 0, region="statement")  # [0, 1]
pete_question  = get_entity_token_indices_structured(input_sample, pipeline, config, 0, 0, region="question")   # [12]
pete_first     = get_entity_token_indices_structured(input_sample, pipeline, config, 0, 0, region=None)         # [0, 1]
assert pete_statement != pete_question
```

`get_entity_token_indices_structured(input_sample, pipeline, config, group_idx, entity_idx, region=None, token_idx=None)` (673-725) es la API de alto nivel: crea un `PromptParser`, parsea, tokeniza y llama a `parsed.get_entity_tokens(group_idx, entity_idx, region)`.

`ParsedPrompt.get_entity_segments` (67-106) filtra segmentos de tipo entidad con ese `(group_idx, entity_idx)` y, si `region` es `'statement'` o `'question'`, exige que `char_start` caiga dentro de la región. `get_entity_tokens` (108-146) toma el **primer** segmento que matchea (por eso `region=None` devuelve la primera aparición, la del enunciado) y lanza `ValueError` si no hay ninguno.

Ojo que cada llamada a `get_entity_token_indices_structured` vuelve a parsear y tokenizar todo; para datasets grandes esto es lento, pero es simple.

### Part 4: verificación visual

Repite la consulta para las dos regiones y recorre los tokens imprimiendo `PETE(statement)` o `PETE(question)` si el índice está en la lista correspondiente. Ahora los marcadores **sí** vienen del sistema (no hardcodeados como en la Part 1) y coinciden: 0-1 y 12.

### Part 5: todas las entidades

Lista de tuplas `(group_idx, entity_idx, nombre, región)` y, para cada una, `get_entity_token_indices_structured` dentro de `try/except ValueError`. Salida guardada: Pete `[0, 1]`, jam `[3]`, Ann `[6]`, pie `[8]` en el enunciado y Pete `[12]` en la pregunta. Nota que para `Ann` en la pregunta lanzaría `ValueError`, porque la pregunta solo menciona entidades del `query_group`.

### Part 6: entidades multi-token

```python
multi_token_sample = {
    "entity_g0_e0": "Elizabeth", "entity_g0_e1": "strawberry jam",
    "entity_g1_e0": "Margaret",  "entity_g1_e1": "apple pie", ...,
    "raw_input": "Elizabeth loves strawberry jam, and Margaret loves apple pie. What does Elizabeth love?",
}
jam_tokens = get_entity_token_indices_structured(multi_token_sample, pipeline, config, 0, 1, region="statement")   # [2, 3]
first_token = get_entity_token_indices_structured(multi_token_sample, pipeline, config, 0, 1, region="statement", token_idx=0)  # [2]
```

- `"strawberry jam"` ocupa dos tokens (`' strawberry'`, `' jam'`), índices 2 y 3.
- `token_idx=k` devuelve solo `[token_positions[k]]` (con chequeo de rango). Es útil para hipótesis tipo "la información está en el **último** token de la entidad" (`token_idx=-1` no está permitido por el chequeo `token_idx < 0`, así que habría que usar `len-1`).
- Las entidades no salen de los pools de la config, pero el parser no lo necesita: trabaja con los valores del sample.

### Part 7: tarea action (3 entidades)

```python
action_config = create_sample_action_config()
action_model = create_direct_causal_model(action_config)
action_sample = sample_valid_entity_binding_input(action_config)
action_output = action_model.new_trace(action_sample)
action_sample['raw_input'] = action_output['raw_input']
```

- El sampler devuelve solo entradas, así que hay que correr el modelo causal y copiar `raw_input` al sample (el parser lo necesita).
- Salida guardada: `"Kate put watch in the box, Dan put coin in the basket, and Pete put key in the pocket. Where was watch put?"` → `box`.

```python
action_parser = PromptParser(action_config)
action_parsed = action_parser.parse_prompt(action_sample)
action_parsed = PromptTokenizer(pipeline).tokenize_prompt(action_parsed)
...
for e in range(action_config.max_entities_per_group):
    tokens = get_entity_token_indices_structured(action_sample, pipeline, action_config, query_group, e, region="statement")
```

21 segmentos, región del enunciado `(0, 86)`, de la pregunta `(87, 107)`; las tres entidades del grupo 0 (`Kate` [0], `watch` [2], `box` [5]). Aquí la pregunta usa `{object}` (rol), y `_parse_question_template` deduce `entity_idx=1` buscando el rol en `entity_roles`.

### Part 8: metadata del parseo

Usa el `parsed` del ejemplo de Pete para listar entidades (5), delimitadores (2: `', and '` en chars 14-20 y `'.'` en 33-34) y los segmentos de `get_entity_segments(0, 0)` (sin filtro de región) con su región y tokens. Muestra que la estructura sirve para más que entidades (por ejemplo, intervenir en el delimitador `"."`, que es una posición típica donde los LMs "resumen" información).

### Part 9: ejemplos aleatorios

Para 5 samples del modelo directo: copia `raw_input`, y obtiene los tokens de la entidad preguntada (`query_group`, `query_indices[0]`) en enunciado y en pregunta. Salida guardada, por ejemplo: `"Kate loves tea, Tim loves jam, and Ann loves pie. What does Ann love?"` → enunciado `[9]`, pregunta `[15]`. En el ejemplo 4, `Sue` al principio del texto sale como `[0, 1]` (otra vez la división por falta de espacio inicial).

### Part 10: manejo de errores

```python
simple_sample = {"entity_g0_e0": "Pete", "entity_g0_e1": "jam", "active_groups": 1,
                 "entities_per_group": 2, "raw_input": "Pete loves jam."}
statement_tokens = get_entity_token_indices_structured(simple_sample, pipeline, config, 0, 0, region="statement")  # [0, 1]
try:
    get_entity_token_indices_structured(simple_sample, pipeline, config, 0, 0, region="question")
except ValueError as e:
    print(e)   # Entity g0_e0 not found in question region
```

Sin `query_group` en el sample, `parse_prompt` no parsea pregunta, `question_region` queda en `None` y `get_entity_segments` salta todos los segmentos para `region='question'` → lista vacía → `ValueError` desde `get_entity_tokens`.

### Summary (markdown)

Resume las ventajas (sin ambigüedad, metadata, basado en plantillas) y deja un snippet de uso: tokens del enunciado "para testear la representación" y de la pregunta "para testear la recuperación".

---

## Cómo se usaría en un experimento

`get_entity_token_indices_structured` devuelve una lista de índices; para usarla en causalab hay que envolverla en un `TokenPosition` (`causalab/causalab/neural/token_position_builder.py`, clase `TokenPosition(ComponentIndexer)`), que es básicamente "una función `input -> List[int]` más el pipeline y un id". Algo como:

```python
TokenPosition(lambda x: get_entity_token_indices_structured(x, pipeline, config, 0, 1, region="statement"),
              pipeline, id="g0_e1_statement")
```

La notebook no lo hace explícitamente; la notebook 04 habla de "last token of e1 entities" pero tampoco construye el `TokenPosition`. (Los scripts de `tasks/entity_binding/experiments/` probablemente lo hacen, pero no los revisé en detalle.)

Ojo con el padding: estos índices se calculan con `add_special_tokens=False` y sin padding (GPT-2 sin `max_length`). Si en el experimento el pipeline rellena a la izquierda hasta `max_length`, los índices absolutos se corren. Las funciones de causalab que trabajan con offsets (`get_substring_token_ids`) llaman a `pipeline.load` con el mismo padding que el experimento, cosa que este parser no garantiza.

---

## Ejercicios

No hay ejercicios ni celdas `# TERRON` / `FILL IN`.

---

## Ojo / errores comunes / dudas

- **API vieja de `pipeline.load`.** La Part 1 hace `pipeline.load(input_sample)` con un `dict`. En el `pipeline.py` actual, `load` espera una **lista de trazas** (`[item["raw_input"] for item in input]`); con un dict iteraría sobre las claves y fallaría. `PromptTokenizer` sí usa la forma nueva (`[trace]`).
- **`statement_conjunction_function`** se importa de `causalab.causal.causal_utils`, donde ya no existe (igual que en la notebook 01).
- **Separador de una sola posición.** `parse_prompt` suma `len(statement_text) + 1` para ubicar la pregunta, asumiendo `statement_question_separator = " "` y sin `prompt_prefix`. En la notebook 04 se usa prefijo `"We will ask a question about the following sentences.\n\n"` y separador `"\n\n"`: con esa config, los offsets de caracteres del parser **no** coincidirían con el `raw_input` real (el enunciado no empieza en el carácter 0 y la pregunta empieza 2 caracteres después). Habría que revisar ese caso antes de confiar en los índices.
- **`find` del texto.** `_parse_filled_statement_at_position` busca cada entidad con `find` desde `current_pos`, pero `current_pos` no se actualiza dentro de ese primer loop (vale siempre 0). Si una entidad fuera substring de la plantilla o de otra entidad del mismo enunciado (p. ej. una comida llamada "love"), podría encontrar la posición equivocada. Con los pools de la notebook no pasa.
- **Primera letra capitalizada.** El parser asume que la conjunción pone mayúscula al primer enunciado; si la entidad inicial empieza con minúscula (p. ej. otra plantilla que empiece por comida), el texto de la entidad en el prompt sería distinto del valor.
- **Los marcadores de la Part 1 están hardcodeados**; no confundirlos con la salida del sistema.
- **Índices de token dependen del tokenizer.** Todo lo visto es para GPT-2; con otro modelo "Pete" al inicio podría ser un token, o el BOS desplazaría todo en 1 si se tokeniza con `add_special_tokens=True`.

---

## Resumen de una línea por sección

- **Part 1:** "Pete" aparece dos veces; en GPT-2 el primero es `'P'+'ete'` (tokens 0-1) y el segundo `' Pete'` (12).
- **Part 2:** `PromptParser` reconstruye el prompt desde las plantillas y lo divide en segmentos tipados con rangos de caracteres y regiones.
- **Tokenización:** `PromptTokenizer` mapea caracteres a tokens por decodificación incremental y asigna tokens a cada segmento.
- **Part 3-5:** `get_entity_token_indices_structured(..., region=...)` distingue la entidad del enunciado de la de la pregunta; `region=None` = primera aparición.
- **Part 6:** entidades multi-token devuelven varios índices; `token_idx` elige uno.
- **Part 7:** funciona igual con la tarea action y plantillas por rol.
- **Part 8:** el parseo también expone delimitadores y regiones.
- **Part 9-10:** prueba con ejemplos aleatorios y errores claros cuando la entidad no está en la región pedida.
