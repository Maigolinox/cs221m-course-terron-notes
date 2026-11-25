# General addition 02: token positions para dígitos (Llama vs Gemma)

Notebook: `causalab/causalab/tasks/general_addition/notebooks/02_token_positions_for_digits.ipynb`

Para intervenir "el dígito de las decenas de 23" en un LM hay que saber **qué token** contiene ese dígito, y eso depende del tokenizer. Esta notebook compara dos modelos: **Llama 3.1 8B Instruct**, que tokeniza números enteros como un solo token (hasta 3 dígitos), y **Gemma 2 9B**, que tokeniza dígito por dígito. Luego usa las funciones de `tasks/general_addition/token_positions.py` para construir los `TokenPosition` que usarían los experimentos. La pregunta que responde: ¿qué estoy interviniendo realmente cuando digo "intervengo `digit_0_1`" en cada modelo?

Spoiler importante: las funciones de token positions de esta tarea **devuelven todos los tokens del número** en ambos modelos; no aíslan un dígito individual ni siquiera en Gemma, aunque el texto de la notebook diga lo contrario. Las propias salidas guardadas lo muestran.

> Tarea **DEPRECATED**. Código: `tasks/general_addition/token_positions.py`, `tasks/general_addition/experiments/tokenization_config.py` y `causalab/causalab/neural/token_position_builder.py`. Apuntes generales: `../01_causal.md` y siguientes.

---

## Ideas clave / teoría

### Tokenización de números

- **Llama 3** usa un pre-tokenizador que separa los números en grupos de hasta 3 dígitos y separa el espacio previo. Por eso `" 23"` es `" "` + `"23"`, `" 123"` es `" "` + `"123"`, y `" 1234"` es `" "` + `"123"` + `"4"` (así aparece en la salida guardada de la Part 13).
- **Gemma 2** separa cada dígito en su propio token: `"23"` → `"2"`, `"3"`.

Consecuencia para causal abstraction: en Llama, el vector del token `"23"` mezcla ambos dígitos. Una intervención ahí cambia "el número completo", no un dígito; no puedes testear una hipótesis sobre `digit_0_1` sin tocar también `digit_0_0`. En Gemma en principio sí se puede, si eliges el token correcto.

### Padding a la izquierda

`LMPipeline` (`causalab/causalab/neural/pipeline.py`) pone `pad_token = eos_token` y `padding_side = "left"` por defecto, y si se le da `max_length` rellena todas las secuencias hasta esa longitud. Con `max_length=256`, una secuencia de ~10 tokens queda con ~246 tokens de padding **al principio** (`<|eot_id|>` en Llama Instruct, `<eos>` en Gemma) y el texto al final. Por eso los índices de los dígitos son ~249-251 y no ~4-5. Una ventaja del padding izquierdo: el último token real siempre está en la posición `max_length - 1`.

### `TokenPosition`

En causalab, un `TokenPosition` (`token_position_builder.py`, clase `TokenPosition(ComponentIndexer)`) es una función `input -> List[int]` + el pipeline + un `id`. Los experimentos lo usan para saber en qué posiciones aplicar la intervención en cada ejemplo (las posiciones cambian de ejemplo a ejemplo).

---

## Recorrido por la notebook

### Setup

```python
import torch
from causalab.neural.pipeline import LMPipeline
from causalab.tasks.general_addition.config import (create_two_number_two_digit_config,
    create_two_number_three_digit_config, create_general_config)
from causalab.tasks.general_addition.causal_models import create_basic_addition_model, sample_valid_addition_input
from causalab.tasks.general_addition.token_positions import get_digit_token_position, create_token_positions
from causalab.tasks.general_addition.experiments.tokenization_config import get_tokens_per_number, TOKENIZATION_CONFIG
```

Lo nuevo: las funciones de `token_positions.py` de la tarea y la tabla de `tokenization_config.py`.

### Part 1: cargar los modelos

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
llama_pipeline = LMPipeline("meta-llama/Meta-Llama-3.1-8B-Instruct", max_new_tokens=5, device=device,
                            dtype=torch.bfloat16 if device == "cuda" else torch.float32, max_length=256)
gemma_pipeline = LMPipeline("google/gemma-2-9b", max_new_tokens=5, device=device, dtype=..., max_length=256)
```

- `max_new_tokens=5`: cuántos tokens generaría en `generate` (no se usa en esta notebook).
- `dtype=bfloat16` en GPU para que dos modelos de 8-9B entren en memoria; `float32` en CPU.
- `max_length=256`: longitud de padding (ver arriba).
- Se cargan los modelos completos aunque solo se necesita el tokenizer (comentado en la markdown: "the pipeline includes the tokenizer").

### Part 2: el prompt de ejemplo

```python
config = create_two_number_two_digit_config()
model = create_basic_addition_model(config)
input_sample = {"digit_0_0": 2, "digit_0_1": 3, "digit_1_0": 4, "digit_1_1": 5,
                "num_addends": 2, "num_digits": 2, "template": config.templates[0]}
output = model.new_trace(input_sample)
prompt = output['raw_input']
input_sample["raw_input"] = output['raw_input']
```

Corre el modelo básico (notebook 01) para generar `'The sum of 23 and 45 is'` (respuesta `'68'`) y guarda `raw_input` dentro del sample, porque las funciones de token position lo leen de ahí.

### Parts 3-5: tokenización con cada modelo

```python
llama_token_dict = llama_pipeline.load({"raw_input": prompt}, add_special_tokens=False)
llama_tokens = llama_token_dict["input_ids"][0].tolist()
for i, token_id in enumerate(llama_tokens):
    token_str = llama_pipeline.tokenizer.decode([token_id])
    marker = " <-- FIRST NUMBER ..." if "23" in token_str else ...
    print(f"Token {i}: {repr(token_str):20s} (ID: {token_id}){marker}")
```

- `pipeline.load(...)` tokeniza; `input_ids` tiene shape `[1, 256]` por el padding; `[0].tolist()` da 256 ids.
- Se decodifica e imprime cada token. Las salidas guardadas están truncadas: solo se ven las primeras ~30-70 líneas de padding (`<|eot_id|>` con id 128009 en Llama, `<eos>` con id 1 en Gemma), así que **los tokens reales no aparecen en la salida guardada de estas celdas**. Los marcadores (`'23' in token_str`) son heurísticas de la celda, no del sistema.
- La celda de comparación lado a lado (Part 5) tiene el mismo problema y su markdown dice que Llama rellena a la derecha; con `padding_side="left"` por defecto (y la salida mostrando padding al principio en ambos), los dos rellenan a la izquierda.
- Ojo de API: aquí `load` recibe un `dict`; en el `pipeline.py` actual espera una lista de trazas (`[item["raw_input"] for item in input]`).

Lo que sí se ve de la tokenización real está en la Part 6 y en la Part 13 (highlights).

### Part 6: `get_digit_token_position`

```python
llama_digit_0_0 = get_digit_token_position(input_sample, llama_pipeline, 0, 0)   # [251]  -> ['23']
gemma_digit_0_0 = get_digit_token_position(input_sample, gemma_pipeline, 0, 0)   # [249, 250] -> ['2', '3']
llama_digit_0_1 = get_digit_token_position(input_sample, llama_pipeline, 0, 1)   # [251]
gemma_digit_0_1 = get_digit_token_position(input_sample, gemma_pipeline, 0, 1)   # [249, 250]
```

Qué hace `get_digit_token_position(input_sample, pipeline, addend_idx, digit_idx)` (`token_positions.py` 16-86), línea por línea:

1. Lee `raw_input`, `num_addends`, `num_digits`.
2. Junta los dígitos del sumando `addend_idx` y calcula su valor (`number_value`), luego `number_str = str(number_value)` (sin ceros a la izquierda).
3. Arma un `context` (`"of 23 and"`, `"and 45 is"`, `", 34,"`) "para que sea único"... pero **`context` nunca se usa** en el resto de la función.
4. `number_tokens = get_substring_token_ids(prompt, number_str, pipeline, add_special_tokens=False)`: tokens que se solapan con la **primera aparición** del string del número en el prompt.
5. Si es un solo token, lo devuelve; si son varios, **también los devuelve todos** (el comentario lo admite: "For now, we'll return all tokens for the number"). `digit_idx` no se usa para nada.

`get_substring_token_ids` (`token_position_builder.py` 204-331):
- busca todas las apariciones del substring con `text.find`, elige la `occurrence` pedida (0 por defecto);
- tokeniza el texto con `pipeline.load([trace], add_special_tokens=..., return_offsets_mapping=True)` (mismo padding que el pipeline, aquí 256 a la izquierda);
- `get_tokens_in_char_range(offsets, start, end)` (161-201) devuelve los índices de tokens cuyo rango de caracteres `(start, end)` se solapa con `[start_char, end_char)`, saltando los de padding (offset `(0, 0)`).

Salida guardada: Llama `[251]` → `'23'` para ambos dígitos; Gemma `[249, 250]` → `'2', '3'` **para ambos dígitos**. El texto impreso "Gemma: digit_0_0 and digit_0_1 return DIFFERENT tokens" es falso según la propia salida.

### Part 7: implicaciones

La celda imprime para `digit_0_1`: Llama `[251]` = `23` (correcto: es el número entero) y Gemma `[249, 250]` y dice "This token represents: 2", "We're intervening on JUST the ones digit!". Pero:
- decodifica `gemma_tokens[gemma_digit_0_1[0]]`, el **primer** token de la lista, que es `'2'`, el dígito de las **decenas**, no de las unidades;
- la intervención usaría **los dos** tokens.

O sea, en esta implementación intervenir `digit_0_1` en Gemma también es intervenir el número completo. La idea conceptual de la notebook (con Gemma se *podría* aislar un dígito) es válida, pero requiere cambiar `get_digit_token_position` para quedarse con `number_tokens[digit_idx]` cuando hay un token por dígito.

### Parts 8-9: 3 y 4 dígitos

Repiten la tokenización con `'The sum of 123 and 456 is'` y `'The sum of 1234 and 5678 is'` (generados con `create_two_number_three_digit_config` y `create_general_config(2, 4)`). Las salidas guardadas otra vez solo muestran padding. Las conclusiones impresas:
- Llama: 1 token para 3 dígitos; 2 tokens para 4 dígitos. El texto dice que se parte en `'12'` y `'34'`, pero el highlight de la Part 13 muestra `**123****4**`: se parte en `123` + `4`.
- Gemma: un token por dígito siempre.

### Part 10: tabla de configuración

```python
print(TOKENIZATION_CONFIG["llama"])   # {2: 1, 3: 1, 4: 2}
print(TOKENIZATION_CONFIG["gemma"])   # {2: 2, 3: 3, 4: 4}
for num_digits in [2, 3, 4]:
    get_tokens_per_number("meta-llama/Meta-Llama-3.1-8B-Instruct", num_digits)
    get_tokens_per_number("google/gemma-2-9b", num_digits)
```

`tokenization_config.py`:
- `TOKENIZATION_CONFIG`: tabla fija "familia → {dígitos: tokens por número}", medida empíricamente (comentario: "Based on empirical testing from analyze_tokenization.py"). Incluye también OLMo (igual que Llama).
- `get_model_family(name)`: busca `"llama"`, `"gemma"` u `"olmo"` en el nombre (minúsculas); si no, `ValueError`.
- `get_tokens_per_number(model_name, num_digits)`: consulta la tabla; error si no hay entrada.
- `get_all_number_token_indices(...)` (no se usa en la notebook): llama a `get_digit_token_position(..., addend_idx, 0)` y recorta a `tokens_per_number`. Es decir, la tabla sirve para saber cuántos tokens tiene un número, no para elegir un dígito.

### Part 11: `create_token_positions`

```python
llama_positions = create_token_positions(llama_pipeline, num_addends=2, num_digits=2)
print(list(llama_positions.keys()))
# ['digit_0_0', 'digit_0_1', 'digit_1_0', 'digit_1_1', 'delimiter_and', 'delimiter_is', 'last_token']
```

`create_token_positions(pipeline, num_addends, num_digits)` (`token_positions.py` 178-207) arma un diccionario nombre → `TokenPosition`:
- `digit_{k}_{d}`: `create_digit_token_position(pipeline, k, d)` → `TokenPosition(lambda x: get_digit_token_position(x, pipeline, k, d), pipeline, id=f"digit_{k}_{d}")`. Los `k, d` se capturan bien porque se pasan como argumentos a la función fábrica (no como variables del loop dentro de la lambda).
- `delimiter_and`, `delimiter_is` (y `delimiter_comma` si hay más de 2 sumandos): `get_delimiter_token_position` (108-140) busca `" and "`, `" is"` o `", "` con `get_substring_token_ids` (primera aparición).
- `last_token`: `get_last_token_index(x, pipeline)` (`token_position_builder.py` 97-100), que tokeniza y devuelve `[len(ids) - 1]`; con padding izquierdo es el último token real.

### Part 12: ejemplos aleatorios

Para 3 sumas aleatorias (`sample_valid_addition_input` + `model.new_trace` + copiar `raw_input`) se busca `digit_0_0` con ambos modelos y se decodifica el **primer** índice devuelto. Salida guardada: Llama encuentra `'42'` en `[251]`; Gemma encuentra `'4'` pero la lista es `[249, 250]`. "✓ Token position functions work correctly" es un print fijo; lo que se verifica es que el primer token cae en el número correcto.

### Part 13: resaltar los tokens seleccionados

```python
gemma_positions_2d = create_token_positions(gemma_pipeline, num_addends=2, num_digits=2)
for digit_name in ["digit_0_0", "digit_0_1", "digit_1_0", "digit_1_1"]:
    highlighted = gemma_positions_2d[digit_name].highlight_selected_token(input_sample)
```

`TokenPosition.highlight_selected_token(input)` (`token_position_builder.py` 71-94): tokeniza con `pipeline.load([input])` (con tokens especiales por defecto), llama a `self.index(input)` para obtener los índices, y reconstruye el texto decodificando cada token no-padding, envolviendo en `**...**` los seleccionados.

Salidas guardadas (las más informativas de la notebook):
- Gemma 2 dígitos: `digit_0_0` y `digit_0_1` → `The sum of **2****3** and 45 is` (los **dos** tokens en ambos casos).
- Gemma 3 y 4 dígitos: igual, todos los dígitos del número resaltados para cualquier `digit_idx`.
- `delimiter_and` → `23** and**** **45`: se resaltan `" and"` y un token `" "` (el espacio antes del número es su propio token, en ambos modelos).
- `delimiter_is` y `last_token` → `** is**`: coinciden (el último token es " is").
- Llama 2 y 3 dígitos: `**23**`, `**123**` para cualquier dígito; 4 dígitos: `**123****4**`.

Detalle sutil: `get_substring_token_ids` se llama con `add_special_tokens=False`, pero `highlight_selected_token` tokeniza **con** `<bos>`. Los índices coinciden igual porque, con padding a la izquierda hasta 256, agregar un token al principio del texto solo come un token de padding y no corre las posiciones del final. Con padding a la derecha o sin `max_length`, los índices quedarían desfasados en 1.

### Summary (markdown)

Tabla de tokens por número (Llama 1/1/2, Gemma 2/3/4) y la conclusión: con Llama las intervenciones "por dígito" son en realidad por número; con Gemma se podría intervenir dígito por dígito. Como se vio, con el código actual de `get_digit_token_position` tampoco en Gemma se aísla un dígito.

---

## Ejercicios

No hay ejercicios ni celdas `# TERRON` / `FILL IN`. Para practicar: modificar (en tu copia) `get_digit_token_position` para devolver `[number_tokens[digit_idx]]` cuando `len(number_tokens) == num_digits` y comprobar con `highlight_selected_token` que en Gemma `digit_0_1` resalta solo el `3`.

---

## Ojo / errores comunes / dudas

- **`digit_idx` y `context` no se usan** en `get_digit_token_position`: siempre se devuelven todos los tokens del número.
- **Primera aparición del string.** Se busca `str(number)` sin contexto. Si los dos sumandos son iguales ("The sum of 23 and 23 is"), `digit_1_*` apuntaría al **primer** 23. Lo mismo si el número del segundo sumando aparece antes como substring (p. ej. sumandos 12 y 2: el "2" se encuentra dentro de "12"). El `context` estaba pensado para esto, pero no se aplica.
- **Textos de la notebook vs salidas.** "Gemma returns DIFFERENT tokens", "This token represents: 2 ... JUST the ones digit", "Llama right-padding" y "'12' and '34'" no coinciden con las salidas guardadas.
- **Las celdas de tokenización completas no muestran los tokens reales** (salida truncada en el padding).
- **API vieja de `pipeline.load`** con dict en vez de lista de trazas; las funciones de `token_positions.py` usan la API de `token_position_builder.py`, que sí está actualizada.
- **Cargar modelos de 8-9B** solo para mirar tokenizers es caro; con `AutoTokenizer.from_pretrained` bastaría para las Parts 3-9 (pero `TokenPosition` pide un pipeline).

---

## Resumen de una línea por sección

- **Part 1:** se cargan Llama 3.1 8B y Gemma 2 9B con padding izquierdo hasta 256.
- **Part 2:** el modelo básico genera `'The sum of 23 and 45 is'`.
- **Parts 3-5:** tokenización completa; las salidas guardadas solo muestran padding.
- **Part 6:** `get_digit_token_position` devuelve todos los tokens del número: Llama `[251]` (`'23'`), Gemma `[249, 250]` (`'2','3'`) para cualquier dígito.
- **Part 7:** la idea de aislar dígitos en Gemma es correcta, pero el código no lo hace.
- **Parts 8-9:** Llama usa 1 token hasta 3 dígitos y parte 4 dígitos en `123`+`4`; Gemma siempre dígito a dígito.
- **Part 10:** `TOKENIZATION_CONFIG` es una tabla fija de tokens por número por familia de modelo.
- **Part 11:** `create_token_positions` arma `TokenPosition`s para dígitos, delimitadores y último token.
- **Part 12-13:** los highlights confirman qué tokens se intervendrían realmente (todo el número).
