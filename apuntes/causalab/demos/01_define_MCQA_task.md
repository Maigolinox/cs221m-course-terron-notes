# 01 — Definir la tarea MCQA: modelo causal y datasets contrafácticos (`01_define_MCQA_task.ipynb`)

Notebook: `causalab/demos/onboarding_tutorial/01_define_MCQA_task.ipynb`.

Primera notebook del tutorial de onboarding (siete notebooks que se encadenan). Todavía **no se carga ningún LLM**. Se define:
1. una tarea trivial de *multiple choice question answering* (MCQA): "The cauliflower is white. What color is the cauliflower? A. yellow / B. white / Answer:" → " B";
2. un **modelo causal** con una única variable intermedia, `answer_position` (en qué posición está la respuesta correcta), que alimenta a `answer` (qué letra es);
3. **pares original/contrafáctico** y una herramienta (`can_distinguish_with_dataset`) para medir si un dataset de pares sirve para distinguir una variable de otra.

La pregunta que responde: ¿cómo diseño contrafácticos que me permitan, más adelante, localizar `answer` y `answer_position` dentro de la red por separado?

Viene de [causal_model_demo.md](causal_model_demo.md) (la API de `CausalModel`/`Mechanism`/interchange). Sigue en [02_trace_residual_stream.md](02_trace_residual_stream.md), donde estas mismas interchange interventions se aplican al residual stream de Llama-3.2-1B. Referencia de la librería: [../01_causal.md](../01_causal.md).

---

## Ideas clave / teoría

**Hipótesis posicional.** El modelo causal dice: el LM primero averigua *en qué posición* (0 o 1) está el color correcto, y después usa esa posición para "desreferenciar" el símbolo (la letra) que está en esa posición. Grafo:

```
color, choice0, choice1 ──► answer_position ─┐
                     symbol0, symbol1 ───────┴─► answer ──► raw_output
template, object, color, symbol*, choice* ──► raw_input
```

**Interchange intervention en el modelo causal.** Con base `b` y contrafáctico `s`, `IntInv(b, s, answer_position)` = correr `b` pero con `answer_position := answer_position(s)`. Como `answer = symbol_{answer_position}` usa los **símbolos del base**, el resultado es "la letra del base que está en la posición donde estaba la respuesta en `s`". En cambio `IntInv(b, s, answer)` copia directamente la **letra** de `s`.

**Confounding (confusión) entre variables.** Un par (b, s) *confunde* dos hipótesis H1 y H2 si la interchange sobre H1 y la interchange sobre H2 producen la **misma salida**. Si eso pasa, cuando veamos que un patch en la red produce esa salida, no vamos a saber cuál de las dos variables vive ahí. Caso particular: si la interchange sobre una variable produce la misma salida que **no intervenir**, el par no da evidencia sobre esa variable (cualquier lugar irrelevante de la red "acertaría").

**Poder discriminativo de un dataset.** Para un dataset de pares, la proporción de pares que *sí* distinguen H1 de H2:

  `proportion = (1/N) · #{ i : salida(IntInv(b_i, s_i, H1)) ≠ salida(IntInv(b_i, s_i, H2)) }`

donde "H2 = None" significa comparar contra la salida sin intervención, `salida(b_i)`. Esto es exactamente lo que calcula `CausalModel.can_distinguish_with_dataset`. Un dataset bueno para localizar `answer` tiene proporción 1 contra `None` y contra `answer_position`.

**El modelo causal y los contrafácticos son el "experimento".** Todo esto se diseña sin red neuronal. Si la hipótesis es correcta, los datasets bien diseñados van a encontrar representaciones de las variables en la red; si no, los resultados sugieren hipótesis nuevas.

---

## El código de la librería que usa esta notebook

### `causalab/causalab/tasks/MCQA/causal_models.py`

- Constantes: `OBJECTS` (10 objetos), `COLORS` (10 colores), `NUM_CHOICES = 2`, `ALPHABET = "A..Z"`, y un único template:
  ```
  "The {object} is {color}. What color is the {object}?\n{symbol0}. {choice0}\n{symbol1}. {choice1}\nAnswer:"
  ```
- `values`: `choice0/1 ∈ COLORS`, `symbol0/1 ∈ ALPHABET`, `answer_position ∈ {0,1}`, `answer ∈ ALPHABET`, `template`, `object`, `color`, `raw_input/raw_output = None`.
- Mecanismos:
  - inputs (con `input_var`): `template`, `object`, `color`, `symbol0`, `symbol1`, `choice0`, `choice1`.
  - `raw_input` = `_fill_template(t)`: reemplaza `{object}`, `{color}`, `{symbol_i}`, `{choice_i}` en el template.
  - `answer_position` = `_get_answer_position(t)`: el índice i tal que `choice_i == color`; si ninguna opción es el color, **lanza `ValueError`**.
  - `answer` = `_get_answer(t)` = `t[f"symbol{answer_position}"]`.
  - `raw_output` = `" " + answer` (con espacio adelante, porque el LM genera el token " B", no "B").
- `positional_causal_model = CausalModel(mechanisms, values, id="2_answer_MCQA")`.

### `causalab/causalab/tasks/MCQA/counterfactuals.py`

- `sample_answerable_question()`: arma a mano un input donde el color correcto **sí** está entre las opciones (por eso no usa `sample_input()`, que podría samplear opciones sin el color y hacer que `_get_answer_position` explote). Samplea objeto, color, un color distractor distinto, una posición al azar para el correcto, y dos símbolos **distintos** con `random.sample(ALPHABET, 2)`. Devuelve `positional_causal_model.new_trace(input_dict)`, un `CausalTrace` con todo calculado.
- `different_symbol()`: base = pregunta answerable; contrafáctico = mismo input pero con **dos símbolos nuevos** sacados del complemento de los símbolos del base. Misma posición, letra distinta.
- `same_symbol_different_position()` (se usa desde la 03): intercambia choices **y** símbolos entre la posición correcta y la otra. Resultado: la letra correcta es la misma, pero está en la otra posición.
- `random_counterfactual()`: dos preguntas answerable independientes.
- Todos devuelven un `CounterfactualExample`: `{"input": trace, "counterfactual_inputs": [trace_cf]}` (un `TypedDict` definido en `causalab/causalab/causal/counterfactual_dataset.py`).

### `CausalModel.run_interchange` y `can_distinguish_with_dataset` (`causalab/causalab/causal/causal_model.py`)

- `run_interchange(input_trace, {var: cf_trace})`: `trace = input_trace.copy()` y por cada var `trace.intervene(var, cf_trace[var])`. Devuelve el trace intervenido.
- `can_distinguish_with_dataset(examples, vars1, vars2)`: para cada ejemplo exige **un solo** contrafáctico (`assert len(counterfactual_traces) == 1`), corre `setting1 = run_interchange(trace, {v: cf for v in vars1})`; si `vars2` no es `None` corre `setting2` igual y cuenta el ejemplo si `setting1["raw_output"] != setting2["raw_output"]`; si es `None`, compara contra `trace["raw_output"]`. Devuelve `{"proportion": count/N, "count": count}` y loguea en nivel DEBUG.

---

## Recorrido por la notebook

### Celda 1 — imports y logging

```python
%load_ext autoreload
%autoreload 2

import logging
import random

from causalab.causal.causal_viz import print_structure
from causalab.tasks.MCQA.causal_models import positional_causal_model
from causalab.tasks.MCQA.counterfactuals import sample_answerable_question

logging.basicConfig(level=logging.INFO)
logging.getLogger("causal.causal_model").setLevel(logging.DEBUG)
```

- `autoreload 2`: recarga módulos modificados antes de cada celda (útil si editás la librería).
- `print_structure` (en `causal_viz.py`): dibujo estático con NetworkX + matplotlib usando `model.print_pos`.
- La intención de las dos últimas líneas es ver los mensajes DEBUG de `can_distinguish_with_dataset`. **Ojo**: el logger real del módulo se llama `causalab.causal.causal_model` (es `logging.getLogger(__name__)`), no `causal.causal_model`, así que esa línea no tiene efecto y en la salida guardada efectivamente no aparecen esos mensajes.

### Celdas 3–7 — el modelo y un ejemplo

```python
causal_model = positional_causal_model
print_structure(causal_model)
```

- Dibuja el grafo (salida guardada: solo `<Figure size 1000x1000 ...>`).

```python
example = sample_answerable_question()
for k, v in example.to_dict().items():
    print(f'{k}:\n "{v}"\n')
```

- `example` es un `CausalTrace`. `to_dict()` devuelve una copia de `_values`.
- El orden de impresión (template, object, color, choice0, symbol0, choice1, answer_position, symbol1, raw_input, answer, raw_output) es el **orden en que se fueron calculando**: `CausalTrace.__init__` setea los inputs en el orden del dict, y apenas una variable tiene todos sus padres listos se calcula. `answer_position` aparece justo después de `choice1` (ya tiene color, choice0 y choice1); `raw_input`, `answer` y `raw_output` aparecen después de `symbol1`.
- En la ejecución guardada: "The flower is black... K. orange / T. black", `answer_position = 1`, `answer = "T"`, `raw_output = " T"`.

```python
full_setting = example
print(f'"answer_position":\n "{full_setting["answer_position"]}"\n')
print(f'"answer":\n "{full_setting["answer"]}"\n')
```

- `full_setting` es simplemente un alias: el trace ya tiene todas las variables calculadas.

### Celda 9 — intervención directa

```python
new_pos = int(not full_setting["answer_position"])
intervened = example.copy().intervene("answer_position", new_pos)
```

- `int(not pos)` da la otra posición (solo funciona porque hay 2 opciones).
- `.copy()` antes de `.intervene` para no mutar `example` (recordá que `intervene` modifica in place y devuelve `self`).
- Resultado guardado: `answer_position = 0`, `answer = "K"` (la letra del base en la posición 0). `raw_input` **no cambia**: no es descendiente de `answer_position`.

### Celda 11 — interchange intervention

```python
new_example = sample_answerable_question()
while new_example["answer_position"] == full_setting["answer_position"]:
    new_example = sample_answerable_question()

intervened_full_setting = causal_model.run_interchange(
    example, {"answer_position": new_example}
)
```

- Resamplea hasta tener un contrafáctico con la respuesta en la **otra** posición (si no, la interchange no cambiaría nada).
- `run_interchange` fija `answer_position` del base al valor del contrafáctico. En la ejecución guardada: el contrafáctico es "The ball is white... R. white / S. purple" (posición 0), y el resultado es `answer = "K"`: la letra del **base** en la posición 0. La letra "R" del contrafáctico no aparece, porque `answer` usa los símbolos del base.

### Celda 13 — un par que confunde `answer` con `answer_position`

```python
example = sample_answerable_question()
cf_inputs = {var: example[var] for var in causal_model.inputs}
cf_inputs["object"] = "toy"
cf_inputs["color"] = example[f"choice{int(not example['answer_position'])}"]
new_example = causal_model.new_trace(cf_inputs)

print(causal_model.run_interchange(example, {"answer_position": new_example})["answer"])
print(causal_model.run_interchange(example, {"answer": new_example})["answer"])
```

- `causal_model.inputs` es la lista de variables sin padres (template, object, color, symbol0, symbol1, choice0, choice1). Copiamos solo inputs para construir un trace nuevo desde cero (así los intermedios se recalculan).
- Cambia el objeto a "toy" (no está en `OBJECTS`, pero nada lo valida) y el color al **distractor** del base. Así el contrafáctico tiene la respuesta correcta en la otra posición, con **los mismos símbolos**.
- Interchange de `answer_position` → letra del base en la posición nueva. Interchange de `answer` → letra del contrafáctico, que (mismos símbolos) es la misma letra. Salida guardada: "I" y "I". **El par confunde las dos variables.**

### Celda 15 — deconfundir cambiando símbolos

```python
cf_inputs2 = {var: example[var] for var in causal_model.inputs}
cf_inputs2["object"] = "toy"
cf_inputs2["color"] = example[f"choice{int(not example['answer_position'])}"]
new_symbols = list({"A", "B", "C", "D"}.difference({example["symbol0"], example["symbol1"]}))
cf_inputs2["symbol0"] = random.choice(new_symbols)
new_example2 = causal_model.new_trace(cf_inputs2)
```

- Igual que antes pero `symbol0` se reemplaza por una letra de {A,B,C,D} que no esté en el base.
- Ahora interchange de `answer_position` sigue dando la letra del base ("I"), pero interchange de `answer` da la letra nueva del contrafáctico ("C" en la ejecución guardada). Distintas → el par **distingue** las variables.
- Detalle: solo se cambia `symbol0`. Si la respuesta del contrafáctico quedara en la posición 1, la letra seguiría siendo la del base y no deconfundiría. Acá funciona porque la respuesta del contrafáctico cae en la posición 0 (en el ejemplo guardado, la base tenía la respuesta en la posición 1 y el contrafáctico en la 0). Con otra muestra donde la base tenga la respuesta en la posición 0, el contrafáctico la tendría en la 1 y el truco **no** funcionaría.

### Celda 17 — `can_distinguish_with_dataset` sobre un solo par

```python
causal_model.can_distinguish_with_dataset(
    [{"input": example, "counterfactual_inputs": [new_example]}],
    ["answer_position"], ["answer"],
)
```

- Dataset de un ejemplo. Salida guardada: `{'proportion': 0.0, 'count': 0}` para el primer par (confunde) y `{'proportion': 1.0, 'count': 1}` para el segundo (deconfunde).

### Celdas 19–21 — dataset de pares aleatorios

```python
from causalab.tasks.MCQA.counterfactuals import random_counterfactual
random_pairs = [random_counterfactual() for _ in range(64)]

print(causal_model.can_distinguish_with_dataset(random_pairs, ["answer"], None))
print(causal_model.can_distinguish_with_dataset(random_pairs, ["answer_position"], None))
print(causal_model.can_distinguish_with_dataset(random_pairs, ["answer"], ["answer_position"]))
```

- 64 pares independientes.
- Razonamiento (de la notebook):
  - `answer` vs nada: solo se confunde si la letra correcta del contrafáctico coincide con la del base (≈ 1/26).
  - `answer_position` vs nada: se confunde si ambas respuestas están en la misma posición (1/2).
  - `answer` vs `answer_position`: se confunde si la letra del contrafáctico coincide con la letra del base que está en la posición del contrafáctico (≈ 1/26).
- Salida guardada: 0.9375, 0.46875, 0.96875. Coherente con esos órdenes de magnitud (con 64 muestras hay ruido).

### Celdas 23–25 — dataset `different_symbol`

```python
from causalab.tasks.MCQA.counterfactuals import different_symbol
different_symbol_pairs = [different_symbol() for _ in range(64)]
```

- Contrafáctico = misma pregunta, mismas opciones, símbolos nuevos (disjuntos).
- Por construcción:
  - `answer` vs nada: siempre distingue (la letra nueva nunca coincide).
  - `answer_position` vs nada: **nunca** distingue (misma posición → misma letra del base).
  - `answer` vs `answer_position`: siempre distingue.
- Salida guardada: 1.0, 0.0, 1.0. Es un dataset diseñado para **localizar `answer`**, y es ciego a `answer_position`.

### Celda 26 — Takeaways

El modelo causal es la hipótesis; los contrafácticos son el instrumento experimental; todo se diseña sin tocar la red.

---

## Ejercicios

No hay ejercicios explícitos. Lo que conviene hacer para la 03 es calcular a mano (o con `can_distinguish_with_dataset`) la tabla para `same_symbol_different_position`: por construcción la letra correcta del contrafáctico es la misma que la del base pero en la otra posición, así que `answer` vs nada debería dar 0, `answer_position` vs nada 1, y `answer` vs `answer_position` 1. En la 03 se imprimen esos números para los datasets ya filtrados.

---

## Ojo / errores comunes / dudas

- **`sample_input()` no sirve para esta tarea**: podría generar opciones que no contienen el color, y entonces `_get_answer_position` lanza `ValueError`. Por eso existe `sample_answerable_question`.
- **`raw_output` lleva un espacio adelante** (`" T"`). Las métricas de las notebooks siguientes comparan con `in`/`strip()` justamente por esto.
- **`answer` usa los símbolos del base**: intervenir `answer_position` nunca puede producir una letra que no esté en el prompt base. Esta es la clave de por qué `different_symbol` distingue `answer` de `answer_position`.
- **El logger configurado en la celda 1 tiene el nombre equivocado** (`causal.causal_model` en vez de `causalab.causal.causal_model`), por eso no se ven los mensajes de debug.
- **El truco de la celda 15 depende de la muestra**: solo reemplaza `symbol0`.
- **`can_distinguish_with_dataset` exige exactamente un contrafáctico por ejemplo** (hay un `assert`).
- **Ojo con los números de las celdas**: son de la ejecución guardada; como todo es aleatorio (sin semilla), al re-ejecutar cambian los ejemplos y las proporciones de `random_pairs`.

---

## Resumen de una línea por sección

- **Intro**: tarea MCQA de colores con dos opciones; el modelo causal tiene una variable intermedia, `answer_position`.
- **Causal model**: `answer_position = índice de la opción que es el color`; `answer = símbolo en esa posición`; `raw_output = " " + answer`.
- **Interventions / interchange**: intervenir `answer_position` produce la letra del **base** en la otra posición.
- **Confounding**: con los mismos símbolos, intervenir `answer` y `answer_position` da lo mismo; con símbolos nuevos, no.
- **Discriminative power**: `can_distinguish_with_dataset` mide qué fracción de pares distingue dos hipótesis; `random_counterfactual` es mediocre para `answer_position`, `different_symbol` es perfecto para `answer` y ciego a `answer_position`.
