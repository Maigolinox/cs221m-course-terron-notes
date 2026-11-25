# Entity binding 03: contrafactuales y el modelo posicional

Notebook: `causalab/causalab/tasks/entity_binding/notebooks/03_counterfactuals_and_positional_variables.ipynb`

Esta notebook plantea dos **hipótesis causales** sobre cómo un LM responde *"Pete loves jam, Ann loves pie, and Bob loves cake. What does Ann love?"*, las escribe como dos modelos causales distintos (**directo** y **posicional**) y muestra cómo construir **datasets contrafactuales** (pares base/contrafactual) que permitan distinguir esas hipótesis mediante interchange interventions. La pregunta que responde: ¿qué variables intermedias propone cada hipótesis y qué tipo de contrafactual hace que "intervenir la variable X" y "intervenir la variable Y" den resultados distintos?

> Tarea **DEPRECATED**; las salidas guardadas son de una versión anterior de causalab (ver "Ojo"). Código: `tasks/entity_binding/causal_models.py` y `tasks/entity_binding/counterfactuals.py`. Para teoría general de interchange interventions ver los apuntes generales (`../01_causal.md` y siguientes).

---

## Ideas clave / teoría

### Dos hipótesis de recuperación

- **Directa** (`create_direct_causal_model`): el modelo "sabe" el índice del grupo preguntado (`query_group`) y lee la respuesta de ahí: `raw_output = entity_g{query_group}_e{answer_index}`. No hay variables intermedias.
- **Posicional** (`create_positional_causal_model`): el modelo primero extrae **qué entidad** se pregunta (por su valor, `query_entity = ("Ann",)`), luego **busca** en qué grupo aparece esa entidad en el rol preguntado (`positional_query_group`), y lee la respuesta de ese grupo. Es recuperación por contenido.

En entradas normales las dos dan la misma respuesta (la notebook lo muestra). Se diferencian en su **estructura interna**, y eso solo se observa intervieniendo.

### Interchange intervention en el modelo causal

Dada una base `b` y un contrafactual `c`, la interchange intervention sobre la variable `V` es: correr el modelo en `b`, pero con `V` fijada al valor que toma en `c`, y recalcular los descendientes de `V`:

$$\text{II}(b, c, V) = \text{salida de } M \text{ con } V \leftarrow V(c) \text{ y el resto calculado desde } b.$$

En causalab esto es `CausalModel.run_interchange(base_trace, {"V": cf_trace})` (`causalab/causal/causal_model.py`): copia la traza base y hace `trace.intervene(V, cf_trace[V])`, que reemplaza el mecanismo de `V` por una constante y recalcula descendientes (`causalab/causal/trace.py`, `intervene` y `_recompute_descendants`).

### Distinguibilidad

`CausalModel.can_distinguish_with_dataset(dataset, vars1, vars2)` cuenta en cuántos pares el `raw_output` tras intervenir `vars1` es **distinto** del `raw_output` tras intervenir `vars2` (o de la salida sin intervenir, si `vars2=None`). Si dos variables dan la misma salida en todo el dataset, un experimento con la red no puede decir cuál de las dos está localizada en un sitio: el dataset no tiene poder para distinguirlas. Diseñar contrafactuales es buscar datasets con proporción alta para las comparaciones que nos interesan.

---

## Recorrido por la notebook

### Setup

```python
from causalab.tasks.entity_binding.config import create_sample_love_config
from causalab.tasks.entity_binding.causal_models import create_direct_causal_model, create_positional_causal_model
from causalab.tasks.entity_binding.counterfactual import swap_query_group, random_counterfactual
```

Ojo: importa `...entity_binding.counterfactual` (singular), pero el archivo que existe es `counterfactuals.py` (plural). La notebook 04 sí usa el nombre correcto. Con el repo actual esta celda daría `ModuleNotFoundError`.

### Part 1: los dos modelos sobre la misma entrada

```python
config = create_sample_love_config()
direct_model = create_direct_causal_model(config)
positional_model = create_positional_causal_model(config)
```

IDs: `entity_binding_direct_3g_2e` y `entity_binding_positional_3g_2e`.

#### Qué agrega el modelo posicional (`causal_models.py` 254-576)

Todo es igual al directo (mismas entradas, mismo `raw_input`) salvo dos variables nuevas y el cambio de padres de `raw_output`:

- `query_entity`, padres: entidades + `query_group` + `query_indices`. Mecanismo `extract_query_entity` (435-450): para cada índice en `query_indices` lee `entity_g{query_group}_e{idx}` y devuelve una **tupla** (por eso la salida muestra `('Ann',)` y no `'Ann'`).
- `positional_query_group`, padres: entidades + `query_entity` + `query_indices` + `active_groups` + `entities_per_group`. Mecanismo `search_for_query_entity` (455-507):
  1. Si `query_entity` es `None`, devuelve `None`.
  2. Para cada grupo activo `g`, verifica que en **todas** las posiciones `query_indices` la entidad del grupo sea igual a la de `query_entity`.
  3. Si matchea más de un grupo lanza `ValueError("Ambiguous query ...")` (por eso el sampler exige entidades distintas por posición).
  4. Si matchea uno, devuelve ese `g`; si ninguno, `None`.
- `raw_output`, padres: entidades + **`positional_query_group`** + `answer_index` + ... Mecanismo (557-570): `entity_g{positional_query_group}_e{answer_index}` o `"UNKNOWN"`.

El grafo queda: `entidades, query_group, query_indices → query_entity → positional_query_group → raw_output`. `query_group` solo influye en la salida **a través** de qué entidad se nombra.

```python
input_sample = {
    "entity_g0_e0": "Pete", "entity_g0_e1": "jam",
    "entity_g1_e0": "Ann",  "entity_g1_e1": "pie",
    "entity_g2_e0": "Bob",  "entity_g2_e1": "cake",
    "query_group": 1, "query_indices": (0,), "answer_index": 1,
    "active_groups": 3, "entities_per_group": 2,
}
direct_output = direct_model.new_trace(input_sample)
positional_output = positional_model.new_trace(input_sample)
```

- Ambos `new_trace` calculan todo el grafo a partir de las entradas.
- La celda imprime los pasos del modelo posicional: `query_entity = ('Ann',)`, buscar Ann en la posición e0, encontrada en el grupo 1, leer `g1_e1 = pie`. Directo: `entity_g1_e1 = pie`. Coinciden (`True`).
- Prompt: `"Pete loves jam, Ann loves pie, and Bob loves cake. What does Ann love?"`.

### Part 2: ¿qué pasa si intercambio grupos?

```python
swapped_sample = input_sample.copy()
swapped_sample["entity_g1_e0"] = "Bob";  swapped_sample["entity_g1_e1"] = "cake"
swapped_sample["entity_g2_e0"] = "Ann";  swapped_sample["entity_g2_e1"] = "pie"
# query_group STAYS 1
direct_swapped = direct_model.new_trace(swapped_sample)
positional_swapped = positional_model.new_trace(swapped_sample)
```

- Se intercambian los grupos 1 y 2 pero `query_group` sigue en 1, así que la pregunta pasa a ser por **Bob** (porque `raw_input` arma la pregunta con la entidad del `query_group`): `"Pete loves jam, Bob loves cake, and Ann loves pie. What does Bob love?"`.
- Directo: `entity_g1_e1 = cake`. Posicional: `query_entity = ('Bob',)`, encontrado en el grupo 1, `cake`. Vuelven a coincidir.

La markdown "Why Do They Still Agree?" lo explica bien: corriendo el modelo hacia adelante sobre una entrada nueva, las dos hipótesis siempre calculan la respuesta correcta; la diferencia aparece cuando intervenimos una variable interna. (La frase de la markdown sobre "might not find it correctly" es vaga; lo concreto se ve en la Part 4.)

### Part 3: generar un contrafactual con `swap_query_group`

```python
example = swap_query_group(config)
input_ex = example["input"]
counter_ex = example["counterfactual_inputs"][0]
```

Qué hace `swap_query_group(config)` (`counterfactuals.py` 17-103), línea por línea:

1. `model = create_direct_causal_model(config)`: se crea pero **no se usa** (resto de una versión anterior).
2. `input_sample = sample_valid_entity_binding_input(config)`: base válida.
3. `other_groups = [g for g in range(active_groups) if g != query_group]`; si no hay (un solo grupo), devuelve un contrafactual aleatorio.
4. `swap_group = random.choice(other_groups)`.
5. `counterfactual = input_sample.copy()` y para cada rol `e` intercambia `entity_g{query_group}_e{e}` con `entity_g{swap_group}_e{e}`: el grupo preguntado **completo** se muda de lugar.
6. `counterfactual["query_group"] = swap_group`: el `query_group` del contrafactual **sigue a la entidad**, así que la pregunta es la misma ("What does X love?" con el mismo X) y la respuesta también.
7. Borra `raw_input` si estaba y devuelve `{"input": ..., "counterfactual_inputs": [counterfactual]}`.

Ojo con el docstring: dice que el contrafactual "keeps the SAME QUERY POSITION" y que pregunta por otra persona. El **código** hace lo contrario: cambia `query_group` a `swap_group` y mantiene la misma persona en la pregunta. La salida guardada confirma lo que hace el código:

- Input: `"Bob loves jam, and Tim loves bread. Who loves jam?"`, `query_group 0`.
- Counterfactual: `"Tim loves bread, and Bob loves jam. Who loves jam?"`, `query_group 1`.

(La línea impresa "(same position!)" es engañosa: la posición cambió; lo que se mantiene es la entidad preguntada y la respuesta.)

```python
for e in range(config.max_entities_per_group):
    key = f"entity_g{query_group}_e{e}"
    ... changed = "<-- CHANGED" if input_val != counter_val else ""
```

Compara el grupo `query_group` de la base entre base y contrafactual: ambas entidades cambian (en esa posición ahora vive el otro grupo).

Por qué este contrafactual es útil: base y contrafactual tienen **la misma `query_entity`** pero **distinto `positional_query_group`**. Es justo el par que separa "qué entidad" de "en qué posición está".

### Part 4: interchange interventions en el modelo posicional

```python
counterfactual_sample = {
    "entity_g0_e0": "Tim", "entity_g0_e1": "soup",
    "entity_g1_e0": "Ann", "entity_g1_e1": "bread",
    "entity_g2_e0": "Pete", "entity_g2_e1": "tea",
    "query_group": 2, "query_indices": (0,), "answer_index": 1,
    "active_groups": 3, "entities_per_group": 2,
}
original_output = positional_model.new_trace(input_sample)
counter_output = positional_model.new_trace(counterfactual_sample)
intervened = positional_model.run_interchange(input_sample, {"query_entity": counterfactual_sample})
```

Salida guardada y su explicación:

| | `query_entity` | `positional_query_group` | `raw_output` |
|---|---|---|---|
| Base (Pete/jam, Ann/pie, Bob/cake; pregunta por Ann) | `('Ann',)` | 1 | pie |
| Contrafactual (Tim/soup, Ann/bread, Pete/tea; pregunta por Pete) | `('Pete',)` | 2 | tea |
| Base con `query_entity ← ('Pete',)` | `('Pete',)` | **0** | **jam** |

La intervención reemplaza la entidad buscada por "Pete" pero **la búsqueda se hace en la base**, donde Pete está en el grupo 0, y se lee `entity_g0_e1 = jam`. Ni la respuesta de la base (pie) ni la del contrafactual (tea): una tercera respuesta que solo predice la hipótesis posicional. Eso es lo que hace a las interchange interventions informativas.

La celda termina con otra intervención que no se imprime: `run_interchange(input_sample, {"positional_query_group": counterfactual_sample})`. Por el código, fijaría `positional_query_group = 2` en la base y daría `entity_g2_e1 = cake`.

Nota de API: acá se pasan `dict`s a `run_interchange`; en la API actual debe ser una `CausalTrace` (hace `input_trace.copy()`), o sea `positional_model.new_trace(input_sample)`.

### Part 5: dataset contrafactual

```python
from causalab.causal.counterfactual_dataset import CounterfactualDataset
config.num_groups = 3
swap_dataset = CounterfactualDataset.from_sampler(64, lambda: swap_query_group(config))
```

- `CounterfactualDataset.from_sampler(n, sampler)` llama `n` veces al sampler y junta los pares en un dataset indexable (`swap_dataset[0]['input']`). En el `counterfactual_dataset.py` actual solo quedan los `TypedDict` `CounterfactualExample` / `LabeledCounterfactualExample` (un dataset ahora es simplemente una `list[CounterfactualExample]`); `from_sampler` es de la API vieja.
- `config.num_groups = 3` agrega un atributo que **ningún código usa** (el campo real es `max_groups`). No tiene efecto.
- Salida guardada: 64 pares, p. ej. `"Kate loves tea, Ann loves pie, and Bob loves cake. What does Kate love?"` vs `"Ann loves pie, Kate loves tea, and Bob loves cake. What does Kate love?"`.

### Part 6: distinguibilidad con el dataset de swaps

```python
positional_model.can_distinguish_with_dataset(swap_dataset, ["query_entity"], ["positional_query_group"])  # 64/64
positional_model.can_distinguish_with_dataset(swap_dataset, ["positional_query_group"], ["raw_output"])   # 64/64
positional_model.can_distinguish_with_dataset(swap_dataset, ["positional_query_group"], None)             # 64/64
```

Cómo funciona `can_distinguish_with_dataset` (`causal_model.py`): para cada par toma el único contrafactual, corre `run_interchange(base, {v: cf for v in vars1})` y lo mismo con `vars2`, y cuenta si los `raw_output` difieren. Con `vars2=None` compara contra la salida de la base sin intervenir.

Razonamiento para cada resultado (en swap, el contrafactual tiene la misma entidad preguntada X, pero en el grupo `s` en vez de `q`):

1. **`query_entity` vs `positional_query_group`.** Intervenir `query_entity` con el valor del contrafactual no cambia nada (es la misma X) → salida = respuesta de la base. Intervenir `positional_query_group ← s` hace que la base lea del grupo `s`, donde en la base vive **otro** grupo → otra respuesta. Distintas siempre → 100%.
2. **`positional_query_group` vs `raw_output`.** Intervenir `raw_output` copia la respuesta del contrafactual, que es la **misma** que la de la base (el grupo entero se mudó con su respuesta). La intervención en `positional_query_group` da otra respuesta. → 100%.
3. **`positional_query_group` vs nada.** Igual que arriba → 100%.

Ojo con las etiquetas impresas: la primera celda dice "Can we distinguish query_entity from raw_output?" pero compara contra `positional_query_group`; la tercera dice "query_entity from no intervention" pero interviene `positional_query_group`. Hay que leer los argumentos, no los `print`.

Lectura importante: en este dataset intervenir `query_entity` **no cambia la salida** (es indistinguible de no intervenir). Así que con `swap_dataset` no podrías localizar `query_entity` en la red, pero sí `positional_query_group`.

### Part 7: contrafactuales aleatorios (baseline)

```python
random_dataset = CounterfactualDataset.from_sampler(64, lambda: random_counterfactual(config))
```

`random_counterfactual(config)` (`counterfactuals.py` 257-282) samplea dos entradas válidas independientes. Salida guardada: `"Tim loves cake, and Sue loves bread. What does Sue love?"` vs `"Sue loves soup, and Bob loves bread. What does Bob love?"`.

```python
positional_model.can_distinguish_with_dataset(random_dataset, ["query_entity"], ["positional_query_group"])  # 56/64
positional_model.can_distinguish_with_dataset(random_dataset, ["positional_query_group"], ["raw_output"])   # 58/64
positional_model.can_distinguish_with_dataset(random_dataset, ["query_entity"], None)                       # 60/64
```

Con pares aleatorios casi siempre las intervenciones cambian algo, pero no siempre (0.88, 0.91, 0.94 en la ejecución guardada). Por ejemplo, si la `query_entity` del contrafactual no aparece en la base (o el contrafactual pregunta por otro rol), la búsqueda devuelve `None` y la salida es `"UNKNOWN"`; y hay pares donde por casualidad dos intervenciones coinciden. Fíjate en la diferencia cualitativa: el dataset aleatorio "distingue" `query_entity` de no intervenir en 94% de los casos, pero mayormente produciendo salidas `"UNKNOWN"` (entradas fuera de distribución para la tarea), mientras que el dataset de swaps da comparaciones limpias del 100% para las variables posicionales.

---

## Ejercicios

No hay ejercicios ni celdas `# TERRON` / `FILL IN`. Para practicar: calcular a mano qué devuelve `run_interchange(base, {"positional_query_group": cf})` en la Part 4 (lo dejé arriba) y luego verificarlo; o construir un contrafactual que distinga `query_entity` de `positional_query_group` en el sentido contrario (misma posición, distinta entidad: por ejemplo, cambiar solo el nombre de la persona del grupo preguntado por uno nuevo).

---

## Ojo / errores comunes / dudas

- **Import roto:** `counterfactual` vs `counterfactuals`.
- **Docstring vs código en `swap_query_group`:** el código mueve el grupo preguntado y hace que `query_group` lo siga. La pregunta y la respuesta del contrafactual son las mismas que en la base.
- **`swap_query_group_preserve_answer` es idéntica a `swap_query_group`** en el código actual (mismo cuerpo); su docstring promete otra cosa. Y `swap_non_query_groups` modifica `input_sample["active_groups"]` a 3 sin volver a samplear las entidades del tercer grupo, que quedan en `None` si estaba inactivo.
- **Diccionarios sin `raw_input`.** El código actual de los samplers no calcula `raw_input` (incluso lo borra del contrafactual), pero la notebook imprime `input_ex['raw_input']`. En la versión vieja el dataset probablemente corría el modelo para completar las trazas. Con el código actual, habría que pasar cada dict por `model.new_trace(...)`.
- **`can_distinguish_with_dataset` necesita exactamente un contrafactual por ejemplo** (`assert len(counterfactual_traces) == 1`).
- **Distinguibilidad es una propiedad del dataset + modelo causal, no de la red.** Todavía no se tocó ningún LM; esto solo dice qué experimentos *podrían* ser informativos.
- **La API de `CausalModel` cambió** (constructor con `Mechanism`), así que reproducir estas salidas requiere la versión vieja de causalab o adaptar `causal_models.py`.

---

## Resumen de una línea por sección

- **Part 1:** directo indexa por `query_group`; posicional extrae `query_entity`, la busca (`positional_query_group`) y lee de ahí. En entradas normales coinciden.
- **Part 2:** intercambiar grupos manteniendo `query_group` cambia la pregunta; ambos modelos siguen coincidiendo.
- **Part 3:** `swap_query_group` mueve el grupo preguntado a otra posición y hace que `query_group` lo siga: misma entidad, distinta posición.
- **Part 4:** intervenir `query_entity` en el posicional produce una tercera respuesta (jam) que ni la base ni el contrafactual tenían.
- **Part 5:** `from_sampler` arma datasets de pares (API vieja).
- **Part 6:** con swaps, `positional_query_group` se distingue 100% de `query_entity`, de `raw_output` y de no intervenir; `query_entity` no tiene efecto.
- **Part 7:** pares aleatorios distinguen "casi siempre" pero de forma menos limpia (muchos `"UNKNOWN"`).
