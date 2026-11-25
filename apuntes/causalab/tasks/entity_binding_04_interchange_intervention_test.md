# Entity binding 04: interchange interventions sobre variables `positional_entity`

Notebook: `causalab/causalab/tasks/entity_binding/notebooks/04_interchange_intervention_test.ipynb`

Esta notebook usa un tercer modelo causal, el **positional entity model** (`create_positional_entity_causal_model`), que hace explícito "en qué grupo está cada entidad" con una variable por celda (`positional_entity_g{g}_e{e}`). Luego prueba interchange interventions con la sintaxis de flecha (`"A<-B"`: poner en `A` de la base el valor de `B` del contrafactual) y mide distinguibilidad. La pregunta que intenta responder: si esas variables parecen triviales en el modelo causal (siempre valen el índice del grupo), ¿tiene sentido intervenirlas, y qué testean en una red neuronal?

Aviso desde ya: las salidas guardadas de esta notebook son **inconsistentes entre sí** y la explicación de la notebook ("la intervención no tiene efecto porque los valores siempre son 0 y 1") no es la razón correcta. Abajo lo analizo con el código.

> Tarea **DEPRECATED**. Código: `tasks/entity_binding/causal_models.py` (líneas 711-1081) y `tasks/entity_binding/counterfactuals.py`. Para la teoría general, ver los apuntes generales (`../01_causal.md` y siguientes).

---

## Ideas clave / teoría

### El modelo `positional_entity`

Extiende el modelo posicional (notebook 03) con más pasos intermedios. Para la config de la notebook (2 grupos, 2 roles) las variables nuevas son:

- `positional_entity_g{g}_e{e}` (una por celda). Padre: `entity_g{g}_e{e}`. Mecanismo (`causal_models.py` 906-910): devuelve `g` si la entidad no es `None`, si no `None`. O sea: "la posición (grupo) de esta entidad".
- `positional_query_e{e}` (una por rol). Padres: todas las entidades, todas las `positional_entity`, `query_group`, `query_indices`, ... Mecanismo `create_positional_query_mechanism(e)` (939-973):
  - si el rol `e` no está en `query_indices`, devuelve `()`;
  - si está, toma `query_entity = entity_g{query_group}_e{e}`, recorre los grupos activos y, para cada `g` donde `entity_g{g}_e{e} == query_entity`, agrega **`positional_entity_g{g}_e{e}`** (no `g` directamente) a una tupla.
- `positional_answer`. Padres: las `positional_query_e*` + `query_indices`. Mecanismo `compute_positional_answer` (976-1014): intersección de los conjuntos de posiciones de los roles preguntados; si queda exactamente una, la devuelve; si queda más de una, error de ambigüedad; si ninguna, `None`.
- `raw_output` (1062-1077): `entity_g{positional_answer}_e{answer_index}`.

El punto clave: la posición que se usa para leer la respuesta **viene de la variable `positional_entity` de la entidad preguntada**. Si intervienes esa variable, cambias de dónde se lee la respuesta, aunque la entidad en sí no se mueva. Es una formalización de la hipótesis de "punteros de posición" (la red guarda, en el token de cada entidad, un identificador del grupo, y la recuperación usa ese identificador).

### Sintaxis de flecha en `run_interchange`

`CausalModel.run_interchange(input_trace, counterfactual_inputs)` (`causalab/causal/causal_model.py`) acepta claves `"A<-B"`: hace `trace.intervene("A", cf_trace["B"])`. Sirve para intervenciones "cruzadas": tomar el valor de una variable en el contrafactual y ponerlo en **otra** variable de la base. Con contrafactuales que intercambian grupos, esto permite "seguir" a una entidad que cambió de lugar.

### Qué estoy testeando realmente

Un resultado de distinguibilidad en el modelo causal depende de dos cosas: (1) si la intervención **cambia el valor** de la variable y (2) si esa variable está **en un camino causal** hacia `raw_output` para la pregunta que se hace. La notebook mezcla las dos.

---

## Recorrido por la notebook

### Setup

```python
from causalab.tasks.entity_binding.config import create_sample_love_config
from causalab.tasks.entity_binding.causal_models import create_positional_entity_causal_model
from causalab.tasks.entity_binding.counterfactuals import swap_query_group
from causalab.causal.counterfactual_dataset import CounterfactualDataset
```

Aquí el import de contrafactuales sí usa el nombre correcto del archivo (`counterfactuals`).

### Step 1: config y modelo

```python
config = create_sample_love_config()
config.max_groups = 2
config.prompt_prefix = "We will ask a question about the following sentences.\n\n"
config.statement_question_separator = "\n\n"
config.prompt_suffix = "\nAnswer:"
config.fixed_query_indices = (0,)
causal_model = create_positional_entity_causal_model(config)
```

- `max_groups = 2`: siempre dos grupos (el sampler hace `randint(2, max_groups)` = 2).
- Prefijo, separador y sufijo dan un formato de "instrucción": `"We will ask...\n\nPete loves pie, and Ann loves bread.\n\nWhat does Ann love?\nAnswer:"`. Esto es lo que se le daría a un LM real.
- `fixed_query_indices = (0,)`: siempre se pregunta por la **persona** y se responde la comida (`answer_index` tiene que ser 1, porque el sampler descarta `answer_index in query_indices`).
- ID resultante: `entity_binding_positional_entity_2g_2e`.

### Step 2: un par contrafactual

```python
example = swap_query_group(config)
input_sample = example["input"]
counterfactual_sample = example["counterfactual_inputs"][0]
```

`swap_query_group` (ver notebook 03) intercambia el grupo preguntado con el otro y actualiza `query_group` para seguir a la entidad. Con 2 grupos esto equivale a invertir el orden de los dos enunciados.

Salida guardada:
- Base: `Pete/pie`, `Ann/bread`; `"Pete loves pie, and Ann loves bread.\n\nWhat does Ann love?"` (así que `query_group = 1`).
- Contrafactual: `Ann/bread`, `Pete/pie`; misma pregunta (`query_group = 0`).

### Step 3: el modelo sin intervenir

```python
original_output = causal_model.new_trace(input_sample)
counter_output = causal_model.new_trace(counterfactual_sample)
print(original_output['positional_entity_g0_e1'], original_output['positional_entity_g1_e1'],
      original_output['positional_answer'], original_output['raw_output'])
```

Salida guardada:

| | `pe_g0_e1` | `pe_g1_e1` | `positional_answer` | `raw_output` |
|---|---|---|---|---|
| Base | 0 | 1 | 1 | bread |
| Contrafactual | 0 | 1 | 0 | bread |

Las `positional_entity` valen 0 y 1 en ambos (por construcción, valen el índice de su celda). `positional_answer` sí cambia: Ann está en el grupo 1 en la base y en el 0 en el contrafactual. La respuesta es la misma (bread), como siempre en `swap_query_group`.

### Step 4: interchange con flechas

```python
target_variables = [
    "positional_entity_g0_e1<-positional_entity_g1_e1",
    "positional_entity_g1_e1<-positional_entity_g0_e1",
]
intervened_output = causal_model.run_interchange(
    input_sample,
    {
        "positional_entity_g0_e1<-positional_entity_g1_e1": counterfactual_sample,
        "positional_entity_g1_e1<-positional_entity_g0_e1": counterfactual_sample,
        "positional_entity_g0_e0<-positional_entity_g1_e0": counterfactual_sample,
        "positional_entity_g1_e0<-positional_entity_g0_e0": counterfactual_sample,
    },
)
for variable in intervened_output:
    print(f"{variable}: {intervened_output[variable]}")
```

- `target_variables` (que se usa más adelante en el Step 6) tiene solo las dos flechas del rol **e1** (comidas).
- Pero la llamada a `run_interchange` de esta celda usa **cuatro** flechas: las de e1 **y** las de e0 (personas).
- Cada flecha `"A<-B"` hace `trace.intervene(A, cf[B])`. En el contrafactual `pe_g1_* = 1` y `pe_g0_* = 0`, así que en la base quedan `pe_g0_* = 1` y `pe_g1_* = 0`: **los valores sí cambian** (se cruzan).

Salida guardada (es consistente con el código):
```
positional_entity_g0_e0: 1
positional_entity_g0_e1: 1
positional_entity_g1_e0: 0
positional_entity_g1_e1: 0
positional_query_e0: (0,)
positional_query_e1: ()
positional_answer: 0
raw_output: pie
```

Trazando el mecanismo: la pregunta es por Ann (`query_group = 1`, rol 0). `positional_query_e0` busca Ann en la columna e0 de la base, la encuentra en el grupo 1, y agrega `positional_entity_g1_e0`, que ahora vale **0**. Entonces `positional_answer = 0` y la respuesta es `entity_g0_e1 = pie`. Es decir: con los "punteros" de posición cruzados, el modelo causal predice que la red respondería con la comida del **otro** grupo. `positional_query_e1 = ()` porque el rol 1 no se pregunta.

### Step 5: "por qué la intervención no tiene efecto"

```python
print(f"  Intervened: g0_e1={intervened_output['positional_entity_g0_e1']}, g1_e1={intervened_output['positional_entity_g1_e1']}")
...
if intervened_output["positional_answer"] != original_output["positional_answer"]: ...
if intervened_output["raw_output"] != original_output["raw_output"]: ...
```

La salida guardada de esta celda dice `Intervened: g0_e1=0, g1_e1=1`, "No change!", `positional_answer unchanged: 1`, `raw_output unchanged: 'bread'`. Eso **contradice** la salida guardada del Step 4 (que muestra `g0_e1=1`, `g1_e1=0`, `positional_answer=0`, `pie`), y `intervened_output` es la misma variable. Lo más probable es que la celda del Step 4 se haya editado (agregando las flechas de e0) y re-ejecutado después, sin volver a correr el Step 5. No lo puedo confirmar sin ejecutar.

Lo que dice el código:
1. La intervención **sí cambia** los valores de `positional_entity` (de 0/1 a 1/0). La afirmación "both original and counterfactual have the same positional values (0, 1)" es cierta, pero la flecha no copia "el mismo" valor: copia el valor de la **otra** celda.
2. Si solo intervienes las de **e1** (lo que hay en `target_variables`), la salida no cambia, pero por otra razón: con `fixed_query_indices = (0,)`, `positional_answer` solo depende de `positional_query_e0`, que solo lee `positional_entity_g*_e0`. Las `positional_entity_*_e1` alimentan a `positional_query_e1`, que devuelve `()` porque el rol 1 nunca se pregunta. **No están en ningún camino activo hacia `raw_output`.**
3. Si intervienes también las de e0 (como en el Step 4), la salida sí cambia (bread → pie).

El print "positional_entity_g0_e1 = lambda entity_val: 0 if ..." de esta celda es una descripción simplificada, no el código real.

### Step 6: distinguibilidad contra "no intervenir"

```python
dataset = CounterfactualDataset.from_sampler(16, lambda: swap_query_group(config), id="swap_query_group_test")
result = causal_model.can_distinguish_with_dataset(dataset, target_variables, None)
# 0 out of 16
```

- `can_distinguish_with_dataset` acepta las flechas porque internamente arma `{var: cf_trace for var in target_variables}` y llama a `run_interchange`.
- Resultado guardado: 0/16. La explicación correcta es el punto 2 de arriba: las variables de e1 no afectan la salida cuando se pregunta por el rol 0. Si hubieras puesto en `target_variables` las flechas de **e0**, cada par daría la comida del otro grupo, distinta de la de la base (las comidas son distintas por la unicidad posicional del sampler), y el resultado sería 16/16 (deducido del código, no ejecutado).

### Step 7: distinguibilidad contra `positional_answer`

```python
result2 = causal_model.can_distinguish_with_dataset(dataset, target_variables, ["positional_answer"])
# 16 out of 16
```

- Intervenir `positional_answer` con el valor del contrafactual pone en la base el grupo donde la entidad preguntada está **en el contrafactual**, que en swap es el otro grupo → la base lee la comida del otro grupo → salida distinta de la base.
- Intervenir las e1 deja la salida de la base.
- Distintas siempre → 16/16. La conclusión de la notebook ("positional_entity and positional_answer are different variables") es correcta en el sentido de que el dataset separa ambas intervenciones; pero lo que en realidad muestra es que una de ellas no tiene efecto y la otra sí.

### Key Takeaways (markdown)

La parte de "Why This Still Matters for Neural Networks" tiene una idea válida, aunque la ilustración falle: una variable del modelo causal puede tener valores "triviales" (siempre el índice del grupo) y aun así ser interesante, porque lo que se testea con la red es si **intercambiar las representaciones** en ciertos tokens (por ejemplo el último token de cada entidad) produce el efecto que predice intervenir esas variables. Con la sintaxis de flecha y el contrafactual de swap, el modelo causal predice un efecto concreto (responder la comida del otro grupo) cuando se cruzan los punteros de la entidad preguntada. Si la red se comporta así al cruzar las activaciones en esos tokens, eso es evidencia de que guarda información posicional ahí.

Lo que me llevo: para que el experimento con la red sea informativo, las variables intervenidas tienen que estar en el camino causal de la pregunta. Con `fixed_query_indices = (0,)` las relevantes son las `positional_entity_*_e0` (tokens de las personas), no las `_e1`.

---

## Ejercicios

No hay ejercicios ni celdas `# TERRON` / `FILL IN`. Para practicar con el código:
- Cambiar `target_variables` por las flechas de e0 y predecir el resultado de los Steps 6 y 7 antes de correrlo (por el razonamiento de arriba: Step 6 debería dar 100%; en el Step 7 ambas intervenciones darían la comida del otro grupo, así que deberían ser indistinguibles, 0%).
- Cambiar `fixed_query_indices = (1,)` (preguntar por la comida, "Who loves X?") y ver que ahora las `_e1` sí importan.

---

## Ojo / errores comunes / dudas

- **Salidas inconsistentes** entre el Step 4 y el Step 5 (ver arriba). Confía en el código, no en los textos impresos.
- **Bug de closure en `positional_entity`.** La línea 910 de `causal_models.py` es `lambda t, g=g: g if t[f"entity_g{g}_e{e}"] is not None else None`: `g` se captura con valor por defecto, pero `e` no, así que todas las lambdas usan el **último** `e` del loop (e = 1 con dos roles). En la práctica `positional_entity_g{g}_e0` chequea si `entity_g{g}_e1` no es `None`. Para grupos activos ambas entidades existen, así que no cambia el resultado acá, pero el padre declarado (`entity_g{g}_e0`) no es la variable que realmente se lee.
- **Las variables `positional_entity` no son "triviales" bajo intervención**, solo lo son en el forward normal. Lo que importa es que `positional_query_e{e}` lee el puntero (`positional_entity`) en vez de calcular el grupo por su cuenta.
- **`swap_query_group` no deja `raw_input` en el contrafactual** en el código actual, pero la notebook lo imprime; viene de la versión vieja. Lo mismo con `CounterfactualDataset.from_sampler`.
- **API de `CausalModel` cambiada:** igual que en las notebooks 01-03, el constructor de la tarea no coincide con `causalab/causal/causal_model.py` actual.
- **Token positions para la red:** la markdown habla de "last token of e1 entities", pero la notebook no construye ni usa ningún `TokenPosition`. Además, el parser de la notebook 02 asume separador `" "` y sin prefijo, que no es la config de esta notebook (ver "Ojo" de la 02).

---

## Resumen de una línea por sección

- **Step 1:** 2 grupos, formato de instrucción, siempre se pregunta por la persona; modelo `positional_entity_2g_2e`.
- **Step 2:** `swap_query_group` invierte el orden de los dos enunciados; misma pregunta y misma respuesta.
- **Step 3:** sin intervenir, `positional_entity` vale el índice de su celda; `positional_answer` sigue a la entidad.
- **Step 4:** las flechas cruzan los punteros; cruzando los de e0 la salida pasa de bread a pie (salida guardada consistente con el código).
- **Step 5:** su salida guardada dice "no change", contradice el Step 4; la razón real de "no efecto" es que las `_e1` no están en el camino causal.
- **Step 6:** 0/16 contra no intervenir, porque solo se intervienen las `_e1`.
- **Step 7:** 16/16 contra `positional_answer`, porque esa sí cambia la salida.
- **Takeaways:** intervenir variables de valor trivial puede tener sentido para la red, siempre que estén en el camino causal de la pregunta.
