# General addition 03: variables de acarreo, contrafactuales y cascadas

> **Actualización (noviembre 2026):** esta tarea ya corre con la versión de `causalab` incluida en el repo. Cambios principales:
> - el modelo causal se construye con objetos `Mechanism`, la API actual de `CausalModel`;
> - los generadores de contrafactuales devuelven `CausalTrace` y aceptan `model=` para armar los traces con el modelo que se analiza;
> - los datasets son listas simples, no `CounterfactualDataset.from_sampler`;
> - se restauró `statement_conjunction_function`.
>
> Lo que abajo se describe como "no corre" o "API vieja" ya está resuelto. La explicación de las ideas sigue valiendo.

Notebook: `causalab/causalab/tasks/general_addition/notebooks/03_counterfactuals_and_carry_variables.ipynb`

Esta notebook trabaja solo con el **modelo causal intermedio** de la suma de 3 dígitos (variables de acarreo `C_1, C_2, C_3` y dígitos de salida `O_1..O_4`). Primero muestra cómo se propagan los acarreos y qué pasa al intervenirlos; después construye dos datasets contrafactuales (**aleatorio** y **sum-to-nine**) y mide con `can_distinguish_with_dataset` cuánto permite cada uno distinguir `C_1` de otras variables. La pregunta: ¿qué pares de entradas hacen visible el efecto de un acarreo, y cómo se ve una **cascada** de acarreos (forzar `C_1 = 1` cambia `C_2` y `C_3`)?

> Tarea **DEPRECATED**. Código: `tasks/general_addition/causal_models.py` (`create_intermediate_addition_model`, `sample_sum_to_nine_input`) y `tasks/general_addition/counterfactuals.py`. Salidas guardadas de una versión anterior de causalab. Apuntes generales: `../01_causal.md` y siguientes.

---

## Ideas clave / teoría

### Repaso del modelo intermedio (D = 3)

Con índices desde la derecha (i = 1 unidades) y $x_i, y_i$ los dígitos de cada sumando en la posición i:

$$C_1 = [x_1 + y_1 \ge 10],\quad C_i = [x_i + y_i + C_{i-1} \ge 10],\quad O_1 = (x_1+y_1) \bmod 10,\quad O_i = (x_i+y_i+C_{i-1}) \bmod 10,\quad O_4 = C_3.$$

En el código las entradas son `digit_{k}_{d}` con d desde la izquierda, así que la posición i usa `digit_0_{3-i}` y `digit_1_{3-i}` (unidades = `digit_*_2`).

### Cuándo se propaga un acarreo

Forzar $C_{i-1}$ de 0 a 1 suma 1 en la posición i. Eso cambia $C_i$ **solo si** $x_i + y_i = 9$ (pasa de 9 a 10). Si $x_i + y_i \le 8$, $C_i$ sigue en 0 (solo cambia $O_i$); si $x_i + y_i \ge 10$, $C_i$ ya era 1 y sigue en 1. Por eso las posiciones que **suman 9** son las que propagan: una **cascada** completa $C_1 \to C_2 \to C_3$ requiere que las decenas y las centenas sumen 9.

### Interchange intervention y distinguibilidad (recordatorio)

Para un par (base $b$, contrafactual $c$) e intervención sobre $V$: correr el modelo en $b$ con $V \leftarrow V(c)$. `can_distinguish_with_dataset(dataset, [V1], [V2])` (`causalab/causal/causal_model.py`) cuenta los pares donde $\text{II}(b,c,V_1)$ y $\text{II}(b,c,V_2)$ dan `raw_output` distinto. Si una variable casi nunca se distingue de otra en un dataset, ese dataset no sirve para localizarla por separado en la red.

Ojo: en esta notebook hay dos tipos de "intervención" que no son lo mismo:
- **Intervención dura** con un valor fijo (`C_1 = 1`), Parts 1 y 3 con `new_trace({..., 'C_1': 1})`.
- **Interchange intervention** (valor tomado del contrafactual), Parts 2 y 3 con `can_distinguish_with_dataset`.

---

## Recorrido por la notebook

### Setup

```python
sys.path.append("../..")
from causalab.tasks.general_addition.config import create_two_number_three_digit_config
from causalab.tasks.general_addition.causal_models import create_intermediate_addition_model
from causalab.tasks.general_addition.counterfactuals import random_counterfactual, sum_to_nine_counterfactual
from causalab.causal.counterfactual_dataset import CounterfactualDataset
```

### Part 1: variables de acarreo

```python
config = create_two_number_three_digit_config()
model = create_intermediate_addition_model(config)
# id: addition_intermediate_2n_3d, 17 variables
carry_vars = [v for v in model.variables if v.startswith("C_")]   # ['C_1', 'C_2', 'C_3']
output_vars = [v for v in model.variables if v.startswith("O_")]  # ['O_1', 'O_2', 'O_3', 'O_4']
```

17 = 6 dígitos + `num_digits` + `template` + 3 carries + 4 salidas + `raw_input` + `raw_output`.

#### 368 + 475 = 843

```python
example = {"digit_0_0": 3, "digit_0_1": 6, "digit_0_2": 8, "digit_1_0": 4, "digit_1_1": 7, "digit_1_2": 5,
           "num_digits": 3, "template": config.templates[0]}
output = model.new_trace(example)
```

Salida guardada: `C_1 = 1` (8+5 = 13), `O_1 = 3`; `C_2 = 1` (6+7+1 = 14), `O_2 = 4`; `C_3 = 0` (3+4+1 = 8), `O_3 = 8`; `O_4 = 0`. Respuesta `843`. Las cuentas en los `print` están escritas a mano; los valores de las variables sí salen del modelo.

#### Intervenir `C_1` y `C_2` en 123 + 234 = 357

```python
output_original = model.new_trace(no_carry_example)                       # C_1=C_2=C_3=0, 357
output_intervened = model.new_trace({**no_carry_example, 'C_1': 1})       # O_2: 5 -> 6, 367
output_intervened_c2 = model.new_trace({**no_carry_example, 'C_2': 1})    # O_3: 3 -> 4, 457
```

- `{**no_carry_example, 'C_1': 1}` agrega `C_1` al final del diccionario de entradas: se calcula `C_1` desde los dígitos y luego se pisa con 1, recalculando descendientes (ver "Ojo" sobre por qué esto funciona solo por el orden).
- Forzar `C_1 = 1`: decenas 2+3+1 = 6 → `O_2 = 6`, `C_2` sigue 0 (2+3 = 5, lejos de 9). Respuesta 367.
- Forzar `C_2 = 1`: centenas 1+2+1 = 4 → `O_3 = 4`, `C_3` sigue 0. Respuesta 457.

Cada carry afecta solo a la posición siguiente porque ninguna posición suma 9.

### Part 2: contrafactuales aleatorios

```python
random_dataset = CounterfactualDataset.from_sampler(
    128, lambda: random_counterfactual(config, num_addends=2, num_digits=3))
```

- `random_counterfactual(config, num_addends, num_digits)` (`counterfactuals.py` 20-41): crea un modelo básico que **no usa** y devuelve `{"input": a, "counterfactual_inputs": [b]}` con `a` y `b` muestras independientes de `sample_valid_addition_input` (dígito inicial 1-9, resto 0-9).
- `CounterfactualDataset.from_sampler(n, f)` llama `n` veces a `f` (API vieja; en el `counterfactual_dataset.py` actual solo están los `TypedDict` y un dataset es una lista de `CounterfactualExample`).
- Salida guardada: `"The sum of 981 and 234 is"` vs `"The sum of 955 and 550 is"`.

#### `C_1` contra todas las variables

```python
test_vars = [v for v in model.variables
             if v not in ["C_1", "raw_input", "raw_output", "template", "num_addends", "num_digits"]]
for var in test_vars:
    result = model.can_distinguish_with_dataset(random_dataset, ["C_1"], [var])
    status = "✓" if result["proportion"] > 0 else "✗"
```

- `test_vars` = los 6 dígitos, `C_2`, `C_3`, `O_1..O_4` (`num_addends` no existe en este modelo; se excluye por las dudas).
- Para cada `var`, compara "intervenir `C_1` con el valor del contrafactual" vs "intervenir `var` con el valor del contrafactual".
- Criterio de ✓: proporción > 0, o sea que **un solo** par distinguible ya cuenta. Es un criterio muy laxo.

Resultados guardados (fragmento): `C_1` vs `digit_0_0` 96.88%, vs `digit_0_1` 89.84%, vs `O_1` 90.62%, vs `C_2` 69.53%; la salida está truncada después. Cómo razonar estos números:
- Intervenir `C_1` solo cambia algo cuando $C_1(b) \ne C_1(c)$, que con dígitos aleatorios pasa cerca de la mitad de las veces. Cuando $C_1(b) = C_1(c)$, intervenir `C_1` deja la salida de la base, y entonces "se distingue" de `var` si y solo si intervenir `var` cambia la salida.
- Intervenir un dígito de entrada casi siempre cambia la salida (y además cambia `raw_input`, aunque eso no se compara). Por eso las proporciones contra dígitos son altas.
- `C_1` vs `C_2` es más bajo (~70%): hay muchos pares donde ambos carries coinciden entre base y contrafactual y ninguna intervención cambia nada.

#### `C_1` contra las salidas

```python
for output_var in ["O_1", "O_2", "O_3", "O_4"]:
    result = model.can_distinguish_with_dataset(random_dataset, ["C_1"], [output_var])
```

Guardado: `O_1` 90.62%, `O_2` 92.19%, `O_3` 92.97%, `O_4` 73.44%. `O_4` es el más bajo porque $O_4 = C_3$ casi siempre vale 0 en base y contrafactual (para que valga 1 hace falta que las centenas sumen ≥ 10), así que intervenirlo muchas veces no cambia nada, igual que `C_1` cuando coincide. La markdown dice que `C_1` "debería" ser muy distinguible de `O_2`/`O_3` "porque afecta esas posiciones vía propagación": en realidad lo distinguible es que intervenir `O_2` copia el dígito de decenas del contrafactual (casi siempre distinto), mientras que intervenir `C_1` a lo sumo suma 1 en las decenas.

### Part 3: contrafactuales sum-to-nine y cascadas

La markdown propone el "vector binario [0, 1, 0]" y dice que forzar `C_1 = 1` llevaría a `[1, 1, 1]`. Por el razonamiento de la teoría eso no puede pasar: si `C_2` ya era 1, forzar `C_1` no cambia `C_2`, y por lo tanto no puede cambiar `C_3`. La propia notebook lo descubre en las celdas siguientes.

#### El dataset sum-to-nine

```python
sum_to_nine_dataset = CounterfactualDataset.from_sampler(
    128, lambda: sum_to_nine_counterfactual(config, digit_position=2))
```

- `sum_to_nine_counterfactual(config, digit_position)` (`counterfactuals.py` 44-71): la base sale de `sample_sum_to_nine_input(config, digit_position)` y el contrafactual es aleatorio (`sample_valid_addition_input(config, 2, max_digits)`).
- `sample_sum_to_nine_input` (`causal_models.py` 498-560): 2 sumandos, `num_digits = max_digits`; en la columna `digit_position` (índice desde la izquierda) samplea `digit_0` y fija `digit_1 = 9 - digit_0`; si la columna es la más significativa y saldría 0, re-samplea `digit_0` entre 1 y 8; el resto de los dígitos al azar (el inicial 1-9).
- `digit_position=2` es la columna de las **unidades** (para D = 3, d = 2 → i = 1). Salida guardada: `985 + 224` (5+4 = 9), `282 + 137` (2+7 = 9), `704 + 725` (4+5 = 9).

Ojo con lo que esto garantiza: unidades que suman 9 significa `C_1 = 0` y `O_1 = 9`, pero como no entra ningún acarreo a las unidades, **eso no hace que `C_1` propague**. Para que intervenir `C_1` cambie `C_2` haría falta que sumen 9 las **decenas** (`digit_position=1`). Con `digit_position=2` lo único seguro es que $C_1(b) = 0$, de modo que la interchange de `C_1` tiene efecto exactamente cuando el contrafactual tiene $C_1(c) = 1$.

#### 369 + 540: [0, 1, 0] → [1, 1, 0]

```python
cascade_example = {... 369 y 540 ...}
output_normal = model.new_trace(cascade_example)             # C = [0, 1, 0], 909
output_cascade = model.new_trace({**cascade_example, 'C_1': 1})  # C = [1, 1, 0], 919
```

- Normal: unidades 9+0 = 9 → `C_1 = 0`, `O_1 = 9`; decenas 6+4 = 10 → `C_2 = 1`, `O_2 = 0`; centenas 3+5+1 = 9 → `C_3 = 0`, `O_3 = 9`. Respuesta 909.
- Forzando `C_1 = 1`: decenas 6+4+1 = 11 → `O_2 = 1`, `C_2` sigue en 1 → centenas sin cambio. Respuesta 919. La celda reconoce "propagated through C_2 but stopped at C_3" (en rigor no se propagó a través de `C_2`: `C_2` no cambió; solo cambió `O_2`).

#### 459 + 550: el ejemplo "full cascade" que no lo es

```python
full_cascade_example = {... 459 y 550 ...}
output_before = model.new_trace(full_cascade_example)   # C = [0, 1, 1], 1009
```

Unidades 9 → `C_1 = 0`; decenas 5+5 = 10 → `C_2 = 1`; centenas 4+5+1 = 10 → `C_3 = 1`. Ya tiene `C_3 = 1` sin intervenir, así que no sirve para mostrar la cascada (la celda siguiente lo admite: "Hmm, that naturally has C_3=1").

#### 369 + 630: la cascada completa [0, 0, 0] → [1, 1, 1]

```python
perfect_cascade = {... 369 y 630 ...}
output_pre_cascade = model.new_trace(perfect_cascade)          # C = [0, 0, 0], 999
output_full_cascade = model.new_trace({**perfect_cascade, 'C_1': 1})
```

Las tres posiciones suman 9 (9+0, 6+3, 3+6). Sin intervenir: todos los carries 0, respuesta 999. Forzando `C_1 = 1`:
- `O_1 = 9` (no depende de `C_1`);
- decenas 6+3+1 = 10 → `C_2 = 1`, `O_2 = 0`;
- centenas 3+6+1 = 10 → `C_3 = 1`, `O_3 = 0`;
- `O_4 = C_3 = 1`.

Respuesta 1009. Este es el ejemplo correcto de cascada y confirma la regla: hacen falta posiciones que sumen 9 **aguas abajo** de la intervención.

#### Distinguibilidad en el dataset sum-to-nine

```python
model.can_distinguish_with_dataset(sum_to_nine_dataset, ["C_1"], ["C_2"])   # 94/128
model.can_distinguish_with_dataset(sum_to_nine_dataset, ["C_1"], ["C_3"])   # 90/128
model.can_distinguish_with_dataset(sum_to_nine_dataset, ["C_1"], ["O_3"])   # 121/128
model.can_distinguish_with_dataset(sum_to_nine_dataset, ["C_1"], ["O_4"])   # 90/128
```

Proporciones guardadas: 73%, 70%, 95%, 70%. Son del mismo orden que con el dataset aleatorio (por ejemplo `C_1` vs `C_2`: 73% acá, 70% en el aleatorio). Con el análisis de arriba eso tiene sentido: con unidades sumando 9 no se fuerza ninguna cascada, así que este dataset no es mucho más "fino" que el aleatorio para separar `C_1` de los carries siguientes. La conclusión impresa ("C_1 should be distinguishable from downstream carries ... because it can trigger cascades") no está realmente respaldada por este dataset. Si quisieras un dataset donde intervenir `C_1` dispare cascadas, la base debería tener decenas (y centenas) sumando 9, y el contrafactual `C_1 = 1`.

### Summary (markdown)

Resume las tres ideas (carries intervenibles, contrafactuales aleatorios como baseline, sum-to-nine para cascadas) y la pregunta para la red: si al intercambiar activaciones en ciertos tokens/capas la red muestra los mismos efectos que intercambiar `C_1` (incluidas cascadas), eso es evidencia de una representación explícita de acarreo. Ojo que el resumen dice "Binary vector [0, 0, 0] with ones digits summing to 9", pero el ejemplo que da [0,0,0] → [1,1,1] es el que tiene **todas** las posiciones sumando 9, no solo las unidades.

---

## Ejercicios

No hay ejercicios ni celdas `# TERRON` / `FILL IN`. Para practicar con el código:
- Generar el dataset con `digit_position=1` (decenas suman 9) y repetir `C_1` vs `C_2`. Antes de correrlo, razona: si el contrafactual tiene `C_1 = 1`, intervenir `C_1` ahora sí cambia `C_2` en la base.
- Para cada uno de los ejemplos de la Part 3, predecir el vector de carries tras forzar `C_1 = 1` usando solo la regla "propaga si la posición suma 9".

---

## Ojo / errores comunes / dudas

- **Intervención vía `new_trace({..., 'C_1': 1})`.** Con la `CausalTrace` actual (`causalab/causal/trace.py`), pasar una variable no-entrada en `inputs` la fija con `_set`, que no reemplaza el mecanismo. Funciona porque `C_1` va al final del diccionario, después de sus padres. Lo robusto es `t = model.new_trace(inputs); t.intervene("C_1", 1)` (o `t["C_1"] = 1`).
- **`can_distinguish_with_dataset` hace interchange, no "forzar a 1".** El valor intervenido es el del contrafactual, que puede coincidir con el de la base (y entonces la intervención no hace nada).
- **Criterio "✓ si proporción > 0"** en la Part 2: no mide nada útil; hay que mirar las proporciones.
- **`digit_position` es desde la izquierda** (2 = unidades para 3 dígitos), al revés que los índices de `C_i`/`O_i`.
- **Unidades que suman 9 no habilitan cascadas desde `C_1`.** Las cascadas dependen de las posiciones **siguientes**.
- **`random_counterfactual` y `sum_to_nine_counterfactual` incluyen `num_addends`** en los dicts, que el modelo intermedio no tiene: con la librería actual daría `KeyError` al crear la traza. Las salidas guardadas son de la versión vieja (igual que el constructor `CausalModel(variables, values, parents, mechanisms)`, que ya no existe así).
- Las cuentas escritas en los `print` ("6 + 4 + 1 = 11") están hardcodeadas; si cambias los dígitos del ejemplo, el texto no cambia.

---

## Resumen de una línea por sección

- **Part 1:** el modelo intermedio de 3 dígitos tiene `C_1..C_3` y `O_1..O_4`; forzar `C_1` o `C_2` cambia solo la posición siguiente si ninguna suma 9.
- **Part 2:** con contrafactuales aleatorios `C_1` se distingue de casi todo en 70-97% de los pares; `C_2` y `O_4` son los más difíciles de separar.
- **Part 3 (dataset):** sum-to-nine en unidades garantiza `C_1 = 0` en la base, pero no habilita cascadas.
- **Part 3 (ejemplos):** [0,1,0] → [1,1,0]; la cascada completa [0,0,0] → [1,1,1] aparece con 369 + 630, donde todas las posiciones suman 9.
- **Part 3 (distinguibilidad):** proporciones parecidas a las del dataset aleatorio; la conclusión sobre cascadas no queda respaldada.
- **Summary:** el objetivo con la red es buscar los mismos efectos (incluidas cascadas) al intercambiar activaciones.
