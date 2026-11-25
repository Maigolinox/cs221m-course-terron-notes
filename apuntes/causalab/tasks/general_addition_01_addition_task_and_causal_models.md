# General addition 01: la tarea de suma y sus dos modelos causales

> **Actualización (noviembre 2026):** esta tarea ya corre con la versión de `causalab` incluida en el repo. Cambios principales:
> - el modelo causal se construye con objetos `Mechanism`, la API actual de `CausalModel`;
> - los generadores de contrafactuales devuelven `CausalTrace` y aceptan `model=` para armar los traces con el modelo que se analiza;
> - los datasets son listas simples, no `CounterfactualDataset.from_sampler`;
> - se restauró `statement_conjunction_function`.
>
> Lo que abajo se describe como "no corre" o "API vieja" ya está resuelto. La explicación de las ideas sigue valiendo.

Notebook: `causalab/causalab/tasks/general_addition/notebooks/01_addition_task_and_causal_models.ipynb`

Esta notebook construye la tarea de **suma de varios dígitos** para experimentos de causal abstraction: prompts como *"The sum of 23 and 45 is"* que el LM debería completar con *" 68"*. Define la configuración (K números de D dígitos), el procesamiento de plantillas, y dos **modelos causales** que representan dos hipótesis sobre cómo se computa la suma: uno **básico** (entradas → dígitos de salida, sin nada en el medio) y uno **intermedio** con variables explícitas de **acarreo** (carry) `C_i` y dígitos de salida `O_i`. La pregunta que responde: ¿cómo se escribe "la suma con acarreos" como grafo causal, y qué pasa si intervengo un acarreo?

> Tarea **DEPRECATED** ("See `causalab/tasks/MCQA/`"). Código: `causalab/causalab/tasks/general_addition/` (`config.py`, `templates.py`, `causal_models.py`). Salidas guardadas de una versión anterior de causalab (ver "Ojo"). Apuntes generales de la librería: `../01_causal.md` y siguientes.

---

## Ideas clave / teoría

### La suma con acarreo como algoritmo

Para sumar dos números de D dígitos se va de derecha a izquierda. En la posición i (i = 1 son las unidades, i = 2 las decenas, ...), con $x_i, y_i$ los dígitos de cada número en esa posición:

$$C_1 = [x_1 + y_1 \ge 10], \qquad O_1 = (x_1 + y_1) \bmod 10$$

$$C_i = [x_i + y_i + C_{i-1} \ge 10], \qquad O_i = (x_i + y_i + C_{i-1}) \bmod 10 \quad (2 \le i \le D)$$

$$O_{D+1} = C_D$$

- $C_i \in \{0,1\}$ es el acarreo que **sale** de la posición i (y entra a la i+1).
- $O_i$ es el dígito de la respuesta en la posición i.
- $[\cdot]$ es 1 si la condición se cumple y 0 si no.
- $O_{D+1}$ es el dígito extra que puede aparecer (99 + 1 = 100).

Ejemplo 58 + 67: unidades 8+7 = 15 → $C_1=1, O_1=5$; decenas 5+6+1 = 12 → $C_2=1, O_2=2$; $O_3 = C_2 = 1$ → 125.

### Dos hipótesis

- **Básica:** la red mapea los dígitos de entrada a los de salida "de una". En el grafo, cada `output_digit_d` depende de **todos** los dígitos de entrada, sin variables intermedias.
- **Intermedia:** la red calcula acarreos explícitos que se propagan. En el grafo, `O_2` depende de `C_1`, `C_2` depende de `C_1`, etc.

Las dos tienen el mismo comportamiento entrada-salida; solo se distinguen con intervenciones sobre las variables internas. Si intervengo `C_1 = 1` en 23 + 45 (donde no hay acarreo), el modelo intermedio predice 78 en vez de 68. Esa predicción es lo que se va a buscar en la red.

### Convenciones de índices (fuente típica de confusión)

- Entradas: `digit_{k}_{d}` con k = número de sumando y d = posición **desde la izquierda**, empezando en 0 (d = 0 es el dígito más significativo). En 23: `digit_0_0 = 2`, `digit_0_1 = 3`.
- Intermedias y salidas del modelo intermedio: `C_i`, `O_i` con i **desde la derecha**, empezando en 1 (i = 1 unidades).
- Conversión (en `causal_models.py`): `d = D - i`.
- Salidas del modelo básico: `output_digit_{d}` con d desde la izquierda, D+1 dígitos (`output_digit_0` es el de las centenas para D = 2).

---

## Recorrido por la notebook

### Step 1: configuraciones

```python
sys.path.append("../..")
from causalab.tasks.general_addition.config import (
    create_two_number_two_digit_config, create_two_number_three_digit_config,
    create_three_number_two_digit_config, create_general_config,
)
config_2n_2d = create_two_number_two_digit_config()
print(config_2n_2d.max_numbers, config_2n_2d.max_digits, config_2n_2d.templates)
```

`AdditionTaskConfig` (`config.py` 15-34) es un dataclass con `max_numbers` (K), `max_digits` (D), `templates` (lista de strings con `{num0}`, `{num1}`, ...), `prompt_prefix`, `prompt_suffix`. Las fábricas:

- `create_two_number_two_digit_config()`: K = 2, D = 2, plantilla `"The sum of {num0} and {num1} is"`.
- `create_two_number_three_digit_config()`: K = 2, D = 3, misma plantilla.
- `create_three_number_two_digit_config()`: K = 3, D = 2, `"The sum of {num0}, {num1}, and {num2} is"`.
- `create_general_config(K, D)` (97-129): genera la plantilla según K: para K ≥ 3 arma `"The sum of {num0}, {num1}, ..., and {numK-1} is"` uniendo partes con espacios.

La celda de "Different Task Configurations" solo imprime las plantillas; los ejemplos ("Example: The sum of 23 and 45 is 68") están escritos a mano en los `print`.

### Step 2: plantillas

```python
from causalab.tasks.general_addition.templates import AdditionTemplateProcessor
processor = AdditionTemplateProcessor(config_2n_2d)
```

#### Representación por dígitos

```python
processor.digits_to_number([2, 3])        # 23
processor.digits_to_number([0, 5, 6])     # 56
processor.number_to_digits(68, 2)         # [6, 8]
processor.number_to_digits(68, 3)         # [0, 6, 8]
```

- `digits_to_number` (`templates.py` 24-40): `result = result*10 + digit` recorriendo de izquierda a derecha (índice 0 = más significativo). Ceros a la izquierda desaparecen.
- `number_to_digits(n, num_digits)` (42-61): saca `n % 10` y divide por 10 `num_digits` veces, y luego invierte la lista; rellena con ceros a la izquierda. No chequea overflow: si el número tiene más dígitos que `num_digits`, los de más se pierden.

#### Llenar la plantilla

```python
prompt = processor.fill_template("The sum of {num0} and {num1} is", [[2, 3], [4, 5]])
# 'The sum of 23 and 45 is'
```

`fill_template` (81-113) arma `{"num0": "23", "num1": "45"}` con `format_number` (que pasa por entero, así que quita ceros a la izquierda) y hace `template.format(**substitutions)`. La rama `else: "0"` del loop nunca se ejecuta (la condición `i < len(numbers)` siempre es verdadera dentro de `enumerate(numbers)`).

#### Calcular la suma

```python
output_digits = processor.compute_sum([number1, number2])   # [0, 6, 8]
output_str = processor.format_output(output_digits)         # '68'
```

`compute_sum` (161-182) convierte a enteros, suma, y vuelve a dígitos con **`num_digits + 1`** posiciones (para el posible acarreo final). Por eso 68 sale `[0, 6, 8]`. `format_output` quita los ceros a la izquierda.

#### Prompt desde el diccionario de dígitos

```python
digit_dict = {"digit_0_0": 2, "digit_0_1": 3, "digit_1_0": 4, "digit_1_1": 5}
full_prompt = processor.generate_prompt(digit_dict, num_addends=2, num_digits=2)
# 'The sum of 23 and 45 is'
```

`generate_prompt` (115-159) es el puente con el modelo causal: lee `digit_{k}_{d}` para `k < num_addends`, `d < num_digits` (0 si falta), arma la lista de números, elige plantilla (la primera si no se pasa) y agrega prefijo/sufijo. Ejemplos con 3 dígitos (`'The sum of 123 and 456 is'`) y 3 sumandos (`'The sum of 12, 34, and 56 is'`).

### Step 3: el modelo causal básico

```python
from causalab.tasks.general_addition.causal_models import create_basic_addition_model, sample_valid_addition_input
basic_model = create_basic_addition_model(config_2n_2d)
# id: addition_basic_2n_2d, 12 variables
```

Qué arma `create_basic_addition_model` (`causal_models.py` 19-203):

- **Variables:** K·D dígitos de entrada (`digit_0_0 ... digit_1_1`), controles `num_addends`, `num_digits`, `template`, D+1 salidas `output_digit_0..D`, y `raw_input`, `raw_output`. Para 2×2: 4 + 3 + 3 + 2 = 12 (coincide con la salida).
- **Valores:** dígitos 0-9, `num_addends ∈ 1..K`, `num_digits ∈ 1..D`.
- **Padres:**
  - entradas y controles: ninguno;
  - cada `output_digit_d` ← **todos** los dígitos de entrada + `num_addends` + `num_digits`;
  - `raw_input` ← dígitos + controles + `template`;
  - `raw_output` ← los `output_digit_*`.
- **Mecanismos:**
  - dígitos: `random.randint(0, 9)`;
  - `compute_output_digits(trace)` (140-157): toma los primeros `num_addends` números con `num_digits` dígitos y llama a `processor.compute_sum`;
  - cada `output_digit_d` se crea con una fábrica `make_output_digit_mechanism(digit_idx)` (160-171) para que cada uno capture su propio índice (si se usara una lambda directa en el loop, todas verían el último `d`). Cada mecanismo **recalcula toda la suma** y se queda con el dígito `digit_idx`;
  - `raw_input`: `processor.generate_prompt(...)`;
  - `raw_output` (193-197): `" " + format_output(output_digits)`. Ojo al **espacio inicial**: es lo que el LM genera después de "is". Por eso la notebook hace `.strip()` al comparar.

La conclusión de la celda: "No explicit intermediate variables for sum or carry".

#### Samplear y correr

```python
sample = sample_valid_addition_input(config_2n_2d, num_addends=2, num_digits=2)
output = basic_model.new_trace(sample)
print(output['raw_input'], output['raw_output'])
num1 = processor.digits_to_number([sample["digit_0_0"], sample["digit_0_1"]])
...
print(f"Correct: {str(expected_sum) == output['raw_output'].strip()}")
```

`sample_valid_addition_input(config, num_addends, num_digits)` (450-495):
- valida que no se pase de K y D;
- fija `num_addends`, `num_digits` y una plantilla al azar;
- para dígitos activos: el más significativo (`d == 0`) va de 1 a 9 (así un número "de 2 dígitos" es realmente 10-99), el resto 0-9;
- dígitos inactivos (sumandos o posiciones de más): 0.

`new_trace(sample)` calcula todo el grafo desde esas entradas (`causalab/causal/trace.py`). Salida guardada: 14 + 73 → prompt `'The sum of 14 and 73 is'`, respuesta `' 87'`, `Correct: True`. La celda de "Multiple Examples" repite esto 5 veces (20+25 = 45, 53+93 = 146, ...); el `✓` final está escrito fijo en el `print`, no es un chequeo.

### Step 4: el modelo intermedio con acarreos

```python
from causalab.tasks.general_addition.causal_models import create_intermediate_addition_model
intermediate_model = create_intermediate_addition_model(config_2n_2d)
# id: addition_intermediate_2n_2d, 13 variables
carry_vars = [v for v in intermediate_model.variables if v.startswith("C_")]   # ['C_1', 'C_2']
output_vars = [v for v in intermediate_model.variables if v.startswith("O_")]  # ['O_1', 'O_2', 'O_3']
```

`create_intermediate_addition_model` (`causal_models.py` 206-447), solo para **2 sumandos** (`K = 2` fijo aunque la config tenga más):

- **Variables:** 2·D dígitos, `num_digits`, `template` (no hay `num_addends`), `C_1..C_D`, `O_1..O_{D+1}`, `raw_input`, `raw_output`. Para D = 2: 4 + 2 + 2 + 3 + 2 = 13.
- **Padres** (312-343), con `d = D - i`:
  - `C_1` ← `digit_0_{D-1}`, `digit_1_{D-1}` (las unidades);
  - `C_i` ← `digit_0_{D-i}`, `digit_1_{D-i}`, `C_{i-1}`;
  - `O_1` ← las unidades;
  - `O_i` (2 ≤ i ≤ D) ← dígitos de la posición i y `C_{i-1}`;
  - `O_{D+1}` ← `C_D`;
  - `raw_output` ← todos los `O_i`.
- **Mecanismos** (360-410): exactamente las fórmulas de arriba. Para que cada mecanismo capture su `d` y su `i` se usan argumentos por defecto (`def mechanism(trace, d=d)`) o fábricas (`make_carry_mechanism(pos, d=d)`, `make_output_mechanism(pos, d=d)`). `O_{D+1}` devuelve `trace[f"C_{D}"]`.
- `raw_input`: `generate_prompt(digit_dict, 2, num_digits, template)`.
- `raw_output` (431-441): junta `[O_1, ..., O_{D+1}]` (derecha→izquierda), **invierte** la lista para pasarla a izquierda→derecha, y formatea con espacio inicial.

Este grafo es una cadena: `C_1 → C_2 → ... → C_D`, y cada `C_{i-1}` alimenta a `O_i`. Esa estructura es la hipótesis "la red propaga acarreos".

#### 27 + 45 = 72 paso a paso

```python
example_input = {"digit_0_0": 2, "digit_0_1": 7, "digit_1_0": 4, "digit_1_1": 5,
                 "num_digits": 2, "template": config_2n_2d.templates[0]}
intermediate_output = intermediate_model.new_trace(example_input)
```

Salida guardada: `C_1 = 1` (7+5 = 12), `O_1 = 2`, `C_2 = 0` (2+4+1 = 7), `O_2 = 7`, `O_3 = 0`, respuesta `' 72'`. Los textos explicativos ("since 7 + 5 = 12 >= 10") están escritos a mano; los números entre llaves sí salen del modelo.

#### 58 + 67 = 125 (dos acarreos)

Igual con `5,8 / 6,7`: `C_1 = 1, O_1 = 5, C_2 = 1, O_2 = 2, O_3 = 1` → `' 125'`. Muestra la propagación unidades → decenas → centenas.

### Step 5: básico vs intermedio

```python
test_input = {"digit_0_0": 3, "digit_0_1": 4, "digit_1_0": 5, "digit_1_1": 9,
              "num_addends": 2, "num_digits": 2, "template": ...}
basic_result = basic_model.new_trace(test_input)
intermediate_result = intermediate_model.new_trace(test_input)
```

34 + 59: el básico da `output_digit_* = [0, 9, 3]` y `' 93'`; el intermedio `C_1=1, O_1=3, C_2=0, O_2=9, O_3=0` y `' 93'`. Mismo comportamiento, distinta estructura. Ojo: `test_input` incluye `num_addends`, que el modelo intermedio no tiene (la versión vieja de la API lo toleraba; ver "Ojo").

### Step 6: intervenir un acarreo

```python
input_no_carry = {"digit_0_0": 2, "digit_0_1": 3, "digit_1_0": 4, "digit_1_1": 5,
                  "num_digits": 2, "template": ...}
output_original = intermediate_model.new_trace(input_no_carry)            # 68, C_1 = 0
output_intervened = intermediate_model.new_trace({**input_no_carry, 'C_1': 1})
```

- `{**input_no_carry, 'C_1': 1}` agrega `C_1 = 1` al diccionario de entradas. Como `C_1` va **al final** del diccionario, se fija después de que los dígitos ya lo calcularon, y pisa su valor; luego se recalculan sus descendientes (`O_2`, `C_2`, `O_3`, `raw_output`).
- Salida guardada: `C_1 = 1`, `O_1 = 8` (no depende de `C_1`), `O_2 = 7` (2+4+1), respuesta `78`.

Esto es una **intervención** en el sentido causal: el mundo "23 + 45" pero con el acarreo de las unidades forzado. El modelo básico no tiene dónde hacer esta intervención: es la ventaja del intermedio para formular hipótesis.

### Step 7: 3 dígitos

```python
config_3d = create_two_number_three_digit_config()
model_3d = create_intermediate_addition_model(config_3d)
result_3d = model_3d.new_trace({... 123 y 456 ..., "num_digits": 3, ...})
```

`C_1 = C_2 = C_3 = 0`, `O_1 = 9, O_2 = 7, O_3 = 5, O_4 = 0` → `' 579'`. La misma construcción escala a cualquier D.

### "What We've Built"

Markdown de resumen: configuración, plantillas, dos modelos (básico e intermedio), la pregunta de investigación (¿la red aprende un mapeo directo o propaga acarreos?) y las convenciones de nombres (entradas desde la izquierda en 0, carries/salidas desde la derecha en 1).

---

## Ejercicios

No hay ejercicios ni celdas `# TERRON` / `FILL IN`. Para practicar: predecir a mano qué da forzar `C_1 = 1` en 45 + 54 (las dos posiciones suman 9, así que el acarreo forzado debería propagarse: `O_1` sigue en 9 porque no depende de `C_1`, pero `O_2` pasa a 0, `C_2` a 1 y `O_3` a 1: respuesta 109; esto es lo que explora la notebook 03) y en 23 + 45 forzar `C_2 = 1` (solo cambia `O_3`, respuesta 168).

---

## Ojo / errores comunes / dudas

- **API vieja de `CausalModel`.** Ambos modelos terminan con `CausalModel(variables, values, parents, mechanisms, id=...)`, pero el constructor actual (`causalab/causal/causal_model.py`) es `CausalModel(mechanisms, values, print_pos=None, id=...)` con objetos `Mechanism(parents, compute)`. Con la librería actual esta llamada falla. Las salidas guardadas son de la versión anterior; no las pude reproducir.
- **Intervenir pasando la variable como "entrada".** `new_trace({**inputs, 'C_1': 1})` funciona solo porque `C_1` queda después de sus padres en el diccionario. En la `CausalTrace` actual, `_set` no reemplaza el mecanismo; si después se fijara un padre, `C_1` se recalcularía y se perdería la intervención. La forma robusta es `trace = model.new_trace(inputs); trace.intervene("C_1", 1)` (o `trace["C_1"] = 1`, que llama a `intervene`).
- **Variables que el modelo no tiene.** Pasar `num_addends` al modelo intermedio (Step 5) da `KeyError` con la `CausalTrace` actual (busca los hijos de una variable inexistente).
- **`num_digits` en el modelo intermedio** solo afecta a `raw_input`; los acarreos siempre usan las D posiciones de la config (`d = D - i`). Si `num_digits < D`, el prompt y la respuesta calculada no corresponderían al mismo número.
- **Espacio inicial en `raw_output`** (`' 68'`): importa al comparar con lo que genera el LM.
- **Modelo básico ineficiente pero correcto:** cada `output_digit_d` recalcula toda la suma. Y como cada salida depende de todos los dígitos, intervenir un dígito de entrada mueve todas las salidas a la vez; no hay variable "parcial" que intervenir.
- **"Correct ✓" impreso fijo** en la celda de varios ejemplos: no es un chequeo real.

---

## Resumen de una línea por sección

- **Step 1:** `AdditionTaskConfig` define K, D y plantillas; `create_general_config` arma la plantilla para cualquier K.
- **Step 2:** `AdditionTemplateProcessor` convierte entre dígitos y números, llena la plantilla y calcula la suma con D+1 dígitos.
- **Step 3:** modelo básico: cada `output_digit_d` depende de todos los dígitos; `raw_output` con espacio inicial; `sample_valid_addition_input` evita ceros a la izquierda.
- **Step 4:** modelo intermedio (2 sumandos): cadena de acarreos `C_1 → ... → C_D`, `O_i` con índices desde la derecha, `O_{D+1} = C_D`.
- **Step 5:** ambos modelos dan la misma respuesta; difieren en estructura interna.
- **Step 6:** forzar `C_1 = 1` en 23+45 da 78: los acarreos son variables intervenibles.
- **Step 7:** la construcción escala a 3 dígitos (123 + 456 = 579, sin acarreos).
