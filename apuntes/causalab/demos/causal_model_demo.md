# ¿Qué es un modelo causal? (`causal_model_demo.ipynb`)

Notebook: `causalab/demos/causal_model_demo.ipynb`.

Es la notebook "cero" de la librería: no carga ningún modelo de lenguaje. Construye un **modelo causal** de juguete (sumar tres dígitos A, B, C módulo 10) con la API de `causalab`, lo visualiza, y muestra las dos operaciones que vamos a usar en todo el tutorial de onboarding: la **intervención** (fijar el valor de una variable) y la **intervención de intercambio** (*interchange intervention*: fijar una variable al valor que tomaría con *otro* input). La pregunta que responde es: ¿qué objeto es un modelo causal en esta librería y qué significa "intervenir" sobre él?

Conexión con lo que sigue: en [01_define_MCQA_task.md](01_define_MCQA_task.md) se usa exactamente esta misma maquinaria (`CausalModel`, `Mechanism`, `input_var`, `run_interchange`) pero para una tarea de preguntas de opción múltiple, y a partir de la 02 se hacen las mismas intervenciones de intercambio, pero dentro de un LLM. Para el detalle de las clases ver también [../01_causal.md](../01_causal.md).

---

## Ideas clave / teoría

**Modelo causal (grafo causal).** Un conjunto de variables, cada una con:
- un conjunto de **valores** posibles,
- un conjunto de **padres** (las variables de las que depende directamente),
- un **mecanismo**: la función que calcula su valor a partir de los valores de sus padres.

Las variables sin padres son **inputs**. Si hay un camino dirigido de α a β, cambiar α *puede* cambiar β ("α tiene efecto causal sobre β"). Correr el modelo "hacia adelante" (*forward*) es: fijar inputs y aplicar los mecanismos en orden topológico.

**Intervención** `do(X = x)`: reemplazar el mecanismo de X por la constante x. X deja de depender de sus padres, y todo lo que está "aguas abajo" de X se recalcula. Lo que está aguas arriba no cambia.

**Interchange intervention.** En vez de decir a mano "X = 4", tomás un segundo input (el *counterfactual* o *source*), corrés el modelo con él, leés el valor que toma X ahí, y lo metés en la corrida con el input original (*base*). Formalmente, si `b` es el input base y `s` el source:

  `IntInv(b, s, X) = M(b) con do(X = X(s))`

donde `X(s)` es el valor de X al correr el modelo sobre `s`. La gracia es que no necesitás saber "qué valor tiene sentido" para X: el valor viene de una corrida real. Esto es lo que hace posible intervenir redes neuronales, donde no sabemos qué valor "razonable" tiene una activación pero sí podemos copiarla de otra corrida (eso es *activation patching*, notebook 02).

**Hipótesis como grafo.** El modelo causal es una *hipótesis* sobre cómo un sistema resuelve una tarea (por ejemplo, "primero suma A+B y después le suma C"). Más adelante vamos a testear si una red neuronal "implementa" esa hipótesis comparando sus respuestas a interchange interventions con las del modelo causal.

**`raw_input` y `raw_output`.** `CausalModel` exige que existan estas dos variables (lo chequea con `assert` en `causalab/causalab/causal/causal_model.py`). `raw_input` es lo que se le pasaría a una red y `raw_output` lo que se compara con la salida de la red. Acá son triviales (una lista y un entero), pero en la tarea MCQA son el prompt de texto y el token de respuesta.

---

## Cómo está implementado en `causalab` (lo que de verdad pasa por debajo)

Antes del recorrido conviene ver las tres piezas de `causalab/causalab/causal/trace.py` y `causal_model.py`:

- `Mechanism` (dataclass en `trace.py`): tiene `parents: list[str]` y `compute: Callable[[CausalTrace], Any]`. `compute` recibe el *trace* completo (`t`) y lee lo que necesita con `t["A"]`. Implementa `__call__` así que `mechanism(trace)` llama a `compute(trace)`.
- `input_var(values)` (en `trace.py`): devuelve `Mechanism(parents=[], compute=lambda t: random.choice(values))`. O sea, un input "sin padres" cuyo mecanismo por defecto es samplear al azar. Cuando le das un valor explícito al crear el trace, ese muestreo no se usa.
- `CausalTrace` (en `trace.py`): el **estado de una corrida**. Guarda `self._values` (dict variable → valor) y una copia de los mecanismos. Al construirse con `inputs`, para cada input llama `_set`, que guarda el valor y llama `_recompute_descendants`: recorre recursivamente los hijos y calcula cada hijo **solo si todos sus padres ya tienen valor**. Así, al terminar de setear todos los inputs, quedan calculadas todas las variables.
  - `trace["X"]` → `get`, lanza `KeyError` si X no se calculó.
  - `trace.intervene(X, v)`: guarda el valor **y reemplaza el mecanismo de X por una constante** (`Mechanism(parents=[], compute=lambda t, v=value: v)`), y recalcula descendientes. Reemplazar el mecanismo es lo que "rompe la flecha" desde los padres: si después cambiás un ancestro, X no se recalcula.
  - `trace["X"] = v` (`__setitem__`) llama a `intervene`, o sea que asignar ya tiene semántica de intervención.
  - `copy()` hace deepcopy de mecanismos y copia superficial de valores.
- `CausalModel(mechanisms, values, id=...)` (en `causal_model.py`): en el constructor
  - deriva `variables`, `parents`, `children`;
  - `inputs` = variables sin padres; `outputs` = variables que no son padres de nadie;
  - calcula `timesteps` (profundidad en el grafo, inputs en 0) y valida que cada padre tenga timestep menor que el hijo (es decir, que el grafo sea acíclico y consistente);
  - exige que todas las variables estén en `values`;
  - ordena `variables` por timestep y arma posiciones para graficar.
  - `new_trace(inputs)` crea un `CausalTrace` con una deepcopy de los mecanismos.
  - `run_interchange(input_trace, {var: cf_trace})`: copia el trace base y, por cada variable, hace `trace.intervene(var, cf_trace[var])`. Soporta además la sintaxis `"A<-B"` (fijar A del base al valor de B del counterfactual).

Ojo: `values` **no se usa para validar** los valores que asignás; solo se usa en `sample_input` (que samplea inputs de `values`) y para chequear que todas las variables estén declaradas. Por eso en la notebook se puede intervenir `Y = 10` aunque `values["Y"]` sea 0..9.

---

## Recorrido por la notebook

### 1. "Let's set up a causal model" (celda 4)

```python
from causalab.causal.causal_model import CausalModel
from causalab.causal.trace import Mechanism, input_var
from causalab.causal.causal_viz import (
    display_structure,
    display_forward_pass,
    display_interchange,
)

DIGITS = list(range(0, 10))
```

- Importa la clase del modelo, los constructores de mecanismos y tres funciones de visualización de `causalab/causalab/causal/causal_viz.py` (son apps interactivas de Dash + Cytoscape; por eso la salida guardada es solo `<IPython.lib.display.IFrame ...>`).
- `DIGITS` = 0..9: todo es aritmética módulo 10.

```python
values = {
    "A": DIGITS, "B": DIGITS, "C": DIGITS,
    "A+B": DIGITS, "C'": DIGITS, "Y": DIGITS,
    "raw_input": None, "raw_output": None,
}
```

- Dominio de cada variable. `None` para `raw_input`/`raw_output` porque son "strings crudos" sin dominio finito. Como dije arriba, solo sirve para declarar las variables (y para `sample_input`).

```python
mechanisms = {
    "A": input_var(DIGITS),
    "B": input_var(DIGITS),
    "C": input_var(DIGITS),
    "A+B": Mechanism(parents=["A", "B"], compute=lambda t: (t["A"] + t["B"]) % 10),
    "C'": Mechanism(parents=["C"], compute=lambda t: t["C"]),
    "Y": Mechanism(parents=["A+B", "C'"], compute=lambda t: (t["A+B"] + t["C'"]) % 10),
    "raw_input": Mechanism(parents=["A", "B", "C"], compute=lambda t: [t["A"], t["B"], t["C"]]),
    "raw_output": Mechanism(parents=["Y"], compute=lambda t: t["Y"]),
}
causal_model = CausalModel(mechanisms, values, id="Hierarchical addition")
```

- `A`, `B`, `C`: inputs (sin padres).
- `A+B`: suma parcial; depende de A y B.
- `C'`: una copia de C. Parece inútil, pero sirve para que el grafo tenga dos variables intermedias "en paralelo" (A+B y C') que luego alimentan a Y. Esto modela un algoritmo en dos pasos: el resultado final **no** depende directamente de A, B, C sino de las variables intermedias.
- `Y = (A+B + C') mod 10`. Notar que usa las intermedias, no los inputs: si intervenís `A+B`, Y cambia aunque A y B sigan igual.
- `raw_input` = la lista `[A, B, C]` (lo que "vería" una red), `raw_output` = Y.
- `CausalModel(...)` valida la estructura y calcula timesteps: A, B, C = 0; A+B, C' = 1; Y = 2; raw_output = 3. Las variables que no son padres de nadie (`outputs`: acá `raw_input` y `raw_output`) reciben al final `end_time` = 3, así que `raw_input` termina con timestep 3 aunque sus padres sean inputs. Esto solo afecta el orden de `variables` y las posiciones del dibujo.

### 2. Visualizar el grafo y correr hacia adelante (celdas 6 y 8)

```python
display_structure(causal_model)

inputs = {"A": 1, "B": 2, "C": 3}
display_forward_pass(causal_model, inputs)
```

- `display_structure` dibuja nodos y aristas padre → hijo, con layout `breadthfirst` desde `raw_input`.
- `display_forward_pass(model, inputs, intervention=None)`: por dentro hace `trace = model.new_trace(inputs)`, después `trace.intervene(var, val)` para cada intervención, y `trace.to_dict()` para tener todos los valores. Con 1, 2, 3: A+B = 3, C' = 3, Y = 6.
- Colores: inputs en teal oscuro, valores calculados en teal claro.

### 3. Una red neuronal también es un modelo causal (celdas 10–11)

```python
l1 = torch.nn.Linear(3, 3)
l2 = torch.nn.Linear(3, 3)
l3 = torch.nn.Linear(3, 1)
variables = ["A", "B", "C", *[f"h1{i+1}" for i in range(3)], *[f"h2{i+1}" for i in range(3)],
             "h31", "raw_input", "raw_output"]
values = {var: None for var in variables}
```

- Tres capas lineales con pesos aleatorios (no entrenadas: el punto es estructural, no que sume bien).
- Cada neurona es una variable: `h11..h13` (capa 1), `h21..h23` (capa 2), `h31` (salida).

```python
"h11": Mechanism(
    parents=["A", "B", "C"],
    compute=lambda t: l1(torch.FloatTensor([t["A"], t["B"], t["C"]])).round()[0].item(),
),
...
"h21": Mechanism(parents=["h11", "h12", "h13"],
    compute=lambda t: l2(torch.FloatTensor([t["h11"], t["h12"], t["h13"]])).round()[0].item()),
...
"h31": Mechanism(parents=["h21", "h22", "h23"],
    compute=lambda t: l3(torch.FloatTensor([t["h21"], t["h22"], t["h23"]])).round().item()),
"raw_output": Mechanism(parents=["h31"], compute=lambda t: t["h31"]),
```

- Cada `h1i` aplica toda la capa `l1` al vector `[A, B, C]` (tensor de shape `(3,)`), redondea, y se queda con la componente i (`[i]`), convertida a float de Python con `.item()`. Es ineficiente (recalcula la capa 3 veces) pero deja clarísimo que **cada neurona es una variable con mecanismo = su fila de la capa**.
- La capa 2 depende solo de la capa 1, la 3 solo de la 2: la estructura del grafo es la del MLP.
- El `.round()` es para que los valores sean "discretos" y se vean lindos en la visualización; no tiene importancia conceptual.

Mensaje: las mismas operaciones (intervenir, interchange) que hacemos sobre el modelo de alto nivel se pueden hacer sobre la red. La teoría de **abstracción causal** compara ambos.

### 4. Intervenciones (celdas 17, 20, 23, 26)

```python
inputs = {"A": 1, "B": 2, "C": 3}
intervention = {"A+B": 4}
display_forward_pass(causal_model, inputs, intervention=intervention)
```

- `do(A+B = 4)`: el mecanismo de A+B se reemplaza por la constante 4; Y = (4 + 3) mod 10 = 7 en vez de 6.
- Colores: la variable intervenida en magenta oscuro. Para el resto, `display_forward_pass` usa `_get_descendants(model, intervention)` dos veces: con `strict=True` obtiene las variables **todas** cuyas causas están en el conjunto intervenido (magenta claro: dependen solo de la intervención), y con `strict=False` las que tienen **al menos un** padre afectado (violeta: mezcla de intervención y valores base). Y depende de A+B (intervenido) y de C' (base), así que sale violeta.

```python
intervention = {"Y": 10}
```

- Intervenir directamente la salida. `raw_output` depende solo de Y, así que queda magenta (no violeta). Y = 10 está fuera de `values["Y"]` y no pasa nada: la librería no valida.

```python
intervention = {"C": 9}
```

- Intervenir un input. Equivale a correr con C = 9, pero el modelo te dice además cuánto vale cada intermedio (C' = 9, Y = (3 + 9) mod 10 = 2).

```python
intervention = {"A+B": 8, "C'": 4}
```

- Intervención múltiple: ambos padres de Y intervenidos, así que Y = (8 + 4) mod 10 = 2 sin importar los inputs. Los inputs quedan "desconectados" de la salida.

### 5. Interchange interventions (celdas 31 y 34)

```python
inputs = {"A": 1, "B": 2, "C": 3}
counterfactual_inputs = {
    "A+B": {"A": 3, "B": 1, "C": 3},
}
display_interchange(causal_model, inputs, counterfactual_inputs)
```

- Base: (1, 2, 3). Source para la variable A+B: (3, 1, 3).
- `display_interchange` (en `causal_viz.py`) convierte cada dict en un trace con `model.new_trace`, copia el trace base y hace `result_trace.intervene(var, cf_trace[var])`. Es decir: corre el source, lee A+B(source) = 4, y hace `do(A+B = 4)` en la corrida base. Resultado Y = 7, igual que en la intervención de la celda 17, pero sin haber "inventado" el 4.
- Esto es exactamente lo que hace `CausalModel.run_interchange` (que se usa en la 01).

```python
counterfactual_inputs = {
    "A+B": {"A": 3, "B": 1, "C": 3},  # A+B = 4
    "C'": {"A": 1, "B": 2, "C": 8},   # C' = 8
}
```

- Interchange múltiple: cada variable puede tener **su propio source**. A+B viene del primer source (4), C' del segundo (8), Y = 2.
- Esto va a reaparecer en la librería: en un `InterchangeTarget` cada "grupo" de unidades del modelo neural comparte un counterfactual, y distintos grupos pueden tener distintos counterfactuals (ver [04_train_DAS_and_DBM.md](04_train_DAS_and_DBM.md) y `causalab/causalab/neural/model_units.py`).

---

## Ejercicios

La notebook no tiene ejercicios formales; solo invita a cambiar los inputs ("play around with different inputs!"). Cosas que vale la pena probar para fijar ideas:
- Cambiar el source de A+B a uno donde A+B(source) = A+B(base): la interchange no cambia nada. Esa es la idea de **confounding** que se formaliza en la 01: un par base/source que no cambia la salida no te da información sobre la variable.
- Intervenir `raw_input` y ver que no afecta a Y: `raw_input` es una hoja (un output del grafo), no está en el camino hacia Y.

---

## Ojo / errores comunes / dudas

- **`values` no valida nada**: podés intervenir con valores fuera del dominio (como `Y = 10`).
- **`intervene` muta el trace**. `trace.intervene(...)` modifica el objeto y devuelve `self`. Si querés conservar el original, hacé `trace.copy().intervene(...)` (así se hace en la 01).
- **`_recompute_descendants` solo calcula un hijo si todos sus padres están listos.** Si creás un trace con inputs incompletos, las variables que dependen de los inputs faltantes simplemente no existen y `trace["Y"]` lanza `KeyError`.
- **En la red de juguete los pesos son aleatorios** (`torch.nn.Linear` sin entrenar), así que `raw_output` no es la suma. El ejemplo es puramente estructural.
- **Las visualizaciones son apps Dash** (necesitan `dash` y `dash_cytoscape`); en la salida guardada solo se ve el objeto `IFrame`, no la imagen. Si la querés ver, hay que ejecutar la notebook.
- Interchange ≠ intervención "cualquiera": la interchange garantiza que el valor que ponés es un valor que la variable **realmente toma** con algún input. En una red neuronal esto evita meter activaciones "fuera de distribución".

---

## Resumen de una línea por sección

- **Set up a causal model**: un `CausalModel` es un dict de `Mechanism` (padres + función) más un dict de dominios; `input_var` crea inputs sin padres.
- **Visualizar / forward pass**: `new_trace(inputs)` propaga los valores por el grafo en orden topológico.
- **Red neuronal como modelo causal**: cada neurona es una variable cuyo mecanismo es su fila de la capa; misma API.
- **Intervenciones**: `intervene` reemplaza el mecanismo por una constante y recalcula solo lo de aguas abajo.
- **Interchange interventions**: el valor de la variable se toma de una corrida con otro input (`run_interchange`); cada variable puede tener su propio source.
