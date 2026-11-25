# Lecture 8: Introducción a la abstracción causal

Esta notebook presenta la teoría de **causal abstraction** (abstracción causal): el marco que dice cuándo un proceso causal simple (un algoritmo simbólico) describe fielmente a un proceso causal más complejo (una red neuronal). La pregunta que responde es: ¿qué significa exactamente que una red neuronal *implemente* un algoritmo? La respuesta que construye es: la red implementa el algoritmo cuando, para todo input y toda interchange intervention, ambos modelos se comportan igual "módulo traducción". En la Lecture 7 usamos interchange interventions (activation patching) de forma más o menos empírica; acá se les da la teoría que las justifica y que dice cuáles hay que correr para probar una hipótesis.

Todo se trabaja con un ejemplo de juguete: la **tarea de igualdad jerárquica**, un algoritmo para resolverla (modelo causal $\mathcal{A}$) y un MLP construido a mano que la resuelve perfectamente (modelo causal $\mathcal{N}$). Al final medimos la **interchange intervention accuracy (IIA)** de una alineación correcta y de una incorrecta.

Archivos de `causalab` que usa la notebook:

- `causalab/causalab/causal/causal_model.py` → clase `CausalModel`
- `causalab/causalab/causal/trace.py` → `Mechanism`, `input_var`, `CausalTrace`
- `causalab/causalab/causal/causal_viz.py` → `print_structure`, `print_setting`

---

## Ideas clave / teoría

### Modelo causal

Un **modelo causal** es un grafo dirigido acíclico (DAG) con:

- **Variables**: los nodos (inputs, cómputos intermedios, outputs).
- **Valores**: el conjunto de estados posibles de cada variable.
- **Padres / hijos**: las aristas. Si $X$ es padre de $Y$, el valor de $Y$ depende directamente del de $X$.
- **Mecanismos**: una función por variable que calcula su valor a partir de los valores de sus padres. Las variables sin padres son los inputs: su valor viene "de afuera".

Lo importante es que esta definición sirve igual para un algoritmo simbólico (variables = pasos intermedios del algoritmo) y para una red neuronal (variables = neuronas, aristas = pesos distintos de cero, mecanismos = "suma ponderada + activación"). Esa simetría es la que permite comparar los dos con el mismo lenguaje.

### Trace

Un **trace** (o *setting*) es la asignación completa de un valor a cada variable que resulta de correr el modelo sobre un input. Se escribe $\mathcal{A}(i)$. En `causalab` es un objeto `CausalTrace`.

### Intervención

Una **intervención** $\mathcal{A}_{X \leftarrow x}$ fuerza la variable $X$ al valor $x$ y **corta** la dependencia de $X$ respecto de sus padres: el mecanismo de $X$ se reemplaza por la constante $x$. Todo lo que está aguas abajo de $X$ se recalcula con el valor nuevo. Una **hard intervention** es justamente eso: fijar a constante.

Las intervenciones son lo que permite hablar de causalidad y no solo de correlación: si al cambiar $X$ (y solo $X$) cambia el output, $X$ tiene un efecto causal sobre el output.

### Interchange intervention

Es una intervención donde el valor que se fija es el que $X$ habría tomado con **otro input**:

$$\mathcal{A}(i_{\text{orig}})_{X \leftarrow \mathcal{A}(i_{\text{counter}})}$$

- $i_{\text{orig}}$: el input **base** (original); es el que se corre.
- $i_{\text{counter}}$: el input **source** (contrafactual); de su corrida se toma el valor de $X$.
- $X$: la variable objetivo.

Se lee: "corré el modelo sobre el base, pero con $X$ puesto en el valor que tendría sobre el source". Es exactamente el activation patching de la Lecture 7, solo que expresado sobre un modelo causal arbitrario.

### Alineación $(\Pi, \pi)$ y abstracción constructiva

Para comparar el algoritmo $\mathcal{A}$ (alto nivel, $\mathcal{H}$) con la red $\mathcal{N}$ (bajo nivel, $\mathcal{L}$) hace falta una **alineación** con dos ingredientes:

1. **Conjuntos alineados $\Pi$**: a cada variable de alto nivel $V$ se le asigna un *conjunto* de variables de bajo nivel $\Pi_V$ (neuronas). Ojo: un conjunto, no una neurona. Y los conjuntos de distintas variables son disjuntos.
2. **Mapas de traducción $\pi_V$**: funciones que leen el valor de alto nivel de $V$ a partir de los valores de las neuronas en $\Pi_V$.

De ahí salen tres mapas derivados:

- $\tau_\pi$ (**traducción de traces**): toma un trace completo de la red y produce un trace del algoritmo aplicando cada $\pi_V$.
- $\omega_\pi$ (**traducción de intervenciones**): toma una intervención en la red (sobre un conjunto de neuronas) y devuelve la intervención correspondiente en el algoritmo (sobre una variable).
- $\rho_\pi$ (**traducción de inputs**): toma un input del algoritmo y lo pasa al formato de input de la red.

Esto es la **abstracción constructiva**: las variables de alto nivel se "construyen" agrupando (clustering) variables de bajo nivel y traduciendo sus valores.

### La ecuación central

$\mathcal{H}$ es una abstracción de $\mathcal{L}$ si para todo input $i$ y toda intervención $X \leftarrow x$:

$$\tau_\pi\big(\textsf{Run}(\mathcal{L}_{X \leftarrow x}(i))\big) = \textsf{Run}\big(\mathcal{H}_{\omega_\pi(X \leftarrow x)}(\rho_{\pi}(i))\big)$$

Término a término:

- Lado izquierdo: intervengo la red ($\mathcal{L}_{X \leftarrow x}$), la corro sobre $i$ (Run), y traduzco el trace resultante al lenguaje del algoritmo ($\tau_\pi$).
- Lado derecho: traduzco la intervención ($\omega_\pi$) y el input ($\rho_\pi$) al nivel alto, y corro el algoritmo intervenido.
- La igualdad exige que "intervenir y después traducir" sea lo mismo que "traducir y después intervenir". Es un diagrama que conmuta. La idea viene de Rubenstein et al. (2017).

La notebook agrega una observación de Geiger et al. (2026): todas las interchange interventions caen dentro del dominio de $\omega_\pi$ por construcción. La razón es que cada $\pi_X$ tiene que estar definida al menos para la intervención vacía (la corrida normal), y una interchange intervention justamente fija las neuronas de $\Pi_X$ a valores que salen de una corrida normal sobre otro input, así que $\pi_X$ sabe traducirlos.

### Interchange Intervention Accuracy (IIA)

En la práctica no se verifica la ecuación para *todas* las intervenciones, sino que se mide en un dataset $\mathcal{D}$ de pares (base, source):

$$\text{IIA} = \frac{1}{|\mathcal{D}|} \sum_{(b, s) \in \mathcal{D}} \mathbf{1}\!\left[\pi_O\!\left(\mathcal{N}(b)_{\Pi_V \leftarrow \mathcal{N}(s)}\right) = \mathcal{A}(\rho(b))_{V \leftarrow \mathcal{A}(\rho(s))}\right]$$

- $\mathcal{N}(b)_{\Pi_V \leftarrow \mathcal{N}(s)}$: corro la red sobre el base $b$ con las neuronas $\Pi_V$ reemplazadas por sus valores sobre el source $s$.
- $\pi_O(\cdot)$: traduzco el output de la red a un valor del algoritmo (acá, True/False).
- $\mathcal{A}(\rho(b))_{V \leftarrow \mathcal{A}(\rho(s))}$: lo que predice el algoritmo con la interchange en $V$ (en la práctica se compara la variable de output $O$).
- $\mathbf{1}[\cdot]$: 1 si coinciden, 0 si no. Se promedia.

Dos detalles: (1) solo se compara el **output**, no el trace completo, así que IIA es una versión debilitada de la ecuación central; (2) IIA = 1 indica abstracción perfecta *sobre ese dataset*. Qué tan fuerte es la evidencia depende de qué pares hay en $\mathcal{D}$ (tema de la Lecture 9).

**KL-IIA** reemplaza el indicador por una divergencia KL entre distribuciones de output, útil cuando la red da probabilidades (logits de un LM):

$$\text{KL-IIA} = 1 - \frac{1}{|\mathcal{D}|} \sum_{(b,s)} D_{\text{KL}}\big(\mathcal{A}(\ldots)_{V\leftarrow\ldots} \,\|\, \pi_O(\mathcal{N}(b)_{\Pi_V \leftarrow \mathcal{N}(s)})\big)$$

Para el MLP determinista de la notebook coincide con IIA. La **abstracción aproximada** de Beckers et al. (2019) toma en cambio la distancia del peor caso sobre todas las intervenciones e inputs y pide que sea $\le \alpha$; IIA es una versión promediada, solo sobre outputs y solo sobre interchange interventions.

---

## Recorrido por la notebook

### Setup (celdas 3-5)

La celda 3 detecta si está en Colab: si sí, clona el repo e instala `requirements/08_causal_abstraction_i.txt`; si no, corre `uv sync` y `plotly_get_chrome` (para exportar plots de plotly a PNG). `clear_output()` borra la salida de la instalación.

```python
import random
import numpy as np

from causalab.causal.causal_model import CausalModel
from causalab.causal.trace import Mechanism, input_var
from causalab.causal.causal_viz import print_structure, print_setting

seed = 42
np.random.seed(seed)
random.seed(seed)
```

- Importa las tres piezas de `causalab` que se usan en toda la notebook (las explico abajo cuando aparecen).
- Fija las semillas de `numpy` y `random`. Ojo: `random` importa porque `input_var` y `CausalModel.sample_input` usan `random.choice` por dentro, y la celda de IIA usa `random.sample`.

### 1. Algoritmos como modelos causales

#### La tarea de igualdad jerárquica

Input: cuatro letras. Output: `True` si los dos pares tienen la **misma relación de igualdad**. O sea, `True` si (primer par igual y segundo par igual) o (primer par distinto y segundo par distinto).

| Input | Par 1 | Par 2 | Output |
|---|---|---|---|
| AABB | igual | igual | True |
| ABCD | distinto | distinto | True |
| ABCC | distinto | igual | False |
| AABC | igual | distinto | False |

Es "jerárquica" porque es igualdad de igualdades: primero se compara dentro de cada par, después se comparan los dos resultados. Eso da un algoritmo con dos variables intermedias obvias:

- $I_1..I_4$: las letras.
- $E_1 = (I_1 = I_2)$, $E_2 = (I_3 = I_4)$.
- $O = (E_1 = E_2)$.

#### Celda 10: definir el algoritmo $\mathcal{A}$

```python
letters = list("ABCDEFGHIJ")

values = {
    "I1": letters,
    "I2": letters,
    "I3": letters,
    "I4": letters,
    "E1": [True, False],
    "E2": [True, False],
    "O": [True, False],
    "raw_input": [None],
    "raw_output": [True, False],
}
```

- `letters`: alfabeto de 10 letras, `['A', ..., 'J']`.
- `values`: diccionario variable → lista de valores posibles. Las letras para los inputs, booleanos para los intermedios y el output.
- `raw_input` / `raw_output`: variables de "infraestructura". `CausalModel` exige que existan (ver abajo, hay un `assert`). Dan un nombre estable al input completo y al output del modelo, para poder conectar el modelo causal con una red neuronal después (la red recibe un `raw_input` y produce un `raw_output`). El valor `[None]` de `raw_input` es un placeholder: la lista de valores solo se usa para muestrear inputs, y `raw_input` acá no es un input (tiene padres), así que nunca se muestrea.

```python
mechanisms = {
    "I1": input_var(letters),
    "I2": input_var(letters),
    "I3": input_var(letters),
    "I4": input_var(letters),
    "E1": Mechanism(parents=["I1", "I2"], compute=lambda t: t["I1"] == t["I2"]),
    "E2": Mechanism(parents=["I3", "I4"], compute=lambda t: t["I3"] == t["I4"]),
    "O": Mechanism(parents=["E1", "E2"], compute=lambda t: t["E1"] == t["E2"]),
    "raw_input": Mechanism(
        parents=["I1", "I2", "I3", "I4"],
        compute=lambda t: (t["I1"], t["I2"], t["I3"], t["I4"])
    ),
    "raw_output": Mechanism(parents=["O"], compute=lambda t: t["O"]),
}
```

Acá aparecen `Mechanism` e `input_var`. En `causalab/causalab/causal/trace.py`:

- `Mechanism` es un `@dataclass` con dos campos: `parents: list[str]` y `compute: Callable[[CausalTrace], Any]`. Su `__call__(trace)` simplemente devuelve `self.compute(trace)`. O sea, un mecanismo es "lista de padres + función que recibe el trace y lee de él lo que necesite". La función recibe el trace entero, no solo los padres; la lista `parents` es la que define el grafo, y es responsabilidad de uno que `compute` solo lea los padres declarados.
- `input_var(values)` devuelve `Mechanism(parents=[], compute=lambda t: random.choice(values))`. Es un mecanismo sin padres. En la práctica su `compute` casi nunca se ejecuta: cuando uno crea un trace con inputs, los valores se fijan directamente (ver `CausalTrace._set`), y un nodo sin padres nunca es "hijo" de nadie, así que nunca se recalcula. Lo que importa de `input_var` es que tenga `parents=[]`, que es lo que hace que `CausalModel` lo clasifique como input.

Línea por línea:

- `I1..I4`: inputs, sin padres.
- `E1`: padres `I1, I2`; `compute` lee los dos valores del trace `t` y devuelve si son iguales. Ídem `E2` con `I3, I4`.
- `O`: padres `E1, E2`, devuelve `E1 == E2`.
- `raw_input`: padres los cuatro inputs, empaqueta la tupla `(I1, I2, I3, I4)`.
- `raw_output`: copia `O`.

```python
pos = {"I1": (0, 0), ..., "O": (1.5, 2)}

equality_model = CausalModel(mechanisms, values, print_pos=pos, id="hierarchical_equality")

print("Variables:", equality_model.variables)
print("Inputs:", equality_model.inputs)
print("Outputs:", equality_model.outputs)
```

- `pos`: coordenadas (x, y) para dibujar cada nodo. No afecta el cómputo.
- `CausalModel(...)`: construye el modelo. Qué hace el constructor (`causalab/causalab/causal/causal_model.py`, líneas 32-167):
  1. Guarda `mechanisms`, `values`, `id`. `self.variables = list(mechanisms.keys())`.
  2. `assert` que existan `raw_input` y `raw_output`.
  3. `self.parents[var] = mechanism.parents` para cada variable, y construye `self.children` invirtiendo esa relación.
  4. `self.inputs` = variables con cero padres.
  5. `self.outputs` = variables que **no son padre de nadie** (arranca con todas y va sacando las que aparecen como padre). Ojo: esto es "nodos hoja" del grafo, no "la salida del algoritmo".
  6. Calcula `self.timesteps`: inputs en 0; en cada paso, a los hijos de variables con timestep `step-1` les pone `step`. Como un hijo puede ser reasignado en pasos posteriores, termina valiendo la longitud del camino más largo desde un input. Después fuerza **todos los outputs (hojas) al último timestep** (`end_time`).
  7. Valida consistencia: toda variable tiene valores, hijos, mecanismo y timestep; padres/hijos son recíprocos; un padre siempre tiene timestep estrictamente menor que su hijo (esto en la práctica chequea que sea un DAG bien ordenado).
  8. Ordena `self.variables` por timestep.
  9. Completa posiciones de dibujo que falten (`raw_input` va por defecto a `(0, -2)`).

Salida guardada:

```
Variables: ['I1', 'I2', 'I3', 'I4', 'E1', 'E2', 'O', 'raw_input', 'raw_output']
Inputs: ['I1', 'I2', 'I3', 'I4']
Outputs: ['raw_input', 'raw_output']
```

Lo de `Outputs` confunde la primera vez: `raw_input` sale como "output" porque nadie depende de él (es una hoja del grafo), aunque conceptualmente sea el input. Por el punto 6, termina en el último timestep y por eso aparece al final de `variables`.

#### Celda 12: visualizar la estructura

```python
print_structure(equality_model)
print("Timesteps:", equality_model.timesteps)
```

`print_structure` (`causalab/causalab/causal/causal_viz.py`, línea 644) arma un `networkx.DiGraph` con una arista `(padre, hijo)` por cada padre de cada variable y lo dibuja con matplotlib usando `model.print_pos`. Nada más.

Salida guardada de timesteps: `{'I1': 0, 'I2': 0, 'I3': 0, 'I4': 0, 'E1': 1, 'raw_input': 3, 'E2': 1, 'O': 2, 'raw_output': 3}`. Se ve lo explicado: `raw_input` tendría timestep 1 por camino, pero como es hoja lo empujan a 3 (`end_time`).

#### Celdas 14-18: traces

```python
trace = equality_model.new_trace({"I1": "A", "I2": "A", "I3": "B", "I4": "B"})
print(f"E1 (I1 == I2): {trace['E1']}")
...
```

`CausalModel.new_trace(inputs)` (línea 171) devuelve `CausalTrace(mechanisms=copy.deepcopy(self.mechanisms), inputs=inputs)`. El `deepcopy` es clave: cada trace tiene su propia copia de los mecanismos, así que intervenir un trace no modifica el modelo ni otros traces.

Qué hace `CausalTrace.__init__` (`trace.py`):

1. Guarda los mecanismos y calcula `self.children` a partir de los `parents`.
2. `self._values = {}`.
3. Para cada input, llama `self._set(var, val)`, que guarda el valor y llama `_recompute_descendants(var)`.

`_recompute_descendants(variable)` recorre los hijos de `variable`; para cada uno, **si todos sus padres ya tienen valor**, lo calcula con su mecanismo y recursivamente recalcula sus descendientes. Por eso, al setear `I1` no pasa nada con `E1` (falta `I2`); al setear `I2` ya se calcula `E1`; y cuando están los cuatro, se completa todo hasta `raw_output`. Es una propagación hacia adelante perezosa, sin orden topológico explícito.

`trace['E1']` usa `__getitem__` → `get`, que tira `KeyError` si la variable no se calculó.

Salidas guardadas: para AABB, `E1=True, E2=True, O=True`; para ABCC, `E1=False, E2=True, O=False`. Esto es $\mathcal{A}(AABB)$ y $\mathcal{A}(ABCC)$.

```python
print_setting(equality_model, trace.to_dict())
```

- `trace.to_dict()` devuelve `self.copy()._values`, el diccionario variable → valor (copiado).
- `print_setting` (`causal_viz.py`, línea 686) dibuja el mismo grafo pero renombra cada nodo a `"var:\n valor"`. Ojo: usa `model.parents`, o sea la estructura del **modelo**, no la del trace. Si el trace fue intervenido, el dibujo sigue mostrando las aristas originales hacia la variable intervenida aunque esa dependencia ya esté cortada.

#### Celdas 23-24: hard intervention

```python
trace_intervened = equality_model.new_trace({"I1": "A", "I2": "A", "I3": "B", "I4": "B"})
trace_intervened.intervene("E1", False)
```

`CausalTrace.intervene(variable, value)` (`trace.py`):

```python
self._values[variable] = value
self.mechanisms[variable] = Mechanism(parents=[], compute=lambda t, v=value: v)
self._recompute_descendants(variable)
```

- Pone el valor.
- **Reemplaza el mecanismo** de la variable por una constante sin padres. Esto es literalmente "romper el link causal con los padres". Gracias a esto, si después se cambia un ancestro, el valor intervenido persiste (en cambio `_set` no toca el mecanismo).
- Recalcula todo lo que está aguas abajo.

Detalle de implementación: `self.children` no se actualiza, así que `I1` sigue listando a `E1` como hijo; pero si `I1` cambiara, se recalcularía `E1` con el mecanismo constante, que devuelve `False`. El efecto es el correcto.

También existe `trace["E1"] = False` (`__setitem__`), que llama a `intervene`: asignar después de la inicialización tiene semántica de intervención.

Salida guardada: antes `E1=True, E2=True, O=True`; después de `do(E1=False)`: `E1=False, E2=True, O=False`. Aunque `I1 = I2 = A`, `E1` quedó en `False` y `O` se recalculó con ese valor. Esto es $\mathcal{A}_{E_1 \leftarrow \text{False}}(AABB)$.

#### Celda 28: interchange intervention en el algoritmo

```python
base = equality_model.new_trace({"I1": "A", "I2": "A", "I3": "B", "I4": "B"})
source = equality_model.new_trace({"I1": "A", "I2": "B", "I3": "C", "I4": "C"})
result = equality_model.run_interchange(base, {"E1": source})
```

- `base`: AABB (E1=True, E2=True, O=True).
- `source`: ABCC (E1=False, E2=True, O=False).
- `run_interchange(input_trace, counterfactual_inputs)` (`causal_model.py`, línea 192):
  1. `trace = input_trace.copy()` (copia el base: `CausalTrace.copy` hace deepcopy de los mecanismos y copia superficial de los valores).
  2. Para cada clave `var` del diccionario: si la clave tiene la forma `"A<-B"`, interviene `A` en el base con el valor de `B` del trace contrafactual; si no, interviene `var` con `cf_trace[var]`.
  3. Devuelve el trace intervenido.

  O sea, `run_interchange(base, {"E1": source})` es exactamente `base.copy()` + `intervene("E1", source["E1"])`. El docstring lo marca como *deprecated* salvo para la sintaxis `"A<-B"` (tomar el valor de una variable con otro nombre en el source), y sugiere usar `copy` + asignación directa.

Salida guardada: `E1 = False (from source)`, `E2 = True (from base)`, `O = False (recomputed)`. Esto es $\mathcal{A}(AABB)_{E_1 \leftarrow \mathcal{A}(ABCC)}$. La celda 29 lo dibuja con `print_setting`.

Lo que hay que llevarse: una interchange intervention es una hard intervention cuyo valor no se elige a mano, sino que viene de correr el modelo sobre otro input. Por eso el valor intervenido siempre es un valor "realista" que la variable puede tomar.

### 2. Redes neuronales como modelos causales

La idea central de esta sección: la red neuronal **también es un `CausalModel`**. Cada neurona es una variable, cada peso distinto de cero es una arista, y el mecanismo de cada neurona es "suma ponderada de los padres + ReLU".

Variables del MLP:

- `X1..X8`: neuronas de embedding. Cada letra tiene un embedding de 2 dimensiones; `X1,X2` = embed(letra 1), `X3,X4` = embed(letra 2), etc. Padre: `raw_input`.
- `H1_1..H1_8`: primera capa oculta (ReLU).
- `H2_1..H2_8`: segunda capa oculta.
- `Y1, Y2`: logits. `raw_output = bool(Y2 > Y1)`.

$W_1$ es **diagonal por bloques**: `H1_1..H1_4` dependen solo de `X1..X4` (par 1) y `H1_5..H1_8` solo de `X5..X8` (par 2). Eso crea dos subgrafos desconectados en la capa 1 que reflejan $E_1$ y $E_2$.

#### Celda 32: construir el MLP como modelo causal

```python
embedding_dim = 2
np.random.seed(42)
letter_embeddings = {
    letter: np.round(np.random.uniform(-1, 1, size=embedding_dim), 2)
    for letter in letters
}

def relu(x):
    return max(0.0, x)
```

- Cada letra recibe un vector aleatorio en $[-1,1]^2$, redondeado a 2 decimales. `np.random.seed(42)` de nuevo para que sean reproducibles.
- `relu` escalar, porque cada neurona es una variable escalar separada.

```python
W1 = np.array([
    [ 1,  0, -1,  0,  0,  0,  0,  0],
    [ 0,  1,  0, -1,  0,  0,  0,  0],
    [-1,  0,  1,  0,  0,  0,  0,  0],
    [ 0, -1,  0,  1,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  1,  0, -1,  0],
    ...
], dtype=float)
```

Fila $i$ = pesos de `H1_{i+1}` sobre `X1..X8`. Escribiendo $a = \text{embed}(I_1)$ y $b = \text{embed}(I_2)$:

- `H1_1 = ReLU(a[0] - b[0])`, `H1_3 = ReLU(b[0] - a[0])` → `H1_1 + H1_3 = |a[0] - b[0]|`.
- `H1_2 = ReLU(a[1] - b[1])`, `H1_4 = ReLU(b[1] - a[1])` → `H1_2 + H1_4 = |a[1] - b[1]|`.
- Entonces `H1_1 + ... + H1_4` es la **distancia L1** entre los embeddings de las dos primeras letras. Es 0 exactamente cuando las letras son iguales (y, salvo coincidencias raras de embeddings, distinto de 0 cuando no). Lo mismo con `H1_5..H1_8` para el segundo par.

Esta es la razón por la que la representación de $E_1$ está en las neuronas `H1_1..H1_4`: todas juntas codifican "¿el primer par es igual?" vía "¿son todas cero?".

```python
W2 = np.array([
    [ 1,  1,  1,  1, -1, -1, -1, -1],
    [-1, -1, -1, -1,  1,  1,  1,  1],
    [ 0,  0,  0,  0,  1,  1,  1,  1],
    [ 1,  1,  1,  1,  0,  0,  0,  0],
    [0]*8, [0]*8, [0]*8, [0]*8,
], dtype=float)
```

Llamando $d_1$ = distancia L1 del par 1 y $d_2$ = la del par 2:

- `H2_1 = ReLU(d1 - d2)`, `H2_2 = ReLU(d2 - d1)` → `H2_1 + H2_2 = |d1 - d2|`.
- `H2_3 = ReLU(d2) = d2`, `H2_4 = ReLU(d1) = d1`.
- `H2_5..H2_8`: fila de ceros, constantes 0.

```python
W3 = np.array([
    [1, 1, -0.999999, -0.999999, 0, 0, 0, 0],
    [0, 0,  0,         0,        0, 0, 0, 0],
], dtype=float)
b3 = np.array([0.0, 1e-14])
```

- `Y1 = |d1 - d2| - 0.999999 (d1 + d2)`.
- `Y2 = 1e-14` (constante).
- Predicción `True` sii `Y2 > Y1`, o sea sii `Y1 < 1e-14`. Casos:
  - ambos pares iguales ($d_1 = d_2 = 0$): `Y1 = 0 < 1e-14` → True.
  - un par igual y otro no (digamos $d_1 = 0 < d_2$): `Y1 = d2 - 0.999999 d2 = 10^{-6} d_2 > 0` → False (para $d_2$ no minúsculo).
  - ambos distintos ($d_1, d_2 > 0$): $|d_1 - d_2| < d_1 + d_2$, y con el factor 0.999999 queda `Y1 < 0` salvo que un $d$ sea muchísimo más chico que el otro → True.

  El `0.999999` en vez de 1 y el sesgo `1e-14` son trucos para que los empates caigan del lado correcto. Esto lo deduzco del código; la notebook no lo explica.

```python
nn_mechanisms = {}
nn_values = {
    "raw_input": [None],
    **{f"X{i+1}": [None] for i in range(8)},
    **{f"H1_{i+1}": [None] for i in range(8)},
    **{f"H2_{i+1}": [None] for i in range(8)},
    "Y1": [None], "Y2": [None],
    "raw_output": [True, False],
}
nn_mechanisms["raw_input"] = input_var([None])
```

- Los valores de las neuronas son `[None]` porque son continuos y no se van a muestrear; `CausalModel` solo exige que la clave exista. No hay validación de que los valores del trace pertenezcan a esa lista.
- Acá `raw_input` **sí** es el input (mecanismo sin padres), a diferencia del algoritmo donde era derivado de `I1..I4`.

```python
for i in range(8):
    letter_idx, dim_idx = i // 2, i % 2
    nn_mechanisms[f"X{i+1}"] = Mechanism(
        parents=["raw_input"],
        compute=lambda t, li=letter_idx, di=dim_idx:
            float(letter_embeddings[t["raw_input"][li]][di]),
    )
```

- `X{i+1}` toma la letra número `i // 2` del `raw_input` y la dimensión `i % 2` de su embedding. Así `X1, X2` = embed(letra 0), `X3, X4` = embed(letra 1), etc.
- Ojo con `li=letter_idx, di=dim_idx` como argumentos por defecto del lambda: es el truco estándar para "congelar" el valor del loop. Sin eso, todos los lambdas leerían el `i` final por el late binding de Python. Se repite en todo el bloque.

```python
for i in range(8):
    row = W1[i]
    parent_idx = [j for j in range(8) if row[j] != 0]
    parents = [f"X{j+1}" for j in parent_idx]
    nn_mechanisms[f"H1_{i+1}"] = Mechanism(
        parents=parents,
        compute=lambda t, r=row, pi=parent_idx:
            relu(sum(r[j] * t[f"X{j+1}"] for j in pi)),
    )
```

- Los padres de `H1_{i+1}` son solo las `X` con peso no nulo en la fila `i`. Esto es lo que convierte "pesos no nulos" en "aristas del grafo causal", y lo que hace visible la estructura diagonal por bloques en `print_structure`.
- El mecanismo es `ReLU(fila · padres)`.

```python
for i in range(8):
    row = W2[i]
    parent_idx = [j for j in range(8) if row[j] != 0]
    parents = [f"H1_{j+1}" for j in parent_idx]
    if len(parents) == 0:
        nn_mechanisms[f"H2_{i+1}"] = Mechanism(parents=["H1_1"], compute=lambda t: 0.0)
    else:
        ...relu(sum(r[j] * t[f"H1_{j+1}"] for j in pi))
```

- Igual que antes con `W2`.
- Caso especial: `H2_5..H2_8` tienen fila nula, no tienen padres reales. Si se les pusiera `parents=[]`, `CausalModel` los trataría como **inputs**, y `CausalTrace` nunca los calcularía (solo se calculan los hijos de variables que se setean), así que leerlos daría `KeyError`. Por eso se les pone un **padre trivial** `H1_1` que no se usa en `compute`: así entran en la cascada de cómputo. El comentario del código lo dice: "trivial parent keeps them in the computation cascade".

```python
for i, name in enumerate(["Y1", "Y2"]):
    ...
    if len(parents) == 0:
        nn_mechanisms[name] = Mechanism(parents=["H2_1"], compute=lambda t, b=bias: b)
    else:
        ...sum(r[j] * t[f"H2_{j+1}"] for j in pi) + b
nn_mechanisms["raw_output"] = Mechanism(parents=["Y1", "Y2"], compute=lambda t: bool(t["Y2"] > t["Y1"]))
```

- `Y1` y `Y2` son lineales (sin ReLU) más bias. `Y2` tiene fila nula, así que se le da el padre trivial `H2_1` y devuelve la constante `1e-14`.
- `raw_output = Y2 > Y1`.

```python
nn_model = CausalModel(nn_mechanisms, nn_values, print_pos=nn_pos, id="mlp")

for name in ["AABB", "ABCC", "ABCD", "AABC"]:
    t = nn_model.new_trace({"raw_input": tuple(name)})
    ...
correct = total = 0
for i1 in "ABCD": ... for i4 in "ABCD":
    t = nn_model.new_trace({"raw_input": (i1, i2, i3, i4)})
    correct += (t["raw_output"] == ((i1 == i2) == (i3 == i4)))
    total += 1
print_structure(nn_model, font=8, node_size=400)
```

- `new_trace({"raw_input": tuple(name)})`: el input de la red es la tupla de 4 letras; se propaga X → H1 → H2 → Y → raw_output.
- Imprime la predicción y los valores de H1 de cada mitad.
- El chequeo exhaustivo recorre las $4^4 = 256$ combinaciones de {A,B,C,D} y compara con la definición de la tarea.

Salidas guardadas:

```
AABB: prediction=True   H1_1-H1_4 = [0,0,0,0]        H1_5-H1_8 = [0,0,0,0]
ABCC: prediction=False  H1_1-H1_4 = [0,0.7,0.71,0]   H1_5-H1_8 = [0,0,0,0]
ABCD: prediction=True   H1_1-H1_4 = [0,0.7,0.71,0]   H1_5-H1_8 = [0.19,0,0,1.42]
AABC: prediction=False  H1_1-H1_4 = [0,0,0,0]        H1_5-H1_8 = [1.15,0.89,0,0]
Accuracy on {A,B,C,D}^4 (256 inputs): 256/256
```

Se ve el patrón: par igual → las 4 neuronas en cero; par distinto → alguna positiva. (Los `np.float64(...)` en el print son solo cómo numpy 2 representa sus escalares dentro de una lista.)

#### Celda 35: hard intervention en la red

$$\mathcal{N}(i)_{H_{1,1},\dots,H_{1,4} \leftarrow v_1,\dots,v_4}$$

```python
base = nn_model.new_trace({"raw_input": ("A", "A", "B", "B")})
source = nn_model.new_trace({"raw_input": ("A", "B", "C", "C")})

result = base.copy()
for v in ["H1_1", "H1_2", "H1_3", "H1_4"]:
    result.intervene(v, source[v])
```

- `base.copy()` para no tocar el trace original.
- Interviene las cuatro neuronas una por una con los valores que tienen en el source. Cada `intervene` reemplaza el mecanismo por constante y recalcula aguas abajo (H2, Y, raw_output). Intervenir de a una funciona porque las cuatro no dependen entre sí.

Salida guardada: el base daba `True`; tras la intervención la predicción es `False`, con `H1_1..H1_4 = [0, 0.7, 0.71, 0]` (del source) y `H1_5..H1_8 = [0,0,0,0]` (del base). Mismo efecto que $\mathcal{A}(AABB)_{E_1 \leftarrow \text{False}}$.

#### Celda 38: interchange intervention en la red

$$\mathcal{N}(i_{\text{orig}})_{H_{1,1},\dots,H_{1,4} \leftarrow \mathcal{N}(i_{\text{counter}})}$$

```python
nn_result = nn_model.run_interchange(base, {
    "H1_1": source, "H1_2": source, "H1_3": source, "H1_4": source,
})
...
algo_result = equality_model.run_interchange(algo_base, {"E1": algo_source})
```

- Lo mismo que la celda anterior pero con `run_interchange`: cada neurona del diccionario se interviene con su valor en el trace `source`. Las cuatro claves apuntan al mismo source porque juntas representan una sola variable de alto nivel.
- Después corre la interchange equivalente en el algoritmo (sobre `E1`).

Salida guardada: red → `prediction = False`; algoritmo → `O = False`. Ambos invierten el output de True a False: intervenir 4 neuronas a la vez "imita" intervenir $E_1$. Este es el experimento de Geiger et al. (2021): alinear representaciones neuronales con variables de un modelo interpretable y verificar con interchange interventions que tienen el mismo rol causal.

### 3. Implementación vía abstracción causal

#### Celda 41: alineación y mapas de traducción

```python
alignment = {
    "E1": ["H1_1", "H1_2", "H1_3", "H1_4"],
    "E2": ["H1_5", "H1_6", "H1_7", "H1_8"],
    "O":  ["Y1", "Y2"],
}
```

Esto es $\Pi$: cada variable de alto nivel → conjunto de neuronas. La alineación apunta a la primera capa oculta, donde la estructura de $W_1$ separa limpiamente los pares. (Los inputs $I_k$ implícitamente se alinean con los pares de neuronas X correspondientes, por cómo está escrito `pi` abajo, aunque no aparecen en `alignment`.)

```python
def closest_letter(embedding):
    return min(letter_embeddings, key=lambda l: np.sum(np.abs(letter_embeddings[l] - embedding)))
```

Búsqueda inversa del embedding: devuelve la letra cuyo embedding está más cerca en distancia L1 del vector dado. `min` sobre un diccionario itera sus claves (las letras).

```python
pi = {
    "I1": lambda t: closest_letter(np.array([t["X1"], t["X2"]])),
    ...
    "E1": lambda t: bool(t["H1_1"] + t["H1_2"] + t["H1_3"] + t["H1_4"] == 0),
    "E2": lambda t: bool(t["H1_5"] + t["H1_6"] + t["H1_7"] + t["H1_8"] == 0),
    "O":  lambda t: bool(t["Y2"] > t["Y1"]),
}
```

Los $\pi_V$: cada uno recibe un trace de la red y devuelve el valor de alto nivel.

- $\pi_{I_k}$: arma el vector de 2 dimensiones desde las X correspondientes y lo decodifica a letra.
- $\pi_{E_1}$: `True` sii las cuatro neuronas suman exactamente 0. Funciona con `== 0` exacto porque ReLU devuelve exactamente `0.0`. Como son no negativas, suma cero ⇔ todas cero.
- $\pi_O$: compara logits.

```python
def tau(nn_trace):
    return {var: fn(nn_trace) for var, fn in pi.items()}
```

$\tau_\pi$: aplica cada $\pi_V$ y arma un "trace del algoritmo" como diccionario.

```python
def omega(nn_vars, source_nn_trace):
    for algo_var, aligned_neurons in alignment.items():
        if sorted(nn_vars) == sorted(aligned_neurons):
            return {algo_var: pi[algo_var](source_nn_trace)}
    return {}
```

$\omega_\pi$ restringido a interchange interventions: si el conjunto de neuronas intervenidas coincide exactamente con algún $\Pi_V$, la intervención de alto nivel es "$V \leftarrow \pi_V(\text{trace del source})$". Si no coincide con ninguno (p. ej. una mezcla de neuronas de dos variables), devuelve `{}`: esa intervención **no tiene traducción** en esta implementación. Esto es importante para entender por qué una alineación incorrecta falla.

```python
def rho(algo_input):
    return {"raw_input": (algo_input["I1"], algo_input["I2"], algo_input["I3"], algo_input["I4"])}
```

$\rho_\pi$: empaqueta las 4 letras en la tupla que la red espera.

#### Celda 43: verificar los mapas

```python
algo_input = {"I1": "A", "I2": "B", "I3": "C", "I4": "C"}
nn_trace = nn_model.new_trace(rho(algo_input))
translated = tau(nn_trace)
algo = equality_model.new_trace(algo_input)
for var in [...]:
    ...match...
source_nn = nn_model.new_trace(rho({...ABCC...}))
algo_intervention = omega(["H1_1", "H1_2", "H1_3", "H1_4"], source_nn)
```

- Chequea $\tau_\pi(\mathcal{N}(\rho_\pi(\text{ABCC}))) = \mathcal{A}(\text{ABCC})$ variable por variable. Es la ecuación central con la intervención vacía.
- Chequea que $\omega$ mande `H1_1..H1_4` a `E1`.

Salida guardada: las 7 variables dan `match`; `omega(...) = {'E1': False}`; `rho(...) = {'raw_input': ('A', 'B', 'C', 'C')}`.

#### Celda 46: IIA con alineación correcta e incorrecta

```python
import itertools

test_inputs = list(itertools.product("ABCD", repeat=4))
random.seed(42)
pairs = random.sample(list(itertools.product(test_inputs, test_inputs)), 500)
```

- `test_inputs`: las 256 tuplas de {A,B,C,D}^4.
- `itertools.product(test_inputs, test_inputs)`: los $256^2 = 65536$ pares (base, source). Se muestrean 500 sin reemplazo. Este es $\mathcal{D}$.

```python
def compute_iia(pairs, nn_vars, algo_var):
    correct = 0
    for base_letters, source_letters in pairs:
        algo_base_input = dict(zip(["I1","I2","I3","I4"], base_letters))
        algo_source_input = dict(zip(["I1","I2","I3","I4"], source_letters))

        nn_base = nn_model.new_trace(rho(algo_base_input))
        nn_source = nn_model.new_trace(rho(algo_source_input))
        nn_result = nn_model.run_interchange(nn_base, {v: nn_source for v in nn_vars})

        algo_base = equality_model.new_trace(algo_base_input)
        algo_source = equality_model.new_trace(algo_source_input)
        algo_result = equality_model.run_interchange(algo_base, {algo_var: algo_source})

        if pi["O"](nn_result) == algo_result["O"]:
            correct += 1
    return correct / len(pairs)
```

Es la fórmula de IIA literal:

- `dict(zip(...))`: convierte la tupla de letras en el input del algoritmo.
- `nn_result`: $\mathcal{N}(b)_{\Pi_V \leftarrow \mathcal{N}(s)}$, con $\Pi_V$ = `nn_vars` (las neuronas que se están probando).
- `algo_result`: $\mathcal{A}(\rho(b))_{V \leftarrow \mathcal{A}(\rho(s))}$.
- Compara $\pi_O$ del resultado de la red con `O` del algoritmo, y promedia.

Notar que esta función no usa `omega`: decide "a mano" que las neuronas `nn_vars` corresponden a `algo_var`. Eso permite probar alineaciones incorrectas, que es justamente el punto.

```python
compute_iia(pairs, ['H1_1','H1_2','H1_3','H1_4'], 'E1')
compute_iia(pairs, ['H1_5','H1_6','H1_7','H1_8'], 'E2')
compute_iia(pairs, ['H1_1','H1_2','H1_7','H1_4'], 'E1')   # incorrecta
compute_iia(pairs, ['H1_5','H1_6','H1_3','H1_8'], 'E2')   # incorrecta
```

Salidas guardadas: correctas → 1.000 y 1.000. Incorrectas → 0.850 y 0.826.

Por qué la incorrecta falla: en `['H1_1','H1_2','H1_7','H1_4']` se deja `H1_3` del base y se mete `H1_7`, que pertenece al segundo par. El resultado es un estado "mezclado": la distancia L1 del par 1 queda como una combinación de neuronas del base y del source, y además se perturba el par 2. En algunos pares (base, source) esa mezcla cambia el output de forma distinta a lo que predice "$E_1 \leftarrow$ valor del source".

Por qué no da 0.5 sino ~0.85: en muchos pares la interchange no cambia nada relevante (por ejemplo, base y source tienen el mismo valor de $E_1$ y la mezcla no rompe el cero, o la neurona intercambiada vale cero en ambos), y entonces red y algoritmo coinciden "de casualidad". Esto es una interpretación mía; la notebook no lo desglosa. Es justamente lo que motiva la Lecture 9: la evidencia que da una IIA depende de qué pares hay en el dataset.

#### Celda 48: más allá de IIA

Solo texto: KL-IIA y abstracción aproximada (ver Ideas clave).

---

## Ejercicios

Esta notebook no tiene ejercicios ni celdas `FILL IN YOUR ANSWER HERE` ni soluciones marcadas `# TERRON`. Todo el código ya viene completo. Si quiero practicar, cosas que se pueden probar sin cambiar la teoría:

- Correr `compute_iia` con `nn_vars = ['H1_1', 'H1_2', 'H1_3', 'H1_4']` pero `algo_var = 'E2'` y razonar por qué falla.
- Probar alineaciones con un subconjunto, por ejemplo solo `['H1_1', 'H1_3']` para `E1`, y pensar qué parte de la distancia L1 se está intercambiando.
- Probar alinear `E1` con neuronas de la segunda capa (`H2_*`) y ver si existe algún conjunto que dé IIA 1.

---

## Ojo / errores comunes / dudas

- **`outputs` de `CausalModel` no es "la salida"**: son las hojas del grafo. Por eso `raw_input` aparece en `Outputs` del algoritmo y en el último timestep.
- **Variables sin padres = inputs**: una variable con mecanismo `parents=[]` nunca se calcula sola. Por eso las neuronas constantes (`H2_5..H2_8`, `Y2`) tienen un padre trivial. Si un día defino un modelo y obtengo `KeyError: Variable ... has not been computed yet`, probablemente es esto.
- **`input_var` muestrea con `random.choice`**, pero en estas celdas nunca se ejecuta su `compute`, porque los inputs se pasan explícitos a `new_trace`.
- **`compute` recibe el trace entero**: nada impide leer variables que no están en `parents`, pero eso rompería el grafo (las intervenciones no se propagarían bien). La lista `parents` tiene que ser honesta.
- **`print_setting` dibuja la estructura del modelo, no la del trace intervenido**.
- **Lambdas en loops**: siempre con argumentos por defecto (`r=row, pi=parent_idx`) para congelar valores.
- **IIA compara solo el output**, no el trace completo; es una versión débil de la ecuación de abstracción. IIA = 1 sobre 500 pares no es prueba sobre todos los pares, aunque acá la red está construida a mano para que funcione.
- **Una alineación es a conjuntos de neuronas**: la gracia del ejemplo es que $E_1$ no vive en una neurona sino en cuatro a la vez. Pero siguen siendo neuronas del eje estándar. Cuando la variable vive en una dirección "rotada" que mezcla neuronas, esta abstracción constructiva no alcanza: eso motiva la Lecture 10 (abstracción bajo traducción y DAS).
- **$\omega$ solo traduce conjuntos exactos**: si intervengo un conjunto de neuronas que no coincide con ningún $\Pi_V$, no hay intervención de alto nivel correspondiente. La ecuación de abstracción no dice nada directamente de esas intervenciones.
- **Detalles de los pesos** (0.999999, 1e-14): son para que los casos borde caigan del lado correcto; mi análisis de por qué está arriba, la notebook no lo comenta.
- No usa `CounterfactualExample` ni `label_counterfactual_data`, aunque existen en `causal_model.py` / `counterfactual_dataset.py`. `CounterfactualExample` es solo un `TypedDict` con `input` (trace base) y `counterfactual_inputs` (lista de traces source); `label_counterfactual_data` corre `run_interchange` para cada ejemplo y devuelve el `raw_output` como `label`. Es la versión "en lote" de lo que acá se hace a mano dentro de `compute_iia`.

---

## Resumen de una línea por sección

- **Setup**: instala dependencias, importa `CausalModel`, `Mechanism`, `input_var`, `print_structure`, `print_setting` y fija semillas.
- **1. Algoritmos como modelos causales**: el algoritmo de igualdad jerárquica es un DAG `I → E → O` con mecanismos; un trace guarda todos los valores, y `intervene` reemplaza un mecanismo por una constante.
- **Interchange en el algoritmo**: `run_interchange(base, {"E1": source})` = copiar el base e intervenir `E1` con su valor en el source.
- **2. Redes como modelos causales**: un MLP a mano donde cada neurona es una variable, cada peso no nulo una arista, y $W_1$ diagonal por bloques separa los dos pares.
- **Intervenciones en la red**: intercambiar `H1_1..H1_4` desde ABCC a AABB invierte la predicción igual que intercambiar $E_1$ en el algoritmo.
- **3. Abstracción constructiva**: alineación $(\Pi, \pi)$ y mapas $\tau, \omega, \rho$; la abstracción exige que "intervenir y traducir" = "traducir e intervenir".
- **IIA**: con la alineación correcta da 1.0 y con una incorrecta ~0.83-0.85 (ejecución guardada).
- **Más allá de IIA**: KL-IIA para outputs probabilísticos y abstracción aproximada (Beckers et al.) para cuando no hay acuerdo perfecto.
