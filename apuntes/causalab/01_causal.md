# causalab/causal: modelos causales, trazas y datasets contrafactuales

Este submódulo es la mitad "de alto nivel" de la librería: define qué es un modelo causal, cómo se ejecuta (una **traza**), cómo se interviene, y cómo se arman, guardan y etiquetan datasets contrafactuales. No toca redes neuronales. Responde a la pregunta: dado mi algoritmo hipotético, ¿qué salida debería dar si fuerzo la variable X al valor que tiene en otro input?

Archivos (todos en `causalab/causalab/causal/`):

- `trace.py`: `Mechanism`, `input_var`, `CausalTrace`.
- `causal_model.py`: `CausalModel`.
- `counterfactual_dataset.py`: tipos `CounterfactualExample` y `LabeledCounterfactualExample`.
- `causal_utils.py`: utilidades (generar/guardar/cargar datasets, scoring post-hoc, caminos causales vivos, samplers).
- `causal_viz.py`: visualización del grafo.
- `__init__.py`: solo un docstring, no reexporta nada (hay que importar desde los submódulos).

## Ideas clave / teoría

- Un **modelo causal** es un DAG de variables. Las variables sin padres son **inputs** (exógenas); las que no tienen hijos son **outputs**. Cada variable no-input tiene un **mecanismo** `f(valores de los padres) -> valor`.
- Una **ejecución** (forward) asigna valores a los inputs y propaga hacia abajo. En causalab, el estado de una ejecución es un `CausalTrace`.
- Una **intervención** `do(X = v)` reemplaza el mecanismo de X por la constante v (corta la flecha desde los padres) y recalcula los descendientes.
- Una **interchange intervention** en el modelo causal es `do(X = valor de X en el contrafactual)` aplicado a la ejecución base. El `raw_output` resultante es el **label** contra el que se compara al LM.

## `trace.py`

### `Mechanism`

```python
@dataclass
class Mechanism:
    parents: list[str]
    compute: Callable[["CausalTrace"], Any]

    def __call__(self, trace: "CausalTrace") -> Any:
        return self.compute(trace)
```

- `parents`: nombres de las variables padre.
- `compute`: función que **recibe la traza entera** (no los valores de los padres sueltos) y devuelve el valor. Adentro uno lee `t["padre"]`. Esto es importante: el mecanismo puede leer cualquier variable de la traza, pero solo se recalcula cuando cambia alguno de los `parents` declarados.
- `__call__` permite hacer `mechanism(trace)`.

### `input_var(values)`

```python
def input_var(values: list[Any]) -> Mechanism:
    return Mechanism(parents=[], compute=lambda t: random.choice(values))
```

Atajo para declarar un input: sin padres, con un `compute` que muestrea al azar. Ojo: en la práctica ese `compute` casi nunca se ejecuta, porque `CausalTrace` solo recalcula **hijos** de variables que cambian; los inputs se fijan a mano vía el dict `inputs`. El muestreo de inputs lo hace `CausalModel.sample_input` usando `values`, no este lambda.

### `CausalTrace`

Es el estado de una ejecución: un diccionario de valores más los mecanismos, con recálculo automático de descendientes.

```python
def __init__(self, mechanisms, inputs=None):
    if inputs is None:
        inputs = {}
    self.mechanisms = mechanisms
    self.children = {var: [] for var in mechanisms}
    for var, mechanism in mechanisms.items():
        for parent in mechanism.parents:
            self.children[parent].append(var)
    self._values = {}
    for var, val in inputs.items():
        self._set(var, val)
```

- Guarda los mecanismos (ojo: `CausalModel.new_trace` le pasa un **deepcopy**, así una intervención en una traza no contamina al modelo).
- Construye el mapa `children` a partir de los `parents` de cada mecanismo. Si un mecanismo declara un padre que no es una variable, esto lanza `KeyError`.
- Arranca vacío y va seteando cada input con `_set`, que dispara el recálculo de descendientes.

```python
def _set(self, variable, value):
    self._values[variable] = value
    self._recompute_descendants(variable)
    return self

def _recompute_descendants(self, variable):
    for child in self.children[variable]:
        mechanism = self.mechanisms[child]
        if all(p in self._values for p in mechanism.parents):
            self._values[child] = mechanism(self)
            self._recompute_descendants(child)
```

Este es el corazón del forward:

- Por cada hijo de la variable que cambió, **solo si todos sus padres ya tienen valor**, calcula el hijo con su mecanismo y recursivamente sus descendientes.
- Consecuencia: mientras voy seteando inputs uno por uno, las variables intermedias aparecen recién cuando el último padre se completa. Si no paso algún input, todo lo que dependa de él queda **sin calcular** (y `trace["X"]` lanza `KeyError("... has not been computed yet ...")`).
- Es una recursión en profundidad, no un orden topológico: si una variable tiene varios caminos desde la que cambió, puede recalcularse más de una vez. No es incorrecto (el último cálculo usa los valores finales de los padres... siempre que la recursión llegue en un orden razonable), pero es redundante.

Acceso tipo diccionario:

- `get(var)` / `trace[var]`: devuelve el valor o `KeyError`.
- `trace[var] = v` (`__setitem__`): **llama a `intervene`**, o sea, asignar después de la construcción tiene semántica de intervención (corta el link con los padres).
- `var in trace`, `del trace[var]` (borra el valor cacheado, no el mecanismo).

```python
def copy(self):
    new_trace = CausalTrace(copy_module.deepcopy(self.mechanisms))
    new_trace._values = copy_module.copy(self._values)
    return new_trace
```

Copia con mecanismos deep-copiados y valores shallow-copiados. Se usa antes de intervenir para no modificar la traza base.

```python
def intervene(self, variable, value):
    self._values[variable] = value
    self.mechanisms[variable] = Mechanism(parents=[], compute=lambda t, v=value: v)
    self._recompute_descendants(variable)
    return self
```

- Fija el valor.
- **Reemplaza el mecanismo** por una constante sin padres (`v=value` como default del lambda para capturar el valor actual y no la variable). Esto es el `do()` de verdad: si después cambio un ancestro, esta variable no se recalcula (ya no es hija de nadie... aunque ojo, `self.children` no se actualiza, así que el padre todavía la lista como hija; lo que evita el recálculo es que el nuevo mecanismo tiene `parents=[]`, por lo que `all(...)` es trivialmente verdadero y se recalcula a la constante, que da el mismo valor).
- Recalcula los descendientes.

`to_dict()` devuelve `self.copy()._values`: un dict plano con todos los valores (es lo que se serializa a JSON).

## `causal_model.py`: `CausalModel`

### Constructor

```python
def __init__(self, mechanisms, values, print_pos=None, id="null"):
    self.mechanisms = mechanisms
    self.values = values
    self.id = id
    self.variables = list(self.mechanisms.keys())
    assert "raw_input" in self.variables
    assert "raw_output" in self.variables
```

Argumentos:

- `mechanisms: dict[str, Mechanism]`: la definición del grafo. Las variables salen de las keys.
- `values: dict[str, Any]`: valores posibles. Para inputs es una lista (de ahí muestrea `sample_input`); para variables calculadas puede ser `None` (por ejemplo `raw_input: None`).
- `print_pos`: posiciones (x, y) para dibujar con networkx.
- `id`: identificador.

Exige que existan `raw_input` y `raw_output`.

```python
    self.parents = {var: mechanism.parents for var, mechanism in self.mechanisms.items()}
    self.children = {var: [] for var in self.variables}
    for variable in self.variables:
        for parent in self.parents[variable]:
            self.children[parent].append(variable)

    self.inputs = [var for var in self.variables if len(self.parents[var]) == 0]
    self.outputs = copy.deepcopy(self.variables)
    for child in self.variables:
        for parent in self.parents[child]:
            if parent in self.outputs:
                self.outputs.remove(parent)
```

- `parents` y `children` derivados de los mecanismos.
- `inputs`: variables sin padres.
- `outputs`: arranca con todas las variables y saca toda variable que sea padre de alguien; quedan las hojas.

```python
    self.timesteps = {input_var: 0 for input_var in self.inputs}
    step = 1
    change = True
    while change:
        change = False
        copytimesteps = copy.deepcopy(self.timesteps)
        for parent in self.timesteps:
            if self.timesteps[parent] == step - 1:
                for child in self.children[parent]:
                    copytimesteps[child] = step
                    change = True
        self.timesteps = copytimesteps
        step += 1
    self.end_time = step - 2
    for output in self.outputs:
        self.timesteps[output] = self.end_time
```

Calcula un "tiempo" por variable, tipo BFS por niveles: inputs en 0; en cada vuelta, los hijos de las variables del nivel anterior reciben el nivel actual. Como se sobreescribe, una variable alcanzable por un camino largo termina con el nivel del **camino más largo** desde algún input. Termina cuando ya no hay hijos nuevos. `end_time` es el último nivel con variables, y todos los outputs se fuerzan a ese nivel (así quedan alineados abajo en el dibujo). Si el grafo tuviera un ciclo, este loop no terminaría.

Después hay un bloque largo de validación (líneas 103-148): cada variable debe estar en `values`, `children`, `mechanisms`, `timesteps`; y para cada par padre-hijo, la relación tiene que ser consistente en ambos sentidos y el padre tener timestep estrictamente menor. Es la verificación de que el grafo es un DAG bien armado.

Finalmente ordena `self.variables` por timestep, completa `print_pos` (raw_input en (0, -2) y el resto en una grilla (índice dentro del nivel, nivel)) e inicializa `self.equiv_classes = {}` (cache para `generate_equiv_classes`, que no se usa desde el constructor).

### `new_trace(inputs=None) -> CausalTrace`

```python
return CausalTrace(mechanisms=copy.deepcopy(self.mechanisms), inputs=inputs)
```

La forma estándar de correr el modelo: le paso un dict de inputs y obtengo una traza con todo calculado. Deep-copia los mecanismos para que intervenir esta traza no toque el modelo.

### `run_interchange(input_trace, counterfactual_inputs) -> CausalTrace`

```python
trace = input_trace.copy()
for var in counterfactual_inputs:
    if "<-" in var:
        original_var, counterfactual_var = var.split("<-")
        original_var = original_var.strip()
        counterfactual_var = counterfactual_var.strip()
        cf_trace = counterfactual_inputs[var]
        trace.intervene(original_var, cf_trace[counterfactual_var])
    else:
        cf_trace = counterfactual_inputs[var]
        trace.intervene(var, cf_trace[var])
return trace
```

- Copia la traza base.
- Para cada entrada del dict `{variable: traza_contrafactual}`:
  - Caso normal: `do(var = cf[var])`.
  - Caso con sintaxis **`"A<-B"`**: `do(A = cf[B])`, o sea, le pongo a la variable A de la base el valor que tiene la variable B en el contrafactual. Esto permite intervenciones "cruzadas" (se usa en entity_binding: `"positional_entity_g0_e1<-positional_entity_g1_e1"`).
- Está marcado como deprecated para el caso simple (el docstring sugiere `result = trace.copy(); result["A"] = cf["A"]`), pero sigue siendo el que usa todo el resto del código.

Ojo con el orden: las intervenciones se aplican en secuencia. Si intervengo X y después Y, y Y es descendiente de X, Y queda fijo (bien). Si intervengo Y primero y después X, como el mecanismo de Y ya es constante, sigue fijo (también bien). Es correcto en ambos órdenes gracias a que `intervene` reemplaza el mecanismo.

### `sample_input(filter_func=None) -> CausalTrace`

```python
inputs = {var: random.choice(self.values[var]) for var in self.inputs}
trace = self.new_trace(inputs)
while not filter_func(trace):
    ...  # vuelve a muestrear
return trace
```

Rejection sampling: muestrea cada input uniformemente de su lista de `values`, corre el modelo, y repite hasta que `filter_func(trace)` sea verdadero. Ojo: si un mecanismo lanza excepción con algún input (MCQA lo hace cuando el color no está entre las opciones), `sample_input` explota en vez de reintentar; por eso MCQA usa su propio sampler.

### `label_counterfactual_data(examples, target_variables) -> list[dict]`

La función que genera los **labels** para entrenar y evaluar.

```python
for example in examples:
    trace = example["input"]
    counterfactual_traces = example["counterfactual_inputs"]
    if len(counterfactual_traces) == 1 and len(target_variables) > 1:
        counterfactual_traces = counterfactual_traces * len(target_variables)
    assert len(target_variables) <= len(counterfactual_traces)

    counterfactual_dict = {}
    for i, var_element in enumerate(target_variables):
        cf_trace = counterfactual_traces[i]
        if isinstance(var_element, list):
            for var in var_element:
                counterfactual_dict[var] = cf_trace
        else:
            counterfactual_dict[var_element] = cf_trace

    setting = self.run_interchange(trace, counterfactual_dict)
    labels.append(setting["raw_output"])
    settings.append(setting)
```

La semántica de `target_variables` es posicional respecto a los contrafactuales:

- El elemento `i` de `target_variables` se interviene con `counterfactual_inputs[i]`.
- Un elemento puede ser un string (una variable) o una lista de strings (varias variables que toman su valor **del mismo** contrafactual).
- Caso especial: si hay un solo contrafactual y varias variables objetivo, se replica el contrafactual, así todas las variables vienen de él. Este es el caso más común: `target_variables=["answer"]` o `["answer", "answer_position"]` con un contrafactual.

Devuelve una lista de dicts: cada ejemplo original más `"label"` (el `raw_output` de la traza intervenida) y `"setting"` (dict con todos los valores de la traza intervenida).

### `can_distinguish_with_dataset(examples, target_variables1, target_variables2, prints=True)`

Cuenta en qué proporción de ejemplos intervenir `target_variables1` produce un `raw_output` distinto que intervenir `target_variables2` (o distinto del output sin intervenir, si `target_variables2` es `None`). Exige exactamente un contrafactual por ejemplo. Devuelve `{"proportion", "count"}`. El argumento `prints` no se usa (solo hay `logger.debug`).

Sirve para diseñar datasets: si un dataset no distingue "answer" de "answer_position", un experimento con ese dataset no puede decidir cuál de las dos variables está en un lugar de la red.

## `counterfactual_dataset.py`

Solo tipos:

```python
class CounterfactualExample(TypedDict):
    input: "CausalTrace"
    counterfactual_inputs: list["CausalTrace"]

class LabeledCounterfactualExample(CounterfactualExample):
    label: Any
```

Un dataset es simplemente `list[CounterfactualExample]`. No hay clase `Dataset` propia (antes la librería usaba HF `datasets`; los restos se ven en docstrings que dicen "Dataset containing...").

## `causal_utils.py`

### Generación, display, guardado y carga

```python
def generate_counterfactual_samples(size, sampler, filter=None):
    examples = []
    while len(examples) < size:
        sample = sampler()
        if filter is None or filter(sample):
            examples.append(sample)
    return examples
```

Llama al sampler (una función sin argumentos que devuelve un `CounterfactualExample`, p.ej. `MCQA.counterfactuals.different_symbol`) hasta juntar `size` ejemplos que pasen el filtro.

`display_counterfactual_examples(examples, num_examples=1, verbose=True, name="dataset")` imprime los primeros ejemplos (input y cada contrafactual) y devuelve `{índice: ejemplo}`.

```python
def save_counterfactual_examples(examples, path):
    def serialize_example(ex):
        return {
            "input": ex["input"].to_dict(),
            "counterfactual_inputs": [t.to_dict() for t in ex["counterfactual_inputs"]],
        }
    ...json.dump(serialized, f, indent=2)
```

Guarda a JSON convirtiendo cada traza en dict plano (todas las variables, incluidas las calculadas). Los valores tienen que ser serializables a JSON.

```python
def load_counterfactual_examples(path, causal_model):
    data = json.load(f)
    return deserialize_counterfactual_examples(data, causal_model)

def deserialize_counterfactual_examples(dataset, causal_model):
    for example in dataset:
        input_trace = causal_model.new_trace(input_data) if isinstance(input_data, dict) else input_data
        ...
```

Al cargar, cada dict se vuelve a pasar por `causal_model.new_trace(dict)`. Detalle fino: `new_trace` recibe **todas** las variables (no solo inputs) y las setea con `_set` en orden; las calculadas se recalculan a medida que se completan los padres y pueden ser pisadas por el valor guardado en el JSON (que es el mismo, si el modelo no cambió). Si los mecanismos cambiaron entre guardar y cargar, el resultado depende del orden de las keys; no es algo que el código controle.

### `compute_interchange_scores(raw_results, causal_model, datasets, target_variables_list, checker)`

Scoring post-hoc de resultados de intervenciones con el formato viejo (`results["dataset"][name]["model_unit"][unit_str]`). Para cada dataset y unit: aplana los `raw_outputs` por batch a un dict por ejemplo (`sequences`, `string`, y `scores` en formato top-k si existen), etiqueta con `label_counterfactual_data` para cada grupo de variables, aplica `checker(output_dict, label)` y guarda `{"scores": [...], "average_score": mean}` bajo la key `"-".join(target_variables)`. El docstring menciona `experiment.perform_interventions` y `intervention_experiment.py`, que ya no existen; la ruta moderna de scoring es `experiments/metric.py` (ver `03_experiments.md`).

### `can_distinguish_with_dataset(dataset, causal_model1, target_variables1, causal_model2=None, target_variables2=None)`

Versión "función suelta" del método de `CausalModel`, marcada "currently unused". Ojo: aunque recibe `causal_model2`, no lo usa para correr nada; ambas intervenciones se hacen sobre las trazas del dataset (`input_data.copy().intervene(...)`).

### Funciones "extraídas de CausalModel"

- `sample_intervention(model, filter_func=None)`: arma un dict de intervención eligiendo, para cada variable que no es input ni output, con probabilidad 1/2, un valor al azar de `values`. Repite hasta que no esté vacío y pase el filtro.
- `label_data_with_variables(model, data, target_variables)`: para cada ejemplo, concatena los `str` de los valores de las variables objetivo en la traza `input` y le asigna un id entero (un id nuevo por combinación distinta). Devuelve `([{"input": dict, "label": id}], {clave_concatenada: id})`. Útil para clasificación / probing. Ojo: concatena sin separador, así que `("1","23")` y `("12","3")` colisionan.
- `generate_equiv_classes(model)`: para cada variable no-input, enumera todas las combinaciones de valores de los padres y agrupa según el valor resultante. Ojo: llama `model.mechanisms[var](*parent_values)`, pero con la API actual `Mechanism.__call__` recibe una traza, no los valores sueltos; con mecanismos de la API nueva esto falla. Además requiere que `values` de cada padre sea una lista finita.
- `find_live_paths(model, intervention)`: encuentra **caminos vivos**: secuencias de variables donde cambiar la primera cambia efectivamente la siguiente. Algoritmo: arranca con caminos de longitud 1 (cada variable); para extender un camino `[..., v]` con un hijo `c`, prueba cada valor posible de `v` (intervenido sobre una traza nueva con los mismos inputs) y si alguno cambia `c` respecto a la ejecución real, `c` es "causa real" y el camino se extiende. Para cuando no hay caminos nuevos. Devuelve `{longitud: [caminos]}` sin la longitud 1.
- `sample_input_tree_balanced(model, equiv_classes, output_var=None, output_var_value=None)`: muestrea inputs "balanceados por la salida": elige primero el valor del output y baja recursivamente por las clases de equivalencia eligiendo una combinación de padres que produzca ese valor. Completa al azar los inputs que no aparecieron. Depende de `generate_equiv_classes`.
- `get_path_maxlen_filter(model, lengths)`, `get_partial_filter(partial_setting)`, `get_specific_path_filter(model, start, end)`: fábricas de filtros (funciones `setting -> bool`) para usar con samplers: "el camino vivo más largo tiene longitud en `lengths`", "el setting coincide con estos valores", "hay un camino vivo de `start` a `end`".

## `causal_viz.py`

Visualizaciones del grafo; todas reciben el modelo como primer argumento.

- `DEFAULT_COLORS`: colores para input base, valores base, input source, valores source y valores contrafactuales.
- `_get_descendants(model, intervention, strict=True)`: BFS desde las variables intervenidas. Con `strict=True` solo incluye hijos cuyos **todos** padres estén afectados (variables totalmente determinadas por la intervención); con `strict=False`, cualquier hijo con al menos un padre afectado.
- `display_structure(model)`: app Dash con Cytoscape (layout `breadthfirst` desde `raw_input`) que muestra el DAG. Lanza un servidor (`app.run()`).
- `display_forward_pass(model, inputs, intervention=None)`: corre una traza con los inputs, aplica la intervención, y dibuja el grafo con los valores en nodos que se pueden mover; colorea distinto lo puramente intervenido (`strict=True`) y lo "contrafactual" (`strict=False`).
- `display_interchange(model, inputs, counterfactual_inputs)`: dibuja la ejecución base intervenida más una copia del grafo por cada contrafactual (nodos con sufijo `-source-i`), para ver de dónde viene cada valor.
- `print_structure(model, font=12, node_size=1000)`: versión estática con networkx + matplotlib usando `model.print_pos`.
- `print_setting(model, total_setting, ...)`: igual pero con etiquetas `"var:\n valor"` en cada nodo.

Las funciones Dash tienen callbacks internos (`onload`, `on_moving_nodes`) para agregar los nodos de valores y moverlos junto con su variable; es boilerplate de UI.

## Ejemplo mínimo de uso (armado con la API real)

```python
from causalab.causal.causal_model import CausalModel
from causalab.causal.trace import Mechanism, input_var

mechanisms = {
    "x": input_var([0, 1, 2]),
    "y": input_var([0, 1, 2]),
    "s": Mechanism(parents=["x", "y"], compute=lambda t: t["x"] + t["y"]),
    "raw_input": Mechanism(parents=["x", "y"], compute=lambda t: f"{t['x']}+{t['y']}="),
    "raw_output": Mechanism(parents=["s"], compute=lambda t: str(t["s"])),
}
values = {"x": [0, 1, 2], "y": [0, 1, 2], "s": list(range(5)), "raw_input": None, "raw_output": None}
model = CausalModel(mechanisms, values)

base = model.new_trace({"x": 1, "y": 2})   # raw_input "1+2=", raw_output "3"
cf = model.new_trace({"x": 0, "y": 0})
out = model.run_interchange(base, {"x": cf})  # do(x = 0) -> raw_output "2"
```

- `inputs` queda `["x", "y"]`, `outputs` queda `["raw_input", "raw_output"]` (raw_input no tiene hijos, así que técnicamente es una hoja).
- `run_interchange` copia `base` y hace `intervene("x", 0)`; `s` y `raw_output` se recalculan, `raw_input` **también** se recalcula a `"0+2="` porque es hijo de `x`. Esto es normal: en el modelo causal la intervención en `x` también cambia el texto, pero el label que importa es `raw_output`.

## Ojo / errores comunes / dudas

- **Mecanismos reciben la traza, no valores.** `compute=lambda t: t["a"] + t["b"]`. Cualquier código viejo que asuma `f(*parent_values)` (como `generate_equiv_classes`) no es compatible.
- **Inputs faltantes = variables sin calcular.** Si a `new_trace` le falta un input, todo lo que depende de él no existe en la traza. Esto pasa, por ejemplo, en los contrafactuales de IOI (ver `04_tasks.md`).
- **`trace[x] = v` es una intervención**, no una asignación inocente: reemplaza el mecanismo. Si querés "cambiar un input y recalcular", eso también funciona (los inputs no tienen padres, así que el efecto es el mismo).
- **`outputs` incluye `raw_input`** si nadie depende de él (lo normal). No hay que confundir `model.outputs` con "la salida de la tarea", que es `raw_output`.
- **El orden de `target_variables` importa** en `label_counterfactual_data`: el elemento i usa el contrafactual i (salvo el caso de un solo contrafactual, que se replica).
- **Deep copies por todos lados.** `new_trace` y `copy` deep-copian los mecanismos; con modelos grandes (entity_binding tiene muchas variables) esto puede ser lento.

## Resumen de una línea por sección

- **`Mechanism` / `input_var`**: padres + función `traza -> valor`; los inputs son mecanismos sin padres.
- **`CausalTrace`**: dict de valores con recálculo recursivo de descendientes; `intervene` reemplaza el mecanismo por una constante.
- **`CausalModel.__init__`**: deriva padres, hijos, inputs, outputs, timesteps y valida que sea un DAG consistente.
- **`run_interchange`**: copia la base e interviene con valores de contrafactuales, soporta `"A<-B"`.
- **`label_counterfactual_data`**: produce el label (raw_output intervenido) para cada ejemplo; el elemento i de `target_variables` usa el contrafactual i.
- **`can_distinguish_with_dataset`**: mide si un dataset separa dos hipótesis de variables.
- **`causal_utils`**: generar/guardar/cargar datasets, scoring viejo, caminos vivos, samplers balanceados y filtros.
- **`causal_viz`**: DAG estático (networkx) o interactivo (Dash/Cytoscape) con valores e intervenciones.
