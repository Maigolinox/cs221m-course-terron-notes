# Lecture 9: Diseño de contrafactuales para interchange interventions

Esta notebook (`09_counterfactual_design.ipynb`) no presenta un método nuevo de interpretabilidad, sino una pregunta metodológica: **cómo elegir los pares base/source** para que una interchange intervention diga algo real sobre dónde vive una variable causal dentro de la red. La tesis es que un heatmap de IIA no es evidencia por sí solo: es evidencia **relativa al contraste contrafactual que induce el dataset**. Si el dataset no separa la variable objetivo $V$ de una variable competidora $W$, un IIA alto en un sitio es compatible con "este sitio codifica $V$" y con "este sitio codifica $W$".

La notebook aplica la misma receta a tres tareas: hierarchical equality (Parte 1), MCQA (Parte 2, mecanismo pointer-value) y entity binding (Parte 3, dos modelos causales que compiten). Todo el código pesado vive en `utils/lecture_09_utils.py` (explicado función por función en la sección final de estos apuntes) y en la librería `causalab` (`causalab/causalab/`).

---

## 1. Ideas clave / teoría

### 1.1 Vocabulario

- **Causal model** ($\mathcal{A}$ o $\mathcal{H}$): variables + mecanismos que calculan cada variable a partir de sus padres. En causalab es la clase `CausalModel` (`causalab/causalab/causal/causal_model.py`).
- **Trace**: una corrida concreta del modelo causal (inputs, intermedias y output). En causalab es `CausalTrace` (`causalab/causalab/causal/trace.py`).
- **Counterfactual example**: un trace base + uno o más traces contrafactuales (source). En causalab es un `TypedDict` con claves `"input"` y `"counterfactual_inputs"` (`causalab/causalab/causal/counterfactual_dataset.py`).
- **Intervención** $\mathcal{A}_{X\leftarrow x}$: fijar a la fuerza $X=x$, cortando el enlace con sus padres.
- **Interchange intervention** $\mathcal{A}(b)_{V\leftarrow\mathcal{A}(s)}$: la intervención donde el valor que se fija viene de correr el modelo sobre el source $s$. En la red: $\mathcal{N}(b)_{\Pi_V\leftarrow\mathcal{N}(s)}$, donde $\Pi_V$ es el conjunto de sitios (capa, posición) que alineamos con $V$. Acá casi siempre $\Pi_V$ es un único sitio del residual stream.
- **Localization hypothesis**: "el sitio $\Pi_V$ de la red juega el rol causal de $V$".
- $\rho$: la traducción del input de alto nivel al prompt que ve el LM (en el código: los templates y las funciones que generan `raw_input`).

### 1.2 Causal abstraction e IIA

La condición general de causal abstraction que la notebook recuerda es

$$
\tau_\pi(\textsf{Run}(\mathcal{N}_{X \leftarrow x}(i))) = \textsf{Run}(\mathcal{A}_{\omega_\pi(X \leftarrow x)}(\rho_\pi(i)))
$$

para todo input $i$ y toda intervención: correr la red intervenida y traducir el trace ($\tau_\pi$) tiene que dar lo mismo que correr el algoritmo con la intervención traducida ($\omega_\pi$) sobre el input traducido ($\rho_\pi$). En la práctica solo se mira el output, y eso es la **IIA**:

$$
\mathrm{IIA}(V,\Pi_V;\mathcal{D}) = \frac{1}{|\mathcal{D}|}\sum_{(b,s)\in\mathcal{D}} \mathbf{1}\!\left[\pi_O\!\left(\mathcal{N}(b)_{\Pi_V \leftarrow \mathcal{N}(s)}\right) = \mathcal{A}(\rho(b))_{V \leftarrow \mathcal{A}(\rho(s))}\right]
$$

Término a término: $\mathcal{D}$ es el dataset de pares; $\mathcal{N}(b)_{\Pi_V\leftarrow\mathcal{N}(s)}$ es correr el LM en $b$ pegando en $\Pi_V$ la activación que tuvo en $s$; $\pi_O$ lee el output del LM (en el código: el string generado, comparado con un `metric`); el lado derecho es el output que predice el modelo causal tras el mismo interchange en $V$. IIA = 1 significa que en **ese** dataset la red intervenida siempre coincide con el algoritmo intervenido.

### 1.3 Los tres criterios de un par diagnóstico

Definiciones (son las que implementa el código, ver §4):

$$
Y_V(b,s) := \mathcal{A}(\rho(b))_{V \leftarrow \mathcal{A}(\rho(s))}[O], \qquad Y_{\mathrm{base}}(b) := \mathcal{A}(\rho(b))[O]
$$

1. **Variable-level change**: $\Delta_V(b,s) = \mathbf{1}[\mathcal{A}(\rho(b))[V] \neq \mathcal{A}(\rho(s))[V]]$. Si $V$ vale lo mismo en base y source, el interchange no cambia nada y el par no testea nada.
2. **Output-changing test intervention**: $Y_V(b,s) \neq Y_{\mathrm{base}}(b)$. Que $V$ cambie no basta: el interchange tiene que predecir un output distinto al de la base. Si no, un sitio que no hace nada (y deja la base intacta) sacaría IIA perfecta.
3. **Distinguishability** respecto de un competidor $W$: $Y_V(b,s) \neq Y_W(b,s)$. Si el interchange en $V$ y el interchange en $W$ predicen el mismo output, el par es **confounding**: un IIA alto no permite decidir si el sitio realiza $V$ o $W$.

A nivel dataset, el criterio 3 se resume con la **distinguishability rate**

$$
\mathrm{Dist}_{V,W}(\mathcal{D}) = \frac{1}{|\mathcal{D}|}\sum_{(b,s)\in\mathcal{D}} \mathbf{1}[Y_V(b,s) \neq Y_W(b,s)].
$$

El punto fino: como no sabemos de antemano dónde está $V$ en la red (eso es justamente lo que buscamos), la única forma de que un IIA alto apunte a $V$ y no a $W$ es que el dataset ya los separe **a nivel del modelo causal**, antes de mirar la red. Por eso todos los diagnósticos se calculan solo con el modelo causal, sin LM.

### 1.4 Los tres roles de dataset

| Dataset | Qué impone | Para qué sirve |
|---|---|---|
| `random` | nada | baseline: cumple criterios solo por casualidad |
| `<V>_targeted` | (1) + (2) | $V$ cambia y su patch cambia el output, pero $W$ queda libre |
| `<V>_isolated` | (1) + (2) + (3), y además $W$ fijo | separa $V$ de $W$ |

Las parejas $(V, W)$ de cada parte:

| Parte | Tarea | $V$ | $W$ |
|---|---|---|---|
| 1 | Hierarchical equality | `left_eq` | `right_eq` |
| 2 | MCQA | `answer_position` | `answer` |
| 3 | Entity binding | `query_entity` | `positional_query_group` |

La comparación principal en los heatmaps es siempre `isolated` vs `targeted`: ambos cumplen (1) y (2), solo `isolated` cumple (3). Un sitio que tenía IIA alto en `targeted` y lo pierde en `isolated` era un sitio cuya señal estaba confundida con $W$.

### 1.5 Cómo funciona un interchange en el modelo causal de causalab (importante para todo lo demás)

Todo el diagnóstico descansa en `CausalTrace.intervene` y en el recálculo de descendientes. En `causalab/causalab/causal/trace.py`:

- `Mechanism(parents, compute)` es un dataclass: lista de padres + función que recibe el trace y devuelve el valor.
- `input_var(values)` crea un mecanismo sin padres que hace `random.choice(values)` (solo se usa si no le pasás el input explícitamente).
- `CausalTrace.__init__(mechanisms, inputs)` calcula los hijos de cada variable y llama `_set(var, val)` por cada input; `_set` guarda el valor y llama `_recompute_descendants`, que recorre los hijos y calcula cada uno **solo si todos sus padres ya tienen valor**, recursivamente. Así, al terminar de setear los inputs, todas las intermedias y `raw_input`/`raw_output` quedan calculadas.
- `intervene(variable, value)` guarda el valor, **reemplaza el mecanismo por una constante** (`Mechanism(parents=[], compute=lambda t, v=value: v)`) y recalcula descendientes. Reemplazar el mecanismo es lo que "corta el enlace con los padres".
- `copy()` hace deepcopy de los mecanismos y copia superficial de los valores, para que intervenir la copia no toque el original.

Y en `causalab/causalab/causal/causal_model.py`, `CausalModel.run_interchange(input_trace, counterfactual_inputs)` hace `trace = input_trace.copy()` y, para cada `var` del dict, `trace.intervene(var, cf_trace[var])`. O sea: $Y_V(b,s)$ = `model.run_interchange(base, {V: source})["raw_output"]`. También soporta la sintaxis `"A<-B"` (tomar `B` del source y ponerlo en `A` de la base), que acá no se usa.

---

## 2. Recorrido por la notebook

### 2.0 Setup

La celda 24 detecta si estamos en Colab (clona el repo e instala `requirements/09_counterfactual_design.txt`) o local (configura el renderer de plotly e instala chrome para exportar). Puro setup.

Celda 26 (imports):

```python
from causalab.causal.causal_model import CausalModel
from causalab.causal.trace import Mechanism, input_var
from utils.lecture_09_utils import (set_seed, make_dataset, ..., diagnose_entity_binding_sites)
set_seed(0)
```

- Importa las dos piezas de causalab para escribir el modelo causal a mano (Parte 1) y todos los helpers de `utils/lecture_09_utils.py`.
- Al importar el módulo de utils pasan cosas: busca la raíz del repo y agrega `causalab` al `sys.path`, y parchea `causal_utils.statement_conjunction_function` si no existe (lo necesita la tarea de entity binding). Ver §4.
- `set_seed(0)` fija `random` y `numpy`. Ojo: **no** fija `torch`, pero acá la generación es greedy (`do_sample=False`), así que no importa.

Celda 28 (carga del modelo):

```python
FILTER_BATCH_SIZE = 4
PATCH_BATCH_SIZE = 4
pipeline, _device = load_activation_patching_pipeline(max_new_tokens=5, max_length=768)
print(pipeline.model.name_or_path)
```

- `load_activation_patching_pipeline` (utils) prueba cargar, en orden, `$CAUSALAB_PATCHING_MODEL`, `meta-llama/Llama-3.2-1B-Instruct`, `Qwen/Qwen2.5-0.5B-Instruct` y `TinyLlama`, y devuelve el primero que carga como `LMPipeline`. En la ejecución guardada se cargó `meta-llama/Llama-3.2-1B-Instruct`.
- `max_new_tokens=5`: cada generación produce 5 tokens; el metric después mira solo el principio.
- `max_length=768`: todos los prompts se paddean a 768 tokens (el prompt de Parte 1 tiene few-shot largo).

Qué hace `LMPipeline` de verdad (`causalab/causalab/neural/pipeline.py`):
- `_setup_model`: carga tokenizer y `AutoModelForCausalLM`, fuerza `attn_implementation="eager"` y `use_cache=False` en la config, define `pad_token = eos_token` y `padding_side="left"` por defecto.
- `load(traces)`: toma `trace["raw_input"]` de cada trace, tokeniza con `padding="max_length"` si hay `max_length` (si no, padding dinámico), con truncation. Con padding a la izquierda, el último token real siempre está en el índice `len-1`, lo que simplifica "last_token".
- `generate(traces)`: `model.generate` greedy con `max_new_tokens`, devuelve `{"scores", "sequences" (los últimos max_new_tokens ids), "string" (decodificado)}`.
- `compute_outputs(dataset, batch_size)`: corre `generate` sobre todas las bases y todos los contrafactuales, en batches, y devuelve listas planas de dicts por ejemplo. Lo usa el filtrado.

### 2.1 Sección conceptual (celdas 29-48)

Son markdown con los tres criterios, las fórmulas y la tabla de $(V,W)$, que ya resumí en §1.3-1.4. Un detalle de lectura de los diagnósticos: las filas $\Delta W$ y $\texttt{patch}(W)\to\Delta Y$ son **diagnósticas**, no requisitos; la fila que hay que mirar primero es $Y_V \neq Y_W$.

### 2.2 Parte 1: Hierarchical equality

#### El modelo causal (celda 55)

```python
HIER_EQ_TEMPLATE = """Determine if the first two numbers are equal, ... 
5 5 8 8: First pair equal (5=5). Last pair equal (8=8). Both equal? Yes
Answer: Yes
...
{left_a} {left_b} {right_a} {right_b}:
Answer:"""
```

Template few-shot con 4 ejemplos razonados (cubren las 4 combinaciones de igualdad) y los 4 huecos `{left_a}`... al final. Esos huecos con llaves importan después: el builder de token positions de causalab parsea `{nombre}` para saber dónde cae cada variable.

```python
def make_hierarchical_equality_toy_model(*, n_values: int = 50) -> CausalModel:
    vals = list(range(n_values))
    values = {
        "left_a": vals, "left_b": vals, "right_a": vals, "right_b": vals,
        "left_eq": [False, True], "right_eq": [False, True],
        "raw_input": None, "raw_output": [" No", " Yes"],
    }
```

- `values` declara el dominio de cada variable. `raw_input` es `None` porque es texto libre. `raw_output` lleva espacio adelante (`" Yes"`) porque así tokeniza la respuesta tras `Answer:`.

```python
    mechanisms = {
        "left_a": input_var(vals), ...,
        "left_eq": Mechanism(parents=["left_a", "left_b"], compute=lambda t: t["left_a"] == t["left_b"]),
        "right_eq": Mechanism(parents=["right_a", "right_b"], compute=lambda t: t["right_a"] == t["right_b"]),
        "raw_output": Mechanism(parents=["left_eq", "right_eq"],
                                compute=lambda t: " Yes" if (t["left_eq"] == t["right_eq"]) else " No"),
        "raw_input": Mechanism(parents=["left_a", "left_b", "right_a", "right_b"],
                               compute=lambda t: HIER_EQ_TEMPLATE.format(...)),
    }
    return CausalModel(mechanisms, values, id=f"hierarchical_equality_{n_values}")
```

- Las 4 hojas son `input_var`. `left_eq` $=E_1=(I_1=I_2)$, `right_eq` $=E_2=(I_3=I_4)$, `raw_output` $=O=(E_1=E_2)$. `raw_input` es $\rho$: rellena el template.
- `CausalModel.__init__` (`causal_model.py`): deriva `variables` de las claves de `mechanisms`, exige que existan `raw_input` y `raw_output` (assert), deriva `parents` y `children`, define `inputs` como las variables sin padres, calcula un `timestep` por variable (0 para inputs, +1 por nivel; las variables sin hijos, los `outputs`, se mandan al último timestep), valida consistencia padres/hijos/timesteps, ordena `variables` por timestep y arma `print_pos` (posiciones para dibujar el grafo).

```python
hier_eq = make_hierarchical_equality_toy_model(n_values=50)
show_localization_view(hier_eq, target_vars=["left_eq"], control_vars=["right_eq"], title=...)
```

- `show_localization_view` (utils) dibuja el DAG con networkx coloreando `left_eq` (target, azul), `right_eq` (control, violeta) y `raw_output` (naranja). Usa `model.print_pos` como layout.

#### Step 1: datasets y diagnósticos (celda 58)

```python
def _sample_pair_with_target_eq(is_eq, *, n_values):
    a = random.randrange(n_values)
    if is_eq:
        return a, a
    b = random.randrange(n_values - 1)
    return a, (b + 1 if b >= a else b)
```

- Devuelve un par `(a, b)` con `a == b` exactamente cuando `is_eq`. El truco `b+1 if b >= a` samplea uniforme entre los `n_values-1` valores distintos de `a` sin rechazo.

```python
def _sample_hier_eq_with_eq(left_is_eq, right_is_eq, *, n_values=50):
    la, lb = _sample_pair_with_target_eq(left_is_eq, n_values=n_values)
    ra, rb = _sample_pair_with_target_eq(right_is_eq, n_values=n_values)
    return {"left_a": la, "left_b": lb, "right_a": ra, "right_b": rb}
```

- Arma un dict de inputs con los booleanos `left_eq`/`right_eq` que le pidas. Es un sampler controlado por las variables intermedias, no por las hojas.

```python
def hier_eq_random_pair(*, n_values=50):
    base_l = random.choice([False, True]); base_r = random.choice([False, True])
    cf_l = random.choice([False, True]);   cf_r = random.choice([False, True])
    base = _sample_hier_eq_with_eq(base_l, base_r, n_values=n_values)
    cf = _sample_hier_eq_with_eq(cf_l, cf_r, n_values=n_values)
    return {"input": hier_eq.new_trace(base), "counterfactual_inputs": [hier_eq.new_trace(cf)]}
```

- Sortea los 4 booleanos de forma independiente, construye base y source y los convierte en traces con `hier_eq.new_trace(dict)`. `CausalModel.new_trace` crea un `CausalTrace` con una **deepcopy** de los mecanismos y los inputs dados; al setearlos se calcula todo el resto.
- Devuelve un `CounterfactualExample`: `{"input": trace, "counterfactual_inputs": [trace]}`.
- Ojo: esto es "random balanceado en las intermedias" (50% de pares iguales), no uniforme sobre las hojas (que daría ~2% de pares iguales con 50 valores). El markdown dice "resample leaves uniformly", pero el código no hace eso.

```python
def hier_eq_target_isolated_pair(*, target="left_eq"):
    base_l = random.choice([False, True]); base_r = random.choice([False, True])
    if target == "left_eq":
        cf_l, cf_r = not base_l, base_r
    else:
        cf_l, cf_r = base_l, not base_r
    ...
```

- Para el dataset aislado: el source **invierte** `left_eq` y **copia** `right_eq`. Por construcción $\Delta V = 1$ y $\Delta W = 0$. Como $O = (E_1 = E_2)$, invertir $E_1$ siempre invierte $O$ (criterio 2), y el patch en $W$ no cambia nada (deja $Y_{base}$), así que $Y_V \neq Y_W$ siempre (criterio 3).

```python
N = 64 if _device == "cuda" else 16
hier_eq_datasets = {
    "left_eq_targeted": build_counterfactual_dataset(causal_model=hier_eq, sampler=hier_eq_random_pair, n=N,
        target="left_eq", other="right_eq", require_distinguishability=False, require_other_fixed=False),
    "left_eq_isolated": build_counterfactual_dataset(causal_model=hier_eq, sampler=hier_eq_target_isolated_pair, n=N,
        target="left_eq", other="right_eq", require_distinguishability=True, require_other_fixed=True),
    "random": make_dataset(N, hier_eq_random_pair),
}
```

- `build_counterfactual_dataset` (utils) es **rejection sampling**: llama al sampler hasta juntar `n` pares que pasen `_satisfies_criteria` (máx. `300·n` intentos, si no, `RuntimeError`). `_satisfies_criteria` chequea en orden: (1) `base[target] != cf[target]`; (2) `run_interchange(base, {target: cf})["raw_output"] != base["raw_output"]`; si `require_other_fixed`, `base[other] == cf[other]`; si `require_distinguishability`, que el patch en `target` y el patch en `other` den outputs distintos.
- `targeted` usa el sampler random y filtra por (1)+(2): en esta tarea (2) es automático cuando cambia `left_eq`, así que `targeted` = pares random con `left_eq` distinto, con `right_eq` libre.
- `isolated` usa el sampler controlado y además exige (3) y `right_eq` fijo (redundante con el sampler, pero es una verificación).
- `random` = `make_dataset(N, sampler)`, que es literalmente `[sampler() for _ in range(n)]` sin filtro.

```python
target_var, competing_var = "left_eq", "right_eq"
display_change_rate_table(hier_eq_datasets, [target_var, competing_var], title=...)
display_effect_rate_table(hier_eq, hier_eq_datasets, {f"patch({target_var})": [target_var], f"patch({competing_var})": [competing_var]}, title=...)
show_localization_view(...)
display_localization_scorecard(hier_eq, hier_eq_datasets, target_vars=[target_var], control_vars=[competing_var], title=...)
plot_dataset_criteria_bars(hier_eq, hier_eq_datasets, target_vars=[target_var], control_vars=[competing_var], title=...)
```

- **Change-rate table**: % de pares con `base[var] != cf[var]` (criterio 1 para $V$ y para $W$).
- **Effect-rate table**: % de pares donde `patch(var)` cambia el output respecto de la base (criterio 2). Internamente llama `compute_testable_rate(model, ds, [var], None)` → `can_distinguish_with_dataset(..., vars2=None)` (ver abajo).
- **Scorecard**: junta los anteriores + "separates V vs W" = criterio 3, vía `can_distinguish_with_dataset(..., vars1=[V], vars2=[W])`.
- `plot_dataset_criteria_bars`: el mismo scorecard como barras horizontales agrupadas (filas = criterios, una barra por dataset).

**`can_distinguish_with_dataset` (el corazón de los diagnósticos).** Está en `causalab/causalab/causal/causal_utils.py` (hay además un método homónimo en `CausalModel`, pero utils importa la función del módulo `causal_utils`):

```python
def can_distinguish_with_dataset(dataset, causal_model1, target_variables1, causal_model2=None, target_variables2=None):
    count = 0
    for example in dataset:
        input_data = example["input"]
        counterfactual_inputs = example["counterfactual_inputs"]
        assert len(counterfactual_inputs) == 1
        cf_trace = counterfactual_inputs[0]
        setting1 = input_data.copy()
        for var in target_variables1:
            setting1.intervene(var, cf_trace[var])
        if causal_model2 is not None and target_variables2 is not None:
            setting2 = input_data.copy()
            for var in target_variables2:
                setting2.intervene(var, cf_trace[var])
            if setting1["raw_output"] != setting2["raw_output"]:
                count += 1
        else:
            if setting1["raw_output"] != input_data["raw_output"]:
                count += 1
    proportion = count / len(dataset)
    return {"proportion": proportion, "count": count}
```

Línea por línea:
- Recorre cada par; exige exactamente un contrafactual por ejemplo (`assert`).
- `setting1 = input_data.copy()` y luego `intervene` en cada variable de `target_variables1` con el valor que tiene en el source: eso es $\mathcal{A}(b)_{V\leftarrow\mathcal{A}(s)}$, y `setting1["raw_output"]` es $Y_V(b,s)$.
- Si se pasa un segundo conjunto de variables (y un segundo modelo no nulo), hace lo mismo para $W$ y cuenta el par si $Y_V \neq Y_W$ → **criterio 3**.
- Si no, compara $Y_V$ contra `input_data["raw_output"]` = $Y_{base}$ → **criterio 2**.
- Devuelve proporción y conteo.

Un detalle que no es obvio: los argumentos `causal_model1` y `causal_model2` **no se usan para calcular nada**. La intervención se hace sobre las copias de los traces, que ya traen sus propios mecanismos (copiados del modelo con que se crearon). `causal_model2` solo funciona como bandera "¿comparo contra $W$ o contra la base?". Por eso `compute_testable_rate` en utils pasa `model if vars2 is not None else None`. Si quisieras comparar dos modelos causales distintos de verdad, esta función no lo haría (usaría los mecanismos de los traces en ambos casos). Para esta notebook no importa porque $V$ y $W$ son del mismo modelo.

**Salida guardada** (N=64): `left_eq_targeted` tiene ΔV=100%, ΔW=54.7%, patch(V) cambia el output 100%, patch(W) 54.7%, separación 45.3%. `left_eq_isolated`: 100/0/100/0 y separación 100%. `random`: ΔV=54.7%, ΔW=48.4%, separación 43.8%. Fijate que en esta tarea, para `targeted`, separación = 100% − ΔW: cada vez que `right_eq` también cambia, el patch en $W$ también invierte el output y el par queda confundido.

#### Step 2: leer pares a mano (celda 61)

```python
print_one_pair_per_dataset(hier_eq_datasets, causal_model=hier_eq, variables=["left_eq", "right_eq"],
                           target="left_eq", other="right_eq", title=...)
```

- Para cada dataset imprime el primer par: `raw_input` de base y source, valor de cada variable (con CHANGED/same), outputs, y los dos patches `patch(left_eq)` y `patch(right_eq)` calculados con `_patch_outputs` (que llama `model.run_interchange`).

```python
def _hier_matches_confounding(ex):
    b, c = ex["input"], ex["counterfactual_inputs"][0]
    return bool(b["left_eq"]) and not b["right_eq"] and not c["left_eq"] and bool(c["right_eq"])

def _hier_matches_nonconfounding(ex):
    ...
    return bool(b["left_eq"]) and not b["right_eq"] and not c["left_eq"] and not c["right_eq"]

print_confounding_and_nonconfounding(causal_model=hier_eq, ..., confounding_source=hier_eq_datasets["left_eq_targeted"],
    nonconfounding_source=hier_eq_datasets["left_eq_isolated"], confounding_filter=_hier_matches_confounding,
    nonconfounding_filter=_hier_matches_nonconfounding, title=...)
```

- Los dos predicados piden el patrón pedagógico: base = (T, F) (output No). Source confounding = (F, T): parchear $V$ da (F,F) → Yes; parchear $W$ da (T,T) → Yes. Mismo output, confundido. Source no-confounding = (F, F): parchear $V$ da (F,F) → Yes, parchear $W$ da (T,F) → No. Distintos.
- `print_confounding_and_nonconfounding` busca con `next(...)` el primer ejemplo de cada fuente que cumpla "es confounding" (`y_t == y_o`) + el filtro, y lo imprime con el veredicto.

En la salida guardada, el par de `targeted` (base `32 32 31 31`, source `25 19 30 30`) tiene `patch(left_eq) -> No` y `patch(right_eq) -> Yes`: ese par en particular sí separa, porque `right_eq` no cambió. La celda 63 repite el ejemplo mínimo a mano: base (7,7,2,5), source (3,8,4,4) confunde; source (3,8,2,5) no.

#### Step 3: interchange en el residual stream (celdas 66-67)

```python
_hier_filtered = filter_and_report(datasets=hier_eq_datasets, pipeline=pipeline, causal_model=hier_eq,
                                   metric=hier_metric, batch_size=FILTER_BATCH_SIZE, label="hier")
```

- `filter_and_report` (utils) llama, por dataset, a `causalab.experiments.filter.filter_dataset` y reporta cuántos quedan; si alguno queda en 0 lanza `RuntimeError`, y si quedan menos de `min_kept_warn` (4 por defecto) avisa.
- `filter_dataset` (`causalab/causalab/experiments/filter.py`): corre `pipeline.compute_outputs(dataset)`, y conserva un ejemplo solo si `metric(output_LM_base, base["raw_output"])` **y** `metric(output_LM_cf, cf["raw_output"])` para todos sus contrafactuales (`validate_counterfactuals=True` por defecto). O sea: nos quedamos con pares donde el LM resuelve bien **base y source** sin intervenir. Esto da el punto de referencia: si el LM no hace la tarea, no tiene sentido preguntar dónde la implementa.
- `hier_metric` (utils): normaliza y compara la **primera palabra** generada con `"yes"`/`"no"`.

Salida guardada: `targeted` 26/64, `isolated` 11/64, `random` 22/64. El modelo de 1B resuelve mal la tarea y el dataset aislado queda con solo 11 pares: los heatmaps de Parte 1 son ruidosos (cada par vale ~9 puntos de IIA).

```python
_hier_sites = build_hier_token_positions(pipeline, prompt_template=HIER_EQ_TEMPLATE)
_hier_token_positions = _hier_sites["positions"]
_hier_pos_id_to_label = _hier_sites["label_map"]
_layers_hier = default_layers(pipeline, _device)
```

- `build_hier_token_positions` (utils) arma 5 sitios con el sistema declarativo de causalab: el primer token (índice 0 dentro del span) de cada uno de `left_a`, `left_b`, `right_a`, `right_b`, más `last_token` (posición −1).
- `default_layers` devuelve `[-1] + list(range(num_layers))`. La capa −1 significa "embeddings" (ver abajo). Llama-3.2-1B tiene 16 capas → 17 filas.

**Cómo resuelve causalab las posiciones** (`causalab/causalab/neural/token_position_builder.py`):
- `build_token_position_factories(specs, template)` devuelve, por nombre, una *factory* que recibe el pipeline y devuelve un `TokenPosition`.
- `{"type": "index", "position": -1}` sin scope → `_build_absolute_index_factory`: tokeniza el prompt con `pipeline.load` y devuelve `[len(ids) - 1]`.
- `{"type": "index", "position": 0, "scope": {"variable": "left_a"}}` → `_build_scoped_index_factory`: primero obtiene todos los tokens de la variable y devuelve el de índice 0.
- Los tokens de una variable se obtienen con `Template.get_variable_positions`: parsea el template en literales y `{variables}`, rellena con los valores del trace llevando la cuenta de los rangos de caracteres de cada variable, tokeniza el texto completo con `pipeline.load(..., return_offsets_mapping=True)` y mapea rangos de caracteres a índices de token. Tiene un cache por (tokenizer, texto).
- `TokenPosition` (subclase de `ComponentIndexer`, `causalab/causalab/neural/model_units.py`) envuelve una función `input -> list[int]` con un `id`. Su `index(input, batch, is_original)` le pasa `is_original` al indexer si este lo acepta (lo usa `both_e1_swap` en la Parte 3).

```python
results_hier, targets_hier = run_patching_for_datasets(pipeline=pipeline, causal_model=hier_eq,
    datasets=_hier_filtered, token_positions=_hier_token_positions, target_variable_groups=[("left_eq",)],
    metric=hier_metric, batch_size=PATCH_BATCH_SIZE, layers=_layers_hier)
```

`run_patching_for_datasets` (utils) hace:
1. `build_residual_stream_targets(pipeline, layers, token_positions, mode="one_target_per_unit")` (`causalab/causalab/experiments/interchange_targets.py`): por cada (capa, posición) crea un `ResidualStream(layer, token_indices=pos, featurizer=None, shape=(hidden_size,), target_output=True)`. Caso especial: capa −1 → capa 0 con `target_output=False`, es decir el **input del bloque 0** = embeddings. Con `one_target_per_unit` devuelve `{(layer, pos.id): InterchangeTarget([[unit]])}`: un target por celda del heatmap. Por eso la barra de progreso de MCQA dice 119 = 17 capas × 7 posiciones.
2. `ResidualStream` (`causalab/causalab/neural/LM_units.py`) es un `AtomicModelUnit` con `component_type="block_output"` (o `"block_input"`), unidad `"pos"` y featurizer identidad (`Featurizer()` por defecto, `causalab/causalab/neural/featurizers.py`). Featurizer identidad + `feature_indices=None` = se reemplaza **el vector completo** del residual stream en esa posición.
3. Guarda cada dataset como JSON (`save_counterfactual_examples`: serializa cada trace con `to_dict()`), y llama `run_interchange_score_heatmap(...)` con `save_results=False`.

`run_interchange_score_heatmap` (`causalab/causalab/experiments/jobs/interchange_score_grid.py`):
- Detecta el tipo de componente mirando el id de la primera unidad (`"ResidualStream" in id`).
- Recarga el dataset con `load_counterfactual_examples` (JSON → `causal_model.new_trace(dict)` por cada trace).
- Para cada target, `run_interchange_interventions(pipeline, dataset, target, batch_size, output_scores=False)`.
- Puntúa con `causal_score_intervention_outputs(raw_results, dataset, causal_model, target_variable_groups, metric)`.
- Devuelve `{"scores": {var_group: {(layer, pos_id): IIA}}, ...}`. utils se queda con `result["scores"]`.

`run_interchange_interventions` (`causalab/causalab/neural/pyvene_core/interchange.py`):
- `prepare_intervenable_model(pipeline, target, "interchange")` (`intervenable_model.py`): por cada unidad crea un config de pyvene `{"component": "block_output", "unit": "pos", "layer": L, "group_key": i, "intervention_type": <clase>}` y arma un `pv.IntervenableModel` sobre `pipeline.model`. La clase de intervención la da `Featurizer.get_interchange_intervention()`.
- Por batch, `batched_interchange_intervention`: `prepare_intervenable_inputs` calcula los índices de token **por separado** en las bases (`is_original=True`) y en los sources (`is_original=False`), tokeniza ambos y arma `unit_locations={"sources->base": (idx_source, idx_base)}`. Luego `pipeline.intervenable_generate` llama `intervenable_model.generate(base, sources=..., unit_locations=..., subspaces=feature_indices, intervene_on_prompt=True, do_sample=False, ...)` y decodifica.
- La intervención (`FeatureInterchangeIntervention` en `featurizers.py`) hace: `f_base, err = featurizer(base)`, `f_src = featurizer(source)`; si no hay subespacios, `f_out = f_src`; devuelve `inverse(f_out, err)`. Con featurizer identidad eso es literalmente "pegar la activación del source".

`causal_score_intervention_outputs` (`causalab/causalab/experiments/metric.py`):
- Por cada grupo de variables crea `make_causal_metric(metric, var_group)` y llama `score_intervention_outputs`.
- `score_intervention_outputs` calcula las etiquetas esperadas con `causal_model.label_counterfactual_data(dataset, target_variables)`, que por cada ejemplo hace `run_interchange(base, {var: cf})` y guarda `setting["raw_output"]` como `label`. Eso es el lado derecho de la IIA.
- Luego aplana los strings generados, aplica `metric(output, label)` (1.0/0.0) por ejemplo y promedia por target. Ese promedio es la IIA de la celda.

#### Step 4: leer los heatmaps (celda 70)

```python
plot_iia_target_control_random(results=results_hier, targets=targets_hier, var_name="left_eq",
    dataset_targeted="left_eq_isolated", dataset_control="left_eq_targeted", dataset_random="random",
    pos_id_to_label=_hier_pos_id_to_label, title_prefix=...)
```

- Ojo con los nombres de argumentos: `dataset_targeted` recibe el dataset **isolated** (el "bueno") y `dataset_control` el **targeted**. La función dibuja dos tripletes: (targeted, isolated, ΔIIA = isolated − targeted) y (random, isolated, ΔIIA = isolated − random).
- Cada panel: filas = capas (arriba la última, abajo "Embed"), columnas = posiciones. Rojo en ΔIIA = el aislado da más; azul = la comparación da más. Negrita si |valor| > 0.5.

Qué mirar: un sitio azul fuerte en ΔIIA (alto en `targeted`, bajo en `isolated`) es un sitio cuyo IIA en `targeted` venía de pares confundidos con `right_eq`. Un candidato serio para `left_eq` tiene IIA alto en `isolated`. Esperable por el grafo: `left_eq` solo puede formarse en posiciones donde ya se vieron `left_a` y `left_b` (en `left_b` o después). La celda 71 solo imprime que no hay segundo heatmap.

### 2.3 Parte 2: MCQA y el mecanismo pointer-value

#### La tarea y el modelo causal (celda 73)

```python
from causalab.tasks.MCQA.causal_models import positional_causal_model
from causalab.tasks.MCQA.counterfactuals import different_symbol, random_counterfactual, same_symbol_different_position, sample_answerable_question
show_localization_view(positional_causal_model, target_vars=["answer_position"], control_vars=["answer"], title=...)
example = sample_answerable_question()
print(example["raw_input"]); print(example["raw_output"].strip()); print(example["answer_position"]); print(example["answer"])
```

El modelo está en `causalab/causalab/tasks/MCQA/causal_models.py`:
- `NUM_CHOICES = 2` (ojo: el markdown habla de "A/B/C/D", pero el código usa **dos** opciones), `ALPHABET` = 26 letras, 10 objetos, 10 colores.
- Template: `"The {object} is {color}. What color is the {object}?\n{symbol0}. {choice0}\n{symbol1}. {choice1}\nAnswer:"`.
- Inputs: `template`, `object`, `color`, `symbol0`, `symbol1`, `choice0`, `choice1`.
- `answer_position` (padres `color`, `choice0`, `choice1`): el índice `i` tal que `choice_i == color` (si no hay, `ValueError`).
- `answer` (padres `answer_position`, `symbol0`, `symbol1`): `t[f"symbol{answer_position}"]`.
- `raw_output` (padre `answer`): `" " + answer`.

Esto es exactamente un **pointer-value**: `answer_position` es un puntero (una dirección, 0 ó 1) y `answer` es el valor que se obtiene al desreferenciarlo en la tabla de símbolos de **la base**. Por eso el interchange en `answer_position` no copia la letra del source: copia la dirección y luego lee la letra que la base tiene en esa dirección. El interchange en `answer`, en cambio, copia la letra del source directamente.

`sample_answerable_question` (`causalab/causalab/tasks/MCQA/counterfactuals.py`): samplea objeto y color, elige una posición al azar para el color correcto, rellena las otras opciones con colores distintos, samplea `NUM_CHOICES` letras **distintas** con `random.sample`, y construye el trace con `positional_causal_model.new_trace`. Garantiza que la pregunta tenga respuesta. Salida guardada: "The house is pink... G. white / M. pink" → output M, position 1.

#### Step 1: datasets (celda 75)

```python
N_MCQA = 64
mcqa_datasets_diag = {
    "answer_position_targeted": build_counterfactual_dataset(causal_model=positional_causal_model, sampler=random_counterfactual, n=N_MCQA,
        target="answer_position", other="answer", require_distinguishability=False, require_other_fixed=False),
    "answer_position_isolated": build_counterfactual_dataset(causal_model=positional_causal_model, sampler=same_symbol_different_position, n=N_MCQA,
        target="answer_position", other="answer", require_distinguishability=True, require_other_fixed=True),
    "random": make_dataset(N_MCQA, random_counterfactual),
}
```

- `random_counterfactual`: dos `sample_answerable_question()` independientes.
- `same_symbol_different_position`: samplea una base, elige otra posición `new_pos`, copia los inputs y **intercambia tanto las opciones como los símbolos** entre `pos` y `new_pos`. Resultado: el color correcto se mueve de posición pero sigue pegado a la misma letra. $\Delta$`answer_position` = 1, $\Delta$`answer` = 0.
- Ejemplo de la salida guardada (isolated): base "D. black / E. yellow" (pos 1, letra E), source "E. yellow / D. black" (pos 0, letra E). `patch(answer_position)` → la base leída en posición 0 → **D**. `patch(answer)` → E. Son distintos, y además D difiere tanto de la salida de la base (E) como de la del source (también E). Esta es la idea de Mueller et al. / Prakash et al. que cita la notebook: con estos pares, parchear el puntero predice una respuesta distinta **tanto** del output original **como** del output contrafactual, y un sitio que solo transporta "la letra de la respuesta" no puede sacar IIA alta.
- Los otros generadores importados (`different_symbol`) no se usan en esta notebook.

**Salida guardada:** `targeted` ΔP=100%, ΔA=96.9%, separación 93.8%; `isolated` 100/0, separación 100%; `random` ΔP=40.6%, ΔA=95.3%, separación 98.4%.

Por qué `targeted` ya separa ~94% aunque no lo exija: las letras se samplean al azar en cada prompt, así que $Y_V$ (letra de la base en la nueva posición) casi nunca coincide con $Y_W$ (letra del source). El confounding del ejemplo didáctico de la celda 78 (letras fijas A/B en las dos preguntas) es raro con este generador. Y en `random` la separación es alta incluso cuando la posición no cambia: ahí $Y_V = Y_{base}$ y $Y_W$ = letra del source, distintas. O sea, en MCQA el criterio 3 casi viene gratis; lo que de verdad diferencia `isolated` es que **fija** $W$ (Δanswer = 0) y que source y base tienen el mismo output.

#### Step 2: pares a mano (celda 77)

Mismo esquema que en Parte 1, sin filtros extra. En la salida guardada, el par `targeted` (base "C. green / A. blue", source "L. purple / V. brown") da `patch(answer_position) -> A` y `patch(answer) -> V`.

#### Step 3: filtrado y patching (celdas 81-82)

```python
size = 32 if _device == "cuda" else 16
mcqa_counterfactual_datasets = {
    "answer_position_targeted": mcqa_datasets_diag["answer_position_targeted"][:size],
    "answer_position_isolated": mcqa_datasets_diag["answer_position_isolated"][:size],
    "random": mcqa_datasets_diag["random"][: max(size, min(len(mcqa_datasets_diag["random"]), 2 * size))],
}
```

- Recorta a 32 pares (64 para `random`, el doble, porque se espera que filtre más y es más ruidoso).

```python
mcqa_filtered = filter_and_report(datasets=mcqa_counterfactual_datasets, pipeline=pipeline, causal_model=positional_causal_model,
                                  metric=mcqa_metric, batch_size=FILTER_BATCH_SIZE, label="mcqa", min_kept_warn=8)
_mcqa_sites = build_mcqa_token_positions(pipeline)
token_positions_mcqa = _mcqa_sites["positions"]
mcqa_pos_id_to_label = _mcqa_sites["label_map"]
layers_mcqa = default_layers(pipeline, _device)
```

- `mcqa_metric` (utils): pasa a mayúsculas el texto generado, busca letras sueltas con `\b([A-Z])\b` y compara la primera con la esperada; si no encuentra, compara el primer carácter.
- Salida guardada: 27/32, 28/32 y 54/64. Acá el LM sí hace bien la tarea.
- `build_mcqa_token_positions` (utils) toma `causalab.tasks.MCQA.token_positions.create_token_positions(pipeline)` y se queda con 7 sitios en orden: `correct_symbol`, `correct_symbol_period`, `symbol0`, `symbol0_period`, `symbol1`, `symbol1_period`, `last_token`.
- `create_token_positions` (`causalab/causalab/tasks/MCQA/token_positions.py`): `symbol{i}` = `{"type": "variable", "name": "symbol{i}"}` (todos los tokens de la letra); `symbol{i}_period` = `{"type": "index", "position": +1, "relative_to": {"variable": "symbol{i}"}}` (el token siguiente al último token de la letra, el punto). `correct_symbol` es **dinámico**: una lambda que mira el trace, busca qué `choice_i == color` y devuelve el spec de `symbol{i}`. Ojo con esto: como el índice se calcula por separado en base y en source, `correct_symbol` puede apuntar a `symbol0` en la base y a `symbol1` en el source (justo lo que pasa en todo par del dataset aislado, porque la posición cambia). Es decir, ese sitio parchea "el símbolo correcto del source" en "el símbolo correcto de la base", aunque estén en posiciones distintas del prompt.

```python
results_mcqa, targets_mcqa = run_patching_for_datasets(..., target_variable_groups=[("answer_position",)], metric=mcqa_metric, ...)
```

Mismo pipeline que en Parte 1; la IIA se mide contra la etiqueta de `patch(answer_position)`.

#### Step 4: heatmaps (celda 84)

Misma función, con `isolated` como "bueno" y `targeted`/`random` como comparación. Para razonar los resultados (no están en el texto guardado, solo como imagen):
- En `isolated` la etiqueta esperada es la **otra** letra de la base (ej. D). Un sitio que lleva "la letra correcta" del source pegaría E, que es el output original: IIA 0. Un sitio que lleva "la posición correcta" haría que la base lea su letra en la otra posición: IIA alta.
- En `targeted` y `random`, como el source tiene otras letras, un sitio que transporta la letra del source predice la letra del source, que casi siempre difiere de la etiqueta. Pero hay que mirar dónde se separan ambos datasets: diferencias en ΔIIA indican sitios cuya IIA depende del contraste.
- Las columnas `symbol{i}_period` y `last_token` son los lugares naturales donde mirar el puntero; la notebook no da la conclusión numérica.

### 2.4 Parte 3: Entity binding y dos modelos causales que compiten

#### Configuración y los dos modelos (celda 87)

```python
from causalab.tasks.entity_binding.config import create_sample_love_config
from causalab.tasks.entity_binding.causal_models import create_direct_causal_model, create_positional_causal_model
install_entity_binding_shims()
random.seed(0); np.random.seed(0)
```

- `install_entity_binding_shims` (utils) aplica dos parches de compatibilidad, porque esta tarea de causalab está marcada como **DEPRECATED** en su propio código (`config.py`, `causal_models.py`):
  1. `install_entity_binding_legacy_adapter`: las funciones `create_*_causal_model` llaman `CausalModel(variables, values, parents, mechanisms, id=...)`, la firma vieja. El adaptador reemplaza el símbolo `CausalModel` dentro del módulo `causalab.tasks.entity_binding.causal_models` por una función que convierte: variable sin padres con lista de valores → `input_var(values)`; resto → `Mechanism(parents, compute=fn)`; y llama al `CausalModel` nuevo. Funciona porque las funciones buscan `CausalModel` en el namespace del módulo en tiempo de llamada.
  2. `install_entity_binding_parser_patch`: el `PromptParser.parse_prompt` de `causalab/causalab/tasks/entity_binding/token_positions.py` calcula offsets de caracteres sin contar el `prompt_prefix` y asumiendo un separador de un espacio (`q_start = len(statement_text) + 1`). Con un prefijo, la entidad 0 caería en tokens del prefijo ("will", "question"...). El parche corre el parser original y desplaza los offsets de los segmentos del statement por `len(prefix)` y los de la pregunta por `len(prefix) + len(separator) - 1`. Solo se activa si el código fuente original no contiene `"statement_start"` (lo verifica con `inspect.getsource`); en esta copia de causalab no lo contiene, así que el parche se aplica.

```python
config = create_sample_love_config()
config.max_groups = 2
config.prompt_prefix = "We will ask a question about the following sentences.\n\n"
config.statement_question_separator = "\n\n"
config.prompt_suffix = "\nAnswer:"
config.fixed_query_indices = (0,)
```

- `create_sample_love_config` (`config.py`): 3 grupos máx., 2 entidades por grupo (persona, comida), pools de 6 nombres y 6 comidas, statement `"{entity_e0} loves {entity_e1}"`, preguntas `"What does {query_entity} love?"` (query índice 0, respuesta índice 1) y `"Who loves {query_entity}?"`.
- La notebook lo reduce a 2 grupos, agrega prefijo, separador y sufijo, y fija `fixed_query_indices=(0,)`: siempre se pregunta por la persona y se responde la comida.

```python
direct_model = create_direct_causal_model(config)
positional_model = create_positional_causal_model(config)
pipeline_eb = load_entity_binding_pipeline(device=_device)
```

Los dos modelos (`causalab/causalab/tasks/entity_binding/causal_models.py`):
- Comparten inputs: `entity_g{g}_e{e}`, `query_group`, `query_indices`, `answer_index`, `active_groups`, `entities_per_group`, `statement_template`; `question_template` depende de `(query_indices, answer_index)`; `raw_input` arma prefijo + statement + separador + pregunta + sufijo usando `TemplateProcessor`.
- **Direct**: `raw_output = entity_g{query_group}_e{answer_index}`. Usa el índice de grupo del input directamente como dirección.
- **Positional** (el "search model"): agrega dos intermedias.
  - `query_entity` = tupla de `entity_g{query_group}_e{i}` para cada `i` de `query_indices`, ej. `('Ann',)`.
  - `positional_query_group` = busca, entre los grupos activos **de este trace**, el grupo cuya entidad en `query_indices` coincide con `query_entity`. Si hay más de uno lanza `ValueError` (ambiguo); si no hay ninguno devuelve `None`.
  - `raw_output = entity_g{positional_query_group}_e{answer_index}`, o `"UNKNOWN"` si el grupo es `None`.

Por qué compiten: en inputs normales `positional_query_group == query_group`, así que ambos modelos dan la misma respuesta (celda 89 lo muestra: los dos responden "pie"). Solo bajo intervenciones divergen.

`load_entity_binding_pipeline` (utils) carga **GPT-2** (float32, `max_length=256`), porque el parser estructurado de posiciones fue desarrollado contra la tokenización de GPT-2.

#### Step 1: un input concreto (celda 89)

```python
input_sample = {"entity_g0_e0": "Pete", "entity_g0_e1": "jam", "entity_g1_e0": "Ann", "entity_g1_e1": "pie",
                "query_group": 1, "query_indices": (0,), "answer_index": 1, "active_groups": config.max_groups,
                "entities_per_group": 2, "statement_template": config.statement_template}
_direct_out = direct_model.new_trace(input_sample)
_positional_out = positional_model.new_trace(input_sample)
```

- Arma el input a mano y lo corre en los dos modelos. Salida guardada: prompt "Pete loves jam, and Ann loves pie. / What does Ann love? / Answer:", ambos dicen "pie", `query_entity=('Ann',)`, `positional_query_group=1`. La celda subraya que un ejemplo es **descriptivo, no discriminativo**.

#### Step 2: generadores (celda 91)

```python
_eb_gens = make_entity_binding_generators(config, positional_model)
query_swap_pair = _eb_gens["query_swap"]
random_pair = _eb_gens["random"]
```

`make_entity_binding_generators` (utils):
- `_clean_inputs`: convierte a dict, **borra** las variables calculadas (`raw_input`, `raw_output`, `query_entity`, `positional_query_group`), convierte `query_indices` a tupla y agrega `statement_template`. Así lo que queda son inputs puros.
- `_materialize`: `positional_model.new_trace(inputs)`. Todos los traces se calculan con el modelo **posicional**.
- `_sample_base`: llama `sample_valid_entity_binding_input(config)` hasta que `active_groups == max_groups` (los 2 grupos llenos). Esa función de causalab samplea `active_groups` en [2, max_groups], `query_group` dentro de los activos, usa `fixed_query_indices`, exige `answer_index ∉ query_indices` y que exista template, y samplea entidades distintas dentro de cada grupo y (por defecto) distintas por posición entre grupos, para que la búsqueda posicional no sea ambigua.
- `query_swap()`: toma una base, `qg = query_group`, elige otro grupo `swap_group`, y en el source **intercambia las entidades de índice de query** (la persona) entre `qg` y `swap_group`. Las comidas quedan en su lugar. Como `query_group` del source sigue siendo `qg`, la pregunta del source nombra a la persona que ahora está en `qg`.
- `random_pair()`: usa `causalab.tasks.entity_binding.counterfactuals.random_counterfactual(config)` (dos `sample_valid_entity_binding_input` independientes) y rechaza hasta que ambos tengan los dos grupos activos.

Salida guardada del sanity check `query_swap`: base "Tim loves bread, and Bob loves tea. What does Bob love?" (output tea), source "Bob loves bread, and Tim loves tea. What does Tim love?" (output tea). `query_entity` cambia (Bob → Tim), `positional_query_group` = 1 en ambos.

Hagamos el interchange a mano para entender:
- `patch(query_entity)`: en la base se fija `query_entity = ('Tim',)`. Se recalcula `positional_query_group` **buscando Tim en las entidades de la base** → grupo 0. `raw_output` = comida del grupo 0 de la base = **bread**.
- `patch(positional_query_group)`: se fija en 1 (el mismo valor) → **tea**, igual a la base.
- $Y_V$ = bread ≠ $Y_W$ = tea: par no confundido. Y otra vez, base y source tienen el mismo output (tea), pero el patch del target predice otra cosa.

El par `random` del sanity check tiene `query_entity` igual en base y source, así que ese par ni siquiera cumple el criterio 1.

#### Step 3: validar datasets (celda 93)

```python
N_ENTITY = 128
entity_datasets = {
    "query_entity_targeted": build_counterfactual_dataset(causal_model=positional_model, sampler=random_pair, n=N_ENTITY,
        target="query_entity", other="positional_query_group", require_distinguishability=False, require_other_fixed=False, max_attempts_per_example=400),
    "query_entity_isolated": build_counterfactual_dataset(causal_model=positional_model, sampler=query_swap_pair, n=N_ENTITY,
        target="query_entity", other="positional_query_group", require_distinguishability=True, require_other_fixed=True, max_attempts_per_example=400),
    "random": make_dataset(N_ENTITY, random_pair),
}
```

Mismo esquema; luego tablas, scorecard, barras, un par por dataset y el ejemplo confounding/no-confounding. Salida guardada: `targeted` Δquery_entity=100%, Δgroup=53.9%, patch(V) cambia el output 100%, patch(W) 53.9%; `isolated` 100/0/100/0; `random` 89.1/56.2/89.1/56.2. La fila de separación quedó truncada en el texto guardado.

Un detalle que conviene tener en cuenta al interpretar `targeted` y `random`: en `random_pair` el source es un input independiente, así que la persona que pregunta el source muchas veces **no aparece** en la base (con 6 nombres y 2 grupos, a menudo pasa). En ese caso, `patch(query_entity)` hace que la búsqueda en la base devuelva `None` y la etiqueta esperada es `"UNKNOWN"`. Eso cuenta como "cambia el output" (criterio 2 cumplido), pero el LM nunca va a generar "UNKNOWN", así que esos pares tiran la IIA hacia abajo en cualquier sitio. No lo cuantifica la notebook; lo deduzco del código de `search_for_query_entity` y `generate_raw_output`. El dataset `isolated` no tiene este problema porque `query_swap` siempre usa una persona que está en la base.

Celda 94, ejemplo mínimo: base "Pete loves jam, and Ann loves pie. What does Ann love?". Source confounding "...What does Pete love?" (cambian $V$ y $W$, ambos patches dan jam). Source no-confounding "Ann loves jam, and Pete loves pie. What does Pete love?" ($V$ cambia, $W=1$ fijo; $Y_V$ = jam, $Y_W$ = pie). Celda 95 conecta con Gur-Arieh et al.: `positional_query_group` ≈ mecanismo **posicional**, `query_entity` ≈ mecanismo **léxico**, y existe un tercero **reflexivo**.

**Cómo esto distingue al modelo direct del positional.** La notebook no lo ejecuta explícitamente, pero se sigue del código: en el modelo direct no existe `query_entity`; la salida depende de `query_group` directamente. En un par `query_swap`, el `query_group` de base y source es el mismo, así que cualquier interchange del modelo direct en `query_group` predice el output de la base (tea), mientras el modelo positional predice bread al parchear `query_entity`. Por eso un sitio con IIA alta para `query_entity` en el dataset aislado es evidencia a favor de una búsqueda por contenido (positional/léxica) y en contra de un acceso directo por índice en ese sitio.

#### Step 4: filtrado, sitios y patching (celdas 97-98)

```python
from causalab.tasks.entity_binding.experiment_config import get_checker
size_eb = 64
entity_counterfactual_datasets = {...[:size_eb]..., "random": entity_datasets["random"][: max(size_eb, min(len(...), 2 * size_eb))]}
entity_filtered = filter_and_report(datasets=entity_counterfactual_datasets, pipeline=pipeline_eb, causal_model=positional_model,
                                    metric=get_checker(), batch_size=min(FILTER_BATCH_SIZE, 64), label="entity", min_kept_warn=8)
```

- `get_checker()` (`causalab/causalab/tasks/entity_binding/experiment_config.py`) devuelve `checker(neural_output, causal_output)` = `causal_output in neural_output["string"] or neural_output["string"].strip() in causal_output`. Es una comparación por inclusión de substrings en ambas direcciones. Ojo: si el LM genera un string vacío (o solo espacios), `"" in causal_output` es `True` y cuenta como acierto. Con GPT-2 y 5 tokens nuevos es poco probable, pero es una debilidad del checker.
- Salida guardada: 13/64, 24/64 y 20/128. GPT-2 resuelve mal la tarea; `targeted` queda con solo 13 pares.

```python
eb_sites = build_entity_binding_token_positions(config, pipeline_eb)
token_positions_eb = eb_sites["positions"]
_eb_pos_id_to_label = eb_sites["label_map"]
diagnose_entity_binding_sites(filtered_datasets=entity_filtered, pipeline=pipeline_eb, config=config, site_panel=eb_sites)
layers_eb = default_layers(pipeline_eb, _device)
```

- `build_entity_binding_token_positions` (utils) arma 7 sitios: `g{g}_e{e}_stmt` (último token de cada entidad en el statement, 4 sitios), `both_e1_swap`, `query_entity_q` (último token de la persona en la pregunta) y `last_token`. Usa `get_entity_token_indices_structured` de causalab (parsea el prompt en segmentos con offsets de caracteres, tokeniza con `pipeline.load` y mapea caracteres a tokens decodificando incrementalmente). Si el resolver falla, cae a `last_token` y registra el error en `resolver_errors`.
- `both_e1_swap`: devuelve los tokens de la comida de los grupos 0 y 1; en el source (`is_original=False`) **invierte el orden**. Como pyvene empareja posiciones source→base en orden, el efecto es pegar la comida del grupo 1 del source en la posición de la comida del grupo 0 de la base y viceversa: intercambia las representaciones de los dos $e_1$.
- `diagnose_entity_binding_sites` resuelve todos los sitios en un ejemplo filtrado, imprime índice y token decodificado, y aborta si hubo fallback o si algún sitio cae dentro del span del prefijo. Salida guardada: `g0_e0_stmt -> 'ue'` (último token de "Sue"), `g0_e1_stmt -> ' tea'`, `g1_e0_stmt -> ' Tim'`, `g1_e1_stmt -> ' pie'`, `both_e1_swap -> [' tea', ' pie']`, `query_entity_q -> ' Sue'`, `last_token -> ':'`. Los índices (~237-255) son grandes porque el prompt se paddea a la izquierda hasta 256.
- GPT-2 tiene 12 capas → 13 filas × 7 sitios = 91 targets (coincide con la barra "91/91").

#### Step 5: heatmaps (celda 100)

Igual que antes, `query_entity_isolated` vs `targeted` y vs `random`. Qué razonar mirando la imagen:
- `query_entity_q` es el candidato natural para el mecanismo léxico: es donde el modelo lee qué persona se pregunta. En el par aislado, pegar la representación de "Tim" del source ahí debería llevar a la base a responder bread si el modelo busca por contenido.
- `last_token` mezcla todo (la notebook lo dice en la docstring del builder: conflaciona `query_entity` y `positional_query_group`).
- Las columnas del statement (`g*_e*_stmt`) prueban si la información relevante está en la representación de las entidades ligadas.
- `both_e1_swap` merece cuidado: en los pares aislados las comidas del source están en los mismos lugares que en la base, así que intercambiar las representaciones de las dos comidas hace que la base "vea" bread en el grupo de Bob, lo que también predice bread. Es decir, ese sitio puede dar IIA alta para `query_entity` por un camino que no es "representar la entidad preguntada". Esto es razonamiento mío a partir del código, no algo que afirme la notebook.

### 2.5 Wrap-up y lecturas (celdas 102-104)

La conclusión: el diseño del dataset contrafactual es parte del método, análogo a los minimal pairs del análisis conductual. Hay que cambiar lo que importa, fijar lo que se pueda y verificar de antemano qué conclusión permite el contraste. Lecturas: Mueller et al. 2025 (MIB, con la tarea MCQA y la separación posición/respuesta; DAS gana en causal variable localization y las SAEs no superan a neuronas individuales), Prakash et al. 2025 (mecanismo **lookback**: pointer → address → payload), Gur-Arieh et al. 2025 (mecanismos posicional, léxico y reflexivo). La próxima clase (10) suelta el supuesto de que cada variable vive en un sitio entero y busca subespacios con DAS.

---

## 3. Ejercicios

Esta notebook no tiene celdas `> FILL IN YOUR ANSWER HERE` ni soluciones marcadas `# TERRON`. Las "preguntas para tener en mente" de los Steps 2 son el ejercicio implícito; para cada par impreso:
1. ¿Cambia $V$? Mirar la línea `- V: base=... | source=... | CHANGED/same`.
2. ¿Cambia $W$? Ídem.
3. ¿El patch de $V$ cambia el output respecto de la base? Comparar `patch(V) -> ...` con `base output`.
4. ¿$Y_V \neq Y_W$? Comparar `patch(V)` con `patch(W)`.
5. ¿Qué hipótesis de localización toca el par? Si 4 falla, el par no discrimina entre $V$ y $W$.

---

## 4. `utils/lecture_09_utils.py` función por función

(El docstring del módulo dice "Lecture 08", pero es el de la clase 9.)

### Arranque del módulo
- **Búsqueda de la raíz** (líneas 14-24): recorre `cwd` y sus padres buscando un `pyproject.toml` junto a una carpeta `causalab`, o `causalab/pyproject.toml` + `causalab/causalab/`; en el segundo caso agrega `.../causalab` al `sys.path`. Así `import causalab` funciona sin instalar el paquete.
- **Parche `statement_conjunction_function`** (31-53): si `causalab.causal.causal_utils` no tiene esa función (la usa la tarea de entity binding para unir statements), define una: capitaliza el primer statement y los une con los delimitadores (`delimiters=[", ", "FILL", ", and ", "."]` → `"A, B, and C."`; con 2 statements usa `delimiters[-2]` = `", and "`). Con la config "love" produce "Pete loves jam, and Ann loves pie." El parche no es opcional: `causalab/causalab/tasks/entity_binding/templates.py` hace `from causalab.causal.causal_utils import statement_conjunction_function` al importarse, y en esta copia de causalab `causal_utils.py` no define esa función, así que sin el parche importar la tarea fallaría. Por eso hay que importar `utils.lecture_09_utils` antes que cualquier módulo de entity binding (la notebook lo hace).

### Grafos
- **`print_structure(model, highlight, ...)`**: arma un `nx.DiGraph` con las aristas padre→hijo del modelo, colorea los nodos según `highlight`, usa `model.print_pos` como layout (o `spring_layout`), dibuja y agrega leyenda. Si fallan los imports de plotting, la versión de respaldo imprime las aristas como texto.
- **`_wrap_graph_label`**: cambia `_` por espacios y hace wrap del texto.
- **`show_localization_view(model, target_vars, control_vars, title, output_var="raw_output")`**: define colores (target azul, control violeta, output naranja) y la leyenda, y llama `print_structure`.

### Básicos
- **`set_seed(seed)`**: `random.seed` + `np.random.seed`.
- **`make_dataset(n, sampler)`**: `[sampler() for _ in range(n)]`.
- **`report_distinguishability(model, dataset, vars1, vars2, label)`**: llama `can_distinguish_with_dataset` e imprime `count/len (proportion)`. No se usa en la notebook.
- **`proportion_true(dataset, var, which)`**: fracción de traces (base o source) donde una variable booleana es `True`; error si no es booleana. No se usa.

### Diagnósticos de pares
- **`_any_value_changed(base, cf, vars)`**: `True` si alguna variable difiere.
- **`_tuple_value(t, vars)`**: valor de una variable, o tupla si son varias.
- **`classify_pair_for_hypothesis(model, ex, vars1, vars2)`**: calcula $y_0$ (base), $y_1$ (patch vars1) y, si hay `vars2`, $y_2$; clasifica en "clean: distinguishable", "confounded: both match baseline" o "confounded: both same non-baseline" (o "effect"/"no effect" sin vars2).
- **`plot_boolean_rate`, `plot_boolean_contingency`, `plot_category_bar`, `plot_confound_breakdown`, `plot_clean_rate_by_base_values`**: gráficos de barras/heatmaps de seaborn sobre esas clasificaciones. No se usan en esta notebook.
- **`summarize_hypothesis_dataset(...)`**: aplica `classify_pair_for_hypothesis` a todo el dataset, imprime conteos por categoría, opcionalmente grafica y muestra ejemplos. No se usa.
- **`compute_testable_rate(model, dataset, vars1, vars2)`**: envoltorio de `can_distinguish_with_dataset` que devuelve `{"testable", "total", "percent"}`. Con `vars2=None` mide el criterio 2; con `vars2` mide el criterio 3. Es la base de todas las tablas.
- **`plot_testable_rate_by_dataset`**: barras con ese porcentaje por dataset. No se usa.

### Tablas y scorecard
- **`build_change_rate_table(datasets, variables)`**: por dataset, `100·mean(base[var] != cf[var])` para cada variable (columnas `Δvar`). Mantiene el orden de los datasets con un `Categorical`.
- **`display_change_rate_table`**: formatea como porcentajes, imprime el título y hace `display`. Devuelve el DataFrame numérico.
- **`plot_change_rate_heatmap`**: versión heatmap. No se usa.
- **`build_effect_rate_table(model, datasets, effect_groups)`**: por dataset y por grupo de variables, `compute_testable_rate(..., None)["percent"]` (criterio 2). `effect_groups` puede ser dict etiqueta→variables o lista.
- **`display_effect_rate_table`** / **`plot_effect_rate_heatmap`**: análogo.
- **`_change_rate_for_vars(dataset, vars_)`**: % de pares donde cambia alguna de las variables.
- **`build_localization_scorecard(model, datasets, target_vars, control_vars)`**: por dataset, columnas `Δ target`, `patch(target) changes output`, `Δ control`, `patch(control) changes output` y `separates target vs control` (criterio 3).
- **`display_localization_scorecard`** / **`plot_localization_scorecard`**: formateo/heatmap.
- **`plot_dataset_criteria_bars(...)`**: reutiliza el scorecard, lo pasa a formato largo con `melt`, renombra las columnas a "(1) Δ V", "(2) patch(V) → ΔY", "(1') ... (diag.)", "(2') ... (diag.)", "(3) Y_V(b,s) ≠ Y_W(b,s)", y dibuja barras horizontales agrupadas por dataset con el porcentaje anotado. Devuelve el DataFrame.
- **`_short_text`, `preview_counterfactual_pairs`**: tabla compacta de pares. No se usa.

### Patching
- **`default_layers(pipeline, device)`**: `[-1] + range(num_hidden_layers)`.
- **`run_patching_for_datasets(...)`**: explicada en §2.2 Step 3. Si no se da `output_root`, crea un directorio temporal; guarda cada dataset como JSON y llama `run_interchange_score_heatmap`. Devuelve `(results, targets)` con `results[name][(var,)][(layer, pos_id)] = IIA`.
- **`scores_to_matrix(scores_dict, targets)`**: usa `detect_component_type_from_targets` y `extract_grid_dimensions_from_targets` (causalab) para obtener la lista ordenada de capas y posiciones, y llena una matriz `(n_layers, n_positions)` con NaN donde falte.
- **`_plot_trace_style_matrix(ax, mat, ...)`**: `imshow` con anotaciones; texto blanco si el valor supera el punto medio de la escala, negrita si |valor| > 0.5.
- **`plot_iia_triplet(results, targets, var_name, dataset_good, dataset_bad, ...)`**: arma las dos matrices, las da vuelta (`flipud`) para que la última capa quede arriba, calcula `diff = good - bad`, y dibuja tres paneles (bad, good, diff con colormap `RdBu_r` simétrico).
- **`plot_iia_target_control_random(...)`**: llama `plot_iia_triplet` con (isolated, targeted) y, si hay, con (isolated, random).

### Impresión de pares
- **`print_counterfactual_pair_text(example, variables, title)`**: imprime inputs, valores por variable con CHANGED/same y outputs.
- **`_patch_outputs(model, ex, target, other)`**: `run_interchange` en target y en other, devuelve los dos `raw_output` sin espacios.
- **`print_one_pair_per_dataset(...)`**: primer par de cada dataset + los dos patches.
- **`print_confounding_and_nonconfounding(...)`**: busca un par confundido ($Y_V = Y_W$) en una fuente y uno no confundido en otra (con filtros opcionales) e imprime ambos con veredicto.

### Builder de datasets
- **`_satisfies_criteria(...)`** y **`build_counterfactual_dataset(...)`**: explicados en §2.2 Step 1. Orden de chequeos: (1), (2), `other` fijo, (3). El presupuesto de intentos es `max_attempts_per_example · n`.

### Carga de pipelines y filtrado
- **`load_activation_patching_pipeline(...)`**: lista de candidatos, `LMPipeline(..., dtype=bfloat16 en cuda / float32 en cpu)`, primer éxito. Importa `torch` adentro.
- **`load_entity_binding_pipeline(...)`**: `LMPipeline("gpt2", dtype=float32, max_length=256)`.
- **`filter_and_report(...)`**: `filter_dataset` por dataset, reporta, falla si queda vacío.

### Métricas y posiciones
- **`hier_metric`**: primera palabra en minúsculas == "yes"/"no".
- **`mcqa_metric`**: primera letra suelta en mayúsculas == letra esperada.
- **`build_hier_token_positions(pipeline, prompt_template)`**: specs declarativas de causalab para el primer token de cada número y el último token; `label_map` id→etiqueta con saltos de línea.
- **`build_mcqa_token_positions(pipeline)`**: los 7 sitios de `create_token_positions` en orden fijo con etiquetas cortas.

### Entity binding
- **`install_entity_binding_legacy_adapter`**, **`install_entity_binding_parser_patch`**, **`install_entity_binding_shims`**: explicados en §2.4.
- **`make_entity_binding_generators(config, positional_model)`**: explicado en §2.4 Step 2.
- **`build_entity_binding_token_positions(config, pipeline)`**: §2.4 Step 4. Internamente: `_normalize` pasa el trace a dict asegurando `raw_input` y `query_indices` como tupla; `_resolve` llama `get_entity_token_indices_structured` y registra errores; `_make_entity_indexer` devuelve el último token de la entidad o cae a `get_last_token_index`; `_query_entity_indexer` hace lo mismo en la región "question" para el grupo y el índice preguntados; `_both_e1_indexer` invierte en el source. Devuelve `positions`, `positions_dict`, `position_order`, `label_map` (id→etiqueta) y `resolver_errors`.
- **`diagnose_entity_binding_sites(...)`**: §2.4 Step 4. Para detectar el prefijo: tokeniza `config.prompt_prefix` sin tokens especiales, busca dónde termina el padding (último `pad_token_id` desde la izquierda) y define el span del prefijo como `[content_start, content_start + len(prefix_ids))`; si algún sitio cae ahí, `RuntimeError`.

---

## 5. Ojo / errores comunes / dudas

- **IIA alta no implica localización.** Solo cuenta si el dataset ya cumple los tres criterios a nivel del modelo causal. Mirar primero la fila $Y_V \neq Y_W$.
- **El criterio 2 evita un falso positivo trivial**: si el patch de $V$ no cambia el output esperado, un sitio inútil (que no altera nada) saca IIA perfecta.
- **`can_distinguish_with_dataset` ignora los argumentos de modelo** (usa los mecanismos copiados en los traces). Sirve para comparar variables del mismo modelo, no dos modelos distintos.
- **Argumentos confusos en `plot_iia_target_control_random`**: `dataset_targeted` es el aislado y `dataset_control` el targeted.
- **"random" de Parte 1 no es uniforme sobre hojas**: balancea `left_eq`/`right_eq` al 50%.
- **MCQA tiene 2 opciones, no 4**, aunque el markdown diga A/B/C/D.
- **En MCQA, el criterio 3 casi viene gratis** porque las letras se samplean al azar; lo distintivo de `isolated` es fijar `answer` y que base y source tengan el mismo output.
- **`correct_symbol` es un sitio dinámico** que puede caer en posiciones distintas en base y source.
- **Pocos ejemplos tras filtrar** (11 en `left_eq_isolated`, 13 en `query_entity_targeted`): las diferencias de ΔIIA pueden ser ruido.
- **"UNKNOWN" como etiqueta** en `targeted`/`random` de entity binding cuando la persona del source no está en la base: el LM no puede acertarla.
- **Checker de entity binding permisivo**: inclusión de substrings en ambos sentidos; un string vacío cuenta como acierto.
- **La tarea de entity binding está deprecada en causalab** y requiere los shims; si actualizás causalab el parche del parser se desactiva solo (busca `"statement_start"`), pero el adaptador del constructor se aplica siempre.
- **Capa −1 = embeddings** (block_input de la capa 0), no la última capa.
- **Los índices de token en base y source se calculan por separado**, así que un mismo "sitio" puede ser distinto número de token en cada prompt; para `both_e1_swap` el orden se invierte a propósito.

---

## 6. Resumen de una línea por sección

- **Conceptos**: IIA mide acuerdo entre la red intervenida y el algoritmo intervenido; solo es evidencia relativa al contraste del dataset.
- **Criterios**: (1) $V$ cambia, (2) su patch cambia el output, (3) su patch predice distinto que el de $W$.
- **Roles**: `random` (nada), `targeted` (1+2), `isolated` (1+2+3 con $W$ fijo); comparar `isolated` contra `targeted`.
- **Parte 1**: en hierarchical equality, `targeted` separa solo cuando `right_eq` no cambia (45%); `isolated` invierte `left_eq` y fija `right_eq` (100%).
- **Parte 2**: MCQA es pointer-value; parchear `answer_position` copia la dirección y la base lee su propia letra; `same_symbol_different_position` mueve la posición sin cambiar la letra.
- **Parte 3**: entity binding tiene un modelo direct (índice de grupo) y uno positional (busca la entidad); `query_swap` cambia la persona preguntada sin cambiar el grupo y separa `query_entity` de `positional_query_group`.
- **Patching**: `run_patching_for_datasets` → un `ResidualStream` por (capa, posición), reemplazo del vector completo vía pyvene, etiquetas con `label_counterfactual_data`, IIA por celda.
- **Wrap-up**: diseñar el contrafactual es parte del método, como los minimal pairs.
