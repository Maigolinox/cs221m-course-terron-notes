# Lecture 10: Abstracción bajo traducción

Esta notebook es la continuación de la Lecture 8. Allá vimos **abstracción constructiva**: una variable de alto nivel se alinea con un *conjunto* de variables de bajo nivel (neuronas). Acá se agrega un paso previo: antes de agrupar, se permite **traducir** el modelo de bajo nivel a un sistema de coordenadas distinto pero causalmente equivalente. Eso es la **abstracción bajo traducción** (*abstraction-under-translation*). La pregunta que responde es: ¿qué hacemos cuando la variable de alto nivel no vive en un conjunto de neuronas, sino en una dirección o subespacio que mezcla muchas neuronas? La respuesta: buscar una traducción (en redes, una rotación) en la que sí quede aislada, y después aplicar abstracción constructiva sobre las coordenadas nuevas.

Tiene dos partes:

1. **Traducción entre modelos causales** con un ejemplo exacto: dos circuitos $\mathcal{M}$ y $\mathcal{M}^*$ para igualdad jerárquica binaria, con distinto "corte" de variables internas, y una biyección $\tau$ entre ellos. Se verifica que $\tau$ conmuta con las corridas normales y con intervenciones, y se ve que una hard intervention en un modelo corresponde a una intervención más compleja en el otro.
2. **DAS (Distributed Alignment Search) como descubrimiento de la traducción**: se entrena un MLP en la tarea, y se aprende con gradiente una rotación ortogonal de la capa oculta tal que intercambiar subespacios rotados reproduce las interchange interventions de $E_1$ y $E_2$.

Archivos de `causalab` que usa: `causalab/causalab/causal/causal_model.py`, `causalab/causalab/causal/trace.py`, `causalab/causalab/causal/causal_viz.py`. La parte de DAS **no** usa `causalab.neural` (ni `featurizers` ni `pyvene_core`): usa `pyvene` directamente. Igual explico al final cómo `causalab` implementa la misma idea con `SubspaceFeaturizer` (`causalab/causalab/neural/featurizers.py`), porque es lo que se usa en las notebooks siguientes y es literalmente la fórmula de la notebook.

---

## Ideas clave / teoría

### Por qué no alcanza la abstracción constructiva

En la Lecture 8 la red estaba construida a mano para que $E_1$ viviera en `H1_1..H1_4` y $E_2$ en `H1_5..H1_8`. En una red entrenada no hay ninguna razón para que eso pase. El grafo computacional de una red está escrito en la base estándar (una coordenada por neurona), pero la siguiente capa lee la activación $h$ a través de una matriz $W h$: si roto $h$ con una matriz ortogonal $Q$ y reemplazo $W$ por $W Q^\top$, la red calcula exactamente lo mismo. O sea, la base de neuronas no tiene nada de privilegiado desde el punto de vista de quien lee la capa, y la información de una variable puede quedar repartida en una dirección arbitraria (**representación distribuida**). Si $E_1$ vive en una dirección que mezcla las 16 neuronas, no hay ningún subconjunto de neuronas que la aísle, y toda alineación constructiva sobre neuronas va a fallar.

La notebook cita el número de Geiger et al. (2023): en este mismo experimento (MLP entrenado en igualdad jerárquica), la mejor alineación sobre neuronas individuales llegaba a IIA 0.73, mientras que DAS encontró una alineación distribuida con IIA 1.00. Ojo: ese 0.73 es del paper; la notebook no corre el baseline por neuronas.

### Traducción

Una **traducción** $\tau$ es una **biyección** sobre los settings completos de las variables: cambia de coordenadas sin perder ni comprimir información. Se le pide que sea una **transformación exacta**:

$$\tau(\mathsf{Run}(\mathcal{L}_i)) = \mathsf{Run}(\mathcal{H}_{\omega(i)})$$

- $i$: una intervención (o *interventional*) sobre el modelo de bajo nivel $\mathcal{L}$; la corrida normal es el caso de la intervención vacía.
- $\mathsf{Run}(\mathcal{L}_i)$: el setting completo que resulta de correr $\mathcal{L}$ intervenido.
- $\tau(\cdot)$: traducido a las variables de $\mathcal{H}$.
- $\omega(i)$: la intervención correspondiente en $\mathcal{H}$.
- La igualdad dice: intervenir abajo y traducir = traducir la intervención e intervenir arriba. Es la misma ecuación de la Lecture 8; la diferencia es que acá $\tau$ es biyectiva (un cambio de coordenadas) y no un agrupamiento que pierde información.

Diferencia con la Lecture 8: allá se iba de bajo a alto nivel **agrupando** (clustering) variables. Acá se **re-expresa** el modelo en otras variables, del mismo tamaño.

### Interventionals "pulled back"

Lo interesante: $\omega$ no tiene por qué mandar hard interventions a hard interventions. Una hard intervention simple en el modelo traducido ("fijá $E_1$") puede corresponder, en el original, a cambiar **varios mecanismos** a la vez. A esa intervención del modelo original se le llama la intervención **pulled back** (tirada hacia atrás a través de $\tau$). En general se habla de *interventionals*: transformaciones de mecanismos, no solo fijar valores.

### Abstracción bajo traducción

$\mathcal{H}$ es abstracción bajo traducción de $\mathcal{L}$ si existe una traducción $\tau(\mathcal{L})$ y $\mathcal{H}$ es abstracción **constructiva** de $\tau(\mathcal{L})$:

$$\mathcal{L} \xrightarrow{\text{traducción } \tau} \tau(\mathcal{L}) \xrightarrow{\text{abstracción constructiva}} \mathcal{H}$$

### DAS como traducción aprendida

DAS aprende una matriz ortogonal $\mathbf{R} \in \mathbb{R}^{k \times d}$ (filas ortonormales; $d$ = dimensión de la capa, $k$ = dimensión del subespacio de la variable). Con ella se define una biyección sobre la activación $h \in \mathbb{R}^d$:

$$f = (\mathbf{R}h,\; h - \mathbf{R}^\top(\mathbf{R}h))$$

- $f_1 = \mathbf{R}h \in \mathbb{R}^k$: las coordenadas de $h$ en el subespacio aprendido (lo que "es" la variable de alto nivel).
- $f_2 = h - \mathbf{R}^\top\mathbf{R}h \in \mathbb{R}^d$: el residuo, la parte de $h$ ortogonal al subespacio. $\mathbf{R}^\top\mathbf{R}$ es el proyector ortogonal sobre el subespacio (porque $\mathbf{R}\mathbf{R}^\top = I_k$).

Inversa:

$$h = \mathbf{R}^\top f_1 + f_2$$

Se comprueba sustituyendo: $\mathbf{R}^\top\mathbf{R}h + h - \mathbf{R}^\top\mathbf{R}h = h$. Como no se pierde nada, es una traducción en el sentido de arriba: el resto de la red queda igual y solo esta capa se re-expresa en coordenadas $(f_1, f_2)$.

La **interchange intervention en el espacio rotado** es una hard interchange sobre $f_1$: tomar $f_1$ del source y $f_2$ del base, y volver:

$$h_{\text{nuevo}} = \mathbf{R}^\top \mathbf{R} h_s + (h_b - \mathbf{R}^\top\mathbf{R}h_b) = h_b + \mathbf{R}^\top\mathbf{R}(h_s - h_b)$$

En la base de neuronas esto cambia, en general, **todas** las neuronas de la capa a la vez: es el análogo neuronal de la intervención pulled back de la Parte 1 (simple arriba, compleja abajo).

**Qué se entrena**: solo $\mathbf{R}$, con la red congelada. Pérdida = cross-entropy entre el output de la red intervenida y la etiqueta contrafactual que predice el modelo causal. La IIA en datos nuevos mide qué tan bien se cumple la ecuación de abstracción en el espacio traducido.

---

## Recorrido por la notebook

### Setup (celdas 3-5)

Igual que la Lecture 8: instala dependencias (`requirements/10_causal_abstraction_ii.txt` en Colab o `uv sync` local), importa `CausalModel`, `Mechanism`, `input_var`, `print_structure`, `print_setting` y fija semillas de `random` y `numpy` en 42. Estas clases están explicadas en detalle en los apuntes de la Lecture 8 (`apuntes/curso/08_causal_abstraction_i.md`); acá solo repito lo necesario.

### 1. Recap y ejemplo

#### Celda 8: helpers

```python
def eq(a, b):
    return int(a == b)

binary = [0, 1]

def project_trace(trace, variables):
    return {var: trace[var] for var in variables}

def all_binary_inputs():
    for bits in itertools.product(binary, repeat=4):
        yield {"A1": bits[0], "A2": bits[1], "A3": bits[2], "A4": bits[3]}
```

- `eq`: igualdad que devuelve `1`/`0` en lugar de `True`/`False`. Se usan enteros para que los valores sean directamente numéricos (compatibles con el MLP después).
- `binary`: el dominio de todas las variables.
- `project_trace`: arma un dict con solo algunas variables de un trace (para comparar sin `raw_input`/`raw_output`). Funciona tanto con un `CausalTrace` (usa `__getitem__`) como con un dict.
- `all_binary_inputs`: generador de los $2^4 = 16$ inputs posibles.

### 2. El algoritmo $\mathcal{M}$

#### Celda 10

```python
M_values = {"A1": binary, ..., "E1": binary, "E2": binary, "C": binary,
            "raw_input": [None], "raw_output": binary}

M_mechanisms = {
    "A1": input_var(binary), ...,
    "E1": Mechanism(parents=["A1", "A2"], compute=lambda t: eq(t["A1"], t["A2"])),
    "E2": Mechanism(parents=["A3", "A4"], compute=lambda t: eq(t["A3"], t["A4"])),
    "C": Mechanism(parents=["E1", "E2"], compute=lambda t: eq(t["E1"], t["E2"])),
    "raw_input": Mechanism(parents=["A1", "A2", "A3", "A4"], compute=lambda t: (t["A1"], t["A2"], t["A3"], t["A4"])),
    "raw_output": Mechanism(parents=["C"], compute=lambda t: t["C"]),
}
M_model = CausalModel(M_mechanisms, M_values, print_pos=M_pos, id="M")
```

Es el mismo algoritmo de la Lecture 8 pero con inputs binarios:

- $A_1..A_4 \in \{0,1\}$ (inputs: `input_var` crea un `Mechanism` sin padres, `causalab/causalab/causal/trace.py`).
- $E_1 = (A_1 = A_2)$, $E_2 = (A_3 = A_4)$, $C = (E_1 = E_2)$.
- `raw_input`/`raw_output`: obligatorias en `CausalModel` (hay un `assert` en el constructor, `causal_model.py` líneas 59-64).

Recordatorio de lo que hace el constructor de `CausalModel` (`causal_model.py`, líneas 32-167): deriva `parents` y `children` de los mecanismos, define `inputs` = variables sin padres, `outputs` = variables que no son padre de nadie (hojas), calcula `timesteps` (largo del camino más largo desde un input, con las hojas forzadas al último paso), valida que padres tengan timestep menor que hijos y ordena `variables` por timestep.

Salida guardada: `Variables: ['A1','A2','A3','A4','E1','E2','C','raw_input','raw_output']`, `Inputs: ['A1','A2','A3','A4']`, `Outputs: ['raw_input','raw_output']` (de nuevo `raw_input` es "output" por ser hoja).

#### Celdas 11-14

- `print_structure(M_model)` dibuja el DAG (`causal_viz.py`, línea 644: un `networkx.DiGraph` con aristas padre→hijo, posicionado con `print_pos`). Timesteps guardados: `E1, E2` en 1, `C` en 2, `raw_input` y `raw_output` en 3.
- `M_model.new_trace({"A1": 1, "A2": 1, "A3": 0, "A4": 0})`: crea un `CausalTrace` con copia profunda de los mecanismos, setea los inputs y propaga hacia adelante (`_recompute_descendants` calcula cada hijo cuando todos sus padres tienen valor). Salida guardada: `Input: (1, 1, 0, 0)`, `E1 = 1`, `E2 = 1`, `C = 1`.
- `print_setting(M_model, trace_M.to_dict())`: el grafo con el valor de cada nodo en la etiqueta.

### 3. Otro corte: el circuito $\mathcal{M}^*$

#### Celda 17

```python
M_star_mechanisms = {
    "A1": input_var(binary), ...,
    "D1": Mechanism(parents=["A1", "A2"], compute=lambda t: eq(t["A1"], t["A2"])),
    "D2": Mechanism(
        parents=["A1", "A2", "A3", "A4"],
        compute=lambda t: eq(eq(t["A1"], t["A2"]), eq(t["A3"], t["A4"])),
    ),
    "C": Mechanism(parents=["D2"], compute=lambda t: t["D2"]),
    "raw_input": ..., "raw_output": ...,
}
M_star_model = CausalModel(M_star_mechanisms, M_star_values, print_pos=M_star_pos, id="M_star")
```

- $D_1 = (A_1 = A_2)$: igual que $E_1$.
- $D_2 = ((A_1 = A_2) = (A_3 = A_4))$: depende de los **cuatro** inputs y ya contiene la respuesta final. Es una variable "entrelazada": mezcla la info del par 1 con la del par 2.
- $C = D_2$: el output solo copia $D_2$.
- Notar que $D_1$ **no tiene hijos**: nadie la usa. El output no depende de ella.

Salida guardada: `Variables: ['A1','A2','A3','A4','D2','C','D1','raw_input','raw_output']`. El orden raro tiene explicación en el constructor: como `D1` no es padre de nadie, `CausalModel` la clasifica como **output** (hoja) y la fuerza al último timestep (3), junto con `raw_input` y `raw_output`. `D2` queda en 1 y `C` en 2. Como el `sort` de Python es estable, dentro del timestep 3 se mantiene el orden de inserción (`D1`, `raw_input`, `raw_output`). La notebook no lo comenta, pero es lo que explica la salida.

#### Celdas 20-21: mismo comportamiento, distinto interior

```python
sample_input = {"A1": 1, "A2": 0, "A3": 1, "A4": 0}
trace_M_sample = M_model.new_trace(sample_input)
trace_M_star_sample = M_star_model.new_trace(sample_input)
print("M trace     :", project_trace(trace_M_sample, [...E1, E2, C]))
print("M* trace    :", project_trace(trace_M_star_sample, [...D1, D2, C]))
```

Salida guardada para (1,0,1,0): en $\mathcal{M}$, `E1=0, E2=0, C=1`; en $\mathcal{M}^*$, `D1=0, D2=1, C=1`. Mismo output, pero $D_2 \neq E_2$: no juegan el mismo rol.

```python
for setting in all_binary_inputs():
    t_M = M_model.new_trace(setting)
    t_M_star = M_star_model.new_trace(setting)
    rows.append((tuple(...), (t_M["E1"], t_M["E2"], t_M["C"]), (t_M_star["D1"], t_M_star["D2"], t_M_star["C"])))
    if t_M["C"] == t_M_star["C"]:
        matches += 1
```

Recorre los 16 inputs, guarda las tripletas internas y cuenta coincidencias de output. Salida guardada: `16/16`. Las primeras filas muestran que `E1 = D1` siempre, mientras que `E2` y `D2` difieren cuando $E_1 = 0$ (por ejemplo input (0,1,0,0): `M=(0,1,0)`, `M*=(0,0,0)`).

Moraleja: comportamiento input-output idéntico no implica estructura interna idéntica. Si mirara solo outputs no podría distinguir estos dos modelos; con intervenciones sí.

### 4. Traducción como cambio de variables

La biyección:

$$(a_1,a_2,a_3,a_4,d_1,d_2,c) \overset{\tau}{\mapsto} (a_1,a_2,a_3,a_4,d_1,(d_1 = d_2),c)$$

O sea $E_1 = D_1$, $E_2 = (D_1 = D_2)$, $C = C$.

Por qué funciona: $D_2 = (E_1 = E_2)$ con $E_1 = D_1$. Para binarios, "$x = (x = y)$" vale exactamente $y$ (si $x = 1$, $(1 = y) = y$; si $x = 0$, $(0 = (0 = y))$ = negar dos veces = $y$). Así que $(D_1 = D_2) = (E_1 = (E_1 = E_2)) = E_2$. La operación "$= $" sobre bits es XNOR y es su propia inversa una vez fijado un argumento; por eso la inversa tiene la misma forma: $D_2 = (E_1 = E_2)$.

#### Celda 23

```python
def tau_star_to_M(trace_star):
    return {
        "A1": trace_star["A1"], ..., "A4": trace_star["A4"],
        "E1": trace_star["D1"],
        "E2": eq(trace_star["D1"], trace_star["D2"]),
        "C": trace_star["C"],
    }

def tau_inverse_from_M(setting_M):
    return {
        ..., "D1": setting_M["E1"],
        "D2": eq(setting_M["E1"], setting_M["E2"]),
        "C": setting_M["C"],
    }
```

- `tau_star_to_M`: implementa $\tau$. Recibe un setting de $\mathcal{M}^*$ (un `CausalTrace` o un dict) y devuelve uno de $\mathcal{M}$. Los inputs y `C` pasan igual.
- `tau_inverse_from_M`: implementa $\tau^{-1}$.

#### Celda 25: chequear que es biyección

```python
all_M_star_states = [ {...} for a1, a2, a3, a4, d1, d2, c in itertools.product(binary, repeat=7) ]
all_M_states = [ {...} for ... in itertools.product(binary, repeat=7) ]

left_inverse_ok = all(tau_inverse_from_M(tau_star_to_M(s)) == s for s in all_M_star_states)
right_inverse_ok = all(tau_star_to_M(tau_inverse_from_M(s)) == s for s in all_M_states)
```

- Genera los $2^7 = 128$ settings posibles de cada espacio, **incluidos los que nunca salen de una corrida normal** (por ejemplo, $C \neq D_2$). Una traducción se define sobre todo el espacio de settings, no solo sobre los alcanzables, porque bajo intervenciones se pueden llegar a estados "imposibles" en la corrida normal.
- Verifica $\tau^{-1} \circ \tau = \text{id}$ y $\tau \circ \tau^{-1} = \text{id}$. Las dos cosas juntas ⇒ biyección.

Salida guardada: `True` y `True`.

#### Celdas 27-28: la traducción conmuta con las corridas normales

```python
for setting in all_binary_inputs():
    t_M = M_model.new_trace(setting)
    t_M_star = M_star_model.new_trace(setting)
    translated = tau_star_to_M(t_M_star)
    target = project_trace(t_M, ["A1", "A2", "A3", "A4", "E1", "E2", "C"])
    if translated == target:
        translated_matches += 1
```

Verifica $\tau(\mathsf{Run}(\mathcal{M}^*(i))) = \mathsf{Run}(\mathcal{M}(i))$ para los 16 inputs (la ecuación con intervención vacía). Notar que acá el input se pasa tal cual a los dos modelos: los inputs se traducen con la identidad. Salida guardada: `16/16`.

La celda 28 lo muestra para (1,0,1,1): el trace de $\mathcal{M}^*$ es `D1=0, D2=0, C=0`; traducido da `E1=0, E2=1, C=0`, idéntico al trace de $\mathcal{M}$.

La diferencia con la Lecture 8 está en la celda 29: los modelos no se relacionan agrupando variables en macrovariables, sino usando **coordenadas internas distintas**.

### 5. Hacer "pull back" de una hard intervention

Pregunta: si en $\mathcal{M}$ hago la hard intervention $E_1 \leftarrow e$, ¿qué operación en $\mathcal{M}^*$ hace lo mismo?

#### Celda 31

```python
def make_M_do_E1(value):
    mechanisms = {
        ...,
        "E1": Mechanism(parents=["A1"], compute=lambda t, v=value: v),
        "E2": Mechanism(parents=["A3", "A4"], compute=lambda t: eq(t["A3"], t["A4"])),
        "C": Mechanism(parents=["E1", "E2"], compute=lambda t: eq(t["E1"], t["E2"])),
        ...
    }
    return CausalModel(mechanisms, M_values, print_pos=M_pos, id=f"M_do_E1_{value}")
```

- Construye un **modelo nuevo** $\mathcal{M}_{E_1 \leftarrow v}$ donde el mecanismo de `E1` es la constante `v`.
- `parents=["A1"]` es un **padre trivial**: el `compute` no lo lee. Si se pusiera `parents=[]`, `CausalModel` trataría a `E1` como input y `CausalTrace` nunca la calcularía (solo se calculan hijos de variables seteadas), así que `C` nunca tendría todos sus padres. Mismo truco que las neuronas constantes de la Lecture 8.
- `v=value` en el lambda congela el valor.
- En $\mathcal{M}$ esto es equivalente a `trace.intervene("E1", v)` (que reemplaza el mecanismo por una constante sin padres, `trace.py`), pero acá se arma como modelo completo para que sea paralelo a lo que viene, que no se puede expresar con `intervene`.

```python
def make_M_star_pullback_E1(value):
    mechanisms = {
        ...,
        "D1": Mechanism(parents=["A1"], compute=lambda t, v=value: v),
        "D2": Mechanism(
            parents=["A3", "A4"],
            compute=lambda t, v=value: eq(v, eq(t["A3"], t["A4"])),
        ),
        "C": Mechanism(parents=["D2"], compute=lambda t: t["D2"]),
        ...
    }
    return CausalModel(mechanisms, M_star_values, print_pos=M_star_pos, id=f"M_star_pullback_E1_{value}")
```

El interventional pulled back en $\mathcal{M}^*$ cambia **dos** mecanismos:

- $D_1 := e$ (constante, con padre trivial `A1`).
- $D_2 := (e = (A_3 = A_4))$: ya no depende de $A_1, A_2$, solo de $A_3, A_4$, y tiene $e$ "horneado" adentro.

Por qué hace falta tocar $D_2$: en $\mathcal{M}^*$ la info de $E_1$ está en $D_1$ **y también** adentro de $D_2$ (que calcula $(A_1 = A_2)$ por su cuenta). Si solo fijara $D_1$, $D_2$ seguiría usando el $(A_1 = A_2)$ real, y la traducción $E_2 = (D_1 = D_2)$ daría basura. Para que, después de traducir, quede $E_1 = e$, $E_2 = (A_3 = A_4)$ y $C = (e = (A_3 = A_4))$, hay que reescribir $D_2$.

Esto es el punto central de la sección: una hard intervention en el modelo traducido se convierte en un **interventional multi-mecanismo** en el original. No es una hard intervention: $D_2$ no se fija a un valor, se le cambia la función.

#### Celdas 33-36

- `print_structure(M_star_pullback_1)`: se ve que `D1` ahora cuelga solo de `A1` (el padre trivial; el dibujo muestra la arista aunque el valor sea constante) y `D2` depende solo de `A3, A4`.
- Celda 35: para el input (0,1,1,1) compara el trace de $\mathcal{M}_{E_1 \leftarrow 1}$, el del pullback en $\mathcal{M}^*$ y el pullback traducido. Salidas guardadas: alto nivel `E1=1, E2=1, C=1`; pullback `D1=1, D2=1, C=1`; traducido `E1=1, E2=1, C=1`. Coinciden.
- Celda 36: `print_setting` del pullback con sus valores.

#### Celda 38: verificar la conmutación para toda la familia

$$\tau(\mathsf{Run}(\mathcal{M}^*_{i_{E_1 \leftarrow b}}(x))) = \mathsf{Run}(\mathcal{M}_{E_1 \leftarrow b}(x)) \quad \forall x$$

con $\omega_\tau(i_{E_1 \leftarrow b}) = (E_1 \leftarrow b)$.

```python
def check_pullback_E1(value):
    M_do = make_M_do_E1(value)
    M_star_pull = make_M_star_pullback_E1(value)
    for setting in all_binary_inputs():
        low = M_star_pull.new_trace(setting)
        high = M_do.new_trace(setting)
        translated_low = tau_star_to_M(low)
        high_trace = project_trace(high, ["A1", "A2", "A3", "A4", "E1", "E2", "C"])
        if translated_low != high_trace:
            return False, setting, translated_low, high_trace
    return True, None, None, None

for b in binary:
    ok, ... = check_pullback_E1(b)
```

- Para cada $b \in \{0,1\}$ construye los dos modelos intervenidos.
- Para cada uno de los 16 inputs: corre ambos, traduce el de abajo, compara con el de arriba. Si falla, devuelve el contraejemplo.

Salida guardada: `commutation holds: True` para $b = 0$ y $b = 1$.

Verificación a mano para entenderlo: en el pullback, $D_1 = b$, $D_2 = (b = E)$ con $E = (A_3 = A_4)$. Traduciendo: $E_1 = b$, $E_2 = (b = (b = E)) = E$, $C = D_2 = (b = E)$. Arriba: $E_1 = b$, $E_2 = E$, $C = (b = E)$. Iguales.

### 6. De traducción a abstracción bajo traducción

Solo texto: $\mathcal{H}$ es abstracción bajo traducción de $\mathcal{L}$ si hay una traducción $\tau(\mathcal{L})$ de la que $\mathcal{H}$ es abstracción constructiva. En la Parte 1 se aisló la primera flecha (la traducción); en el ejemplo, una vez traducido, $\mathcal{M}^*$ es literalmente $\mathcal{M}$ (la abstracción constructiva es la identidad).

### 7. DAS como aprendizaje de la traducción

En el ejemplo anterior $\tau$ era conocida. En una red entrenada no: $E_1$ y $E_2$ pueden estar codificadas en un sistema de coordenadas entrelazado (como el $(D_1, D_2)$ de $\mathcal{M}^*$). DAS aprende la traducción.

#### Por qué embeddings aleatorios (celdas 43-44)

Cada ejemplo usa **vectores aleatorios nuevos**. Posiciones con el mismo valor binario dentro de un par reciben el mismo vector; con valor distinto, vectores independientes. Así la red no puede memorizar embeddings de letras: la única señal estable es la relación geométrica (identidad vs. diferencia) dentro de cada par. Es el primer experimento de Geiger et al. (2023), el paper que introdujo DAS.

#### Celda 45: imports y construcción del input

```python
import torch
from torch.utils.data import DataLoader
from datasets import Dataset
from tqdm import tqdm, trange

from pyvene.models.mlp.modelings_mlp import MLPConfig
from pyvene import create_mlp_classifier
from pyvene import IntervenableModel, RotatedSpaceIntervention, RepresentationConfig, IntervenableConfig

torch.manual_seed(seed)
```

- `pyvene` es la librería de intervenciones de Stanford (la misma que usa `causalab` por debajo en `causalab/causalab/neural/pyvene_core/`). Acá se usa directo:
  - `MLPConfig` / `create_mlp_classifier`: un MLP clasificador estilo HuggingFace que trae `pyvene` para experimentos de juguete.
  - `IntervenableModel`: envuelve un modelo de PyTorch y le inserta intervenciones en ciertos puntos (hooks) según un config.
  - `IntervenableConfig` / `RepresentationConfig`: describen dónde intervenir (capa, componente, unidad) y con qué tipo de intervención.
  - `RotatedSpaceIntervention`: la intervención de DAS (rotación ortogonal + intercambio de dimensiones en el espacio rotado + rotación inversa).
- `Dataset`, `tqdm`, `trange` se importan pero no se usan en las celdas que siguen.

```python
embedding_dim = 4

def randvec(n=4, lower=-1, upper=1):
    return np.array([round(random.uniform(lower, upper), 2) for _ in range(n)])
```

Vector aleatorio de 4 componentes uniformes en $[-1,1]$, redondeadas a 2 decimales.

```python
def make_input_tensor(trace):
    e = [randvec(embedding_dim) for _ in range(4)]
    vecs = [None] * 4
    vecs[0] = e[0]
    vecs[1] = e[0] if trace["A1"] == trace["A2"] else e[1]
    vecs[2] = e[2]
    vecs[3] = e[2] if trace["A3"] == trace["A4"] else e[3]
    return torch.tensor(np.concatenate(vecs), dtype=torch.float32)
```

- Cuatro vectores nuevos por llamada.
- Posición 1 (A1) siempre `e[0]`; posición 2 (A2) repite `e[0]` si $A_1 = A_2$, si no usa `e[1]`. Lo mismo con el segundo par usando `e[2]`/`e[3]`.
- El par 2 siempre usa vectores distintos de los del par 1, así que no hay igualdades entre pares que confundan.
- Concatena → tensor de shape `[16]` (4 posiciones × 4 dims).
- Ojo: el input solo depende del **patrón de igualdad** (E1, E2), no de los bits. (0,0,1,1) y (1,1,0,0) tienen la misma distribución de input. Está bien porque el output solo depende de E1 y E2.

Salida guardada: `Embedding dim: 4, MLP input dim: 16`.

#### Celda 47: datos de entrenamiento factuales

```python
def generate_factual_data(causal_model, n):
    X_list, y_list = [], []
    for _ in range(n):
        trace = causal_model.sample_input()
        X_list.append(make_input_tensor(trace))
        y_list.append(trace["C"])
    return torch.stack(X_list), torch.tensor(y_list, dtype=torch.long)

n_train = 1048576
batch_size = 1024
X_train, y_train = generate_factual_data(M_model, n_train)
```

- `CausalModel.sample_input(filter_func=None)` (`causal_model.py`, línea 256): elige un valor al azar (`random.choice`) de `self.values[var]` para cada input y crea un trace con `new_trace`. Si se pasa `filter_func`, repite hasta que el trace lo cumpla (rejection sampling). Acá sin filtro: $A_1..A_4$ uniformes en {0,1}, así $E_1, E_2$ son 1 con probabilidad 1/2 y $C$ queda balanceado.
- Cada ejemplo: input `[16]`, etiqueta `C`. `torch.stack` → `X_train` de shape `[1048576, 16]`, `y_train` `[1048576]` (long, para cross-entropy).

Salida guardada: `1048576 examples, input dim = 16`; `524937 positive, 523639 negative`.

#### Celda 48: entrenar el MLP

```python
h_dim = 16
mlp_config = MLPConfig(
    h_dim=h_dim, n_layer=3, pdrop=0.0,
    include_bias=True, num_classes=2,
    squeeze_output=False, input_dim=embedding_dim * 4
)
_, _, mlp = create_mlp_classifier(mlp_config)
```

- MLP con 3 capas ocultas de 16, sin dropout, con bias, 2 clases, input 16.
- `create_mlp_classifier` devuelve una tupla de 3; el tercero es el modelo (los otros dos se descartan; no puedo verificar qué son sin el código de pyvene, probablemente config y tokenizer).
- `squeeze_output=False`: por cómo se usan los logits más abajo (`cf_out[0].squeeze(1)`), los logits parecen salir con una dimensión de posición extra cuando el input viene con forma `[batch, 1, 16]`.

```python
for name, p in mlp.named_parameters():
    if 'weight' in name and p.dim() >= 2:
        torch.nn.init.kaiming_normal_(p, nonlinearity='relu')
    elif 'bias' in name:
        torch.nn.init.zeros_(p)
```

Inicialización Kaiming (varianza escalada para ReLU) en las matrices de pesos y bias en cero, para que el gradiente fluya bien.

```python
mlp.to("cuda")
optimizer = torch.optim.Adam(mlp.parameters(), lr=0.01)

n_epochs = 3
for epoch in range(n_epochs):
    correct, total = 0, 0
    perm = torch.randperm(X_train.shape[0])
    for start in range(0, X_train.shape[0], batch_size):
        idx = perm[start:start + batch_size]
        xb = X_train[idx].to("cuda")
        yb = y_train[idx].to("cuda")
        loss, logits = mlp(inputs_embeds=xb, labels=yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        correct += (logits.argmax(1) == yb).sum().item()
        total += yb.shape[0]
    print(...)
```

- Requiere GPU (`"cuda"` hardcodeado).
- Loop clásico: permutación aleatoria por época, batches de 1024 (`xb`: `[1024, 16]`).
- `mlp(inputs_embeds=xb, labels=yb)`: estilo HuggingFace; con `labels` devuelve `(loss, logits)`. `inputs_embeds` porque el input ya es un vector continuo, no IDs de tokens.
- Acumula accuracy de entrenamiento con `argmax`.
- Al final imprime el rango de logits sobre 1000 ejemplos.

Salida guardada: accuracy 0.9510 → 0.9988 → 0.9992; `Logit range: [-56.1, 60.7]` (la red está muy confiada). La línea `loaded model` la imprime pyvene al crear el MLP.

#### Celda 49: test

Genera 10000 ejemplos nuevos (con vectores aleatorios nuevos), pone `mlp.eval()`, calcula accuracy con `torch.no_grad()`. Salida guardada: `Test accuracy: 0.9994`. La red resuelve la tarea, pero todavía no sabemos cómo codifica $E_1$ y $E_2$ en la capa oculta.

#### Celda 50: dataset contrafactual

```python
def generate_counterfactual_data(causal_model, n, intervention_id=2):
    examples = []
    for _ in range(n):
        base_trace = causal_model.sample_input()
        source1_trace = causal_model.sample_input()
        source2_trace = causal_model.sample_input()

        base_tensor = make_input_tensor(base_trace)
        source1_tensor = make_input_tensor(source1_trace)
        source2_tensor = make_input_tensor(source2_trace)

        cf_E1 = source1_trace["E1"] if intervention_id in (0, 2) else base_trace["E1"]
        cf_E2 = source2_trace["E2"] if intervention_id in (1, 2) else base_trace["E2"]
        cf_label = eq(cf_E1, cf_E2)

        examples.append({
            "input_ids": base_tensor,
            "source_input_ids": torch.stack([source1_tensor, source2_tensor]),
            "labels": torch.tensor(cf_label, dtype=torch.long),
            "intervention_id": intervention_id,
        })
    return examples
```

- Tres inputs por ejemplo: un base y **dos sources**. `source1` aporta $E_1$, `source2` aporta $E_2$.
- `intervention_id`: 0 = intercambiar solo $E_1$, 1 = solo $E_2$, 2 = ambos.
- La etiqueta contrafactual se calcula a mano: es $C$ de $\mathcal{M}(\text{base})_{E_1 \leftarrow \mathcal{M}(s_1),\, E_2 \leftarrow \mathcal{M}(s_2)}$ (o la versión con una sola variable). Equivale a lo que daría `M_model.run_interchange(base_trace, {"E1": source1_trace, "E2": source2_trace})["C"]`, pero sin pasar por `run_interchange` (y no usa `CounterfactualExample` ni `label_counterfactual_data` de `causalab`, aunque es exactamente el caso de uso de esas funciones).
- Shapes por ejemplo: `input_ids` `[16]`, `source_input_ids` `[2, 16]`, `labels` escalar.
- Ojo: el nombre `input_ids` es engañoso, no son IDs sino vectores.

```python
cf_batch_size = 6400
n_cf = 1280000
train_cf_dataset = generate_counterfactual_data(M_model, n_cf, intervention_id=2)
```

Se entrena **solo con intervention_id = 2** (ambas variables a la vez). Esta celda no tiene salida guardada.

Detalle importante: con `intervention_id = 2`, la etiqueta es `eq(source1.E1, source2.E2)` y **no depende del base para nada**. O sea, en entrenamiento la red intervenida tiene que sacar toda la información relevante de los subespacios intercambiados; lo que quede en el resto del espacio del base tiene que ser irrelevante. Esto es una consecuencia del código que la notebook no remarca.

#### Celda 51: helpers de DAS

```python
def create_das_model(mlp, layer, hidden_dim, emb_dim):
    config = IntervenableConfig(
        model_type=type(mlp),
        representations=[
            RepresentationConfig(layer, "block_output", "pos", 1,
                                 subspace_partition=None, intervention_link_key=0),
            RepresentationConfig(layer, "block_output", "pos", 1,
                                 subspace_partition=None, intervention_link_key=0),
        ],
        intervention_types=RotatedSpaceIntervention,
    )
    return IntervenableModel(config, mlp, use_fast=True)
```

- Dos `RepresentationConfig`, una por variable ($E_1$, $E_2$), las dos en **el mismo lugar**: capa `layer`, componente `"block_output"` (la salida del bloque de esa capa del MLP, o sea la activación oculta de 16 dims), unidad `"pos"` (posición) y `1` = número máximo de unidades intervenidas.
- `intervention_link_key=0` en ambas: **vincula** las dos intervenciones para que compartan la misma instancia de intervención, o sea la misma matriz de rotación $\mathbf{R}$. Esto es clave: hay una sola rotación de la capa y cada variable se queda con una parte de las dimensiones rotadas.
- `intervention_types=RotatedSpaceIntervention`: tipo de intervención para todas.
- `use_fast=True`: asume que la ubicación de la intervención es la misma para todo el batch (de ahí el warning de la salida de la celda 53).

```python
def run_das_forward(das_model, batch, batch_size, emb_dim):
    iid = batch["intervention_id"][0].item()
    if iid == 2:
        _, cf_out = das_model(
            {"inputs_embeds": batch["input_ids"]},
            [
                {"inputs_embeds": batch["source_input_ids"][:, 0]},
                {"inputs_embeds": batch["source_input_ids"][:, 1]},
            ],
            {"sources->base": (
                [[[0]] * batch_size, [[0]] * batch_size],
                [[[0]] * batch_size, [[0]] * batch_size],
            )},
            subspaces=[
                [[_ for _ in range(0, emb_dim * 2)]] * batch_size,
                [[_ for _ in range(emb_dim * 2, emb_dim * 4)]] * batch_size,
            ],
        )
    elif iid == 0: ...
    elif iid == 1: ...
    return cf_out
```

La llamada a `IntervenableModel` es `das_model(base, sources, unit_locations, subspaces=...)`:

- `iid = batch["intervention_id"][0].item()`: asume que todo el batch tiene el mismo tipo de intervención (cierto, porque cada dataset se genera con un solo `intervention_id`).
- Primer argumento, el **base**: `{"inputs_embeds": batch["input_ids"]}` con shape `[bs, 1, 16]` (la dimensión de "posición" de tamaño 1 la agrega el `unsqueeze` del loop de entrenamiento).
- Segundo argumento, la **lista de sources**, una por intervención: `source_input_ids[:, 0]` (`[bs, 1, 16]`, source1 → $E_1$) y `source_input_ids[:, 1]` (source2 → $E_2$).
- Tercer argumento, **ubicaciones** `"sources->base"`: una tupla (ubicaciones en los sources, ubicaciones en el base). Cada una es una lista con un elemento por intervención; cada elemento es `[[0]] * batch_size`, o sea, "posición 0" para cada ejemplo del batch. Como el MLP tiene una sola "posición", siempre es 0.
- `subspaces`: por intervención, por ejemplo, la lista de **dimensiones del espacio rotado** que se intercambian. $E_1$ usa dims 0-7 (`range(0, 8)`) y $E_2$ dims 8-15 (`range(8, 16)`). Las 16 dimensiones rotadas se parten en dos mitades.
- Devuelve `(salida_base_original, salida_contrafactual)`; se descarta la primera y se usa `cf_out`.
- Para `iid == 0` o `1`: se pasa `None` en el lugar de la intervención que no se usa (en sources, ubicaciones y subspaces). Para $E_2$ se usa source2 (índice 1).

Qué hace `RotatedSpaceIntervention` por dentro: no lo pude verificar en el código (pyvene no está instalado en este entorno y no está incluido en el repo). Por la documentación de pyvene y el paper de DAS, la idea es: rotar base y source con una matriz ortogonal $Q$ (`rotate_layer`, de tamaño `16 × 16` acá), reemplazar en el base rotado las dimensiones indicadas en `subspaces` por las del source rotado, y deshacer la rotación con $Q^\top$. Con dos intervenciones vinculadas que comparten $Q$, el efecto conceptual es: dims 0-7 del base rotado vienen de source1, dims 8-15 de source2. Cómo exactamente pyvene compone dos intervenciones en el mismo punto (si las aplica en secuencia o juntas) no lo pude comprobar.

Relación con la fórmula $f = (\mathbf{R}h, h - \mathbf{R}^\top\mathbf{R}h)$: acá la rotación es completa ($16 \times 16$), y las filas 0-7 de $Q$ juegan el rol de $\mathbf{R}_{E_1} \in \mathbb{R}^{8 \times 16}$ y las filas 8-15 el de $\mathbf{R}_{E_2}$. Intercambiar solo dims 0-7 en el espacio rotado es exactamente $h_b + \mathbf{R}_{E_1}^\top\mathbf{R}_{E_1}(h_s - h_b)$: la parte de las otras 8 dimensiones es el "residuo" $f_2$, que se conserva del base.

```python
def batched_random_sampler(data, batch_size):
    batch_indices = list(range(int(len(data) / batch_size)))
    random.shuffle(batch_indices)
    for b_i in batch_indices:
        for i in range(b_i * batch_size, (b_i + 1) * batch_size):
            yield i
```

Sampler que baraja el **orden de los batches** pero no los ejemplos dentro de cada batch (bloques contiguos de 6400). Descarta el resto si `len(data)` no es múltiplo de `batch_size` (acá sí es múltiplo: 1280000 / 6400 = 200 batches). Como los ejemplos ya se generaron al azar, barajar dentro del batch no aporta nada.

`compute_metrics` (no se usa después) y `compute_loss` (cross-entropy) son triviales.

#### Celda 53: entrenar la rotación

```python
das_model = create_das_model(mlp, layer=0, hidden_dim=h_dim, emb_dim=embedding_dim)
das_model.set_device("cuda")
das_model.disable_model_gradients()
```

- Interviene en `layer=0`: la salida de la **primera** capa oculta (16 dims).
- `disable_model_gradients()`: congela los pesos del MLP. Lo único entrenable es la intervención.

```python
optimizer_params = []
for k, v in das_model.interventions.items():
    optimizer_params += [{"params": v.rotate_layer.parameters()}]
    break
das_optimizer = torch.optim.Adam(optimizer_params, lr=0.001)
print(f"Trainable parameters: {das_model.count_parameters()}")
```

- Toma la `rotate_layer` de la primera intervención y corta (`break`): como están vinculadas, comparten la rotación, así que agregar la segunda duplicaría los parámetros en el optimizador.
- Salida guardada: `Trainable parameters: 256` = $16 \times 16$. Esto confirma que la rotación es **completa** (`16 × 16`), no low-rank $k \times d$ como dice el texto; el "low rank" aparece al partir las 16 dimensiones rotadas en dos bloques de 8.

```python
for epoch in range(das_epochs):
    das_model.model.train()
    loader = DataLoader(train_cf_dataset, batch_size=cf_batch_size,
                        sampler=batched_random_sampler(train_cf_dataset, cf_batch_size))
    for batch in loader:
        batch["input_ids"] = batch["input_ids"].unsqueeze(1)
        batch["source_input_ids"] = batch["source_input_ids"].unsqueeze(2)
        bs = batch["input_ids"].shape[0]
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to("cuda")

        cf_out = run_das_forward(das_model, batch, bs, embedding_dim)
        logits = cf_out[0].squeeze(1)
        labels = batch["labels"].squeeze().long()
        preds = logits.argmax(1)
        loss = compute_loss(logits, labels)

        das_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(das_model.parameters(), 1.0)
        das_optimizer.step()
        das_model.set_zero_grad()
        ...
    iia = epoch_correct / epoch_total
    avg_loss = epoch_loss / (epoch_total / cf_batch_size)
```

Línea por línea:

- `DataLoader` con el sampler custom: el `collate` por defecto apila los dicts → `input_ids` `[6400, 16]`, `source_input_ids` `[6400, 2, 16]`, `labels` `[6400]`, `intervention_id` `[6400]`.
- `unsqueeze(1)` en el base → `[6400, 1, 16]`; `unsqueeze(2)` en los sources → `[6400, 2, 1, 16]`. Se agrega la dimensión de "posición" (secuencia de largo 1) que pyvene espera para la unidad `"pos"`. Así `source_input_ids[:, 0]` queda `[6400, 1, 16]`.
- Pasa todo a GPU.
- `run_das_forward`: corre la red con la intervención en el espacio rotado.
- `cf_out[0]`: logits contrafactuales, `[6400, 1, 2]` → `squeeze(1)` → `[6400, 2]`.
- `loss`: cross-entropy contra la etiqueta contrafactual del modelo causal. Si la rotación es buena, la red intervenida hace lo que predice $\mathcal{M}$.
- `clip_grad_norm_` a 1.0 para estabilidad; `step` actualiza solo la rotación (es lo único en el optimizador).
- `das_model.set_zero_grad()`: limpia gradientes también de lo que no está en el optimizador.
- IIA de entrenamiento = fracción de aciertos sobre la etiqueta contrafactual. `avg_loss` = suma de losses / número de batches.

Cómo se mantiene ortogonal $\mathbf{R}$: en pyvene, `rotate_layer` va envuelta en `torch.nn.utils.parametrizations.orthogonal` (lo mismo hace `causalab` en `SubspaceFeaturizer`, ver abajo), así que el optimizador mueve un parámetro libre y la matriz que se usa es siempre ortogonal. Esto lo afirmo por `causalab/causalab/neural/featurizers.py` y por cómo funciona pyvene; el código de pyvene no lo pude abrir.

Salidas guardadas: IIA 0.6784 (época 1) → 0.8437 → 0.9601 → ... → 0.9992 (época 10); loss de 7.71 a 0.0036. El warning de `use_fast=True` avisa que la ubicación de la intervención se asume fija en el batch.

#### Celda 54: IIA en datos contrafactuales nuevos

Mismo loop pero con `torch.no_grad()`, `das_model.model.eval()` y 12800 ejemplos nuevos con `intervention_id=2` (vectores aleatorios nunca vistos). Salida guardada: `Test IIA (both E1 & E2): 0.9994`. El comentario del código dice que la diferencia con 1.0 se debe a entrenamiento finito y optimización aproximada.

#### Celda 55: IIA por variable

```python
def eval_iia(das_model, causal_model, intervention_id, n=12800, batch_size=6400):
    dataset = generate_counterfactual_data(causal_model, n, intervention_id=intervention_id)
    ... mismo loop de evaluación ...
    return correct / total

iia_e1 = eval_iia(das_model, M_model, intervention_id=0)
iia_e2 = eval_iia(das_model, M_model, intervention_id=1)
iia_both = eval_iia(das_model, M_model, intervention_id=2)
```

- Genera datasets donde solo se intercambia $E_1$ (dims 0-7 desde source1), solo $E_2$ (dims 8-15 desde source2), o ambas.
- Esto es más exigente que el entrenamiento: con `iid = 0`, el base sí importa (aporta $E_2$ por las dims 8-15 que no se tocan). Si la partición estuviera mal (por ejemplo, $E_1$ repartida entre las dos mitades), esto fallaría aunque la IIA conjunta fuera alta.

Salidas guardadas: E1 solo 0.9992, E2 solo 0.9995, ambas 0.9991. La rotación aprendida parte la capa limpiamente: las primeras 8 dimensiones rotadas llevan $E_1$ y las otras 8 llevan $E_2$, con poco cruce. Eso es lo que hace "buena" a una traducción: cada subespacio es causalmente autocontenido.

#### Celdas 57-58: conexión con la Parte 1

- $\mathbf{R}$ define una traducción $\tau$ (biyección sobre la capa oculta), igual que $\tau$ recortaba $(D_1, D_2)$ en $(E_1, E_2)$.
- Un intercambio de subespacio en la base rotada (hard intervention sobre $f_1$) es, en la base de neuronas, una perturbación distribuida que toca las 16 neuronas. Mismo patrón que el pullback de la Parte 1, donde fijar $E_1$ requería reescribir dos mecanismos de $\mathcal{M}^*$.
- IIA alta = la ecuación de abstracción constructiva vale (aproximadamente) en el espacio traducido.

La celda 52 también menciona **Boundless DAS** (Wu et al., 2023): en vez de elegir a mano cuántas dimensiones rotadas le tocan a cada variable (acá 8 de 16, una asignación enorme), las aprende. En Alpaca 7B encontraron que cada variable booleana necesitaba solo 5-10% del espacio.

---

## Cómo implementa esto `causalab`: featurizers

La notebook no usa `causalab.neural`, pero la fórmula de la Parte 2 está implementada literalmente en `causalab/causalab/neural/featurizers.py`, y es lo que usan los experimentos de DAS de la librería. Vale la pena entenderlo porque es la "traducción" en código.

**`Featurizer`**: contenedor de dos módulos de PyTorch:

- `featurizer`: $x \mapsto (f, \text{error})$, donde `error` es el residuo de reconstrucción.
- `inverse_featurizer`: $(f, \text{error}) \mapsto \hat{x}$.

Por defecto ambos son la identidad (`IdentityFeaturizerModule` devuelve `(x, None)`), con `id="null"`; `is_trivial()` chequea justamente ese `id`. O sea: el featurizer identidad es la "traducción trivial" que deja las neuronas en la base estándar (abstracción constructiva pura, como la Lecture 8).

**`SubspaceFeaturizerModule.forward`**:

```python
r = self.rotate.weight.T      # (k, d)
f = x.to(r.dtype) @ r.T       # f = R x, shape (..., k)
error = x - (f @ r).to(x.dtype)  # x - R^T R x
return f, error
```

Es exactamente $f_1 = \mathbf{R}h$ y $f_2 = h - \mathbf{R}^\top\mathbf{R}h$. `self.rotate` es una `pv.models.layers.LowRankRotateLayer` de pyvene con `weight` de shape `(d, k)` (columnas ortonormales), por eso se transpone.

**`SubspaceInverseFeaturizerModule.forward`**: `(f @ r) + error` = $\mathbf{R}^\top f_1 + f_2$. La inversa de la notebook.

**`SubspaceFeaturizer(shape=(d, k), trainable=True)`**: crea `LowRankRotateLayer(d, k, init_orth=True)`, la envuelve con `torch.nn.utils.parametrizations.orthogonal` (así se mantiene ortogonal durante el entrenamiento), y arma el featurizer con `n_features = k`. También acepta `rotation_subspace` para cargar una matriz ya calculada (por ejemplo, de PCA/SVD con `build_SVD_featurizers`). En `causalab/causalab/experiments/train.py` (`_initialize_featurizers`) es lo que se usa para DAS: `SubspaceFeaturizer(shape=(unit_shape[0], n_features), trainable=True, id=f"DAS_{unit.id}")`, con `n_features` por defecto 32.

**La intervención de interchange** (`build_feature_interchange_intervention`): devuelve una *clase* (no una instancia, porque pyvene instancia las intervenciones por su cuenta) cuyo `forward(base, source, subspaces)` hace:

1. `f_base, base_err = featurizer(base)`; `f_src, _ = featurizer(source)`.
2. Si no hay `subspaces`, `f_out = f_src` (se reemplaza todo el espacio de features); si hay, intercambia solo esas coordenadas con `pv.models.intervention_utils._do_intervention_by_swap`.
3. `inverse(f_out, base_err)`: reconstruye con el **residuo del base**.

O sea: coordenadas del subespacio del source + todo lo demás del base. Es la interchange en el espacio traducido, exactamente $h_b + \mathbf{R}^\top\mathbf{R}(h_s - h_b)$.

Otros featurizers del mismo archivo: `SAEFeaturizer` (usa encode/decode de un sparse autoencoder como traducción, con el error de reconstrucción como residuo, así sigue siendo invertible) y la intervención de **mask** (`build_feature_mask_intervention`, DBM: aprende una compuerta sigmoide por feature para decidir cuáles vienen del source, con pérdida de sparsity L1). Todos encajan en la misma idea: elegir un featurizer = elegir la traducción $\tau$; intervenir features = hacer abstracción constructiva sobre el modelo traducido.

En `causalab/causalab/neural/model_units.py`, `create_intervention_config` pone como `intervention_type` la clase que devuelve `featurizer.get_interchange_intervention()`, y `causalab/causalab/neural/pyvene_core/intervenable_model.py` (`prepare_intervenable_model`) arma el `pv.IntervenableModel` con esos configs, agrupando unidades con `group_key` para que compartan source. Es el mismo patrón que la notebook arma a mano con `IntervenableConfig`, `RepresentationConfig` e `intervention_link_key`.

---

## Ejercicios

La notebook no tiene ejercicios, ni celdas `FILL IN YOUR ANSWER HERE`, ni soluciones marcadas `# TERRON`. Ideas para practicar (sin respuestas):

- Escribir el pullback de $E_2 \leftarrow e$ en $\mathcal{M}^*$ y verificarlo con una función tipo `check_pullback_E1`. Pista para razonarlo: ¿cuál de $D_1$, $D_2$ contiene información de $E_2$?
- Pensar si la ecuación con la traducción vale para intervenciones sobre $D_2$ en $\mathcal{M}^*$: ¿a qué intervención de $\mathcal{M}$ corresponde "fijar $D_2$"?
- Cambiar el reparto de dimensiones (por ejemplo 2 dims para cada variable en lugar de 8) y ver cómo cambia la IIA por variable.
- Correr un baseline por neuronas: intercambiar subconjuntos de neuronas de la capa 0 sin rotación y comparar con el 0.73 del paper.

---

## Ojo / errores comunes / dudas

- **Traducción ≠ abstracción constructiva**: la traducción es biyectiva (mismo número de estados, no se pierde info); la abstracción constructiva agrupa y puede perder detalle. La abstracción bajo traducción es la composición de las dos.
- **La traducción se define en todos los settings**, no solo en los alcanzables; por eso la celda 25 chequea los 128 estados. Si solo se chequearan corridas normales, una función no invertible podría pasar el test.
- **Una hard intervention arriba puede ser un interventional abajo**: el pullback de $E_1 \leftarrow e$ reescribe dos mecanismos, y uno de ellos ($D_2$) no queda constante. En redes: intercambiar un subespacio rotado toca todas las neuronas.
- **Padres triviales**: `E1`/`D1` constantes llevan `parents=["A1"]` para no ser tratadas como inputs por `CausalModel`/`CausalTrace`.
- **`D1` como output**: en $\mathcal{M}^*$, `D1` no tiene hijos, así que `CausalModel` la mete en `outputs` y en el último timestep. Es la razón del orden raro de `M_star_model.variables`.
- **Rotación completa vs. low-rank**: el texto habla de $\mathbf{R} \in \mathbb{R}^{k\times d}$; el código usa una rotación $16 \times 16$ (256 parámetros) y la parte en dos bloques de 8. Son lo mismo visto de dos maneras: cada bloque de filas es un $\mathbf{R}$ low-rank.
- **El entrenamiento con `iid = 2` no usa información del base** (la etiqueta solo depende de los sources). La evaluación por variable (`iid = 0` o `1`) es la que realmente prueba que la partición separa $E_1$ de $E_2$.
- **8 de 16 dimensiones por variable es muchísimo**: con tanto espacio, una rotación "casi cualquiera" podría acomodar mucha información. En modelos grandes las variables ocupan una fracción pequeña. IIA alta con subespacios grandes es evidencia más débil (esto es un comentario mío inspirado en la celda 52).
- **DAS puede encontrar alineaciones aunque la red no "use" la variable de forma natural**: el subespacio se entrena con supervisión del modelo causal, así que siempre conviene evaluar en datos nuevos (celdas 54-55) y desconfiar de subespacios demasiado grandes.
- **Requiere GPU**: `"cuda"` está hardcodeado en todo el bloque de DAS.
- **`input_ids` no son IDs**: son vectores continuos de 16 dims, y se pasan como `inputs_embeds`.
- **Ubicaciones de pyvene**: `[[0]] * batch_size` es "posición 0 para cada ejemplo"; los `unsqueeze` agregan esa dimensión de posición de largo 1.
- **No verificado**: el comportamiento interno de `RotatedSpaceIntervention`, `create_mlp_classifier`, `"block_output"` y de cómo pyvene combina dos intervenciones vinculadas en el mismo lugar. pyvene no está instalado en este entorno ni incluido en el repo; lo que digo sale de la notebook, de `causalab/causalab/neural/featurizers.py` (que usa las mismas piezas de pyvene) y del conocimiento general de la librería.

---

## Resumen de una línea por sección

- **Setup**: mismo entorno e imports que la Lecture 8.
- **1. Recap**: abstracción bajo traducción = primero recarvar el modelo de bajo nivel, después abstracción constructiva.
- **2. Modelo $\mathcal{M}$**: igualdad jerárquica binaria con $E_1$, $E_2$, $C$ como en la Lecture 8.
- **3. Modelo $\mathcal{M}^*$**: mismo comportamiento con variables internas distintas ($D_2$ ya contiene la respuesta, $D_1$ no tiene hijos).
- **4. Traducción**: $E_1 = D_1$, $E_2 = (D_1 = D_2)$ es una biyección sobre los 128 settings y conmuta con las 16 corridas normales.
- **5. Pullback**: $E_1 \leftarrow e$ en $\mathcal{M}$ corresponde a reescribir $D_1$ y $D_2$ en $\mathcal{M}^*$, y la conmutación se verifica para $e \in \{0,1\}$.
- **6. Abstracción bajo traducción**: composición de traducción + abstracción constructiva.
- **7. DAS**: se entrena un MLP con embeddings aleatorios (test acc 0.9994) y una rotación ortogonal $16\times16$ de la capa 0 cuyas dims 0-7 y 8-15 realizan $E_1$ y $E_2$ con IIA ≈ 0.999 por variable (ejecución guardada).
- **Featurizers de `causalab`**: `SubspaceFeaturizer` implementa $f = (\mathbf{R}h, h - \mathbf{R}^\top\mathbf{R}h)$ y su inversa; elegir featurizer = elegir la traducción.
