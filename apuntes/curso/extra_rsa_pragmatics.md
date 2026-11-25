# Extra: How do models perform pragmatic inference? (RSA y pragmática)

Esta notebook extra junta pragmática computacional con interpretabilidad. Primero arma, con un juego de referencia de juguete, el modelo **Rational Speech Acts (RSA)**: cómo un oyente "racional" infiere a qué objeto se refiere alguien poniéndose en el lugar del hablante. Después pregunta si un LM grande (Qwen3.5-27B, corrido en remoto vía NDIF) **se comporta** como un oyente pragmático y, con interchange interventions sobre el residual stream, empieza a explorar si su **cómputo interno** se parece al de RSA. No hay ejercicios ni celdas `# TERRON`; es una demo para jugar.

Versión de nnsight fijada en `requirements/extra_rsa_pragmatics.txt`: `nnsight==0.8.2` (más nueva que la de la lecture 7, ojo con diferencias de API).

---

## Ideas clave / teoría

### Pragmática vs semántica

La **semántica** de una palabra dice a qué objetos se aplica literalmente: "blue" es verdadero para todo lo azul. La **pragmática** es lo que inferimos del hecho de que el hablante eligió *esa* palabra y no otra. Si había una palabra más específica disponible y no la usó, eso es información.

### Reference game

Un **reference game** es el escenario mínimo para estudiar esto: hay un conjunto de objetos (referentes $r \in R$), el hablante quiere señalar uno y solo puede decir una palabra de un conjunto fijo de utterances $u \in U$. El oyente tiene que adivinar el objeto.

En la notebook: $R$ = {blue square, blue circle, green square}, $U$ = {"blue", "green", "square", "circle"}. La semántica se codifica en una **matriz de significados** $\llbracket u \rrbracket(r) \in \{0,1\}$ (fila = utterance, columna = referente):

| | blue square | blue circle | green square |
|---|---|---|---|
| blue | 1 | 1 | 0 |
| green | 0 | 0 | 1 |
| square | 1 | 0 | 1 |
| circle | 0 | 1 | 0 |

El razonamiento intuitivo: si escucho "blue", literalmente puede ser cualquiera de los dos azules. Pero si el hablante quisiera el blue circle, podría haber dicho "circle", que es inequívoco. No lo dijo, así que probablemente quiere el blue square (para el cual no existe una palabra inequívoca: "square" también aplica al verde).

### Las tres capas de RSA

RSA formaliza "ponerse en el lugar del otro" como una recursión de agentes probabilísticos ([Frank & Goodman 2012](https://doi.org/10.1126/science.1218633); el libro [problang.org](https://www.problang.org/)). La figura `figures/extra_rsa_pragmatics/rsa.png` muestra las burbujas anidadas: L1 piensa en S1, que piensa en L0.

**Literal listener (L0):**

$$P_{L_0}(r \mid u) \propto \llbracket u \rrbracket(r)$$

- Dada la palabra $u$, elige uniformemente entre los referentes para los que $u$ es verdadera.
- "$\propto$" significa "proporcional a": hay que normalizar para que sume 1 sobre $r$. En la matriz: normalizar cada **fila**.
- "blue" → (0.5, 0.5, 0).

**Pragmatic speaker (S1):**

$$P_{S_1}(u \mid r) \propto P_{L_0}(r \mid u)$$

- El hablante que quiere comunicar $r$ elige $u$ en proporción a qué tan probable es que el literal listener acierte $r$ al oírla. O sea, prefiere palabras **informativas**.
- Normalización sobre $u$ para un $r$ fijo: tomar la **columna** $r$ de la matriz L0 y normalizarla (en el código: transponer y normalizar filas).
- Para blue circle: L0 da 0.5 con "blue" y 1 con "circle" → normalizado (1/3, 0, 0, 2/3).
- La notebook aclara que es una simplificación del modelo de Frank & Goodman: el modelo completo suele tener un parámetro de racionalidad $\alpha$ (softmax de $\alpha \cdot \log P_{L_0}$ menos un costo de la utterance) y un prior sobre referentes. Acá $\alpha = 1$, sin costos y con prior uniforme.

**Pragmatic listener (L1):**

$$P_{L_1}(r \mid u) \propto P_{S_1}(u \mid r)$$

- El oyente pragmático hace Bayes sobre el hablante: $P(r \mid u) \propto P(u \mid r)\,P(r)$, con prior $P(r)$ uniforme (por eso no aparece).
- Normalización sobre $r$ para $u$ fijo: tomar la columna $u$ de S1 (otra vez transponer y normalizar filas).
- Para "blue": S1(blue | blue square) = 1/2 (el blue square solo tiene "blue" y "square", ambas con L0 = 0.5), S1(blue | blue circle) = 1/3 → normalizado (0.6, 0.4, 0). Es exactamente la salida guardada.

Lo importante: cada nivel es la misma operación (normalizar en la otra dirección), aplicada sobre el resultado del nivel anterior. Toda la "pragmática" sale de esa doble normalización: la palabra "circle" le "roba" probabilidad al blue circle en S1, y eso se propaga a L1.

### ¿Cómo se prueba si un LM es un listener pragmático?

1. **Conductualmente**: se le da al LM un reference game en texto, con nombres para los objetos, y se lee la distribución del siguiente token sobre los nombres. Si la palabra es ambigua literalmente pero RSA la desambigua, un LM pragmático debería poner más probabilidad en el referente que predice L1.
2. **Controlando sesgos**: un solo prompt no alcanza; el LM podría preferir un nombre, un color o un animal por sí mismo. Hay que variar contextos de forma que la predicción de RSA cambie y ver si el LM cambia en la misma dirección.
3. **Mecanísticamente (causal abstraction)**: RSA es un modelo causal de alto nivel con variables (la utterance, el contexto/las alternativas, L0, S1, L1). La pregunta es si hay activaciones del LM que jueguen el mismo rol causal que esas variables. Para eso se usan **interchange interventions**: dos prompts que difieren en un solo factor (por ejemplo solo la utterance, o solo el contexto que determina qué alternativas son inequívocas), se pisa una activación del target con la del source, y se mira si el output se mueve como predice RSA si esa activación codificara esa variable ([Geiger et al. 2021](https://arxiv.org/abs/2106.02997)). Cambiar distintos factores es una forma de apuntar a distintos "pasos" de la inferencia: cambiar la utterance afecta el input de L1; cambiar el contexto afecta qué hubiera dicho S1 (las alternativas).

### NDIF

**NDIF** (National Deep Inference Fabric) es una infraestructura que hostea modelos grandes y deja correr trazas de nnsight **en remoto**: uno escribe el mismo código de `model.trace(...)` y nnsight lo serializa, lo manda al servidor, se ejecuta junto al modelo en sus GPUs, y vuelven solo los valores marcados con `.save()`. Así se puede intervenir un modelo de 27B sin cargarlo localmente. Hace falta una API key (login en https://login.api.ndif.us/).

---

## Recorrido por la notebook

### 0. Setup

Igual que en las lectures: detecta Colab, clona el repo e instala `requirements/extra_rsa_pragmatics.txt`; si no, configura plotly para renderizar con PNG. Sin nada especial.

### Pragmatic inference (texto)

Plantea el juego (figura `reference_game.png`: cuadrado azul, círculo azul, cuadrado verde) y el razonamiento intuitivo explicado arriba: "circle" → blue circle, "green" → green square, "blue" → más probablemente blue square.

### Rational Speech Acts: literal listener

```python
import numpy as np

REFERENCES = ["blue square", "blue circle", "green square"] # R
UTTERANCES = ["blue", "green", "square", "circle"] # U

MEANINGS = np.array([
    [1, 1, 0], # blue
    [0, 0, 1], # green
    [1, 0, 1], # square
    [0, 1, 0]  # circle
])
```

- `REFERENCES` y `UTTERANCES` fijan el orden de columnas y filas.
- `MEANINGS`: la matriz semántica $\llbracket u \rrbracket(r)$, shape `(4, 3)` = (utterances, referentes). Un 1 indica que la palabra es literalmente verdadera para el objeto.

```python
def literal_listener(u : int):
    L0 = MEANINGS / MEANINGS.sum(axis=1)[:, None]
    return L0[u]
```

- `MEANINGS.sum(axis=1)`: suma por fila (cuántos objetos describe cada palabra), shape `(4,)`: [2, 1, 2, 1].
- `[:, None]` la convierte en columna `(4, 1)` para que el broadcasting divida cada fila por su suma.
- `L0` queda `(4, 3)`, cada fila es $P_{L_0}(\cdot \mid u)$ y suma 1.
- `L0[u]`: la fila de la utterance pedida.

```python
utterance = "blue"
u = UTTERANCES.index(utterance)
l0 = literal_listener(u)
print(...)
```

Convierte el string a índice y muestra la distribución. Salida guardada: blue square 0.50, blue circle 0.50, green square 0.00.

### Pragmatic speaker

```python
def pragmatic_speaker(r : int):
    L0 = MEANINGS / MEANINGS.sum(axis=1)[:, None]
    S1 = L0.T / L0.T.sum(axis=1)[:, None]
    return S1[r]
```

- Recalcula L0 `(4, 3)`.
- `L0.T`: `(3, 4)` = (referentes, utterances). La fila $r$ contiene $P_{L_0}(r \mid u)$ para cada $u$.
- Dividir cada fila por su suma normaliza sobre $u$: eso es $P_{S_1}(u \mid r)$. `S1` es `(3, 4)`.
- `S1[r]`: distribución sobre utterances para el referente $r$.

Salida guardada para "blue circle": blue 0.33, green 0.00, square 0.00, circle 0.67. La columna del blue circle en L0 es (0.5, 0, 0, 1) → suma 1.5 → (1/3, 0, 0, 2/3).

### Pragmatic listener

```python
def pragmatic_listener(u : int):
    L0 = MEANINGS / MEANINGS.sum(axis=1)[:, None]
    S1 = L0.T / L0.T.sum(axis=1)[:, None]
    L1 = S1.T / S1.T.sum(axis=1)[:, None]
    return L1[u]
```

- `S1.T`: `(4, 3)` = (utterances, referentes). Fila $u$ = $P_{S_1}(u \mid r)$ para cada $r$.
- Normalizar filas = normalizar sobre $r$ con prior uniforme: $P_{L_1}(r \mid u)$. `L1` es `(4, 3)`, mismo formato que L0.

Cuentas para "blue" (para entender de dónde sale 0.6/0.4):
- S1 del blue square: su columna de L0 es (0.5, 0, 0.5, 0) → (0.5, 0, 0.5, 0). Entonces S1(blue | blue square) = 0.5.
- S1(blue | blue circle) = 1/3 (calculado arriba).
- S1(blue | green square) = 0.
- Normalizando (0.5, 0.333, 0): (0.6, 0.4, 0).

Salida guardada: literal 0.50/0.50/0.00, pragmático 0.60/0.40/0.00. El oyente pragmático "descuenta" al blue circle porque el hablante tenía "circle" disponible.

### What about language models?

```python
import nnsight
nnsight.login()
```

Pide/guarda la API key de NDIF (salida guardada: "NDIF API key saved.").

```python
model = nnsight.TransformersModel("Qwen/Qwen3.5-27B")
tokenizer = model.tokenizer
```

- Wrapper de nnsight 0.8 sobre un modelo de HuggingFace (en la lecture 7 se usaba `LanguageModel`; acá la clase es `TransformersModel`, no pude verificar en qué difieren exactamente). Como se va a correr remoto, no se descargan los pesos completos; se usa la estructura del modelo para armar las trazas.
- La notebook comenta que se puede probar con modelos más chicos siempre que respondan bien, y adelanta que "even a 27B model doesn't act all that pragmatically".

```python
NAMES = ["Gary", "Charles", "Frank"]
token_ids = [ids[1] if len(ids) > 1 else ids[0] for ids in tokenizer(NAMES).input_ids]

PROMPT_TEMPLATE = """Alice and Bob are playing a collaborative game where Alice says a word and Bob has to pick the thing that Alice is referring to.

In front of Bob are a blue dog called Gary, a green dog called Frank, and a blue cat called Charles. 

Alice sees the animals, but doesn't know their names. Instead, she can only say \"blue\", \"green\", \"dog\", or \"cat\".

Alice says, \"{utterance}\".

What animal is Alice referring to?
<think></think>
Answer: the animal that Alice is referring to is called **"""

def get_prompt(utterance : str):
    return PROMPT_TEMPLATE.format(utterance=utterance)
```

- El reference game traducido a texto. Mapeo con el ejemplo de figuras: color = color, forma → animal. Objetos: Gary = blue dog, Frank = green dog, Charles = blue cat. Utterances: "blue", "green", "dog", "cat".
- **Los nombres** son el truco para leer la respuesta: el LM no tiene que generar "the blue dog", sino un nombre, que es un token y se puede leer directo de la distribución del siguiente token.
- "Alice ... doesn't know their names": para que tenga sentido que use una sola palabra.
- `<think></think>`: Qwen3.5 es un modelo con modo de razonamiento; poner el bloque de pensamiento vacío hace que conteste directo, sin cadena de razonamiento. Así la respuesta depende solo del forward pass sobre el prompt.
- `... is called **`: el prompt termina justo antes del nombre (en markdown, negrita), así el siguiente token es el nombre. Por eso los nombres se tokenizan **sin espacio** adelante (van pegados a `**`).
- `token_ids`: tokeniza cada nombre; si da más de un token toma el **segundo**, si da uno toma ese. La idea parece ser saltear un token BOS si el tokenizer lo agrega; pero si el tokenizer no agrega BOS (Qwen normalmente no lo hace) y un nombre se partiera en dos, tomaría el segundo pedazo, que no es lo que queremos. Con estos tres nombres funciona si cada uno es un único token. Más adelante (`get_output_ids`) esto se reemplaza por un `assert` más seguro.
- Ojo: el orden de `NAMES` (Gary, Charles, Frank) no es el orden en que aparecen en el prompt (Gary, Frank, Charles). No importa, porque se imprime con el nombre al lado.

La figura `reference_game_lm.png` muestra los tres animales.

Predicción RSA para este contexto (la calculé con las mismas funciones de la notebook y la matriz correspondiente; no está en la notebook): "dog" → Gary 0.6, Frank 0.4 (porque Frank tiene "green", inequívoco); "blue" → Gary 0.6, Charles 0.4 (porque Charles tiene "cat"); "cat" → Charles 1. La notebook lo plantea como pregunta: si Alice dice "dog", ¿Gary o Frank? ¿Para cuál podría haber dicho algo más simple?

```python
utterance = "dog"
prompt = get_prompt(utterance)
with model.session(remote=True):
    with model.trace(prompt):
        logits = model.output.logits[:, -1, :].save()
```

- `model.session(remote=True)`: abre una sesión de NDIF; todo lo que va adentro (una o varias trazas) se empaqueta en **un solo request** al servidor. Es lo que hace que la ejecución sea remota.
- `model.trace(prompt)`: forward pass sobre el prompt (en el servidor).
- `model.output.logits[:, -1, :]`: logits del siguiente token, `(1, vocab)`. `.save()` hace que se devuelvan al cliente. Lo que no se guarda no viaja.
- La salida `<IPython.core.display.HTML object>` es la barra de progreso/estado de NDIF.

```python
probs = torch.softmax(logits, dim=-1)
print(f"Heard: {utterance}")
print("\t|\t".join(n.strip() for n in NAMES))
print("\t|\t".join(f"{p:.2f}" for p in probs[0, token_ids]))
```

Softmax sobre el vocabulario completo y lectura de las tres probabilidades con indexado avanzado `probs[0, token_ids]` → tensor de 3. Salida guardada: Gary 0.46, Charles 0.01, Frank 0.40.

```python
name_probs = probs[0, token_ids]
print(f"Total probability on the three names: {name_probs.sum():.2f}")
print("\t|\t".join(f"{p:.2f}" for p in name_probs / name_probs.sum()))
```

Las tres probabilidades no suman 1 porque el softmax es sobre todo el vocabulario (algo de masa va a otros tokens). Se **renormaliza** dividiendo por la suma para ver cómo reparte el modelo entre los tres animales. Salida guardada: masa total 0.87; renormalizado Gary 0.52, Charles 0.01, Frank 0.46. La notebook lo lee como "por poco, Qwen prefiere Gary", en la dirección de RSA (0.6/0.4).

Mismo código con "cat": salida guardada Charles 0.99 (es inequívoco, cualquier oyente lo resuelve). Con "blue": Gary 0.61, Charles 0.20, Frank 0.09: prefiere Gary sobre Charles, de nuevo en la dirección de RSA.

La notebook advierte: son pocos datos. No se puede saber si es inferencia pragmática o un **sesgo** hacia el nombre Gary, hacia los perros o hacia el azul. Notar que en este contexto Gary es la respuesta pragmática tanto de "dog" como de "blue", así que un sesgo pro-Gary es indistinguible de pragmática. Eso motiva la sección siguiente.

### Causal abstraction

```python
NAMES = ["Gary", "Frank", "Charles"]
OBJECTS = ["blue dog", "green dog", "blue cat"]
UTTERANCES = ["blue", "green", "dog", "cat"]

PROMPT_TEMPLATE = """... In front of Bob are {referents}. 
... she can only say {utterances}.
Alice says, \"{utterance}\".
...
Answer: the animal that Alice is most likely referring to is the one called **"""
```

- Ahora el template está **parametrizado**: los objetos, sus nombres y la lista de utterances se rellenan desde listas. Eso permite armar pares de prompts que difieren en un solo factor.
- El final cambió a "is most likely referring to is the one called **" (empuja un poco a pensar en "lo más probable").

```python
def get_prompt(utterance, names=NAMES, objects=OBJECTS, utterances=UTTERANCES):
    referents = [f"a {obj} called {name}" for obj, name in zip(objects, names)]
    referents[-1] = "and " + referents[-1]
    referents_str = ", ".join(referents)

    utterances = [f"\"{u}\"" for u in utterances]
    utterances[-1] = "or " + utterances[-1]
    utterances_str = ", ".join(utterances)

    return PROMPT_TEMPLATE.format(referents=referents_str, utterances=utterances_str, utterance=utterance)
```

- Arma "a blue dog called Gary, a green dog called Frank, and a blue cat called Charles": zip de objetos y nombres, "and " delante del último, unidos con ", ".
- Lo mismo para las utterances con comillas y "or " delante de la última.
- `format` rellena el template. La salida guardada con "dog" muestra el prompt resultante, igual al anterior salvo el final.

```python
def get_output_ids(names=NAMES):
    ids = tokenizer(names).input_ids
    assert all(len(i) == 1 for i in ids), f"pick names that are a single token: {names}"
    return [i[0] for i in ids]
```

Versión segura del `token_ids` de antes: exige que cada nombre sea **un único token** (sin espacio adelante, porque va después de `**`). El comentario da el ejemplo: Qwen3.5 parte "Eddie" en "E" + "ddie", y estaríamos leyendo P("E"), que no es la probabilidad del nombre.

```python
def print_output(prompt, names, output_ids):
    with model.session(remote=True):
        with model.trace(prompt):
            logits = model.output.logits[:, -1, :].save()
    probs = torch.softmax(logits, dim=-1)
    print("\t|\t".join(n.strip() for n in names))
    print("\t|\t".join(f"{p:.2f}" for p in probs[0, output_ids]))
```

Empaqueta lo de antes: corre remoto, saca la distribución del siguiente token y muestra las probabilidades de los nombres (sin renormalizar).

#### Experimento 1: mismo "blue", distinto contexto

```python
names = ["Gary", "Tom", "David"]
output_ids = get_output_ids(names)

source_prompt = get_prompt("blue", objects=["green dog", "blue cat", "blue dog"], names=names)
target_prompt = get_prompt("blue", objects=["green cat", "blue cat", "blue dog"], names=names)
```

- Nombres nuevos (todos de un token) para evitar el posible sesgo hacia "Gary" de antes, y Gary pasa a ser el objeto verde (el distractor que nunca es "blue").
- **Source**: Gary = green dog, Tom = blue cat, David = blue dog.
- **Target**: Gary = green **cat**, Tom = blue cat, David = blue dog.
- Única diferencia: el animal de Gary. La utterance ("blue") y los dos candidatos azules son iguales. Pero eso cambia **qué alternativa es inequívoca**:
  - En el source, "cat" solo describe a Tom → si Alice quisiera a Tom podría haber dicho "cat" → RSA L1("blue") = Tom 0.4, David 0.6.
  - En el target, "dog" solo describe a David → RSA L1("blue") = Tom 0.6, David 0.4.

  (Estos números los calculé con las funciones RSA de la notebook sobre las matrices correspondientes; la notebook no los muestra.)
- Es decir, el par está diseñado para que **la respuesta pragmática se invierta** sin cambiar la utterance ni los candidatos: lo que cambia es el razonamiento de S1 sobre alternativas.

Salidas guardadas:
- source: Gary 0.01, Tom 0.47, David 0.41
- target: Gary 0.02, Tom 0.33, David 0.54

El modelo **sí** cambia su preferencia entre source y target (es sensible al contexto), pero comparando con RSA, en ambos casos prefiere el candidato **opuesto** al que predice L1 (source: Tom sobre David; target: David sobre Tom). Esto encaja con el comentario de la notebook de que ni un 27B se comporta muy pragmáticamente. Una lectura posible (mía, no de la notebook): el modelo podría estar asociando "blue" con el candidato cuyo *otro* rasgo se comparte con el distractor, o simplemente respondiendo a otros factores superficiales del prompt; con un solo par no se puede saber.

#### La función de interchange intervention

```python
def interchange_intervention(source_prompt, target_prompt, token_indices, layers, output_ids):
    counterfactual_outputs = []
    with model.session(remote=True):
        with torch.no_grad():
            # STEP 1: source run
            with model.trace(source_prompt):
                hidden_states = {layer_idx: model.model.language_model.layers[layer_idx].output.save()
                                 for layer_idx in layers}

            # STEP 2: counterfactual runs (source -> target)
            for layer_idx in layers:
                layer_counterfactual_outputs = []
                for token_idx in token_indices:
                    with model.trace(target_prompt):
                        model.model.language_model.layers[layer_idx].output[:, token_idx, :] = hidden_states[layer_idx][:, token_idx, :]

                        counterfactual_logits = model.output.logits[:, -1, :]
                        result = nnsight.save(torch.softmax(counterfactual_logits, dim=-1)[0, output_ids].detach().cpu().tolist())
                    layer_counterfactual_outputs.append(result)
                counterfactual_outputs.append(layer_counterfactual_outputs)

            counterfactual_outputs = nnsight.save(counterfactual_outputs)

    return counterfactual_outputs
```

Nomenclatura: acá **target** es lo que en la lecture 7 se llamaba **base** (el prompt que se corre de verdad y se interviene); **source** es de donde se copian activaciones.

- `model.session(remote=True)`: toda la función (1 traza del source + `len(layers) × len(token_indices)` trazas del target) se manda como **una** sesión a NDIF. Dentro de una sesión, valores de una traza (`hidden_states`) se pueden usar en trazas posteriores sin bajarlos al cliente: todo vive en el servidor.
- `torch.no_grad()`: sin gradientes.
- **STEP 1**: un forward sobre el source guardando la salida de cada capa de `layers` en un dict `{capa: tensor (1, seq, hidden)}`. `model.model.language_model.layers`: en Qwen3.5 el modelo de lenguaje está anidado bajo `language_model` (la arquitectura tiene un envoltorio, probablemente por ser multimodal; no lo verifiqué contra el código de transformers). Cada elemento es un decoder layer y su `.output` es el residual stream después de esa capa.
- **STEP 2**: para cada capa y cada posición:
  - `with model.trace(target_prompt)`: forward sobre el target.
  - La asignación pisa el residual stream del target en esa capa y esa posición con el del source. Igual que en la lecture 7. Ojo: con `token_idx = -1` se pisa la **última** posición en ambos; como los prompts tienen el mismo largo (mismo template, los nombres y animales son de un token cada uno; asumo que se cumple), las posiciones se corresponden.
  - `counterfactual_logits`: logits finales con la intervención.
  - `softmax(...)[0, output_ids]`: las probabilidades de los 3 nombres; `.detach().cpu().tolist()` las convierte en lista de Python (en el servidor).
  - `nnsight.save(...)`: la forma de guardar objetos que no son tensores (una lista) para que vuelvan al cliente. Es la API funcional equivalente a `.save()`.
- `counterfactual_outputs` termina siendo una lista anidada `[capa][posición] → [p_nombre1, p_nombre2, p_nombre3]`, y se guarda entera.
- El comentario inicial explica una decisión de ingeniería: hay una versión más rápida que corre todas las capas como varios "invokes" de una sola traza (batching), sincronizados con barriers, pero el deployment multi-GPU de Qwen3.5-27B en NDIF falla con trazas multi-invoke en capas tardías, así que se usa la versión "lenta" de una traza por corrida. El comentario menciona `interchange_intervention_slow in utils.py`, pero ese archivo no está en el repo (en `utils/` solo hay `lecture_09_utils.py`).

```python
token_indices = [-1]
layers = list(range(0, len(model.model.language_model.layers), 5))
counterfactual_outputs = interchange_intervention(source_prompt, target_prompt, token_indices, layers, output_ids)
```

- Solo se interviene el **último token** (el `**` antes del nombre).
- Capas 0, 5, 10, ... hasta el total (cada 5 para ahorrar tiempo; no anoté el número total de capas de Qwen3.5-27B porque no lo puedo verificar sin el config).

```python
df = pd.DataFrame([c[0] for c in counterfactual_outputs], columns=names, index=layers)
px.line(df, template="simple_white", width=600, markers=True,
        labels={"index": "Layer", "value": "P(name) after patching", "variable": "Name"})
```

- `c[0]`: para cada capa, el resultado de la única posición (-1). DataFrame con filas = capas, columnas = nombres.
- Line plot: una curva por nombre, $P(\text{nombre})$ después de patchear en función de la capa.

**Cómo leer el plot** (el gráfico no está guardado como texto, así que describo qué esperar):
- En capas tempranas, patchear el último token casi no debería cambiar nada: ese token es idéntico en ambos prompts y todavía no juntó información del contexto. Las curvas deberían estar cerca de los valores del **target** (Tom 0.33, David 0.54).
- En capas tardías, el último token ya tiene "la decisión" del modelo, así que patchearlo debería llevar las curvas a los valores del **source** (Tom 0.47, David 0.41). En la última capa es casi forzoso.
- La **capa de transición** es donde la información que distingue los contextos (qué alternativa es inequívoca / qué animal es Gary) llega al último token. Si el modelo calculara algo tipo "alternativas → S1 → L1" en el último token, uno buscaría que la transición sea gradual o tenga etapas.

#### Experimento 2: mismo contexto, distinta utterance

```python
names = ["Gary", "Tom", "David"]
objects = ["green dog", "blue cat", "blue dog"]
output_ids = get_output_ids(names)

source_prompt = get_prompt("blue", objects=objects, names=names)
target_prompt = get_prompt("dog", objects=objects, names=names)
```

- Mismos objetos que el source del experimento 1 (Gary = green dog, Tom = blue cat, David = blue dog).
- Ahora lo que cambia es la **utterance**: source "blue", target "dog".
- RSA (calculado por mí con las funciones de la notebook): "blue" → Tom 0.4, David 0.6; "dog" → Gary 0.4, David 0.6 (Gary tiene "green", inequívoco).

Salidas guardadas:
- source ("blue"): Gary 0.01, Tom 0.47, David 0.41
- target ("dog"): Gary 0.27, Tom 0.04, David 0.64

Para "dog" el modelo sí va en la dirección de RSA (David > Gary). `print(target_prompt)` muestra el prompt completo para chequear.

Se corre la misma `interchange_intervention` (último token, capas cada 5) y el mismo line plot (`df_2`).

**Qué esperar / comparar:** acá el factor cambiado es la palabra dicha, que vive en una posición concreta del prompt (dentro de `Alice says, "..."`). El patcheo del último token debería mover las curvas de los valores del target ("dog": Gary/David) a los del source ("blue": Tom/David) a partir de cierta capa. Comparar la capa de transición con la del experimento 1 es la idea de "intervención por paso de inferencia": si la información de la utterance llega al último token antes o después que la información del contexto/alternativas, eso dice algo sobre el orden en que el modelo combina esos factores. La notebook no saca conclusiones explícitas sobre estos plots; queda abierto.

---

## Ojo / errores comunes / dudas

- **Los nombres tienen que ser un solo token**, y sin espacio adelante porque el prompt termina en `**`. Si no, se lee la probabilidad del primer pedazo. `get_output_ids` lo verifica con un `assert`; el `token_ids` de la primera parte no.
- **Las probabilidades no suman 1**: el softmax es sobre todo el vocabulario. Para comparar con RSA conviene renormalizar sobre los nombres (la notebook lo hace solo una vez).
- **Sesgos vs pragmática**: un solo contexto donde el referente pragmático coincide con un posible sesgo (Gary, azul, perro) no prueba nada. Los pares del experimento 1 están armados para que la respuesta RSA se invierta; ahí el modelo va en contra de RSA en los dos casos.
- **RSA simplificado**: sin $\alpha$, sin costos, prior uniforme. Los valores 0.6/0.4 dependen de esas elecciones; con $\alpha > 1$ la preferencia pragmática sería más fuerte.
- **Target vs base**: en esta notebook el prompt intervenido se llama target (en la 7 era base). Misma idea.
- **Solo se interviene la última posición** y cada 5 capas. Con eso se ve *cuándo* la información llega al último token, no *dónde* se calcula antes (podría estar en los tokens de los objetos o de la utterance). Para eso habría que barrer posiciones como en la lecture 7.
- **Alineación de posiciones**: se asume que source y target tienen el mismo número de tokens (cambian palabras de un token por otras de un token). No hay `assert` que lo verifique.
- **NDIF y sesiones**: lo que no se guarda con `.save()` / `nnsight.save()` no vuelve. Dentro de una `session` se pueden reutilizar valores entre trazas en el servidor. Las trazas multi-invoke fallan para este modelo en NDIF (por eso la versión lenta).
- **`utils.py` mencionado en un comentario no existe** en el repo.
- **`<think></think>`** desactiva el razonamiento explícito de Qwen; sin eso, el siguiente token sería el comienzo de un razonamiento, no el nombre.
- **API de nnsight 0.8** (`TransformersModel`, `nnsight.save`) difiere de la usada en la lecture 7 (0.5, `LanguageModel`, `.save()`).

---

## Resumen de una línea por sección

- **Setup**: instala dependencias, configura plotly.
- **Pragmatic inference**: reference game con 3 objetos y 4 palabras; "blue" debería apuntar al blue square porque el círculo tenía "circle".
- **RSA – L0**: normalizar filas de la matriz semántica; "blue" → 0.5/0.5.
- **RSA – S1**: normalizar columnas de L0; el hablante prefiere palabras informativas (blue circle → "circle" 0.67).
- **RSA – L1**: normalizar columnas de S1; "blue" → blue square 0.6, blue circle 0.4.
- **What about LMs**: Qwen3.5-27B vía NDIF con un prompt de juego de referencia; prefiere el referente pragmático con "dog" y "blue", pero podría ser sesgo.
- **Causal abstraction – experimento 1**: mismo "blue", cambia qué alternativa es inequívoca; la respuesta del modelo cambia pero en sentido contrario a RSA; se patchea el último token por capa para ver cuándo llega esa información.
- **Causal abstraction – experimento 2**: mismo contexto, "blue" vs "dog"; mismo patcheo para ver cuándo llega la información de la utterance y comparar con el experimento 1.
