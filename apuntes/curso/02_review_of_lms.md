# Lecture 2: Review of Language Models

> **Actualización:** el modelo se carga con `attn_implementation="eager"`. Con transformers 5 (la versión del `.venv` local), cambiar `model.config._attn_implementation` después de cargar no devuelve los mapas de atención. La notebook ya está ejecutada y tiene las respuestas escritas.

Esta notebook es un repaso de cómo funciona un language model (LM), pero pensado desde la interpretabilidad: no le interesa tanto por qué el transformer se diseñó así, sino qué función computacional cumple cada pieza. Va "pelando capas" de a una: primero el modelo como caja negra (GPT-2 generando texto), después qué entra y qué sale (tokens y logits), y al final el interior: residual stream, MLP y attention, con `nnsight` para mirar activaciones. La pregunta de fondo es: ¿por dónde viaja la información dentro del modelo y qué componente hace qué? La respuesta que deja la notebook es "las MLP guardan y procesan información de cada token, la attention es lo único que mueve información entre tokens, y todo fluye de izquierda a derecha".

---

## Ideas clave / teoría

### 1. Autoregresión: un token a la vez

Un LM base recibe una secuencia de tokens y devuelve una **distribución de probabilidad sobre el siguiente token**. Para generar texto se repite: se elige un token (con `argmax`, que se llama **greedy decoding**, o muestreando de la distribución), se agrega a la entrada y se vuelve a correr. Esto vale incluso para modelos instruct o de razonamiento; lo que cambia es qué significa "probable" para ellos (porque se entrenaron con otros objetivos además del de predecir texto).

### 2. Tokenización

El **tokenizer** parte un string en **tokens** (pedazos de palabra) y cada token tiene un **token id** único: es básicamente una tabla de búsqueda token → entero. A la lista de ids de un texto se le llama **input ids**. GPT-2 tiene un vocabulario de 50257 tokens.

Dos cosas que hay que tener claras:

- En GPT-2 (y en muchos tokenizers tipo BPE de byte) el carácter `Ġ` que aparece al imprimir tokens representa **un espacio adelante**. `'ĠTower'` es " Tower", con espacio.
- `" Paris"` y `"Paris"` son **tokens distintos** con ids distintos (6342 y 40313 en la salida guardada). Por eso un espacio al final del prompt cambia completamente lo que el modelo ve.

### 3. Logits, softmax y top-k

La salida del modelo para un input de `T` tokens es una matriz de shape `(T, V)` donde `V` es el tamaño del vocabulario. La fila `t` son los **logits** (puntajes sin normalizar) para el token que viene después de la posición `t`. Se calcula para todas las posiciones en paralelo; para predecir el siguiente token del prompt solo importa la última fila.

Los logits se convierten en probabilidades con **softmax**:

$$
p_i = \frac{e^{z_i}}{\sum_{j=1}^{V} e^{z_j}}
$$

- $z_i$ es el logit del token $i$ del vocabulario.
- $e^{z_i}$ lo vuelve positivo y exagera las diferencias (un logit un poco más grande pesa mucho más).
- El denominador suma sobre todo el vocabulario, así que $\sum_i p_i = 1$.

Softmax es monótona: el token con mayor logit es también el de mayor probabilidad. Por eso `argmax` sobre logits o sobre probabilidades da lo mismo. **Top-k** (`torch.topk`) devuelve los `k` valores más grandes y sus índices; los índices son token ids, que se pasan por `tokenizer.decode` para ver el texto.

### 4. Weights vs. activations

- **Weights / parameters**: los números fijos del modelo (las matrices de cada capa). No cambian con el input.
- **Activations / representations**: los vectores intermedios que se producen al pasar un input concreto. Cambian con cada input.

Cada activación es la salida de aplicar algún parámetro a otra activación. Casi todos los métodos de interpretabilidad del curso trabajan con activaciones. La analogía de la notebook: los parámetros son como las neuronas del cerebro (estables en el día), las activaciones son lo que "se prende" ante un estímulo (como en un fMRI).

### 5. Residual stream

Es la activación que pasa de una capa a la siguiente. Tiene shape `(batch, tokens, hidden_dim)`; en SmolLM2-135M `hidden_dim = 576`. Cada capa lee del residual stream y le **suma** lo que computa (por eso "residual": hay conexiones residuales `x + f(x)`). Dos propiedades importantes:

- **La información fluye de abajo hacia arriba (capas) y de izquierda a derecha (tokens).** La activación del token `t` en la capa `l` solo puede depender de activaciones en tokens `t' ≤ t` y capas `l' < l`. Cambiar tokens *posteriores* a "Paris" no cambia en nada la representación de "Paris"; cambiar tokens *anteriores* sí.
- **Las activaciones son embeddings contextuales**: la representación de un token refleja su significado en ese contexto. En capa 0 una palabra tiene siempre el mismo vector; más arriba se diferencia según el contexto.

### 6. Estructura de una capa (Llama / SmolLM2)

Cada `LlamaDecoderLayer` hace, a grandes rasgos (pre-norm):

```
h = x + Attention(RMSNorm(x))
y = h + MLP(RMSNorm(h))
```

- **RMSNorm** (`input_layernorm`, `post_attention_layernorm`): normaliza el vector dividiéndolo por su norma RMS y lo escala por un vector aprendido de 576 números. Ojo: aunque se llame `post_attention_layernorm`, en Llama se aplica *antes* de la MLP (es la norma de la entrada de la MLP).
- **MLP**: en Llama es una **SwiGLU**: `down_proj( SiLU(gate_proj(x)) * up_proj(x) )`. `gate_proj` y `up_proj` llevan de 576 a 1536 dimensiones, se multiplican elemento a elemento (una pasa por la no linealidad SiLU, la otra no) y `down_proj` vuelve a 576. Las 1536 dimensiones intermedias son lo que se suele llamar "neuronas" de la MLP.
- **Attention** (`self_attn`): proyecciones `q_proj`, `k_proj`, `v_proj`, `o_proj`.

**MLP vs attention**, que es la idea central:

- La **MLP opera token por token**: el vector de la posición `t` entra, sale otro vector para la posición `t`. Nunca mira otras posiciones. Concentra la mayoría de los parámetros y es donde se cree que vive el conocimiento factual (la notebook cita a Meng et al. 2022, ROME, donde editar MLPs cambia los hechos que recuerda el modelo).
- La **attention es el único componente que mueve información entre tokens.** Si el modelo usa algo que apareció antes en el contexto, eso pasó por attention. La notebook lo resume como: MLP = memoria de largo plazo, attention = memoria de trabajo.

### 7. Atención: queries, keys, values

Para cada token, la attention calcula tres proyecciones de su activación:

- **query** $q_j$: "qué estoy buscando" (desde el token que mira).
- **key** $k_i$: "qué ofrezco" (desde el token que es mirado).
- **value** $v_i$: "qué información entrego si me miran".

La notebook escribe el mapa de atención como

$$
QK_{i,j} = k_i \cdot q_j
$$

con $i$ = token mirado (key) y $j$ = token que mira (query). Después se aplica softmax "por columna" (sobre las keys $i$, para cada query $j$) y se hace un promedio ponderado de values:

$$
h_j = \sum_{i \le j} \mathrm{softmax}(QK)_{i,j}\, v_i
$$

- $h_j$: salida de la attention para el token $j$ (antes de `o_proj`).
- La suma es solo sobre $i \le j$: es la **máscara causal**, ningún token mira al futuro. Por eso "la mitad de abajo" del mapa (según cómo se dibuje) queda vacía.
- Los pesos $\mathrm{softmax}(QK)_{\cdot, j}$ suman 1: cada query reparte "una unidad" de atención entre los tokens anteriores (incluido él mismo).

Detalles que la fórmula de la notebook omite pero están en el código (`source` de la celda 131): los scores se multiplican por `scaling` (que en la implementación estándar es $1/\sqrt{d_{head}}$), las queries y keys reciben **RoPE** (`apply_rotary_pos_emb`, codificación de posición por rotación) antes del producto punto, y al final la concatenación de todas las cabezas pasa por `o_proj`. Además, la notebook dice "cuanto más cerca en similitud coseno", pero lo que se calcula es un **producto punto**, no coseno (no se normalizan los vectores).

Un conjunto (q, k, v) con su mapa de atención es una **attention head**. Cada capa tiene varias en paralelo (9 en SmolLM2-135M), cada una puede mover un tipo distinto de información.

**GQA (grouped-query attention).** En el print del modelo, `q_proj` sale a 576 pero `k_proj` y `v_proj` salen a 192. Con 9 heads, `head_dim = 576 / 9 = 64`, así que 192 = 3 × 64: hay 3 cabezas de key/value compartidas entre las 9 de query (cada key/value la usan 3 queries). Esto es deducción mía a partir de los shapes; la notebook no lo menciona, pero explica por qué las proyecciones K y V son más chicas.

### 8. QK circuit, OV circuit, skip-trigrams (Elhage et al. 2021)

En el marco de "A Mathematical Framework for Transformer Circuits", cada head se separa en dos circuitos:

- **QK circuit**: decide *a dónde* mira cada token (viene de $W_Q$ y $W_K$).
- **OV circuit**: decide *qué* se escribe en el residual stream una vez que mira (viene de $W_V$ y $W_O$).

Leído así, una head de una capa implementa un **skip-trigram**: "si estoy en el token B y más atrás hubo un A, aumentar la probabilidad de C".

### 9. Induction heads

Una **induction head** busca en el contexto una aparición anterior del token actual, atiende al token que vino *después* de esa aparición y lo copia como predicción. Patrón: `[A][B] ... [A] → predice [B]`. Requiere dos heads en capas distintas que se componen (una "previous-token head" que escribe en cada posición cuál era el token anterior, y la induction head que usa esa info en su key). Se cree que es un mecanismo central del in-context learning. La notebook aclara que un patrón de **offset fijo** (cada token mira al que está 2 posiciones atrás) **no** es una induction head, porque la induction head matchea por *contenido*, no por posición.

### 10. Attention sinks

Una head es un **attention sink** cuando casi todos los tokens ponen casi toda su atención en el primer token (o en un token de padding/BOS). Como softmax obliga a repartir exactamente 1 de atención, una head que "no quiere hacer nada" en cierto contexto necesita un lugar donde tirar esa atención sin que afecte la salida; el primer token (que todos pueden ver) suele cumplir ese rol, típicamente con un value que aporta poco. Es una forma de implementar un "no-op".

---

## Recorrido por la notebook

### 0. Setup (celda 5)

```python
try:
    import google.colab
    is_colab = True
except ImportError:
    is_colab = False

if is_colab:
    pio.renderers.default = "colab"
    !git clone https://github.com/Maigolinox/cs221m-course-terron-notes.git
    %cd cs221m-course-terron-notes
    !pip install -r requirements/02_review_of_lms.txt
else:
    pio.renderers.default = "plotly_mimetype+png"
    !uv sync
    !plotly_get_chrome -y
```

Detecta si corre en Colab intentando importar `google.colab`. En Colab clona el repo e instala las dependencias de `requirements/02_review_of_lms.txt` (que fija `nnsight==0.5.15`, `transformers==4.57.6`, `accelerate==1.12.0`). Fuera de Colab usa `uv sync` con el `pyproject.toml` del repo. El renderer de plotly `"plotly_mimetype+png"` guarda los gráficos también como PNG (por eso se ven las imágenes en la notebook guardada); `plotly_get_chrome` instala el Chrome que usa kaleido para exportar PNG.

### 1. Talking to language models

#### Cargar GPT-2 como pipeline (celda 17)

```python
from transformers import pipeline
gpt2 = pipeline("text-generation", model="gpt2", device_map='auto')
```

`pipeline` de 🤗 Transformers empaqueta tokenizer + modelo + bucle de generación en una sola función. `"text-generation"` es la tarea; `device_map='auto'` pone el modelo en GPU si hay. Este es el nivel "caja negra": texto entra, texto sale.

#### Generar (celda 19)

```python
output = gpt2(
    'Once upon a time',
    max_new_tokens=20,
    do_sample=False
)
print(output[0]['generated_text'])
```

- `max_new_tokens=20`: genera 20 tokens nuevos (no palabras).
- `do_sample=False`: greedy, en cada paso toma el `argmax`. Hace la salida determinista.
- La salida es una lista de dicts; `generated_text` incluye el prompt más lo generado.

La salida guardada: "Once upon a time, the world was a place of great beauty and great danger. The world was a place of great". Ya se ve un rasgo típico del greedy decoding en modelos chicos: repetición ("The world was a place of great").

#### Ejercicio 1 (celda 22, `# TERRON`)

```python
prompts = [
    "The capital of France is",              # hechos del mundo
    "The Eiffel Tower is located in",        # otro hecho conocido
    "2 + 2 =",                               # matemática básica
    "17 * 23 =",                             # matemática un poco más difícil
    "Translate to Spanish: I like cats ->",  # seguir instrucciones / traducir
    "Q: What is the opposite of hot? A:",   # formato pregunta-respuesta
]

for prompt in prompts:
    output = gpt2(prompt, max_new_tokens=20, do_sample=False)
    print(repr(output[0]['generated_text']))
    print('-' * 60)
```

Recorre una lista de prompts que tocan distintas capacidades: conocimiento factual (dos prompts), aritmética fácil y difícil, seguir una instrucción (traducir) y un formato Q/A. Usa greedy para que el resultado sea reproducible y `repr` para ver saltos de línea y espacios tal cual. Esta celda no tiene salida guardada, así que no sé qué responde GPT-2 en cada caso.

Qué mirar para responder la pregunta de la celda 23 (3 capacidades, 3 fallas):

- Si completa los hechos con la entidad correcta en la primera palabra generada, o si se va por las ramas.
- Si la aritmética da el número correcto; en la multiplicación, si da un número plausible pero incorrecto (eso indica que no está calculando).
- Si "sigue la instrucción" o simplemente continúa el texto como si fuera un documento (GPT-2 es un modelo base, no fue entrenado para seguir instrucciones).
- Si respeta el formato Q/A (eso es pattern matching del formato, algo que los modelos base suelen hacer bien) aunque el contenido sea malo.
- Repeticiones y loops, que con greedy son frecuentes.

Para comparar con GPT-5: pensar en instruction tuning/RLHF, tamaño, y datos de entrenamiento.

#### Liberar memoria (celda 25)

`del gpt2` borra la referencia para que el pipeline pueda ser recolectado y no ocupe GPU.

### Tokens y logits

#### Cargar tokenizer y modelo por separado (celda 30)

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("gpt2").to("cuda")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
```

- `AutoModelForCausalLM`: carga la arquitectura correcta para un modelo causal (de izquierda a derecha) con su cabeza de LM. `.to("cuda")` lo mueve a GPU (sin GPU esta línea falla).
- `AutoTokenizer`: carga el tokenizer que corresponde a `"gpt2"`.

#### Tokenizar (celda 32)

```python
tokens = tokenizer.tokenize("The Eiffel Tower is in the city of")
token_ids = tokenizer("The Eiffel Tower is in the city of")["input_ids"]

print("Vocabulary size of our tokenizer:", tokenizer.vocab_size)
print("Number of tokens in our input:", len(token_ids))
print("Tokens:", tokens)
print("Token ids:", token_ids)
```

- `tokenizer.tokenize(text)`: devuelve los tokens como strings (sin convertir a ids).
- `tokenizer(text)`: devuelve un dict con `input_ids` (y `attention_mask`). `["input_ids"]` es la lista de enteros.
- `tokenizer.vocab_size`: 50257.

Salida guardada: 10 tokens, `['The', 'ĠE', 'iff', 'el', 'ĠTower', 'Ġis', 'Ġin', 'Ġthe', 'Ġcity', 'Ġof']` con ids `[464, 412, 733, 417, 8765, 318, 287, 262, 1748, 286]`. Notar que "Eiffel" se parte en 3 tokens (`ĠE`, `iff`, `el`): las palabras poco frecuentes se dividen en subpalabras. GPT-2 no agrega token BOS automáticamente, por eso el primer token es directamente `The`.

#### Forward pass y shape de logits (celda 35)

```python
import torch
input_ids = tokenizer("The Eiffel Tower is in the city of", return_tensors="pt")["input_ids"]

with torch.no_grad():
    outputs = model(input_ids=input_ids.to(model.device)).logits[0]

outputs.shape
```

- `return_tensors="pt"`: devuelve tensores de PyTorch en vez de listas. `input_ids` tiene shape `(1, 10)`: una dimensión de batch al frente.
- `torch.no_grad()`: desactiva el cálculo de gradientes (no vamos a entrenar; ahorra memoria y tiempo).
- `input_ids.to(model.device)`: el input tiene que estar en el mismo device que el modelo.
- `model(...)` devuelve un objeto con `.logits` de shape `(1, 10, 50257)`. `[0]` saca la dimensión de batch → `(10, 50257)`.

Salida guardada: `torch.Size([10, 50257])` = (número de tokens del input × tamaño del vocabulario). Hay una predicción por posición porque el transformer procesa todas las posiciones en paralelo (y con la máscara causal, la fila `t` solo usa tokens `≤ t`).

#### Softmax y gráfico de la distribución (celda 38)

```python
next_token_probability = outputs[-1]
next_token_probability = next_token_probability.softmax(dim=-1)
fig = px.line(x=range(len(next_token_probability)),
              y=next_token_probability.cpu().numpy(), ...)
```

- `outputs[-1]`: la fila de la última posición, shape `(50257,)`. Son los logits del token que sigue a "of". (El comentario del código dice `(50527,)`, es un typo: es 50257.)
- `.softmax(dim=-1)`: softmax sobre el vocabulario; ahora suma 1.
- `.cpu().numpy()`: plotly necesita un array de numpy en CPU.

El resto es estética del gráfico. El eje x no tiene significado (los ids son arbitrarios); lo que se ve es que casi toda la masa está en unos pocos picos. (El título dice "The Eiffel Tower is in ___" aunque el prompt real termina en "the city of".)

#### Top-3 (celda 40)

```python
most_likely_tokens = torch.topk(next_token_probability, 3, sorted=True)
for v, i in zip(most_likely_tokens.values, most_likely_tokens.indices):
    print(f'{tokenizer.decode(i.item())} ({v.item():.2f})')
```

- `torch.topk(x, 3, sorted=True)`: devuelve un namedtuple con `.values` (las 3 probabilidades más altas, ordenadas de mayor a menor) e `.indices` (sus posiciones, que son token ids).
- `i.item()`: pasa el tensor de un elemento a int de Python.
- `tokenizer.decode(id)`: id → texto.

Salida guardada: ` Paris (0.07)`, ` London (0.06)`, ` New (0.03)`. Vale la pena notar lo baja que es la probabilidad de Paris (7%) y lo cerca que está London: GPT-2 small no está nada seguro. " New" es el comienzo de "New York" (un solo token no puede ser toda la ciudad).

#### Argmax (celda 42)

```python
output_token = tokenizer.decode(next_token_probability.argmax(dim=-1))
```

`argmax` devuelve el índice del valor máximo: el token greedy. Salida guardada: `" Paris"`, con espacio adelante.

#### Paris con y sin espacio (celda 44)

```python
token_ids_no_space = tokenizer(" Paris")["input_ids"]
token_ids_with_space = tokenizer("Paris")["input_ids"]
print("Paris with a space:", token_ids_no_space)
print("Paris without a space:", token_ids_with_space)
```

Ojo con los nombres: la variable `token_ids_no_space` es la de `" Paris"` (que *sí* tiene espacio) y `token_ids_with_space` la de `"Paris"`. Los nombres están invertidos, pero los prints son correctos: con espacio → `[6342]`, sin espacio → `[40313]`. Son tokens completamente distintos para el modelo.

#### Ejercicio 2 (celda 47, `# TERRON`)

```python
input_ids = tokenizer("The Eiffel Tower is in the city of ", return_tensors="pt")["input_ids"]

with torch.no_grad():
    outputs = model(input_ids=input_ids.to(model.device)).logits[0]

next_token_probability = outputs[-1].softmax(dim=-1)

most_likely_tokens = torch.topk(next_token_probability, 3, sorted=True)
for v, i in zip(most_likely_tokens.values, most_likely_tokens.indices):
    print(f'{tokenizer.decode(i.item())!r} ({v.item():.2f})')
```

Es exactamente el pipeline anterior (tokenizar → forward sin gradientes → última fila → softmax → topk) con el prompt terminado en espacio. Detalles:

- Sobrescribe `input_ids`, `outputs` y `next_token_probability`, así que si después se vuelve a correr una celda anterior que los usa, va a mostrar los resultados del prompt con espacio.
- `!r` en el f-string imprime el `repr` del token decodificado, con comillas, para que se vea si empieza con espacio o no.

No hay salida guardada. Qué observar: el espacio final se tokeniza como un token propio (un espacio suelto), y en el entrenamiento casi nunca aparece " " seguido de una palabra, porque las palabras ya traen su espacio pegado. Hay que mirar si los top-3 son ciudades sin espacio adelante, si la probabilidad queda más repartida (más baja) que antes, o si aparecen tokens raros. Comparar con ` Paris (0.07)` de la celda 40 y razonar a partir de cómo funciona la tokenización.

#### Takeaways y `del model` (celdas 50-53)

Input = token ids, output = distribución sobre el siguiente token; detalles como un espacio cambian los tokens y por lo tanto la predicción. `del model` libera GPT-2.

### 2. The anatomy of language models

#### Cargar SmolLM2 con nnsight (celda 63)

```python
from nnsight import LanguageModel
model = LanguageModel("HuggingFaceTB/SmolLM2-135M")
```

**nnsight** es una librería para inspeccionar e intervenir los internals de modelos de PyTorch. `LanguageModel` envuelve un modelo de Hugging Face (y su tokenizer, accesible como `model.tokenizer`). Por defecto carga los pesos de forma perezosa (cuando se usa por primera vez), por eso la primera traza muestra logs de carga que se limpian con `clear_output()`. SmolLM2-135M es un modelo de arquitectura Llama de 135M de parámetros.

#### Imprimir el modelo (celdas 69 y 81)

La salida guardada muestra la estructura (la copio resumida):

```
LlamaForCausalLM(
  (model): LlamaModel(
    (embed_tokens): Embedding(49152, 576)
    (layers): 30 x LlamaDecoderLayer(
        (self_attn): q_proj 576→576, k_proj 576→192, v_proj 576→192, o_proj 576→576
        (mlp): gate_proj 576→1536, up_proj 576→1536, down_proj 1536→576, act_fn SiLU
        (input_layernorm), (post_attention_layernorm): RMSNorm(576)
    )
    (norm): RMSNorm(576)
    (rotary_emb): LlamaRotaryEmbedding()
  )
  (lm_head): Linear(576 → 49152)
)
```

Cómo leerlo:

- `embed_tokens`: tabla de 49152 (vocabulario de SmolLM2, distinto al de GPT-2) × 576 (hidden dim). Convierte cada token id en un vector de 576.
- `layers`: 30 bloques idénticos en estructura (pesos distintos).
- `norm`: RMSNorm final antes de la cabeza.
- `rotary_emb`: calcula los senos/cosenos de RoPE; no tiene parámetros entrenables.
- `lm_head`: proyecta el vector final de 576 a 49152 logits. No confundir con las attention heads.
- `bias=False` en todas las Linear: no hay términos de sesgo, solo matrices.
- `generator`/`streamer`: módulos que agrega nnsight para poder trazar `generate`.

#### La API básica de nnsight (celda 71)

```python
with model.trace("<INPUT>"):
  component = model.thing.we.want.to.inspect
  thing_to_save = component.output.save()
```

- `model.trace(input)`: abre un contexto. Todo lo que se escribe adentro se registra y se ejecuta durante el forward pass con ese input. Al salir del `with`, el forward ya corrió.
- Se navega a los módulos con la misma ruta de atributos que en el print (`model.model.layers[10].mlp`).
- `.output`: la salida del módulo en el forward; `.input`: su entrada; `.weight`: el parámetro.
- `.save()`: marca el valor para que siga disponible **después** de cerrar el contexto. Sin `.save()`, el valor se descarta. En nnsight 0.5 lo que queda guardado es el tensor real.

#### Pesos: estáticos (celdas 72 y 75)

```python
with model.trace("This is an example"):
  layer_10_mlp = model.model.layers[10].mlp.gate_proj
  layer_10_mlp_weight = layer_10_mlp.weight.save()
```

- `model.model.layers[10].mlp.gate_proj`: la Linear de 576→1536 de la MLP de la capa 10.
- `.weight`: su matriz. En PyTorch, `nn.Linear(in, out)` guarda el peso con shape `(out, in)`, así que es `(1536, 576)`.

La salida guardada es un `Parameter` en `bfloat16` (el dtype en que se cargó el modelo). La celda 75 repite con otro input ("An entirely different input!") y la salida guardada es idéntica: **los pesos no dependen del input**.

#### Activaciones: dependen del input (celda 77)

```python
with model.trace("This is an example"):
  layer_10_mlp = model.model.layers[10].mlp.gate_proj.output.save()

with model.trace("An entire different input!"):
  layer_10_mlp_1 = model.model.layers[10].mlp.gate_proj.output.save()
```

Ahora se guarda `.output` de `gate_proj`, shape `(1, num_tokens, 1536)`. En la salida guardada la primera tiene 4 filas (4 tokens: "This", " is", " an", " example") y la segunda 5, con valores distintos. Notar que el nombre `layer_10_mlp` se reutiliza: en la celda 72 era el módulo, acá es un tensor.

#### Residual stream: shape (celda 85)

```python
with model.trace("The city Paris is in France"):
  layer_10_output = model.model.layers[10].output.save()
layer_10_output.shape
```

`layers[10].output` es la salida del bloque entero = residual stream después de la capa 10. Salida guardada: `torch.Size([1, 6, 576])` = (batch, tokens, hidden). En esta versión de transformers la capa devuelve directamente el tensor (en versiones viejas devolvía una tupla y había que hacer `output[0]`); por eso se puede indexar `[0, 2]` directo más abajo. Acá los 6 tokens coinciden con las 6 palabras, y no hay BOS (si lo hubiera, serían 7).

#### Activación de "Paris" (celda 88)

```python
paris_activation = layer_10_output[0, 2]
paris_activation[:20]
```

`[0, 2]`: batch 0, posición 2 ("The"=0, " city"=1, " Paris"=2). Resultado: vector de 576; se muestran los primeros 20. Los valores sueltos no tienen interpretación obvia; eso es justamente el problema que ataca el resto del curso.

#### Autoregresión en acción (celdas 90 y 92)

```python
with model.trace("The city Paris is in France"):
  paris_original = model.model.layers[10].output[0, 2].save()

with model.trace("The city Paris is in London"):
  paris_london = model.model.layers[10].output[0, 2].save()

with model.trace("The beautiful Paris is in France"):
  paris_beautiful = model.model.layers[10].output[0, 2].save()
```

Se puede indexar dentro del trace (`.output[0, 2]`) y guardar solo ese pedazo. Tres frases:

1. Original.
2. Cambia una palabra **después** de Paris.
3. Cambia una palabra **antes** de Paris.

```python
print("Changing tokens AFTER Paris:", torch.equal(paris_original, paris_london))
print("Changing tokens BEFORE Paris:", torch.equal(paris_original, paris_beautiful))
```

`torch.equal` chequea igualdad exacta de todos los elementos. Salida guardada: `True` y `False`. Cambiar el futuro no afecta en absolutamente nada (bit a bit) a la representación de Paris; cambiar el pasado sí. Esto es la máscara causal de la attention. Ojo: es igualdad exacta porque el cómputo de las primeras posiciones es idéntico; con padding o batch distinto podrían aparecer diferencias numéricas mínimas.

La notebook saca dos conclusiones: la información fluye de izquierda a derecha (cualquier activación en `(t, l)` se puede atribuir a activaciones en `t' ≤ t`, `l' < l`, base de la causal mediation de la Lecture 7), y el contexto previo moldea las activaciones (embeddings contextuales).

#### "break" polisémico a través de las capas (celda 95)

```python
with model.trace("The package is very fragile - it will break easily"):
  break_one = [layer.output[0, -2] for layer in model.model.layers].save()

with model.trace("If you throw the ball at the window you might break it"):
  break_two = [layer.output[0, -2] for layer in model.model.layers].save()

with model.trace("This relationship is hard and I think we should break up"):
  break_three = [layer.output[0, -2] for layer in model.model.layers].save()
```

- La list comprehension recorre las 30 capas y toma, de cada una, el vector en la posición `-2` (el penúltimo token, que en las tres frases es " break": "break easily", "break it", "break up").
- `.save()` sobre la lista entera: en la versión de nnsight que usa el curso funciona (la celda tiene gráfico guardado). Resultado: lista de 30 tensores de shape `(576,)`.

```python
cosine_similarities_one_vs_two = [
    torch.nn.functional.cosine_similarity(act_one, act_two, dim=0).item()
    for act_one, act_two in zip(break_one, break_two)
]
```

**Similitud coseno**: $\cos(a, b) = \frac{a \cdot b}{\|a\|\|b\|}$, entre -1 y 1; mide si dos vectores apuntan en la misma dirección sin importar su norma. `dim=0` porque los vectores son 1D. Se calcula capa por capa: break1 vs break2 (dos sentidos físicos) y break1 vs break3 (físico vs figurado). El resto de la celda es plotly: dos líneas, eje x = capa, eje y = coseno.

Lo que la notebook dice que se ve en el gráfico:

- **Capa 0: todas iguales** (similitud alta), porque todavía es casi solo el embedding de la palabra.
- **Al final: bastante distintas**, porque las últimas capas se especializan en predecir el siguiente token, y el siguiente token es distinto en cada frase.
- **En el medio (alrededor de la capa 10) aparece la semántica**: break1 (intransitivo, "se rompe") y break2 (transitivo, "rompe la ventana") difieren en sintaxis pero comparten significado, mientras break1 y break3 ("break up") difieren en significado. Ahí es donde más se separan las curvas.

Detalle técnico: `layer.output` de la capa 0 ya es *después* de la primera capa (incluye una attention), no el embedding puro; aun así está muy cerca del embedding.

#### Ejercicio 3 (celda 99, `# TERRON`)

Pide encontrar un uso de "break" cuya representación en la **capa 10** tenga similitud coseno entre 0.5 y 0.8 con los otros dos usos ("break the window" y "break up").

```python
with model.trace("If you throw the ball at the window you might break it"):
  break_window = model.model.layers[10].output[0, -2].save()

with model.trace("This relationship is hard and I think we should break up"):
  break_up = model.model.layers[10].output[0, -2].save()
```

Las dos referencias, solo capa 10, posición `-2`.

```python
candidates = [
    ("He promised he would never break the law", -3),            # violar una regla
    ("I worked all morning so now I need to take a break", -1),  # break como sustantivo (descanso)
    ("The news anchor was the first to break the story", -3),    # dar a conocer una noticia
    ("She trained for years to finally break the record", -3),   # superar una marca
]
```

Lista de (frase, posición de "break" contada desde el final). La posición cambia según cuántos tokens hay después: "break the law" → `-3`; "a break" al final → `-1`. Esto supone que cada palabra posterior es un solo token, por eso después hay un chequeo.

```python
for sentence, pos in candidates:
  with model.trace(sentence):
    your_break = model.model.layers[10].output[0, pos].save()

  sim_window = torch.nn.functional.cosine_similarity(break_window, your_break, dim=0).item()
  sim_up = torch.nn.functional.cosine_similarity(break_up, your_break, dim=0).item()
  token_at_pos = model.tokenizer.tokenize(sentence)[pos]
  print(f"{sentence!r} -> token {token_at_pos!r}: window={sim_window:.3f}, up={sim_up:.3f}")

  if 0.5 <= sim_window <= 0.8 and 0.5 <= sim_up <= 0.8:
    print("  ^ esta cumple el rango, me quedo con esta")
    break
```

- Por cada candidata: un trace, guarda el vector de la capa 10 en la posición de "break" (la capa se deja en 10 como pide la consigna).
- Calcula coseno contra las dos referencias.
- `model.tokenizer.tokenize(sentence)[pos]`: chequeo de que en esa posición está realmente el token "break" (debería imprimir algo como `'Ġbreak'`). Esto funciona porque, como vimos, SmolLM2 no agrega BOS, así que los tokens de `tokenize` coinciden posición a posición con los del trace. Además, indexar desde el final hace que un BOS al principio no importaría.
- Si ambas similitudes caen en [0.5, 0.8], imprime y sale del loop. Ojo con el `break` de Python: es la palabra clave que corta el `for`, no tiene nada que ver con la palabra "break" del ejercicio.
- Si ninguna cumple, el loop termina sin cortar y `your_break` queda con la **última** candidata; los prints de abajo (código original de la consigna) mostrarían esa, que no cumple el rango. Hay que leer la salida de las líneas del loop para saber cuál cumplió.

No hay salida guardada. Para responder "por qué tiene esa similitud": pensar en qué comparte cada sentido con "romper una ventana" (verbo transitivo con objeto, acción de romper algo) y con "break up" (sentido figurado/abstracto, terminar algo), y que la capa 10 es donde, según la notebook, mejor se refleja la semántica. Un sentido que esté "en el medio" de los dos (por ejemplo, abstracto pero transitivo, o un sustantivo) es razonable que dé similitudes intermedias. El paper de la pista (Findings EACL 2023) estudia sentidos de verbos en representaciones contextuales.

### The MLP layer

#### Capa 10 por dentro (celda 106)

Print de `model.model.layers[10]`: la misma estructura que arriba. Sirve de referencia para los conteos.

#### Contar parámetros de la MLP (celda 110)

```python
mlp_components = [
    parameter for name, parameter in model.model.layers[10].named_parameters()
    if "mlp" in name
]
num_parameters = sum(p.numel() for p in mlp_components)
```

- `named_parameters()`: generador de pares `(nombre, tensor)` de todos los parámetros del módulo, con nombres tipo `"mlp.gate_proj.weight"`, `"self_attn.q_proj.weight"`, `"input_layernorm.weight"`.
- Filtra los que tienen `"mlp"` en el nombre.
- `p.numel()`: número de elementos del tensor.

Salida guardada: **2.654.208**. Se verifica a mano: tres matrices de 576 × 1536 → $3 \times 576 \times 1536 = 3 \times 884.736 = 2.654.208$. Sin biases.

#### Contar parámetros de attention (celda 112)

Mismo código con `"attn"`. Salida guardada: **884.736**. A mano:

| Proyección | Shape | Parámetros |
|---|---|---|
| `q_proj` | 576 × 576 | 331.776 |
| `k_proj` | 576 × 192 | 110.592 |
| `v_proj` | 576 × 192 | 110.592 |
| `o_proj` | 576 × 576 | 331.776 |
| **Total** | | **884.736** |

El filtro `"attn"` agarra `self_attn.*` pero no las normas: `post_attention_layernorm` contiene "attention", no la subcadena "attn" (después de "att" viene "e"). Las normas (576 parámetros cada una) quedan fuera de los dos conteos. MLP / attention = 2.654.208 / 884.736 = 3 exacto, que es lo que dice la notebook. Si no hubiera GQA (K y V de 576×576), la attention tendría $4 \times 331.776 = 1.327.104$ parámetros y la razón sería 2.

#### "Sparsity" de la MLP (celda 115)

```python
with model.trace("The city Paris is in France"):
  mlp_activations = [layer.mlp.output[0, 2] for layer in model.model.layers].save()

fig = ff.create_distplot(
    [[a.item() for mlp_activation in mlp_activations for a in mlp_activation]],
    group_labels=['dist_plot'],
    show_hist=False,
)
print("Number of neurons of MLP activation in a single layer:", len(mlp_activations[0]))
```

- Para cada una de las 30 capas guarda la salida de la MLP en la posición 2 (" Paris"). Cada una es un vector de 576.
- La doble comprensión aplana todo: 30 × 576 = 17.280 números, cada uno pasado a float con `.item()` (lento pero da igual para esto).
- `ff.create_distplot(..., show_hist=False)`: dibuja una estimación de densidad (KDE) de esos valores.
- Salida guardada del print: 576.

La notebook observa que casi todos los valores están cerca de 0 y hay uno por encima de 300. Interpretación que propone: si una "neurona" se prende solo con ciertas entradas (por ejemplo, solo con Paris), podría estar guardando un hecho; eso cuadra con la idea de que las MLP hacen factual recall. Y aclara que un valor grande no implica efecto causal (eso se mide con intervenciones, más adelante en el curso).

Ojo con un matiz: `mlp.output` es la salida de `down_proj`, que vive en el espacio del residual stream (576), no las 1536 "neuronas" intermedias de la MLP. Las neuronas en sentido estricto serían la entrada a `down_proj` (`act_fn(gate_proj(x)) * up_proj(x)`). Así que el print "Number of neurons... 576" habla en realidad de dimensiones del residual stream. Además, valores enormes y aislados en una dimensión fija del residual stream son un fenómeno conocido en LLMs ("massive activations"), que no necesariamente tiene que ver con Paris. La notebook no distingue esto.

#### Ejercicio 4 (celda 120, `# TERRON`)

Pide el porcentaje de parámetros del modelo que están en MLPs.

```python
all_params = list(model.named_parameters())

total_parameters = sum(p.numel() for name, p in all_params)
mlp_parameters = sum(p.numel() for name, p in all_params if "mlp" in name)
attn_parameters = sum(p.numel() for name, p in all_params if "attn" in name)
```

- `model.named_parameters()` sobre el `LanguageModel` de nnsight recorre todo el modelo envuelto. Se convierte a lista porque se recorre varias veces (un generador se agotaría después del primer `sum`).
- Tres sumas: total, las de "mlp" (todas las capas) y las de "attn".

```python
print(f"MLP percentage: {100 * mlp_parameters / total_parameters:.1f}%")

non_embedding = sum(p.numel() for name, p in all_params if "embed" not in name and "lm_head" not in name)
print(f"MLP percentage (without embeddings): {100 * mlp_parameters / non_embedding:.1f}%")
```

- Primer porcentaje: sobre el total.
- Segundo: sobre los parámetros que no son embeddings ni `lm_head`. La razón (comentario del código): en un modelo tan chico, la tabla de embeddings es una fracción grande del total y distorsiona la comparación. En la literatura de scaling laws se suele reportar "non-embedding parameters" justamente por esto.

No hay salida guardada. Para chequear lo que debería salir, se puede armar la cuenta con lo que ya sabemos:

- MLP: 30 capas × 2.654.208.
- Attention: 30 capas × 884.736.
- Normas: 2 por capa × 30 capas × 576, más la norma final de 576.
- Embedding: 49152 × 576.
- `lm_head`: 576 × 49152, **pero** SmolLM2 usa *tied embeddings* (la `lm_head` comparte la matriz con `embed_tokens`), y `named_parameters()` no repite parámetros compartidos, así que no debería aparecer por separado. Esto no está verificado en la notebook (no hay salida); si los pesos no estuvieran atados, el total aumentaría en otros 28 millones. El comentario del código también asume que están atados. Para comprobarlo: imprimir los nombres de `all_params` y ver si hay alguno con `lm_head`, o mirar `model.config.tie_word_embeddings`.

Con eso se puede verificar que el total dé cerca de los 135M que dice el nombre del modelo y sacar los dos porcentajes. Para la segunda parte de la pregunta (qué otras capacidades necesitan muchos parámetros además del factual recall), pensar en lo que una MLP hace token por token: transformaciones no lineales de features (por ejemplo, combinar "es plural" + "es sustantivo"), aritmética, conocimiento lingüístico (morfología, sintaxis), traducir entre representaciones internas y el vocabulario de salida, etc.

### Attention

#### El código fuente de la attention (celda 131)

```python
print(model.model.layers[10].self_attn.source)
```

`.source` es una funcionalidad de nnsight 0.5: muestra el `forward` del módulo con cada llamada a función etiquetada (`self_q_proj_0`, `apply_rotary_pos_emb_0`, `attention_interface_0`, `self_o_proj_0`...). Esas etiquetas son accesibles como atributos para leer o modificar el resultado de esa operación específica dentro del forward. Lo que muestra la salida guardada, línea por línea:

- Líneas 8-9: `input_shape = (batch, seq)`, `hidden_shape = (batch, seq, -1, head_dim)`.
- Líneas 11-13: `q_proj`, `k_proj`, `v_proj` aplicados al hidden state, `.view(hidden_shape)` separa en heads, `.transpose(1, 2)` deja `(batch, heads, seq, head_dim)`. Queries: 9 heads; keys/values: 3 heads (GQA).
- Línea 16: RoPE sobre queries y keys.
- Líneas 18-21: KV cache (solo relevante al generar).
- Líneas 23-25: elige la implementación de attention según `config._attn_implementation` (eager, sdpa, flash...).
- Línea 27: `attn_output, attn_weights = attention_interface(...)`, con `scaling=self.scaling`. **Esto devuelve los pesos de atención (el mapa)**.
- Líneas 38-39: junta las heads (`reshape(*input_shape, -1)` → `(batch, seq, 576)`) y aplica `o_proj`.

#### Extraer el mapa de atención (celda 134)

```python
model.config._attn_implementation = "eager"
prompt = "a is a & b is b & c is c & d is"

with model.trace(prompt):
  outputs = model.model.layers[10].self_attn.source.attention_interface_0.output.save()

attn_outputs, attn_weights = outputs
attn_weights.shape
```

- `_attn_implementation = "eager"`: fuerza la implementación "a mano" de PyTorch (la de las fórmulas). Las implementaciones optimizadas (SDPA, flash attention) no materializan la matriz de pesos y devuelven `None` en `attn_weights`, así que sin esta línea no habría mapa para mirar.
- `prompt`: un patrón repetitivo "x is x &" con un último "d is" incompleto; un modelo que entienda el patrón debería predecir " d".
- `.source.attention_interface_0.output`: la salida de la llamada a `attention_interface` dentro del forward, la tupla `(attn_output, attn_weights)`.
- Se desempaqueta afuera del trace.

Salida guardada: `torch.Size([1, 9, 14, 14])` = (batch, heads, query, key). 14 tokens: `a, is, a, &, b, is, b, &, c, is, c, &, d, is`; no hay BOS (si no, serían 15). `attn_weights[0, h, j, i]` es cuánto atiende el token `j` al token `i` en la head `h`; para cada `j`, la suma sobre `i` es 1 y es 0 para `i > j`.

#### Graficar la head 3 de la capa 10 (celda 137)

```python
tokens =[t.replace('Ġ', '') for t in model.tokenizer.tokenize(prompt)]

fig = px.imshow(
    attn_weights[0, 3].detach().cpu().float().T,
    x=[f"{i}) {l}" for i, l in enumerate(tokens)],
    y=[f"{i}) {l}" for i, l in enumerate(tokens)],
    ...
)
```

- `tokens`: los tokens como strings sin el `Ġ`, para las etiquetas de los ejes.
- `attn_weights[0, 3]`: head 3, shape `(14 query, 14 key)`.
- `.detach()`: saca el tensor del grafo de gradientes; `.cpu()`; `.float()`: de bfloat16 a float32 (numpy/plotly no manejan bfloat16).
- `.T`: transpone a `(key, query)`. En `px.imshow` las filas son el eje y y las columnas el eje x, así que queda: **eje x = token que mira (query), eje y = token mirado (key)**. Cada columna es una distribución que suma 1. Esto coincide con la convención $QK_{i,j}$ de la fórmula (i = key, j = query) y con lo de "softmax por columna".

La notebook describe que en la head 3 cada token atiende al que está **2 posiciones antes**: la segunda "b" a la primera "b", "&" a "is", "is" a "&". Una diagonal desplazada en 2.

La notebook dice que un offset fijo no es una induction head (correcto) pero agrega que con un prompt tan regular "ambos dibujarían lo mismo". Haciendo la cuenta, no me cierra: el patrón tiene período 4 (`x is x &`), así que una induction head en el token `t` atendería al token después de la aparición anterior del mismo token, que está en `t − 4 + 1 = t − 3` (por ejemplo, en el segundo "&" (pos 7) buscaría el primer "&" (pos 3) y atendería a "b" (pos 4)). Eso sería una diagonal a offset 3, no 2. Puede ser que la notebook se refiera a otra cosa; no queda claro.

#### Ejercicio 5 (celda 140, `# TERRON`): attention sinks

Pide recorrer heads y capas y encontrar un **attention sink** (todos los tokens atienden al primero).

```python
print("layers:", model.config.num_hidden_layers, "| heads:", model.config.num_attention_heads)
```

Imprime 30 capas y 9 heads (de la config).

```python
with model.trace(prompt):
  all_attn = [layer.self_attn.source.attention_interface_0.output[1] for layer in model.model.layers].save()
```

En una sola pasada guarda el mapa de atención de las 30 capas. `output[1]` es `attn_weights` de la tupla. Resultado: lista de 30 tensores `(1, 9, 14, 14)`. Usa el mismo `prompt` de la celda 134 y depende de que `_attn_implementation` siga en `"eager"`.

```python
sink_scores = torch.stack([
  attn[0, :, 1:, 0].float().mean(dim=-1) for attn in all_attn
])  # (layers, heads)
```

Esta es la línea clave. Para cada capa:

- `attn[0]`: saca el batch → `(9 heads, 14 query, 14 key)`.
- `[:, 1:, 0]`: todas las heads, queries desde la 1 en adelante, key 0 → `(9, 13)`: cuánta atención pone cada query en el primer token.
- Se saltea la query 0 porque por la máscara causal solo puede mirarse a sí misma: su atención al token 0 es siempre 1 y inflaría el promedio.
- `.float().mean(dim=-1)`: promedio sobre las 13 queries → `(9,)`. Un valor cerca de 1 = esa head es un sink.
- `torch.stack` de las 30 → `(30, 9)`.

```python
px.imshow(sink_scores.detach().cpu(), labels=dict(x="head", y="layer", color="atención al token 0"), ...).show()
```

Heatmap capa × head: las celdas oscuras son heads que mandan casi todo al token 0.

```python
best = sink_scores.argmax()
sink_layer, sink_head = divmod(best.item(), sink_scores.shape[1])
print(f"sink más fuerte: layer {sink_layer}, head {sink_head} ({sink_scores[sink_layer, sink_head]:.2f})")
```

- `argmax()` sin `dim` devuelve el índice en el tensor **aplanado** (0..269).
- `divmod(idx, 9)` lo convierte en (fila, columna) = (capa, head). Equivale a `torch.unravel_index`.
- El f-string con `:.2f` sobre un tensor de 0 dimensiones funciona (PyTorch delega en `.item()`).

```python
px.imshow(all_attn[sink_layer][0, sink_head].detach().cpu().float().T, ...).show()
```

Dibuja el mapa de esa head con la misma convención que la celda 137. Un sink se ve como una **fila superior** (key 0) casi toda oscura.

No hay salida guardada, así que no sé qué capa/head gana. Detalles para razonar la respuesta:

- En este prompt el token 0 es "a" (no hay BOS), así que el sink cae sobre un token de contenido normal. Que igual funcione como sink dice que lo que importa es la *posición* (es visible para todos), no el contenido.
- Qué significa para la salida de la head: si todas las queries ponen ~1 en el token 0, entonces $h_j \approx v_0$ para todo `j`: la head escribe casi el mismo vector en todas las posiciones, sin importar el contexto. Si ese $v_0$ es chico o constante, la head efectivamente no aporta información específica: es un "no-op".
- Por qué un modelo aprendería esto: softmax obliga a repartir 1 de atención; si una head es útil solo en ciertos contextos, en los demás necesita un "basurero" donde poner la atención sin dañar el residual stream. La pista de la consigna pregunta en qué capas pasaría: comparar en el heatmap capas tempranas, medias y tardías.
- El score es un promedio: una head puede tener score alto por un par de queries muy concentradas y otras no. Conviene confirmar mirando el mapa.

### Putting it all together (celda 145)

```python
for prompt in ["A robin is a type of", "A robin is not a type of"]:
    with model.generate(prompt, max_new_tokens=8):
        out = model.generator.output.save()
    print(model.tokenizer.decode(out[0]))
```

- `model.generate(...)`: como `trace` pero corre la generación completa (varios forward passes). Acepta los mismos argumentos que `generate` de Hugging Face.
- `model.generator.output`: el módulo `generator` que agrega nnsight; su output son los token ids finales (prompt + generados), shape `(1, len)`.
- `decode(out[0])`: pasa a texto.

Sin salida guardada. Según la notebook, el segundo prompt genera algo contradictorio: afirma algo y su opuesto. El punto: toda la maquinaria funcionó "como se espera" (attention llevó "robin" a la última posición, la MLP recuperó qué es un robin, la `lm_head` puntuó "bird" alto), y aun así la oración no tiene sentido porque el "not" no se integró. Un modelo puede ser fluido e incoherente, y no siempre se puede saber por qué mirando solo la salida: de ahí la necesidad de mirar los internals.

### Takeaways (celdas 147-150)

- Los LMs trabajan un token a la vez: toda la información relevante tiene que llegar a la última capa del último token antes de predecir.
- Son autoregresivos: las activaciones dependen solo del contexto previo.
- MLPs guardan información, attention la mueve. Nada más puede mover información entre tokens.

La próxima lecture (03) va "de afuera hacia adentro": behavioral analysis.

---

## Ojo / errores comunes / dudas

- **Espacios y tokenización.** `" Paris"` ≠ `"Paris"`. Un espacio al final del prompt crea un token extra y cambia la predicción. En la celda 44 los nombres de las variables están invertidos respecto a su contenido.
- **Logits vs probabilidades.** El modelo devuelve logits; las probabilidades salen de softmax. `argmax` y `topk` dan el mismo orden en ambos.
- **Una fila de logits por posición.** Para predecir el siguiente token del prompt solo se usa la última (`[-1]`). Las demás son predicciones "del token siguiente" en cada posición intermedia.
- **Typo en comentarios.** `(50527,)` en la celda 38 es 50257.
- **Pesos `(out, in)`.** `nn.Linear(576, 1536).weight` tiene shape `(1536, 576)`.
- **`.save()` es obligatorio** para usar algo fuera del `with model.trace(...)`.
- **`layers[i].output` vs `mlp.output` vs `gate_proj.output`.** El primero es el residual stream completo después de la capa; el segundo es solo la contribución de la MLP (576 dims); el tercero es una activación intermedia de 1536 dims.
- **"Neuronas" de la MLP.** El gráfico de la celda 115 usa la salida de la MLP (576), no las 1536 neuronas intermedias.
- **Conteo con substrings.** `"attn" in name` no agarra `post_attention_layernorm`; si se filtrara por `"attention"` sí lo agarraría.
- **Tied embeddings.** Si `lm_head` y `embed_tokens` comparten pesos, `named_parameters()` cuenta la matriz una vez. No está verificado con una salida en esta notebook.
- **Eager attention.** Sin `model.config._attn_implementation = "eager"` no hay `attn_weights` (vienen `None`). La celda 140 depende de que la 134 se haya corrido antes.
- **Convención del gráfico de atención.** Por el `.T`, el eje x es la query (quien mira) y el eje y la key (a quien mira). Es fácil leerlo al revés.
- **Producto punto, no coseno.** La notebook habla de similitud coseno entre q y k, pero el score es un producto punto escalado.
- **Offset fijo ≠ induction head.** La head 3 de la capa 10 muestra un offset de 2; una induction head matchea por contenido. Por mi cuenta, en este prompt de período 4 una induction head daría offset 3, no lo mismo que un offset de 2.
- **Sin BOS.** SmolLM2 y GPT-2 no agregan BOS en estos ejemplos, por eso las posiciones (Paris = 2) y el tamaño 14×14 cuadran con `tokenize`. Con otro modelo que sí lo agrega, todos los índices se corren en 1.
- **GPU.** La celda 30 hace `.to("cuda")`; sin GPU falla.
- **Celdas `# TERRON` sin ejecutar.** Ninguna de las soluciones (celdas 22, 47, 99, 120, 140) tiene salida guardada; los resultados hay que obtenerlos corriéndolas.

---

## Resumen de una línea por sección

- **Setup**: instala dependencias (Colab con pip pineado, local con `uv sync`) y configura plotly para guardar PNG.
- **Talking to LMs**: GPT-2 como caja negra con `pipeline`; greedy con `do_sample=False`; el ejercicio 1 prueba hechos, aritmética, instrucciones y formato.
- **Tokens y logits**: texto → token ids; el modelo devuelve logits `(T, V)`; softmax de la última fila da la distribución del siguiente token; `topk`/`argmax` para leerla; un espacio cambia todo.
- **Weights vs activations**: los pesos son fijos, las activaciones cambian con el input; nnsight (`trace`, `.output`, `.weight`, `.save()`) permite leer ambos.
- **Residual stream**: shape `(batch, tokens, 576)`; la información fluye solo hacia adelante (cambiar tokens posteriores no altera nada); las representaciones son contextuales y la semántica aparece en capas medias ("break").
- **MLP**: SwiGLU token por token, 3 veces más parámetros que la attention por capa (2.654.208 vs 884.736), se le atribuye el factual recall; el ejercicio 4 calcula qué fracción del modelo es MLP.
- **Attention**: único mecanismo que mueve información entre tokens (q, k, v, softmax causal); mapa de la capa 10 head 3 con offset 2; el ejercicio 5 busca attention sinks promediando la atención al token 0.
- **Putting it all together**: "A robin is not a type of" muestra un modelo fluido pero incoherente, y motiva mirar los internals.
