# Lecture 4: Decodificar información en modelos de lenguaje (probes y steering)

Esta notebook es el primer contacto con los "internos" del modelo. La pregunta de fondo es doble: **¿qué información podemos leer de las activaciones intermedias de un LM?** (probing) y **¿podemos usar lo que leemos para controlar lo que hace el modelo?** (steering). Recorre tres formas lineales de "mirar" las activaciones (logit lens, PCA, probes supervisados), después hace un desvío conceptual sobre correlación vs. causalidad con un ejemplo de juguete (Prof. X), y termina usando **difference-in-means** para hacer que el modelo hable en francés (localmente, con SmolLM2-1.7B) o en emojis (remotamente, con Gemma-2-9B vía NDIF).

---

## Ideas clave / teoría

### Activaciones y residual stream

Un transformer decoder procesa cada token en una columna: embedding → capa 0 → capa 1 → ... → capa final → norm final → `lm_head` (unembedding) → logits. Lo que sale de cada `model.model.layers[i]` es el **residual stream** en esa capa: un vector de tamaño `hidden_dim` (2048 en SmolLM2-1.7B) por cada token. Con transformers 5.x (el `uv.lock` del repo fija `transformers 5.5.0` y `nnsight 0.6.3`), la salida de una decoder layer es directamente un tensor de shape `(batch, seq_len, hidden_dim)`, no una tupla. Por eso en la notebook se indexa `layers[i].output[:, -1, :]` sin un `[0]` previo.

### Probing

Un **probe** es una función (acá siempre lineal) que toma una activación interna y predice algo que nosotros entendemos: un token, un idioma, una etiqueta. Si un probe lineal logra predecir la propiedad, decimos que la información está **linealmente decodificable** en esa capa. Ojo: que se pueda decodificar no implica que el modelo la *use* (eso es lo que discute la sección 2).

Las tres variantes de la notebook:

| Método | ¿Cómo se obtiene la transformación lineal? | ¿Qué devuelve? |
|---|---|---|
| Logit lens | Se reutiliza el final del propio modelo (`norm` + `lm_head`) | Distribución sobre tokens |
| PCA | No supervisado: direcciones de máxima varianza del dataset | Coordenadas en k dimensiones |
| Probe supervisado | Se entrena (regresión logística) con etiquetas | Clase predicha |

### Logit lens

Idea: "saltar al final". Tomo la activación `h_ℓ` de la capa ℓ y la paso por lo que el modelo usaría al final:

$$
\text{logits}_\ell = W_U \cdot \text{RMSNorm}(h_\ell), \qquad p_\ell = \text{softmax}(\text{logits}_\ell)
$$

- `h_ℓ`: residual stream en la capa ℓ, en una posición de token.
- `RMSNorm`: la normalización final del modelo (`model.model.norm`). Divide el vector por su raíz cuadrática media y multiplica por un vector de escala aprendido γ: `RMSNorm(h) = γ ⊙ h / sqrt(mean(h²) + ε)`. Es importante aplicarla: el `lm_head` fue entrenado para recibir vectores ya normalizados, y las normas del residual stream crecen mucho a lo largo de las capas (fijate los valores tipo 78.0, -57.5 que imprime la celda 23).
- `W_U`: la matriz del `lm_head`, shape `(vocab_size, hidden_dim)`.
- `softmax`: convierte logits en probabilidades.

Supuesto fuerte: que las capas intermedias "hablan el mismo idioma" que la última. No está garantizado; por eso muchas celdas del mapa salen con basura.

### PCA

PCA busca las direcciones ortogonales (componentes principales) que capturan la mayor varianza de los datos. Con datos centrados `X̃ = X − μ` (shape `(n, d)`), las componentes son los autovectores de la covarianza `X̃ᵀX̃/(n−1)` con mayor autovalor. Proyectar a k dimensiones es:

$$
z = (x - \mu)\, C^\top, \qquad C \in \mathbb{R}^{k \times d}
$$

En sklearn `C` es `pca.components_` (shape `(k, d)`) y `μ` es `pca.mean_`. `fit_transform` ajusta y proyecta en un paso; `transform` proyecta datos nuevos con la PCA ya ajustada. Dos cosas a recordar: PCA es no supervisada (no sabe de etiquetas; si separa clases es porque esa diferencia es la que más varianza tiene en *ese* dataset), y el signo de cada componente es arbitrario.

### Probe supervisado (regresión logística)

`LogisticRegression` de sklearn. Para dos clases calcula `p(clase positiva | x) = σ(w·x + b)`. Para más de dos clases (como acá, 4 idiomas) usa por defecto la versión multinomial: un vector de pesos `w_k` por clase y `p(k | x) = softmax_k(w_k · x)`. Datos clave de la API:

- `fit(X, y)`: entrena. Por defecto usa regularización L2 con `C=1.0` y el solver `lbfgs`.
- `fit_intercept=False`: sin término `b`. La frontera pasa por el origen.
- `classes_`: las etiquetas **ordenadas** (alfabéticamente para strings). Para idiomas: `['de', 'en', 'es', 'fr']`.
- `coef_`: shape `(n_clases, d)` en multiclase. En binario es `(1, d)` y la fila corresponde a la clase positiva, que es `classes_[1]`.
- `score(X, y)`: accuracy. `predict(X)`: etiquetas.

### Correlación vs. causalidad: probing ≠ steering

Un probe busca **la dirección que mejor separa** las clases. Eso no tiene por qué ser **la dirección por la que hay que mover** un punto para cambiarlo de clase de forma "natural". Cuando dos features están correlacionadas y una de ellas separa con menos ruido, la regresión logística puede apoyarse en una combinación que discrimina muy bien pero que como intervención no tiene sentido. Ese es el punto del ejemplo de Prof. X.

### Difference-in-means

Con dos grupos de activaciones A y B:

$$
v = \mu_B - \mu_A = \frac{1}{|B|}\sum_{x\in B} x - \frac{1}{|A|}\sum_{x\in A} x
$$

Usado como clasificador, la frontera es el hiperplano perpendicular a `v` que pasa por el punto medio de las medias. Usado como intervención (steering), sumamos `α·v` a las activaciones: `h' = h + α v`. La intuición es que `v` es literalmente el desplazamiento promedio que lleva "del grupo A al grupo B", así que mueve la distribución entera en vez de solo cruzar una frontera. No necesita entrenar nada.

---

## Recorrido por la notebook

### 0. Setup

**Celda 5**: detecta si está en Colab (`import google.colab`). Si sí, clona el repo e instala `requirements/04_probes.txt` (que solo contiene `nnsight`); si no, configura plotly para renderizar como PNG e instala Chrome para kaleido. `clear_output()` limpia la salida. Setup puro.

**Celda 7**: carga del modelo.

```python
from nnsight import LanguageModel

model = LanguageModel("HuggingFaceTB/SmolLM2-1.7B")

with model.generate("Once upon a", max_new_tokens=1):
    outputs = model.generator.output.save()

model.to("cuda")
clear_output()
print(model.tokenizer.decode(outputs[0]))
```

- `LanguageModel(...)`: wrapper de nnsight sobre un modelo de Hugging Face. Te da acceso a los submódulos con la misma jerarquía que el modelo de HF (`model.model.layers[i]`, `model.model.norm`, `model.lm_head`) y a `model.tokenizer`. nnsight carga los pesos de forma perezosa ("meta" device) hasta el primer uso real; por eso hacen una generación de prueba antes de `.to("cuda")`.
- `model.generate(prompt, max_new_tokens=1)`: abre un contexto de **generación**. Todo lo que escribas adentro se ejecuta "durante" la generación. Al salir del `with` se corre el modelo de verdad.
- `model.generator.output`: los ids de salida de la generación (prompt + tokens nuevos), shape `(batch, seq_len_total)`.
- `.save()`: le dice a nnsight que ese valor tiene que sobrevivir al contexto. Lo que no se guarda con `.save()` se descarta al salir del `with`.
- `decode(outputs[0])`: decodifica la primera (única) secuencia del batch. La salida guardada imprime "Once upon a time, there was a little girl named Lily who", más de un token; no queda claro por qué, porque `max_new_tokens=1`. Puede ser la config de generación por defecto del modelo; no es importante.

---

### 1. Decodificar información con probes

#### Logit lens: un token, la última capa (celdas 21–25)

```python
with model.generate("ES: amor, FR:"):
  last_layer_last_token = model.model.layers[-1].output[0, -1].save()
  output = model.generator.output.save()
print(model.tokenizer.decode(output))
```

- `model.model.layers[-1].output`: salida de la última decoder layer, shape `(1, seq_len, 2048)`. En un `generate`, si no se usa `tracer.all()` (ver más abajo), lo que se accede corresponde al **primer** forward, que procesa el prompt completo.
- `[0, -1]`: batch 0, último token (el ":"). Queda un vector `(2048,)`.
- `decode(output)` sin `[0]`: decodifica el batch entero y devuelve una lista; por eso la salida aparece entre corchetes: `['ES: amor, FR: amour, G: Liebe, IT: amore, RU: люб']`. El modelo sigue el patrón y agrega más idiomas.

**Celda 23** imprime la shape (`torch.Size([2048])`) y los primeros 10 valores. Lo único que hay que sacar de acá es que el vector crudo no se entiende a ojo y que tiene magnitudes grandes.

```python
unembedding_layer = model.lm_head
logits = unembedding_layer(model.model.norm(last_layer_last_token))
next_token_prediction = logits.argmax(dim=-1)
next_token = model.tokenizer.decode(next_token_prediction)
```

- `model.lm_head`: el `Linear(2048 → vocab)` final.
- `model.model.norm(...)`: el RMSNorm final. Acá se llama **fuera** de un trace sobre un tensor real (ya guardado), así que simplemente ejecuta el módulo.
- `logits`: shape `(vocab_size,)`. `argmax` da el id del token más probable.
- Resultado guardado: `" am"`, el primer pedazo de "amour". Es exactamente lo que el modelo predijo, lo cual tiene sentido: aplicar `norm` + `lm_head` a la última capa es reproducir a mano el final del forward. Esta celda es la "prueba de cordura" del logit lens.

#### Logit lens en todas las capas y tokens (celda 28)

```python
prompt= "ES: amor, FR:"
layer_range = list(range(10, len(model.model.layers)))

decoded_layers = []
with model.trace(prompt):
    for layer_index in layer_range:
        layer_activation = model.model.layers[layer_index].output
        layer_activation = model.model.norm(layer_activation)
        decoded_layer_logits = unembedding_layer(layer_activation)
        decoded_layer = torch.nn.functional.softmax(decoded_layer_logits, dim=-1)
        decoded_layers.append(decoded_layer.save())

decoded_layers = torch.cat(decoded_layers) # (num_layers, num_tokens, vocab_size)
probabilities, decoded_tokens = decoded_layers.max(dim=-1)
```

- `layer_range`: de la capa 10 a la última (para que el gráfico no sea enorme). `len(model.model.layers)` es el número de capas; para SmolLM2-1.7B debería ser 24 (no lo imprime la notebook).
- `model.trace(prompt)`: a diferencia de `generate`, hace **un solo forward** sobre el prompt. Es el contexto que se usa para leer/modificar activaciones sin generar.
- Dentro del trace se recorren las capas **en orden creciente**. En nnsight 0.5+ esto importa: el código del trace se ejecuta en paralelo al forward, y tenés que pedir las activaciones en el orden en que el modelo las produce (pedir la capa 5 después de la 10 da error).
- `model.model.norm(layer_activation)` y `unembedding_layer(...)`: dentro del trace también se pueden llamar módulos del modelo sobre activaciones intermedias. Shapes: `(1, seq, 2048)` → norm → `(1, seq, 2048)` → lm_head → `(1, seq, vocab)`.
- `softmax(dim=-1)`: probabilidades sobre el vocabulario.
- `.save()` en cada capa; al salir del trace la lista tiene tensores reales.
- `torch.cat` sobre la dim 0: cada elemento es `(1, seq, vocab)`, así que queda `(num_layers, seq, vocab)`.
- `.max(dim=-1)`: devuelve el par `(valores, índices)`: la probabilidad máxima y el token argmax por (capa, posición). Ambos `(num_layers, seq)`.

```python
decoded_tokens = [[model.tokenizer.decode(t.item()) for t in layer_tokens]
                  for layer_tokens in decoded_tokens]
input_tokens = [t.replace('Ġ', '') for t in model.tokenizer.tokenize(prompt)]
```

- Convierte la matriz de ids en una matriz de strings para escribir en cada celda del heatmap.
- `tokenize` devuelve tokens BPE crudos; `Ġ` es como el tokenizer byte-level marca un espacio al inicio. Se quita para las etiquetas del eje x.

El resto (`px.imshow`, `update_layout`, `update_traces(text=decoded_tokens)`) es ploteo: un heatmap con color = probabilidad máxima y texto = token decodificado. `origin="lower"` pone la capa 10 abajo, así que se lee como el modelo: de abajo (capas tempranas) hacia arriba y de izquierda a derecha.

Lo que hay que mirar: en la columna del último ":", en capas medias–tardías el lens decodifica "love" (inglés) antes de que aparezca "am". Ni el input ni el output piden inglés.

#### Ejercicio 1 (celda 34, `# TERRON`)

Pide probar el logit lens con otros prompts (traducciones, hechos, aritmética) y buscar patrones. La solución empaqueta la celda 28 en una función:

```python
def logit_lens(prompt, start_layer=10):
    layer_range = list(range(start_layer, len(model.model.layers)))
    decoded_layers = []
    with model.trace(prompt):
        for layer_index in layer_range:
            layer_activation = model.model.norm(model.model.layers[layer_index].output)
            decoded_layers.append(unembedding_layer(layer_activation).softmax(dim=-1).save())
    decoded_layers = torch.cat(decoded_layers)  # (num_layers, num_tokens, vocab)
    probabilities, decoded_tokens = decoded_layers.max(dim=-1)
    ...
```

- Es idéntico a la celda 28, compactado: `norm` y `lm_head` en una sola expresión, y `.softmax(dim=-1)` como método del tensor en vez de `torch.nn.functional.softmax`.
- El heatmap es el mismo con título dinámico (`{prompt!r}` imprime el prompt con comillas).
- Agrega un resumen en texto:

```python
for layer_index, row_tokens, row_probs in zip(layer_range, decoded_tokens, probabilities):
    print(f"  layer {layer_index:2d}: {row_tokens[-1]!r} ({row_probs[-1].item():.2f})")
```

  Para cada capa imprime solo el token decodificado en la **última posición** (la que predice la respuesta) y su probabilidad. Es la columna que interesa, y así se puede leer sin mirar el gráfico.

- Prompts elegidos: tres traducciones que no involucran inglés (`"FR: chat, ES:"` → gato, `"DE: Hund, FR:"` → chien, `"ES: casa, DE:"` → Haus), un hecho (`"The capital of Italy is"`) y aritmética (`"3 + 4 ="`).

La celda no está ejecutada, así que no hay resultados. Qué mirar al correrla: en las traducciones, si en las capas medias de la última columna aparece la palabra en inglés (cat, dog, house) antes que la del idioma destino; en el hecho y la suma, en qué capa aparece por primera vez la respuesta correcta (" Rome", " 7") y si antes aparecen tokens relacionados.

**Pregunta de la celda 35** (¿piensa el modelo en inglés?). Para razonarla: el lens solo dice qué token está más cerca de la activación *según el `lm_head`*; que el más cercano sea inglés puede deberse a que el inglés domina el entrenamiento y por eso su región del espacio de salida es "la más grande", no necesariamente a que haya un paso de traducción interna al inglés. La pista apunta a Wendler et al. (2024), que separan tres fases (en la tercera aparece el idioma destino) y concluyen que el "espacio de conceptos" intermedio está más cerca del inglés, que no es lo mismo que decir que el modelo traduce al inglés. Para testearlo de verdad habría que pasar de decodificar a **intervenir** (por ejemplo, ¿cambia la salida si manipulo esa dirección "inglesa"?), y comparar contra modelos con distinta mezcla de idiomas.

#### PCA (celdas 42–48)

**Celda 42**: dos listas de 10 prompts con los mismos animales en inglés; una termina en `Francais:` y la otra en `Espanol:`. La única diferencia entre pares es el idioma destino.

**Celda 43**: recolecta activaciones.

```python
LAYER = 10
with torch.no_grad():
  french_activations = []
  for french_prompt in tqdm(ENGLISH_TO_FRENCH, desc="French"):
    with model.trace(french_prompt):
      french_activations.append(model.model.layers[LAYER].output[:, -1, :].save())
  ...
french_activations = torch.cat(french_activations) # (10, hidden_dim)
```

- `torch.no_grad()`: no se construye grafo de gradientes (ahorra memoria; solo leemos).
- Un `trace` por prompt (no en batch, así no hay padding).
- `output[:, -1, :]`: todos los batch (1), último token (el ":"), todo el hidden → `(1, 2048)`. Con `torch.cat` queda `(10, 2048)` por idioma.
- `tqdm`: barra de progreso.

**Celda 45**:

```python
pca = PCA(n_components=2)
all_activations = torch.cat([french_activations, spanish_activations]).cpu().float().numpy() # (20, 2048)
low_dim_activations = pca.fit_transform(all_activations) # (20, 2)
```

- Se concatenan las 20 activaciones: filas 0–9 francés, 10–19 español. Ese orden importa después.
- `.cpu().float().numpy()`: sklearn trabaja con numpy en CPU; `.float()` convierte a float32 por si el modelo está en bf16 (numpy no tiene bf16).
- `fit_transform`: centra, calcula las 2 componentes y proyecta → `(20, 2)`.

**Celda 47**: scatter con `low_dim_activations[:10]` como francés (cuadrados) y `[10:]` como español (círculos). `hovertext` muestra el prompt al pasar el mouse. Según el texto, PCA 1 sola ya separa los dos clusters: las activaciones en la capa 10 son linealmente separables por idioma destino.

#### Ejercicio 2 (celda 52, `# TERRON`)

Pide un prompt inglés→español que la PCA clasifique como francés. Pista: palabras que no necesitan traducción.

```python
YOUR_PROMPT = "English: chocolate, Espanol:"
candidates = [YOUR_PROMPT, "English: hotel, Espanol:", "English: taxi, Espanol:",
              "English: radio, Espanol:", "English: piano, Espanol:"]

french_center = low_dim_activations[:10, 0].mean()
spanish_center = low_dim_activations[10:, 0].mean()
```

- La idea: "chocolate", "hotel", "taxi", "radio", "piano" se escriben igual (o casi) en inglés, francés y español. Si el modelo no tiene que "traducir", quizás la activación del ":" no codifique tan fuerte el idioma destino.
- En vez de usar la regla de la notebook ("debajo de 0 en PCA 1 es francés"), calcula el **centro de cada cluster en PCA 1** y clasifica por cercanía. Es más robusto, porque el signo de las componentes de PCA es arbitrario: en otra corrida francés podría quedar a la derecha.

```python
for prompt in candidates:
  with torch.no_grad():
    with model.trace(prompt):
      activation = model.model.layers[LAYER].output[:, -1, :].save()
  activation_low_dim = pca.transform(activation.cpu().float().numpy())
  pc1 = activation_low_dim[0, 0]
  closest = "French" if abs(pc1 - french_center) < abs(pc1 - spanish_center) else "Spanish"
```

- Misma extracción que en la celda 43 (capa `LAYER`, último token).
- `pca.transform`: proyecta con la PCA **ya ajustada** sobre los animales (no se reajusta). Shape `(1, 2)`.
- `pc1 = [0, 0]`: primera fila, primera componente.
- Clasificación por vecino más cercano en PCA 1, con un aviso si sale "French".
- Después agrega cada candidato como estrella al gráfico `fig` de la celda 47 (`name` = la palabra sin "English: ").

Para la pregunta de la celda 53: hay que mirar cuáles candidatos caen del lado francés y pensar por qué (¿la palabra es idéntica en francés? ¿el modelo "decide" el idioma más tarde para esas palabras?). El argumento de "cómo lo sé" es que la PCA se ajustó solo con animales, así que PCA 1 es "la dirección que separa estos 20 prompts", no "la dirección del idioma destino" en general.

#### Probe supervisado (celdas 61–66)

**Celda 61**: `languages_df`, un DataFrame de 10 filas × 4 columnas (`en`, `es`, `fr`, `de`), cada fila es la misma frase corta en los 4 idiomas.

**Celda 62**:

```python
LAYER = 20
all_activations = []
with torch.no_grad():
  for i in trange(languages_df.shape[0]):
    row = languages_df.iloc[i]
    with model.trace(row.values.tolist()):
      all_activations.append(model.model.layers[LAYER].output[:, -1, :].save())
all_activations = torch.cat(all_activations, dim=0).cpu().float().numpy() # (40, hidden dim)
```

- Ahora capa 20 (más profunda).
- `row.values.tolist()`: las 4 traducciones de una frase, en el orden de las columnas `[en, es, fr, de]`.
- `model.trace(lista)`: un trace con **batch** de 4 strings. nnsight tokeniza con padding. Para que `[:, -1, :]` sea el último token real (y no un token de padding), el padding tiene que ser a la izquierda. Hasta donde sé, el `LanguageModel` de nnsight configura `padding_side="left"` por defecto, pero no lo verifiqué en el código.
- Shape por iteración: `(4, 2048)`; concatenado `(40, 2048)` en orden `[en_1, es_1, fr_1, de_1, en_2, ...]`.

**Celda 64**:

```python
X = all_activations
y = ['en', 'es', 'fr', 'de'] * len(languages_df)
linear_probe = LogisticRegression(random_state=12, fit_intercept=False)
linear_probe.fit(X, y)
linear_probe.score(X, y)
```

- `y` repite el patrón de 4 etiquetas 10 veces, alineado con el orden de las filas de `X`.
- Probe multinomial de 4 clases, sin intercepto, semilla fija.
- `score` sobre el mismo set de entrenamiento: 1.0 en la salida guardada. Ojo: 40 puntos en 2048 dimensiones son casi siempre separables, así que 100% de train accuracy dice poco por sí solo. La notebook lo reconoce ("it's quite fragile").

**Celda 66**: pasa "Bon appetit" por el modelo (capa 20, último token), lo convierte a numpy y hace `linear_probe.predict(...)[0]`. La salida guardada predice `fr`. Notar que esta frase no termina en "." como las del dataset.

#### Ejercicio 3 (celdas 70, 72, 75)

**Celda 70 (`# TERRON`)**: ajustar una PCA de 2 componentes a las 40 activaciones.

```python
input_to_pca = all_activations
pca = PCA(n_components=2)
low_dim_activations = pca.fit_transform(input_to_pca)  # (40, 2)
```

- Sobrescribe `pca` y `low_dim_activations` del ejercicio 2 (ahora es la PCA de 4 idiomas en la capa 20). Si volvés a correr la celda 52 después de esta, va a usar la PCA equivocada y además `LAYER` ya vale 20.
- El ploteo que sigue usa `low_dim_activations[index::4]`: slicing con paso 4 a partir de `index`, que justamente agarra un idioma (0 = en, 1 = es, 2 = fr, 3 = de) gracias al orden intercalado de la celda 62.

**Celda 72**: proyectar la frontera del probe para inglés en el plano PCA (no hay que escribir código, pero hay que entenderla).

```python
language_idx = list(linear_probe.classes_).index('en')
linear_weights = linear_probe.coef_[language_idx] # (2048,)
weights_pca = pca.components_ @ linear_weights # (2,)
x = np.linspace(-300, 300, 200)
y = -(weights_pca[0] / weights_pca[1]) * x
```

- `classes_` está ordenado alfabéticamente (`['de', 'en', 'es', 'fr']`), así que `'en'` es el índice 1. Por eso se busca el índice en vez de asumir 0.
- `coef_[language_idx]`: el vector `w_en` de 2048 pesos.
- `pca.components_ @ w`: `(2, 2048) @ (2048,)` → `(2,)`. Por qué funciona: si una activación vive (aprox.) en el plano PCA, `x ≈ μ + z₀c₀ + z₁c₁`, entonces `w·x ≈ w·μ + z₀(c₀·w) + z₁(c₁·w)`. Las coordenadas `(c₀·w, c₁·w)` son justamente `weights_pca`: los pesos del probe "vistos" en coordenadas PCA.
- La frontera `w₀ z₀ + w₁ z₁ = 0` despejada da la recta `z₁ = −(w₀/w₁) z₀`, que se dibuja punteada.

Aproximaciones escondidas que vale la pena tener presentes:
1. Ignora el término `w·μ` (la PCA centra los datos y el probe no), así que la recta real estaría desplazada del origen.
2. Ignora todo lo que queda fuera del plano de 2 componentes.
3. En multiclase la predicción es `argmax_k w_k·x`; "score de inglés > 0" no es exactamente "predice inglés". El texto lo dice: es la frontera de inglés "manteniendo constantes las predicciones de los otros idiomas".

Así que la línea es una visualización aproximada, no la frontera exacta.

**Celda 75 (`# TERRON`)**: encontrar una frase que el probe clasifique mal en la frontera inglés / no inglés.

```python
candidates = [(YOUR_PROMPT, "fr"), ("Déjà vu.", "fr"), ("Kindergarten.", "de"),
              ("Fiesta.", "es"), ("Le weekend.", "fr"), ("Das Internet.", "de")]
```

- Estrategia: préstamos léxicos. "croissant", "déjà vu", "kindergarten", "fiesta" se usan en inglés pero vienen de otro idioma; "Le weekend." y "Das Internet." son francés/alemán con una palabra inglesa. Cada una lleva su idioma "real" para comparar. Todas terminan en "." como pide la consigna, para que el último token se parezca al del dataset.

```python
  with torch.no_grad():
    with model.trace(prompt):
      activation = model.model.layers[LAYER].output[:, -1, :].save()
  activation = activation.cpu().float().numpy()

  prediction = linear_probe.predict(activation)[0]
  wrong = (prediction == "en") != (true_language == "en")
  print(...)
  activation_low_dim = pca.transform(activation)  # (1, 2)
  fig.add_trace(go.Scatter(x=activation_low_dim[:, 0], y=activation_low_dim[:, 1],
                           mode='markers', marker=dict(symbol='star', size=15), ...))
```

- Paso 1 (dado): activación en la capa 20, último token.
- Paso 2: `predict` y un flag `wrong`. El `!=` entre dos booleanos es un XOR: marca error solo si el probe dice "inglés" y no lo es, o al revés. Confundir francés con alemán no cuenta, porque la consigna es sobre la frontera de inglés.
- Paso 3: `pca.transform` con la PCA de la celda 70.
- Paso 4: estrella en el gráfico, con la predicción en la leyenda.

Para responder la celda 77: mirar qué frases salen con el aviso y dónde caen las estrellas respecto a la recta punteada. Para separar "cómo representa el modelo" de "cómo es el probe", preguntate: ¿el modelo, al generar a continuación, sigue en inglés o en el otro idioma? Si el modelo continúa en francés pero el probe dice inglés, el error es del probe (entrenado con 40 frases, posiblemente apoyado en features espurias). Si el modelo también "cree" que está en inglés, el probe refleja algo de la representación. Otra forma: entrenar el probe con más datos, o en otra capa, y ver si el error persiste.

---

### 2. Probes e intervenciones: correlación y causalidad (Prof. X)

#### Datos (celda 86)

```python
np.random.seed(12)
N = 20
low_group_effort = np.random.normal(loc=5, scale=1.8, size=N)
high_group_effort = np.random.normal(loc=10, scale=1.8, size=N)
low_group_grade = 0.5 * low_group_effort + np.random.normal(loc=5, scale=0.3, size=N)
high_group_grade = 0.5 * high_group_effort + np.random.normal(loc=7, scale=0.3, size=N)
X = np.stack((np.concatenate((low_group_effort, high_group_effort)),
              np.concatenate((low_group_grade, high_group_grade))), axis=1)
X = X - X.mean(axis=0)
y = ['low'] * N + ['high'] * N
```

- Dos grupos de 20 estudiantes. Feature 1 ("esfuerzo", # de tareas entregadas): media 5 vs. 10, con mucho ruido (desvío 1.8).
- Feature 2 (nota promedio): `0.5 · esfuerzo + ruido`, con ruido de media 5 para el grupo bajo y 7 para el alto, pero con desvío chico (0.3).
- O sea: dentro de cada grupo, nota y esfuerzo están fuertemente correlacionadas (recta de pendiente 0.5); entre grupos, a igual esfuerzo, el grupo alto tiene ~2 puntos más de nota. En esperanza, la diferencia de medias entre grupos es `(10−5, 0.5·5 + 2) = (5, 4.5)`.
- `np.stack(..., axis=1)`: matriz `(40, 2)`, columnas = (esfuerzo, nota). Filas 0–19 grupo bajo ("failed final"), 20–39 alto ("passed").
- `X - X.mean(axis=0)`: centra. Esto es lo que justifica usar probes sin intercepto (fronteras que pasan por el origen).

**Celda 87**: scatter de los dos grupos con ejes en escala 1:1 (`scaleanchor`), para que los ángulos de las flechas y rectas que se agregan después sean visualmente correctos.

#### Probe de regresión logística (celdas 91–94)

```python
lr_probe = LogisticRegression(fit_intercept=False, random_state=12)
lr_probe.fit(X, y)
lr_probe.score(X, y)          # 0.975 en la salida guardada
theta = lr_probe.coef_[0]     # array([ 0.4126429 , -2.12310183])
```

- Binario: `coef_` es `(1, 2)`. Como `classes_ = ['high', 'low']` (orden alfabético), la clase positiva es `classes_[1] = 'low'`. Entonces `θ` apunta hacia **"low" (reprobó)**: `θ·x > 0` ⇒ predice "low".
- Leer `θ = (0.41, −2.12)`: la nota pesa mucho (y con signo negativo: más nota ⇒ menos "low") y el esfuerzo pesa poco y con signo positivo. ¿Por qué? Porque a igual esfuerzo, la nota separa los grupos con poco ruido (0.3), mientras que el esfuerzo es ruidoso (1.8). El probe aprende algo como "nota alta *para tu nivel de esfuerzo*", que discrimina muy bien, y por eso el esfuerzo termina con peso positivo: más esfuerzo con la misma nota te acerca a "low".

```python
theta_orthogonal = np.array([-theta[1], theta[0]])
t = np.linspace(-2, 2, 100)
probe_x = t * theta_orthogonal[0]
probe_y = t * theta_orthogonal[1]
```

- La frontera de decisión (sin intercepto) es `{x : θ·x = 0}`, una recta por el origen **perpendicular a θ**. Rotar `θ = (a, b)` 90° da `(−b, a)`, que es la dirección de esa recta. `t` la recorre. Se dibuja en bordó punteado.

#### Intervenir con el probe (celda 96)

```python
example_student_index = 7
example_student_stats = X[example_student_index]
example_student_stats_intervened = example_student_stats - theta
```

- El estudiante 7 está en las primeras 20 filas: es del grupo que reprobó.
- `x − θ`: moverse en contra de la dirección "low", es decir, el cambio más directo para que el probe diga "high". `−θ = (−0.41, +2.12)`: **subir la nota y entregar menos tareas**.
- `add_annotation(...)` dibuja la flecha de `x` a `x − θ` (los `+0.05` / `−0.1` son ajustes cosméticos de la punta).

La conclusión (celdas 97–98): esa intervención cruza la frontera del probe, pero no tiene sentido como política ("subirle la nota a todos y que entreguen menos"). La dirección del probe **distingue** pero no **acerca** las distribuciones.

#### Difference in means (celdas 102–104)

```python
mean_low = X[:N].mean(axis=0)
mean_high = X[N:].mean(axis=0)
difference_in_means = mean_high - mean_low
difference_in_means_boundary = np.array([-difference_in_means[1], difference_in_means[0]])
```

- Media de cada grupo `(2,)`; la diferencia apunta de "low" a "high". Esperable cerca de `(5, 4.5)`: más tareas **y** más nota.
- La "frontera" es la perpendicular a `v` por el origen. Como los datos están centrados y los grupos tienen el mismo tamaño, la media global (el origen) es exactamente el punto medio entre `mean_low` y `mean_high`, así que esta recta es exactamente la frontera de "clasificar por la media más cercana" (la mediatriz entre las dos medias).

```python
example_student_diff_in_means = example_student_stats + difference_in_means
```

- Mismo estudiante, ahora se le suma `v` (flecha verde). Lo lleva al centro de la nube de los que aprobaron. Esto es exactamente lo que se va a hacer con activaciones: `h' = h + v`.

La notebook aclara que esto tampoco garantiza causalidad (puede haber causas no observadas de por qué el estudiante no entrega), y cita Marks & Tegmark (2023), que encontraron que en LMs reales los probes de difference-in-means clasifican parecido a otros pero dan direcciones con más efecto causal al intervenir.

---

### 3. Controlar el comportamiento con steering

#### Inglés → francés en SmolLM2 (celdas 111–117)

**Celda 111**: `english_sentences` y `french_sentences` = las columnas `en` y `fr` de `languages_df` (10 frases cada una).

**Celda 113**:

```python
LAYER = 20
with torch.no_grad():
    with model.trace(english_sentences):
        english_activations = model.model.layers[LAYER].output[:, -1, :].save() # (10, hidden dim)
with torch.no_grad():
    with model.trace(french_sentences):
        french_activations = model.model.layers[LAYER].output[:, -1, :].save() # (10, hidden dim)
```

- Batch de 10 frases por trace (otra vez depende del padding a la izquierda para que `-1` sea el "." final). Misma capa que el probe.
- Se toma el último token porque es donde el modelo "planea" qué viene después.

**Celda 115**:

```python
difference_in_means = french_activations.mean(dim=0) - english_activations.mean(dim=0) # (hidden dim,)
```

- `mean(dim=0)` promedia sobre las 10 frases → `(2048,)`. Resta: francés − inglés.

**Celda 117**, la intervención:

```python
with model.generate("Bon appetit", max_new_tokens=20):
    original_output = model.generator.output.save()

with model.generate("Bon appetit", max_new_tokens=20) as tracer:
    with tracer.all():
        model.model.layers[LAYER].output += difference_in_means
    steered_output = model.generator.output.save()
```

- Primera generación: baseline, sin tocar nada.
- `as tracer`: el objeto del contexto, que da acceso a utilidades como `tracer.all()`.
- **`tracer.all()`**: en una generación hay un forward por cada token nuevo (el primero procesa el prompt entero; los siguientes, con KV cache, procesan solo el token recién generado). Sin `tracer.all()`, una intervención se aplica solo al primer forward. Con `tracer.all()`, lo que está adentro del bloque se re-aplica **en cada paso** de generación. Por eso el texto dice que el vector se suma en todos los tokens: en el paso 1 a todas las posiciones del prompt, y en cada paso siguiente al token nuevo.
- `output += difference_in_means`: suma in-place. Shapes: `(1, seq, 2048) + (2048,)` broadcastea a todas las posiciones. Al ser in-place, lo que recibe la capa 21 es la activación modificada; así se "inyecta" la intervención en el forward.
- El `.save()` de `generator.output` va fuera del bloque `all()`, porque es la salida final y queremos un solo valor.

Salida guardada: original "Bon appetit!\n\nThe first thing I did was to make a list of all the ingredients I needed." (inglés); steered "Bon appetit!\n\nP.S. - Bonne anniversaire à toutes les personnes qui" (francés). Notar que la fuerza es 1 (se suma `v` sin escalar).

#### Steering de un modelo grande vía NDIF (celdas 120–136, opcional)

**NDIF** (National Deep Inference Fabric) es un cluster que corre modelos grandes del lado del servidor; nnsight manda el "programa" de intervención y recibe solo lo que se guardó con `.save()`. Requiere API key de NDIF y aceptar la licencia de Gemma en Hugging Face.

- **Celda 123**: `huggingface_hub.login()` (pide el token de HF; hace falta para bajar config y tokenizer de un modelo con acceso restringido).
- **Celda 124**: `CONFIG.set_default_api_key("...")` configura la key de NDIF para nnsight.
- **Celda 126**: `LanguageModel("google/gemma-2-9b-it")`. No se cargan los pesos localmente: sirve para tener la estructura. La salida guardada muestra la arquitectura: 42 `Gemma2DecoderLayer`, hidden 3584, vocab 256000, cuatro RMSNorm por capa.
- **Celda 128**: `generate(..., remote=True)`. `remote=True` es lo único que cambia: la ejecución pasa a NDIF. Salida: "<bos>The Eiffel Tower is in the city of Paris".

**Celda 130**: `steering_examples`, 9 tríos `[pregunta, respuesta en inglés, respuesta en emojis]`.

**Celda 132**:

```python
def apply_template(user, assistant):
    messages = [{"role": "user", "content": user},
                {"role": "assistant", "content": assistant}]
    return ndif_model.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

english_sentences = [apply_template(s[0], s[1]) for s in steering_examples]
emoji_sentences = [apply_template(s[0], s[2]) for s in steering_examples]
LAYER = 20

with ndif_model.trace(english_sentences, remote=True):
    english_activations = ndif_model.model.layers[LAYER].output[:, -5:-1, :].mean(dim=1).save() # (9, hidden dim)
with ndif_model.trace(emoji_sentences, remote=True):
    emoji_activations = ndif_model.model.layers[LAYER].output[:, -5:-1, :].mean(dim=1).save()
```

- `apply_chat_template(..., tokenize=False)`: arma el string con el formato de chat de Gemma (`<start_of_turn>user ... <end_of_turn> <start_of_turn>model ... <end_of_turn>`). Es un modelo instruct: hay que darle el formato con el que fue entrenado. `add_generation_prompt=False` porque la respuesta del asistente ya está incluida.
- Dos sets contrastivos: misma pregunta, respuesta en inglés vs. en emojis.
- `output[:, -5:-1, :]`: las posiciones de la quinta a la segunda desde el final (4 tokens), que caen sobre el final de la respuesta y los tokens de cierre del turno. `.mean(dim=1)` las promedia → `(9, 3584)`. Promediar varias posiciones da un vector más estable que usar una sola. No queda claro exactamente qué tokens caen en esa ventana sin tokenizar; depende del template.
- El `.mean` se calcula del lado del servidor y solo baja el resultado chico.
- El comentario dice "use a late layer", pero 20 de 42 es más bien una capa media.

**Celda 134**: `difference_in_means = emoji_activations.mean(dim=0) - english_activations.mean(dim=0)`, shape `(3584,)`. Ojo: sobrescribe el `difference_in_means` del francés.

**Celda 136**:

```python
messages = [{"role": "user", "content": "What sport does LeBron James play?"}]
STEER_STRENGTH = 5.
prompt = ndif_model.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

with ndif_model.generate(prompt, max_new_tokens=10, do_sample=False, remote=True):
    original_output = ndif_model.generator.output.save()

with ndif_model.generate(prompt, max_new_tokens=10, do_sample=False, remote=True) as tracer:
    with tracer.all():
        ndif_model.model.layers[LAYER].output += STEER_STRENGTH * difference_in_means.to(ndif_model.model.layers[LAYER].device)
    steered_output = ndif_model.generator.output.save()
```

- `add_generation_prompt=True`: deja el template abierto en `<start_of_turn>model` para que el modelo responda.
- `do_sample=False`: decodificación greedy, así la comparación original vs. steered es determinista.
- `STEER_STRENGTH = 5`: acá se escala el vector (`h + 5v`). La fuerza es un hiperparámetro: muy chica no hace nada, muy grande rompe el texto.
- `.to(...device)`: el vector se calculó y bajó localmente; hay que mandarlo al device donde vive esa capa en el servidor.
- Salida guardada: original "LeBron James plays **basketball**."; steered "🤔🤔🤔🤔🤔 ... 🤔🤔🤔". Funcionó en el sentido de que habla en emojis, pero perdió el contenido (no responde la pregunta). Bueno para la pregunta del ejercicio 4.
- Detalle: la salida muestra `<bos><bos>`. `apply_chat_template` ya mete un `<bos>` en el string y el tokenizer agrega otro al tokenizar. No parece romper nada acá, pero es un clásico.

#### Ejercicio 4 (celdas 140–142, `# TERRON`)

Pide replicar el pipeline: 10 positivos + 10 negativos, activaciones, diferencia de medias, sumar durante la generación.

**Celda 140**: el dataset.

- `steer_negative`: 10 frases en inglés; `steer_positive`: las mismas en español. Son pares traducidos (como en `languages_df`), así que la diferencia de medias aísla sobre todo el idioma y no el contenido.
- Deliberadamente distintas de `languages_df`, para no reusar las frases del probe.
- `assert` para asegurar 10 y 10.
- Usa el modelo local (`model`, SmolLM2). El comentario de la celda original dice "`ndif_model` (remote, ~70B)", pero el modelo remoto cargado es de 9B; es un resto de otra versión del curso.

**Celda 141**:

```python
STEER_LAYER = 20
with torch.no_grad():
    with model.trace(steer_negative):
        neg_acts = model.model.layers[STEER_LAYER].output[:, -1, :].save()  # (10, hidden)
    with model.trace(steer_positive):
        pos_acts = model.model.layers[STEER_LAYER].output[:, -1, :].save()  # (10, hidden)
spanish_direction = pos_acts.mean(dim=0) - neg_acts.mean(dim=0)
print("steering vector norm:", spanish_direction.norm().item())
```

- Igual que la celda 113, con nombres propios (`STEER_LAYER` para no pisar `LAYER`, que a esa altura vale 20 igual).
- Dos traces seguidos dentro del mismo `no_grad`.
- Imprime la norma L2 del vector para tener una escala de referencia: si la norma es grande comparada con la de las activaciones, fuerzas altas van a romper el texto.

**Celda 142**:

```python
STEER_PROMPT = "My favorite thing to do on weekends is"
with model.generate(STEER_PROMPT, max_new_tokens=25, do_sample=False):
    original_output = model.generator.output.save()
...
for strength in [0.5, 1.0, 2.0, 4.0]:
    with model.generate(STEER_PROMPT, max_new_tokens=25, do_sample=False) as tracer:
        with tracer.all():
            model.model.layers[STEER_LAYER].output += strength * spanish_direction
        steered_output = model.generator.output.save()
    print(model.tokenizer.decode(steered_output[0]))
```

- Prompt neutro en inglés; baseline greedy.
- Barrido de fuerzas 0.5, 1, 2, 4, con la misma mecánica de `tracer.all()` + `+=`.
- `decode(steered_output[0])`: primera secuencia del batch (string, no lista).

No está ejecutado. Qué mirar para la celda 143: desde qué fuerza la continuación pasa a español, si a fuerzas altas el texto se degenera (repeticiones, tokens sin sentido, como los 🤔 de Gemma), y si en algún punto cambia el idioma pero se pierde la coherencia con el prompt. Un caso "no exitoso" puede ser una fuerza baja que no cambia nada, o una alta que produce basura.

---

## Ojo / errores comunes / dudas

- **`.save()`**: sin él, el valor desaparece al salir del `with`. Pasa mucho con listas: hay que hacer `.save()` en cada elemento que se agrega (como en la celda 28).
- **Orden de acceso en el trace**: con nnsight nuevo, pedir activaciones en orden distinto al del forward falla. Los loops sobre capas van en orden creciente.
- **`trace` vs. `generate`**: `trace` = un forward; `generate` = varios forwards. En `generate`, sin `tracer.all()` las intervenciones afectan solo al primer paso.
- **Padding en batch**: `output[:, -1, :]` sobre un batch de strings de distinto largo depende de que el padding esté a la izquierda. Con padding a la derecha estarías leyendo tokens de padding para las frases cortas.
- **`classes_` ordenado**: en binario, `coef_[0]` es para `classes_[1]`. En Prof. X eso es "low", por eso la intervención es `x − θ` y no `x + θ`.
- **Accuracy de entrenamiento**: 40 puntos en 2048 dims son separables casi por construcción. El 100% no dice que el probe generalice.
- **Signo de PCA arbitrario**: "debajo de 0 es francés" depende de la corrida; mejor comparar con los centros de los clusters (como hace la solución del ejercicio 2).
- **Variables pisadas**: `pca`, `low_dim_activations`, `LAYER`, `X`, `y`, `difference_in_means`, `english_sentences`, `fig` se redefinen varias veces. Si ejecutás fuera de orden, los resultados cambian sin aviso. En especial: `X`/`y` de Prof. X pisan los del probe de idiomas, y la celda 52 usa `LAYER` (10 cuando se escribió).
- **Frontera proyectada en PCA (celda 72)**: es aproximada (ignora el intercepto implícito por el centrado, lo que queda fuera del plano y el carácter multiclase).
- **Decodificable ≠ usado**: probing muestra correlación; para hablar de causalidad hay que intervenir (que es lo que hace el steering, y lo que la Lecture 6 profundiza).
- **Logit lens sin `norm`**: si te olvidás de `model.model.norm`, los resultados de las capas intermedias son mucho peores, porque `lm_head` espera vectores normalizados.
- **Fuerza de steering**: Gemma con fuerza 5 se fue a 🤔 repetidos. Siempre conviene barrer fuerzas.

---

## Resumen por sección

- **Setup**: carga SmolLM2-1.7B con nnsight; `generate` + `.save()` para obtener salidas.
- **Logit lens**: `lm_head(norm(h_ℓ))` en cada capa y token; en traducción ES→FR aparecen tokens en inglés ("love") en capas medias.
- **Ejercicio 1**: la misma celda como función, para varios prompts, imprimiendo el token de la última posición por capa.
- **PCA**: 20 activaciones (capa 10, último token) proyectadas a 2D; PCA 1 separa destino francés vs. español.
- **Ejercicio 2**: palabras que no cambian entre idiomas (chocolate, hotel...) proyectadas con la PCA de animales, clasificadas por el centro más cercano en PCA 1.
- **Probe supervisado**: regresión logística multinomial sin intercepto sobre 40 activaciones (capa 20); 100% en train.
- **Ejercicio 3**: PCA de 4 idiomas, frontera de inglés proyectada con `components_ @ coef_[en]`, y préstamos léxicos para buscar errores del probe.
- **Prof. X**: la regresión logística elige "nota relativa al esfuerzo" (θ ≈ (0.41, −2.12)); como intervención sugiere algo absurdo. La diferencia de medias sugiere "más tareas y más nota".
- **Steering local**: `v = mean(fr) − mean(en)` en la capa 20, sumado en cada paso con `tracer.all()`; la continuación pasa a francés.
- **NDIF**: `remote=True` corre Gemma-2-9B en el servidor; steering a emojis con fuerza 5 (sale 🤔 repetido).
- **Ejercicio 4**: steering inglés → español en SmolLM2 con pares traducidos y un barrido de fuerzas 0.5–4.
