# Lecture 7: Causal Mediation Analysis

> **Actualización:** el modelo se carga con `attn_implementation="eager"` (ver la nota de la 02). La notebook ya está ejecutada y tiene las respuestas escritas.

Esta notebook pasa de "mirar" adentro del modelo (logit lens, probes) a **intervenir** adentro del modelo de forma sistemática. La pregunta central es: cuando cambio un pedacito del input (por ejemplo "Eiffel" por "Tokyo") y la predicción cambia ("France" por "Japan"), ¿por qué componentes internos pasa ese cambio, en qué capa y en qué token? La herramienta es la **causal mediation analysis**: hacer una **interchange intervention** (copiar una activación de una corrida "source" a una corrida "base") en cada componente, uno por uno, y medir cuánto se mueve la predicción. Con eso se arma un mapa capa × token del flujo de información, se identifica la cabeza de atención que mueve la información entre tokens, se mira cómo el modelo trackea el estado de un objeto a lo largo de un texto y, al final, se combinan dos intervenciones para obtener un efecto composicional ("Tokyo").

Modelo: `HuggingFaceTB/SmolLM2-360M` (32 capas, 15 cabezas de atención por capa), manejado con `nnsight` (versión fijada en `requirements/07_causal_mediation_analysis.txt`: `nnsight==0.5.15`).

---

## Ideas clave / teoría

### Correlacional vs causal

El **logit lens** (lecture 4) toma el residual stream de una capa intermedia, le aplica la norma final y el `lm_head` (la matriz de unembedding) y mira qué token "decodifica". Es correlacional: que en la capa 27 se pueda leer "France" no implica que esa activación sea la que el modelo usa para producir "France". Puede ser información presente pero no usada. Para hablar de **causa** hay que intervenir: cambiar la activación y ver si cambia el output.

### Log odds como métrica

La notebook formaliza la conducta como una comparación entre dos tokens candidatos:

$$\text{LogOdds}(\text{France},\text{Japan} \mid c_E) := \log p(\text{France} \mid c_E) - \log p(\text{Japan} \mid c_E)$$

- $c_E$ es el contexto "Eiffel Tower is in the country of"; $c_T$ es "Tokyo Tower is in the country of".
- $p(\cdot \mid c)$ es la distribución del siguiente token (softmax de los logits en la última posición).
- La resta de log-probabilidades es el logaritmo del cociente $p(\text{France})/p(\text{Japan})$, o sea un log odds entre esas dos opciones.

Por qué esta métrica y no, por ejemplo, la probabilidad de "France" sola:

1. **Tiene signo**: $>0$ significa que el modelo prefiere France, $<0$ que prefiere Japan. La conducta correcta es $\text{LogOdds}(\cdot \mid c_E) > 0$ y $\text{LogOdds}(\cdot \mid c_T) < 0$. Lo que queremos explicar es justamente **cómo el modelo da vuelta el signo** según el contexto.
2. **Se cancela el resto del vocabulario**: $\log p(a) - \log p(b) = \text{logit}_a - \text{logit}_b$ (el término de normalización del softmax, el log-sum-exp, aparece en las dos y se resta). Entonces la métrica no depende de cuánta masa se va a otros tokens ("the", "Paris", etc.). Ojo: en el código igual se calcula con `log_softmax`, pero el resultado es idéntico a restar logits.
3. Es una escala aditiva y sin saturación (las probabilidades se aplastan cerca de 0 y 1, los log odds no).

### El efecto de una interchange intervention

Una **interchange intervention** (también "activation patching") necesita dos inputs:

- **base**: el input que corremos de verdad ("Eiffel Tower is in the country of").
- **source**: un input mínimamente distinto ("Tokyo Tower is in the country of").

Corremos el source, guardamos una activación interna $h$, y después corremos el base reemplazando su activación en ese mismo lugar por la del source. El resto del forward pass sigue normal.

La fórmula para el efecto de la última capa en el último token:

$$\text{LogOdds}(\text{France},\text{Japan} \mid c_E)-\text{LogOdds}(\text{France},\text{Japan} \mid c_E,\ h^{\text{last},32}_E \leftarrow h^{\text{last},32}_T)$$

Término a término:

- $h^{\text{last},32}_E$: el hidden state (residual stream, salida del bloque) de la capa 32 (la última; en el código son índices 0..31) en la **última posición** del contexto Eiffel.
- $h^{\text{last},32}_T$: lo mismo pero en el contexto Tokyo.
- $c_E,\ h^{\text{last},32}_E \leftarrow h^{\text{last},32}_T$: "corré el modelo sobre $c_E$, pero cuando llegues a ese vector, pisalo con el de Tokyo".
- El primer término es el log odds limpio (positivo). El segundo es el log odds intervenido. La resta es el **efecto**: cuánto bajó la preferencia por France gracias a meter esa activación de Tokyo.

Lectura de valores:
- Efecto ≈ 0: esa activación no lleva la información que distingue Eiffel de Tokyo (o no se usa).
- Efecto grande y positivo: la activación media la diferencia; meterla de Tokyo empuja al modelo hacia Japan. El máximo "natural" es cuando el log odds intervenido llega al de Tokyo, o sea efecto ≈ $\text{LogOdds}(c_E) - \text{LogOdds}(c_T)$.
- Efecto negativo: la intervención reforzó France (raro, pero posible).

En el vocabulario de [Vig et al. (2020)](https://arxiv.org/abs/2004.12265), esto es un **indirect effect** a través de un **mediator** (el componente que pisamos): el efecto del cambio de input ($c_E \to c_T$) que pasa *por* ese componente. El "total effect" sería cambiar el input entero; el "direct effect" sería lo que pasa por todos los demás caminos.

### Por qué capa × token

Repetimos la intervención para cada capa $\ell$ y cada posición $t$, y obtenemos una matriz de efectos. Dos cosas son ciertas siempre (la notebook lo remarca):

- **En la capa 0** solo tiene efecto la posición que cambiamos (Eiffel/Tokyo), porque en las demás posiciones el residual stream todavía no recibió información de otros tokens; son idénticas en base y source.
- **En la última capa** solo importa el último token, porque la predicción sale de ahí.

Entonces en algún momento la información tiene que **viajar** desde el token cambiado hasta el último token. Como el único mecanismo que mueve información entre posiciones en un Transformer es la **atención** (el MLP actúa posición por posición), ese salto lo hace alguna cabeza de atención. El heatmap muestra *cuándo* pasa.

### Cómo leer los heatmaps capa × token

- Eje x: grupos de tokens (con la etiqueta "base ↑ source", ej. "Eiffel ↑ Tokyo").
- Eje y: capa (0 abajo; `autorange='min'` en el código invierte el orden por defecto de `imshow`).
- Color: efecto en log odds (blanco = 0 / mínimo, rosa = máximo).
- Patrón típico: una columna rosa en el token cambiado desde la capa 0 hasta cierta capa $\ell^*$, y desde ahí la columna del último token se vuelve rosa hasta arriba. La capa donde "se cruza" es donde la atención transporta la información. Si una posición intermedia queda rosa en capas medias, el modelo está guardando ahí estado intermedio (sección 4).

### Mediadores más finos: cabezas de atención

En vez de pisar todo el residual stream de una capa, se puede pisar la salida de **una cabeza** de atención. En arquitecturas tipo Llama (SmolLM2 lo es), la atención calcula para cada cabeza un vector de tamaño `head_dim`, los **concatena** en un vector de tamaño `num_heads * head_dim` y después aplica `o_proj` (una lineal) para volver al residual stream. La **entrada** de `o_proj` es entonces la concatenación de las salidas de las cabezas, y la cabeza $h$ ocupa el slice `[h*attn_dim : (h+1)*attn_dim]`. Pisar ese slice = intervenir solo esa cabeza. Elegir el mediador (capa entera, cabeza, neurona, dirección) es una de las decisiones centrales del área ([Mueller et al. 2025](https://arxiv.org/abs/2408.01416)).

### Multi-source interventions

Si dos intervenciones distintas codifican variables distintas (entidad y atributo), combinarlas debería componer: "entidad = Tokyo Tower" + "atributo = city" → "Tokyo", aunque ninguna de las dos corridas source predecía "Tokyo". Eso descarta que las intervenciones solo estén "copiando la respuesta".

---

## Recorrido por la notebook

### 0. Setup

```python
try:
    import google.colab
    is_colab = True
except ImportError:
    is_colab = False
if is_colab:
    pio.renderers.default = "colab"
    !git clone ...; %cd ...; !pip install -r requirements/07_causal_mediation_analysis.txt
else:
    pio.renderers.default = "plotly_mimetype+png"
    !plotly_get_chrome -y
clear_output()
```

Detecta si estamos en Colab; ahí clona el repo e instala dependencias. Fuera de Colab configura plotly para que renderice como mimetype + PNG (por eso hace falta Chrome para exportar imágenes). Puro setup.

```python
from nnsight import LanguageModel

model = LanguageModel("HuggingFaceTB/SmolLM2-360M")

prompt = 'A Transformer is'
with model.generate(prompt) as tracer:
  output = model.generator.output.save()

model.to('cuda')
```

- `LanguageModel(...)` es el wrapper de nnsight sobre un modelo de HuggingFace. Carga el modelo de forma perezosa ("meta"), por eso se corre un `generate` de prueba para forzar la carga de pesos.
- `model.generate(prompt)` abre un contexto de generación. `model.generator.output` es la secuencia de tokens generada; `.save()` marca el valor para que sobreviva al salir del contexto (todo lo que no se guarda se descarta).
- `model.to('cuda')` pasa el modelo a GPU.

### 1. Revisiting probing: logit lens

```python
with model.generate(prompt, max_new_tokens=1) as tracer:
  output = model.generator.output.save()
print(model.tokenizer.decode(output)[0])
```

Genera un solo token y lo imprime. Salidas guardadas: "Eiffel Tower is in the country of France" y "Tokyo Tower is in the country of Japan". Ojo con `decode(output)[0]`: `output` tiene shape `(1, seq)`; por la salida guardada (la frase completa) el `[0]` toma el primer texto del batch, no el primer carácter. No verifiqué el detalle de cómo el tokenizer de nnsight maneja el tensor 2D.

**Logit lens sobre las capas 20–31:**

```python
layer_range = list(range(20, len(model.model.layers)))  # capas 20..31
decoded_layers = []
with model.trace(prompt):
    for layer_index in layer_range:
        layer_activation = model.model.layers[layer_index].output
        layer_activation = model.model.norm(layer_activation)
        decoded_layer_logits = model.lm_head(layer_activation)
        decoded_layer = torch.nn.functional.softmax(decoded_layer_logits, dim=-1)
        decoded_layers.append(decoded_layer.save())

decoded_layers = torch.cat(decoded_layers)  # (num_layers, num_tokens, vocab_size)
probabilities, decoded_tokens = decoded_layers.max(dim=-1)
decoded_tokens = [[model.tokenizer.decode(t.item()) for t in layer_tokens]
                  for layer_tokens in decoded_tokens]
```

- `model.trace(prompt)`: contexto de nnsight que corre **un** forward pass sobre el prompt. Dentro del bloque se escribe código que se "intercala" con la ejecución del modelo: cuando pedís `module.output`, nnsight espera a que ese módulo se ejecute y te da su salida. Por eso hay que acceder a los módulos en el orden en que se ejecutan (capa 20, 21, ...).
- `model.model.layers[i].output`: salida del bloque decoder $i$ = residual stream después de ese bloque, shape `(1, seq, hidden)`. (El comentario dice "start from layer 10" pero el código empieza en 20: el comentario está desactualizado.)
- `model.model.norm(...)`: aplica la RMSNorm final. Hay que hacerlo porque el `lm_head` espera activaciones normalizadas.
- `model.lm_head(...)`: proyecta al vocabulario, shape `(1, seq, vocab)`. Llamar módulos así dentro del trace es aplicar el módulo a un valor arbitrario (no engancha el forward real).
- `softmax` sobre el vocabulario y `.save()` de cada capa.
- `torch.cat` sobre la dimensión 0 apila las 12 capas: `(12, seq, vocab)`.
- `.max(dim=-1)` da, para cada (capa, posición), la probabilidad máxima y el id del token top.
- Se decodifican los ids para usarlos como texto en las celdas del heatmap.

El ploteo (`px.imshow` con `origin="lower"`, escala RdYlBu, `texttemplate` con los tokens decodificados) es estándar. `input_tokens` saca la marca `Ġ` (espacio en BPE tipo GPT-2) de los tokens.

Lo mismo se repite para "Tokyo Tower". La notebook interpreta que "France"/"Japan" aparecen en el último token alrededor de las capas 26–27 en ambos casos (y que "Paris" aparece sobre "in" más o menos al mismo tiempo). Pero esto es **decodificable**, no necesariamente **causal**: eso motiva la sección siguiente.

### 2. Causal Mediation Analysis

La parte teórica (log odds, fórmula con $h^{\text{last},32}$) está explicada arriba. Vamos al código.

```python
base_prompt = "Eiffel Tower is in the country of"
source_prompt = "Tokyo Tower is in the country of"
base_next_word = "France"
source_next_word = "Japan"

base_output = model.tokenizer(" " + base_next_word.strip())["input_ids"][0]
source_output = model.tokenizer(" " + source_next_word.strip())["input_ids"][0]
```

- Base y source son un **par mínimo**: difieren solo en la entidad. Tienen que tener la misma cantidad de tokens para que las posiciones se alineen (se asume que "Tokyo" ocupa los mismos 2 tokens que "Eiffel"; la notebook no lo chequea).
- Para los ids de los tokens objetivo se tokeniza `" France"` **con espacio adelante**, porque en el texto la palabra viene después de "of " y en BPE el espacio va pegado al token siguiente. `[0]` toma el primer token (se asume que " France" y " Japan" son un solo token; si no lo fueran estaríamos midiendo solo el primer pedazo). SmolLM2 no agrega BOS, por eso `[0]` es el token de la palabra.

```python
with model.trace(base_prompt) as tracer:
    original_logits = model.output.logits[:, -1, :].save()

original_logprobs = original_logits.log_softmax(dim=-1)
original_logodds = (original_logprobs[0, base_output] - original_logprobs[0, source_output]).item()
```

- `model.output.logits`: logits finales, shape `(1, seq, vocab)`. `[:, -1, :]` se queda con la última posición → `(1, vocab)`.
- `log_softmax` → log-probabilidades. Restar las de France y Japan da el log odds limpio (un float, por `.item()`).

```python
source_hidden_states = []
with torch.no_grad():
    with model.trace(source_prompt):
        source_hidden_states.append([
            layer.output.save()
            for layer in model.model.layers
        ])
source_hidden_states = source_hidden_states[0]
```

- Corre el source una vez y guarda la salida de **cada** capa: una lista de 32 tensores de shape `(1, seq, hidden)`.
- `torch.no_grad()` evita armar el grafo de gradientes (más rápido, menos memoria).
- El truco de meter la lista dentro de otra lista y después sacar `[0]` no aporta nada funcional; es solo estilo.

```python
token_idx_list = [[0,1],[2],[3],[4],[5],[6],[7]]

causal_effects = []
for layer_idx in range(model.config.num_hidden_layers):        # 32 capas
    causal_effect_per_layer = []
    for token_idx in token_idx_list:                            # 7 grupos
        with torch.no_grad():
            with model.trace(base_prompt) as tracer:
                model.model.layers[layer_idx].output[:, token_idx, :] = \
                    source_hidden_states[layer_idx][:, token_idx, :]

                intervened_logits = model.output.logits[:, -1, :]
                intervened_logprobs = intervened_logits.log_softmax(dim=-1)
                intervened_logodds = (intervened_logprobs[0, base_output] - intervened_logprobs[0, source_output]).item()
                effect = (original_logodds - intervened_logodds).save()
            causal_effect_per_layer.append(effect)
    causal_effects.append(causal_effect_per_layer)
```

Línea por línea:

- `token_idx_list`: grupos de posiciones a intervenir juntas. El prompt tiene 8 tokens; "Eiffel" ocupa las posiciones 0 y 1 (se tokeniza en dos pedazos), y se agrupan para que la columna represente la palabra entera. Las demás son "Tower", "is", "in", "the", "country", "of" (posiciones 2–7). Esto supone que "Tokyo" también ocupa exactamente 2 tokens, si no las posiciones quedarían corridas.
- Doble loop: 32 capas × 7 grupos = **224 forward passes** del base, cada uno con una intervención distinta.
- `model.model.layers[layer_idx].output[:, token_idx, :] = ...`: **la intervención**. Asigna in-place, en el residual stream de salida de esa capa y en esas posiciones, el valor guardado del source. Indexar con una lista (`token_idx`) en la dimensión de secuencia selecciona esas posiciones. Como es la salida del bloque, la capa siguiente ya recibe el valor pisado y todo lo de arriba se recalcula a partir de ahí. Ojo: esto asume que en la versión de transformers instalada la salida del decoder layer es un tensor; en versiones viejas era una tupla y había que usar `.output[0]`.
- `intervened_logits = model.output.logits[:, -1, :]`: el output final *con* la intervención aplicada.
- Se calcula el log odds intervenido igual que el limpio.
- `effect = original_logodds - intervened_logodds`, y `.save()` para sacarlo del trace. En nnsight 0.5 el código del trace se ejecuta con valores reales, así que `.item()` ya devuelve un float; nnsight permite llamar `.save()` también sobre objetos que no son tensores.
- Resultado: `causal_effects` es una lista 32 × 7 de floats.

**Etiquetas del plot:**

```python
base_tokens = [model.tokenizer.decode([token]).strip() for token in model.tokenizer(base_prompt+' '+base_next_word).input_ids]
source_tokens = [...]  # igual con el source
label_tokens = [''.join([base_tokens[i] for i in tokens]) + '<br>↑<br>' + ''.join([source_tokens[i] for i in tokens])
                for tokens in token_idx_list]
```

Decodifica cada token por separado y, para cada grupo, junta los pedazos ("E"+"iffel" → "Eiffel") y arma la etiqueta "base ↑ source" (la flecha indica "el source se mete en el base"). `<br>` es salto de línea en plotly.

**Heatmap:** `px.imshow(causal_effects, x=label_tokens, y=capas, color_continuous_scale=[[0,'#FFFFFF'],[1,'#E59EDD']])`, escala de blanco a rosa. `yaxis=dict(autorange='min')` pone la capa 0 abajo. Ojo: la escala va del mínimo al máximo de los datos, así que efectos negativos también se ven blancos.

**Interpretación de la notebook** (de la ejecución original; el gráfico no está guardado como texto):
1. Pocos componentes importan, y están localizados en posiciones concretas.
2. Hay una capa (**26**) donde la información "Eiffel vs Tokyo" pasa de la columna de la entidad a la columna del último token ("of").
3. Coincide con el logit lens.

Además se remarcan las dos verdades "triviales" (capa 0: solo el token cambiado; última capa: solo el último token). El contenido real del gráfico es **dónde y cuándo** se produce el salto entre ambas.

### Ejercicio 1 (resuelto, celdas `# TERRON`)

**Qué pide:** en vez de cambiar la entidad, cambiar el **atributo** que se pregunta (country vs city), para ver cómo el modelo sabe qué propiedad de la Torre Eiffel tiene que recuperar.

```python
base_prompt = "Eiffel Tower is in the country of"
source_prompt = "Eiffel Tower is in the city of"

base_next_word = "France"
source_next_word = "Paris"
```

Par mínimo: mismo monumento, cambia solo "country" → "city". El token candidato del source pasa a ser " Paris". La métrica ahora es $\text{LogOdds}(\text{France},\text{Paris})$: positiva en el base, y un efecto grande significa que la intervención empujó al modelo a contestar la ciudad.

```python
base_output = model.tokenizer(" "+base_next_word.strip())["input_ids"][0]
source_output = model.tokenizer(" "+source_next_word.strip())["input_ids"][0]

base_tokens_ids = model.tokenizer(base_prompt)["input_ids"]
source_tokens_ids = model.tokenizer(source_prompt)["input_ids"]
assert len(base_tokens_ids) == len(source_tokens_ids), (len(base_tokens_ids), len(source_tokens_ids))
print("base tokens:  ", model.tokenizer.convert_ids_to_tokens(base_tokens_ids))
print("source tokens:", model.tokenizer.convert_ids_to_tokens(source_tokens_ids))
```

Agregado de la solución: un **chequeo de alineación**. La interchange intervention posición por posición solo tiene sentido si base y source tienen el mismo largo (si "city" fueran dos tokens, la posición 7 ya no sería "of" en ambos). El `assert` corta si no coinciden y los `print` muestran la tokenización para verificarlo a ojo (con los tokens crudos, `Ġ` incluido).

```python
with model.trace(base_prompt) as tracer:
    original_logits = model.output.logits[:, -1, :].save()
original_logprobs = original_logits.log_softmax(dim=-1)
original_logodds = (original_logprobs[0, base_output] - original_logprobs[0, source_output]).item()
print("original log odds (France vs Paris):", original_logodds)
```

Igual que antes, pero ahora France vs Paris. Conviene mirar que sea positivo (el modelo efectivamente prefiere France en el base); si fuera cercano a 0, los efectos serían chicos y difíciles de leer.

```python
source_hidden_states = []
with torch.no_grad():
    with model.trace(source_prompt):
        source_hidden_states.append([layer.output.save() for layer in model.model.layers])
source_hidden_states = source_hidden_states[0]
```

Residual stream de las 32 capas para el prompt "city".

```python
token_idx_list = [[0,1],[2],[3],[4],[5],[6],[7]]
# ... mismo doble loop que en la sección 2 ...
```

El loop es idéntico al de la sección 2. Lo único que cambió es qué token difiere entre base y source: ahora es la posición 6 ("country"/"city"), que corresponde al grupo de índice 5 dentro de `token_idx_list`. En la capa 0 la columna rosa debería ser esa, no la de "Eiffel".

Las etiquetas se arman igual (la columna de interés queda "country ↑ city"). La celda del plot agrega un análisis numérico:

```python
effects = np.array(causal_effects, dtype=float)  # (32, 7)
changed_col, last_col = 5, 6
for layer_idx, row in enumerate(effects):
    print(f"layer {layer_idx:2d} | country/city: {row[changed_col]:6.2f} | last token (of): {row[last_col]:6.2f}")
crossed = np.where(effects[:, last_col] > effects[:, changed_col])[0]
print("primera capa donde el último token tiene más efecto que 'country/city':",
      int(crossed[0]) if len(crossed) else "ninguna")
```

- Convierte la lista a array `(capas, grupos)`.
- `changed_col=5` es el grupo `[6]` ("country/city"); `last_col=6` es el grupo `[7]` ("of").
- Imprime, por capa, el efecto en esas dos columnas: es la versión numérica de mirar dos columnas del heatmap.
- `crossed`: índices de capas donde el efecto del último token supera al del token cambiado; `crossed[0]` es la primera, una definición operativa de "la capa donde la información se mueve".

**Qué mirar para responder** (celda `> FILL IN YOUR ANSWER HERE`): la capa del cruce, y compararla con la 26 que se encontró para Eiffel vs Tokyo y con lo que mostraba el logit lens. Para razonar el "por qué": pensá qué tan lejos está "country" de "of" (es el token inmediatamente anterior) comparado con "Eiffel", y si tiene sentido que el modelo necesite saber *qué atributo* le piden antes o después de recuperar *de qué entidad* habla. Ojo con la definición de `crossed`: es "la primera capa donde last > changed", que puede ser ruidosa si los dos valores están cerca; mirá también el heatmap. La sección 5 de la notebook (que interviene el atributo en capas 0–5) da una pista fuerte de qué esperar.

### 3. Otros componentes: cabezas de atención

Como la información salta de la entidad al último token en la capa ~26, y eso solo lo puede hacer la atención, ahora se busca **qué cabeza** lo hace.

```python
source_hidden_states = []
with torch.no_grad():
    with model.trace(source_prompt):
        source_hidden_states.append([
            layer.self_attn.o_proj.input.save()
            for layer in model.model.layers
        ])
source_hidden_states = source_hidden_states[0]
```

- `layer.self_attn.o_proj.input`: en nnsight, `.input` de un módulo es su primer argumento posicional. Para `o_proj` es la concatenación de las salidas de todas las cabezas (ya multiplicadas por los valores y ponderadas por la atención), shape `(1, seq, num_heads * head_dim)`.
- Se guarda para las 32 capas del source (Tokyo).

```python
attn_dim = model.config.hidden_size // model.config.num_attention_heads

causal_effects = []
for layer_idx in range(24,32):
    causal_effect_per_layer = []
    for head_index in range(model.config.num_attention_heads):
        with torch.no_grad():
            with model.trace(base_prompt) as tracer:
                attention_value = model.model.layers[layer_idx].self_attn.o_proj.input
                attention_value[:, -1, head_index * attn_dim:(head_index + 1) * attn_dim] = \
                source_hidden_states[layer_idx][:, -1, head_index * attn_dim:(head_index + 1) * attn_dim]
                model.model.layers[layer_idx].self_attn.o_proj.input = attention_value

                intervened_logits = model.output.logits[:, -1, :]
                intervened_logprobs = intervened_logits.log_softmax(dim=-1)
                intervened_logodds = (intervened_logprobs[0, base_output] - intervened_logprobs[0, source_output]).item()
                effect = (original_logodds - intervened_logodds).save()
            causal_effect_per_layer.append(effect)
    causal_effects.append(causal_effect_per_layer)
```

- `attn_dim`: dimensión por cabeza = `hidden_size / num_attention_heads`. Para SmolLM2-360M serían 960/15 = 64 (no lo verifiqué contra el config, pero la notebook confirma 15 cabezas en la salida `torch.Size([1, 15, 6, 6])`). Esto funciona porque en este modelo `num_heads * head_dim == hidden_size`; en modelos donde no es así habría que usar `head_dim` del config.
- Solo capas 24–31 "para ahorrar tiempo" (la sección anterior dijo que lo interesante pasa cerca de la 26). 8 capas × 15 cabezas = 120 forward passes.
- `attention_value = ...o_proj.input`: agarra el tensor de entrada de `o_proj` en la corrida base.
- La asignación con el slice `[:, -1, h*attn_dim:(h+1)*attn_dim]`: **solo la última posición** y **solo las 64 dimensiones de la cabeza h** se pisan con las del source. Es decir, el mediador es "la salida de la cabeza h de la capa ℓ en el último token". Se interviene en el último token porque es ahí donde la cabeza *escribe* la información que trajo de "Eiffel" (la atención mueve info desde las keys/values de otras posiciones hacia la query, que es la última).
- `model.model.layers[layer_idx].self_attn.o_proj.input = attention_value`: vuelve a asignar el input. Como la asignación anterior ya fue in-place, esta línea es más que nada explícita/defensiva.
- El resto es igual: log odds intervenido, efecto, `.save()`.
- Resultado: matriz 8 × 15 (capa × cabeza). En el plot el eje x dice `'token'` pero en realidad son **cabezas** (etiqueta heredada; en el ejercicio 2 está corregido a `'head'`).

**Interpretación de la notebook:** gran efecto en capa 26, **solo en la cabeza 10**. Esa cabeza transporta la información de la entidad al último token. Lo notable es la **esparsidad**: muy pocos componentes hacen casi todo el trabajo para una tarea dada (como en el circuito IOI de [Wang et al. 2023](https://arxiv.org/abs/2211.00593), con 26 cabezas en 7 clases).

### Patrones de atención

```python
model.config._attn_implementation = "eager"
prompt = "The Eiffel Tower is in"

with model.trace(prompt):
  outputs = model.model.layers[26].self_attn.source.attention_interface_0.output.save()

attn_outputs, attn_weights = outputs
attn_weights.shape   # salida guardada: torch.Size([1, 15, 6, 6])
```

- `_attn_implementation = "eager"`: fuerza la implementación "a mano" de la atención. Con SDPA/flash-attention los pesos de atención no se materializan (vuelven como `None`), así que para verlos hace falta eager.
- `self_attn.source`: feature de nnsight 0.5 que expone las **operaciones internas** del `forward` de un módulo (las llamadas a funciones dentro de su código fuente). `attention_interface_0` es la primera llamada a `attention_interface(...)` dentro de `LlamaAttention.forward`; su `.output` es la tupla `(attn_output, attn_weights)`.
- `attn_weights`: shape `(batch=1, heads=15, query=6, key=6)`. El prompt tiene 6 tokens: "The", "E", "iffel" (Eiffel en 2 tokens), "Tower", "is", "in". Ojo que este prompt es distinto al de la sección anterior ("The ... is in" en vez de "... is in the country of"), pero la cabeza se supone que hace lo mismo.

```python
fig = px.imshow(
    attn_weights[0, 10].detach().cpu().float().T,
    x=[f"{i}) {l}" for i, l in enumerate(tokens)],
    y=[f"{i}) {l}" for i, l in enumerate(tokens)],
    ...
)
```

- `attn_weights[0, 10]`: matriz `(query, key)` de la cabeza 10. Fila $i$ = cómo distribuye la atención el token $i$ sobre las keys (suma 1, y es triangular inferior por la máscara causal).
- `.T` la transpone: ahora filas = keys (eje y) y columnas = queries (eje x). Entonces **cada columna** es la distribución de atención de un token query. Por eso la notebook dice "mirá la columna arriba de 'in'": esa columna dice a qué tokens mira "in".
- El comentario `# look at attention head #3` está desactualizado (mira la 10).

Lectura: en la cabeza 10, la columna de "in" pone casi todo el peso sobre los tokens de "Eiffel". Es la firma de una cabeza que **mueve información de la entidad al último token**, coherente con el resultado causal. Se compara con la cabeza 11 (mismo código con `attn_weights[0, 11]`), que no tiene ese patrón.

Ojo: el patrón de atención es evidencia **correlacional** de nuevo (dónde mira la cabeza), pero acá complementa el resultado causal: sabemos que la cabeza importa y además vemos de dónde lee.

### Ejercicio 2 (resuelto, celdas `# TERRON`)

**Qué pide:** repetir el análisis por cabezas de atención pero con el par country/city, para ver qué cabeza mueve el **atributo** y si es la misma que movía la entidad (capa 26, cabeza 10).

```python
base_prompt = "Eiffel Tower is in the country of"
source_prompt = "Eiffel Tower is in the city of"
base_next_word = "France"
source_next_word = "Paris"
```

Mismo par que el ejercicio 1 (lo sugiere el hint).

La celda de log odds original es igual a la del ejercicio 1 (con `print`). La de activaciones del source guarda `o_proj.input` de las 32 capas, igual que en la sección 3.

```python
attn_dim = model.config.hidden_size // model.config.num_attention_heads
LAYERS_TO_CHECK = list(range(model.config.num_hidden_layers))

causal_effects = []
for layer_idx in LAYERS_TO_CHECK:
    causal_effect_per_layer = []
    for head_index in range(model.config.num_attention_heads):
        head_slice = slice(head_index * attn_dim, (head_index + 1) * attn_dim)
        with torch.no_grad():
            with model.trace(base_prompt) as tracer:
                attention_value = model.model.layers[layer_idx].self_attn.o_proj.input
                attention_value[:, -1, head_slice] = source_hidden_states[layer_idx][:, -1, head_slice]
                model.model.layers[layer_idx].self_attn.o_proj.input = attention_value
                ...
                effect = (original_logodds - intervened_logodds).save()
            causal_effect_per_layer.append(effect)
    causal_effects.append(causal_effect_per_layer)
```

Diferencias con la sección 3:

- `LAYERS_TO_CHECK` = **todas** las capas (0–31), porque no hay por qué suponer que el atributo se mueve en la misma capa que la entidad. Son 32 × 15 = 480 forward passes (tarda más).
- `head_slice = slice(...)`: el mismo slice de antes pero en un objeto `slice` con nombre, más legible. `x[:, -1, head_slice]` equivale a `x[:, -1, a:b]`.
- Se sigue interviniendo solo el último token ("of"). Ojo: acá "country" está en la posición inmediatamente anterior, así que una cabeza que lea "country" desde "of" puede actuar muy temprano.

El plot (eje x ahora bien etiquetado como `'head'`) y después:

```python
effects = np.array(causal_effects, dtype=float)       # (32, 15)
top = np.argsort(effects.ravel())[::-1][:5]
for flat in top:
    l, h = divmod(int(flat), effects.shape[1])
    print(f"layer {LAYERS_TO_CHECK[l]}, head {h}: effect = {effects[l, h]:.2f}")
print("efecto de layer 26 / head 10 (la de Eiffel vs Tokyo):", f"{effects[LAYERS_TO_CHECK.index(26), 10]:.2f}")
```

- `effects.ravel()` aplana la matriz a 480 valores; `argsort(...)[::-1][:5]` da los índices de los 5 efectos más grandes.
- `divmod(flat, 15)` convierte el índice plano en (fila, columna) = (capa, cabeza).
- Último print: el efecto de la cabeza 26.10 en este nuevo experimento, para comparar directamente.

**Qué mirar para responder:** cuáles son las top cabezas (en qué capas están: tempranas, medias, tardías) y si 26.10 aparece o tiene efecto chico. Para el "por qué": pensá que son dos variables distintas (qué entidad vs qué atributo), que en el texto vienen de tokens distintos a distinta distancia del final, y que el modelo podría necesitar el atributo *antes* de hacer el lookup de la entidad. Ojo que las intervenciones son sobre el último token solamente: una cabeza que mueve el atributo a otra posición intermedia no se vería acá.

### 4. Tracking states ("the cup")

Hasta acá la info pasaba directo de la entidad al último token. Ahora se pregunta si el modelo guarda **estado intermedio** en otras posiciones cuando el texto es más largo (una especie de "working memory").

```python
base_prompt = "Alice poured coffee into a cup. Then, she put it in the fridge. Now, the cup is"
source_prompt = "Alice poured coffee into a cup. Then, she drank it in one go. Now, the cup is"
```

Salidas guardadas: el base continúa con "cold" y el source con "empty". El par mínimo cambia la **situación** (heladera vs se lo tomó), y con eso el estado de la taza.

```python
base_output = model.tokenizer(" cold")["input_ids"][0]
source_output = model.tokenizer(" empty")["input_ids"][0]

with model.generate(base_prompt) as tracer:
    original_logits = model.output.logits[:, -1, :].save()
```

- Métrica: $\text{LogOdds}(\text{cold}, \text{empty})$.
- Ojo: acá se usa `model.generate` en vez de `model.trace`. Dentro de generate, `model.output` corresponde al primer paso (el forward sobre el prompt completo), así que `logits[:, -1, :]` sigue siendo la predicción del siguiente token. Funciona, pero `model.trace` sería lo consistente con el resto.

La celda de activaciones del source es igual a la sección 2 (`layer.output` de las 32 capas).

```python
token_idx_list = [
    [0,1,2,3,4,5,6],            # setup: Alice poured the cup
    [7,8,9,10,11,12,13,14,15],  # situation: fridge vs. drinking
    [16,17],                    # "Now,"
    [18,19],                    # "the cup"
    [20]                        # "is" (final token!)
]
```

- En vez de un token por columna, se agrupan por **segmentos** del texto: el setup (7 tokens: "Alice poured coffee into a cup."), la situación (9 tokens: "Then, she put it in the fridge." / "Then, she drank it in one go."; los dos tienen que tener 9 tokens para que esto cuadre), "Now,", "the cup", "is".
- Intervenir un grupo = pisar todas esas posiciones a la vez en esa capa. Es más barato (5 columnas en vez de 21) a costa de resolución.
- El loop es idéntico al de la sección 2: 32 × 5 = 160 forward passes.

El plot usa etiquetas a mano y `zmax=4`, que **satura** la escala de color en 4: cualquier efecto ≥ 4 se ve igual de rosa. Sirve para que efectos medianos no queden blancos al lado de uno enorme, pero hay que tenerlo en cuenta al comparar intensidades.

**Interpretación de la notebook:** el estado de la taza (fría vs vacía) aparece representado en los tokens **"the cup"**. O sea, el patrón no es "situación → último token" directo, sino situación → "the cup" en capas medias → "is". El modelo **agrega información contextual** en el token que nombra a la entidad cuyo estado importa. La lectura en el heatmap: la columna de la situación es rosa abajo (es lo que cambiamos), después se prende "the cup" en un rango de capas, y recién después "is".

Referencia: [Li et al. (2025)](https://arxiv.org/abs/2503.02854) estudian state tracking con secuencias de swaps entre objetos y encuentran dos mecanismos (tipo associative scan, o usando paridad de permutaciones para recortar opciones antes del scan).

### Ejercicio 3: Winograd (en workbench, sin código)

Se hace en la web **workbench** (https://workbench-git-v2-ndif.vercel.app/), con `llama-3.3-70B-Instruct` para que el modelo sea capaz de resolver la tarea. Las dos oraciones:

1. "Alice returned the book to Sarah because she **finished** it. Question: Who does 'she' refer to? Answer:" → Alice.
2. "Alice returned the book to Sarah because she **needed** it. ..." → Sarah.

Es un **Winograd schema**: un par de oraciones que difieren en una palabra y donde el referente del pronombre cambia, y para resolverlo hace falta conocimiento del mundo (un modelo de la situación: por qué alguien devuelve un libro). No alcanza con sintaxis.

Pasos y qué razonar en cada uno:

- **[1] Swap del último token.** Es el control de sanidad: en la última capa el último token siempre tiene efecto (de ahí sale la predicción). Si el swap del último token no cambia la respuesta de Alice a Sarah, algo está mal en el setup (par mal armado, o el modelo no distingue los casos).
- **[2] Swap de palabras del medio.** Hay que buscar qué posición intermedia tiene efecto en capas medias. El hint dice que no conviene probar todas: pensá dónde *debería* agregar información el modelo, a la luz de "the cup" en la sección 4. Candidatos naturales: el pronombre "she" (en la oración y en la pregunta), la palabra que cambia ("finished"/"needed"), "it". [Yamakoshi et al. (2023)](https://arxiv.org/abs/2306.03882) encontraron justamente un pequeño circuito de cabezas que lleva la info de la palabra decisiva hacia el **pronombre**.
- **[3] Cambiar los nombres del source a Bec y Cynthia** (dejando el base con Alice y Sarah) y hacer swap del "she" de la pregunta. Acá la pregunta es qué codifica la representación del pronombre: si guardara el **nombre** concreto ("Sarah"), patchearla desde un source con Bec/Cynthia debería empujar a responder un nombre del source (o romper la respuesta); si guarda algo **posicional/relacional** (por ejemplo "el primer mencionado" vs "el segundo"), entonces el swap debería dar Alice/Sarah según la posición del referente en el source, aunque los nombres sean otros. Hay que formular la expectativa antes y comparar.

Para la pregunta final (¿guarda info intermedia en tokens específicos? ¿cuáles? ¿por qué aprendería eso?), lo que hay que traer es lo observado en [2]/[3] y conectarlo con la idea de que agregar el estado en el token de la entidad/pronombre permite que después cualquier pregunta sobre esa entidad lo lea con una sola cabeza de atención, en vez de re-procesar todo el contexto.

### 5. Multi-source intervention

Objetivo: con el base "Eiffel Tower is in the country of", hacer que el modelo diga **"Tokyo"**, combinando:
- source 1: "Tokyo Tower is in the country of" → cambia la **entidad** (Japan),
- source 2: "Eiffel Tower is in the city of" → cambia el **atributo** (Paris).

Salidas guardadas: el base genera "France", source 1 "Japan", source 2 "Paris". Ninguno genera "Tokyo".

```python
base_output = model.tokenizer(" France")["input_ids"][0]
source_output_1 = model.tokenizer(" Japan")["input_ids"][0]
source_output_2 = model.tokenizer(" Paris")["input_ids"][0]
source_output = model.tokenizer(" Tokyo")["input_ids"][0]
```

Ids de los cuatro tokens a comparar (ojo: el nombre `source_output` acá es el de "Tokyo", el objetivo compuesto).

Dos celdas guardan `o_proj.input` de las 32 capas para cada source (`source_hidden_states_1` y `_2`), igual que en la sección 3.

```python
token_idx = -1
logprobs = []
for interv_type in ['no_intervention','source_country','source_property','combined']:
    if interv_type=='no_intervention':  interv_country, interv_property = False, False
    if interv_type=='source_country':   interv_country, interv_property = True, False
    if interv_type=='source_property':  interv_country, interv_property = False, True
    if interv_type=='combined':         interv_country, interv_property = True, True
    with torch.no_grad():
        with model.trace(base_prompt) as tracer:
            for layer_idx in range(model.config.num_hidden_layers):
                if layer_idx in [24,25,26,27] and interv_country:
                    model.model.layers[layer_idx].self_attn.o_proj.input[:, token_idx, :] = \
                        source_hidden_states_1[layer_idx][:, token_idx, :]
                if layer_idx in [0,1,2,3,4,5] and interv_property:
                    model.model.layers[layer_idx].self_attn.o_proj.input[:, token_idx, :] = \
                        source_hidden_states_2[layer_idx][:, token_idx, :]
            intervened_logits = model.output.logits[:, -1, :]
            intervened_logprobs = intervened_logits.log_softmax(dim=-1).save()
    logprobs.extend([[interv_type, 'France', intervened_logprobs[0,base_output].item()],
                     [interv_type, 'Japan', intervened_logprobs[0,source_output_1].item()],
                     [interv_type, 'Paris', intervened_logprobs[0,source_output_2].item()],
                     [interv_type, 'Tokyo', intervened_logprobs[0,source_output].item()]])
df = pd.DataFrame(logprobs, columns=['intervention','next token','log probability'])
```

- 4 condiciones: sin intervención, solo entidad, solo atributo, ambas. Los flags `interv_country`/`interv_property` deciden qué se aplica.
- `token_idx = -1`: siempre en el último token.
- El loop sobre capas recorre de 0 a 31 **en orden**, lo cual importa en nnsight (hay que tocar los módulos en el orden de ejecución).
- **Entidad**: en capas 24–27 se reemplaza **toda** la entrada de `o_proj` (todas las cabezas, sin slice) en el último token por la del source 1. Es una versión gruesa del hallazgo "capa 26 mueve la entidad".
- **Atributo**: en capas 0–5 se reemplaza toda la entrada de `o_proj` en el último token por la del source 2. Estas capas tempranas son la elección de la notebook para la info de "country vs city" (y es coherente con lo que debería salir del ejercicio 2).
- Se guardan las log-probs completas y se extraen las de los 4 tokens.
- `df` en formato largo: (intervención, token, log-prob), 16 filas, listo para un bar chart agrupado.

El `px.bar(..., barmode="group")` muestra, para cada intervención, las log-probs de los 4 tokens.

**Interpretación de la notebook:** sin intervención gana France; con la de entidad gana Japan (réplica de la sección 3); con la de atributo gana Paris (réplica del ejercicio 2); combinadas gana **Tokyo**.

Por qué importa: una explicación alternativa sería que las intervenciones son **específicas de ítem**, que lo que meten es "decí Japan" o "decí Paris". Si fuera así, combinarlas daría Japan o Paris, nunca Tokyo, porque ningún source tiene Tokyo como respuesta. Que aparezca Tokyo muestra que una intervención lleva la variable abstracta "entidad = Tokyo Tower" y la otra "atributo = city", y el resto del modelo las **compone**. Esto prepara el terreno para la causal abstraction (lecture 8), donde las activaciones se alinean con variables de un modelo causal de alto nivel.

---

## Ojo / errores comunes / dudas

- **Alineación de tokens.** Toda la notebook interviene posición por posición. Si base y source no tienen el mismo número de tokens, las posiciones no se corresponden y el resultado no significa nada. La solución del ejercicio 1 agrega un `assert` para esto; en los demás casos se asume.
- **Espacio en el token de respuesta.** `" France"` y `"France"` son tokens distintos. Se usa con espacio porque viene después de "of ". Y se asume que es **un** token; si no, `[0]` mide solo el primer pedazo.
- **El signo del efecto.** Efecto = original − intervenido. Positivo = la intervención empujó hacia la respuesta del source. Con la escala blanco→rosa, los negativos no se distinguen de cero.
- **`zmax=4`** en la sección 4 satura colores: no compares intensidades por encima de 4.
- **Etiqueta `'token'`** en el plot por cabezas de la sección 3: en realidad el eje x son cabezas.
- **Comentarios desactualizados**: "start from layer 10" (empieza en 20), "look at attention head #3" (mira 10 y 11). No confundirse.
- **Intervenir el residual stream vs la salida de una cabeza.** Pisar `layers[i].output` reemplaza todo lo acumulado hasta esa capa en esa posición; pisar un slice de `o_proj.input` reemplaza solo la contribución de una cabeza en esa capa (el resto del residual sigue siendo del base). Son mediadores distintos y miden cosas distintas.
- **Por qué `o_proj.input` y no `o_proj.output`.** La salida de `o_proj` ya mezcló todas las cabezas con una matriz; ahí no hay un slice por cabeza. En la entrada sí están separadas.
- **Patrones de atención con SDPA.** Sin `_attn_implementation = "eager"` los pesos no se calculan. Además, cambiar esto en el config afecta las corridas siguientes (más lentas).
- **El prompt del attention pattern es otro** ("The Eiffel Tower is in"), así que se está asumiendo que la misma cabeza hace lo mismo en un prompt parecido.
- **`.output` de un decoder layer**: el código lo trata como tensor. En versiones de transformers donde el layer devuelve una tupla, esto falla y hay que usar `.output[0]`.
- **nnsight y el orden**: dentro de un `trace`, acceder a módulos en un orden distinto al de ejecución da error (el módulo ya corrió). Por eso los loops van de capa baja a alta.
- **Un solo par de prompts.** Todo se basa en un único par (o un par de pares). Lo de "cabeza 10 en capa 26" es para *este* input; generalizar requiere muchos pares (lo que hace causal abstraction).
- **Celda de la sección 4 con `generate`**: funciona porque `model.output` es el primer forward, pero es una inconsistencia menor.

---

## Resumen de una línea por sección

- **0. Setup**: carga SmolLM2-360M con nnsight y lo pasa a GPU.
- **1. Logit lens**: "France"/"Japan" se pueden decodificar en el último token hacia las capas 26–27, pero eso es correlación.
- **2. Causal mediation**: efecto = log odds limpio − log odds con una activación del source; barrido capa × token muestra que la info salta de "Eiffel/Tokyo" al último token en la capa ~26.
- **Ejercicio 1**: mismo barrido con country vs city; se busca la capa donde la info pasa de "country/city" a "of" y se compara con la 26.
- **3. Cabezas de atención**: parchear slices de `o_proj.input` por cabeza en el último token encuentra que la cabeza 10 de la capa 26 mueve la entidad; su patrón de atención mira a "Eiffel" desde "in".
- **Ejercicio 2**: barrido por cabeza en todas las capas con country vs city; se compara la cabeza top con 26.10.
- **4. Tracking states**: con fridge vs drank, el estado de la taza se guarda en los tokens "the cup" antes de llegar al último token.
- **Ejercicio 3**: Winograd en workbench; buscar en qué tokens intermedios (pronombre) se agrega la info y si codifica nombres o posiciones.
- **5. Multi-source**: entidad (capas 24–27, source Tokyo) + atributo (capas 0–5, source city) combinadas hacen decir "Tokyo", lo que muestra un efecto composicional y no una copia de respuesta.
