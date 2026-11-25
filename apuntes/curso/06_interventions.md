# Lecture 6: Intervenciones sobre redes neuronales

> **Actualización:** la solución del Ejercicio 1 importa `torch`, que esa celda necesitaba. La notebook ya está ejecutada (sin la sección de ReFT, que instala `pyreft`) y tiene las respuestas escritas.

Si la Lecture 4 preguntaba "¿qué podemos **leer** de las activaciones?", esta pregunta "¿qué pasa si las **cambiamos**?". La notebook recorre tres familias de intervenciones: **ablations** (sacar información, con INLP para quitar un sesgo de género en OLMo-2-1B), **interchange interventions** (cambiar activaciones entre inputs para descubrir qué componente es causalmente relevante, usando una red de juguete que calcula el máximo de dos números) y **steering** entrenado (ReFT/LoReFT con `pyreft`, para que TinyLlama responda en emojis). El hilo conductor es que toda intervención externa hace supuestos sobre cómo el modelo representa la información (por ejemplo, "0 significa nada" o "la información es lineal"), y esos supuestos pueden fallar.

---

## Ideas clave / teoría

### Tipos de intervención

- **Ablation**: modificar una activación para *quitar* información. Las variantes clásicas son poner en cero (**zero-out**), reemplazar por la media, o proyectar para borrar una dirección (INLP).
- **Interchange intervention** (también **activation patching**): correr el modelo con un input **base**, pero reemplazando una activación interna por el valor que tomó con otro input **source**. Si la salida cambia, esa activación tiene un efecto causal en la salida. La ventaja es que no inventa valores: usa valores que el propio modelo produce.
- **Steering**: modificar activaciones para *inducir* un comportamiento. En la Lecture 4 fue difference-in-means (sin entrenar); acá es ReFT (entrenado con descenso por gradiente).

### Métrica de sesgo (minimal pairs)

Siguiendo la Lecture 3, se compara `p(she | "The [profesión] said that")` vs. `p(he | ...)`. Un **minimal pair** son dos inputs que difieren solo en la palabra que nos interesa (acá, la profesión), así que la diferencia en la salida se le atribuye a eso.

### INLP (iterative nullspace projection)

Ravfogel et al. (2020). Objetivo: que ningún clasificador lineal pueda recuperar un atributo (acá, si la profesión es estereotípicamente masculina o femenina) a partir de las activaciones. Algoritmo:

1. Entrenar un probe lineal `w` que clasifique el atributo.
2. Proyectar todas las activaciones al **nullspace** de `w` (el subespacio de vectores ortogonales a `w`), es decir, borrar la componente a lo largo de `w`.
3. Repetir hasta que el probe no pueda clasificar.

La proyección de un vector `x` al nullspace de `w`:

$$
x' = x - \frac{x \cdot w}{w \cdot w}\, w = \left(I - \frac{w w^\top}{\lVert w \rVert^2}\right) x
$$

- `x · w`: cuánto tiene `x` en la dirección de `w` (sin normalizar).
- `/ (w · w)`: normaliza, para que `(x·w / w·w) w` sea exactamente la proyección de `x` sobre la recta de `w`.
- Restar esa proyección deja solo la parte ortogonal: `x' · w = x·w − (x·w/w·w)(w·w) = 0`. El probe `w` ya no ve nada (con `fit_intercept=False` su score es 0 para todos, o sea que queda en el punto de indecisión).
- `I − wwᵀ/‖w‖²` es la matriz de proyección (simétrica, idempotente, de rango `d−1`).

Cada iteración borra una dirección. La justificación es: el modelo no puede usar información que no está en sus activaciones. El límite: solo borra información **lineal**; algo codificado de forma no lineal sobrevive.

### MaxNN: el "0" no significa "nada"

Red de juguete con ReLU que calcula `max(x, y)` para `x, y > 0`. Con `r = ReLU`:

- Capa 1 (4 neuronas): `l1 = [x − y, y − x, x, y]`
- `r1 = ReLU(l1) = [ReLU(x−y), ReLU(y−x), ReLU(x), ReLU(y)]`
- Capa 2 (2 neuronas): `l2 = [−100·r1[0] + r1[3], −100·r1[1] + r1[2]]`
- Salida: `ReLU(l2[0]) + ReLU(l2[1])`

Interpretación neurona por neurona:
- `r1[0] = ReLU(x−y)`: positivo si `x > y`, 0 si no. Es una "bandera" de "ganó x".
- `r1[1] = ReLU(y−x)`: bandera de "ganó y".
- `r1[2], r1[3]`: copias de `x` e `y` (para positivos, `ReLU` no los toca).
- `l2[0] = −100·r1[0] + y`: si ganó `x`, se le resta un número enorme y después de ReLU queda 0; si ganó `y` (`r1[0] = 0`), vale `y`. O sea "y si x < y, si no 0".
- `l2[1] = −100·r1[1] + x`: "x si y < x, si no 0".
- La suma final elige la que sobrevivió.

Acá el 0 de `r1[0]` no significa "sin información": significa "x no es el mayor", y es justamente lo que deja pasar a `y`. Por eso un zero-out puede *escribir* información en vez de borrarla.

### ReFT / LoReFT

Representation fine-tuning (Wu et al., 2024): en vez de entrenar pesos (fine-tuning, LoRA), se deja el modelo **congelado** y se entrena una intervención sobre las activaciones de ciertas capas y posiciones. La variante LoReFT:

$$
h_{\text{intervened}} = h + R^\top (W h + b - R h)
$$

- `h ∈ ℝ^d`: la activación original (residual stream en una capa y posición).
- `R ∈ ℝ^{r×d}`, con filas ortonormales (`R Rᵀ = I_r`): define un subespacio de dimensión `r` (acá `r = 4`, con `d` = hidden size del modelo).
- `R h ∈ ℝ^r`: las coordenadas actuales de `h` en ese subespacio.
- `W h + b ∈ ℝ^r`: las coordenadas "objetivo", una función lineal aprendida de `h`. Es el "source" aprendido.
- `W h + b − R h`: cuánto hay que mover en cada coordenada del subespacio.
- `Rᵀ(...)`: lleva ese desplazamiento de vuelta a `ℝ^d`.

Lectura: es un **interchange intervention en un subespacio de dimensión r**, donde el valor del source no viene de otro input sino de una función aprendida. Se puede verificar: `R h_intervened = R h + R Rᵀ (Wh + b − Rh) = Wh + b`, y la parte de `h` ortogonal al subespacio queda intacta. Parámetros por intervención: `R` (`r·d`), `W` (`r·d`) y `b` (`r`), o sea `2rd + r`.

---

## Recorrido por la notebook

### 0. Setup (celda 7)

Igual que en la Lecture 4: detecta Colab, instala `requirements/06_interventions.txt` (que solo contiene `nnsight`) o configura plotly para PNG. Setup puro.

---

### 1. Ablations: identificar información quitándola

#### Cargar OLMo (celda 17)

```python
from nnsight import LanguageModel
model = LanguageModel("allenai/OLMo-2-0425-1B")
```

Modelo de 1B de AllenAI, totalmente abierto (pesos, código, datos). Misma API de nnsight que en la Lecture 4 (`model.trace`, `model.model.layers[i].output`, `.save()`).

#### Primer vistazo al sesgo (celda 20)

```python
prompt_f = "The nurse said that"
prompt_m = "The doctor said that"
he_token_id = model.tokenizer(" he").input_ids[0]
she_token_id = model.tokenizer(" she").input_ids[0]

with model.trace(prompt_f):
    logits_f = model.output.logits.save() # (1, num tokens, vocab size)
probs_f = logits_f.softmax(dim=-1)
...
print(f"P(he | F): {probs_f[0, -1, he_token_id]:.3f}")
```

- `" he"` y `" she"` **con espacio adelante**: en BPE la palabra después de un espacio es un token distinto de la palabra sin espacio. Como siguen a "that", el token relevante es el que lleva espacio.
- `.input_ids[0]`: el primer id. Esto supone que el tokenizer no agrega un BOS al inicio (si lo agregara, `[0]` sería el BOS para ambos). Que las dos probabilidades impresas sean distintas confirma que son ids distintos.
- `model.output.logits`: la salida del modelo HF (un objeto con `.logits`), shape `(1, seq, vocab)`.
- `softmax` fuera del trace, sobre el tensor guardado.
- `probs[0, -1, id]`: batch 0, última posición (lo que predice después de "that"), el token pedido.

Salida guardada: nurse → P(he)=0.057, P(she)=0.118; doctor → P(he)=0.112, P(she)=0.046. El ratio se invierte con la profesión.

#### Ejercicio 1: métrica de sesgo (celda 23, `# TERRON`)

```python
def measure_bias(probs, stereotype : str):
  p_he = probs[he_token_id]
  p_she = probs[she_token_id]
  if stereotype == "m":
    ratio = p_he / p_she
  else:
    ratio = p_she / p_he
  return torch.log(ratio).item()
```

La métrica elegida es el **log-ratio orientado al estereotipo**:

$$
\text{bias} = \log \frac{p(\text{pronombre del estereotipo})}{p(\text{pronombre contrario})}
$$

- Para profesiones "m" (doctor, pilot...): `log p(he)/p(she)`. Para "f" (nurse, nanny...): `log p(she)/p(he)`.
- 0 = el modelo es indiferente entre "he" y "she"; > 0 = sigue el estereotipo; < 0 = va en contra.
- Por qué log: hace la métrica simétrica. "2 veces más probable" da `+log 2` y "2 veces menos probable" da `−log 2`; con el ratio crudo serían 2 y 0.5, que no se cancelan al promediar.
- Ignora el resto del vocabulario (no importa que p(he) + p(she) sea chico); solo mira la comparación relativa.
- `.item()`: tensor escalar → float de Python.

Como referencia, con los números de la celda 20 (redondeados): nurse ≈ log(0.118/0.057) ≈ 0.73 y doctor ≈ log(0.112/0.046) ≈ 0.89. Esto es una cuenta mía sobre la salida guardada de la celda 20; la celda 23 no tiene salida, y en batch los valores pueden variar un poco.

El resto de la celda:

```python
with model.trace(prompts_m):
  logits_m = model.output.logits[:, -1, :].save()
with model.trace(prompts_f):
  logits_f = model.output.logits[:, -1, :].save()
```

- Batch de 6 prompts cada uno. `[:, -1, :]` → `(6, vocab)`. Igual que en la Lecture 4, depende de que el padding sea a la izquierda ("flight attendant" tiene más tokens que el resto).
- Itera fila por fila (`for logits in logits_m`), hace softmax `(vocab,)` y llama a `measure_bias`. La lista `biases` queda en orden `professions_m + professions_f`.
- Imprime el sesgo medio y, agregado en la solución, el sesgo por profesión con signo (`{b:+.2f}`), para ver cuáles pesan más.

Para la celda 25: describir la métrica (lo de arriba), el valor de un modelo sin sesgo (0) y mirar si la media y la mayoría de las profesiones son positivas.

#### Bias en activaciones con PCA (celda 28)

```python
LAYER = 5
n = len(professions_m)
with torch.no_grad():
    with model.trace(prompts_m):
        activations_m = model.model.layers[LAYER].output[:, -3, :].save() # (n, hidden dim)
    with model.trace(prompts_f):
        activations_f = model.model.layers[LAYER].output[:, -3, :].save()
pca = PCA(n_components=2)
all_activations = torch.cat([activations_m, activations_f]).cpu().float().numpy() # (2n, hidden_dim)
low_dim_activations = pca.fit_transform(all_activations) # (2n, 2)
```

- Capa 5, posición `-3`: en "The [profesión] said that", las últimas dos posiciones son " said" y " that", así que `-3` es el **último token de la profesión** (para "flight attendant", " attendant"). Se mira la representación de la profesión, no la del punto de predicción.
- `(6, hidden)` por grupo → `(12, hidden)` → PCA → `(12, 2)`. Filas 0–5 "m", 6–11 "f".
- El scatter (cuadrados verdes "m", círculos bordó "f") muestra, según el texto, una separación lineal bastante clara.

El texto (celda 29) hace la advertencia importante: que sea separable no prueba que el modelo use esa información; pero si *garantizamos* que no es separable linealmente, al menos un lector lineal no podría usarla.

#### INLP (celda 34)

```python
def inlp(activations_m, activations_f, n_iters=10):
    results = []
    for i in trange(n_iters):
        X = np.concatenate([activations_m, activations_f], axis=0)
        y = np.array(["m"] * len(activations_m) + ["f"] * len(activations_f))

        probe = LogisticRegression(fit_intercept=False, random_state=42, max_iter=1000)
        probe.fit(X, y)
```

- Cada iteración arma `X` `(12, d)` e `y` `(12,)` con las activaciones **actuales** (ya proyectadas en iteraciones anteriores).
- Probe binario sin intercepto; `max_iter=1000` para que lbfgs converja. Como `classes_ = ['f', 'm']`, `coef_[0]` apunta hacia "m" (no importa para la proyección: borrar `w` o `−w` es lo mismo).

```python
        probe_accuracy = probe.score(X, y)
        results.append({"activation_m": activations_m.copy(),
                        "activation_f": activations_f.copy(),
                        "iteration": i, "probe_accuracy": probe_accuracy})
        if probe_accuracy < 0.51:
            print(f"Stopping at round {i} (probe accuracy = {probe_accuracy:.2f})")
            break
```

- Guarda una **foto** (`.copy()`, para que no se modifique después) de las activaciones al inicio de la ronda `i`, junto con la accuracy del probe entrenado sobre ellas.
- Criterio de parada: accuracy < 51%, o sea, el probe ya no supera el azar (con 12 puntos balanceados, 50%).

```python
        probe_weights = probe.coef_[0] # (hidden dim,)
        probe_magnitude = np.dot(probe_weights, probe_weights)
        activations_m = activations_m - np.outer(activations_m @ probe_weights, probe_weights) / probe_magnitude
        activations_f = activations_f - np.outer(activations_f @ probe_weights, probe_weights) / probe_magnitude
    return results
```

- `probe_magnitude = w·w = ‖w‖²`.
- `activations_m @ w`: `(6, d) @ (d,)` → `(6,)`, el producto `x_i · w` de cada fila.
- `np.outer(a, w)`: `(6,) ⊗ (d,)` → `(6, d)`, donde la fila `i` es `(x_i·w) · w`.
- Dividir por `‖w‖²` y restar: es la fórmula `x' = x − (x·w/w·w) w` aplicada a cada fila a la vez.
- Se aplica igual a ambos grupos.

Detalle fino: esta versión proyecta secuencialmente (primero borra `w₁`, después `w₂` sobre los datos ya proyectados). Si `w₂` tuviera componente en `w₁`, la segunda proyección podría reintroducir algo de `w₁`. En la práctica no pasa porque, con regularización L2, el `w` óptimo de la regresión logística vive en el espacio generado por los datos, que ya es ortogonal a `w₁`. El paper original proyecta con el nullspace de todas las direcciones juntas para garantizarlo.

**Celda 36**: corre `inlp` y grafica cada foto en un subplot.

```python
results = inlp(activations_m.cpu().float().numpy(), activations_f.cpu().float().numpy())
...
for col, r in enumerate(results, start=1):
    pca = PCA(n_components=2)
    proj = pca.fit_transform(np.concatenate([acts_m_snap, acts_f_snap], axis=0))
```

- Las activaciones se pasan como numpy float32.
- Para cada foto se **reajusta** una PCA nueva (el comentario explica por qué: después de borrar direcciones, la varianza que queda es otra, y una PCA vieja mostraría ejes ya vacíos).
- `make_subplots(rows=1, cols=len(results))`, títulos con la ronda y la accuracy; `showlegend=(col == 1)` para no repetir la leyenda.

Salida guardada: `Stopping at round 2 (probe accuracy = 0.42)`. Entonces `results` tiene 3 fotos (rondas 0, 1 y 2); la de la ronda 2 son las activaciones después de **dos** proyecciones, y ahí el probe ya no clasifica.

#### Ejercicio 2: ¿INLP redujo el sesgo? (celda 39)

La idea: correr el modelo sobre los mismos prompts pero **reemplazando** la activación de la capa 5 en la posición `-3` por la versión INLP, y volver a medir el sesgo con `measure_bias`.

```python
LAYER = 5
TOKEN_POSITION = -3
inlp_activations_m = torch.tensor(results[-1]['activation_m']).to(model.device) # (6, hidden dim)
inlp_activations_f = torch.tensor(results[-1]['activation_f']).to(model.device)

with model.trace(prompts_m):
  ############################# YOUR CODE HERE #############################
  # replace the model's activations at layer 5 and token -3 with the INLP activations!
  # mismo lugar de donde sacamos las activaciones para INLP: capa 5, token de la profesión
  model.model.layers[LAYER].output[:, TOKEN_POSITION, :] = inlp_activations_m
  ##########################################################################
  logits_m = model.output.logits[:, -1, :].save()

with model.trace(prompts_f):
  ############################# YOUR CODE HERE #############################
  # repeat for `inlp_activations_f`
  model.model.layers[LAYER].output[:, TOKEN_POSITION, :] = inlp_activations_f
  ##########################################################################
  logits_f = model.output.logits[:, -1, :].save()
```

- `results[-1]`: la última foto (la de la ronda en que se cortó).
- `torch.tensor(numpy)`: queda float32; `.to(model.device)` lo manda al device del modelo.
- `model.model.layers[LAYER].output[:, TOKEN_POSITION, :] = ...`: dentro del trace, escribe sobre el residual stream de la capa 5, en todos los prompts del batch (`:`), solo en la posición `-3` (el último token de la profesión, antes de " said that"), y en todas las dimensiones ocultas. Las capas siguientes ya leen la versión INLP. `inlp_activations_m` es `(6, hidden)`: una fila por prompt, en el mismo orden en que se recolectaron.
- Después se mide el sesgo igual que en el ejercicio 1 y se imprime por profesión para comparar.

Ojo con dos cosas. El orden de los prompts tiene que coincidir con el que se usó para recolectar, y así es. Y el dtype: la asignación indexada de torch convierte el float32 al dtype del modelo, así que si OLMo carga en bf16 no debería romperse. Igual no verifiqué con qué dtype carga OLMo en esta configuración.

Para la celda 41, qué mirar cuando funcione: comparar el sesgo medio y por profesión antes y después. Preguntas para razonar: si baja pero no llega a 0, ¿por dónde se "escapa" la información? Algunas pistas:
- Solo se reemplazó **una posición** en **una capa**. Las posiciones " said" y " that" ya leyeron la profesión vía atención en las capas 0–5, y en profesiones de dos tokens también está " flight".
- INLP solo garantiza que *el probe usado* no separa. Con 12 puntos en miles de dimensiones, en principio siempre existe algún hiperplano que los separa; que la regresión logística regularizada no lo encuentre no es lo mismo que "ningún clasificador lineal puede".
- Información no lineal no se borra.

#### Takeaway (celdas 43–45)

INLP es un ejemplo de intervención que supone que la información relevante es lineal. `del model` libera memoria antes de pasar a la red de juguete.

---

### 2. Interchange interventions

#### La red MaxNN (celda 56)

```python
class MaxNN(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.l1 = torch.nn.Linear(2, 4, bias=False)
    self.l2 = torch.nn.Linear(4, 2, bias=False)
    self.l3 = torch.nn.Linear(2, 1, bias=False)
    self.relu1 = torch.nn.ReLU()
    self.relu2 = torch.nn.ReLU()
```

- Tres capas lineales sin bias y dos ReLU. `Linear(in, out)` guarda `weight` con shape `(out, in)` y calcula `W x`.

```python
    self.l1.weight.data = torch.tensor([[1., -1.], [-1., 1.], [1., 0.], [0., 1.]])
    self.l2.weight.data = torch.tensor([[-100., 0., 0., 1.], [0., -100., 1., 0.]])
    self.l3.weight.data = torch.tensor([[1., 1.]])
```

- Los pesos se fijan a mano (`.data` para escribir sin pasar por autograd).
- Fila por fila de `l1`: `x − y`, `y − x`, `x`, `y`.
- `l2`: fila 0 = `−100·r[0] + r[3]`; fila 1 = `−100·r[1] + r[2]`.
- `l3`: suma de las dos.

```python
  def forward(self, x):
    x = self.l1(x); x = self.relu1(x)
    x = self.l2(x); x = self.relu2(x)
    x = self.l3(x)
    return x
```

Importante para las intervenciones: `model.l1.output` es la salida de `l1` **antes** de `relu1`. Intervenir ahí significa que el ReLU se aplica después sobre los valores modificados.

**Ejemplo completo a mano**, `(x, y) = (1, 3)`:

| Paso | Valor |
|---|---|
| `l1` | `[1−3, 3−1, 1, 3] = [−2, 2, 1, 3]` |
| `relu1` | `[0, 2, 1, 3]` |
| `l2` | `[−100·0 + 3, −100·2 + 1] = [3, −199]` |
| `relu2` | `[3, 0]` |
| `l3` | `3` |

Y `(7, 2)`: `l1 = [5, −5, 7, 2]` → `relu1 = [5, 0, 7, 2]` → `l2 = [−500 + 2, 0 + 7] = [−498, 7]` → `relu2 = [0, 7]` → `7`.

#### Ejercicio 3: casos borde (celda 61, `# TERRON`)

```python
maxnn = MaxNN()
test_inputs = [[1., 3.], [7., 2.], [4., 4.], [0., 5.], [-1., -3.], [-2., 3.], [0.5, 0.2]]
for x in test_inputs:
  inputs = torch.tensor(x)
  out = maxnn(inputs).item()
  expected = max(x)
  flag = "" if abs(out - expected) < 1e-6 else "   <-- no da el max"
```

- Acá `maxnn` es el módulo de PyTorch "pelado" (todavía sin nnsight). Input sin dimensión de batch: shape `(2,)`; salida `(1,)`, `.item()` la pasa a float.
- Compara contra `max(x)` de Python con tolerancia `1e-6` y marca los casos que fallan.

La celda no está ejecutada. Hice las cuentas a mano (la red es determinista) con un script en numpy que replica los pesos:

| Input | `l1` | `relu1` | `l2` | Salida | `max` |
|---|---|---|---|---|---|
| (1, 3) | [−2, 2, 1, 3] | [0, 2, 1, 3] | [3, −199] | 3 | 3 |
| (7, 2) | [5, −5, 7, 2] | [5, 0, 7, 2] | [−498, 7] | 7 | 7 |
| (4, 4) | [0, 0, 4, 4] | [0, 0, 4, 4] | [4, 4] | **8** | 4 |
| (0, 5) | [−5, 5, 0, 5] | [0, 5, 0, 5] | [5, −500] | 5 | 5 |
| (−1, −3) | [2, −2, −1, −3] | [2, 0, 0, 0] | [−200, 0] | **0** | −1 |
| (−2, 3) | [−5, 5, −2, 3] | [0, 5, 0, 3] | [3, −500] | 3 | 3 |
| (0.5, 0.2) | [0.3, −0.3, 0.5, 0.2] | [0.3, 0, 0.5, 0.2] | [−29.8, 0.5] | 0.5 | 0.5 |

Qué mirar: cuando `x = y` ninguna bandera se activa y pasan las dos copias, así que la red suma; con negativos el ReLU de las copias las anula. Hay otro caso que la celda no prueba: si `x` e `y` son muy parecidos, `100·|x − y|` puede no alcanzar para anular la copia. Por ejemplo `(1, 1.005)`: `l2 = [1.005, −0.5 + 1] = [1.005, 0.5]`, salida ≈ 1.505. El "−100" funciona como un `if` solo si `100·|x−y|` es mayor que el número que hay que anular. La celda 64 aclara que de acá en adelante se asumen positivos con un máximo único.

#### Zero-out interventions (celdas 70–73)

**Celda 70**: `model = nnsight.NNsight(MaxNN())`. `NNsight` es el wrapper genérico de nnsight para cualquier `torch.nn.Module` (el `LanguageModel` es la versión con tokenizer para HF). Da `model.l1.output`, `model.output`, `model.trace(...)` igual que antes. Llamar `model(inputs)` fuera de un trace corre el modelo normal.

**Celda 71**:

```python
inputs = torch.tensor([1., 3.])
original_output = model(inputs)
with model.trace(inputs):
  layer3 = model.l3.output
  layer3[0] = 0.
  intervention_output = model.output.save()
```

- `model.l3.output`: shape `(1,)` (input sin batch). `layer3[0] = 0.` escribe in-place la única neurona de salida.
- `model.output.save()`: la salida final del modelo, después de la intervención.
- Resultado guardado: antes 3.0, después 0.0. Trivial: es la última capa.

**Celda 73**: lo mismo pero `layer1 = model.l1.output; layer1[-1] = 0.`. Como el input no tiene batch, `l1.output` es `(4,)` y `[-1]` es la **neurona 3**, la copia de `y`. Para (1, 3): `l1 = [−2, 2, 1, 0]` → `relu1 = [0, 2, 1, 0]` → `l2 = [0, −199]` → salida 0. Salida guardada: 3.0 → 0.0. Parece que "borramos la respuesta", pero en realidad borramos la copia de `y`, que era justo la que iba a ganar.

#### Ejercicio 4 (celda 77, `# TERRON`)

Pide 2 inputs más donde el zero-out de la neurona 3 dé 0, y 2 donde tenga otro efecto.

```python
test_inputs = [[1., 3.], [2., 5.], [4., 9.], [5., 2.], [9., 4.]]
for x in test_inputs:
  ...
  with model.trace(inputs):
    layer1 = model.l1.output
    layer1[-1] = 0.
    intervention_output = model.output.save()
```

- La solución elige dos con `y > x` (esperando el mismo efecto) y dos con `x > y` (esperando otro efecto). Es el mismo bloque de la celda 73 dentro de un loop.

Cuentas a mano:

| Input | `l1` intervenido | `relu1` | `l2` | Antes | Después |
|---|---|---|---|---|---|
| (1, 3) | [−2, 2, 1, 0] | [0, 2, 1, 0] | [0, −199] | 3 | 0 |
| (2, 5) | [−3, 3, 2, 0] | [0, 3, 2, 0] | [0, −298] | 5 | 0 |
| (4, 9) | [−5, 5, 4, 0] | [0, 5, 4, 0] | [0, −496] | 9 | 0 |
| (5, 2) | [3, −3, 5, 0] | [3, 0, 5, 0] | [−300, 5] | 5 | 5 |
| (9, 4) | [5, −5, 9, 0] | [5, 0, 9, 0] | [−500, 9] | 9 | 9 |

Para la celda 79: fijate qué tienen en común los inputs de cada grupo y qué rol juega la neurona 3 en cada caso (¿se usa la copia de `y` cuando gana `x`?).

#### Zero-out de varias neuronas (celdas 84 y 87)

**Celda 84**: `layer1[:2] = 0.` (las dos banderas). Para (1, 3): `relu1 = [0, 0, 1, 3]` → `l2 = [3, 1]` → salida **4** = `x + y`. Sin banderas, ninguna copia se anula y la red suma. Es el mismo mecanismo que el caso `x = y` del ejercicio 3. En general da `x + y` para cualquier par positivo.

**Celda 87**: `layer1[[1, 3]] = 0.` (indexado con lista: neuronas 1 y 3, la bandera "ganó y" y la copia de `y`). Para (1, 3): `relu1 = [0, 0, 1, 0]` → `l2 = [0, 1]` → salida **1**. La notebook lo describe como "devuelve el mínimo". Ojo: eso es cierto para este input, pero en general esta intervención devuelve siempre `x`:
- Si `y > x`: `relu1 = [0, 0, x, 0]`, `l2 = [0, x]` → `x` (que es el mínimo).
- Si `x > y`: `relu1 = [x−y, 0, x, 0]`, `l2 = [−100(x−y), x]` → `x` (que es el máximo).

Lo comprobé con el script: (5, 2) → 5 y (9, 4) → 9. O sea que "min" es una coincidencia de que en (1, 3) el `x` es el menor.

La lista de la celda 88 resume: el zero-out puede dar 0, no hacer nada, sumar o devolver el menor. Moraleja: con no linealidades, 0 es un valor con significado, no "ausencia de información".

#### Interchange interventions (celdas 97 y 101)

**Celda 97**, swap completo de `l1`:

```python
source_input = torch.tensor([7., 2.])
base_input = torch.tensor([1., 3.])

with model.trace(source_input):
  source_activation = model.l1.output.save()

with model.trace(base_input):
  base_activation = model.l1.output
  base_activation[:] = source_activation[:]
  intervention_output = model.output.save()
```

- Paso 1: un trace sobre el source y se guarda `l1.output` = `[5, −5, 7, 2]`.
- Paso 2: un trace sobre el base y se sobrescribe **in-place** todo el vector (`[:]`) con el del source. `source_activation` ya es un tensor real (se guardó en el trace anterior), así que se puede usar adentro del segundo trace.
- Resultado guardado: base 3, source 7, intervención 7. Si reemplazás toda la capa, todo lo que viene después es idéntico al forward del source.

**Celda 101**, swap de las dos últimas neuronas (`[2:]`, las copias):

- `l1` intervenido = `[−2, 2, 7, 2]` (banderas del base, copias del source) → `relu1 = [0, 2, 7, 2]` → `l2 = [−0 + 2, −200 + 7] = [2, −193]` → **2**.
- Salida guardada: 2.0. La bandera del base dice "ganó y", y la copia de `y` que hay ahora es la del source (2). Resultado: "el `y` del source", ni el max del base ni el del source.

#### Ejercicio 5 (celdas 107 y 108, `# TERRON`)

**Celda 107**: el mismo patrón con `[:2]` (las banderas) en vez de `[2:]`:

```python
with model.trace(source_input):
  source_activation = model.l1.output.save()
with model.trace(base_input):
  base_activation = model.l1.output
  base_activation[:2] = source_activation[:2]
  intervention_output = model.output.save()
```

Cuenta a mano para source (7, 2) y base (1, 3): `l1` = `[5, −5, 1, 3]` (banderas del source, copias del base) → `relu1 = [5, 0, 1, 3]` → `l2 = [−500 + 3, 0 + 1] = [−497, 1]` → **1**.

**Celda 108**: más pares para ver el patrón, con una forma más compacta (asigna directo `model.l1.output[:2] = source_activation[:2]` sin variable intermedia).

```python
pairs = [([7., 2.], [1., 3.]), ([2., 7.], [1., 3.]), ([2., 7.], [5., 4.]), ([9., 1.], [6., 8.])]
```

Cuentas a mano (formato `(source) → (base)`):

| Source | Base | ¿Quién gana en el source? | `l1` intervenido | Salida |
|---|---|---|---|---|
| (7, 2) | (1, 3) | x | [5, −5, 1, 3] | 1 |
| (2, 7) | (1, 3) | y | [−5, 5, 1, 3] | 3 |
| (2, 7) | (5, 4) | y | [−5, 5, 5, 4] | 4 |
| (9, 1) | (6, 8) | x | [8, −8, 6, 8] | 6 |

Para la celda 110: mirá la tercera columna contra la salida. ¿De cuál de los dos números del **base** sale la respuesta, y qué lo decide? Compará también con la tabla de la celda 101 (donde se cambiaban las copias). Los takeaways de la notebook (celdas 117–118) describen el resultado general en términos de **función** vs. **contenido**.

Caveat de magnitudes: esto funciona porque la diferencia del source multiplicada por 100 es mucho mayor que los valores del base. Con un source de diferencia muy chica y un base de números grandes (por ejemplo source (1, 1.001) y base (50, 80)), la bandera no alcanzaría a anular la copia.

#### Ejercicio 6: intervención max → min (celda 113, `# TERRON`)

Pide una intervención que haga que el modelo calcule `min`. La pista: un valor constante no sirve, tiene que depender del input.

```python
def compute_intervention(l1_output):
  intervention = l1_output.clone()
  intervention[0] = l1_output[1]
  intervention[1] = l1_output[0]
  return intervention
```

- `clone()`: copia el tensor, para no modificar `l1_output` mientras se lo lee (si se hiciera in-place, `intervention[1] = l1_output[0]` leería el valor ya pisado).
- **Intercambia las dos banderas**: la neurona 0 recibe `y − x` y la 1 recibe `x − y`. La red "cree" que ganó el otro número y deja pasar la copia del menor.
- Por qué no sirve una constante: el valor correcto de las banderas depende de cuál de los dos es mayor en *cada* input. Cualquier constante fija la decisión en un sentido para todos los inputs.
- Como `l1[1] = −l1[0]`, intercambiarlas es lo mismo que negarlas (`-l1_output[:2]`).

```python
for x in [[1., 3.], [7., 2.], [5., 9.], [10., 4.]]:
  inputs = torch.tensor(x)
  original_output = model(inputs)
  with model.trace(inputs):
    base_activation = model.l1.output
    intervention = compute_intervention(base_activation)
    model.l1.output = intervention
    intervention_output = model.output.save()
```

- A diferencia de las celdas anteriores, acá no se escribe in-place: se **reasigna** la salida entera del módulo (`model.l1.output = intervention`). nnsight reemplaza lo que `l1` le pasa a `relu1` por el nuevo tensor.
- `compute_intervention` se llama con la activación de dentro del trace; operaciones como `clone()` e indexado funcionan igual que con un tensor normal.
- Se prueba con cuatro inputs, mezclando `x > y` y `y > x`.

Cuentas a mano:

| Input | `l1` original | `l1` intervenido | `relu1` | `l2` | Salida | `min` |
|---|---|---|---|---|---|---|
| (1, 3) | [−2, 2, 1, 3] | [2, −2, 1, 3] | [2, 0, 1, 3] | [−197, 1] | 1 | 1 |
| (7, 2) | [5, −5, 7, 2] | [−5, 5, 7, 2] | [0, 5, 7, 2] | [2, −493] | 2 | 2 |
| (5, 9) | [−4, 4, 5, 9] | [4, −4, 5, 9] | [4, 0, 5, 9] | [−391, 5] | 5 | 5 |
| (10, 4) | [6, −6, 10, 4] | [−6, 6, 10, 4] | [0, 6, 10, 4] | [4, −590] | 4 | 4 |

Para la celda 115 ("¿se podría descubrir con interchange interventions?"): pensá qué valores puede poner un interchange en las banderas. Solo puede copiar banderas de otro input source. ¿Existe un único source que funcione para todos los base, o el source adecuado depende del base? ¿Qué pasa si para cada base elegís como source el mismo par con `x` e `y` invertidos? La diferencia entre "un valor de un input real" y "una función arbitraria de la activación" es la clave.

#### Takeaways de la sección (celdas 117–121)

Intervenir las banderas cambia **qué función** calcula la red; intervenir las copias cambia **sobre qué valores** la calcula. Esto se descubrió sin mirar los valores de las activaciones, solo cambiándolos entre inputs. La notebook menciona como ejemplo a gran escala el **causal tracing** de Meng et al. (2022): agregan ruido a los tokens del sujeto, restauran un hidden state por vez a su valor limpio (un interchange con el prompt limpio como source) y miden cuánto se recupera la respuesta; encuentran que importan los MLP de capas medias en el último token del sujeto, y de ahí sale ROME.

---

### 3. Steering con ReFT

#### Instalar y cargar (celdas 134–136)

**Celda 134**: `pip install pyreft`. Advierte que baja la versión de `transformers`; al terminar se puede restaurar con `pip install -U transformers`. Esto choca con el resto del curso (el repo fija transformers 5.5.0), así que conviene correrlo en un entorno aparte o al final.

**Celda 136**:

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
prompt_no_input_template = """\n<|user|>:%s</s>\n<|assistant|>:"""
model_name_or_path = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
model = transformers.AutoModelForCausalLM.from_pretrained(model_name_or_path, dtype=torch.bfloat16, device_map=device)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path, model_max_length=2048,
                                                       padding_side="right", use_fast=False)
tokenizer.pad_token = tokenizer.unk_token
```

- Acá ya no se usa nnsight: el modelo es un `AutoModelForCausalLM` de HF puro, porque `pyreft` lo envuelve a su manera (sobre `pyvene`).
- `prompt_no_input_template`: template de chat a mano; `%s` se reemplaza por la instrucción. Es el del tutorial de pyreft.
- `dtype=torch.bfloat16`: pesos en bf16 para ahorrar memoria.
- Tokenizer con `padding_side="right"` (para entrenamiento causal es lo normal) y `use_fast=False` (tokenizer lento de SentencePiece, como en el tutorial).
- TinyLlama no tiene token de padding; se usa `unk` como pad.

#### Configurar la intervención (celda 137)

```python
reft_config = pyreft.ReftConfig(representations={
    "layer": 15,
    "component": "block_output",
    "low_rank_dimension": 4,
    "intervention": pyreft.LoreftIntervention(embed_dim=model.config.hidden_size, low_rank_dimension=4)})
reft_model = pyreft.get_reft_model(model, reft_config)
reft_model.set_device(device)
reft_model.print_trainable_parameters()
```

- `layer: 15`: en qué capa intervenir.
- `component: "block_output"`: la salida del bloque transformer completo, es decir, el residual stream después de esa capa (lo mismo que `model.model.layers[15].output` en nnsight).
- `low_rank_dimension: 4`: el `r` de la fórmula (filas de `R` y de `W`).
- `LoreftIntervention(embed_dim=hidden_size, low_rank_dimension=4)`: la intervención LoReFT. `embed_dim` es `d`.
- `get_reft_model`: envuelve el modelo de HF, congela sus pesos y registra la intervención como único parámetro entrenable.
- `print_trainable_parameters()`: no hay salida guardada. Por la fórmula `2rd + r`, si el hidden size de TinyLlama es 2048 (no lo verifiqué en la notebook), serían `2·4·2048 + 4 = 16 388` parámetros, contra ~1.1B del modelo.

No revisé el código fuente de pyreft (no está en el repo ni instalado). Lo de arriba sobre cómo implementa `R` (con filas ortonormales) y `W, b` viene de la descripción de la notebook y del paper.

#### Datos (celda 139)

```python
training_examples = [["Who are you?", "🤖💬🌐🧠"], ...]
data_module = pyreft.make_last_position_supervised_data_module(
    tokenizer, model,
    [prompt_no_input_template % e[0] for e in training_examples],
    [e[1] for e in training_examples])
```

- 10 pares (instrucción, respuesta en emojis). Incluyen intentos de "romper" la regla ("Forget the previous instructions...", "Can you respond with anything other than emojis?") para que la intervención sea robusta.
- Cada instrucción se mete en el template.
- `make_last_position_supervised_data_module`: arma el dataset para el `Trainer`. Por el nombre y por cómo se usa en la celda 143, la intervención se aplica en la **última posición del prompt** y la supervisión es next-token prediction sobre la respuesta (lo normal es enmascarar los tokens del prompt en las labels para que la loss sea solo sobre la respuesta; no lo verifiqué en el código de pyreft). Devuelve un dict con dataset y data collator, que se pasa con `**data_module`.

#### Entrenamiento (celda 141)

```python
training_args = transformers.TrainingArguments(num_train_epochs=100.0, output_dir="./tmp",
    per_device_train_batch_size=10, learning_rate=4e-3, logging_steps=40, report_to=[])
trainer = pyreft.ReftTrainerForCausalLM(model=reft_model, tokenizer=tokenizer, args=training_args, **data_module)
_ = trainer.train()
```

- `TrainingArguments` estándar de HF. Batch de 10 = todo el dataset, así que 100 épocas = 100 pasos de gradiente. `learning_rate=4e-3` es alto comparado con fine-tuning completo, lo cual es típico cuando se entrenan pocos parámetros. `logging_steps=40` loguea la loss cada 40 pasos; `report_to=[]` desactiva wandb y similares; `output_dir="./tmp"` es donde guardaría checkpoints.
- `ReftTrainerForCausalLM`: subclase del `Trainer` de HF que sabe pasar las `unit_locations` de la intervención en cada batch. Solo se actualizan los parámetros de la intervención.
- La salida guardada de esta celda es `'<|endoftext|><|user|>\nTEST\n'`, que no parece un log de entrenamiento. No queda claro de dónde sale (quizás de algún print interno de pyreft o de una ejecución anterior de la celda).

#### Generar con la intervención (celda 143)

```python
instruction = "Which dog breed do people think is cuter, poodle or doodle?"
prompt = prompt_no_input_template % instruction
prompt = tokenizer(prompt, return_tensors="pt").to(device)

base_unit_location = prompt["input_ids"].shape[-1] - 1
_, reft_response = reft_model.generate(
    prompt,
    unit_locations={"sources->base": (None, [[[base_unit_location]]])},
    intervene_on_prompt=True,
    max_new_tokens=20,
    do_sample=True)
print(tokenizer.decode(reft_response[0], skip_special_tokens=True))
```

- Instrucción nueva, que no está en el entrenamiento, para ver si generaliza.
- `tokenizer(..., return_tensors="pt")`: dict con `input_ids` y `attention_mask`, shape `(1, seq)`.
- `base_unit_location = seq − 1`: índice de la **última posición del prompt**, la misma en la que se entrenó la intervención.
- `unit_locations={"sources->base": (fuentes, base)}`: la forma de `pyvene` de decir dónde se interviene. Es una tupla `(ubicaciones en los sources, ubicaciones en el base)`.
  - `None` en los sources: LoReFT no usa un input source; el "source" es `Wh + b`, calculado del propio `h`.
  - `[[[base_unit_location]]]`: tres niveles de anidamiento = `[por intervención][por ejemplo del batch][posiciones]`. Una intervención, un ejemplo, una posición.
- `intervene_on_prompt=True`: aplica la intervención durante el forward del prompt. Como solo se toca la última posición del prompt, los tokens generados después ven el cambio a través de la atención (KV cache).
- `do_sample=True`: muestreo, así que la salida varía entre corridas.
- `generate` devuelve una tupla `(salida sin intervenir, salida intervenida)`; la primera se descarta con `_` (según la convención de pyvene; no lo verifiqué en el código).
- `decode(..., skip_special_tokens=True)`: no hay salida guardada. Lo esperado según el tutorial es una respuesta en emojis.

La comparación con la Lecture 4: difference-in-means es un vector fijo sumado en todos los tokens; ReFT es una función aprendida (depende de `h`), en un subespacio de dimensión 4 y en **una sola posición**, entrenada directamente para producir el comportamiento.

---

## Ojo / errores comunes / dudas

- **Ejercicio 2, una sola posición**: el reemplazo INLP se hace solo en la capa 5 y la posición −3; " said" y " that" ya pudieron leer la profesión por atención antes de esa capa.
- **`l1.output` es pre-ReLU**: todas las intervenciones de MaxNN escriben antes del ReLU. Un `0` ahí sigue siendo `0` después del ReLU, pero un valor negativo copiado del source se anula.
- **Sin dimensión de batch en MaxNN**: `torch.tensor([1., 3.])` es `(2,)`, así que `l1.output[-1]` es una neurona. Con un batch `(1, 2)` tendrías que indexar `[:, -1]`; si no, estarías tocando el ejemplo entero.
- **In-place vs. reasignar en nnsight**: `model.l1.output[:2] = ...` modifica in-place; `model.l1.output = nuevo` reemplaza la salida. En el ejercicio 6 hace falta el `clone()` para no leer valores ya pisados.
- **"min" de la celda 87**: zero-out de las neuronas 1 y 3 devuelve siempre `x`, no el mínimo; coincide con el mínimo solo cuando `y > x`.
- **Casos borde de MaxNN**: `x = y` suma; negativos dan 0; números muy parecidos (diferencia menor a ~1/100 del valor) también "filtran" la copia que debería anularse.
- **INLP con pocos datos**: 12 puntos en miles de dimensiones. "El probe no separa" es una afirmación sobre esta regresión logística regularizada, no sobre todos los clasificadores lineales.
- **INLP borra lineal, no no lineal**, y en el ejercicio 2 solo se reemplaza una posición de una capa.
- **Padding**: los batches de profesiones (`flight attendant` es más largo) dependen del padding a la izquierda para que `-1` y `-3` caigan en el lugar correcto.
- **pyreft y transformers**: pyreft baja la versión de transformers; puede romper nnsight u otras notebooks si se usa el mismo entorno.
- **Variables pisadas**: `model` pasa de OLMo (nnsight) a MaxNN (nnsight) a TinyLlama (HF puro); `pca`, `biases`, `logits_m/f` se reusan.

---

## Resumen por sección

- **Setup**: instalación y renderer de plotly.
- **Sesgo en OLMo**: P(she) > P(he) para nurse y al revés para doctor.
- **Ejercicio 1**: métrica `log p(pronombre estereotipado)/p(contrario)`, 0 = sin sesgo, promediada sobre 12 profesiones.
- **Bias en activaciones**: PCA de la capa 5 en el token de la profesión separa "m" y "f".
- **INLP**: entrenar probe, proyectar con `x − (x·w/w·w) w`, repetir; se corta en la ronda 2 con accuracy 0.42.
- **Ejercicio 2**: reemplazar la activación de capa 5 / posición −3 por la versión INLP y medir de nuevo.
- **MaxNN**: `l1 = [x−y, y−x, x, y]`, banderas de comparación + copias; `l2` usa −100 como un `if`.
- **Ejercicio 3**: casos borde: iguales (suma), negativos (0), números muy cercanos.
- **Zero-out**: según la neurona da 0, nada, suma o `x`; el 0 tiene significado.
- **Ejercicio 4**: zero-out de la copia de `y` da 0 si `y > x` y no cambia nada si `x > y`.
- **Interchange**: swap completo = salida del source; swap de copias `[2:]` = el valor del source elegido por la comparación del base.
- **Ejercicio 5**: swap de banderas `[:2]` = el valor del base elegido por la comparación del source.
- **Ejercicio 6**: intercambiar (o negar) las banderas convierte max en min; necesita depender del input.
- **ReFT**: LoReFT `h + Rᵀ(Wh + b − Rh)` de rango 4 en la capa 15, última posición del prompt, entrenado 100 pasos para responder en emojis.
