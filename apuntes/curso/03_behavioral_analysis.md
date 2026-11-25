# Lecture 3: Behavioral Analysis

Esta notebook es sobre evaluar a un LLM **desde afuera**, sin mirar sus internals, pero de una forma más inteligente que reportar un número de accuracy. Toma un benchmark de opción múltiple (MMLU, temas de matemática), evalúa a `microsoft/Phi-3-small-8k-instruct` de la forma estándar (5-shot, comparando las probabilidades de las letras A/B/C/D) y después le aplica **stress tests**: cambios controlados en el input que o bien *no deberían* cambiar la respuesta (permutar opciones, cambiar letras) o bien *sí deberían* (cambiar un número del enunciado y recalcular la respuesta). La pregunta que responde es: cuando el modelo acierta, ¿está razonando o está recordando pares pregunta-respuesta que vio en el preentrenamiento? El cierre es un **probe de memorización**: se mete la respuesta original como distractor en las variantes y se mide cuántas veces el modelo la elige aunque ya no sea correcta.

---

## Ideas clave / teoría

### 1. Los benchmarks sobreestiman la competencia

Un modelo puede sacar buena accuracy explotando patrones superficiales (**shortcut learning**) en vez de hacer el razonamiento que el test quiere medir. La notebook cita cuatro trabajos:

- **McCoy et al. (2019), "Right for the Wrong Reasons" (HANS)**: modelos de NLI (incluido BERT) usan heurísticas sintácticas como el solapamiento léxico entre premisa e hipótesis ("si comparten palabras, hay entailment"). Con un test construido para romper esas heurísticas, fallan sistemáticamente.
- **Jia & Liang (2017)**: en reading comprehension, agregar una sola oración distractora (que no cambia la respuesta pero comparte palabras con la pregunta) baja el F1 promedio de 16 modelos de 75.4 a 36.4; en el 96.6% de los fallos de los 4 modelos más analizados, la respuesta salió de la oración agregada.
- **Zheng et al. (2024)**: los LLMs tienen **selection bias** en opción múltiple: preferencia por ciertos IDs de opción (por ejemplo, "A"), así que permutar las opciones puede cambiar la predicción.
- **Mirzadeh et al. (2024), GSM-Symbolic**: reescriben problemas de GSM8K como templates y cambian solo los números; la accuracy cae y varía mucho entre instancias, lo que sugiere pattern matching contra templates conocidos.

Conclusión: **evaluando solo en test sets naturales no se puede distinguir razonamiento robusto de atajos**. Hacen falta perturbaciones pequeñas y controladas.

### 2. Minimal contrasting pairs

Un **par contrastivo mínimo** es (pregunta original, variante mínimamente modificada). Se cambia una sola cosa y se mantiene todo lo demás igual (redacción, estructura, estilo de los distractores). Si el desempeño cambia, se le puede atribuir al cambio. Si se cambian dos cosas a la vez y la accuracy baja, ya no se sabe cuál fue la causa.

### 3. Dos tipos de perturbación, dos tipos de falla

La notebook toma de Jia & Liang dos términos:

- **Oversensitivity** (sobresensibilidad): el modelo reacciona a un cambio que *no debería importar*. Se prueba cambiando variables irrelevantes ("de fondo"): permutar opciones, cambiar las letras. La respuesta correcta (su contenido) no cambia.
- **Overstability** (sobreestabilidad): el modelo *no* reacciona a un cambio que *sí debería importar*. Se prueba cambiando variables **causalmente relevantes**: un número del enunciado, recalculando la respuesta. Si el modelo sigue respondiendo lo mismo de antes, no está usando ese número.

Notar el vocabulario causal: la etiqueta (respuesta correcta) es un efecto de ciertas variables del input. Cambiar una variable que no es causa no debería mover la etiqueta; cambiar una que sí es causa sí la mueve. Esto anticipa el lenguaje de intervenciones del resto del curso.

### 4. Contaminación y memorización

MMLU se considera **contaminado**: es tan usado que es probable que muchas de sus preguntas (o paráfrasis) estén en los datos de preentrenamiento. Un acierto puede venir de recordar la pregunta. Dos variantes de memorización que conviene distinguir:

- Memorizar **la letra** de la respuesta ("esta pregunta → C"). Lo expone la permutación de opciones.
- Memorizar **el valor** de la respuesta ("esta pregunta → 5+2√13"). Lo expone cambiar los números y meter el valor viejo como distractor.

### 5. Evaluación por log-probabilidades de las letras

En vez de generar texto y parsearlo, se mira el **siguiente token** después de `"Answer:"` y se comparan los logits de los cuatro tokens `" A"`, `" B"`, `" C"`, `" D"`. La predicción es el de mayor logit.

Formalmente, si $z_\ell$ es el logit del token de la letra $\ell$ en la última posición, la log-probabilidad es

$$
\log p(\ell \mid \text{prompt}) = z_\ell - \log \sum_{w \in V} e^{z_w}
$$

El segundo término (log-sum-exp sobre todo el vocabulario $V$) es el mismo para las cuatro letras, así que comparar log-probs es exactamente lo mismo que comparar logits crudos. Por eso el código hace `argmax` directo sobre los logits aunque la notebook hable de log-probabilidades. Una consecuencia: el modelo podría asignar probabilidad alta a algo que no es ninguna letra (por ejemplo, un salto de línea) y este método igual elige una de las cuatro; es una evaluación "forzada".

### 6. Few-shot prompting

**5-shot** = se ponen 5 preguntas resueltas (con su respuesta) antes de la pregunta de test. Le enseña al modelo el formato (que después de `Answer:` viene una letra) y mejora el desempeño. Es el protocolo estándar de MMLU.

---

## Recorrido por la notebook

### 0. Setup (celda 4)

Igual que en la Lecture 2: detecta Colab, clona el repo e instala `requirements/03_behavioral_analysis.txt` (que fija `transformers==4.51.3`, `torch==2.6.0`, `flash-attn==2.7.4.post1`), o hace `uv sync` en local. Configura plotly para exportar PNG.

### Cargar MMLU (celda 7)

```python
from datasets import load_dataset, concatenate_datasets

subjects = ['abstract_algebra', 'college_mathematics', 'high_school_mathematics']
N_SHOTS = 5

ds_list = []
dev_examples = {}
for subj in subjects:
    subj_ds = load_dataset('cais/mmlu', subj, split='test')
    subj_ds = subj_ds.map(lambda _: {"subject": subj}, batched=False)
    ds_list.append(subj_ds)

    subj_dev = load_dataset('cais/mmlu', subj, split='validation')
    dev_examples[subj] = list(subj_dev.select(range(min(N_SHOTS, len(subj_dev)))))
    print(...)

ds = concatenate_datasets(ds_list)
```

- `load_dataset('cais/mmlu', subj, split='test')`: librería `datasets` de Hugging Face. `'cais/mmlu'` es el repo del dataset, `subj` es la configuración (una por materia), `split='test'` el conjunto de evaluación. Cada ejemplo tiene `question` (str), `choices` (lista de 4 str) y `answer` (int 0-3).
- `.map(lambda _: {"subject": subj}, batched=False)`: agrega una columna `subject` a cada fila. La lambda ignora el ejemplo (`_`) y devuelve un dict con la columna nueva; `map` mezcla ese dict con las columnas existentes. Como `map` se ejecuta en el momento, no hay problema de que la lambda capture `subj` del loop.
- Para los few-shot usa el split `'validation'` y toma los primeros 5 con `.select(range(...))`; `list(...)` lo convierte en una lista de dicts. El comentario dice "dev set"; MMLU en Hugging Face tiene además un split llamado `dev` (5 ejemplos por materia, que es el que usa el protocolo original). Acá se usan los 5 primeros de `validation`. No sé si coinciden con los de `dev`; no cambia la lógica pero vale saberlo.
- `concatenate_datasets`: une los tres datasets en uno. El orden queda: abstract_algebra (índices 0-99), college_mathematics (100-199), high_school_mathematics (200-469). Esto importa más adelante: los índices 200+ son de high school.

Salida guardada: 100 + 100 + 270 = 470 preguntas de test, 5 shots por materia.

### Cargar el modelo (celda 9)

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch_type = torch.float16

model_id = 'microsoft/Phi-3-small-8k-instruct'
revision = 'main'

tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side='left', trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch_type,
                                             trust_remote_code=True, revision=revision)
model = model.to(device).eval()
```

- `float16`: la mitad de memoria que float32. Un modelo de ~7B parámetros en fp16 ocupa ~14 GB.
- `padding_side='left'`: al armar un batch de prompts de distinto largo, rellena **a la izquierda**. Es clave para `predict` (ver abajo).
- `trust_remote_code=True`: Phi-3-small trae código propio (arquitectura y tokenizer) en el repo de Hugging Face; esto permite ejecutarlo. Implica confiar en ese código.
- `tokenizer.pad_token = tokenizer.eos_token`: el tokenizer no tiene token de padding definido; se reutiliza el de fin de secuencia. Como hay `attention_mask`, los tokens de padding se ignoran igual.
- `revision='main'`: la rama del repo del modelo. Para reproducibilidad exacta se pondría un hash de commit.
- `.eval()`: modo evaluación (desactiva dropout).

La notebook dice que es el modelo de escala 7B con mejor accuracy en MMLU según HELM.

### 1. Evaluación estándar

#### `format_mmlu_prompt` (celda 14)

```python
LABELS = [" A", " B", " C", " D"]

def format_mmlu_prompt(question, choices, subject, labels=LABELS, few_shot_examples=None):
    subject_str = subject.replace("_", " ")
    prompt = f"The following are multiple choice questions (with answers) about {subject_str}.\n\n"

    if few_shot_examples:
        for ex in few_shot_examples:
            prompt += ex["question"] + "\n"
            for i, choice in enumerate(ex["choices"]):
                prompt += f"{labels[i]}. {choice}\n"
            prompt += f"Answer:{labels[ex['answer']]}\n\n"

    prompt += question + "\n"
    for i, choice in enumerate(choices):
        prompt += f"{labels[i]}. {choice}\n"
    prompt += "Answer:"
    return prompt
```

Línea por línea:

- `LABELS` tiene las letras **con un espacio adelante**. No es estético: el token que el modelo va a predecir después de `"Answer:"` es `" A"` (con espacio), igual que en GPT-2 vimos que `" Paris"` ≠ `"Paris"`.
- `subject.replace("_", " ")`: "abstract_algebra" → "abstract algebra".
- Encabezado estándar de MMLU: "The following are multiple choice questions (with answers) about ...".
- Por cada ejemplo few-shot: la pregunta, las 4 opciones como `" A. 4"` (con el espacio de la etiqueta, por eso en la salida las opciones aparecen con un espacio al principio), y `"Answer: A"` (`"Answer:"` + `" A"`), seguido de línea en blanco.
- `labels[ex['answer']]`: la letra de la respuesta correcta del ejemplo de demostración.
- La pregunta de test se formatea igual pero termina en `"Answer:"` sin letra: el modelo tiene que completar.
- Importante: **el parámetro `labels` se usa tanto en los few-shot como en la pregunta de test**. Si se pasan etiquetas distintas, las demostraciones también las usan. Esto es relevante en el Ejercicio 1.

La salida guardada de esta celda (que en realidad corresponde al print de la celda 16; las salidas de esta notebook están corridas de celda, ver "Ojo") muestra el prompt completo de `ds[0]`: 5 preguntas de álgebra abstracta resueltas y al final "Find the degree for the given field extension Q(sqrt(2), sqrt(3), sqrt(18)) over Q." con opciones 0, 4, 2, 6 y `Answer:`.

#### `predict` (celda 14)

```python
def predict(model, tokenizer, prompts, labels=LABELS):
    # arreglo: `labels` no estaba definido, ahora es un parámetro (por defecto A-D)
    label_token_ids = [tokenizer.encode(label, add_special_tokens=False)[0] for label in labels]

    inputs = tokenizer(prompts, return_tensors="pt", padding='longest', padding_side='left').to(model.device)
    with torch.no_grad():
        logits = model(**inputs).logits

    last_logits = logits[torch.arange(len(prompts)), -1]

    answer_logits = last_logits[:, label_token_ids]
    predictions = answer_logits.argmax(dim=-1).cpu().tolist()
    return predictions
```

**Arreglo de bug**: en la versión original, `predict` usaba una variable `labels` que no estaba definida dentro de la función (ni como parámetro). Se agregó `labels=LABELS` como parámetro con valor por defecto, así funciona con las letras A-D sin cambiar las llamadas existentes.

Línea por línea:

- `tokenizer.encode(label, add_special_tokens=False)[0]`: convierte `" A"` en su lista de token ids sin agregar tokens especiales (como BOS) y toma el primero. Para letras con espacio en un tokenizer tipo BPE es un solo token, así que `[0]` es *el* token. Si una etiqueta se tokenizara en varios tokens (por ejemplo `" (iv)"`), `[0]` tomaría solo el primero y la comparación podría ser ambigua (varias etiquetas podrían compartir el primer token). Resultado: lista de 4 ids.
- `tokenizer(prompts, return_tensors="pt", padding='longest', padding_side='left')`: tokeniza el batch. `padding='longest'` rellena todos al largo del prompt más largo del batch. `padding_side='left'` pone el relleno a la izquierda. Devuelve `input_ids` y `attention_mask`, shape `(B, L)`.
- `.to(model.device)`: al mismo device del modelo.
- `model(**inputs)`: desempaqueta el dict como argumentos (`input_ids=..., attention_mask=...`). `.logits` tiene shape `(B, L, V)`.
- `logits[torch.arange(len(prompts)), -1]`: indexación avanzada; para cada fila `b` del batch toma la posición `-1`. Shape `(B, V)`. Equivale a `logits[:, -1]`. **Esto solo es correcto porque el padding está a la izquierda**: así, la última posición de cada fila es siempre el último token real (`"Answer:"`). Con padding a la derecha, en los prompts más cortos la posición `-1` sería un token de padding.
- `last_logits[:, label_token_ids]`: selecciona las 4 columnas de las letras → `(B, 4)`. La columna `k` corresponde a `labels[k]`.
- `.argmax(dim=-1)`: índice 0-3 de la letra con mayor logit (equivalente a mayor log-prob, ver teoría). `.cpu().tolist()` → lista de ints.

Lo que devuelve es un **índice en el orden de `labels`**: 0 significa "eligió `labels[0]`". Con el default, 0 = " A".

#### Prompt de ejemplo (celda 16)

```python
example_prompt = format_mmlu_prompt(
    ds[0]["question"], ds[0]["choices"], ds[0]["subject"],
    few_shot_examples=dev_examples[ds[0]["subject"]],
)
print(example_prompt)
```

Arma el prompt de la primera pregunta con los 5 shots de su materia. Es el texto que se ve en la salida guardada descrita arriba.

#### Evaluación en batch (celda 19)

```python
BATCH_SIZE = 16

prompts = [
    format_mmlu_prompt(ex["question"], ex["choices"], ex["subject"],
                       few_shot_examples=dev_examples[ex["subject"]])
    for ex in ds
]
gold_labels = [ex["answer"] for ex in ds]
example_subjects = [ex["subject"] for ex in ds]

all_preds = []
for i in tqdm(range(0, len(prompts), BATCH_SIZE), desc="Evaluating"):
    batch = prompts[i : i + BATCH_SIZE]
    preds = predict(model, tokenizer, batch)
    all_preds.extend(preds)
```

- Arma los 470 prompts (cada uno con los shots de su materia).
- `gold_labels`: las respuestas correctas (0-3). `example_subjects`: la materia de cada pregunta, para el desglose.
- Recorre de a 16 (`range(0, 470, 16)` → 30 batches, como muestra la barra de `tqdm`). `prompts[i:i+16]` toma el batch; el último tiene 470 − 29×16 = 6 prompts.
- `tqdm` solo dibuja la barra de progreso.
- `all_preds` termina con 470 predicciones en el mismo orden que `ds`.

(El texto dice "one at a time" pero el código evalúa en batches de 16.)

#### Accuracy total (celda 21)

```python
correct = sum(p == g for p, g in zip(all_preds, gold_labels))
accuracy = correct / len(gold_labels)
```

`p == g` es un booleano; `sum` de booleanos cuenta los `True`. Salida guardada: **221/470 = 47.0%**. Con 4 opciones, el azar es 25%, así que está por encima del azar pero lejos de ser bueno. La variable `accuracy` se reutiliza después como referencia.

#### Accuracy por materia (celda 23)

```python
subject_correct = defaultdict(int)
subject_total = defaultdict(int)
for p, g, s in zip(all_preds, gold_labels, example_subjects):
    subject_total[s] += 1
    if p == g:
        subject_correct[s] += 1
```

`defaultdict(int)` arranca cada clave en 0, así no hay que inicializar. Salida guardada: abstract_algebra 46/100 = 46.0%, college_mathematics 42/100 = 42.0%, high_school_mathematics 133/270 = 49.3%.

La celda 24 plantea la pregunta: ¿por encima del azar razonando, o reconociendo preguntas vistas en el entrenamiento? Los stress tests están para separar esas dos hipótesis.

### 2. Stress testing I: input perturbations (variables irrelevantes)

#### `permute_choices` (celda 29)

```python
def permute_choices(choices, answer, seed=None):
    rng = random.Random(seed)
    indices = list(range(len(choices)))
    rng.shuffle(indices)
    permuted_choices = [choices[i] for i in indices]
    new_answer = indices.index(answer)
    return permuted_choices, new_answer
```

- `random.Random(seed)`: un generador de números aleatorios propio, con su semilla. No toca el estado global de `random`, así que es reproducible y no interfiere con otras celdas.
- `indices = [0, 1, 2, 3]` y se mezcla in place. Por ejemplo `[2, 0, 1, 3]`.
- `permuted_choices[j] = choices[indices[j]]`: en la posición nueva `j` va la opción que estaba en `indices[j]`.
- `new_answer = indices.index(answer)`: la posición `j` donde quedó la respuesta original (la `j` tal que `indices[j] == answer`).

Demo con `ds[0]` y `seed=0`, salida guardada: original A.0, **B.4**, C.2, D.6 → permutado A.2, B.0, **C.4**, D.6. El valor correcto (4) no cambió, solo la letra.

#### Stress test con permutaciones (celda 32)

```python
N_TRIALS = 3
trial_preds = []
trial_gold = []

for trial in range(N_TRIALS):
    perm_prompts = []
    perm_gold = []
    for idx, ex in enumerate(ds):
        seed = trial * len(ds) + idx
        perm_choices, perm_answer = permute_choices(ex["choices"], ex["answer"], seed=seed)
        perm_prompts.append(format_mmlu_prompt(
            ex["question"], perm_choices, ex["subject"],
            few_shot_examples=dev_examples[ex["subject"]],
        ))
        perm_gold.append(perm_answer)

    preds = []
    for i in tqdm(range(0, len(perm_prompts), BATCH_SIZE), desc=...):
        batch = perm_prompts[i : i + BATCH_SIZE]
        preds.extend(predict(model, tokenizer, batch))

    trial_preds.append(preds)
    trial_gold.append(perm_gold)
```

- 3 trials; en cada uno, cada pregunta recibe una permutación distinta.
- `seed = trial * len(ds) + idx`: semilla única por (trial, pregunta): trial 0 usa 0..469, trial 1 usa 470..939, etc. Reproducible y sin repetir.
- Solo se permutan las opciones de la pregunta de test; los few-shot quedan como estaban.
- `perm_gold`: la nueva posición correcta de cada pregunta.
- Después, la misma inferencia en batches que antes.
- Al final, `trial_preds[t][i]` y `trial_gold[t][i]` son la predicción y la respuesta correcta de la pregunta `i` en el trial `t`.

Una permutación aleatoria puede ser la identidad (probabilidad 1/24), así que algunas "variantes" son idénticas al original. No afecta mucho, pero existe.

#### Accuracy por trial (celda 34)

```python
trial_accs = []
for t in range(N_TRIALS):
    acc = sum(p == g for p, g in zip(trial_preds[t], trial_gold[t])) / len(ds)
    trial_accs.append(acc)

print(f"Permuted accuracy:    {np.mean(trial_accs):.1%} ± {np.std(trial_accs):.1%} ...")
print(f"Accuracy drop:        {accuracy - np.mean(trial_accs):+.1%}\n")
```

- Accuracy de cada trial, después media y desviación estándar. `np.std` por defecto es la desviación poblacional (`ddof=0`); con solo 3 trials es una estimación muy gruesa.
- `:+.1%` imprime el signo siempre: `+1.7%` significa que la accuracy *bajó* 1.7 puntos (original − permutada).

Salida guardada (aparece pegada a la celda 32): original 47.0%, permutada 45.3% ± 1.0%, por trial 46.6%, 45.1%, 44.3%, caída +1.7%.

#### Por materia (celda 36)

```python
for s in subjects:
    orig_acc = subject_correct[s] / subject_total[s]
    subj_mask = [i for i, es in enumerate(example_subjects) if es == s]
    subj_accs = []
    for t in range(N_TRIALS):
        sc = sum(trial_preds[t][i] == trial_gold[t][i] for i in subj_mask)
        subj_accs.append(sc / len(subj_mask))
```

`subj_mask` es la lista de índices de esa materia (no una máscara booleana, a pesar del nombre). Para cada trial cuenta aciertos solo en esos índices. Salida guardada: abstract_algebra 46.0% → 46.0% ± 0.0%; college_mathematics 42.0% → 42.3% ± 3.4%; high_school_mathematics 49.3% → 46.2% ± 1.0%. La notebook interpreta que high school es donde menos robusto es (quizás algunas respuestas memorizadas).

#### Robustez por pregunta (celda 38)

```python
correct_counts = []
for i in range(len(ds)):
    n_correct = sum(trial_preds[t][i] == trial_gold[t][i] for t in range(N_TRIALS))
    correct_counts.append(n_correct)

orig_correct = [int(p == g) for p, g in zip(all_preds, gold_labels)]

for k in range(N_TRIALS + 1, -1, -1):
    count = sum((oc + cc) == k for oc, cc in zip(orig_correct, correct_counts))
    print(f"  {k}/{N_TRIALS + 1} trials correct: {count} questions")
```

- `correct_counts[i]`: en cuántos de los 3 trials permutados acertó la pregunta `i` (0-3).
- `orig_correct[i]`: 1 si acertó en la versión original, 0 si no.
- `oc + cc` va de 0 a 4: aciertos sobre 4 versiones (original + 3 permutadas).
- `range(4, -1, -1)` = 4, 3, 2, 1, 0: imprime de más robusto a menos.

Salida guardada: 4/4 → 117, 3/4 → 64, 2/4 → 59, 1/4 → 82, 0/4 → 148 (suman 470). Cómo leerlo: las 117 que acierta siempre son candidatas a "sabe resolverlas" (o memorizó el valor, no la letra); las 148 que nunca acierta, no las sabe; las intermedias son inestables. Las que acierta solo en el original y en ninguna permutación serían las sospechosas de "memorizó la letra", pero este print no las separa (un 1/4 puede ser acertar el original o acertar una permutada).

**Análisis de la notebook (celda 39)**: la caída es de 1.7 puntos con una desviación de 1.0 entre trials, así que el modelo es mayormente robusto a permutar opciones; no muestra el sesgo posicional fuerte de Zheng et al.

#### Ejercicio 1 (celdas 43, 45, 47, `# TERRON`)

Pide otra perturbación que mantenga el contenido de la respuesta, por ejemplo cambiar las etiquetas (1)-(4) o (i)-(iv).

**Solución (celda 43):** en vez de permutar las opciones, **se permutan las letras** y las opciones quedan en el orden original.

```python
def manipulate_choices(labels, choices, answer, seed=None):
    rng = random.Random(seed)
    new_labels = list(labels)
    rng.shuffle(new_labels)

    new_choices = list(choices)

    new_answer = labels.index(new_labels[answer])
    return new_labels, new_choices, new_answer
```

- `new_labels = list(labels)` hace una copia (si no, `shuffle` mezclaría la lista global `LABELS` in place y rompería todo lo demás). Se mezcla: por ejemplo `[" C", " A", " B", " D"]`. Entonces, de arriba a abajo, las opciones se presentan como C, A, B, D.
- `new_choices = list(choices)`: las opciones no se tocan (copia por prolijidad).
- `new_labels[answer]`: la letra que le tocó a la opción correcta (que sigue en la posición `answer`).
- `labels.index(...)`: la posición de esa letra en la lista **original** `[" A", " B", " C", " D"]`. ¿Por qué? Porque `predict` (con su default `labels=LABELS`) devuelve un índice en el orden A, B, C, D. Si la respuesta correcta quedó con la letra " C", el modelo acierta si `predict` devuelve 2, así que `new_answer` tiene que ser 2. Esta es la parte sutil: **la respuesta se expresa en el espacio de las letras, no de las posiciones**.

Reproduciendo la función en Python puro (sin el modelo) con `seed=0` y las opciones de `ds[0]` (`['0', '4', '2', '6']`, respuesta 1): `new_labels = [' C', ' A', ' B', ' D']`, y `new_answer = 0`, porque a la opción correcta "4" le toca " A". Este cálculo es mío, la celda no tiene salida guardada.

La demo:

```python
orig_choices = ds[0]["choices"]
orig_answer = ds[0]["answer"]
orig_labels = LABELS
new_labels, new_choices, new_answer = manipulate_choices(orig_labels, orig_choices, orig_answer, seed=0)
```

**Arreglo de bug**: la demo original llamaba a `permute_choices` en vez de a `manipulate_choices`; `permute_choices` tiene otra firma (recibe `choices, answer` y devuelve 2 valores, no 3), así que el desempaquetado de 3 valores fallaba. Ahora llama a `manipulate_choices`.

```python
for i, c in enumerate(new_choices):
    marker = " <-- correct" if i == orig_answer else ""
    print(f"  {new_labels[i]}. {c}{marker}")

print("\ncorrect letter:", LABELS[new_answer], "| label next to the correct choice:", new_labels[orig_answer])
```

- El marcador usa `orig_answer` porque las opciones no se movieron: la correcta sigue en la misma posición.
- El último print es un chequeo de consistencia: `LABELS[new_answer]` (la letra que debería predecir el modelo) tiene que ser igual a `new_labels[orig_answer]` (la letra que aparece al lado de la opción correcta). Si son distintos, la función tiene un error.

**Celda 45 (código dado, no se edita):**

```python
N_TRIALS = 1
for trial in range(N_TRIALS):
    ...
    for idx, ex in enumerate(ds):
        seed = trial * len(ds) + idx
        perm_labels, perm_choices, perm_answer = manipulate_choices(LABELS, ex["choices"], ex["answer"], seed=seed)
        perm_prompts.append(format_mmlu_prompt(
            ex["question"],
            perm_choices,
            ex["subject"],
            labels=perm_labels,  # arreglo: faltaba esta coma
            few_shot_examples=dev_examples[ex["subject"]],
        ))
        perm_gold.append(perm_answer)
    ...
        preds.extend(predict(model, tokenizer, batch))
```

**Arreglo de bug**: faltaba la coma después de `labels=perm_labels`, lo que era un `SyntaxError` (dos argumentos keyword pegados). Ahora está.

Puntos para entender qué se está midiendo:

- `labels=perm_labels` hace que **los 5 few-shot también usen las letras mezcladas** (porque `format_mmlu_prompt` usa el mismo `labels` para todo). Las demostraciones son consistentes: por ejemplo, si una demo tiene respuesta en la posición 0 y `perm_labels[0] = " C"`, la demo dice `Answer: C`. Así el modelo ve el mismo esquema raro en las demos y en el test. Cada pregunta tiene su propia permutación (semilla distinta), así que el esquema cambia de prompt a prompt.
- `predict` se llama con el default `labels=LABELS`: compara los logits de " A", " B", " C", " D" en ese orden y devuelve el índice. Como `perm_answer` está en ese mismo espacio (por el `labels.index(...)`), la comparación `p == g` es correcta. Si la perturbación usara etiquetas distintas a A-D (por ejemplo " 1", " 2"...), habría que pasarle a `predict` esas etiquetas (`labels=...`) y definir `new_answer` como índice en esa lista: para eso sirvió el arreglo de agregar el parámetro `labels` a `predict`.
- `N_TRIALS = 1` y se **sobrescriben** `trial_preds`, `trial_gold` y `N_TRIALS` de la sección anterior. Si se vuelven a correr las celdas 34-38 después de esto, van a mostrar los resultados de esta perturbación, no de la permutación.

**Celda 47**: mismo código de accuracy que la 34 (con un solo trial, la desviación estándar es 0). No hay salida guardada.

**Qué mirar para responder (celda 48)**: qué se cambió (la correspondencia letra-posición y el orden alfabético de las letras), si cambió el contenido de la respuesta (no: el valor correcto es el mismo, solo cambia qué letra lo acompaña) y comparar la accuracy con el 47.0% original y con el 45.3% ± 1.0% de las permutaciones. Para decidir si la diferencia es "real", compararla con la variación entre trials de la permutación (y si se quiere, subir `N_TRIALS`). Una pista para interpretar: este cambio rompe una regularidad muy fuerte del preentrenamiento (A, B, C, D en orden) que la permutación de opciones no rompía; si el modelo cae mucho, sugiere que depende del orden canónico de las letras o de la posición, más que de leer la etiqueta que está al lado de cada opción.

### 3. Stress testing II: primitive substitutions (variables causalmente relevantes)

Ahora se cambia algo que **sí** cambia la respuesta: un número del enunciado. Procedimiento: tomar preguntas que el modelo acertó, sustituir un número, recalcular la respuesta y los distractores, y ver si sigue acertando. La suposición es que la variante sigue siendo una instancia de la misma "tarea".

#### Elegir los ejemplos (celda 55)

```python
correct_indices_with_numbers = [
    i for i, (p, g) in enumerate(zip(all_preds, gold_labels))
    if (p == g and
        len(re.findall(r'\d+', ds[i]['question'])) > 1 and
        ds[i]['subject'] == 'high_school_mathematics')
]
target_indices = correct_indices_with_numbers[:5]
```

- Filtra preguntas que el modelo acertó en la evaluación original (`p == g`), de high school mathematics, y con números en el enunciado.
- `re.findall(r'\d+', texto)`: todas las secuencias de dígitos. `> 1` exige **al menos dos** números (el comentario dice "at least one", pero el código pide más de uno).
- Toma los primeros 5.

Salida guardada: índices `[200, 207, 208, 215, 225]`:

1. **200**: reflejar un pentágono con vértices (– 2, – 4), (– 4, 1), (–1, 4), (2, 4), (3, 0) sobre y = x; correcta D. (– 4, –2).
2. **207**: suma de los n positivos con n² divisor de 1200; correcta A. 42.
3. **208**: dado justo lanzado 5 veces, probabilidad de sacar un 6 en a lo sumo 2 tiros; correcta C. 625/648.
4. **215**: hormiga de (−4, 6) al origen y luego a (4, 3), distancia total; correcta D. 5+2√13.
5. **225**: mayor entero impar que divide a 5!; correcta C. 15.

El modelo acertó los 5.

#### Generadores de variantes (celda 59)

La idea: una función por ejemplo que recibe `trial` (0-4) y devuelve `(question, choices, answer)` con un número cambiado y todo recalculado. `N_VARIANTS = 5`. Se guardan en el dict `variant_generators[idx] = función`.

Un patrón común a los 5:

```python
answer_pos = trial % 4
options = [d1, d2, d3]
options.insert(answer_pos, correct)
return question, options, answer_pos
```

La respuesta correcta se inserta en la posición `trial % 4`: trial 0 → A, 1 → B, 2 → C, 3 → D, 4 → A. Así la letra correcta rota entre variantes y el modelo no puede acertar por preferir siempre una letra (la notebook lo relaciona con Zheng et al. en las lecturas). `list.insert(i, x)` mete `x` en el índice `i` y corre el resto.

**Ejemplo 200 (reflexión sobre y = x):**

```python
def gen_example_200(trial):
    first_vertices = [(-3, -5), (-1, -3), (-5, -2), (-2, -6), (-6, -1)]
    v = first_vertices[trial]
    correct = (v[1], v[0])

    d1 = (-v[0], -v[1])  # reflection through origin
    d2 = (-v[0], v[1])   # reflection across y-axis
    d3 = (v[0], -v[1])   # reflection across x-axis
    ...
    question = (f"If a pentagon P with vertices at {fmt(v)}, (– 4, 1), (–1, 4), "
                f"(2, 4), and (3, 0) is reflected across the line y = x ...")
```

- Reflejar (x, y) sobre y = x da (y, x). Se cambia el **primer vértice** y la respuesta correcta es su reflejo.
- Distractores: los errores típicos (reflejar por el origen, por el eje y, por el eje x).
- `fmt(p)` da `"(x, y)"` con guion ASCII.
- Los otros 4 vértices quedan iguales. La pregunta original decía "one of the vertices of P' is", y como solo una opción es reflejo de algún vértice, sigue habiendo una sola respuesta correcta.

**Ejemplo 207 (n² divide a N):**

```python
targets = [1800, 2400, 3600, 4800, 7200]
N = targets[trial]
ns = [n for n in range(1, int(math.isqrt(N)) + 1) if N % (n * n) == 0]
correct_sum = sum(ns)
d1 = correct_sum + 6
d2 = correct_sum - 3 if correct_sum > 3 else correct_sum + 9
d3 = sum(ns[:-1]) if len(ns) > 1 else correct_sum + 12
for lst in [d1, d2, d3]:
    while d3 in (correct_sum, d1, d2):
        d3 += 1
```

- `math.isqrt(N)`: raíz cuadrada entera. Si n² divide a N, entonces n ≤ √N, así que alcanza con buscar hasta ahí.
- `ns`: todos los n que cumplen. La respuesta es su suma.
- Distractores: suma + 6, suma − 3 y "la suma sin el n más grande" (`ns[:-1]`, un error de olvidarse el último).
- El `for lst in [...]` no usa `lst`; en la práctica solo corre el `while` que corre `d3` hacia arriba si choca con otra opción. `d1` y `d2` nunca chocan entre sí ni con la correcta.

Reproduciendo los generadores en Python puro (son funciones sin dependencias del modelo; los resultados son míos, no salidas de la notebook): con N = 2400 la suma da **42**, igual que la respuesta original (1200 y 2400 tienen los mismos n: 1, 2, 4, 5, 10, 20). O sea, la variante del trial 1 **no cambia la respuesta**: no es un par contrastivo de verdad. Además, con N = 1800 el distractor "suma sin el último" también da 42, el valor original (la salida guardada de la celda 59 lo muestra: A. 72, B. 78, C. 69, D. 42). Y 3600 y 7200 dan la misma suma (168).

**Ejemplo 208 (binomial):**

```python
roll_counts = [4, 6, 7, 3, 8]
n = roll_counts[trial]
k = 2
p = Fraction(1, 6); q = Fraction(5, 6)
correct_prob = sum(comb(n, i) * p**i * q**(n - i) for i in range(k + 1))
p_exact_0 = q ** n
p_exact_1 = comb(n, 1) * p * q ** (n - 1)
p_exact_2 = comb(n, 2) * p**2 * q ** (n - 2)
```

- La cantidad de seises en `n` tiros es $X \sim \mathrm{Bin}(n, 1/6)$. Lo que se pide es $P(X \le 2) = \sum_{i=0}^{2} \binom{n}{i} (1/6)^i (5/6)^{n-i}$.
  - $\binom{n}{i}$ (`comb(n, i)`): de cuántas formas elegir cuáles `i` tiros son seis.
  - $(1/6)^i$: esos `i` tiros son seis.
  - $(5/6)^{n-i}$: los demás no.
- `Fraction` hace aritmética racional exacta, así la respuesta es una fracción simplificada exacta (no un float).
- Distractores: $P(X=0)$, $P(X=1)$, $P(X=2)$ por separado: el error de calcular solo un término en vez de sumar.
- `fmt_frac` la formatea en LaTeX `\frac{num}{den}`, igual que el original. Las llaves dobles `{{ }}` en el f-string producen llaves literales.

Salida guardada para n = 4: correcta 425/432.

**Ejemplo 215 (distancia de la hormiga):**

```python
configs = [((-3, 4), (8, 6)), ((-6, 8), (5, 12)), ((-8, 15), (4, 3)),
           ((-5, 12), (8, 6)), ((-9, 12), (3, 4))]
...
def simplify_sqrt(n):
    coeff, rem = 1, n
    for p in range(2, int(math.isqrt(n)) + 1):
        while rem % (p * p) == 0:
            rem //= p * p
            coeff *= p
    return coeff, rem
```

- Distancia total = $\sqrt{x_1^2 + y_1^2} + \sqrt{x_2^2 + y_2^2}$ (inicio→origen + origen→fin).
- `simplify_sqrt(n)`: escribe $\sqrt{n} = c\sqrt{r}$ sacando los factores cuadrados (por ejemplo, $\sqrt{52} = 2\sqrt{13}$). Recorre p desde 2 y mientras p² divida a lo que queda, lo saca y multiplica el coeficiente por p.
- `fmt_sqrt` lo muestra como entero si r = 1, o como `c\sqrt{r}`.
- El bloque `if sr == 1 and er == 1 ... else` arma la respuesta: si ambas distancias son enteras, las suma; si una es entera, `"entero+raíz"`; si no, `"raíz+raíz"`.
- Ojo: **todas las configuraciones elegidas son ternas pitagóricas** (3-4-5, 6-8-10, 5-12-13, 8-15-17, 9-12-15), así que todas las respuestas son enteros (15, 23, 22, 23, 20). La original era `5+2\sqrt{13}` (una parte irracional). La variante cambia también la *forma* de la respuesta, no solo el número.
- Distractores: distancia Manhattan (sumar |x|+|y|), solo el primer tramo, solo el segundo tramo.

**Ejemplo 225 (mayor divisor impar de n!):**

```python
factorials = [6, 7, 8, 9, 10]
nfact = math.factorial(n)
odd_part = nfact
while odd_part % 2 == 0:
    odd_part //= 2
```

- El mayor divisor impar de un número es lo que queda al sacarle todos los factores 2. Para 5! = 120 = 2³ · 15 es 15.
- `product_str` arma `"6\cdot 5\cdot ... 1"` para el enunciado, como en el original.
- Distractores: el mayor impar ≤ n, `odd_part // 3` (o `odd_part − 2`), `odd_part // 5` (o `odd_part + 2`); se filtran repetidos y positivos, y si faltan se rellena con impares desde 3.

Según mi reproducción en Python puro: 7! y 8! dan la misma respuesta (315), y para 6! la respuesta es 45 y uno de los distractores es **15**, el valor de la respuesta original (la salida guardada lo muestra: A. 45, B. 5, C. 15, D. 9).

**Sanity check** (final de la celda 59): imprime la variante 0 de cada generador. Salida guardada coherente con lo de arriba, todas con `answer_pos=0`.

#### Evaluar las variantes (celda 62)

```python
for rank, idx in enumerate(target_indices):
    ex = ds[idx]
    subj = ex["subject"]
    gen = variant_generators[idx]
    shots = dev_examples[subj]

    orig_pred = all_preds[idx]
    orig_gold = gold_labels[idx]
    orig_is_correct = orig_pred == orig_gold
    all_orig_correct += int(orig_is_correct)

    var_prompts = []
    var_golds = []
    for t in range(N_VARIANTS):
        q, c, a = gen(t)
        var_prompts.append(format_mmlu_prompt(q, c, subj, few_shot_examples=shots))
        var_golds.append(a)

    var_preds = predict(model, tokenizer, var_prompts)
    var_correct = sum(p == g for p, g in zip(var_preds, var_golds))
    ...
```

- Para cada uno de los 5 ejemplos: busca el resultado original (ya calculado en `all_preds`, no vuelve a correr el modelo).
- Genera las 5 variantes con los mismos 5 shots y las evalúa en un solo batch de 5 (`predict` con 5 prompts).
- Acumula aciertos y totales.
- El resto imprime, por trial, predicción vs. respuesta correcta y los primeros 100 caracteres de la pregunta. Para la impresión llama otra vez a `gen(t)` (es determinista, da lo mismo).

Salida guardada:

| Ejemplo | Original | Variantes correctas | Predicciones (trial 0-4) |
|---|---|---|---|
| 200 (reflexión) | D, correcto | 1/5 | D, D, D, D, D (solo acierta cuando la correcta es D) |
| 207 (n² \| N) | A, correcto | 1/5 | B, A, C, A, B |
| 208 (dado) | C, correcto | 3/5 | A, C, A, D, A |
| 215 (hormiga) | D, correcto | 1/5 | B, A, C, A, B |
| 225 (n!) | C, correcto | 4/5 | A, B, C, B, A |

Resumen (celda 64, salida pegada en la 62): original 5/5 = 100%, variantes 10/25 = 40%, caída +60%.

Cosas para leer en esa tabla: en el 200 el modelo dice D siempre, sin importar el vértice: eso es exactamente la **overstability** (no reacciona a un cambio que sí importa) y además parece una preferencia por la posición/letra en esta pregunta. En el 207, el trial 1 (N = 2400) tiene la misma respuesta que el original (42) y el modelo eligió A, que es 48: falló incluso una variante cuya respuesta no cambió de valor.

#### Ejercicio 2 (celdas 68, 70, 72, `# TERRON`)

La celda 68 repite `gen_example_200` original como referencia. El ejercicio pide reescribirlo para cambiar el **segundo** vértice en vez del primero: hay varias variables causalmente relevantes en el input (cada vértice), y testear cada una dice algo de cómo resuelve el modelo.

**Solución (celda 70):**

```python
def gen_example_200(trial):
    second_vertices = [(-5, 2), (-3, 1), (-6, 3), (-2, 5), (-5, 1)]
    v = second_vertices[trial]
    correct = (v[1], v[0])

    d1 = (-v[0], -v[1])
    d2 = (-v[0], v[1])
    d3 = (v[0], -v[1])
    ...
    question = (
        f"If a pentagon P with vertices at (– 2, – 4), {fmt(v)}, (–1, 4), "
        f"(2, 4), and (3, 0) is reflected across the line y = x to get a "
        f"new pentagon, P', then one of the vertices of P' is"
    )

    answer_pos = trial % 4
    options = [fmt(d1), fmt(d2), fmt(d3)]
    options.insert(answer_pos, fmt(correct))
    return question, options, answer_pos
```

- El primer vértice vuelve a ser el original `(– 2, – 4)` (con el mismo formato de guiones largos del enunciado original) y el que varía es el segundo (originalmente `(– 4, 1)`).
- La respuesta correcta es el reflejo del vértice nuevo; los distractores son los mismos tres errores típicos aplicados a ese vértice.
- La correcta sigue rotando de letra con `trial % 4`.
- El comentario dice que se revisó que ningún distractor sea el reflejo de otro vértice del pentágono (si lo fuera, habría dos respuestas correctas). Lo verifiqué reproduciendo el generador: los reflejos de los vértices fijos son (−4, −2), (4, −1), (4, 2), (0, 3), y ningún distractor de las 5 variantes coincide con ellos.
- Al final imprime la variante 0 como vistazo.

Notar que la función se llama igual que la original: **la redefine**. Desde acá, `gen_example_200` es la versión del segundo vértice.

**Celda 72 (dada):** es la celda 62 restringida al ejemplo 200, con:

```python
ex2_generators = {200: gen_example_200}
...
    gen = ex2_generators[idx]
```

En la versión original esta línea era `variant_generators = {200: gen_example_200}`, que **reemplazaba el dict completo** por uno con solo el 200. Ahora usa un dict aparte (`ex2_generators`) para no pisar los cinco generadores que necesita la sección 4. Como `variant_generators[200]` guarda el objeto función que existía cuando se corrió la celda 59, redefinir `gen_example_200` acá no lo toca: la sección 4 sigue usando la versión del primer vértice.

No hay salida guardada. **Qué mirar (celda 73)**: comparar con el 1/5 del primer vértice y, sobre todo, mirar las predicciones: si sigue diciendo la misma letra en las 5 variantes, el modelo no está usando el vértice que se cambió (overstability). Preguntas para razonar: ¿cuál es la variable causal que se sustituyó? ¿Su efecto sobre la respuesta es directo (la correcta es literalmente su reflejo)? Si el modelo es insensible a ella, ¿qué está usando para responder? Un dato que hay que tener en cuenta al interpretar: en esta versión, el primer vértice original (– 2, – 4) sigue en el enunciado, así que la respuesta original (– 4, –2) sigue siendo un vértice válido de P' (aunque no está entre las opciones). También considerar diferencias superficiales: los vértices fijos usan "–" (guion largo con espacios) y el vértice nuevo y las opciones usan "-" ASCII, y el original escribe `P’` con apóstrofo tipográfico mientras el generador usa `P'`.

### 4. Memorization probe: inyectar la respuesta original como distractor

Si el modelo memorizó el par pregunta-respuesta, al cambiar los números va a tender a elegir el **valor original**. El probe: en cada variante se reemplaza un distractor por la respuesta original y se cuentan tres resultados:

1. Elige la **nueva respuesta correcta** → resuelve la tarea.
2. Elige el **valor original** → señal de memorización.
3. Elige **otro distractor** → no resuelve, pero tampoco se pega al original.

#### Código (celda 77)

```python
original_answer_strings = {}
for idx in target_indices:
    ex = ds[idx]
    original_answer_strings[idx] = ex["choices"][ex["answer"]]
```

Guarda el **string** de la respuesta original de cada ejemplo (por ejemplo `"42"`, `"5+2\sqrt{13}"`).

```python
def inject_original_answer(variant_choices, variant_answer_pos, orig_answer_str):
    choices = list(variant_choices)
    correct_val = choices[variant_answer_pos]

    if orig_answer_str == correct_val:
        return None

    for i in range(4):
        if i != variant_answer_pos:
            choices[i] = orig_answer_str
            orig_pos = i
            break

    return choices, variant_answer_pos, orig_pos
```

- Copia las opciones de la variante.
- Si la respuesta original es igual a la nueva correcta (como en 207 con N = 2400), devuelve `None`: esa variante no sirve para el probe, porque elegir "la original" sería también elegir la correcta.
- Si no, reemplaza el **primer** slot que no es el correcto: si la correcta está en A, reemplaza B; si no, reemplaza A. `break` corta después del primer reemplazo.
- Devuelve las opciones nuevas, la posición correcta (no cambia) y la posición donde quedó el valor original.
- La comparación es por string exacto: si la variante escribiera el mismo valor con otro formato (por ejemplo, otros guiones), no lo detectaría como igual.

El loop del probe:

```python
for rank, idx in enumerate(target_indices):
    ...
    gen = variant_generators[idx]
    orig_ans_str = original_answer_strings[idx]

    for t in range(N_VARIANTS):
        q, c, a = gen(t)
        result = inject_original_answer(c, a, orig_ans_str)
        if result is None:
            continue
        new_choices, correct_pos, orig_pos = result
        probe_prompts.append(format_mmlu_prompt(q, new_choices, subj, few_shot_examples=shots))
        probe_correct_pos.append(correct_pos)
        probe_orig_pos.append(orig_pos)

    if not probe_prompts:
        print("Skipped: ...")
        continue

    preds = predict(model, tokenizer, probe_prompts)

    n_correct = sum(p == cp for p, cp in zip(preds, probe_correct_pos))
    n_original = sum(p == op for p, op in zip(preds, probe_orig_pos))
    n_other = len(preds) - n_correct - n_original
```

- Por cada ejemplo arma las variantes con el original inyectado, saltea las que dan `None`.
- Si todas se saltearon, imprime "Skipped" y sigue con el siguiente ejemplo.
- Una llamada a `predict` por ejemplo.
- Cuenta cuántas predicciones coinciden con la posición correcta, cuántas con la del original, y el resto es "other".
- En el print por variante, el orden de los `if` le da prioridad a "ORIGINAL" (aunque `p` no puede ser igual a ambos, porque las posiciones siempre son distintas).
- `Variant {t_i}` enumera sobre las variantes **usables**: si se salteó alguna, la numeración ya no coincide con el `trial` del generador.

Salida guardada:

| Ejemplo | Usables | Nueva correcta | Original | Otra |
|---|---|---|---|---|
| 200 | 5 | 0 | 1 | 4 |
| 207 | 4 | 2 | 1 | 1 |
| 208 | 5 | 3 | 1 | 1 |
| 215 | 5 | 0 | **5** | 0 |
| 225 | 5 | 1 | 1 | 3 |
| **Total** | **24** | 6 (25%) | 9 (38%) | 9 (38%) |

24 usables = 25 − 1 (la variante de 207 con N = 2400).

**Análisis de la notebook (celda 78)**: el modelo elige el valor original 38% de las veces, más que el 25% en que elige la nueva correcta: sugiere pattern matching contra asociaciones memorizadas. El caso más fuerte es el **215**: elige `5+2√13` en las 5 variantes, aunque las coordenadas cambiaron y ya no es la distancia correcta; nunca elige la correcta ni otro distractor. Combinado con la robustez a la permutación, la notebook concluye que el modelo memoriza el **valor** de la respuesta más que la **letra**.

Matices que vale la pena tener en cuenta al leer estos números (son observaciones mías a partir del código y las salidas guardadas, no de la notebook):

- **215**: como todas las variantes tienen respuestas enteras (ternas pitagóricas), `5+2\sqrt{13}` es la **única opción con raíz** entre cuatro números enteros. El modelo podría estar eligiendo "la que tiene forma de respuesta de este tipo de problema" y no necesariamente recordando el valor exacto. Es una señal de memorización fuerte, pero también hay un cambio de forma, no solo de número.
- **207, variante 0** (N = 1800): las opciones son 72, 78, 69, 42 y el 42 ya estaba como distractor en D. La inyección reemplaza la B por "42", así que quedan **dos opciones "42"** (B y D). El modelo eligió B (contado como "ORIGINAL"); en la celda 62, sin inyección, en la misma variante también había elegido B (que ahí era 78). Si hubiera elegido D, se contaría como "other" aunque es el mismo valor.
- **225, variante 0** (6!): pasa lo mismo, 15 ya era el distractor C y se inyecta en B → dos "15". El modelo eligió B (contado como "ORIGINAL").
- **200**: el modelo dice D en 4 de 5 casos, como en la celda 62; su único "ORIGINAL" es la variante 3, donde la correcta era D y el original se inyectó en A, y ahí sí se movió de D a A. Además, la respuesta original inyectada `(– 4, –2)` usa guiones largos mientras las otras opciones usan "-" ASCII, o sea, se ve distinta de las demás.

Nada de esto invalida la conclusión general, pero con 24 variantes cada caso pesa ~4 puntos porcentuales.

### Takeaways (celda 80)

- La accuracy esconde las razones: el mismo modelo acierta las 5 originales y le va mucho peor en las variantes.
- Las perturbaciones controladas hacen los resultados interpretables: cambiar una cosa a la vez.
- Dos formas de fallar, dos tests: **oversensitivity** (permutar opciones) y **overstability** (sustituir números).
- Estos tests muestran que hubo un atajo, no *dónde* se origina. Para eso hacen falta métodos de atribución (integrated gradients, Sundararajan et al. 2017) y el análisis causal de internals del resto del curso.

**Integrated gradients** (mencionado en las lecturas): asigna a cada token del input un puntaje de cuánto contribuyó a la predicción, acumulando gradientes a lo largo de un camino desde un input "en blanco" (baseline) hasta el input real. No se usa en esta notebook.

---

## Ojo / errores comunes / dudas

- **Los tres bugs arreglados**: (1) `predict` usaba `labels` sin definir; ahora es un parámetro con default `LABELS`. (2) La demo del Ejercicio 1 llamaba a `permute_choices` (firma distinta, devuelve 2 valores) y ahora llama a `manipulate_choices`. (3) Faltaba una coma después de `labels=perm_labels` en la celda 45 (era un `SyntaxError`).
- **Cuarto arreglo: la celda 72 ya no pisa `variant_generators`.** En el original hacía `variant_generators = {200: gen_example_200}`, que borraba los generadores de 207, 208, 215 y 225, y la celda 77 tiraba `KeyError` en el 207 al correr la notebook en orden. Ahora usa `ex2_generators`. Ojo: si después de correr la celda 72 volvés a correr la 59, todo queda como al principio; si corrés la 70 y luego la 59, la 59 redefine `gen_example_200` con la versión del primer vértice.
- **Con el generador del Ejercicio 2, el probe del 200 no tiene sentido**: el primer vértice original sigue en el enunciado, así que inyectar `(– 4, –2)` agregaría una segunda respuesta correcta.
- **Salidas corridas de celda.** En la notebook guardada, varias salidas aparecen pegadas a la celda anterior: el prompt de ejemplo aparece en la celda 14 (es de la 16), las accuracies aparecen en la 19 (son de la 21 y 23), las estadísticas de permutación en la 32 (son de la 34, 36 y 38), el resumen en la 62 (es de la 64). Ninguna celda tiene `execution_count`. Los números son coherentes entre sí, pero no hay que buscarlos en la celda "correcta".
- **Padding a la izquierda es obligatorio** para que `logits[:, -1]` sea el último token real. Hay una sutileza que no verifiqué: con padding a la izquierda, algunos modelos calculan las posiciones (para RoPE) sin tener en cuenta el padding y eso puede cambiar levemente las predicciones según con qué otros prompts cayó en el batch.
- **Logits vs log-probs**: el código compara logits; es equivalente a comparar log-probs entre las 4 letras. Pero es una evaluación forzada: siempre elige una letra aunque el modelo quisiera decir otra cosa.
- **`[0]` en `label_token_ids`** asume que cada etiqueta es un token. Con etiquetas multi-token (como `" (iv)"`) el método se rompe silenciosamente.
- **Índice en espacio de letras.** `predict` devuelve índices en el orden de `labels` que se le pasa. En el Ejercicio 1, `new_answer` debe estar en ese mismo espacio (por eso `labels.index(new_labels[answer])`).
- **Variables sobrescritas.** El Ejercicio 1 pisa `N_TRIALS`, `trial_preds` y `trial_gold`; el Ejercicio 2 pisa `gen_example_200` y `variant_generators`. Correr celdas fuera de orden da resultados engañosos.
- **Variantes que no cambian la respuesta.** 207 con N = 2400 da 42 igual que el original; en la celda 62 se cuenta como variante igual. 7! y 8! (225) comparten respuesta, igual que 3600 y 7200 (207) y dos configuraciones del 215 (23): no es un problema para "cambiar respecto al original", pero reduce la diversidad.
- **Distractores repetidos tras inyectar** (207 variante 0, 225 variante 0): quedan dos opciones con el mismo valor.
- **Comentario vs código en la celda 55**: dice "at least one number" pero exige más de uno.
- **"Dev set"** en los comentarios, pero se carga el split `validation`.
- **Tamaño de muestra.** 5 preguntas × 5 variantes es poco; los porcentajes de la sección 3 y 4 se mueven mucho con un solo caso. 3 trials de permutación también es poco para la desviación estándar.
- **`random.Random(seed)`** en vez de `random.seed`: no toca el estado global, por eso las permutaciones son reproducibles aunque se corran otras cosas aleatorias en el medio.

---

## Resumen de una línea por sección

- **Setup**: instala dependencias (pip pineado en Colab o `uv sync`) y configura plotly.
- **Dataset y modelo**: 470 preguntas de matemática de MMLU (3 materias) con 5 shots de `validation`; Phi-3-small-8k-instruct en fp16, padding a la izquierda.
- **Evaluación estándar**: prompt 5-shot terminado en `Answer:`, se elige la letra con mayor logit entre " A"/" B"/" C"/" D"; 47.0% total.
- **Stress test I (permutar opciones)**: cambia la letra de la respuesta sin cambiar su valor; cae solo 1.7 puntos (± 1.0): el modelo es mayormente robusto a este cambio (no es oversensitive acá).
- **Ejercicio 1**: se mezclan las letras manteniendo el orden de las opciones (también en los shots); `new_answer` se expresa en el orden A-D para que `predict` lo compare bien.
- **Stress test II (sustituir números)**: 5 generadores recalculan respuesta y distractores y rotan la letra correcta; de 5/5 se pasa a 10/25 (overstability: por ejemplo, en el 200 dice D siempre).
- **Ejercicio 2**: variar el segundo vértice en vez del primero para ver si el modelo es sensible a otra variable causal.
- **Probe de memorización**: se inyecta la respuesta original como distractor; la elige 9/24 veces contra 6/24 de la nueva correcta, con el caso extremo 215 (5/5), lo que sugiere memorización del valor (con los matices de distractores repetidos y cambio de forma).
- **Takeaways**: la accuracy no explica el porqué; perturbar una cosa a la vez; oversensitivity vs overstability; para saber *dónde* está el atajo hay que ir a los internals.
