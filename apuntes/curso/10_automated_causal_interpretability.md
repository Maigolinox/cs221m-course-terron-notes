# Localizing and Disentangling High-level Concepts (Lecture 10, notebook ii)

Esta notebook (`10_automated_causal_interpretability.ipynb`) ataca el **problema de disentanglement**: encontrar dentro de una activación un feature (dirección, subespacio o conjunto de neuronas) que controle **un** atributo de una entidad (por ejemplo, el país de una ciudad) sin tocar los demás atributos que esa misma activación también codifica (continente, idioma, zona horaria...). Usa la tarea de disentanglement de **RAVEL** sobre `allenai/OLMo-2-0425-1B` y compara cuatro métodos cada vez más finos: full vector patching, **DBM** (Differential Binary Masking), **DAS** (Distributed Alignment Search) y **MDAS** (Multi-task DAS).

Advertencia importante antes de empezar: la propia notebook dice que es **read-only**. Sus figuras se generaron con scripts de entrenamiento que no son públicos (`01_generate_and_filter_dataset.py`, `02_run_full_vector_patching.py`, `03_train_dbm.py`, `04_train_das.py`, `05_train_mdas.py`, `00_compute_scores_and_visualize.py`, dentro de una carpeta `causalab-internal`). Esos scripts **no están en este repo**, y la copia de `causalab/` incluida no tiene una tarea RAVEL. Lo que sí tiene es la maquinaria genérica para entrenar DAS y DBM (`causalab/causalab/experiments/train.py`, `jobs/DAS_grid.py`, `jobs/DBM_*`, `neural/featurizers.py`, `neural/pyvene_core/interchange.py`). En estos apuntes explico la notebook tal cual y, para "cómo se entrena y evalúa", explico lo que hace esa maquinaria de causalab, aclarando que no puedo confirmar que los scripts privados la usen exactamente así.

---

## 1. Ideas clave / teoría

### 1.1 El problema: polisemanticidad y entanglement

En clases anteriores localizábamos una variable eligiendo un sitio (capa, posición) del residual stream y parcheando el vector entero. Sabemos que la información de una entidad (sus atributos) se concentra en el residual stream del **último token de la entidad**. Pero ese vector codifica a la vez muchos atributos de "Paris": país, continente, idioma, latitud, zona horaria. Parchear el vector entero cambia todos juntos.

Elegir algunas dimensiones (neuronas) supone que cada atributo vive en un conjunto disjunto de neuronas. Como las neuronas suelen ser **polysemantic** (una neurona responde a varios conceptos), eso en general falla. La salida es mapear las activaciones a otro espacio de features (por ejemplo, una rotación) donde los atributos sí queden separados.

### 1.2 La tarea RAVEL y sus dos criterios

Cada ejemplo es un prompt de recuperación de un atributo de una entidad (ciudades). Para localizar el atributo objetivo (en las figuras guardadas, `country`) se usan dos tipos de pares base/source:

- **Cause**: la base pregunta por el atributo objetivo ("Stockholm is in the country of" → Sweden). Tras intervenir el feature con un source (prompt neutro "Barcelona is a large city"), el modelo debería decir el país del source (Spain). Es el mismo objetivo que la IIA de las clases 7-9.
- **Isolate**: la base pregunta por **otro** atributo ("Zurich is in the continent of" → Europe). Tras la misma intervención, el output debería **no cambiar** (Europe). Este es el criterio nuevo: garantiza que el feature no arrastra otros atributos.

El **disentangle score** de Huang et al. (2024) es el promedio de las accuracies de Cause e Isolate; es el tercer heatmap de cada método.

La conexión con la clase 9 es directa: Isolate es el análogo del criterio 3 (distinguishability). Un sitio con Cause alto pero Isolate bajo es un sitio donde el feature parcheado es compatible con "codifica país" y con "codifica toda la entidad", igual que un par confundido.

### 1.3 El modelo causal

```
entity  ──┐
          └──► value
attribute ┘
```

Tres variables: `entity` (la ciudad, input), `attribute` (qué se pregunta, input), `value` (el valor recuperado, output). Son inputs independientes que determinan juntos el valor. La variable objetivo de la intervención es `entity`.

Un punto que conviene pensar: si en este modelo de 3 variables intervenís `entity` en un ejemplo Isolate, el `value` **sí** cambiaría (el continente de Montreal no es Europa). O sea, el criterio Isolate no es "interchange de `entity` en este grafo". Lo que codifica es que el feature buscado no es la entidad entera sino **la parte de la entidad que determina el atributo objetivo**. Los nombres de las figuras lo reflejan: `cause_country_entity_heatmap.png` (Cause: target `entity`) e `isolate_country_none_heatmap.png` (Isolate: target "none", es decir, el output esperado es el de la base). Esto último es mi lectura de los nombres de archivo; el código que genera esas etiquetas no está disponible.

### 1.4 Los cuatro métodos

Todos intervienen en un sitio (capa, posición) a la vez y producen heatmaps capa × posición.

**Full vector patching.** $h_{\text{base}} \leftarrow h_{\text{src}}$ entero. Hipótesis de la notebook: Cause alto, Isolate bajo, porque cambia todos los atributos a la vez.

**DBM (Differential Binary Masking).** Aprende una máscara $m \in \{0,1\}^d$ sobre las dimensiones del residual stream:

$$h' = (1-m)\odot h_{\text{base}} + m \odot h_{\text{src}}$$

Durante el entrenamiento la máscara es continua, $m = \sigma(\text{logits}/T)$, con temperatura $T$ que se va bajando (annealing) para que termine casi binaria, más un término de sparsity que empuja a usar pocas dimensiones. Selecciona neuronas **alineadas con los ejes**. Si un atributo vive en una combinación lineal de neuronas, DBM no puede aislarlo.

**DAS (Distributed Alignment Search).** Aprende una rotación ortonormal y parchea solo un subespacio de dimensión $k$. Los pasos de la notebook, con $h \in \mathbb{R}^d$:
1. Featurize: $f = hR^\top$.
2. Patch: reemplazar las primeras $k$ coordenadas de $f_{\text{base}}$ por las de $f_{\text{src}}$.
3. Invertir: $h' = fR$.
4. Optimizar $R$ (manteniéndola ortonormal) para que el output intervenido coincida con el del modelo causal.

Como $R$ es ortonormal, las primeras $k$ coordenadas pueden llevar el atributo y las otras $d-k$ preservan todo lo demás. DAS encuentra **direcciones** (combinaciones lineales de neuronas), así que puede separar atributos que comparten neuronas.

**MDAS (Multi-task DAS).** Una sola rotación entrenada sobre una mezcla de ejemplos Cause e Isolate. Tiene que hacer las dos cosas a la vez: el subespacio transfiere el atributo objetivo (Cause) y deja intactos los demás (Isolate). Single-task DAS entrena por separado en Cause e Isolate, y nada garantiza que el subespacio entrenado para Cause no rompa otros atributos.

### 1.5 La forma exacta de DAS en causalab

En `causalab/causalab/neural/featurizers.py`, DAS se implementa con un `SubspaceFeaturizer`:

```python
rotate = pv.models.layers.LowRankRotateLayer(*shape, init_orth=True)   # shape = (d, k)
rotate = torch.nn.utils.parametrizations.orthogonal(rotate)
...
n_features=rotate.weight.shape[1]                                        # = k
```

- No es una rotación $d\times d$ completa: es una matriz $W \in \mathbb{R}^{d\times k}$ con **columnas ortonormales** (la parametrización `orthogonal` de PyTorch mantiene esa restricción durante el entrenamiento, así que no hace falta proyectar a mano). `init_orth=True` la inicializa ortogonal.
- `SubspaceFeaturizerModule.forward(x)`: `r = W.T`, `f = x @ W` (las $k$ coordenadas en el subespacio) y `error = x - f @ W.T` (la componente de $x$ ortogonal al subespacio).
- `SubspaceInverseFeaturizerModule.forward(f, error)`: `f @ W.T + error`.
- `FeatureInterchangeIntervention.forward(base, source)`: `f_base, err_base = featurizer(base)`, `f_src, _ = featurizer(source)`, y si no hay subconjunto de features `f_out = f_src`; devuelve `inverse(f_out, err_base)`.

Poniendo todo junto:

$$h' = W W^\top h_{\text{src}} + (I - W W^\top)\, h_{\text{base}} = h_{\text{base}} + W W^\top (h_{\text{src}} - h_{\text{base}}).$$

Es decir: la proyección sobre el subespacio viene del source, la componente ortogonal viene de la base. Es exactamente lo mismo que la versión "rotar con $R$ completa, cambiar las primeras $k$ coordenadas, rotar de vuelta" de la notebook (las primeras $k$ filas de $R$ son $W^\top$), pero sin necesitar las otras $d-k$ filas. El tamaño por defecto del subespacio es `n_features = 32` (en `DEFAULT_CONFIG` de `causalab/causalab/experiments/configs/train_config.py` y en `train_DAS`). No sé qué $k$ usaron los scripts privados.

### 1.6 La forma exacta de DBM en causalab

`build_feature_mask_intervention` en `featurizers.py`:

```python
self.mask = torch.nn.Parameter(torch.zeros(n_features))      # un logit por dimensión (o uno solo si tie_masks)
...
if self.training:
    gate = torch.sigmoid(mask / temp)
else:
    gate = (torch.sigmoid(mask) > 0.5).float()
f_out = (1.0 - gate) * f_base + gate * f_src
return self._inverse(f_out, base_err)

def get_sparsity_loss(self):
    gate = torch.sigmoid(self.mask / self.temperature)
    return torch.norm(gate, p=1)
```

- Los logits arrancan en 0, o sea gate = 0.5 en todas las dimensiones.
- En entrenamiento el gate es continuo y se vuelve más abrupto a medida que baja `temp`; en evaluación es un umbral duro en 0.5.
- La pérdida de sparsity es la norma L1 de los gates. Con featurizer identidad (DBM sobre neuronas), `f = h` y las dimensiones son neuronas del residual stream.
- `tie_masks=True` usa un solo logit para toda la unidad (selecciona o no el sitio completo); el DBM de la notebook ("selecciona un subconjunto de las d_model dimensiones") corresponde a `tie_masks=False`, que es lo que hace `causalab/causalab/experiments/jobs/DBM_feature_masks_grid.py`.

---

## 2. Recorrido por la notebook

### 2.1 Celda 0-1: introducción, RAVEL y modelo causal

Puro markdown, resumido en §1.1-1.3. La figura 1 viene del repo de RAVEL. El ejemplo canónico: base Cause "Paris is in the continent of → Europe", base Isolate "Paris is in the country of → France", source "Tokyo is a large city". Tras intervenir el feature de continente: "Asia" en Cause y "France" (sin cambio) en Isolate.

### 2.2 Celda 2: setup

```python
%load_ext autoreload
%autoreload 2
import sys
CAUSALAB_DIR = "causalab-internal"
sys.path.append(CAUSALAB_DIR)
sys.path.append('pyvene')
```

- `autoreload 2` recarga módulos editados sin reiniciar el kernel.
- Agrega al path una carpeta `causalab-internal` (versión interna de causalab con los scripts) y una copia local de `pyvene`. Ninguna de las dos carpetas existe en este repo, por eso la notebook no corre.

### 2.3 Celdas 3-4: generar y filtrar el dataset

```bash
python 01_generate_and_filter_dataset.py --model allenai/OLMo-2-0425-1B --output-dir ravel_exp
```

Por el nombre y por lo que se carga después, el script arma pares Cause e Isolate para cada atributo, los filtra quedándose con los que el modelo responde bien (como `filter_dataset` en la clase 9) y los guarda en `ravel_exp/cause_<attr>/` y `ravel_exp/isolate_<attr>/`, con splits train/test. No puedo ver su código.

```python
cause_data = json.load(open(os.path.join(CAUSALAB_DIR, "ravel_exp/cause_country/train_dataset.json")))
isolate_data = json.load(open(os.path.join(CAUSALAB_DIR, "ravel_exp/isolate_country/train_dataset.json")))
display(cause_data[0]); display(isolate_data[0])
```

- Carga el JSON crudo (no usa `load_counterfactual_examples`, así que son dicts, no `CausalTrace`). El formato es el mismo que produce `save_counterfactual_examples` de causalab: `{"input": {...}, "counterfactual_inputs": [{...}]}` con todos los valores del trace.

Salida guardada:
- Cause: input `{'entity': 'Stockholm', 'attribute': 'country', 'raw_input': 'Stockholm is in the country of', 'value': 'Sweden', 'raw_output': ' Sweden'}`; source `{'entity': 'Barcelona', 'attribute': None, 'raw_input': 'Barcelona is a large city', 'value': 'UNKNOWN_VALUE', ...}`.
- Isolate: input `Zurich ... continent → ' Europe'`; source `Montreal is a large city`.

Cosas a notar:
- El source tiene `attribute: None` y un prompt neutro que no pregunta nada; su `value` es `'UNKNOWN_VALUE'`. El source solo aporta la **entidad**; su output no importa. Lo que se compara es el output de la base intervenida contra la etiqueta del modelo causal (país de Barcelona en Cause; Europe en Isolate).
- `raw_output` lleva espacio adelante (`' Sweden'`), coherente con la tokenización tras "of".
- El dataset `isolate_country` contiene bases que preguntan por otros atributos (acá continent): "isolate respecto de country".

### 2.4 Sección 2: full vector patching (celdas 5-8)

```bash
python 02_run_full_vector_patching.py --model allenai/OLMo-2-0425-1B \
    --cause-test-dataset ravel_exp/cause_country/test_dataset.json \
    --isolate-test-dataset ravel_exp/isolate_country/test_dataset.json \
    --output-dir ravel_exp/full_vector_country
python 00_compute_scores_and_visualize.py --results .../cause/raw_results.pkl \
    --isolate-results .../isolate/raw_results.pkl --output-dir full_vector_viz
```

- No hay entrenamiento: solo se corre en los splits de **test**. Es el mismo experimento que `run_patching_for_datasets` de la clase 9 (un `ResidualStream` con featurizer identidad por sitio), una vez sobre Cause y otra sobre Isolate. El script de visualización toma los resultados crudos de ambos, calcula los scores y produce tres heatmaps (Cause, Isolate y Disentangle = promedio).
- Que guarde `raw_results.pkl` separado de los scores coincide con el diseño de causalab: `run_interchange_interventions` produce las generaciones crudas, y el puntaje se calcula después (`causal_score_intervention_outputs` o `compute_interchange_scores` en `causal_utils.py`, que explícitamente separan "correr intervenciones caro" de "puntuar con distintas variables barato"). No puedo verificar que el script use esas funciones.

Celda 7 (y sus copias en 11, 15, 19):

```python
heatmaps_dir = Path(CAUSALAB_DIR) / "full_vector_viz/heatmaps"
paths = [heatmaps_dir / "cause_country_entity_heatmap.png",
         heatmaps_dir / "isolate_country_none_heatmap.png",
         heatmaps_dir / "disentangle_heatmap.png"]
fig, axes = plt.subplots(1, 3, figsize=(16, 7))
for ax, path, title in zip(axes, paths, titles):
    ax.imshow(Image.open(path)); ax.set_title(title); ax.axis("off")
```

- Solo abre los tres PNG precomputados con PIL y los muestra lado a lado. No calcula nada.

Interpretación (celda 8): según el texto, Cause alto y Isolate bajo, confirmando la hipótesis. Parchear el vector de la entidad cambia todos los atributos: en un ejemplo Isolate, la base pasa a "pensar en Montreal" y responde North America en lugar de Europe.

### 2.5 Sección 3: DBM (celdas 9-12)

```bash
python 03_train_dbm.py --model allenai/OLMo-2-0425-1B \
    --train-dataset ravel_exp/cause_country/train_dataset.json \
    --cause-test-dataset ravel_exp/cause_country/test_dataset.json \
    --isolate-test-dataset ravel_exp/isolate_country/test_dataset.json \
    --output-dir ravel_exp/dbm_cause_country
```

- Se entrena **solo con datos Cause** (`--train-dataset` es cause) y se evalúa en test de Cause y de Isolate. O sea: el método solo ve el objetivo Cause durante el entrenamiento, e Isolate mide si "por casualidad" quedó desenredado.
- La notebook dice que se entrena un modelo DBM por (capa, posición) "para cada objetivo (cause e isolate)", pero el comando muestra un solo `--train-dataset` de cause. Lo dejo anotado como inconsistencia: no queda claro si hay otra corrida con isolate.
- Salen heatmaps de accuracy y de cantidad de dimensiones seleccionadas.

**Cómo se entrena una máscara DBM con causalab** (ver §3 para el loop genérico): `intervention_type="mask"`, un `Featurizer(n_features=d, tie_masks=False)` identidad por unidad, gate continuo con temperatura que baja linealmente de 1.0 a 0.001 durante la primera mitad de los pasos (`temperature_annealing_fraction=0.5`, `temperature_schedule=(1.0, 0.001)` en `DEFAULT_CONFIG`) y queda fija después; pérdida = cross-entropy de la etiqueta + `regularization_coefficient · (L1 de gates / número total de elementos de máscara)`. Al final se toman como seleccionadas las dimensiones con `sigmoid(mask) > 0.5` (`set_feature_indices`).

Interpretación (celda 12):
- Cause alto + Isolate alto en un sitio: hay neuronas distintas para atributos distintos (**neuron-level disentanglement**).
- Cause alto + Isolate bajo: las neuronas elegidas también llevan otros atributos; hace falta buscar direcciones (DAS).
- Pocas dimensiones con accuracy alta es lo mejor: el atributo está concentrado.

### 2.6 Sección 4: DAS (celdas 13-16)

```bash
python 04_train_das.py --model allenai/OLMo-2-0425-1B \
    --train-dataset ravel_exp/cause_country/train_dataset.json \
    --cause-test-dataset ... --isolate-test-dataset ... \
    --output-dir ravel_exp/das_cause_country
```

- Igual que DBM: se entrena con Cause, se evalúa en Cause e Isolate, un modelo DAS (una $W$) independiente por (capa, posición).
- El equivalente en causalab es `train_DAS` (`causalab/causalab/experiments/jobs/DAS_grid.py`): recibe un dict `{(layer, pos_id): InterchangeTarget}` (de `build_residual_stream_targets(..., mode="one_target_per_unit")`), rutas de train/test, el modelo causal, `target_variable_group` (acá sería `("entity",)`) y un `metric`. Si no le pasás config usa `intervention_type="interchange"`, `train_batch_size=32`, `evaluation_batch_size=64`, `training_epoch=4`, `init_lr=0.001`, `DAS.n_features=32`. Llama `train_interventions` y después dibuja heatmaps de train y test y guarda metadata con la mejor ubicación (`max` del score).

Interpretación (celda 16):
- Cause + Isolate altos: hay un subespacio de baja dimensión con el atributo y ortogonal a los demás (**directional disentanglement**).
- Cause alto + Isolate bajo: ni una rotación lineal los separa en ese sitio, o `n_features` está mal elegido.
- DAS > DBM en un sitio: el atributo está distribuido en varias neuronas (lo típico por polisemanticidad).
- DAS ≈ DBM: la representación ya estaba alineada con los ejes.

Ojo con la segunda viñeta: como DAS se entrenó solo con Cause, un Isolate bajo no prueba que "no se pueda" separar; puede ser que la rotación encontrada simplemente no tenía incentivo para preservar los otros atributos. Esa es justamente la motivación de MDAS.

### 2.7 Sección 5: MDAS (celdas 17-20)

```bash
python 05_train_mdas.py --model allenai/OLMo-2-0425-1B \
    --cause-train-dataset ravel_exp/cause_country/train_dataset.json \
    --cause-test-dataset ravel_exp/cause_country/test_dataset.json \
    --isolate-train-dataset ravel_exp/isolate_country/train_dataset.json \
    --isolate-test-dataset ravel_exp/isolate_country/test_dataset.json \
    --output-dir ravel_exp/mdas_cause_country
```

- La diferencia clave: dos datasets de **entrenamiento**, Cause e Isolate, para una sola rotación por sitio.
- En causalab no hay una función MDAS específica. `train_interventions` recibe **un** `train_dataset_path` y **un** `target_variable_group`, y etiqueta todo el train con ese grupo. Una forma natural de hacer MDAS con esta maquinaria sería etiquetar por separado (Cause con el interchange de `entity`, Isolate con el output de la base) y concatenar los ejemplos ya etiquetados antes del loop de `train_interventions_pyvene`, que solo necesita que cada ejemplo tenga su `label`. Es una hipótesis mía sobre cómo podría implementarse; el script real no está.

Interpretación (celda 20):
- Cause e Isolate altos **en el mismo sitio con una sola rotación** es la evidencia fuerte de disentanglement: el subespacio de dimensión $k$ codifica el atributo objetivo y nada más.
- Sitios donde MDAS tiene menos Cause que DAS: optimizar Isolate le cuesta efecto causal; puede haber estructura más compleja (un concepto derivado de otro).
- Huang et al. reportan pares difíciles (país/idioma, latitud/longitud) por **ripple effects**: cambiar un atributo implica, por los hechos del mundo, cambiar el otro. Si cambiás el país de una ciudad a Japón, "el idioma de esa ciudad" debería pasar a japonés, así que una etiqueta Isolate que exige que el idioma no cambie puede pedir algo incoherente con el conocimiento del modelo.

### 2.8 Sección 6-7: resumen y lecturas

La tabla final:

| Método | Interviene en | Disentanglement |
|---|---|---|
| Full vector | vector entero | ninguno |
| DBM | máscara binaria por dimensión | a nivel neurona |
| DAS | subespacio de dimensión $k$ | direccional |
| MDAS | subespacio de dimensión $k$ entrenado con Cause + Isolate | direccional, separado del resto |

Lecturas: Huang et al. 2024 (RAVEL, Cause/Isolate, MDAS; los métodos con supervisión contrafactual superan a PCA y SAEs), De Cao et al. 2020 y 2022 (differentiable masking sobre tokens y luego sobre neuronas con penalización L0), Davies et al. 2023 (máscaras sobre heads/MLPs según **desiderata**; las condiciones Cause/Isolate son desiderata de ese tipo), Geiger et al. 2023 (DAS con rotación ortogonal), Wu et al. 2023 (Boundless DAS, que aprende también el tamaño del subespacio y escala a 7B).

---

## 3. Cómo se entrena y evalúa una intervención automatizada (causalab)

Esto es lo que hace la librería incluida; los scripts privados de la notebook seguramente hacen algo equivalente, pero no lo puedo confirmar.

### 3.1 Orquestación: `train_interventions` (`causalab/causalab/experiments/train.py`)

```python
config = merge_with_defaults(config)
intervention_type = config.get("intervention_type", "mask")
...
train_dataset = load_counterfactual_examples(train_dataset_path, causal_model)
_initialize_featurizers(interchange_targets, config)
test_dataset = load_counterfactual_examples(test_dataset_path, causal_model)
labeled_train_dataset = causal_model.label_counterfactual_data(copy.deepcopy(train_dataset), list(target_variable_group))
```

- `merge_with_defaults` mezcla la config del usuario con `DEFAULT_CONFIG` (merge profundo en los dicts anidados `DAS`, `masking`, `featurizer_kwargs`).
- `intervention_type`: `"mask"` = DBM, `"interchange"` = DAS. Para mask exige `featurizer_kwargs`.
- Carga train y test como `CausalTrace` (JSON → `causal_model.new_trace`).
- `_initialize_featurizers`: a cada unidad sin featurizer propio (id `"null"`) le pone `Featurizer(n_features=d, tie_masks=...)` si es mask, o `SubspaceFeaturizer(shape=(d, n_features), trainable=True)` si es DAS. Si la unidad ya tenía un featurizer (por ejemplo PCA), lo respeta.
- `label_counterfactual_data` (`causal_model.py`): por cada ejemplo hace `run_interchange(base, {var: cf})` sobre las variables objetivo y agrega `"label"` = `raw_output` resultante y `"setting"` = el trace intervenido. Esas etiquetas son el objetivo de entrenamiento: lo que el modelo causal dice que debería salir tras el interchange.

```python
for key, target in pbar:
    train_interventions_pyvene(pipeline=pipeline, interchange_target=target, counterfactual_dataset=labeled_train_dataset,
        intervention_type=intervention_type, config=config,
        loss_and_metric_fn=lambda p, m, b, t, sp, sim: LM_loss_and_metric_fn(p, m, b, t, metric, source_pipeline=sp, source_intervenable_model=sim),
        source_pipeline=source_pipeline)
    feature_indices = target.get_feature_indices()
    train_raw_results = {key: run_interchange_interventions(pipeline=pipeline, counterfactual_dataset=train_dataset, interchange_target=target, ...)}
    train_eval = causal_score_intervention_outputs(raw_results=train_raw_results, dataset=train_dataset, causal_model=causal_model,
                                                   target_variable_groups=[target_variable_group], metric=metric)
    test_raw_results = {...test_dataset...}
    test_eval = causal_score_intervention_outputs(...)
    results_by_key[key] = {"train_score": ..., "test_score": ..., "feature_indices": ..., ...}
```

- Un entrenamiento **independiente por sitio** `(layer, pos_id)`: eso es lo que da un heatmap de DAS/DBM.
- Después de entrenar, evalúa con **generación** (`run_interchange_interventions`, la misma función del patching de la clase 9 pero ahora con el featurizer entrenado dentro de la intervención) en train y en test, y puntúa con `causal_score_intervention_outputs` (etiquetas recalculadas con el modelo causal, `metric(output, label)`, promedio). Ese promedio es la IIA (o accuracy) del sitio.
- Al final calcula promedios, cuenta unidades seleccionadas (DBM) y guarda resultados, metadata y modelos.

La separación train/test es importante: DAS tiene muchos parámetros ($d\times k$ por sitio), así que puede **sobreajustar** y "fabricar" un subespacio que haga pasar el entrenamiento. Por eso los heatmaps de la notebook son de test.

### 3.2 El loop de entrenamiento: `train_interventions` (`causalab/causalab/neural/pyvene_core/interchange.py`)

```python
intervenable_model = prepare_intervenable_model(pipeline, interchange_target, intervention_type=intervention_type)
intervenable_model.disable_model_gradients()
intervenable_model.eval()
...
optimizer_params = []
for k, v in intervenable_model.interventions.items():
    optimizer_params += list(v.parameters())
optimizer = torch.optim.AdamW(optimizer_params, lr=config["init_lr"], weight_decay=0)
scheduler = transformers.get_scheduler(scheduler_type, optimizer=optimizer, num_training_steps=num_epoch * num_batches)
```

- Crea el `IntervenableModel` de pyvene con la intervención correspondiente (interchange con featurizer DAS, o mask).
- **Congela el LM** (`disable_model_gradients`) y lo pone en `eval()`: solo se entrenan los parámetros de las intervenciones (la $W$ de DAS o los logits de la máscara).
- Optimizador AdamW sin weight decay; scheduler `"constant"` por defecto.
- Si es mask: arma el schedule de temperaturas (lineal y luego constante) y lo setea en cada intervención.

```python
for epoch in train_iterator:
    indices = list(range(len(counterfactual_dataset))); random.shuffle(indices)
    for step, start in enumerate(epoch_iterator):
        examples = [counterfactual_dataset[i] for i in indices[start:start + train_batch_size]]
        loss, eval_metrics, _ = loss_and_metric_fn(pipeline, intervenable_model, examples, interchange_target, source_pipeline, source_intervenable_model)
        if intervention_type == "mask":
            ... total_sparsity += mask_intervention.get_sparsity_loss(); total_mask_elements += mask.numel(); set_temperature(temp)
            loss = loss + regularization_coefficient * (total_sparsity / total_mask_elements)
        loss.backward(); optimizer.step(); scheduler.step(); current_step += 1
        intervenable_model.set_zero_grad()
```

- Epochs con shuffle, batches, pérdida del LM intervenido, regularización de sparsity normalizada (solo DBM), backward, paso del optimizador.
- Early stopping opcional por `patience` sobre la pérdida promedio de la época.
- Al terminar, para mask: `indices = nonzero(sigmoid(mask) > 0.5)` y `model_unit.set_feature_indices(indices)` (con `tie_masks`, la unidad entera entra o no). Para DAS no se toca `feature_indices` (queda `None` = usar las $k$ features del subespacio).

### 3.3 La pérdida: `LM_loss_and_metric_fn` (`causalab/causalab/experiments/metric.py`)

```python
batched_base, batched_counterfactuals, inv_locations, feature_indices = prepare_intervenable_inputs(pipeline, examples, interchange_target)
batched_inv_label_strs = [ex["label"] for ex in examples]
batched_inv_label_traces = [CausalTrace(mechanisms={"raw_input": ...}, inputs={"raw_input": label_str}) for label_str in batched_inv_label_strs]
batched_inv_label = pipeline.load(batched_inv_label_traces, max_length=pipeline.max_new_tokens, padding_side="right", add_special_tokens=False, use_chat_template=False)
for k in batched_base:
    batched_base[k] = torch.cat([batched_base[k], batched_inv_label[k]], dim=-1)
_, counterfactual_logits = intervenable_model(batched_base, batched_counterfactuals, unit_locations=inv_locations, subspaces=feature_indices, ...)
labels = batched_inv_label["input_ids"]
logits = counterfactual_logits.logits[:, -labels.shape[-1] - 1 : -1]
pred_ids = torch.argmax(logits, dim=-1)
...
loss = compute_cross_entropy_loss(logits, labels, pipeline.tokenizer.pad_token_id)
```

Línea por línea:
- `prepare_intervenable_inputs`: tokeniza bases y sources (left padding a `max_length`) y calcula índices de token por separado para base y source. Shapes: `input_ids` de la base `(B, L)`.
- Toma la etiqueta de cada ejemplo (ej. `" Spain"` en Cause, `" Europe"` en Isolate) y la tokeniza **con padding a la derecha** hasta `max_new_tokens` tokens, sin tokens especiales. Shape `(B, T)`.
- Concatena la etiqueta al final de la base: `(B, L+T)`. Esto es **teacher forcing**: en un solo forward el modelo ve la respuesta correcta y podemos leer la predicción de cada token de la respuesta.
- Forward del modelo intervenido (pyvene aplica la intervención en los sitios del prompt). `counterfactual_logits.logits` tiene shape `(B, L+T, V)`.
- `logits[:, -T-1:-1]`: los logits en las posiciones **anteriores** a cada token de la etiqueta, que son los que predicen esos tokens. Shape `(B, T, V)`.
- `argmax` para la accuracy (decodifica y aplica el `checker` contra la etiqueta), y `compute_cross_entropy_loss`: cross-entropy aplanada `(B·T, V)` vs `(B·T,)`, ignorando el `pad_token_id`. Como la etiqueta se paddeó a la derecha hasta `max_new_tokens` y `pad = eos`, los tokens de relleno no aportan a la pérdida: solo cuentan los tokens reales de la respuesta.

Así el gradiente fluye: pérdida → logits del LM congelado → activación intervenida $h' = h_{base} + WW^\top(h_{src} - h_{base})$ → $W$. DAS aprende el subespacio que hace que el LM intervenido diga lo que el modelo causal predice.

### 3.4 Resumen del pipeline de una intervención automatizada

1. Dataset contrafactual filtrado (el LM resuelve base y source), dividido en train/test.
2. Etiquetas del modelo causal para el interchange de las variables objetivo (`label_counterfactual_data`).
3. Por cada sitio: inicializar featurizer (DAS: $W$ ortonormal $d\times k$; DBM: logits de máscara en 0).
4. Entrenar solo esos parámetros con cross-entropy de la etiqueta bajo teacher forcing (+ sparsity y annealing en DBM).
5. Evaluar con generación greedy en test, comparar con la etiqueta usando un `metric`, promediar → heatmap.
6. Para RAVEL: repetir la evaluación sobre Cause e Isolate y promediar para el disentangle score.

---

## 4. Ejercicios

La notebook no tiene ejercicios ni celdas para completar (`FILL IN` / `# TERRON`). Lo que sí hay que hacer es leer los heatmaps guardados (son imágenes) con estas preguntas:
- Full vector: ¿en qué capas/posiciones Cause es alto? ¿Isolate es bajo en esos mismos sitios? El disentangle score debería quedar mediocre en todos lados.
- DBM: ¿hay sitios con Cause e Isolate altos? ¿Cuántas dimensiones usa (heatmap de feature counts)?
- DAS vs DBM: ¿en qué sitios DAS mejora? Eso indica atributo distribuido.
- MDAS vs DAS: ¿sube Isolate? ¿baja Cause en algún sitio? Lo segundo sugiere atributos que no se dejan separar linealmente o ripple effects.

---

## 5. Ojo / errores comunes / dudas

- **La notebook no se puede ejecutar**: los scripts y la carpeta `causalab-internal` no están; tampoco hay tarea RAVEL en la copia de causalab del repo.
- **Isolate no es "interchange de `entity`"** en el grafo de 3 variables; es "el output de la base no debe cambiar". El nombre `isolate_country_none` lo sugiere, pero no lo pude verificar en código.
- **DBM y DAS de la notebook se entrenan solo con Cause** (según los comandos). Isolate bajo en esos métodos no significa que el atributo sea inseparable; solo que nada los forzó a preservar el resto. MDAS es el test justo.
- **La "rotación" de causalab es de rango bajo**: $W\in\mathbb{R}^{d\times k}$ con columnas ortonormales, y $h' = h_{base} + WW^\top(h_{src}-h_{base})$. Equivale a la descripción de la notebook con una $R$ completa.
- **Train vs test**: con $d\times k$ parámetros por sitio, DAS puede sobreajustar; hay que mirar el test.
- **El tamaño del subespacio importa**: con `n_features` muy grande DAS se parece a full vector patching (y pierde Isolate); muy chico puede no alcanzar para Cause. Boundless DAS lo aprende.
- **Ripple effects**: en pares como país/idioma, una etiqueta Isolate puede ser incoherente con el mundo; un Isolate bajo ahí no es necesariamente un fracaso del método.
- **Duda que no pude verificar**: `train_interventions` llama `intervenable_model.eval()` antes de entrenar, y la máscara de DBM usa `self.training` para elegir entre gate suave (entrenamiento) y umbral duro (evaluación). Si pyvene registra las intervenciones como submódulos del `IntervenableModel`, el `eval()` las pondría en modo evaluación y el gate sería el umbral duro, sin gradiente útil para los logits. No tengo pyvene instalado para comprobar cómo guarda las intervenciones, así que no sé si esto ocurre en la práctica.
- **El texto de la notebook dice "un modelo DBM por (capa, posición) para cada objetivo (cause e isolate)"**, pero el comando mostrado solo entrena con cause.

---

## 6. Resumen de una línea por sección

- **Problema**: un vector de entidad codifica muchos atributos; queremos un feature que controle solo uno.
- **RAVEL**: Cause (el atributo objetivo cambia al del source) + Isolate (los otros atributos no cambian); disentangle = promedio.
- **Modelo causal**: `entity` + `attribute` → `value`; se interviene `entity`, pero Isolate exige que el output de la base se mantenga.
- **Full vector**: Cause alto, Isolate bajo; cambia todo.
- **DBM**: máscara por neurona con gates sigmoides, annealing de temperatura y L1; separa solo si los atributos viven en neuronas distintas.
- **DAS**: subespacio ortonormal de dimensión $k$; $h' = h_{base} + WW^\top(h_{src}-h_{base})$; separa atributos distribuidos.
- **MDAS**: una sola rotación entrenada con Cause e Isolate; la evidencia más fuerte de disentanglement.
- **Entrenamiento en causalab**: etiquetas del modelo causal, LM congelado, AdamW sobre los parámetros de la intervención, cross-entropy con teacher forcing, evaluación por generación en test.
