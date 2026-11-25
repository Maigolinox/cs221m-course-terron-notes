# causalab/neural: el LM, dónde intervenir, featurizers y pyvene

Este submódulo es la mitad "de bajo nivel": cómo cargo y corro el LM, cómo describo un lugar de la red (capa, componente, tokens), cómo describo el espacio de features donde intervengo (featurizer), y cómo todo eso se traduce a un `pyvene.IntervenableModel`. Responde a: ¿cómo hago, en código, "reemplazá la activación H del input base por la del contrafactual"?

Archivos (en `causalab/causalab/neural/`):

- `pipeline.py`: `Pipeline` (ABC) y `LMPipeline`.
- `model_units.py`: `ComponentIndexer`, `AtomicModelUnit`, `InterchangeTarget`.
- `LM_units.py`: `ResidualStream`, `MLP`, `AttentionHead`.
- `featurizers.py`: `Featurizer`, identidad, `SubspaceFeaturizer`, `SAEFeaturizer`, y las fábricas de intervenciones pyvene.
- `token_position_builder.py`: `TokenPosition`, `Template`, specs declarativos, combinadores.
- `pyvene_core/intervenable_model.py`, `pyvene_core/interchange.py`, `pyvene_core/collect.py`.
- `__init__.py` reexporta: `ResidualStream`, `AttentionHead`, las clases de featurizers y las fábricas `build_feature_interchange_intervention` / `build_feature_collect_intervention`, `ComponentIndexer`, `AtomicModelUnit`, `InterchangeTarget`, `Pipeline`, `LMPipeline`. (No reexporta `MLP` ni `TokenPosition`.)

## Ideas clave / teoría

### pyvene en dos párrafos

**pyvene** es la librería de Stanford NLP para intervenir modelos de PyTorch. Uno construye un `IntervenableConfig` con una lista de "representaciones" a intervenir; cada una es un dict con:

- `layer`: número de capa,
- `component`: qué tensor dentro de la capa (`"block_input"`, `"block_output"`, `"mlp_output"`, `"head_attention_value_output"`, ...),
- `unit`: cómo se indexa ese tensor (`"pos"` = por posición de token; `"h.pos"` = por cabeza y posición),
- `group_key`: qué fuente (source) alimenta esta intervención,
- `intervention_type`: **una clase** de intervención (no una instancia) que pyvene instancia.

Después `pv.IntervenableModel(config, model)` envuelve al modelo. Al llamarlo se le pasa el input base, una lista de inputs fuente, `unit_locations={"sources->base": (posiciones_en_fuentes, posiciones_en_base)}` y opcionalmente `subspaces` (qué coordenadas intercambiar). pyvene corre las fuentes, captura las activaciones en las posiciones indicadas, corre la base y en las mismas capas llama `intervencion.forward(base_activ, source_activ, subspaces)`, reemplazando la activación base por lo que devuelva.

### Featurizer invertible

Un **featurizer** es un par de módulos:

- forward: `x -> (f, error)`
- inverse: `(f, error) -> x̂`

con la propiedad de que `inverse(*forward(x)) == x` (reconstrucción exacta, gracias al término `error`). Intervenir "en el espacio de features" es:

```
f_base, err_base = F(h_base)
f_src,  _        = F(h_src)
f_out = f_base con algunas coordenadas reemplazadas por las de f_src
h_out = F^{-1}(f_out, err_base)
```

El `error` es lo que el featurizer no captura (residuo de reconstrucción). Se toma **siempre de la base**: la parte de la activación que está fuera del espacio de features no se toca. Para la identidad `error = None`; para una rotación de rango bajo es la componente ortogonal al subespacio; para un SAE es el error de reconstrucción del SAE.

### DAS como featurizer de rotación

DAS (Distributed Alignment Search) busca un subespacio de dimensión k del espacio de activaciones (dimensión d) donde vive la variable causal. Con una matriz W de d x k con columnas ortonormales:

- `f = x W` (coordenadas de x en el subespacio, shape k),
- `error = x - x W Wᵀ` (lo que queda fuera del subespacio),
- `x̂ = f Wᵀ + error`.

Intercambiar todas las coordenadas de f entre base y fuente da:

`h_out = h_base + (h_src - h_base) W Wᵀ`

o sea, se reemplaza la proyección de h_base sobre el subespacio por la proyección de h_src. Entrenar DAS = optimizar W (manteniéndola ortogonal) para que esta intervención produzca los labels del modelo causal.

## `pipeline.py`

### `_infer_device_and_dtype(requested_device, requested_dtype)`

Si no se especifica device usa `"cuda"` si hay GPU, si no `"cpu"`. Si no se especifica dtype usa `"auto"` (transformers toma el dtype de la config del modelo).

### `Pipeline` (ABC)

Clase base con `__init__(model_or_name)` que guarda el argumento y llama `_setup_model()`. Métodos abstractos: `_setup_model`, `load(raw_input)`, `dump(model_output)`, `generate(prompt)`, `intervenable_generate(intervenable_model, base, sources, map, feature_indices, source_representations=None)`. La idea es que podría haber pipelines para otros tipos de modelo; en el código solo existe `LMPipeline`.

### `LMPipeline.__init__`

```python
def __init__(self, model_or_name, *, max_new_tokens=3, max_length=None,
             logit_labels=False, position_ids=False, use_chat_template=False,
             padding_side="left", **kwargs):
```

- `model_or_name`: nombre de HF (`"gpt2"`, `"meta-llama/..."`) o un `PreTrainedModel` ya cargado.
- `max_new_tokens`: cuántos tokens genera (`generate` e `intervenable_generate`). También se usa como largo de los labels en la loss.
- `max_length`: si se da, **todo** input se paddea/trunca a esta longitud (`padding="max_length"`).
- `logit_labels`: se guarda pero no se usa en el archivo.
- `position_ids`: si es True, `load` agrega `position_ids` calculados por `model.prepare_inputs_for_generation`.
- `use_chat_template`: si es True, cada prompt se envuelve como mensaje de usuario con el chat template.
- `padding_side`: por defecto `"left"` (lo natural para generación autoregresiva: el último token real queda en la última posición).
- `**kwargs`: se leen `device`, `dtype`, `hf_token`, `config`.

### `_setup_model`

```python
if isinstance(self.model_or_name, str):
    hf_token = kwargs.get("hf_token") or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    self.tokenizer = AutoTokenizer.from_pretrained(self.model_or_name, token=hf_token)
    self.model = AutoModelForCausalLM.from_pretrained(..., torch_dtype=dtype).to(device=device)
    if hasattr(self.model.config, "_attn_implementation"):
        self.model.config._attn_implementation = "eager"
    if hasattr(self.model.config, "use_cache"):
        self.model.config.use_cache = False
else:
    self.model = self.model_or_name.to(device)
    if isinstance(dtype, torch.dtype):
        self.model = self.model.to(dtype)
    self.tokenizer = AutoTokenizer.from_pretrained(self.model.config.name_or_path)

self.tokenizer.pad_token = self.tokenizer.eos_token
self.tokenizer.pad_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)
if self.padding_side is not None:
    self.tokenizer.padding_side = self.padding_side
```

- Si se pasa un string, descarga/carga tokenizer y modelo.
- Fuerza atención `"eager"` (la implementación naïve, no SDPA/flash): pyvene necesita poder engancharse a los tensores intermedios de atención (p.ej. la salida de values por cabeza), cosa que las implementaciones fusionadas no exponen.
- `use_cache = False` en la config del modelo (aunque `generate` después pasa `use_cache=True` explícito).
- Con modelo pre-cargado, solo lo mueve de device y castea dtype si es un `torch.dtype` explícito.
- **`pad_token = eos_token`**: muchos LMs (GPT-2, Llama) no tienen pad token. Consecuencia: en la loss, los tokens EOS se ignoran como si fueran padding (ver `03_experiments.md`).

### `load(input, ...) -> dict` (tokenización)

```python
raw_input = [item["raw_input"] for item in input]
if use_chat_template: ...  # aplica apply_chat_template a cada texto
if max_length is None and not no_padding:
    max_length = self.max_length
...
enc = self.tokenizer(
    raw_input,
    padding=False if no_padding else ("max_length" if max_length else True),
    max_length=max_length,
    truncation=max_length is not None,
    return_tensors="pt",
    add_special_tokens=add_special_tokens,
    return_offsets_mapping=return_offsets_mapping,
)
if self.position_ids:
    enc["position_ids"] = self.model.prepare_inputs_for_generation(...)["position_ids"]
offset_mapping = enc.pop("offset_mapping", None)
for k, v in enc.items():
    enc[k] = v.to(self.model.device)
if offset_mapping is not None:
    enc["offset_mapping"] = offset_mapping
```

- Recibe una **lista de trazas** (o de dicts): de cada una solo usa `item["raw_input"]`. Por eso en varios lugares se arman "trazas" mínimas con un solo mecanismo `raw_input`.
- Padding: sin padding (`no_padding=True`), a `max_length` fijo, o al más largo del batch (`True`).
- Puede cambiar temporalmente el `padding_side` (se usa `"right"` para los labels).
- `return_offsets_mapping=True` devuelve, por token, el rango de caracteres `(start, end)` en el texto; es la base del sistema de token positions. Se saca del dict antes de mover a GPU (queda en CPU) y se vuelve a poner.
- Devuelve el dict de HF (`input_ids`, `attention_mask`, ...) en el device del modelo.

### `dump(model_output, is_logits=True) -> str | list[str]`

Decodifica a texto. Acepta tensor, lista/tupla de tensores o dict (usa `"sequences"` o `"scores"`). Si el tensor tiene 3 dims y son logits, hace `argmax(-1)`; si tiene 2 dims son ids; si tiene 1, le agrega dim de batch. Decodifica con `skip_special_tokens=True`. Ojo: si hay un solo elemento devuelve un **string**, no una lista de un elemento.

### `generate(input, **gen_kwargs) -> dict`

```python
inputs = self.load(input)
defaults = dict(max_new_tokens=self.max_new_tokens, pad_token_id=..., return_dict_in_generate=True,
                output_scores=True, do_sample=False, use_cache=True)
defaults.update(gen_kwargs)
with torch.no_grad():
    out = self.model.generate(**inputs, **defaults)
scores = [s.detach().cpu() for s in (out.scores or [])]
seq = out.sequences[:, -self.max_new_tokens :].detach().cpu()
...
return {"scores": scores, "sequences": seq, "string": self.dump(seq, is_logits=False)}
```

Generación greedy (`do_sample=False`). `scores` es una lista de largo `max_new_tokens` de tensores `(batch, vocab)`. `sequences` se recorta a los últimos `max_new_tokens` tokens (solo lo generado, porque con padding a izquierda el prompt ocupa el resto). Ojo: si la generación termina antes (EOS), HF paddea, así que siempre hay `max_new_tokens` columnas. Libera memoria GPU al final.

### `intervenable_generate(intervenable_model, base, sources, map, feature_indices, source_representations=None, **gen_kwargs)`

```python
defaults = dict(unit_locations=map, subspaces=feature_indices, max_new_tokens=..., pad_token_id=...,
                return_dict_in_generate=True, output_scores=True, intervene_on_prompt=True,
                do_sample=False, use_cache=True)
defaults.update(gen_kwargs)
with torch.no_grad():
    out = intervenable_model.generate(base, sources=sources,
                                      source_representations=source_representations, **defaults)
sequences = out[-1].sequences[:, -self.max_new_tokens :].detach().cpu()
```

- `base`: dict tokenizado de los inputs base.
- `sources`: lista de dicts tokenizados (uno por grupo/contrafactual), o `None` si se pasan `source_representations` (activaciones ya recolectadas, para cross-model patching).
- `map`: `{"sources->base": (cf_indices, base_indices)}`.
- `feature_indices`: los `subspaces` de pyvene.
- `intervene_on_prompt=True`: la intervención se aplica en el forward del prompt (en las posiciones dadas), y después se genera normalmente.
- pyvene devuelve una tupla; `out[-1]` es la salida de `generate` del modelo intervenido.
- `defaults.update(gen_kwargs)` pisa el `output_scores=True` por defecto con lo que pase el llamador. Con `False`, HF no calcula scores y el resultado no trae la clave `"scores"`. Con un entero k (modo top-k), a HF le llega `output_scores=k`, que es truthy, así que se calculan los logits completos y la reducción a top-k se hace después en `run_interchange_interventions`.

### `compute_outputs(dataset, batch_size=32)`

Corre `generate` sobre todas las bases y todos los contrafactuales (aplanados en orden: todos los cf del ejemplo 0, después los del 1, ...) y devuelve `{"base_outputs": [...], "counterfactual_outputs": [...]}`, una entrada por input con `sequences` `(1, max_new_tokens)`, `scores` (lista de `(1, vocab)`) y `string`. Lo usa `filter_dataset`.

`get_num_layers()` y `get_num_attention_heads()` leen la config.

## `model_units.py`

### `ComponentIndexer`

```python
class ComponentIndexer:
    def __init__(self, indexer, id="null"):
        self.indexer = indexer
        self.id = id

    def index(self, input, batch=False, is_original=None):
        if batch:
            return [self._call_indexer(i, is_original) for i in input]
        return self._call_indexer(input, is_original)

    def _call_indexer(self, input, is_original):
        if is_original is not None:
            try:
                return self.indexer(input, is_original=is_original)
            except TypeError:
                return self.indexer(input)
        else:
            return self.indexer(input)
```

Envuelve una función `input -> list[int]` (normalmente posiciones de tokens). Con `batch=True` la aplica a cada elemento de la lista, **uno por uno** (esto importa por el padding, ver más abajo). Si se pasa `is_original`, intenta llamarla con ese kwarg y si la función no lo acepta (TypeError) cae a la llamada sin él. Ojo: ese `except TypeError` también se traga cualquier TypeError legítimo que ocurra *dentro* del indexer y reintenta sin el kwarg.

### `AtomicModelUnit`

La unidad básica de intervención: **dónde** (capa, componente, índices) y **en qué espacio** (featurizer + subconjunto de features).

```python
def __init__(self, layer, component_type, indices_func, unit="pos", featurizer=None,
             feature_indices=None, shape=None, *, id="null"):
    ...
    if isinstance(indices_func, list):
        constant = indices_func
        self._static_indices = True
        def _const(_):
            return constant
        self._indices_func = ComponentIndexer(_const, id=f"constant_{constant}")
    else:
        self._static_indices = False
        self._indices_func = indices_func
    self.featurizer = featurizer or Featurizer()
    self.shape = shape
    self.feature_indices = None
    if feature_indices is not None:
        self.set_feature_indices(feature_indices)
```

- `layer`, `component_type` (nombre de componente de pyvene), `unit` (`"pos"` o `"h.pos"`).
- `indices_func`: un `ComponentIndexer` (índices dinámicos que dependen del input) o una lista fija (se envuelve en un indexer constante y la unit queda marcada como **estática**).
- `featurizer`: por defecto la identidad.
- `feature_indices`: subconjunto de features a intervenir; se valida que `max(indices) < featurizer.n_features` si `n_features` está definido.
- `shape`: dimensión del vector (p.ej. `(hidden_size,)`); el docstring dice "reservado", pero `train.py` lo usa para crear featurizers del tamaño correcto.

Métodos:

- `index_component(input, **kwargs)`: delega en el indexer (acepta `batch=`, `is_original=`).
- `get/set_feature_indices`, `set_featurizer` (revalida los índices contra el nuevo featurizer), `get_shape`, `get_index_id`, `is_static`, `set_layer/get_layer`.

```python
def create_intervention_config(self, group_key, intervention_type):
    config = {"component": self.component_type, "unit": self.unit,
              "layer": self.layer, "group_key": group_key}
    if intervention_type == "interchange":
        config["intervention_type"] = self.featurizer.get_interchange_intervention()
    elif intervention_type == "collect":
        config["intervention_type"] = self.featurizer.get_collect_intervention()
    elif intervention_type == "mask":
        config["intervention_type"] = self.featurizer.get_mask_intervention()
    ...
    return config
```

Esta es la traducción a pyvene: un dict por unit, con la **clase** de intervención que le da el featurizer según el tipo.

### `InterchangeTarget`

Contenedor de `list[list[AtomicModelUnit]]`:

- Lista exterior = **grupos**; cada grupo comparte un input contrafactual (grupo i ↔ `counterfactual_inputs[i]`).
- Lista interior = las units de ese grupo (todas se intervienen a la vez).

Se comporta como lista (`[]`, `len`, iteración). Utilidades:

- `flatten()`: todas las units en una lista plana (el orden de `flatten` es el orden en que se crean las intervenciones en pyvene, y por lo tanto el orden de `feature_indices`, de `intervenable_model.interventions`, etc.).
- `nest_to_match(flat_list)`: reshapea una lista plana a la estructura de grupos.
- `set_featurizer`, `set_feature_indices`: aplican a todas las units. Ojo: `set_featurizer` pone **la misma instancia** de featurizer en todas, o sea, compartirían parámetros.
- `get_feature_indices() -> {unit.id: indices}`.
- `save(parent_dir)`: escribe `units_metadata.json` (por unit: id, feature_indices, layer, component_type, unit, index_id, info del featurizer, shape, versión `"2.0"`) y, si hay featurizers no triviales, `featurizers.pt` con `{unit.id: featurizer.to_dict()}`.
- `load(parent_dir, ignore_version_mismatch=False)`: lee ambos archivos y para cada unit **existente** (tiene que construirse antes con los mismos ids) setea featurizer, feature_indices y shape. Valida la versión.

## `LM_units.py`

Subclases de `AtomicModelUnit` para transformers. Las tres arman un `id` legible que después usan las visualizaciones para parsear capa/posición/cabeza.

### `ResidualStream(layer, token_indices, *, featurizer=None, shape=None, feature_indices=None, target_output=False)`

```python
component_type = "block_output" if target_output else "block_input"
uid = f"ResidualStream(Layer-{layer},{component_type},Token-{tok_id})"
unit = "pos"
```

- `block_input` de la capa L = residual stream antes del bloque L; `block_output` = después. `block_input` de la capa 0 son los embeddings. Por eso los builders mapean "capa -1" a `layer=0, block_input`.
- El id incluye el componente para no confundir `block_input` de L con `block_output` de L-1 (que son el mismo tensor conceptual pero distinto hook).
- `load_modules(base_name, dir, token_positions)`: classmethod de formato viejo que parsea el id, busca la token position con ese id y carga el featurizer desde archivos `<path>_featurizer`/`_inverse_featurizer`.

### `MLP(layer, token_indices, *, ..., location="mlp_output")`

`location` en `("mlp_input", "mlp_output", "mlp_activation")`; se usa directamente como componente pyvene. Id `MLP(Layer-L,location,Token-...)`. `load_modules` análogo.

### `AttentionHead(layer, head, token_indices, *, ..., target_output=True)`

```python
component_type = "head_attention_value_output" if target_output else "head_attention_value_input"
uid = f"AttentionHead(Layer-{layer},Head-{head},Token-{tok_id})"
unit = "h.pos"
```

La salida de una cabeza (valores ponderados por atención, antes de la proyección de salida). Lo especial es el indexado:

```python
def index_component(self, input, batch=False, **kwargs):
    if batch:
        return [
            [[self.head]] * len(input),
            [self._indices_func.index(x, **kwargs) for x in input],
        ]
    return [[[self.head]], [self._indices_func.index(input, **kwargs)]]
```

Con `unit="h.pos"` pyvene espera dos niveles de índices: primero las cabezas por ejemplo (`[[head]]*batch`) y después las posiciones por ejemplo. Por eso devuelve un par `[cabezas, posiciones]`.

## `featurizers.py`

### Identidad

```python
class IdentityFeaturizerModule(torch.nn.Module):
    def forward(self, x): return x, None

class IdentityInverseFeaturizerModule(torch.nn.Module):
    def forward(self, x, error): return x
```

### `Featurizer`

```python
def __init__(self, featurizer=None, inverse_featurizer=None, *, n_features=None, id="null", tie_masks=False):
```

- `featurizer` / `inverse_featurizer`: módulos de torch (por defecto identidad).
- `n_features`: dimensión del espacio de features; obligatorio para máscaras.
- `id`: **convención importante**: `"null"` = featurizer trivial (identidad sin parámetros). `is_trivial()` es literalmente `self.id == "null"`. `train.py` usa esto para saber si una unit ya tiene un featurizer "de verdad" (no lo pisa).
- `tie_masks`: para la máscara, una sola compuerta escalar para todas las features (selecciona la unit entera) o una por feature.

Métodos que construyen intervenciones (cacheadas en el objeto con `hasattr`):

- `get_interchange_intervention()` → `build_feature_interchange_intervention(featurizer, inverse, id)`.
- `get_collect_intervention()` → `build_feature_collect_intervention(featurizer, id)`.
- `get_mask_intervention()` → `build_feature_mask_intervention(featurizer, inverse, n_features, id, tie_masks)`; lanza error si `n_features is None`.

Ojo con el cache: si cambio `tie_masks` después de haber pedido la intervención, se sigue usando la clase cacheada.

`featurize(x)` y `inverse_featurize(f, error)` llaman a los módulos.

Serialización:

- `to_dict()`: `None` para SAE y para triviales. Si no, devuelve `model_info` (nombres de clases, `n_features`, id, y para subspace la matriz de rotación y si requiere grad) más los `state_dict` de forward e inverse.
- `from_dict(data)`: reconstruye. Para `SubspaceFeaturizerModule` crea un `LowRankRotateLayer(in_dim, out_dim, init_orth=False)`, copia la matriz, lo envuelve con `torch.nn.utils.parametrizations.orthogonal`, y crea forward e inverse **compartiendo la misma capa**. Después carga los state dicts.
- `save_modules(path)` / `load_modules(path)`: formato viejo en dos archivos (`<path>_featurizer`, `<path>_inverse_featurizer`), misma lógica.

### Por qué las fábricas devuelven clases (no instancias)

El comentario del código lo explica: pyvene recibe `intervention_type` como **clase** y la instancia él mismo pasando sus kwargs (`embed_dim`, etc.). Pero nuestras intervenciones necesitan el featurizer, que pyvene no conoce. La solución es crear la clase dinámicamente dentro de una función, capturando el featurizer en un **closure**. El código lo llama "type crime" porque rompe el tipado estático.

### `build_feature_interchange_intervention(featurizer, inverse_featurizer, featurizer_id)`

```python
class FeatureInterchangeIntervention(pv.TrainableIntervention, pv.DistributedRepresentationIntervention):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._featurizer = featurizer
        self._inverse = inverse_featurizer

    def forward(self, base, source, subspaces=None):
        f_base, base_err = self._featurizer(base)
        f_src, _ = self._featurizer(source)
        if subspaces is None or _subspace_is_all_none(subspaces):
            f_out = f_src
        else:
            f_out = pv.models.intervention_utils._do_intervention_by_swap(
                f_base, f_src, "interchange", self.interchange_dim, subspaces,
                subspace_partition=self.subspace_partition, use_fast=self.use_fast)
        return self._inverse(f_out, base_err).to(base.dtype)
```

Línea por línea:

- Hereda de `TrainableIntervention` (tiene parámetros entrenables: pyvene los incluye y los mueve de device) y `DistributedRepresentationIntervention` (marca que opera sobre una representación distribuida, con subespacios).
- En `__init__` asigna los módulos capturados como atributos. Como la intervención es un `nn.Module` y los featurizers también, quedan **registrados como submódulos**: sus parámetros (p.ej. la rotación de DAS) aparecen en `intervention.parameters()`. Así es como el optimizador de `train_interventions` llega a la rotación.
- `forward`: `base` y `source` son las activaciones en las posiciones elegidas (shape aprox. `(batch, n_pos, d)` o `(batch, d)` según pyvene). Se featurizan ambas.
- Si no hay subespacios (`feature_indices=None` para todas las units), `f_out = f_src`: se reemplazan **todas** las features por las de la fuente.
- Si hay subespacios, `_do_intervention_by_swap` (función interna de pyvene) reemplaza en `f_base` solo las coordenadas indicadas por las de `f_src`.
- Se invierte usando el **error de la base** y se castea al dtype original.

Con featurizer identidad y `subspaces=None`: `h_out = h_src`. Es el activation patching clásico (full vector).

### `build_feature_collect_intervention(featurizer, featurizer_id)`

```python
class FeatureCollectIntervention(pv.CollectIntervention):
    def forward(self, base, source=None, subspaces=None):
        f_base, _ = self._featurizer(base)
        return pv.models.intervention_utils._do_intervention_by_swap(
            f_base, source, "collect", self.interchange_dim, subspaces, ...)
```

No modifica nada: featuriza la activación y la devuelve (opcionalmente solo las coordenadas de `subspaces`) para que pyvene la guarde. Es lo que usa `collect_features`.

### `build_feature_mask_intervention(featurizer, inverse_featurizer, n_features, featurizer_id, tie_masks=False)`

La intervención de **DBM**:

```python
class FeatureMaskIntervention(pv.TrainableIntervention):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        ...
        if tie_masks:
            self.mask = torch.nn.Parameter(torch.zeros(1), requires_grad=True)
        else:
            self.mask = torch.nn.Parameter(torch.zeros(n_features), requires_grad=True)
        self.temperature = None

    def forward(self, base, source, subspaces=None):
        f_base, base_err = self._featurizer(base)
        f_src, _ = self._featurizer(source)
        mask = self.mask.to(f_base.device)
        temp = self.temperature.to(f_base.device)
        f_base = f_base.to(mask.dtype)
        f_src = f_src.to(mask.dtype)
        if self.training:
            gate = torch.sigmoid(mask / temp)
        else:
            gate = (torch.sigmoid(mask) > 0.5).float()
        f_out = (1.0 - gate) * f_base + gate * f_src
        return self._inverse(f_out.to(base.dtype), base_err).to(base.dtype)

    def get_sparsity_loss(self):
        gate = torch.sigmoid(self.mask / self.temperature)
        return torch.norm(gate, p=1)
```

- Parámetro `mask` (logits de la compuerta), inicializado en 0 → compuerta inicial `sigmoid(0)=0.5`: arranca mezclando mitad base y mitad fuente.
- `tie_masks=True`: un solo escalar (selecciona o no la unit entera). `False`: un logit por feature.
- La temperatura tiene que setearse desde afuera (`set_temperature`); sin ella `forward` falla.
- En **modo training** la compuerta es suave: `sigmoid(mask / T)`. Con T grande es casi 0.5; a medida que T → 0 se vuelve casi un escalón (0 o 1). Esto es la relajación continua de una máscara binaria.
- En **modo eval** la compuerta es dura: 1 si `mask > 0`, 0 si no.
- `f_out` es interpolación por feature entre base y fuente: con gate 1 la feature viene de la fuente (se intercambia), con gate 0 queda la de la base.
- Ojo: **ignora `subspaces`**. La selección la decide la máscara, no `feature_indices`.
- `get_sparsity_loss`: norma L1 de las compuertas (suaves). Penaliza tener muchas features "encendidas"; el loop de entrenamiento la suma a la loss.

### `SubspaceFeaturizerModule` / `SubspaceInverseFeaturizerModule` / `SubspaceFeaturizer`

```python
class SubspaceFeaturizerModule(torch.nn.Module):
    def __init__(self, rotate_layer):
        super().__init__()
        self.rotate = rotate_layer

    def forward(self, x):
        r = self.rotate.weight.T          # (k, d)
        f = x.to(r.dtype) @ r.T           # x @ W  -> (..., k)
        error = x - (f @ r).to(x.dtype)   # x - x W Wᵀ
        return f, error

class SubspaceInverseFeaturizerModule(torch.nn.Module):
    def forward(self, f, error):
        r = self.rotate.weight.T
        return (f.to(r.dtype) @ r).to(f.dtype) + error.to(f.dtype)   # f Wᵀ + error
```

- `rotate` es un `pv.models.layers.LowRankRotateLayer(d, k)` de pyvene, cuyo `weight` tiene shape `(d, k)` (de ahí que `n_features = weight.shape[1] = k`). Llamemos W a ese weight.
- Forward: `r = Wᵀ` (k x d); `f = x rᵀ = x W` (coordenadas en el subespacio); `error = x - f r = x - x W Wᵀ` (componente ortogonal al subespacio, si W tiene columnas ortonormales).
- Inverse: `f r + error = f Wᵀ + error`. Si f no cambió, devuelve exactamente x.
- Las dos clases comparten **la misma** `rotate_layer`, así que entrenar una entrena la otra.

```python
class SubspaceFeaturizer(Featurizer):
    def __init__(self, *, shape=None, rotation_subspace=None, trainable=True, id="subspace", **kwargs):
        assert shape is not None or rotation_subspace is not None
        if shape is not None:
            rotate = pv.models.layers.LowRankRotateLayer(*shape, init_orth=True)
        else:
            shape = rotation_subspace.shape
            rotate = pv.models.layers.LowRankRotateLayer(*shape, init_orth=False)
            rotate.weight.data.copy_(rotation_subspace)
        rotate = torch.nn.utils.parametrizations.orthogonal(rotate)
        rotate.requires_grad_(trainable)
        super().__init__(SubspaceFeaturizerModule(rotate), SubspaceInverseFeaturizerModule(rotate),
                         n_features=rotate.weight.shape[1], id=id, **kwargs)
```

- Dos formas de construirlo: `shape=(d, k)` (inicialización ortogonal aleatoria, para entrenar DAS) o `rotation_subspace` (una matriz `(d, k)` dada, p.ej. componentes de PCA).
- `torch.nn.utils.parametrizations.orthogonal` reparametriza `weight` para que **siempre** sea ortogonal (columnas ortonormales en el caso d x k): el optimizador mueve un parámetro libre y torch lo mapea a una matriz ortogonal. Así DAS optimiza sobre rotaciones sin tener que reproyectar a mano.
- `trainable=False` congela la rotación (PCA fijo). `trainable=True` es DAS.
- Ojo: cuando se pasa `rotation_subspace`, la parametrización ortogonal puede no reproducir exactamente la matriz copiada si esta no era ortonormal; las componentes de `TruncatedSVD` sí lo son.

### `SAEFeaturizerModule` / `SAEInverseFeaturizerModule` / `SAEFeaturizer`

```python
def forward(self, x):                       # SAEFeaturizerModule
    features = self.sae.encode(x.to(self.sae.dtype))
    error = x - self.sae.decode(features).to(x.dtype)
    return features.to(x.dtype), error

def forward(self, features, error):         # SAEInverseFeaturizerModule
    return self.sae.decode(features.to(self.sae.dtype)).to(features.dtype) + error.to(features.dtype)
```

Envuelve un SAE (con API de `sae_lens`: `encode`, `decode`, `dtype`, `cfg`). `error` es el error de reconstrucción del SAE, así la intervención no pierde información. `SAEFeaturizer(sae, trainable=False)` congela el SAE y toma `n_features = sae.cfg.to_dict()["d_sae"]`; id `"sae"`. Está marcado "currently unused". No se serializa (`to_dict` devuelve None; el docstring dice que se carga de sae_lens).

### Utilidades

- `_subspace_is_all_none(subspaces)`: `True` si `subspaces` es `None` o si **todas** las entradas son `None` (o listas de `None`). Importante: una lista vacía `[]` devuelve `False`. Entonces, para una unit con `feature_indices=[]` (DBM la deseleccionó) se va al camino de `_do_intervention_by_swap` con subespacio vacío, que no intercambia nada (hay un test de regresión específico: `tests/test_pyvene_core/test_empty_feature_indices_regression.py`).
- `build_SVD_featurizers(model_units, svd_results, trainable=False, featurizer_id="SVD")`: para cada unit, toma `svd_results[unit.id]["rotation"]`, crea `SubspaceFeaturizer(rotation_subspace=...)` y lo setea con `feature_indices=None`. Marcado "currently unused".

## `token_position_builder.py`

Resuelve "¿qué índice de token corresponde a esta parte del prompt?" de forma robusta a la tokenización.

### `TokenPosition(ComponentIndexer)`

```python
def __init__(self, indexer, pipeline, is_original=True, **kwargs):
    super().__init__(indexer, **kwargs)
    self.pipeline = pipeline
    self.is_original = is_original
```

Un `ComponentIndexer` que además conoce el pipeline (tokenizer). `is_original` se guarda pero lo que realmente decide es el kwarg que se pasa en cada llamada a `index` (ver `paired_token_position`).

`highlight_selected_token(input)`: tokeniza el prompt, obtiene las posiciones y devuelve el texto con esos tokens en `**negrita**` (salteando el padding). Útil para verificar a ojo.

### Indexers de conveniencia

- `get_last_token_index(input, pipeline)`: `[len(input_ids) - 1]` del prompt tokenizado. Con padding a la izquierda el último token real es el último índice.
- `get_all_tokens(input, pipeline, padding=False)`: devuelve una `TokenPosition` (id `"all_tokens"`) cuyo indexer da todos los índices no-pad (o todos, con `padding=True`).
- `get_list_of_each_token(input, pipeline)`: una `TokenPosition` por cada token no-pad del prompt, con id `tok_{i}_{texto}`. Es lo que usa el tracing de residual stream. Ojo: las posiciones quedan **fijas** (las del prompt usado para construirlas).

### Del carácter al token

```python
def get_tokens_in_char_range(offsets, start_char, end_char):
    if end_char <= start_char:
        return []
    starts = offsets[:, 0]
    ends = offsets[:, 1]
    nonpad = ~((starts == 0) & (ends == 0))
    overlap = (starts < end_char) & (ends > start_char)
    idx = torch.nonzero(nonpad & overlap, as_tuple=False).flatten()
    return idx.tolist()
```

- `offsets`: `(seq_len, 2)` del `offset_mapping` del tokenizer: por token, rango de caracteres `[start, end)`.
- Los tokens de padding (y algunos especiales) tienen offset `(0, 0)` y se descartan.
- Un token pertenece al rango si se solapa con `[start_char, end_char)`. Es inclusivo: si una palabra comparte token con un espacio o puntuación vecina, ese token entra.

`get_substring_token_ids(text, substring, pipeline, add_special_tokens=False, occurrence=0, strict=False)`: encuentra todas las apariciones del substring, elige la `occurrence` (soporta índices negativos), opcionalmente falla si hay más de una (`strict`), tokeniza con `pipeline.load(..., return_offsets_mapping=True)` y aplica `get_tokens_in_char_range`. Ojo: por defecto tokeniza **sin** special tokens, lo que puede desalinear índices con la tokenización real (que usa `add_special_tokens=True`) en modelos que agregan BOS.

### `Template`

Parsea un template con `{variable}` y sabe dónde cae cada variable en los tokens.

```python
def _parse(self):
    pattern = r"\{([^}]+)\}"
    for match in re.finditer(pattern, self.template_str):
        if match.start() > last_end:
            self.parts.append(("literal", self.template_str[last_end:match.start()]))
        self.parts.append(("variable", match.group(1)))
        last_end = match.end()
    if last_end < len(self.template_str):
        self.parts.append(("literal", self.template_str[last_end:]))
```

Divide el template en una lista alternada de `("literal", texto)` y `("variable", nombre)`.

`fill(values)`: concatena literales y `str(values[var])`; error si falta una variable ("You probably put a non-input variable in your template").

```python
def get_variable_positions(self, values, pipeline):
    char_positions = {}
    current_pos = 0
    full_text = []
    for part_type, content in self.parts:
        if part_type == "literal":
            full_text.append(content); current_pos += len(content)
        else:
            value_str = str(values[content])
            start_char = current_pos
            end_char = current_pos + len(value_str)
            char_positions.setdefault(content, []).append((start_char, end_char))
            full_text.append(value_str); current_pos = end_char
    full_text_str = "".join(full_text)

    cache_key = (id(pipeline.tokenizer), full_text_str)
    if cache_key in Template._position_cache:
        return Template._position_cache[cache_key]

    trace = CausalTrace(mechanisms={"raw_input": Mechanism(parents=[], compute=lambda t: t["raw_input"])},
                        inputs={"raw_input": full_text_str})
    tokenized = pipeline.load([trace], return_offsets_mapping=True)
    offsets = tokenized["offset_mapping"][0]

    variable_tokens = {}
    for var_name, char_ranges in char_positions.items():
        variable_tokens[var_name] = []
        for start_char, end_char in char_ranges:
            variable_tokens[var_name].extend(get_tokens_in_char_range(offsets, start_char, end_char))
    Template._position_cache[cache_key] = variable_tokens
    return variable_tokens
```

Esta es la pieza clave del sistema de posiciones:

1. Rellena el template **llevando la cuenta de en qué caracteres cae cada variable**. Una variable que aparece varias veces (en MCQA, `{object}` aparece dos veces) acumula varios rangos.
2. Cachea por `(id del tokenizer, texto)` en un dict **de clase** (compartido entre todos los templates, nunca se vacía).
3. Tokeniza el texto completo con `pipeline.load` y `offset_mapping` (la misma tokenización que se usa en las intervenciones, incluido padding a `max_length` si está configurado).
4. Mapea cada rango de caracteres a tokens.

La ventaja frente a "tokenizar el valor por separado y buscarlo" es que respeta cómo el tokenizer junta espacios y palabras en contexto.

Ojo: asume que el `raw_input` del modelo causal es exactamente `template.fill(values)`. Si el mecanismo de `raw_input` hace otra cosa (prefijos, chat template...), las posiciones de caracteres no coinciden. Y si `use_chat_template=True`, `load` envuelve el texto y los offsets se refieren al texto envuelto: los rangos calculados sobre el texto sin envolver quedan corridos.

### Specs declarativos: `build_token_position_factories(specs, template)`

```python
token_positions = {
    "last": {"type": "index", "position": -1},
    "x": {"type": "variable", "name": "x"},
    "second_token_of_x": {"type": "index", "position": 1, "scope": {"variable": "x"}},
    "token_after_x": {"type": "index", "position": +1, "relative_to": {"variable": "x"}},
    "correct_answer": lambda setting: {"type": "variable", "name": "option_Z" if ... else "option_X"},
}
factories = build_token_position_factories(token_positions, template)
tp = factories["x"](pipeline)   # -> TokenPosition
```

Devuelve `{nombre: factory}`, donde cada factory recibe el pipeline y devuelve un `TokenPosition`. `_build_factory` despacha:

- **callable** → `_build_dynamic_factory`: el indexer llama `spec_func(input_sample)` para obtener un spec concreto **por ejemplo**, construye una factory temporal con ese spec y la evalúa. Permite "la posición del símbolo correcto", que depende de la traza.
- `"type": "index"` → `_build_index_factory`, que según las claves va a:
  - `_build_absolute_index_factory(name, position)`: índice en la secuencia tokenizada completa (negativos desde el final, incluyendo padding si hay `max_length`, aunque con padding a izquierda `-1` sigue siendo el último token real). Valida rango.
  - `_build_scoped_index_factory` (`scope: {"variable": x}`): primero obtiene todos los tokens de la variable x y después indexa dentro de esa lista (`position=0` primer token de x, `-1` último).
  - `_build_relative_index_factory` (`relative_to: {"variable": x}`): offset `>= 0` se cuenta desde el **último** token de x (`+1` = el token siguiente a x); offset negativo desde el **primer** token de x (`-1` = el token anterior). Ojo: `position: 0` con `relative_to` da el último token de x. Valida que quede dentro de la secuencia.
- `"type": "variable"` → `_build_variable_factory`: valida en construcción que la variable exista en el template y, por ejemplo, devuelve `template_obj.get_variable_positions(input_sample, pipeline)[var]` (todos los tokens de la variable, incluyendo todas sus apariciones).

### Combinadores

```python
def paired_token_position(original_position, counterfactual_position, id="paired"):
    def indexer(input_sample, is_original=True):
        if is_original:
            return original_position.index(input_sample)
        else:
            return counterfactual_position.index(input_sample)
    return TokenPosition(indexer, original_position.pipeline, id=id)
```

Posición distinta en base y contrafactual: "intervengo el último token de la base usando la activación del token de la respuesta en el contrafactual". Funciona porque `prepare_intervenable_inputs` llama con `is_original=True` para bases y `False` para contrafactuales, y `ComponentIndexer._call_indexer` le pasa el kwarg.

`combined_token_position(token_positions, id="combined")`: concatena los índices de varias posiciones (propaga `is_original`).

## `pyvene_core/intervenable_model.py`

### `prepare_intervenable_model(pipeline, model_units, intervention_type="interchange") -> pv.IntervenableModel`

```python
if isinstance(model_units, list):
    interchange_target = InterchangeTarget([model_units])
else:
    interchange_target = model_units

static = True
for group in interchange_target:
    for model_unit in group:
        if not model_unit.is_static():
            static = False

configs = []
for i, group in enumerate(interchange_target):
    for model_unit in group:
        config = model_unit.create_intervention_config(i, intervention_type)
        configs.append(config)

intervention_config = pv.IntervenableConfig(configs)
intervenable_model = pv.IntervenableModel(intervention_config, model=pipeline.model, use_fast=static)
intervenable_model.set_device(pipeline.model.device)
```

- Una lista plana se envuelve en un único grupo (todas las units comparten el contrafactual 0).
- `static`: si **todas** las units tienen índices fijos, se pasa `use_fast=True` a pyvene (un camino más eficiente que asume las mismas posiciones en todo el batch).
- Por cada unit, en orden de grupos, crea el dict de config con `group_key = i` (índice del grupo = índice del contrafactual) y la clase de intervención del featurizer.
- Construye el `IntervenableModel` sobre el modelo del pipeline (no lo copia: es el mismo modelo) y lo pone en el device.

`delete_intervenable_model(m)`: mueve las intervenciones a CPU (`set_model=False` para no mover el LM), borra, `gc.collect()` y vacía la cache de CUDA. Se llama siempre al terminar, para no acumular memoria entre los muchos targets de una grilla.

## `pyvene_core/interchange.py`

### `prepare_intervenable_inputs(pipeline, examples, interchange_target)`

La función que arma todo lo que pyvene necesita para un batch:

```python
batched_base = [ex["input"] for ex in examples]
batched_counterfactuals = [
    list(cf_tuple) for cf_tuple in zip(*[ex["counterfactual_inputs"] for ex in examples])
]
```

- `batched_base`: lista de trazas base (largo `batch`).
- `batched_counterfactuals`: **transpuesta**: de `(batch, num_cf)` a `(num_cf, batch)`. El elemento j es la lista de los contrafactuales j de todos los ejemplos. Esto se alinea con los grupos del target.

```python
base_indices = [
    model_unit.index_component(batched_base, batch=True, is_original=True)
    for group in interchange_target for model_unit in group
]
counterfactual_indices = [
    model_unit.index_component(batched_counterfactual, batch=True, is_original=False)
    for group, batched_counterfactual in zip(interchange_target, batched_counterfactuals)
    for model_unit in group
]
feature_indices = [
    [model_unit.get_feature_indices() for _ in range(len(batched_base))]
    for group in interchange_target for model_unit in group
]
```

- `base_indices`: shape `(num_units, batch, num_posiciones)`: para cada unit, las posiciones en cada ejemplo base.
- `counterfactual_indices`: igual, pero cada unit se indexa sobre los contrafactuales **de su grupo** (el `zip` empareja grupo i con contrafactuales i). Ojo: si hay menos contrafactuales que grupos, el `zip` corta y la lista queda más corta que el número de units.
- `feature_indices`: `(num_units, batch, ...)`: los mismos índices de features repetidos por ejemplo (o `None`).

```python
batched_base = pipeline.load(batched_base)
batched_counterfactuals = [pipeline.load(bc) for bc in batched_counterfactuals]
inv_locations = {"sources->base": (counterfactual_indices, base_indices)}
return batched_base, batched_counterfactuals, inv_locations, feature_indices
```

Tokeniza base y cada grupo de contrafactuales, y arma el `unit_locations` de pyvene.

Ojo con el padding: las posiciones se calculan **ejemplo por ejemplo** (cada indexer tokeniza un solo prompt), pero después el batch se tokeniza junto. Si `pipeline.max_length` está definido, todo se paddea a la misma longitud en ambos casos y las posiciones coinciden. Si no, cada prompt individual no tiene padding, pero en el batch los más cortos reciben padding a la izquierda y sus posiciones quedan corridas. Todos los tests y scripts del repo construyen el pipeline con `max_length`, así que conviene hacer lo mismo.

### `batched_interchange_intervention(pipeline, intervenable_model, examples, interchange_target, output_scores=True, source_pipeline=None, source_intervenable_model=None)`

1. `prepare_intervenable_inputs` con el pipeline target.
2. Si hay `source_pipeline` (cross-model): `collect_source_representations` recolecta las activaciones de los contrafactuales en el modelo fuente y se pone `batched_counterfactuals = None`.
3. `pipeline.intervenable_generate(...)` con `output_scores`.
4. Mueve los inputs tokenizados a CPU.

Devuelve el dict `{"sequences", "scores"?, "string"}` del batch.

### `run_interchange_interventions(pipeline, counterfactual_dataset, interchange_target, batch_size=32, output_scores=True, source_pipeline=None) -> dict[str, list]`

```python
intervenable_model = prepare_intervenable_model(pipeline, interchange_target, intervention_type="interchange")
source_intervenable_model = None
if source_pipeline is not None:
    source_intervenable_model = prepare_intervenable_model(source_pipeline, interchange_target, intervention_type="collect")
all_outputs = []
for start in range(0, len(counterfactual_dataset), batch_size):
    examples = counterfactual_dataset[start : start + batch_size]
    with torch.no_grad():
        output_dict = batched_interchange_intervention(...)
        all_outputs.append(output_dict)
delete_intervenable_model(intervenable_model)
...
if not isinstance(output_scores, bool) and output_scores > 0:
    all_outputs = _convert_to_top_k(all_outputs, pipeline, k=output_scores)
all_outputs = _move_outputs_to_cpu(all_outputs)
all_outputs = {k: [output[k] for output in all_outputs] for k in all_outputs[0].keys()}
return all_outputs
```

- Construye el modelo intervenible una vez por target, recorre el dataset en batches sin gradientes, y libera.
- `output_scores`: `False` (sin logits), `True` (logits de vocabulario completo, caro) o un entero k (convierte a top-k).
- `_convert_to_top_k`: por cada paso generado, `torch.topk` sobre `(batch, vocab)` y decodifica los k tokens; guarda `top_k_logits`, `top_k_indices`, `top_k_tokens`. El comentario dice "while still on GPU", pero en realidad `intervenable_generate` ya devolvió los scores en CPU.
- `_move_outputs_to_cpu`: recorre recursivamente y hace `detach().cpu()`.
- Al final **"des-batchea" por clave**: devuelve `{"sequences": [tensor_batch0, tensor_batch1, ...], "string": [strs_batch0, ...], ...}`. Ojo: es una lista **por batch**, no por ejemplo; `string` de cada batch es una lista de strings (o un string si el batch tenía un solo ejemplo, por cómo funciona `dump`). `score_intervention_outputs` aplana esto.

### `train_interventions(pipeline, interchange_target, counterfactual_dataset, intervention_type, config, loss_and_metric_fn, source_pipeline=None) -> str`

El loop de entrenamiento de DAS y DBM. Lo explico en detalle en `03_experiments.md` (sección de entrenamiento), porque ahí se ve junto con la loss. En resumen: construye el modelo intervenible del tipo pedido, congela el LM, optimiza con AdamW solo los parámetros de las intervenciones, maneja el schedule de temperatura y la penalización de sparsity para máscaras, early stopping opcional, y al final (para máscaras) convierte las máscaras en `feature_indices` de cada unit.

## `pyvene_core/collect.py`

### `_collect_activations_single_batch(intervenable_model, loaded_inputs, indices)`

```python
location_map = {"sources->base": (indices, indices)}
activations = intervenable_model(loaded_inputs, unit_locations=location_map)[0][1]
return activations
```

Con un modelo intervenible en modo `"collect"`, se corre solo la base (sin fuentes) con las mismas posiciones como "fuente" y "base". pyvene devuelve `((base_outputs, activaciones_recolectadas), cf_outputs)`; `[0][1]` es la lista de activaciones, **una por unit** en el orden de creación.

### `collect_features(dataset, pipeline, model_units, batch_size=32) -> dict[str, Tensor]`

```python
intervenable_model = prepare_intervenable_model(pipeline, model_units, intervention_type="collect")
data = {model_unit.id: [] for model_unit in model_units}
for start in range(0, len(dataset), batch_size):
    batch = dataset[start : start + batch_size]
    batched_inputs = [example["input"] for example in batch]
    indices = [mu.index_component(batched_inputs, batch=True, is_original=True) for mu in model_units]
    loaded_inputs = pipeline.load(batched_inputs)
    activations = _collect_activations_single_batch(intervenable_model, loaded_inputs, indices)
    ...
    for activation_idx, model_unit in enumerate(model_units):
        unit_activations = activations[activation_idx]
        hidden_size = unit_activations.shape[-1]
        reshaped_activations = unit_activations.reshape(-1, hidden_size)
        data[model_unit.id].extend(reshaped_activations.cpu())
...
result = {unit_id: torch.stack(acts) for unit_id, acts in data.items()}
```

- Solo usa los `input` del dataset (no los contrafactuales).
- Por unit, aplana la activación a `(-1, hidden)`: si la token position devuelve varias posiciones, **cada posición pasa a ser una muestra** (el número de filas deja de ser el número de ejemplos).
- Devuelve `{unit.id: tensor (n_muestras, n_features)}`. Las features son las del featurizer de cada unit (identidad = activación cruda).
- Ojo: las keys son `unit.id`, así que dos units con el mismo id se pisan.

### `collect_source_representations(source_pipeline, examples, interchange_target, source_intervenable_model=None)`

Para cross-model patching: transpone los contrafactuales (`(num_cf, batch)`), los tokeniza con el **pipeline fuente**, calcula las posiciones con `is_original=False` (con el tokenizer fuente, a través de las token positions de cada unit... ojo, las `TokenPosition` llevan adentro su propio pipeline, que normalmente es el del target; si los tokenizers difieren, hay que construir token positions adecuadas) y llama a `collect_batch_representations`.

### `collect_batch_representations(pipeline, batched_counterfactuals, interchange_target, counterfactual_indices, intervenable_model=None)`

Si no se pasa un modelo, crea uno en modo collect con todas las units (aplanadas, un solo grupo). Después recorre los grupos: para el grupo i usa los contrafactuales i y las posiciones de las units de ese grupo, recolecta, y concatena todo en una lista plana (una entrada por unit, en orden). Ese es el formato de `source_representations` de pyvene.

Detalle: el modelo collect se construye con todas las units y en cada llamada por grupo se pasan solo las posiciones de ese grupo; esto depende de que pyvene acepte `unit_locations` parciales. No queda claro solo desde este código cómo pyvene maneja las units de otros grupos en esa llamada.

### `compute_svd(features_dict, n_components=None, normalize=False, algorithm="randomized")`

```python
for unit_id, features in features_dict.items():
    n_samples, n_features = features.shape
    max_components = min(n_samples, n_features) - 1
    n = min(max_components, n_components) if n_components is not None else max_components
    if normalize:
        mean = features.mean(dim=0, keepdim=True)
        std_vals = torch.clamp(features.var(dim=0) ** 0.5, min=1e-6)
        features = (features - mean) / std_vals
    svd = TruncatedSVD(n_components=n, algorithm=algorithm)
    svd.fit(features)
    components = svd.components_.copy()             # (n, d)
    rotation = torch.tensor(components).to(features.dtype)
    svd_results[unit_id] = {"components": components, "explained_variance_ratio": ...,
                            "rotation": rotation.T,  # (d, n) para SubspaceFeaturizer
                            "mean": mean, "std": std, "n_components": n}
```

- `TruncatedSVD` de sklearn. Con `normalize=True` se centra y estandariza cada dimensión, lo que la vuelve PCA (sobre la matriz de correlación).
- `n` se limita a `min(n_samples, n_features) - 1` (restricción de TruncatedSVD).
- `rotation` traspuesta queda `(d, n)`: el formato que espera `SubspaceFeaturizer(rotation_subspace=...)`.
- Ojo: la media y el desvío se devuelven pero el `SubspaceFeaturizer` **no** los usa: proyecta las activaciones crudas (sin centrar ni escalar) sobre direcciones calculadas en el espacio estandarizado. Para intervenir esto no rompe nada (la inversa es exacta por el término `error`), pero las direcciones no son exactamente "las componentes principales de las activaciones crudas".

## Ojo / errores comunes / dudas

- **`max_length` en el pipeline.** Sin él, las posiciones calculadas por ejemplo pueden no coincidir con las del batch paddeado a la izquierda (ver `prepare_intervenable_inputs`).
- **`feature_indices`: `None` vs `[]`.** `None` = todas las features (intercambio completo); `[]` = ninguna. Tras DBM con `tie_masks=True`, las units seleccionadas quedan con `None` y las no seleccionadas con `[]`.
- **La máscara ignora `subspaces`**: en modo `"mask"` no importa qué `feature_indices` tenga la unit durante el entrenamiento.
- **Modo train/eval de la máscara.** `train_interventions` llama `intervenable_model.eval()` antes del loop y nunca `train()`. La compuerta suave (`sigmoid(mask/T)`) solo se usa si la intervención está en modo `training`. Que la máscara reciba gradiente depende de si el `eval()` de pyvene se propaga a las intervenciones; eso no se puede verificar mirando solo causalab (pyvene no está instalado en este entorno). Hay tests de integración de DBM (`tests/integration/test_DAS_and_DBM_integration.py`) que sugieren que el flujo funciona, pero conviene revisarlo si DBM no aprende.
- **Los featurizers compartidos**: `InterchangeTarget.set_featurizer` pone la misma instancia en todas las units; si es entrenable, todas comparten la rotación.
- **El cache de `Template`** es global y crece sin límite; con datasets muy grandes y textos distintos puede ocupar memoria.
- **Chat template + token positions**: el sistema de offsets asume que el texto tokenizado es el `raw_input` del template. Con `use_chat_template=True` eso deja de ser cierto.
- **`component_type` es un string de pyvene**: los nombres válidos (`block_input`, `head_attention_value_output`, ...) dependen del soporte de pyvene para cada arquitectura.

## Resumen de una línea por sección

- **pyvene**: config con (layer, component, unit, group_key, clase de intervención); se llama con base, fuentes, `unit_locations` y `subspaces`.
- **Featurizer invertible**: `x -> (f, error)` y `(f, error) -> x`; la intervención cambia `f` y conserva el `error` de la base.
- **`LMPipeline`**: carga HF con atención eager, pad=eos a la izquierda; `load` tokeniza `raw_input`, `generate`/`intervenable_generate` generan greedy.
- **`AtomicModelUnit` / `InterchangeTarget`**: unidad = dónde + featurizer + features; target = grupos de units, grupo i ↔ contrafactual i.
- **`ResidualStream` / `MLP` / `AttentionHead`**: units concretas con componentes de pyvene; las cabezas usan `unit="h.pos"`.
- **Intervenciones**: interchange (swap de features), collect (leer features), mask (interpolación con compuerta sigmoide + L1).
- **`SubspaceFeaturizer`**: W (d x k) ortogonal parametrizada; `f = xW`, `error = x - xWWᵀ`; base de DAS y PCA.
- **Token positions**: specs declarativos (index / variable / scope / relative_to / dinámico) resueltos con offsets de caracteres del tokenizer.
- **`pyvene_core`**: preparar el modelo intervenible, armar posiciones por batch, correr interchange en lotes, recolectar activaciones y SVD.
