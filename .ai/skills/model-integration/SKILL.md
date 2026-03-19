---
name: Model Integration
description: >
  Patterns for integrating a new model into diffusers: standard pipeline setup,
  modular pipeline conversion, file structure templates, checklists, and conventions.
  Trigger: adding a new model, converting to modular pipeline, setting up file structure.
---

## Standard Pipeline Integration

### File structure for a new model

```
src/diffusers/
  models/transformers/transformer_<model>.py     # The core model
  schedulers/scheduling_<model>.py               # If model needs a custom scheduler
  pipelines/<model>/
    __init__.py
    pipeline_<model>.py                          # Main pipeline
    pipeline_<model>_<variant>.py                # Variant pipelines (e.g. pyramid, distilled)
    pipeline_output.py                           # Output dataclass
  loaders/lora_pipeline.py                       # LoRA mixin (add to existing file)

tests/
  models/transformers/test_models_transformer_<model>.py
  pipelines/<model>/test_<model>.py
  lora/test_lora_layers_<model>.py

docs/source/en/api/
  pipelines/<model>.md
  models/<model>_transformer3d.md                # or appropriate name
```

### Integration checklist

- [ ] Implement transformer model with `from_pretrained` support
- [ ] Implement or reuse scheduler
- [ ] Implement pipeline(s) with `__call__` method
- [ ] Add LoRA support if applicable
- [ ] Register all classes in `__init__.py` files (lazy imports)
- [ ] Write unit tests (model, pipeline, LoRA)
- [ ] Write docs
- [ ] Run `make style` and `make quality`
- [ ] Test parity with reference implementation (see `parity-testing` skill)

### Attention pattern

Attention must follow the diffusers pattern: both the `Attention` class and its processor are defined in the model file. The processor's `__call__` handles the actual compute and must use `dispatch_attention_fn` rather than calling `F.scaled_dot_product_attention` directly. The attention class inherits `AttentionModuleMixin` and declares `_default_processor_cls` and `_available_processors`.

```python
# transformer_mymodel.py

class MyModelAttnProcessor:
    _attention_backend = None
    _parallel_config = None

    def __call__(self, attn, hidden_states, attention_mask=None, ...):
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)
        # reshape, apply rope, etc.
        hidden_states = dispatch_attention_fn(
            query, key, value,
            attn_mask=attention_mask,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        hidden_states = hidden_states.flatten(2, 3)
        return attn.to_out[0](hidden_states)


class MyModelAttention(nn.Module, AttentionModuleMixin):
    _default_processor_cls = MyModelAttnProcessor
    _available_processors = [MyModelAttnProcessor]

    def __init__(self, query_dim, heads=8, dim_head=64, ...):
        super().__init__()
        self.to_q = nn.Linear(query_dim, heads * dim_head, bias=False)
        self.to_k = nn.Linear(query_dim, heads * dim_head, bias=False)
        self.to_v = nn.Linear(query_dim, heads * dim_head, bias=False)
        self.to_out = nn.ModuleList([nn.Linear(heads * dim_head, query_dim), nn.Dropout(0.0)])
        self.set_processor(MyModelAttnProcessor())

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        return self.processor(self, hidden_states, attention_mask, **kwargs)
```

Consult the implementations in `src/diffusers/models/transformers/` if you need further references.

### Pipeline rules

- All pipelines must inherit from `DiffusionPipeline`. Consult implementations in `src/diffusers/pipelines` in case you need references.
- DO NOT use an existing pipeline class (e.g., `FluxPipeline`) to override another pipeline (e.g., `FluxImg2ImgPipeline` which will be a part of the core codebase (`src`).

### Test setup

- Slow tests gated with `@slow` and `RUN_SLOW=1`
- All model-level tests must use the `BaseModelTesterConfig`, `ModelTesterMixin`, `MemoryTesterMixin`, `AttentionTesterMixin`, `LoraTesterMixin`, and `TrainingTesterMixin` classes initially to write the tests. Any additional tests should be added after discussions with the maintainers. Use `tests/models/transformers/test_models_transformer_flux.py` as a reference.

### Common diffusers conventions

- Pipelines inherit from `DiffusionPipeline`
- Models use `ModelMixin` with `register_to_config` for config serialization
- Schedulers use `SchedulerMixin` with `ConfigMixin`
- Use `@torch.no_grad()` on pipeline `__call__`
- Support `output_type="latent"` for skipping VAE decode
- Support `generator` parameter for reproducibility
- Use `self.progress_bar(timesteps)` for progress tracking
### Misc

**Dealing with new dependencies:**

Don't arbitrarily add new dependencies even if the reference code has it. Always try to implement operations with pure PyTorch first. For example, anything implemented with `einops` can be implemented with just PyTorch.

If you need to rely on a particularly dependency its import should be guarded:

`if is_my_dependency_available(): import my_dependency`

---

## Modular Pipeline Conversion

### When to use

Modular pipelines break a monolithic `__call__` into composable blocks. Convert when:
- The model supports multiple workflows (T2V, I2V, V2V, etc.)
- Users need to swap guidance strategies (CFG, CFG-Zero*, PAG)
- You want to share blocks across pipeline variants

### File structure

```
src/diffusers/modular_pipelines/<model>/
  __init__.py                          # Lazy imports
  modular_pipeline.py                  # Pipeline class (tiny, mostly config)
  encoders.py                          # Text encoder + image/video VAE encoder blocks
  before_denoise.py                    # Pre-denoise setup blocks
  denoise.py                           # The denoising loop blocks
  decoders.py                          # VAE decode block
  modular_blocks_<model>.py            # Block assembly (AutoBlocks)
```

### Block types decision tree

```
Is this a single operation?
  YES -> ModularPipelineBlocks (leaf block)

Does it run multiple blocks in sequence?
  YES -> SequentialPipelineBlocks
    Does it iterate (e.g. chunk loop)?
      YES -> LoopSequentialPipelineBlocks

Does it choose ONE block based on which input is present?
  Is the selection 1:1 with trigger inputs?
    YES -> AutoPipelineBlocks (simple trigger mapping)
    NO  -> ConditionalPipelineBlocks (custom select_block method)
```

### Build order (easiest first)

1. `decoders.py` -- Takes latents, runs VAE decode, returns images/videos
2. `encoders.py` -- Takes prompt, returns prompt_embeds. Add image/video VAE encoder if needed
3. `before_denoise.py` -- Timesteps, latent prep, noise setup. Each logical operation = one block
4. `denoise.py` -- The hardest. Convert guidance to guider abstraction

### Key pattern: Guider abstraction

Original pipeline has guidance baked in:
```python
for i, t in enumerate(timesteps):
    noise_pred = self.transformer(latents, prompt_embeds, ...)
    if self.do_classifier_free_guidance:
        noise_uncond = self.transformer(latents, negative_prompt_embeds, ...)
        noise_pred = noise_uncond + scale * (noise_pred - noise_uncond)
    latents = self.scheduler.step(noise_pred, t, latents).prev_sample
```

Modular pipeline separates concerns:
```python
guider_inputs = {
    "encoder_hidden_states": (prompt_embeds, negative_prompt_embeds),
}

for i, t in enumerate(timesteps):
    components.guider.set_state(step=i, num_inference_steps=num_steps, timestep=t)
    guider_state = components.guider.prepare_inputs(guider_inputs)

    for batch in guider_state:
        components.guider.prepare_models(components.transformer)
        cond_kwargs = {k: getattr(batch, k) for k in guider_inputs}
        context_name = getattr(batch, components.guider._identifier_key)
        with components.transformer.cache_context(context_name):
            batch.noise_pred = components.transformer(
                hidden_states=latents, timestep=timestep,
                return_dict=False, **cond_kwargs, **shared_kwargs,
            )[0]
        components.guider.cleanup_models(components.transformer)

    noise_pred = components.guider(guider_state)[0]
    latents = components.scheduler.step(noise_pred, t, latents, generator=generator)[0]
```

### Key pattern: Chunk loops for video models

Use `LoopSequentialPipelineBlocks` for outer loop:
```python
class ChunkDenoiseStep(LoopSequentialPipelineBlocks):
    block_classes = [PrepareChunkStep, NoiseGenStep, DenoiseInnerStep, UpdateStep]
```

Note: blocks inside `LoopSequentialPipelineBlocks` receive `(components, block_state, k)` where `k` is the loop iteration index.

### Key pattern: Workflow selection

```python
class AutoDenoise(ConditionalPipelineBlocks):
    block_classes = [V2VDenoiseStep, I2VDenoiseStep, T2VDenoiseStep]
    block_trigger_inputs = ["video_latents", "image_latents"]
    default_block_name = "text2video"
```

### Standard InputParam/OutputParam templates

```python
# Inputs
InputParam.template("prompt")              # str, required
InputParam.template("negative_prompt")     # str, optional
InputParam.template("image")               # PIL.Image, optional
InputParam.template("generator")           # torch.Generator, optional
InputParam.template("num_inference_steps") # int, default=50
InputParam.template("latents")             # torch.Tensor, optional

# Outputs
OutputParam.template("prompt_embeds")
OutputParam.template("negative_prompt_embeds")
OutputParam.template("image_latents")
OutputParam.template("latents")
OutputParam.template("videos")
OutputParam.template("images")
```

### ComponentSpec patterns

```python
# Heavy models - loaded from pretrained
ComponentSpec("transformer", YourTransformerModel)
ComponentSpec("vae", AutoencoderKL)

# Lightweight objects - created inline from config
ComponentSpec(
    "guider", 
    ClassifierFreeGuidance,
    config=FrozenDict({"guidance_scale": 7.5}),
    default_creation_method="from_config"
)
```

### Conversion checklist

- [ ] Read original pipeline's `__call__` end-to-end, map stages
- [ ] Write test scripts (reference + target) with identical seeds
- [ ] Create file structure under `modular_pipelines/<model>/`
- [ ] Write decoder block (simplest)
- [ ] Write encoder blocks (text, image, video)
- [ ] Write before_denoise blocks (timesteps, latent prep, noise)
- [ ] Write denoise block with guider abstraction (hardest)
- [ ] Create pipeline class with `default_blocks_name`
- [ ] Assemble blocks in `modular_blocks_<model>.py`
- [ ] Wire up `__init__.py` with lazy imports
- [ ] Run `make style`
- [ ] Test all workflows for parity with reference

---

## Weight Conversion Tips

<!-- TODO: Add concrete examples as we encounter them. Common patterns to watch for:
  - Fused QKV weights that need splitting into separate Q, K, V
  - Scale/shift ordering differences (reference stores [shift, scale], diffusers expects [scale, shift])
  - Weight transpositions (linear stored as transposed conv, or vice versa)
  - Interleaved head dimensions that need reshaping
  - Bias terms absorbed into different layers
  Add each with a before/after code snippet showing the conversion. -->
