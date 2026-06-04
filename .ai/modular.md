# Modular pipeline conventions and rules

Shared reference for modular pipeline conventions, patterns, and gotchas.

## Common modular conventions

When adding a new modular pipeline (or reviewing one), skim `src/diffusers/modular_pipelines/qwenimage/`, `src/diffusers/modular_pipelines/flux2/`, `src/diffusers/modular_pipelines/wan/`, and `src/diffusers/modular_pipelines/helios/` first to establish the pattern. Most conventions (file split between `encoders.py` / `before_denoise.py` / `denoise.py` / `decoders.py`, how `expected_components` / `inputs` / `intermediate_outputs` are declared, the denoise-loop wrapping with `LoopSequentialPipelineBlocks`, top-level assembly via `AutoPipelineBlocks` / `SequentialPipelineBlocks` in `modular_blocks_<model>.py`, the `ModularPipeline` subclass shape, the guider-abstracted denoise body, `kwargs_type="denoiser_input_fields"` plumbing) are easiest to internalize by comparison rather than from a fixed list.

## File structure

```
src/diffusers/modular_pipelines/<model>/
  __init__.py                          # Lazy imports
  modular_pipeline.py                  # Pipeline class (tiny, mostly config)
  encoders.py                          # Text encoder + image/video VAE encoder blocks
  before_denoise.py                    # Pre-denoise setup blocks (timesteps, latent prep, noise)
  denoise.py                           # The denoising loop blocks
  decoders.py                          # VAE decode block
  modular_blocks_<model>.py            # Block assembly (AutoBlocks)
```

## Block types decision tree

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

## Build order (easiest first)

1. `decoders.py` -- Takes latents, runs VAE decode, returns images/videos
2. `encoders.py` -- Takes prompt, returns prompt_embeds. Add image/video VAE encoder if needed
3. `before_denoise.py` -- Timesteps, latent prep, noise setup. Each logical operation = one block
4. `denoise.py` -- The hardest. Convert guidance to guider abstraction

## Key pattern: Guider abstraction

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

## Key pattern: Denoising loop

All models use `LoopSequentialPipelineBlocks` for the denoising loop (iterating over timesteps):
```python
class MyModelDenoiseLoopWrapper(LoopSequentialPipelineBlocks):
    block_classes = [LoopBeforeDenoiser, LoopDenoiser, LoopAfterDenoiser]
```

Autoregressive video models (e.g. Helios) also use it for an outer chunk loop:
```python
class HeliosChunkDenoiseStep(HeliosChunkLoopWrapper):
    block_classes = [
        HeliosChunkHistorySliceStep,
        HeliosChunkNoiseGenStep,
        HeliosChunkSchedulerResetStep,
        HeliosChunkDenoiseInner,
        HeliosChunkUpdateStep,
    ]
```

Note: sub-blocks inside `LoopSequentialPipelineBlocks` receive `(components, block_state, i, t)` for denoise loops or `(components, block_state, k)` for chunk loops.

## Key pattern: Workflow selection

```python
class AutoDenoise(ConditionalPipelineBlocks):
    block_classes = [V2VDenoiseStep, I2VDenoiseStep, T2VDenoiseStep]
    block_trigger_inputs = ["video_latents", "image_latents"]
    default_block_name = "text2video"
```

## Key pattern: Standalone block reusability

One of the core reason a pipeline is split into blocks at all: each block (text encoder, VAE encoder, prepare-latents, denoise, decoder) must be runnable on its own, and its output must be reusable as the input to a different downstream chain.

Concretely:
- The text encoder block returns `prompt_embeds`. A user can run only that block, save the embeddings, and feed them to the denoise loop later — possibly with a different `num_images_per_prompt`, possibly across multiple runs.
- The VAE encoder is its own block in `encoders.py` (e.g. `WanVaeEncoderStep`) returning `image_latents`. The prepare-latents block accepts `image_latents`, not raw images, so users can swap in pre-encoded latents.
- The decoder block accepts denoised latents from any source — directly from the denoise loop, or after an injected step (upscale, latent edit). Don't bundle decoding into the denoise loop.

Two consequences for input plumbing:

1. **Encoder / VAE-encoder blocks accept raw inputs only** (`prompt`, `image`, ...) and emit per-prompt outputs (`prompt_embeds`, `image_latents`). They do **not** bake in `num_images_per_prompt`.
2. **Per-prompt expansion happens in a dedicated input step** inside the core denoise sequence (e.g. `<Model>TextInputStep`). That keeps pre-encoded embeds reusable across runs with different `num_images_per_prompt`. See `qwenimage/before_denoise.py` for the canonical input step.

Standard pipelines accept `prompt_embeds` / `image_latents` as `__call__` inputs so users can skip encoding. In modular pipelines this is unnecessary — users just pop out the encoder block and run it standalone. Don't accept pre-computed encoder outputs as `__call__` inputs of an encoder block.

## Key pattern: Flat block assembly

Prefer flat sequences over nested compositions. Put the `Auto` / `Conditional` selection at the top level and make each workflow variant a flat `InsertableDict` of leaf blocks. Try not to nest `AutoPipelineBlocks` inside `SequentialPipelineBlocks` inside `AutoPipelineBlocks` — debugging which workflow was selected, and which block inside which sub-block touched which state, becomes painful. See `flux2/modular_blocks_flux2_klein.py` for the canonical shape.

## InputParam / OutputParam

Use `.template("<name>")` for params with a canonical meaning (`prompt`, `negative_prompt`, `image`, `generator`, `num_inference_steps`, `latents`, `prompt_embeds`, `images`, `videos`, etc.) — the template carries a vetted description and type hint. The full registry lives in [`src/diffusers/modular_pipelines/modular_pipeline_utils.py`](../src/diffusers/modular_pipelines/modular_pipeline_utils.py) (`INPUT_PARAM_TEMPLATES`, `OUTPUT_PARAM_TEMPLATES`); read that file rather than relying on a hardcoded list here, since names get added.

For params that don't match a template (model-specific names, custom semantics), declare the field directly:

```python
# Inputs
InputParam(
    "text_lens",
    required=True,
    type_hint=torch.Tensor,
    description="Per-prompt text lengths used by the transformer attention mask.",
)

# Outputs
OutputParam(
    "text_bth",
    type_hint=torch.Tensor,
    kwargs_type="denoiser_input_fields",
    description="Padded text hidden states of shape (B, T_max, H) fed into the transformer.",
)
```

If a template's predefined description doesn't fit (e.g. the `"latents"` output template means "Denoised latents", which is wrong for the noisy latents out of a prepare-latents step) — drop the template and declare the field directly with an accurate description. See gotcha #5.

## ComponentSpec patterns

```python
# models (with weights) - loaded from pretrained
ComponentSpec("transformer", YourTransformerModel)
ComponentSpec("vae", AutoencoderKL)

# weightless objects - created inline from config
ComponentSpec(
    "guider",
    ClassifierFreeGuidance,
    config=FrozenDict({"guidance_scale": 7.5}),
    default_creation_method="from_config"
)
```

## Gotchas

1. **Importing from standard pipelines.** The modular and standard pipeline systems are parallel — modular blocks must not import from `diffusers.pipelines.*`. For shared utility methods (e.g. `_pack_latents`, `retrieve_timesteps`), either redefine as standalone functions or use `# Copied from diffusers.pipelines.<model>...` headers. See `wan/before_denoise.py` and `helios/before_denoise.py` for examples.

2. **Cross-importing between modular pipelines.** Don't import utilities from another model's modular pipeline (e.g. SD3 importing from `qwenimage.inputs`). If a utility is shared, move it to `modular_pipeline_utils.py` or copy it with a `# Copied from` header.

3. **Accepting `guidance_scale` as a pipeline input.** Users configure the guider separately (see [guider docs](https://huggingface.co/docs/diffusers/main/en/api/guiders)). Different guider types have different parameters; forwarding them through the pipeline doesn't scale. Don't manually set `components.guider.guidance_scale = ...` inside blocks. Same applies to computing `do_classifier_free_guidance` — that logic belongs in the guider. **Exception:** some pipeline only support distilled checkpoints (e.g. distilled Flux) skip CFG entirely and don't carry a guider — `guidance_scale` is then a real model input, not a guider knob, and accepting it as a pipeline input is fine. If you're reviewing a pipeline that doesn't have a `guider` in `expected_components`, flag it explicitly so the choice is intentional.

4. **Instantiating components inline.** If a class like `VideoProcessor` is needed, register it as a `ComponentSpec` and access via `components.video_processor`. Don't create new instances inside block `__call__`.

5. **Using `InputParam.template()` / `OutputParam.template()` when semantics don't match.** Templates carry predefined descriptions — e.g. the `"latents"` output template means "Denoised latents". Don't use it for initial noisy latents from a prepare-latents step. Use a plain `InputParam(...)` / `OutputParam(...)` with an accurate description instead.

6. **Test model paths pointing to contributor repos.** Tiny test models must live under `hf-internal-testing/`, not personal repos like `username/tiny-model`. Move the model before merge.

7. **Respect the declared IO system.** Components in `expected_components`, fields in `inputs` / `intermediate_outputs` — once declared, the modular framework guarantees them. So:
    - **Don't read defensively.** Declared components are always set as attributes (possibly `None`); declared upstream outputs are always populated in `block_state` after the upstream block runs. `getattr(components, "vae", None)`, `hasattr(self, "vae")`, `getattr(block_state, "prompt_embeds", None)` are dead code that hides typos. Use `components.vae` / `block_state.prompt_embeds` directly. Check `is not None` only when nullability is meaningful (a component the user might not have loaded).
    - **Don't write undeclared.** If a block sets `block_state.foo = ...`, declare `OutputParam("foo", ...)` in `intermediate_outputs`. The declarations are the public contract — undeclared writes can't be wired to downstream blocks.
    - **Don't call `state.set()` directly inside a block.** Write to state only through declared `intermediate_outputs` via `self.get_block_state(state)` / `self.set_block_state(state, block_state)`. A direct `state.set("foo", value)` bypasses the block's interface entirely — the field never appears as a declared output, so downstream blocks can't see it through the normal wiring and the framework can't generate docs / validate types for it.

8. **No-op skip logic inside an optional block.** If a step is conditional (e.g. an optional prompt enhancer), don't have the block check a flag at the top of `__call__` and `return` early. Wrap it in an `AutoPipelineBlocks` with `block_trigger_inputs = ["use_xxx"]` so the block is only assembled into the pipeline when the trigger input is provided. The block's own `__call__` should always assume its components and inputs are present.

## Conversion checklist

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
- [ ] Add `# auto_docstring` above all assembled blocks (SequentialPipelineBlocks, AutoPipelineBlocks, etc.), run `python utils/modular_auto_docstring.py --fix_and_overwrite`, and verify the generated docstrings — all parameters should have proper descriptions with no "TODO" placeholders indicating missing definitions
- [ ] Run `make style` and `make quality`
- [ ] Test all workflows for parity with reference
