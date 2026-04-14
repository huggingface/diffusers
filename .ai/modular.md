# Modular pipeline conventions and rules

Shared reference for modular pipeline conventions, patterns, and gotchas.

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

## Standard InputParam/OutputParam templates

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

## ComponentSpec patterns

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

## Gotchas

1. **Importing from standard pipelines.** The modular and standard pipeline systems are parallel — modular blocks must not import from `diffusers.pipelines.*`. For shared utility methods (e.g. `_pack_latents`, `retrieve_timesteps`), either redefine as standalone functions or use `# Copied from diffusers.pipelines.<model>...` headers. See `wan/before_denoise.py` and `helios/before_denoise.py` for examples.

2. **Cross-importing between modular pipelines.** Don't import utilities from another model's modular pipeline (e.g. SD3 importing from `qwenimage.inputs`). If a utility is shared, move it to `modular_pipeline_utils.py` or copy it with a `# Copied from` header.

3. **Accepting `guidance_scale` as a pipeline input.** Users configure the guider separately (see [guider docs](https://huggingface.co/docs/diffusers/main/en/api/guiders)). Different guider types have different parameters; forwarding them through the pipeline doesn't scale. Don't manually set `components.guider.guidance_scale = ...` inside blocks. Same applies to computing `do_classifier_free_guidance` — that logic belongs in the guider.

4. **Accepting pre-computed outputs as inputs to skip encoding.** In standard pipelines we accept `prompt_embeds`, `negative_prompt_embeds`, `image_latents`, etc. so users can skip encoding steps. In modular pipelines this is unnecessary — users just pop out the encoder block and run it separately. Encoder blocks should only accept raw inputs (`prompt`, `image`, etc.).

5. **VAE encoding inside prepare-latents.** Image encoding should be its own block in `encoders.py` (e.g. `MyModelVaeEncoderStep`). The prepare-latents block should accept `image_latents`, not raw images. This lets users run encoding standalone. See `WanVaeEncoderStep` for reference.

6. **Instantiating components inline.** If a class like `VideoProcessor` is needed, register it as a `ComponentSpec` and access via `components.video_processor`. Don't create new instances inside block `__call__`.

7. **Deeply nested block structure.** Prefer flat sequences over nesting Auto blocks inside Sequential blocks inside Auto blocks. Put the `Auto` selection at the top level and make each workflow variant a flat `InsertableDict` of leaf blocks. See `flux2/modular_blocks_flux2_klein.py` for the pattern.

8. **Using `InputParam.template()` / `OutputParam.template()` when semantics don't match.** Templates carry predefined descriptions — e.g. the `"latents"` output template means "Denoised latents". Don't use it for initial noisy latents from a prepare-latents step. Use a plain `InputParam(...)` / `OutputParam(...)` with an accurate description instead.

9. **Test model paths pointing to contributor repos.** Tiny test models must live under `hf-internal-testing/`, not personal repos like `username/tiny-model`. Move the model before merge.

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
