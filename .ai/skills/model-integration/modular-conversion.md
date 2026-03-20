# Modular Pipeline Conversion Reference

## When to use

Modular pipelines break a monolithic `__call__` into composable blocks. Convert when:
- The model supports multiple workflows (T2V, I2V, V2V, etc.)
- Users need to swap guidance strategies (CFG, CFG-Zero*, PAG)
- You want to share blocks across pipeline variants

## File structure

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

## Key pattern: Chunk loops for video models

Use `LoopSequentialPipelineBlocks` for outer loop:
```python
class ChunkDenoiseStep(LoopSequentialPipelineBlocks):
    block_classes = [PrepareChunkStep, NoiseGenStep, DenoiseInnerStep, UpdateStep]
```

Note: blocks inside `LoopSequentialPipelineBlocks` receive `(components, block_state, k)` where `k` is the loop iteration index.

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
- [ ] Run `make style` and `make quality`
- [ ] Test all workflows for parity with reference
