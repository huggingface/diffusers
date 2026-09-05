# Modular pipeline conventions and rules

Shared reference for modular pipeline conventions, patterns, and gotchas.

## Common modular conventions

When adding a new modular pipeline (or reviewing one), skim `src/diffusers/modular_pipelines/qwenimage/`, `src/diffusers/modular_pipelines/flux2/`, `src/diffusers/modular_pipelines/wan/`, and `src/diffusers/modular_pipelines/helios/` first to establish the pattern. Most conventions (file split between `encoders.py` / `before_denoise.py` / `denoise.py` / `decoders.py`, how `expected_components` / `inputs` / `intermediate_outputs` are declared, the denoise-loop wrapping with `LoopSequentialPipelineBlocks`, top-level assembly via `AutoPipelineBlocks` / `SequentialPipelineBlocks` in `modular_blocks_<model>.py`, the `ModularPipeline` subclass shape, the guider-abstracted denoise body, `kwargs_type="denoiser_input_fields"` plumbing) are easiest to internalize by comparison rather than from a fixed list.

## Running a modular pipeline

This section provides guidance on how to execute pipelines and blocks — in scripts, debugging sessions, and tests alike.

- **Full pipeline from a repo**: `ModularPipeline.from_pretrained(repo_id)` — the base class, not the model subclass; it resolves the right class from the repo's `modular_model_index.json` (falling back to a standard `model_index.json`). Then `pipe.load_components()` and call it. New modular repositories should include `modular_model_index.json` because it records modular block and component metadata. `model_index.json` remains supported for compatibility with standard repositories, but it cannot express all modular metadata.
- **A single block or sub-workflow**: convert it to a pipeline first with `init_pipeline()`. Blocks are never executed directly.

```python
# one block
pipe = MyTextEncoderStep().init_pipeline("some-org/tiny-model")  # repo optional if the block needs no pretrained components
pipe.load_components()                                           # init_pipeline only wires specs; this materializes components

# a chain of blocks
blocks = SequentialPipelineBlocks.from_blocks_dict({"vision": VisionStep(), "sound": SoundStep()})
pipe = blocks.init_pipeline()

# run it: declared InputParams are call kwargs; `output=` selects what comes back
ids = pipe(prompt="a robot", output="cond_input_ids")   # one value (any declared output/intermediate)
state = pipe(prompt="a robot")                          # or the full state — read values with state.get("name")
```

- **Swap components and config values** with `pipe.update_components(scheduler=new_scheduler, my_config_flag=False)` — it handles both, keeping the specs and the saved `modular_model_index.json` in sync. Read config via `pipe.config.<name>` (direct attribute access is deprecated).
- **Don't call a block directly** (`block(components, state)`) and don't hand-build a `PipelineState` to feed it. That is the executor's internal protocol — it only *appears* to work for blocks that never touch `components`, and breaks the moment the block gains a component or config dependency. If you find yourself constructing a `PipelineState`, you want `init_pipeline()` and a normal call instead.

## File structure

```
src/diffusers/modular_pipelines/<model>/
  __init__.py                          # Lazy imports
  modular_pipeline.py                  # Pipeline class (tiny, mostly config)
  encoders.py                          # Text encoder + image/video VAE encoder blocks
  before_denoise.py                    # Pre-denoise setup blocks (timesteps, latent prep, noise)
  denoise.py                           # The denoising loop blocks
  decoders.py                          # VAE decode block
  modular_blocks_<model>.py            # Blocksets (AutoBlocks)
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

Is it a different CHECKPOINT (distilled / turbo / a variant with its own schedule)?
  YES -> create a separate blockset for the variant, unless it behaves
         literally the same (see Key pattern: Checkpoint variants)
```

## Build order (easiest first)

1. `decoders.py` -- Takes latents, runs VAE decode, returns images/videos
2. `encoders.py` -- Takes prompt, returns prompt_embeds. Add image/video VAE encoder if needed
3. `before_denoise.py` -- Timesteps, latent prep, noise setup. Each logical operation = one block
4. `denoise.py` -- The hardest. Convert guidance to guider abstraction

## Growing a pipeline: one workflow at a time

Build one workflow end-to-end first (e.g. t2v), then add the next workflow — and later the next blockset / checkpoint variant — one at a time. Each addition should **introduce new blocks rather than modify existing ones**: existing blocks are already wired into working workflows, and a new leaf block plus a new blockset entry can't break them. See `flux2/` for the shape: `modular_blocks_flux2.py` builds the base workflows, and `modular_blocks_flux2_klein.py` adds the klein (distilled) variant as new blocksets composing the same leaf blocks, with new block classes only for the steps that differ.

The one good reason to touch an existing block is to make it strictly more **general**. Be honest about which direction the edit goes:

- **Adding a branch is not generalizing — it's specializing.** `if components.config.foo:` or `if block_state.image is not None:` inside an existing block means the block now does two things. Add a new block for the new case instead, and let workflow selection or a variant blockset pick between them.
- **Collapsing duplicates is generalizing.** If two blocks are identical except for which conditioning inputs they pass to the denoiser, don't keep both — rework the one block to take `kwargs_type="denoiser_input_fields"` (see the `kwargs_type` pattern below) so the same block serves every workflow, as the Cosmos3 denoise step does.

Rule of thumb: a generalizing edit *removes* an if/else or a duplicate block. If your edit *adds* an if/else, it's a new block trying to get out.

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

## Key pattern: `kwargs_type` inputs (`denoiser_input_fields`)

The conditioning inputs a denoiser needs often vary by workflow — especially for omni models like Cosmos3, where the action workflow requires additional action conditioning, and a workflow that generates sound along with video requires additional sound inputs. Tag these outputs with `kwargs_type="denoiser_input_fields"` when they are written; the denoiser then declares a single input with that `kwargs_type` and receives every tagged value collected into one dict. This avoids creating a new denoiser block for each workflow just to list its specific inputs:

```python
# producer side: standard conditioning outputs already carry the tag via their templates
OutputParam.template("prompt_embeds")  # kwargs_type="denoiser_input_fields"
# workflow-specific fields declare it explicitly
OutputParam(
    "action_embeds",
    kwargs_type="denoiser_input_fields",
    type_hint=torch.Tensor,
    description="Action conditioning fed into the transformer.",
)

# consumer side (the loop denoiser): declare the kwargs_type input once
InputParam.template("denoiser_input_fields")

# inside the denoiser __call__: every tagged value arrives in one dict —
# and also individually (block_state.prompt_embeds, block_state.action_embeds, ...)
block_state.denoiser_input_fields  # {"prompt_embeds": ..., "action_embeds": ...}
```

The denoiser typically filters this dict against the transformer's forward signature and forwards the matches — so a new block can add conditioning just by tagging its output (no change to the denoiser), and tagged fields the transformer doesn't accept are silently ignored (see `qwenimage/denoise.py` or `helios/denoise.py`; `z_image/denoise.py` is a minimal consumer).

How the tagging works (behavior is pinned down in `tests/modular_pipelines/test_modular_pipelines_custom_blocks.py::TestBlockKwargsTypeInputs`):

- A value gets its tag when it is **written** to pipeline state: a block output is tagged if declared with `OutputParam(..., kwargs_type=...)`; a user-passed input is tagged if the pipeline-level `InputParam` it matches declares a kwargs_type.
- Users can always pass all the tagged values as a dict under the kwargs_type name — `pipe(denoiser_input_fields={"prompt_embeds": ...})` — and every entry gets tagged. In a full pipeline this is rarely needed: named inputs and tagged block outputs get tagged on their own; the dict form matters mainly for standalone runs (below).
- **Gotcha — standalone runs:** a named input declared *without* the kwargs_type lands in state by name but never gets tagged, so it never reaches the consumer's dict. So when a denoise block runs standalone (without the upstream blocks whose tagged outputs normally supply these values), passing them as plain named inputs silently does nothing — they must go through the `denoiser_input_fields={...}` dict, or the block must declare them as named `InputParam(..., kwargs_type="denoiser_input_fields")` inputs.

## Key pattern: Workflow selection

```python
class AutoDenoise(ConditionalPipelineBlocks):
    block_classes = [V2VDenoiseStep, I2VDenoiseStep, T2VDenoiseStep]
    block_trigger_inputs = ["video_latents", "image_latents"]
    default_block_name = "text2video"
```

## Key pattern: Checkpoint variants

A different checkpoint (distilled / turbo / a variant with its own schedule) can have its own blockset mapped to it: give the variant a `ModularPipeline` subclass carrying its `default_blocks_name`, and checkpoints route to it automatically — via `_class_name` in `modular_model_index.json`, or, for repos that only ship a standard `model_index.json`, a config-keyed map fn in `MODULAR_PIPELINE_MAPPING` (see `_flux2_klein_map_fn`).

A variant blockset also declares its **own `model_name`** (every block class carries one), with its own `_create_default_map_fn` entry in `MODULAR_PIPELINE_MAPPING` — e.g. `wan-animate-2` / `wan-animate-2-distilled`. Sharing the base's name means `blocks.init_pipeline()` resolves the *base* pipeline class (the map fn gets no config on that path) and `save_pretrained` then round-trips the wrong `_class_name` — see huggingface/diffusers#14451.

Default to taking that option. The only reason not to split is when the variant behaves literally the same. If the split buys anything at all — the distilled variant doesn't have to declare `negative_prompt`, doesn't carry a guider, and its docs describe exactly what the checkpoint does — make the separate blockset. It costs almost nothing: blocksets compose the same shared leaf blocks, and only the steps that truly differ need new block classes. See `modular_blocks_flux2_klein.py`, which reuses the base flux2 leaf blocks and swaps in just a `negative_prompt`-free text encoder and a guider-free denoise step.

Don't fall back to the standard-pipeline habit of a config flag branching inside a shared block (`ConfigSpec(name="is_distilled")` + `if components.config.is_distilled:`). That keeps both variants' behavior bundled in one blockset — and the input surface is the one thing it can never fix: a repo can override components and config values per checkpoint, but never which inputs the blocks declare, so the distilled checkpoint would still accept `negative_prompt` and silently ignore it.

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

## Key pattern: Flat blocksets

Prefer flat sequences over nested compositions. Put the `Auto` / `Conditional` selection at the top level and make each workflow variant a flat `InsertableDict` of leaf blocks. Try not to nest `AutoPipelineBlocks` inside `SequentialPipelineBlocks` inside `AutoPipelineBlocks` — debugging which workflow was selected, and which block inside which sub-block touched which state, becomes painful. See `flux2/modular_blocks_flux2_klein.py` for the canonical shape.

The default blockset's top-level children are exactly the steps worth running standalone — `text_encoder` / `image_encoder` / `vae_encoder` / `denoise` / `decode` — each poppable and usable on its own. Multi-step children (a preprocess + encode pair, a prepare + loop pair) are assembled as a module-level `InsertableDict` plus a `SequentialPipelineBlocks` reading it:

```python
MyImageEncoderBlocks = InsertableDict([("preprocess", MyProcessImagesInputStep()), ("encode", MyImageClipEncoderStep())])

# auto_docstring
class MyImageEncodeStep(SequentialPipelineBlocks):
    model_name = "my-model"
    block_classes = MyImageEncoderBlocks.values()
    block_names = MyImageEncoderBlocks.keys()
```

Preset files never import from each other: each `modular_blocks_*.py` self-assembles its groups from the leaf files (`encoders.py`, `denoise.py`, ...), even when a group is identical to the sibling preset's — see `modular_blocks_wan_animate_2_distilled.py`, `modular_blocks_flux2_klein.py`.

## InputParam / OutputParam

Use `.template("<name>")` for params with a canonical meaning (`prompt`, `negative_prompt`, `image`, `generator`, `num_inference_steps`, `latents`, `prompt_embeds`, `images`, `videos`, etc.) — the template carries a vetted description and type hint. The full registry lives in [`src/diffusers/modular_pipelines/modular_pipeline_utils.py`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/modular_pipelines/modular_pipeline_utils.py) (`INPUT_PARAM_TEMPLATES`, `OUTPUT_PARAM_TEMPLATES`); read that file rather than relying on a hardcoded list here, since names get added.

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

**Declare defaults in the `InputParam`, not inside `__call__`.**

```python
# yes
InputParam(name="num_frames", type_hint=int, default=189)

# no — works, but the assembled pipeline is not aware of it
if block_state.num_frames is None:
    block_state.num_frames = 189
```

A declared default is part of the block's contract, so the assembled pipeline is aware of it: the generated docstring shows it and `default_call_parameters` reports it. Resolved inside the body instead, the input renders as `*optional*` with no default, and nothing at the pipeline level can report what the block will actually do. Don't worry about branches of a conditional blockset declaring different defaults for the same input — each branch resolves its own at runtime. Resolve inside `__call__` only when the default is *computed* — derived from other inputs or component config (`height = components.default_sample_size * components.vae_scale_factor`). And when several blocks in a sequence share an input, declare the same default on each (or only on the first block that reads it): in a sequence the input is one shared value, so disagreeing declarations are silently resolved first-block-wins.

**A composed `SequentialPipelineBlocks` can override `inputs` / `outputs`.** Two uses: narrow `outputs` to what downstream actually consumes, so the docstring shows the step's product instead of every internal intermediate (Wan-Animate-2's core denoise exposes only `segment_frames`); and change one input default per preset by mapping over `super().inputs` (the distilled core denoise turns `num_inference_steps` into `default=10`):

```python
@property
def inputs(self):
    # The distilled checkpoint samples in few steps.
    return [
        InputParam.template("num_inference_steps", default=10) if param.name == "num_inference_steps" else param
        for param in super().inputs
    ]
```

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

6. **Test model paths pointing to contributor repos.** Tiny test models ultimately live under `hf-internal-testing/`, not personal repos like `username/tiny-model`. Developing against a personal repo is fine and not merge-blocking — a maintainer moves the model (before or after merge) and updates the path.

7. **Respect the declared IO system.** Components in `expected_components`, fields in `inputs` / `intermediate_outputs` — once declared, the modular framework guarantees them. So:
    - **Don't read defensively.** Declared components are always set as attributes (possibly `None`); declared upstream outputs are always populated in `block_state` after the upstream block runs. `getattr(components, "vae", None)`, `hasattr(self, "vae")`, `getattr(block_state, "prompt_embeds", None)` are dead code that hides typos. Use `components.vae` / `block_state.prompt_embeds` directly. Check `is not None` only when nullability is meaningful (a component the user might not have loaded).
    - **Don't write undeclared.** If a block sets `block_state.foo = ...`, declare `OutputParam("foo", ...)` in `intermediate_outputs`. The declarations are the public contract — undeclared writes can't be wired to downstream blocks.
    - **Don't call `state.set()` directly inside a block.** Write to state only through declared `intermediate_outputs` via `self.get_block_state(state)` / `self.set_block_state(state, block_state)`. A direct `state.set("foo", value)` bypasses the block's interface entirely — the field never appears as a declared output, so downstream blocks can't see it through the normal wiring and the framework can't generate docs / validate types for it.

8. **No-op skip logic inside an optional block.** If a step is conditional (e.g. an optional prompt enhancer), don't have the block check a flag at the top of `__call__` and `return` early. Wrap it in an `AutoPipelineBlocks` with `block_trigger_inputs = ["use_xxx"]` so the block is only assembled into the pipeline when the trigger input is provided. The block's own `__call__` should always assume its components and inputs are present.

9. **Serving a checkpoint variant through a config flag in a shared block.** `ConfigSpec(name="is_distilled")` plus `if components.config.is_distilled:` bundles two checkpoints' behavior into one blockset — and it can't change the input surface at all (the distilled variant would still accept `negative_prompt`). Suggest a separate blockset for the variant instead (see Key pattern: Checkpoint variants).

10. **Declaring a pretrained model component just to read a config value from it.** Everything in `expected_components` gets loaded, so an encoder or decoder block should not declare the `transformer` just to read its patch size, and a denoise block should not declare the `vae` just to read its compression ratio: a block run on its own would then load a model it never calls. Put such values on the `ModularPipeline` subclass as a property that reads the component when it is loaded and falls back to a constant otherwise (`vae_spatial_compression_ratio`, `latents_mean` in `ltx2/modular_pipeline.py`); the fallback lets a block run on its own, and the loaded component wins whenever it is there.

11. **Raw `torch.randn(device=...)` for noise.** Use `randn_tensor(...)` from `utils/torch_utils`: it draws on the generator's device and moves the result, so CPU generators (what the test mixins pass) work, and the CUDA-generator path is bit-identical to `torch.randn`.

12. **Latent form drifting across block boundaries.** Two transforms sit between a VAE and a transformer: *normalizing* (the VAE's latent statistics / `scaling_factor`) and *packing* (`[B, C, F, H, W]` → a token sequence `[B, S, D]`). Whoever applies one must have a mirror block that undoes it, and a core denoise group must hand back `latents` in the same form it took them -- otherwise latents get normalized twice, unpacked against the wrong geometry, or reach a decoder in a form it cannot read. The convention across the modular pipelines:
    - **Normalize / denormalize live on the VAE blocks.** The VAE encoder emits normalized `image_latents`; the decoder denormalizes right before `vae.decode`. Nothing inside the denoise group applies or removes latent statistics -- that block would need the VAE's stats (gotcha 10) and the encoder's output would no longer be usable as-is.
    - **Pack / unpack live inside the core denoise group.** The prepare-latents / input step packs, and a dedicated after-denoise step at the end of the group unpacks (`QwenImageAfterDenoiseStep`, `Flux2UnpackLatentsStep`, `MiniMaxH3AfterDenoiseStep`, `LTX2UnpackLatentsStep`). The group's `latents` output is then the VAE form its input had, so decoders, upsamplers and a second denoise pass take it as-is and need no `height` / `width` / `num_frames` just to unpack. Unpacking in the decoder instead (`flux`, `krea2`, `ltx`) leaves packed latents in state that no other block can consume and makes the decoder carry geometry inputs it does not otherwise need.
    - If the transformer patchifies internally (SD3, SDXL, Wan, HunyuanVideo, Helios, Cosmos, Anima, Z-Image), don't pack at block level at all.
    - Say which form a tensor is in wherever it crosses a boundary: `"packed, normalized [B, S, D]"` / `"[B, C, F, H, W], normalized"` in the `InputParam` / `OutputParam` descriptions.

    Known reasons to deviate, worth a comment where they apply: the statistics are stored over the *packed* channels, so denormalizing has to happen before unpacking or tile the stats (`ernie_image`, `ideogram4`); unpacking needs per-token ids rather than `height` / `width` (`flux2`, whose unpack step consumes `latent_ids`); or a later step conditions on decoded *pixels*, so decode has to run inside the loop (`wan_animate_2`, Cosmos transfer).

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
- [ ] `--fix_and_overwrite` regenerates **every** modular family — revert the files outside your model's folder before committing (careful with path filters: `grep -v pipelines/wan/` also matches `modular_pipelines/wan/`)
- [ ] `python utils/check_forward_call_docstrings.py` must pass (CI gates it): every `forward` / `__call__` parameter needs its own docstring entry (no `a (…), b (…):` fused entries) and a `Returns:` section
- [ ] Run `make style` and `make quality`
- [ ] Test all workflows for parity with reference
