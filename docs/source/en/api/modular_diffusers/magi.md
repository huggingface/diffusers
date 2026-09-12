# MAGI-1 base generation

`MagiTextToVideoBlocks` connects T5 text encoding, official HQ/duration conditioning, chunk denoising, and chunk-wise
VAE decoding. Separate `MagiImageToVideoBlocks` and `MagiVideoToVideoBlocks` support image/video prefixes with the
same base checkpoint. Distilled sampling is not supported by these workflows.

## Convert and load

Convert the official base checkpoint directly into a self-contained pipeline with one command:

```bash
python scripts/convert_magi_to_diffusers.py pipeline \
    --transformer_path /path/to/ckpt/magi/4.5B_base/inference_weight \
    --config_path /path/to/MAGI-1/example/4.5B/4.5B_base_config.json \
    --vae_path /path/to/ckpt/vae \
    --t5_path /path/to/t5-v1_1-xxl \
    --special_tokens_path /path/to/MAGI-1/example/assets/special_tokens.npz \
    --output_path /path/to/magi-base-pipeline
```

The converter saves T5, the tokenizer, learned null-caption features, and the official HQ/duration vectors with the
models. Runtime inference does not import the official repository or read its special-token archive.
The script converts sharded T5 `.bin` checkpoints with `torch.load(weights_only=True)` into temporary safetensors
before loading them. Only convert trusted checkpoints. The destination must be empty. After saving, the script
reloads all model components and verifies their tensor values exactly. Use the `transformer` or `vae` subcommand
to convert those components separately.

```python
import torch
from diffusers import MagiModularPipeline
from diffusers.utils import export_to_video

pipeline = MagiModularPipeline.from_pretrained("/path/to/magi-base-pipeline")
pipeline.load_components(dtype={
    "default": torch.float32,
    "transformer": torch.bfloat16,
    "vae": torch.bfloat16,
})
pipeline.to("cuda")
pipeline.vae.enable_tiling(tile_sample_min_length=12)

videos = pipeline(
    prompt="A cat walks through a sunlit garden.",
    height=512,
    width=512,
    num_frames=96,
    num_inference_steps=64,
    generator=torch.Generator("cpu").manual_seed(42),
    output_type="np",
    output="videos",
)
export_to_video(videos[0], "magi.mp4", fps=24)
```

This example requires enough memory for all components. For smaller devices, arrange component offloading before
inference. Keep T5, text-conditioning features, initial noise, and Euler state in FP32. Load the Transformer with
the desired dtype instead of casting the entire pipeline: some Transformer weights must remain FP32.
The decoder uses autocast when the VAE runs in FP16 or BF16, matching the reference decode precision context.
For official CUDA BF16 comparisons, select `pipeline.transformer.set_attention_backend("flash_varlen")` and
`pipeline.vae.set_attention_backend("flash")`; use the same seed and generator device as the reference.

Prompt cleaning requires `ftfy` and `beautifulsoup4`. The default follows the official two-pass cleaning, including
its removal of CJK characters. Set `clean_caption=False` to use only lowercasing and whitespace trimming.
T5 processes each prompt separately and pads to 800 tokens. Each chunk receives
`[duration, HQ, T5 tokens...]`, truncated back to 800 positions. Duration counts remaining chunks and saturates at
eight. Unconditional features come from the learned null caption, with the first 50 tokens unmasked; no negative
prompt is encoded.

Height and width must be compatible with both the VAE compression ratio and Transformer patch size.
`num_frames` must be divisible by the temporal compression ratio; generation rounds up to complete latent chunks.
The defaults are six latent frames per chunk and four active chunks per window. `num_images_per_prompt`
controls how many videos are generated per prompt.

The top-level blocks are `text_encoder`, `prepare_latents`, `denoise`, and `decode`. They can be initialized
independently with `block.init_pipeline(checkpoint_path)` and `load_components()`.
The decoder divides latents by 0.18215 and decodes each generated chunk independently. Enable VAE tiling before
running large videos. Outputs follow Diffusers conventions: `pt` is normalized float video with shape
`(batch, frames, channels, height, width)`, `np` uses channels last, `pil` is a list of frame lists, and
`latent` bypasses decoding. Standard PIL conversion is not the original MAGI uint8 truncation path; compressed
video bytes are not a parity target.

## Prepared-latent denoising

`MagiDenoiseStep` generates latents for MAGI-1 base models. It accepts prepared text features and initial noise;
it does not encode prompts, encode prefix videos, or decode generated latents. Distilled models and prefixes that
do not contain a whole number of latent chunks are not supported by this block.

### Usage

Load a converted `MagiTransformer3DModel` checkpoint and provide it alongside the scheduler and guider:

```python
import torch
from diffusers import MagiClassifierFreeGuidance, MagiDenoiseStep, MagiEulerScheduler, MagiTransformer3DModel

transformer = MagiTransformer3DModel.from_pretrained(transformer_path, torch_dtype=torch.bfloat16).to("cuda")
pipeline = MagiDenoiseStep().init_pipeline()
pipeline.update_components(
    transformer=transformer,
    scheduler=MagiEulerScheduler(),
    guider=MagiClassifierFreeGuidance(),
)
result = pipeline(
    latents=initial_latents,
    prompt_embeds=prompt_embeds,
    prompt_attention_mask=prompt_attention_mask,
    negative_prompt_embeds=null_embeds,
    negative_prompt_attention_mask=null_mask,
    num_inference_steps=64,
    chunk_width=6,
    window_size=4,
    output=["latents", "clean_kv_cache", "completed_chunks"],
)
```

The example assumes that the input tensors are already on the Transformer's execution device. Keep initial noise
and sampling state in FP32. The Transformer handles its internal mixed precision; do not cast the whole model
with `.bfloat16()`, which would also cast weights that need FP32.

The official mixed-precision parity checks use the `flash_varlen` attention backend. If Flash Attention is
installed, select it with `transformer.set_attention_backend("flash_varlen")` before running the pipeline.
The default native backend remains available, but its BF16 arithmetic is not guaranteed to match the official
Flash Attention path bit for bit.

`initial_latents` has shape `(batch, channels, chunks * chunk_width, height, width)`. Conditional text features have
shape `(batch, length, caption_channels)` or `(batch, chunks, length, caption_channels)`, with a matching boolean
keep-mask. The latter form supports chunk-dependent text features prepared by the caller.

The negative inputs are the model's **learned null-caption features**, not an empty prompt passed through a text
encoder. They have shape `(batch, length, caption_channels)` and `(batch, length)` and are shared across chunks.
Use the same padded text length for conditional and null features. The official text preparation uses 800 padded
tokens and keeps the first 50 tokens for the null caption. Text preprocessing and special tokens must be prepared
separately; this block does not reconstruct them.

### Window and guidance

`num_inference_steps` counts Euler updates per generated chunk. It must be divisible by both `window_size` and the
length of `noise2clean_kvrange`, whose default is `(5, 4, 3, 2)`. Each entry is a positive attention-window size in
chunks, including the current chunk. The active generation window first expands, then moves forward, then shrinks.
For `K` total chunks, `P` prefix chunks, `W` window size, and `N` updates per chunk, there are
`(N // W) * (K + W - 1 - P)` window iterations.

The guider combines three velocity predictions at each active chunk's own timestep:

```python
velocity = (1 - prefix_scale) * independent_velocity
velocity += (prefix_scale - text_scale) * prefix_velocity + text_scale * text_and_prefix_velocity
```

The independent branch treats each active chunk as a separate batch item, with no cached prefix and temporal
positions starting at zero. Configure thresholds and scales through `MagiClassifierFreeGuidance`, not through
pipeline inputs. Guidance is applied before the FP32 Euler update.

### Clean-prefix cache

Optionally pass `prefix_latents` with shape `(batch, channels, prefix_chunks * chunk_width, height, width)`.
These replace the corresponding leading slots of `initial_latents` and remain unchanged. At least one chunk must
remain to generate. Caller-owned latent tensors are not modified.

Initial prefix KV is computed with null-caption conditioning at `clean_t=0.9999`. When a generated chunk leaves
the active window, the next iteration recomputes it at `clean_t`. Both prefix-conditioned branches read the same
previous cache; only the null-text branch supplies newly reusable KV. Each layer copies only the requested clean
prefix before processing the next layer, so full noisy-window caches do not accumulate across the model.
`clean_chunk_kvrange` defaults to 1 and must be positive.

Set `cache_device="cpu"` on a pipeline call to keep clean KV on CPU. Each Transformer layer transfers only its own
input cache to the compute device and returns its updated clean slice to CPU. This changes storage, not attention
ranges or sampling math. The example maps the official engine config's `kv_offload` setting to this argument.
Component CPU offload and clean-KV offload are separate controls; long videos may need both.

`clean_kv_cache` contains per-layer `(key, value)` tensors with shape `(batch, tokens, kv_heads, head_dim)`. It keeps
the complete clean prefix to preserve absolute temporal positions; there is no cache eviction. The final generated
chunk is not refreshed because no later chunk needs it, so the returned cache covers all but the last chunk (or is
`None` for a one-chunk video). It is an inspection output, not a resumable generation checkpoint.

Every fresh pipeline call resets the schedule and prefix cache. `completed_chunks` includes supplied prefix chunks
and all finalized generated chunks. Only `result["latents"]` should be passed to later VAE decoding, with the
appropriate MAGI latent scaling applied separately.

## MagiDenoiseStep

[[autodoc]] MagiDenoiseStep

## MagiClassifierFreeGuidance

[[autodoc]] MagiClassifierFreeGuidance


## Image and video prefixes

The same base checkpoint can run image- or video-conditioned continuation with `MagiImageToVideoBlocks` or
`MagiVideoToVideoBlocks`. Select the blockset explicitly; loading the existing checkpoint normally still selects
text-to-video.

```python
from diffusers import MagiImageToVideoBlocks

pipe = MagiImageToVideoBlocks().init_pipeline("/path/to/MAGI-1-diffusers")
pipe.load_components()
pipe.vae.enable_tiling(tile_sample_min_length=12)
# Configure precision, attention backends, and device/offload as in the example script.
videos = pipe(
    prompt="Good Boy",
    image=image_tensor,
    height=256,
    width=256,
    num_frames=24,
    output_type="pt",
    output="videos",
)
```

The image input is pre-resized RGB `torch.uint8` with shape `(batch, 3, height, width)`. The video blockset instead
accepts `video` with shape `(batch, 3, frames, height, width)`. File decoding and fps resampling are outside the
core blocks. The example script `examples/magi/inference_magi.py` handles these with FFmpeg; `--image` selects I2V
and `--video` selects V2V. The video loader uses the first 32 frames after fps resampling, as in the official loader.

The VAE encoder emits deterministic, scaled `conditioning_latents`. Complete prefix chunks initialize the clean
cache. Partial-prefix values are injected before every evaluation, including clean-cache refresh, but only active
chunks are updated by Euler integration. Finalized output is not overwritten by a later cache refresh.

For these workflows, `num_frames` requests new frames. Prefix plus new frames rounds up to full latent chunks.
Prefix latents are removed before each output chunk is decoded, except for the first chunk of a one-latent-frame
image prefix. The official decoder returns one frame for a tile containing only one latent position; the prefix
decoder preserves that convention. Actual decoded length therefore depends on both chunk rounding and VAE tiling.
At chunk width 6 with 12-frame VAE tiles, an image plus 24 requested new frames produces 48 frames; a 32-frame video
prefix plus 24 requested new frames produces 37 frames.

The `latents` intermediate retains prefix slots. `output_type="latent", output="videos"` returns the cropped latent
suffix, with the image-prefix exception above. The `vae_encoder`, `prepare_latents`, `denoise`, and `decode` blocks
are independently reusable through `init_pipeline()`. The standalone `MagiDenoiseStep` described above still accepts
only complete prefix chunks; use the new workflow's `denoise` block for partial prefixes.

## MagiImageToVideoBlocks

[[autodoc]] MagiImageToVideoBlocks

## MagiVideoToVideoBlocks

[[autodoc]] MagiVideoToVideoBlocks
