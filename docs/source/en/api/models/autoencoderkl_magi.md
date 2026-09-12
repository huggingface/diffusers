# AutoencoderKLMagi

MAGI-1 uses a transformer VAE with 16 latent channels, 8x spatial compression, and 4x temporal compression. The encoder
and decoder each contain 24 transformer blocks. Both use full attention within the input tile, learned position
embeddings, and the reference normalization formula `(x - mean) / (std + eps)` on Q, K, and V.

The implementation retains the official fused `qkv`, `proj`, `mlp.fc1`, `mlp.fc2`, and `patch_embed.proj` layers. CLS and
position parameters are grouped in a small module so device placement and offloading hooks move them together with
their inputs. The conversion script maps these four parameter names and translates the configuration; it does not
split or reshape the checkpoint tensors. The encoder preserves the reference's channel-last storage layout. After
tiling, posterior means are packed separately from log-variance storage, as in the reference's mean-only concatenation;
this avoids changing the reduced-precision decoder's matrix multiplication path.

## Convert and load

Convert the official `ckpt/vae` directory before loading it:

```bash
python scripts/convert_magi_to_diffusers.py vae --checkpoint_path /path/to/ckpt/vae --output_path /path/to/magi-vae-diffusers
```

The script loads all weights strictly and verifies that saving and reloading preserves every tensor.

```python
import torch
from diffusers import AutoencoderKLMagi

vae = AutoencoderKLMagi.from_pretrained("/path/to/magi-vae-diffusers", torch_dtype=torch.bfloat16).to("cuda")
vae.set_attention_backend("flash")

# video has shape (batch, 3, frames, height, width) and values normalized to [-1, 1].
video = video.to(device="cuda", dtype=torch.bfloat16)
with torch.no_grad():
    latents = vae.encode(video).latent_dist.mode()
    reconstruction = vae.decode(latents, num_frames=video.shape[2]).sample
```

QKV is always fused, as in the reference. The `flash` backend requires FlashAttention and matches the official attention
kernel. The default backend works without this optional dependency; reduced-precision results can differ between
kernels. Latents here are raw VAE latents. The diffusion pipeline's latent scaling is applied outside this model.

## Frame counts and sampling

Spatial dimensions must be divisible by the spatial patch size. Without tiling, frame counts must be divisible by the
temporal patch length, except that a single input frame is repeated to fill one patch. With temporal tiling, the final
tile may also contain exactly one frame. Other incomplete temporal patches are rejected instead of silently dropping
frames.

`encode` returns a posterior distribution. Use `.mode()` to match the official inference pipeline, which patches the
original VAE to encode deterministically. `.sample(generator=...)` follows Diffusers' generator and dtype conventions.
The original standalone VAE samples CPU float32 noise, so its stochastic outputs are not guaranteed to match for the
same seed. The deterministic mode is the reference inference path.

By default, `decode` follows the original VAE convention: a tile containing one latent time position returns only the
first decoded frame. This also applies to the last tile of a longer video. For example, a four-frame video becomes one
latent time position and decodes to one frame by default. Pass `num_frames=4` to retain all four frames. The supplied
length must fit the latent patch count. `vae(video).sample` automatically supplies the original frame count.

## Temporal and spatial tiling

Enable single-device tiling to limit the attention sequence length:

```python
vae.enable_tiling(
    tile_sample_min_length=12,
    tile_sample_min_height=256,
    tile_sample_min_width=256,
    temporal_tile_overlap_factor=0.0,
    spatial_tile_overlap_factor=0.25,
)
with torch.no_grad():
    latents = vae.encode(video).latent_dist.mode()
    reconstruction = vae.decode(latents, num_frames=video.shape[2]).sample
```

All tile dimensions are expressed in input video pixels or frames. The official video helper uses half the configured
FPS as its temporal tile length and enables spatial tiling; 12 corresponds to a configured FPS of 24. Match these
settings when comparing against the reference. Set `allow_spatial_tiling=False` for temporal tiling only.

Tile dimensions and overlaps must align with the latent grid. Tiles at the boundary may be shorter. The tiler follows
the official frame/height/width iteration and blending order: encoding blends against preceding already-blended tiles,
while decoding blends against the original decoded neighbors. Decoder blending uses FP32 intermediates for low-precision
tensors, matching the accumulation precision of the compiled reference blend operations. In the reference, compiler
fallbacks can instead use low-precision eager arithmetic; numerical comparisons should isolate compiler state between
different tiling configurations.

Tiling changes the attention context, so tiled outputs need not match whole-input outputs. The tiled posterior blends
mean and log-variance channels independently; sampling this distribution is not equivalent to blending independently
sampled tiles. Use `.mode()` for reference inference parity.

Use `vae.enable_slicing()` to process batch items individually, `vae.disable_slicing()` to restore full-batch processing,
and `vae.disable_tiling()` to restore whole-input processing. Distributed tile processing is not implemented.

## AutoencoderKLMagi

[[autodoc]] AutoencoderKLMagi
    - all
