# MAGI-1 image and video prefixes

The same converted base checkpoint supports T2V, I2V, and V2V. The default saved workflow remains T2V.
Select `MagiImageToVideoBlocks` or `MagiVideoToVideoBlocks` explicitly to use a prefix.

## Official example inputs

Run from the repository root using the environment described in `README.md`:

```bash
PYTHONPATH=src /path/to/magi-env/bin/python examples/magi/inference_magi.py \
    --model /path/to/MAGI-1-diffusers \
    --config /path/to/MAGI-1/example/4.5B/4.5B_base_config.json \
    --image /path/to/MAGI-1/example/assets/image.jpeg \
    --height 256 --width 256 --num-frames 24 \
    --output /path/to/new-i2v-output --save-latents
```

For V2V, replace `--image ...` with `--video /path/to/input.mp4`. The example uses FFmpeg to stretch frames to
the requested resolution. For video it resamples to the config's fps and uses at most the first 32 frames,
matching the official prefix loader. FFmpeg must be installed. Image and video inputs are mutually exclusive.
These small dimensions are smoke-test settings, not a visual-quality recommendation.

## Python interface

```python
from diffusers import MagiImageToVideoBlocks

pipe = MagiImageToVideoBlocks().init_pipeline("/path/to/MAGI-1-diffusers")
pipe.load_components()
# Configure device, per-component precision, offload, and attention backends as in inference_magi.py.
pipe.vae.enable_tiling(tile_sample_min_length=12)
videos = pipe(
    prompt="Good Boy", image=image_tensor, height=256, width=256,
    num_frames=24, output_type="pt", output="videos",
)
```

The core image encoder takes pre-resized RGB `torch.uint8` pixels shaped `(batch, 3, height, width)`.
The video encoder takes `(batch, 3, frames, height, width)`. It encodes exactly the supplied frames; file
decoding, fps resampling, and the 32-frame limit belong to the example loader. One prefix may be broadcast
over several prompts. Otherwise its batch must match the prompt batch. Spatial dimensions must match
`height` and `width`; video length must satisfy the VAE's temporal patch requirements.

The `vae_encoder` block can run independently and returns `conditioning_latents`. The `prepare_latents`
block expands these per-prompt latents to the generated video batch. The `denoise` block accepts the full
scaled prefix, including a partial final chunk. Complete prefix chunks initialize the clean KV cache;
partial-prefix values are reinjected at every model evaluation. The partial prefix still participates in the
Euler update, as in the official sampler: it is not permanently frozen in the output state.

`num_frames` requests **new** frames. Prefix plus requested frames rounds up to a full latent chunk. V2V
omits prefix latents before decoding each output chunk. A one-latent-frame prefix retains the first four
decoded frames, matching official I2V behavior. Thus, at the default chunk width of 6:

- I2V with 24 requested new frames returns 48 frames.
- V2V with a 32-frame prefix and 24 requested new frames returns 37 frames with 12-frame VAE tiles.

The official VAE emits only one image frame for a final tile containing one latent position. For V2V's
first cropped chunk, this can make the decoded output shorter than four times its latent length. The prefix
decoder preserves this behavior rather than forcing a frame count.

The `latents` intermediate retains prefix slots. `output_type="latent", output="videos"` returns the cropped
output suffix, with the I2V exception above. `--save-latents` saves full latents, conditioning latents, and the
exact input pixels for reference comparisons.
