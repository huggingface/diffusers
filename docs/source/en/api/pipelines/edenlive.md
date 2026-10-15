<!-- Copyright 2026 The VAST-AI Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# EdenLive

[EdenLive](https://github.com/VAST-AI-Research/EdenLive-Official) generates video from a reference image, text, and a
proxy video. It uses a trained DINOv3 encoder and projection to condition a Wan-based diffusion transformer.

[`EdenLivePipeline`] supports both bidirectional generation and successive causal chunks with clean latent history.
Select a released configuration with `profile` when loading the original checkpoint repository. Defaults apply when
`num_inference_steps` and `guidance_scale` are omitted:

| Profile | Sampling | Default steps | Guidance scale | Default height x width |
|---|---|---|---|---|
| `5b-bidir` | UniPC over the full video | 48 | 3 | 704 x 1280 |
| `14b-bidir` | Euler over the full video | 30 | 3 | 480 x 832 |
| `5b-causal` | Stochastic Euler per chunk | 4 | 1 | 704 x 1280 |
| `14b-causal` | Stochastic Euler per chunk | 4 | 1 | 480 x 832 |

The released causal checkpoints use fixed four-step schedules. Changing `num_inference_steps` alone to another
value raises an error; it does not select a new distilled schedule. Guidance greater than one enables the negative
prompt branch.

## Generate a video

```py
import torch
from diffusers import EdenLivePipeline
from diffusers.utils import export_to_video, load_image, load_video

pipe = EdenLivePipeline.from_pretrained(
    "VAST-AI/EdenLive", profile="5b-causal"
).to("cuda")

frames = pipe(
    prompt="A character explores a snowy landscape.",
    image=load_image("reference.png"),
    video=load_video("proxy-24fps.mp4"),
    num_frames=49,
    generator=torch.Generator("cuda").manual_seed(42),
).frames[0]
export_to_video(frames, "edenlive.mp4", fps=24)
```

## Load and save checkpoints

Use `EdenLivePipeline.from_pretrained()` for both formats:

| Format | Required metadata | Profile selection |
|---|---|---|
| Original release | Root `model_index.json` with the EdenLive original-format marker | Pass one of the four complete profile names |
| Saved Diffusers pipeline | `model_index.json` and saved component directories | Omit `profile` to use the saved value |

The original release loader reads the existing checkpoint files without writing converted weights. It fetches only
the files needed for the selected profile and components. The same entry accepts a Hub repository ID or local
directory, plus `revision`, `cache_dir`, `token`, and `local_files_only=True` for cached or offline loading.

The original release's default precision is BF16 for the transformer, VAE, UMT5 text encoder, CLIP encoder, and
LightTAE, with FP32 parameters for the DINO encoder and projection. An explicit `dtype` (or `torch_dtype`) can be
a single dtype or a mapping by component name. Initialized component overrides retain their identity and dtype.
Active `device_map`, `variant`, and quantization options are not supported by the original-format loader. Load on CPU
and then place the pipeline on the desired device, as in the example above.

Save and reload a standard pipeline with:

```py
pipe.save_pretrained("edenlive-saved")
pipe = EdenLivePipeline.from_pretrained("edenlive-saved").to("cuda")
```

The saved configuration retains the full profile and decoder choice. A conflicting explicit profile raises an error.
Older standard snapshots with size-only profiles or the former causal class name are normalized in memory; newly
saved snapshots use `EdenLivePipeline`. The generic `DiffusionPipeline.from_pretrained()` entry is not supported for
original releases.

## Prepare the inputs

The reference image and proxy video should depict the same composition. Supply a full RGB proxy video already
sampled at the desired output frame rate. The Python pipeline does not resample timestamps; the command-line
example below does. The pipeline then selects proxy frames at the VAE temporal stride once.

| Input | Accepted layout and values |
|---|---|
| `image` | RGB PIL image, or floating-point NumPy/tensor image with values in [0, 1] |
| `video` | List of RGB PIL frames; tensor `(T, 3, H, W)` or `(B, T, 3, H, W)`; NumPy `(T, H, W, 3)` or `(B, T, H, W, 3)` with floating-point values in [0, 1] |
| `prompt` | String or list of strings; alternatively provide `prompt_embeds` or a supported temporal prompt format |
| `height`, `width` | Output dimensions, both multiples of 32 for 5B or 16 for 14B |
| `num_frames` | Positive frame count of the form \\(4k+1\\), no greater than the input video length |

Here `B` is the video batch size, `T` the decoded frame count, and `H` and `W` the frame height and width. The reference
image is resized and center-cropped to the output dimensions. Proxy RGB frames are processed at `proxy_image_size`
(480 x 832 by default) with the profile's `proxy_resize_mode` before DINO encoding. The command-line example also
crops and resizes proxy frames to the output aspect ratio before passing them to the pipeline. Omitting `num_frames`
selects the largest valid input prefix; missing frames are not padded or repeated by the pipeline.

Both model sizes use temporal compression by four. A video of \\(F\\) frames has
\\(T_{\mathrm{latent}}=(F-1)/4+1\\) latent frames. The 5B model uses 48 latent channels and spatial compression by 16;
the 14B model uses 16 channels and spatial compression by 8.

## Choose a decoder

Original releases load the matching [`EdenLiveLightTAEDecoder`] by default. Use the full Wan VAE with:

```py
pipe = EdenLivePipeline.from_pretrained(
    "VAST-AI/EdenLive", profile="14b-causal", decoder=None
)
```

Pass `decoder=my_decoder` to use an initialized compatible decoder. The full VAE remains available to encode the
reference image. A custom decoder must implement `decode(latents, return_dict=False)` and return a tuple whose
first item is an RGB tensor in `(B, 3, T, H, W)` layout and `[-1, 1]` range. Its `config.latent_channels`, `scale_factor_temporal`, and
`scale_factor_spatial` must match the VAE. Use a compatible `ModelMixin` decoder for pipeline-wide device placement
and standard serialization. Plain `torch.nn.Module` decoders support instance loading and decoding with explicit
component placement. A decoder receives raw, unnormalized VAE latents; the pipeline applies the inverse latent
normalization before calling it. Selecting `output_type="latent"` bypasses this conversion and decoding.

## Temporal prompts

The `5b-bidir` profile accepts `text_timeline` instead of a single prompt. Each segment contains `start_sec`, `end_sec`,
and `text`. For example, this timeline describes 49 frames sampled at 24 fps:

```py
timeline = {
    "fps": 24,
    "num_frames": 49,
    "segments": [
        {"start_sec": 0.0, "end_sec": 1.0, "text": "A character walks through snow."},
        {"start_sec": 1.0, "end_sec": 2.0, "text": "The character approaches a warm campfire."},
    ],
}
```

Pass it as `text_timeline=timeline` with the image and video. Segments must start at zero, be contiguous, and cover
the timestamp of the last requested frame. Each segment must receive at least one latent frame. A latent frame
exactly at a segment end belongs to the segment on its left. The timeline frame count must match the requested
video, and its `fps` describes the supplied frames; it does not trigger resampling. For a batch, provide one timeline
dictionary per video. Timeline prompts cannot be combined with precomputed text embeddings. Other profiles reject
`text_timeline`.

Causal profiles accept `chunk_prompts`, a list containing one complete prompt per latent chunk. For the released
three-frame chunk size, 49 decoded frames produce 13 latent frames and require five prompts; the last chunk has
one latent frame. This interface requires batch size one and cannot be combined with precomputed positive text
embeddings. A single `prompt` applies to all chunks when `chunk_prompts` is omitted. Bidirectional profiles reject
`chunk_prompts`.

Each pipeline call starts with fresh causal history. The released 18-frame latent context includes six retained
sink frames and the current chunk. Within a call, completed clean chunks are committed to the cache before the
next chunk is generated.

## Multiple GPUs and the command-line example

The inference example resamples video timestamps and uses Ulysses context parallelism with PyTorch FSDP2 on the
same ranks. Run with Python for one GPU or `torchrun` for two or four GPUs:

```bash
torchrun --standalone --nproc_per_node=2 examples/edenlive/inference_edenlive.py \
  --model VAST-AI/EdenLive --profile 14b-causal \
  --video proxy.mp4 --image reference.png \
  --prompt "A character explores a snowy landscape." \
  --num-frames 81 --output edenlive.mp4
```

Use `--full-vae` for full VAE decoding. Only rank zero exports the video; the example also records timing and peak
memory. Keep the parallel degree fixed when comparing reproducible outputs: BF16 matrix multiplication shapes
can change predictions, as also observed in the reference implementation.

The example defaults to 24 fps for 5B and 16 fps for 14B. See the
[example README](https://github.com/huggingface/diffusers/tree/main/examples/edenlive) for installation, single-GPU
usage, temporal prompt JSON files, and the timing report format.

## EdenLivePipeline

The constructor registers initialized components. `from_pretrained()` loads a release or saved pipeline;
`__call__()` prepares the image, proxy features, text, and initial noise before running the profile's sampling path.
Use `generator` for reproducible noise and `callback_on_step_end` to inspect or replace supported tensors after a
step. In causal mode callbacks receive the current chunk, while their step indices count across all chunks.

[[autodoc]] EdenLivePipeline
    - all
    - __call__

## EdenLiveLightTAEDecoder

This decoder implements the released Tiny decoding path without an encoder. The 5B variant uses 48 latent channels
and `patch_size=2`; the 14B variant uses 16 channels and `patch_size=1`. `decode()` accepts raw VAE latents in
`(B, C, T_latent, H_latent, W_latent)` layout and returns `DecoderOutput.sample` in `(B, 3, T, H, W)` layout with
values in `[-1, 1]`. Unlike pipeline frame output, its channel axis precedes the time axis.

To decode `output_type="latent"` yourself, invert the VAE normalization first:

\\[
z_{\mathrm{raw}} = z_{\mathrm{diffusion}} \odot \sigma + \mu.
\\]

Here \\(\mu\\) and \\(\sigma\\) are the channel-wise `vae.config.latents_mean` and `vae.config.latents_std`, and
\\(\odot\\) denotes element-wise multiplication with broadcasting. Move the unnormalized latents to the decoder's
device and dtype before decoding. The pipeline handles this when returning decoded frames. Every `decode()` call
creates fresh temporal state and supports a short final chunk; separate calls do not share decoder history.

[[autodoc]] EdenLiveLightTAEDecoder
    - decode

## EdenLivePipelineOutput

With `return_dict=True`, read the result from `.frames`:

| `output_type` | `.frames` layout | Values |
|---|---|---|
| `"np"` (default) | NumPy `(B, T, H, W, 3)` | Floating-point RGB in [0, 1] |
| `"pt"` | Tensor `(B, T, 3, H, W)` | Floating-point RGB in [0, 1] |
| `"pil"` | One list of PIL RGB frames per video | PIL images |
| `"latent"` | Tensor `(B, C, T_latent, H_latent, W_latent)` | Normalized diffusion latents, without RGB bounds |

With `return_dict=False`, the pipeline returns `(frames,)` with the same selected representation. Select
`result.frames[0]` to export the first decoded video with `export_to_video()`.

[[autodoc]] pipelines.edenlive.pipeline_output.EdenLivePipelineOutput
