<!-- Copyright 2026 The VAST-AI Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# EdenLive

[EdenLive](https://github.com/VAST-AI-Research/EdenLive-Official) generates video from a reference image, prompt, and
RGB proxy video. [`EdenLivePipeline`] supports bidirectional generation and causal generation with persistent history.

## Basic usage

Prepare an RGB reference image and a proxy video with at least 49 frames sampled at 24 fps. The reference image
provides appearance; the proxy video provides scene structure and motion. For installation, see the
[example README](https://github.com/huggingface/diffusers/tree/main/examples/edenlive#setup).

```python
import torch
from diffusers import EdenLivePipeline
from diffusers.utils import export_to_video, load_image, load_video

pipe = EdenLivePipeline.from_pretrained("VAST-AI/EdenLive", profile="5b-causal")
pipe.to("cuda")

image = load_image("reference.png")
video = load_video("proxy-24fps.mp4")
prompt = "A character explores a snowy landscape."

frames = pipe(
    image=image,
    video=video,
    prompt=prompt,
    num_frames=49,
    generator=torch.Generator().manual_seed(42),
).frames[0]
export_to_video(frames, "edenlive.mp4", fps=24)
```

`from_pretrained()` loads the model, `pipe(...)` generates the video, and `.frames[0]` selects the first video in
the output batch. Each offline call starts a new video. Use the [streaming API](#streaming-causal-video) to continue
generation across calls.

## Choose a profile

Select `profile` when loading an original release. These defaults apply when generation options are omitted:

| Profile | Sampling | Steps | Guidance scale | Height × width | Example fps |
|---|---|---|---|---|---|
| `5b-bidir` | Bidirectional | 48 | 3 | 704 × 1280 | 24 |
| `14b-bidir` | Bidirectional | 30 | 3 | 480 × 832 | 16 |
| `5b-causal` | Causal | 4 per chunk | 1 | 704 × 1280 | 24 |
| `14b-causal` | Causal | 4 per chunk | 1 | 480 × 832 | 16 |

Causal profiles use four fixed denoising steps per chunk. Bidirectional profiles support `num_inference_steps`.
Set `guidance_scale` above one to use a `negative_prompt`.

## Prepare the inputs

Supply proxy frames sampled at the desired output frame rate. The pipeline selects frames at the VAE temporal
stride; timestamp resampling is handled by the [CLI example](#multiple-gpus-and-the-command-line-example).

| Input | Accepted layout and values |
|---|---|
| `image` | RGB PIL image, or floating-point NumPy/tensor image in [0, 1] |
| `video` | List of RGB PIL frames; tensor `(T, 3, H, W)` or `(B, T, 3, H, W)`; NumPy `(T, H, W, 3)` or `(B, T, H, W, 3)` with floating-point values in [0, 1] |
| `prompt` | String or list of strings; use `prompt_embeds` for precomputed text embeddings |
| `height`, `width` | Output dimensions, multiples of 32 for 5B or 16 for 14B |
| `num_frames` | Positive frame count of the form $4k+1$, within the input video length |

Here `B` is batch size, `T` is frame count, and `H` and `W` are image height and width. The reference image is
resized and center-cropped to the output dimensions. Omitting `num_frames` uses the largest valid input prefix.

## Change prompts during a video

### Causal chunk prompts

Use `chunk_prompts` for either causal profile. This example reuses the pipeline, image, and video from
[Basic usage](#basic-usage). A 49-frame video has five chunks, so it takes five prompts:

```python
chunk_prompts = [
    "A character walks through snow.",
    "A character walks through snow.",
    "The character approaches a campfire.",
    "The character warms their hands beside the campfire.",
    "The character rests beside the campfire.",
]

frames = pipe(
    image=image,
    video=video,
    chunk_prompts=chunk_prompts,
    num_frames=49,
    generator=torch.Generator().manual_seed(42),
).frames[0]
export_to_video(frames, "chunk-prompts.mp4", fps=24)
```

`chunk_prompts` supplies one text prompt per chunk for a single video.

### Bidirectional text timeline

For `5b-bidir`, use `text_timeline` to assign prompts to time ranges. Reuse the image and video from Basic usage
and load the bidirectional profile:

```python
pipe = EdenLivePipeline.from_pretrained("VAST-AI/EdenLive", profile="5b-bidir")
pipe.to("cuda")

timeline = {
    "fps": 24,
    "num_frames": 49,
    "segments": [
        {"start_sec": 0.0, "end_sec": 1.0, "text": "A character walks through snow."},
        {"start_sec": 1.0, "end_sec": 2.0, "text": "The character approaches a warm campfire."},
    ],
}

frames = pipe(
    image=image,
    video=video,
    text_timeline=timeline,
    num_frames=49,
    generator=torch.Generator().manual_seed(42),
).frames[0]
export_to_video(frames, "timeline.mp4", fps=24)
```

Match the timeline's `fps` and `num_frames` to the input and requested output. Use contiguous segments from zero
through the last frame timestamp, with at least one latent frame per segment.

## Streaming causal video

Both causal profiles support successive calls with shared history. Initialize one state, then pass three already
sampled RGB proxy images per chunk. This example changes the prompt for the second chunk and saves the 21 output
frames as one video:

```python
import numpy as np
import torch
from diffusers import EdenLivePipeline
from diffusers.utils import export_to_video, load_image

pipe = EdenLivePipeline.from_pretrained("VAST-AI/EdenLive", profile="5b-causal")
pipe.to("cuda")

proxy_paths = [
    ["proxy/chunk-0-0.png", "proxy/chunk-0-1.png", "proxy/chunk-0-2.png"],
    ["proxy/chunk-1-0.png", "proxy/chunk-1-1.png", "proxy/chunk-1-2.png"],
]
prompts = ["A character walks through snow.", "The character approaches a campfire."]

state = pipe.create_stream_state(
    image=load_image("reference.png"),
    prompt=prompts[0],
    generator=torch.Generator().manual_seed(42),
    output_type="np",
)
chunks = []
try:
    for paths, current_prompt in zip(proxy_paths, prompts):
        result = pipe(
            stream_state=state,
            proxy_frames=[load_image(path) for path in paths],
            prompt=current_prompt,
        )
        chunks.append(result.frames[0])
finally:
    pipe.close_stream(state)

export_to_video(np.concatenate(chunks, axis=0), "stream.mp4", fps=24)
```

Each call returns its chunk immediately: 9 RGB frames initially, then 12 per call. For incremental video-file
writing, see [`inference_edenlive_streaming.py`](https://github.com/huggingface/diffusers/blob/main/examples/edenlive/inference_edenlive_streaming.py).

Set resolution, guidance, negative prompt, generator, and output format in `create_stream_state()`. Keep these
settings and the pipeline's device and dtype fixed for the stream. Update `prompt` or `prompt_embeds` in each
generation call; omitting both retains the current prompt. Run calls sharing a pipeline serially.

Streaming uses batch size one and three proxy images per call. Tensor inputs have shape `(3, 3, H, W)`; NumPy
inputs use `(3, H, W, 3)`, with finite floating-point RGB values in [0, 1]. RGB output uses LightTAE.
Set `output_type="latent"` at initialization to receive three latent frames per call. For 14B, keep VAE slicing
and tiling disabled. Close a state after a model execution failure and initialize a new one.

## Load and save checkpoints

`EdenLivePipeline.from_pretrained()` accepts a Hub repository or a local directory. Original releases use a root
`model_index.json` and the selected profile's original weights. Saved Diffusers pipelines retain their profile and
decoder choice:

```python
pipe.save_pretrained("edenlive-saved")
pipe = EdenLivePipeline.from_pretrained("edenlive-saved").to("cuda")
```

For local or cached weights:

```python
pipe = EdenLivePipeline.from_pretrained(
    "/path/to/EdenLive", profile="14b-causal", local_files_only=True
).to("cuda")
```

Original releases load the transformer, VAE, text encoder, CLIP, and LightTAE in BF16, with DINO and its projection
in FP32. Use `dtype` for an explicit dtype or per-component mapping. Original-release loading supports
`revision`, `cache_dir`, and `token`; `device_map`, `variant`, and quantization options require a saved
Diffusers pipeline.

### CPU offload

Load the pipeline on CPU and enable model offload before generation:

```python
pipe = EdenLivePipeline.from_pretrained("VAST-AI/EdenLive", profile="5b-causal")
pipe.enable_model_cpu_offload()
```

The offline and streaming calls above use the same interface with CPU offload enabled.

### Offline decoder selection

Original releases use [`EdenLiveLightTAEDecoder`] by default. To decode offline videos with the full Wan VAE,
pass `decoder=None` when loading:

```python
pipe = EdenLivePipeline.from_pretrained(
    "VAST-AI/EdenLive", profile="5b-causal", decoder=None
).to("cuda")
```

Pass an initialized decoder as `decoder=my_decoder` to use a compatible custom decoder.
`output_type="latent"` returns normalized diffusion latents before decoding.

## Multiple GPUs and the command-line example

The CLI resamples proxy video timestamps and supports Ulysses context parallelism with PyTorch FSDP2 on two or four
GPUs. From the Diffusers repository root:

```bash
torchrun --standalone --nproc_per_node=2 examples/edenlive/inference_edenlive.py \
  --model VAST-AI/EdenLive --profile 14b-causal \
  --video proxy.mp4 --image reference.png \
  --prompt "A character explores a snowy landscape." \
  --num-frames 81 --output edenlive.mp4
```

See the [example README](https://github.com/huggingface/diffusers/tree/main/examples/edenlive) for installation,
single-GPU commands, prompt files, and output reports.

## EdenLivePipeline

Load weights with `from_pretrained()`. Generate a complete video with `pipe(image=..., video=..., prompt=...)`,
or initialize a stream with `create_stream_state()` and generate chunks with
`pipe(stream_state=..., proxy_frames=..., prompt=...)`. Call `close_stream()` when the stream ends.

[[autodoc]] EdenLivePipeline
    - all
    - __call__
    - create_stream_state
    - close_stream

## EdenLiveLightTAEDecoder

`decode()` takes raw VAE latents in `(B, C, T_latent, H_latent, W_latent)` layout and returns RGB in
`(B, 3, T, H, W)`, with values in [-1, 1]. To decode the pipeline's `output_type="latent"` yourself, first invert
the VAE normalization:

$$
z_{\mathrm{raw}} = z_{\mathrm{diffusion}} \odot \sigma + \mu.
$$

Here $\mu$ and $\sigma$ are `vae.config.latents_mean` and `vae.config.latents_std`, broadcast over batch, time,
and spatial dimensions; $\odot$ is element-wise multiplication. Match the latents to the decoder's device and dtype.

Pass an `EdenLiveLightTAEDecodeCache` to retain decoder history across calls. Without a cache, each decode starts
a new video. The pipeline manages this cache for streaming generation.

[[autodoc]] EdenLiveLightTAEDecoder
    - decode

## EdenLivePipelineOutput

Read generated frames from `.frames`, or use `return_dict=False` to receive `(frames,)`:

| `output_type` | `.frames` layout | Values |
|---|---|---|
| `"np"` (default) | NumPy `(B, T, H, W, 3)` | Floating-point RGB in [0, 1] |
| `"pt"` | Tensor `(B, T, 3, H, W)` | Floating-point RGB in [0, 1] |
| `"pil"` | One list of PIL RGB frames per video | PIL images |
| `"latent"` | Tensor `(B, C, T_latent, H_latent, W_latent)` | Normalized diffusion latents |

Use `result.frames[0]` to select the first video for `export_to_video()`.

[[autodoc]] pipelines.edenlive.pipeline_output.EdenLivePipelineOutput
