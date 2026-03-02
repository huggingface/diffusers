<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Remote inference

> [!TIP]
> This is currently an experimental feature, and if you have any feedback, please feel free to leave it [here](https://github.com/huggingface/diffusers/issues/new?template=remote-vae-pilot-feedback.yml).

Remote inference offloads the decoding and encoding process to a remote endpoint to relax the memory requirements for local inference with large models. This feature is powered by [Inference Endpoints](https://huggingface.co/docs/inference-endpoints/index). Refer to the table below for the supported models and endpoint.

| Model | Endpoint | Checkpoint | Support |
|---|---|---|---|
| Stable Diffusion v1 | https://q1bj3bpq6kzilnsu.us-east-1.aws.endpoints.huggingface.cloud | [stabilityai/sd-vae-ft-mse](https://huggingface.co/stabilityai/sd-vae-ft-mse) | encode/decode |
| Stable Diffusion XL | https://x2dmsqunjd6k9prw.us-east-1.aws.endpoints.huggingface.cloud | [madebyollin/sdxl-vae-fp16-fix](https://huggingface.co/madebyollin/sdxl-vae-fp16-fix) | encode/decode |
| Flux | https://whhx50ex1aryqvw6.us-east-1.aws.endpoints.huggingface.cloud | [black-forest-labs/FLUX.1-schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell) | encode/decode |
| HunyuanVideo | https://o7ywnmrahorts457.us-east-1.aws.endpoints.huggingface.cloud | [hunyuanvideo-community/HunyuanVideo](https://huggingface.co/hunyuanvideo-community/HunyuanVideo) | decode |

This guide will show you how to encode and decode latents with remote inference.

## Encoding

Encoding converts images and videos into latent representations. Refer to the table below for the supported VAEs.

Pass an image to [`~utils.remote_encode`] to encode it. The specific `scaling_factor` and `shift_factor` values for each model can be found in the [Remote inference](../hybrid_inference/api_reference) API reference.

```py
import torch
from diffusers import FluxPipeline
from diffusers.utils import load_image
from diffusers.utils.remote_utils import remote_encode

pipeline = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell",
    torch_dtype=torch.float16,
    vae=None,
    device_map="cuda"
)

init_image = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/astronaut.jpg"
)
init_image = init_image.resize((768, 512))

init_latent = remote_encode(
    endpoint="https://whhx50ex1aryqvw6.us-east-1.aws.endpoints.huggingface.cloud",
    image=init_image,
    scaling_factor=0.3611,
    shift_factor=0.1159
)
```

## Decoding

Decoding converts latent representations back into images or videos. Refer to the table below for the available and supported VAEs.

Set the output type to `"latent"` in the pipeline and set the `vae` to `None`. Pass the latents to the [`~utils.remote_decode`] function. For Flux, the latents are packed so the `height` and `width` also need to be passed. The specific `scaling_factor` and `shift_factor` values for each model can be found in the [Remote inference](../hybrid_inference/api_reference) API reference.

<hfoptions id="decode">
<hfoption id="Flux">

```py
from diffusers import FluxPipeline

pipeline = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell",
    torch_dtype=torch.bfloat16,
    vae=None,
    device_map="cuda"
)

prompt = """
A photorealistic Apollo-era photograph of a cat in a small astronaut suit with a bubble helmet, standing on the Moon and holding a flagpole planted in the dusty lunar soil. The flag shows a colorful paw-print emblem. Earth glows in the black sky above the stark gray surface, with sharp shadows and high-contrast lighting like vintage NASA photos.
"""

latent = pipeline(
    prompt=prompt,
    guidance_scale=0.0,
    num_inference_steps=4,
    output_type="latent",
).images
image = remote_decode(
    endpoint="https://whhx50ex1aryqvw6.us-east-1.aws.endpoints.huggingface.cloud/",
    tensor=latent,
    height=1024,
    width=1024,
    scaling_factor=0.3611,
    shift_factor=0.1159,
)
image.save("image.jpg")
```

</hfoption>
<hfoption id="HunyuanVideo">

```py
import torch
from diffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel

transformer = HunyuanVideoTransformer3DModel.from_pretrained(
    "hunyuanvideo-community/HunyuanVideo", subfolder="transformer", torch_dtype=torch.bfloat16
)
pipeline = HunyuanVideoPipeline.from_pretrained(
    model_id, transformer=transformer, vae=None, torch_dtype=torch.float16, device_map="cuda"
)

latent = pipeline(
    prompt="A cat walks on the grass, realistic",
    height=320,
    width=512,
    num_frames=61,
    num_inference_steps=30,
    output_type="latent",
).frames

video = remote_decode(
    endpoint="https://o7ywnmrahorts457.us-east-1.aws.endpoints.huggingface.cloud/",
    tensor=latent,
    output_type="mp4",
)

if isinstance(video, bytes):
    with open("video.mp4", "wb") as f:
        f.write(video)
```

</hfoption>
</hfoptions>

## Queuing

Remote inference supports queuing to process multiple generation requests. While the current latent is being decoded, you can queue the next prompt.

```py
import queue
import threading
from IPython.display import display
from diffusers import StableDiffusionXLPipeline

def decode_worker(q: queue.Queue):
    while True:
        item = q.get()
        if item is None:
            break
        image = remote_decode(
            endpoint="https://q1bj3bpq6kzilnsu.us-east-1.aws.endpoints.huggingface.cloud/",
            tensor=item,
            scaling_factor=0.13025,
        )
        display(image)
        q.task_done()

q = queue.Queue()
thread = threading.Thread(target=decode_worker, args=(q,), daemon=True)
thread.start()

def decode(latent: torch.Tensor):
    q.put(latent)

prompts = [
    "A grainy Apollo-era style photograph of a cat in a snug astronaut suit with a bubble helmet, standing on the lunar surface and gripping a flag with a paw-print emblem. The gray Moon landscape stretches behind it, Earth glowing vividly in the black sky, shadows crisp and high-contrast.",
    "A vintage 1960s sci-fi pulp magazine cover illustration of a heroic cat astronaut planting a flag on the Moon. Bold, saturated colors, exaggerated space gear, playful typography floating in the background, Earth painted in bright blues and greens.",
    "A hyper-detailed cinematic shot of a cat astronaut on the Moon holding a fluttering flag, fur visible through the helmet glass, lunar dust scattering under its feet. The vastness of space and Earth in the distance create an epic, awe-inspiring tone.",
    "A colorful cartoon drawing of a happy cat wearing a chunky, oversized spacesuit, proudly holding a flag with a big paw print on it. The Moon’s surface is simplified with craters drawn like doodles, and Earth in the sky has a smiling face.",
    "A monochrome 1969-style press photo of a “first cat on the Moon” moment. The cat, in a tiny astronaut suit, stands by a planted flag, with grainy textures, scratches, and a blurred Earth in the background, mimicking old archival space photos."
]


pipeline = StableDiffusionXLPipeline.from_pretrained(
    "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    vae=None,
    device_map="cuda"
)

pipeline.unet = pipeline.unet.to(memory_format=torch.channels_last)
pipeline.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)

_ = pipeline(
    prompt=prompts[0],
    output_type="latent",
)

for prompt in prompts:
    latent = pipeline(
        prompt=prompt,
        output_type="latent",
    ).images
    decode(latent)

q.put(None)
thread.join()
```

## Benchmarks

The tables demonstrate the memory requirements for encoding and decoding with Stable Diffusion v1.5 and SDXL on different GPUs.

For the majority of these GPUs, the memory usage dictates whether other models (text encoders, UNet/transformer) need to be offloaded or required tiled encoding. The latter two techniques increases inference time and impacts quality.

<details><summary>Encoding - Stable Diffusion v1.5</summary>

| GPU                           | Resolution   |   Time (seconds) |   Memory (%) |   Tiled Time (secs) |   Tiled Memory (%) |
|:------------------------------|:-------------|-----------------:|-------------:|--------------------:|-------------------:|
| NVIDIA GeForce RTX 4090       | 512x512      |            0.015 |      3.51901 |               0.015 |            3.51901 |
| NVIDIA GeForce RTX 4090       | 256x256      |            0.004 |      1.3154  |               0.005 |            1.3154  |
| NVIDIA GeForce RTX 4090       | 2048x2048    |            0.402 |     47.1852  |               0.496 |            3.51901 |
| NVIDIA GeForce RTX 4090       | 1024x1024    |            0.078 |     12.2658  |               0.094 |            3.51901 |
| NVIDIA GeForce RTX 4080 SUPER | 512x512      |            0.023 |      5.30105 |               0.023 |            5.30105 |
| NVIDIA GeForce RTX 4080 SUPER | 256x256      |            0.006 |      1.98152 |               0.006 |            1.98152 |
| NVIDIA GeForce RTX 4080 SUPER | 2048x2048    |            0.574 |     71.08    |               0.656 |            5.30105 |
| NVIDIA GeForce RTX 4080 SUPER | 1024x1024    |            0.111 |     18.4772  |               0.14  |            5.30105 |
| NVIDIA GeForce RTX 3090       | 512x512      |            0.032 |      3.52782 |               0.032 |            3.52782 |
| NVIDIA GeForce RTX 3090       | 256x256      |            0.01  |      1.31869 |               0.009 |            1.31869 |
| NVIDIA GeForce RTX 3090       | 2048x2048    |            0.742 |     47.3033  |               0.954 |            3.52782 |
| NVIDIA GeForce RTX 3090       | 1024x1024    |            0.136 |     12.2965  |               0.207 |            3.52782 |
| NVIDIA GeForce RTX 3080       | 512x512      |            0.036 |      8.51761 |               0.036 |            8.51761 |
| NVIDIA GeForce RTX 3080       | 256x256      |            0.01  |      3.18387 |               0.01  |            3.18387 |
| NVIDIA GeForce RTX 3080       | 2048x2048    |            0.863 |     86.7424  |               1.191 |            8.51761 |
| NVIDIA GeForce RTX 3080       | 1024x1024    |            0.157 |     29.6888  |               0.227 |            8.51761 |
| NVIDIA GeForce RTX 3070       | 512x512      |            0.051 |     10.6941  |               0.051 |           10.6941  |
| NVIDIA GeForce RTX 3070       | 256x256      |            0.015 |      3.99743 |               0.015 |            3.99743 |
| NVIDIA GeForce RTX 3070       | 2048x2048    |            1.217 |     96.054   |               1.482 |           10.6941  |
| NVIDIA GeForce RTX 3070       | 1024x1024    |            0.223 |     37.2751  |               0.327 |           10.6941  |

</details>

<details><summary>Encoding SDXL</summary>

| GPU                           | Resolution   |   Time (seconds) |   Memory Consumed (%) |   Tiled Time (seconds) |   Tiled Memory (%) |
|:------------------------------|:-------------|-----------------:|----------------------:|-----------------------:|-------------------:|
| NVIDIA GeForce RTX 4090       | 512x512      |            0.029 |               4.95707 |                  0.029 |            4.95707 |
| NVIDIA GeForce RTX 4090       | 256x256      |            0.007 |               2.29666 |                  0.007 |            2.29666 |
| NVIDIA GeForce RTX 4090       | 2048x2048    |            0.873 |              66.3452  |                  0.863 |           15.5649  |
| NVIDIA GeForce RTX 4090       | 1024x1024    |            0.142 |              15.5479  |                  0.143 |           15.5479  |
| NVIDIA GeForce RTX 4080 SUPER | 512x512      |            0.044 |               7.46735 |                  0.044 |            7.46735 |
| NVIDIA GeForce RTX 4080 SUPER | 256x256      |            0.01  |               3.4597  |                  0.01  |            3.4597  |
| NVIDIA GeForce RTX 4080 SUPER | 2048x2048    |            1.317 |              87.1615  |                  1.291 |           23.447   |
| NVIDIA GeForce RTX 4080 SUPER | 1024x1024    |            0.213 |              23.4215  |                  0.214 |           23.4215  |
| NVIDIA GeForce RTX 3090       | 512x512      |            0.058 |               5.65638 |                  0.058 |            5.65638 |
| NVIDIA GeForce RTX 3090       | 256x256      |            0.016 |               2.45081 |                  0.016 |            2.45081 |
| NVIDIA GeForce RTX 3090       | 2048x2048    |            1.755 |              77.8239  |                  1.614 |           18.4193  |
| NVIDIA GeForce RTX 3090       | 1024x1024    |            0.265 |              18.4023  |                  0.265 |           18.4023  |
| NVIDIA GeForce RTX 3080       | 512x512      |            0.064 |              13.6568  |                  0.064 |           13.6568  |
| NVIDIA GeForce RTX 3080       | 256x256      |            0.018 |               5.91728 |                  0.018 |            5.91728 |
| NVIDIA GeForce RTX 3080       | 2048x2048    |          OOM     |             OOM       |                  1.866 |           44.4717  |
| NVIDIA GeForce RTX 3080       | 1024x1024    |            0.302 |              44.4308  |                  0.302 |           44.4308  |
| NVIDIA GeForce RTX 3070       | 512x512      |            0.093 |              17.1465  |                  0.093 |           17.1465  |
| NVIDIA GeForce RTX 3070       | 256x256      |            0.025 |               7.42931 |                  0.026 |            7.42931 |
| NVIDIA GeForce RTX 3070       | 2048x2048    |          OOM     |             OOM       |                  2.674 |           55.8355  |
| NVIDIA GeForce RTX 3070       | 1024x1024    |            0.443 |              55.7841  |                  0.443 |           55.7841  |

</details>

<details><summary>Decoding - Stable Diffusion v1.5</summary>

| GPU | Resolution | Time (seconds) | Memory (%) | Tiled Time (secs) | Tiled Memory (%) |
| --- | --- | --- | --- | --- | --- |
| NVIDIA GeForce RTX 4090 | 512x512 | 0.031 | 5.60% | 0.031 (0%) | 5.60% |
| NVIDIA GeForce RTX 4090 | 1024x1024 | 0.148 | 20.00% | 0.301 (+103%) | 5.60% |
| NVIDIA GeForce RTX 4080 | 512x512 | 0.05 | 8.40% | 0.050 (0%) | 8.40% |
| NVIDIA GeForce RTX 4080 | 1024x1024 | 0.224 | 30.00% | 0.356 (+59%) | 8.40% |
| NVIDIA GeForce RTX 4070 Ti | 512x512 | 0.066 | 11.30% | 0.066 (0%) | 11.30% |
| NVIDIA GeForce RTX 4070 Ti | 1024x1024 | 0.284 | 40.50% | 0.454 (+60%) | 11.40% |
| NVIDIA GeForce RTX 3090 | 512x512 | 0.062 | 5.20% | 0.062 (0%) | 5.20% |
| NVIDIA GeForce RTX 3090 | 1024x1024 | 0.253 | 18.50% | 0.464 (+83%) | 5.20% |
| NVIDIA GeForce RTX 3080 | 512x512 | 0.07 | 12.80% | 0.070 (0%) | 12.80% |
| NVIDIA GeForce RTX 3080 | 1024x1024 | 0.286 | 45.30% | 0.466 (+63%) | 12.90% |
| NVIDIA GeForce RTX 3070 | 512x512 | 0.102 | 15.90% | 0.102 (0%) | 15.90% |
| NVIDIA GeForce RTX 3070 | 1024x1024 | 0.421 | 56.30% | 0.746 (+77%) | 16.00% |

</details>

<details><summary>Decoding SDXL</summary>

| GPU | Resolution | Time (seconds) | Memory Consumed (%) | Tiled Time (seconds) | Tiled Memory (%) |
| --- | --- | --- | --- | --- | --- |
| NVIDIA GeForce RTX 4090 | 512x512 | 0.057 | 10.00% | 0.057 (0%) | 10.00% |
| NVIDIA GeForce RTX 4090 | 1024x1024 | 0.256 | 35.50% | 0.257 (+0.4%) | 35.50% |
| NVIDIA GeForce RTX 4080 | 512x512 | 0.092 | 15.00% | 0.092 (0%) | 15.00% |
| NVIDIA GeForce RTX 4080 | 1024x1024 | 0.406 | 53.30% | 0.406 (0%) | 53.30% |
| NVIDIA GeForce RTX 4070 Ti | 512x512 | 0.121 | 20.20% | 0.120 (-0.8%) | 20.20% |
| NVIDIA GeForce RTX 4070 Ti | 1024x1024 | 0.519 | 72.00% | 0.519 (0%) | 72.00% |
| NVIDIA GeForce RTX 3090 | 512x512 | 0.107 | 10.50% | 0.107 (0%) | 10.50% |
| NVIDIA GeForce RTX 3090 | 1024x1024 | 0.459 | 38.00% | 0.460 (+0.2%) | 38.00% |
| NVIDIA GeForce RTX 3080 | 512x512 | 0.121 | 25.60% | 0.121 (0%) | 25.60% |
| NVIDIA GeForce RTX 3080 | 1024x1024 | 0.524 | 93.00% | 0.524 (0%) | 93.00% |
| NVIDIA GeForce RTX 3070 | 512x512 | 0.183 | 31.80% | 0.183 (0%) | 31.80% |
| NVIDIA GeForce RTX 3070 | 1024x1024 | 0.794 | 96.40% | 0.794 (0%) | 96.40% |

</details>


## Resources

- Remote inference is also supported in [SD.Next](https://github.com/vladmandic/sdnext) and [ComfyUI-HFRemoteVae](https://github.com/kijai/ComfyUI-HFRemoteVae).
- Refer to the [Remote VAEs for decoding with Inference Endpoints](https://huggingface.co/blog/remote_vae) blog post to learn more.