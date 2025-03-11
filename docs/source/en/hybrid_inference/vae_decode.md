# Getting Started: VAE Decode with Hybrid Inference

VAE decode is an essential component of diffusion models - turning latent representations into images or videos.

## Memory

These tables demonstrate the VRAM requirements for VAE decode with SD v1 and SD XL on different GPUs.

For the majority of these GPUs the memory usage % dictates other models (text encoders, UNet/Transformer) must be offloaded, or tiled decoding has to be used which increases time taken and impacts quality.

<details><summary>SD v1.5</summary>

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

<details><summary>SDXL</summary>

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

## Available VAEs

|   | **Endpoint** | **Model** |
|:-:|:-----------:|:--------:|
| **Stable Diffusion v1** | [https://q1bj3bpq6kzilnsu.us-east-1.aws.endpoints.huggingface.cloud](https://q1bj3bpq6kzilnsu.us-east-1.aws.endpoints.huggingface.cloud) | [`stabilityai/sd-vae-ft-mse`](https://hf.co/stabilityai/sd-vae-ft-mse) |
| **Stable Diffusion XL** | [https://x2dmsqunjd6k9prw.us-east-1.aws.endpoints.huggingface.cloud](https://x2dmsqunjd6k9prw.us-east-1.aws.endpoints.huggingface.cloud) | [`madebyollin/sdxl-vae-fp16-fix`](https://hf.co/madebyollin/sdxl-vae-fp16-fix) |
| **Flux** | [https://whhx50ex1aryqvw6.us-east-1.aws.endpoints.huggingface.cloud](https://whhx50ex1aryqvw6.us-east-1.aws.endpoints.huggingface.cloud) | [`black-forest-labs/FLUX.1-schnell`](https://hf.co/black-forest-labs/FLUX.1-schnell) |
| **HunyuanVideo** | [https://o7ywnmrahorts457.us-east-1.aws.endpoints.huggingface.cloud](https://o7ywnmrahorts457.us-east-1.aws.endpoints.huggingface.cloud) | [`hunyuanvideo-community/HunyuanVideo`](https://hf.co/hunyuanvideo-community/HunyuanVideo) |


> [!TIP]
> Model support can be requested [here](https://github.com/huggingface/diffusers/issues/new?template=remote-vae-pilot-feedback.yml).


## Code

> [!TIP]
> Install `diffusers` from `main` to run the code: `pip install git+https://github.com/huggingface/diffusers@main`


A helper method simplifies interacting with Hybrid Inference.

```python
from diffusers.utils.remote_utils import remote_decode
```

### Basic example

Here, we show how to use the remote VAE on random tensors.

<details><summary>Code</summary>

```python
image = remote_decode(
    endpoint="https://q1bj3bpq6kzilnsu.us-east-1.aws.endpoints.huggingface.cloud/",
    tensor=torch.randn([1, 4, 64, 64], dtype=torch.float16),
    scaling_factor=0.18215,
)
```

</details>

<figure class="image flex flex-col items-center justify-center text-center m-0 w-full">
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/remote_vae/output.png"/>
</figure>

Usage for Flux is slightly different. Flux latents are packed so we need to send the `height` and `width`.

<details><summary>Code</summary>

```python
image = remote_decode(
    endpoint="https://whhx50ex1aryqvw6.us-east-1.aws.endpoints.huggingface.cloud/",
    tensor=torch.randn([1, 4096, 64], dtype=torch.float16),
    height=1024,
    width=1024,
    scaling_factor=0.3611,
    shift_factor=0.1159,
)
```

</details>

<figure class="image flex flex-col items-center justify-center text-center m-0 w-full">
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/remote_vae/flux_random_latent.png"/>
</figure>

Finally, an example for HunyuanVideo.

<details><summary>Code</summary>

```python
video = remote_decode(
    endpoint="https://o7ywnmrahorts457.us-east-1.aws.endpoints.huggingface.cloud/",
    tensor=torch.randn([1, 16, 3, 40, 64], dtype=torch.float16),
    output_type="mp4",
)
with open("video.mp4", "wb") as f:
    f.write(video)
```

</details>

<figure class="image flex flex-col items-center justify-center text-center m-0 w-full">
   <video
      alt="queue.mp4"
      autoplay loop autobuffer muted playsinline
    >
    <source src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/remote_vae/video_1.mp4" type="video/mp4">
  </video>
</figure>


### Generation

But we want to use the VAE on an actual pipeline to get an actual image, not random noise. The example below shows how to do it with SD v1.5. 

<details><summary>Code</summary>

```python
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    variant="fp16",
    vae=None,
).to("cuda")

prompt = "Strawberry ice cream, in a stylish modern glass, coconut, splashing milk cream and honey, in a gradient purple background, fluid motion, dynamic movement, cinematic lighting, Mysterious"

latent = pipe(
    prompt=prompt,
    output_type="latent",
).images
image = remote_decode(
    endpoint="https://q1bj3bpq6kzilnsu.us-east-1.aws.endpoints.huggingface.cloud/",
    tensor=latent,
    scaling_factor=0.18215,
)
image.save("test.jpg")
```

</details>

<figure class="image flex flex-col items-center justify-center text-center m-0 w-full">
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/remote_vae/test.jpg"/>
</figure>

Here’s another example with Flux.

<details><summary>Code</summary>

```python
from diffusers import FluxPipeline

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell",
    torch_dtype=torch.bfloat16,
    vae=None,
).to("cuda")

prompt = "Strawberry ice cream, in a stylish modern glass, coconut, splashing milk cream and honey, in a gradient purple background, fluid motion, dynamic movement, cinematic lighting, Mysterious"

latent = pipe(
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
image.save("test.jpg")
```

</details>

<figure class="image flex flex-col items-center justify-center text-center m-0 w-full">
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/remote_vae/test_1.jpg"/>
</figure>

Here’s an example with HunyuanVideo.

<details><summary>Code</summary>

```python
from diffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel

model_id = "hunyuanvideo-community/HunyuanVideo"
transformer = HunyuanVideoTransformer3DModel.from_pretrained(
    model_id, subfolder="transformer", torch_dtype=torch.bfloat16
)
pipe = HunyuanVideoPipeline.from_pretrained(
    model_id, transformer=transformer, vae=None, torch_dtype=torch.float16
).to("cuda")

latent = pipe(
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

</details>

<figure class="image flex flex-col items-center justify-center text-center m-0 w-full">
   <video
      alt="queue.mp4"
      autoplay loop autobuffer muted playsinline
    >
    <source src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/remote_vae/video.mp4" type="video/mp4">
  </video>
</figure>


### Queueing

One of the great benefits of using a remote VAE is that we can queue multiple generation requests. While the current latent is being processed for decoding, we can already queue another one. This helps improve concurrency. 


<details><summary>Code</summary>

```python
import queue
import threading
from IPython.display import display
from diffusers import StableDiffusionPipeline

def decode_worker(q: queue.Queue):
    while True:
        item = q.get()
        if item is None:
            break
        image = remote_decode(
            endpoint="https://q1bj3bpq6kzilnsu.us-east-1.aws.endpoints.huggingface.cloud/",
            tensor=item,
            scaling_factor=0.18215,
        )
        display(image)
        q.task_done()

q = queue.Queue()
thread = threading.Thread(target=decode_worker, args=(q,), daemon=True)
thread.start()

def decode(latent: torch.Tensor):
    q.put(latent)

prompts = [
    "Blueberry ice cream, in a stylish modern glass , ice cubes, nuts, mint leaves, splashing milk cream, in a gradient purple background, fluid motion, dynamic movement, cinematic lighting, Mysterious",
    "Lemonade in a glass, mint leaves, in an aqua and white background, flowers, ice cubes, halo, fluid motion, dynamic movement, soft lighting, digital painting, rule of thirds composition, Art by Greg rutkowski, Coby whitmore",
    "Comic book art, beautiful, vintage, pastel neon colors, extremely detailed pupils, delicate features, light on face, slight smile, Artgerm, Mary Blair, Edmund Dulac, long dark locks, bangs, glowing, fashionable style, fairytale ambience, hot pink.",
    "Masterpiece, vanilla cone ice cream garnished with chocolate syrup, crushed nuts, choco flakes, in a brown background, gold, cinematic lighting, Art by WLOP",
    "A bowl of milk, falling cornflakes, berries, blueberries, in a white background, soft lighting, intricate details, rule of thirds, octane render, volumetric lighting",
    "Cold Coffee with cream, crushed almonds, in a glass, choco flakes, ice cubes, wet, in a wooden background, cinematic lighting, hyper realistic painting, art by Carne Griffiths, octane render, volumetric lighting, fluid motion, dynamic movement, muted colors,",
]

pipe = StableDiffusionPipeline.from_pretrained(
    "Lykon/dreamshaper-8",
    torch_dtype=torch.float16,
    vae=None,
).to("cuda")

pipe.unet = pipe.unet.to(memory_format=torch.channels_last)
pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)

_ = pipe(
    prompt=prompts[0],
    output_type="latent",
)

for prompt in prompts:
    latent = pipe(
        prompt=prompt,
        output_type="latent",
    ).images
    decode(latent)

q.put(None)
thread.join()
```

</details>


<figure class="image flex flex-col items-center justify-center text-center m-0 w-full">
   <video
      alt="queue.mp4"
      autoplay loop autobuffer muted playsinline
    >
    <source src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/remote_vae/queue.mp4" type="video/mp4">
  </video>
</figure>

## Integrations

* **[SD.Next](https://github.com/vladmandic/sdnext):** All-in-one UI with direct supports Hybrid Inference.
* **[ComfyUI-HFRemoteVae](https://github.com/kijai/ComfyUI-HFRemoteVae):** ComfyUI node for Hybrid Inference.
