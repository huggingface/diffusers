# Getting Started: VAE Encode with Hybrid Inference

VAE encode is used for training, image-to-image and image-to-video - turning into images or videos into latent representations.

## Memory

These tables demonstrate the VRAM requirements for VAE encode with SD v1 and SD XL on different GPUs.

For the majority of these GPUs the memory usage % dictates other models (text encoders, UNet/Transformer) must be offloaded, or tiled encoding has to be used which increases time taken and impacts quality.

<details><summary>SD v1.5</summary>

| GPU | Resolution | Time (seconds) | Memory (%) | Tiled Time (secs) | Tiled Memory (%) |
| --- | --- | --- | --- | --- | --- |
TODO

</details>

<details><summary>SDXL</summary>

| GPU | Resolution | Time (seconds) | Memory Consumed (%) | Tiled Time (seconds) | Tiled Memory (%) |
| --- | --- | --- | --- | --- | --- |
TODO

</details>

## Available VAEs

|   | **Endpoint** | **Model** |
|:-:|:-----------:|:--------:|
| **Stable Diffusion v1** | [https://qc6479g0aac6qwy9.us-east-1.aws.endpoints.huggingface.cloud](https://qc6479g0aac6qwy9.us-east-1.aws.endpoints.huggingface.cloud) | [`stabilityai/sd-vae-ft-mse`](https://hf.co/stabilityai/sd-vae-ft-mse) |
| **Stable Diffusion XL** | [https://xjqqhmyn62rog84g.us-east-1.aws.endpoints.huggingface.cloud](https://xjqqhmyn62rog84g.us-east-1.aws.endpoints.huggingface.cloud) | [`madebyollin/sdxl-vae-fp16-fix`](https://hf.co/madebyollin/sdxl-vae-fp16-fix) |
| **Flux** | [https://ptccx55jz97f9zgo.us-east-1.aws.endpoints.huggingface.cloud](https://ptccx55jz97f9zgo.us-east-1.aws.endpoints.huggingface.cloud) | [`black-forest-labs/FLUX.1-schnell`](https://hf.co/black-forest-labs/FLUX.1-schnell) |


> [!TIP]
> Model support can be requested [here](https://github.com/huggingface/diffusers/issues/new?template=remote-vae-pilot-feedback.yml).


## Code

> [!TIP]
> Install `diffusers` from `main` to run the code: `pip install git+https://github.com/huggingface/diffusers@main`


A helper method simplifies interacting with Hybrid Inference.

```python
from diffusers.utils.remote_utils import remote_encode
```

### Basic example

Let's encode an image, then decode it to demonstrate.

<figure class="image flex flex-col items-center justify-center text-center m-0 w-full">
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/astronaut.jpg"/>
</figure>

<details><summary>Code</summary>

```python
from diffusers.utils import load_image
from diffusers.utils.remote_utils import remote_decode

image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/astronaut.jpg?download=true")

latent = remote_encode(
    endpoint="https://ptccx55jz97f9zgo.us-east-1.aws.endpoints.huggingface.cloud/",
    scaling_factor=0.3611,
    shift_factor=0.1159,
)

decoded = remote_decode(
    endpoint="https://whhx50ex1aryqvw6.us-east-1.aws.endpoints.huggingface.cloud/",
    tensor=latent,
    scaling_factor=0.3611,
    shift_factor=0.1159,
)
```

</details>

<figure class="image flex flex-col items-center justify-center text-center m-0 w-full">
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/remote_vae/decoded.png"/>
</figure>


### Generation

Now let's look at a generation example, we'll encode the image, generate then remotely decode too!

<details><summary>Code</summary>

```python
import torch
from diffusers import StableDiffusionImg2ImgPipeline
from diffusers.utils import load_image
from diffusers.utils.remote_utils import remote_decode, remote_encode

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    variant="fp16",
    vae=None,
).to("cuda")

init_image = load_image(
    "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"
)
init_image = init_image.resize((768, 512))

init_latent = remote_encode(
    endpoint="https://qc6479g0aac6qwy9.us-east-1.aws.endpoints.huggingface.cloud/",
    image=init_image,
    scaling_factor=0.18215,
)

prompt = "A fantasy landscape, trending on artstation"
latent = pipe(
    prompt=prompt,
    image=init_latent,
    strength=0.75,
    output_type="latent",
).images

image = remote_decode(
    endpoint="https://q1bj3bpq6kzilnsu.us-east-1.aws.endpoints.huggingface.cloud/",
    tensor=latent,
    scaling_factor=0.18215,
)
image.save("fantasy_landscape.jpg")
```

</details>

<figure class="image flex flex-col items-center justify-center text-center m-0 w-full">
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/remote_vae/fantasy_landscape.png"/>
</figure>

## Integrations

* **[SD.Next](https://github.com/vladmandic/sdnext):** All-in-one UI with direct supports Hybrid Inference.
* **[ComfyUI-HFRemoteVae](https://github.com/kijai/ComfyUI-HFRemoteVae):** ComfyUI node for Hybrid Inference.
