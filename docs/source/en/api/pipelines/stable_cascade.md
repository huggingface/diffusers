<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Stable Cascade

This model is built upon the [Würstchen](https://openreview.net/forum?id=gU58d5QeGv) architecture and its main
difference to other models like Stable Diffusion is that it is working at a much smaller latent space. Why is this
important? The smaller the latent space, the **faster** you can run inference and the **cheaper** the training becomes.
How small is the latent space? Stable Diffusion uses a compression factor of 8, resulting in a 1024x1024 image being
encoded to 128x128. Stable Cascade achieves a compression factor of 42, meaning that it is possible to encode a
1024x1024 image to 24x24, while maintaining crisp reconstructions. The text-conditional model is then trained in the
highly compressed latent space. Previous versions of this architecture, achieved a 16x cost reduction over Stable
Diffusion 1.5.

Therefore, this kind of model is well suited for usages where efficiency is important. Furthermore, all known extensions
like finetuning, LoRA, ControlNet, IP-Adapter, LCM etc. are possible with this method as well.

The original codebase can be found at [Stability-AI/StableCascade](https://github.com/Stability-AI/StableCascade).

## Model Overview
Stable Cascade consists of three models: Stage A, Stage B and Stage C, representing a cascade to generate images,
hence the name "Stable Cascade".

Stage A & B are used to compress images, similar to what the job of the VAE is in Stable Diffusion.
However, with this setup, a much higher compression of images can be achieved. While the Stable Diffusion models use a
spatial compression factor of 8, encoding an image with resolution of 1024 x 1024 to 128 x 128, Stable Cascade achieves
a compression factor of 42. This encodes a 1024 x 1024 image to 24 x 24, while being able to accurately decode the
image. This comes with the great benefit of cheaper training and inference. Furthermore, Stage C is responsible
for generating the small 24 x 24 latents given a text prompt.

The Stage C model operates on the small 24 x 24 latents and denoises the latents conditioned on text prompts. The model is also the largest component in the Cascade pipeline and is meant to be used with the `StableCascadePriorPipeline`

The Stage B and Stage A models are used with the `StableCascadeDecoderPipeline` and are responsible for generating the final image given the small 24 x 24 latents.

<Tip warning={true}>

There are some restrictions on data types that can be used with the Stable Cascade models. The official checkpoints for the  `StableCascadePriorPipeline` do not support the `torch.float16` data type. Please use `torch.bfloat16` instead.

In order to use the `torch.bfloat16` data type with the `StableCascadeDecoderPipeline` you need to have PyTorch 2.2.0 or higher installed. This also means that using the `StableCascadeCombinedPipeline` with `torch.bfloat16` requires PyTorch 2.2.0 or higher, since it calls the `StableCascadeDecoderPipeline` internally.

If it is not possible to install PyTorch 2.2.0 or higher in your environment, the `StableCascadeDecoderPipeline` can be used on its own with the `torch.float16` data type. You can download the full precision or `bf16` variant weights for the pipeline and cast the weights to `torch.float16`.

</Tip>

## Usage example

```python
import torch
from diffusers import StableCascadeDecoderPipeline, StableCascadePriorPipeline

prompt = "an image of a shiba inu, donning a spacesuit and helmet"
negative_prompt = ""

prior = StableCascadePriorPipeline.from_pretrained("stabilityai/stable-cascade-prior", variant="bf16", torch_dtype=torch.bfloat16)
decoder = StableCascadeDecoderPipeline.from_pretrained("stabilityai/stable-cascade", variant="bf16", torch_dtype=torch.float16)

prior.enable_model_cpu_offload()
prior_output = prior(
    prompt=prompt,
    height=1024,
    width=1024,
    negative_prompt=negative_prompt,
    guidance_scale=4.0,
    num_images_per_prompt=1,
    num_inference_steps=20
)

decoder.enable_model_cpu_offload()
decoder_output = decoder(
    image_embeddings=prior_output.image_embeddings.to(torch.float16),
    prompt=prompt,
    negative_prompt=negative_prompt,
    guidance_scale=0.0,
    output_type="pil",
    num_inference_steps=10
).images[0]
decoder_output.save("cascade.png")
```

## Using the Lite Versions of the Stage B and Stage C models

```python
import torch
from diffusers import (
    StableCascadeDecoderPipeline,
    StableCascadePriorPipeline,
    StableCascadeUNet,
)

prompt = "an image of a shiba inu, donning a spacesuit and helmet"
negative_prompt = ""

prior_unet = StableCascadeUNet.from_pretrained("stabilityai/stable-cascade-prior", subfolder="prior_lite")
decoder_unet = StableCascadeUNet.from_pretrained("stabilityai/stable-cascade", subfolder="decoder_lite")

prior = StableCascadePriorPipeline.from_pretrained("stabilityai/stable-cascade-prior", prior=prior_unet)
decoder = StableCascadeDecoderPipeline.from_pretrained("stabilityai/stable-cascade", decoder=decoder_unet)

prior.enable_model_cpu_offload()
prior_output = prior(
    prompt=prompt,
    height=1024,
    width=1024,
    negative_prompt=negative_prompt,
    guidance_scale=4.0,
    num_images_per_prompt=1,
    num_inference_steps=20
)

decoder.enable_model_cpu_offload()
decoder_output = decoder(
    image_embeddings=prior_output.image_embeddings,
    prompt=prompt,
    negative_prompt=negative_prompt,
    guidance_scale=0.0,
    output_type="pil",
    num_inference_steps=10
).images[0]
decoder_output.save("cascade.png")
```

## Loading original checkpoints with `from_single_file`

Loading the original format checkpoints is supported via `from_single_file` method in the StableCascadeUNet.

```python
import torch
from diffusers import (
    StableCascadeDecoderPipeline,
    StableCascadePriorPipeline,
    StableCascadeUNet,
)

prompt = "an image of a shiba inu, donning a spacesuit and helmet"
negative_prompt = ""

prior_unet = StableCascadeUNet.from_single_file(
    "https://huggingface.co/stabilityai/stable-cascade/resolve/main/stage_c_bf16.safetensors",
    torch_dtype=torch.bfloat16
)
decoder_unet = StableCascadeUNet.from_single_file(
    "https://huggingface.co/stabilityai/stable-cascade/blob/main/stage_b_bf16.safetensors",
    torch_dtype=torch.bfloat16
)

prior = StableCascadePriorPipeline.from_pretrained("stabilityai/stable-cascade-prior", prior=prior_unet, torch_dtype=torch.bfloat16)
decoder = StableCascadeDecoderPipeline.from_pretrained("stabilityai/stable-cascade", decoder=decoder_unet, torch_dtype=torch.bfloat16)

prior.enable_model_cpu_offload()
prior_output = prior(
    prompt=prompt,
    height=1024,
    width=1024,
    negative_prompt=negative_prompt,
    guidance_scale=4.0,
    num_images_per_prompt=1,
    num_inference_steps=20
)

decoder.enable_model_cpu_offload()
decoder_output = decoder(
    image_embeddings=prior_output.image_embeddings,
    prompt=prompt,
    negative_prompt=negative_prompt,
    guidance_scale=0.0,
    output_type="pil",
    num_inference_steps=10
).images[0]
decoder_output.save("cascade-single-file.png")
```

## Uses

### Direct Use

The model is intended for research purposes for now. Possible research areas and tasks include

- Research on generative models.
- Safe deployment of models which have the potential to generate harmful content.
- Probing and understanding the limitations and biases of generative models.
- Generation of artworks and use in design and other artistic processes.
- Applications in educational or creative tools.

Excluded uses are described below.

### Out-of-Scope Use

The model was not trained to be factual or true representations of people or events,
and therefore using the model to generate such content is out-of-scope for the abilities of this model.
The model should not be used in any way that violates Stability AI's [Acceptable Use Policy](https://stability.ai/use-policy).

## Limitations and Bias

### Limitations
- Faces and people in general may not be generated properly.
- The autoencoding part of the model is lossy.


## StableCascadeCombinedPipeline

[[autodoc]] StableCascadeCombinedPipeline
	- all
	- __call__

## StableCascadePriorPipeline

[[autodoc]] StableCascadePriorPipeline
	- all
	- __call__

## StableCascadePriorPipelineOutput

[[autodoc]] pipelines.stable_cascade.pipeline_stable_cascade_prior.StableCascadePriorPipelineOutput

## StableCascadeDecoderPipeline

[[autodoc]] StableCascadeDecoderPipeline
	- all
	- __call__

