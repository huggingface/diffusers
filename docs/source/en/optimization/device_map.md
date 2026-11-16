<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Split model components across GPUs with device_map

The `device_map` argument places model components on separate GPUs. Use it when a model doesn't fit on one GPU and your GPUs don't have a fast interconnect. Refer to [Distributed inference](../training/distributed_inference) to compare it with the other strategies, and to [Device placement](./memory#device-placement) for more `device_map` options.

The example below loads each component only for the stage that needs it, and uses `device_map` to spread the large ones across two GPUs.

## Encode the prompt

Set `device_map="balanced"` to place the text encoders on separate GPUs. Use `max_memory` to cap how much memory each GPU uses. Set `transformer` and `vae` to `None` so only the text encoders load.

```py
from diffusers import FluxPipeline
import torch

prompt = """
cinematic film still of a cat sipping a margarita in a pool in Palm Springs, California
highly detailed, high budget hollywood movie, cinemascope, moody, epic, gorgeous, film grain
"""

pipeline = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    transformer=None,
    vae=None,
    device_map="balanced",
    max_memory={0: "16GB", 1: "16GB"},
    dtype=torch.bfloat16
)
with torch.no_grad():
    print("Encoding prompts.")
    prompt_embeds, pooled_prompt_embeds, text_ids = pipeline.encode_prompt(
        prompt=prompt, prompt_2=None, max_sequence_length=512
    )
```

Once you have the text embeddings, remove the text encoders from the GPU to make room for the transformer.

```py
import gc

def flush():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()

del pipeline.text_encoder
del pipeline.text_encoder_2
del pipeline.tokenizer
del pipeline.tokenizer_2
del pipeline

flush()
```

## Denoise with the sharded transformer

Set `device_map="auto"` to let Accelerate place the transformer. It fills the first GPU before moving to the next, and uses the CPU or disk only when the GPUs run out of memory. On two smaller GPUs, the transformer ends up split across both. Weights on the CPU or disk slow down inference.

```py
from diffusers import AutoModel
import torch

transformer = AutoModel.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    subfolder="transformer",
    device_map="auto",
    dtype=torch.bfloat16
)
```

> [!TIP]
> Check `transformer.hf_device_map` to see how the transformer is distributed across devices. A pipeline loaded with `device_map` also has a `hf_device_map` attribute that shows where each component is placed.

Load a pipeline with only the transformer, and set `output_type="latent"` so it returns latents instead of images.

```py
pipeline = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    text_encoder=None,
    text_encoder_2=None,
    tokenizer=None,
    tokenizer_2=None,
    vae=None,
    transformer=transformer,
    dtype=torch.bfloat16
)

print("Running denoising.")
height, width = 768, 1360
latents = pipeline(
    prompt_embeds=prompt_embeds,
    pooled_prompt_embeds=pooled_prompt_embeds,
    num_inference_steps=50,
    guidance_scale=3.5,
    height=height,
    width=width,
    output_type="latent",
).images
```

## Decode the latents

Remove the pipeline and transformer from memory and load a VAE to decode the latents. The VAE is small enough to fit on one GPU.

```py
import torch
from diffusers import AutoencoderKL
from diffusers.image_processor import VaeImageProcessor

del pipeline.transformer
del pipeline
del transformer

flush()

vae = AutoencoderKL.from_pretrained(
    "black-forest-labs/FLUX.1-dev", subfolder="vae", dtype=torch.bfloat16
).to("cuda")  # or "mps", "xpu", "cpu"
vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)

with torch.no_grad():
    print("Running decoding.")
    latents = FluxPipeline._unpack_latents(latents, height, width, vae_scale_factor)
    latents = (latents / vae.config.scaling_factor) + vae.config.shift_factor

    image = vae.decode(latents, return_dict=False)[0]
    image = image_processor.postprocess(image, output_type="pil")
    image[0].save("split_transformer.png")
```

By selectively loading and unloading the models you need at a given stage and sharding the largest models across multiple GPUs, you can run inference with large models on consumer GPUs.
