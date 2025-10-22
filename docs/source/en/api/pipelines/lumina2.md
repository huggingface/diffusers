<!-- Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License. -->

# Lumina2

<div class="flex flex-wrap space-x-1">
  <img alt="LoRA" src="https://img.shields.io/badge/LoRA-d8b4fe?style=flat"/>
</div>

[Lumina Image 2.0: A Unified and Efficient Image Generative Model](https://huggingface.co/Alpha-VLLM/Lumina-Image-2.0) is a 2 billion parameter flow-based diffusion transformer capable of generating diverse images from text descriptions.

The abstract from the paper is:

*We introduce Lumina-Image 2.0, an advanced text-to-image model that surpasses previous state-of-the-art methods across multiple benchmarks, while also shedding light on its potential to evolve into a generalist vision intelligence model. Lumina-Image 2.0 exhibits three key properties: (1) Unification – it adopts a unified architecture that treats text and image tokens as a joint sequence, enabling natural cross-modal interactions and facilitating task expansion. Besides, since high-quality captioners can provide semantically better-aligned text-image training pairs, we introduce a unified captioning system, UniCaptioner, which generates comprehensive and precise captions for the model. This not only accelerates model convergence but also enhances prompt adherence, variable-length prompt handling, and task generalization via prompt templates. (2) Efficiency – to improve the efficiency of the unified architecture, we develop a set of optimization techniques that improve semantic learning and fine-grained texture generation during training while incorporating inference-time acceleration strategies without compromising image quality. (3) Transparency – we open-source all training details, code, and models to ensure full reproducibility, aiming to bridge the gap between well-resourced closed-source research teams and independent developers.*

> [!TIP]
> Make sure to check out the Schedulers [guide](../../using-diffusers/schedulers) to learn how to explore the tradeoff between scheduler speed and quality, and see the [reuse components across pipelines](../../using-diffusers/loading#reuse-a-pipeline) section to learn how to efficiently load the same components into multiple pipelines.

## Using Single File loading with Lumina Image 2.0

Single file loading for Lumina Image 2.0 is available for the `Lumina2Transformer2DModel`

```python
import torch
from diffusers import Lumina2Transformer2DModel, Lumina2Pipeline

ckpt_path = "https://huggingface.co/Alpha-VLLM/Lumina-Image-2.0/blob/main/consolidated.00-of-01.pth"
transformer = Lumina2Transformer2DModel.from_single_file(
    ckpt_path, torch_dtype=torch.bfloat16
)

pipe = Lumina2Pipeline.from_pretrained(
    "Alpha-VLLM/Lumina-Image-2.0", transformer=transformer, torch_dtype=torch.bfloat16
)
pipe.enable_model_cpu_offload()
image = pipe(
    "a cat holding a sign that says hello",
    generator=torch.Generator("cpu").manual_seed(0),
).images[0]
image.save("lumina-single-file.png")

```

## Using GGUF Quantized Checkpoints with Lumina Image 2.0

GGUF Quantized checkpoints for the `Lumina2Transformer2DModel` can be loaded via `from_single_file` with the `GGUFQuantizationConfig` 

```python
from diffusers import Lumina2Transformer2DModel, Lumina2Pipeline, GGUFQuantizationConfig 

ckpt_path = "https://huggingface.co/calcuis/lumina-gguf/blob/main/lumina2-q4_0.gguf"
transformer = Lumina2Transformer2DModel.from_single_file(
    ckpt_path,
    quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
    torch_dtype=torch.bfloat16,
)

pipe = Lumina2Pipeline.from_pretrained(
    "Alpha-VLLM/Lumina-Image-2.0", transformer=transformer, torch_dtype=torch.bfloat16
)
pipe.enable_model_cpu_offload()
image = pipe(
    "a cat holding a sign that says hello",
    generator=torch.Generator("cpu").manual_seed(0),
).images[0]
image.save("lumina-gguf.png")
```

## Lumina Accessory

Lumina-Accessory is a multi-task instruction fine-tuning framework designed for the Lumina series. The official repository is from [Alpha-VLLM/Lumina-Accessory](https://github.com/Alpha-VLLM/Lumina-Accessory)

```python
import torch
from diffusers import Lumina2AccessoryPipeline, Lumina2AccessoryTransformer2DModel
from diffusers.utils import load_image

ckpt_path = "https://huggingface.co/Alpha-VLLM/Lumina-Accessory/blob/main/consolidated.00-of-01.pth"
transformer = Lumina2AccessoryTransformer2DModel.from_single_file(ckpt_path, torch_dtype=torch.bfloat16)
pipe = Lumina2AccessoryPipeline.from_pretrained(
    "Alpha-VLLM/Lumina-Image-2.0", transformer=transformer, torch_dtype=torch.bfloat16
    )

# Enable memory optimizations.
pipe.enable_model_cpu_offload()

img = load_image("https://github.com/Alpha-VLLM/Lumina-Accessory/blob/main/examples/case_1_condition.jpg?raw=true")
prompt = "A classical oil painting of a young woman dressed in a modern DARK BLACK leather jacket."
system_prompt = "You are an assistant designed to generate superior images with the highest degree of image-text alignment based on textual prompts and a partially masked image."
image = pipe(
        image=img,
        prompt=prompt,
        system_prompt=system_prompt,
        width=img.size[0],
        height=img.size[1],
        negative_prompt=" ",
        num_inference_steps=25,
        num_images_per_prompt=1,
        guidance_scale=4.0,
        cfg_trunc_ratio=1.0,
        cfg_normalization=True,
        generator=torch.Generator().manual_seed(42),
    ).images[0]
image.save("lumina2_accessory_image_infliling.png")
```

## Lumina2Pipeline

[[autodoc]] Lumina2Pipeline
  - all
  - __call__


## Lumina2AccessoryPipeline

[[autodoc]] Lumina2AccessoryPipeline
  - all
  - __call__