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

# SANA-Sprint

<div class="flex flex-wrap space-x-1">
  <img alt="LoRA" src="https://img.shields.io/badge/LoRA-d8b4fe?style=flat"/>
</div>

[SANA-Sprint: One-Step Diffusion with Continuous-Time Consistency Distillation](https://huggingface.co/papers/2503.09641) from NVIDIA, MIT HAN Lab, and Hugging Face by Junsong Chen, Shuchen Xue, Yuyang Zhao, Jincheng Yu, Sayak Paul, Junyu Chen, Han Cai, Enze Xie, Song Han

The abstract from the paper is:

*This paper presents SANA-Sprint, an efficient diffusion model for ultra-fast text-to-image (T2I) generation. SANA-Sprint is built on a pre-trained foundation model and augmented with hybrid distillation, dramatically reducing inference steps from 20 to 1-4. We introduce three key innovations: (1) We propose a training-free approach that transforms a pre-trained flow-matching model for continuous-time consistency distillation (sCM), eliminating costly training from scratch and achieving high training efficiency. Our hybrid distillation strategy combines sCM with latent adversarial distillation (LADD): sCM ensures alignment with the teacher model, while LADD enhances single-step generation fidelity. (2) SANA-Sprint is a unified step-adaptive model that achieves high-quality generation in 1-4 steps, eliminating step-specific training and improving efficiency. (3) We integrate ControlNet with SANA-Sprint for real-time interactive image generation, enabling instant visual feedback for user interaction. SANA-Sprint establishes a new Pareto frontier in speed-quality tradeoffs, achieving state-of-the-art performance with 7.59 FID and 0.74 GenEval in only 1 step — outperforming FLUX-schnell (7.94 FID / 0.71 GenEval) while being 10× faster (0.1s vs 1.1s on H100). It also achieves 0.1s (T2I) and 0.25s (ControlNet) latency for 1024×1024 images on H100, and 0.31s (T2I) on an RTX 4090, showcasing its exceptional efficiency and potential for AI-powered consumer applications (AIPC). Code and pre-trained models will be open-sourced.*

> [!TIP]
> Make sure to check out the Schedulers [guide](../../using-diffusers/schedulers) to learn how to explore the tradeoff between scheduler speed and quality, and see the [reuse components across pipelines](../../using-diffusers/loading#reuse-a-pipeline) section to learn how to efficiently load the same components into multiple pipelines.

This pipeline was contributed by [lawrence-cj](https://github.com/lawrence-cj), [shuchen Xue](https://github.com/scxue) and [Enze Xie](https://github.com/xieenze). The original codebase can be found [here](https://github.com/NVlabs/Sana). The original weights can be found under [hf.co/Efficient-Large-Model](https://huggingface.co/Efficient-Large-Model/).

Available models:

|                                                                    Model                                                                    | Recommended dtype |
|:-------------------------------------------------------------------------------------------------------------------------------------------:|:-----------------:|
| [`Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers`](https://huggingface.co/Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers) | `torch.bfloat16`  |
| [`Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers`](https://huggingface.co/Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers) | `torch.bfloat16`  |

Refer to [this](https://huggingface.co/collections/Efficient-Large-Model/sana-sprint-67d6810d65235085b3b17c76) collection for more information.

Note: The recommended dtype mentioned is for the transformer weights. The text encoder must stay in `torch.bfloat16` and VAE weights must stay in `torch.bfloat16` or `torch.float32` for the model to work correctly. Please refer to the inference example below to see how to load the model with the recommended dtype. 


## Quantization

Quantization helps reduce the memory requirements of very large models by storing model weights in a lower precision data type. However, quantization may have varying impact on video quality depending on the video model.

Refer to the [Quantization](../../quantization/overview) overview to learn more about supported quantization backends and selecting a quantization backend that supports your use case. The example below demonstrates how to load a quantized [`SanaSprintPipeline`] for inference with bitsandbytes.

```py
import torch
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig, SanaTransformer2DModel, SanaSprintPipeline
from transformers import BitsAndBytesConfig as BitsAndBytesConfig, AutoModel

quant_config = BitsAndBytesConfig(load_in_8bit=True)
text_encoder_8bit = AutoModel.from_pretrained(
    "Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers",
    subfolder="text_encoder",
    quantization_config=quant_config,
    torch_dtype=torch.bfloat16,
)

quant_config = DiffusersBitsAndBytesConfig(load_in_8bit=True)
transformer_8bit = SanaTransformer2DModel.from_pretrained(
    "Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers",
    subfolder="transformer",
    quantization_config=quant_config,
    torch_dtype=torch.bfloat16,
)

pipeline = SanaSprintPipeline.from_pretrained(
    "Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers",
    text_encoder=text_encoder_8bit,
    transformer=transformer_8bit,
    torch_dtype=torch.bfloat16,
    device_map="balanced",
)

prompt = "a tiny astronaut hatching from an egg on the moon"
image = pipeline(prompt).images[0]
image.save("sana.png")
```

## Setting `max_timesteps`

Users can tweak the `max_timesteps` value for experimenting with the visual quality of the generated outputs. The default `max_timesteps` value was obtained with an inference-time search process. For more details about it, check out the paper.

## Image to Image 

The [`SanaSprintImg2ImgPipeline`] is a pipeline for image-to-image generation. It takes an input image and a prompt, and generates a new image based on the input image and the prompt.

```py
import torch
from diffusers import SanaSprintImg2ImgPipeline
from diffusers.utils.loading_utils import load_image

image = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/penguin.png"
)

pipe = SanaSprintImg2ImgPipeline.from_pretrained(
    "Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers", 
    torch_dtype=torch.bfloat16)
pipe.to("cuda")

image = pipe(
    prompt="a cute pink bear", 
    image=image, 
    strength=0.5, 
    height=832, 
    width=480
).images[0]
image.save("output.png")
```

## SanaSprintPipeline

[[autodoc]] SanaSprintPipeline
  - all
  - __call__

## SanaSprintImg2ImgPipeline

[[autodoc]] SanaSprintImg2ImgPipeline
  - all
  - __call__


## SanaPipelineOutput

[[autodoc]] pipelines.sana.pipeline_output.SanaPipelineOutput
