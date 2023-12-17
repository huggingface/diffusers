<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Consistency Models

Consistency Models were proposed in [Consistency Models](https://huggingface.co/papers/2303.01469) by Yang Song, Prafulla Dhariwal, Mark Chen, and Ilya Sutskever.

The abstract from the paper is:

*Diffusion models have significantly advanced the fields of image, audio, and video generation, but they depend on an iterative sampling process that causes slow generation. To overcome this limitation, we propose consistency models, a new family of models that generate high quality samples by directly mapping noise to data. They support fast one-step generation by design, while still allowing multistep sampling to trade compute for sample quality. They also support zero-shot data editing, such as image inpainting, colorization, and super-resolution, without requiring explicit training on these tasks. Consistency models can be trained either by distilling pre-trained diffusion models, or as standalone generative models altogether. Through extensive experiments, we demonstrate that they outperform existing distillation techniques for diffusion models in one- and few-step sampling, achieving the new state-of-the-art FID of 3.55 on CIFAR-10 and 6.20 on ImageNet 64x64 for one-step generation. When trained in isolation, consistency models become a new family of generative models that can outperform existing one-step, non-adversarial generative models on standard benchmarks such as CIFAR-10, ImageNet 64x64 and LSUN 256x256.*

The original codebase can be found at [openai/consistency_models](https://github.com/openai/consistency_models), and additional checkpoints are available at [openai](https://huggingface.co/openai).

The pipeline was contributed by [dg845](https://github.com/dg845) and [ayushtues](https://huggingface.co/ayushtues). ❤️

## Tips

For an additional speed-up, use `torch.compile` to generate multiple images in <1 second:

```diff
  import torch
  from diffusers import ConsistencyModelPipeline

  device = "cuda"
  # Load the cd_bedroom256_lpips checkpoint.
  model_id_or_path = "openai/diffusers-cd_bedroom256_lpips"
  pipe = ConsistencyModelPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float16)
  pipe.to(device)

+ pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)

  # Multistep sampling
  # Timesteps can be explicitly specified; the particular timesteps below are from the original GitHub repo:
  # https://github.com/openai/consistency_models/blob/main/scripts/launch.sh#L83
  for _ in range(10):
      image = pipe(timesteps=[17, 0]).images[0]
      image.show()
```


## ConsistencyModelPipeline
[[autodoc]] ConsistencyModelPipeline
    - all
    - __call__

## ImagePipelineOutput
[[autodoc]] pipelines.ImagePipelineOutput
