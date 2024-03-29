<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Perturbed-Attention Guidance

[Self-Rectifying Diffusion Sampling with Perturbed-Attention Guidance](https://arxiv.org/abs/2403.17377)

Abstract

*Recent studies have demonstrated that diffusion models are capable of generating high-quality samples, but their quality heavily depends on sampling guidance techniques, such as classifier guidance (CG) and classifier-free guidance (CFG). These techniques are often not applicable in unconditional generation or in various downstream tasks such as image restoration. In this paper, we propose a novel sampling guidance, called Perturbed-Attention Guidance (PAG), which improves diffusion sample quality across both unconditional and conditional settings, achieving this without requiring additional training or the integration of external modules. PAG is designed to progressively enhance the structure of samples throughout the denoising process. It involves generating intermediate samples with degraded structure by substituting selected self-attention maps in diffusion U-Net with an identity matrix, by considering the self-attention mechanisms' ability to capture structural information, and guiding the denoising process away from these degraded samples. In both ADM and Stable Diffusion, PAG surprisingly improves sample quality in conditional and even unconditional scenarios. Moreover, PAG significantly improves the baseline performance in various downstream tasks where existing guidances such as CG or CFG cannot be fully utilized, including ControlNet with empty prompts and image restoration such as inpainting and deblurring.*

This implementation is based on [Diffusers](https://huggingface.co/docs/diffusers/index). StableDiffusionPAGPipeline is a modification of StableDiffusionPipeline to support Perturbed-Attention Guidance (PAG). For more information about PAG, please refer to our project page / paper / GitHub.

[Project](https://ku-cvlab.github.io/Perturbed-Attention-Guidance/) / [arXiv](https://arxiv.org/abs/2403.17377) / [GitHub](https://github.com/KU-CVLAB/Perturbed-Attention-Guidance)

## Loading StableDiffusionPAGPipeline
```
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPAGPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
)

device="cuda"
pipe = pipe.to(device)
```

## Sampling with PAG
```
output = pipe(
        prompts,
        width=512,
        height=512,
        num_inference_steps=50,
        guidance_scale=0.0,
        pag_scale=5.0,
        pag_applied_layers_index=['m0']
    ).images
```

<Tip>

Make sure to check out the Schedulers [guide](../../using-diffusers/schedulers) to learn how to explore the tradeoff between scheduler speed and quality, and see the [reuse components across pipelines](../../using-diffusers/loading#reuse-components-across-pipelines) section to learn how to efficiently load the same components into multiple pipelines.

</Tip>

## StableDiffusionPAGPipeline
[[autodoc]] StableDiffusionPAGPipeline
	- __call__
	- all
