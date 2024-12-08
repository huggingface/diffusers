<!-- Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License. -->

# SparseControlNetModel

SparseControlNetModel is an implementation of ControlNet for [AnimateDiff](https://arxiv.org/abs/2307.04725).

ControlNet was introduced in [Adding Conditional Control to Text-to-Image Diffusion Models](https://huggingface.co/papers/2302.05543) by Lvmin Zhang, Anyi Rao, and Maneesh Agrawala.

The SparseCtrl version of ControlNet was introduced in [SparseCtrl: Adding Sparse Controls to Text-to-Video Diffusion Models](https://arxiv.org/abs/2311.16933) for achieving controlled generation in text-to-video diffusion models by Yuwei Guo, Ceyuan Yang, Anyi Rao, Maneesh Agrawala, Dahua Lin, and Bo Dai.

The abstract from the paper is:

*The development of text-to-video (T2V), i.e., generating videos with a given text prompt, has been significantly advanced in recent years. However, relying solely on text prompts often results in ambiguous frame composition due to spatial uncertainty. The research community thus leverages the dense structure signals, e.g., per-frame depth/edge sequences, to enhance controllability, whose collection accordingly increases the burden of inference. In this work, we present SparseCtrl to enable flexible structure control with temporally sparse signals, requiring only one or a few inputs, as shown in Figure 1. It incorporates an additional condition encoder to process these sparse signals while leaving the pre-trained T2V model untouched. The proposed approach is compatible with various modalities, including sketches, depth maps, and RGB images, providing more practical control for video generation and promoting applications such as storyboarding, depth rendering, keyframe animation, and interpolation. Extensive experiments demonstrate the generalization of SparseCtrl on both original and personalized T2V generators. Codes and models will be publicly available at [this https URL](https://guoyww.github.io/projects/SparseCtrl).*

## Example for loading SparseControlNetModel

```python
import torch
from diffusers import SparseControlNetModel

# fp32 variant in float16
# 1. Scribble checkpoint
controlnet = SparseControlNetModel.from_pretrained("guoyww/animatediff-sparsectrl-scribble", torch_dtype=torch.float16)

# 2. RGB checkpoint
controlnet = SparseControlNetModel.from_pretrained("guoyww/animatediff-sparsectrl-rgb", torch_dtype=torch.float16)

# For loading fp16 variant, pass `variant="fp16"` as an additional parameter
```

## SparseControlNetModel

[[autodoc]] SparseControlNetModel

## SparseControlNetOutput

[[autodoc]] models.controlnet_sparsectrl.SparseControlNetOutput
