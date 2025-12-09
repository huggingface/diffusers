<!-- Copyright 2025 The ChronoEdit Team and HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License. -->

# ChronoEditTransformer3DModel

A Diffusion Transformer model for 3D video-like data from [ChronoEdit: Towards Temporal Reasoning for Image Editing and World Simulation](https://huggingface.co/papers/2510.04290) from NVIDIA and University of Toronto, by Jay Zhangjie Wu, Xuanchi Ren, Tianchang Shen, Tianshi Cao, Kai He, Yifan Lu, Ruiyuan Gao, Enze Xie, Shiyi Lan, Jose M. Alvarez, Jun Gao, Sanja Fidler, Zian Wang, Huan Ling.

> **TL;DR:** ChronoEdit reframes image editing as a video generation task, using input and edited images as start/end frames to leverage pretrained video models with temporal consistency. A temporal reasoning stage introduces reasoning tokens to ensure physically plausible edits and visualize the editing trajectory.

The model can be loaded with the following code snippet.

```python
from diffusers import ChronoEditTransformer3DModel

transformer = ChronoEditTransformer3DModel.from_pretrained("nvidia/ChronoEdit-14B-Diffusers", subfolder="transformer", torch_dtype=torch.bfloat16)
```

## ChronoEditTransformer3DModel

[[autodoc]] ChronoEditTransformer3DModel

## Transformer2DModelOutput

[[autodoc]] models.modeling_outputs.Transformer2DModelOutput
