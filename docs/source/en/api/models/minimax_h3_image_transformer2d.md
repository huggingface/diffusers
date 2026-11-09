<!-- Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License. -->

# MiniMaxH3ImageTransformer2DModel

A Diffusion Transformer model for image generation and editing, introduced in [MiniMax-H3-Image](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image) by MiniMax.

It is the [`MiniMaxH3Transformer3DModel`] without its audio stream: a single stack of blocks runs over **one packed 1-D sequence** that holds the text conditioning, the rows of the reference images and the rows of the generated image. Building the packed layout is the caller's job, which is why the forward signature takes the layout apart from the latents: the `(t, h, w)` position grid, the per-row modality tags, the per-row timestep indices and the two index tensors that address the image and text rows. [`MiniMaxH3ImageBlocks`] builds all of it.

The checkpoint is distilled to six sampling steps with Parallel Decoding Distillation (PDD), which keeps one output head per step. `forward` therefore takes `pdd_step_index`, the step it is running, and projects with that step's head.

```python
import torch
from diffusers import MiniMaxH3ImageTransformer2DModel

transformer = MiniMaxH3ImageTransformer2DModel.from_pretrained(
    "MiniMaxAI/MiniMax-H3-Image", subfolder="transformer", dtype=torch.bfloat16
).to("cuda")  # or "mps", "xpu", "cpu"
```

The checkpoint is mixed precision: the input patch projection, the timestep MLP and the output heads are float32 while the block stack is bfloat16. `from_pretrained` keeps that layout through `_keep_in_fp32_modules`, so pass `dtype=torch.bfloat16` and let it place the float32 modules rather than casting the model with `.to(torch.bfloat16)` afterwards.

## MiniMaxH3ImageTransformer2DModel

[[autodoc]] MiniMaxH3ImageTransformer2DModel
