<!-- Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License. -->

# MiniMaxH3Transformer3DModel

A Diffusion Transformer model for joint video and audio generation, introduced in [MiniMax-H3](https://huggingface.co/MiniMaxAI) by MiniMax.

MiniMax-H3 runs a single stack of blocks over **one packed 1-D sequence** that holds the text conditioning, the conditioning image and video rows, the audio rows and the target video rows at once. Attention is full self-attention over that sequence, so there is no cross-attention and no per-modality block weights. Modality-specific behaviour comes only from the two input patch projections, the per-row modality tag that selects the AdaLN modulation parameters, and the two output heads.

Building the packed layout is the caller's job, which is why the forward signature takes the layout apart from the latents: the `(t, h, w)` position grid, the per-row modality tags, the per-row timestep indices and the three index tensors that address the video, audio and text rows. [`MiniMaxH3Pipeline`] and [`MiniMaxH3Ref2VAPipeline`] build all of it.

A layout that carries padding rows (tag `-1`) needs a masked attention backend, since those rows are kept in their own attention document by a boolean mask; a padless sequence needs no mask and keeps every backend available.

One repository holds both released checkpoint partitions, so the subfolder is what selects the task: `transformer/` for the text and keyframe tasks, `transformer_ref/` for the omni-reference task.

```python
import torch
from diffusers import MiniMaxH3Transformer3DModel

transformer = MiniMaxH3Transformer3DModel.from_pretrained(
    "MiniMaxAI/MiniMax-H3", subfolder="transformer", dtype=torch.bfloat16
).to("cuda")
```

The checkpoint is mixed precision: the two input patch projections, the timestep MLP and the two output heads are float32 while the block stack is bfloat16. `from_pretrained` keeps that layout through `_keep_in_fp32_modules`, so pass `dtype=torch.bfloat16` and let it place the float32 modules rather than casting the model with `.to(torch.bfloat16)` afterwards.

## MiniMaxH3Transformer3DModel

[[autodoc]] MiniMaxH3Transformer3DModel

## MiniMaxH3TransformerOutput

[[autodoc]] models.transformers.transformer_minimax_h3.MiniMaxH3TransformerOutput
