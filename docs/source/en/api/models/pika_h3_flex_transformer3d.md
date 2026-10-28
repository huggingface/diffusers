<!-- Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License. -->

# PikaH3FlexTransformer3DModel

The Diffusion Transformer of [Pika H3 Flex](https://huggingface.co/Pika-Labs/Pika-H3-Flex-Early-Access) by Pika: the [`MiniMaxH3Transformer3DModel`] architecture and weights, post-trained into a six-evaluation generator through a pair of rank-128 LoRA adapters.

The architecture is MiniMax-H3's — one stack of blocks over a single packed sequence of text, conditioning, audio and video rows, with full self-attention and per-row `(timestep, modality)` AdaLN modulation — and so is the checkpoint layout, so the model is loaded from the `MiniMaxAI/MiniMax-H3` partitions (`transformer/` for the text and keyframe tasks, `transformer_ref/` for the omni-reference task) and the Pika adapter is loaded on top. What this class changes is the precision contract of the AdaLN projections: every AdaLN activation runs in float32 unconditionally, which is what the released adapters were validated against.

```python
import torch
from diffusers import PikaH3FlexTransformer3DModel

transformer = PikaH3FlexTransformer3DModel.from_pretrained(
    "MiniMaxAI/MiniMax-H3", subfolder="transformer", dtype=torch.bfloat16
).to("cuda")  # or "mps", "xpu", "cpu"
```

The checkpoint is mixed precision: the two input patch projections, the timestep MLP and the two output heads are float32 while the block stack is bfloat16. `from_pretrained` keeps that layout through `_keep_in_fp32_modules`, so pass `dtype=torch.bfloat16` and let it place the float32 modules rather than casting the model with `.to(torch.bfloat16)` afterwards. The adapters are loaded at the pipeline level with [`~loaders.PikaH3FlexLoraLoaderMixin.load_lora_weights`].

## PikaH3FlexTransformer3DModel

[[autodoc]] PikaH3FlexTransformer3DModel

## PikaH3FlexTransformerOutput

[[autodoc]] models.transformers.transformer_pika_h3_flex.PikaH3FlexTransformerOutput
