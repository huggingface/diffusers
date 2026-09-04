<!-- Copyright 2025 The HuggingFace Team and SANA-WM Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License. -->

# SanaWMLTX2RefinerTransformer3DModel

The chunk-causal autoregressive refiner DiT used as stage 2 of [`SanaWMPipeline`], driven by
[`SanaWMLTX2Refiner`].

It is architecturally identical to [`LTX2VideoTransformer3DModel`] — same config arguments, same submodules, same
parameter names — so a released LTX-2 checkpoint loads into it unchanged. The forward pass differs:

* only the video stream is run (the audio and audio/video cross-attention branches are skipped),
* self-attention runs against an explicit sliding-window KV cache ([`SanaWMRefinerKVCache`]) holding the attention
  sink plus the recent refined history, so per-block compute is bounded and total refinement cost scales linearly
  with video length,
* the caller supplies the video RoPE, which lets each autoregressive window keep every frame's absolute index in the
  source video (see
  [`SanaWMLTX2RefinerTransformer3DModel.build_rotary_emb_for_absolute_positions`]).

The model can be loaded with:

```python
import torch
from diffusers import SanaWMLTX2RefinerTransformer3DModel

transformer = SanaWMLTX2RefinerTransformer3DModel.from_pretrained(
    "Efficient-Large-Model/SANA-WM_bidirectional-diffusers",
    subfolder="refiner/transformer",
    torch_dtype=torch.bfloat16,
)
```

## SanaWMLTX2RefinerTransformer3DModel

[[autodoc]] SanaWMLTX2RefinerTransformer3DModel

## SanaWMRefinerKVCache

[[autodoc]] models.transformers.transformer_sana_wm_refiner.SanaWMRefinerKVCache

## SanaWMRefinerKVLayerCache

[[autodoc]] models.transformers.transformer_sana_wm_refiner.SanaWMRefinerKVLayerCache

## Transformer2DModelOutput

[[autodoc]] models.modeling_outputs.Transformer2DModelOutput
