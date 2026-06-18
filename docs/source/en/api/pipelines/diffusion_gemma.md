<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# DiffusionGemma

DiffusionGemma is a block-diffusion encoder-decoder language model. A causal encoder reads the clean prompt (and any
previously generated blocks) into a KV cache, and a bidirectional decoder denoises a fixed-size "canvas" of
`canvas_length` tokens by cross-attending to that cache. Generation alternates an outer autoregressive loop over
canvases with an inner denoising loop, where each step samples candidate tokens, commits the most confident ones via
[`BlockRefinementScheduler`] in uniform corruption mode, and renoises the rest. The model itself lives in
`transformers` as `DiffusionGemmaForBlockDiffusion`.

## Usage

```py
import torch
from transformers import AutoProcessor, DiffusionGemmaForBlockDiffusion

from diffusers import BlockRefinementScheduler, DiffusionGemmaPipeline

model_id = "google/diffusiongemma-26B-A4B-it"
model = DiffusionGemmaForBlockDiffusion.from_pretrained(model_id, dtype=torch.bfloat16, device_map="auto")
processor = AutoProcessor.from_pretrained(model_id)
scheduler = BlockRefinementScheduler()

pipe = DiffusionGemmaPipeline(model=model, scheduler=scheduler, processor=processor)
output = pipe(
    prompt="Why is the sky blue?",
    gen_length=256,
    num_inference_steps=32,
    temperature=0.0,
)
print(output.texts[0])
```

## Callbacks

Callbacks run after each denoising step. Pass `callback_on_step_end_tensor_inputs` to select which tensors are
included in `callback_kwargs`; `canvas` (the current block tokens) and `logits` are available. Return `{"canvas": ...}`
from the callback to replace the canvas.

```py
def on_step_end(pipe, step, timestep, callback_kwargs):
    canvas = callback_kwargs["canvas"]
    # Inspect or modify `canvas` here.
    return {"canvas": canvas}


out = pipe(
    prompt="Why is the sky blue?",
    callback_on_step_end=on_step_end,
    callback_on_step_end_tensor_inputs=["canvas"],
)
```

## DiffusionGemmaPipeline
[[autodoc]] DiffusionGemmaPipeline
    - all
    - __call__

## DiffusionGemmaPipelineOutput
[[autodoc]] pipelines.DiffusionGemmaPipelineOutput
