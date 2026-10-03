<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# LLaDA2

[LLaDA2](https://huggingface.co/collections/inclusionAI/llada21) is a family of discrete diffusion language models
that generate text through block-wise iterative refinement. Instead of autoregressive token-by-token generation,
LLaDA2 starts with a fully masked sequence and progressively unmasks tokens by confidence over multiple refinement
steps.

## Usage

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from diffusers import BlockRefinementScheduler, LLaDA2Pipeline

model_id = "inclusionAI/LLaDA2.1-mini"
model = AutoModelForCausalLM.from_pretrained(
    model_id, trust_remote_code=True, dtype=torch.bfloat16, device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
scheduler = BlockRefinementScheduler()

pipe = LLaDA2Pipeline(model=model, scheduler=scheduler, tokenizer=tokenizer)
output = pipe(
    prompt="Write a short poem about the ocean.",
    gen_length=256,
    block_length=32,
    num_inference_steps=32,
    threshold=0.7,
    editing_threshold=0.5,
    max_post_steps=16,
    temperature=0.0,
)
print(output.texts[0])
```

## Callbacks

Callbacks run after each refinement step. Pass `callback_on_step_end_tensor_inputs` to select which tensors are
included in `callback_kwargs`. In the current implementation, `block_x` (the sequence window being refined) and
`transfer_index` (mask-filling commit mask) are provided; return `{"block_x": ...}` from the callback to replace the
window.

```py
def on_step_end(pipe, step, timestep, callback_kwargs):
    block_x = callback_kwargs["block_x"]
    # Inspect or modify `block_x` here.
    return {"block_x": block_x}

out = pipe(
    prompt="Write a short poem.",
    callback_on_step_end=on_step_end,
    callback_on_step_end_tensor_inputs=["block_x"],
)
```

## Recommended parameters

LLaDA2.1 models support two modes:

| Mode | `threshold` | `editing_threshold` | `max_post_steps` |
|------|-------------|---------------------|------------------|
| Quality | 0.7 | 0.5 | 16 |
| Speed | 0.5 | `None` | 16 |

Pass `editing_threshold=None`, `0.0`, or a negative value to turn off post-mask editing.

For LLaDA2.0 models, disable editing by passing `editing_threshold=None` or `0.0`.

For all models: `block_length=32`, `temperature=0.0`, `num_inference_steps=32`.

## LLaDA2Pipeline
[[autodoc]] LLaDA2Pipeline
    - all
    - __call__

## LLaDA2PipelineOutput
[[autodoc]] pipelines.LLaDA2PipelineOutput
