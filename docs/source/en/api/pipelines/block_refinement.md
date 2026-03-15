<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Block Refinement

`BlockRefinementPipeline` generates text through block-wise iterative refinement. It starts with a fully masked
sequence and processes it in fixed-size blocks. Within each block, the model predicts all tokens simultaneously,
commits the most confident ones, and re-masks the rest for further refinement. This is the core pipeline behind
[`LLaDA2Pipeline`].

## Usage

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from diffusers import BlockRefinementPipeline, BlockRefinementScheduler

model_id = "inclusionAI/LLaDA2.1-mini"
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, dtype=torch.bfloat16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
scheduler = BlockRefinementScheduler()

pipe = BlockRefinementPipeline(model=model, scheduler=scheduler, tokenizer=tokenizer)
out = pipe(
    prompt="Explain gradient descent.",
    gen_length=256,
    block_length=32,
    steps=32,
    temperature=0.0,
    mask_token_id=tokenizer.mask_token_id,
)
print(out.texts[0])
```

## Callbacks

Callbacks run after each refinement step and can inspect or modify the current tokens.

```py
def on_step_end(pipe, step, timestep, callback_kwargs):
    cur_x = callback_kwargs["cur_x"]
    # Inspect or modify `cur_x` here.
    return {"cur_x": cur_x}

out = pipe(
    prompt="Write a short poem.",
    callback_on_step_end=on_step_end,
    callback_on_step_end_tensor_inputs=["cur_x"],
)
```

## BlockRefinementPipeline
[[autodoc]] BlockRefinementPipeline
    - all
    - __call__

## BlockRefinementPipelineOutput
[[autodoc]] pipelines.BlockRefinementPipelineOutput
