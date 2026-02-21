<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Block Refinement

`BlockRefinementPipeline` performs block-wise iterative refinement over a masked token template, sampling and
committing tokens based on confidence.

## Config defaults

You can set default sampling parameters when creating the pipeline. Passing `None` for a parameter in `__call__`
falls back to `pipe.config`.

```py
from diffusers import BlockRefinementPipeline

pipe = BlockRefinementPipeline(
    model=model,
    tokenizer=tokenizer,
    gen_length=256,
    block_length=32,
    steps=16,
    temperature=0.8,
    sampling_method="multinomial",
)

out = pipe(prompt="Explain gradient descent.")
print(out.texts[0])
```

## Callbacks

Callbacks run after each refinement step and can inspect or override the current tokens.

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
