<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# DFlash

[DFlash](https://huggingface.co/collections/z-lab/dflash) is a block-diffusion speculative decoding scheme. A small
diffusion *draft* model proposes a block of tokens conditioned on hidden features extracted from intermediate layers
of a frozen *target* causal LM; the target then verifies the proposed block in a single forward pass and accepts the
longest matching prefix. The draft model is shared with the target's tokenizer, so no calibration is needed.

`DFlashPipeline` ties the two models together: prefill on the target, draft a block, verify against the target's
posterior via [`DFlashTokenDiffusionScheduler`], commit the accepted prefix and the next-token resample, and repeat
until `max_new_tokens` or a stop token. Compatible draft/target pairs include `z-lab/Qwen3-8B-DFlash-b16` with
`Qwen/Qwen3-8B`, and `z-lab/Qwen3.5-4B-DFlash` with `Qwen/Qwen3.5-4B` (the latter is a hybrid-attention target — see
the rollback note below).

## Usage

```py
import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from diffusers import DFlashPipeline

# Draft ships custom modeling code via `auto_map` — `trust_remote_code=True` is required.
draft = AutoModel.from_pretrained(
    "z-lab/Qwen3.5-4B-DFlash", trust_remote_code=True, dtype=torch.bfloat16, device_map="auto"
)
target = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-4B", dtype=torch.bfloat16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B")

pipe = DFlashPipeline(draft_model=draft, target_model=target, tokenizer=tokenizer)
output = pipe(
    prompt="What is 2 + 2? Answer in one sentence.",
    max_new_tokens=128,
    temperature=0.0,
    chat_template_kwargs={"enable_thinking": False},
)
print(output.texts[0])
```

`DFlashPipeline` currently runs `batch_size=1` only. Multi-prompt batching requires per-row partial-accept tracking
and is not yet supported.

## Hybrid-attention targets

For target models with linear-attention layers (e.g. Qwen3.5's gated-delta-net), `DynamicCache.crop()` is a
documented no-op on those layers (see `transformers.cache_utils.LinearAttentionCacheLayerMixin.crop`), so a
partial-accept block would otherwise leak rejected speculative tokens into the recurrent state. The pipeline
detects linear-attention caches via [`DFlashTokenDiffusionScheduler.cache_has_linear_attention`] and uses a
snapshot/restore + accepted-prefix re-forward pattern to advance both layer types cleanly. This adds one extra
target forward per partial-accept block on hybrid targets; full-attention targets use a plain `cache.crop()`.

## Callbacks

Callbacks run after each block-verify step. Pass `callback_on_step_end_tensor_inputs` to select which tensors are
included in `callback_kwargs`. Allowed keys: `block_output_ids` (the drafted block), `draft_logits`,
`accepted_length`, `next_token`, and `output_ids` (the running output buffer). Return `{"output_ids": ...}` from the
callback to replace the buffer.

```py
def on_step_end(pipe, step, timestep, callback_kwargs):
    output_ids = callback_kwargs["output_ids"]
    return {"output_ids": output_ids}

out = pipe(
    prompt="...",
    callback_on_step_end=on_step_end,
    callback_on_step_end_tensor_inputs=["output_ids"],
)
```

## DFlashPipeline
[[autodoc]] DFlashPipeline
    - all
    - __call__

## DFlashPipelineOutput
[[autodoc]] pipelines.DFlashPipelineOutput
