<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# DFlash

[DFlash: Block Diffusion for Flash Speculative Decoding](https://huggingface.co/papers/2602.06036) is by Jian Chen, Yesheng Liang, and Zhijian Liu.

The abstract from the paper is:

*Autoregressive large language models (LLMs) deliver strong performance but require inherently sequential decoding, leading to high inference latency and poor GPU utilization. Speculative decoding mitigates this bottleneck by using a fast draft model whose outputs are verified in parallel by the target LLM. However, existing methods still rely on autoregressive drafting, which remains sequential and constrains practical speedups. Diffusion LLMs offer a promising alternative by enabling parallel generation, but current diffusion models typically underperform compared with autoregressive models. In this paper, we introduce DFlash, a speculative decoding framework that employs a lightweight block diffusion model for parallel drafting. We show that speculative decoding provides a natural and effective setting for diffusion models. By generating draft tokens in a single forward pass, DFlash enables efficient drafting, and by conditioning the draft model on context features extracted from the target model, it achieves high-quality drafts with higher acceptance rates. Experiments show that DFlash achieves over 6× lossless acceleration across a range of models and tasks, delivering up to 2.5× higher speedup than the state-of-the-art speculative decoding method EAGLE-3.*

`DFlashPipeline` ties the two models together: prefill on the target, draft a block, verify against the target's
posterior via [`DFlashTokenDiffusionScheduler`], commit the accepted prefix and the next-token resample, and repeat
until `max_new_tokens` or a stop token. Pretrained draft/target pairs are available in the
[z-lab/dflash collection](https://huggingface.co/collections/z-lab/dflash); the canonical pair is
`z-lab/Qwen3-8B-DFlash-b16` with `Qwen/Qwen3-8B`.

## Usage

```py
import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from diffusers import DFlashPipeline

# Draft ships custom modeling code via `auto_map` — `trust_remote_code=True` is required.
draft = AutoModel.from_pretrained(
    "z-lab/Qwen3-8B-DFlash-b16", trust_remote_code=True, dtype=torch.bfloat16, device_map="auto"
)
target = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", dtype=torch.bfloat16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

pipe = DFlashPipeline(draft_model=draft, target_model=target, tokenizer=tokenizer)
output = pipe(
    prompt="What is 2 + 2? Answer in one sentence.",
    max_new_tokens=256,
    temperature=0.0,
    chat_template_kwargs={"enable_thinking": False},
)
print(output.texts[0])
```

> **Note:** Qwen3 is a reasoning model and generates a `<think>...</think>` block before the answer. Pass
> `chat_template_kwargs={"enable_thinking": False}` to suppress thinking mode, or increase `max_new_tokens`
> (e.g. `8192`) when you want the full reasoning trace.

`DFlashPipeline` currently runs `batch_size=1` only. Multi-prompt batching requires per-row partial-accept tracking
and is not yet supported.

## Hybrid-attention targets

Target models with linear-attention layers (e.g. Qwen3.5's gated-delta-net) are not yet supported. For those
targets, `DynamicCache.crop()` silently no-ops on linear-attention layers, which would leak rejected speculative
tokens into the recurrent state. The current pipeline targets full-attention models only (e.g. `Qwen/Qwen3-8B`).

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
