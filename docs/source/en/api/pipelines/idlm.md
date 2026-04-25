<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# I-DLM

[Introspective Diffusion Language Models (I-DLM)](https://arxiv.org/abs/2604.11035) are diffusion LLMs that recover the AR self-consistency property (the "introspective acceptance rate" of ~0.98) via strict causal attention, Dream-style logit shift, and all-masked training. At inference time, *Introspective Strided Decoding* (ISD) runs a single forward per round that both **verifies** previously-proposed speculative tokens (against the anchor distribution `p` at now-visible clean positions) and **generates** the next batch of specs (from the MASK-position proposal distribution `q`). Acceptance via `min(1, p(x) / (alpha * q(x)))` guarantees the output matches the base AR distribution.

Published I-DLM checkpoints (e.g. [`yifanyu/I-DLM-8B`](https://huggingface.co/yifanyu/I-DLM-8B)) are finetuned from standard Qwen3 weights and load via `AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)`.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from diffusers import IDLMPipeline, IDLMBlockDiffusionScheduler

model = AutoModelForCausalLM.from_pretrained("yifanyu/I-DLM-8B", trust_remote_code=True, dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained("yifanyu/I-DLM-8B", trust_remote_code=True)

scheduler = IDLMBlockDiffusionScheduler(gen_block_size=4)
pipe = IDLMPipeline(model=model, tokenizer=tokenizer, scheduler=scheduler)
out = pipe(prompt="Prove that sqrt(2) is irrational.", max_new_tokens=256, use_chat_template=True)
print(out.texts[0])
```

## IDLMPipeline
[[autodoc]] IDLMPipeline
    - all
    - __call__

## IDLMPipelineOutput
[[autodoc]] pipelines.IDLMPipelineOutput
