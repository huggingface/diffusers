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
    num_inference_steps=48,
    temperature=0.0,
)
print(output.texts[0])
```

`num_inference_steps` is the number of denoising steps per canvas (48 matches the released checkpoint); fewer steps are
faster but lower quality. For multimodal prompts, pass an `image` alongside the `prompt` (or put the image content in a
raw `messages` conversation), and the processor turns it into the model's image inputs automatically.

## Schedulers

The scheduler is the sampler that denoises each canvas, and it is interchangeable: swap it to change the sampling
strategy without touching anything else. Three schedulers are available:

- `BlockRefinementScheduler` (default): commits the most confident tokens each step (above `threshold`, plus an even
  per-step quota) and renoises the rest. `editing_threshold` additionally lets it re-edit already committed tokens.
- `DiscreteDDIMScheduler`: samples each position from the exact discrete posterior of the uniform corruption process
  (D3PM). It is parameter free, and the final step deterministically commits the predicted tokens.
- `EntropyBoundScheduler`: commits the lowest-entropy positions whose joint entropy stays under `entropy_bound`, so
  roughly independent tokens are accepted together.

```py
from diffusers import DiscreteDDIMScheduler, EntropyBoundScheduler

pipe.scheduler = DiscreteDDIMScheduler()
# or: pipe.scheduler = EntropyBoundScheduler(entropy_bound=0.1)
output = pipe(prompt="Why is the sky blue?", gen_length=256, num_inference_steps=48)
print(output.texts[0])
```

Scheduler-specific sampling knobs (the block-refinement `threshold`/`top_k`, the entropy bound, ...) are set on the
scheduler config:

```py
from diffusers import BlockRefinementScheduler

pipe.scheduler = BlockRefinementScheduler.from_config(pipe.scheduler.config, threshold=0.9)
```

### Predictor-corrector sampling

`DiscreteDDIMScheduler` supports the leave-one-out predictor-corrector of [Reparameterizing Uniform Diffusion Models](https://huggingface.co/papers/2605.22765). After each predictor step the pipeline runs `corrector_steps` Gibbs sweeps that resample the least-confident positions from the one-coordinate conditional of the noisy marginal, which leaves that marginal invariant and improves generation at no extra training cost. It works directly on the released checkpoint: for uniform diffusion the denoiser and the leave-one-out posterior are interchangeable in closed form, so the corrector recovers the leave-one-out quantities it needs without any retraining.

```py
from diffusers import DiscreteDDIMScheduler

pipe.scheduler = DiscreteDDIMScheduler(corrector_steps=2, corrector_k=12)
output = pipe(prompt="Why is the sky blue?", gen_length=256, num_inference_steps=48)
print(output.texts[0])
```

## Static cache and compilation

The pipeline prefills the encoder once per block into a reusable cache (a `DynamicCache` by default). Pass
`cache_implementation="static"` to use a fixed-shape `StaticCache` instead, whose shapes let you `torch.compile` the
decoder for a further speedup:

```py
pipe.model.model.decoder = torch.compile(pipe.model.model.decoder, fullgraph=True)
output = pipe(prompt="Why is the sky blue?", gen_length=256, cache_implementation="static")
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
