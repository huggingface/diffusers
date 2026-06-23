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

### Temperature annealing

By default the pipeline anneals the sampling temperature linearly from `t_max` (`0.8`) on the first denoising step
down to `t_min` (`0.4`), matching the sampler the released checkpoint was tuned with (sharper sampling as it
denoises). It matters most for stochastic schedulers like `EntropyBoundScheduler` at a loose `entropy_bound`, where
flat high-temperature sampling can degrade. Set both `t_min` and `t_max` to `None` to instead use a flat `temperature`
(`0.0` for greedy):

```py
output = pipe(prompt="Why is the sky blue?", gen_length=256)                       # annealed 0.8 -> 0.4 (default)
output = pipe(prompt="Why is the sky blue?", gen_length=256, t_min=None, t_max=None, temperature=0.0)  # greedy
```

### Predictor-corrector sampling

`DiscreteDDIMScheduler` supports the leave-one-out predictor-corrector of [Reparameterizing Uniform Diffusion Models](https://huggingface.co/papers/2605.22765). It refines the canvas with `corrector_steps` Gibbs sweeps that resample the least-confident positions from the one-coordinate conditional of the noisy marginal, which leaves that marginal invariant and improves generation at no extra training cost. It works directly on the released checkpoint: for uniform diffusion the denoiser and the leave-one-out posterior are interchangeable in closed form, so the corrector recovers the leave-one-out quantities it needs without any retraining.

The corrector sweeps are folded into the `num_inference_steps` budget rather than added on top: the pipeline runs fewer predictor steps and spends the freed forwards on correctors, so the total number of model forwards stays `num_inference_steps` and the predictor-corrector costs the same as plain ancestral sampling.

```py
from diffusers import DiscreteDDIMScheduler

pipe.scheduler = DiscreteDDIMScheduler(corrector_steps=2, corrector_k=12)
output = pipe(prompt="Why is the sky blue?", gen_length=256, num_inference_steps=48)
print(output.texts[0])
```

## PEFT adapters

The denoiser is a 🤗 Transformers model, so adapters are loaded through its native [PEFT](https://huggingface.co/docs/peft) integration rather than the diffusers `load_lora_weights` API. Because that integration is adapter-type-agnostic, the same calls load LoRA, DoRA, or any other PEFT adapter (e.g. the output of TRL's `SFTTrainer`). Manage adapters on the model component directly:

```py
pipe.model.load_adapter("path/to/adapter", adapter_name="sft")  # LoRA, DoRA, ...
pipe.model.set_adapter("sft")
output = pipe(prompt="Why is the sky blue?", gen_length=256)

pipe.model.disable_adapters()  # run the base model
pipe.model.delete_adapter("sft")
```

Adapters stay active and unmerged: DiffusionGemma ties the encoder and decoder base weights, so fusing an adapter into them would corrupt both branches.

## Static cache and compilation

The pipeline prefills the encoder once per block into a reusable cache (a `DynamicCache` by default). Pass
`cache_implementation="static"` to use a fixed-shape `StaticCache` instead, whose shapes let you `torch.compile` the
decoder with cudagraphs for a further speedup (the pipeline marks each step and clones the logits so cudagraph memory
is not overwritten):

```py
pipe.model.model.decoder = torch.compile(pipe.model.model.decoder, mode="reduce-overhead", fullgraph=True)
output = pipe(prompt="Why is the sky blue?", gen_length=256, cache_implementation="static")
```

## Adaptive stopping

A block usually converges before all `num_inference_steps` are spent, so by default the pipeline leaves a block's
denoising loop early once every example's argmax prediction is stable for `stability_threshold` steps and the mean
per-token entropy falls below `confidence_threshold` (`0.005`, the value used by the released checkpoint). This roughly
halves the number of decoder forwards at matched quality and is the largest single throughput lever. Pass
`confidence_threshold=None` to always run the full `num_inference_steps`:

```py
output = pipe(prompt="Why is the sky blue?", gen_length=256, confidence_threshold=None)  # disable adaptive stopping
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
