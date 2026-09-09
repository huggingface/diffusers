<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# TorchTPU

[TorchTPU](https://github.com/google-pytorch/torch_tpu/) is a PyTorch backend for Google's Tensor Processing Units (TPUs), which lets you run Diffusers pipelines on Cloud TPUs (v6e, v5p, etc.) with minimal code changes.

Two execution modes are available:

| Mode | Constant | How to activate | Notes |
|---|---|---|---|
| Strict eager (default) | `EagerMode.DEFER_NEVER` | `import torch_tpu` | Operations dispatched one at a time, asynchronous |
| Compile | — | `pipe.enable_tpu_compile()` | AOT compilation with `TpuBackend` |

Follow the [TorchTPU installation guide](https://github.com/google-pytorch/torch_tpu/). After installation,
`import torch_tpu` registers the `"tpu"` device automatically.

## Eager mode

```python
import gc
import torch
import torch_tpu  # noqa: F401

from diffusers import FluxPipeline

pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)

# 1. Encode on TPU.
pipe.text_encoder.to("tpu")
pipe.text_encoder_2.to("tpu")
with torch.no_grad():
    prompt_embeds, pooled_prompt_embeds, _ = pipe.encode_prompt(
        prompt="a golden retriever surfing a wave, photorealistic",
        prompt_2="a golden retriever surfing a wave, photorealistic",
        device=torch.device("tpu"),
        max_sequence_length=512,
    )

# 2. Free the text encoders — nothing below needs them.
pipe.text_encoder = None
pipe.text_encoder_2 = None
gc.collect()

# 3. Move the transformer and VAE in, then denoise with the precomputed embeddings.
pipe.transformer.to("tpu")
pipe.vae.to("tpu")
image = pipe(
    prompt_embeds=prompt_embeds,
    pooled_prompt_embeds=pooled_prompt_embeds,
    height=1024,
    width=1024,
    num_inference_steps=4,
    guidance_scale=0.0,
).images[0]

image.save("output.png")
```

If the text encoder alone is too large for a single chip(eg. FLUX.2-dev's Mistral-3-Small is ~45GB),
shard it across multiple chips with [`~diffusers.hooks.tensor_parallel.apply_tensor_parallel`], the
same mechanism [`~ModelMixin.enable_parallelism`] uses for the transformer (see [Tensor
parallelism](../training/distributed_inference#tensor-parallelism)). It only requires `model:
torch.nn.Module`, so it works directly on a `transformers.PreTrainedModel` text encoder too, not
just a diffusers `ModelMixin`. The text encoder doesn't define a `_tp_plan`, so supply one: pair
each attention/MLP projection that expands the hidden dimension (`"colwise"`) with the one that
contracts it back (`"rowwise"`), matching the `transformers` model's actual module names.

## Compiled mode

[`enable_tpu_compile`] runs `torch.compile` with `TpuBackend` on each pipeline module that is already on TPU. The first call (warmup) is slow because it compiles. Later calls reuse the compiled graph. Where it's supported, it replaces SDP-based attention with `AttnProcessor` for XLA tracing.

> [!IMPORTANT]
> TorchTPU requires **static shapes** — `torch.compile` is called with `dynamic=False`
> internally. Every time `height`, `width`, or `num_inference_steps` changes, the graph is
> recompiled from scratch. Keep these values constant across all calls after warmup, or call
> [`tpu_warmup`] again before changing them.

```python
import torch
import torch_tpu  # noqa: F401

from diffusers import FluxPipeline

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell",
    torch_dtype=torch.bfloat16,
)
pipe.transformer.to("tpu")
pipe.vae.to("tpu")

pipe.enable_tpu_compile()

# Warmup — triggers static graph compilation.
pipe.tpu_warmup(
    prompt="warmup",
    height=1024,
    width=1024,
    num_inference_steps=4,
    guidance_scale=0.0,
)

# Timed inference reuses the compiled graph.
image = pipe(
    prompt="a golden retriever surfing a wave, photorealistic",
    height=1024,
    width=1024,
    num_inference_steps=4,
    guidance_scale=0.0,
).images[0]

image.save("output.png")
```
