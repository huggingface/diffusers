<!-- Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License. -->

# PAN2

PAN2 is a foundation world model that simulates how a scene evolves under a time-grounded plan of actions and camera movements. Its 29B generative simulator is a dual-stream multimodal diffusion transformer, integrated here; the [reasoning backbone](https://huggingface.co/IFM/PAN2#reasoning-backbone) that writes its prompts is not.

PAN2 is integrated as [Modular Diffusers](../../modular_diffusers/overview) blocks only. [`PAN2Blocks`] supports two workflows:

| Workflow | Inputs |
|---|---|
| `t2v` | `prompt` |
| `i2v` | `prompt`, `image` |

```py
import torch
from diffusers import ModularPipeline
from diffusers.utils import export_to_video, load_image

pipe = ModularPipeline.from_pretrained("IFM/PAN2")
pipe.load_components(dtype=torch.bfloat16)
pipe.enable_safety_checker()
pipe.to("cuda")

prompt = "A red fox trots through fresh snow in a pine forest at dawn."

video = pipe(prompt=prompt, generator=torch.Generator().manual_seed(0), output="videos")[0]
export_to_video(video, "pan2_t2v.mp4", fps=24)

image = load_image("path/to/first_frame.png")
video = pipe(prompt=prompt, image=image, generator=torch.Generator().manual_seed(0), output="videos")[0]
export_to_video(video, "pan2_i2v.mp4", fps=24)
```

To load a specific revision of the checkpoint, pass `revision` to `load_components` as well, since `from_pretrained` only reads the pipeline index at that revision: `pipe.load_components(dtype=torch.bfloat16, revision="<commit>")`.

Classifier-free guidance is configured on the guider (`guidance_scale=3.0` by default):

```py
pipe.update_components(guider=pipe.guider.new(guidance_scale=5.0))
```

Every call takes a single prompt, which is encoded without padding. The transformer therefore never needs an attention mask, and every [attention backend](../../optimization/attention_backends) is available:

```py
pipe.transformer.set_attention_backend("flash_hub")  # FlashAttention-2
```

## Safety checker

PAN2 runs guardrails on the prompt before generation and on every generated video after it, and a blocked prompt or video raises a `ValueError`. The checks are on by default, so the pipeline raises until a checker is loaded with `enable_safety_checker()`, which needs the [PAN2 guardrail](https://github.com/ifm-ai/pan2-guardrail) package:

```bash
pip install git+https://github.com/ifm-ai/pan2-guardrail
```

## Context parallelism

[`PAN2Transformer3DModel`] supports [context parallelism](../../training/distributed_inference#context-parallelism). The text and video token counts are usually not divisible by the number of devices (the number of text tokens depends on the prompt), so use `ulysses_anything` or `ring_anything` for inference. Plain `ulysses_degree` / `ring_degree`, for example for training, requires both token counts to be divisible by the context parallel size.

```py
# torchrun --nproc_per_node=2 pan2_context_parallel.py
import torch
import torch.distributed as dist
from diffusers import ContextParallelConfig, ModularPipeline

dist.init_process_group("nccl")
rank = dist.get_rank()
torch.cuda.set_device(rank)

pipe = ModularPipeline.from_pretrained("IFM/PAN2")
pipe.load_components(dtype=torch.bfloat16)
pipe.enable_safety_checker()
pipe.to(f"cuda:{rank}")
pipe.transformer.enable_parallelism(config=ContextParallelConfig(ulysses_degree=2, ulysses_anything=True))

video = pipe(prompt="A red fox trots through fresh snow.", generator=torch.Generator().manual_seed(0), output="videos")[0]
dist.destroy_process_group()
```

## PAN2ModularPipeline

[[autodoc]] PAN2ModularPipeline

## PAN2Blocks

[[autodoc]] PAN2Blocks
