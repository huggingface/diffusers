<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# ComponentsManager

The [`ComponentsManager`] is a model registry and management system for Modular Diffusers. It adds and tracks models, stores useful metadata (model size, device placement, adapters), and supports offloading.

This guide will show you how to use [`ComponentsManager`] to manage components and device memory.

## Connect to a pipeline

Create a [`ComponentsManager`] and pass it to a [`ModularPipeline`] with either [`~ModularPipeline.from_pretrained`] or [`~ModularPipelineBlocks.init_pipeline`]. 


<hfoptions id="create">
<hfoption id="from_pretrained">

```py
from diffusers import ModularPipeline, ComponentsManager
import torch

manager = ComponentsManager()
pipe = ModularPipeline.from_pretrained("Tongyi-MAI/Z-Image-Turbo", components_manager=manager)
pipe.load_components(torch_dtype=torch.bfloat16)
```

</hfoption>
<hfoption id="init_pipeline">

```py
from diffusers import ModularPipelineBlocks, ComponentsManager
import torch
manager = ComponentsManager()
blocks = ModularPipelineBlocks.from_pretrained("diffusers/Florence2-image-Annotator", trust_remote_code=True)
pipe= blocks.init_pipeline(components_manager=manager)
pipe.load_components(torch_dtype=torch.bfloat16)
```

</hfoption>
</hfoptions>

Components loaded by the pipeline are automatically registered in the manager. You can inspect them right away.

## Inspect components

Print the [`ComponentsManager`] to see all registered components, including their class, device placement, dtype, memory size, and load ID.

The output below corresponds to the `from_pretrained` example above.

```py
Components:
=======================================================================================================================================================================
Models:
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
Name_ID                      | Class                    | Device: act(exec) | Dtype          | Size      | Load ID                                         | Collection
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
vae_140458257515376          | AutoencoderKL            | cpu               | torch.bfloat16 | 159.87 MB | Tongyi-MAI/Z-Image-Turbo|vae|null|null          | N/A
text_encoder_140458257514752 | Qwen3Model               | cpu               | torch.bfloat16 | 7.49 GB   | Tongyi-MAI/Z-Image-Turbo|text_encoder|null|null | N/A
transformer_140458257515616  | ZImageTransformer2DModel | cpu               | torch.bfloat16 | 11.46 GB  | Tongyi-MAI/Z-Image-Turbo|transformer|null|null  | N/A
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------

Other Components:
------------------------------------------------------------------------
ID                        | Class                           | Collection
------------------------------------------------------------------------
scheduler_140461023555264 | FlowMatchEulerDiscreteScheduler | N/A
tokenizer_140458256346432 | Qwen2Tokenizer                  | N/A
------------------------------------------------------------------------
```

The table shows models (with device, dtype, and memory info) separately from other components like schedulers and tokenizers. If any models have LoRA adapters, IP-Adapters, or quantization applied, that information is displayed in an additional section at the bottom.

## Offloading

The [`~ComponentsManager.enable_auto_cpu_offload`] method is a global offloading strategy that works across all models regardless of which pipeline is using them. Once enabled, you don't need to worry about device placement if you add or remove components.

```py
manager.enable_auto_cpu_offload(device="cuda")
```

All models begin on the CPU and [`ComponentsManager`] moves them to the appropriate device right before they're needed, and moves other models back to the CPU when the incoming model wouldn't fit. Each offloading decision checks the memory actually available on the device at that moment and keeps `memory_reserve` (3GB unless you change it) of it free as headroom for activations, which scale with resolution, batch size, and sequence length rather than with the number of models — image models around 1024px are comfortable with a few GB, while video models need more.

```py
manager.enable_auto_cpu_offload(device="cuda", memory_reserve="1GB")
```

Set `memory_reserve=0` to keep as much on the device as possible. If a forward pass runs out of memory anyway, the manager offloads the smallest model on the device and retries it, escalating one model at a time until it fits — pass `retry_on_oom=False` to raise the error instead. A model that still doesn't fit once everything else is offloaded is too large for the device on its own; use [group offloading](../optimization/memory#group-offloading) for it.

Components added while offloading is enabled join the managed set without disturbing it: the new component starts on CPU and everything already resident stays where it is; removing a component likewise detaches only that component.

### Inspecting what the offloader did

Every move is recorded. Print [`ComponentsManager.offload_record`] after a run to see the sequence, what each move cost, and where a forward pass ran out of memory.

```py
manager.enable_auto_cpu_offload(device="cuda")
pipe(prompt="a cat")
print(manager.offload_record)
```

On a 20GB card, the Z-Image pipeline from the example above records this:

```py
# | Onload                                 | Offloaded                              | Available | Reason
--------------------------------------------------------------------------------------------------------
1 | text_encoder_140458257514752 (7.49 GB) | -                                      | 18.02 GB  |
2 | transformer_140458257515616 (11.46 GB) | text_encoder_140458257514752 (7.49 GB) | 10.38 GB  |
3 | vae_140458257515376 (159.87 MB)        | -                                      | 6.36 GB   |
--------------------------------------------------------------------------------------------------------
```

Each row is one decision: the model that loaded, what was offloaded to make room for it, and the free device memory (`Available`, read just before the decision's first move) it was based on. Here the transformer found 10.38 GB free — not enough for its 11.46 GB of weights while keeping the 3GB `memory_reserve` — so the text encoder was offloaded first. The VAE then fit into the 6.36 GB left next to the transformer, so it loaded without pushing anything off. The `Reason` column only speaks on moves an onload did not cause — an OOM retry (`oom_retry:<model>`), disabling offloading — which appear as their own rows. To watch the moves live as they happen instead, enable info logging with `diffusers.logging.set_verbosity_info()`.

A model appearing repeatedly in this table is thrashing — it is being offloaded and re-loaded every step, which costs a PCIe transfer each way. That usually means `memory_reserve` is too large (models are pushed off that would have fit) or too small (each step ends in an OOM retry).

To find the right value, measure your workflow once: run it at the resolution, batch size, and sequence length you intend to use (activations scale with all three), and read the device's peak memory afterwards:

```py
manager.enable_auto_cpu_offload(device="cuda")
pipe(prompt="a cat")

torch.cuda.max_memory_allocated()  # peak of the run: resident weights + activations
```

The peak is the heaviest moment of the run: the weights resident at that point plus the running model's activations. The record shows the weights side — which models were on the device together and their sizes — so subtracting them from the peak gives the activation headroom, and rounding that up generously gives the `memory_reserve`. In the 20GB run above the peak reads 14.15 GB and falls in the decode stage, where the transformer (11.46 GB) and the VAE (159.87 MB) are both resident; the ~2.5 GB left over is what the VAE's decode needed on top of the weights, so the default 3GB reserve covers this workload.

Call [`~ComponentsManager.disable_auto_cpu_offload`] to disable offloading.

```py
manager.disable_auto_cpu_offload()
```
