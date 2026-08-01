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
pipe.load_components(dtype=torch.bfloat16)
```

</hfoption>
<hfoption id="init_pipeline">

```py
from diffusers import ModularPipelineBlocks, ComponentsManager
import torch
manager = ComponentsManager()
blocks = ModularPipelineBlocks.from_pretrained("diffusers/Florence2-image-Annotator", trust_remote_code=True)
pipe= blocks.init_pipeline(components_manager=manager)
pipe.load_components(dtype=torch.bfloat16)
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

The reserve is an estimate, so a forward pass can still run out of memory. By default the manager recovers on its own: it offloads the smallest model on the device and retries the forward pass, escalating one model at a time until it fits — pass `retry_on_oom=False` to raise the error instead. A model that still doesn't fit once everything else is offloaded is too large for the device on its own; use [group offloading](../optimization/memory#group-offloading) for it.

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
# | Onload                                 | Offloaded                              | Available | Peak     | Reason
-------------------------------------------------------------------------------------------------------------------
1 | text_encoder_129824856400544 (7.49 GB) | -                                      | 18.02 GB  | 8.50 KB  |
2 | transformer_129824854851072 (11.46 GB) | text_encoder_129824856400544 (7.49 GB) | 10.38 GB  | 7.77 GB  |
3 | vae_129824905851968 (159.87 MB)        | -                                      | 6.36 GB   | 12.07 GB |
-------------------------------------------------------------------------------------------------------------------
```

Each row is one decision: the model that loaded, what was offloaded to make room for it, and the memory picture (`Available` and `Peak`, read just before the decision's first move) it was based on. Here the transformer found 10.38 GB free — not enough for its 11.46 GB of weights while keeping the 3GB `memory_reserve` — so the text encoder was offloaded first. The VAE then fit into the 6.36 GB left next to the transformer, so it loaded without pushing anything off. The `Reason` column stays empty for these planned moves; it is filled when a model is offloaded for any other reason: `oom_retry:<model>` when a forward pass ran out of memory and offloading was the rescue (covered below), or `offloading_disabled` when [`~ComponentsManager.disable_auto_cpu_offload`] moves everything back to CPU. To watch the moves live as they happen instead, enable info logging with `diffusers.logging.set_verbosity_info()`.

A model appearing repeatedly in this table is thrashing — it is being offloaded and re-loaded every step, which costs a PCIe transfer each way. That usually means `memory_reserve` is too large (models are pushed off that would have fit) or too small (each step ends in an OOM retry). It can also mean the default strategy is making the wrong call for your workload: a model that runs again inside the same denoise loop should stay resident even when evicting it looks fine on memory alone. That is a case for a custom strategy, which the next section covers.

### Writing a custom offload strategy

Every decision in the record above was made by an *offload strategy*: a callable consulted each time a model needs to move onto the device. It receives the hooks of the models currently on the device (`hooks`), the model about to load (`model_id`, `model`), and the `execution_device`, and returns which of the resident hooks to offload first.

The default strategy, `AutoOffloadStrategy`, is what produced the records so far: it works out how much room the incoming model needs — its weights plus `memory_reserve`, measured against the memory currently free — and frees the smallest combination of resident models that covers it, or nothing if the model already fits.

To make different decisions, write your own callable with the same signature and set it with [`~ComponentsManager.set_offload_strategy`]. For example, we can create a sequential strategy with one line:

```py
class OffloadEverything:
    def __call__(self, hooks, model_id, model, execution_device):
        return hooks  # offload every other resident model before each load
```

This strategy offloads and onloads models in the sequence they are called (pretty much what [`~DiffusionPipeline.enable_model_cpu_offload`] does for standard pipelines). Use it like this:

```py
manager.enable_auto_cpu_offload(device="cuda")
manager.set_offload_strategy(OffloadEverything())

pipe(prompt="a cat")
print(manager.offload_record)
```

The record verifies it did what it promised — every load now evicts the previous resident, even on a large card with plenty of free memory:

```py
# | Onload                                 | Offloaded                              | Available | Peak     | Reason
-------------------------------------------------------------------------------------------------------------------
1 | text_encoder_136361472535648 (7.49 GB) | -                                      | 78.57 GB  | 8.50 KB  |
2 | transformer_136361472167088 (11.46 GB) | text_encoder_136361472535648 (7.49 GB) | 70.93 GB  | 7.77 GB  |
3 | vae_136361468822832 (159.87 MB)        | transformer_136361472167088 (11.46 GB) | 66.91 GB  | 12.07 GB |
-------------------------------------------------------------------------------------------------------------------
```

The strategy is plain Python over the resident hooks — each carries its `model_id` (the component name plus a unique suffix) and its `model` — so it can encode policy the memory numbers can't express. Say your pipeline has two DiTs that alternate every denoising step: evicting one to make room for the other is exactly the thrashing described above, so keep them resident together by never offloading a `dit` for a `dit`, whatever the memory picture says:

```py
class KeepDitsTogether:
    def __call__(self, hooks, model_id, model, execution_device):
        if model_id.startswith("dit"):
            return [hook for hook in hooks if not hook.model_id.startswith("dit")]
        return hooks
```

### Finding the bottleneck with the peak column

`Peak` is the device's peak allocated memory as of that moment, so reading down the column tells you *when* the run's heaviest moments happened. In the first record above: nothing had run yet when the text encoder loaded (8.50 KB); by the transformer's turn the peak was the text encoder's weights plus its activations (7.77 GB); by the VAE's turn the transformer and its denoising steps had pushed it to 12.07 GB. The end of the run isn't a row (nothing loads after the VAE), so read the final peak directly — it's the same counter:

```py
torch.cuda.max_memory_allocated()  # 14.15 GB — the peak of the whole run
```

That last jump, from 12.07 GB to 14.15 GB with the transformer (11.46 GB) and VAE (159.87 MB) resident, is the decode stage: the ~2.4 GB it needed on top of the weights is exactly the headroom `memory_reserve` exists to protect, so the default 3GB covers this workload. That is the general recipe for sizing the reserve: run your workflow once at the resolution, batch size, and sequence length you intend to use (activations scale with all three), find the largest jump in the peak column, subtract the weights that loaded in that stretch, and round up generously.

### When a forward pass still runs out of memory

The strategy plans with weights. It frees enough room for the incoming model's weights plus `memory_reserve` — but it cannot see how much the forward pass it just made room for is about to allocate, so the reserve *is* the activation budget. When the reserve under-covers the real need, the decisions all succeed and the forward pass itself runs out of memory; that is when the OOM retry from earlier steps in, offloading the smallest resident model and rerunning the forward pass until it fits. In other words: an `oom_retry` row in the record means `memory_reserve` was set smaller than what the activations actually needed.

Here is the same Z-Image workflow on a 13.5GB card with `memory_reserve=0`:

```py
# | Onload                                 | Offloaded                              | Available | Peak     | Reason
-------------------------------------------------------------------------------------------------------------------
1 | text_encoder_133045385135840 (7.49 GB) | -                                      | 12.90 GB  | 8.50 KB  |
2 | transformer_133045391586896 (11.46 GB) | text_encoder_133045385135840 (7.49 GB) | 5.25 GB   | 7.77 GB  |
3 | vae_133044657178144 (159.87 MB)        | -                                      | 1.29 GB   | 12.07 GB |
4 | -                                      | transformer_133045391586896 (11.46 GB) | 693.77 MB | 12.65 GB | oom_retry:vae_133044657178144
-------------------------------------------------------------------------------------------------------------------
```

The first three rows are ordinary: the text encoder yields to the transformer, and with a reserve of 0 the VAE is allowed to squeeze into the 1.29 GB next to it. But decode needs ~2.4 GB of activations (the jump we measured earlier), so the forward pass dies, and row 4 is the rescue: the transformer — the smallest (and only) other resident model — is pushed off mid-run and decode reruns with the whole card to itself. The run completes, but it paid for the bad plan at the worst moment: an unplanned 11.46 GB transfer in the middle of decode, and a re-load next time the transformer runs.

The record already told us the right value: decode's ~2.4 GB of activations need covering, so the default `"3GB"` reserve fits this workload. Same card, `memory_reserve="3GB"`:

```py
# | Onload                                 | Offloaded                              | Available | Peak     | Reason
-------------------------------------------------------------------------------------------------------------------
1 | text_encoder_129850895513456 (7.49 GB) | -                                      | 12.90 GB  | 8.50 KB  |
2 | transformer_129850972639344 (11.46 GB) | text_encoder_129850895513456 (7.49 GB) | 5.25 GB   | 7.77 GB  |
3 | vae_129850924782688 (159.87 MB)        | transformer_129850972639344 (11.46 GB) | 1.29 GB   | 12.07 GB |
-------------------------------------------------------------------------------------------------------------------
```

Now the eviction happens *before* decode instead of in the middle of it — the VAE's row offloads the transformer up front because 1.29 GB free minus the reserve leaves no room for it, and there is no OOM row. The result is the same set of transfers, planned instead of forced. If a model OOMs with nothing left to offload, it does not fit on the device by itself — that terminal case and the group-offloading escape are covered [above](#offloading).

Call [`~ComponentsManager.disable_auto_cpu_offload`] to disable offloading.

```py
manager.disable_auto_cpu_offload()
```
