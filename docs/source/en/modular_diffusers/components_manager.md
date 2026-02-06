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
=============================================================================================================================
Models:
-----------------------------------------------------------------------------------------------------------------------------
Name_ID                      | Class                    | Device: act(exec) | Dtype          | Size (GB) | Load ID
-----------------------------------------------------------------------------------------------------------------------------
text_encoder_140458257514752 | Qwen3Model               | cpu               | torch.bfloat16 | 7.49      | Tongyi-MAI/Z-Image-Turbo|text_encoder|null|null
vae_140458257515376          | AutoencoderKL            | cpu               | torch.bfloat16 | 0.16      | Tongyi-MAI/Z-Image-Turbo|vae|null|null
transformer_140458257515616  | ZImageTransformer2DModel | cpu               | torch.bfloat16 | 11.46     | Tongyi-MAI/Z-Image-Turbo|transformer|null|null
-----------------------------------------------------------------------------------------------------------------------------

Other Components:
-----------------------------------------------------------------------------------------------------------------------------
ID                           | Class                           | Collection
-----------------------------------------------------------------------------------------------------------------------------
scheduler_140461023555264    | FlowMatchEulerDiscreteScheduler | N/A
tokenizer_140458256346432    | Qwen2Tokenizer                  | N/A
-----------------------------------------------------------------------------------------------------------------------------
```

The table shows models (with device, dtype, and memory info) separately from other components like schedulers and tokenizers. If any models have LoRA adapters, IP-Adapters, or quantization applied, that information is displayed in an additional section at the bottom.

## Offloading

The [`~ComponentsManager.enable_auto_cpu_offload`] method is a global offloading strategy that works across all models regardless of which pipeline is using them. Once enabled, you don't need to worry about device placement if you add or remove components.

```py
manager.enable_auto_cpu_offload(device="cuda")
```

All models begin on the CPU and [`ComponentsManager`] moves them to the appropriate device right before they're needed, and moves other models back to the CPU when GPU memory is low.

Call [`~ComponentsManager.disable_auto_cpu_offload`] to disable offloading.

```py
manager.disable_auto_cpu_offload()
```
