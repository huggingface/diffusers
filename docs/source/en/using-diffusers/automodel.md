<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# AutoModel

The [`AutoModel`] class automatically detects and loads the correct model class (UNet, transformer, VAE) from a `config.json` file. You don't need to know the specific model class name ahead of time. It supports data types and device placement, and works across model types and libraries.

The example below loads a transformer from Diffusers and a text encoder from Transformers. Use the `subfolder` parameter to specify where to load the `config.json` file from.

```py
import torch
from diffusers import AutoModel, DiffusionPipeline

transformer = AutoModel.from_pretrained(
    "Qwen/Qwen-Image", subfolder="transformer", torch_dtype=torch.bfloat16, device_map="cuda"
)

text_encoder = AutoModel.from_pretrained(
    "Qwen/Qwen-Image", subfolder="text_encoder", torch_dtype=torch.bfloat16, device_map="cuda"
)
```

[`AutoModel`] also loads models from the [Hub](https://huggingface.co/models) that aren't included in Diffusers. Set `trust_remote_code=True` in [`AutoModel.from_pretrained`] to load custom models.

```py
import torch
from diffusers import AutoModel

transformer = AutoModel.from_pretrained(
    "custom/custom-transformer-model", trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="cuda"
)
```

If the custom model inherits from the [`ModelMixin`] class, it gets access to the same features as Diffusers model classes, like [regional compilation](../optimization/fp16#regional-compilation) and [group offloading](../optimization/memory#group-offloading).

> [!NOTE]
> Learn more about implementing custom models in the [Community components](../using-diffusers/custom_pipeline_overview#community-components) guide.