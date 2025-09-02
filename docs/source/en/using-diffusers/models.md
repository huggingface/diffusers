<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

[[open-in-colab]]

# Models

A diffusion model relies on a few individual models working together to generate an output. These models are responsible for denoising, encoding inputs, and decoding latents into the actual outputs.

This guide will show you how to load models.

## Loading a model

All models are loaded with the [`~ModelMixin.from_pretrained`] method, which downloads and caches the latest model version. If the latest files are available in the local cache, [`~ModelMixin.from_pretrained`] reuses files in the cache.

Pass the `subfolder` argument to [`~ModelMixin.from_pretrained`] to specify where to load the model weights from. Omit the `subfolder` argument if the repository doesn't have a subfolder structure or if you're loading a standalone model.

```py
from diffusers import QwenImageTransformer2DModel

model = QwenImageTransformer2DModel.from_pretrained("Qwen/Qwen-Image", subfolder="transformer")
```

## AutoModel

[`AutoModel`] detects the model class from a `model_index.json` file or a model's `config.json` file. It fetches the correct model class from these files and delegates the actual loading to the model class. [`AutoModel`] is useful for automatic model type detection without needing to know the exact model class beforehand.

```py
from diffusers import AutoModel

model = AutoModel.from_pretrained(
    "Qwen/Qwen-Image", subfolder="transformer"
)
```

## Model data types

Use the `torch_dtype` argument in [`~ModelMixin.from_pretrained`] to load a model with a specific data type. This allows you to load a model in a lower precision to reduce memory usage.

```py
import torch
from diffusers import QwenImageTransformer2DModel

model = QwenImageTransformer2DModel.from_pretrained(
    "Qwen/Qwen-Image",
    subfolder="transformer",
    torch_dtype=torch.bfloat16
)
```

[nn.Module.to](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.to) can also convert to a specific data type on the fly. However, it converts *all* weights to the requested data type unlike `torch_dtype` which respects `_keep_in_fp32_modules`. This argument preserves layers in `torch.float32` for numerical stability and best generation quality (see example [_keep_in_fp32_modules](https://github.com/huggingface/diffusers/blob/f864a9a352fa4a220d860bfdd1782e3e5af96382/src/diffusers/models/transformers/transformer_wan.py#L374))

```py
from diffusers import QwenImageTransformer2DModel

model = QwenImageTransformer2DModel.from_pretrained(
    "Qwen/Qwen-Image", subfolder="transformer"
)
model = model.to(dtype=torch.float16) 
```

## Device placement

Use the `device_map` argument in [`~ModelMixin.from_pretrained`] to place a model on an accelerator like a GPU. It is especially helpful where there are multiple GPUs.

Diffusers currently provides three options to `device_map` for individual models, `"cuda"`, `"balanced"` and `"auto"`. Refer to the table below to compare the three placement strategies.

| parameter | description |
|---|---|
| `"cuda"` | places pipeline on a supported accelerator (CUDA) |
| `"balanced"` | evenly distributes pipeline on all GPUs |
| `"auto"` | distribute model from fastest device first to slowest |

Use the `max_memory` argument in [`~ModelMixin.from_pretrained`] to allocate a maximum amount of memory to use on each device. By default, Diffusers uses the maximum amount available.

```py
import torch
from diffusers import QwenImagePipeline

max_memory = {0: "16GB", 1: "16GB"}
pipeline = QwenImagePipeline.from_pretrained(
    "Qwen/Qwen-Image", 
    torch_dtype=torch.bfloat16,
    device_map="cuda",
    max_memory=max_memory
)
```

The `hf_device_map` attribute allows you to access and view the `device_map`.

```py
print(transformer.hf_device_map)
# {'': device(type='cuda')}
```

## Saving models

Save a model with the [`~ModelMixin.save_pretrained`] method.

```py
from diffusers import QwenImageTransformer2DModel

model = QwenImageTransformer2DModel.from_pretrained("Qwen/Qwen-Image", subfolder="transformer")
model.save_pretrained("./local/model")
```

For large models, it is helpful to use `max_shard_size` to save a model as multiple shards. A shard can be loaded faster and save memory (refer to the [parallel loading](./loading#parallel-loading) docs for more details), especially if there is more than one GPU.

```py
model.save_pretrained("./local/model", max_shard_size="5GB")
```
