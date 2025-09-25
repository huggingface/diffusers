<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

-->

# Quanto

[Quanto](https://github.com/huggingface/optimum-quanto) is a PyTorch quantization backend for [Optimum](https://huggingface.co/docs/optimum/index). It has been designed with versatility and simplicity in mind:

- All features are available in eager mode (works with non-traceable models)
- Supports quantization aware training
- Quantized models are compatible with `torch.compile`
- Quantized models are Device agnostic (e.g CUDA,XPU,MPS,CPU)

Although the Quanto library does allow quantizing `nn.Conv2d` and `nn.LayerNorm` modules, currently, Diffusers only supports quantizing the weights in the `nn.Linear` layers of a model.

Make sure Quanto and [Accelerate](https://huggingface.co/docs/optimum/index) are installed.

```bash
pip install -U optimum-quanto accelerate
```

Create and pass `weights_dtype` to [`QuantoConfig`] configure the target data type to quantize a model to. The example below quantizes the model to `float8`. Check [`QuantoConfig`] for a list of supported weight types.

```python
import torch
from diffusers import AutoModel, QuantoConfig, FluxPipeline

quantization_config = QuantoConfig(weights_dtype="float8")
transformer = FluxTransformer2DModel.from_pretrained(
      "black-forest-labs/FLUX.1-dev",
      subfolder="transformer",
      quantization_config=quantization_config,
      torch_dtype=torch.bfloat16,
)

pipeline = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    transformer=transformer,
    torch_dtype=torch.bfloat16,
    device_map="cuda"
)
prompt = """
cinematic film still of a cat sipping a margarita in a pool in Palm Springs, California
highly detailed, high budget hollywood movie, cinemascope, moody, epic, gorgeous, film grain
"""
image = pipeline(prompt).images[0]
image.save("flux-quanto.png")
```

[`QuantoConfig`] also works with single files with [`~loaders.FromOriginalModelMixin.from_single_file`].

```python
import torch
from diffusers import AutoModel, QuantoConfig

quantization_config = QuantoConfig(weights_dtype="float8")
transformer = AutoModel.from_single_file(
    "https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/flux1-dev.safetensors",
    quantization_config=quantization_config,
    torch_dtype=torch.bfloat16
)
```

## torch.compile

Quanto supports torch.compile for `int8` weights only.

```python
import torch
from diffusers import FluxPipeline, AutoModel, QuantoConfig

quantization_config = QuantoConfig(weights_dtype="int8")
transformer = FluxTransformer2DModel.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    subfolder="transformer",
    quantization_config=quantization_config,
    torch_dtype=torch.bfloat16,
)
transformer = torch.compile(transformer, mode="max-autotune", fullgraph=True)
pipeline = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    transformer=transformer,
    torch_dtype=torch.bfloat16,
    device_map="cuda"
)
```

## Skipping quantization on specific modules

Use `modules_to_not_convert` to skip quantization on specific modules. The modules passed to this argument must match the module keys in `state_dict`.

```python
import torch
from diffusers import AutoModel, QuantoConfig

quantization_config = QuantoConfig(weights_dtype="float8", modules_to_not_convert=["proj_out"])
transformer = AutoModel.from_pretrained(
      "black-forest-labs/FLUX.1-dev",
      subfolder="transformer",
      quantization_config=quantization_config,
      torch_dtype=torch.bfloat16,
)
```

## Saving quantized models

Save a Quanto model with [`~ModelMixin.save_pretrained`]. Models quantized directly with the Quanto library - not as a backend in Diffusers - can't be loaded in Diffusers with [`~ModelMixin.from_pretrained`].

```python
import torch
from diffusers import AutoModel, QuantoConfig

quantization_config = QuantoConfig(weights_dtype="float8")
transformer = AutoModel.from_pretrained(
      "black-forest-labs/FLUX.1-dev",
      subfolder="transformer",
      quantization_config=quantization_config,
      torch_dtype=torch.bfloat16,
)
transformer.save_pretrained("path/to/saved/model")

# Reload quantized model
model = AutoModel.from_pretrained("path/to/saved/model")
```