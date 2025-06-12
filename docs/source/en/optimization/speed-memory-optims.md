<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Compile and offloading

There are trade-offs associated with optimizing solely for [inference speed](./fp16) or [memory-usage](./memory). For example, [caching](./cache) increases inference speed but requires more memory to store the intermediate outputs from the attention layers.

If your hardware is sufficiently powerful, you can choose to focus on one or the other. For a more balanced approach that doesn't sacrifice too much in terms of inference speed and memory-usage, try compiling and offloading a model.

Refer to the table below for the latency and memory-usage of each combination.

| combination | latency | memory usage |
|---|---|---|
| quantization, torch.compile |  |  |
| quantization, torch.compile, model CPU offloading |  |  |
| quantization, torch.compile, group offloading |  |  |

This guide will show you how to compile and offload a model to improve both inference speed and memory-usage.

## Quantization and torch.compile

> [!TIP]
> The quantization backend, such as [bitsandbytes](../quantization/bitsandbytes#torchcompile), must be compatible with torch.compile. Refer to the quantization [overview](https://huggingface.co/docs/transformers/quantization/overview#overview) table to see which backends support torch.compile.

Start by [quantizing](../quantization/overview) a model to reduce the memory required to store it and [compiling](./fp16#torchcompile) it to accelerate inference.

```py
import torch
from diffusers import DiffusionPipeline
from diffusers.quantizers import PipelineQuantizationConfig

# quantize
pipeline_quant_config = PipelineQuantizationConfig(
    quant_backend="bitsandbytes_4bit",
    quant_kwargs={"load_in_4bit": True, "bnb_4bit_quant_type": "nf4", "bnb_4bit_compute_dtype": torch.bfloat16},
    components_to_quantize=["transformer", "text_encoder_2"],
)
pipeline = DiffusionPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    quantization_config=pipeline_quant_config,
    torch_dtype=torch.bfloat16,
).to("cuda")

# compile
pipeline.transformer.to(memory_format=torch.channels_last)
pipeline.transformer = torch.compile(
    pipeline.transformer, mode="ax-autotune", fullgraph=True
)
pipeline("""
    cinematic film still of a cat sipping a margarita in a pool in Palm Springs, California
    highly detailed, high budget hollywood movie, cinemascope, moody, epic, gorgeous, film grain
"""
).images[0]
```

## Quantization, torch.compile, and offloading

In addition to quantization and torch.compile, try offloading if you need to reduce memory-usage further. Offloading moves various layers or model components from the CPU to the GPU as needed for computations.

<hfoptions id="offloading">
<hfoption id="model CPU offloading">

[Model CPU offloading](./memory#model-offloading) moves an individual pipeline component, like the transformer model, to the GPU when it is needed for computation. Otherwise, it is offloaded to the CPU.

```py
import torch
from diffusers import DiffusionPipeline
from diffusers.quantizers import PipelineQuantizationConfig

# quantize
pipeline_quant_config = PipelineQuantizationConfig(
    quant_backend="bitsandbytes_4bit",
    quant_kwargs={"load_in_4bit": True, "bnb_4bit_quant_type": "nf4", "bnb_4bit_compute_dtype": torch.bfloat16},
    components_to_quantize=["transformer", "text_encoder_2"],
)
pipeline = DiffusionPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    quantization_config=pipeline_quant_config,
    torch_dtype=torch.bfloat16,
).to("cuda")

# model CPU offloading
pipeline.enable_model_cpu_offload()

# compile
pipeline.transformer.to(memory_format=torch.channels_last)
pipeline.transformer = torch.compile(
    pipeline.transformer, mode="ax-autotune", fullgraph=True
)
pipeline(
    "cinematic film still of a cat sipping a margarita in a pool in Palm Springs, California, highly detailed, high budget hollywood movie, cinemascope, moody, epic, gorgeous, film grain"
).images[0]
```

</hfoption>
<hfoption id="group offloading">

[Group offloading](./memory#group-offloading) moves the internal layers of an individual pipeline component, like the transformer model, to the GPU for computation and offloads it when it's not required. At the same time, it uses the [CUDA stream](./memory#cuda-stream) feature to prefetch the next layer for execution.

By overlapping computation and data transfer, it is faster than model CPU offloading while also saving memory. 

```py
import torch
from diffusers import DiffusionPipeline
from diffusers.hooks import apply_group_offloading
from diffusers.quantizers import PipelineQuantizationConfig

# quantize
pipeline_quant_config = PipelineQuantizationConfig(
    quant_backend="bitsandbytes_4bit",
    quant_kwargs={"load_in_4bit": True, "bnb_4bit_quant_type": "nf4", "bnb_4bit_compute_dtype": torch.bfloat16},
    components_to_quantize=["transformer", "text_encoder_2"],
)
pipeline = DiffusionPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    quantization_config=pipeline_quant_config,
    torch_dtype=torch.bfloat16,
).to("cuda")

# group offloading
onload_device = torch.device("cuda")
offload_device = torch.device("cpu")

pipeline.transformer.enable_group_offload(onload_device=onload_device, offload_device=offload_device, offload_type="leaf_level", use_stream=True)
pipeline.vae.enable_group_offload(onload_device=onload_device, offload_type="leaf_level", use_stream=True)
apply_group_offloading(pipeline.text_encoder, onload_device=onload_device, offload_type="block_level", num_blocks_per_group=1, use_stream=True)

# compile
pipeline.transformer.to(memory_format=torch.channels_last)
pipeline.transformer = torch.compile(
    pipeline.transformer, mode="ax-autotune", fullgraph=True
)
pipeline(
    "cinematic film still of a cat sipping a margarita in a pool in Palm Springs, California, highly detailed, high budget hollywood movie, cinemascope, moody, epic, gorgeous, film grain"
).images[0]
```

</hfoption>
</hfoptions>