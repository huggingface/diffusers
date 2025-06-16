<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Compile and offloading quantized models

When optimizing models, you often face trade-offs between [inference speed](./fp16) and [memory-usage](./memory). For instance, while [caching](./cache) can boost inference speed, it comes at the cost of increased memory consumption since it needs to store intermediate attention layer outputs.

A more balanced optimization strategy combines [torch.compile](./fp16#torchcompile) with various [offloading methods](./memory#offloading) on a quantized model. This approach not only accelerates inference but also helps lower memory-usage.

For image generation, combining quantization and [model offloading](./memory#model-offloading) can often give the best trade-off between quality, speed, and memory. Group offloading is not as effective because it is usually not possible to *fully* overlap data transfer if the compute kernel finishes faster. This results in some communication overhead between the CPU and GPU.

For video generation, combining quantization and [group-offloading](./memory#group-offloading) tends to be better because video models are more compute-bound. 

The table below provides a comparison of optimization strategy combinations and their impact on latency and memory-usage for Flux.

| combination | latency (s) | memory-usage (GB) |
|---|---|---|
| quantization | 32.602 | 14.9453 |
| quantization, torch.compile | 25.847 | 14.9448 |
| quantization, torch.compile, model CPU offloading | 32.312 | 12.2369 |
| quantization, torch.compile, group offloading | 60.235 | 12.2369 |
<small>These results are benchmarked on Flux with a RTX 4090. The `transformer` and `text_encoder_2` components are quantized. Refer to the [benchmarking script](https://gist.github.com/sayakpaul/0db9d8eeeb3d2a0e5ed7cf0d9ca19b7d) if you're interested in evaluating your own model.</small>

> [!TIP]
> We recommend installing [PyTorch nightly](https://pytorch.org/get-started/locally/) for better torch.compile support.

This guide will show you how to compile and offload a quantized model.

## Quantization and torch.compile

> [!TIP]
> The quantization backend, such as [bitsandbytes](../quantization/bitsandbytes#torchcompile), must be compatible with torch.compile. Refer to the quantization [overview](https://huggingface.co/docs/transformers/quantization/overview#overview) table to see which backends support torch.compile.

Start by [quantizing](../quantization/overview) a model to reduce the memory required for storage and [compiling](./fp16#torchcompile) it to accelerate inference.

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
pipeline.transformer.compile( mode="max-autotune", fullgraph=True)
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
pipeline.transformer.compile( mode="max-autotune", fullgraph=True)
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
apply_group_offloading(pipeline.text_encoder, onload_device=onload_device, offload_type="leaf_level", use_stream=True)
apply_group_offloading(pipeline.text_encoder_2, onload_device=onload_device, offload_type="leaf_level", use_stream=True)

# compile
pipeline.transformer.to(memory_format=torch.channels_last)
pipeline.transformer.compile( mode="max-autotune", fullgraph=True)
pipeline(
    "cinematic film still of a cat sipping a margarita in a pool in Palm Springs, California, highly detailed, high budget hollywood movie, cinemascope, moody, epic, gorgeous, film grain"
).images[0]
```

</hfoption>
</hfoptions>