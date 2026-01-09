<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Compiling and offloading quantized models

Optimizing models often involves trade-offs between [inference speed](./fp16) and [memory-usage](./memory). For instance, while [caching](./cache) can boost inference speed, it also increases memory consumption since it needs to store the outputs of intermediate attention layers. A more balanced optimization strategy combines quantizing a model, [torch.compile](./fp16#torchcompile) and various [offloading methods](./memory#offloading).

> [!TIP]
> Check the [torch.compile](./fp16#torchcompile) guide to learn more about compilation and how they can be applied here. For example, regional compilation can significantly reduce compilation time without giving up any speedups. 

For image generation, combining quantization and [model offloading](./memory#model-offloading) can often give the best trade-off between quality, speed, and memory. Group offloading is not as effective for image generation because it is usually not possible to *fully* overlap data transfer if the compute kernel finishes faster. This results in some communication overhead between the CPU and GPU.

For video generation, combining quantization and [group-offloading](./memory#group-offloading) tends to be better because video models are more compute-bound. 

The table below provides a comparison of optimization strategy combinations and their impact on latency and memory-usage for Flux.

| combination | latency (s) | memory-usage (GB) |
|---|---|---|
| quantization  | 32.602 | 14.9453 |
| quantization, torch.compile  | 25.847 | 14.9448 |
| quantization, torch.compile, model CPU offloading | 32.312 | 12.2369 |

<small>These results are benchmarked on Flux with a RTX 4090. The transformer and text_encoder components are quantized. Refer to the <a href="https://gist.github.com/sayakpaul/0db9d8eeeb3d2a0e5ed7cf0d9ca19b7d">benchmarking script</a> if you're interested in evaluating your own model.</small>

This guide will show you how to compile and offload a quantized model with [bitsandbytes](../quantization/bitsandbytes#torchcompile). Make sure you are using [PyTorch nightly](https://pytorch.org/get-started/locally/) and the latest version of bitsandbytes.

```bash
pip install -U bitsandbytes
```

## Quantization and torch.compile

Start by [quantizing](../quantization/overview) a model to reduce the memory required for storage and [compiling](./fp16#torchcompile) it to accelerate inference.

Configure the [Dynamo](https://docs.pytorch.org/docs/stable/torch.compiler_dynamo_overview.html) `capture_dynamic_output_shape_ops = True` to handle dynamic outputs when compiling bitsandbytes models.

```py
import torch
from diffusers import DiffusionPipeline
from diffusers.quantizers import PipelineQuantizationConfig

torch._dynamo.config.capture_dynamic_output_shape_ops = True

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
pipeline.transformer.compile(mode="max-autotune", fullgraph=True)
pipeline("""
    cinematic film still of a cat sipping a margarita in a pool in Palm Springs, California
    highly detailed, high budget hollywood movie, cinemascope, moody, epic, gorgeous, film grain
"""
).images[0]
```

## Quantization, torch.compile, and offloading

In addition to quantization and torch.compile, try offloading if you need to reduce memory-usage further. Offloading moves various layers or model components from the CPU to the GPU as needed for computations.

Configure the [Dynamo](https://docs.pytorch.org/docs/stable/torch.compiler_dynamo_overview.html) `cache_size_limit` during offloading to avoid excessive recompilation and set `capture_dynamic_output_shape_ops = True` to handle dynamic outputs when compiling bitsandbytes models.

<hfoptions id="offloading">
<hfoption id="model CPU offloading">

[Model CPU offloading](./memory#model-offloading) moves an individual pipeline component, like the transformer model, to the GPU when it is needed for computation. Otherwise, it is offloaded to the CPU.

```py
import torch
from diffusers import DiffusionPipeline
from diffusers.quantizers import PipelineQuantizationConfig

torch._dynamo.config.cache_size_limit = 1000
torch._dynamo.config.capture_dynamic_output_shape_ops = True

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
pipeline.transformer.compile()
pipeline(
    "cinematic film still of a cat sipping a margarita in a pool in Palm Springs, California, highly detailed, high budget hollywood movie, cinemascope, moody, epic, gorgeous, film grain"
).images[0]
```

</hfoption>
<hfoption id="group offloading">

[Group offloading](./memory#group-offloading) moves the internal layers of an individual pipeline component, like the transformer model, to the GPU for computation and offloads it when it's not required. At the same time, it uses the [CUDA stream](./memory#cuda-stream) feature to prefetch the next layer for execution.

By overlapping computation and data transfer, it is faster than model CPU offloading while also saving memory. 

```py
# pip install ftfy
import torch
from diffusers import AutoModel, DiffusionPipeline
from diffusers.hooks import apply_group_offloading
from diffusers.utils import export_to_video
from diffusers.quantizers import PipelineQuantizationConfig
from transformers import UMT5EncoderModel

torch._dynamo.config.cache_size_limit = 1000
torch._dynamo.config.capture_dynamic_output_shape_ops = True

# quantize
pipeline_quant_config = PipelineQuantizationConfig(
    quant_backend="bitsandbytes_4bit",
    quant_kwargs={"load_in_4bit": True, "bnb_4bit_quant_type": "nf4", "bnb_4bit_compute_dtype": torch.bfloat16},
    components_to_quantize=["transformer", "text_encoder"],
)

text_encoder = UMT5EncoderModel.from_pretrained(
    "Wan-AI/Wan2.1-T2V-14B-Diffusers", subfolder="text_encoder", torch_dtype=torch.bfloat16
)
pipeline = DiffusionPipeline.from_pretrained(
    "Wan-AI/Wan2.1-T2V-14B-Diffusers",
    quantization_config=pipeline_quant_config,
    torch_dtype=torch.bfloat16,
).to("cuda")

# group offloading
onload_device = torch.device("cuda")
offload_device = torch.device("cpu")

pipeline.transformer.enable_group_offload(
    onload_device=onload_device,
    offload_device=offload_device,
    offload_type="leaf_level",
    use_stream=True,
    non_blocking=True
)
pipeline.vae.enable_group_offload(
    onload_device=onload_device,
    offload_device=offload_device,
    offload_type="leaf_level",
    use_stream=True,
    non_blocking=True
)
apply_group_offloading(
    pipeline.text_encoder,
    onload_device=onload_device,
    offload_type="leaf_level",
    use_stream=True,
    non_blocking=True
)

# compile
pipeline.transformer.compile()

prompt = """
The camera rushes from far to near in a low-angle shot, 
revealing a white ferret on a log. It plays, leaps into the water, and emerges, as the camera zooms in 
for a close-up. Water splashes berry bushes nearby, while moss, snow, and leaves blanket the ground. 
Birch trees and a light blue sky frame the scene, with ferns in the foreground. Side lighting casts dynamic 
shadows and warm highlights. Medium composition, front view, low angle, with depth of field.
"""
negative_prompt = """
Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, 
low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, 
misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards
"""

output = pipeline(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_frames=81,
    guidance_scale=5.0,
).frames[0]
export_to_video(output, "output.mp4", fps=16)
```

</hfoption>
</hfoptions>