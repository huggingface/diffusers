<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Compiling and offloading quantized models

Quantization, [torch.compile](./fp16#torchcompile), and [offloading](./memory#offloading) can be combined to balance [inference speed](./fp16) and [memory usage](./memory). Quantization reduces the memory needed to store weights, torch.compile speeds up inference, and offloading keeps inactive layers or models on the CPU until they're needed. Other techniques trade one for the other. For example, [caching](./cache) speeds up inference but increases memory usage because it stores intermediate outputs.

> [!TIP]
> Refer to the [torch.compile](./fp16#torchcompile) guide to learn more about compilation. For example, [regional compilation](./fp16#regional-compilation) significantly reduces compilation time without giving up the speedup.

The offloading method to combine with quantization depends on the workload.

- For image generation, use [model offloading](./memory#model-offloading). Image models do less compute per layer, so with group offloading, the current layer often finishes before the next layer has transferred and the GPU waits on the CPU.
- For video generation, use [group offloading](./memory#group-offloading). Video models are more compute-bound, so data transfer overlaps with computation.

The table below shows the latency and memory usage of each combination on Flux.

| Combination | Latency (s) | Memory usage (GB) |
|---|---|---|
| quantization  | 32.602 | 14.9453 |
| quantization, torch.compile  | 25.847 | 14.9448 |
| quantization, torch.compile, model CPU offloading | 32.312 | 12.2369 |

<small>Benchmarked on Flux with an RTX 4090, with the `transformer` and `text_encoder_2` (T5) components quantized. Use the <a href="https://gist.github.com/sayakpaul/0db9d8eeeb3d2a0e5ed7cf0d9ca19b7d">benchmarking script</a> to evaluate your own model.</small>

The examples below use [bitsandbytes](../quantization/bitsandbytes#torchcompile), but other quantization backends, such as [TorchAO](../quantization/torchao), also support compilation and offloading. Install the latest version of bitsandbytes. [PyTorch nightly](https://pytorch.org/get-started/locally/) is also recommended.

```bash
pip install -U bitsandbytes
```

## Quantization and torch.compile

[Quantize](../quantization/overview) a model to reduce the memory needed to store its weights, then [compile](./fp16#torchcompile) it to speed up inference.

Set `torch._dynamo.config.capture_dynamic_output_shape_ops = True` so [Dynamo](https://docs.pytorch.org/docs/stable/torch.compiler_dynamo_overview.html) can compile bitsandbytes ops whose output shapes depend on the input.

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
    dtype=torch.bfloat16,
).to("cuda")  # or "mps", "xpu", "cpu"

# compile
pipeline.transformer.to(memory_format=torch.channels_last)
pipeline.transformer.compile(mode="max-autotune", fullgraph=True)
pipeline(
    "cinematic film still of a cat sipping a margarita in a pool in Palm Springs, California, highly detailed, high budget hollywood movie, cinemascope, moody, epic, gorgeous, film grain"
).images[0]
```

## Quantization, torch.compile, and offloading

Add offloading to quantization and torch.compile to reduce memory usage further. Offloading keeps layers or model components on the CPU and moves them to the GPU only when they're needed.

Raise the [Dynamo](https://docs.pytorch.org/docs/stable/torch.compiler_dynamo_overview.html) `cache_size_limit` to avoid excessive recompilation with offloading, and set `capture_dynamic_output_shape_ops = True` to compile bitsandbytes ops whose output shapes depend on the input.

<hfoptions id="offloading">
<hfoption id="model CPU offloading">

[Model offloading](./memory#model-offloading) moves a whole pipeline component, like the transformer, to the GPU only when it's needed for computation. Otherwise, the component stays on the CPU.

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
    dtype=torch.bfloat16,
)

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

[Group offloading](./memory#group-offloading) moves the internal layers of a component, like the transformer, to the GPU only when they run. With `use_stream=True`, it uses [CUDA streams](./memory#cuda-stream) to prefetch the next layer while the current one runs. For compute-bound video models, this overlap makes group offloading faster than model offloading while also using less memory.

```py
# pip install ftfy
import torch
from diffusers import DiffusionPipeline
from diffusers.hooks import apply_group_offloading
from diffusers.utils import export_to_video
from diffusers.quantizers import PipelineQuantizationConfig

torch._dynamo.config.cache_size_limit = 1000
torch._dynamo.config.capture_dynamic_output_shape_ops = True

# quantize
pipeline_quant_config = PipelineQuantizationConfig(
    quant_backend="bitsandbytes_4bit",
    quant_kwargs={"load_in_4bit": True, "bnb_4bit_quant_type": "nf4", "bnb_4bit_compute_dtype": torch.bfloat16},
    components_to_quantize=["transformer", "text_encoder"],
)

pipeline = DiffusionPipeline.from_pretrained(
    "Wan-AI/Wan2.1-T2V-14B-Diffusers",
    quantization_config=pipeline_quant_config,
    dtype=torch.bfloat16,
)

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

## Next steps

- Learn more about each offloading method in the [Reduce memory usage](./memory) guide.
- Speed up inference further with the [torch.compile](./fp16#torchcompile) guide.
- Compare quantization backends in the [quantization overview](../quantization/overview).
