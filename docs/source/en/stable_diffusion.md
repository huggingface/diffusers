<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Overview

Diffusion inference is computationally expensive, and you often run a [`DiffusionPipeline`] more than once before you like the result. This page provides an overview of the main Diffusers optimization techniques, what they do, and when to use them.

## Starter path

When the model fits on one GPU, start with this baseline load. Set `dtype` and place the pipeline on an accelerator. Reach for model CPU offload only when memory is tight. You could also speed up inference with fewer steps or a faster scheduler.

When you omit `dtype`, Diffusers loads components in `float32`. Pass `dtype=torch.bfloat16` (or `torch.float16` if bfloat16 is unsupported), then place the pipeline on an accelerator with `pipeline.to("cuda")`.

```py
import torch
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    dtype=torch.bfloat16,
)
pipeline.to("cuda")  # or "mps", "xpu"

prompt = """
cinematic film still of a cat sipping a margarita in a pool in Palm Springs, California
highly detailed, high budget hollywood movie, cinemascope, moody, epic, gorgeous, film grain
"""
pipeline(prompt).images[0]
```

If the pipeline does not fit, or memory is tight, call [`~DiffusionPipeline.enable_model_cpu_offload`] instead of keeping everything on the GPU. It places the active model on the GPU and keeps the other components on the CPU.

Skip it when the model fits. Offloading is slower when you do not need it.

For more offloading options, see [Reduce memory usage](./optimization/memory#offloading).

```py
pipeline = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    dtype=torch.bfloat16,
)
pipeline.enable_model_cpu_offload()
```

Lower latency with fewer `num_inference_steps` or a faster scheduler such as [`DPMSolverMultistepScheduler`]. That usually speeds up generation but can reduce image quality versus a slower, higher-quality scheduler. See [Accelerate inference](./optimization/fp16) for more speed techniques.

```py
import time
from diffusers import DPMSolverMultistepScheduler

pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)

start_time = time.perf_counter()
image = pipeline(prompt, num_inference_steps=25).images[0]
end_time = time.perf_counter()

print(f"Image generation took {end_time - start_time:.3f} seconds")
```

## Optimization techniques

When the starter path is not enough, use these techniques. If you are out of memory, start with offloading or quantization. If inference is too slow, start with caching, attention backends, `torch.compile`, or regional compilation.

- [Caching](./optimization/cache) — Reuse intermediates across denoising steps when you want more speed and can spend memory.
- [Attention backends](./optimization/attention_backends) — Swap Diffusers attention implementations through a unified API when attention is the bottleneck.
- [Quantization](./quantization/overview) — Load smaller weights to cut memory (and often speed up inference). [GGUF](./quantization/gguf) is a common starting point.
- [Regional compilation](./optimization/fp16#regional-compilation) — Compile repeated blocks to cut `torch.compile` cold-start latency and reuse compiled artifacts.
- [torch.compile](./optimization/fp16#torchcompile) — Compile the UNet, transformer, or VAE into optimized kernels.
- [Kernels](./optimization/fp16#kernels) — Load optimized Hub compute kernels (attention and custom CUDA ops such as RMSNorm or RoPE) when you need hardware-specific speedups beyond stock PyTorch.
- [Offloading](./optimization/memory#offloading) — Move inactive models or layers to the CPU with CPU, model, or group offloading.
- [Quantize + compile + offload](./optimization/speed-memory-optims) — Combine quantization, `torch.compile`, and offloading when one technique is not enough.
