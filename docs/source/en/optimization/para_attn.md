# ParaAttention

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/para-attn/flux-performance.png">
</div>
<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/para-attn/hunyuan-video-performance.png">
</div>


Large image and video generation models, such as [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) and [HunyuanVideo](https://huggingface.co/tencent/HunyuanVideo), can be an inference challenge for real-time applications and deployment because of their size.

[ParaAttention](https://github.com/chengzeyi/ParaAttention) is a library that implements **context parallelism** and **first block cache**, and can be combined with other techniques (torch.compile, fp8 dynamic quantization), to accelerate inference.

This guide will show you how to apply ParaAttention to FLUX.1-dev and HunyuanVideo on NVIDIA L20 GPUs.
No optimizations are applied for our baseline benchmark, except for HunyuanVideo to avoid out-of-memory errors.

Our baseline benchmark shows that FLUX.1-dev is able to generate a 1024x1024 resolution image in 28 steps in 26.36 seconds, and HunyuanVideo is able to generate 129 frames at 720p resolution in 30 steps in 3675.71 seconds.

> [!TIP]
> For even faster inference with context parallelism, try using NVIDIA A100 or H100 GPUs (if available) with NVLink support, especially when there is a large number of GPUs.

## First Block Cache

Caching the output of the transformers blocks in the model and reusing them in the next inference steps reduces the computation cost and makes inference faster.

However, it is hard to decide when to reuse the cache to ensure quality generated images or videos. ParaAttention directly uses the **residual difference of the first transformer block output** to approximate the difference among model outputs. When the difference is small enough, the residual difference of previous inference steps is reused. In other words, the denoising step is skipped.

This achieves a 2x speedup on FLUX.1-dev and HunyuanVideo inference with very good quality.

<figure>
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/para-attn/ada-cache.png" alt="Cache in Diffusion Transformer" />
    <figcaption>How AdaCache works, First Block Cache is a variant of it</figcaption>
</figure>

<hfoptions id="first-block-cache">
<hfoption id="FLUX-1.dev">

To apply first block cache on FLUX.1-dev, call `apply_cache_on_pipe` as shown below. 0.08 is the default residual difference value for FLUX models.

```python
import time
import torch
from diffusers import FluxPipeline

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16,
).to("cuda")

from para_attn.first_block_cache.diffusers_adapters import apply_cache_on_pipe

apply_cache_on_pipe(pipe, residual_diff_threshold=0.08)

# Enable memory savings
# pipe.enable_model_cpu_offload()
# pipe.enable_sequential_cpu_offload()

begin = time.time()
image = pipe(
    "A cat holding a sign that says hello world",
    num_inference_steps=28,
).images[0]
end = time.time()
print(f"Time: {end - begin:.2f}s")

print("Saving image to flux.png")
image.save("flux.png")
```

| Optimizations | Original | FBCache rdt=0.06 | FBCache rdt=0.08 | FBCache rdt=0.10 | FBCache rdt=0.12 |
| - | - | - | - | - | - |
| Preview | ![Original](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/para-attn/flux-original.png) | ![FBCache rdt=0.06](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/para-attn/flux-fbc-0.06.png) | ![FBCache rdt=0.08](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/para-attn/flux-fbc-0.08.png) | ![FBCache rdt=0.10](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/para-attn/flux-fbc-0.10.png) | ![FBCache rdt=0.12](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/para-attn/flux-fbc-0.12.png) |
| Wall Time (s) | 26.36 | 21.83 | 17.01 | 16.00 | 13.78 |

First Block Cache reduced the inference speed to 17.01 seconds compared to the baseline, or 1.55x faster, while maintaining nearly zero quality loss.

</hfoption>
<hfoption id="HunyuanVideo">

To apply First Block Cache on HunyuanVideo, `apply_cache_on_pipe` as shown below. 0.06 is the default residual difference value for HunyuanVideo models.

```python
import time
import torch
from diffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel
from diffusers.utils import export_to_video

model_id = "tencent/HunyuanVideo"
transformer = HunyuanVideoTransformer3DModel.from_pretrained(
    model_id,
    subfolder="transformer",
    torch_dtype=torch.bfloat16,
    revision="refs/pr/18",
)
pipe = HunyuanVideoPipeline.from_pretrained(
    model_id,
    transformer=transformer,
    torch_dtype=torch.float16,
    revision="refs/pr/18",
).to("cuda")

from para_attn.first_block_cache.diffusers_adapters import apply_cache_on_pipe

apply_cache_on_pipe(pipe, residual_diff_threshold=0.6)

pipe.vae.enable_tiling()

begin = time.time()
output = pipe(
    prompt="A cat walks on the grass, realistic",
    height=720,
    width=1280,
    num_frames=129,
    num_inference_steps=30,
).frames[0]
end = time.time()
print(f"Time: {end - begin:.2f}s")

print("Saving video to hunyuan_video.mp4")
export_to_video(output, "hunyuan_video.mp4", fps=15)
```

<video controls>
  <source src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/para-attn/hunyuan-video-original.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

<small> HunyuanVideo without FBCache </small>

<video controls>
  <source src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/para-attn/hunyuan-video-fbc.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

<small> HunyuanVideo with FBCache </small>

First Block Cache reduced the inference speed to 2271.06 seconds compared to the baseline, or 1.62x faster, while maintaining nearly zero quality loss.

</hfoption>
</hfoptions>

## fp8 quantization

fp8 with dynamic quantization further speeds up inference and reduces memory usage. Both the activations and weights must be quantized in order to use the 8-bit [NVIDIA Tensor Cores](https://www.nvidia.com/en-us/data-center/tensor-cores/).

Use `float8_weight_only` and `float8_dynamic_activation_float8_weight` to quantize the text encoder and transformer model.

The default quantization method is per tensor quantization, but if your GPU supports row-wise quantization, you can also try it for better accuracy.

Install [torchao](https://github.com/pytorch/ao/tree/main) with the command below.

```bash
pip3 install -U torch torchao
```

[torch.compile](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html) with `mode="max-autotune-no-cudagraphs"` or `mode="max-autotune"` selects the best kernel for performance. Compilation can take a long time if it's the first time the model is called, but it is worth it once the model has been compiled.

This example only quantizes the transformer model, but you can also quantize the text encoder to reduce memory usage even more.

> [!TIP]
> Dynamic quantization can significantly change the distribution of the model output, so you need to change the `residual_diff_threshold` to a larger value for it to take effect.

<hfoptions id="fp8-quantization">
<hfoption id="FLUX-1.dev">

```python
import time
import torch
from diffusers import FluxPipeline

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16,
).to("cuda")

from para_attn.first_block_cache.diffusers_adapters import apply_cache_on_pipe

apply_cache_on_pipe(
    pipe,
    residual_diff_threshold=0.12,  # Use a larger value to make the cache take effect
)

from torchao.quantization import quantize_, float8_dynamic_activation_float8_weight, float8_weight_only

quantize_(pipe.text_encoder, float8_weight_only())
quantize_(pipe.transformer, float8_dynamic_activation_float8_weight())
pipe.transformer = torch.compile(
   pipe.transformer, mode="max-autotune-no-cudagraphs",
)

# Enable memory savings
# pipe.enable_model_cpu_offload()
# pipe.enable_sequential_cpu_offload()

for i in range(2):
    begin = time.time()
    image = pipe(
        "A cat holding a sign that says hello world",
        num_inference_steps=28,
    ).images[0]
    end = time.time()
    if i == 0:
        print(f"Warm up time: {end - begin:.2f}s")
    else:
        print(f"Time: {end - begin:.2f}s")

print("Saving image to flux.png")
image.save("flux.png")
```

fp8 dynamic quantization and torch.compile reduced the inference speed to 7.56 seconds compared to the baseline, or 3.48x faster.

</hfoption>
<hfoption id="HunyuanVideo">

```python
import time
import torch
from diffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel
from diffusers.utils import export_to_video

model_id = "tencent/HunyuanVideo"
transformer = HunyuanVideoTransformer3DModel.from_pretrained(
    model_id,
    subfolder="transformer",
    torch_dtype=torch.bfloat16,
    revision="refs/pr/18",
)
pipe = HunyuanVideoPipeline.from_pretrained(
    model_id,
    transformer=transformer,
    torch_dtype=torch.float16,
    revision="refs/pr/18",
).to("cuda")

from para_attn.first_block_cache.diffusers_adapters import apply_cache_on_pipe

apply_cache_on_pipe(pipe)

from torchao.quantization import quantize_, float8_dynamic_activation_float8_weight, float8_weight_only

quantize_(pipe.text_encoder, float8_weight_only())
quantize_(pipe.transformer, float8_dynamic_activation_float8_weight())
pipe.transformer = torch.compile(
   pipe.transformer, mode="max-autotune-no-cudagraphs",
)

# Enable memory savings
pipe.vae.enable_tiling()
# pipe.enable_model_cpu_offload()
# pipe.enable_sequential_cpu_offload()

for i in range(2):
    begin = time.time()
    output = pipe(
        prompt="A cat walks on the grass, realistic",
        height=720,
        width=1280,
        num_frames=129,
        num_inference_steps=1 if i == 0 else 30,
    ).frames[0]
    end = time.time()
    if i == 0:
        print(f"Warm up time: {end - begin:.2f}s")
    else:
        print(f"Time: {end - begin:.2f}s")

print("Saving video to hunyuan_video.mp4")
export_to_video(output, "hunyuan_video.mp4", fps=15)
```

A NVIDIA L20 GPU only has 48GB memory and could face out-of-memory (OOM) errors after compilation and if `enable_model_cpu_offload` isn't called because HunyuanVideo has very large activation tensors when running with high resolution and large number of frames. For GPUs with less than 80GB of memory, you can try reducing the resolution and number of frames to avoid OOM errors.

Large video generation models are usually bottlenecked by the attention computations rather than the fully connected layers. These models don't significantly benefit from quantization and torch.compile.

</hfoption>
</hfoptions>

## Context Parallelism

Context Parallelism parallelizes inference and scales with multiple GPUs. The ParaAttention compositional design allows you to combine Context Parallelism with First Block Cache and dynamic quantization.

> [!TIP]
> Refer to the [ParaAttention](https://github.com/chengzeyi/ParaAttention/tree/main) repository for detailed instructions and examples of how to scale inference with multiple GPUs.

If the inference process needs to be persistent and serviceable, it is suggested to use [torch.multiprocessing](https://pytorch.org/docs/stable/multiprocessing.html) to write your own inference processor. This can eliminate the overhead of launching the process and loading and recompiling the model.

<hfoptions id="context-parallelism">
<hfoption id="FLUX-1.dev">

The code sample below combines First Block Cache, fp8 dynamic quantization, torch.compile, and Context Parallelism for the fastest inference speed.

```python
import time
import torch
import torch.distributed as dist
from diffusers import FluxPipeline

dist.init_process_group()

torch.cuda.set_device(dist.get_rank())

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16,
).to("cuda")

from para_attn.context_parallel import init_context_parallel_mesh
from para_attn.context_parallel.diffusers_adapters import parallelize_pipe
from para_attn.parallel_vae.diffusers_adapters import parallelize_vae

mesh = init_context_parallel_mesh(
    pipe.device.type,
    max_ring_dim_size=2,
)
parallelize_pipe(
    pipe,
    mesh=mesh,
)
parallelize_vae(pipe.vae, mesh=mesh._flatten())

from para_attn.first_block_cache.diffusers_adapters import apply_cache_on_pipe

apply_cache_on_pipe(
    pipe,
    residual_diff_threshold=0.12,  # Use a larger value to make the cache take effect
)

from torchao.quantization import quantize_, float8_dynamic_activation_float8_weight, float8_weight_only

quantize_(pipe.text_encoder, float8_weight_only())
quantize_(pipe.transformer, float8_dynamic_activation_float8_weight())
torch._inductor.config.reorder_for_compute_comm_overlap = True
pipe.transformer = torch.compile(
   pipe.transformer, mode="max-autotune-no-cudagraphs",
)

# Enable memory savings
# pipe.enable_model_cpu_offload(gpu_id=dist.get_rank())
# pipe.enable_sequential_cpu_offload(gpu_id=dist.get_rank())

for i in range(2):
    begin = time.time()
    image = pipe(
        "A cat holding a sign that says hello world",
        num_inference_steps=28,
        output_type="pil" if dist.get_rank() == 0 else "pt",
    ).images[0]
    end = time.time()
    if dist.get_rank() == 0:
        if i == 0:
            print(f"Warm up time: {end - begin:.2f}s")
        else:
            print(f"Time: {end - begin:.2f}s")

if dist.get_rank() == 0:
    print("Saving image to flux.png")
    image.save("flux.png")

dist.destroy_process_group()
```

Save to `run_flux.py` and launch it with [torchrun](https://pytorch.org/docs/stable/elastic/run.html).

```bash
# Use --nproc_per_node to specify the number of GPUs
torchrun --nproc_per_node=2 run_flux.py
```

Inference speed is reduced to 8.20 seconds compared to the baseline, or 3.21x faster, with 2 NVIDIA L20 GPUs. On 4 L20s, inference speed is 3.90 seconds, or 6.75x faster.

</hfoption>
<hfoption id="HunyuanVideo">

The code sample below combines First Block Cache and Context Parallelism for the fastest inference speed.

```python
import time
import torch
import torch.distributed as dist
from diffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel
from diffusers.utils import export_to_video

dist.init_process_group()

torch.cuda.set_device(dist.get_rank())

model_id = "tencent/HunyuanVideo"
transformer = HunyuanVideoTransformer3DModel.from_pretrained(
    model_id,
    subfolder="transformer",
    torch_dtype=torch.bfloat16,
    revision="refs/pr/18",
)
pipe = HunyuanVideoPipeline.from_pretrained(
    model_id,
    transformer=transformer,
    torch_dtype=torch.float16,
    revision="refs/pr/18",
).to("cuda")

from para_attn.context_parallel import init_context_parallel_mesh
from para_attn.context_parallel.diffusers_adapters import parallelize_pipe
from para_attn.parallel_vae.diffusers_adapters import parallelize_vae

mesh = init_context_parallel_mesh(
    pipe.device.type,
)
parallelize_pipe(
    pipe,
    mesh=mesh,
)
parallelize_vae(pipe.vae, mesh=mesh._flatten())

from para_attn.first_block_cache.diffusers_adapters import apply_cache_on_pipe

apply_cache_on_pipe(pipe)

# from torchao.quantization import quantize_, float8_dynamic_activation_float8_weight, float8_weight_only
#
# torch._inductor.config.reorder_for_compute_comm_overlap = True
#
# quantize_(pipe.text_encoder, float8_weight_only())
# quantize_(pipe.transformer, float8_dynamic_activation_float8_weight())
# pipe.transformer = torch.compile(
#    pipe.transformer, mode="max-autotune-no-cudagraphs",
# )

# Enable memory savings
pipe.vae.enable_tiling()
# pipe.enable_model_cpu_offload(gpu_id=dist.get_rank())
# pipe.enable_sequential_cpu_offload(gpu_id=dist.get_rank())

for i in range(2):
    begin = time.time()
    output = pipe(
        prompt="A cat walks on the grass, realistic",
        height=720,
        width=1280,
        num_frames=129,
        num_inference_steps=1 if i == 0 else 30,
        output_type="pil" if dist.get_rank() == 0 else "pt",
    ).frames[0]
    end = time.time()
    if dist.get_rank() == 0:
        if i == 0:
            print(f"Warm up time: {end - begin:.2f}s")
        else:
            print(f"Time: {end - begin:.2f}s")

if dist.get_rank() == 0:
    print("Saving video to hunyuan_video.mp4")
    export_to_video(output, "hunyuan_video.mp4", fps=15)

dist.destroy_process_group()
```

Save to `run_hunyuan_video.py` and launch it with [torchrun](https://pytorch.org/docs/stable/elastic/run.html).

```bash
# Use --nproc_per_node to specify the number of GPUs
torchrun --nproc_per_node=8 run_hunyuan_video.py
```

Inference speed is reduced to 649.23 seconds compared to the baseline, or 5.66x faster, with 8 NVIDIA L20 GPUs.

</hfoption>
</hfoptions>

## Benchmarks

<hfoptions id="conclusion">
<hfoption id="FLUX-1.dev">

| GPU Type | Number of GPUs | Optimizations | Wall Time (s) | Speedup |
| - | - | - | - | - |
| NVIDIA L20 | 1 | Baseline | 26.36 | 1.00x |
| NVIDIA L20 | 1 | FBCache (rdt=0.08) | 17.01 | 1.55x |
| NVIDIA L20 | 1 | FP8 DQ | 13.40 | 1.96x |
| NVIDIA L20 | 1 | FBCache (rdt=0.12) + FP8 DQ | 7.56 | 3.48x |
| NVIDIA L20 | 2 | FBCache (rdt=0.12) + FP8 DQ + CP | 4.92 | 5.35x |
| NVIDIA L20 | 4 | FBCache (rdt=0.12) + FP8 DQ + CP | 3.90 | 6.75x |

</hfoption>
<hfoption id="HunyuanVideo">

| GPU Type | Number of GPUs | Optimizations | Wall Time (s) | Speedup |
| - | - | - | - | - |
| NVIDIA L20 | 1 | Baseline | 3675.71 | 1.00x |
| NVIDIA L20 | 1 | FBCache | 2271.06 | 1.62x |
| NVIDIA L20 | 2 | FBCache + CP | 1132.90 | 3.24x |
| NVIDIA L20 | 4 | FBCache + CP | 718.15 | 5.12x |
| NVIDIA L20 | 8 | FBCache + CP | 649.23 | 5.66x |

</hfoption>
</hfoptions>
