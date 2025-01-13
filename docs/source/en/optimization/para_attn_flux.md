# Fastest FLUX Inference with ParaAttention

[![](https://mermaid.ink/img/pako:eNqNUu9r2zAQ_VcOQUjDbMeWEycxbNAmKxS6kTH2g1X9oFgXW2BLQVbaZMH_-85J6WDsQ_VB0j096U733okVViHLWRiGwhTWbHWZCwM0DsdlJZ1_ifrxrJWvckiSOP4LVqjLyufApwSeXxkMTtpogk5DX2GDwxyGW-uw9cMOusFAmMOx6J8ON-glVNbp39Z4WQvjta8RBPsh6xq8bhDoIpRo0EmvTQnWIOhGlkjF-Apu77_9jJJQ4VMAScwnh34KgM-h9bhrA-LD5-93q7truOdxcLm0lk5ee4-UzRrBqJxQHnQLD4LdyBZrbVCwgKq4vVnKosIrp_z7OIrnoxd4PYfVl_9REj6Cd_C2c9os11d89DbehHiPwhwvlQrWImmlWsEghjD8ACk1X5iNdPDAsyjNqB2zKE5oSaMJfXwWTQmbRAseQBot4kcWsAZdI7Ui8U-9nIKd1RIsp-2GGtG3piOe3Hv79WgKlnu3x4A5uy8rlm9l3VK03ynpcaVl6WTziu6k-WVt8w_ro9LeulewtlIhhSfmj7vehKVuPSW82LDH964muPJ-1-bjcX8clSThfhMVthm3WvU2qp4W2Tjj2VzyFLNZKqdpqopNsphv-STZqlmccMm6LmB4zv_p4viz8bs_BMbpYw?type=png)](https://mermaid.live/edit#pako:eNqNUu9r2zAQ_VcOQUjDbMeWEycxbNAmKxS6kTH2g1X9oFgXW2BLQVbaZMH_-85J6WDsQ_VB0j096U733okVViHLWRiGwhTWbHWZCwM0DsdlJZ1_ifrxrJWvckiSOP4LVqjLyufApwSeXxkMTtpogk5DX2GDwxyGW-uw9cMOusFAmMOx6J8ON-glVNbp39Z4WQvjta8RBPsh6xq8bhDoIpRo0EmvTQnWIOhGlkjF-Apu77_9jJJQ4VMAScwnh34KgM-h9bhrA-LD5-93q7truOdxcLm0lk5ee4-UzRrBqJxQHnQLD4LdyBZrbVCwgKq4vVnKosIrp_z7OIrnoxd4PYfVl_9REj6Cd_C2c9os11d89DbehHiPwhwvlQrWImmlWsEghjD8ACk1X5iNdPDAsyjNqB2zKE5oSaMJfXwWTQmbRAseQBot4kcWsAZdI7Ui8U-9nIKd1RIsp-2GGtG3piOe3Hv79WgKlnu3x4A5uy8rlm9l3VK03ynpcaVl6WTziu6k-WVt8w_ro9LeulewtlIhhSfmj7vehKVuPSW82LDH964muPJ-1-bjcX8clSThfhMVthm3WvU2qp4W2Tjj2VzyFLNZKqdpqopNsphv-STZqlmccMm6LmB4zv_p4viz8bs_BMbpYw)

## Introduction

During the past year, we have seen the rapid development of image generation models with the release of several open-source models, such as [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) and [SD3.5-large](https://huggingface.co/stabilityai/stable-diffusion-3.5-large).
It is very exciting to see that open source image generation models are going to beat closed source.
However, the inference speed of these models is still a bottleneck for real-time applications and deployment.

In this article, we will use [ParaAttention](https://github.com/chengzeyi/ParaAttention), a library implements **Context Parallelism** and **First Block Cache**, as well as other techniques like `torch.compile` and **FP8 Dynamic Quantization**, to achieve the fastest inference speed for FLUX.1-dev.

**We set up our experiments on NVIDIA L20 GPUs, which only have PCIe support.**
**If you have NVIDIA A100 or H100 GPUs with NVLink support, you can achieve a better speedup with context parallelism, especially when the number of GPUs is large.**

## FLUX.1-dev Inference with `diffusers`

Like many other generative AI models, FLUX.1-dev has its official code repository and is supported by other frameworks like `diffusers` and `ComfyUI`.
In this article, we will focus on optimizing the inference speed of FLUX.1-dev with `diffusers`.
To use FLUX.1-dev with `diffusers`, we need to install its latest version:

```bash
pip3 install -U diffusers
```

Then, we can load the model and generate images with the following code:

```python
import time
import torch
from diffusers import FluxPipeline

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16,
).to("cuda")

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

This is our baseline.
On one single NVIDIA L20 GPU, we can generate 1 image with 1024x1024 resolution in 28 inference steps in 26.36 seconds.

## Apply First Block Cache on FLUX.1-dev

By caching the output of the transformer blocks in the transformer model and resuing them in the next inference steps, we can reduce the computation cost and make the inference faster.
However, it is hard to decide when to reuse the cache to ensure the quality of the generated image.
Recently, [TeaCache](https://github.com/ali-vilab/TeaCache) suggests that we can use the timestep embedding to approximate the difference among model outputs.
And [AdaCache](https://adacache-dit.github.io) also shows that caching can contribute grant significant inference speedups without sacrificing the generation quality, across multiple image and video DiT baselines.
However, TeaCache is still a bit complex as it needs a rescaling strategy to ensure the accuracy of the cache.
In ParaAttention, we find that we can directly use **the residual difference of the first transformer block output** to approximate the difference among model outputs.
When the difference is small enough, we can reuse the residual difference of previous inference steps, meaning that we in fact skip this denoising step.
This has been proved to be effective in our experiments and we can achieve an up to 1.5x speedup on FLUX.1-dev inference with very good quality.

<figure>
    <img src="https://adacache-dit.github.io/clarity/images/adacache.png" alt="Cache in Diffusion Transformer" />
    <figcaption>How AdaCache works, First Block Cache is a variant of it</figcaption>
</figure>

To apply the first block cache on FLUX.1-dev, we can call `apply_cache_on_pipe` with `residual_diff_threshold=0.08`, which is the default value for FLUX models.

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
| Preview | ![Original](https://github.com/chengzeyi/ParaAttention/blob/main/assets/flux_original.png) | ![FBCache rdt=0.06](https://github.com/chengzeyi/ParaAttention/blob/main/assets/flux_fbc_0.06.png) | ![FBCache rdt=0.08](https://github.com/chengzeyi/ParaAttention/blob/main/assets/flux_fbc_0.08.png) | ![FBCache rdt=0.10](https://github.com/chengzeyi/ParaAttention/blob/main/assets/flux_fbc_0.10.png) | ![FBCache rdt=0.12](https://github.com/chengzeyi/ParaAttention/raw/main/assets/flux_fbc_0.12.png) |
| Wall Time (s) | 26.36 | 21.83 | 17.01 | 16.00 | 13.78 |

We observe that the first block cache is very effective in speeding up the inference, and maintaining nearly no quality loss in the generated image.
Now, on one single NVIDIA L20 GPU, we can generate 1 image with 1024x1024 resolution in 28 inference steps in 17.01 seconds. This is a 1.55x speedup compared to the baseline.

## Quantize the model into FP8

To further speed up the inference and reduce memory usage, we can quantize the model into FP8 with dynamic quantization.
We must quantize both the activation and weight of the transformer model to utilize the 8-bit **Tensor Cores** on NVIDIA GPUs.
Here, we use  `float8_weight_only` and `float8_dynamic_activation_float8_weight`to quantize the text encoder and transformer model respectively.
The default quantization method is per tensor quantization. If your GPU supports row-wise quantization, you can also try it for better accuracy.
[diffusers-torchao](https://github.com/sayakpaul/diffusers-torchao) provides a really good tutorial on how to quantize models in `diffusers` and achieve a good speedup.
Here, we simply install the latest `torchao` that is capable of quantizing FLUX.1-dev.
If you are not familiar with `torchao` quantization, you can refer to this [documentation](https://github.com/pytorch/ao/blob/main/torchao/quantization/README.md).

```bash
pip3 install -U torch torchao
```

We also need to pass the model to `torch.compile` to gain actual speedup.
`torch.compile` with `mode="max-autotune-no-cudagraphs"` or `mode="max-autotune"` can help us to achieve the best performance by generating and selecting the best kernel for the model inference.
The compilation process could take a long time, but it is worth it.
If you are not familiar with `torch.compile`, you can refer to the [official tutorial](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html).
In this example, we only quantize the transformer model, but you can also quantize the text encoder to reduce more memory usage.
We also need to notice that the actually compilation process is done on the first time the model is called, so we need to warm up the model to measure the speedup correctly.

**Note**: we find that dynamic quantization can significantly change the distribution of the model output, so we need to change the `residual_diff_threshold` to a larger value to make it take effect.

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

We can see that the quantization and compilation process can further speed up the inference.
Now, on one single NVIDIA L20 GPU, we can generate 1 image with 1024x1024 resolution in 28 inference steps in 7.56s, which is a 3.48x speedup compared to the baseline.

## Parallelize the inference with Context Parallelism

A lot faster than before, right? But we are not satisfied with the speedup we have achieved so far.
If we want to accelerate the inference further, we can use context parallelism to parallelize the inference.
Libraries like [xDit](https://github.com/xdit-project/xDiT) and our [ParaAttention](https://github.com/chengzeyi/ParaAttention) provide ways to scale up the inference with multiple GPUs.
In ParaAttention, we design our API in a compositional way so that we can combine context parallelism with first block cache and dynamic quantization all together.
We provide very detailed instructions and examples of how to scale up the inference with multiple GPUs in our ParaAttention repository.
Users can easily launch the inference with multiple GPUs by calling `torchrun`.
If there is a need to make the inference process persistent and serviceable, it is suggested to use `torch.multiprocessing` to write your own inference processor, which can eliminate the overhead of launching the process and loading and recompiling the model.

Below is our ultimate code to achieve the fastest FLUX.1-dev inference:

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

We save the above code to `run_flux.py` and run it with `torchrun`:

```bash
# Use --nproc_per_node to specify the number of GPUs
torchrun --nproc_per_node=2 run_flux.py
```

With 2 NVIDIA L20 GPUs, we can generate 1 image with 1024x1024 resolution in 28 inference steps in 8.20 seconds, which is a 3.21x speedup compared to the baseline.
And with 4 NVIDIA L20 GPUs, we can generate 1 image with 1024x1024 resolution in 28 inference steps in 3.90 seconds, which is a 6.75x speedup compared to the baseline.

## Conclusion

| GPU Type | Number of GPUs | Optimizations | Wall Time (s) | Speedup |
| - | - | - | - | - |
| NVIDIA L20 | 1 | Baseline | 26.36 | 1.00x |
| NVIDIA L20 | 1 | FBCache (rdt=0.08) | 17.01 | 1.55x |
| NVIDIA L20 | 1 | FP8 DQ | 13.40 | 1.96x |
| NVIDIA L20 | 1 | FBCache (rdt=0.12) + FP8 DQ | 7.56 | 3.48x |
| NVIDIA L20 | 2 | FBCache (rdt=0.12) + FP8 DQ + CP | 4.92 | 5.35x |
| NVIDIA L20 | 4 | FBCache (rdt=0.12) + FP8 DQ + CP | 3.90 | 6.75x |
