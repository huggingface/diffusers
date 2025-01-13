# Fastest HunyuanVideo Inference with ParaAttention

[![](https://mermaid.ink/img/pako:eNptktuK2zAQhl9lEIS01HZsOXESXxT20NKFtgQKW-hqLxR7YgtsKcjjbbwh795x0m4PVCDQfNJoTv9RFK5EkYswDJUtnN2ZKlcWeB2Gm1p7-mmN67spqc4hSeL4N6zRVDXlIBcMz79MJkdjDaPjlGpscZrDdOc8djQ9wWkyUfYwFOPX4RZJQ-28eXaWdKMsGWoQlPiqmwbItAjsCBVa9JqMrcBZhCdTouNkqIYPvbYD7_sRBZDIVXxYyviQyHUAaQwd4b4L2As-39_d3l3BRxkHF9eN9vqKCDmms0pwUqE-mA4elLjWHTbGohIB5_L--kYX9d8GvIGbzSv5-j9w_gtu_oArho_KDpcQSnTIrS47JSCGMHwL83hsqbJb7eEhzWQWpWkAUi6TKM64riSV0ZozXyarKFkEkM3XkUwfRSBa9K02JU_wOM5EiXPLlcj5uOU6xspO_E735L4MthA5-R4D4V1f1SLf6aZjq9-XmvDW6Mrr9oXutf3mXPvPq3elIedfYON0iWweBQ37UUmV6YgDXrQ08t43jGuifZfPZuN1VPEE-m1UuHbWmXLUQv20zmZc-ErLFLNlqhdpWhbbZL3ayXmyK5dxIrU4nQKB5_ifLrI9q_f0AzkD3HY?type=png)](https://mermaid.live/edit#pako:eNptktuK2zAQhl9lEIS01HZsOXESXxT20NKFtgQKW-hqLxR7YgtsKcjjbbwh795x0m4PVCDQfNJoTv9RFK5EkYswDJUtnN2ZKlcWeB2Gm1p7-mmN67spqc4hSeL4N6zRVDXlIBcMz79MJkdjDaPjlGpscZrDdOc8djQ9wWkyUfYwFOPX4RZJQ-28eXaWdKMsGWoQlPiqmwbItAjsCBVa9JqMrcBZhCdTouNkqIYPvbYD7_sRBZDIVXxYyviQyHUAaQwd4b4L2As-39_d3l3BRxkHF9eN9vqKCDmms0pwUqE-mA4elLjWHTbGohIB5_L--kYX9d8GvIGbzSv5-j9w_gtu_oArho_KDpcQSnTIrS47JSCGMHwL83hsqbJb7eEhzWQWpWkAUi6TKM64riSV0ZozXyarKFkEkM3XkUwfRSBa9K02JU_wOM5EiXPLlcj5uOU6xspO_E735L4MthA5-R4D4V1f1SLf6aZjq9-XmvDW6Mrr9oXutf3mXPvPq3elIedfYON0iWweBQ37UUmV6YgDXrQ08t43jGuifZfPZuN1VPEE-m1UuHbWmXLUQv20zmZc-ErLFLNlqhdpWhbbZL3ayXmyK5dxIrU4nQKB5_ifLrI9q_f0AzkD3HY)

## Introduction

During the past year, we have seen the rapid development of video generation models with the release of several open-source models, such as [HunyuanVideo](https://huggingface.co/tencent/HunyuanVideo), [CogVideoX](https://huggingface.co/THUDM/CogVideoX-5b) and [Mochi](https://huggingface.co/genmo/mochi-1-preview).
It is very exciting to see that open source video models are going to beat closed source.
However, the inference speed of these models is still a bottleneck for real-time applications and deployment.

In this article, we will use [ParaAttention](https://github.com/chengzeyi/ParaAttention), a library implements **Context Parallelism** and **First Block Cache**, as well as other techniques like `torch.compile` and **FP8 Dynamic Quantization**, to achieve the fastest inference speed for HunyuanVideo.
If you want to speed up other models like `CogVideoX`, `Mochi` or `FLUX`, you can also follow the same steps in this article.

**We set up our experiments on NVIDIA L20 GPUs, which only have PCIe support.**
**If you have NVIDIA A100 or H100 GPUs with NVLink support, you can achieve a better speedup with context parallelism, especially when the number of GPUs is large.**

## HunyuanVideo Inference with `diffusers`

Like many other generative AI models, HunyuanVideo has its official code repository and is supported by other frameworks like `diffusers` and `ComfyUI`.
In this article, we will focus on optimizing the inference speed of HunyuanVideo with `diffusers`.
To use HunyuanVideo with `diffusers`, we need to install its latest version:

```bash
pip3 install -U diffusers
```

Then, we can load the model and generate video frames with the following code:

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

However, most people will experience OOM (Out of Memory) errors when running the above code.
This is because the HunyuanVideo transformer model is relatively large and it has a quite large text encoder.
Besides, HunyuanVideo requires a variable length of text conditions and the `diffusers` library implements this feature with a `attn_mask` in `scaled_dot_product_attention`.
The size of `attn_mask` is proportional to the square of the input sequence length, which is crazy when we increase the resolution and the number of frames of the inference!
Luckily, we can use ParaAttention to solve this problem.
In ParaAttention, we patch the original implementation in `diffusers` to cut the text conditions before calling `scaled_dot_product_attention`.
We implement this in our `apply_cache_on_pipe` function so we can call it after loading the model:

```bash
pip3 install -U para-attn
```

```python
pipe = HunyuanVideoPipeline.from_pretrained(
    model_id,
    transformer=transformer,
    torch_dtype=torch.float16,
    revision="refs/pr/18",
).to("cuda")

from para_attn.first_block_cache.diffusers_adapters import apply_cache_on_pipe

apply_cache_on_pipe(pipe, residual_diff_threshold=0.0)
```

We pass `residual_diff_threshold=0.0` to `apply_cache_on_pipe` to disable the cache mechanism now, because we will enable it later.
Here, we only want it to cut the text conditions to avoid OOM errors.
If you still experience OOM errors, you can try calling `pipe.enable_model_cpu_offload` or `pipe.enable_sequential_cpu_offload` after calling `apply_cache_on_pipe`.

This is our baseline.
On one single NVIDIA L20 GPU, we can generate 129 frames with 720p resolution in 30 inference steps in 3675.71 seconds.

## Apply First Block Cache on HunyuanVideo

By caching the output of the transformer blocks in the transformer model and resuing them in the next inference steps, we can reduce the computation cost and make the inference faster.
However, it is hard to decide when to reuse the cache to ensure the quality of the generated video.
Recently, [TeaCache](https://github.com/ali-vilab/TeaCache) suggests that we can use the timestep embedding to approximate the difference among model outputs.
And [AdaCache](https://adacache-dit.github.io) also shows that caching can contribute grant significant inference speedups without sacrificing the generation quality, across multiple video DiT baselines.
However, TeaCache is still a bit complex as it needs a rescaling strategy to ensure the accuracy of the cache.
In ParaAttention, we find that we can directly use **the residual difference of the first transformer block output** to approximate the difference among model outputs.
When the difference is small enough, we can reuse the residual difference of previous inference steps, meaning that we in fact skip this denoising step.
This has been proved to be effective in our experiments and we can achieve an up to 2x speedup on HunyuanVideo inference with very good quality.

<figure>
    <img src="https://adacache-dit.github.io/clarity/images/adacache.png" alt="Cache in Diffusion Transformer" />
    <figcaption>How AdaCache works, First Block Cache is a variant of it</figcaption>
</figure>

To apply the first block cache on HunyuanVideo, we can call `apply_cache_on_pipe` with `residual_diff_threshold=0.06`, which is the default value for HunyuanVideo.

```python
apply_cache_on_pipe(pipe, residual_diff_threshold=0.06)
```

### HunyuanVideo without FBCache

https://github.com/user-attachments/assets/883d771a-e74e-4081-aa2a-416985d6c713

### HunyuanVideo with FBCache

https://github.com/user-attachments/assets/f77c2f58-2b59-4dd1-a06a-a36974cb1e40

We observe that the first block cache is very effective in speeding up the inference, and maintaining nearly no quality loss in the generated video.
Now, on one single NVIDIA L20 GPU, we can generate 129 frames with 720p resolution in 30 inference steps in 2271.06 seconds. This is a 1.62x speedup compared to the baseline.

## Quantize the model into FP8

To further speed up the inference and reduce memory usage, we can quantize the model into FP8 with dynamic quantization.
We must quantize both the activation and weight of the transformer model to utilize the 8-bit **Tensor Cores** on NVIDIA GPUs.
Here, we use  `float8_weight_only` and `float8_dynamic_activation_float8_weight`to quantize the text encoder and transformer model respectively.
The default quantization method is per tensor quantization. If your GPU supports row-wise quantization, you can also try it for better accuracy.
[diffusers-torchao](https://github.com/sayakpaul/diffusers-torchao) provides a really good tutorial on how to quantize models in `diffusers` and achieve a good speedup.
Here, we simply install the latest `torchao` that is capable of quantizing HunyuanVideo.
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

**Note**: we find that dynamic quantization can significantly change the distribution of the model output, so you might need to tweak the `residual_diff_threshold` to a larger value to make it take effect.

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

The NVIDIA L20 GPU only has 48GB memory and could face OOM errors after compiling the model and not calling `enable_model_cpu_offload`,
because the HunyuanVideo has very large activation tensors when running with high resolution and large number of frames.
So here we skip measuring the speedup with quantization and compilation on one single NVIDIA L20 GPU and choose to use context parallelism to release the memory pressure.
If you want to run HunyuanVideo with `torch.compile` on GPUs with less than 80GB memory, you can try reducing the resolution and the number of frames to avoid OOM errors.

Due to the fact that large video generation models usually have performance bottleneck on the attention computation rather than the fully connected layers, we don't observe a significant speedup with quantization and compilation.
However, models like `FLUX` and `SD3` can benefit a lot from quantization and compilation, it is suggested to try it for these models.

## Parallelize the inference with Context Parallelism

A lot faster than before, right? But we are not satisfied with the speedup we have achieved so far.
If we want to accelerate the inference further, we can use context parallelism to parallelize the inference.
Libraries like [xDit](https://github.com/xdit-project/xDiT) and our [ParaAttention](https://github.com/chengzeyi/ParaAttention) provide ways to scale up the inference with multiple GPUs.
In ParaAttention, we design our API in a compositional way so that we can combine context parallelism with first block cache and dynamic quantization all together.
We provide very detailed instructions and examples of how to scale up the inference with multiple GPUs in our ParaAttention repository.
Users can easily launch the inference with multiple GPUs by calling `torchrun`.
If there is a need to make the inference process persistent and serviceable, it is suggested to use `torch.multiprocessing` to write your own inference processor, which can eliminate the overhead of launching the process and loading and recompiling the model.

Below is our ultimate code to achieve the fastest HunyuanVideo inference:

```python
import time
import torch
import torch.distributed as dist
from diffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel
from diffusers.utils import export_to_video

dist.init_process_group()

torch.cuda.set_device(dist.get_rank())

# [rank1]: RuntimeError: Expected mha_graph->execute(handle, variant_pack, workspace_ptr.get()).is_good() to be true, but got false.  (Could this error message be improved?  If so, please report an enhancement request to PyTorch.)
torch.backends.cuda.enable_cudnn_sdp(False)

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

We save the above code to `run_hunyuan_video.py` and run it with `torchrun`:

```bash
torchrun --nproc_per_node=8 run_hunyuan_video.py
```

With 8 NVIDIA L20 GPUs, we can generate 129 frames with 720p resolution in 30 inference steps in 649.23 seconds. This is a 5.66x speedup compared to the baseline!

## Conclusion

| GPU Type | Number of GPUs | Optimizations | Wall Time (s) | Speedup |
| - | - | - | - | - |
| NVIDIA L20 | 1 | Baseline | 3675.71 | 1.00x |
| NVIDIA L20 | 1 | FBCache | 2271.06 | 1.62x |
| NVIDIA L20 | 2 | FBCache + CP | 1132.90 | 3.24x |
| NVIDIA L20 | 4 | FBCache + CP | 718.15 | 5.12x |
| NVIDIA L20 | 8 | FBCache + CP | 649.23 | 5.66x |
