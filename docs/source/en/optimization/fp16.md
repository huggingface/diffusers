<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Accelerate inference

Diffusion models are slow at inference because generation is an iterative process where noise is gradually refined into an image or video over a certain number of "steps". To speedup this process, you can try experimenting with different [schedulers](../api/schedulers/overview), reduce the precision of the model weights for faster computations, use more memory-efficient attention mechanisms, and more.

Combine and use these techniques together to make inference faster than using any single technique on its own.

This guide will go over how to accelerate inference.

## Model data type

The precision and data type of the model weights affect inference speed because a higher precision requires more memory to load and more time to perform the computations. PyTorch loads model weights in float32 or full precision by default, so changing the data type is a simple way to quickly get faster inference.

<hfoptions id="dtypes">
<hfoption id="bfloat16">

bfloat16 is similar to float16 but it is more robust to numerical errors. Hardware support for bfloat16 varies, but most modern GPUs are capable of supporting bfloat16.

```py
import torch
from diffusers import StableDiffusionXLPipeline

pipeline = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.bfloat16
).to("cuda")

prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
pipeline(prompt, num_inference_steps=30).images[0]
```

</hfoption>
<hfoption id="float16">

float16 is similar to bfloat16 but may be more prone to numerical errors.

```py
import torch
from diffusers import StableDiffusionXLPipeline

pipeline = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
).to("cuda")

prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
pipeline(prompt, num_inference_steps=30).images[0]
```

</hfoption>
<hfoption id="TensorFloat-32">

[TensorFloat-32 (tf32)](https://blogs.nvidia.com/blog/2020/05/14/tensorfloat-32-precision-format/) mode is supported on NVIDIA Ampere GPUs and it computes the convolution and matrix multiplication operations in tf32. Storage and other operations are kept in float32. This enables significantly faster computations when combined with bfloat16 or float16.

PyTorch only enables tf32 mode for convolutions by default and you'll need to explicitly enable it for matrix multiplications.

```py
import torch
from diffusers import StableDiffusionXLPipeline

torch.backends.cuda.matmul.allow_tf32 = True

pipeline = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.bfloat16
).to("cuda")

prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
pipeline(prompt, num_inference_steps=30).images[0]
```

Refer to the [mixed precision training](https://huggingface.co/docs/transformers/en/perf_train_gpu_one#mixed-precision) docs for more details.

</hfoption>
</hfoptions>

## Scaled dot product attention

> [!TIP]
> Memory-efficient attention optimizes for inference speed *and* [memory usage](./memory#memory-efficient-attention)!

[Scaled dot product attention (SDPA)](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) implements several attention backends, [FlashAttention](https://github.com/Dao-AILab/flash-attention), [xFormers](https://github.com/facebookresearch/xformers), and a native C++ implementation. It automatically selects the most optimal backend for your hardware.

SDPA is enabled by default if you're using PyTorch >= 2.0 and no additional changes are required to your code. You could try experimenting with other attention backends though if you'd like to choose your own. The example below uses the [torch.nn.attention.sdpa_kernel](https://pytorch.org/docs/stable/generated/torch.nn.attention.sdpa_kernel.html) context manager to enable efficient attention.

```py
from torch.nn.attention import SDPBackend, sdpa_kernel
import torch
from diffusers import StableDiffusionXLPipeline

pipeline = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.bfloat16
).to("cuda")

prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"

with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
  image = pipeline(prompt, num_inference_steps=30).images[0]
```

## torch.compile

[torch.compile](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html) accelerates inference by compiling PyTorch code and operations into optimized kernels. Diffusers typically compiles the more compute-intensive models like the UNet, transformer, or VAE.

Enable the following compiler settings for maximum speed (refer to the [full list](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/config.py) for more options).

```py
import torch
from diffusers import StableDiffusionXLPipeline

torch._inductor.config.conv_1x1_as_mm = True
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.epilogue_fusion = False
torch._inductor.config.coordinate_descent_check_all_directions = True
```

Load and compile the UNet and VAE. There are several different modes you can choose from, but `"max-autotune"` optimizes for the fastest speed by compiling to a CUDA graph. CUDA graphs effectively reduces the overhead by launching multiple GPU operations through a single CPU operation.

> [!TIP]
> With PyTorch 2.3.1, you can control the caching behavior of torch.compile. This is particularly beneficial for compilation modes like `"max-autotune"` which performs a grid-search over several compilation flags to find the optimal configuration. Learn more in the [Compile Time Caching in torch.compile](https://pytorch.org/tutorials/recipes/torch_compile_caching_tutorial.html) tutorial.

Changing the memory layout to [channels_last](./memory#torchchannels_last) also optimizes memory and inference speed.

```py
pipeline = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
).to("cuda")
pipeline.unet.to(memory_format=torch.channels_last)
pipeline.vae.to(memory_format=torch.channels_last)
pipeline.unet = torch.compile(
    pipeline.unet, mode="max-autotune", fullgraph=True
)
pipeline.vae.decode = torch.compile(
    pipeline.vae.decode,
    mode="max-autotune",
    fullgraph=True
)

prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
pipeline(prompt, num_inference_steps=30).images[0]
```

Compilation is slow the first time, but once compiled, it is significantly faster. Try to only use the compiled pipeline on the same type of inference operations. Calling the compiled pipeline on a different image size retriggers compilation which is slow and inefficient.

### Dynamic shape compilation

> [!TIP]
> Make sure to always use the nightly version of PyTorch for better support.

`torch.compile` keeps track of input shapes and conditions, and if these are different, it recompiles the model. For example, if a model is compiled on a 1024x1024 resolution image and used on an image with a different resolution, it triggers recompilation.

To avoid recompilation, add `dynamic=True` to try and generate a more dynamic kernel to avoid recompilation when conditions change.

```diff
+ torch.fx.experimental._config.use_duck_shape = False
+ pipeline.unet = torch.compile(
    pipeline.unet, fullgraph=True, dynamic=True
)
```

Specifying `use_duck_shape=False` instructs the compiler if it should use the same symbolic variable to represent input sizes that are the same. For more details, check out this [comment](https://github.com/huggingface/diffusers/pull/11327#discussion_r2047659790).

Not all models may benefit from dynamic compilation out of the box and may require changes. Refer to this [PR](https://github.com/huggingface/diffusers/pull/11297/) that improved the [`AuraFlowPipeline`] implementation to benefit from dynamic compilation.

Feel free to open an issue if dynamic compilation doesn't work as expected for a Diffusers model.

### Regional compilation

[Regional compilation](https://docs.pytorch.org/tutorials/recipes/regional_compilation.html) trims cold-start latency by only compiling the *small and frequently-repeated block(s)* of a model - typically a transformer layer - and enables reusing compiled artifacts for every subsequent occurrence.
For many diffusion architectures, this delivers the same runtime speedups as full-graph compilation and reduces compile time by 8–10x.

Use the [`~ModelMixin.compile_repeated_blocks`] method, a helper that wraps `torch.compile`, on any component such as the transformer model as shown below.

```py
# pip install -U diffusers
import torch
from diffusers import StableDiffusionXLPipeline

pipeline = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
).to("cuda")

# compile only the repeated transformer layers inside the UNet
pipeline.unet.compile_repeated_blocks(fullgraph=True)
```

To enable regional compilation for a new model, add a `_repeated_blocks` attribute to a model class containing the class names (as strings) of the blocks you want to compile.

```py
class MyUNet(ModelMixin):
    _repeated_blocks = ("Transformer2DModel",)  # ← compiled by default
```

> [!TIP]
> For more regional compilation examples, see the reference [PR](https://github.com/huggingface/diffusers/pull/11705).

There is also a [compile_regions](https://github.com/huggingface/accelerate/blob/273799c85d849a1954a4f2e65767216eb37fa089/src/accelerate/utils/other.py#L78) method in [Accelerate](https://huggingface.co/docs/accelerate/index) that automatically selects candidate blocks in a model to compile. The remaining graph is compiled separately. This is useful for quick experiments because there aren't as many options for you to set which blocks to compile or adjust compilation flags.

```py
# pip install -U accelerate
import torch
from diffusers import StableDiffusionXLPipeline
from accelerate.utils import compile_regions

pipeline = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
).to("cuda")
pipeline.unet = compile_regions(pipeline.unet, mode="reduce-overhead", fullgraph=True)
```

[`~ModelMixin.compile_repeated_blocks`] is intentionally explicit. List the blocks to repeat in `_repeated_blocks` and the helper only compiles those blocks. It offers predictable behavior and easy reasoning about cache reuse in one line of code.

### Graph breaks

It is important to specify `fullgraph=True` in torch.compile to ensure there are no graph breaks in the underlying model. This allows you to take advantage of torch.compile without any performance degradation. For the UNet and VAE, this changes how you access the return variables.

```diff
- latents = unet(
-   latents, timestep=timestep, encoder_hidden_states=prompt_embeds
-).sample

+ latents = unet(
+   latents, timestep=timestep, encoder_hidden_states=prompt_embeds, return_dict=False
+)[0]
```

### GPU sync

The `step()` function is [called](https://github.com/huggingface/diffusers/blob/1d686bac8146037e97f3fd8c56e4063230f71751/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl.py#L1228) on the scheduler each time after the denoiser makes a prediction, and the `sigmas` variable is [indexed](https://github.com/huggingface/diffusers/blob/1d686bac8146037e97f3fd8c56e4063230f71751/src/diffusers/schedulers/scheduling_euler_discrete.py#L476). When placed on the GPU, it introduces latency because of the communication sync between the CPU and GPU. It becomes more evident when the denoiser has already been compiled.

In general, the `sigmas` should [stay on the CPU](https://github.com/huggingface/diffusers/blob/35a969d297cba69110d175ee79c59312b9f49e1e/src/diffusers/schedulers/scheduling_euler_discrete.py#L240) to avoid the communication sync and latency.

> [!TIP]
> Refer to the [torch.compile and Diffusers: A Hands-On Guide to Peak Performance](https://pytorch.org/blog/torch-compile-and-diffusers-a-hands-on-guide-to-peak-performance/) blog post for maximizing performance with `torch.compile` for diffusion models.

### Benchmarks

Refer to the [diffusers/benchmarks](https://huggingface.co/datasets/diffusers/benchmarks) dataset to see inference latency and memory usage data for compiled pipelines.

The [diffusers-torchao](https://github.com/sayakpaul/diffusers-torchao#benchmarking-results) repository also contains benchmarking results for compiled versions of Flux and CogVideoX.

## Dynamic quantization

[Dynamic quantization](https://pytorch.org/tutorials/recipes/recipes/dynamic_quantization.html) improves inference speed by reducing precision to enable faster math operations. This particular type of quantization determines how to scale the activations based on the data at runtime rather than using a fixed scaling factor. As a result, the scaling factor is more accurately aligned with the data.

The example below applies [dynamic int8 quantization](https://pytorch.org/tutorials/recipes/recipes/dynamic_quantization.html) to the UNet and VAE with the [torchao](../quantization/torchao) library.

> [!TIP]
> Refer to our [torchao](../quantization/torchao) docs to learn more about how to use the Diffusers torchao integration.

Configure the compiler tags for maximum speed.

```py
import torch
from torchao import apply_dynamic_quant
from diffusers import StableDiffusionXLPipeline

torch._inductor.config.conv_1x1_as_mm = True
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.epilogue_fusion = False
torch._inductor.config.coordinate_descent_check_all_directions = True
torch._inductor.config.force_fuse_int_mm_with_mul = True
torch._inductor.config.use_mixed_mm = True
```

Filter out some linear layers in the UNet and VAE which don't benefit from dynamic quantization with the [dynamic_quant_filter_fn](https://github.com/huggingface/diffusion-fast/blob/0f169640b1db106fe6a479f78c1ed3bfaeba3386/utils/pipeline_utils.py#L16).

```py
pipeline = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.bfloat16
).to("cuda")

apply_dynamic_quant(pipeline.unet, dynamic_quant_filter_fn)
apply_dynamic_quant(pipeline.vae, dynamic_quant_filter_fn)

prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
pipeline(prompt, num_inference_steps=30).images[0]
```

## Fused projection matrices

> [!WARNING]
> The [fuse_qkv_projections](https://github.com/huggingface/diffusers/blob/58431f102cf39c3c8a569f32d71b2ea8caa461e1/src/diffusers/pipelines/pipeline_utils.py#L2034) method is experimental and support is limited to mostly Stable Diffusion pipelines. Take a look at this [PR](https://github.com/huggingface/diffusers/pull/6179) to learn more about how to enable it for other pipelines

An input is projected into three subspaces, represented by the projection matrices Q, K, and V, in an attention block. These projections are typically calculated separately, but you can horizontally combine these into a single matrix and perform the projection in a single step. It increases the size of the matrix multiplications of the input projections and also improves the impact of quantization.

```py
pipeline.fuse_qkv_projections()
```

## Resources

- Read the [Presenting Flux Fast: Making Flux go brrr on H100s](https://pytorch.org/blog/presenting-flux-fast-making-flux-go-brrr-on-h100s/) blog post to learn more about how you can combine all of these optimizations with [TorchInductor](https://docs.pytorch.org/docs/stable/torch.compiler.html) and [AOTInductor](https://docs.pytorch.org/docs/stable/torch.compiler_aot_inductor.html) for a ~2.5x speedup using recipes from [flux-fast](https://github.com/huggingface/flux-fast).

    These recipes support AMD hardware and [Flux.1 Kontext Dev](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev).
- Read the [torch.compile and Diffusers: A Hands-On Guide to Peak Performance](https://pytorch.org/blog/torch-compile-and-diffusers-a-hands-on-guide-to-peak-performance/) blog post
to maximize performance when using `torch.compile`.