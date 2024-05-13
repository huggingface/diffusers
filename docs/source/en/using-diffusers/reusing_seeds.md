<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Reproducible pipelines

Diffusion models are inherently random which is what allows it to generate different outputs every time it is run. But there are certain times when you want to generate the same output every time, like when you're testing, replicating results, and even [improving image quality](#deterministic-batch-generation). While you can't expect to get identical results across platforms, you can expect reproducible results across releases and platforms within a certain tolerance range (though even this may vary).

This guide will show you how to control randomness for deterministic generation on a CPU and GPU.

> [!TIP]
> We strongly recommend reading PyTorch's [statement about reproducibility](https://pytorch.org/docs/stable/notes/randomness.html):
>
> "Completely reproducible results are not guaranteed across PyTorch releases, individual commits, or different platforms. Furthermore, results may not be reproducible between CPU and GPU executions, even when using identical seeds."

## Control randomness

During inference, pipelines rely heavily on random sampling operations which include creating the
Gaussian noise tensors to denoise and adding noise to the scheduling step.

Take a look at the tensor values in the [`DDIMPipeline`] after two inference steps.

```python
from diffusers import DDIMPipeline
import numpy as np

ddim = DDIMPipeline.from_pretrained( "google/ddpm-cifar10-32", use_safetensors=True)
image = ddim(num_inference_steps=2, output_type="np").images
print(np.abs(image).sum())
```

Running the code above prints one value, but if you run it again you get a different value.

Each time the pipeline is run, [torch.randn](https://pytorch.org/docs/stable/generated/torch.randn.html) uses a different random seed to create the Gaussian noise tensors. This leads to a different result each time it is run and enables the diffusion pipeline to generate a different random image each time.

But if you need to reliably generate the same image, that depends on whether you're running the pipeline on a CPU or GPU.

> [!TIP]
> It might seem unintuitive to pass `Generator` objects to a pipeline instead of the integer value representing the seed. However, this is the recommended design when working with probabilistic models in PyTorch because a `Generator` is a *random state* that can be passed to multiple pipelines in a sequence. As soon as the `Generator` is consumed, the *state* is changed in place which means even if you passed the same `Generator` to a different pipeline, it won't produce the same result because the state is already changed.

<hfoptions id="hardware">
<hfoption id="CPU">

To generate reproducible results on a CPU, you'll need to use a PyTorch [Generator](https://pytorch.org/docs/stable/generated/torch.Generator.html) and set a seed. Now when you run the code, it always prints a value of `1491.1711` because the `Generator` object with the seed is passed to all the random functions in the pipeline. You should get a similar, if not the same, result on whatever hardware and PyTorch version you're using.

```python
import torch
import numpy as np
from diffusers import DDIMPipeline

ddim = DDIMPipeline.from_pretrained("google/ddpm-cifar10-32", use_safetensors=True)
generator = torch.Generator(device="cpu").manual_seed(0)
image = ddim(num_inference_steps=2, output_type="np", generator=generator).images
print(np.abs(image).sum())
```

</hfoption>
<hfoption id="GPU">

Writing a reproducible pipeline on a GPU is a bit trickier, and full reproducibility across different hardware is not guaranteed because matrix multiplication - which diffusion pipelines require a lot of - is less deterministic on a GPU than a CPU. For example, if you run the same code example from the CPU example, you'll get a different result even though the seed is identical. This is because the GPU uses a different random number generator than the CPU.

```python
import torch
import numpy as np
from diffusers import DDIMPipeline

ddim = DDIMPipeline.from_pretrained("google/ddpm-cifar10-32", use_safetensors=True)
ddim.to("cuda")
generator = torch.Generator(device="cuda").manual_seed(0)
image = ddim(num_inference_steps=2, output_type="np", generator=generator).images
print(np.abs(image).sum())
```

To avoid this issue, Diffusers has a [`~utils.torch_utils.randn_tensor`] function for creating random noise on the CPU, and then moving the tensor to a GPU if necessary. The [`~utils.torch_utils.randn_tensor`] function is used everywhere inside the pipeline. Now you can call [torch.manual_seed](https://pytorch.org/docs/stable/generated/torch.manual_seed.html) which automatically creates a CPU `Generator` that can be passed to the pipeline even if it is being run on a GPU.

```python
import torch
import numpy as np
from diffusers import DDIMPipeline

ddim = DDIMPipeline.from_pretrained("google/ddpm-cifar10-32", use_safetensors=True)
ddim.to("cuda")
generator = torch.manual_seed(0)
image = ddim(num_inference_steps=2, output_type="np", generator=generator).images
print(np.abs(image).sum())
```

> [!TIP]
> If reproducibility is important to your use case, we recommend always passing a CPU `Generator`. The performance loss is often negligible and you'll generate more similar values than if the pipeline had been run on a GPU.

Finally, more complex pipelines such as [`UnCLIPPipeline`], are often extremely
susceptible to precision error propagation. You'll need to use
exactly the same hardware and PyTorch version for full reproducibility.

</hfoption>
</hfoptions>

## Deterministic algorithms

You can also configure PyTorch to use deterministic algorithms to create a reproducible pipeline. The downside is that deterministic algorithms may be slower than non-deterministic ones and you may observe a decrease in performance.

Non-deterministic behavior occurs when operations are launched in more than one CUDA stream. To avoid this, set the environment variable [CUBLAS_WORKSPACE_CONFIG](https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility) to `:16:8` to only use one buffer size during runtime.

PyTorch typically benchmarks multiple algorithms to select the fastest one, but if you want reproducibility, you should disable this feature because the benchmark may select different algorithms each time. Set Diffusers [enable_full_determinism](https://github.com/huggingface/diffusers/blob/142f353e1c638ff1d20bd798402b68f72c1ebbdd/src/diffusers/utils/testing_utils.py#L861) to enable deterministic algorithms.

```py
enable_full_determinism()
```

Now when you run the same pipeline twice, you'll get identical results.

```py
import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", use_safetensors=True).to("cuda")
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
g = torch.Generator(device="cuda")

prompt = "A bear is playing a guitar on Times Square"

g.manual_seed(0)
result1 = pipe(prompt=prompt, num_inference_steps=50, generator=g, output_type="latent").images

g.manual_seed(0)
result2 = pipe(prompt=prompt, num_inference_steps=50, generator=g, output_type="latent").images

print("L_inf dist =", abs(result1 - result2).max())
"L_inf dist = tensor(0., device='cuda:0')"
```

## Deterministic batch generation

A practical application of creating reproducible pipelines is *deterministic batch generation*. You generate a batch of images and select one image to improve with a more detailed prompt. The main idea is to pass a list of [Generator's](https://pytorch.org/docs/stable/generated/torch.Generator.html) to the pipeline and tie each `Generator` to a seed so you can reuse it.

Let's use the [runwayml/stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5) checkpoint and generate a batch of images.

```py
import torch
from diffusers import DiffusionPipeline
from diffusers.utils import make_image_grid

pipeline = DiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, use_safetensors=True
)
pipeline = pipeline.to("cuda")
```

Define four different `Generator`s and assign each `Generator` a seed (`0` to `3`). Then generate a batch of images and pick one to iterate on.

> [!WARNING]
> Use a list comprehension that iterates over the batch size specified in `range()` to create a unique `Generator` object for each image in the batch. If you multiply the `Generator` by the batch size integer, it only creates *one* `Generator` object that is used sequentially for each image in the batch.
>
> ```py
> [torch.Generator().manual_seed(seed)] * 4
> ```

```python
generator = [torch.Generator(device="cuda").manual_seed(i) for i in range(4)]
prompt = "Labrador in the style of Vermeer"
images = pipeline(prompt, generator=generator, num_images_per_prompt=4).images[0]
make_image_grid(images, rows=2, cols=2)
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/diffusers/diffusers-images-docs/resolve/main/reusabe_seeds.jpg"/>
</div>

Let's improve the first image (you can choose any image you want) which corresponds to the `Generator` with seed `0`. Add some additional text to your prompt and then make sure you reuse the same `Generator` with seed `0`. All the generated images should resemble the first image.

```python
prompt = [prompt + t for t in [", highly realistic", ", artsy", ", trending", ", colorful"]]
generator = [torch.Generator(device="cuda").manual_seed(0) for i in range(4)]
images = pipeline(prompt, generator=generator).images
make_image_grid(images, rows=2, cols=2)
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/diffusers/diffusers-images-docs/resolve/main/reusabe_seeds_2.jpg"/>
</div>
