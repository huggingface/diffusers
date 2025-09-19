<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Reproducibility

Diffusion is a random process that generates a different output every time. For certain situations like testing and replicating results, you want to generate the same result each time, across releases and platforms within a certain tolerance range.

This guide will show you how to control sources of randomness and enable deterministic algorithms.

## Generator

Pipelines rely on [torch.randn](https://pytorch.org/docs/stable/generated/torch.randn.html), which uses a different random seed each time, to create the initial noisy tensors. To generate the same output on a CPU or GPU, use a [Generator](https://docs.pytorch.org/docs/stable/generated/torch.Generator.html) to manage how random values are generated.

> [!TIP]
> If reproducibility is important to your use case, we recommend always using a CPU `Generator`. The performance loss is often negligible and you'll generate more similar values.

<hfoptions id="generator">
<hfoption id="GPU">

The GPU uses a different random number generator than the CPU. Diffusers solves this issue with the [`~utils.torch_utils.randn_tensor`] function to create the random tensor on a CPU and then moving it to the GPU. This function is used everywhere inside the pipeline and you don't need to explicitly call it.

Use [manual_seed](https://docs.pytorch.org/docs/stable/generated/torch.manual_seed.html) as shown below to set a seed.

```py
import torch
import numpy as np
from diffusers import DDIMPipeline

ddim = DDIMPipeline.from_pretrained("google/ddpm-cifar10-32", device_map="cuda")
generator = torch.manual_seed(0)
image = ddim(num_inference_steps=2, output_type="np", generator=generator).images
print(np.abs(image).sum())
```

</hfoption>
<hfoption id="CPU">

Set `device="cpu"` in the `Generator` and use [manual_seed](https://docs.pytorch.org/docs/stable/generated/torch.manual_seed.html) to set a seed for generating random numbers.

```py
import torch
import numpy as np
from diffusers import DDIMPipeline

ddim = DDIMPipeline.from_pretrained("google/ddpm-cifar10-32")
generator = torch.Generator(device="cpu").manual_seed(0)
image = ddim(num_inference_steps=2, output_type="np", generator=generator).images
print(np.abs(image).sum())
```

</hfoption>
</hfoptions>

The `Generator` object should be passed to the pipeline instead of an integer seed. `Generator` maintains a *random state* that is consumed and modified when used. Once consumed, the same `Generator` object produces different results in subsequent calls, even across different pipelines, because it's *state* has changed.

```py
generator = torch.manual_seed(0)

for _ in range(5):
-    image = pipeline(prompt, generator=generator)
+    image = pipeline(prompt, generator=torch.manual_seed(0))
```

## Deterministic algorithms

PyTorch supports [deterministic algorithms](https://docs.pytorch.org/docs/stable/notes/randomness.html#avoiding-nondeterministic-algorithms) - where available - for certain operations so they produce the same results. Deterministic algorithms may be slower and decrease performance.

Use Diffusers' [enable_full_determinism](https://github.com/huggingface/diffusers/blob/142f353e1c638ff1d20bd798402b68f72c1ebbdd/src/diffusers/utils/testing_utils.py#L861) function to enable deterministic algorithms.

```py
import torch
from diffusers_utils import enable_full_determinism

enable_full_determinism()
```

Under the hood, `enable_full_determinism` works by:

- Setting the environment variable [CUBLAS_WORKSPACE_CONFIG](https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility) to `:16:8` to only use one buffer size during rntime. Non-deterministic behavior occurs when operations are used in more than one CUDA stream.
- Disabling benchmarking to find the fastest convolution operation by setting `torch.backends.cudnn.benchmark=False`. Non-deterministic behavior occurs because the benchmark may select different algorithms each time depending on hardware or benchmarking noise.
- Disabling TensorFloat32 (TF32) operations in favor of more precise and consistent full-precision operations.


## Resources

We strongly recommend reading PyTorch's developer notes about [Reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html). You can try to limit randomness, but it is not *guaranteed* even with an identical seed.