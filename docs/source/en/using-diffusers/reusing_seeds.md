<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Reproducibility

Diffusion is a random process that generates a different output every time. For use cases like testing and replicating results, you want to generate the same result each time, across releases and platforms within a certain tolerance range.

This guide will show you how to control sources of randomness and enable deterministic algorithms.

## Generator

Pipelines rely on [torch.randn](https://pytorch.org/docs/stable/generated/torch.randn.html), which uses a different random seed each time, to create the initial noisy tensors. To generate the same output on a CPU or GPU, use a [Generator](https://docs.pytorch.org/docs/stable/generated/torch.Generator.html) to manage how random values are generated.

> [!TIP]
> If reproducibility is important, you should use a CPU `Generator`. The performance loss is often negligible and you'll generate more similar values.

<hfoptions id="generator">
<hfoption id="GPU">

Use a CPU `Generator` when you care about reproducibility. CPU RNG is more stable across machines.

When you pass a CPU `Generator`, Diffusers’ [`~utils.torch_utils.randn_tensor`] samples on the CPU and moves the tensor to the GPU inside the pipeline. You do not call `randn_tensor` or `.to("cuda")` yourself. A GPU `Generator` samples on-device instead and can diverge from CPU results.

Use [manual_seed](https://docs.pytorch.org/docs/stable/generated/torch.manual_seed.html) to set a seed.

```py
import torch
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained(
    "Qwen/Qwen-Image", dtype=torch.bfloat16, device_map="cuda"  # or "mps", "xpu", "cpu"
)
generator = torch.manual_seed(0)
image = pipeline(
    prompt="a red apple on a wooden table",
    generator=generator,
    num_inference_steps=4,
).images[0]
```

</hfoption>
<hfoption id="CPU">

Create a CPU `Generator` and set a seed with [Generator.manual_seed](https://docs.pytorch.org/docs/stable/generated/torch.Generator.html#torch.Generator.manual_seed).

```py
import torch
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained(
    "Qwen/Qwen-Image", dtype=torch.bfloat16, device_map="cpu"
)
generator = torch.Generator(device="cpu").manual_seed(0)
image = pipeline(
    prompt="a red apple on a wooden table",
    generator=generator,
    num_inference_steps=4,
).images[0]
```

</hfoption>
</hfoptions>

Pass a `Generator` object to the pipeline instead of an integer seed. A `Generator` keeps a random state that is consumed and updated when you use it. After that, the same object produces different results on later calls, even across pipelines, because its state has changed. Reseed it or create a new `Generator` before each call when you need the same seed again.

```py
import torch

prompt = "a red apple on a wooden table"

for _ in range(5):
    generator = torch.manual_seed(0)
    image = pipeline(prompt, generator=generator, num_inference_steps=4).images[0]
```

## Deterministic algorithms

PyTorch supports [deterministic algorithms](https://docs.pytorch.org/docs/stable/notes/randomness.html#avoiding-nondeterministic-algorithms) (where available) for certain operations so they produce the same results. Deterministic algorithms may be slower and decrease performance.

Use Diffusers' [`~utils.torch_utils.enable_full_determinism`] to enable deterministic algorithms.

```py
from diffusers.utils.torch_utils import enable_full_determinism

enable_full_determinism()
```

`enable_full_determinism` works by:

- Setting the environment variable `CUDA_LAUNCH_BLOCKING` to `1`
- Setting the environment variable [CUBLAS_WORKSPACE_CONFIG](https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility) to `:16:8` so cuBLAS uses a fixed workspace layout at runtime
- Calling `torch.use_deterministic_algorithms(True)`
- Setting `torch.backends.cudnn.deterministic = True`
- Setting `torch.backends.cudnn.benchmark = False` so cuDNN does not pick a different convolution algorithm each run
- Disabling TensorFloat32 (TF32) with `torch.backends.cuda.matmul.allow_tf32 = False` in favor of more precise full-precision matmul

## Next steps

You should read PyTorch's developer notes about [Reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html). You can try to limit randomness, but it is not *guaranteed* even with an identical seed.
