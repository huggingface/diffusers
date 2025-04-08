<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Metal Performance Shaders (MPS)

> [!TIP]
> Pipelines with a <img alt="MPS" src="https://img.shields.io/badge/MPS-000000?style=flat&logo=apple&logoColor=white%22"> badge indicate a model can take advantage of the MPS backend on Apple silicon devices for faster inference. Feel free to open a [Pull Request](https://github.com/huggingface/diffusers/compare) to add this badge to pipelines that are missing it.

🤗 Diffusers is compatible with Apple silicon (M1/M2 chips) using the PyTorch [`mps`](https://pytorch.org/docs/stable/notes/mps.html) device, which uses the Metal framework to leverage the GPU on MacOS devices. You'll need to have:

- macOS computer with Apple silicon (M1/M2) hardware
- macOS 12.6 or later (13.0 or later recommended)
- arm64 version of Python
- [PyTorch 2.0](https://pytorch.org/get-started/locally/) (recommended) or 1.13 (minimum version supported for `mps`)

The `mps` backend uses PyTorch's `.to()` interface to move the Stable Diffusion pipeline on to your M1 or M2 device:

```python
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5")
pipe = pipe.to("mps")

# Recommended if your computer has < 64 GB of RAM
pipe.enable_attention_slicing()

prompt = "a photo of an astronaut riding a horse on mars"
image = pipe(prompt).images[0]
image
```

<Tip warning={true}>

The PyTorch [mps](https://pytorch.org/docs/stable/notes/mps.html) backend does not support NDArray sizes greater than `2**32`. Please open an [Issue](https://github.com/huggingface/diffusers/issues/new/choose) if you encounter this problem so we can investigate.

</Tip>

If you're using **PyTorch 1.13**, you need to "prime" the pipeline with an additional one-time pass through it. This is a temporary workaround for an issue where the first inference pass produces slightly different results than subsequent ones. You only need to do this pass once, and after just one inference step you can discard the result.

```diff
  from diffusers import DiffusionPipeline

  pipe = DiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5").to("mps")
  pipe.enable_attention_slicing()

  prompt = "a photo of an astronaut riding a horse on mars"
  # First-time "warmup" pass if PyTorch version is 1.13
+ _ = pipe(prompt, num_inference_steps=1)

  # Results match those from the CPU device after the warmup pass.
  image = pipe(prompt).images[0]
```

## Troubleshoot

This section lists some common issues with using the `mps` backend and how to solve them.

### Attention slicing

M1/M2 performance is very sensitive to memory pressure. When this occurs, the system automatically swaps if it needs to which significantly degrades performance.

To prevent this from happening, we recommend *attention slicing* to reduce memory pressure during inference and prevent swapping. This is especially relevant if your computer has less than 64GB of system RAM, or if you generate images at non-standard resolutions larger than 512×512 pixels. Call the [`~DiffusionPipeline.enable_attention_slicing`] function on your pipeline:

```py
from diffusers import DiffusionPipeline
import torch

pipeline = DiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16, variant="fp16", use_safetensors=True).to("mps")
pipeline.enable_attention_slicing()
```

Attention slicing performs the costly attention operation in multiple steps instead of all at once. It usually improves performance by ~20% in computers without universal memory, but we've observed *better performance* in most Apple silicon computers unless you have 64GB of RAM or more.

### Batch inference

Generating multiple prompts in a batch can crash or fail to work reliably. If this is the case, try iterating instead of batching.