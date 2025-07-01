<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# DeepCache
[DeepCache](https://huggingface.co/papers/2312.00858) accelerates [`StableDiffusionPipeline`] and [`StableDiffusionXLPipeline`] by strategically caching and reusing high-level features while efficiently updating low-level features by taking advantage of the U-Net architecture.

Start by installing [DeepCache](https://github.com/horseee/DeepCache):
```bash
pip install DeepCache
```

Then load and enable the [`DeepCacheSDHelper`](https://github.com/horseee/DeepCache#usage):

```diff
  import torch
  from diffusers import StableDiffusionPipeline
  pipe = StableDiffusionPipeline.from_pretrained('stable-diffusion-v1-5/stable-diffusion-v1-5', torch_dtype=torch.float16).to("cuda")

+ from DeepCache import DeepCacheSDHelper
+ helper = DeepCacheSDHelper(pipe=pipe)
+ helper.set_params(
+     cache_interval=3,
+     cache_branch_id=0,
+ )
+ helper.enable()

  image = pipe("a photo of an astronaut on a moon").images[0]
```

The `set_params` method accepts two arguments: `cache_interval` and `cache_branch_id`. `cache_interval` means the frequency of feature caching, specified as the number of steps between each cache operation. `cache_branch_id` identifies which branch of the network (ordered from the shallowest to the deepest layer) is responsible for executing the caching processes.
Opting for a lower `cache_branch_id` or a larger `cache_interval` can lead to faster inference speed at the expense of reduced image quality (ablation experiments of these two hyperparameters can be found in the [paper](https://huggingface.co/papers/2312.00858)). Once those arguments are set, use the `enable` or `disable` methods to activate or deactivate the `DeepCacheSDHelper`.

<div class="flex justify-center">
    <img src="https://github.com/horseee/Diffusion_DeepCache/raw/master/static/images/example.png">
</div>

You can find more generated samples (original pipeline vs DeepCache) and the corresponding inference latency in the [WandB report](https://wandb.ai/horseee/DeepCache/runs/jwlsqqgt?workspace=user-horseee). The prompts are randomly selected from the [MS-COCO 2017](https://cocodataset.org/#home) dataset.

## Benchmark

We tested how much faster DeepCache accelerates [Stable Diffusion v2.1](https://huggingface.co/stabilityai/stable-diffusion-2-1) with 50 inference steps on an NVIDIA RTX A5000, using different configurations for resolution, batch size, cache interval (I), and cache branch (B).

| **Resolution** | **Batch size** | **Original** | **DeepCache(I=3, B=0)** | **DeepCache(I=5, B=0)** | **DeepCache(I=5, B=1)** |
|----------------|----------------|--------------|-------------------------|-------------------------|-------------------------|
|             512|               8|         15.96|              6.88(2.32x)|              5.03(3.18x)|              7.27(2.20x)|
|                |               4|          8.39|              3.60(2.33x)|              2.62(3.21x)|              3.75(2.24x)|
|                |               1|          2.61|              1.12(2.33x)|              0.81(3.24x)|              1.11(2.35x)|
|             768|               8|         43.58|             18.99(2.29x)|             13.96(3.12x)|             21.27(2.05x)|
|                |               4|         22.24|              9.67(2.30x)|              7.10(3.13x)|             10.74(2.07x)|
|                |               1|          6.33|              2.72(2.33x)|              1.97(3.21x)|              2.98(2.12x)|
|            1024|               8|        101.95|             45.57(2.24x)|             33.72(3.02x)|             53.00(1.92x)|
|                |               4|         49.25|             21.86(2.25x)|             16.19(3.04x)|             25.78(1.91x)|
|                |               1|         13.83|              6.07(2.28x)|              4.43(3.12x)|              7.15(1.93x)|
