<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# DeepCache
[DeepCache](https://huggingface.co/papers/2312.00858) accelerates [`StableDiffusionPipeline`] by strategically caching and reusing high-level features, while efficiently updating low-level features, which leverages the unique properties of the U-Net architecture. 

Install DeepCache from `pip`:
```bash
pip install DeepCache
```

You can use [`DeepCache`](https://github.com/horseee/DeepCache) by loading and enabling the [`DeepCacheSDHelper`](https://github.com/horseee/DeepCache#usage):

```diff
  import torch
  from diffusers import StableDiffusionPipeline
  pipe = StableDiffusionPipeline.from_pretrained('runwayml/stable-diffusion-v1-5', torch_dtype=torch.float16).to("cuda")

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
Opting for a lower `cache_branch_id` or a larger `cache_interval` can lead to faster inference speed; however, this may come at the cost of reduced image quality. Once those arguments are set, use the `enable` or `disable` methods to activate or deactivate the DeepCacheSDHelper respectively.

<div class="flex justify-center">
    <img src="https://github.com/horseee/Diffusion_DeepCache/raw/master/static/images/example.png">
</div>

You can find more generated samples (Original Pipeline v.s. DeepCache) and the corresponding inference latency in the [WandB report](https://wandb.ai/horseee/DeepCache/runs/jwlsqqgt?workspace=user-horseee). The prompts are randomly selected from the [MS-COCO 2017](https://cocodataset.org/#home) dataset.

## Benchmark

We measure the acceleration ratio achievable using DeepCache. All evaluations are based on the [Stable Diffusion v2.1](https://huggingface.co/stabilityai/stable-diffusion-2-1) with 50 inference steps, using NVIDIA RTX A5000. The results show the speed enhancements that can be expected under different configurations of resolution, batch size, the interval for cache(I) and the branch for cache(B).

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
