<!--版权所有 2025 The HuggingFace Team。保留所有权利。

根据 Apache 许可证 2.0 版本（"许可证"）授权；除非遵守许可证，否则不得使用此文件。您可以在以下网址获取许可证副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，否则根据许可证分发的软件按"原样"分发，无任何明示或暗示的担保或条件。有关许可证的具体语言，请参阅许可证中的权限和限制。
-->

# DeepCache
[DeepCache](https://huggingface.co/papers/2312.00858) 通过策略性地缓存和重用高级特征，同时利用 U-Net 架构高效更新低级特征，来加速 [`StableDiffusionPipeline`] 和 [`StableDiffusionXLPipeline`]。

首先安装 [DeepCache](https://github.com/horseee/DeepCache)：
```bash
pip install DeepCache
```

然后加载并启用 [`DeepCacheSDHelper`](https://github.com/horseee/DeepCache#usage)：

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

`set_params` 方法接受两个参数：`cache_interval` 和 `cache_branch_id`。`cache_interval` 表示特征缓存的频率，指定为每次缓存操作之间的步数。`cache_branch_id` 标识网络的哪个分支（从最浅层到最深层排序）负责执行缓存过程。
选择较低的 `cache_branch_id` 或较大的 `cache_interval` 可以加快推理速度，但会降低图像质量（这些超参数的消融实验可以在[论文](https://huggingface.co/papers/2312.00858)中找到）。一旦设置了这些参数，使用 `enable` 或 `disable` 方法来激活或停用 `DeepCacheSDHelper`。

<div class="flex justify-center">
    <img src="https://github.com/horseee/Diffusion_DeepCache/raw/master/static/images/example.png">
</div>

您可以在 [WandB 报告](https://wandb.ai/horseee/DeepCache/runs/jwlsqqgt?workspace=user-horseee) 中找到更多生成的样本（原始管道 vs DeepCache）和相应的推理延迟。提示是从 [MS-COCO 2017](https://cocodataset.org/#home) 数据集中随机选择的。

## 基准测试

我们在 NVIDIA RTX A5000 上测试了 DeepCache 使用 50 个推理步骤加速 [Stable Diffusion v2.1](https://huggingface.co/stabilityai/stable-diffusion-2-1) 的速度，使用不同的配置，包括分辨率、批处理大小、缓存间隔（I）和缓存分支(B)。

| **分辨率** | **批次大小** | **原始** | **DeepCache(I=3, B=0)** | **DeepCache(I=5, B=0)** | **DeepCache(I=5, B=1)** |
|----------------|----------------|--------------|-------------------------|-------------------------|-------------------------|
|             512|               8|         15.96|              6.88(2.32倍)|              5.03(3.18倍)|              7.27(2.20x)|
|                |               4|          8.39|              3.60(2.33倍)|              2.62(3.21倍)|              3.75(2.24x)|
|                |               1|          2.61|              1.12(2.33倍)|              0.81(3.24倍)|              1.11(2.35x)|
|             768|               8|         43.58|             18.99(2.29倍)|             13.96(3.12倍)|             21.27(2.05x)|
|                |               4|         22.24|              9.67(2.30倍)|              7.10(3.13倍)|             10.74(2.07x)|
|                |               1|          6.33|              2.72(2.33倍)|              1.97(3.21倍)|              2.98(2.12x)|
|            1024|               8|        101.95|             45.57(2.24倍)|             33.72(3.02倍)|             53.00(1.92x)|
|                |               4|         49.25|             21.86(2.25倍)|             16.19(3.04倍)|             25.78(1.91x)|
|                |               1|         13.83|              6.07(2.28倍)|              4.43(3.12倍)|              7.15(1.93x)|