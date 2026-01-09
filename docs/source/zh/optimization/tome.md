<!--版权所有 2025 The HuggingFace Team。保留所有权利。

根据 Apache 许可证 2.0 版（“许可证”）授权；除非遵守许可证，否则不得使用此文件。
您可以在以下网址获取许可证副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，根据许可证分发的软件按“原样”分发，不附带任何明示或暗示的担保或条件。请参阅许可证以了解具体的语言管理权限和限制。
-->

# 令牌合并

[令牌合并](https://huggingface.co/papers/2303.17604)（ToMe）在基于 Transformer 的网络的前向传递中逐步合并冗余令牌/补丁，这可以加速 [`StableDiffusionPipeline`] 的推理延迟。

从 `pip` 安装 ToMe：

```bash
pip install tomesd
```

您可以使用 [`tomesd`](https://github.com/dbolya/tomesd) 库中的 [`apply_patch`](https://github.com/dbolya/tomesd?tab=readme-ov-file#usage) 函数：

```diff
  from diffusers import StableDiffusionPipeline
  import torch
  import tomesd

  pipeline = StableDiffusionPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16, use_safetensors=True,
  ).to("cuda")
+ tomesd.apply_patch(pipeline, ratio=0.5)

  image = pipeline("a photo of an astronaut riding a horse on mars").images[0]
```

`apply_patch` 函数公开了多个[参数](https://github.com/dbolya/tomesd#usage)，以帮助在管道推理速度和生成令牌的质量之间取得平衡。最重要的参数是 `ratio`，它控制在前向传递期间合并的令牌数量。

如[论文](https://huggingface.co/papers/2303.17604)中所述，ToMe 可以在显著提升推理速度的同时，很大程度上保留生成图像的质量。通过增加 `ratio`，您可以进一步加速推理，但代价是图像质量有所下降。

为了测试生成图像的质量，我们从 [Parti Prompts](https://parti.research.google/) 中采样了一些提示，并使用 [`StableDiffusionPipeline`] 进行了推理，设置如下：

<div class="flex justify-center">
      <img src="https://huggingface.co/datasets/diffusers/docs-images/resolve/main/tome/tome_samples.png">
</div>

我们没有注意到生成样本的质量有任何显著下降，您可以在此 [WandB 报告](https://wandb.ai/sayakpaul/tomesd-results/runs/23j4bj3i?workspace=)中查看生成的样本。如果您有兴趣重现此实验，请使用此[脚本](https://gist.github.com/sayakpaul/8cac98d7f22399085a060992f411ecbd)。

## 基准测试

我们还在启用 [xFormers](https://huggingface.co/docs/diffusers/optimization/xformers) 的情况下，对 [`StableDiffusionPipeline`] 上 `tomesd` 的影响进行了基准测试，涵盖了多个图像分辨率。结果
结果是从以下开发环境中的A100和V100 GPU获得的：

```bash
- `diffusers` 版本：0.15.1
- Python 版本：3.8.16
- PyTorch 版本（GPU？）：1.13.1+cu116 (True)
- Huggingface_hub 版本：0.13.2
- Transformers 版本：4.27.2
- Accelerate 版本：0.18.0
- xFormers 版本：0.0.16
- tomesd 版本：0.1.2
```

要重现此基准测试，请随意使用此[脚本](https://gist.github.com/sayakpaul/27aec6bca7eb7b0e0aa4112205850335)。结果以秒为单位报告，并且在适用的情况下，我们报告了使用ToMe和ToMe + xFormers时相对于原始管道的加速百分比。

| **GPU**  | **分辨率** | **批处理大小** | **原始** | **ToMe**       | **ToMe + xFormers** |
|----------|----------------|----------------|-------------|----------------|---------------------|
| **A100** |            512 |             10 |        6.88 | 5.26 (+23.55%) |      4.69 (+31.83%) |
|          |            768 |             10 |         OOM |          14.71 |                  11 |
|          |                |              8 |         OOM |          11.56 |                8.84 |
|          |                |              4 |         OOM |           5.98 |                4.66 |
|          |                |              2 |        4.99 | 3.24 (+35.07%) |       2.1 (+37.88%) |
|          |                |              1 |        3.29 | 2.24 (+31.91%) |       2.03 (+38.3%) |
|          |           1024 |             10 |         OOM |            OOM |                 OOM |
|          |                |              8 |         OOM |            OOM |                 OOM |
|          |                |              4 |         OOM |          12.51 |                9.09 |
|          |                |              2 |         OOM |           6.52 |                4.96 |
|          |                |              1 |         6.4 | 3.61 (+43.59%) |      2.81 (+56.09%) |
| **V100** |            512 |             10 |         OOM |          10.03 |                9.29 |
|          |                |              8 |         OOM |           8.05 |                7.47 |
|          |                |              4 |         5.7 |  4.3 (+24.56%) |      3.98 (+30.18%) |
|          |                |              2 |        3.14 | 2.43 (+22.61%) |      2.27 (+27.71%) |
|          |                |              1 |        1.88 | 1.57 (+16.49%) |      1.57 (+16.49%) |
|          |            768 |             10 |         OOM |            OOM |               23.67 |
|          |                |              8 |         OOM |            OOM |               18.81 |
|          |                |              4 |         OOM |          11.81 |                 9.7 |
|          |                |              2 |         OOM |           6.27 |                 5.2 |
|          |                |              1 |        5.43 | 3.38 (+37.75%) |      2.82 (+48.07%) |
|          |           1024 |             10 |         OOM |            
如上表所示，`tomesd` 带来的加速效果在更大的图像分辨率下变得更加明显。有趣的是，使用 `tomesd` 可以在更高分辨率如 1024x1024 上运行管道。您可能还可以通过 [`torch.compile`](fp16#torchcompile) 进一步加速推理。