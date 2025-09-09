<!-- 版权所有 2025 HuggingFace 团队。保留所有权利。

根据 Apache 许可证 2.0 版本（“许可证”）授权；除非符合许可证，否则不得使用此文件。您可以在以下网址获取许可证副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，否则根据许可证分发的软件按“原样”分发，不附带任何明示或暗示的担保或条件。请参阅许可证以了解具体的语言管理权限和限制。 -->

# 缓存

缓存通过存储和重用不同层的中间输出（如注意力层和前馈层）来加速推理，而不是在每个推理步骤执行整个计算。它显著提高了生成速度，但以更多内存为代价，并且不需要额外的训练。

本指南向您展示如何在 Diffusers 中使用支持的缓存方法。

## 金字塔注意力广播

[金字塔注意力广播 (PAB)](https://huggingface.co/papers/2408.12588) 基于这样一种观察：在生成过程的连续时间步之间，注意力输出差异不大。注意力差异在交叉注意力层中最小，并且通常在一个较长的时间步范围内被缓存。其次是时间注意力和空间注意力层。

> [!TIP]
> 并非所有视频模型都有三种类型的注意力（交叉、时间和空间）！

PAB 可以与其他技术（如序列并行性和无分类器引导并行性（数据并行性））结合，实现近乎实时的视频生成。

设置并传递一个 [`PyramidAttentionBroadcastConfig`] 到管道的变换器以启用它。`spatial_attention_block_skip_range` 控制跳过空间注意力块中注意力计算的频率，`spatial_attention_timestep_skip_range` 是要跳过的时间步范围。注意选择一个合适的范围，因为较小的间隔可能导致推理速度变慢，而较大的间隔可能导致生成质量降低。

```python
import torch
from diffusers import CogVideoXPipeline, PyramidAttentionBroadcastConfig

pipeline = CogVideoXPipeline.from_pretrained("THUDM/CogVideoX-5b", torch_dtype=torch.bfloat16)
pipeline.to("cuda")

config = PyramidAttentionBroadcastConfig(
    spatial_attention_block_skip_range=2,
    spatial_attention_timestep_skip_range=(100, 800),
    current_timestep_callback=lambda: pipe.current_timestep,
)
pipeline.transformer.enable_cache(config)
```

## FasterCache

[FasterCache](https://huggingface.co/papers/2410.19355) 缓存并重用注意力特征，类似于 [PAB](#pyramid-attention-broadcast)，因为每个连续时间步的输出差异很小。

此方法在使用无分类器引导进行采样时（在大多数基础模型中常见），也可能选择跳过无条件分支预测，并且
如果连续时间步之间的预测潜在输出存在显著冗余，则从条件分支预测中估计它。

设置并将 [`FasterCacheConfig`] 传递给管道的 transformer 以启用它。

```python
import torch
from diffusers import CogVideoXPipeline, FasterCacheConfig

pipe line= CogVideoXPipeline.from_pretrained("THUDM/CogVideoX-5b", torch_dtype=torch.bfloat16)
pipeline.to("cuda")

config = FasterCacheConfig(
    spatial_attention_block_skip_range=2,
    spatial_attention_timestep_skip_range=(-1, 681),
    current_timestep_callback=lambda: pipe.current_timestep,
    attention_weight_callback=lambda _: 0.3,
    unconditional_batch_skip_range=5,
    unconditional_batch_timestep_skip_range=(-1, 781),
    tensor_format="BFCHW",
)
pipeline.transformer.enable_cache(config)
```