<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->
# Helios

[Helios](https://github.com/PKU-YuanGroup/Helios) 是首个能够在单张 NVIDIA H100 GPU 上以 19.5 FPS 运行的 14B 视频生成模型。它在支持分钟级视频生成的同时，拥有媲美强大基线模型的生成质量，并在统一架构下原生集成了文生视频（T2V）、图生视频（I2V）和视频生视频（V2V）任务。Helios 的主要特性包括：

- 无需常用的防漂移策略（例如：自强制/self-forcing、误差库/error-banks、关键帧采样或逆采样），我们的模型即可生成高质量且高度连贯的分钟级视频。
- 无需标准的加速技术（例如：KV 缓存、因果掩码、稀疏/线性注意力机制、TinyVAE、渐进式噪声调度、隐藏状态缓存或量化），作为一款 14B 规模的视频生成模型，我们在单张 H100 GPU 上的端到端推理速度便达到了 19.5 FPS。
- 引入了多项优化方案，在降低显存消耗的同时，显著提升了训练与推理的吞吐量。这些改进使得我们无需借助并行或分片（sharding）等基础设施，即可使用与图像模型相当的批大小（batch sizes）来训练 14B 的视频生成模型。

本指南将引导您完成 Helios 在不同场景下的使用。

## Load Model Checkpoints

模型权重可以存储在Hub上或本地的单独子文件夹中，在这种情况下，您应该使用 [`~DiffusionPipeline.from_pretrained`] 方法。

```python
# !pip install Helios_eva_clip insightface facexlib
import torch
from diffusers import HeliosPipeline
from huggingface_hub import snapshot_download

# For Best Quality
snapshot_download(repo_id="BestWishYsh/Helios-Base", local_dir="BestWishYsh/Helios-Base")
pipe = HeliosPipeline.from_pretrained("BestWishYsh/Helios-Base", torch_dtype=torch.bfloat16)
pipe.to("cuda")

# Intermediate Weight
snapshot_download(repo_id="BestWishYsh/Helios-Mid", local_dir="BestWishYsh/Helios-Mid")
pipe = HeliosPipeline.from_pretrained("BestWishYsh/Helios-Mid", torch_dtype=torch.bfloat16)
pipe.to("cuda")

# For Best Efficiency
snapshot_download(repo_id="BestWishYsh/Helios-Distilled", local_dir="BestWishYsh/Helios-Distilled")
pipe = HeliosPipeline.from_pretrained("BestWishYsh/Helios-Distilled", torch_dtype=torch.bfloat16)
pipe.to("cuda")
```

## Text-to-Video Showcases

<table>
</table>

## Image-to-Video Showcases

<table>
</table>

## Interactive-Video Showcases

<table>
</table>

## Resources

通过以下资源了解有关 Helios 的更多信息：

- 一段 [视频](https://www.youtube.com/watch?v=) 演示了 Helios 的主要功能;
- 有关更多详细信息，请参阅研究论文 [Helios: 14B Real-Time Long Video Generation Model can be Cheaper, Faster but Keep Stronger than 1.3B ones](https://huggingface.co/papers/)。
