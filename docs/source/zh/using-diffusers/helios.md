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
import torch
from diffusers import HeliosPipeline, HeliosPyramidPipeline
from huggingface_hub import snapshot_download

# For Best Quality
snapshot_download(repo_id="BestWishYsh/Helios-Base", local_dir="BestWishYsh/Helios-Base")
pipe = HeliosPipeline.from_pretrained("BestWishYsh/Helios-Base", torch_dtype=torch.bfloat16)
pipe.to("cuda")

# Intermediate Weight
snapshot_download(repo_id="BestWishYsh/Helios-Mid", local_dir="BestWishYsh/Helios-Mid")
pipe = HeliosPyramidPipeline.from_pretrained("BestWishYsh/Helios-Mid", torch_dtype=torch.bfloat16)
pipe.to("cuda")

# For Best Efficiency
snapshot_download(repo_id="BestWishYsh/Helios-Distilled", local_dir="BestWishYsh/Helios-Distilled")
pipe = HeliosPyramidPipeline.from_pretrained("BestWishYsh/Helios-Distilled", torch_dtype=torch.bfloat16)
pipe.to("cuda")
```

## Text-to-Video Showcases

<table>
  <tr>
    <th style="text-align: center;">Prompt</th>
    <th style="text-align: center;">Generated Video</th>
  </tr>
  <tr>
    <td><small>A Viking warrior driving a modern city bus filled with passengers. The Viking has long blonde hair tied back, a beard, and is adorned with a fur-lined helmet and armor. He wears a traditional tunic and trousers, but also sports a seatbelt as he focuses on navigating the busy streets. The interior of the bus is typical, with rows of seats occupied by diverse passengers going about their daily routines. The exterior shots show the bustling urban environment, including tall buildings and traffic. Medium shot focusing on the Viking at the wheel, with occasional close-ups of his determined expression.
    </small></td>
    <td>
      <video width="4000" controls>
        <source src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/helios/t2v_showcases1.mp4" type="video/mp4">
      </video>
    </td>
  </tr>
  <tr>
    <td><small>A documentary-style nature photography shot from a camera truck moving to the left, capturing a crab quickly scurrying into its burrow. The crab has a hard, greenish-brown shell and long claws, moving with determined speed across the sandy ground. Its body is slightly arched as it burrows into the sand, leaving a small trail behind. The background shows a shallow beach with scattered rocks and seashells, and the horizon features a gentle curve of the coastline. The photo has a natural and realistic texture, emphasizing the crab's natural movement and the texture of the sand. A close-up shot from a slightly elevated angle.
    </small></td>
    <td>
      <video width="4000" controls>
        <source src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/helios/t2v_showcases2.mp4" type="video/mp4">
      </video>
    </td>
  </tr>
</table>

## Image-to-Video Showcases

<table>
  <tr>
    <th style="text-align: center;">Image</th>
    <th style="text-align: center;">Prompt</th>
    <th style="text-align: center;">Generated Video</th>
  </tr>
  <tr>
    <td><img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/helios/i2v_showcases1.jpg" style="height: auto; width: 300px;"></td>
    <td><small>A sleek red Kia car speeds along a rural road under a cloudy sky, its modern design and dynamic movement emphasized by the blurred motion of the surrounding fields and trees stretching into the distance. The car's glossy exterior reflects the overcast sky, highlighting its aerodynamic shape and sporty stance. The license plate reads "KIA 626," and the vehicle's headlights are on, adding to the sense of motion and energy. The road curves gently, with the car positioned slightly off-center, creating a sense of forward momentum. A dynamic front three-quarter view captures the car's powerful presence against the serene backdrop of rolling hills and scattered trees.
    </small></td>
    <td>
      <video width="2000" controls>
        <source src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/helios/i2v_showcases1.mp4" type="video/mp4">
      </video>
    </td>
  </tr>
  <tr>
    <td><img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/helios/i2v_showcases2.jpg" style="height: auto; width: 300px;"></td>
    <td><small>A close-up captures a fluffy orange cat with striking green eyes and white whiskers, gazing intently towards the camera. The cat's fur is soft and well-groomed, with a mix of warm orange and cream tones. Its large, expressive eyes are a vivid green, reflecting curiosity and alertness. The cat's nose is small and pink, and its mouth is slightly open, revealing a hint of its pink tongue. The background is softly blurred, suggesting a cozy indoor setting with neutral tones. The photo has a shallow depth of field, focusing sharply on the cat's face while the background remains out of focus. A close-up shot from a slightly elevated perspective.
    </small></td>
    <td>
      <video width="2000" controls>
        <source src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/helios/i2v_showcases2.mp4" type="video/mp4">
      </video>
    </td>
  </tr>
</table>

## Interactive-Video Showcases

<table>
  <tr>
    <th style="text-align: center;">Prompt</th>
    <th style="text-align: center;">Generated Video</th>
  </tr>
  <tr>
    <td><small>The prompt can be found <a href="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/helios/interactive_showcases1.txt">here</a></small></td>
    <td>
      <video width="680" controls>
        <source src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/helios/interactive_showcases1.mp4" type="video/mp4">
      </video>
    </td>
  </tr>
  <tr>
    <td><small>The prompt can be found <a href="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/helios/interactive_showcases2.txt">here</a></small></td>
    <td>
      <video width="680" controls>
        <source src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/helios/interactive_showcases2.mp4" type="video/mp4">
      </video>
    </td>
  </tr>
</table>

## Resources

通过以下资源了解有关 Helios 的更多信息：

- [视频1](https://www.youtube.com/watch?v=vd_AgHtOUFQ)和[视频2](https://www.youtube.com/watch?v=1GeIU2Dn7UY)演示了 Helios 的主要功能;
- 有关更多详细信息，请参阅研究论文 [Helios: Real Real-Time Long Video Generation Model](https://huggingface.co/papers/)。
