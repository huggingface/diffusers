<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->
# Helios

[Helios](https://github.com/PKU-YuanGroup/Helios) is the first 14B video generation model that runs at 19.5 FPS on a single NVIDIA H100 GPU and supports minute-scale generation while matching the quality of a strong baseline, natively integrating T2V, I2V, and V2V tasks within a unified architecture. The main features of Helios are:

- Without commonly used anti-drifting strategies (eg, self-forcing, error-banks, keyframe sampling, or inverted sampling), Helios generates minute-scale videos with high quality and strong coherence.
- Without standard acceleration techniques (eg, KV-cache, causal masking, sparse/linear attention, TinyVAE, progressive noise schedules, hidden-state caching, or quantization), Helios achieves 19.5 FPS in end-to-end inference for a 14B video generation model on a single H100 GPU.
- Introducing optimizations that improve both training and inference throughput while reducing memory consumption. These changes enable training a 14B video generation model without parallelism or sharding infrastructure, with batch sizes comparable to image models.

This guide will walk you through using Helios for use cases.

## Load Model Checkpoints

Model weights may be stored in separate subfolders on the Hub or locally, in which case, you should use the [`~DiffusionPipeline.from_pretrained`] method.

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

Learn more about Helios with the following resources.
- Watch [video1](https://www.youtube.com/watch?v=vd_AgHtOUFQ) and [video2](https://www.youtube.com/watch?v=1GeIU2Dn7UY) for a demonstration of Helios's key features.
- The research paper, [Helios: Real Real-Time Long Video Generation Model](https://huggingface.co/papers/2603.04379) for more details.
