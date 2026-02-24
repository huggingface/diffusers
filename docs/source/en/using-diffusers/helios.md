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

- Without commonly used anti-drifting strategies (\eg, self-forcing, error-banks, keyframe sampling, or inverted sampling), \ours generates minute-scale videos with high quality and strong coherence.
- Without standard acceleration techniques (\eg, KV-cache, causal masking, sparse/linear attention, TinyVAE, progressive noise schedules, hidden-state caching, or quantization), \ours achieves 19.5 FPS in end-to-end inference for a 14B video generation model on a single H100 GPU.
- Introducing optimizations that improve both training and inference throughput while reducing memory consumption. These changes enable training a 14B video generation model without parallelism or sharding infrastructure, with batch sizes comparable to image models.

This guide will walk you through using Helios for use cases.

## Load Model Checkpoints

Model weights may be stored in separate subfolders on the Hub or locally, in which case, you should use the [`~DiffusionPipeline.from_pretrained`] method.

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

Learn more about Helios with the following resources.
- A [video](https://www.youtube.com/watch?v=) demonstrating Helios's main features.
- The research paper, [Helios: 14B Real-Time Long Video Generation Model can be Cheaper, Faster but Keep Stronger than 1.3B ones](https://huggingface.co/papers/) for more details.
