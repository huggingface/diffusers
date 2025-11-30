<!-- Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License. -->

<div style="float: right;">
  <div class="flex flex-wrap space-x-1">
    <a href="https://huggingface.co/docs/diffusers/main/en/tutorials/using_peft_for_inference" target="_blank" rel="noopener">
      <img alt="LoRA" src="https://img.shields.io/badge/LoRA-d8b4fe?style=flat"/>
    </a>
  </div>
</div>

# HunyuanVideo-1.5

HunyuanVideo-1.5 is a lightweight yet powerful video generation model that achieves state-of-the-art visual quality and motion coherence with only 8.3 billion parameters, enabling efficient inference on consumer-grade GPUs. This achievement is built upon several key components, including meticulous data curation, an advanced DiT architecture with selective and sliding tile attention (SSTA), enhanced bilingual understanding through glyph-aware text encoding, progressive pre-training and post-training, and an efficient video super-resolution network. Leveraging these designs, we developed a unified framework capable of high-quality text-to-video and image-to-video generation across multiple durations and resolutions. Extensive experiments demonstrate that this compact and proficient model establishes a new state-of-the-art among open-source models.

You can find all the original HunyuanVideo checkpoints under the [Tencent](https://huggingface.co/tencent) organization.

> [!TIP]
> Click on the HunyuanVideo models in the right sidebar for more examples of video generation tasks.
>
> The examples below use a checkpoint from [hunyuanvideo-community](https://huggingface.co/hunyuanvideo-community) because the weights are stored in a layout compatible with Diffusers.

The example below demonstrates how to generate a video optimized for memory or inference speed.

<hfoptions id="usage">
<hfoption id="memory">

Refer to the [Reduce memory usage](../../optimization/memory) guide for more details about the various memory saving techniques.


```py
import torch
from diffusers import AutoModel, HunyuanVideo15Pipeline
from diffusers.utils import export_to_video


pipeline = HunyuanVideo15Pipeline.from_pretrained(
    "HunyuanVideo-1.5-Diffusers-480p_t2v",
    torch_dtype=torch.bfloat16,
)

# model-offloading and tiling
pipeline.enable_model_cpu_offload()
pipeline.vae.enable_tiling()

prompt = "A fluffy teddy bear sits on a bed of soft pillows surrounded by children's toys."
video = pipeline(prompt=prompt, num_frames=61, num_inference_steps=30).frames[0]
export_to_video(video, "output.mp4", fps=15)
```


## HunyuanVideo15Pipeline

[[autodoc]] HunyuanVideo15Pipeline
  - all
  - __call__

## HunyuanVideo15ImageToVideoPipeline

[[autodoc]] HunyuanVideo15ImageToVideoPipeline
  - all
  - __call__

## HunyuanVideo15PipelineOutput

[[autodoc]] pipelines.hunyuan_video1_5.pipeline_output.HunyuanVideo15PipelineOutput
