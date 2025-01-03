<!--Copyright 2024 The HuggingFace Team. All rights reserved.
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
# limitations under the License.
-->

# CogVideoX

[CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer](https://arxiv.org/abs/2408.06072) from Tsinghua University & ZhipuAI, by Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, Da Yin, Xiaotao Gu, Yuxuan Zhang, Weihan Wang, Yean Cheng, Ting Liu, Bin Xu, Yuxiao Dong, Jie Tang.

The abstract from the paper is:

*We introduce CogVideoX, a large-scale diffusion transformer model designed for generating videos based on text prompts. To efficently model video data, we propose to levearge a 3D Variational Autoencoder (VAE) to compresses videos along both spatial and temporal dimensions. To improve the text-video alignment, we propose an expert transformer with the expert adaptive LayerNorm to facilitate the deep fusion between the two modalities. By employing a progressive training technique, CogVideoX is adept at producing coherent, long-duration videos characterized by significant motion. In addition, we develop an effectively text-video data processing pipeline that includes various data preprocessing strategies and a video captioning method. It significantly helps enhance the performance of CogVideoX, improving both generation quality and semantic alignment. Results show that CogVideoX demonstrates state-of-the-art performance across both multiple machine metrics and human evaluations. The model weight of CogVideoX-2B is publicly available at https://github.com/THUDM/CogVideo.*

<Tip>

Make sure to check out the Schedulers [guide](../../using-diffusers/schedulers) to learn how to explore the tradeoff between scheduler speed and quality, and see the [reuse components across pipelines](../../using-diffusers/loading#reuse-a-pipeline) section to learn how to efficiently load the same components into multiple pipelines.

</Tip>

This pipeline was contributed by [zRzRzRzRzRzRzR](https://github.com/zRzRzRzRzRzRzR). The original codebase can be found [here](https://huggingface.co/THUDM). The original weights can be found under [hf.co/THUDM](https://huggingface.co/THUDM).

There are three official CogVideoX checkpoints for text-to-video and video-to-video.

| checkpoints | recommended inference dtype |
|:---:|:---:|
| [`THUDM/CogVideoX-2b`](https://huggingface.co/THUDM/CogVideoX-2b) | torch.float16 |
| [`THUDM/CogVideoX-5b`](https://huggingface.co/THUDM/CogVideoX-5b) | torch.bfloat16 |
| [`THUDM/CogVideoX1.5-5b`](https://huggingface.co/THUDM/CogVideoX1.5-5b) | torch.bfloat16 |

There are two official CogVideoX checkpoints available for image-to-video.

| checkpoints | recommended inference dtype |
|:---:|:---:|
| [`THUDM/CogVideoX-5b-I2V`](https://huggingface.co/THUDM/CogVideoX-5b-I2V) | torch.bfloat16 |
| [`THUDM/CogVideoX-1.5-5b-I2V`](https://huggingface.co/THUDM/CogVideoX-1.5-5b-I2V) | torch.bfloat16 |

For the CogVideoX 1.5 series:
- Text-to-video (T2V) works best at a resolution of 1360x768 because it was trained with that specific resolution.
- Image-to-video (I2V) works for multiple resolutions. The width can vary from 768 to 1360, but the height must be 768. The height/width must be divisible by 16.
- Both T2V and I2V models support generation with 81 and 161 frames and work best at this value. Exporting videos at 16 FPS is recommended.

There are two official CogVideoX checkpoints that support pose controllable generation (by the [Alibaba-PAI](https://huggingface.co/alibaba-pai) team).

| checkpoints | recommended inference dtype |
|:---:|:---:|
| [`alibaba-pai/CogVideoX-Fun-V1.1-2b-Pose`](https://huggingface.co/alibaba-pai/CogVideoX-Fun-V1.1-2b-Pose) | torch.bfloat16 |
| [`alibaba-pai/CogVideoX-Fun-V1.1-5b-Pose`](https://huggingface.co/alibaba-pai/CogVideoX-Fun-V1.1-5b-Pose) | torch.bfloat16 |

## Inference

Use [`torch.compile`](https://huggingface.co/docs/diffusers/main/en/tutorials/fast_diffusion#torchcompile) to reduce the inference latency.

First, load the pipeline:

```python
import torch
from diffusers import CogVideoXPipeline, CogVideoXImageToVideoPipeline
from diffusers.utils import export_to_video,load_image
pipe = CogVideoXPipeline.from_pretrained("THUDM/CogVideoX-5b").to("cuda") # or "THUDM/CogVideoX-2b" 
```

If you are using the image-to-video pipeline, load it as follows:

```python
pipe = CogVideoXImageToVideoPipeline.from_pretrained("THUDM/CogVideoX-5b-I2V").to("cuda")
```

Then change the memory layout of the pipelines `transformer` component to `torch.channels_last`:

```python
pipe.transformer.to(memory_format=torch.channels_last)
```

Compile the components and run inference:

```python
pipe.transformer = torch.compile(pipeline.transformer, mode="max-autotune", fullgraph=True)

# CogVideoX works well with long and well-described prompts
prompt = "A panda, dressed in a small, red jacket and a tiny hat, sits on a wooden stool in a serene bamboo forest. The panda's fluffy paws strum a miniature acoustic guitar, producing soft, melodic tunes. Nearby, a few other pandas gather, watching curiously and some clapping in rhythm. Sunlight filters through the tall bamboo, casting a gentle glow on the scene. The panda's face is expressive, showing concentration and joy as it plays. The background includes a small, flowing stream and vibrant green foliage, enhancing the peaceful and magical atmosphere of this unique musical performance."
video = pipe(prompt=prompt, guidance_scale=6, num_inference_steps=50).frames[0]
```

The [T2V benchmark](https://gist.github.com/a-r-r-o-w/5183d75e452a368fd17448fcc810bd3f) results on an 80GB A100 machine are:

```
Without torch.compile(): Average inference time: 96.89 seconds.
With torch.compile(): Average inference time: 76.27 seconds.
```

### Memory optimization

CogVideoX-2b requires about 19 GB of GPU memory to decode 49 frames (6 seconds of video at 8 FPS) with output resolution 720x480 (W x H), which makes it not possible to run on consumer GPUs or free-tier T4 Colab. The following memory optimizations could be used to reduce the memory footprint. For replication, you can refer to [this](https://gist.github.com/a-r-r-o-w/3959a03f15be5c9bd1fe545b09dfcc93) script.

- `pipe.enable_model_cpu_offload()`:
  - Without enabling cpu offloading, memory usage is `33 GB`
  - With enabling cpu offloading, memory usage is `19 GB`
- `pipe.enable_sequential_cpu_offload()`:
  - Similar to `enable_model_cpu_offload` but can significantly reduce memory usage at the cost of slow inference
  - When enabled, memory usage is under `4 GB`
- `pipe.vae.enable_tiling()`:
  - With enabling cpu offloading and tiling, memory usage is `11 GB`
- `pipe.vae.enable_slicing()`

## Quantization

Quantization helps reduce the memory requirements of very large models by storing model weights in a lower precision data type. However, quantization may have varying impact on video quality depending on the video model.

Refer to the [Quantization](../../quantization/overview) overview to learn more about supported quantization backends and selecting a quantization backend that supports your use case. The example below demonstrates how to load a quantized [`CogVideoXPipeline`] for inference with bitsandbytes.

```py
import torch
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig, CogVideoXTransformer3DModel, CogVideoXPipeline
from diffusers.utils import export_to_video
from transformers import BitsAndBytesConfig as BitsAndBytesConfig, T5EncoderModel

quant_config = BitsAndBytesConfig(load_in_8bit=True)
text_encoder_8bit = T5EncoderModel.from_pretrained(
    "THUDM/CogVideoX-2b",
    subfolder="text_encoder",
    quantization_config=quant_config,
    torch_dtype=torch.float16,
)

quant_config = DiffusersBitsAndBytesConfig(load_in_8bit=True)
transformer_8bit = CogVideoXTransformer3DModel.from_pretrained(
    "THUDM/CogVideoX-2b",
    subfolder="transformer",
    quantization_config=quant_config,
    torch_dtype=torch.float16,
)

pipeline = CogVideoXPipeline.from_pretrained(
    "THUDM/CogVideoX-2b",
    text_encoder=text_encoder_8bit,
    transformer=transformer_8bit,
    torch_dtype=torch.float16,
    device_map="balanced",
)

prompt = "A detailed wooden toy ship with intricately carved masts and sails is seen gliding smoothly over a plush, blue carpet that mimics the waves of the sea. The ship's hull is painted a rich brown, with tiny windows. The carpet, soft and textured, provides a perfect backdrop, resembling an oceanic expanse. Surrounding the ship are various other toys and children's items, hinting at a playful environment. The scene captures the innocence and imagination of childhood, with the toy ship's journey symbolizing endless adventures in a whimsical, indoor setting."
video = pipeline(prompt=prompt, guidance_scale=6, num_inference_steps=50).frames[0]
export_to_video(video, "ship.mp4", fps=8)
```

## CogVideoXPipeline

[[autodoc]] CogVideoXPipeline
  - all
  - __call__

## CogVideoXImageToVideoPipeline

[[autodoc]] CogVideoXImageToVideoPipeline
  - all
  - __call__

## CogVideoXVideoToVideoPipeline

[[autodoc]] CogVideoXVideoToVideoPipeline
  - all
  - __call__

## CogVideoXFunControlPipeline

[[autodoc]] CogVideoXFunControlPipeline
  - all
  - __call__

## CogVideoXPipelineOutput

[[autodoc]] pipelines.cogvideo.pipeline_output.CogVideoXPipelineOutput
