<!-- Copyright 2024 The HuggingFace Team. All rights reserved.
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
    <img alt="LoRA" src="https://img.shields.io/badge/LoRA-d8b4fe?style=flat"/>
  </div>
</div>

# HunyuanVideo

[HunyuanVideo](https://huggingface.co/papers/2412.03603) is a 13B diffusion transformer model designed to be competitive with closed-source video foundation models and enable wider community access. This model uses a "dual-stream to single-stream" architecture to separately process the video and text tokens first, before concatenating and feeding them to the transformer to fuse the multimodal information. A pretrained multimodal large language model (MLLM) is used as the encoder because it has better image-text alignment, better image detail description and reasoning, and it can be used as a zero-shot learner if system instructions are added to user prompts. Finally, HunyuanVideo uses a 3D causal variational autoencoder to more efficiently process video data at the original resolution and frame rate.

You can find all the original HunyuanVideo checkpoints under the Tencent [organization](https://huggingface.co/tencent).

> [!TIP]
> The examples below use a checkpoint from [hunyuanvideo-community](https://huggingface.co/hunyuanvideo-community) because the weights are stored in a layout compatible with Diffusers.

The example below demonstrates how to generate a video optimized for memory or inference speed.

<hfoptions id="usage">
<hfoption id="memory">

```py
import torch
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig, HunyuanVideoTransformer3DModel, HunyuanVideoPipeline
from diffusers.utils import export_to_video

# quantization
quant_config = DiffusersBitsAndBytesConfig(load_in_4bit=True)
transformer = HunyuanVideoTransformer3DModel.from_pretrained(
    "hunyuanvideo-community/HunyuanVideo",
    subfolder="transformer",
    quantization_config=quant_config,
    torch_dtype=torch.bfloat16,
)

pipeline = HunyuanVideoPipeline.from_pretrained(
    "hunyuanvideo-community/HunyuanVideo",
    transformer=transformer,
    torch_dtype=torch.float16,
)

# model-offloading
pipeline.enable_model_cpu_offload()
pipeline.vae.enable_tiling()

prompt = "A fluffy teddy bear sits on a bed of soft pillows surrounded by children's toys."
video = pipeline(prompt=prompt, num_frames=61, num_inference_steps=30).frames[0]
export_to_video(video, "output.mp4", fps=15)
```

</hfoptions>
<hfoption id="inference speed">

```py
import torch
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig, HunyuanVideoTransformer3DModel, HunyuanVideoPipeline
from diffusers.utils import export_to_video

# quantization
quant_config = DiffusersBitsAndBytesConfig(load_in_4bit=True)
transformer = HunyuanVideoTransformer3DModel.from_pretrained(
    "hunyuanvideo-community/HunyuanVideo",
    subfolder="transformer",
    quantization_config=quant_config,
    torch_dtype=torch.bfloat16,
)

pipeline = HunyuanVideoPipeline.from_pretrained(
    "hunyuanvideo-community/HunyuanVideo",
    transformer=transformer,
    torch_dtype=torch.float16,
)

# model-offloading
pipeline.enable_model_cpu_offload()
pipeline.vae.enable_tiling()

# torch.compile
pipeline.transformer.to(memory_format=torch.channels_last)
pipeline.transformer = torch.compile(
    pipeline.transformer, mode="max-autotune", fullgraph=True
)

prompt = "A fluffy teddy bear sits on a bed of soft pillows surrounded by children's toys."
video = pipeline(prompt=prompt, num_frames=61, num_inference_steps=30).frames[0]
export_to_video(video, "output.mp4", fps=15)
```

</hfoption>
</hfoptions>

## Notes

- HunyuanVideo supports LoRAs with [`~loaders.HunyuanVideoLoraLoaderMixin.load_lora_weights`].

  ```py
  import torch
  from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig, HunyuanVideoTransformer3DModel, HunyuanVideoPipeline
  from diffusers.utils import export_to_video

  # quantize weights to int4 with bitsandbytes
  quant_config = DiffusersBitsAndBytesConfig(load_in_4bit=True)
  transformer = HunyuanVideoTransformer3DModel.from_pretrained(
      "hunyuanvideo-community/HunyuanVideo",
      subfolder="transformer",
      quantization_config=quant_config,
      torch_dtype=torch.bfloat16,
  )

  pipeline = HunyuanVideoPipeline.from_pretrained(
      "hunyuanvideo-community/HunyuanVideo",
      transformer=transformer,
      torch_dtype=torch.float16,
  )

  # load LoRA weights
  pipeline.load_lora_weights("https://huggingface.co/lucataco/hunyuan-steamboat-willie-10", adapter_name="steamboat-willie")
  pipeline.set_adapters("steamboat-willie", 0.9)

  # model-offloading
  pipeline.enable_model_cpu_offload()
  pipeline.vae.enable_tiling()

  prompt = """
  In the style of SWR. A black and white animated scene featuring a fluffy teddy bear sits on a bed of soft pillows surrounded by children's toys.
  """
  video = pipeline(prompt=prompt, num_frames=61, num_inference_steps=30).frames[0]
  export_to_video(video, "output.mp4", fps=15)
  ```

- Refer to the table below for recommended inference values.

  | parameter | recommended value |
  |---|---|
  | text encoder dtype | `torch.float16` |
  | transformer dtype | `torch.bfloat16` |
  | vae dtype | `torch.float16` |
  | `num_frames` | 4 * k + 1 |

- Try lower `shift` values (`2.0` to `5.0`) for lower resolution videos, and try higher `shift` values (`7.0` to `12.0`) for higher resolution images.

## HunyuanVideoPipeline

[[autodoc]] HunyuanVideoPipeline
  - all
  - __call__

## HunyuanVideoPipelineOutput

[[autodoc]] pipelines.hunyuan_video.pipeline_output.HunyuanVideoPipelineOutput
