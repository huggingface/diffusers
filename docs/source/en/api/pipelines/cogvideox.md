<!--Copyright 2025 The HuggingFace Team. All rights reserved.
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

<div style="float: right;">
  <div class="flex flex-wrap space-x-1">
    <a href="https://huggingface.co/docs/diffusers/main/en/tutorials/using_peft_for_inference" target="_blank" rel="noopener">
      <img alt="LoRA" src="https://img.shields.io/badge/LoRA-d8b4fe?style=flat"/>
    </a>
  </div>
</div>

# CogVideoX

[CogVideoX](https://huggingface.co/papers/2408.06072) is a large diffusion transformer model - available in 2B and 5B parameters - designed to generate longer and more consistent videos from text. This model uses a 3D causal variational autoencoder to more efficiently process video data by reducing sequence length (and associated training compute) and preventing flickering in generated videos. An "expert" transformer with adaptive LayerNorm improves alignment between text and video, and 3D full attention helps accurately capture motion and time in generated videos.

You can find all the original CogVideoX checkpoints under the [CogVideoX](https://huggingface.co/collections/THUDM/cogvideo-66c08e62f1685a3ade464cce) collection.

> [!TIP]
> Click on the CogVideoX models in the right sidebar for more examples of other video generation tasks.

The example below demonstrates how to generate a video optimized for memory or inference speed.

<hfoptions id="usage">
<hfoption id="memory">

Refer to the [Reduce memory usage](../../optimization/memory) guide for more details about the various memory saving techniques.

The quantized CogVideoX 5B model below requires ~16GB of VRAM.

```py
import torch
from diffusers import CogVideoXPipeline, AutoModel
from diffusers.quantizers import PipelineQuantizationConfig
from diffusers.hooks import apply_group_offloading
from diffusers.utils import export_to_video

# quantize weights to int8 with torchao
pipeline_quant_config = PipelineQuantizationConfig(
  quant_backend="torchao",
  quant_kwargs={"quant_type": "int8wo"},
  components_to_quantize="transformer"
)

# fp8 layerwise weight-casting
transformer = AutoModel.from_pretrained(
    "THUDM/CogVideoX-5b",
    subfolder="transformer",
    torch_dtype=torch.bfloat16
)
transformer.enable_layerwise_casting(
    storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16
)

pipeline = CogVideoXPipeline.from_pretrained(
    "THUDM/CogVideoX-5b",
    transformer=transformer,
    quantization_config=pipeline_quant_config,
    torch_dtype=torch.bfloat16
)
pipeline.to("cuda")

# model-offloading
pipeline.enable_model_cpu_offload()

prompt = """
A detailed wooden toy ship with intricately carved masts and sails is seen gliding smoothly over a plush, blue carpet that mimics the waves of the sea. 
The ship's hull is painted a rich brown, with tiny windows. The carpet, soft and textured, provides a perfect backdrop, resembling an oceanic expanse. 
Surrounding the ship are various other toys and children's items, hinting at a playful environment. The scene captures the innocence and imagination of childhood, 
with the toy ship's journey symbolizing endless adventures in a whimsical, indoor setting.
"""

video = pipeline(
    prompt=prompt,
    guidance_scale=6,
    num_inference_steps=50
).frames[0]
export_to_video(video, "output.mp4", fps=8)
```

</hfoption>
<hfoption id="inference speed">

[Compilation](../../optimization/fp16#torchcompile) is slow the first time but subsequent calls to the pipeline are faster.

The average inference time with torch.compile on a 80GB A100 is 76.27 seconds compared to 96.89 seconds for an uncompiled model.

```py
import torch
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video

pipeline = CogVideoXPipeline.from_pretrained(
    "THUDM/CogVideoX-2b",
    torch_dtype=torch.float16
).to("cuda")

# torch.compile
pipeline.transformer.to(memory_format=torch.channels_last)
pipeline.transformer = torch.compile(
    pipeline.transformer, mode="max-autotune", fullgraph=True
)

prompt = """
A detailed wooden toy ship with intricately carved masts and sails is seen gliding smoothly over a plush, blue carpet that mimics the waves of the sea. 
The ship's hull is painted a rich brown, with tiny windows. The carpet, soft and textured, provides a perfect backdrop, resembling an oceanic expanse. 
Surrounding the ship are various other toys and children's items, hinting at a playful environment. The scene captures the innocence and imagination of childhood, 
with the toy ship's journey symbolizing endless adventures in a whimsical, indoor setting.
"""

video = pipeline(
    prompt=prompt,
    guidance_scale=6,
    num_inference_steps=50
).frames[0]
export_to_video(video, "output.mp4", fps=8)
```

</hfoption>
</hfoptions>

## Notes

- CogVideoX supports LoRAs with [`~loaders.CogVideoXLoraLoaderMixin.load_lora_weights`].

  <details>
  <summary>Show example code</summary>

  ```py
  import torch
  from diffusers import CogVideoXPipeline
  from diffusers.hooks import apply_group_offloading
  from diffusers.utils import export_to_video

  pipeline = CogVideoXPipeline.from_pretrained(
      "THUDM/CogVideoX-5b",
      torch_dtype=torch.bfloat16
  )
  pipeline.to("cuda")

  # load LoRA weights
  pipeline.load_lora_weights("finetrainers/CogVideoX-1.5-crush-smol-v0", adapter_name="crush-lora")
  pipeline.set_adapters("crush-lora", 0.9)

  # model-offloading
  pipeline.enable_model_cpu_offload()

  prompt = """
  PIKA_CRUSH A large metal cylinder is seen pressing down on a pile of Oreo cookies, flattening them as if they were under a hydraulic press.
  """
  negative_prompt = "inconsistent motion, blurry motion, worse quality, degenerate outputs, deformed outputs"

  video = pipeline(
      prompt=prompt, 
      negative_prompt=negative_prompt, 
      num_frames=81, 
      height=480,
      width=768,
      num_inference_steps=50
  ).frames[0]
  export_to_video(video, "output.mp4", fps=16)
  ```

  </details>

- The text-to-video (T2V) checkpoints work best with a resolution of 1360x768 because that was the resolution it was pretrained on.

- The image-to-video (I2V) checkpoints work with multiple resolutions. The width can vary from 768 to 1360, but the height must be 758. Both height and width must be divisible by 16.

- Both T2V and I2V checkpoints work best with 81 and 161 frames. It is recommended to export the generated video at 16fps.

- Refer to the table below to view memory usage when various memory-saving techniques are enabled.

  | method | memory usage (enabled) | memory usage (disabled) |
  |---|---|---|
  | enable_model_cpu_offload | 19GB | 33GB |
  | enable_sequential_cpu_offload | <4GB | ~33GB (very slow inference speed) |
  | enable_tiling | 11GB (with enable_model_cpu_offload) | --- |
 
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
