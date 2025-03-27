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

# LTX-Video

[LTX-Video](https://huggingface.co/Lightricks/LTX-Video) is a diffusion transformer designed for fast and real-time generation of high-resolution videos from text and images. The main feature of LTX-Video is the Video-VAE. The Video-VAE has a higher pixel to latent compression ratio (1:192) which enables more efficient video data processing and faster generation speed. To support and prevent finer details from being lost during generation, the Video-VAE decoder performs the latent to pixel conversion *and* the last denoising step.

You can find all the original LTX-Video checkpoints under the [Lightricks](https://huggingface.co/Lightricks) organization.

> [!TIP]
> Click on the LTX-Video models in the right sidebar for more examples of other video generation tasks.

The example below demonstrates how to generate a video optimized for memory or inference speed.

<hfoptions id="usage">
<hfoption id="memory">

Refer to the [Reduce memory usage](../../optimization/memory) guide for more details about the various memory saving techniques.

The LTX-Video model below requires ~10GB of VRAM.

```py
import torch
from diffusers import LTXPipeline, LTXVideoTransformer3DModel
from diffusers.hooks import apply_group_offloading
from diffusers.utils import export_to_video

# fp8 layerwise weight-casting
transformer = LTXVideoTransformer3DModel.from_pretrained(
  "Lightricks/LTX-Video",
  subfolder="transformer",
  torch_dtype=torch.bfloat16
)
transformer.enable_layerwise_casting(
  storage_dtype=torch.float8_e4m3fn,
  compute_dtype=torch.bfloat16
)

pipeline = LTXPipeline.from_pretrained("Lightricks/LTX-Video", transformer=transformer, torch_dtype=torch.bfloat16)

# group-offloading
onload_device = torch.device("cuda")
offload_device = torch.device("cpu")
pipeline.transformer.enable_group_offload(onload_device=onload_device, offload_device=offload_device, offload_type="leaf_level", use_stream=True)
apply_group_offloading(pipeline.text_encoder, onload_device=onload_device, offload_type="block_level", num_blocks_per_group=2)
apply_group_offloading(pipeline.vae, onload_device=onload_device, offload_type="leaf_level")

prompt = """
A woman with long brown hair and light skin smiles at another woman with long blonde hair. The woman with brown hair wears a black jacket and has a small, barely noticeable mole on her right cheek. The camera angle is a close-up, focused on the woman with brown hair's face. The lighting is warm and natural, likely from the setting sun, casting a soft glow on the scene. The scene appears to be real-life footage
"""
negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted"

video = pipeline(
    prompt=prompt,
    negative_prompt=negative_prompt,
    width=768,
    height=512,
    num_frames=161,
    decode_timestep=0.03,
    decode_noise_scale=0.025,
    num_inference_steps=50,
).frames[0]
export_to_video(video, "output.mp4", fps=24)
```

</hfoption>
<hfoption id="inference speed">

Compilation is slow the first time but subsequent calls to the pipeline are faster.

```py
import torch
from diffusers import LTXPipeline
from diffusers.utils import export_to_video

pipeline = LTXPipeline.from_pretrained(
    "Lightricks/LTX-Video", torch_dtype=torch.bfloat16
)

# torch.compile
pipeline.transformer.to(memory_format=torch.channels_last)
pipeline.transformer = torch.compile(
    pipeline.transformer, mode="max-autotune", fullgraph=True
)

prompt = """
A woman with long brown hair and light skin smiles at another woman with long blonde hair. The woman with brown hair wears a black jacket and has a small, barely noticeable mole on her right cheek. The camera angle is a close-up, focused on the woman with brown hair's face. The lighting is warm and natural, likely from the setting sun, casting a soft glow on the scene. The scene appears to be real-life footage
"""
negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted"

video = pipeline(
    prompt=prompt,
    negative_prompt=negative_prompt,
    width=768,
    height=512,
    num_frames=161,
    decode_timestep=0.03,
    decode_noise_scale=0.025,
    num_inference_steps=50,
).frames[0]
export_to_video(video, "output.mp4", fps=24)
```

</hfoption>
</hfoptions>

## Notes

- LTX-Video supports LoRAs with [`~loaders.LTXVideoLoraLoaderMixin.load_lora_weights`].

  ```py
  import torch
  from diffusers import LTXConditionPipeline
  from diffusers.utils import export_to_video, load_image

  pipeline = LTXConditionPipeline.from_pretrained(
      "Lightricks/LTX-Video-0.9.5", torch_dtype=torch.bfloat16
  )

  pipeline.load_lora_weights("Lightricks/LTX-Video-Cakeify-LoRA", adapter_name="cakeify")
  pipeline.set_adapters("cakeify")

  # use "CAKEIFY" to trigger the LoRA
  prompt = "CAKEIFY a person using a knife to cut a cake shaped like a cereal box"
  image = load_image("https://i5.walmartimages.com/asr/c0463def-4995-47a7-9486-294fff8cf9fc.f9779f3fc4c621cf1fe86465af1d2ecd.jpeg")

  video = pipeline(
      prompt=prompt,
      image=image,
      width=768,
      height=512,
      num_frames=161,
      decode_timestep=0.03,
      decode_noise_scale=0.025,
      num_inference_steps=50,
  ).frames[0]
  export_to_video(video, "output.mp4", fps=24)
  ```

- LTX-Video supports loading from single files, such as [GGUF checkpoints](../../quantization/gguf), with [`loaders.FromOriginalModelMixin.from_single_file`] or [`loaders.FromSingleFileMixin.from_single_file`].

  ```py
  import torch
  from diffusers.utils import export_to_video
  from diffusers import LTXPipeline, LTXVideoTransformer3DModel, GGUFQuantizationConfig

  transformer = LTXVideoTransformer3DModel.from_single_file(
    "https://huggingface.co/city96/LTX-Video-gguf/blob/main/ltx-video-2b-v0.9-Q3_K_S.gguf"
    quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
    torch_dtype=torch.bfloat16
  )
  pipeline = LTXPipeline.from_pretrained(
    "Lightricks/LTX-Video",
    transformer=transformer,
    torch_dtype=torch.bfloat16
  )
  ```

## LTXPipeline

[[autodoc]] LTXPipeline
  - all
  - __call__

## LTXImageToVideoPipeline

[[autodoc]] LTXImageToVideoPipeline
  - all
  - __call__

## LTXConditionPipeline

[[autodoc]] LTXConditionPipeline
  - all
  - __call__

## LTXLatentUpsamplePipeline

[[autodoc]] LTXLatentUpsamplePipeline
  - all
  - __call__

## LTXPipelineOutput

[[autodoc]] pipelines.ltx.pipeline_output.LTXPipelineOutput
