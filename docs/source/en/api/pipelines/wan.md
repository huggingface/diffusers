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

# Wan

[Wan-2.1](https://huggingface.co/papers/2503.20314) by the Wan Team.

*This report presents Wan, a comprehensive and open suite of video foundation models designed to push the boundaries of video generation. Built upon the mainstream diffusion transformer paradigm, Wan achieves significant advancements in generative capabilities through a series of innovations, including our novel VAE, scalable pre-training strategies, large-scale data curation, and automated evaluation metrics. These contributions collectively enhance the model's performance and versatility. Specifically, Wan is characterized by four key features: Leading Performance: The 14B model of Wan, trained on a vast dataset comprising billions of images and videos, demonstrates the scaling laws of video generation with respect to both data and model size. It consistently outperforms the existing open-source models as well as state-of-the-art commercial solutions across multiple internal and external benchmarks, demonstrating a clear and significant performance superiority. Comprehensiveness: Wan offers two capable models, i.e., 1.3B and 14B parameters, for efficiency and effectiveness respectively. It also covers multiple downstream applications, including image-to-video, instruction-guided video editing, and personal video generation, encompassing up to eight tasks. Consumer-Grade Efficiency: The 1.3B model demonstrates exceptional resource efficiency, requiring only 8.19 GB VRAM, making it compatible with a wide range of consumer-grade GPUs. Openness: We open-source the entire series of Wan, including source code and all models, with the goal of fostering the growth of the video generation community. This openness seeks to significantly expand the creative possibilities of video production in the industry and provide academia with high-quality video foundation models. All the code and models are available at [this https URL](https://github.com/Wan-Video/Wan2.1).*

You can find all the original Wan2.1 checkpoints under the [Wan-AI](https://huggingface.co/Wan-AI) organization.

The following Wan models are supported in Diffusers:

- [Wan 2.1 T2V 1.3B](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers)
- [Wan 2.1 T2V 14B](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B-Diffusers)
- [Wan 2.1 I2V 14B - 480P](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P-Diffusers)
- [Wan 2.1 I2V 14B - 720P](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P-Diffusers)
- [Wan 2.1 FLF2V 14B - 720P](https://huggingface.co/Wan-AI/Wan2.1-FLF2V-14B-720P-diffusers)
- [Wan 2.1 VACE 1.3B](https://huggingface.co/Wan-AI/Wan2.1-VACE-1.3B-diffusers)
- [Wan 2.1 VACE 14B](https://huggingface.co/Wan-AI/Wan2.1-VACE-14B-diffusers)
- [Wan 2.2 T2V 14B](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers)
- [Wan 2.2 I2V 14B](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B-Diffusers)
- [Wan 2.2 TI2V 5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers)
- [Wan 2.2 Animate 14B](https://huggingface.co/Wan-AI/Wan2.2-Animate-14B-Diffusers)

> [!TIP]
> Click on the Wan models in the right sidebar for more examples of video generation.

### Text-to-Video Generation

The example below demonstrates how to generate a video from text optimized for memory or inference speed.

<hfoptions id="T2V usage">
<hfoption id="T2V memory">

Refer to the [Reduce memory usage](../../optimization/memory) guide for more details about the various memory saving techniques.

The Wan2.1 text-to-video model below requires ~13GB of VRAM.

```py
# pip install ftfy
import torch
import numpy as np
from diffusers import AutoModel, WanPipeline
from diffusers.quantizers import PipelineQuantizationConfig
from diffusers.hooks.group_offloading import apply_group_offloading
from diffusers.utils import export_to_video, load_image
from transformers import UMT5EncoderModel

text_encoder = UMT5EncoderModel.from_pretrained("Wan-AI/Wan2.1-T2V-14B-Diffusers", subfolder="text_encoder", torch_dtype=torch.bfloat16)
vae = AutoModel.from_pretrained("Wan-AI/Wan2.1-T2V-14B-Diffusers", subfolder="vae", torch_dtype=torch.float32)
transformer = AutoModel.from_pretrained("Wan-AI/Wan2.1-T2V-14B-Diffusers", subfolder="transformer", torch_dtype=torch.bfloat16)

# group-offloading
onload_device = torch.device("cuda")
offload_device = torch.device("cpu")
apply_group_offloading(text_encoder,
    onload_device=onload_device,
    offload_device=offload_device,
    offload_type="block_level",
    num_blocks_per_group=4
)
transformer.enable_group_offload(
    onload_device=onload_device,
    offload_device=offload_device,
    offload_type="leaf_level",
    use_stream=True
)

pipeline = WanPipeline.from_pretrained(
    "Wan-AI/Wan2.1-T2V-14B-Diffusers",
    vae=vae,
    transformer=transformer,
    text_encoder=text_encoder,
    torch_dtype=torch.bfloat16
)
pipeline.to("cuda")

prompt = """
The camera rushes from far to near in a low-angle shot,
revealing a white ferret on a log. It plays, leaps into the water, and emerges, as the camera zooms in
for a close-up. Water splashes berry bushes nearby, while moss, snow, and leaves blanket the ground.
Birch trees and a light blue sky frame the scene, with ferns in the foreground. Side lighting casts dynamic
shadows and warm highlights. Medium composition, front view, low angle, with depth of field.
"""
negative_prompt = """
Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality,
low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured,
misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards
"""

output = pipeline(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_frames=81,
    guidance_scale=5.0,
).frames[0]
export_to_video(output, "output.mp4", fps=16)
```

</hfoption>
<hfoption id="T2V inference speed">

[Compilation](../../optimization/fp16#torchcompile) is slow the first time but subsequent calls to the pipeline are faster. [Caching](../../optimization/cache) may also speed up inference by storing and reusing intermediate outputs.

```py
# pip install ftfy
import torch
import numpy as np
from diffusers import AutoModel, WanPipeline
from diffusers.hooks.group_offloading import apply_group_offloading
from diffusers.utils import export_to_video, load_image
from transformers import UMT5EncoderModel

text_encoder = UMT5EncoderModel.from_pretrained("Wan-AI/Wan2.1-T2V-14B-Diffusers", subfolder="text_encoder", torch_dtype=torch.bfloat16)
vae = AutoModel.from_pretrained("Wan-AI/Wan2.1-T2V-14B-Diffusers", subfolder="vae", torch_dtype=torch.float32)
transformer = AutoModel.from_pretrained("Wan-AI/Wan2.1-T2V-14B-Diffusers", subfolder="transformer", torch_dtype=torch.bfloat16)

pipeline = WanPipeline.from_pretrained(
    "Wan-AI/Wan2.1-T2V-14B-Diffusers",
    vae=vae,
    transformer=transformer,
    text_encoder=text_encoder,
    torch_dtype=torch.bfloat16
)
pipeline.to("cuda")

# torch.compile
pipeline.transformer.to(memory_format=torch.channels_last)
pipeline.transformer = torch.compile(
    pipeline.transformer, mode="max-autotune", fullgraph=True
)

prompt = """
The camera rushes from far to near in a low-angle shot,
revealing a white ferret on a log. It plays, leaps into the water, and emerges, as the camera zooms in
for a close-up. Water splashes berry bushes nearby, while moss, snow, and leaves blanket the ground.
Birch trees and a light blue sky frame the scene, with ferns in the foreground. Side lighting casts dynamic
shadows and warm highlights. Medium composition, front view, low angle, with depth of field.
"""
negative_prompt = """
Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality,
low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured,
misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards
"""

output = pipeline(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_frames=81,
    guidance_scale=5.0,
).frames[0]
export_to_video(output, "output.mp4", fps=16)
```

</hfoption>
</hfoptions>

### First-Last-Frame-to-Video Generation

The example below demonstrates how to use the image-to-video pipeline to generate a video using a text description, a starting frame, and an ending frame.

<hfoptions id="FLF2V usage">
<hfoption id="usage">

```python
import numpy as np
import torch
import torchvision.transforms.functional as TF
from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
from diffusers.utils import export_to_video, load_image
from transformers import CLIPVisionModel


model_id = "Wan-AI/Wan2.1-FLF2V-14B-720P-diffusers"
image_encoder = CLIPVisionModel.from_pretrained(model_id, subfolder="image_encoder", torch_dtype=torch.float32)
vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
pipe = WanImageToVideoPipeline.from_pretrained(
    model_id, vae=vae, image_encoder=image_encoder, torch_dtype=torch.bfloat16
)
pipe.to("cuda")

first_frame = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/flf2v_input_first_frame.png")
last_frame = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/flf2v_input_last_frame.png")

def aspect_ratio_resize(image, pipe, max_area=720 * 1280):
    aspect_ratio = image.height / image.width
    mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
    height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
    width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
    image = image.resize((width, height))
    return image, height, width

def center_crop_resize(image, height, width):
    # Calculate resize ratio to match first frame dimensions
    resize_ratio = max(width / image.width, height / image.height)

    # Resize the image
    width = round(image.width * resize_ratio)
    height = round(image.height * resize_ratio)
    size = [width, height]
    image = TF.center_crop(image, size)

    return image, height, width

first_frame, height, width = aspect_ratio_resize(first_frame, pipe)
if last_frame.size != first_frame.size:
    last_frame, _, _ = center_crop_resize(last_frame, height, width)

prompt = "CG animation style, a small blue bird takes off from the ground, flapping its wings. The bird's feathers are delicate, with a unique pattern on its chest. The background shows a blue sky with white clouds under bright sunshine. The camera follows the bird upward, capturing its flight and the vastness of the sky from a close-up, low-angle perspective."

output = pipe(
    image=first_frame, last_image=last_frame, prompt=prompt, height=height, width=width, guidance_scale=5.5
).frames[0]
export_to_video(output, "output.mp4", fps=16)
```

</hfoption>
</hfoptions>

### Any-to-Video Controllable Generation

Wan VACE supports various generation techniques which achieve controllable video generation. Some of the capabilities include:
- Control to Video (Depth, Pose, Sketch, Flow, Grayscale, Scribble, Layout, Boundary Box, etc.). Recommended library for preprocessing videos to obtain control videos: [huggingface/controlnet_aux]()
- Image/Video to Video (first frame, last frame, starting clip, ending clip, random clips)
- Inpainting and Outpainting
- Subject to Video (faces, object, characters, etc.)
- Composition to Video (reference anything, animate anything, swap anything, expand anything, move anything, etc.)

The code snippets available in [this](https://github.com/huggingface/diffusers/pull/11582) pull request demonstrate some examples of how videos can be generated with controllability signals.

The general rule of thumb to keep in mind when preparing inputs for the VACE pipeline is that the input images, or frames of a video that you want to use for conditioning, should have a corresponding mask that is black in color. The black mask signifies that the model will not generate new content for that area, and only use those parts for conditioning the generation process. For parts/frames that should be generated by the model, the mask should be white in color.

</hfoption>
</hfoptions>

### Wan-Animate: Unified Character Animation and Replacement with Holistic Replication

[Wan-Animate](https://huggingface.co/papers/2509.14055) by the Wan Team.

*We introduce Wan-Animate, a unified framework for character animation and replacement. Given a character image and a reference video, Wan-Animate can animate the character by precisely replicating the expressions and movements of the character in the video to generate high-fidelity character videos. Alternatively, it can integrate the animated character into the reference video to replace the original character, replicating the scene's lighting and color tone to achieve seamless environmental integration. Wan-Animate is built upon the Wan model. To adapt it for character animation tasks, we employ a modified input paradigm to differentiate between reference conditions and regions for generation. This design unifies multiple tasks into a common symbolic representation. We use spatially-aligned skeleton signals to replicate body motion and implicit facial features extracted from source images to reenact expressions, enabling the generation of character videos with high controllability and expressiveness. Furthermore, to enhance environmental integration during character replacement, we develop an auxiliary Relighting LoRA. This module preserves the character's appearance consistency while applying the appropriate environmental lighting and color tone. Experimental results demonstrate that Wan-Animate achieves state-of-the-art performance. We are committed to open-sourcing the model weights and its source code.*

The project page: https://humanaigc.github.io/wan-animate

This model was mostly contributed by [M. Tolga Cangöz](https://github.com/tolgacangoz).

#### Usage

The Wan-Animate pipeline supports two modes of operation:

1. **Animation Mode** (default): Animates a character image based on motion and expression from reference videos
2. **Replacement Mode**: Replaces a character in a background video with a new character while preserving the scene

##### Prerequisites

Before using the pipeline, you need to preprocess your reference video to extract:
- **Pose video**: Contains skeletal keypoints representing body motion
- **Face video**: Contains facial feature representations for expression control

For replacement mode, you additionally need:
- **Background video**: The original video containing the scene
- **Mask video**: A mask indicating where to generate content (white) vs. preserve original (black)

> [!NOTE]
> The preprocessing tools are available in the original Wan-Animate repository. Integration of these preprocessing steps into Diffusers is planned for a future release.

The example below demonstrates how to use the Wan-Animate pipeline:

<hfoptions id="Animate usage">
<hfoption id="Animation mode">

```python
import numpy as np
import torch
from diffusers import AutoencoderKLWan, WanAnimatePipeline
from diffusers.utils import export_to_video, load_image, load_video
from transformers import CLIPVisionModel

model_id = "Wan-AI/Wan2.2-Animate-14B-Diffusers"
vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
pipe = WanAnimatePipeline.from_pretrained(
    model_id, vae=vae, torch_dtype=torch.bfloat16
)
pipe.to("cuda")

# Load character image and preprocessed videos
image = load_image("path/to/character.jpg")
pose_video = load_video("path/to/pose_video.mp4")  # Preprocessed skeletal keypoints
face_video = load_video("path/to/face_video.mp4")  # Preprocessed facial features

# Resize image to match VAE constraints
def aspect_ratio_resize(image, pipe, max_area=720 * 1280):
    aspect_ratio = image.height / image.width
    mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
    height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
    width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
    image = image.resize((width, height))
    return image, height, width

image, height, width = aspect_ratio_resize(image, pipe)

prompt = "A person dancing energetically in a studio with dynamic lighting and professional camera work"
negative_prompt = "blurry, low quality, distorted, deformed, static, poorly drawn"

# Generate animated video
output = pipe(
    image=image,
    pose_video=pose_video,
    face_video=face_video,
    prompt=prompt,
    negative_prompt=negative_prompt,
    height=height,
    width=width,
    num_frames=81,
    guidance_scale=5.0,
    mode="animation",  # Animation mode (default)
).frames[0]
export_to_video(output, "animated_character.mp4", fps=16)
```

</hfoption>
<hfoption id="Replacement mode">

```python
import numpy as np
import torch
from diffusers import AutoencoderKLWan, WanAnimatePipeline
from diffusers.utils import export_to_video, load_image, load_video
from transformers import CLIPVisionModel

model_id = "Wan-AI/Wan2.2-Animate-14B-Diffusers"
image_encoder = CLIPVisionModel.from_pretrained(model_id, subfolder="image_encoder", torch_dtype=torch.float16)
vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
pipe = WanAnimatePipeline.from_pretrained(
    model_id, vae=vae, image_encoder=image_encoder, torch_dtype=torch.bfloat16
)
pipe.to("cuda")

# Load all required inputs for replacement mode
image = load_image("path/to/new_character.jpg")
pose_video = load_video("path/to/pose_video.mp4")  # Preprocessed skeletal keypoints
face_video = load_video("path/to/face_video.mp4")  # Preprocessed facial features
background_video = load_video("path/to/background_video.mp4")  # Original scene
mask_video = load_video("path/to/mask_video.mp4")  # Black: preserve, White: generate

# Resize image to match video dimensions
def aspect_ratio_resize(image, pipe, max_area=720 * 1280):
    aspect_ratio = image.height / image.width
    mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
    height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
    width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
    image = image.resize((width, height))
    return image, height, width

image, height, width = aspect_ratio_resize(image, pipe)

prompt = "A person seamlessly integrated into the scene with consistent lighting and environment"
negative_prompt = "blurry, low quality, inconsistent lighting, floating, disconnected from scene"

# Replace character in background video
output = pipe(
    image=image,
    pose_video=pose_video,
    face_video=face_video,
    background_video=background_video,
    mask_video=mask_video,
    prompt=prompt,
    negative_prompt=negative_prompt,
    height=height,
    width=width,
    num_frames=81,
    guidance_scale=5.0,
    mode="replacement",  # Replacement mode
).frames[0]
export_to_video(output, "character_replaced.mp4", fps=16)
```

</hfoption>
<hfoption id="Advanced options">

```python
import numpy as np
import torch
from diffusers import AutoencoderKLWan, WanAnimatePipeline
from diffusers.utils import export_to_video, load_image, load_video
from transformers import CLIPVisionModel

model_id = "Wan-AI/Wan2.2-Animate-14B-Diffusers"
image_encoder = CLIPVisionModel.from_pretrained(model_id, subfolder="image_encoder", torch_dtype=torch.float16)
vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
pipe = WanAnimatePipeline.from_pretrained(
    model_id, vae=vae, image_encoder=image_encoder, torch_dtype=torch.bfloat16
)
pipe.to("cuda")

image = load_image("path/to/character.jpg")
pose_video = load_video("path/to/pose_video.mp4")
face_video = load_video("path/to/face_video.mp4")

def aspect_ratio_resize(image, pipe, max_area=720 * 1280):
    aspect_ratio = image.height / image.width
    mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
    height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
    width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
    image = image.resize((width, height))
    return image, height, width

image, height, width = aspect_ratio_resize(image, pipe)

prompt = "A person dancing energetically in a studio"
negative_prompt = "blurry, low quality"

# Advanced: Use temporal guidance and custom callback
def callback_fn(pipe, step_index, timestep, callback_kwargs):
    # You can modify latents or other tensors here
    print(f"Step {step_index}, Timestep {timestep}")
    return callback_kwargs

output = pipe(
    image=image,
    pose_video=pose_video,
    face_video=face_video,
    prompt=prompt,
    negative_prompt=negative_prompt,
    height=height,
    width=width,
    num_frames=81,
    num_inference_steps=50,
    guidance_scale=5.0,
    num_frames_for_temporal_guidance=5,  # Use 5 frames for temporal guidance (1 or 5 recommended)
    callback_on_step_end=callback_fn,
    callback_on_step_end_tensor_inputs=["latents"],
).frames[0]
export_to_video(output, "animated_advanced.mp4", fps=16)
```

</hfoption>
</hfoptions>

#### Key Parameters

- **mode**: Choose between `"animation"` (default) or `"replacement"`
- **num_frames_for_temporal_guidance**: Number of frames for temporal guidance (1 or 5 recommended). Using 5 provides better temporal consistency but requires more memory
- **guidance_scale**: Controls how closely the output follows the text prompt. Higher values (5-7) produce results more aligned with the prompt
- **num_frames**: Total number of frames to generate. Should be divisible by `vae_scale_factor_temporal` (default: 4)


## Notes

- Wan2.1 supports LoRAs with [`~loaders.WanLoraLoaderMixin.load_lora_weights`].

  <details>
  <summary>Show example code</summary>

  ```py
  # pip install ftfy
  import torch
  from diffusers import AutoModel, WanPipeline
  from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
  from diffusers.utils import export_to_video

  vae = AutoModel.from_pretrained(
      "Wan-AI/Wan2.1-T2V-1.3B-Diffusers", subfolder="vae", torch_dtype=torch.float32
  )
  pipeline = WanPipeline.from_pretrained(
      "Wan-AI/Wan2.1-T2V-1.3B-Diffusers", vae=vae, torch_dtype=torch.bfloat16
  )
  pipeline.scheduler = UniPCMultistepScheduler.from_config(
      pipeline.scheduler.config, flow_shift=5.0
  )
  pipeline.to("cuda")

  pipeline.load_lora_weights("benjamin-paine/steamboat-willie-1.3b", adapter_name="steamboat-willie")
  pipeline.set_adapters("steamboat-willie")

  pipeline.enable_model_cpu_offload()

  # use "steamboat willie style" to trigger the LoRA
  prompt = """
  steamboat willie style, golden era animation, The camera rushes from far to near in a low-angle shot,
  revealing a white ferret on a log. It plays, leaps into the water, and emerges, as the camera zooms in
  for a close-up. Water splashes berry bushes nearby, while moss, snow, and leaves blanket the ground.
  Birch trees and a light blue sky frame the scene, with ferns in the foreground. Side lighting casts dynamic
  shadows and warm highlights. Medium composition, front view, low angle, with depth of field.
  """

  output = pipeline(
      prompt=prompt,
      num_frames=81,
      guidance_scale=5.0,
  ).frames[0]
  export_to_video(output, "output.mp4", fps=16)
  ```

  </details>

- [`WanTransformer3DModel`] and [`AutoencoderKLWan`] supports loading from single files with [`~loaders.FromSingleFileMixin.from_single_file`].

  <details>
  <summary>Show example code</summary>

  ```py
  # pip install ftfy
  import torch
  from diffusers import WanPipeline, WanTransformer3DModel, AutoencoderKLWan

  vae = AutoencoderKLWan.from_single_file(
      "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/blob/main/split_files/vae/wan_2.1_vae.safetensors"
  )
  transformer = WanTransformer3DModel.from_single_file(
      "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/blob/main/split_files/diffusion_models/wan2.1_t2v_1.3B_bf16.safetensors",
      torch_dtype=torch.bfloat16
  )
  pipeline = WanPipeline.from_pretrained(
      "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
      vae=vae,
      transformer=transformer,
      torch_dtype=torch.bfloat16
  )
  ```

  </details>

- Set the [`AutoencoderKLWan`] dtype to `torch.float32` for better decoding quality.

- The number of frames per second (fps) or `k` should be calculated by `4 * k + 1`.

- Try lower `shift` values (`2.0` to `5.0`) for lower resolution videos and higher `shift` values (`7.0` to `12.0`) for higher resolution images.

- Wan 2.1 and 2.2 support using [LightX2V LoRAs](https://huggingface.co/Kijai/WanVideo_comfy/tree/main/Lightx2v) to speed up inference. Using them on Wan 2.2 is slightly more involed. Refer to [this code snippet](https://github.com/huggingface/diffusers/pull/12040#issuecomment-3144185272) to learn more.

- Wan 2.2 has two denoisers. By default, LoRAs are only loaded into the first denoiser. One can set `load_into_transformer_2=True` to load LoRAs into the second denoiser. Refer to [this](https://github.com/huggingface/diffusers/pull/12074#issue-3292620048) and [this](https://github.com/huggingface/diffusers/pull/12074#issuecomment-3155896144) examples to learn more.

## WanPipeline

[[autodoc]] WanPipeline
  - all
  - __call__

## WanImageToVideoPipeline

[[autodoc]] WanImageToVideoPipeline
  - all
  - __call__

## WanVACEPipeline

[[autodoc]] WanVACEPipeline
  - all
  - __call__

## WanVideoToVideoPipeline

[[autodoc]] WanVideoToVideoPipeline
  - all
  - __call__

## WanAnimatePipeline

[[autodoc]] WanAnimatePipeline
  - all
  - __call__

## WanPipelineOutput

[[autodoc]] pipelines.wan.pipeline_output.WanPipelineOutput
