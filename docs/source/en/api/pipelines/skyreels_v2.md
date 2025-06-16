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
    <a href="https://huggingface.co/docs/diffusers/main/en/tutorials/using_peft_for_inference" target="_blank" rel="noopener">
      <img alt="LoRA" src="https://img.shields.io/badge/LoRA-d8b4fe?style=flat"/>
    </a>
  </div>
</div>

# SkyReels-V2: Infinite-length Film Generative model

[SkyReels-V2](https://huggingface.co/papers/2504.13074) by the SkyReels Team.

*Recent advances in video generation have been driven by diffusion models and autoregressive frameworks, yet critical challenges persist in harmonizing prompt adherence, visual quality, motion dynamics, and duration: compromises in motion dynamics to enhance temporal visual quality, constrained video duration (5-10 seconds) to prioritize resolution, and inadequate shot-aware generation stemming from general-purpose MLLMs' inability to interpret cinematic grammar, such as shot composition, actor expressions, and camera motions. These intertwined limitations hinder realistic long-form synthesis and professional film-style generation. To address these limitations, we propose SkyReels-V2, an Infinite-length Film Generative Model, that synergizes Multi-modal Large Language Model (MLLM), Multi-stage Pretraining, Reinforcement Learning, and Diffusion Forcing Framework. Firstly, we design a comprehensive structural representation of video that combines the general descriptions by the Multi-modal LLM and the detailed shot language by sub-expert models. Aided with human annotation, we then train a unified Video Captioner, named SkyCaptioner-V1, to efficiently label the video data. Secondly, we establish progressive-resolution pretraining for the fundamental video generation, followed by a four-stage post-training enhancement: Initial concept-balanced Supervised Fine-Tuning (SFT) improves baseline quality; Motion-specific Reinforcement Learning (RL) training with human-annotated and synthetic distortion data addresses dynamic artifacts; Our diffusion forcing framework with non-decreasing noise schedules enables long-video synthesis in an efficient search space; Final high-quality SFT refines visual fidelity. All the code and models are available at [this https URL](https://github.com/SkyworkAI/SkyReels-V2).*

You can find all the original SkyReels-V2 checkpoints under the [Skywork](https://huggingface.co/collections/Skywork/skyreels-v2-6801b1b93df627d441d0d0d9) organization.

The following SkyReels-V2 models are supported in Diffusers:
- [SkyReels-V2 DF 1.3B - 540P](https://huggingface.co/Skywork/SkyReels-V2-DF-1.3B-540P-Diffusers)
- [SkyReels-V2 DF 14B - 540P](https://huggingface.co/Skywork/SkyReels-V2-DF-14B-540P-Diffusers)
- [SkyReels-V2 DF 14B - 720P](https://huggingface.co/Skywork/SkyReels-V2-DF-14B-720P-Diffusers)
- [SkyReels-V2 T2V 14B - 540P](https://huggingface.co/Skywork/SkyReels-V2-T2V-14B-540P-Diffusers)
- [SkyReels-V2 T2V 14B - 720P](https://huggingface.co/Skywork/SkyReels-V2-T2V-14B-720P-Diffusers)
- [SkyReels-V2 I2V 1.3B - 540P](https://huggingface.co/Skywork/SkyReels-V2-I2V-1.3B-540P-Diffusers)
- [SkyReels-V2 I2V 14B - 540P](https://huggingface.co/Skywork/SkyReels-V2-I2V-14B-540P-Diffusers)
- [SkyReels-V2 I2V 14B - 720P](https://huggingface.co/Skywork/SkyReels-V2-I2V-14B-720P-Diffusers)

> [!TIP]
> Click on the SkyReels-V2 models in the right sidebar for more examples of video generation.

### Text-to-Video Generation

The example below demonstrates how to generate a video from text optimized for memory or inference speed.

<hfoptions id="T2V usage">
<hfoption id="T2V memory">

Refer to the [Reduce memory usage](../../optimization/memory) guide for more details about the various memory saving techniques.

```py
# pip install ftfy
import torch
import numpy as np
from diffusers import AutoModel, SkyReelsV2DiffusionForcingPipeline
from diffusers.hooks.group_offloading import apply_group_offloading
from diffusers.utils import export_to_video
from transformers import UMT5EncoderModel

text_encoder = UMT5EncoderModel.from_pretrained("Skywork/SkyReels-V2-DF-14B-540P-Diffusers", subfolder="text_encoder", torch_dtype=torch.bfloat16)
vae = AutoModel.from_pretrained("Skywork/SkyReels-V2-DF-14B-540P-Diffusers", subfolder="vae", torch_dtype=torch.float32)
transformer = AutoModel.from_pretrained("Skywork/SkyReels-V2-DF-14B-540P-Diffusers", subfolder="transformer", torch_dtype=torch.bfloat16)

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

pipeline = SkyReelsV2DiffusionForcingPipeline.from_pretrained(
    "Skywork/SkyReels-V2-DF-14B-540P-Diffusers",
    vae=vae,
    transformer=transformer,
    text_encoder=text_encoder,
    torch_dtype=torch.bfloat16
)
pipe = pipe.to("cuda")
pipe.transformer.set_ar_attention(causal_block_size=5)

prompt = "A cat and a dog baking a cake together in a kitchen. The cat is carefully measuring flour, while the dog is stirring the batter with a wooden spoon. The kitchen is cozy, with sunlight streaming through the window."

output = pipe(
    prompt=prompt,
    num_inference_steps=30,
    height=544,
    width=960,
    num_frames=97,
    ar_step=5,  # Controls asynchronous inference (0 for synchronous mode)
    overlap_history=None,  # Number of frames to overlap for smooth transitions in long videos; 17 for long
    addnoise_condition=20,  # Improves consistency in long video generation
).frames[0]
export_to_video(output, "T2V.mp4", fps=24, quality=8)
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
from diffusers import AutoencoderKLWan, SkyReelsV2DiffusionForcingImageToVideoPipeline
from diffusers.utils import export_to_video, load_image


model_id = "Skywork/SkyReels-V2-DF-14B-720P-Diffusers"
vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
pipe = SkyReelsV2DiffusionForcingImageToVideoPipeline.from_pretrained(
    model_id, vae=vae, torch_dtype=torch.bfloat16
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
    image=first_frame, last_image=last_frame, prompt=prompt, height=height, width=width, guidance_scale=5.0
).frames[0]
export_to_video(output, "output.mp4", fps=24)
```

</hfoption>
</hfoptions>


## Notes

- SkyReels-V2 supports LoRAs with [`~loaders.SkyReelsV2LoraLoaderMixin.load_lora_weights`].

  <details>
  <summary>Show example code</summary>

  ```py
  # pip install ftfy
  import torch
  from diffusers import AutoModel, SkyReelsV2DiffusionForcingPipeline
  from diffusers.utils import export_to_video

  vae = AutoModel.from_pretrained(
      "Skywork/SkyReels-V2-DF-1.3B-540P-Diffusers", subfolder="vae", torch_dtype=torch.float32
  )
  pipeline = SkyReelsV2DiffusionForcingPipeline.from_pretrained(
      "Skywork/SkyReels-V2-DF-1.3B-540P-Diffusers", vae=vae, torch_dtype=torch.bfloat16
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
      num_frames=97,
      guidance_scale=6.0,
  ).frames[0]
  export_to_video(output, "output.mp4", fps=24)
  ```

  </details>


## SkyReelsV2DiffusionForcingPipeline

[[autodoc]] SkyReelsV2DiffusionForcingPipeline
  - all
  - __call__

## SkyReelsV2DiffusionForcingImageToVideoPipeline

[[autodoc]] SkyReelsV2DiffusionForcingImageToVideoPipeline
  - all
  - __call__

## SkyReelsV2DiffusionForcingVideoToVideoPipeline

[[autodoc]] SkyReelsV2DiffusionForcingVideoToVideoPipeline
  - all
  - __call__

## SkyReelsV2Pipeline

[[autodoc]] SkyReelsV2Pipeline
  - all
  - __call__

## SkyReelsV2ImageToVideoPipeline

[[autodoc]] SkyReelsV2ImageToVideoPipeline
  - all
  - __call__

## SkyReelsV2PipelineOutput

[[autodoc]] pipelines.skyreels_v2.pipeline_output.SkyReelsV2PipelineOutput