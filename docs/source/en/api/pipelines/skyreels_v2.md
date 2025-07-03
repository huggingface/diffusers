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
- [SkyReels-V2 FLF2V 1.3B - 540P](https://huggingface.co/Skywork/SkyReels-V2-FLF2V-1.3B-540P-Diffusers)

> [!TIP]
> Click on the SkyReels-V2 models in the right sidebar for more examples of video generation.

### A _Visual_ Demonstration

        An example with these parameters:
        base_num_frames=97, num_frames=97, num_inference_steps=30, ar_step=5, causal_block_size=5

        vae_scale_factor_temporal -> 4
        num_latent_frames: (97-1)//vae_scale_factor_temporal+1 = 25 frames -> 5 blocks of 5 frames each

        base_num_latent_frames = (97-1)//vae_scale_factor_temporal+1 = 25 → blocks = 25//5 = 5 blocks
        This 5 blocks means the maximum context length of the model is 25 frames in the latent space.

        Asynchronous Processing Timeline:
        ┌─────────────────────────────────────────────────────────────────┐
        │ Steps:    1    6   11   16   21   26   31   36   41   46   50   │
        │ Block 1: [■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■]                       │
        │ Block 2:      [■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■]                  │
        │ Block 3:           [■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■]             │
        │ Block 4:                [■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■]        │
        │ Block 5:                     [■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■]   │
        └─────────────────────────────────────────────────────────────────┘

        For Long Videos (num_frames > base_num_frames):
        base_num_frames acts as the "sliding window size" for processing long videos.

        Example: 257-frame video with base_num_frames=97, overlap_history=17
        ┌──── Iteration 1 (frames 1-97) ────┐
        │ Processing window: 97 frames      │ → 5 blocks, async processing
        │ Generates: frames 1-97            │
        └───────────────────────────────────┘
                    ┌────── Iteration 2 (frames 81-177) ──────┐
                    │ Processing window: 97 frames            │
                    │ Overlap: 17 frames (81-97) from prev    │ → 5 blocks, async processing
                    │ Generates: frames 98-177                │
                    └─────────────────────────────────────────┘
                                ┌────── Iteration 3 (frames 161-257) ──────┐
                                │ Processing window: 97 frames             │
                                │ Overlap: 17 frames (161-177) from prev   │ → 5 blocks, async processing
                                │ Generates: frames 178-257                │
                                └──────────────────────────────────────────┘

        Each iteration independently runs the asynchronous processing with its own 5 blocks.
        base_num_frames controls:
        1. Memory usage (larger window = more VRAM)
        2. Model context length (must match training constraints)
        3. Number of blocks per iteration (base_num_latent_frames // causal_block_size)

        Each block takes 30 steps to complete denoising.
        Block N starts at step: 1 + (N-1) x ar_step
        Total steps: 30 + (5-1) x 5 = 50 steps


        Synchronous mode (ar_step=0) would process all blocks/frames simultaneously:
        ┌──────────────────────────────────────────────┐
        │ Steps:       1            ...            30  │
        │ All blocks: [■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■] │
        └──────────────────────────────────────────────┘
        Total steps: 30 steps


        An example on how the step matrix is constructed for asynchronous processing:
        Given the parameters: (num_inference_steps=30, flow_shift=8, num_frames=97, ar_step=5, causal_block_size=5)
        - num_latent_frames = (97 frames - 1) // (4 temporal downsampling) + 1 = 25
        - step_template = [999, 995, 991, 986, 980, 975, 969, 963, 956, 948,
                           941, 932, 922, 912, 901, 888, 874, 859, 841, 822,
                           799, 773, 743, 708, 666, 615, 551, 470, 363, 216]

        The algorithm creates a 50x25 step_matrix where:
        - Row 1:  [999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999]
        - Row 2:  [995, 995, 995, 995, 995, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999]
        - Row 3:  [991, 991, 991, 991, 991, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999]
        - ...
        - Row 7:  [969, 969, 969, 969, 969, 995, 995, 995, 995, 995, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999]
        - ...
        - Row 21: [799, 799, 799, 799, 799, 888, 888, 888, 888, 888, 941, 941, 941, 941, 941, 975, 975, 975, 975, 975, 999, 999, 999, 999, 999]
        - ...
        - Row 35: [  0,   0,   0,   0,   0, 216, 216, 216, 216, 216, 666, 666, 666, 666, 666, 822, 822, 822, 822, 822, 901, 901, 901, 901, 901]
        - ...
        - Row 42: [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 551, 551, 551, 551, 551, 773, 773, 773, 773, 773]
        - ...
        - Row 50: [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 216, 216, 216, 216, 216]

        Detailed Row 6 Analysis:
        - step_matrix[5]:       [ 975, 975, 975, 975, 975, 999, 999, 999, 999, 999, 999,  ...,  999]
        - step_index[5]:        [   6,   6,   6,   6,   6,   1,   1,   1,   1,   1,   0,  ...,    0]
        - step_update_mask[5]:  [True,True,True,True,True,True,True,True,True,True,False, ...,False]
        - valid_interval[5]:    (0, 25)

        Key Pattern: Block i lags behind Block i-1 by exactly ar_step=5 timesteps, creating the
        staggered "diffusion forcing" effect where later blocks condition on cleaner earlier blocks.

### Text-to-Video Generation

The example below demonstrates how to generate a video from text.

<hfoptions id="T2V usage">
<hfoption id="T2V memory">

Refer to the [Reduce memory usage](../../optimization/memory) guide for more details about the various memory saving techniques.

From the original repo:
>You can use --ar_step 5 to enable asynchronous inference. When asynchronous inference, --causal_block_size 5 is recommended while it is not supposed to be set for synchronous generation... Asynchronous inference will take more steps to diffuse the whole sequence which means it will be SLOWER than synchronous mode. In our experiments, asynchronous inference may improve the instruction following and visual consistent performance.

```py
# pip install ftfy
import torch
from diffusers import AutoModel, SkyReelsV2DiffusionForcingPipeline, UniPCMultistepScheduler
from diffusers.utils import export_to_video

vae = AutoModel.from_pretrained("Skywork/SkyReels-V2-DF-14B-540P-Diffusers", subfolder="vae", torch_dtype=torch.float32)
transformer = AutoModel.from_pretrained("Skywork/SkyReels-V2-DF-14B-540P-Diffusers", subfolder="transformer", torch_dtype=torch.bfloat16)

pipeline = SkyReelsV2DiffusionForcingPipeline.from_pretrained(
    "Skywork/SkyReels-V2-DF-14B-540P-Diffusers",
    vae=vae,
    transformer=transformer,
    torch_dtype=torch.bfloat16
)
flow_shift = 8.0  # 8.0 for T2V, 5.0 for I2V
pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config, flow_shift=flow_shift)
pipeline = pipeline.to("cuda")

prompt = "A cat and a dog baking a cake together in a kitchen. The cat is carefully measuring flour, while the dog is stirring the batter with a wooden spoon. The kitchen is cozy, with sunlight streaming through the window."

output = pipeline(
    prompt=prompt,
    num_inference_steps=30,
    height=544,  # 720 for 720P
    width=960,   # 1280 for 720P
    num_frames=97,
    base_num_frames=97,  # 121 for 720P
    ar_step=5,  # Controls asynchronous inference (0 for synchronous mode)
    causal_block_size=5,  # Number of frames in each block for asynchronous processing
    overlap_history=None,  # Number of frames to overlap for smooth transitions in long videos; 17 for long video generations
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
from diffusers import AutoencoderKLWan, SkyReelsV2DiffusionForcingImageToVideoPipeline, UniPCMultistepScheduler
from diffusers.utils import export_to_video, load_image


model_id = "Skywork/SkyReels-V2-DF-14B-720P-Diffusers"
vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
pipeline = SkyReelsV2DiffusionForcingImageToVideoPipeline.from_pretrained(
    model_id, vae=vae, torch_dtype=torch.bfloat16
)
flow_shift = 5.0  # 8.0 for T2V, 5.0 for I2V
pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config, flow_shift=flow_shift)
pipeline.to("cuda")

first_frame = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/flf2v_input_first_frame.png")
last_frame = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/flf2v_input_last_frame.png")

def aspect_ratio_resize(image, pipeline, max_area=720 * 1280):
    aspect_ratio = image.height / image.width
    mod_value = pipeline.vae_scale_factor_spatial * pipeline.transformer.config.patch_size[1]
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

first_frame, height, width = aspect_ratio_resize(first_frame, pipeline)
if last_frame.size != first_frame.size:
    last_frame, _, _ = center_crop_resize(last_frame, height, width)

prompt = "CG animation style, a small blue bird takes off from the ground, flapping its wings. The bird's feathers are delicate, with a unique pattern on its chest. The background shows a blue sky with white clouds under bright sunshine. The camera follows the bird upward, capturing its flight and the vastness of the sky from a close-up, low-angle perspective."

output = pipeline(
    image=first_frame, last_image=last_frame, prompt=prompt, height=height, width=width, guidance_scale=5.0
).frames[0]
export_to_video(output, "output.mp4", fps=24, quality=8)
```

</hfoption>
</hfoptions>


### Video-to-Video Generation

<hfoptions id="V2V usage">
<hfoption id="usage">

`SkyReelsV2DiffusionForcingVideoToVideoPipeline` extends a given video.

```python
import numpy as np
import torch
import torchvision.transforms.functional as TF
from diffusers import AutoencoderKLWan, SkyReelsV2DiffusionForcingVideoToVideoPipeline, UniPCMultistepScheduler
from diffusers.utils import export_to_video, load_video


model_id = "Skywork/SkyReels-V2-DF-14B-540P-Diffusers"
vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
pipeline = SkyReelsV2DiffusionForcingVideoToVideoPipeline.from_pretrained(
    model_id, vae=vae, torch_dtype=torch.bfloat16
)
flow_shift = 5.0  # 8.0 for T2V, 5.0 for I2V
pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config, flow_shift=flow_shift)
pipeline.to("cuda")

video = load_video("input_video.mp4")

prompt = "CG animation style, a small blue bird takes off from the ground, flapping its wings. The bird's feathers are delicate, with a unique pattern on its chest. The background shows a blue sky with white clouds under bright sunshine. The camera follows the bird upward, capturing its flight and the vastness of the sky from a close-up, low-angle perspective."

output = pipeline(
    video=video, prompt=prompt, height=544, width=960, guidance_scale=5.0,
    num_inference_steps=30, num_frames=257, base_num_frames=97#, ar_step=5, causal_block_size=5,
).frames[0]
export_to_video(output, "output.mp4", fps=24, quality=8)
# Total frames will be the number of frames of given video + 257
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