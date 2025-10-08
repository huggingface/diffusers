<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Text-to-Video Generation with AnimateDiff

<div class="flex flex-wrap space-x-1">
  <img alt="LoRA" src="https://img.shields.io/badge/LoRA-d8b4fe?style=flat"/>
</div>

## Overview

[AnimateDiff: Animate Your Personalized Text-to-Image Diffusion Models without Specific Tuning](https://huggingface.co/papers/2307.04725) by Yuwei Guo, Ceyuan Yang, Anyi Rao, Yaohui Wang, Yu Qiao, Dahua Lin, Bo Dai.

The abstract of the paper is the following:

*With the advance of text-to-image models (e.g., Stable Diffusion) and corresponding personalization techniques such as DreamBooth and LoRA, everyone can manifest their imagination into high-quality images at an affordable cost. Subsequently, there is a great demand for image animation techniques to further combine generated static images with motion dynamics. In this report, we propose a practical framework to animate most of the existing personalized text-to-image models once and for all, saving efforts in model-specific tuning. At the core of the proposed framework is to insert a newly initialized motion modeling module into the frozen text-to-image model and train it on video clips to distill reasonable motion priors. Once trained, by simply injecting this motion modeling module, all personalized versions derived from the same base T2I readily become text-driven models that produce diverse and personalized animated images. We conduct our evaluation on several public representative personalized text-to-image models across anime pictures and realistic photographs, and demonstrate that our proposed framework helps these models generate temporally smooth animation clips while preserving the domain and diversity of their outputs. Code and pre-trained weights will be publicly available at [this https URL](https://animatediff.github.io/).*

## Available Pipelines

| Pipeline | Tasks | Demo
|---|---|:---:|
| [AnimateDiffPipeline](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/animatediff/pipeline_animatediff.py) | *Text-to-Video Generation with AnimateDiff* |
| [AnimateDiffControlNetPipeline](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/animatediff/pipeline_animatediff_controlnet.py) | *Controlled Video-to-Video Generation with AnimateDiff using ControlNet* |
| [AnimateDiffSparseControlNetPipeline](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/animatediff/pipeline_animatediff_sparsectrl.py) | *Controlled Video-to-Video Generation with AnimateDiff using SparseCtrl* |
| [AnimateDiffSDXLPipeline](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/animatediff/pipeline_animatediff_sdxl.py) | *Video-to-Video Generation with AnimateDiff* |
| [AnimateDiffVideoToVideoPipeline](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/animatediff/pipeline_animatediff_video2video.py) | *Video-to-Video Generation with AnimateDiff* |
| [AnimateDiffVideoToVideoControlNetPipeline](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/animatediff/pipeline_animatediff_video2video_controlnet.py) | *Video-to-Video Generation with AnimateDiff using ControlNet* |

## Available checkpoints

Motion Adapter checkpoints can be found under [guoyww](https://huggingface.co/guoyww/). These checkpoints are meant to work with any model based on Stable Diffusion 1.4/1.5.

## Usage example

### AnimateDiffPipeline

AnimateDiff works with a MotionAdapter checkpoint and a Stable Diffusion model checkpoint. The MotionAdapter is a collection of Motion Modules that are responsible for adding coherent motion across image frames. These modules are applied after the Resnet and Attention blocks in Stable Diffusion UNet.

The following example demonstrates how to use a *MotionAdapter* checkpoint with Diffusers for inference based on StableDiffusion-1.4/1.5.

```python
import torch
from diffusers import AnimateDiffPipeline, DDIMScheduler, MotionAdapter
from diffusers.utils import export_to_gif

# Load the motion adapter
adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2", torch_dtype=torch.float16)
# load SD 1.5 based finetuned model
model_id = "SG161222/Realistic_Vision_V5.1_noVAE"
pipe = AnimateDiffPipeline.from_pretrained(model_id, motion_adapter=adapter, torch_dtype=torch.float16)
scheduler = DDIMScheduler.from_pretrained(
    model_id,
    subfolder="scheduler",
    clip_sample=False,
    timestep_spacing="linspace",
    beta_schedule="linear",
    steps_offset=1,
)
pipe.scheduler = scheduler

# enable memory savings
pipe.enable_vae_slicing()
pipe.enable_model_cpu_offload()

output = pipe(
    prompt=(
        "masterpiece, bestquality, highlydetailed, ultradetailed, sunset, "
        "orange sky, warm lighting, fishing boats, ocean waves seagulls, "
        "rippling water, wharf, silhouette, serene atmosphere, dusk, evening glow, "
        "golden hour, coastal landscape, seaside scenery"
    ),
    negative_prompt="bad quality, worse quality",
    num_frames=16,
    guidance_scale=7.5,
    num_inference_steps=25,
    generator=torch.Generator("cpu").manual_seed(42),
)
frames = output.frames[0]
export_to_gif(frames, "animation.gif")
```

Here are some sample outputs:

<table>
    <tr>
        <td><center>
        masterpiece, bestquality, sunset.
        <br>
        <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/animatediff-realistic-doc.gif"
            alt="masterpiece, bestquality, sunset"
            style="width: 300px;" />
        </center></td>
    </tr>
</table>

> [!TIP]
> AnimateDiff tends to work better with finetuned Stable Diffusion models. If you plan on using a scheduler that can clip samples, make sure to disable it by setting `clip_sample=False` in the scheduler as this can also have an adverse effect on generated samples. Additionally, the AnimateDiff checkpoints can be sensitive to the beta schedule of the scheduler. We recommend setting this to `linear`.

### AnimateDiffControlNetPipeline

AnimateDiff can also be used with ControlNets ControlNet was introduced in [Adding Conditional Control to Text-to-Image Diffusion Models](https://huggingface.co/papers/2302.05543) by Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. With a ControlNet model, you can provide an additional control image to condition and control Stable Diffusion generation. For example, if you provide depth maps, the ControlNet model generates a video that'll preserve the spatial information from the depth maps. It is a more flexible and accurate way to control the video generation process.

```python
import torch
from diffusers import AnimateDiffControlNetPipeline, AutoencoderKL, ControlNetModel, MotionAdapter, LCMScheduler
from diffusers.utils import export_to_gif, load_video

# Additionally, you will need a preprocess videos before they can be used with the ControlNet
# HF maintains just the right package for it: `pip install controlnet_aux`
from controlnet_aux.processor import ZoeDetector

# Download controlnets from https://huggingface.co/lllyasviel/ControlNet-v1-1 to use .from_single_file
# Download Diffusers-format controlnets, such as https://huggingface.co/lllyasviel/sd-controlnet-depth, to use .from_pretrained()
controlnet = ControlNetModel.from_single_file("control_v11f1p_sd15_depth.pth", torch_dtype=torch.float16)

# We use AnimateLCM for this example but one can use the original motion adapters as well (for example, https://huggingface.co/guoyww/animatediff-motion-adapter-v1-5-3)
motion_adapter = MotionAdapter.from_pretrained("wangfuyun/AnimateLCM")

vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16)
pipe: AnimateDiffControlNetPipeline = AnimateDiffControlNetPipeline.from_pretrained(
    "SG161222/Realistic_Vision_V5.1_noVAE",
    motion_adapter=motion_adapter,
    controlnet=controlnet,
    vae=vae,
).to(device="cuda", dtype=torch.float16)
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config, beta_schedule="linear")
pipe.load_lora_weights("wangfuyun/AnimateLCM", weight_name="AnimateLCM_sd15_t2v_lora.safetensors", adapter_name="lcm-lora")
pipe.set_adapters(["lcm-lora"], [0.8])

depth_detector = ZoeDetector.from_pretrained("lllyasviel/Annotators").to("cuda")
video = load_video("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/animatediff-vid2vid-input-1.gif")
conditioning_frames = []

with pipe.progress_bar(total=len(video)) as progress_bar:
    for frame in video:
        conditioning_frames.append(depth_detector(frame))
        progress_bar.update()

prompt = "a panda, playing a guitar, sitting in a pink boat, in the ocean, mountains in background, realistic, high quality"
negative_prompt = "bad quality, worst quality"

video = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_frames=len(video),
    num_inference_steps=10,
    guidance_scale=2.0,
    conditioning_frames=conditioning_frames,
    generator=torch.Generator().manual_seed(42),
).frames[0]

export_to_gif(video, "animatediff_controlnet.gif", fps=8)
```

Here are some sample outputs:

<table align="center">
    <tr>
      <th align="center">Source Video</th>
      <th align="center">Output Video</th>
    </tr>
    <tr>
        <td align="center">
          raccoon playing a guitar
          <br />
          <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/animatediff-vid2vid-input-1.gif" alt="racoon playing a guitar" />
        </td>
        <td align="center">
          a panda, playing a guitar, sitting in a pink boat, in the ocean, mountains in background, realistic, high quality
          <br/>
          <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/animatediff-controlnet-output.gif" alt="a panda, playing a guitar, sitting in a pink boat, in the ocean, mountains in background, realistic, high quality" />
        </td>
    </tr>
</table>

### AnimateDiffSparseControlNetPipeline

[SparseCtrl: Adding Sparse Controls to Text-to-Video Diffusion Models](https://huggingface.co/papers/2311.16933) for achieving controlled generation in text-to-video diffusion models by Yuwei Guo, Ceyuan Yang, Anyi Rao, Maneesh Agrawala, Dahua Lin, and Bo Dai.

The abstract from the paper is:

*The development of text-to-video (T2V), i.e., generating videos with a given text prompt, has been significantly advanced in recent years. However, relying solely on text prompts often results in ambiguous frame composition due to spatial uncertainty. The research community thus leverages the dense structure signals, e.g., per-frame depth/edge sequences, to enhance controllability, whose collection accordingly increases the burden of inference. In this work, we present SparseCtrl to enable flexible structure control with temporally sparse signals, requiring only one or a few inputs, as shown in Figure 1. It incorporates an additional condition encoder to process these sparse signals while leaving the pre-trained T2V model untouched. The proposed approach is compatible with various modalities, including sketches, depth maps, and RGB images, providing more practical control for video generation and promoting applications such as storyboarding, depth rendering, keyframe animation, and interpolation. Extensive experiments demonstrate the generalization of SparseCtrl on both original and personalized T2V generators. Codes and models will be publicly available at [this https URL](https://guoyww.github.io/projects/SparseCtrl).*

SparseCtrl introduces the following checkpoints for controlled text-to-video generation:

- [SparseCtrl Scribble](https://huggingface.co/guoyww/animatediff-sparsectrl-scribble)
- [SparseCtrl RGB](https://huggingface.co/guoyww/animatediff-sparsectrl-rgb)

#### Using SparseCtrl Scribble

```python
import torch

from diffusers import AnimateDiffSparseControlNetPipeline
from diffusers.models import AutoencoderKL, MotionAdapter, SparseControlNetModel
from diffusers.schedulers import DPMSolverMultistepScheduler
from diffusers.utils import export_to_gif, load_image


model_id = "SG161222/Realistic_Vision_V5.1_noVAE"
motion_adapter_id = "guoyww/animatediff-motion-adapter-v1-5-3"
controlnet_id = "guoyww/animatediff-sparsectrl-scribble"
lora_adapter_id = "guoyww/animatediff-motion-lora-v1-5-3"
vae_id = "stabilityai/sd-vae-ft-mse"
device = "cuda"

motion_adapter = MotionAdapter.from_pretrained(motion_adapter_id, torch_dtype=torch.float16).to(device)
controlnet = SparseControlNetModel.from_pretrained(controlnet_id, torch_dtype=torch.float16).to(device)
vae = AutoencoderKL.from_pretrained(vae_id, torch_dtype=torch.float16).to(device)
scheduler = DPMSolverMultistepScheduler.from_pretrained(
    model_id,
    subfolder="scheduler",
    beta_schedule="linear",
    algorithm_type="dpmsolver++",
    use_karras_sigmas=True,
)
pipe = AnimateDiffSparseControlNetPipeline.from_pretrained(
    model_id,
    motion_adapter=motion_adapter,
    controlnet=controlnet,
    vae=vae,
    scheduler=scheduler,
    torch_dtype=torch.float16,
).to(device)
pipe.load_lora_weights(lora_adapter_id, adapter_name="motion_lora")
pipe.fuse_lora(lora_scale=1.0)

prompt = "an aerial view of a cyberpunk city, night time, neon lights, masterpiece, high quality"
negative_prompt = "low quality, worst quality, letterboxed"

image_files = [
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/animatediff-scribble-1.png",
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/animatediff-scribble-2.png",
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/animatediff-scribble-3.png"
]
condition_frame_indices = [0, 8, 15]
conditioning_frames = [load_image(img_file) for img_file in image_files]

video = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=25,
    conditioning_frames=conditioning_frames,
    controlnet_conditioning_scale=1.0,
    controlnet_frame_indices=condition_frame_indices,
    generator=torch.Generator().manual_seed(1337),
).frames[0]
export_to_gif(video, "output.gif")
```

Here are some sample outputs:

<table align="center">
    <tr>
        <center>
          <b>an aerial view of a cyberpunk city, night time, neon lights, masterpiece, high quality</b>
        </center>
    </tr>
    <tr>
        <td>
          <center>
            <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/animatediff-scribble-1.png" alt="scribble-1" />
          </center>
        </td>
        <td>
          <center>
            <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/animatediff-scribble-2.png" alt="scribble-2" />
          </center>
        </td>
        <td>
          <center>
            <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/animatediff-scribble-3.png" alt="scribble-3" />
          </center>
        </td>
    </tr>
    <tr>
        <td colspan=3>
          <center>
            <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/animatediff-sparsectrl-scribble-results.gif" alt="an aerial view of a cyberpunk city, night time, neon lights, masterpiece, high quality" />
          </center>
        </td>
    </tr>
</table>

#### Using SparseCtrl RGB

```python
import torch

from diffusers import AnimateDiffSparseControlNetPipeline
from diffusers.models import AutoencoderKL, MotionAdapter, SparseControlNetModel
from diffusers.schedulers import DPMSolverMultistepScheduler
from diffusers.utils import export_to_gif, load_image


model_id = "SG161222/Realistic_Vision_V5.1_noVAE"
motion_adapter_id = "guoyww/animatediff-motion-adapter-v1-5-3"
controlnet_id = "guoyww/animatediff-sparsectrl-rgb"
lora_adapter_id = "guoyww/animatediff-motion-lora-v1-5-3"
vae_id = "stabilityai/sd-vae-ft-mse"
device = "cuda"

motion_adapter = MotionAdapter.from_pretrained(motion_adapter_id, torch_dtype=torch.float16).to(device)
controlnet = SparseControlNetModel.from_pretrained(controlnet_id, torch_dtype=torch.float16).to(device)
vae = AutoencoderKL.from_pretrained(vae_id, torch_dtype=torch.float16).to(device)
scheduler = DPMSolverMultistepScheduler.from_pretrained(
    model_id,
    subfolder="scheduler",
    beta_schedule="linear",
    algorithm_type="dpmsolver++",
    use_karras_sigmas=True,
)
pipe = AnimateDiffSparseControlNetPipeline.from_pretrained(
    model_id,
    motion_adapter=motion_adapter,
    controlnet=controlnet,
    vae=vae,
    scheduler=scheduler,
    torch_dtype=torch.float16,
).to(device)
pipe.load_lora_weights(lora_adapter_id, adapter_name="motion_lora")

image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/animatediff-firework.png")

video = pipe(
    prompt="closeup face photo of man in black clothes, night city street, bokeh, fireworks in background",
    negative_prompt="low quality, worst quality",
    num_inference_steps=25,
    conditioning_frames=image,
    controlnet_frame_indices=[0],
    controlnet_conditioning_scale=1.0,
    generator=torch.Generator().manual_seed(42),
).frames[0]
export_to_gif(video, "output.gif")
```

Here are some sample outputs:

<table align="center">
    <tr>
        <center>
          <b>closeup face photo of man in black clothes, night city street, bokeh, fireworks in background</b>
        </center>
    </tr>
    <tr>
        <td>
          <center>
            <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/animatediff-firework.png" alt="closeup face photo of man in black clothes, night city street, bokeh, fireworks in background" />
          </center>
        </td>
        <td>
          <center>
            <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/animatediff-sparsectrl-rgb-result.gif" alt="closeup face photo of man in black clothes, night city street, bokeh, fireworks in background" />
          </center>
        </td>
    </tr>
</table>

### AnimateDiffSDXLPipeline

AnimateDiff can also be used with SDXL models. This is currently an experimental feature as only a beta release of the motion adapter checkpoint is available.

```python
import torch
from diffusers.models import MotionAdapter
from diffusers import AnimateDiffSDXLPipeline, DDIMScheduler
from diffusers.utils import export_to_gif

adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-sdxl-beta", torch_dtype=torch.float16)

model_id = "stabilityai/stable-diffusion-xl-base-1.0"
scheduler = DDIMScheduler.from_pretrained(
    model_id,
    subfolder="scheduler",
    clip_sample=False,
    timestep_spacing="linspace",
    beta_schedule="linear",
    steps_offset=1,
)
pipe = AnimateDiffSDXLPipeline.from_pretrained(
    model_id,
    motion_adapter=adapter,
    scheduler=scheduler,
    torch_dtype=torch.float16,
    variant="fp16",
).to("cuda")

# enable memory savings
pipe.enable_vae_slicing()
pipe.enable_vae_tiling()

output = pipe(
    prompt="a panda surfing in the ocean, realistic, high quality",
    negative_prompt="low quality, worst quality",
    num_inference_steps=20,
    guidance_scale=8,
    width=1024,
    height=1024,
    num_frames=16,
)

frames = output.frames[0]
export_to_gif(frames, "animation.gif")
```

### AnimateDiffVideoToVideoPipeline

AnimateDiff can also be used to generate visually similar videos or enable style/character/background or other edits starting from an initial video, allowing you to seamlessly explore creative possibilities.

```python
import imageio
import requests
import torch
from diffusers import AnimateDiffVideoToVideoPipeline, DDIMScheduler, MotionAdapter
from diffusers.utils import export_to_gif
from io import BytesIO
from PIL import Image

# Load the motion adapter
adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2", torch_dtype=torch.float16)
# load SD 1.5 based finetuned model
model_id = "SG161222/Realistic_Vision_V5.1_noVAE"
pipe = AnimateDiffVideoToVideoPipeline.from_pretrained(model_id, motion_adapter=adapter, torch_dtype=torch.float16)
scheduler = DDIMScheduler.from_pretrained(
    model_id,
    subfolder="scheduler",
    clip_sample=False,
    timestep_spacing="linspace",
    beta_schedule="linear",
    steps_offset=1,
)
pipe.scheduler = scheduler

# enable memory savings
pipe.enable_vae_slicing()
pipe.enable_model_cpu_offload()

# helper function to load videos
def load_video(file_path: str):
    images = []

    if file_path.startswith(('http://', 'https://')):
        # If the file_path is a URL
        response = requests.get(file_path)
        response.raise_for_status()
        content = BytesIO(response.content)
        vid = imageio.get_reader(content)
    else:
        # Assuming it's a local file path
        vid = imageio.get_reader(file_path)

    for frame in vid:
        pil_image = Image.fromarray(frame)
        images.append(pil_image)

    return images

video = load_video("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/animatediff-vid2vid-input-1.gif")

output = pipe(
    video = video,
    prompt="panda playing a guitar, on a boat, in the ocean, high quality",
    negative_prompt="bad quality, worse quality",
    guidance_scale=7.5,
    num_inference_steps=25,
    strength=0.5,
    generator=torch.Generator("cpu").manual_seed(42),
)
frames = output.frames[0]
export_to_gif(frames, "animation.gif")
```

Here are some sample outputs:

<table>
    <tr>
      <th align=center>Source Video</th>
      <th align=center>Output Video</th>
    </tr>
    <tr>
        <td align=center>
          raccoon playing a guitar
          <br />
          <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/animatediff-vid2vid-input-1.gif"
              alt="racoon playing a guitar"
              style="width: 300px;" />
        </td>
        <td align=center>
          panda playing a guitar
          <br/>
          <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/animatediff-vid2vid-output-1.gif"
              alt="panda playing a guitar"
              style="width: 300px;" />
        </td>
    </tr>
    <tr>
        <td align=center>
          closeup of margot robbie, fireworks in the background, high quality
          <br />
          <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/animatediff-vid2vid-input-2.gif"
              alt="closeup of margot robbie, fireworks in the background, high quality"
              style="width: 300px;" />
        </td>
        <td align=center>
          closeup of tony stark, robert downey jr, fireworks
          <br/>
          <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/animatediff-vid2vid-output-2.gif"
              alt="closeup of tony stark, robert downey jr, fireworks"
              style="width: 300px;" />
        </td>
    </tr>
</table>



### AnimateDiffVideoToVideoControlNetPipeline

AnimateDiff can be used together with ControlNets to enhance video-to-video generation by allowing for precise control over the output. ControlNet was introduced in [Adding Conditional Control to Text-to-Image Diffusion Models](https://huggingface.co/papers/2302.05543) by Lvmin Zhang, Anyi Rao, and Maneesh Agrawala, and allows you to condition Stable Diffusion with an additional control image to ensure that the spatial information is preserved throughout the video. 

This pipeline allows you to condition your generation both on the original video and on a sequence of control images.

```python
import torch
from PIL import Image
from tqdm.auto import tqdm

from controlnet_aux.processor import OpenposeDetector
from diffusers import AnimateDiffVideoToVideoControlNetPipeline
from diffusers.utils import export_to_gif, load_video
from diffusers import AutoencoderKL, ControlNetModel, MotionAdapter, LCMScheduler

# Load the ControlNet
controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-openpose", torch_dtype=torch.float16)
# Load the motion adapter
motion_adapter = MotionAdapter.from_pretrained("wangfuyun/AnimateLCM")
# Load SD 1.5 based finetuned model
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16)
pipe = AnimateDiffVideoToVideoControlNetPipeline.from_pretrained(
    "SG161222/Realistic_Vision_V5.1_noVAE",
    motion_adapter=motion_adapter,
    controlnet=controlnet,
    vae=vae,
).to(device="cuda", dtype=torch.float16)

# Enable LCM to speed up inference
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config, beta_schedule="linear")
pipe.load_lora_weights("wangfuyun/AnimateLCM", weight_name="AnimateLCM_sd15_t2v_lora.safetensors", adapter_name="lcm-lora")
pipe.set_adapters(["lcm-lora"], [0.8])

video = load_video("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/dance.gif")
video = [frame.convert("RGB") for frame in video]

prompt = "astronaut in space, dancing"
negative_prompt = "bad quality, worst quality, jpeg artifacts, ugly"

# Create controlnet preprocessor
open_pose = OpenposeDetector.from_pretrained("lllyasviel/Annotators").to("cuda")

# Preprocess controlnet images
conditioning_frames = []
for frame in tqdm(video):
    conditioning_frames.append(open_pose(frame))

strength = 0.8
with torch.inference_mode():
    video = pipe(
        video=video,
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=10,
        guidance_scale=2.0,
        controlnet_conditioning_scale=0.75,
        conditioning_frames=conditioning_frames,
        strength=strength,
        generator=torch.Generator().manual_seed(42),
    ).frames[0]

video = [frame.resize(conditioning_frames[0].size) for frame in video]
export_to_gif(video, f"animatediff_vid2vid_controlnet.gif", fps=8)
```

Here are some sample outputs:

<table align="center">
    <tr>
      <th align="center">Source Video</th>
      <th align="center">Output Video</th>
    </tr>
    <tr>
        <td align="center">
          anime girl, dancing
          <br />
          <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/dance.gif" alt="anime girl, dancing" />
        </td>
        <td align="center">
          astronaut in space, dancing
          <br/>
          <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/animatediff_vid2vid_controlnet.gif" alt="astronaut in space, dancing" />
        </td>
    </tr>
</table>

**The lights and composition were transferred from the Source Video.**

## Using Motion LoRAs

Motion LoRAs are a collection of LoRAs that work with the `guoyww/animatediff-motion-adapter-v1-5-2` checkpoint. These LoRAs are responsible for adding specific types of motion to the animations.

```python
import torch
from diffusers import AnimateDiffPipeline, DDIMScheduler, MotionAdapter
from diffusers.utils import export_to_gif

# Load the motion adapter
adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2", torch_dtype=torch.float16)
# load SD 1.5 based finetuned model
model_id = "SG161222/Realistic_Vision_V5.1_noVAE"
pipe = AnimateDiffPipeline.from_pretrained(model_id, motion_adapter=adapter, torch_dtype=torch.float16)
pipe.load_lora_weights(
    "guoyww/animatediff-motion-lora-zoom-out", adapter_name="zoom-out"
)

scheduler = DDIMScheduler.from_pretrained(
    model_id,
    subfolder="scheduler",
    clip_sample=False,
    beta_schedule="linear",
    timestep_spacing="linspace",
    steps_offset=1,
)
pipe.scheduler = scheduler

# enable memory savings
pipe.enable_vae_slicing()
pipe.enable_model_cpu_offload()

output = pipe(
    prompt=(
        "masterpiece, bestquality, highlydetailed, ultradetailed, sunset, "
        "orange sky, warm lighting, fishing boats, ocean waves seagulls, "
        "rippling water, wharf, silhouette, serene atmosphere, dusk, evening glow, "
        "golden hour, coastal landscape, seaside scenery"
    ),
    negative_prompt="bad quality, worse quality",
    num_frames=16,
    guidance_scale=7.5,
    num_inference_steps=25,
    generator=torch.Generator("cpu").manual_seed(42),
)
frames = output.frames[0]
export_to_gif(frames, "animation.gif")
```

<table>
    <tr>
        <td><center>
        masterpiece, bestquality, sunset.
        <br>
        <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/animatediff-zoom-out-lora.gif"
            alt="masterpiece, bestquality, sunset"
            style="width: 300px;" />
        </center></td>
    </tr>
</table>

## Using Motion LoRAs with PEFT

You can also leverage the [PEFT](https://github.com/huggingface/peft) backend to combine Motion LoRA's and create more complex animations.

First install PEFT with

```shell
pip install peft
```

Then you can use the following code to combine Motion LoRAs.

```python
import torch
from diffusers import AnimateDiffPipeline, DDIMScheduler, MotionAdapter
from diffusers.utils import export_to_gif

# Load the motion adapter
adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2", torch_dtype=torch.float16)
# load SD 1.5 based finetuned model
model_id = "SG161222/Realistic_Vision_V5.1_noVAE"
pipe = AnimateDiffPipeline.from_pretrained(model_id, motion_adapter=adapter, torch_dtype=torch.float16)

pipe.load_lora_weights(
    "diffusers/animatediff-motion-lora-zoom-out", adapter_name="zoom-out",
)
pipe.load_lora_weights(
    "diffusers/animatediff-motion-lora-pan-left", adapter_name="pan-left",
)
pipe.set_adapters(["zoom-out", "pan-left"], adapter_weights=[1.0, 1.0])

scheduler = DDIMScheduler.from_pretrained(
    model_id,
    subfolder="scheduler",
    clip_sample=False,
    timestep_spacing="linspace",
    beta_schedule="linear",
    steps_offset=1,
)
pipe.scheduler = scheduler

# enable memory savings
pipe.enable_vae_slicing()
pipe.enable_model_cpu_offload()

output = pipe(
    prompt=(
        "masterpiece, bestquality, highlydetailed, ultradetailed, sunset, "
        "orange sky, warm lighting, fishing boats, ocean waves seagulls, "
        "rippling water, wharf, silhouette, serene atmosphere, dusk, evening glow, "
        "golden hour, coastal landscape, seaside scenery"
    ),
    negative_prompt="bad quality, worse quality",
    num_frames=16,
    guidance_scale=7.5,
    num_inference_steps=25,
    generator=torch.Generator("cpu").manual_seed(42),
)
frames = output.frames[0]
export_to_gif(frames, "animation.gif")
```

<table>
    <tr>
        <td><center>
        masterpiece, bestquality, sunset.
        <br>
        <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/animatediff-zoom-out-pan-left-lora.gif"
            alt="masterpiece, bestquality, sunset"
            style="width: 300px;" />
        </center></td>
    </tr>
</table>

## Using FreeInit

[FreeInit: Bridging Initialization Gap in Video Diffusion Models](https://huggingface.co/papers/2312.07537) by Tianxing Wu, Chenyang Si, Yuming Jiang, Ziqi Huang, Ziwei Liu.

FreeInit is an effective method that improves temporal consistency and overall quality of videos generated using video-diffusion-models without any addition training. It can be applied to AnimateDiff, ModelScope, VideoCrafter and various other video generation models seamlessly at inference time, and works by iteratively refining the latent-initialization noise. More details can be found it the paper.

The following example demonstrates the usage of FreeInit.

```python
import torch
from diffusers import MotionAdapter, AnimateDiffPipeline, DDIMScheduler
from diffusers.utils import export_to_gif

adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2")
model_id = "SG161222/Realistic_Vision_V5.1_noVAE"
pipe = AnimateDiffPipeline.from_pretrained(model_id, motion_adapter=adapter, torch_dtype=torch.float16).to("cuda")
pipe.scheduler = DDIMScheduler.from_pretrained(
    model_id,
    subfolder="scheduler",
    beta_schedule="linear",
    clip_sample=False,
    timestep_spacing="linspace",
    steps_offset=1
)

# enable memory savings
pipe.enable_vae_slicing()
pipe.enable_vae_tiling()

# enable FreeInit
# Refer to the enable_free_init documentation for a full list of configurable parameters
pipe.enable_free_init(method="butterworth", use_fast_sampling=True)

# run inference
output = pipe(
    prompt="a panda playing a guitar, on a boat, in the ocean, high quality",
    negative_prompt="bad quality, worse quality",
    num_frames=16,
    guidance_scale=7.5,
    num_inference_steps=20,
    generator=torch.Generator("cpu").manual_seed(666),
)

# disable FreeInit
pipe.disable_free_init()

frames = output.frames[0]
export_to_gif(frames, "animation.gif")
```

> [!WARNING]
> FreeInit is not really free - the improved quality comes at the cost of extra computation. It requires sampling a few extra times depending on the `num_iters` parameter that is set when enabling it. Setting the `use_fast_sampling` parameter to `True` can improve the overall performance (at the cost of lower quality compared to when `use_fast_sampling=False` but still better results than vanilla video generation models).

> [!TIP]
> Make sure to check out the Schedulers [guide](../../using-diffusers/schedulers) to learn how to explore the tradeoff between scheduler speed and quality, and see the [reuse components across pipelines](../../using-diffusers/loading#reuse-a-pipeline) section to learn how to efficiently load the same components into multiple pipelines.

<table>
    <tr>
      <th align=center>Without FreeInit enabled</th>
      <th align=center>With FreeInit enabled</th>
    </tr>
    <tr>
        <td align=center>
          panda playing a guitar
          <br />
          <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/animatediff-no-freeinit.gif"
              alt="panda playing a guitar"
              style="width: 300px;" />
        </td>
        <td align=center>
          panda playing a guitar
          <br/>
          <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/animatediff-freeinit.gif"
              alt="panda playing a guitar"
              style="width: 300px;" />
        </td>
    </tr>
</table>

## Using AnimateLCM

[AnimateLCM](https://animatelcm.github.io/) is a motion module checkpoint and an [LCM LoRA](https://huggingface.co/docs/diffusers/using-diffusers/inference_with_lcm_lora) that have been created using a consistency learning strategy that decouples the distillation of the image generation priors and the motion generation priors.

```python
import torch
from diffusers import AnimateDiffPipeline, LCMScheduler, MotionAdapter
from diffusers.utils import export_to_gif

adapter = MotionAdapter.from_pretrained("wangfuyun/AnimateLCM")
pipe = AnimateDiffPipeline.from_pretrained("emilianJR/epiCRealism", motion_adapter=adapter)
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config, beta_schedule="linear")

pipe.load_lora_weights("wangfuyun/AnimateLCM", weight_name="sd15_lora_beta.safetensors", adapter_name="lcm-lora")

pipe.enable_vae_slicing()
pipe.enable_model_cpu_offload()

output = pipe(
    prompt="A space rocket with trails of smoke behind it launching into space from the desert, 4k, high resolution",
    negative_prompt="bad quality, worse quality, low resolution",
    num_frames=16,
    guidance_scale=1.5,
    num_inference_steps=6,
    generator=torch.Generator("cpu").manual_seed(0),
)
frames = output.frames[0]
export_to_gif(frames, "animatelcm.gif")
```

<table>
    <tr>
        <td><center>
        A space rocket, 4K.
        <br>
        <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/animatelcm-output.gif"
            alt="A space rocket, 4K"
            style="width: 300px;" />
        </center></td>
    </tr>
</table>

AnimateLCM is also compatible with existing [Motion LoRAs](https://huggingface.co/collections/dn6/animatediff-motion-loras-654cb8ad732b9e3cf4d3c17e).

```python
import torch
from diffusers import AnimateDiffPipeline, LCMScheduler, MotionAdapter
from diffusers.utils import export_to_gif

adapter = MotionAdapter.from_pretrained("wangfuyun/AnimateLCM")
pipe = AnimateDiffPipeline.from_pretrained("emilianJR/epiCRealism", motion_adapter=adapter)
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config, beta_schedule="linear")

pipe.load_lora_weights("wangfuyun/AnimateLCM", weight_name="sd15_lora_beta.safetensors", adapter_name="lcm-lora")
pipe.load_lora_weights("guoyww/animatediff-motion-lora-tilt-up", adapter_name="tilt-up")

pipe.set_adapters(["lcm-lora", "tilt-up"], [1.0, 0.8])
pipe.enable_vae_slicing()
pipe.enable_model_cpu_offload()

output = pipe(
    prompt="A space rocket with trails of smoke behind it launching into space from the desert, 4k, high resolution",
    negative_prompt="bad quality, worse quality, low resolution",
    num_frames=16,
    guidance_scale=1.5,
    num_inference_steps=6,
    generator=torch.Generator("cpu").manual_seed(0),
)
frames = output.frames[0]
export_to_gif(frames, "animatelcm-motion-lora.gif")
```

<table>
    <tr>
        <td><center>
        A space rocket, 4K.
        <br>
        <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/animatelcm-motion-lora.gif"
            alt="A space rocket, 4K"
            style="width: 300px;" />
        </center></td>
    </tr>
</table>

## Using FreeNoise

[FreeNoise: Tuning-Free Longer Video Diffusion via Noise Rescheduling](https://huggingface.co/papers/2310.15169) by Haonan Qiu, Menghan Xia, Yong Zhang, Yingqing He, Xintao Wang, Ying Shan, Ziwei Liu.

FreeNoise is a sampling mechanism that can generate longer videos with short-video generation models by employing noise-rescheduling, temporal attention over sliding windows, and weighted averaging of latent frames. It also can be used with multiple prompts to allow for interpolated video generations. More details are available in the paper.

The currently supported AnimateDiff pipelines that can be used with FreeNoise are:
- [`AnimateDiffPipeline`]
- [`AnimateDiffControlNetPipeline`]
- [`AnimateDiffVideoToVideoPipeline`]
- [`AnimateDiffVideoToVideoControlNetPipeline`]

In order to use FreeNoise, a single line needs to be added to the inference code after loading your pipelines.

```diff
+ pipe.enable_free_noise()
```

After this, either a single prompt could be used, or multiple prompts can be passed as a dictionary of integer-string pairs. The integer keys of the dictionary correspond to the frame index at which the influence of that prompt would be maximum. Each frame index should map to a single string prompt. The prompts for intermediate frame indices, that are not passed in the dictionary, are created by interpolating between the frame prompts that are passed. By default, simple linear interpolation is used. However, you can customize this behaviour with a callback to the `prompt_interpolation_callback` parameter when enabling FreeNoise.

Full example:

```python
import torch
from diffusers import AutoencoderKL, AnimateDiffPipeline, LCMScheduler, MotionAdapter
from diffusers.utils import export_to_video, load_image

# Load pipeline
dtype = torch.float16
motion_adapter = MotionAdapter.from_pretrained("wangfuyun/AnimateLCM", torch_dtype=dtype)
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=dtype)

pipe = AnimateDiffPipeline.from_pretrained("emilianJR/epiCRealism", motion_adapter=motion_adapter, vae=vae, torch_dtype=dtype)
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config, beta_schedule="linear")

pipe.load_lora_weights(
    "wangfuyun/AnimateLCM", weight_name="AnimateLCM_sd15_t2v_lora.safetensors", adapter_name="lcm_lora"
)
pipe.set_adapters(["lcm_lora"], [0.8])

# Enable FreeNoise for long prompt generation
pipe.enable_free_noise(context_length=16, context_stride=4)
pipe.to("cuda")

# Can be a single prompt, or a dictionary with frame timesteps
prompt = {
    0: "A caterpillar on a leaf, high quality, photorealistic",
    40: "A caterpillar transforming into a cocoon, on a leaf, near flowers, photorealistic",
    80: "A cocoon on a leaf, flowers in the background, photorealistic",
    120: "A cocoon maturing and a butterfly being born, flowers and leaves visible in the background, photorealistic",
    160: "A beautiful butterfly, vibrant colors, sitting on a leaf, flowers in the background, photorealistic",
    200: "A beautiful butterfly, flying away in a forest, photorealistic",
    240: "A cyberpunk butterfly, neon lights, glowing",
}
negative_prompt = "bad quality, worst quality, jpeg artifacts"

# Run inference
output = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_frames=256,
    guidance_scale=2.5,
    num_inference_steps=10,
    generator=torch.Generator("cpu").manual_seed(0),
)

# Save video
frames = output.frames[0]
export_to_video(frames, "output.mp4", fps=16)
```

### FreeNoise memory savings

Since FreeNoise processes multiple frames together, there are parts in the modeling where the memory required exceeds that available on normal consumer GPUs. The main memory bottlenecks that we identified are spatial and temporal attention blocks, upsampling and downsampling blocks, resnet blocks and feed-forward layers. Since most of these blocks operate effectively only on the channel/embedding dimension, one can perform chunked inference across the batch dimensions. The batch dimension in AnimateDiff are either spatial (`[B x F, H x W, C]`) or temporal (`B x H x W, F, C`) in nature (note that it may seem counter-intuitive, but the batch dimension here are correct, because spatial blocks process across the `B x F` dimension while the temporal blocks process across the `B x H x W` dimension). We introduce a `SplitInferenceModule` that makes it easier to chunk across any dimension and perform inference. This saves a lot of memory but comes at the cost of requiring more time for inference.

```diff
# Load pipeline and adapters
# ...
+ pipe.enable_free_noise_split_inference()
+ pipe.unet.enable_forward_chunking(16)
```

The call to `pipe.enable_free_noise_split_inference` method accepts two parameters: `spatial_split_size` (defaults to `256`) and `temporal_split_size` (defaults to `16`). These can be configured based on how much VRAM you have available. A lower split size results in lower memory usage but slower inference, whereas a larger split size results in faster inference at the cost of more memory.

## Using `from_single_file` with the MotionAdapter

`diffusers>=0.30.0` supports loading the AnimateDiff checkpoints into the `MotionAdapter` in their original format via `from_single_file`

```python
from diffusers import MotionAdapter

ckpt_path = "https://huggingface.co/Lightricks/LongAnimateDiff/blob/main/lt_long_mm_32_frames.ckpt"

adapter = MotionAdapter.from_single_file(ckpt_path, torch_dtype=torch.float16)
pipe = AnimateDiffPipeline.from_pretrained("emilianJR/epiCRealism", motion_adapter=adapter)
```

## AnimateDiffPipeline

[[autodoc]] AnimateDiffPipeline
  - all
  - __call__

## AnimateDiffControlNetPipeline

[[autodoc]] AnimateDiffControlNetPipeline
  - all
  - __call__

## AnimateDiffSparseControlNetPipeline

[[autodoc]] AnimateDiffSparseControlNetPipeline
  - all
  - __call__

## AnimateDiffSDXLPipeline

[[autodoc]] AnimateDiffSDXLPipeline
  - all
  - __call__

## AnimateDiffVideoToVideoPipeline

[[autodoc]] AnimateDiffVideoToVideoPipeline
  - all
  - __call__

## AnimateDiffVideoToVideoControlNetPipeline

[[autodoc]] AnimateDiffVideoToVideoControlNetPipeline
  - all
  - __call__

## AnimateDiffPipelineOutput

[[autodoc]] pipelines.animatediff.AnimateDiffPipelineOutput
