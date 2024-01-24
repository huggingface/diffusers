<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Text or image-to-video

Driven by the success of text-to-image diffusion models, generative video models are able to generate short clips of video from a text prompt or an initial image. These models extend a pretrained diffusion model to generate videos by adding some type of temporal and/or spatial convolution layer to the architecture. A mixed dataset of images and videos are used to train the model which learns to output a series of video frames based on the text or image conditioning.

This guide will show you how to generate videos with several types of video diffusion models, how to configure their parameters, and how to control video generation.

## Popular models

[Stable Video Diffusions](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid), [AnimateDiff](https://huggingface.co/guoyww/animatediff), [Text2Video-Zero](https://huggingface.co/docs/diffusers/main/en/api/pipelines/text_to_video_zero), and [ModelScopeT2V](https://huggingface.co/ali-vilab/text-to-video-ms-1.7b) are all used for video diffusion. Each model is distinct; for example, AnimateDiff inserts a motion modeling module into a frozen text-to-image model to generate personalized animated images whereas Stable Video Diffusion is entirely pretrained from scratch following a three stage training process to generate short high-quality videos.

### Stable Video Diffusion

Stable Video Diffusion is based on the Stable Diffusion 2.1 model and it is trained on images, then low-resolution videos, and finally a smaller dataset of high-resolution videos. This model generates a short 2-4 second video from an initial image. Learn more details about Stable Video Diffusion such as micro-conditioning in the [Stable Video Diffusion](../using-diffusers/svd) guide!

You can use it by loading the [`StableVideoDiffusionPipeline`].

```py
import torch

from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video

pipeline = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16"
)
pipeline.enable_model_cpu_offload()

image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/svd/rocket.png")
image = image.resize((1024, 576))

generator = torch.manual_seed(42)
frames = pipeline(image, decode_chunk_size=8, generator=generator).frames[0]

export_to_video(frames, "generated.mp4", fps=7)
```

<div class="flex gap-4">
  <div>
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/svd/rocket.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">"initial image of a rocket"</figcaption>
  </div>
  <div>
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/svd/output_rocket.gif"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">"generated video"</figcaption>
  </div>
</div>

### AnimateDiff

AnimateDiff is an adapter model that inserts a motion module into a pretrained diffusion model to animate an image. The adapter is trained on video clips to learn motion which is used to condition the generation process to create a video. It is faster and easier to only train the adapter and it can be loaded into most diffusion models, effectively turning them into "video models".

Start by loading a [`MotionAdapter`].

```py
import torch
from diffusers import AnimateDiffPipeline, DDIMScheduler, MotionAdapter
from diffusers.utils import export_to_gif

adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2", torch_dtype=torch.float16)
```

Then load a finetuned Stable Diffusion model with the [`AnimateDiffPipeline`].

```py
pipeline = AnimateDiffPipeline.from_pretrained("emilianJR/epiCRealism", motion_adapter=adapter, torch_dtype=torch.float16)
scheduler = DDIMScheduler.from_pretrained(
    "digiplay/Photon_v1",
    subfolder="scheduler",
    clip_sample=False,
    timestep_spacing="linspace",
    beta_schedule="linear",
    steps_offset=1,
)
pipeline.scheduler = scheduler
pipeline.enable_vae_slicing()
pipeline.enable_model_cpu_offload()
```

Create a prompt and generate the video.

```py
output = pipeline(
    prompt="A space rocket with trails of smoke behind it launching into space from the desert, 4k, high resolution",
    negative_prompt="bad quality, worse quality, low resolution",
    num_frames=16,
    guidance_scale=7.5,
    num_inference_steps=50,
    generator=torch.Generator("cpu").manual_seed(49),
)
frames = output.frames[0]
export_to_gif(frames, "animation.gif")
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/animatediff.gif"/>
</div>

### Text2Video-Zero

Text2Video-Zero is a zero-shot video model that takes advantage of what an existing diffusion model has learned to generate a video. It adds motion to the latents and uses cross-frame attention to maintain the context and appearance of objects in a frame. This allows Text2Video-Zero to create consistent videos. Text2Video-Zero can also be extended to other tasks like instruction-guided video editing and conditioning a video on pose or edge control.

```py
import torch
import imageio
from diffusers import TextToVideoZeroPipeline

pipeline = TextToVideoZeroPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
).to("cuda")

output = pipeline(
    prompt="A space rocket with trails of smoke behind it launching into space from the desert, 4k, high resolution",
    negative_prompt="low resolution, bad quality, worse quality",
    video_length=16,
    generator=torch.Generator("cpu").manual_seed(24)
).images
output = [(r * 255).astype("uint8") for r in output]
imageio.mimsave("video.mp4", output)
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/t2v-zero.gif"/>
</div>

### ModelscopeT2V

ModelscopeT2V adds spatial and temporal convolutions and attention to a UNet, and it is trained on image-text and video-text datasets to enhance what it learns during training. The model takes a prompt, encodes it and creates text embeddings which are denoised by the UNet, and then decoded by a VQGAN into a video. Load a ModelScopeT2V checkpoint into the [`DiffusionPipeline`] along with a prompt to generate a video.

<Tip>

ModelScopeT2V generates watermarked videos due to the datasets it was trained on. To use a watermark-free model, try the [cerspense/zeroscope_v2_76w](https://huggingface.co/cerspense/zeroscope_v2_576w) model with the [`TextToVideoSDPipeline`] first, and then upscale it's output with the [cerspense/zeroscope_v2_XL](https://huggingface.co/cerspense/zeroscope_v2_XL) checkpoint using the [`Video2VideoSDPipeline`].

</Tip>

```py
```

## Configure model parameters

There are a few important parameters you can configure in the pipeline that'll affect the video generation process and quality. Let's take a closer look at what these parameters do and how changing them affects the output.

### Number of frames

The `num_frames` parameter determines how many video frames are generated per second. A frame is an image that is played in a sequence of other frames to create motion or a video. This affects video length because the pipeline generates a certain number of frames per second (check a pipeline's API reference for the default value). To increase the video duration, you'll need to increase the `num_frames` parameter.

<Tip>

Instead of `num_frames`, the Text2Video-Zero model uses the `video_length` parameter to control how many video frames are generated.

</Tip>

```py
```

### Guidance scale

The `guidance_scale` parameter controls how closely aligned the generated video and text prompt or initial image is. A higher `guidance_scale` value means your generated video is more aligned with the text prompt or initial image, while a lower `guidance_scale` value means your generated video is less aligned which could give the model more "creativity" to interpret the conditioning input.

<Tip>

Stable Video Diffusion uses the `min_guidance_scale` and `max_guidance_scale` parameters for applying guidance to the first and last frames respectively.

</Tip>

```py
```

### Negative prompt

A negative prompt deters the model from generating things you don’t want it to. This parameter is commonly used to improve overall generation quality by removing poor or bad features such as “low resolution” or “bad details”.

```py
```

### Model-specific parameters

There are some pipeline parameters that are unique to each model such as adjusting the motion in a video or adding noise to the initial image.

<hfoptions id="special-parameters">
<hfoption id="Stable Video Diffusion">

Stable Video Diffusion provides additional micro-conditioning for the frame rate with the `fps` parameter and for motion with the `motion_bucket_id` parameter. Together, these parameters allow for adjusting the amount of motion in the generated video. There is also a `noise_aug_strength` parameter that increases the amount of noise added to the initial image. Varying this parameter affects how similar the generated video and initial image are. A higher `noise_aug_strength` also increases the amount of motion. To learn more, read the [Micro-conditioning](../using-diffusers/svd#micro-conditioning) guide.

</hfoption>
<hfoption id="Text2Video-Zero">

Text2Video-Zero computes the amount of motion to apply to each frame from randomly sampled latents. You can use the `motion_field_strength_x` and `motion_field_strength_y` parameters to control the amount of motion to apply to the x and y-axes of the video. The parameters `t0` and `t1` are the timesteps to apply motion to the latents.

</hfoption>
</hfoptions>

## Control video generation

Video generation can be controlled similar to how text-to-image, image-to-image, and inpainting can be controlled with a [`ControlNetModel`]. The only difference is you need to use the [`CrossFrameAttnProcessor`] so each frame can attend to the first frame.

### Text2Video-Zero[[control]]

Text2Video-Zero video generation can be conditioned on pose and edge images for even greater control over a subject's motion in the generated video or to preserve the identity of a subject/object in the video. You can also use Text2Video-Zero with [InstructPix2Pix](../api/pipelines/pix2pix) for editing videos with text.

<hfoptions id="t2v-zero">
<hfoption id="pose control">

Start by downloading a video and extracting the pose images from it.

```py
from huggingface_hub import hf_hub_download
from PIL import Image
import imageio

filename = "__assets__/poses_skeleton_gifs/dance1_corr.mp4"
repo_id = "PAIR/Text2Video-Zero"
video_path = hf_hub_download(repo_type="space", repo_id=repo_id, filename=filename)

reader = imageio.get_reader(video_path, "ffmpeg")
frame_count = 8
pose_images = [Image.fromarray(reader.get_data(i)) for i in range(frame_count)]
```

Load a [`ControlNetModel`] for pose estimation and a checkpoint into the [`StableDiffusionControlNetPipeline`]. Then you'll use the [`CrossFrameAttnProcessor`] for the UNet and ControlNet.

```py
import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers.pipelines.text_to_video_synthesis.pipeline_text_to_video_zero import CrossFrameAttnProcessor

model_id = "runwayml/stable-diffusion-v1-5"
controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-openpose", torch_dtype=torch.float16)
pipeline = StableDiffusionControlNetPipeline.from_pretrained(
    model_id, controlnet=controlnet, torch_dtype=torch.float16
).to("cuda")

pipeline.unet.set_attn_processor(CrossFrameAttnProcessor(batch_size=2))
pipeline.controlnet.set_attn_processor(CrossFrameAttnProcessor(batch_size=2))
```

Fix the latents for all the frames, and then pass your prompt and extracted pose images to the model to generate a video.

```py
latents = torch.randn((1, 4, 64, 64), device="cuda", dtype=torch.float16).repeat(len(pose_images), 1, 1, 1)

prompt = "Darth Vader dancing in a desert"
result = pipeline(prompt=[prompt] * len(pose_images), image=pose_images, latents=latents).images
imageio.mimsave("video.mp4", result, fps=4)
```

</hfoption>
<hfoption id="edge control">

Download a video and extract the edges from it.

```py
from huggingface_hub import hf_hub_download
from PIL import Image
import imageio

filename = "__assets__/poses_skeleton_gifs/dance1_corr.mp4"
repo_id = "PAIR/Text2Video-Zero"
video_path = hf_hub_download(repo_type="space", repo_id=repo_id, filename=filename)

reader = imageio.get_reader(video_path, "ffmpeg")
frame_count = 8
pose_images = [Image.fromarray(reader.get_data(i)) for i in range(frame_count)]
```

Load a [`ControlNetModel`] for canny edge and a checkpoint into the [`StableDiffusionControlNetPipeline`]. Then you'll use the [`CrossFrameAttnProcessor`] for the UNet and ControlNet.

```py
import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers.pipelines.text_to_video_synthesis.pipeline_text_to_video_zero import CrossFrameAttnProcessor

model_id = "runwayml/stable-diffusion-v1-5"
controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
pipeline = StableDiffusionControlNetPipeline.from_pretrained(
    model_id, controlnet=controlnet, torch_dtype=torch.float16
).to("cuda")

pipeline.unet.set_attn_processor(CrossFrameAttnProcessor(batch_size=2))
pipeline.controlnet.set_attn_processor(CrossFrameAttnProcessor(batch_size=2))
```

Fix the latents for all the frames, and then pass your prompt and extracted edge images to the model to generate a video.

```py
latents = torch.randn((1, 4, 64, 64), device="cuda", dtype=torch.float16).repeat(len(pose_images), 1, 1, 1)

prompt = "Darth Vader dancing in a desert"
result = pipeline(prompt=[prompt] * len(pose_images), image=pose_images, latents=latents).images
imageio.mimsave("video.mp4", result, fps=4)
```

</hfoption>
<hfoption id="InstructPix2Pix">

InstructPix2Pix allows you to use text to describe the changes you want to make to the video. Start by downloading and reading a video.

```py
from huggingface_hub import hf_hub_download
from PIL import Image
import imageio

filename = "__assets__/pix2pix video/camel.mp4"
repo_id = "PAIR/Text2Video-Zero"
video_path = hf_hub_download(repo_type="space", repo_id=repo_id, filename=filename)

reader = imageio.get_reader(video_path, "ffmpeg")
frame_count = 8
video = [Image.fromarray(reader.get_data(i)) for i in range(frame_count)]
```

Load the [`StableDiffusionInstructPix2PixPipeline`] and set the [`CrossFrameAttnProcessor`] for the UNet.

```py
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline
from diffusers.pipelines.text_to_video_synthesis.pipeline_text_to_video_zero import CrossFrameAttnProcessor

pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained("timbrooks/instruct-pix2pix", torch_dtype=torch.float16).to("cuda")
pipeline.unet.set_attn_processor(CrossFrameAttnProcessor(batch_size=3))
```

Pass a prompt describing the change you want to apply to the video.

```py
prompt = "make it Van Gogh Starry Night style"
result = pipeline(prompt=[prompt] * len(video), image=video).images
imageio.mimsave("edited_video.mp4", result, fps=4)
```

</hfoption>
</hfoptions>

## Optimize

Video generation requires a lot of memory because you're generating many video frames at once. You can reduce your memory requirements at the expense of some inference speed. Try:

1. model offloading offloads pipeline components that are no longer needed to the CPU
2. feed-forward chunking runs the feed-forward layer in a loop instead of all at once
3. break up the number of frames the VAE has to decode into chunks instead of decoding them all at once

```diff
- pipeline.enable_model_cpu_offload()
- frames = pipeline(image, decode_chunk_size=8, generator=generator).frames[0]
+ pipeline.enable_model_cpu_offload()
+ pipeline.unet.enable_forward_chunking()
+ frames = pipeline(image, decode_chunk_size=2, generator=generator, num_frames=25).frames[0]
```

If memory is not an issue and you want to optimize for speed, try wrapping the UNet with [`torch.compile`](../optimization/torch2.0#torchcompile).

```diff
- pipeline.enable_model_cpu_offload()
+ pipeline.to("cuda")
+ pipeline.unet = torch.compile(pipeline.unet, mode="reduce-overhead", fullgraph=True)
```
