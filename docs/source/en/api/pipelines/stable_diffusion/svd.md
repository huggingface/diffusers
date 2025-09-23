<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Stable Video Diffusion

Stable Video Diffusion was proposed in [Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets](https://hf.co/papers/2311.15127) by Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, Varun Jampani, Robin Rombach.

The abstract from the paper is:

*We present Stable Video Diffusion - a latent video diffusion model for high-resolution, state-of-the-art text-to-video and image-to-video generation. Recently, latent diffusion models trained for 2D image synthesis have been turned into generative video models by inserting temporal layers and finetuning them on small, high-quality video datasets. However, training methods in the literature vary widely, and the field has yet to agree on a unified strategy for curating video data. In this paper, we identify and evaluate three different stages for successful training of video LDMs: text-to-image pretraining, video pretraining, and high-quality video finetuning. Furthermore, we demonstrate the necessity of a well-curated pretraining dataset for generating high-quality videos and present a systematic curation process to train a strong base model, including captioning and filtering strategies. We then explore the impact of finetuning our base model on high-quality data and train a text-to-video model that is competitive with closed-source video generation. We also show that our base model provides a powerful motion representation for downstream tasks such as image-to-video generation and adaptability to camera motion-specific LoRA modules. Finally, we demonstrate that our model provides a strong multi-view 3D-prior and can serve as a base to finetune a multi-view diffusion model that jointly generates multiple views of objects in a feedforward fashion, outperforming image-based methods at a fraction of their compute budget. We release code and model weights at this https URL.*

<Tip>

To learn how to use Stable Video Diffusion, take a look at the [Stable Video Diffusion](../../../using-diffusers/svd) guide.

<br>

Check out the [Stability AI](https://huggingface.co/stabilityai) Hub organization for the [base](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid) and [extended frame](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt) checkpoints!

</Tip>

The are two variants of this model, [SVD](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid) and [SVD-XT](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt). The SVD checkpoint is trained to generate 14 frames and the SVD-XT checkpoint is further finetuned to generate 25 frames.

You'll use the SVD-XT checkpoint for this guide.

```python
import torch

from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video

pipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16"
)
pipe.enable_model_cpu_offload()

# Load the conditioning image
image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/svd/rocket.png")
image = image.resize((1024, 576))

generator = torch.manual_seed(42)
frames = pipe(image, decode_chunk_size=8, generator=generator).frames[0]

export_to_video(frames, "generated.mp4", fps=7)
```

<div class="flex gap-4">
  <div>
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/svd/rocket.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">"source image of a rocket"</figcaption>
  </div>
  <div>
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/svd/output_rocket.gif"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">"generated video from source image"</figcaption>
  </div>
</div>

## torch.compile

You can gain a 20-25% speedup at the expense of slightly increased memory by [compiling](../optimization/fp16#torchcompile) the UNet.

```diff
- pipe.enable_model_cpu_offload()
+ pipe.to("cuda")
+ pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
```

## Reduce memory usage

Video generation is very memory intensive because you're essentially generating `num_frames` all at once, similar to text-to-image generation with a high batch size. To reduce the memory requirement, there are multiple options that trade-off inference speed for lower memory requirement:

- enable model offloading: each component of the pipeline is offloaded to the CPU once it's not needed anymore.
- enable feed-forward chunking: the feed-forward layer runs in a loop instead of running a single feed-forward with a huge batch size.
- reduce `decode_chunk_size`: the VAE decodes frames in chunks instead of decoding them all together. Setting `decode_chunk_size=1` decodes one frame at a time and uses the least amount of memory (we recommend adjusting this value based on your GPU memory) but the video might have some flickering.

```diff
- pipe.enable_model_cpu_offload()
- frames = pipe(image, decode_chunk_size=8, generator=generator).frames[0]
+ pipe.enable_model_cpu_offload()
+ pipe.unet.enable_forward_chunking()
+ frames = pipe(image, decode_chunk_size=2, generator=generator, num_frames=25).frames[0]
```

Using all these tricks together should lower the memory requirement to less than 8GB VRAM.

## Micro-conditioning

Stable Diffusion Video also accepts micro-conditioning, in addition to the conditioning image, which allows more control over the generated video:

- `fps`: the frames per second of the generated video.
- `motion_bucket_id`: the motion bucket id to use for the generated video. This can be used to control the motion of the generated video. Increasing the motion bucket id increases the motion of the generated video.
- `noise_aug_strength`: the amount of noise added to the conditioning image. The higher the values the less the video resembles the conditioning image. Increasing this value also increases the motion of the generated video.

For example, to generate a video with more motion, use the `motion_bucket_id` and `noise_aug_strength` micro-conditioning parameters:

```python
import torch

from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video

pipe = StableVideoDiffusionPipeline.from_pretrained(
  "stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16"
)
pipe.enable_model_cpu_offload()

# Load the conditioning image
image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/svd/rocket.png")
image = image.resize((1024, 576))

generator = torch.manual_seed(42)
frames = pipe(image, decode_chunk_size=8, generator=generator, motion_bucket_id=180, noise_aug_strength=0.1).frames[0]
export_to_video(frames, "generated.mp4", fps=7)
```

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/svd/output_rocket_with_conditions.gif)

## Tips

Video generation is memory-intensive and one way to reduce your memory usage is to set `enable_forward_chunking` on the pipeline's UNet so you don't run the entire feedforward layer at once. Breaking it up into chunks in a loop is more efficient.

Check out the [Text or image-to-video](text-img2vid) guide for more details about how certain parameters can affect video generation and how to optimize inference by reducing memory usage.

## StableVideoDiffusionPipeline

[[autodoc]] StableVideoDiffusionPipeline

## StableVideoDiffusionPipelineOutput

[[autodoc]] pipelines.stable_video_diffusion.StableVideoDiffusionPipelineOutput
