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

# Helios

[Helios: Real Real-Time Long Video Generation Model](https://huggingface.co/papers/2603.04379) from Peking University & ByteDance & etc, by Shenghai Yuan, Yuanyang Yin, Zongjian Li, Xinwei Huang, Xiao Yang, Li Yuan.

*  <u>We introduce Helios, the first 14B video generation model that runs at 17 FPS on a single NVIDIA H100 GPU and supports minute-scale generation while matching a strong baseline in quality.</u> We make breakthroughs along three key dimensions: (1) robustness to long-video drifting without commonly used anti-drift heuristics such as self-forcing, error banks, or keyframe sampling; (2) real-time generation without standard acceleration techniques such as KV-cache, causal masking, or sparse attention; and (3) training without parallelism or sharding frameworks, enabling image-diffusion-scale batch sizes while fitting up to four 14B models within 80 GB of GPU memory. Specifically, Helios is a 14B autoregressive diffusion model with a unified input representation that natively supports T2V, I2V, and V2V tasks. To mitigate drifting in long-video generation, we characterize its typical failure modes and propose simple yet effective training strategies that explicitly simulate drifting during training, while eliminating repetitive motion at its source. For efficiency, we heavily compress the historical and noisy context and reduce the number of sampling steps, yielding computational costs comparable to—or lower than—those of 1.3B video generative models. Moreover, we introduce infrastructure-level optimizations that accelerate both inference and training while reducing memory consumption. Extensive experiments demonstrate that Helios consistently outperforms prior methods on both short- and long-video generation. All the code and models are available at [this https URL](https://pku-yuangroup.github.io/Helios-Page).

The following Helios models are supported in Diffusers:

- [Helios-Base](https://huggingface.co/BestWishYsh/Helios-Base): Best Quality, with v-prediction, standard CFG and custom HeliosScheduler.
- [Helios-Mid](https://huggingface.co/BestWishYsh/Helios-Mid): Intermediate Weight, with v-prediction, CFG-Zero* and custom HeliosScheduler.
- [Helios-Distilled](https://huggingface.co/BestWishYsh/Helios-Distilled): Best Efficiency, with x0-prediction and custom HeliosDMDScheduler.

> [!TIP]
> Click on the Helios models in the right sidebar for more examples of video generation.

### Optimizing Memory and Inference Speed

The example below demonstrates how to generate a video from text optimized for memory or inference speed.

<hfoptions id="optimization">
<hfoption id="memory">

Refer to the [Reduce memory usage](../../optimization/memory) guide for more details about the various memory saving techniques.

The Helios model below requires ~6GB of VRAM.

```py
import torch
from diffusers import AutoModel, HeliosPipeline
from diffusers.hooks.group_offloading import apply_group_offloading
from diffusers.utils import export_to_video

vae = AutoModel.from_pretrained("BestWishYsh/Helios-Base", subfolder="vae", torch_dtype=torch.float32)

# group-offloading
pipeline = HeliosPipeline.from_pretrained(
    "BestWishYsh/Helios-Base",
    vae=vae,
    torch_dtype=torch.bfloat16
)
pipeline.enable_group_offload(
    onload_device=torch.device("cuda"),
    offload_device=torch.device("cpu"),
    offload_type="leaf_level",
    use_stream=True,
    record_stream=True,
)

prompt = """
A vibrant tropical fish swimming gracefully among colorful coral reefs in a clear, turquoise ocean. The fish has bright blue 
and yellow scales with a small, distinctive orange spot on its side, its fins moving fluidly. The coral reefs are alive with 
a variety of marine life, including small schools of colorful fish and sea turtles gliding by. The water is crystal clear, 
allowing for a view of the sandy ocean floor below. The reef itself is adorned with a mix of hard and soft corals in shades 
of red, orange, and green. The photo captures the fish from a slightly elevated angle, emphasizing its lively movements and 
the vivid colors of its surroundings. A close-up shot with dynamic movement.
"""
negative_prompt = """
Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality,
low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured,
misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards
"""

output = pipeline(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_frames=99,
    num_inference_steps=50,
    guidance_scale=5.0,
    generator=torch.Generator("cuda").manual_seed(42),
).frames[0]
export_to_video(output, "helios_base_t2v_output.mp4", fps=24)
```

</hfoption>
<hfoption id="inference speed">

[Compilation](../../optimization/fp16#torchcompile) is slow the first time but subsequent calls to the pipeline are faster. [Attention Backends](../../optimization/attention_backends) such as FlashAttention and SageAttention can significantly increase speed by optimizing the computation of the attention mechanism. [Context Parallelism](../../training/distributed_inference#context-parallelism) splits the input sequence across multiple devices to enable processing of long contexts in parallel, reducing memory pressure and latency. [Caching](../../optimization/cache) may also speed up inference by storing and reusing intermediate outputs.

```py
import torch
from diffusers import AutoModel, HeliosPipeline
from diffusers.utils import export_to_video

vae = AutoModel.from_pretrained("BestWishYsh/Helios-Base", subfolder="vae", torch_dtype=torch.float32)

pipeline = HeliosPipeline.from_pretrained(
    "BestWishYsh/Helios-Base",
    vae=vae,
    torch_dtype=torch.bfloat16
)
pipeline.to("cuda")

# attention backend
# pipeline.transformer.set_attention_backend("flash")
pipeline.transformer.set_attention_backend("_flash_3_hub") # For Hopper GPUs

# torch.compile
torch.backends.cudnn.benchmark = True
pipeline.text_encoder.compile(mode="max-autotune-no-cudagraphs", dynamic=False)
pipeline.vae.compile(mode="max-autotune-no-cudagraphs", dynamic=False)
pipeline.transformer.compile(mode="max-autotune-no-cudagraphs", dynamic=False)

prompt = """
A vibrant tropical fish swimming gracefully among colorful coral reefs in a clear, turquoise ocean. The fish has bright blue 
and yellow scales with a small, distinctive orange spot on its side, its fins moving fluidly. The coral reefs are alive with 
a variety of marine life, including small schools of colorful fish and sea turtles gliding by. The water is crystal clear, 
allowing for a view of the sandy ocean floor below. The reef itself is adorned with a mix of hard and soft corals in shades 
of red, orange, and green. The photo captures the fish from a slightly elevated angle, emphasizing its lively movements and 
the vivid colors of its surroundings. A close-up shot with dynamic movement.
"""
negative_prompt = """
Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality,
low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured,
misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards
"""

output = pipeline(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_frames=99,
    num_inference_steps=50,
    guidance_scale=5.0,
    generator=torch.Generator("cuda").manual_seed(42),
).frames[0]
export_to_video(output, "helios_base_t2v_output.mp4", fps=24)
```

</hfoption>
</hfoptions>


### Generation with Helios-Base

The example below demonstrates how to use Helios-Base to generate video based on text, image or video.

<hfoptions id="Helios-Base usage">
<hfoption id="usage">

```python
import torch
from diffusers import AutoModel, HeliosPipeline
from diffusers.utils import export_to_video, load_video, load_image

vae = AutoModel.from_pretrained("BestWishYsh/Helios-Base", subfolder="vae", torch_dtype=torch.float32)

pipeline = HeliosPipeline.from_pretrained(
    "BestWishYsh/Helios-Base",
    vae=vae,
    torch_dtype=torch.bfloat16
)
pipeline.to("cuda")

negative_prompt = """
Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality,
low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured,
misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards
"""

# For Text-to-Video
prompt = """
A vibrant tropical fish swimming gracefully among colorful coral reefs in a clear, turquoise ocean. The fish has bright blue 
and yellow scales with a small, distinctive orange spot on its side, its fins moving fluidly. The coral reefs are alive with 
a variety of marine life, including small schools of colorful fish and sea turtles gliding by. The water is crystal clear, 
allowing for a view of the sandy ocean floor below. The reef itself is adorned with a mix of hard and soft corals in shades 
of red, orange, and green. The photo captures the fish from a slightly elevated angle, emphasizing its lively movements and 
the vivid colors of its surroundings. A close-up shot with dynamic movement.
"""

output = pipeline(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_frames=99,
    num_inference_steps=50,
    guidance_scale=5.0,
    generator=torch.Generator("cuda").manual_seed(42),
).frames[0]
export_to_video(output, "helios_base_t2v_output.mp4", fps=24)

# For Image-to-Video
prompt = """
A towering emerald wave surges forward, its crest curling with raw power and energy. Sunlight glints off the translucent water, 
illuminating the intricate textures and deep green hues within the wave’s body. A thick spray erupts from the breaking crest, 
casting a misty veil that dances above the churning surface. As the perspective widens, the immense scale of the wave becomes 
apparent, revealing the restless expanse of the ocean stretching beyond. The scene captures the ocean’s untamed beauty and 
relentless force, with every droplet and ripple shimmering in the light. The dynamic motion and vivid colors evoke both awe and 
respect for nature’s might.
"""
image_path = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/helios/wave.jpg"

output = pipeline(
    prompt=prompt,
    negative_prompt=negative_prompt,
    image=load_image(image_path).resize((640, 384)),
    num_frames=99,
    num_inference_steps=50,
    guidance_scale=5.0,
    generator=torch.Generator("cuda").manual_seed(42),
).frames[0]
export_to_video(output, "helios_base_i2v_output.mp4", fps=24)

# For Video-to-Video
prompt = """
A bright yellow Lamborghini Huracn Tecnica speeds along a curving mountain road, surrounded by lush green trees 
under a partly cloudy sky. The car's sleek design and vibrant color stand out against the natural backdrop, 
emphasizing its dynamic movement. The road curves gently, with a guardrail visible on one side, adding depth to 
the scene. The motion blur captures the sense of speed and energy, creating a thrilling and exhilarating atmosphere. 
A front-facing shot from a slightly elevated angle, highlighting the car's aggressive stance and the surrounding greenery.
"""
video_path = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/helios/car.mp4"

output = pipeline(
    prompt=prompt,
    negative_prompt=negative_prompt,
    video=load_video(video_path),
    num_frames=99,
    num_inference_steps=50,
    guidance_scale=5.0,
    generator=torch.Generator("cuda").manual_seed(42),
).frames[0]
export_to_video(output, "helios_base_v2v_output.mp4", fps=24)
```

</hfoption>
</hfoptions>


### Generation with Helios-Mid

The example below demonstrates how to use Helios-Mid to generate video based on text, image or video.

<hfoptions id="Helios-Mid usage">
<hfoption id="usage">

```python
import torch
from diffusers import AutoModel, HeliosPyramidPipeline
from diffusers.utils import export_to_video, load_video, load_image

vae = AutoModel.from_pretrained("BestWishYsh/Helios-Mid", subfolder="vae", torch_dtype=torch.float32)

pipeline = HeliosPyramidPipeline.from_pretrained(
    "BestWishYsh/Helios-Mid",
    vae=vae,
    torch_dtype=torch.bfloat16
)
pipeline.to("cuda")

negative_prompt = """
Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality,
low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured,
misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards
"""

# For Text-to-Video
prompt = """
A vibrant tropical fish swimming gracefully among colorful coral reefs in a clear, turquoise ocean. The fish has bright blue 
and yellow scales with a small, distinctive orange spot on its side, its fins moving fluidly. The coral reefs are alive with 
a variety of marine life, including small schools of colorful fish and sea turtles gliding by. The water is crystal clear, 
allowing for a view of the sandy ocean floor below. The reef itself is adorned with a mix of hard and soft corals in shades 
of red, orange, and green. The photo captures the fish from a slightly elevated angle, emphasizing its lively movements and 
the vivid colors of its surroundings. A close-up shot with dynamic movement.
"""

output = pipeline(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_frames=99,
    pyramid_num_inference_steps_list=[20, 20, 20],
    guidance_scale=5.0,
    use_zero_init=True,
    zero_steps=1,
    generator=torch.Generator("cuda").manual_seed(42),
).frames[0]
export_to_video(output, "helios_pyramid_t2v_output.mp4", fps=24)

# For Image-to-Video
prompt = """
A towering emerald wave surges forward, its crest curling with raw power and energy. Sunlight glints off the translucent water, 
illuminating the intricate textures and deep green hues within the wave’s body. A thick spray erupts from the breaking crest, 
casting a misty veil that dances above the churning surface. As the perspective widens, the immense scale of the wave becomes 
apparent, revealing the restless expanse of the ocean stretching beyond. The scene captures the ocean’s untamed beauty and 
relentless force, with every droplet and ripple shimmering in the light. The dynamic motion and vivid colors evoke both awe and 
respect for nature’s might.
"""
image_path = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/helios/wave.jpg"

output = pipeline(
    prompt=prompt,
    negative_prompt=negative_prompt,
    image=load_image(image_path).resize((640, 384)),
    num_frames=99,
    pyramid_num_inference_steps_list=[20, 20, 20],
    guidance_scale=5.0,
    use_zero_init=True,
    zero_steps=1,
    generator=torch.Generator("cuda").manual_seed(42),
).frames[0]
export_to_video(output, "helios_pyramid_i2v_output.mp4", fps=24)

# For Video-to-Video
prompt = """
A bright yellow Lamborghini Huracn Tecnica speeds along a curving mountain road, surrounded by lush green trees 
under a partly cloudy sky. The car's sleek design and vibrant color stand out against the natural backdrop, 
emphasizing its dynamic movement. The road curves gently, with a guardrail visible on one side, adding depth to 
the scene. The motion blur captures the sense of speed and energy, creating a thrilling and exhilarating atmosphere. 
A front-facing shot from a slightly elevated angle, highlighting the car's aggressive stance and the surrounding greenery.
"""
video_path = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/helios/car.mp4"

output = pipeline(
    prompt=prompt,
    negative_prompt=negative_prompt,
    video=load_video(video_path),
    num_frames=99,
    pyramid_num_inference_steps_list=[20, 20, 20],
    guidance_scale=5.0,
    use_zero_init=True,
    zero_steps=1,
    generator=torch.Generator("cuda").manual_seed(42),
).frames[0]
export_to_video(output, "helios_pyramid_v2v_output.mp4", fps=24)
```

</hfoption>
</hfoptions>


### Generation with Helios-Distilled

The example below demonstrates how to use Helios-Distilled to generate video based on text, image or video.

<hfoptions id="Helios-Distilled usage">
<hfoption id="usage">

```python
import torch
from diffusers import AutoModel, HeliosPyramidPipeline
from diffusers.utils import export_to_video, load_video, load_image

vae = AutoModel.from_pretrained("BestWishYsh/Helios-Distilled", subfolder="vae", torch_dtype=torch.float32)

pipeline = HeliosPyramidPipeline.from_pretrained(
    "BestWishYsh/Helios-Distilled",
    vae=vae,
    torch_dtype=torch.bfloat16
)
pipeline.to("cuda")

negative_prompt = """
Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality,
low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured,
misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards
"""

# For Text-to-Video
prompt = """
A vibrant tropical fish swimming gracefully among colorful coral reefs in a clear, turquoise ocean. The fish has bright blue 
and yellow scales with a small, distinctive orange spot on its side, its fins moving fluidly. The coral reefs are alive with 
a variety of marine life, including small schools of colorful fish and sea turtles gliding by. The water is crystal clear, 
allowing for a view of the sandy ocean floor below. The reef itself is adorned with a mix of hard and soft corals in shades 
of red, orange, and green. The photo captures the fish from a slightly elevated angle, emphasizing its lively movements and 
the vivid colors of its surroundings. A close-up shot with dynamic movement.
"""

output = pipeline(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_frames=240,
    pyramid_num_inference_steps_list=[2, 2, 2],
    guidance_scale=1.0,
    is_amplify_first_chunk=True,
    generator=torch.Generator("cuda").manual_seed(42),
).frames[0]
export_to_video(output, "helios_distilled_t2v_output.mp4", fps=24)

# For Image-to-Video
prompt = """
A towering emerald wave surges forward, its crest curling with raw power and energy. Sunlight glints off the translucent water, 
illuminating the intricate textures and deep green hues within the wave’s body. A thick spray erupts from the breaking crest, 
casting a misty veil that dances above the churning surface. As the perspective widens, the immense scale of the wave becomes 
apparent, revealing the restless expanse of the ocean stretching beyond. The scene captures the ocean’s untamed beauty and 
relentless force, with every droplet and ripple shimmering in the light. The dynamic motion and vivid colors evoke both awe and 
respect for nature’s might.
"""
image_path = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/helios/wave.jpg"

output = pipeline(
    prompt=prompt,
    negative_prompt=negative_prompt,
    image=load_image(image_path).resize((640, 384)),
    num_frames=240,
    pyramid_num_inference_steps_list=[2, 2, 2],
    guidance_scale=1.0,
    is_amplify_first_chunk=True,
    generator=torch.Generator("cuda").manual_seed(42),
).frames[0]
export_to_video(output, "helios_distilled_i2v_output.mp4", fps=24)

# For Video-to-Video
prompt = """
A bright yellow Lamborghini Huracn Tecnica speeds along a curving mountain road, surrounded by lush green trees 
under a partly cloudy sky. The car's sleek design and vibrant color stand out against the natural backdrop, 
emphasizing its dynamic movement. The road curves gently, with a guardrail visible on one side, adding depth to 
the scene. The motion blur captures the sense of speed and energy, creating a thrilling and exhilarating atmosphere. 
A front-facing shot from a slightly elevated angle, highlighting the car's aggressive stance and the surrounding greenery.
"""
video_path = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/helios/car.mp4"

output = pipeline(
    prompt=prompt,
    negative_prompt=negative_prompt,
    video=load_video(video_path),
    num_frames=240,
    pyramid_num_inference_steps_list=[2, 2, 2],
    guidance_scale=1.0,
    is_amplify_first_chunk=True,
    generator=torch.Generator("cuda").manual_seed(42),
).frames[0]
export_to_video(output, "helios_distilled_v2v_output.mp4", fps=24)
```

</hfoption>
</hfoptions>


## Text-to-Video Showcases

<table>
  <tr>
    <th style="text-align: center;">Prompt</th>
    <th style="text-align: center;">Generated Video</th>
  </tr>
  <tr>
    <td><small>A Viking warrior driving a modern city bus filled with passengers. The Viking has long blonde hair tied back, a beard, and is adorned with a fur-lined helmet and armor. He wears a traditional tunic and trousers, but also sports a seatbelt as he focuses on navigating the busy streets. The interior of the bus is typical, with rows of seats occupied by diverse passengers going about their daily routines. The exterior shots show the bustling urban environment, including tall buildings and traffic. Medium shot focusing on the Viking at the wheel, with occasional close-ups of his determined expression.
    </small></td>
    <td>
      <video width="4000" controls>
        <source src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/helios/t2v_showcases1.mp4" type="video/mp4">
      </video>
    </td>
  </tr>
  <tr>
    <td><small>A documentary-style nature photography shot from a camera truck moving to the left, capturing a crab quickly scurrying into its burrow. The crab has a hard, greenish-brown shell and long claws, moving with determined speed across the sandy ground. Its body is slightly arched as it burrows into the sand, leaving a small trail behind. The background shows a shallow beach with scattered rocks and seashells, and the horizon features a gentle curve of the coastline. The photo has a natural and realistic texture, emphasizing the crab's natural movement and the texture of the sand. A close-up shot from a slightly elevated angle.
    </small></td>
    <td>
      <video width="4000" controls>
        <source src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/helios/t2v_showcases2.mp4" type="video/mp4">
      </video>
    </td>
  </tr>
</table>

## Image-to-Video Showcases

<table>
  <tr>
    <th style="text-align: center;">Image</th>
    <th style="text-align: center;">Prompt</th>
    <th style="text-align: center;">Generated Video</th>
  </tr>
  <tr>
    <td><img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/helios/i2v_showcases1.jpg" style="height: auto; width: 300px;"></td>
    <td><small>A sleek red Kia car speeds along a rural road under a cloudy sky, its modern design and dynamic movement emphasized by the blurred motion of the surrounding fields and trees stretching into the distance. The car's glossy exterior reflects the overcast sky, highlighting its aerodynamic shape and sporty stance. The license plate reads "KIA 626," and the vehicle's headlights are on, adding to the sense of motion and energy. The road curves gently, with the car positioned slightly off-center, creating a sense of forward momentum. A dynamic front three-quarter view captures the car's powerful presence against the serene backdrop of rolling hills and scattered trees.
    </small></td>
    <td>
      <video width="2000" controls>
        <source src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/helios/i2v_showcases1.mp4" type="video/mp4">
      </video>
    </td>
  </tr>
  <tr>
    <td><img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/helios/i2v_showcases2.jpg" style="height: auto; width: 300px;"></td>
    <td><small>A close-up captures a fluffy orange cat with striking green eyes and white whiskers, gazing intently towards the camera. The cat's fur is soft and well-groomed, with a mix of warm orange and cream tones. Its large, expressive eyes are a vivid green, reflecting curiosity and alertness. The cat's nose is small and pink, and its mouth is slightly open, revealing a hint of its pink tongue. The background is softly blurred, suggesting a cozy indoor setting with neutral tones. The photo has a shallow depth of field, focusing sharply on the cat's face while the background remains out of focus. A close-up shot from a slightly elevated perspective.
    </small></td>
    <td>
      <video width="2000" controls>
        <source src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/helios/i2v_showcases2.mp4" type="video/mp4">
      </video>
    </td>
  </tr>
</table>

## Interactive-Video Showcases

<table>
  <tr>
    <th style="text-align: center;">Prompt</th>
    <th style="text-align: center;">Generated Video</th>
  </tr>
  <tr>
    <td><small>The prompt can be found <a href="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/helios/interactive_showcases1.txt">here</a></small></td>
    <td>
      <video width="680" controls>
        <source src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/helios/interactive_showcases1.mp4" type="video/mp4">
      </video>
    </td>
  </tr>
  <tr>
    <td><small>The prompt can be found <a href="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/helios/interactive_showcases2.txt">here</a></small></td>
    <td>
      <video width="680" controls>
        <source src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/helios/interactive_showcases2.mp4" type="video/mp4">
      </video>
    </td>
  </tr>
</table>

## Resources

Learn more about Helios with the following resources.
- Watch [video1](https://www.youtube.com/watch?v=vd_AgHtOUFQ) and [video2](https://www.youtube.com/watch?v=1GeIU2Dn7UY) for a demonstration of Helios's key features.
- The research paper, [Helios: Real Real-Time Long Video Generation Model](https://huggingface.co/papers/2603.04379) for more details.

## HeliosPipeline

[[autodoc]] HeliosPipeline

  - all
  - __call__

## HeliosPyramidPipeline

[[autodoc]] HeliosPyramidPipeline

  - all
  - __call__

## HeliosPipelineOutput

[[autodoc]] pipelines.helios.pipeline_output.HeliosPipelineOutput
