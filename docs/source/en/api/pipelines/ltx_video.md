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
    <img alt="MPS" src="https://img.shields.io/badge/MPS-000000?style=flat&logo=apple&logoColor=white%22">
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
from diffusers import LTXPipeline, AutoModel
from diffusers.hooks import apply_group_offloading
from diffusers.utils import export_to_video

# fp8 layerwise weight-casting
transformer = AutoModel.from_pretrained(
    "Lightricks/LTX-Video",
    subfolder="transformer",
    torch_dtype=torch.bfloat16
)
transformer.enable_layerwise_casting(
    storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16
)

pipeline = LTXPipeline.from_pretrained("Lightricks/LTX-Video", transformer=transformer, torch_dtype=torch.bfloat16)

# group-offloading
onload_device = torch.device("cuda")
offload_device = torch.device("cpu")
pipeline.transformer.enable_group_offload(onload_device=onload_device, offload_device=offload_device, offload_type="leaf_level", use_stream=True)
apply_group_offloading(pipeline.text_encoder, onload_device=onload_device, offload_type="block_level", num_blocks_per_group=2)
apply_group_offloading(pipeline.vae, onload_device=onload_device, offload_type="leaf_level")

prompt = """
A woman with long brown hair and light skin smiles at another woman with long blonde hair.
The woman with brown hair wears a black jacket and has a small, barely noticeable mole on her right cheek.
The camera angle is a close-up, focused on the woman with brown hair's face. The lighting is warm and 
natural, likely from the setting sun, casting a soft glow on the scene. The scene appears to be real-life footage
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

[Compilation](../../optimization/fp16#torchcompile) is slow the first time but subsequent calls to the pipeline are faster. [Caching](../../optimization/cache) may also speed up inference by storing and reusing intermediate outputs.

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
A woman with long brown hair and light skin smiles at another woman with long blonde hair.
The woman with brown hair wears a black jacket and has a small, barely noticeable mole on her right cheek.
The camera angle is a close-up, focused on the woman with brown hair's face. The lighting is warm and 
natural, likely from the setting sun, casting a soft glow on the scene. The scene appears to be real-life footage
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

- Refer to the following recommended settings for generation from the [LTX-Video](https://github.com/Lightricks/LTX-Video) repository.

  - The recommended dtype for the transformer, VAE, and text encoder is `torch.bfloat16`. The VAE and text encoder can also be `torch.float32` or `torch.float16`.
  - For guidance-distilled variants of LTX-Video, set `guidance_scale` to `1.0`. The `guidance_scale` for any other model should be set higher, like `5.0`, for good generation quality.
  - For timestep-aware VAE variants (LTX-Video 0.9.1 and above), set `decode_timestep` to `0.05` and `image_cond_noise_scale` to `0.025`.
  - For variants that support interpolation between multiple conditioning images and videos (LTX-Video 0.9.5 and above), use similar images and videos for the best results. Divergence from the conditioning inputs may lead to abrupt transitionts in the generated video.

- LTX-Video 0.9.7 includes a spatial latent upscaler and a 13B parameter transformer. During inference, a low resolution video is quickly generated first and then upscaled and refined.

  <details>
  <summary>Show example code</summary>

  ```py
  import torch
  from diffusers import LTXConditionPipeline, LTXLatentUpsamplePipeline
  from diffusers.pipelines.ltx.pipeline_ltx_condition import LTXVideoCondition
  from diffusers.utils import export_to_video, load_video

  pipeline = LTXConditionPipeline.from_pretrained("Lightricks/LTX-Video-0.9.7-dev", torch_dtype=torch.bfloat16)
  pipeline_upsample = LTXLatentUpsamplePipeline.from_pretrained("Lightricks/ltxv-spatial-upscaler-0.9.7", vae=pipeline.vae, torch_dtype=torch.bfloat16)
  pipeline.to("cuda")
  pipe_upsample.to("cuda")
  pipeline.vae.enable_tiling()

  def round_to_nearest_resolution_acceptable_by_vae(height, width):
      height = height - (height % pipeline.vae_temporal_compression_ratio)
      width = width - (width % pipeline.vae_temporal_compression_ratio)
      return height, width

  video = load_video(
      "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cosmos/cosmos-video2world-input-vid.mp4"
  )[:21]  # only use the first 21 frames as conditioning
  condition1 = LTXVideoCondition(video=video, frame_index=0)

  prompt = """
  The video depicts a winding mountain road covered in snow, with a single vehicle 
  traveling along it. The road is flanked by steep, rocky cliffs and sparse vegetation. 
  The landscape is characterized by rugged terrain and a river visible in the distance. 
  The scene captures the solitude and beauty of a winter drive through a mountainous region.
  """
  negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted"
  expected_height, expected_width = 768, 1152
  downscale_factor = 2 / 3
  num_frames = 161

  # 1. Generate video at smaller resolution
  # Text-only conditioning is also supported without the need to pass `conditions`
  downscaled_height, downscaled_width = int(expected_height * downscale_factor), int(expected_width * downscale_factor)
  downscaled_height, downscaled_width = round_to_nearest_resolution_acceptable_by_vae(downscaled_height, downscaled_width)
  latents = pipeline(
      conditions=[condition1],
      prompt=prompt,
      negative_prompt=negative_prompt,
      width=downscaled_width,
      height=downscaled_height,
      num_frames=num_frames,
      num_inference_steps=30,
      decode_timestep=0.05,
      decode_noise_scale=0.025,
      image_cond_noise_scale=0.0,
      guidance_scale=5.0,
      guidance_rescale=0.7,
      generator=torch.Generator().manual_seed(0),
      output_type="latent",
  ).frames

  # 2. Upscale generated video using latent upsampler with fewer inference steps
  # The available latent upsampler upscales the height/width by 2x
  upscaled_height, upscaled_width = downscaled_height * 2, downscaled_width * 2
  upscaled_latents = pipe_upsample(
      latents=latents,
      output_type="latent"
  ).frames

  # 3. Denoise the upscaled video with few steps to improve texture (optional, but recommended)
  video = pipeline(
      conditions=[condition1],
      prompt=prompt,
      negative_prompt=negative_prompt,
      width=upscaled_width,
      height=upscaled_height,
      num_frames=num_frames,
      denoise_strength=0.4,  # Effectively, 4 inference steps out of 10
      num_inference_steps=10,
      latents=upscaled_latents,
      decode_timestep=0.05,
      decode_noise_scale=0.025,
      image_cond_noise_scale=0.0,
      guidance_scale=5.0,
      guidance_rescale=0.7,
      generator=torch.Generator().manual_seed(0),
      output_type="pil",
  ).frames[0]

  # 4. Downscale the video to the expected resolution
  video = [frame.resize((expected_width, expected_height)) for frame in video]

  export_to_video(video, "output.mp4", fps=24)
  ```

  </details>

- LTX-Video 0.9.7 distilled model is guidance and timestep-distilled to speedup generation. It requires `guidance_scale` to be set to `1.0` and `num_inference_steps` should be set between `4` and `10` for good generation quality. You should also use the following custom timesteps for the best results.

  - Base model inference to prepare for upscaling: `[1000, 993, 987, 981, 975, 909, 725, 0.03]`.
  - Upscaling: `[1000, 909, 725, 421, 0]`.

  <details>
  <summary>Show example code</summary>

  ```py
  import torch
  from diffusers import LTXConditionPipeline, LTXLatentUpsamplePipeline
  from diffusers.pipelines.ltx.pipeline_ltx_condition import LTXVideoCondition
  from diffusers.utils import export_to_video, load_video

  pipeline = LTXConditionPipeline.from_pretrained("Lightricks/LTX-Video-0.9.7-distilled", torch_dtype=torch.bfloat16)
  pipe_upsample = LTXLatentUpsamplePipeline.from_pretrained("Lightricks/ltxv-spatial-upscaler-0.9.7", vae=pipeline.vae, torch_dtype=torch.bfloat16)
  pipeline.to("cuda")
  pipe_upsample.to("cuda")
  pipeline.vae.enable_tiling()

  def round_to_nearest_resolution_acceptable_by_vae(height, width):
      height = height - (height % pipeline.vae_spatial_compression_ratio)
      width = width - (width % pipeline.vae_spatial_compression_ratio)
      return height, width

  prompt = """
  artistic anatomical 3d render, utlra quality, human half full male body with transparent 
  skin revealing structure instead of organs, muscular, intricate creative patterns, 
  monochromatic with backlighting, lightning mesh, scientific concept art, blending biology 
  with botany, surreal and ethereal quality, unreal engine 5, ray tracing, ultra realistic, 
  16K UHD, rich details. camera zooms out in a rotating fashion
  """
  negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted"
  expected_height, expected_width = 768, 1152
  downscale_factor = 2 / 3
  num_frames = 161

  # 1. Generate video at smaller resolution
  downscaled_height, downscaled_width = int(expected_height * downscale_factor), int(expected_width * downscale_factor)
  downscaled_height, downscaled_width = round_to_nearest_resolution_acceptable_by_vae(downscaled_height, downscaled_width)
  latents = pipeline(
      prompt=prompt,
      negative_prompt=negative_prompt,
      width=downscaled_width,
      height=downscaled_height,
      num_frames=num_frames,
      timesteps=[1000, 993, 987, 981, 975, 909, 725, 0.03],
      decode_timestep=0.05,
      decode_noise_scale=0.025,
      image_cond_noise_scale=0.0,
      guidance_scale=1.0,
      guidance_rescale=0.7,
      generator=torch.Generator().manual_seed(0),
      output_type="latent",
  ).frames

  # 2. Upscale generated video using latent upsampler with fewer inference steps
  # The available latent upsampler upscales the height/width by 2x
  upscaled_height, upscaled_width = downscaled_height * 2, downscaled_width * 2
  upscaled_latents = pipe_upsample(
      latents=latents,
      adain_factor=1.0,
      output_type="latent"
  ).frames

  # 3. Denoise the upscaled video with few steps to improve texture (optional, but recommended)
  video = pipeline(
      prompt=prompt,
      negative_prompt=negative_prompt,
      width=upscaled_width,
      height=upscaled_height,
      num_frames=num_frames,
      denoise_strength=0.999,  # Effectively, 4 inference steps out of 5
      timesteps=[1000, 909, 725, 421, 0],
      latents=upscaled_latents,
      decode_timestep=0.05,
      decode_noise_scale=0.025,
      image_cond_noise_scale=0.0,
      guidance_scale=1.0,
      guidance_rescale=0.7,
      generator=torch.Generator().manual_seed(0),
      output_type="pil",
  ).frames[0]

  # 4. Downscale the video to the expected resolution
  video = [frame.resize((expected_width, expected_height)) for frame in video]

  export_to_video(video, "output.mp4", fps=24)
  ```

  </details>

- LTX-Video 0.9.8 distilled model is similar to the 0.9.7 variant. It is guidance and timestep-distilled, and similar inference code can be used as above. An improvement of this version is that it supports generating very long videos. Additionally, it supports using tone mapping to improve the quality of the generated video using the `tone_map_compression_ratio` parameter. The default value of `0.6` is recommended.

  <details>
  <summary>Show example code</summary>
  
  ```python
  import torch
  from diffusers import LTXConditionPipeline, LTXLatentUpsamplePipeline
  from diffusers.pipelines.ltx.pipeline_ltx_condition import LTXVideoCondition
  from diffusers.pipelines.ltx.modeling_latent_upsampler import LTXLatentUpsamplerModel
  from diffusers.utils import export_to_video, load_video

  pipeline = LTXConditionPipeline.from_pretrained("Lightricks/LTX-Video-0.9.8-13B-distilled", torch_dtype=torch.bfloat16)
  # TODO: Update the checkpoint here once updated in LTX org
  upsampler = LTXLatentUpsamplerModel.from_pretrained("a-r-r-o-w/LTX-0.9.8-Latent-Upsampler", torch_dtype=torch.bfloat16)
  pipe_upsample = LTXLatentUpsamplePipeline(vae=pipeline.vae, latent_upsampler=upsampler).to(torch.bfloat16)
  pipeline.to("cuda")
  pipe_upsample.to("cuda")
  pipeline.vae.enable_tiling()

  def round_to_nearest_resolution_acceptable_by_vae(height, width):
      height = height - (height % pipeline.vae_spatial_compression_ratio)
      width = width - (width % pipeline.vae_spatial_compression_ratio)
      return height, width

  prompt = """The camera pans over a snow-covered mountain range, revealing a vast expanse of snow-capped peaks and valleys.The mountains are covered in a thick layer of snow, with some areas appearing almost white while others have a slightly darker, almost grayish hue. The peaks are jagged and irregular, with some rising sharply into the sky while others are more rounded. The valleys are deep and narrow, with steep slopes that are also covered in snow. The trees in the foreground are mostly bare, with only a few leaves remaining on their branches. The sky is overcast, with thick clouds obscuring the sun. The overall impression is one of peace and tranquility, with the snow-covered mountains standing as a testament to the power and beauty of nature."""
  # prompt = """A woman walks away from a white Jeep parked on a city street at night, then ascends a staircase and knocks on a door. The woman, wearing a dark jacket and jeans, walks away from the Jeep parked on the left side of the street, her back to the camera; she walks at a steady pace, her arms swinging slightly by her sides; the street is dimly lit, with streetlights casting pools of light on the wet pavement; a man in a dark jacket and jeans walks past the Jeep in the opposite direction; the camera follows the woman from behind as she walks up a set of stairs towards a building with a green door; she reaches the top of the stairs and turns left, continuing to walk towards the building; she reaches the door and knocks on it with her right hand; the camera remains stationary, focused on the doorway; the scene is captured in real-life footage."""
  negative_prompt = "bright colors, symbols, graffiti, watermarks, worst quality, inconsistent motion, blurry, jittery, distorted"
  expected_height, expected_width = 480, 832
  downscale_factor = 2 / 3
  # num_frames = 161
  num_frames = 361

  # 1. Generate video at smaller resolution
  downscaled_height, downscaled_width = int(expected_height * downscale_factor), int(expected_width * downscale_factor)
  downscaled_height, downscaled_width = round_to_nearest_resolution_acceptable_by_vae(downscaled_height, downscaled_width)
  latents = pipeline(
      prompt=prompt,
      negative_prompt=negative_prompt,
      width=downscaled_width,
      height=downscaled_height,
      num_frames=num_frames,
      timesteps=[1000, 993, 987, 981, 975, 909, 725, 0.03],
      decode_timestep=0.05,
      decode_noise_scale=0.025,
      image_cond_noise_scale=0.0,
      guidance_scale=1.0,
      guidance_rescale=0.7,
      generator=torch.Generator().manual_seed(0),
      output_type="latent",
  ).frames

  # 2. Upscale generated video using latent upsampler with fewer inference steps
  # The available latent upsampler upscales the height/width by 2x
  upscaled_height, upscaled_width = downscaled_height * 2, downscaled_width * 2
  upscaled_latents = pipe_upsample(
      latents=latents,
      adain_factor=1.0,
      tone_map_compression_ratio=0.6,
      output_type="latent"
  ).frames

  # 3. Denoise the upscaled video with few steps to improve texture (optional, but recommended)
  video = pipeline(
      prompt=prompt,
      negative_prompt=negative_prompt,
      width=upscaled_width,
      height=upscaled_height,
      num_frames=num_frames,
      denoise_strength=0.999,  # Effectively, 4 inference steps out of 5
      timesteps=[1000, 909, 725, 421, 0],
      latents=upscaled_latents,
      decode_timestep=0.05,
      decode_noise_scale=0.025,
      image_cond_noise_scale=0.0,
      guidance_scale=1.0,
      guidance_rescale=0.7,
      generator=torch.Generator().manual_seed(0),
      output_type="pil",
  ).frames[0]

  # 4. Downscale the video to the expected resolution
  video = [frame.resize((expected_width, expected_height)) for frame in video]

  export_to_video(video, "output.mp4", fps=24)
  ```

  </details>

- LTX-Video supports LoRAs with [`~loaders.LTXVideoLoraLoaderMixin.load_lora_weights`].

  <details>
  <summary>Show example code</summary>

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
  prompt = "CAKEIFY a person using a knife to cut a cake shaped like a Pikachu plushie"
  image = load_image("https://huggingface.co/Lightricks/LTX-Video-Cakeify-LoRA/resolve/main/assets/images/pikachu.png")

  video = pipeline(
      prompt=prompt,
      image=image,
      width=576,
      height=576,
      num_frames=161,
      decode_timestep=0.03,
      decode_noise_scale=0.025,
      num_inference_steps=50,
  ).frames[0]
  export_to_video(video, "output.mp4", fps=26)
  ```

  </details>

- LTX-Video supports loading from single files, such as [GGUF checkpoints](../../quantization/gguf), with [`loaders.FromOriginalModelMixin.from_single_file`] or [`loaders.FromSingleFileMixin.from_single_file`].

  <details>
  <summary>Show example code</summary>

  ```py
  import torch
  from diffusers.utils import export_to_video
  from diffusers import LTXPipeline, AutoModel, GGUFQuantizationConfig

  transformer = AutoModel.from_single_file(
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

  </details>

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
