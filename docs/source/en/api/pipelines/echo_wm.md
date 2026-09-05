<!-- Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License. -->

# Echo-WM

[Echo-WM](https://huggingface.co/Echo-Team/Echo-WM) is an omnimodal world model for enterable generative media. Given
an initial image, a text description, and a camera-action program, it jointly generates video and synchronized audio
while responding to the requested camera motion. It supports first-person and third-person scenes with the same
camera-control interface.

For more details, refer to the [project page](https://echo-team-joy-future-academy-jd.github.io/Echo-1.5-Page/wm/),
[paper](https://huggingface.co/papers/2608.23189), and
[original codebase](https://github.com/jd-opensource/JoyAI-Echo/tree/main/echo_wm).

> [!IMPORTANT]
> The released checkpoints are intended for academic research and non-commercial use under the
> [LTX-2 Community License](https://github.com/jd-opensource/JoyAI-Echo/blob/main/LICENSE).

## Available checkpoints

| Checkpoint | Description |
|---|---|
| [Echo-WM Base](https://huggingface.co/Echo-Team/Echo-WM-Base-Diffusers) | Bidirectional audio-video generation with a 30-step diffusion loop for higher quality. |
| [Echo-WM Flash](https://huggingface.co/Echo-Team/Echo-WM-Flash-Diffusers) | Guidance-distilled, four-step autoregressive generation with a bounded KV cache. |

Echo-WM is available as a [Modular Diffusers](../../modular_diffusers/overview) pipeline. Load both checkpoints with
[`ModularPipeline`]; the checkpoint's `modular_model_index.json` automatically selects either
[`EchoWMModularPipeline`] or [`EchoWMFlashModularPipeline`].

## Usage

Install Diffusers and the required runtime dependencies:

```bash
pip install -U diffusers transformers accelerate av
```

### Echo-WM Base

Base supports positive and negative prompts and uses 30 denoising steps by default. Structured prompts with
Environment, Character, Style, Perspective, Sounds, and Speech fields generally provide the most control.
The example below matches the [Base model card](https://huggingface.co/Echo-Team/Echo-WM-Base-Diffusers), using the
official `wm_cases/0010` input image and camera trajectory.

```py
import torch
from diffusers import ComponentsManager, ModularPipeline
from diffusers.utils import encode_video, load_image

components_manager = ComponentsManager()
components_manager.enable_auto_cpu_offload(device="cuda")
pipe = ModularPipeline.from_pretrained(
    "Echo-Team/Echo-WM-Base-Diffusers", components_manager=components_manager
)
pipe.load_components(dtype=torch.bfloat16)

image = load_image(
    "https://raw.githubusercontent.com/jd-opensource/JoyAI-Echo/main/echo_wm/examples/wm_cases/0010/input.png"
)
prompt = """Environment: A still teal pool fills a fantasy canyon of pale limestone, walled by weathered white rock and umbrella-crowned trees with dense blue-green canopies, mossy boulders, and clumps of small golden flowers. A monumental natural arch opens on the far bank, framing a white spired palace and the shallow steps that descend to the water. The cliffs, arch, and palace all invert cleanly in the mirror-flat surface.

Character: A solitary adventurer stands on the pale stone shore in a teal cloak and neck wrap over leather travel gear, holding a tall walking staff and facing the arch across the water. Keep the character centered from behind, with stable anatomy, cloak, backpack, and staff.

Style: Painterly high-end fantasy environment art with soft golden daylight, saturated teal water, finely layered limestone and foliage detail, a serene exploration mood, and controlled cinematic color.

Perspective: Wide third-person rear view at standing height, preserving the centered adventurer, the pool, the stone arch, and the distant palace.

Sounds: Shallow water laps faintly against the stone shore, loose pebbles shift under careful footsteps, birds call and echo between the cliff walls, and broad leaves stir in a light breeze. A calm orchestral theme with soft strings, wooden flute, and low hand percussion supports the sense of discovery without covering the water.

Speech: None."""
negative_prompt = """worst quality, inconsistent motion, blurry, jittery, distorted, game UI,
video game interface, HUD, heads-up display, menu, status bar, health bar,
score, minimap, crosshair, reticle, buttons, icons, subtitles, captions,
watermark, logo, text overlay, user interface"""

result = pipe(
    image=image,
    prompt=prompt,
    negative_prompt=negative_prompt,
    action="w-60,a-60,w-60,d-60",
    num_frames=241,
    num_inference_steps=30,
    generator=torch.Generator(device="cuda").manual_seed(34),
    output=["videos", "audio"],
)
encode_video(
    video=result["videos"][0],
    audio=result["audio"][0].float().cpu(),
    fps=24,
    audio_sample_rate=pipe.vocoder.config.output_sampling_rate,
    output_path="echo_wm_av.mp4",
)
```

### Echo-WM Flash

Flash uses a distilled four-step schedule for every autoregressive chunk. Guidance is baked into the checkpoint, so it
doesn't accept `negative_prompt` or `num_inference_steps`.
The example below matches the [Flash model card](https://huggingface.co/Echo-Team/Echo-WM-Flash-Diffusers), using the
official `wm_causal_cases/0079` image, full prompt, and camera trajectory.

```py
import torch
from diffusers import ComponentsManager, ModularPipeline
from diffusers.utils import encode_video, load_image

components_manager = ComponentsManager()
components_manager.enable_auto_cpu_offload(device="cuda")
pipe = ModularPipeline.from_pretrained(
    "Echo-Team/Echo-WM-Flash-Diffusers", components_manager=components_manager
)
pipe.load_components(dtype=torch.bfloat16)

image = load_image(
    "https://raw.githubusercontent.com/jd-opensource/JoyAI-Echo/main/echo_wm/examples/wm_causal_cases/0079/input.jpg"
)
prompt = (
    "An enchanted crystal cave with massive prismatic crystal formations in purple, teal, and pink. "
    "Bioluminescent fungi glow on the cave floor and walls. Floating light motes drift through the air. "
    "The crystals refract light into rainbow spectra. To the right, a large crystalline cave monster with "
    "glowing purple eyes lurks behind tall crystal clusters. Deep cavern atmosphere with ethereal luminescence. "
    "Further to the right beyond the monster, a subterranean crystal pool glows with turquoise light, fed by a "
    "thin waterfall dripping from a stalactite cluster. The cave opens into a wider chamber with an ancient stone "
    "altar covered in glowing runes. First-person viewer. First-person view with the right hand holding a twisted "
    "wooden magic wand topped with a bright blue-white crystal orb that radiates light. The wand rotates together "
    "with the viewer's perspective when turning."
)
result = pipe(
    image=image,
    prompt=prompt,
    action="l-96,l-96,l-96,l-96",
    num_frames=241,
    generator=torch.Generator(device="cuda").manual_seed(42),
    output=["videos", "audio"],
)
encode_video(
    video=result["videos"][0],
    audio=result["audio"][0].float().cpu(),
    fps=24,
    audio_sample_rate=pipe.vocoder.config.output_sampling_rate,
    output_path="echo_wm_flash_av.mp4",
)
```

## Camera control

An action is a comma-separated sequence of `<keys>-<frames>` segments. Keys are case-insensitive and may be combined,
for example `wj-60` moves forward while turning left.

| Key | Camera action |
|---|---|
| `w` / `s` | Move forward / backward |
| `a` / `d` | Strafe left / right |
| `i` / `k` | Pitch up / down |
| `j` / `l` | Yaw left / right |
| `none` | Hold still |

For a 241-frame video, 240 action frames are used because the input image supplies the first frame. For example,
`w-60,a-60,w-60,d-60` produces a four-segment camera trajectory. Longer action programs are truncated to the requested
video length: the Flash example retains the official `l-96,l-96,l-96,l-96` program but uses only its first 240 action
frames. Shorter programs hold the final camera pose for the remaining frames.

### Action overlay

[`apply_action_overlay`] optionally draws a WASD/IJKL HUD on decoded PIL frames. This is a postprocessing utility and
doesn't change model inference. Apply it before [`encode_video`] to keep the generated audio in the same MP4:

```py
from diffusers.modular_pipelines.echo_wm import apply_action_overlay
from diffusers.utils import encode_video

video = apply_action_overlay(result["videos"][0], action="w-60,a-60,w-60,d-60")
encode_video(
    video=video,
    audio=result["audio"][0].float().cpu(),
    fps=24,
    audio_sample_rate=pipe.vocoder.config.output_sampling_rate,
    output_path="echo_wm_action_av.mp4",
)
```

## Usage notes

- The released checkpoints generate at 1280x704. The UCPE grid is stored in the transformer configuration, so using a
  different resolution requires a checkpoint trained for that grid.
- Flash requires `num_frames = 1 + 24 * n`, such as 121 or 241 frames. The default 241-frame output corresponds to ten
  autoregressive chunks after the initial-frame sink.
- Base accepts `negative_prompt` and `num_inference_steps`. Flash uses its fixed distilled schedule and only consumes
  the positive prompt.
- Base and Flash enable spatial and temporal VAE decode tiling by default. The reference defaults are a 512-pixel
  spatial tile long side (short side scaled to the video aspect ratio), 64-pixel spatial overlap, 64-frame temporal
  tiles, and 24-frame temporal overlap. These parameters use Diffusers' VAE tile boundaries and blending, which
  differ from the original implementation; decoded pixels are not guaranteed to be identical.
  Pass `vae_tiling=False` to disable both, or adjust `vae_tile_size`, `vae_tile_overlap`, `vae_temporal_tile_size`,
  and `vae_temporal_tile_overlap` when calling the pipeline. The settings apply only during video decoding and
  do not change a shared VAE's encoding behavior. Tiling reduces intermediate activation memory, but the decoded
  video is still returned in full rather than streamed to disk.
- `ComponentsManager.enable_auto_cpu_offload()` lowers GPU memory usage but increases generation time. Omit the
  components manager and use `pipe.to("cuda")` when the entire pipeline fits in GPU memory.
- `encode_video` uses PyAV to write the generated video and synchronized audio directly to a single MP4 file.
- Encode audio at `pipe.vocoder.config.output_sampling_rate` (48 kHz for these checkpoints), not
  `pipe.audio_sampling_rate` (16 kHz, used for the audio latent time grid). Using the latter slows playback by 3x.

## EchoWMModularPipeline

[[autodoc]] EchoWMModularPipeline

## EchoWMFlashModularPipeline

[[autodoc]] EchoWMFlashModularPipeline

## EchoWMTransformer3DModel

[[autodoc]] EchoWMTransformer3DModel

## EchoWMBlocks

[[autodoc]] EchoWMBlocks

## EchoWMFlashBlocks

[[autodoc]] EchoWMFlashBlocks

## EchoWMCameraConditionStep

[[autodoc]] modular_pipelines.echo_wm.EchoWMCameraConditionStep

## apply_action_overlay

[[autodoc]] modular_pipelines.echo_wm.apply_action_overlay
