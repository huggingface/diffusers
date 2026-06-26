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

# Cosmos 3

NVIDIA Cosmos 3 is a unified world foundation model (WFM) for Physical AI — a single omni-model that combines world generation, physical reasoning, and action generation. It replaces the separate Predict, Reason, and Transfer models from earlier Cosmos releases: whether you're building for robotics, autonomous vehicles, or smart spaces, Cosmos 3 gives you one foundation to simulate and understand the physical world.

What's shipping with this release:

- Models on the Hugging Face Hub with model cards and licensing
- Cosmos 3 Diffusers integration for generation pipelines (this page)
- Post-training scripts for fine-tuning Cosmos 3 on your own data
- Open synthetic data generation (SDG) datasets for Physical AI

## What's new in Cosmos 3

The biggest change from previous Cosmos releases is that Cosmos 3 is an *omni-model*, built on a Mixture-of-Transformers (MoT) architecture. Previously, developers worked with separate models for world generation (Predict), controlled generation (Transfer), scene understanding (Reason), and action-policy generation. Cosmos 3 unifies all of these in one model that reasons and generates across modalities in a single forward pass.

From one model you can:

- Generate physically plausible video worlds from text, images, or action inputs (image-to-video, text-to-video, action-conditioned video generation).
- Reason about physical properties like motion, causality, and spatial relationships.
- Predict future video and action sequences from the current state.
- Transfer scenes across viewpoints and conditions with structural control (edge, blur, depth, segmentation, world-scenario maps).

Under the hood, a single `Cosmos3OmniTransformer` runs a Qwen-style language model in parallel with a diffusion generation pathway: text tokens flow through a causal "understanding" stream while video and sound latents flow through a bi-directionally-attended "generation" stream, joined by a 3D multimodal RoPE. See the [Cosmos World Foundation Model Platform paper](https://huggingface.co/papers/2501.03575) for the architectural background.

## Available checkpoints

Two checkpoints are released on the Hub — [`nvidia/Cosmos3-Nano`](https://huggingface.co/nvidia/Cosmos3-Nano) (smaller, faster) and [`nvidia/Cosmos3-Super`](https://huggingface.co/nvidia/Cosmos3-Super) (larger, higher quality). The same pipeline class supports text-to-image, text-to-video, image-to-video, and (with a sound-capable checkpoint) text+image-to-video-with-sound — pick a repo and use the per-model tab in each workflow below.

> [!TIP]
> Make sure to check out the Schedulers [guide](../../using-diffusers/schedulers) to learn how to explore the tradeoff between scheduler speed and quality, and see the [reuse components across pipelines](../../using-diffusers/loading#reuse-a-pipeline) section to learn how to efficiently load the same components into multiple pipelines.

## Prompt upsampling

Cosmos 3 was trained on long, highly descriptive captions. For optimal quality, short text prompts should be **upsampled into a specific JSON structure** before they are passed to the pipeline. The upsampler lives in the [cosmos-framework](https://github.com/NVIDIA/cosmos-framework) package.

Start from a short, plain-text prompt and save it to `assets/prompt.txt`. For the text-to-video example below, the original prompt is *"A robotic arm is cleaning a plate in a kitchen"*:

```bash
mkdir -p assets
echo "A robotic arm is cleaning a plate in a kitchen" > assets/prompt.txt
```

Then install the framework and run the upsampler. The example below upsamples for text-to-video using Opus-4.6:

```bash
git clone https://github.com/NVIDIA/cosmos-framework.git packages/cosmos-framework
pip install -e packages/cosmos-framework

export PROMPT_UPSAMPLER_ENDPOINT_URL="https://api.anthropic.com/v1/"
export PROMPT_UPSAMPLER_MODEL_NAME="claude-opus-4-6"
export PROMPT_UPSAMPLER_API_TOKEN="<your_token>"

python -m cosmos_framework.inference.prompt_upsampling \
    --input assets/prompt.txt \
    --output assets/example_t2v_prompt.json \
    --mode text2video \
    --endpoint-url "${PROMPT_UPSAMPLER_ENDPOINT_URL}" \
    --model "${PROMPT_UPSAMPLER_MODEL_NAME}" \
    --api-token "${PROMPT_UPSAMPLER_API_TOKEN}" \
    --resolution 720 \
    --aspect-ratio "16,9"
```

Switch `--mode` to match the workflow you are targeting (`text2image`, `text2video`, `image2video`). The command writes the upsampled prompt(s) to the `--output` file as a JSON array (one object per non-empty line in `--input`); pass a `.jsonl` path instead to get one JSON object per line. For `image2video`, you must also supply the conditioning image via `--image-url` (a URL or local path) or `--image-list` (one image per prompt).

<!-- TODO: Add prompt upsampling support for video inputs (video-to-video) to the upsampler CLI. -->

A pre-upsampled positive prompt (`assets/example_t2v_prompt.json`) and negative prompt (`assets/negative_prompt.json`) are provided for convenience, and are used by the generation examples below. The examples load these JSON files and pass them to the pipeline as JSON strings via `json.dumps(...)`.

## Text-to-video

Multi-frame generation conditioned on text alone. Pick `num_frames` based on the target duration — the default `num_frames=189` produces ≈ 7.9 s at 24 FPS. The prompt and negative prompt are read from the JSON-upsampled files described in [Prompt upsampling](#prompt-upsampling).

<hfoptions id="model">
<hfoption id="Nano">

```python
import json
import torch
from diffusers import Cosmos3OmniPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.utils import export_to_video

# JSON-upsampled positive and negative prompts (see "Prompt upsampling" above).
json_prompt = json.load(open("assets/example_t2v_prompt.json"))
negative_prompt = json.load(open("assets/negative_prompt.json"))

pipe = Cosmos3OmniPipeline.from_pretrained(
    "nvidia/Cosmos3-Nano", torch_dtype=torch.bfloat16, device_map="cuda"
)
pipe.scheduler = UniPCMultistepScheduler.from_config(
    pipe.scheduler.config, flow_shift=10.0, use_karras_sigmas=False
)

result = pipe(
    prompt=json.dumps(json_prompt),
    negative_prompt=json.dumps(negative_prompt),
    num_frames=189,
    height=720,
    width=1280,
    num_inference_steps=35,
    guidance_scale=6.0,
    fps=24.0,
)
# macro_block_size=1 allows arbitrary frame sizes (Cosmos3 outputs are not always divisible by 16).
export_to_video(result.video, "cosmos3_t2v.mp4", fps=24, macro_block_size=1)
```

</hfoption>
<hfoption id="Super">

```python
import json
import torch
from diffusers import Cosmos3OmniPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.utils import export_to_video

# JSON-upsampled positive and negative prompts (see "Prompt upsampling" above).
json_prompt = json.load(open("assets/example_t2v_prompt.json"))
negative_prompt = json.load(open("assets/negative_prompt.json"))

pipe = Cosmos3OmniPipeline.from_pretrained(
    "nvidia/Cosmos3-Super", torch_dtype=torch.bfloat16, device_map="cuda"
)
pipe.scheduler = UniPCMultistepScheduler.from_config(
    pipe.scheduler.config, flow_shift=10.0, use_karras_sigmas=False
)

result = pipe(
    prompt=json.dumps(json_prompt),
    negative_prompt=json.dumps(negative_prompt),
    num_frames=189,
    height=720,
    width=1280,
    num_inference_steps=35,
    guidance_scale=6.0,
    fps=24.0,
)
# macro_block_size=1 allows arbitrary frame sizes (Cosmos3 outputs are not always divisible by 16).
export_to_video(result.video, "cosmos3_t2v.mp4", fps=24, macro_block_size=1)
```

</hfoption>
</hfoptions>

## Text-to-image

Single-frame generation. The model is conditioned only on the text prompt; pass `num_frames=1`. Upsample with `--mode text2image` to produce the JSON prompt.

<hfoptions id="model">
<hfoption id="Nano">

```python
import json
import torch
from diffusers import Cosmos3OmniPipeline

# JSON-upsampled prompt (see "Prompt upsampling" above).
json_prompt = json.load(open("assets/example_t2i_prompt.json"))

pipe = Cosmos3OmniPipeline.from_pretrained(
    "nvidia/Cosmos3-Nano", torch_dtype=torch.bfloat16, device_map="cuda"
)

result = pipe(prompt=json.dumps(json_prompt), num_frames=1, height=720, width=1280)
result.video[0].save("cosmos3_t2i.jpg", format="JPEG", quality=85)
```

</hfoption>
<hfoption id="Super">

```python
import json
import torch
from diffusers import Cosmos3OmniPipeline

# JSON-upsampled prompt (see "Prompt upsampling" above).
json_prompt = json.load(open("assets/example_t2i_prompt.json"))

pipe = Cosmos3OmniPipeline.from_pretrained(
    "nvidia/Cosmos3-Super", torch_dtype=torch.bfloat16, device_map="cuda"
)

result = pipe(prompt=json.dumps(json_prompt), num_frames=1, height=720, width=1280)
result.video[0].save("cosmos3_t2i.jpg", format="JPEG", quality=85)
```

</hfoption>
</hfoptions>

## Image-to-video

Pass a conditioning image via `image=`. The pipeline anchors frame 0 to the supplied image and denoises the rest. Upsample with `--mode image2video` to produce the JSON prompt.

<hfoptions id="model">
<hfoption id="Nano">

```python
import json
import torch
from diffusers import Cosmos3OmniPipeline
from diffusers.utils import export_to_video, load_image

# JSON-upsampled positive and negative prompts (see "Prompt upsampling" above).
json_prompt = json.load(open("assets/example_i2v_prompt.json"))
negative_prompt = json.load(open("assets/negative_prompt_i2v.json"))

pipe = Cosmos3OmniPipeline.from_pretrained(
    "nvidia/Cosmos3-Nano", torch_dtype=torch.bfloat16, device_map="cuda"
)

image = load_image(
    "https://github.com/nvidia-cosmos/cosmos-dependencies/releases/download/assets/robot_153.jpg"
)

result = pipe(
    prompt=json.dumps(json_prompt),
    negative_prompt=json.dumps(negative_prompt),
    image=image,
    num_frames=189,
    height=720,
    width=1280,
    fps=24.0,
)
# macro_block_size=1 allows arbitrary frame sizes (Cosmos3 outputs are not always divisible by 16).
export_to_video(result.video, "cosmos3_i2v.mp4", fps=24, macro_block_size=1)
```

</hfoption>
<hfoption id="Super">

```python
import json
import torch
from diffusers import Cosmos3OmniPipeline
from diffusers.utils import export_to_video, load_image

# JSON-upsampled positive and negative prompts (see "Prompt upsampling" above).
json_prompt = json.load(open("assets/example_i2v_prompt.json"))
negative_prompt = json.load(open("assets/negative_prompt_i2v.json"))

pipe = Cosmos3OmniPipeline.from_pretrained(
    "nvidia/Cosmos3-Super", torch_dtype=torch.bfloat16, device_map="cuda"
)

image = load_image(
    "https://github.com/nvidia-cosmos/cosmos-dependencies/releases/download/assets/robot_153.jpg"
)

result = pipe(
    prompt=json.dumps(json_prompt),
    negative_prompt=json.dumps(negative_prompt),
    image=image,
    num_frames=189,
    height=720,
    width=1280,
    fps=24.0,
)
# macro_block_size=1 allows arbitrary frame sizes (Cosmos3 outputs are not always divisible by 16).
export_to_video(result.video, "cosmos3_i2v.mp4", fps=24, macro_block_size=1)
```

</hfoption>
</hfoptions>

## Video-to-video

Pass a conditioning clip via `video=` (e.g. from `load_video`). The pipeline anchors the leading latent frames given by `condition_frame_indexes_vision` (default `[0, 1]`) to the clip and denoises the rest. Use `condition_video_keep` (`"first"` or `"last"`) to choose which end of a longer source clip the conditioning frames are taken from. As with the other modes, the prompt should follow the descriptive JSON structure described in [Prompt upsampling](#prompt-upsampling).

<!-- TODO: Add prompt upsampling support for video inputs (video-to-video) to the upsampler CLI. -->

<hfoptions id="model">
<hfoption id="Nano">

```python
import json
import torch
from diffusers import Cosmos3OmniPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.utils import export_to_video, load_video

# JSON-upsampled positive and negative prompts (see "Prompt upsampling" above).
json_prompt = json.load(open("assets/example_v2v_prompt.json"))
negative_prompt = json.load(open("assets/negative_prompt_i2v.json"))

pipe = Cosmos3OmniPipeline.from_pretrained(
    "nvidia/Cosmos3-Nano", torch_dtype=torch.bfloat16, device_map="cuda"
)
pipe.scheduler = UniPCMultistepScheduler.from_config(
    pipe.scheduler.config, flow_shift=10.0, use_karras_sigmas=False
)

video = load_video(
    "https://github.com/nvidia-cosmos/cosmos-dependencies/raw/refs/heads/assets/cosmos3/inputs/vision/robot_pouring.mp4"
)

result = pipe(
    prompt=json.dumps(json_prompt),
    negative_prompt=json.dumps(negative_prompt),
    video=video,
    condition_frame_indexes_vision=[0, 1],
    condition_video_keep="first",
    num_frames=189,
    height=720,
    width=1280,
    num_inference_steps=35,
    guidance_scale=6.0,
    fps=24.0,
)
# macro_block_size=1 allows arbitrary frame sizes (Cosmos3 outputs are not always divisible by 16).
export_to_video(result.video, "cosmos3_v2v.mp4", fps=24, macro_block_size=1)
```

</hfoption>
<hfoption id="Super">

```python
import json
import torch
from diffusers import Cosmos3OmniPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.utils import export_to_video, load_video

# JSON-upsampled positive and negative prompts (see "Prompt upsampling" above).
json_prompt = json.load(open("assets/example_v2v_prompt.json"))
negative_prompt = json.load(open("assets/negative_prompt_i2v.json"))

pipe = Cosmos3OmniPipeline.from_pretrained(
    "nvidia/Cosmos3-Super", torch_dtype=torch.bfloat16, device_map="cuda"
)
pipe.scheduler = UniPCMultistepScheduler.from_config(
    pipe.scheduler.config, flow_shift=10.0, use_karras_sigmas=False
)

video = load_video(
    "https://github.com/nvidia-cosmos/cosmos-dependencies/raw/refs/heads/assets/cosmos3/inputs/vision/robot_pouring.mp4"
)

result = pipe(
    prompt=json.dumps(json_prompt),
    negative_prompt=json.dumps(negative_prompt),
    video=video,
    condition_frame_indexes_vision=[0, 1],
    condition_video_keep="first",
    num_frames=189,
    height=720,
    width=1280,
    num_inference_steps=35,
    guidance_scale=6.0,
    fps=24.0,
)
# macro_block_size=1 allows arbitrary frame sizes (Cosmos3 outputs are not always divisible by 16).
export_to_video(result.video, "cosmos3_v2v.mp4", fps=24, macro_block_size=1)
```

</hfoption>
</hfoptions>

## Transfer (structural control)

Transfer generates a target clip that follows a **precomputed control video** (a spatial control signal): edge (Canny), blur, depth, segmentation, or a world-scenario map (WSM). Pass it through `control_videos=` as a mapping from hint name to a loaded video. The control map is resized, temporally padded, normalized, and VAE-encoded into a clean conditioning item placed before the noisy target; the model then generates the target to match it. Transfer is video-only (no `image`, `video`, `action`, or `enable_sound`), and the prompt is a pre-upsampled JSON caption (see [Prompt upsampling](#prompt-upsampling)).

Diffusers does not ship the control assets. Ready-made ones (a control video + matching `prompt.json` per hint, plus a shared `negative_prompt.json`) live in the [Cosmos cookbook](https://github.com/NVIDIA/cosmos/tree/main/cookbooks/cosmos3/generator/transfer/assets). For the edge example below, download them into a local `assets/` folder:

```bash
base=https://github.com/NVIDIA/cosmos/raw/refs/heads/main/cookbooks/cosmos3/generator/transfer/assets
mkdir -p assets/edge
curl -sL "$base/edge/control_edge.mp4" -o assets/edge/control_edge.mp4
curl -sL "$base/edge/prompt.json"      -o assets/edge/prompt.json
curl -sL "$base/negative_prompt.json"  -o assets/negative_prompt.json
```

Guidance uses a nested control/text classifier-free-guidance blend. `guidance_scale` is the usual text CFG; `control_guidance` (`!= 1.0`) additionally amplifies the control signal. Recommended starting values per hint:

| Hint | `guidance_scale` | `control_guidance` | `flow_shift` | Geometry |
| --- | --- | --- | --- | --- |
| Edge / Blur / Depth | 3.0 | 1.5 | 10.0 | 121 frames @ 30 FPS |
| Segmentation | 3.0 | 2.0 | 10.0 | 121 frames @ 30 FPS |
| World scenario (WSM) | 1.0 | 3.0 | 10.0 | 101 frames @ 10 FPS |

Depth, segmentation, and WSM control maps must be precomputed by external models; edge/blur maps can be produced offline with any Canny/blur tool. The shipped cookbook configs use a single hint each; passing several entries in `control_videos` to combine hints is supported by the pipeline but is not a tuned/validated cookbook path (set `guidance_scale` / `control_guidance` explicitly, since the per-hint defaults above assume a single hint). Long clips are generated autoregressively in chunks of `num_video_frames_per_chunk` and stitched automatically.

<hfoptions id="model">
<hfoption id="Nano">

```python
import json
import torch
from diffusers import Cosmos3OmniPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.utils import export_to_video, load_video

# Downloaded into assets/ from the Cosmos cookbook (see the curl snippet above).
json_prompt = json.load(open("assets/edge/prompt.json"))
negative_prompt = json.load(open("assets/negative_prompt.json"))
control_edge = load_video("assets/edge/control_edge.mp4")

pipe = Cosmos3OmniPipeline.from_pretrained(
    "nvidia/Cosmos3-Nano", torch_dtype=torch.bfloat16, device_map="cuda"
)
pipe.scheduler = UniPCMultistepScheduler.from_config(
    pipe.scheduler.config, flow_shift=10.0, use_karras_sigmas=False
)

result = pipe(
    prompt=json.dumps(json_prompt),
    negative_prompt=json.dumps(negative_prompt),
    control_videos={"edge": control_edge},
    num_frames=121,
    height=720,
    width=1280,
    fps=30.0,
    num_inference_steps=35,
    guidance_scale=3.0,
    control_guidance=1.5,
)
# macro_block_size=1 allows arbitrary frame sizes (Cosmos3 outputs are not always divisible by 16).
export_to_video(result.video, "cosmos3_transfer_edge.mp4", fps=30, macro_block_size=1)
```

</hfoption>
<hfoption id="Super">

```python
import json
import torch
from diffusers import Cosmos3OmniPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.utils import export_to_video, load_video

# Downloaded into assets/ from the Cosmos cookbook (see the curl snippet above).
json_prompt = json.load(open("assets/edge/prompt.json"))
negative_prompt = json.load(open("assets/negative_prompt.json"))
control_edge = load_video("assets/edge/control_edge.mp4")

pipe = Cosmos3OmniPipeline.from_pretrained(
    "nvidia/Cosmos3-Super", torch_dtype=torch.bfloat16, device_map="cuda"
)
pipe.scheduler = UniPCMultistepScheduler.from_config(
    pipe.scheduler.config, flow_shift=10.0, use_karras_sigmas=False
)

result = pipe(
    prompt=json.dumps(json_prompt),
    negative_prompt=json.dumps(negative_prompt),
    control_videos={"edge": control_edge},
    num_frames=121,
    height=720,
    width=1280,
    fps=30.0,
    num_inference_steps=35,
    guidance_scale=3.0,
    control_guidance=1.5,
)
# macro_block_size=1 allows arbitrary frame sizes (Cosmos3 outputs are not always divisible by 16).
export_to_video(result.video, "cosmos3_transfer_edge.mp4", fps=30, macro_block_size=1)
```

</hfoption>
</hfoptions>

## Video-to-video with sound

When the checkpoint carries a `sound_tokenizer`, add `enable_sound=True` to the video-to-video call to jointly generate a synchronized audio track. The waveform is returned alongside the video and can be muxed into the MP4 with [`~utils.encode_video`].

<hfoptions id="model">
<hfoption id="Nano">

```python
import json
import torch
from diffusers import Cosmos3OmniPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.utils import encode_video, load_video

# JSON-upsampled positive and negative prompts (see "Prompt upsampling" above).
json_prompt = json.load(open("assets/example_v2v_prompt.json"))
negative_prompt = json.load(open("assets/negative_prompt_i2v.json"))

pipe = Cosmos3OmniPipeline.from_pretrained(
    "nvidia/Cosmos3-Nano", torch_dtype=torch.bfloat16, device_map="cuda"
)
pipe.scheduler = UniPCMultistepScheduler.from_config(
    pipe.scheduler.config, flow_shift=10.0, use_karras_sigmas=False
)

video = load_video(
    "https://github.com/nvidia-cosmos/cosmos-dependencies/raw/refs/heads/assets/cosmos3/inputs/vision/robot_pouring.mp4"
)

result = pipe(
    prompt=json.dumps(json_prompt),
    negative_prompt=json.dumps(negative_prompt),
    video=video,
    condition_frame_indexes_vision=[0, 1],
    condition_video_keep="first",
    num_frames=189,
    height=720,
    width=1280,
    fps=24.0,
    enable_sound=True,
)

encode_video(
    result.video,
    fps=24,
    audio=result.sound,
    audio_sample_rate=pipe.sound_tokenizer.config.sampling_rate,
    output_path="cosmos3_v2v_with_sound.mp4",
)
```

</hfoption>
<hfoption id="Super">

```python
import json
import torch
from diffusers import Cosmos3OmniPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.utils import encode_video, load_video

# JSON-upsampled positive and negative prompts (see "Prompt upsampling" above).
json_prompt = json.load(open("assets/example_v2v_prompt.json"))
negative_prompt = json.load(open("assets/negative_prompt_i2v.json"))

pipe = Cosmos3OmniPipeline.from_pretrained(
    "nvidia/Cosmos3-Super", torch_dtype=torch.bfloat16, device_map="cuda"
)
pipe.scheduler = UniPCMultistepScheduler.from_config(
    pipe.scheduler.config, flow_shift=10.0, use_karras_sigmas=False
)

video = load_video(
    "https://github.com/nvidia-cosmos/cosmos-dependencies/raw/refs/heads/assets/cosmos3/inputs/vision/robot_pouring.mp4"
)

result = pipe(
    prompt=json.dumps(json_prompt),
    negative_prompt=json.dumps(negative_prompt),
    video=video,
    condition_frame_indexes_vision=[0, 1],
    condition_video_keep="first",
    num_frames=189,
    height=720,
    width=1280,
    fps=24.0,
    enable_sound=True,
)

encode_video(
    result.video,
    fps=24,
    audio=result.sound,
    audio_sample_rate=pipe.sound_tokenizer.config.sampling_rate,
    output_path="cosmos3_v2v_with_sound.mp4",
)
```

</hfoption>
</hfoptions>

## Text-to-video with sound

When the checkpoint carries a `sound_tokenizer`, pass `enable_sound=True` to jointly generate a synchronized audio track. The waveform is returned alongside the video and can be muxed into the MP4 with [`~utils.encode_video`].

This is the same call as the text-to-video example above with `enable_sound=True` added:

<hfoptions id="model">
<hfoption id="Nano">

```python
import json
import torch
from diffusers import Cosmos3OmniPipeline
from diffusers.utils import encode_video

# JSON-upsampled positive and negative prompts (see "Prompt upsampling" above).
json_prompt = json.load(open("assets/example_t2v_sound_prompt.json"))
negative_prompt = json.load(open("assets/negative_prompt.json"))

pipe = Cosmos3OmniPipeline.from_pretrained(
    "nvidia/Cosmos3-Nano", torch_dtype=torch.bfloat16, device_map="cuda"
)

result = pipe(
    prompt=json.dumps(json_prompt),
    negative_prompt=json.dumps(negative_prompt),
    num_frames=189,
    height=720,
    width=1280,
    fps=24.0,
    enable_sound=True,
)

encode_video(
    result.video,
    fps=24,
    audio=result.sound,
    audio_sample_rate=pipe.sound_tokenizer.config.sampling_rate,
    output_path="cosmos3_with_sound.mp4",
)
```

</hfoption>
<hfoption id="Super">

```python
import json
import torch
from diffusers import Cosmos3OmniPipeline
from diffusers.utils import encode_video

# JSON-upsampled positive and negative prompts (see "Prompt upsampling" above).
json_prompt = json.load(open("assets/example_t2v_sound_prompt.json"))
negative_prompt = json.load(open("assets/negative_prompt.json"))

pipe = Cosmos3OmniPipeline.from_pretrained(
    "nvidia/Cosmos3-Super", torch_dtype=torch.bfloat16, device_map="cuda"
)

result = pipe(
    prompt=json.dumps(json_prompt),
    negative_prompt=json.dumps(negative_prompt),
    num_frames=189,
    height=720,
    width=1280,
    fps=24.0,
    enable_sound=True,
)

encode_video(
    result.video,
    fps=24,
    audio=result.sound,
    audio_sample_rate=pipe.sound_tokenizer.config.sampling_rate,
    output_path="cosmos3_with_sound.mp4",
)
```

</hfoption>
</hfoptions>

## Action-conditioned generation

Action runs group every action-specific input into a [`CosmosActionCondition`] passed via the `action` argument instead of the top-level `image` / `video` / `height` / `width` arguments. Set `resolution_tier` (`256`/`480`/`704`/`720`) close to the input video's native resolution; it selects the conditioning canvas. Cosmos 3 supports three action modes — `policy`, `forward_dynamics`, and `inverse_dynamics`. `policy` and `forward_dynamics` condition only on the first frame (so an `image` or a `video` both work), while `inverse_dynamics` requires a `video`. The conditioning video for an action run is set on `action.video` (or `action.image`), not on the pipeline's top-level `video` argument.

Pass a plain task description as `prompt` and pick the camera with `action.view_point` (default `"ego_view"`; also `"third_person_view"`, `"wrist_view"`, `"concat_view"`). The pipeline turns these into the structured JSON caption the model was trained on, so action prompts should not be LLM-upsampled.

### Action policy

Action policy generation predicts future video and action tokens from the first observation frame, text prompt, and action domain metadata. The example below uses the Bridge robot domain and writes the predicted action chunk to JSON in model-normalized action space.

<hfoptions id="model">
<hfoption id="Nano">

```python
import json

import torch
from diffusers import Cosmos3OmniPipeline, CosmosActionCondition
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.utils import export_to_video, load_video

pipe = Cosmos3OmniPipeline.from_pretrained(
    "nvidia/Cosmos3-Nano", torch_dtype=torch.bfloat16, device_map="cuda"
)
pipe.scheduler = UniPCMultistepScheduler.from_config(
    pipe.scheduler.config, flow_shift=10.0, use_karras_sigmas=False
)

prompt = "Put the pot to the left of the purple item."
video = load_video(
    "https://github.com/nvidia-cosmos/cosmos-dependencies/raw/refs/heads/assets/cosmos3/inputs/action/bridge_20260501_0.mp4"
)

result = pipe(
    prompt=prompt,
    action=CosmosActionCondition(
        mode="policy",
        chunk_size=16,
        domain_name="bridge_orig_lerobot",
        resolution_tier=480,
        video=video,
        view_point="ego_view",
    ),
    fps=5,
    num_inference_steps=30,
    guidance_scale=1.0,
    use_system_prompt=False,
)

# macro_block_size=1 allows arbitrary frame sizes (Cosmos3 outputs are not always divisible by 16).
export_to_video(result.video, "sample.mp4", fps=5, macro_block_size=1)

if result.action is not None:
    with open("sample_action.json", "w") as f:
        json.dump(result.action[0].tolist(), f)
```

</hfoption>
<hfoption id="Super">

```python
import json

import torch
from diffusers import Cosmos3OmniPipeline, CosmosActionCondition
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.utils import export_to_video, load_video

pipe = Cosmos3OmniPipeline.from_pretrained(
    "nvidia/Cosmos3-Super", torch_dtype=torch.bfloat16, device_map="cuda"
)
pipe.scheduler = UniPCMultistepScheduler.from_config(
    pipe.scheduler.config, flow_shift=10.0, use_karras_sigmas=False
)

prompt = "Put the pot to the left of the purple item."
video = load_video(
    "https://github.com/nvidia-cosmos/cosmos-dependencies/raw/refs/heads/assets/cosmos3/inputs/action/bridge_20260501_0.mp4"
)

result = pipe(
    prompt=prompt,
    action=CosmosActionCondition(
        mode="policy",
        chunk_size=16,
        domain_name="bridge_orig_lerobot",
        resolution_tier=480,
        video=video,
        view_point="ego_view",
    ),
    fps=5,
    num_inference_steps=30,
    guidance_scale=1.0,
    use_system_prompt=False,
)

# macro_block_size=1 allows arbitrary frame sizes (Cosmos3 outputs are not always divisible by 16).
export_to_video(result.video, "sample.mp4", fps=5, macro_block_size=1)

if result.action is not None:
    with open("sample_action.json", "w") as f:
        json.dump(result.action[0].tolist(), f)
```

</hfoption>
</hfoptions>

## Metadata templates

`tokenize_prompt` appends short metadata sentences inside the user message so the LLM sees the conditioning the model was trained with. The positive prompt gets sentences like *"The video is 7.9 seconds long and is of 24 FPS."* and *"This video is of 720x1280 resolution."*; the negative prompt gets the inverse (*"… is not …"*).

Both are on by default. Disable either pair through `__call__`:

```python
result = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_frames=189,
    height=720,
    width=1280,
    fps=24.0,
    add_duration_template=False,    # skip the duration sentence on both prompts
    add_resolution_template=False,  # skip the resolution sentence on both prompts
)
```

`add_duration_template` has no effect when `num_frames == 1` (image mode); only the resolution sentence is appended in that case.

## Safety checker

Cosmos3 wires up the [`cosmos_guardrail`](https://pypi.org/project/cosmos-guardrail/) `CosmosSafetyChecker` and runs it **by default**. The text guardrail rejects unsafe prompts before generation (`ValueError`); the video guardrail runs on the decoded frames and either pixelates detected faces or rejects the output. Audio output is not guardrailed.

Install the optional dependency to enable the default checker:

```
pip install cosmos_guardrail
```

The checker is mandatory under the NVIDIA Open Model License Agreement. The two flags below exist for tests and development workflows where the guardrail would be redundant (e.g., the input has already been cleared, or you are intentionally exercising the pipeline on edge inputs).

**Disable at construction** (no checker is instantiated, so no guardrail models are downloaded or loaded into memory):

```python
import torch
from diffusers import Cosmos3OmniPipeline

pipe = Cosmos3OmniPipeline.from_pretrained(
    "nvidia/Cosmos3-Nano",
    torch_dtype=torch.bfloat16,
    device_map="cuda",
    enable_safety_checker=False,
)
```

**Disable for a single call** (checker stays loaded — useful for one-off bypass while keeping the default on for subsequent calls):

```python
result = pipe(
    prompt=prompt,
    num_frames=189,
    height=720,
    width=1280,
    fps=24.0,
    enable_safety_check=False,
)
```

To supply a custom checker (e.g., a no-op subclass for fast tests), pass it as `safety_checker=`:

```python
pipe = Cosmos3OmniPipeline.from_pretrained(
    "nvidia/Cosmos3-Nano",
    torch_dtype=torch.bfloat16,
    device_map="cuda",
    safety_checker=MyCustomSafetyChecker(),
)
```

## Cosmos3OmniPipeline

[[autodoc]] Cosmos3OmniPipeline

- all
- __call__

## CosmosActionCondition

[[autodoc]] CosmosActionCondition

## Cosmos3OmniPipelineOutput

[[autodoc]] pipelines.cosmos.pipeline_cosmos3_omni.Cosmos3OmniPipelineOutput