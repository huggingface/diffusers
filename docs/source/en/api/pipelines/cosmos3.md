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
- Transfer scenes across viewpoints and conditions with structural control *(coming soon)*.

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

## Context parallelism

For long videos or high resolutions, a single forward pass can exceed the memory and latency budget of one GPU. Cosmos 3 supports **context parallelism (CP)** to shard the sequence dimension across multiple GPUs, splitting the attention computation so each device holds only a slice of the tokens.

Cosmos 3 supports **Ulysses** context parallelism (all-to-all sequence/head exchange). Ring attention is not supported.

Unlike most diffusers models, Cosmos 3 does **not** wire CP into the transformer or the declarative [`~ModelMixin.enable_parallelism`] path: its grouped-query attention, separate understanding/generation streams (the generation stream attends to both), and ragged per-stream lengths can't be expressed as a `_cp_plan`. Instead, the model exposes small no-op shard/gather seams, and the implementation lives in [`examples/cosmos3/cosmos_parallel.py`](https://github.com/huggingface/diffusers/blob/main/examples/cosmos3/cosmos_parallel.py) — a self-contained module you can read end to end and adapt. It offers two orthogonal, composable sharding axes:

| Helper | Shards | Use for |
|---|---|---|
| `enable_cosmos3_context_parallel(transformer, cp_mesh)` | sequence (CP / Ulysses) | latency on a model that fits one GPU (`Nano`) |
| `enable_cosmos3_tensor_parallel(transformer, tp_mesh)` | weights (TP) | fitting a model that doesn't fit one GPU (`Super`) |

Use either alone or both together on a 2-D `(tp, cp)` mesh (see [Fitting large models with tensor parallelism](#fitting-large-models-with-tensor-parallelism)).

Two requirements are specific to Cosmos 3:

- Use the `native` attention backend. Cosmos 3 uses grouped-query attention (GQA), and the native SDPA backend is the only one that accepts `enable_gqa` (cuDNN and flash reject it). The helpers expand the KV heads to the query-head count and call SDPA with `enable_gqa=False` so it still dispatches to the flash kernel (the math fallback would materialize the full `[S, S]` scores and OOM on long sequences).
- The CP (Ulysses) degree must divide the query-head count (32 for `Nano`, 64 for `Super`); for TP, the degree must divide the KV heads (8). The understanding (text) and generation (video/sound) streams are sharded independently along the sequence, and ragged lengths are zero-padded internally to a multiple of the world size.

### Run it

The full CLI [`examples/cosmos3/inference_cosmos3.py`](https://github.com/huggingface/diffusers/blob/main/examples/cosmos3/inference_cosmos3.py) reuses these helpers, so **any modality** (text-to-image/video, image-to-video, sound, action modes) runs multi-GPU via `--tp-degree` / `--cp-degree`. Launch with [torchrun](https://docs.pytorch.org/docs/stable/elastic/run.html); `--tp-degree * --cp-degree` must equal `--nproc_per_node`. Every rank produces the same output; rank 0 writes it.

```bash
# CP only — Nano (fits one GPU); CP degree must divide 32 query heads.
torchrun --nproc_per_node=4 examples/cosmos3/inference_cosmos3.py --model nano --cp-degree 4 --prompt "..."

# TP only — Super; TP degree must divide 64 query heads and 8 KV heads.
torchrun --nproc_per_node=4 examples/cosmos3/inference_cosmos3.py --model super --tp-degree 4 --prompt "..."

# TP + CP — Super, with sound (TP=2 x CP=2 across 4 GPUs).
torchrun --nproc_per_node=4 examples/cosmos3/inference_cosmos3.py \
    --model super --tp-degree 2 --cp-degree 2 --enable-sound --prompt "..."
```

`Super`'s ~120 GB of weights do not fit on one 96 GB GPU, so it needs TP; `Nano` fits on a single GPU, so CP for it is a pure latency optimization. (Omit both flags to run single-GPU.)

### Fitting large models with tensor parallelism

CP shards *activations* but replicates every weight on every rank, so it does not reduce a model's weight footprint — a model that doesn't fit on one GPU still won't fit under CP alone. To shard the **weights**, `enable_cosmos3_tensor_parallel(transformer, tp_mesh)` applies Megatron-style tensor parallelism on a second, orthogonal mesh axis:

- The attention and MLP projections are column/row sharded across the TP group (`to_q/to_k/to_v` + `add_q/k/v` and the MLPs' `gate/up` are column-parallel; `to_out/to_add_out` and the MLPs' `down` are row-parallel with an all-reduce). Each rank ends up owning `query_heads / tp` query heads and `kv_heads / tp` KV heads.
- TP composes with CP on a 2-D `(tp, cp)` device mesh: TP splits heads/weights persistently, CP shards the sequence on top. The constraints are `tp` divides the KV heads (8), and `tp * cp` divides the query heads (32 for `Nano`, 64 for `Super`).
- Weights are loaded to CPU and sharded onto the GPUs layer by layer, so the full model is never materialized on a single device.

> [!TIP]
> TP issues an all-reduce on every attention and MLP block, so it is bandwidth-heavy. On hosts without NVLink it is the dominant cost; prefer the smallest TP degree that makes the weights fit and put the remaining GPUs into CP.

### Use it in your own pipeline

The CLI flags are convenient, but you can call the helpers directly. Build the device mesh, apply TP *before* the model lands on the GPUs, switch to the `native` backend, then enable CP — the rest of your pipeline code is unchanged:

```python
from torch.distributed.device_mesh import init_device_mesh

# Make the helper module importable.
import sys
sys.path.insert(0, "examples/cosmos3")
from cosmos_parallel import (
    enable_cosmos3_context_parallel,
    enable_cosmos3_flash_attention,
    enable_cosmos3_tensor_parallel,
)

# torchrun sets RANK / WORLD_SIZE / LOCAL_RANK. Pick tp_degree * cp_degree == world size.
mesh = init_device_mesh("cuda", (tp_degree, cp_degree), mesh_dim_names=("tp", "cp"))

# Load on CPU first; a TP-sharded model may not fit one GPU.
pipe = Cosmos3OmniPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
if tp_degree > 1:
    enable_cosmos3_tensor_parallel(pipe.transformer, mesh["tp"])  # shard weights -> GPUs
pipe.to(f"cuda:{local_rank}")                                     # move the replicated remainder
pipe.transformer.set_attention_backend("native")
if cp_degree > 1:
    enable_cosmos3_context_parallel(pipe.transformer, mesh["cp"])  # shard the sequence
elif tp_degree > 1:
    enable_cosmos3_flash_attention(pipe.transformer)               # GQA-safe dense attention

# `pipe(...)` is called exactly as in the single-GPU workflows above.
```

For CP only (no weight sharding), use a 1-D mesh: `init_device_mesh("cuda", (world_size,), mesh_dim_names=("cp",))` and just `enable_cosmos3_context_parallel`.

> [!TIP]
> On some multi-GPU topologies the first NCCL all-to-all can hang. If a CP run stalls at the start of the first denoising step, set `NCCL_P2P_DISABLE=1` in the environment before launching `torchrun`.

CP and TP compose with all the workflows above (text-to-video, image-to-video, text-to-video with sound, and action-conditioned generation) and with both the `Nano` and `Super` checkpoints — only the pipeline construction and the parallelism setup lines change.

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