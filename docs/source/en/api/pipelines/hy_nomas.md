<!-- Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License. -->

# Hy-Nomas

Hy-Nomas generates a world one video segment at a time, conditioned on text and optional camera extrinsics. Its
distilled checkpoint uses flow-matching Euler sampling, a Qwen2.5-VL text encoder, and a PixelUMM diffusion decoder.

This implementation supports the released double-stream transformer, action-aware text visibility, camera attention,
and continuation through an explicit generation state. Text prompts can be plain strings or the segment JSON produced
by the model's prompt preprocessing. Action timestamps in JSON must already be relative to their segment.

## Installation

Install PyTorch 2.10.0 with CUDA support for your system, then run the following command from the Diffusers checkout:

```bash
pip install -e ".[hy-nomas]"
```

The `hy-nomas` extra installs the Transformers text encoder, Accelerate loading support, and the `imageio` and
`imageio-ffmpeg` video export dependencies. The original Hy-Nomas code checkout is not required.

## Convert the checkpoints

Run the conversion script from a Diffusers checkout. The decoder input must be the original floating-point checkpoint.
The script checks parameter mappings and verifies the saved components by loading them again.

```bash
python scripts/convert_hy_nomas_to_diffusers.py \
  --transformer_checkpoint /path/to/dit.safetensors \
  --decoder_checkpoint /path/to/pixelumm.safetensors \
  --text_encoder_path /path/to/Qwen2.5-VL-7B-Instruct \
  --output_path /path/to/Hy-Nomas \
  --dtype bfloat16
```

The output contains `modular_model_index.json` and standard component subdirectories. The converted pipeline loads
without importing the original repository or executing remote code. The examples below use `tencent/Hy-Nomas` as the
model ID; to use a local conversion, replace it with `/path/to/Hy-Nomas`.

## Generate video from a prompt

Use a plain string to describe one segment. This example generates 120 frames at 24 FPS with four denoising steps and
returns a session for subsequent interaction.

```python
import torch

from diffusers import HyNomasModularPipeline
from diffusers.utils import export_to_video

pipe = HyNomasModularPipeline.from_pretrained("tencent/Hy-Nomas")
pipe.load_components(dtype=torch.bfloat16)
pipe.to(device="cuda", dtype=torch.bfloat16)

first = pipe(
    prompt=(
        "A small orange fox stands on a stone path in a luminous forest. "
        "Blue fireflies drift between the trees. The fox walks slowly along the path."
    ),
    height=704,
    width=1280,
    num_frames=120,
    fps=24,
    num_inference_steps=4,
    generator=torch.Generator("cpu").manual_seed(42),
    output=["videos", "session"],
)
export_to_video(first["videos"][0], "hy_nomas_video.mp4", fps=24, quality=None)
```

Pass the dtype to both loading and `to` to reproduce the original BF16 inference. The latter also casts the text
encoder's nonpersistent rotary-position buffers, which are recreated in FP32 when loading its weights.

### Structured prompts and multiple segments

Structured prompts preserve scene descriptions and action timing. Serialize each segment with `json.dumps`; `prompt`
accepts a string or a list of strings, not a Python dictionary. Each serialized object must contain exactly one entry
in `segments`. A list describes successive segments of one video, with batch size one.

The following two prompts use the `forest_world` scene and action descriptions from the model assets'
`examples/prompts.json`, with each segment extended to five seconds. Keep action timestamps relative to each segment:
the second segment's 5–10 second interval becomes 0–5 seconds. Prompt enhancement and splitting a complete prompt file
into segments happen before the pipeline call.

```python
import json

prompts = [
    {
        "Style": "Cinematic fantasy",
        "perspective_view": "Third-person view",
        "subject_list": [
            {
                "subject_id": "fox",
                "display_name": "a small orange fox",
                "subject_attribute": "Orange fur, white chest, and a fluffy tail.",
            }
        ],
        "segments": [
            {
                "visual_description_long": (
                    "A small orange fox stands on a stone path in a luminous forest. "
                    "Blue fireflies drift between the trees."
                ),
                "contained_actions": [
                    {
                        "start_time": "00:00.000",
                        "end_time": "00:05.000",
                        "action_description": "The fox walks slowly along the path.",
                    }
                ],
            }
        ],
    },
    {
        "segments": [
            {
                "description_change": "Glowing blue flowers open beside the path as the fox continues forward.",
                "contained_actions": [
                    {
                        "start_time": "00:00.000",
                        "end_time": "00:05.000",
                        "action_description": "The fox passes the flowers and looks around.",
                    }
                ],
            }
        ]
    },
]

videos = pipe(
    prompt=[json.dumps(prompt) for prompt in prompts],
    height=704,
    width=1280,
    num_frames=240,
    segment_latent_lengths=[30, 30],
    fps=24,
    num_inference_steps=4,
    generator=torch.Generator("cpu").manual_seed(42),
    output="videos",
)
export_to_video(videos[0], "hy_nomas_video.mp4", fps=24, quality=None)
```

Each segment contains 30 latent frames, corresponding to 120 pixel frames or five seconds at 24 FPS. To generate only
the first segment, pass `prompt=json.dumps(prompts[0])`, `num_frames=120`, and `segment_latent_lengths=[30]`.

This adapted `forest_world` case covers ten seconds. Given absolute segment boundaries of 0, 5, and 10 seconds,
the original Hy-Nomas event-based inference derives 240 frames and pixel boundaries `[0, 120, 240]`. Use
`num_frames=240` and `segment_latent_lengths=[30, 30]` to generate the same duration in one pipeline call.

| Timestamp location | First segment | Second segment |
|---|---|---|
| Source JSON for this adapted example, relative to the full video | 0–5 seconds | 5–10 seconds |
| Actions in each model-ready prompt, relative to its segment | 0–5 seconds | 0–5 seconds |

The original preprocessor subtracts the segment start time from its action timestamps. The pipeline expects these
segment-relative action times; absolute segment boundaries are supplied separately through frame counts.

`num_frames` is the number of new pixel frames; the latent frame count is `(num_frames - 1) // 4 + 1`.
`segment_latent_lengths` must sum to this count. If omitted, all segments receive equal lengths, and the latent count
must be divisible by the number of prompts. The pipeline uses `fps` to map action timestamps to latent frames; keep
it equal to the FPS passed to `export_to_video`. It does not infer the rollout length from JSON timestamps.
Height and width must be multiples of 32. For the original nominal 720p preset, the decoded height is 704 pixels.

The default PixelUMM decoding window contains 10 latent frames. These examples use five-second segments; output
quality also depends on the prompt, seed, and decoding context.

Use `output_type="latent"` to skip decoding. The `latents` output has shape `(1, 32, T, H/32, W/32)`.
Video output types are `np` (batch, frames, height, width, channels), `pt` (batch, frames, channels, height, width),
and `pil`.

### Match reference attention

The transformer defaults to PyTorch scaled dot product attention. On CUDA, the PixelUMM decoder uses compiled
FlexAttention for local attention, so its first call includes compilation. This default inference path does not require
MagiAttention or FlashAttention. To compare against the original BF16 implementation, install its compatible
MagiAttention, FlashAttention 2, and FlashAttention 3 dependencies and select the reference processors after loading
the transformer:

```python
pipe.transformer.enable_reference_attention(text_backend="flash_varlen", deterministic=True)
```

This uses FlashAttention 2 for the unpadded text refiner, MagiAttention for action and camera visibility ranges, and
FlashAttention 3 for unrestricted history attention. MagiAttention is imported only when its backend runs; the default
processors do not require these optional dependencies. Use `disable_reference_attention()` or
`set_attention_backend("native")` to return to the standard processors. Set
`MAGI_ATTENTION_DETERMINISTIC_MODE=1` in the original implementation as well when comparing outputs. Use the same
checkpoint precision, noise timeline capacity, CPU generator seed, prompts, normalized camera inputs, and full history
on both sides. A matching seed alone does not establish numerical equivalence across different attention kernels.

Enable these processors before running the adapted `forest_world` example: use the two structured prompts above in
one 240-frame call with `segment_latent_lengths=[30, 30]`, 24 FPS, four denoising steps, and CPU seed 42.
Two separate 120-frame calls sample differently shaped noise tensors; matching their
seed and combined duration does not reproduce the single-call noise. For interactive equivalence, reserve the full
60-latent timeline with `noise_capacity=60` on the first call, then continue its session without a new generator.

To match the original MP4 encoding quality as well, export with:

```python
export_to_video(videos[0], "hy_nomas_video.mp4", fps=24, quality=None)
```

The original exporter leaves x264 at its default CRF 23, while `export_to_video` normally passes `quality=5`, which
maps to CRF 25. This encoding setting is separate from model inference. Pixel matching requires the reference
attention kernels; the default PyTorch attention path can produce different floating-point results.

## Generate video from a prompt and camera

Use the same `prompt` input and add `camera_extrinsics`, shaped `(1, num_frames, 4, 4)`. Omitting it disables camera
conditioning; no additional switch is needed. Reuse the loaded `pipe` and the structured `prompts` above:

```python
# Normalized relative poses for forward motion, including first-pose padding.
camera = torch.eye(4).repeat(1, 120, 1, 1)
camera[:, 4:, 2, 3] = 1.0

videos = pipe(
    prompt=json.dumps(prompts[0]),
    camera_extrinsics=camera,
    height=704,
    width=1280,
    num_frames=120,
    fps=24,
    num_inference_steps=4,
    generator=torch.Generator("cpu").manual_seed(42),
    output="videos",
)
export_to_video(videos[0], "hy_nomas_camera.mp4", fps=24, quality=None)
```

These inputs follow the model's training normalization. With framewise normalization enabled in the reference
preprocessing, matrices represent relative motion between neighboring camera poses, rather than absolute world
positions. In this example, positive Z translation specifies normalized forward motion. The first four identity
matrices include the repeated initial pose required by this 120-frame layout; they do not describe four independently
recorded stationary frames.

The pipeline encodes the supplied matrices without estimating camera motion or normalizing an absolute trajectory.
For custom motion, prepare camera-to-world poses in the training coordinate system and apply the reference
normalization, including segment boundaries and initial-pose padding, before calling the pipeline. Supply camera
matrices only for the new frames when continuing a session. The same camera input works with a plain text prompt.

### Display recorded keyboard controls

The `examples/community/hy_nomas_keyboard_overlay.py` example in the Diffusers checkout displays WASD and all four
arrow keys on an existing silent video. It runs on CPU without model weights or a Hy-Nomas checkout. The original video
is preserved and the example refuses to overwrite an existing output.

```bash
pip install numpy pillow imageio imageio-ffmpeg
python examples/community/hy_nomas_keyboard_overlay.py \
  --video video.mp4 \
  --manifest replay_manifest.json \
  --output video_keyboard.mp4
```

The replay manifest contains `sessions[].turns[].params`, with each turn specifying `num_frames`, `video_fps`, and
optional `camera_trajectory`. The trajectory can be a JSON object or a JSON string. Select a session with
`--session-index` when the manifest contains multiple sessions. The input video must contain exactly that session's
concatenated turns at the recorded FPS, without duplicated stable-prefix carry frames.

For a single camera recording, replace `--manifest replay_manifest.json` with `--camera-trajectory camera.json`:

```json
{
  "durationMs": 1000,
  "keyEvents": [
    {"t": 100, "code": "KeyW", "pressed": true},
    {"t": 250, "code": "ArrowUp", "pressed": true},
    {"t": 600, "code": "ArrowUp", "pressed": false},
    {"t": 800, "code": "KeyW", "pressed": false}
  ]
}
```

Event timestamps are milliseconds. Supported codes are `KeyW`, `KeyA`, `KeyS`, `KeyD`, `ArrowUp`, `ArrowDown`,
`ArrowLeft`, and `ArrowRight`. The video frame clock determines playback progress, which maps to the recording's
`durationMs`, matching camera playback. If `durationMs` is absent, timestamps use the turn's video clock directly.
States reset between turns. Segments without `keyEvents` show **NO KEY EVENTS** with every key released; camera motion
does not fabricate recorded input. This example displays recorded events and does not record live keyboard input or
convert keyboard events to model camera poses. Camera conditioning still uses `camera_extrinsics`.

For a custom preview, import the example from the repository root and render frames directly:

```python
from examples.community.hy_nomas_keyboard_overlay import KeyboardTimeline, draw_keyboard_overlay

timeline = KeyboardTimeline([turn["params"] for turn in manifest["sessions"][0]["turns"]])
pressed_keys, label = timeline.state_at_frame(frame_index)
preview = draw_keyboard_overlay(rgb_uint8_frame, pressed_keys, label)
```

The returned preview is a copy. Apply overlays after generation, and use the original frames for numerical comparisons.

## Multi-turn interaction

Pass the returned `session` to the next call to continue the same world with a new prompt. For example, continue the
five-second `first` result from the plain text example with the next scene:

```python
continued = pipe(
    prompt=(
        "Glowing blue flowers open beside the path as the fox continues forward. "
        "The fox passes the flowers and looks around."
    ),
    height=704,
    width=1280,
    num_frames=120,
    fps=24,
    num_inference_steps=4,
    session=first["session"],
    output=["videos", "session"],
)
export_to_video(continued["videos"][0], "hy_nomas_continued.mp4", fps=24, quality=None)
```

The result contains both turns, totaling ten seconds. A structured continuation can instead use
`prompt=json.dumps(prompts[1])`. Its action timestamps remain relative to the new segment. Pass
`session=continued["session"]` for the next turn; a continuation restores the session's generator state when no new
generator is supplied.

The pipeline forks the cache before adding new segments, so an earlier state can be reused for alternative
continuations. Keep the pipeline components, precision, device, and resolution unchanged when reusing a state.

Both `videos` and `latents` include the accumulated history. Each call contributes exactly `num_frames` pixel frames;
padding at the end of a call's last segment is removed. Decoding runs over that history, which allows the temporal
decoder to revise the end of the previous clip. The session is a runtime object and is not part of `save_pretrained`.

For identical noise between a combined rollout and incremental calls, set `noise_capacity` on the first call to the
total latent timeline capacity. Reuse that session without supplying a new generator. This allocates a fixed noise
timeline and rejects requests beyond its capacity. The decoder has independent noise; `decoder_generator` selects
its initial seed, with 42 as the default. PixelUMM does not advance this generator.

For applications that need incremental pixel updates, `pipe.decoder.create_streaming_decode_session()` provides
`push_preview`, `finish`, and `rollback_last_push`. Preview updates identify the pixel offset to replace.

## Reuse individual blocks

```python
from diffusers.modular_pipelines.hy_nomas.encoders import HyNomasTextEncoderStep

encoder = HyNomasTextEncoderStep().init_pipeline("tencent/Hy-Nomas")
encoder.load_components(dtype=torch.bfloat16)
encoder.to(device="cuda", dtype=torch.bfloat16)
encoded = encoder(prompt="A forest path.", output="prompt_embeds")
```

The distilled workflow uses guidance scale one. Training, original application servers, prompt expansion services,
and custom FP8 kernels remain outside this inference integration.

## HyNomasModularPipeline

[[autodoc]] HyNomasModularPipeline

## HyNomasAutoBlocks

[[autodoc]] HyNomasAutoBlocks

## HyNomasGenerationState

[[autodoc]] HyNomasGenerationState
