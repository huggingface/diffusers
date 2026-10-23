<!-- Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License. -->

# Hy-Nomas

Hy-Nomas generates a world one video segment at a time, conditioned on text and optional camera extrinsics. Its
distilled checkpoint uses eight flow-matching Euler steps, a Qwen2.5-VL text encoder, and a PixelUMM diffusion decoder.

This implementation supports the released double-stream transformer, action-aware text visibility, camera attention,
and continuation through an explicit generation state. Text prompts can be plain strings or the segment JSON produced
by the model's prompt preprocessing. Action timestamps in JSON must already be relative to their segment.

## Installation

Install PyTorch 2.10.0 with CUDA support for your system, then run the following command from the Diffusers checkout:

```bash
python -m pip install -e . "transformers==5.20.2" "accelerate==1.13.0" imageio imageio-ffmpeg
```

The text encoder requires `Qwen2_5_VLTextModel`, provided by the Transformers version above. The video export example
also requires `imageio` and `imageio-ffmpeg`. The base `pip install -e .` command does not install these model-specific
dependencies. Use a converted model directory as described below; the original Hy-Nomas code checkout is not required.

## Convert the checkpoints

Run the conversion script from a Diffusers checkout. The decoder input must be the original floating-point checkpoint.
The script checks parameter mappings and verifies the saved components by loading them again.

```bash
python scripts/convert_hy_nomas_to_diffusers.py \
  --transformer_checkpoint /path/to/dit.safetensors \
  --decoder_checkpoint /path/to/pixelumm.safetensors \
  --text_encoder_path /path/to/Qwen2.5-VL-7B-Instruct \
  --output_path /path/to/hy-nomas-diffusers \
  --dtype bfloat16
```

The output contains `modular_model_index.json` and standard component subdirectories. The converted pipeline loads
without importing the original repository or executing remote code.

## Generate video

```python
import torch

from diffusers import ModularPipeline
from diffusers.utils import export_to_video

pipe = ModularPipeline.from_pretrained("/path/to/hy-nomas-diffusers")
pipe.load_components(dtype=torch.bfloat16)
pipe.to(device="cuda", dtype=torch.bfloat16)

result = pipe(
    prompt=["A forest beside a river.", "The view moves along the river toward a waterfall."],
    height=704,
    width=1280,
    num_frames=97,
    segment_latent_lengths=[12, 13],
    num_inference_steps=8,
    generator=torch.Generator("cpu").manual_seed(42),
    output=["videos", "session"],
)
export_to_video(result["videos"][0], "world.mp4", fps=24)
```

Pass the dtype to both loading and `to` to reproduce the original BF16 inference. The latter also casts the text
encoder's nonpersistent rotary-position buffers, which are recreated in FP32 when loading its weights.

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

A list of prompts represents **successive segments of one video**, with batch size one. It is not a batch of independent
videos. `num_frames` is the number of new pixel frames; the latent frame count is `(num_frames - 1) // 4 + 1`.
`segment_latent_lengths` must sum to this count. If omitted, all segments receive equal lengths.
Height and width must be multiples of 32. For the original nominal 720p preset, the decoded height is 704 pixels.

For visual quality checks, use segment lengths similar to the example. Very short segments and decoding windows can
produce oversharpened or oversaturated clips. The default PixelUMM decoding window contains 10 latent frames.

Use `output_type="latent"` to skip decoding. The `latents` output has shape `(1, 32, T, H/32, W/32)`.
Video output types are `np` (batch, frames, height, width, channels), `pt` (batch, frames, channels, height, width),
and `pil`.

## Camera control

Pass `camera_extrinsics` with shape `(1, num_frames, 4, 4)`. These are **normalized camera-to-world matrices in the
model's training coordinate system**. Camera calibration, trajectory generation, and normalization must be performed
before calling the pipeline. Omitting the argument disables camera conditioning.

```python
camera = torch.eye(4).repeat(1, 97, 1, 1)
camera[:, :, 0, 3] = torch.linspace(0, 0.2, 97)
videos = pipe(
    prompt="A forest path.",
    height=704,
    width=1280,
    num_frames=97,
    camera_extrinsics=camera,
    output="videos",
)
```

### Display recorded keyboard controls

The `examples/community/hy_nomas_keyboard_overlay.py` example in the Diffusers checkout displays WASD and all four arrow
keys on an existing silent video. It runs on CPU without model weights or a Hy-Nomas checkout. The original video is
preserved and the example refuses to overwrite an existing output.

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

## Continue or branch a world

Pass the returned `session` to another call. The pipeline forks the cache before adding new segments, so the same
state can be reused for alternative continuations. Keep the pipeline components, precision, device, and resolution
unchanged when reusing a state.

```python
continued = pipe(
    prompt="The river opens into a lake.",
    height=704,
    width=1280,
    num_frames=49,
    session=result["session"],
    output=["videos", "session"],
)
export_to_video(continued["videos"][0], "continued-world.mp4", fps=24)
```

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

encoder = HyNomasTextEncoderStep().init_pipeline("/path/to/hy-nomas-diffusers")
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
