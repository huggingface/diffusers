# Cosmos3 — smoke-test runner

The canonical reference for `Cosmos3OmniPipeline` lives in the diffusers docs:
[`docs/source/en/api/pipelines/cosmos3.md`](../../docs/source/en/api/pipelines/cosmos3.md). Use the
examples there as the source of truth for application code — they cover text-to-image,
text-to-video, image-to-video, and text+sound modes.

This directory provides a small CLI wrapper (`inference_cosmos3.py`) that exercises the full
load → encode → denoise → decode path against either the Hub release or a local checkpoint
during development.

## Setup

```bash
pip install -r examples/cosmos3/requirements.txt
```

## Usage

Text-to-image:

```bash
python examples/cosmos3/inference_cosmos3.py \
    --prompt "A medium shot of a modern robotics research laboratory…" \
    --num-frames 1
```

Text-to-video:

```bash
python examples/cosmos3/inference_cosmos3.py \
    --prompt "A waterfall cascading down a rocky cliff in a lush forest."
```

Image-to-video:

```bash
python examples/cosmos3/inference_cosmos3.py \
    --prompt "The right robotic hand picks up the red sphere…" \
    --vision-path https://github.com/nvidia-cosmos/cosmos-dependencies/releases/download/assets/robot_153.jpg
```

Text-to-video-with-sound (sound-capable checkpoint only):

```bash
python examples/cosmos3/inference_cosmos3.py \
    --prompt "A waterfall in a lush forest." \
    --enable-sound
```

Action forward dynamics, robot domain (predict video from an observation video and a provided action chunk):

```bash
python examples/cosmos3/inference_cosmos3.py \
    --model nano \
    --prompt "Put the pot to the left of the purple item. This video is captured from a first-person perspective looking at the scene." \
    --vision-path "https://github.com/nvidia-cosmos/cosmos-dependencies/raw/refs/heads/assets/cosmos3/inputs/action/bridge_0.mp4" \
    --action-mode forward_dynamics \
    --action-path "https://github.com/nvidia-cosmos/cosmos-dependencies/raw/refs/heads/assets/cosmos3/inputs/action/bridge_0.json" \
    --action-chunk-size 16 \
    --domain-name bridge_orig_lerobot \
    --resolution-tier 480 --fps 5 \
    --num-inference-steps 30 --guidance-scale 1.0 --flow-shift 5.0 --seed 0 \
    --output results/cosmos3_forward_dynamics_robot
```

Action forward dynamics, autonomous-vehicle domain:

```bash
python examples/cosmos3/inference_cosmos3.py \
    --model nano \
    --prompt "You are an autonomous vehicle planning system. This video is captured from a first-person perspective looking at the scene." \
    --vision-path "https://github.com/nvidia-cosmos/cosmos-dependencies/raw/refs/heads/assets/cosmos3/inputs/action/av_vision_25_73d01c91-51f0-46cf-9b76-5682a76fb349.mp4" \
    --action-mode forward_dynamics \
    --action-path "https://github.com/nvidia-cosmos/cosmos-dependencies/raw/refs/heads/assets/cosmos3/inputs/action/av_action_25.json" \
    --action-chunk-size 60 \
    --domain-name av \
    --resolution-tier 480 --fps 10 \
    --num-inference-steps 30 --guidance-scale 1.0 --flow-shift 5.0 --seed 0 \
    --output results/cosmos3_forward_dynamics_av
```

Action inverse dynamics, robot domain (predict actions from an observed video):

```bash
python examples/cosmos3/inference_cosmos3.py \
    --model nano \
    --prompt "Put the pot to the left of the purple item. This video is captured from a first-person perspective looking at the scene." \
    --vision-path "https://github.com/nvidia-cosmos/cosmos-dependencies/raw/refs/heads/assets/cosmos3/inputs/action/bridge_0.mp4" \
    --action-mode inverse_dynamics \
    --action-chunk-size 16 \
    --domain-name bridge_orig_lerobot \
    --resolution-tier 480 --fps 5 \
    --num-inference-steps 30 --guidance-scale 1.0 --flow-shift 5.0 --seed 0 \
    --output results/cosmos3_inverse_dynamics_robot
```

Action inverse dynamics, autonomous-vehicle domain:

```bash
python examples/cosmos3/inference_cosmos3.py \
    --model nano \
    --prompt "You are an autonomous vehicle planning system. This video is captured from a first-person perspective looking at the scene." \
    --vision-path "https://github.com/nvidia-cosmos/cosmos-dependencies/raw/refs/heads/assets/cosmos3/inputs/action/av_vision_25_73d01c91-51f0-46cf-9b76-5682a76fb349.mp4" \
    --action-mode inverse_dynamics \
    --action-chunk-size 60 \
    --domain-name av \
    --resolution-tier 480 --fps 10 \
    --num-inference-steps 30 --guidance-scale 1.0 --flow-shift 5.0 --seed 0 \
    --output results/cosmos3_inverse_dynamics_av
```

Action policy, robot domain (predict both future video and actions from the first observation frame):

```bash
python examples/cosmos3/inference_cosmos3.py \
    --model nano \
    --prompt "Put the pot to the left of the purple item. This video is captured from a first-person perspective looking at the scene." \
    --vision-path "https://github.com/nvidia-cosmos/cosmos-dependencies/raw/refs/heads/assets/cosmos3/inputs/action/bridge_0.mp4" \
    --action-mode policy \
    --action-chunk-size 16 \
    --domain-name bridge_orig_lerobot \
    --resolution-tier 480 --fps 5 \
    --num-inference-steps 30 --guidance-scale 1.0 --flow-shift 5.0 --seed 0 \
    --output results/cosmos3_policy_robot
```

Action policy, autonomous-vehicle domain:

```bash
python examples/cosmos3/inference_cosmos3.py \
    --model nano \
    --prompt "You are an autonomous vehicle planning system. Please go backward. This video is captured from a first-person perspective looking at the scene." \
    --vision-path "https://github.com/nvidia-cosmos/cosmos-dependencies/raw/refs/heads/assets/cosmos3/inputs/action/av_vision_25_73d01c91-51f0-46cf-9b76-5682a76fb349.mp4" \
    --action-mode policy \
    --action-chunk-size 60 \
    --domain-name av \
    --resolution-tier 480 --fps 10 \
    --num-inference-steps 30 --guidance-scale 1.0 --flow-shift 5.0 --seed 0 \
    --output results/cosmos3_policy_av
```

Action modes use `action_chunk_size + 1` conditioning frames. `forward_dynamics` consumes `--action-path`; `inverse_dynamics` and `policy` write predicted actions to `sample_action.json` in model-normalized action space. This script loads `--vision-path` as a video for all action modes; `policy` and `forward_dynamics` condition only on the first frame, while `inverse_dynamics` uses the whole video.

`--resolution-tier` is a resolution *tier* (`256`/`480`/`704`/`720`). The tier keys a table of predefined aspect-ratio canvases; the one closest to the input aspect ratio becomes the padded conditioning canvas. It is not the output frame size: the input is downscaled (never upscaled) and padded to fill the canvas, then the padding is cropped from the latents so the decoded output follows the downscaled input content. `--height` / `--width` (and `--num-frames`) are ignored for action modes.

Pick the tier that matches the native resolution of your conditioning input (`480` for ~480p, `720` for ~720p). A tier below your input downscales it and discards detail; a tier above your input gains no resolution (content is never upscaled), wastes compute on padding, and is a train/inference distribution mismatch that can degrade quality.

### Useful flags

| Flag | Default | Description |
|---|---|---|
| `--prompt` | (required) | Text prompt. |
| `--vision-path` | `None` | URL or local path for an image-conditioning frame (image-to-video), or the image/video conditioning for action modes. |
| `--num-frames` | `189` | `1` = image, otherwise number of video frames (`189` ≈ 7.9 s @ 24 FPS). Ignored for action modes (derived from `--action-chunk-size`). |
| `--height` / `--width` | `720` / `1280` | Output resolution (must be a multiple of the VAE spatial scale factor). Ignored for action modes; use `--resolution-tier`. |
| `--resolution-tier` | `480` | Action resolution tier (`256`/`480`/`704`/`720`): selects the aspect bin / padded conditioning canvas, not the output size. |
| `--fps` | `24.0` | Frame rate of the generated video. |
| `--enable-sound` | off | Generate a synchronized audio track. |
| `--action-mode` | `None` | Enable action conditioning/generation. One of `forward_dynamics`, `inverse_dynamics`, or `policy`. |
| `--action-path` | `None` | URL or local JSON action path for `forward_dynamics`. |
| `--action-chunk-size` | `None` | Number of action tokens. Action runs generate/use `action_chunk_size + 1` video frames. |
| `--domain-name` | `None` | Action embodiment domain, for example `bridge_orig_lerobot` or `av`. |
| `--no-duration-template` | off | Skip the duration metadata sentence appended to the prompt and negative prompt. Ignored for `--num-frames 1`. |
| `--no-resolution-template` | off | Skip the resolution metadata sentence appended to the prompt and negative prompt. |
| `--output` | `.` | Directory to write `sample.jpg` or `sample.mp4`. |
