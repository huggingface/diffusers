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

### Useful flags

| Flag | Default | Description |
|---|---|---|
| `--prompt` | (required) | Text prompt. |
| `--vision-path` | `None` | URL or local path for an image-conditioning frame (image-to-video). |
| `--num-frames` | `189` | `1` = image, otherwise number of video frames (`189` ≈ 7.9 s @ 24 FPS). |
| `--height` / `--width` | `720` / `1280` | Output resolution (must be a multiple of the VAE spatial scale factor). |
| `--fps` | `24.0` | Frame rate of the generated video. |
| `--enable-sound` | off | Generate a synchronized audio track. |
| `--no-duration-template` | off | Skip the duration metadata sentence appended to the prompt and negative prompt. Ignored for `--num-frames 1`. |
| `--no-resolution-template` | off | Skip the resolution metadata sentence appended to the prompt and negative prompt. |
| `--output` | `.` | Directory to write `sample.jpg` or `sample.mp4`. |
