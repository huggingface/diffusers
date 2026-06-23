# Cosmos3 — smoke-test runner

The canonical reference for `Cosmos3OmniPipeline` lives in the diffusers docs:
[`docs/source/en/api/pipelines/cosmos3.md`](../../docs/source/en/api/pipelines/cosmos3.md). Use the
examples there as the source of truth for application code — they cover text-to-image,
text-to-video, image-to-video, and text+sound modes.

This directory provides two files:

- `inference_cosmos3.py` — the runnable CLI (text-to-image/video, image-to-video, sound, action
  modes). Single-GPU by default; pass `--tp-degree` / `--cp-degree` and launch with `torchrun`
  to run any modality multi-GPU (see [Multi-GPU inference](#multi-gpu-inference-context-parallelism)
  below).
- `cosmos_parallel.py` — the importable multi-GPU helpers (context + tensor parallelism). No
  `main`; the CLI imports from it. Read it to understand or adapt the sharding.

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

Video-to-video (condition on the leading frames of a clip and continue it):

```bash
python examples/cosmos3/inference_cosmos3.py \
    --prompt "A robotic arm finishes pouring liquid into the glass." \
    --video-path "https://github.com/nvidia-cosmos/cosmos-dependencies/raw/refs/heads/assets/cosmos3/inputs/vision/robot_pouring.mp4" \
    --condition-frame-indexes-vision 0,1 \
    --condition-video-keep first
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
    --prompt "Put the pot to the left of the purple item." \
    --vision-path "https://github.com/nvidia-cosmos/cosmos-dependencies/raw/refs/heads/assets/cosmos3/inputs/action/bridge_0.mp4" \
    --action-mode forward_dynamics \
    --action-path "https://github.com/nvidia-cosmos/cosmos-dependencies/raw/refs/heads/assets/cosmos3/inputs/action/bridge_0.json" \
    --action-chunk-size 16 \
    --domain-name bridge_orig_lerobot \
    --resolution-tier 480 --fps 5 \
    --num-inference-steps 30 --guidance-scale 1.0 --flow-shift 10.0 --seed 0 \
    --output results/cosmos3_forward_dynamics_robot
```

Action forward dynamics, autonomous-vehicle domain:

```bash
python examples/cosmos3/inference_cosmos3.py \
    --model nano \
    --prompt "You are an autonomous vehicle planning system." \
    --vision-path "https://github.com/nvidia-cosmos/cosmos-dependencies/raw/refs/heads/assets/cosmos3/inputs/action/av_vision_25_73d01c91-51f0-46cf-9b76-5682a76fb349.mp4" \
    --action-mode forward_dynamics \
    --action-path "https://github.com/nvidia-cosmos/cosmos-dependencies/raw/refs/heads/assets/cosmos3/inputs/action/av_action_25.json" \
    --action-chunk-size 60 \
    --domain-name av \
    --resolution-tier 480 --fps 10 \
    --num-inference-steps 30 --guidance-scale 1.0 --flow-shift 10.0 --seed 0 \
    --output results/cosmos3_forward_dynamics_av
```

Action inverse dynamics, robot domain (predict actions from an observed video):

```bash
python examples/cosmos3/inference_cosmos3.py \
    --model nano \
    --prompt "Put the pot to the left of the purple item." \
    --vision-path "https://github.com/nvidia-cosmos/cosmos-dependencies/raw/refs/heads/assets/cosmos3/inputs/action/bridge_0.mp4" \
    --action-mode inverse_dynamics \
    --action-chunk-size 16 \
    --domain-name bridge_orig_lerobot \
    --resolution-tier 480 --fps 5 \
    --num-inference-steps 30 --guidance-scale 1.0 --flow-shift 10.0 --seed 0 \
    --output results/cosmos3_inverse_dynamics_robot
```

Action inverse dynamics, autonomous-vehicle domain:

```bash
python examples/cosmos3/inference_cosmos3.py \
    --model nano \
    --prompt "You are an autonomous vehicle planning system." \
    --vision-path "https://github.com/nvidia-cosmos/cosmos-dependencies/raw/refs/heads/assets/cosmos3/inputs/action/av_vision_25_73d01c91-51f0-46cf-9b76-5682a76fb349.mp4" \
    --action-mode inverse_dynamics \
    --action-chunk-size 60 \
    --domain-name av \
    --resolution-tier 480 --fps 10 \
    --num-inference-steps 30 --guidance-scale 1.0 --flow-shift 10.0 --seed 0 \
    --output results/cosmos3_inverse_dynamics_av
```

Action policy, robot domain (predict both future video and actions from the first observation frame):

```bash
python examples/cosmos3/inference_cosmos3.py \
    --model nano \
    --prompt "Put the pot to the left of the purple item." \
    --vision-path "https://github.com/nvidia-cosmos/cosmos-dependencies/raw/refs/heads/assets/cosmos3/inputs/action/bridge_0.mp4" \
    --action-mode policy \
    --action-chunk-size 16 \
    --domain-name bridge_orig_lerobot \
    --resolution-tier 480 --fps 5 \
    --num-inference-steps 30 --guidance-scale 1.0 --flow-shift 10.0 --seed 0 \
    --output results/cosmos3_policy_robot
```

Action policy, autonomous-vehicle domain:

```bash
python examples/cosmos3/inference_cosmos3.py \
    --model nano \
    --prompt "You are an autonomous vehicle planning system. Please go backward." \
    --vision-path "https://github.com/nvidia-cosmos/cosmos-dependencies/raw/refs/heads/assets/cosmos3/inputs/action/av_vision_25_73d01c91-51f0-46cf-9b76-5682a76fb349.mp4" \
    --action-mode policy \
    --action-chunk-size 60 \
    --domain-name av \
    --resolution-tier 480 --fps 10 \
    --num-inference-steps 30 --guidance-scale 1.0 --flow-shift 10.0 --seed 0 \
    --output results/cosmos3_policy_av
```

Action modes use `action_chunk_size + 1` conditioning frames. `forward_dynamics` consumes `--action-path`; `inverse_dynamics` and `policy` write predicted actions to `sample_action.json` in model-normalized action space. This script loads `--vision-path` as a video for all action modes; `policy` and `forward_dynamics` condition only on the first frame, while `inverse_dynamics` uses the whole video.

Pass `--prompt` as a plain task description and select the camera perspective with `--view-point` (default `ego_view`); the pipeline builds the structured action caption (task, viewpoint, duration, FPS, resolution) the model was trained on. Do not hand-write the viewpoint sentence into `--prompt`.

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
| `--flow-shift` | `None` | Override `UniPCMultistepScheduler.flow_shift` (and force `use_karras_sigmas=False`); left at the checkpoint default when unset. Cosmos3 runs use `10.0`. |
| `--enable-sound` | off | Generate a synchronized audio track. |
| `--action-mode` | `None` | Enable action conditioning/generation. One of `forward_dynamics`, `inverse_dynamics`, or `policy`. |
| `--action-path` | `None` | URL or local JSON action path for `forward_dynamics`. |
| `--action-chunk-size` | `None` | Number of action tokens. Action runs generate/use `action_chunk_size + 1` video frames. |
| `--domain-name` | `None` | Action embodiment domain, for example `bridge_orig_lerobot` or `av`. |
| `--view-point` | `ego_view` | Camera perspective for the action caption's framing (`ego_view`, `third_person_view`, `wrist_view`, `concat_view`). Action only. |
| `--no-duration-template` | off | Skip the duration metadata sentence appended to the prompt and negative prompt. Ignored for `--num-frames 1` and for action modes (which build a structured caption instead). |
| `--no-resolution-template` | off | Skip the resolution metadata sentence appended to the prompt and negative prompt. Ignored for action modes. |
| `--output` | `.` | Directory to write `sample.jpg` or `sample.mp4`. |

## Multi-GPU inference (context parallelism)

Cosmos 3 can be sharded across GPUs on two orthogonal axes (implemented in `cosmos_parallel.py`):

- **Context parallelism (CP)** — `enable_cosmos3_context_parallel`. The *sequence* is sharded
  across GPUs and attention runs with two Ulysses all-to-all collectives per layer, cutting
  per-step latency for long videos / high resolutions. Weights are replicated, so this is for
  models that already fit one GPU (`Nano`).
- **Tensor parallelism (TP)** — `enable_cosmos3_tensor_parallel`. The attention and MLP *weight*
  matrices are sharded across GPUs (Megatron-style), so a checkpoint that doesn't fit one GPU
  (`Super`, ~120 GB) loads. The sequence is not sharded.
- **TP + CP** — both at once on a 2-D `(tp, cp)` mesh: a large model *and* latency.

The model itself carries no parallelism logic — it exposes small no-op shard/gather seams, and
`cosmos_parallel.py` implements the entire path (collectives, GQA KV-head handling, ragged-length
padding, the dual-pathway attention, weight sharding) behind those two helpers. It is meant to be
read end to end and adapted.

The CLI imports these helpers, so you run **any modality** (text-to-image/video, image-to-video,
sound, action modes) multi-GPU by adding `--tp-degree` / `--cp-degree` and launching with
[torchrun](https://docs.pytorch.org/docs/stable/elastic/run.html) — `--tp-degree * --cp-degree`
must equal `--nproc_per_node`:

```bash
# CP only (Nano): CP degree must divide the 32 query heads.
torchrun --nproc_per_node 4 examples/cosmos3/inference_cosmos3.py --model nano --cp-degree 4 --prompt "..."

# TP only (Super): TP degree must divide the 64 query heads and 8 KV heads.
torchrun --nproc_per_node 4 examples/cosmos3/inference_cosmos3.py --model super --tp-degree 4 --prompt "..."

# TP + CP (Super), 4 GPUs as 2 x 2, with sound:
torchrun --nproc_per_node 4 examples/cosmos3/inference_cosmos3.py \
    --model super --tp-degree 2 --cp-degree 2 --enable-sound --prompt "A waterfall in a forest."
```

Notes:

- The helpers use the `native` attention backend (the only one that supports GQA's `enable_gqa`),
  and expand the KV heads to the query-head count so SDPA picks the flash kernel — passing
  `enable_gqa=True` forces the math kernel, which materializes the full `[S, S]` scores and OOMs
  on long sequences.
- Only Ulysses is supported (not ring attention).
- The CP/Ulysses degree must divide the query heads (32 for `Nano`, 64 for `Super`). For TP,
  `tp` must divide the KV heads (8), and `tp * cp` must divide the query heads.
- TP all-reduces on every block, so it's bandwidth-heavy — use the smallest TP degree that makes
  the weights fit and put the remaining GPUs into CP.
- Generation size is set with the usual CLI flags (`--num-frames` / `--height` / `--width`), and
  multi-GPU runs require a seed for reproducibility across ranks (the CLI sets one if you omit `--seed`).
- On some multi-GPU topologies the first NCCL all-to-all can hang; if a run stalls at the first
  denoising step, set `NCCL_P2P_DISABLE=1` before launching.

See the [pipeline docs](../../docs/source/en/api/pipelines/cosmos3.md#context-parallelism) for how
to enable CP and TP from your own pipeline code.
