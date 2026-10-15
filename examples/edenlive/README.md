# EdenLive inference

`inference_edenlive.py` runs offline video generation from a reference image, text, and a proxy video. It supports
the four released profiles through `EdenLivePipeline`, with single-GPU execution or Ulysses context parallelism
and PyTorch FSDP2 across two or four GPUs. See the [pipeline guide](../../docs/source/en/api/pipelines/edenlive.md)
for the Python API, component loading, decoder overrides, and output formats.

## Setup

Run these commands from the Diffusers repository root in an environment with a CUDA-compatible PyTorch build:

```bash
pip install -e .
pip install --upgrade -r examples/edenlive/requirements.txt
```

Transformers must provide `DINOv3ViTModel`, `UMT5EncoderModel`, and `CLIPVisionModel`. Multi-GPU execution also
requires PyTorch's composable FSDP2 API (`torch.distributed.fsdp.fully_shard`). The release directory or Hub repository
must contain EdenLive's root `model_index.json` and the original weights for the selected profile.

## Single GPU

```bash
python examples/edenlive/inference_edenlive.py \
  --model VAST-AI/EdenLive --profile 5b-causal \
  --video proxy.mp4 --image reference.png \
  --prompt "A character explores a snowy landscape." \
  --num-frames 49 --output edenlive.mp4
```

`--model` also accepts a local release or saved Diffusers pipeline. Add `--local-files-only` to use only local or
cached weights. The CLI requires `--profile`; for a saved pipeline it must match the saved configuration.

| Profile | Default height x width | Default fps | Default steps | Guidance scale |
|---|---|---|---|---|
| `5b-bidir` | 704 x 1280 | 24 | 48 | 3 |
| `14b-bidir` | 480 x 832 | 16 | 30 | 3 |
| `5b-causal` | 704 x 1280 | 24 | 4 per chunk | 1 |
| `14b-causal` | 480 x 832 | 16 | 4 per chunk | 1 |

The CLI resamples source timestamps to `--fps`, center-crops to the output aspect ratio, and resizes the frames.
`--num-frames` must be positive, have the form $4k+1$, and fit the clip at the selected frame rate. If omitted, the
largest valid prefix is used. Output height and width must be multiples of 32 for 5B or 16 for 14B. Unlike this CLI,
the Python pipeline expects frames that are already sampled at the desired output frame rate.

The released causal schedules contain four fixed evaluation points. Keep `--num-inference-steps` at four or omit it
for causal profiles. The bidirectional profiles accept a different positive step count. `--guidance-scale` above
one enables the negative prompt branch; supply its text with `--negative-prompt`.

## Multiple GPUs

Launch one process per GPU, with two or four processes:

```bash
torchrun --standalone --nproc_per_node=2 examples/edenlive/inference_edenlive.py \
  --model VAST-AI/EdenLive --profile 14b-causal \
  --video proxy.mp4 --image reference.png \
  --prompt "A character explores a snowy landscape." \
  --num-frames 81 --output edenlive.mp4
```

Set `--nproc_per_node=4` for four GPUs. The transformer uses Ulysses context parallelism and FSDP2 over the same
ranks; other components are replicated. Add `--no-fsdp` to retain context parallelism with replicated transformer
weights. Only rank zero writes the video and JSON report. When comparing seeded results, keep the parallel degree
and attention backend fixed because BF16 matrix multiplication shapes can affect predictions.

## Temporal prompts

For `5b-bidir`, pass `--text-timeline timeline.json`. This example describes 49 frames at 24 fps:

```json
{
  "fps": 24,
  "num_frames": 49,
  "segments": [
    {"start_sec": 0.0, "end_sec": 1.0, "text": "A character walks through snow."},
    {"start_sec": 1.0, "end_sec": 2.0, "text": "The character approaches a warm campfire."}
  ]
}
```

Match the JSON `fps` and `num_frames` to `--fps` and `--num-frames`. Segments start at zero, are contiguous, and
cover the last frame timestamp; each segment must include at least one latent frame. A latent frame at an exact
segment boundary uses the segment on its left. Other profiles do not accept timelines.

For either causal profile, pass `--chunk-prompts prompts.json` containing a JSON list of complete prompts. A 49-frame
video has 13 latent frames, so the released three-frame chunks require five prompts:

```json
[
  "A character walks through snow.",
  "A character walks through snow.",
  "The character approaches a campfire.",
  "The character warms their hands beside the campfire.",
  "The character rests beside the campfire."
]
```

The final chunk may be shorter. The CLI still requires `--prompt` when either temporal format is supplied; the
timeline or chunk prompts determine the positive captions used for those segments or chunks. Each invocation
starts a fresh causal history.

## Decoding and reports

LightTAE is the default decoder for original releases. Add `--full-vae` to use the full Wan VAE for output decoding.
The VAE always encodes the reference image. `--vae-tiling` enables tiling on that VAE; it does not enable LightTAE
tiling. Use `--seed` to select the noise seed and `--attention-backend` to select an available Diffusers attention
backend. Run `python examples/edenlive/inference_edenlive.py --help` for all options.

The output path selects the video filename. A JSON file with the same stem records the CLI arguments, resampled
source frame indices, output fps and frame count, world size, FSDP2 status, and per-rank elapsed seconds and peak
allocated/reserved GPU memory in bytes. Timing covers the pipeline call, including conditioning and decoding;
it excludes checkpoint loading, input video preprocessing, and video export.
